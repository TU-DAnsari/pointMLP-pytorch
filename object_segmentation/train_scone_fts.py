from __future__ import print_function
import os
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from util.mixed_dataset import MixedOccupancyDataset
from util.simple_dataset import SimpleDataset
import torch.nn.functional as F
import models
import numpy as np
from torch.utils.data import DataLoader
from util.util import IOStream
from tqdm import tqdm
from collections import defaultdict
import random
from pathlib import Path
import datetime
from util.util import parse_args, compute_class_weights
from util.progress_plots import save_plots
import shutil
import yaml
from types import SimpleNamespace

DATA_PATH = Path("/home/danish/lobster/ml/data/shapenet/shapenet_partials_mixed.h5")
BACKBONE_DIR = Path("/home/danish/lobster/ml/pointMLP-pytorch/pointMLP-pytorch/object_segmentation/checkpoints/segmentation/pointMLPSegmentationMedium_2026-07-09_16-09")
with open(BACKBONE_DIR / "config.yaml", 'r') as f:
    config_backbone = yaml.safe_load(f)
    args_backbone = SimpleNamespace(**config_backbone)

device = torch.device("cuda")

backbone = models.__dict__[args_backbone.model](4, args_backbone.num_points, 3)
checkpoint = torch.load(BACKBONE_DIR / "best_insiou_model.pth", weights_only=False, map_location=device)
state_dict = checkpoint["model"]
backbone.load_state_dict(state_dict)
backbone.to(device)

def feature_loader(data_loader, norm_stats=None):
    reference_points = []
    reference_features = []
    combined_points = []
    combined_features = []
    combined_labels = []

    with torch.no_grad():
        for reference, _, _, _, combined, labels in tqdm(data_loader, total=len(data_loader), smoothing=0.9):
            reference_tensor = reference.float().permute(0, 2, 1).cuda(non_blocking=True)
            combined_tensor = combined.float().permute(0, 2, 1).cuda(non_blocking=True)

            reference_out = backbone.encoder(reference_tensor, reference_tensor)
            combined_out = backbone.encoder(combined_tensor, combined_tensor)

            features_reference = backbone.seg_head.decode(reference_out)
            features_combined = backbone.seg_head.decode(combined_out)

            features_reference = features_reference.permute(0, 2, 1).cpu().numpy()
            features_combined = features_combined.permute(0, 2, 1).cpu().numpy()

            reference_points.append(reference)
            reference_features.append(features_reference[:, :, :128])
            combined_points.append(combined)
            combined_features.append(features_combined[:, :, :128])

            combined_labels.append(labels)

    reference_points = np.concatenate(reference_points, axis=0)
    reference_features = np.concatenate(reference_features, axis=0)
    combined_points = np.concatenate(combined_points, axis=0)
    combined_features = np.concatenate(combined_features, axis=0)
    combined_labels = np.concatenate(combined_labels, axis=0)

    if norm_stats is None:
        all_features = np.concatenate([reference_features, combined_features], axis=1)
        mean = all_features.mean(axis=(0, 1), keepdims=True)
        std = all_features.std(axis=(0, 1), keepdims=True) + 1e-6
        norm_stats = (mean, std)

    # mean, std = norm_stats
    # reference_features = (reference_features - mean) / std
    # combined_features = (combined_features - mean) / std

    data = [reference_points, reference_features, combined_points, combined_features, combined_labels]
    return SimpleDataset(data), norm_stats


def _empty_history():
    return {
        "train_loss": [],
        "train_acc": [],
        "train_iou": [],
        "val_loss": [],
        "val_acc": [],
        "val_iou": [],
    }

def main():    
    args = parse_args()
    assert torch.cuda.is_available(), "Please ensure codes are executed in cuda."

    if args.exp_name is None:
        args.exp_name = args.model + "_" + f"{datetime.datetime.now():%Y-%m-%d_%H-%M}"
        
    checkpoint_dir = 'checkpoints/occupancy/%s' % args.exp_name
    config_save_path = os.path.join(checkpoint_dir, 'config.yaml')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    if not args.eval:
        shutil.copy(args.config, config_save_path)
        with open(config_save_path, 'a') as f:
            f.write(f"\nDATA_PATH: {DATA_PATH}\n")

    log_name = checkpoint_dir + '/%s_%s.log' % (args.model, 'test' if args.eval else 'train')
    io = IOStream(log_name)
    io.cprint(str(args))

    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)

    if args.manual_seed is not None:
        torch.cuda.manual_seed_all(args.manual_seed)

    if not args.eval:
        train(args, io)
    else:
        pass


def weight_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, (torch.nn.Conv2d, torch.nn.Conv1d)):
        torch.nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)


def train(args, io):
    device = torch.device("cuda")
    checkpoint_dir = 'checkpoints/occupancy/%s' % args.exp_name

    model = models.__dict__[args.model]().to(device)
    model.apply(weight_init)

    train_data_pre = MixedOccupancyDataset(DATA_PATH, split="train", num_points=args.num_points)
    val_data_pre = MixedOccupancyDataset(DATA_PATH, split="val", num_points=args.num_points)

    print("Training samples: %d" % len(train_data_pre))
    print("Validation samples: %d" % len(val_data_pre))

    train_loader_pre = DataLoader(train_data_pre, 
                              batch_size=args.batch_size, 
                              shuffle=True,
                              num_workers=args.workers, 
                              drop_last=False, 
                              pin_memory=True, 
                              persistent_workers=True)
    
    val_loader_pre   = DataLoader(val_data_pre, 
                              batch_size=args.test_batch_size, 
                              shuffle=False,
                              num_workers=args.workers, 
                              drop_last=False,
                              pin_memory=True, 
                              persistent_workers=True)
    
    train_feature_dataset, norm_stats = feature_loader(train_loader_pre)
    val_feature_dataset, _ = feature_loader(val_loader_pre, norm_stats)

    train_loader = DataLoader(train_feature_dataset, 
                              batch_size=args.batch_size, 
                              shuffle=True,
                              num_workers=args.workers, 
                              drop_last=False, 
                              pin_memory=True, 
                              persistent_workers=True)
    
    val_loader   = DataLoader(val_feature_dataset, 
                              batch_size=args.test_batch_size, 
                              shuffle=False,
                              num_workers=args.workers, 
                              drop_last=False,
                              pin_memory=True, 
                              persistent_workers=True)
    
    io.cprint(str(model))

    if args.resume:
        state_dict = torch.load(f"{checkpoint_dir}/best_insiou_model.pth", weights_only=False, map_location='cpu')['model']
        state_dict = {
            k.replace("module.", "", 1).replace("_orig_mod.", "", 1): v
            for k, v in state_dict.items()
        }
        model.load_state_dict(state_dict)
        print("Resuming training...")
    else:
        print("Training from scratch...")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params}")


    if args.use_sgd:
        opt = optim.SGD(model.parameters(), lr=args.lr * 100, momentum=args.momentum, weight_decay=0)
    else:
        opt = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    if args.scheduler == 'cos':
        scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr if args.use_sgd else args.lr / 100)
    else:
        scheduler = StepLR(opt, step_size=args.step, gamma=0.5)

    best_acc = 0
    best_iou = 0

    # label_array = train_data.labels.reshape(-1)
    # class_weights = compute_class_weights(label_array.reshape(-1), len(class_labels), device)

    history = _empty_history()

    for epoch in range(args.epochs):
        train_metrics = train_epoch(args, train_loader, model, opt, scheduler, epoch, io)
        test_metrics = test_epoch(args, val_loader, model, epoch, io)

        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["acc"])
        history["train_iou"].append(train_metrics["iou"])
        history["val_loss"].append(test_metrics["loss"])
        history["val_acc"].append(test_metrics["acc"])
        history["val_iou"].append(test_metrics["iou"])

        # save_plots(history, checkpoint_dir, labels_classes=)

        if test_metrics['acc'] > best_acc:
            best_acc = test_metrics['acc']
            io.cprint('Max Acc: %.5f' % best_acc)
            torch.save({'model': model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict(),
                        'optimizer': opt.state_dict(), 'epoch': epoch, 'test_acc': best_acc},
                       f'{checkpoint_dir}/best_acc_model.pth')

        if test_metrics['iou'] > best_iou:
            best_iou = test_metrics['iou']
            io.cprint('Max instance iou: %.5f' % best_iou)
            torch.save({'model': model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict(),
                        'optimizer': opt.state_dict(), 'epoch': epoch, 'best_iou': best_iou},
                       f'{checkpoint_dir}/best_iou_model.pth')

    io.cprint('Final Max Acc: %.5f' % best_acc)
    io.cprint('Final Max instance iou: %.5f' % best_iou)
    torch.save({'model': model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict(),
                'optimizer': opt.state_dict(), 'epoch': args.epochs - 1, 'test_iou': best_iou},
               f'{checkpoint_dir}/model_ep{args.epochs}.pth')


def train_epoch(args, train_loader, model, opt, scheduler, epoch, io):
    train_loss = 0.0
    count = 0.0
    accuracy = []
    iou = 0.0
    model.train()

    for reference_points, reference_features, combined_points, combined_features, combined_labels in tqdm(train_loader, total=len(train_loader), smoothing=0.9):
        opt.zero_grad(set_to_none=True)

        batch_size, num_point, _ = reference_points.size()

        reference_points = reference_points.float().cuda(non_blocking=True)
        reference_features = reference_features.float().cuda(non_blocking=True)
        combined_points = combined_points.float().cuda(non_blocking=True)
        combined_features = combined_features.float().cuda(non_blocking=True)
        labels = combined_labels.float().cuda(non_blocking=True)

        occ_pred = model(reference_points, reference_features, combined_points, combined_features)
        occ_pred = occ_pred.squeeze(-1)
        occ_prob = torch.sigmoid(occ_pred)

        loss = F.mse_loss(occ_prob, labels)

        loss.backward()
        opt.step()

        with torch.no_grad():
            pred_binary = (occ_prob >= 0.5).float()
            correct = pred_binary.eq(labels).sum()

            tp = (pred_binary * labels).sum(dim=1)              # (B,)
            fp = (pred_binary * (1 - labels)).sum(dim=1)        # (B,)
            fn = ((1 - pred_binary) * labels).sum(dim=1)        # (B,)
            denom = tp + fp + fn
            batch_iou = torch.where(denom > 0, tp / denom, torch.ones_like(tp))
            iou   += batch_iou.sum().item()
            count += batch_size

        train_loss += loss.item() * batch_size
        accuracy.append(correct.item() / (batch_size * num_point))

    metrics = {
        "loss": train_loss / count, 
        "acc": np.mean(accuracy), 
        "iou": iou / count
    }

    if args.scheduler == 'cos':
        scheduler.step()
    elif args.scheduler == 'step':
        if opt.param_groups[0]['lr'] > 0.9e-5:
            scheduler.step()
        if opt.param_groups[0]['lr'] < 0.9e-5:
            for pg in opt.param_groups:
                pg['lr'] = 0.9e-5

    io.cprint('Train %d, loss: %.5f, acc: %.5f, iou: %.5f, lr: %f' % (
        epoch + 1, train_loss / count, np.mean(accuracy), iou / count,
        opt.param_groups[0]['lr']))
    
    return metrics



def test_epoch(args, val_loader, model, epoch, io):
    test_loss = 0.0
    count = 0.0
    accuracy = []
    iou = 0.0
    metrics = defaultdict(lambda: list())
    model.eval()

    with torch.no_grad():
        for reference_points, reference_features, combined_points, combined_features, combined_labels in tqdm(val_loader, total=len(val_loader), smoothing=0.9):
            batch_size, num_point, _ = reference_points.size()

            reference_points = reference_points.float().cuda(non_blocking=True)
            reference_features = reference_features.float().cuda(non_blocking=True)
            combined_points = combined_points.float().cuda(non_blocking=True)
            combined_features = combined_features.float().cuda(non_blocking=True)
            labels = combined_labels.float().cuda(non_blocking=True)

            occ_pred = model(reference_points, reference_features, combined_points, combined_features)
            occ_pred = occ_pred.squeeze(-1)
            occ_prob = torch.sigmoid(occ_pred)

            loss = F.mse_loss(occ_prob, labels)

            pred_binary = (occ_prob >= 0.5).float()
            correct = pred_binary.eq(labels).sum()

            tp = (pred_binary * labels).sum(dim=1)              # (B,)
            fp = (pred_binary * (1 - labels)).sum(dim=1)        # (B,)
            fn = ((1 - pred_binary) * labels).sum(dim=1)        # (B,)
            denom = tp + fp + fn
            batch_iou = torch.where(denom > 0, tp / denom, torch.ones_like(tp))
            iou   += batch_iou.sum().item()
            count += batch_size

            test_loss += loss.item() * batch_size
            accuracy.append(correct.item() / (batch_size * num_point))

    metrics = {
        "loss": test_loss / count,
        "acc": np.mean(accuracy),
        "iou": iou / count,
    }

    io.cprint(
        f"Test {epoch+1}, loss: {test_loss/count:.5f}, acc: {metrics['acc']:.5f}, iou: {metrics['iou']:.5f}"
    )

    return metrics

if __name__ == "__main__":
    main()