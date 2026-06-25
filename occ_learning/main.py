from __future__ import print_function
from ast import arg
import os
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from util.lobrob_dataset import LobRobDataset
import torch.nn.functional as F
import torch.nn as nn
import model as models
import numpy as np
from torch.utils.data import DataLoader
from util.util import IOStream
from tqdm import tqdm
from collections import defaultdict
import random
from pathlib import Path
import datetime
from util.util import parse_args
import shutil


DATA_PATH = Path("/home/danish/lobster/ml_data/lobrob/car.h5")
DATASET_CLASS = LobRobDataset

def main():    
    args = parse_args()
    assert torch.cuda.is_available(), "Please ensure codes are executed in cuda."

    if args.exp_name is None:
        args.exp_name = args.model + "_" + f"{datetime.datetime.now():%Y-%m-%d_%H-%M}"
        
    _init_(args=args)

    checkpoint_dir = 'checkpoints/%s' % args.exp_name
    config_save_path = os.path.join(checkpoint_dir, 'config.yaml')
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


def _init_(args):
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/' + args.exp_name):
        os.makedirs('checkpoints/' + args.exp_name)


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

    train_data = DATASET_CLASS(DATA_PATH,
                            split="train", 
                            )
    
    val_data = DATASET_CLASS(DATA_PATH, 
                            split="val", 
                            )

    print("Training samples: %d" % len(train_data))
    print("Validation samples: %d" % len(val_data))

    train_loader = DataLoader(train_data, 
                              batch_size=args.batch_size, 
                              shuffle=True,
                              num_workers=args.workers, 
                              drop_last=True, 
                              pin_memory=True, 
                              persistent_workers=True)
    
    val_loader   = DataLoader(val_data, 
                              batch_size=args.test_batch_size, 
                              shuffle=False,
                              num_workers=args.workers, 
                              drop_last=False,
                              pin_memory=True, 
                              persistent_workers=True)
    
    model = models.__dict__[args.model](len(train_data[0][0])).to(device)
    model.apply(weight_init)

    io.cprint(str(model))

    if args.resume:
        state_dict = torch.load(f"checkpoints/{args.exp_name}/best_insiou_model.pth", weights_only=False, map_location='cpu')['model']
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

    for epoch in range(args.epochs):
        train_epoch(args, train_loader, model, opt, scheduler, epoch, io)
        test_metrics = test_epoch(args, val_loader, model, epoch, io)

        if test_metrics['accuracy'] > best_acc:
            best_acc = test_metrics['accuracy']
            io.cprint('Max Acc: %.5f' % best_acc)
            torch.save({'model': model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict(),
                        'optimizer': opt.state_dict(), 'epoch': epoch, 'test_acc': best_acc},
                       'checkpoints/%s/best_acc_model.pth' % args.exp_name)

        if test_metrics['avg_iou'] > best_iou:
            best_iou = test_metrics['avg_iou']
            io.cprint('Max instance iou: %.5f' % best_iou)
            torch.save({'model': model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict(),
                        'optimizer': opt.state_dict(), 'epoch': epoch, 'best_iou': best_iou},
                       'checkpoints/%s/best_iou_model.pth' % args.exp_name)

    io.cprint('Final Max Acc: %.5f' % best_acc)
    io.cprint('Final Max instance iou: %.5f' % best_iou)
    torch.save({'model': model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict(),
                'optimizer': opt.state_dict(), 'epoch': args.epochs - 1, 'test_iou': best_iou},
               'checkpoints/%s/model_ep%d.pth' % (args.exp_name, args.epochs))


def train_epoch(args, train_loader, model, opt, scheduler, epoch, io):
    train_loss = 0.0
    count = 0.0
    accuracy = []
    iou = 0.0
    model.train()

    for points_partial, points_proxy, labels in tqdm(train_loader, total=len(train_loader), smoothing=0.9):
        batch_size, num_point, _ = points_partial.size()

        points_partial = points_partial.float().permute(0, 2, 1).cuda(non_blocking=True)
        points_proxy = points_proxy.float().permute(0, 2, 1).cuda(non_blocking=True)
        labels = labels.float().cuda(non_blocking=True)

        opt.zero_grad(set_to_none=True)

        occ_pred = model(points_partial, points_proxy)
        occ_prob = torch.sigmoid(occ_pred)


        loss = F.mse_loss(occ_prob, labels)
        loss.backward()
        opt.step()

        with torch.no_grad():
            pred_binary = (occ_prob >= 0.5).float()
            correct = pred_binary.eq(labels).sum()

            # Binary IoU: TP / (TP + FP + FN), averaged over batch
            tp = (pred_binary * labels).sum(dim=1)              # (B,)
            fp = (pred_binary * (1 - labels)).sum(dim=1)        # (B,)
            fn = ((1 - pred_binary) * labels).sum(dim=1)        # (B,)
            denom = tp + fp + fn
            batch_iou = torch.where(denom > 0, tp / denom, torch.ones_like(tp))
            iou   += batch_iou.sum().item()
            count += batch_size

        train_loss += loss.item() * batch_size
        accuracy.append(correct.item() / (batch_size * num_point))

    if args.scheduler == 'cos':
        scheduler.step()
    elif args.scheduler == 'step':
        if opt.param_groups[0]['lr'] > 0.9e-5:
            scheduler.step()
        if opt.param_groups[0]['lr'] < 0.9e-5:
            for pg in opt.param_groups:
                pg['lr'] = 0.9e-5

    io.cprint('Train %d, loss: %.5f, acc: %.5f, ins_iou: %.5f, lr: %f' % (
        epoch + 1, train_loss / count, np.mean(accuracy), iou / count,
        opt.param_groups[0]['lr']))


def test_epoch(args, val_loader, model, epoch, io):
    test_loss = 0.0
    count = 0.0
    accuracy = []
    iou = 0.0
    metrics = defaultdict(lambda: list())
    model.eval()

    with torch.no_grad():
        for points_partial, points_proxy, labels in tqdm(val_loader, total=len(val_loader), smoothing=0.9):
            batch_size, num_point, _ = points_partial.size()

            points_partial = points_partial.float().permute(0, 2, 1).cuda(non_blocking=True)
            points_proxy = points_proxy.float().permute(0, 2, 1).cuda(non_blocking=True)
            labels = labels.float().cuda(non_blocking=True)

            occ_pred = model(points_partial, points_proxy)   
            occ_prob = torch.sigmoid(occ_pred)

            loss = F.mse_loss(occ_prob, labels)

            pred_binary = (occ_prob >= 0.5).float()
            correct = pred_binary.eq(labels).sum()

            # Binary IoU: TP / (TP + FP + FN), averaged over batch
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
        'accuracy':      np.mean(accuracy),
        'avg_iou': iou / count,
    }

    io.cprint(
        f"Test {epoch+1}, loss: {test_loss/count:.5f}, acc: {metrics['accuracy']:.5f}, iou: {metrics['avg_iou']:.5f}"
    )

    return metrics


# def test(args, io):
#     val_data = DATASET_CLASS(DATA_PATH, 
#                             split="val", 
#                             )
    
#     val_loader = DataLoader(val_data, 
#                               batch_size=args.test_batch_size, 
#                               shuffle=False,
#                               num_workers=args.workers, 
#                               drop_last=False,
#                               pin_memory=True, 
#                               persistent_workers=True)

#     device = torch.device("cuda")
#     model = models.__dict__[args.model](len(val_data[0])).to(device)

#     from collections import OrderedDict
#     state_dict = torch.load("checkpoints/%s/best_%s_model.pth" % (args.exp_name, args.model_type),
#                             map_location='cpu')['model']
#     new_state_dict = OrderedDict()
#     for k, v in state_dict.items():
#         new_state_dict[k.replace('module.', '')] = v
#     model.load_state_dict(new_state_dict)
#     model.eval()

#     accuracy = []
#     iou = []
#     per_class_iou  = np.zeros(n_classes, dtype=np.float32)
#     per_class_seen = np.zeros(n_classes, dtype=np.int32)

#     with torch.no_grad():
#         for _, primary_batch, secondary_batch, label_batch, _ in tqdm(val_loader, total=len(val_loader), smoothing=0.9):
#             batch_size, num_point, _ = primary_batch.size()

#             sampling_input = primary_batch.float().permute(0, 2, 1)

#             if len(args.model_input) == 0 and args.use_normals:
#                 model_input = secondary_batch[:, :, -3:].float().permute(0, 2, 1)
#             if len(args.model_input) == 0 and not args.use_normals:
#                 model_input = primary_batch.float().permute(0, 2, 1)
#             if len(args.model_input) != 0 and args.use_normals:
#                 model_input = secondary_batch.float().permute(0, 2, 1)
#             if len(args.model_input) != 0 and not args.use_normals:
#                 raise NotImplementedError("Should not be using this configuration")
            
#             labels = label_batch.long()

#             sampling_input = sampling_input.cuda(non_blocking=True)
#             model_input = model_input.cuda(non_blocking=True) 
#             labels = labels.cuda(non_blocking=True)

#             seg_pred = model(sampling_input, model_input)
#             batch_shapeious = compute_overall_iou(seg_pred, labels, n_classes)
#             shape_ious += batch_shapeious

#             pred_choice = seg_pred.data.max(2)[1]
#             for cls in range(n_classes):
#                 gt_mask   = (labels == cls)
#                 pred_mask = (pred_choice == cls)
#                 intersection = (gt_mask & pred_mask).sum().item()
#                 union        = (gt_mask | pred_mask).sum().item()
#                 if union > 0:
#                     per_class_iou[cls]  += intersection / union
#                     per_class_seen[cls] += 1

#             pred_flat = seg_pred.view(-1, n_classes).data.max(1)[1]
#             correct = pred_flat.eq(labels.view(-1)).cpu().sum()
#             accuracy.append(correct.item() / (batch_size * num_point))

#     for cls in range(n_classes):
#         if per_class_seen[cls] > 0:
#             per_class_iou[cls] /= per_class_seen[cls]
#         io.cprint('%s iou: %.5f' % (labels_classes[cls], per_class_iou[cls]))

#     io.cprint('Test acc: %.5f  class mIoU: %.5f  instance mIoU: %.5f' % (
#         np.mean(accuracy), np.mean(per_class_iou), np.mean(shape_ious)))


if __name__ == "__main__":
    main()