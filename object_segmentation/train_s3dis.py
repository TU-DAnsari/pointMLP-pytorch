from __future__ import print_function
import os
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from util.s3dis_dataset import S3DISDataset
import torch.nn.functional as F
import models 
import numpy as np
from torch.utils.data import DataLoader
from util.util import compute_overall_iou, IOStream
from tqdm import tqdm
from collections import defaultdict
import random
from pathlib import Path
import datetime
from util.util import parse_args, compute_class_weights
from util.util import compute_overall_iou
from util.neas import search_angles
from util.rotation import rotate_points
from util.feature_consistency import feature_consistency_loss
import shutil
import matplotlib
matplotlib.use('Agg')   # no display needed on a training server

from util.progress_plots import save_plots


"""
0  ceiling      
1  floor       
2  wall        
3  beam
4  column       
5  window      
6  door        
7  table
8  chair        
9  sofa       
10  bookcase    
11  board
12 clutter
"""

label_remap = {
    0: 0,
    1: 1,
    2: 2,
    3: 2,
    4: 2,
    5: 2,
    6: 2,
    7: 3,
    8: 3,
    9: 3,
    10: 3,
    11: 3,
    12: 3,
}

if label_remap:
    n_classes = len(set(label_remap.values()))
    labels_classes = ["ceiling", "floor", "structural", "objects"]
else:
    n_classes = 13
    labels_classes = ["ceiling",
                    "floor", 
                    "wall", 
                    "beam", 
                    "column", 
                    "window", 
                    "door", 
                    "table", 
                    "chair", 
                    "sofa", 
                    "bookcase", 
                    "board",
                    "clutter"]

# train_paths = [Path("/home/danish/lobster/ml/data/s3dis/Area_1.h5"),
#                Path("/home/danish/lobster/ml/data/s3dis/Area_3.h5"),
#                Path("/home/danish/lobster/ml/data/s3dis/Area_6.h5")]

# val_paths = [Path("/home/danish/lobster/ml/data/s3dis/Area_2.h5"),
#                Path("/home/danish/lobster/ml/data/s3dis/Area_4.h5")]

# test_paths = [Path("/home/danish/lobster/ml/data/s3dis/Area_5.h5")]

train_paths = [Path("/home/danish/lobster/ml/data/s3dis/Area_1.h5")]
val_paths = [Path("/home/danish/lobster/ml/data/s3dis/Area_2.h5")]



def _empty_history():
    return {
        "train_loss": [],
        "train_acc": [],
        "train_ins_iou": [],
        "val_loss": [],
        "val_acc": [],
        "val_ins_iou": [],
        "val_cls_iou": [],
        "val_per_class_iou": [],   # list of np.ndarray(n_classes,)
    }


def main():    
    args = parse_args()
    assert torch.cuda.is_available(), "Please ensure codes are executed in cuda."

    if args.exp_name is None:
        args.exp_name = args.model + "_" + f"{datetime.datetime.now():%Y-%m-%d_%H-%M}"
        
    checkpoint_dir = 'checkpoints/segmentation/%s' % args.exp_name
    config_save_path = os.path.join(checkpoint_dir, 'config.yaml')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    if not args.eval:
        shutil.copy(args.config, config_save_path)
        with open(config_save_path, 'a') as f:
            f.write(f"\nDATA_PATH: {train_paths}\n")

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
    checkpoint_dir = 'checkpoints/segmentation/%s' % args.exp_name

    train_data = S3DISDataset(train_paths,
                              num_points=args.num_points,
                              min_points=args.min_points,
                              block_size=args.block_size,
                              stride=args.stride,
                              normalize=args.normalize,
                              label_remap=label_remap,
                              )

    val_data = S3DISDataset(val_paths,
                            num_points=args.num_points,
                            min_points=args.min_points,
                            block_size=args.block_size,
                            stride=args.stride,
                            normalize=args.normalize,
                            label_remap=label_remap,
                            )

    print("Training samples: %d" % len(train_data))
    print("Validation samples: %d" % len(val_data))

    train_loader = DataLoader(train_data, 
                              batch_size=args.batch_size, 
                              shuffle=True,
                              num_workers=args.workers, 
                              drop_last=False, 
                              pin_memory=True, 
                              persistent_workers=True)
    
    val_loader   = DataLoader(val_data, 
                              batch_size=args.test_batch_size, 
                              shuffle=False,
                              num_workers=args.workers, 
                              drop_last=False,
                              pin_memory=True, 
                              persistent_workers=True)
    
    model = models.__dict__[args.model](n_classes, args.num_points, 3).to(device)
    model.apply(weight_init)

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
    best_class_iou = 0
    best_instance_iou = 0

    label_array = train_data.label_blocks.reshape(-1)
    class_weights = compute_class_weights(label_array.reshape(-1), n_classes, device)

    history = _empty_history()

    for epoch in range(args.epochs):
        train_metrics = train_epoch(args, train_loader, class_weights, model, opt, scheduler, epoch, io, ntk_params=None)
        test_metrics, per_class_iou = test_epoch(args, val_loader, model, class_weights, epoch, io)

        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["acc"])
        history["train_ins_iou"].append(train_metrics["ins_iou"])
        history["val_loss"].append(test_metrics["loss"])
        history["val_acc"].append(test_metrics["accuracy"])
        history["val_ins_iou"].append(test_metrics["avg_iou"])
        history["val_cls_iou"].append(float(np.mean(per_class_iou)))
        history["val_per_class_iou"].append(per_class_iou.copy())

        save_plots(history, checkpoint_dir, labels_classes)

        if test_metrics['accuracy'] > best_acc:
            best_acc = test_metrics['accuracy']
            io.cprint('Max Acc: %.5f' % best_acc)
            torch.save({'model': model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict(),
                        'optimizer': opt.state_dict(), 'epoch': epoch, 'test_acc': best_acc},
                        f'{checkpoint_dir}/best_acc_model.pth')

        if test_metrics['avg_iou'] > best_instance_iou:
            best_instance_iou = test_metrics['avg_iou']
            io.cprint('Max instance iou: %.5f' % best_instance_iou)
            torch.save({'model': model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict(),
                        'optimizer': opt.state_dict(), 'epoch': epoch, 'test_instance_iou': best_instance_iou},
                        f'{checkpoint_dir}/best_insiou_model.pth')

        avg_class_iou = np.mean(per_class_iou)
        if avg_class_iou > best_class_iou:
            best_class_iou = avg_class_iou
            for i in range(n_classes):
                io.cprint('%s iou: %.5f' % (labels_classes[i], per_class_iou[i]))
            io.cprint('Max class iou: %.5f' % best_class_iou)
            torch.save({'model': model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict(),
                        'optimizer': opt.state_dict(), 'epoch': epoch, 'test_class_iou': best_class_iou},
                        f'{checkpoint_dir}/best_clsiou_model.pth')

    io.cprint('Final Max Acc: %.5f' % best_acc)
    io.cprint('Final Max instance iou: %.5f' % best_instance_iou)
    io.cprint('Final Max class iou: %.5f' % best_class_iou)
    torch.save({'model': model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict(),
                'optimizer': opt.state_dict(), 'epoch': args.epochs - 1, 'test_iou': best_instance_iou},
                f'{checkpoint_dir}/model_ep{args.epochs}.pth')


def train_epoch(args, train_loader, class_weights, model, opt, scheduler, epoch, io, ntk_params):
    train_loss = 0.0
    count = 0.0
    accuracy = []
    shape_ious = 0.0
    model.train()

    neas_on   = getattr(args, "neas_enable", False)
    max_iter  = getattr(args, "neas_max_iter", 5)
    axes      = tuple(getattr(args, "neas_axes", ("x", "y", "z")))
    feat_w    = float(getattr(args, "neas_feat_weight", 0.0))


    for points_batch, _, labels_batch in tqdm(train_loader, total=len(train_loader), smoothing=0.9):
        batch_size, num_point, _ = points_batch.size()

        points_batch = points_batch.float().permute(0, 2, 1).cuda(non_blocking=True)
        # features_batch = features_batch.float().permute(0, 2, 1).cuda(non_blocking=True)
        labels_batch = labels_batch.long().cuda(non_blocking=True)
    
        if neas_on:
            # search dirties .grad and runs several fwd/bwd internally;
            # it calls model.zero_grad() at the end.
            (ax, ay, az), _chosen = search_angles(
                model, points_batch, points_batch,
                axes=axes, max_iter=max_iter, ntk_params=ntk_params,
            )
            rot_points = rotate_points(points_batch, ax, ay, az).detach()
        else:
            rot_points = None

        opt.zero_grad(set_to_none=True)
        
        seg_pred = model(points_batch, points_batch)           
        seg_pred_flat = seg_pred.contiguous().view(-1, n_classes)        

        loss = F.nll_loss(seg_pred_flat, labels_batch.view(-1), class_weights)

        if rot_points is not None:
            seg_pred_rot = model(rot_points, rot_points)
            seg_flat_rot = seg_pred_rot.contiguous().view(-1, n_classes)
            loss = loss + F.nll_loss(seg_flat_rot, labels_batch.view(-1), class_weights)
 
            # -------- (d) optional feature-consistency term --------
            if feat_w > 0.0:
                loss = loss + feat_w * feature_consistency_loss(
                    model, points_batch, points_batch, ax, ay, az
                )

        loss = torch.mean(loss)
        loss.backward()
        opt.step()

        pred_choice = seg_pred_flat.data.max(1)[1]
        correct = pred_choice.eq(labels_batch.view(-1)).sum()

        count += batch_size

        batch_shapeious = compute_overall_iou(seg_pred, labels_batch, n_classes)
        batch_shapeious = seg_pred.new_tensor([np.sum(batch_shapeious)], dtype=torch.float64)

        shape_ious += batch_shapeious.item()
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

    epoch_loss = train_loss / count
    epoch_acc  = np.mean(accuracy)
    epoch_iou  = shape_ious / count

    io.cprint('Train %d, loss: %.5f, acc: %.5f, ins_iou: %.5f, lr: %f' % (
        epoch + 1, epoch_loss, epoch_acc, epoch_iou,
        opt.param_groups[0]['lr']))

    return {"loss": epoch_loss, "acc": epoch_acc, "ins_iou": epoch_iou}


def test_epoch(args, val_loader, model, class_weights, epoch, io):
    test_loss = 0.0
    count = 0.0
    accuracy = []
    shape_ious = 0.0
    per_class_iou  = np.zeros(n_classes, dtype=np.float32)
    per_class_seen = np.zeros(n_classes, dtype=np.int32)
    metrics = defaultdict(lambda: list())
    model.eval()

    with torch.no_grad():
        for points_batch, _, labels_batch in tqdm(val_loader, total=len(val_loader), smoothing=0.9):
            batch_size, num_point, _ = points_batch.size()

            points_batch = points_batch.float().permute(0, 2, 1).cuda(non_blocking=True)
            # features_batch = features_batch.float().permute(0, 2, 1).cuda(non_blocking=True)
            labels_batch = labels_batch.long().cuda(non_blocking=True)
            
            seg_pred = model(points_batch, points_batch)         
            batch_shapeious = compute_overall_iou(seg_pred, labels_batch, n_classes)

            pred_choice = seg_pred.data.max(2)[1]
            for cls in range(n_classes):
                gt_mask   = (labels_batch == cls)
                pred_mask = (pred_choice == cls)
                intersection = (gt_mask & pred_mask).sum().item()
                union        = (gt_mask | pred_mask).sum().item()
                if union > 0:
                    per_class_iou[cls]  += intersection / union
                    per_class_seen[cls] += 1

            batch_ious = seg_pred.new_tensor([np.sum(batch_shapeious)], dtype=torch.float64)
            seg_pred = seg_pred.contiguous().view(-1, n_classes)
            loss = F.nll_loss(seg_pred.contiguous().view(-1, n_classes), labels_batch.view(-1), class_weights)

            pred_choice = seg_pred.data.max(1)[1]
            correct = pred_choice.eq(labels_batch.data.view(-1)).sum()

            loss = torch.mean(loss)
            shape_ious  += batch_ious.item()
            count       += batch_size
            test_loss   += loss.item() * batch_size
            accuracy.append(correct.item() / (batch_size * num_point))

    for cls in range(n_classes):
        if per_class_seen[cls] > 0:
            per_class_iou[cls] /= per_class_seen[cls]

    metrics = {
        'loss':     test_loss / count,
        'accuracy': np.mean(accuracy),
        'avg_iou':  shape_ious / count,
    }

    io.cprint(
        f"Test {epoch+1}, loss: {metrics['loss']:.5f}, acc: {metrics['accuracy']:.5f}, ins_iou: {metrics['avg_iou']:.5f}, " +
        ", ".join(f"{labels_classes[i]} iou: {per_class_iou[i]:.5f}" for i in range(n_classes))
    )

    return metrics, per_class_iou


# def test(args, io):
#     test_data = S3DISDataset(test_paths,
#                               num_points=args.num_points,
#                               min_points=args.min_points,
#                               block_size=args.block_size,
#                               stride=args.stride,
#                               normalize=args.normalize,
#                               label_remap=label_remap,
#                               )
    
#     test_loader = DataLoader(test_data, 
#                               batch_size=args.test_batch_size, 
#                               shuffle=False,
#                               num_workers=args.workers, 
#                               drop_last=False,
#                               pin_memory=True, 
#                               persistent_workers=True)

#     device = torch.device("cuda")
#     model = models.__dict__[args.model](n_classes).to(device)

#     from collections import OrderedDict
#     state_dict = torch.load("checkpoints/%s/best_%s_model.pth" % (args.exp_name, args.model_type),
#                             map_location='cpu')['model']
#     new_state_dict = OrderedDict()
#     for k, v in state_dict.items():
#         new_state_dict[k.replace('module.', '')] = v
#     model.load_state_dict(new_state_dict)
#     model.eval()

#     accuracy = []
#     shape_ious = []
#     per_class_iou  = np.zeros(n_classes, dtype=np.float32)
#     per_class_seen = np.zeros(n_classes, dtype=np.int32)

#     with torch.no_grad():
#         for points_batch, features_batch, labels_batch in tqdm(test_loader, total=len(test_loader), smoothing=0.9):
#             batch_size, num_point, _ = points_batch.size()

#             points_batch = points_batch.float().permute(0, 2, 1).cuda(non_blocking=True)
#             features_batch = features_batch.float().permute(0, 2, 1).cuda(non_blocking=True)
#             labels_batch = labels_batch.long().cuda(non_blocking=True)
            
#             seg_pred = model(points_batch, features_batch)         
#             batch_shapeious = compute_overall_iou(seg_pred, labels_batch, n_classes)
#             shape_ious += batch_shapeious

#             pred_choice = seg_pred.data.max(2)[1]
#             for cls in range(n_classes):
#                 gt_mask   = (labels_batch == cls)
#                 pred_mask = (pred_choice == cls)
#                 intersection = (gt_mask & pred_mask).sum().item()
#                 union        = (gt_mask | pred_mask).sum().item()
#                 if union > 0:
#                     per_class_iou[cls]  += intersection / union
#                     per_class_seen[cls] += 1

#             pred_flat = seg_pred.view(-1, n_classes).data.max(1)[1]
#             correct = pred_flat.eq(labels_batch.view(-1)).cpu().sum()
#             accuracy.append(correct.item() / (batch_size * num_point))

#     for cls in range(n_classes):
#         if per_class_seen[cls] > 0:
#             per_class_iou[cls] /= per_class_seen[cls]
#         io.cprint('%s iou: %.5f' % (labels_classes[cls], per_class_iou[cls]))

#     io.cprint('Test acc: %.5f  class mIoU: %.5f  instance mIoU: %.5f' % (
#         np.mean(accuracy), np.mean(per_class_iou), np.mean(shape_ious)))


if __name__ == "__main__":
    main()