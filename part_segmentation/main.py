from __future__ import print_function
from ast import arg
import os
import argparse
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from util.dataset import ARKitScenesDataset
import torch.nn.functional as F
import torch.nn as nn
import model as models
import numpy as np
from torch.utils.data import DataLoader
from util.util import compute_overall_iou, IOStream
from tqdm import tqdm
from collections import defaultdict
from torch.autograd import Variable
import random
from pathlib import Path
import datetime


n_classes = 2
labels_classes = ['no_object', 'object']
ARKITSCENES_PATH = Path("/home/danish/lobster/ml_data/ARKitScenes/arkitscenes_small.h5")


def parse_args():
    parser = argparse.ArgumentParser(description='ARKitScenes Scene Segmentation')
    parser.add_argument('--model',              type=str,   default='pointMLP')
    parser.add_argument('--exp_name',           type=str,   default=None)

    parser.add_argument('--batch_size',         type=int,   default=48)
    parser.add_argument('--test_batch_size',    type=int,   default=32)
    parser.add_argument('--epochs',             type=int,   default=350)

    parser.add_argument('--num_points',         type=int,   default=1024)
    parser.add_argument('--block_size',         type=int,   default=2.0)
    parser.add_argument('--stride',             type=int,   default=1.0)
    parser.add_argument('--min_points',         type=int,   default=256)

    parser.add_argument('--use_sgd',            type=bool,  default=False)
    parser.add_argument('--scheduler',          type=str,   default='step')
    parser.add_argument('--step',               type=int,   default=40)
    parser.add_argument('--lr',                 type=float, default=0.003)
    parser.add_argument('--momentum',           type=float, default=0.9)
    parser.add_argument('--manual_seed',        type=int,   default=None)
    parser.add_argument('--eval',               type=bool,  default=False)
    parser.add_argument('--workers',            type=int,   default=12)
    parser.add_argument('--resume',             type=bool,  default=False)
    parser.add_argument('--model_type',         type=str,   default='insiou')
    return parser.parse_args()

def main():    
    args = parse_args()

    assert torch.cuda.is_available(), "Please ensure codes are executed in cuda."

    if args.exp_name is None:
        args.exp_name = args.model + "_" + f"{datetime.datetime.now():%Y-%m-%d_%H-%M}"
        
    _init_(args=args)

    log_name = 'checkpoints/%s/%s_%s.log' % (args.exp_name, args.model, 'test' if args.eval else 'train')
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
        test(args, io)


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

    model = models.__dict__[args.model](n_classes, args.num_points).to(device)
    model.apply(weight_init)


    scaler = torch.amp.GradScaler("cuda") #UNUSED

    io.cprint(str(model))

    if args.resume:
        state_dict = torch.load(f"checkpoints/{args.exp_name}/best_insiou_model.pth", weights_only=False, map_location='cpu')['model']
        state_dict = {
            k.replace("module.", "", 1).replace("_orig_mod.", "", 1): v
            for k, v in state_dict.items()
        }
        # for k in state_dict.keys():
        #     if 'module' not in k:
        #         from collections import OrderedDict
        #         new_state_dict = OrderedDict()
        #         for k in state_dict:
        #             new_state_dict['module.' + k] = state_dict[k]
        #         state_dict = new_state_dict
        #     break
        model.load_state_dict(state_dict)
        print("Resuming training...")
    else:
        print("Training from scratch...")

    # model = torch.compile(model)
    
    train_data = ARKitScenesDataset(ARKITSCENES_PATH, split='train', 
                                    num_points=args.num_points,
                                    block_size=args.block_size, 
                                    stride=args.stride, 
                                    min_points=args.min_points)
    
    val_data   = ARKitScenesDataset(ARKITSCENES_PATH, split='val', 
                                    num_points=args.num_points,
                                    block_size=args.block_size, 
                                    stride=args.stride, 
                                    min_points=args.min_points)

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

    for epoch in range(args.epochs):
        train_epoch(args, scaler, train_loader, model, opt, scheduler, epoch, io)
        test_metrics, per_class_iou = test_epoch(val_loader, model, epoch, io)

        if test_metrics['accuracy'] > best_acc:
            best_acc = test_metrics['accuracy']
            io.cprint('Max Acc: %.5f' % best_acc)
            torch.save({'model': model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict(),
                        'optimizer': opt.state_dict(), 'epoch': epoch, 'test_acc': best_acc},
                       'checkpoints/%s/best_acc_model.pth' % args.exp_name)

        if test_metrics['shape_avg_iou'] > best_instance_iou:
            best_instance_iou = test_metrics['shape_avg_iou']
            io.cprint('Max instance iou: %.5f' % best_instance_iou)
            torch.save({'model': model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict(),
                        'optimizer': opt.state_dict(), 'epoch': epoch, 'test_instance_iou': best_instance_iou},
                       'checkpoints/%s/best_insiou_model.pth' % args.exp_name)

        avg_class_iou = np.mean(per_class_iou)
        if avg_class_iou > best_class_iou:
            best_class_iou = avg_class_iou
            for i in range(n_classes):
                io.cprint('%s iou: %.5f' % (labels_classes[i], per_class_iou[i]))
            io.cprint('Max class iou: %.5f' % best_class_iou)
            torch.save({'model': model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict(),
                        'optimizer': opt.state_dict(), 'epoch': epoch, 'test_class_iou': best_class_iou},
                       'checkpoints/%s/best_clsiou_model.pth' % args.exp_name)

    io.cprint('Final Max Acc: %.5f' % best_acc)
    io.cprint('Final Max instance iou: %.5f' % best_instance_iou)
    io.cprint('Final Max class iou: %.5f' % best_class_iou)
    torch.save({'model': model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict(),
                'optimizer': opt.state_dict(), 'epoch': args.epochs - 1, 'test_iou': best_instance_iou},
               'checkpoints/%s/model_ep%d.pth' % (args.exp_name, args.epochs))


def train_epoch(args, scaler, train_loader, model, opt, scheduler, epoch, io):
    train_loss = 0.0
    count = 0.0
    accuracy = []
    shape_ious = 0.0
    model.train()

    log_counter = 0

    for points, labels in tqdm(train_loader, total=len(train_loader), smoothing=0.9):
        # points: (B, 3, N) — already in correct format from dataset
        # labels: (B, N)
        batch_size, _, num_point = points.size()

        points = points.float().cuda(non_blocking=True)   # (B, 3, N)
        labels = labels.long().cuda(non_blocking=True)    # (B, N)

        opt.zero_grad(set_to_none=True)

        # with torch.amp.autocast("cuda"):

        seg_pred = model(points)     
        seg_pred_flat = seg_pred.contiguous().view(-1, n_classes)        

        loss = F.nll_loss(seg_pred_flat, labels.view(-1))
        loss = torch.mean(loss)

        loss.backward()
        opt.step()

        # scaler.scale(loss).backward()
        # scaler.step(opt)
        # scaler.update()

        pred_choice = seg_pred_flat.data.max(1)[1]          # (B*N,)
        correct = pred_choice.eq(labels.view(-1)).sum()

        count += batch_size

        # if log_counter % 100 == 0:
        batch_shapeious = compute_overall_iou(seg_pred, labels, n_classes)
        batch_shapeious = seg_pred.new_tensor([np.sum(batch_shapeious)], dtype=torch.float64)

        shape_ious += batch_shapeious.item()
        train_loss += loss.item() * batch_size
        accuracy.append(correct.item() / (batch_size * num_point))

        # log_counter += 1

    if args.scheduler == 'cos':
        scheduler.step()
    elif args.scheduler == 'step':
        if opt.param_groups[0]['lr'] > 0.9e-5:
            scheduler.step()
        if opt.param_groups[0]['lr'] < 0.9e-5:
            for pg in opt.param_groups:
                pg['lr'] = 0.9e-5

    io.cprint('Train %d, loss: %.5f, acc: %.5f, ins_iou: %.5f, lr: %f' % (
        epoch + 1, train_loss / count, np.mean(accuracy), shape_ious / count,
        opt.param_groups[0]['lr']))


def test_epoch(val_loader, model, epoch, io):
    test_loss = 0.0
    count = 0.0
    accuracy = []
    shape_ious = 0.0
    per_class_iou  = np.zeros(n_classes, dtype=np.float32)
    per_class_seen = np.zeros(n_classes, dtype=np.int32)
    metrics = defaultdict(lambda: list())
    model.eval()

    with torch.no_grad():
        for points, labels in tqdm(val_loader, total=len(val_loader), smoothing=0.9):
            batch_size, _, num_point = points.size()

            points = points.float().cuda(non_blocking=True)   # (B, 3, N)
            labels = labels.long().cuda(non_blocking=True)    # (B, N)

            # with torch.amp.autocast("cuda"):
            seg_pred = model(points)
            batch_shapeious = compute_overall_iou(seg_pred, labels, n_classes)

            # per-class iou
            pred_choice = seg_pred.data.max(2)[1]             # (B, N)
            for cls in range(n_classes):
                gt_mask   = (labels == cls)
                pred_mask = (pred_choice == cls)
                intersection = (gt_mask & pred_mask).sum().item()
                union        = (gt_mask | pred_mask).sum().item()
                if union > 0:
                    per_class_iou[cls]  += intersection / union
                    per_class_seen[cls] += 1

            batch_ious = seg_pred.new_tensor([np.sum(batch_shapeious)], dtype=torch.float64)
            seg_pred = seg_pred.contiguous().view(-1, n_classes)
            loss = F.nll_loss(seg_pred.contiguous().view(-1, n_classes), labels.view(-1))

            pred_choice = seg_pred.data.max(1)[1]  # b*n
            correct = pred_choice.eq(labels.data.view(-1)).sum()

            loss = torch.mean(loss)
            shape_ious  += batch_ious.item()
            count       += batch_size
            test_loss   += loss.item() * batch_size
            accuracy.append(correct.item() / (batch_size * num_point))

    for cls in range(n_classes):
        if per_class_seen[cls] > 0:
            per_class_iou[cls] /= per_class_seen[cls]

    metrics = {
        'accuracy':      np.mean(accuracy),
        'shape_avg_iou': shape_ious / count,
    }
    io.cprint('Test %d, loss: %.5f, acc: %.5f, ins_iou: %.5f' % (
        epoch + 1, test_loss / count, metrics['accuracy'], metrics['shape_avg_iou']))

    return metrics, per_class_iou


def test(args, io):
    val_data = ARKitScenesDataset(ARKITSCENES_PATH, split='val', num_points=args.num_points,
                                    block_size=1.0, stride=1.0, min_points=512)
    val_loader = DataLoader(val_data, batch_size=args.test_batch_size, shuffle=False,
                            num_workers=args.workers, drop_last=False)

    device = torch.device("cuda")
    model = models.__dict__[args.model](n_classes).to(device)

    # model = torch.compile(model)

    from collections import OrderedDict
    state_dict = torch.load("checkpoints/%s/best_%s_model.pth" % (args.exp_name, args.model_type),
                            map_location='cpu')['model']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_state_dict[k.replace('module.', '')] = v
    model.load_state_dict(new_state_dict)
    model.eval()

    accuracy = []
    shape_ious = []
    per_class_iou  = np.zeros(n_classes, dtype=np.float32)
    per_class_seen = np.zeros(n_classes, dtype=np.int32)

    with torch.no_grad():
        for points, labels in tqdm(val_loader, total=len(val_loader), smoothing=0.9):
            batch_size, _, num_point = points.size()
            points = points.float().cuda(non_blocking=True)
            labels = labels.long().cuda(non_blocking=True)

            seg_pred = model(points)
            batch_shapeious = compute_overall_iou(seg_pred, labels, n_classes)
            shape_ious += batch_shapeious

            pred_choice = seg_pred.data.max(2)[1]
            for cls in range(n_classes):
                gt_mask   = (labels == cls)
                pred_mask = (pred_choice == cls)
                intersection = (gt_mask & pred_mask).sum().item()
                union        = (gt_mask | pred_mask).sum().item()
                if union > 0:
                    per_class_iou[cls]  += intersection / union
                    per_class_seen[cls] += 1

            pred_flat = seg_pred.view(-1, n_classes).data.max(1)[1]
            correct = pred_flat.eq(labels.view(-1)).cpu().sum()
            accuracy.append(correct.item() / (batch_size * num_point))

    for cls in range(n_classes):
        if per_class_seen[cls] > 0:
            per_class_iou[cls] /= per_class_seen[cls]
        io.cprint('%s iou: %.5f' % (labels_classes[cls], per_class_iou[cls]))

    io.cprint('Test acc: %.5f  class mIoU: %.5f  instance mIoU: %.5f' % (
        np.mean(accuracy), np.mean(per_class_iou), np.mean(shape_ious)))


if __name__ == "__main__":
    main()