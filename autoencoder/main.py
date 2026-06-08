"""
train_autoencoder.py
--------------------
Training script for PointMLP denoising autoencoders.

Supports two modes controlled by --model:
  pointMLPSmallAutoEncoder   – deterministic AE
  pointMLPSmallVAE           – variational AE  (adds β·KL term to Chamfer loss)

Dataset contract
----------------
The dataset is expected to yield 5-tuples:
    idx, source_pc, target_pc, labels, meta

  source_pc : (N, C)  – noisy / partial input point cloud
  target_pc : (N, 3)  – clean target XYZ  (reconstruction target)
  labels    : unused here, kept for API compatibility
  meta      : unused here

Usage examples
--------------
# Train deterministic AE
python train_autoencoder.py --config cfg/ae.yaml --model pointMLPSmallAutoEncoder

# Train VAE
python train_autoencoder.py --config cfg/vae.yaml --model pointMLPSmallVAE --beta 1.0

# Resume
python train_autoencoder.py --config cfg/ae.yaml --model pointMLPSmallAutoEncoder --resume

# Eval only
python train_autoencoder.py --config cfg/ae.yaml --model pointMLPSmallAutoEncoder --eval
"""

from __future__ import print_function
import os
import argparse
import random
import datetime
import shutil
from pathlib import Path
from collections import defaultdict
import model as models

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from tqdm import tqdm

from util.lobrob_dataset import LobRobDataset
from util.util import IOStream, parse_args

# ---------------------------------------------------------------------------
# Paths / dataset
# ---------------------------------------------------------------------------
DATA_PATH     = Path("/home/danish/lobster/ml_data/lobrob/ae_data.h5")
DATASET_CLASS = LobRobDataset

# ---------------------------------------------------------------------------
# Chamfer Distance (pure PyTorch, no external deps)
# ---------------------------------------------------------------------------

def chamfer_distance(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Symmetric Chamfer Distance between two point clouds.

    Args:
        pred:   (B, N, 3)
        target: (B, M, 3)
    Returns:
        scalar mean Chamfer distance over the batch
    """
    # (B, N, M)
    diff = pred.unsqueeze(2) - target.unsqueeze(1)          # (B, N, M, 3)
    dist2 = (diff ** 2).sum(dim=-1)                          # (B, N, M)

    # pred → target: for each pred point, nearest target point
    d_pred_to_tgt = dist2.min(dim=2)[0].mean(dim=1)          # (B,)
    # target → pred: for each target point, nearest pred point
    d_tgt_to_pred = dist2.min(dim=1)[0].mean(dim=1)          # (B,)

    return (d_pred_to_tgt + d_tgt_to_pred).mean()


# ---------------------------------------------------------------------------
# KL divergence (closed-form, unit Gaussian prior)
# ---------------------------------------------------------------------------

def kl_divergence(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    """
    KL( q(z|x) || N(0,I) ) = -0.5 * sum(1 + log_var - mu² - exp(log_var))
    Returns the mean over the batch.
    """
    return -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def weight_init(m):
    if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d)):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias,   0)


def _init_(args):
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs(f'checkpoints/{args.exp_name}', exist_ok=True)


def is_vae(model) -> bool:
    """Returns True if model is a VAE (forward returns a 3-tuple)."""
    return hasattr(model, 'bottleneck')


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def build_loaders(args):
    """Build train / val DataLoaders from shared dataset args."""
    train_data = DATASET_CLASS(DATA_PATH, split="train", num_points=args.num_points)
    val_data   = DATASET_CLASS(DATA_PATH, split="val", num_points=args.num_points)

    train_loader = DataLoader(
        train_data,
        batch_size   = args.batch_size,
        shuffle      = True,
        num_workers  = args.workers,
        drop_last    = True,
        pin_memory   = True,
        persistent_workers = True,
    )
    val_loader = DataLoader(
        val_data,
        batch_size   = args.test_batch_size,
        shuffle      = False,
        num_workers  = args.workers,
        drop_last    = False,
        pin_memory   = True,
        persistent_workers = True,
    )

    return train_loader, val_loader, train_data


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    assert torch.cuda.is_available(), "CUDA required."

    if args.exp_name is None:
        args.exp_name = args.model + "_" + f"{datetime.datetime.now():%Y-%m-%d_%H-%M}"

    _init_(args)

    checkpoint_dir   = f'checkpoints/{args.exp_name}'
    config_save_path = os.path.join(checkpoint_dir, 'config.yaml')

    if not args.eval:
        shutil.copy(args.config, config_save_path)
        with open(config_save_path, 'a') as f:
            f.write(f"\nDATA_PATH: {DATA_PATH}\n")

    log_name = checkpoint_dir + f'/{args.model}_{"test" if args.eval else "train"}.log'
    io = IOStream(log_name)
    io.cprint(str(args))

    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)

    if not args.eval:
        train(args, io)
    else:
        test(args, io)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args, io):
    device = torch.device("cuda")

    train_loader, val_loader, train_data = build_loaders(args)
    print(f"Train: {len(train_loader.dataset)}  Val: {len(val_loader.dataset)}")

    # ---- build model -------------------------------------------------------
    # Factory functions: pointMLPSmallAutoEncoder(num_points)
    #                    pointMLPSmallVAE(num_points)
    model = models.__dict__[args.model](args.num_points).to(device)
    model.apply(weight_init)
    io.cprint(str(model))

    if args.resume:
        ckpt = torch.load(
            f"checkpoints/{args.exp_name}/best_model.pth",
            weights_only=False, map_location='cpu'
        )['model']
        ckpt = {k.replace("module.", "", 1).replace("_orig_mod.", "", 1): v
                for k, v in ckpt.items()}
        model.load_state_dict(ckpt)
        io.cprint("Resumed from checkpoint.")
    else:
        io.cprint("Training from scratch.")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    io.cprint(f"Trainable parameters: {trainable:,}")

    # ---- optimiser & scheduler --------------------------------------------
    if args.use_sgd:
        opt = optim.SGD(model.parameters(), lr=args.lr * 100,
                        momentum=args.momentum, weight_decay=0)
    else:
        opt = optim.Adam(model.parameters(), lr=args.lr,
                         betas=(0.9, 0.999), eps=1e-8, weight_decay=0)

    if args.scheduler == 'cos':
        scheduler = CosineAnnealingLR(
            opt, args.epochs,
            eta_min=args.lr if args.use_sgd else args.lr / 100
        )
    else:
        scheduler = StepLR(opt, step_size=args.step, gamma=0.5)

    # β for KL term (VAE only); fall back to 1.0 if not set in args
    beta = getattr(args, 'beta', 1.0)

    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        train_loss, train_cd, train_kl = train_epoch(
            args, train_loader, model, opt, scheduler, epoch, io, beta)

        val_loss, val_cd, val_kl = val_epoch(
            args, val_loader, model, epoch, io, beta)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            io.cprint(f"  → New best val loss: {best_val_loss:.6f} — saving checkpoint.")
            torch.save(
                {'model': model.state_dict(), 'optimizer': opt.state_dict(),
                 'epoch': epoch, 'val_loss': best_val_loss},
                f'checkpoints/{args.exp_name}/best_model.pth'
            )

    io.cprint(f"Training complete. Best val loss: {best_val_loss:.6f}")
    torch.save(
        {'model': model.state_dict(), 'optimizer': opt.state_dict(),
         'epoch': args.epochs - 1, 'val_loss': best_val_loss},
        f'checkpoints/{args.exp_name}/model_ep{args.epochs}.pth'
    )


def _forward_and_loss(model, sampling_input, model_input, target_xyz, beta):
    """
    Unified forward + loss for both AE and VAE.

    Returns:
        loss:    scalar total loss
        cd:      Chamfer distance (float)
        kl:      KL term (float, 0.0 for plain AE)
        recon:   (B, N, 3)
    """
    if is_vae(model):
        recon, mu, log_var = model(sampling_input, model_input)
        cd  = chamfer_distance(recon, target_xyz)
        kl  = kl_divergence(mu, log_var)
        loss = cd + beta * kl
    else:
        recon = model(sampling_input, model_input)
        cd    = chamfer_distance(recon, target_xyz)
        kl    = torch.tensor(0.0, device=cd.device)
        loss  = cd

    return loss, cd.item(), kl.item(), recon


def train_epoch(args, loader, model, opt, scheduler, epoch, io, beta):
    model.train()
    total_loss = total_cd = total_kl = 0.0
    count = 0

    for source_batch, target_batch in tqdm(loader, desc=f"Train {epoch+1}", smoothing=0.9):
        B = source_batch.size(0)

        # source_batch: (B, N, C)  — noisy input
        # target_batch: (B, N, C)  — clean target (we only use XYZ = first 3 dims)

        sampling_input = source_batch.float().permute(0, 2, 1).cuda(non_blocking=True)
        model_input = source_batch.float().permute(0, 2, 1).cuda(non_blocking=True)
        target = target_batch.float().cuda(non_blocking=True)  # (B, N, 3)

        opt.zero_grad(set_to_none=True)
        loss, cd, kl, _ = _forward_and_loss(model, sampling_input, model_input, target, beta)
        loss.backward()
        opt.step()

        total_loss += loss.item() * B
        total_cd   += cd          * B
        total_kl   += kl          * B
        count      += B

    # scheduler step
    if args.scheduler == 'cos':
        scheduler.step()
    elif args.scheduler == 'step':
        if opt.param_groups[0]['lr'] > 9e-6:
            scheduler.step()
        for pg in opt.param_groups:
            pg['lr'] = max(pg['lr'], 9e-6)

    avg_loss = total_loss / count
    avg_cd   = total_cd   / count
    avg_kl   = total_kl   / count
    io.cprint(
        f"Train {epoch+1:3d}  loss={avg_loss:.6f}  cd={avg_cd:.6f}  "
        f"kl={avg_kl:.6f}  lr={opt.param_groups[0]['lr']:.2e}"
    )
    return avg_loss, avg_cd, avg_kl


@torch.no_grad()
def val_epoch(args, loader, model, epoch, io, beta):
    model.eval()
    total_loss = total_cd = total_kl = 0.0
    count = 0

    for source_batch, target_batch in tqdm(loader, desc=f"Train {epoch+1}", smoothing=0.9):
        B = source_batch.size(0)

        # source_batch: (B, N, C)  — noisy input
        # target_batch: (B, N, C)  — clean target (we only use XYZ = first 3 dims)

        sampling_input = source_batch.float().permute(0, 2, 1).cuda(non_blocking=True)
        model_input = source_batch.float().permute(0, 2, 1).cuda(non_blocking=True)
        target = target_batch.float().cuda(non_blocking=True)  # (B, N, 3)

        loss, cd, kl, _ = _forward_and_loss(model, sampling_input, model_input, target, beta)

        total_loss += loss.item() * B
        total_cd   += cd          * B
        total_kl   += kl          * B
        count      += B

    avg_loss = total_loss / count
    avg_cd   = total_cd   / count
    avg_kl   = total_kl   / count
    io.cprint(
        f"Val   {epoch+1:3d}  loss={avg_loss:.6f}  cd={avg_cd:.6f}  "
        f"kl={avg_kl:.6f}"
    )
    return avg_loss, avg_cd, avg_kl


if __name__ == "__main__":
    main()