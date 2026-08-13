"""
Improved training script (v2) -- targets the blur issue and specific metric
goals by adding SSIM + perceptual loss terms to the original L1 loss, plus
data augmentation and a learning rate schedule.

Saves to a SEPARATE checkpoint folder (checkpoints_v2) so your existing v1
results (checkpoints/best_model.pt) are preserved for comparison.

Usage:
    Smoke test first (~2-5 min, confirms no crashes/OOM with the new loss):
        python train_v2.py --epochs 2 --batch_size 8

    Full run (safe to leave overnight):
        python train_v2.py --epochs 300 --batch_size 16

    If interrupted, resume from where it left off:
        python train_v2.py --epochs 300 --batch_size 16 --resume

Note: first run downloads pretrained VGG16 weights (~500MB) -- needs
internet access for that one moment, then it's cached locally.
"""

import argparse
import csv
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import PairedRestorationDataset, get_train_val_filenames
from models.unet import RestorationUNet
from losses import CombinedLoss


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(args):
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    train_files, val_files = get_train_val_filenames(args.gt_dir, val_fraction=args.val_fraction, seed=args.seed)
    print(f"Train pairs: {len(train_files)}  |  Val pairs: {len(val_files)}")

    # augment=True only for training data -- validation must stay unmodified
    train_ds = PairedRestorationDataset(args.gt_dir, args.degraded_dir, train_files, augment=True)
    val_ds = PairedRestorationDataset(args.gt_dir, args.degraded_dir, val_files, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = RestorationUNet(base_ch=args.base_ch).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print("Setting up combined loss (downloads pretrained VGG16 weights on first run)...")
    criterion = CombinedLoss(device, w_l1=args.w_l1, w_ssim=args.w_ssim, w_perceptual=args.w_perceptual)

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch = 1
    best_val_loss = float("inf")

    resume_path = os.path.join(args.checkpoint_dir, "last_checkpoint.pt")
    if args.resume and os.path.exists(resume_path):
        print(f"Resuming from {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint.get("val_loss", float("inf"))
        print(f"Resuming at epoch {start_epoch}, best_val_loss so far: {best_val_loss:.4f}")

    log_path = os.path.join(args.checkpoint_dir, "training_log.csv")
    if not os.path.exists(log_path):
        with open(log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "train_l1", "train_ssim_loss", "train_perceptual",
                              "val_loss", "lr", "time_seconds"])

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()

        model.train()
        train_loss_total = 0.0
        comp_totals = {"l1": 0.0, "ssim_loss": 0.0, "perceptual": 0.0}
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [train]", leave=False)
        for degraded, gt in train_bar:
            degraded, gt = degraded.to(device), gt.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                pred = model(degraded)
                loss, components = criterion(pred, gt)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            bsz = degraded.size(0)
            train_loss_total += loss.item() * bsz
            for k in comp_totals:
                comp_totals[k] += components[k] * bsz
            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = train_loss_total / len(train_ds)
        for k in comp_totals:
            comp_totals[k] /= len(train_ds)

        model.eval()
        val_loss_total = 0.0
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [val]", leave=False)
        with torch.no_grad():
            for degraded, gt in val_bar:
                degraded, gt = degraded.to(device), gt.to(device)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    pred = model(degraded)
                    loss, _ = criterion(pred, gt)
                val_loss_total += loss.item() * degraded.size(0)
                val_bar.set_postfix(loss=f"{loss.item():.4f}")

        val_loss = val_loss_total / len(val_ds)
        scheduler.step()
        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch}/{args.epochs} | train: {train_loss:.4f} "
              f"(l1={comp_totals['l1']:.4f} ssim={comp_totals['ssim_loss']:.4f} perc={comp_totals['perceptual']:.4f}) "
              f"| val: {val_loss:.4f} | lr: {current_lr:.6f} | time: {epoch_time:.1f}s")

        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{train_loss:.6f}", f"{comp_totals['l1']:.6f}",
                              f"{comp_totals['ssim_loss']:.6f}", f"{comp_totals['perceptual']:.6f}",
                              f"{val_loss:.6f}", f"{current_lr:.6f}", f"{epoch_time:.1f}"])

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_loss": val_loss,
            "base_ch": args.base_ch,
        }
        torch.save(checkpoint, os.path.join(args.checkpoint_dir, "last_checkpoint.pt"))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, os.path.join(args.checkpoint_dir, "best_model.pt"))
            print(f"  -> New best model saved (val_loss: {val_loss:.4f})")

    print("Training complete.")
    print(f"Best val_loss: {best_val_loss:.4f}")
    print(f"Best checkpoint: {os.path.join(args.checkpoint_dir, 'best_model.pt')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_dir", type=str, default="data/gt")
    parser.add_argument("--degraded_dir", type=str, default="data/degraded")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_v2")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--base_ch", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--w_l1", type=float, default=1.0, help="Weight for L1 pixel loss")
    parser.add_argument("--w_ssim", type=float, default=1.0, help="Weight for SSIM loss (targets SSIM metric)")
    parser.add_argument("--w_perceptual", type=float, default=0.1, help="Weight for VGG perceptual loss (targets LPIPS metric)")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    train(args)
