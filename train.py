"""
Training script for the KLA image restoration model.

Usage:
    Smoke test first (confirm everything runs, ~2-5 min):
        python train.py --epochs 2 --batch_size 4

    Real training run:
        python train.py --epochs 50 --batch_size 8
"""

import argparse
import csv
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import PairedRestorationDataset, get_train_val_filenames
from models.unet import RestorationUNet


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

    # --- Data ---
    train_files, val_files = get_train_val_filenames(args.gt_dir, val_fraction=args.val_fraction, seed=args.seed)
    print(f"Train pairs: {len(train_files)}  |  Val pairs: {len(val_files)}")

    train_ds = PairedRestorationDataset(args.gt_dir, args.degraded_dir, train_files)
    val_ds = PairedRestorationDataset(args.gt_dir, args.degraded_dir, val_files)

    # num_workers=0 by default: avoids Windows multiprocessing issues.
    # Safe to raise later if data loading becomes the bottleneck.
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # --- Model ---
    model = RestorationUNet(base_ch=args.base_ch).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.L1Loss()

    # Mixed precision: roughly halves VRAM usage, especially important on
    # the 4GB RTX 3050. Free speed on the RTX 4050 too.
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # --- Logging setup ---
    log_path = os.path.join(args.checkpoint_dir, "training_log.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "time_seconds"])

    best_val_loss = float("inf")

    # --- Training loop ---
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        model.train()
        train_loss_total = 0.0
        for degraded, gt in train_loader:
            degraded, gt = degraded.to(device), gt.to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                pred = model(degraded)
                loss = criterion(pred, gt)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss_total += loss.item() * degraded.size(0)

        train_loss = train_loss_total / len(train_ds)

        # --- Validation ---
        model.eval()
        val_loss_total = 0.0
        with torch.no_grad():
            for degraded, gt in val_loader:
                degraded, gt = degraded.to(device), gt.to(device)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    pred = model(degraded)
                    loss = criterion(pred, gt)
                val_loss_total += loss.item() * degraded.size(0)

        val_loss = val_loss_total / len(val_ds)
        epoch_time = time.time() - epoch_start

        print(f"Epoch {epoch}/{args.epochs} | train_loss: {train_loss:.4f} | "
              f"val_loss: {val_loss:.4f} | time: {epoch_time:.1f}s")

        # Log every epoch (training hygiene: full record of the run)
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{train_loss:.6f}", f"{val_loss:.6f}", f"{epoch_time:.1f}"])

        # Save checkpoint every epoch (safety net against interruptions)
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "base_ch": args.base_ch,
        }
        torch.save(checkpoint, os.path.join(args.checkpoint_dir, "last_checkpoint.pt"))

        # Also save best-so-far separately
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
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--base_ch", type=int, default=32, help="Use 16 for RTX 3050 (4GB VRAM)")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train(args)
