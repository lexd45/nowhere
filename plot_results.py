"""
Generates the graphs and visual comparisons for your PPT results slide.

Prerequisites (run in this order):
  1. train.py has finished -> creates checkpoints/training_log.csv
  2. evaluate.py run with --val_only --gt_dir data/gt -> creates outputs/val_restored/metrics_per_image.csv

Usage:
    python plot_results.py --metrics_csv outputs/val_restored/metrics_per_image.csv --restored_dir outputs/val_restored
"""

import argparse
import csv
import os

import numpy as np
import matplotlib.pyplot as plt


def plot_loss_curve(log_csv, out_path="loss_curve.png"):
    epochs, train_losses, val_losses = [], [], []
    with open(log_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_losses.append(float(row["train_loss"]))
            val_losses.append(float(row["val_loss"]))

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("L1 Loss")
    plt.title("Training / Validation Loss Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    print(f"Saved: {out_path}")
    plt.close()


def plot_metrics_distribution(metrics_csv, out_path="metrics_distribution.png"):
    psnr_vals, ssim_vals, lpips_vals = [], [], []
    with open(metrics_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            psnr_vals.append(float(row["psnr"]))
            ssim_vals.append(float(row["ssim"]))
            lpips_vals.append(float(row["lpips"]))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, vals, name, unit in zip(
        axes, [psnr_vals, ssim_vals, lpips_vals], ["PSNR", "SSIM", "LPIPS"], ["dB", "", ""]
    ):
        ax.hist(vals, bins=20, color="steelblue", edgecolor="black")
        ax.axvline(np.mean(vals), color="red", linestyle="--", label=f"mean={np.mean(vals):.3f}")
        ax.set_title(f"{name} distribution ({len(vals)} images)")
        ax.set_xlabel(f"{name} {unit}".strip())
        ax.set_ylabel("Count")
        ax.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    print(f"Saved: {out_path}")
    plt.close()

    print("\nSummary:")
    print(f"  PSNR:  mean={np.mean(psnr_vals):.3f}  min={np.min(psnr_vals):.3f}  max={np.max(psnr_vals):.3f}")
    print(f"  SSIM:  mean={np.mean(ssim_vals):.4f}  min={np.min(ssim_vals):.4f}  max={np.max(ssim_vals):.4f}")
    print(f"  LPIPS: mean={np.mean(lpips_vals):.4f}  min={np.min(lpips_vals):.4f}  max={np.max(lpips_vals):.4f}")


def plot_before_after_grid(degraded_dir, restored_dir, gt_dir, filenames, out_path="before_after_grid.png"):
    n = len(filenames)
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    if n == 1:
        axes = [axes]

    for i, fname in enumerate(filenames):
        degraded = np.load(os.path.join(degraded_dir, fname))
        restored = np.load(os.path.join(restored_dir, fname))
        gt = np.load(os.path.join(gt_dir, fname))

        axes[i][0].imshow(degraded, cmap="gray")
        axes[i][0].set_title("Degraded" if i == 0 else "", fontsize=9)
        axes[i][0].axis("off")

        axes[i][1].imshow(np.clip(restored, 0, 1), cmap="gray")
        axes[i][1].set_title("Restored (ours)" if i == 0 else "", fontsize=9)
        axes[i][1].axis("off")

        axes[i][2].imshow(gt, cmap="gray")
        axes[i][2].set_title("Ground Truth" if i == 0 else "", fontsize=9)
        axes[i][2].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    print(f"Saved: {out_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_csv", type=str, default="checkpoints/training_log.csv")
    parser.add_argument("--metrics_csv", type=str, default=None)
    parser.add_argument("--degraded_dir", type=str, default="data/degraded")
    parser.add_argument("--restored_dir", type=str, default=None)
    parser.add_argument("--gt_dir", type=str, default="data/gt")
    parser.add_argument("--n_samples", type=int, default=4)
    args = parser.parse_args()

    if os.path.exists(args.log_csv):
        plot_loss_curve(args.log_csv)
    else:
        print(f"Skipping loss curve: {args.log_csv} not found")

    if args.metrics_csv and os.path.exists(args.metrics_csv):
        plot_metrics_distribution(args.metrics_csv)
    else:
        print("Skipping metrics distribution: pass --metrics_csv pointing to metrics_per_image.csv")

    if args.restored_dir and os.path.exists(args.restored_dir):
        sample_files = sorted(f for f in os.listdir(args.restored_dir) if f.endswith(".npy"))[: args.n_samples]
        plot_before_after_grid(args.degraded_dir, args.restored_dir, args.gt_dir, sample_files)
    else:
        print("Skipping before/after grid: pass --restored_dir pointing to evaluate.py's output folder")
