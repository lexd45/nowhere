"""
Compares PSNR/SSIM/LPIPS across multiple trained model runs side by side.
Produces a printed table and a bar chart comparison image -- useful directly
for your PPT results slide.

Usage:
    python compare_models.py --runs "v1:outputs/val_restored/metrics_per_image.csv" "v3:outputs/val_restored_v3/metrics_per_image.csv"

Add one --runs entry per completed model evaluation (v1, v2, v3, ...).
"""

import argparse
import csv

import numpy as np
import matplotlib.pyplot as plt


def load_metrics(csv_path):
    psnr, ssim, lpips_ = [], [], []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            psnr.append(float(row["psnr"]))
            ssim.append(float(row["ssim"]))
            lpips_.append(float(row["lpips"]))
    return np.array(psnr), np.array(ssim), np.array(lpips_)


def main(args):
    labels = []
    all_psnr, all_ssim, all_lpips = [], [], []

    print(f"{'Model':<10}{'PSNR (dB)':>16}{'SSIM':>14}{'LPIPS':>14}")
    print("-" * 54)

    for entry in args.runs:
        label, path = entry.split(":", 1)
        psnr, ssim, lpips_ = load_metrics(path)
        labels.append(label)
        all_psnr.append(psnr)
        all_ssim.append(ssim)
        all_lpips.append(lpips_)
        print(f"{label:<10}{np.mean(psnr):>10.3f} +/-{np.std(psnr):>4.2f}"
              f"{np.mean(ssim):>13.4f}{np.mean(lpips_):>14.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, values, name in zip(axes, [all_psnr, all_ssim, all_lpips], ["PSNR (dB)", "SSIM", "LPIPS"]):
        means = [np.mean(v) for v in values]
        stds = [np.std(v) for v in values]
        ax.bar(labels, means, yerr=stds, capsize=5, color="steelblue", edgecolor="black")
        ax.set_title(name)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("model_comparison.png", dpi=120)
    print("\nSaved: model_comparison.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True, help='List of "label:path_to_metrics_csv"')
    args = parser.parse_args()

    main(args)
