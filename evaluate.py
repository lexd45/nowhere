"""
Standalone evaluation / inference script for the KLA restoration task.

REQUIRED USAGE (must run unmodified, as specified in the hackathon brief):
    python evaluate.py --input_dir <path_to_degraded_npy_folder> --output_dir <path_to_save_restored_npy>

OPTIONAL — if ground truth is available, also computes and prints PSNR/SSIM/LPIPS:
    python evaluate.py --input_dir data/degraded --output_dir outputs/val_restored --gt_dir data/gt --val_only

--val_only restricts evaluation to the same held-out validation filenames used during
training (never seen by the model), so metrics reflect real generalization, not
memorized training data. Use this for your own results reporting.

For the actual held-out test set (no ground truth available), run WITHOUT --val_only
and WITHOUT --gt_dir:
    python evaluate.py --input_dir data/test_degraded --output_dir outputs/test_restored
"""

import argparse
import os
import time

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric
import lpips

from models.unet import RestorationUNet
from dataset import get_train_val_filenames


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    base_ch = checkpoint.get("base_ch", 32)
    model = RestorationUNet(base_ch=base_ch).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def restore_image(model, degraded_np, device):
    """degraded_np: (H,W) float32 -> returns (H,W) float32, clipped to [0,1]"""
    x = torch.from_numpy(degraded_np.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(x)
    pred_np = pred.squeeze().cpu().numpy()
    pred_np = np.clip(pred_np, 0.0, 1.0)
    return pred_np


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_model(args.checkpoint, device)
    print(f"Loaded model from {args.checkpoint}")

    os.makedirs(args.output_dir, exist_ok=True)

    if args.val_only:
        if not args.gt_dir:
            raise ValueError("--val_only requires --gt_dir to be set")
        _, filenames = get_train_val_filenames(args.gt_dir, val_fraction=args.val_fraction, seed=args.seed)
        print(f"Restricting evaluation to {len(filenames)} held-out validation files")
    else:
        filenames = sorted(f for f in os.listdir(args.input_dir) if f.endswith(".npy"))
        print(f"Found {len(filenames)} files in {args.input_dir}")

    compute_metrics = args.gt_dir is not None
    lpips_model = None
    if compute_metrics:
        print("Loading LPIPS model (downloads AlexNet weights on first run)...")
        lpips_model = lpips.LPIPS(net="alex").to(device)
        lpips_model.eval()

    psnr_scores, ssim_scores, lpips_scores = [], [], []
    per_image_records = []

    start_time = time.time()
    for fname in filenames:
        degraded_np = np.load(os.path.join(args.input_dir, fname)).astype(np.float32)
        restored_np = restore_image(model, degraded_np, device)

        np.save(os.path.join(args.output_dir, fname), restored_np)

        if compute_metrics:
            gt_np = np.load(os.path.join(args.gt_dir, fname)).astype(np.float32)

            p = psnr_metric(gt_np, restored_np, data_range=1.0)
            s = ssim_metric(gt_np, restored_np, data_range=1.0)

            gt_t = torch.from_numpy(gt_np).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1).to(device)
            pred_t = torch.from_numpy(restored_np).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1).to(device)
            gt_t = gt_t * 2 - 1     # [0,1] -> [-1,1], required by LPIPS
            pred_t = pred_t * 2 - 1
            with torch.no_grad():
                l = lpips_model(pred_t, gt_t).item()

            psnr_scores.append(p)
            ssim_scores.append(s)
            lpips_scores.append(l)
            per_image_records.append((fname, p, s, l))

    total_time = time.time() - start_time
    n_images = len(filenames)
    print(f"\nProcessed {n_images} images in {total_time:.1f}s ({total_time / n_images * 1000:.1f} ms/image)")
    print(f"Restored outputs saved to: {args.output_dir}")

    if compute_metrics:
        print("\n--- Metrics (mean +/- std) ---")
        print(f"PSNR:  {np.mean(psnr_scores):.3f} +/- {np.std(psnr_scores):.3f} dB")
        print(f"SSIM:  {np.mean(ssim_scores):.4f} +/- {np.std(ssim_scores):.4f}")
        print(f"LPIPS: {np.mean(lpips_scores):.4f} +/- {np.std(lpips_scores):.4f}")

        metrics_path = os.path.join(args.output_dir, "metrics_per_image.csv")
        with open(metrics_path, "w") as f:
            f.write("filename,psnr,ssim,lpips\n")
            for fname, p, s, l in per_image_records:
                f.write(f"{fname},{p:.4f},{s:.4f},{l:.4f}\n")
        print(f"Per-image metrics saved to: {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Folder of degraded .npy images")
    parser.add_argument("--output_dir", type=str, required=True, help="Folder to save restored .npy images")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--gt_dir", type=str, default=None, help="Optional: enables PSNR/SSIM/LPIPS computation")
    parser.add_argument("--val_only", action="store_true", help="Evaluate only on held-out validation split")
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    main(args)
