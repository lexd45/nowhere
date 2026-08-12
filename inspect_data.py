"""
Quick inspection script for .npy dataset files.

Usage (run from your project folder, with restoration-env activated):
    python inspect_data.py --path data/train/gt
    python inspect_data.py --path data/train/degraded --n 8
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os


def inspect_folder(folder_path, num_samples=5):
    if not os.path.isdir(folder_path):
        print(f"ERROR: '{folder_path}' is not a valid folder. Check the path.")
        return

    files = sorted([f for f in os.listdir(folder_path) if f.endswith(".npy")])

    if not files:
        print(f"No .npy files found in '{folder_path}'.")
        print("Contents of this folder:", os.listdir(folder_path)[:10])
        return

    print(f"Found {len(files)} .npy files in '{folder_path}'")
    print(f"First few filenames: {files[:5]}")
    print("-" * 60)

    sample_files = files[:num_samples]
    fig, axes = plt.subplots(1, len(sample_files), figsize=(4 * len(sample_files), 4))
    if len(sample_files) == 1:
        axes = [axes]

    for i, fname in enumerate(sample_files):
        arr = np.load(os.path.join(folder_path, fname))
        print(f"\n{fname}")
        print(f"  shape : {arr.shape}")
        print(f"  dtype : {arr.dtype}")
        print(f"  min   : {arr.min():.4f}")
        print(f"  max   : {arr.max():.4f}")
        print(f"  mean  : {arr.mean():.4f}")

        img = arr.squeeze()  # drop extra channel dim if grayscale, e.g. (H,W,1) -> (H,W)
        axes[i].imshow(img, cmap="gray")
        axes[i].set_title(f"{fname}\n{arr.shape}", fontsize=8)
        axes[i].axis("off")

    plt.tight_layout()
    out_path = "npy_preview.png"
    plt.savefig(out_path, dpi=100)
    print(f"\nSaved visual preview image to: {out_path}")
    print("(Open that PNG file directly if a window doesn't pop up automatically.)")

    try:
        plt.show()
    except Exception:
        pass  # some setups can't pop a window; the saved PNG still works


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True, help="Folder containing .npy files")
    parser.add_argument("--n", type=int, default=5, help="Number of samples to preview")
    args = parser.parse_args()

    inspect_folder(args.path, args.n)
