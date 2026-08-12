"""
Dataset loader for the KLA image restoration task.

Pairs are matched by identical filename across the gt/ and degraded/ folders
(e.g. gt/000000.npy <-> degraded/000000.npy), confirmed by matching means
during data inspection.

GT images:       (256, 256), float32, exact range [0, 1]
Degraded images: (128, 128), float32, range roughly [-0.05, 1.4]
                  (values outside [0,1] are expected -- speckle noise artifact,
                  not a bug. Do NOT clip them.)
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset


class PairedRestorationDataset(Dataset):
    def __init__(self, gt_dir, degraded_dir, filenames):
        """
        gt_dir, degraded_dir : folder paths
        filenames            : list of filenames (e.g. ['000000.npy', ...])
                                shared between both folders for this split
        """
        self.gt_dir = gt_dir
        self.degraded_dir = degraded_dir
        self.filenames = filenames

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]

        gt = np.load(os.path.join(self.gt_dir, fname)).astype(np.float32)
        degraded = np.load(os.path.join(self.degraded_dir, fname)).astype(np.float32)

        # Add channel dimension: (H,W) -> (1,H,W), required by PyTorch conv layers
        gt_tensor = torch.from_numpy(gt).unsqueeze(0)
        degraded_tensor = torch.from_numpy(degraded).unsqueeze(0)

        return degraded_tensor, gt_tensor


class UnpairedTestDataset(Dataset):
    """For data/test_degraded -- no ground truth available, used only at
    inference time (evaluate.py), not during training."""

    def __init__(self, degraded_dir):
        self.degraded_dir = degraded_dir
        self.filenames = sorted(
            f for f in os.listdir(degraded_dir) if f.endswith(".npy")
        )

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        degraded = np.load(os.path.join(self.degraded_dir, fname)).astype(np.float32)
        degraded_tensor = torch.from_numpy(degraded).unsqueeze(0)
        return degraded_tensor, fname  # filename returned so outputs can be saved with matching names


def get_train_val_filenames(gt_dir, val_fraction=0.1, seed=42):
    """Splits filenames into train/val lists. Fixed seed = reproducible split
    every time this is run (training hygiene: same split across runs/machines)."""
    filenames = sorted(f for f in os.listdir(gt_dir) if f.endswith(".npy"))

    rng = np.random.RandomState(seed)
    shuffled = filenames.copy()
    rng.shuffle(shuffled)

    n_val = int(len(shuffled) * val_fraction)
    val_files = shuffled[:n_val]
    train_files = shuffled[n_val:]

    return train_files, val_files


if __name__ == "__main__":
    # Quick smoke test -- run this file directly to sanity-check everything
    # loads and shapes come out correctly before training starts.
    train_files, val_files = get_train_val_filenames("data/gt", val_fraction=0.1)
    print(f"Train pairs: {len(train_files)}, Val pairs: {len(val_files)}")

    train_ds = PairedRestorationDataset("data/gt", "data/degraded", train_files)
    degraded, gt = train_ds[0]
    print(f"degraded shape: {degraded.shape}, dtype: {degraded.dtype}")
    print(f"gt shape:       {gt.shape}, dtype: {gt.dtype}")
    print(f"degraded min/max: {degraded.min():.4f} / {degraded.max():.4f}")
    print(f"gt min/max:       {gt.min():.4f} / {gt.max():.4f}")
