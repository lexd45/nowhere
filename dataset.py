
import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset

class PairedRestorationDataset(Dataset):
    def __init__(self, gt_dir, degraded_dir, filenames, augment=False):
        
        self.gt_dir = gt_dir
        self.degraded_dir = degraded_dir
        self.filenames = filenames
        self.augment = augment

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]

        gt = np.load(os.path.join(self.gt_dir, fname)).astype(np.float32)
        degraded = np.load(os.path.join(self.degraded_dir, fname)).astype(np.float32)

        if self.augment:
            h_lr, w_lr = degraded.shape
            crop_lr_size = 64
            crop_hr_size = crop_lr_size * 2

            top_lr = random.randint(0, h_lr - crop_lr_size)
            left_lr = random.randint(0, w_lr - crop_lr_size)
            top_hr, left_hr = top_lr * 2, left_lr * 2

            degraded = degraded[top_lr:top_lr+crop_lr_size, left_lr:left_lr+crop_lr_size]
            gt = gt[top_hr:top_hr+crop_hr_size, left_hr:left_hr+crop_hr_size]

            if random.random() < 0.5:
                gt = np.flip(gt, axis=1).copy()
                degraded = np.flip(degraded, axis=1).copy()
            if random.random() < 0.5:
                gt = np.flip(gt, axis=0).copy()
                degraded = np.flip(degraded, axis=0).copy()
            k = random.choice([0, 1, 2, 3])
            if k > 0:
                gt = np.rot90(gt, k=k).copy()
                degraded = np.rot90(degraded, k=k).copy()

        gt_tensor = torch.from_numpy(gt).unsqueeze(0)
        degraded_tensor = torch.from_numpy(degraded).unsqueeze(0)

        return degraded_tensor, gt_tensor

class UnpairedTestDataset(Dataset):
    
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
        return degraded_tensor, fname

def get_train_val_filenames(gt_dir, val_fraction=0.1, seed=42):
    
    filenames = sorted(f for f in os.listdir(gt_dir) if f.endswith(".npy"))

    rng = np.random.RandomState(seed)
    shuffled = filenames.copy()
    rng.shuffle(shuffled)

    n_val = int(len(shuffled) * val_fraction)
    val_files = shuffled[:n_val]
    train_files = shuffled[n_val:]

    return train_files, val_files

if __name__ == "__main__":
    train_files, val_files = get_train_val_filenames("data/gt", val_fraction=0.1)
    print(f"Train pairs: {len(train_files)}, Val pairs: {len(val_files)}")

    train_ds = PairedRestorationDataset("data/gt", "data/degraded", train_files)
    degraded, gt = train_ds[0]
    print(f"degraded shape: {degraded.shape}, dtype: {degraded.dtype}")
    print(f"gt shape:       {gt.shape}, dtype: {gt.dtype}")
    print(f"degraded min/max: {degraded.min():.4f} / {degraded.max():.4f}")
    print(f"gt min/max:       {gt.min():.4f} / {gt.max():.4f}")
