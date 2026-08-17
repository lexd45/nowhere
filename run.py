import sys
import os
import time
import numpy as np
import torch
from models.unet import RestorationUNet

def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    base_ch = checkpoint.get("base_ch", 32)
    model = RestorationUNet(base_ch=base_ch).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model

def restore_image(model, degraded_np, device, use_tta=True):
    # degraded_np is (H, W). We need (1, 1, H, W)
    x = torch.from_numpy(degraded_np.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        if use_tta:
            preds = []
            for k in range(4):
                rot_x = torch.rot90(x, k, [2, 3])
                out1 = model(rot_x)
                out2 = model(torch.flip(rot_x, [3]))
                out1_inv = torch.rot90(out1, -k, [2, 3])
                out2_inv = torch.rot90(torch.flip(out2, [3]), -k, [2, 3])
                preds.extend([out1_inv, out2_inv])
            pred = torch.mean(torch.stack(preds), dim=0)
        else:
            pred = model(x)
            
    pred_np = pred.squeeze().cpu().numpy()
    pred_np = np.clip(pred_np, 0.0, 1.0)
    # Ensure no NaN or Inf
    pred_np = np.nan_to_num(pred_np, nan=0.0, posinf=1.0, neginf=0.0)
    return pred_np

def main():
    if len(sys.argv) != 3:
        print("Usage: python run.py <input-dir> <output-dir>")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    # Create output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    checkpoint_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints", "best_model.pt")
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Model weights not found at {checkpoint_path}")
        sys.exit(1)

    model = load_model(checkpoint_path, device)
    print(f"Loaded model from {checkpoint_path}")

    filenames = sorted([f for f in os.listdir(input_dir) if f.endswith(".npy")])
    print(f"Found {len(filenames)} files in {input_dir}")

    start_time = time.time()
    for fname in filenames:
        degraded_np = np.load(os.path.join(input_dir, fname)).astype(np.float32)
        restored_np = restore_image(model, degraded_np, device)

        # Force shape to be (H, W) if it accidentally gained an extra dimension
        if restored_np.ndim > 2:
            restored_np = restored_np.squeeze()

        out_npy = os.path.join(output_dir, fname)
        np.save(out_npy, restored_np)

    total_time = time.time() - start_time
    n_images = len(filenames)
    if n_images > 0:
        print(f"\nProcessed {n_images} images in {total_time:.1f}s ({total_time / n_images * 1000:.1f} ms/image)")
    print(f"Restored outputs saved to: {output_dir}")

if __name__ == "__main__":
    main()
