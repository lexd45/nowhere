"""
Combined loss function for restoration training.

L1 alone tends to produce blurry outputs -- it's a per-pixel average, which
rewards "safe" smooth predictions whenever the model is uncertain about
fine detail. This combined loss adds two more terms that directly target
your specific metric goals:

  - SSIM loss: directly optimizes structural similarity (targets your SSIM metric)
  - VGG perceptual loss: compares deep image features rather than raw pixels
    (targets your LPIPS metric, which works the same way)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


def _gaussian_window(window_size, sigma, channels, device):
    coords = torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window_1d = g.unsqueeze(1)
    window_2d = window_1d @ window_1d.t()
    return window_2d.expand(channels, 1, window_size, window_size).contiguous()


def ssim_loss(pred, target, window_size=11, sigma=1.5, C1=0.01 ** 2, C2=0.03 ** 2):
    """Returns (1 - SSIM), so it behaves like a normal loss to minimize."""
    channels = pred.size(1)
    window = _gaussian_window(window_size, sigma, channels, pred.device)

    mu1 = F.conv2d(pred, window, padding=window_size // 2, groups=channels)
    mu2 = F.conv2d(target, window, padding=window_size // 2, groups=channels)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, window, padding=window_size // 2, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=window_size // 2, groups=channels) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=window_size // 2, groups=channels) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return 1 - ssim_map.mean()


class VGGPerceptualLoss(nn.Module):
    """Compares deep VGG features instead of raw pixels -- same principle LPIPS
    itself uses, so optimizing this directly pulls LPIPS down."""

    def __init__(self, device):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features[:16].to(device)
        for p in vgg.parameters():
            p.requires_grad = False
        vgg.eval()
        self.vgg = vgg
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.to(device)  # ensures mean/std buffers actually move to GPU, not just self.vgg

    def forward(self, pred, target):
        pred_3 = pred.repeat(1, 3, 1, 1)
        target_3 = target.repeat(1, 3, 1, 1)
        pred_norm = (pred_3 - self.mean) / self.std
        target_norm = (target_3 - self.mean) / self.std
        return F.l1_loss(self.vgg(pred_norm), self.vgg(target_norm))


class CombinedLoss(nn.Module):
    def __init__(self, device, w_l1=1.0, w_ssim=1.0, w_perceptual=0.1):
        super().__init__()
        self.w_l1 = w_l1
        self.w_ssim = w_ssim
        self.w_perceptual = w_perceptual
        self.l1 = nn.L1Loss()
        self.perceptual = VGGPerceptualLoss(device) if w_perceptual > 0 else None

    def forward(self, pred, target):
        pred = torch.clamp(pred, 0.0, 1.0)  # match evaluate.py's clipping so training optimizes what's actually measured
        l1 = self.l1(pred, target)
        ssim_l = ssim_loss(pred, target) if self.w_ssim > 0 else torch.tensor(0.0, device=pred.device)
        perc = self.perceptual(pred, target) if self.perceptual is not None else torch.tensor(0.0, device=pred.device)

        total = self.w_l1 * l1 + self.w_ssim * ssim_l + self.w_perceptual * perc
        components = {"l1": l1.item(), "ssim_loss": ssim_l.item(), "perceptual": perc.item(), "total": total.item()}
        return total, components
