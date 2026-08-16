
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

def gradient_loss(pred, target):
    
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                            dtype=torch.float32, device=pred.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                            dtype=torch.float32, device=pred.device).view(1, 1, 3, 3)

    pred_gx = F.conv2d(pred, sobel_x, padding=1)
    pred_gy = F.conv2d(pred, sobel_y, padding=1)
    target_gx = F.conv2d(target, sobel_x, padding=1)
    target_gy = F.conv2d(target, sobel_y, padding=1)

    return F.l1_loss(pred_gx, target_gx) + F.l1_loss(pred_gy, target_gy)

class CharbonnierLoss(nn.Module):
    
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        diff = pred - target
        return torch.mean(torch.sqrt(diff * diff + self.eps ** 2))

class VGGPerceptualLoss(nn.Module):
    
    def __init__(self, device):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features[:16].to(device)
        for p in vgg.parameters():
            p.requires_grad = False
        vgg.eval()
        self.vgg = vgg
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.to(device)

    def forward(self, pred, target):
        pred_3 = pred.repeat(1, 3, 1, 1)
        target_3 = target.repeat(1, 3, 1, 1)
        pred_norm = (pred_3 - self.mean) / self.std
        target_norm = (target_3 - self.mean) / self.std
        return F.l1_loss(self.vgg(pred_norm), self.vgg(target_norm))

class FFTLoss(nn.Module):
    
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        pred_fft = torch.fft.rfft2(pred, norm='ortho')
        target_fft = torch.fft.rfft2(target, norm='ortho')
        return F.l1_loss(torch.abs(pred_fft), torch.abs(target_fft))

class CombinedLoss(nn.Module):
    def __init__(self, device, w_l1=1.0, w_ssim=1.0, w_perceptual=0.1, w_gradient=1.0,
                 w_fft=0.0, use_charbonnier=False, use_mse=False):
        
        super().__init__()
        self.w_l1 = w_l1
        self.w_ssim = w_ssim
        self.w_perceptual = w_perceptual
        self.w_gradient = w_gradient
        self.w_fft = w_fft
        self.use_charbonnier = use_charbonnier
        self.use_mse = use_mse
        if self.use_mse:
            self.l1 = nn.MSELoss()
        else:
            self.l1 = CharbonnierLoss() if use_charbonnier else nn.L1Loss()
        self.perceptual = VGGPerceptualLoss(device) if w_perceptual > 0 else None
        self.fft_loss = FFTLoss() if w_fft > 0 else None

    def forward(self, pred, target):
        l1 = self.l1(pred, target)
        ssim_l = ssim_loss(pred, target) if self.w_ssim > 0 else torch.tensor(0.0, device=pred.device)
        perc = self.perceptual(pred, target) if self.perceptual is not None else torch.tensor(0.0, device=pred.device)
        grad = gradient_loss(pred, target) if self.w_gradient > 0 else torch.tensor(0.0, device=pred.device)
        fft_l = self.fft_loss(pred, target) if self.fft_loss is not None else torch.tensor(0.0, device=pred.device)

        total = self.w_l1 * l1 + self.w_ssim * ssim_l + self.w_perceptual * perc + self.w_gradient * grad + self.w_fft * fft_l
        components = {"l1": l1.item(), "ssim_loss": ssim_l.item(), "perceptual": perc.item(),
                      "gradient": grad.item(), "fft": fft_l.item(), "total": total.item()}
        return total, components
