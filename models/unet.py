"""
Small residual U-Net for the KLA restoration task.

Approach (matches KLA's own stated guidance: "direct image to image
regression: image in, image out"):
  1. Bilinear-upsample the 128x128 degraded input to 256x256
  2. Pass it through a U-Net that predicts a *residual correction*
  3. Add the residual to the upsampled input to get the final output

Predicting a residual (rather than the full image from scratch) is a
standard, well-documented technique in image restoration literature --
it's easier to train and converges faster, because if the network learns
"predict zero," the output already equals a reasonable upsampled image.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # pad in case of odd-size mismatch between encoder/decoder feature maps
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class RestorationUNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base_ch=32):
        """
        base_ch controls model size/VRAM usage:
          base_ch=32 -> default, fits comfortably on RTX 4050 (6GB)
          base_ch=16 -> smaller/faster variant, use on RTX 3050 (4GB) for
                        quick parallel experiments
        """
        super().__init__()
        self.inc = DoubleConv(in_ch, base_ch)
        self.down1 = Down(base_ch, base_ch * 2)
        self.down2 = Down(base_ch * 2, base_ch * 4)
        self.down3 = Down(base_ch * 4, base_ch * 8)
        self.up1 = Up(base_ch * 8, base_ch * 4)
        self.up2 = Up(base_ch * 4, base_ch * 2)
        self.up3 = Up(base_ch * 2, base_ch)
        self.outc = nn.Conv2d(base_ch, out_ch, kernel_size=1)

    def forward(self, x_lr):
        # x_lr: (B, 1, 128, 128) degraded input
        x_up = F.interpolate(x_lr, scale_factor=2, mode="bilinear", align_corners=False)  # (B,1,256,256)

        x1 = self.inc(x_up)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)

        residual = self.outc(x)
        return x_up + residual  # residual learning


if __name__ == "__main__":
    # Smoke test: confirm a forward pass runs and produces the right output shape
    model = RestorationUNet(base_ch=32)
    dummy_input = torch.randn(2, 1, 128, 128)  # batch of 2
    output = model(dummy_input)
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == (2, 1, 256, 256), "Output shape mismatch!"
    print("Model smoke test passed.")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")
