import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout),
        )

    def forward(self, x):
        return self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x):
        x = self.pool(x)
        return self.conv(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)

        if x.shape != skip.shape:
            x = nn.functional.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)

        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class SARangNet(nn.Module):
    """
    Lightweight U-Net style colorization model.

    Input channels:
        10 SAR-derived features

    Output channels:
        3 RGB channels
        1 confidence channel
    """

    def __init__(self, in_channels: int = 10, out_channels: int = 4):
        super().__init__()

        self.enc1 = ConvBlock(in_channels, 64)
        self.enc2 = DownBlock(64, 128)
        self.enc3 = DownBlock(128, 256)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.InstanceNorm2d(512, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.2),
        )

        self.up3 = UpBlock(512, 256, 256)
        self.up2 = UpBlock(256, 128, 128)
        self.up1 = UpBlock(128, 64, 64)

        self.final = nn.Conv2d(64, out_channels, kernel_size=1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        b = self.bottleneck(e3)

        d3 = self.up3(b, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)

        out = self.final(d1)
        return self.activation(out)
