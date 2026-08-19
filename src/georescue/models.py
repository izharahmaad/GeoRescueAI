"""Segmentation models used by GeoRescue AI experiments."""
from __future__ import annotations

from typing import Literal

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_channels, out_channels, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class Up(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_channels + skip_channels, out_channels, dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([skip, x], dim=1))


class UNet(nn.Module):
    """Compact U-Net baseline for one modality."""

    def __init__(self, in_channels: int, num_classes: int, base_channels: int = 32, dropout: float = 0.0) -> None:
        super().__init__()
        b = base_channels
        self.inc = ConvBlock(in_channels, b, dropout)
        self.down1 = Down(b, b * 2, dropout)
        self.down2 = Down(b * 2, b * 4, dropout)
        self.down3 = Down(b * 4, b * 8, dropout)
        self.bottleneck = Down(b * 8, b * 16, dropout)
        self.up1 = Up(b * 16, b * 8, b * 8, dropout)
        self.up2 = Up(b * 8, b * 4, b * 4, dropout)
        self.up3 = Up(b * 4, b * 2, b * 2, dropout)
        self.up4 = Up(b * 2, b, b, dropout)
        self.head = nn.Conv2d(b, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.bottleneck(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.head(x)


class FusionUNet(nn.Module):
    """Early feature-fusion U-Net with separate optical and SAR stems."""

    def __init__(
        self,
        optical_channels: int,
        sar_channels: int,
        num_classes: int,
        base_channels: int = 32,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        b = base_channels
        self.optical_stem = ConvBlock(optical_channels, b, dropout)
        self.sar_stem = ConvBlock(sar_channels, b, dropout)
        self.fused_stem = ConvBlock(b * 2, b, dropout)
        self.down1 = Down(b, b * 2, dropout)
        self.down2 = Down(b * 2, b * 4, dropout)
        self.down3 = Down(b * 4, b * 8, dropout)
        self.bottleneck = Down(b * 8, b * 16, dropout)
        self.up1 = Up(b * 16, b * 8, b * 8, dropout)
        self.up2 = Up(b * 8, b * 4, b * 4, dropout)
        self.up3 = Up(b * 4, b * 2, b * 2, dropout)
        self.up4 = Up(b * 2, b, b, dropout)
        self.head = nn.Conv2d(b, num_classes, 1)

    def forward(self, optical: torch.Tensor, sar: torch.Tensor) -> torch.Tensor:
        optical_features = self.optical_stem(optical)
        sar_features = self.sar_stem(sar)
        x1 = self.fused_stem(torch.cat([optical_features, sar_features], dim=1))
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.bottleneck(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.head(x)


ModelName = Literal["optical_unet", "sar_unet", "fusion_unet"]


def build_model(
    name: ModelName,
    optical_channels: int,
    sar_channels: int,
    num_classes: int,
    base_channels: int,
    dropout: float = 0.0,
) -> nn.Module:
    if name == "optical_unet":
        model = UNet(optical_channels, num_classes, base_channels, dropout)
        model._is_sar_model = False
        return model
    if name == "sar_unet":
        model = UNet(sar_channels, num_classes, base_channels, dropout)
        model._is_sar_model = True
        return model
    if name == "fusion_unet":
        return FusionUNet(optical_channels, sar_channels, num_classes, base_channels, dropout)
    raise ValueError(f"Unsupported model: {name}")
