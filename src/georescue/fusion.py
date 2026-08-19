"""Fusion building blocks kept separate for future improved-fusion experiments."""
from __future__ import annotations

import torch
from torch import nn


class FeatureConcatFusion(nn.Module):
    """Concatenate two feature tensors and project them back to a chosen width."""

    def __init__(self, left_channels: int, right_channels: int, out_channels: int) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(left_channels + right_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        if left.shape[-2:] != right.shape[-2:]:
            raise ValueError(f"Fusion feature maps must share spatial size: {left.shape} vs {right.shape}")
        return self.projection(torch.cat([left, right], dim=1))
