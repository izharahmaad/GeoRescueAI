"""Preprocessing contracts shared by future real-dataset adapters.

Real benchmark readers should normalize data into the dataset contract before
it reaches the model. Dataset-specific label mappings belong in adapters and
must follow the benchmark's official definitions.
"""
from __future__ import annotations

import numpy as np


def normalize_minmax(array: np.ndarray, lower: float = 0.0, upper: float = 1.0) -> np.ndarray:
    """Safely scale an array into [lower, upper]."""
    x = np.asarray(array, dtype=np.float32)
    lo = float(np.nanmin(x))
    hi = float(np.nanmax(x))
    if not np.isfinite(lo) or not np.isfinite(hi):
        raise ValueError("Input contains no finite range")
    if hi <= lo:
        return np.full_like(x, lower, dtype=np.float32)
    scaled = (x - lo) / (hi - lo)
    return (scaled * (upper - lower) + lower).astype(np.float32)


def ensure_channel_first(array: np.ndarray, channels: int) -> np.ndarray:
    """Convert HWC to CHW while rejecting ambiguous channel dimensions."""
    x = np.asarray(array)
    if x.ndim != 3:
        raise ValueError(f"Expected a 3D array, got {x.shape}")
    if x.shape[0] == channels:
        return x
    if x.shape[-1] == channels:
        return np.moveaxis(x, -1, 0)
    raise ValueError(f"Could not identify {channels} channels in shape {x.shape}")


def validate_aligned_modalities(optical: np.ndarray, sar: np.ndarray, target: np.ndarray) -> None:
    """Validate spatial correspondence before model training/inference."""
    if optical.shape[-2:] != sar.shape[-2:] or optical.shape[-2:] != target.shape[-2:]:
        raise ValueError(
            f"Spatial alignment failed: optical={optical.shape}, sar={sar.shape}, target={target.shape}"
        )
