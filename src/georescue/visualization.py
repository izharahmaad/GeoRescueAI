"""Lightweight prediction visualization utilities."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def save_mask(mask: np.ndarray, path: str | Path) -> None:
    arr = np.asarray(mask)
    if arr.ndim != 2:
        raise ValueError("mask must be 2D")
    Image.fromarray(arr.astype(np.uint8), mode="L").save(Path(path))


def save_overlay(optical: np.ndarray, mask: np.ndarray, path: str | Path) -> None:
    image = np.asarray(optical)
    if image.ndim != 3:
        raise ValueError("optical must be CHW or HWC")
    if image.shape[0] in (1, 3):
        image = np.moveaxis(image, 0, -1)
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    image = np.clip(image, 0, 1) * 255
    image = image.astype(np.uint8)
    binary = (np.asarray(mask) > 0)
    overlay = image.copy()
    overlay[binary, 0] = 255
    overlay[binary, 1:] = (overlay[binary, 1:] * 0.35).astype(np.uint8)
    Image.fromarray(overlay).save(Path(path))
