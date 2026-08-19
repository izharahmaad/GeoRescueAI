"""Dataset abstractions for matched optical/SAR/target samples."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


class NPZMultiModalDataset(Dataset[dict[str, torch.Tensor]]):
    """Load samples stored as .npz with arrays: optical, sar, target.

    Arrays are expected channel-first for optical/SAR, while target is HxW.
    The loader also accepts channel-last arrays and normalizes them to channel-first.
    """

    def __init__(self, directory: str | Path, optical_channels: int, sar_channels: int) -> None:
        self.directory = Path(directory)
        self.files = sorted(self.directory.glob("*.npz"))
        if not self.files:
            raise FileNotFoundError(f"No .npz samples found in {self.directory}")
        self.optical_channels = optical_channels
        self.sar_channels = sar_channels

    @staticmethod
    def _channel_first(array: np.ndarray, channels: int, name: str) -> np.ndarray:
        arr = np.asarray(array)
        if arr.ndim != 3:
            raise ValueError(f"{name} must be 3D, got shape {arr.shape}")
        if arr.shape[0] == channels:
            return arr
        if arr.shape[-1] == channels:
            return np.moveaxis(arr, -1, 0)
        raise ValueError(f"{name} expected {channels} channels, got shape {arr.shape}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        path = self.files[index]
        with np.load(path) as sample:
            optical = self._channel_first(sample["optical"], self.optical_channels, "optical")
            sar = self._channel_first(sample["sar"], self.sar_channels, "sar")
            target = np.asarray(sample["target"])

        if target.ndim != 2:
            raise ValueError(f"target must be HxW, got {target.shape}")
        if optical.shape[1:] != target.shape or sar.shape[1:] != target.shape:
            raise ValueError(
                f"Spatial mismatch for {path.name}: optical={optical.shape}, "
                f"sar={sar.shape}, target={target.shape}"
            )

        return {
            "optical": torch.from_numpy(optical.astype(np.float32)),
            "sar": torch.from_numpy(sar.astype(np.float32)),
            "target": torch.from_numpy(target.astype(np.int64)),
            "id": path.stem,
        }


def collate_keep_ids(batch: Sequence[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor | list[str]]:
    return {
        "optical": torch.stack([item["optical"] for item in batch]),
        "sar": torch.stack([item["sar"] for item in batch]),
        "target": torch.stack([item["target"] for item in batch]),
        "id": [str(item["id"]) for item in batch],
    }
