"""Training helpers: reproducibility, device selection, checkpoints, and loops."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .metrics import confusion_matrix, metrics_from_confusion


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def save_checkpoint(path: str | Path, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, score: float, meta: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "score": score,
        "meta": meta,
    }, destination)


def load_checkpoint(path: str | Path, model: nn.Module, device: torch.device, optimizer: torch.optim.Optimizer | None = None) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location=device)
    model.load_state_dict(payload["model"])
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    return payload


class SegmentationRunner:
    def __init__(self, model: nn.Module, device: torch.device, num_classes: int, amp: bool = True) -> None:
        self.model = model.to(device)
        self.device = device
        self.num_classes = num_classes
        self.amp_enabled = bool(amp and device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)
        self.criterion = nn.CrossEntropyLoss()

    def _forward(self, batch: dict[str, Any]) -> torch.Tensor:
        optical = batch["optical"].to(self.device, non_blocking=True)
        sar = batch["sar"].to(self.device, non_blocking=True)
        if hasattr(self.model, "optical_stem"):
            return self.model(optical, sar)
        if getattr(self.model, "_is_sar_model", False):
            return self.model(sar)
        return self.model(optical)

    def train_epoch(self, loader: DataLoader, optimizer: torch.optim.Optimizer) -> float:
        self.model.train()
        total_loss = 0.0
        total_items = 0
        for batch in loader:
            target = batch["target"].to(self.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.amp_enabled):
                logits = self._forward(batch)
                loss = self.criterion(logits, target)
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()
            size = target.shape[0]
            total_loss += float(loss.detach()) * size
            total_items += size
        return total_loss / max(total_items, 1)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> tuple[float, dict[str, float | list[float]]]:
        self.model.eval()
        total_loss = 0.0
        total_items = 0
        cm = torch.zeros((self.num_classes, self.num_classes), dtype=torch.int64)
        for batch in loader:
            target = batch["target"].to(self.device, non_blocking=True)
            logits = self._forward(batch)
            loss = self.criterion(logits, target)
            prediction = logits.argmax(dim=1)
            cm += confusion_matrix(prediction.cpu(), target.cpu(), self.num_classes)
            size = target.shape[0]
            total_loss += float(loss) * size
            total_items += size
        return total_loss / max(total_items, 1), metrics_from_confusion(cm)


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
