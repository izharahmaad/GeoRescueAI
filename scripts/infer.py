"""Run single-sample inference for NPZ or image-compatible development inputs."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from georescue.config import load_config
from georescue.models import build_model
from georescue.train_utils import load_checkpoint, select_device
from georescue.visualization import save_mask, save_overlay


def load_array(path: Path, channels: int) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        array = np.load(path)
    elif path.suffix.lower() == ".npz":
        with np.load(path) as data:
            key = "optical" if channels == 3 else "sar"
            if key not in data:
                raise KeyError(f"{path} must contain '{key}'")
            array = data[key]
    else:
        raise ValueError("Development inference currently accepts .npy or .npz arrays")
    if array.ndim == 2:
        array = array[None, ...]
    elif array.ndim == 3 and array.shape[-1] == channels:
        array = np.moveaxis(array, -1, 0)
    if array.shape[0] != channels:
        raise ValueError(f"Expected {channels} channels, got {array.shape}")
    return array.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--optical", required=True)
    parser.add_argument("--sar", required=True)
    parser.add_argument("--output", default="outputs/predictions/prediction.png")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = select_device(cfg.training.device)
    model = build_model(cfg.model.name, cfg.data.optical_channels, cfg.data.sar_channels, cfg.data.num_classes, cfg.model.base_channels, cfg.model.dropout).to(device)
    checkpoint = Path(args.checkpoint) if args.checkpoint else ROOT / cfg.training.checkpoint_dir / cfg.training.best_checkpoint_name
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    load_checkpoint(checkpoint, model, device)
    model.eval()

    optical = load_array(Path(args.optical), cfg.data.optical_channels)
    sar = load_array(Path(args.sar), cfg.data.sar_channels)
    if optical.shape[1:] != sar.shape[1:]:
        raise ValueError("Optical and SAR spatial dimensions must match")

    with torch.no_grad():
        optical_t = torch.from_numpy(optical[None]).to(device)
        sar_t = torch.from_numpy(sar[None]).to(device)
        if hasattr(model, "optical_stem"):
            logits = model(optical_t, sar_t)
        elif getattr(model, "_is_sar_model", False):
            logits = model(sar_t)
        else:
            logits = model(optical_t)
        mask = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

    output = ROOT / args.output if not Path(args.output).is_absolute() else Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_mask(mask, output)
    save_overlay(optical, mask, output.with_name(output.stem + "_overlay.png"))
    print(f"Saved mask: {output.resolve()}")


if __name__ == "__main__":
    main()
