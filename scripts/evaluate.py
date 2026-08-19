"""Evaluate a trained GeoRescue AI model on a held-out split."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from torch.utils.data import DataLoader

from georescue.config import load_config
from georescue.datasets import NPZMultiModalDataset, collate_keep_ids
from georescue.models import build_model
from georescue.train_utils import SegmentationRunner, load_checkpoint, select_device, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = select_device(cfg.training.device)
    test_dir = ROOT / cfg.data.root / cfg.data.test_dir
    dataset = NPZMultiModalDataset(test_dir, cfg.data.optical_channels, cfg.data.sar_channels)
    loader = DataLoader(dataset, batch_size=cfg.training.batch_size, shuffle=False, num_workers=cfg.training.num_workers, collate_fn=collate_keep_ids)
    model = build_model(cfg.model.name, cfg.data.optical_channels, cfg.data.sar_channels, cfg.data.num_classes, cfg.model.base_channels, cfg.model.dropout)
    runner = SegmentationRunner(model, device, cfg.data.num_classes, cfg.training.amp)

    checkpoint = Path(args.checkpoint) if args.checkpoint else ROOT / cfg.training.checkpoint_dir / cfg.training.best_checkpoint_name
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    load_checkpoint(checkpoint, model, device)
    loss, metrics = runner.evaluate(loader)
    result = {"checkpoint": str(checkpoint), "loss": loss, **metrics}
    print(f"loss={loss:.4f}")
    print(f"mIoU={metrics['miou']:.4f}")
    print(f"Dice/F1={metrics['dice']:.4f}")
    print(f"Precision={metrics['precision']:.4f}")
    print(f"Recall={metrics['recall']:.4f}")
    print(f"Per-class IoU={metrics['per_class_iou']}")
    write_json(ROOT / "outputs/experiments/evaluation.json", result)


if __name__ == "__main__":
    main()
