"""Train a GeoRescue AI segmentation baseline."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from torch.utils.data import DataLoader

from georescue.config import load_config
from georescue.datasets import NPZMultiModalDataset, collate_keep_ids
from georescue.models import build_model
from georescue.train_utils import SegmentationRunner, save_checkpoint, seed_everything, select_device, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg.training.seed)
    device = select_device(cfg.training.device)

    train_dir = ROOT / cfg.data.root / cfg.data.train_dir
    val_dir = ROOT / cfg.data.root / cfg.data.val_dir
    train_ds = NPZMultiModalDataset(train_dir, cfg.data.optical_channels, cfg.data.sar_channels)
    val_ds = NPZMultiModalDataset(val_dir, cfg.data.optical_channels, cfg.data.sar_channels)
    loader_kwargs = dict(batch_size=cfg.training.batch_size, num_workers=cfg.training.num_workers, pin_memory=device.type == "cuda", collate_fn=collate_keep_ids)
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    model = build_model(cfg.model.name, cfg.data.optical_channels, cfg.data.sar_channels, cfg.data.num_classes, cfg.model.base_channels, cfg.model.dropout)
    runner = SegmentationRunner(model, device, cfg.data.num_classes, cfg.training.amp)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.training.epochs)

    epochs = min(cfg.training.epochs, 2) if args.smoke_test else cfg.training.epochs
    best_miou = -1.0
    history = []
    for epoch in range(1, epochs + 1):
        train_loss = runner.train_epoch(train_loader, optimizer)
        val_loss, metrics = runner.evaluate(val_loader)
        scheduler.step()
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, **metrics}
        history.append(row)
        print(f"Epoch {epoch}/{epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | mIoU={metrics['miou']:.4f} | Dice={metrics['dice']:.4f}")
        if metrics["miou"] > best_miou:
            best_miou = float(metrics["miou"])
            save_checkpoint(
                cfg.training.checkpoint_dir / cfg.training.best_checkpoint_name,
                model,
                optimizer,
                epoch,
                best_miou,
                {"config": str(args.config), "model_name": cfg.model.name},
            )

    write_json(ROOT / "outputs/experiments/train_history.json", {"device": str(device), "history": history})
    print(f"Best checkpoint: {(ROOT / cfg.training.checkpoint_dir / cfg.training.best_checkpoint_name).resolve()}")


if __name__ == "__main__":
    main()
