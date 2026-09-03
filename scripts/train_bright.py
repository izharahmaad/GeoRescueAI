"""BRIGHT DFC25 training and smoke-test entry point."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from torch.utils.data import DataLoader

from georescue.bright_dataset import (
    BrightSegmentationDataset,
    BrightDatasetError,
)
from georescue.bright_splits import (
    default_dfc25_split_config,
)
from georescue.config import load_config
from georescue.models import build_model
from georescue.train_utils import (
    SegmentationRunner,
    save_checkpoint,
    seed_everything,
    select_device,
    write_json,
)


BRIGHT_CROP_SIZE = 640


def bright_collate(
    batch: list[tuple[torch.Tensor, torch.Tensor, str]],
) -> dict[str, Any]:
    """Convert BRIGHT samples into the existing runner format.

    Dataset sample:

        image     [6, H, W]
        target    [H, W]
        sample_id string

    Returned batch:

        optical   [B, 3, H, W]
        sar       [B, 3, H, W]
        target    [B, H, W]
    """

    if not batch:
        raise ValueError(
            "Cannot collate an empty BRIGHT batch."
        )

    images, targets, sample_ids = zip(*batch)

    image_batch = torch.stack(
        images,
        dim=0,
    )

    target_batch = torch.stack(
        targets,
        dim=0,
    )

    if image_batch.ndim != 4:
        raise ValueError(
            "Expected BRIGHT image batch with shape "
            f"[B, 6, H, W], got {tuple(image_batch.shape)}"
        )

    if image_batch.shape[1] != 6:
        raise ValueError(
            "Expected six BRIGHT input channels, "
            f"got {image_batch.shape[1]}"
        )

    if target_batch.ndim != 3:
        raise ValueError(
            "Expected target batch with shape "
            f"[B, H, W], got {tuple(target_batch.shape)}"
        )

    return {
        "optical": image_batch[:, :3],
        "sar": image_batch[:, 3:6],
        "target": target_batch,
        "sample_ids": list(sample_ids),
    }


def build_bright_loader(
    dataset: BrightSegmentationDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    """Create a DataLoader for BRIGHT."""

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=bright_collate,
    )


def inspect_batch(
    loader: DataLoader,
) -> dict[str, Any]:
    """Inspect and validate one BRIGHT batch."""

    batch = next(iter(loader))

    print()
    print("Batch inspection")
    print("-" * 60)

    print(
        "Optical:",
        tuple(batch["optical"].shape),
        batch["optical"].dtype,
    )

    print(
        "SAR:",
        tuple(batch["sar"].shape),
        batch["sar"].dtype,
    )

    print(
        "Target:",
        tuple(batch["target"].shape),
        batch["target"].dtype,
    )

    print(
        "Sample IDs:",
        batch["sample_ids"],
    )

    if batch["optical"].shape[1] != 3:
        raise RuntimeError(
            "BRIGHT optical input must have 3 channels."
        )

    if batch["sar"].shape[1] != 3:
        raise RuntimeError(
            "BRIGHT SAR input must have 3 channels."
        )

    return batch


def run_smoke_test(
    bright_root: Path,
    split_root: Path,
    device: torch.device,
) -> None:
    """Run one complete real-data training step."""

    split_config = default_dfc25_split_config(
        split_root
    )

    dataset = BrightSegmentationDataset(
        root=bright_root,
        split_file=split_config.test,
        strict_split=False,
        crop_size=None,
        training=False,
    )

    if len(dataset) == 0:
        raise RuntimeError(
            "No locally available BRIGHT test samples were found."
        )

    loader = build_bright_loader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        device=device,
    )

    batch = inspect_batch(
        loader
    )

    image = batch["optical"].new_empty(
        (
            batch["optical"].shape[0],
            6,
            batch["optical"].shape[-2],
            batch["optical"].shape[-1],
        )
    )

    image[:, :3] = batch["optical"]
    image[:, 3:6] = batch["sar"]

    target = batch["target"].to(
        device,
        non_blocking=True,
    )

    model = build_model(
        name="fusion_unet",
        optical_channels=3,
        sar_channels=3,
        num_classes=2,
        base_channels=16,
        dropout=0.0,
    )

    runner = SegmentationRunner(
        model=model,
        device=device,
        num_classes=2,
        amp=False,
    )

    optimizer = torch.optim.AdamW(
        runner.model.parameters(),
        lr=1e-4,
    )

    runner.model.train()

    optimizer.zero_grad(
        set_to_none=True
    )

    logits = runner._forward(
        batch
    )

    print()
    print(
        "Logits:",
        tuple(logits.shape),
    )

    if logits.shape[0] != target.shape[0]:
        raise RuntimeError(
            "Batch dimension mismatch between logits and target."
        )

    if logits.shape[-2:] != target.shape[-2:]:
        raise RuntimeError(
            "Spatial dimension mismatch between logits and target."
        )

    loss = runner.criterion(
        logits,
        target,
    )

    print(
        "Loss:",
        float(loss.detach()),
    )

    if not torch.isfinite(loss):
        raise RuntimeError(
            "Training loss is not finite."
        )

    loss.backward()

    gradient_count = sum(
        1
        for parameter in runner.model.parameters()
        if parameter.grad is not None
    )

    if gradient_count == 0:
        raise RuntimeError(
            "No model gradients were produced."
        )

    optimizer.step()

    print(
        "Parameters with gradients:",
        gradient_count,
    )

    print()
    print(
        "BRIGHT real-data training smoke test: PASS"
    )


def run_training(
    cfg: Any,
    bright_root: Path,
    split_root: Path,
    device: torch.device,
    smoke_test: bool,
) -> None:
    """Train the BRIGHT model when the full dataset is available."""

    split_config = default_dfc25_split_config(
        split_root
    )

    train_dataset = BrightSegmentationDataset(
        root=bright_root,
        split_file=split_config.train,
        strict_split=True,
        crop_size=BRIGHT_CROP_SIZE,
        training=True,
    )

    holdout_dataset = BrightSegmentationDataset(
        root=bright_root,
        split_file=split_config.holdout,
        strict_split=True,
        crop_size=BRIGHT_CROP_SIZE,
        training=False,
    )

    train_loader = build_bright_loader(
        dataset=train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        device=device,
    )

    holdout_loader = build_bright_loader(
        dataset=holdout_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        device=device,
    )

    model = build_model(
        name=cfg.model.name,
        optical_channels=cfg.data.optical_channels,
        sar_channels=cfg.data.sar_channels,
        num_classes=cfg.data.num_classes,
        base_channels=cfg.model.base_channels,
        dropout=cfg.model.dropout,
    )

    runner = SegmentationRunner(
        model=model,
        device=device,
        num_classes=cfg.data.num_classes,
        amp=cfg.training.amp,
    )

    optimizer = torch.optim.AdamW(
        runner.model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.training.epochs,
    )

    epochs = (
        min(
            cfg.training.epochs,
            1,
        )
        if smoke_test
        else cfg.training.epochs
    )

    best_miou = -1.0
    history: list[dict[str, Any]] = []

    for epoch in range(
        1,
        epochs + 1,
    ):
        train_loss = runner.train_epoch(
            train_loader,
            optimizer,
        )

        holdout_loss, metrics = runner.evaluate(
            holdout_loader
        )

        scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "holdout_loss": holdout_loss,
            **metrics,
        }

        history.append(row)

        print(
            f"Epoch {epoch}/{epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"holdout_loss={holdout_loss:.4f} | "
            f"mIoU={metrics['miou']:.4f} | "
            f"Dice={metrics['dice']:.4f}"
        )

        if metrics["miou"] > best_miou:
            best_miou = float(
                metrics["miou"]
            )

            save_checkpoint(
                cfg.training.checkpoint_dir
                / cfg.training.best_checkpoint_name,
                runner.model,
                optimizer,
                epoch,
                best_miou,
                {
                    "config": str(
                        "configs/bright.yaml"
                    ),
                    "dataset": "BRIGHT DFC25",
                    "model_name": cfg.model.name,
                },
            )

    write_json(
        ROOT
        / "outputs/experiments/bright_train_history.json",
        {
            "device": str(device),
            "history": history,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GeoRescue AI BRIGHT DFC25 trainer."
    )

    parser.add_argument(
        "--config",
        default="configs/bright.yaml",
    )

    parser.add_argument(
        "--bright-root",
        required=True,
        help="Path to the BRIGHT dataset root.",
    )

    parser.add_argument(
        "--split-root",
        default="data/splits/dfc25",
        help="Path to DFC25 split files.",
    )

    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run one real-data training step only.",
    )

    args = parser.parse_args()

    cfg = load_config(
        args.config
    )

    seed_everything(
        cfg.training.seed
    )

    device = select_device(
        cfg.training.device
    )

    bright_root = Path(
        args.bright_root
    )

    split_root = Path(
        args.split_root
    )

    print(
        "GeoRescue AI - BRIGHT DFC25"
    )
    print("=" * 60)

    print(
        "Device:",
        device,
    )

    print(
        "BRIGHT root:",
        bright_root,
    )

    if not bright_root.is_dir():
        raise SystemExit(
            f"BRIGHT dataset root not found: {bright_root}"
        )

    try:
        if args.smoke_test:
            run_smoke_test(
                bright_root=bright_root,
                split_root=split_root,
                device=device,
            )
        else:
            run_training(
                cfg=cfg,
                bright_root=bright_root,
                split_root=split_root,
                device=device,
                smoke_test=False,
            )

    except BrightDatasetError as exc:
        raise SystemExit(
            f"BRIGHT dataset error:\n{exc}"
        ) from exc


if __name__ == "__main__":
    main()