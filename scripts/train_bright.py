from __future__ import annotations

import argparse
from pathlib import Path

import torch

from georescue.bright_dataset import (
    BrightDatasetError,
    BrightSegmentationDataset,
)
from georescue.bright_splits import (
    default_dfc25_split_config,
)


def inspect_dataset(
    name: str,
    dataset: BrightSegmentationDataset,
) -> None:
    """Inspect one BRIGHT dataset split."""

    print(f"\n{name}")
    print("-" * 60)

    print("Samples:", len(dataset))
    print("Training:", dataset.training)
    print("Crop size:", dataset.crop_size)

    image, target, sample_id = dataset[0]

    print("First sample:", sample_id)
    print("Image shape:", tuple(image.shape))
    print("Image dtype:", image.dtype)
    print("Target shape:", tuple(target.shape))
    print("Target dtype:", target.dtype)

    print(
        "Image range:",
        float(image.min()),
        "to",
        float(image.max()),
    )

    print(
        "Target classes:",
        torch.unique(target).tolist(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect BRIGHT DFC25 training configuration."
    )

    parser.add_argument(
        "--bright-root",
        required=True,
        help="Path to BRIGHT dataset root.",
    )

    parser.add_argument(
        "--split-root",
        default="data/splits/dfc25",
        help="Path to DFC25 split files.",
    )

    parser.add_argument(
        "--crop-size",
        type=int,
        default=640,
        help="Training crop size.",
    )

    args = parser.parse_args()

    bright_root = Path(args.bright_root)
    split_root = Path(args.split_root)

    if not bright_root.is_dir():
        raise SystemExit(
            f"BRIGHT root not found: {bright_root}"
        )

    split_config = default_dfc25_split_config(
        split_root
    )

    print("GeoRescue AI — BRIGHT DFC25 dataset inspection")
    print("=" * 60)

    print("BRIGHT root:", bright_root)
    print("Split root:", split_root)

    try:
        train_dataset = BrightSegmentationDataset(
            root=bright_root,
            split_file=split_config.train,
            crop_size=args.crop_size,
            training=True,
        )

        holdout_dataset = BrightSegmentationDataset(
            root=bright_root,
            split_file=split_config.holdout,
            crop_size=args.crop_size,
            training=False,
        )

        validation_dataset = BrightSegmentationDataset(
            root=bright_root,
            split_file=split_config.validation,
            crop_size=args.crop_size,
            training=False,
        )

        test_dataset = BrightSegmentationDataset(
            root=bright_root,
            split_file=split_config.test,
            crop_size=None,
            training=False,
        )

    except (
        BrightDatasetError,
        FileNotFoundError,
    ) as exc:
        raise SystemExit(
            f"BRIGHT dataset inspection failed:\n{exc}"
        ) from exc

    inspect_dataset(
        "TRAIN",
        train_dataset,
    )

    inspect_dataset(
        "HOLDOUT",
        holdout_dataset,
    )

    inspect_dataset(
        "VALIDATION",
        validation_dataset,
    )

    inspect_dataset(
        "TEST",
        test_dataset,
    )

    print("\nBRIGHT dataset inspection completed.")


if __name__ == "__main__":
    main()