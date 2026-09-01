from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from georescue.bright_dataset import (
    BrightSegmentationDataset,
)


@dataclass(frozen=True)
class BrightSplitConfig:
    """Paths for the official BRIGHT DFC25 split files."""

    train: Path
    holdout: Path
    validation: Path
    test: Path


@dataclass(frozen=True)
class BrightSplitDatasets:
    """Datasets corresponding to the official DFC25 splits."""

    train: BrightSegmentationDataset
    holdout: BrightSegmentationDataset
    validation: BrightSegmentationDataset
    test: BrightSegmentationDataset


def default_dfc25_split_config(
    split_root: str | Path = "data/splits/dfc25",
) -> BrightSplitConfig:
    """Return the standard DFC25 split-file locations."""

    root = Path(split_root)

    return BrightSplitConfig(
        train=root / "train_set.txt",
        holdout=root / "holdout_set.txt",
        validation=root / "val_set.txt",
        test=root / "test_set.txt",
    )


def create_dfc25_datasets(
    bright_root: str | Path,
    split_root: str | Path = "data/splits/dfc25",
    crop_size: int = 640,
) -> BrightSplitDatasets:
    """Create all official DFC25 dataset views."""

    bright_root = Path(bright_root)

    if not bright_root.is_dir():
        raise FileNotFoundError(
            f"BRIGHT dataset root not found: {bright_root}"
        )

    split_config = default_dfc25_split_config(
        split_root
    )

    for path in (
        split_config.train,
        split_config.holdout,
        split_config.validation,
        split_config.test,
    ):
        if not path.is_file():
            raise FileNotFoundError(
                f"BRIGHT split file not found: {path}"
            )

    return BrightSplitDatasets(
        train=BrightSegmentationDataset(
            root=bright_root,
            split_file=split_config.train,
            crop_size=crop_size,
            training=True,
        ),
        holdout=BrightSegmentationDataset(
            root=bright_root,
            split_file=split_config.holdout,
            crop_size=crop_size,
            training=False,
        ),
        validation=BrightSegmentationDataset(
            root=bright_root,
            split_file=split_config.validation,
            crop_size=crop_size,
            training=False,
        ),
        test=BrightSegmentationDataset(
            root=bright_root,
            split_file=split_config.test,
            crop_size=None,
            training=False,
        ),
    )