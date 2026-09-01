from __future__ import annotations

from pathlib import Path

from georescue.bright_splits import (
    default_dfc25_split_config,
)


def test_default_dfc25_split_config(
    tmp_path: Path,
) -> None:
    config = default_dfc25_split_config(
        tmp_path
    )

    assert config.train == (
        tmp_path / "train_set.txt"
    )

    assert config.holdout == (
        tmp_path / "holdout_set.txt"
    )

    assert config.validation == (
        tmp_path / "val_set.txt"
    )

    assert config.test == (
        tmp_path / "test_set.txt"
    )