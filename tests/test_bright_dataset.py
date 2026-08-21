from __future__ import annotations

from pathlib import Path

import pytest

from georescue.bright_dataset import (
    BrightDatasetError,
    discover_bright_samples,
)


def test_bright_empty_dataset_raises(tmp_path: Path) -> None:
    root = tmp_path / "BRIGHT"

    (root / "pre-event").mkdir(parents=True)
    (root / "post-event").mkdir()
    (root / "target").mkdir()

    with pytest.raises(BrightDatasetError):
        discover_bright_samples(root)


def test_bright_sample_matching(tmp_path: Path) -> None:
    root = tmp_path / "BRIGHT"

    pre_event = root / "pre-event"
    post_event = root / "post-event"
    target = root / "target"

    pre_event.mkdir(parents=True)
    post_event.mkdir()
    target.mkdir()

    sample_id = "example_00000000"

    (pre_event / f"{sample_id}_pre_disaster.tif").touch()
    (post_event / f"{sample_id}_post_disaster.tif").touch()
    (target / f"{sample_id}_building_damage.tif").touch()

    samples = discover_bright_samples(root)

    assert len(samples) == 1
    assert samples[0].sample_id == sample_id
    assert samples[0].optical_path.name == (
        f"{sample_id}_pre_disaster.tif"
    )
    assert samples[0].sar_path.name == (
        f"{sample_id}_post_disaster.tif"
    )
    assert samples[0].target_path.name == (
        f"{sample_id}_building_damage.tif"
    )