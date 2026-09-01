from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from rasterio.transform import from_origin
import torch

from georescue.bright_dataset import (
    BrightDatasetError,
    BrightSegmentationDataset,
    discover_bright_samples,
)


def _create_raster(
    path: Path,
    array: np.ndarray,
) -> None:
    """Create a small GeoTIFF fixture for unit testing."""
    import rasterio

    if array.ndim == 2:
        count = 1
        height, width = array.shape
    elif array.ndim == 3:
        count, height, width = array.shape
    else:
        raise ValueError(
            f"Expected 2-D or 3-D array, got shape {array.shape}"
        )

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=count,
        dtype=array.dtype,
        crs="EPSG:26913",
        transform=from_origin(
            0.0,
            100.0,
            1.0,
            1.0,
        ),
    ) as dst:
        if array.ndim == 2:
            dst.write(array, 1)
        else:
            dst.write(array)


def test_bright_empty_dataset_raises(
    tmp_path: Path,
) -> None:
    """An empty BRIGHT dataset should fail clearly."""
    root = tmp_path / "BRIGHT"

    (root / "pre-event").mkdir(parents=True)
    (root / "post-event").mkdir()
    (root / "target").mkdir()

    with pytest.raises(BrightDatasetError):
        discover_bright_samples(root)


def test_bright_sample_matching(
    tmp_path: Path,
) -> None:
    """Matching filenames should produce one BrightSample."""
    root = tmp_path / "BRIGHT"

    pre_event = root / "pre-event"
    post_event = root / "post-event"
    target = root / "target"

    pre_event.mkdir(parents=True)
    post_event.mkdir()
    target.mkdir()

    sample_id = "example_00000000"

    _create_raster(
        pre_event / f"{sample_id}_pre_disaster.tif",
        np.zeros(
            (3, 8, 8),
            dtype=np.uint8,
        ),
    )

    _create_raster(
        post_event / f"{sample_id}_post_disaster.tif",
        np.zeros(
            (1, 8, 8),
            dtype=np.uint8,
        ),
    )

    _create_raster(
        target / f"{sample_id}_building_damage.tif",
        np.zeros(
            (8, 8),
            dtype=np.uint8,
        ),
    )

    samples = discover_bright_samples(root)

    assert len(samples) == 1

    sample = samples[0]

    assert sample.sample_id == sample_id
    assert sample.optical_path.name == (
        f"{sample_id}_pre_disaster.tif"
    )
    assert sample.sar_path.name == (
        f"{sample_id}_post_disaster.tif"
    )
    assert sample.target_path.name == (
        f"{sample_id}_building_damage.tif"
    )


def test_bright_dataset_returns_six_channels(
    tmp_path: Path,
) -> None:
    """The dataset should return a 6-channel multimodal tensor."""
    root = tmp_path / "BRIGHT"

    pre_event = root / "pre-event"
    post_event = root / "post-event"
    target = root / "target"

    pre_event.mkdir(parents=True)
    post_event.mkdir()
    target.mkdir()

    sample_id = "example_00000001"

    _create_raster(
        pre_event / f"{sample_id}_pre_disaster.tif",
        np.full(
            (3, 16, 16),
            100,
            dtype=np.uint8,
        ),
    )

    _create_raster(
        post_event / f"{sample_id}_post_disaster.tif",
        np.full(
            (1, 16, 16),
            50,
            dtype=np.uint8,
        ),
    )

    target_array = np.zeros(
        (16, 16),
        dtype=np.uint8,
    )

    target_array[4:8, 4:8] = 1

    _create_raster(
        target / f"{sample_id}_building_damage.tif",
        target_array,
    )

    # Evaluation mode preserves the complete raster.
    dataset = BrightSegmentationDataset(
        root=root,
        crop_size=None,
        training=False,
    )

    image, mask, returned_id = dataset[0]

    assert image.shape == (6, 16, 16)
    assert mask.shape == (16, 16)

    # The dataset returns PyTorch tensors.
    assert image.dtype == torch.float32
    assert mask.dtype == torch.int64

    assert returned_id == sample_id


def test_bright_training_crop(
    tmp_path: Path,
) -> None:
    """Training mode should produce the requested crop size."""
    root = tmp_path / "BRIGHT"

    pre_event = root / "pre-event"
    post_event = root / "post-event"
    target = root / "target"

    pre_event.mkdir(parents=True)
    post_event.mkdir()
    target.mkdir()

    sample_id = "example_00000002"

    _create_raster(
        pre_event / f"{sample_id}_pre_disaster.tif",
        np.full(
            (3, 32, 32),
            100,
            dtype=np.uint8,
        ),
    )

    _create_raster(
        post_event / f"{sample_id}_post_disaster.tif",
        np.full(
            (1, 32, 32),
            50,
            dtype=np.uint8,
        ),
    )

    target_array = np.zeros(
        (32, 32),
        dtype=np.uint8,
    )

    target_array[8:16, 8:16] = 1

    _create_raster(
        target / f"{sample_id}_building_damage.tif",
        target_array,
    )

    dataset = BrightSegmentationDataset(
        root=root,
        crop_size=16,
        training=True,
    )

    image, mask, returned_id = dataset[0]

    assert image.shape == (6, 16, 16)
    assert mask.shape == (16, 16)

    assert image.dtype == torch.float32
    assert mask.dtype == torch.int64

    assert returned_id == sample_id