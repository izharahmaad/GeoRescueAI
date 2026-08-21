from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import rasterio


@dataclass(frozen=True)
class BrightSample:
    """One matched BRIGHT optical, SAR, and damage-target sample."""

    sample_id: str
    optical_path: Path
    sar_path: Path
    target_path: Path


class BrightDatasetError(RuntimeError):
    """Raised when the BRIGHT dataset structure is invalid."""


def _index_files(
    directory: Path,
    suffix: str,
) -> dict[str, Path]:
    """Index files by their shared BRIGHT sample ID."""
    files = sorted(directory.glob(f"*{suffix}"))

    index: dict[str, Path] = {}

    for path in files:
        sample_id = path.name.removesuffix(suffix)

        if not sample_id:
            raise BrightDatasetError(
                f"Could not derive a sample ID from: {path}"
            )

        if sample_id in index:
            raise BrightDatasetError(
                f"Duplicate sample ID in {directory}: {sample_id}"
            )

        index[sample_id] = path

    return index


def discover_bright_samples(
    root: str | Path,
) -> list[BrightSample]:
    """Discover matched optical/SAR/target samples in BRIGHT."""

    root = Path(root)

    pre_event = root / "pre-event"
    post_event = root / "post-event"
    target = root / "target"

    required_directories = (
        pre_event,
        post_event,
        target,
    )

    for directory in required_directories:
        if not directory.exists():
            raise BrightDatasetError(
                f"Required BRIGHT directory does not exist: {directory}"
            )

        if not directory.is_dir():
            raise BrightDatasetError(
                f"Expected a directory but found: {directory}"
            )

    optical_index = _index_files(
        pre_event,
        "_pre_disaster.tif",
    )

    sar_index = _index_files(
        post_event,
        "_post_disaster.tif",
    )

    target_index = _index_files(
        target,
        "_building_damage.tif",
    )

    common_ids = (
        set(optical_index)
        & set(sar_index)
        & set(target_index)
    )

    if not common_ids:
        raise BrightDatasetError(
            "No matched optical/SAR/target BRIGHT samples were found."
        )

    samples = [
        BrightSample(
            sample_id=sample_id,
            optical_path=optical_index[sample_id],
            sar_path=sar_index[sample_id],
            target_path=target_index[sample_id],
        )
        for sample_id in sorted(common_ids)
    ]

    return samples


def inspect_raster(
    path: str | Path,
) -> dict[str, object]:
    """Inspect a BRIGHT GeoTIFF without loading its pixel data."""

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Raster file not found: {path}"
        )

    if not path.is_file():
        raise ValueError(
            f"Expected a file but found: {path}"
        )

    try:
        with rasterio.open(path) as dataset:
            return {
                "path": str(path),
                "width": dataset.width,
                "height": dataset.height,
                "count": dataset.count,
                "dtype": tuple(dataset.dtypes),
                "crs": (
                    str(dataset.crs)
                    if dataset.crs is not None
                    else None
                ),
                "transform": tuple(dataset.transform),
                "bounds": tuple(dataset.bounds),
                "nodata": dataset.nodata,
            }
    except rasterio.errors.RasterioIOError as exc:
        raise BrightDatasetError(
            f"Unable to read GeoTIFF: {path}"
        ) from exc


def inspect_sample(
    sample: BrightSample,
) -> dict[str, object]:
    """Inspect all three rasters belonging to one BRIGHT sample."""

    return {
        "sample_id": sample.sample_id,
        "optical": inspect_raster(
            sample.optical_path
        ),
        "sar": inspect_raster(
            sample.sar_path
        ),
        "target": inspect_raster(
            sample.target_path
        ),
    }