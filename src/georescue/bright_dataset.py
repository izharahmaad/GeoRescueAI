from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class BrightSample:
    """Matched BRIGHT pre-event, post-event, and target rasters."""

    sample_id: str
    optical_path: Path
    sar_path: Path
    target_path: Path


class BrightDatasetError(RuntimeError):
    """Raised when the BRIGHT dataset is invalid or incomplete."""


def _index_files(
    directory: Path,
    suffix: str,
) -> dict[str, Path]:
    """Index BRIGHT files using the shared sample identifier."""
    files = sorted(directory.glob(f"*{suffix}"))
    index: dict[str, Path] = {}

    for path in files:
        sample_id = path.name.removesuffix(suffix)

        if not sample_id:
            raise BrightDatasetError(
                f"Unable to derive sample ID from: {path}"
            )

        if sample_id in index:
            raise BrightDatasetError(
                f"Duplicate sample ID: {sample_id}"
            )

        index[sample_id] = path

    return index


def _read_split_ids(
    split_file: str | Path,
) -> list[str]:
    """Read sample IDs from an official BRIGHT split file."""

    split_path = Path(split_file)

    if not split_path.is_file():
        raise BrightDatasetError(
            f"BRIGHT split file does not exist: {split_path}"
        )

    try:
        lines = split_path.read_text(
            encoding="utf-8"
        ).splitlines()
    except OSError as exc:
        raise BrightDatasetError(
            f"Unable to read BRIGHT split file: {split_path}"
        ) from exc

    split_ids = [
        line.strip()
        for line in lines
        if line.strip()
    ]

    if not split_ids:
        raise BrightDatasetError(
            f"BRIGHT split file is empty: {split_path}"
        )

    counts = Counter(split_ids)

    duplicate_ids = sorted(
        sample_id
        for sample_id, count in counts.items()
        if count > 1
    )

    if duplicate_ids:
        preview = duplicate_ids[:10]

        message = (
            "Duplicate sample IDs found in split file: "
            f"{preview}"
        )

        if len(duplicate_ids) > 10:
            message += " ..."

        raise BrightDatasetError(message)

    return split_ids


def discover_bright_samples(
    root: str | Path,
    split_file: str | Path | None = None,
    strict_split: bool = True,
) -> list[BrightSample]:
    """Discover matched BRIGHT samples.

    Args:
        root:
            BRIGHT dataset root containing:
            ``pre-event``, ``post-event``, and ``target``.

        split_file:
            Optional official BRIGHT split file.

        strict_split:
            When True, every ID listed in ``split_file`` must exist
            locally as a fully matched optical/SAR/target sample.

            When False, only IDs that are actually available locally
            are returned, preserving the official split-file order.

    Returns:
        A list of matched BRIGHT samples.
    """

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
        if not directory.is_dir():
            raise BrightDatasetError(
                f"Missing required BRIGHT directory: {directory}"
            )

    optical = _index_files(
        pre_event,
        "_pre_disaster.tif",
    )

    sar = _index_files(
        post_event,
        "_post_disaster.tif",
    )

    targets = _index_files(
        target,
        "_building_damage.tif",
    )

    common_ids = (
        set(optical)
        & set(sar)
        & set(targets)
    )

    if not common_ids:
        raise BrightDatasetError(
            "No matched BRIGHT optical/SAR/target samples found."
        )

    if split_file is None:
        selected_ids = sorted(common_ids)

    else:
        split_ids = _read_split_ids(
            split_file
        )

        missing_ids = [
            sample_id
            for sample_id in split_ids
            if sample_id not in common_ids
        ]

        if strict_split and missing_ids:
            preview = missing_ids[:10]

            message = (
                "Split file contains IDs that are not fully matched "
                f"in the BRIGHT dataset: {preview}"
            )

            if len(missing_ids) > 10:
                message += " ..."

            raise BrightDatasetError(message)

        # In non-strict development mode, use only samples that
        # physically exist while preserving official split order.
        selected_ids = [
            sample_id
            for sample_id in split_ids
            if sample_id in common_ids
        ]

        if not selected_ids:
            raise BrightDatasetError(
                "None of the IDs in the requested BRIGHT split "
                "are available as fully matched local samples."
            )

    return [
        BrightSample(
            sample_id=sample_id,
            optical_path=optical[sample_id],
            sar_path=sar[sample_id],
            target_path=targets[sample_id],
        )
        for sample_id in selected_ids
    ]


def inspect_raster(
    path: str | Path,
) -> dict[str, object]:
    """Inspect raster metadata without loading pixel data."""

    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(path)

    try:
        with rasterio.open(path) as src:
            return {
                "path": str(path),
                "width": src.width,
                "height": src.height,
                "count": src.count,
                "dtype": tuple(src.dtypes),
                "crs": str(src.crs) if src.crs else None,
                "transform": tuple(src.transform),
                "bounds": tuple(src.bounds),
                "nodata": src.nodata,
            }
    except rasterio.errors.RasterioIOError as exc:
        raise BrightDatasetError(
            f"Unable to open raster: {path}"
        ) from exc


def inspect_sample(
    sample: BrightSample,
) -> dict[str, object]:
    """Inspect metadata for one complete BRIGHT sample."""

    return {
        "sample_id": sample.sample_id,
        "optical": inspect_raster(sample.optical_path),
        "sar": inspect_raster(sample.sar_path),
        "target": inspect_raster(sample.target_path),
    }


def validate_sample_geometry(
    sample: BrightSample,
) -> None:
    """Validate dimensions, CRS, transform, and spatial extent.

    Transform and bounds are compared with small absolute tolerances
    because GeoTIFF geospatial metadata uses floating-point values.
    """

    paths = {
        "optical": sample.optical_path,
        "sar": sample.sar_path,
        "target": sample.target_path,
    }

    metadata: dict[str, dict[str, object]] = {}

    for name, path in paths.items():
        metadata[name] = inspect_raster(path)

    reference = metadata["optical"]

    reference_transform = np.asarray(
        reference["transform"],
        dtype=np.float64,
    )

    reference_bounds = np.asarray(
        reference["bounds"],
        dtype=np.float64,
    )

    for name in ("sar", "target"):
        current = metadata[name]

        # Pixel dimensions must match exactly.
        if (
            current["width"] != reference["width"]
            or current["height"] != reference["height"]
        ):
            raise BrightDatasetError(
                f"Dimension mismatch: optical vs {name}"
            )

        # Coordinate reference systems must match exactly.
        if current["crs"] != reference["crs"]:
            raise BrightDatasetError(
                f"CRS mismatch: optical vs {name}"
            )

        current_transform = np.asarray(
            current["transform"],
            dtype=np.float64,
        )

        # Only absolute tolerance is used because projected coordinate
        # values can be large. The BRIGHT sample only differed by
        # approximately 3e-13 in the transform coefficients.
        if not np.allclose(
            current_transform,
            reference_transform,
            rtol=0.0,
            atol=1e-9,
        ):
            raise BrightDatasetError(
                f"Transform mismatch: optical vs {name}"
            )

        current_bounds = np.asarray(
            current["bounds"],
            dtype=np.float64,
        )

        if not np.allclose(
            current_bounds,
            reference_bounds,
            rtol=0.0,
            atol=1e-6,
        ):
            raise BrightDatasetError(
                f"Bounds mismatch: optical vs {name}"
            )


class BrightSegmentationDataset(Dataset):
    """PyTorch dataset for BRIGHT multimodal damage segmentation.

    Each sample returns:

        input:
            6-channel tensor [6, H, W]

        target:
            integer mask [H, W]

        sample_id:
            BRIGHT sample identifier

    Channels:

        0-2: pre-event optical RGB
        3-5: post-event SAR replicated across three channels
    """

    def __init__(
        self,
        root: str | Path,
        samples: list[BrightSample] | None = None,
        split_file: str | Path | None = None,
        strict_split: bool = True,
        crop_size: int | None = 640,
        training: bool = True,
    ) -> None:
        self.root = Path(root)
        self.crop_size = crop_size
        self.training = training
        self.strict_split = strict_split

        if samples is not None and split_file is not None:
            raise ValueError(
                "Provide either samples or split_file, not both."
            )

        if samples is not None:
            self.samples = list(samples)

        else:
            self.samples = discover_bright_samples(
                self.root,
                split_file=split_file,
                strict_split=strict_split,
            )

        if not self.samples:
            raise BrightDatasetError(
                "BrightSegmentationDataset received no samples."
            )

        if crop_size is not None and crop_size <= 0:
            raise ValueError(
                "crop_size must be positive or None."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def _read_sample(
        self,
        sample: BrightSample,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Read one matched optical/SAR/target triplet."""

        try:
            with rasterio.open(sample.optical_path) as src:
                optical = src.read()

            with rasterio.open(sample.sar_path) as src:
                sar = src.read()

            with rasterio.open(sample.target_path) as src:
                target = src.read(1)

        except rasterio.errors.RasterioIOError as exc:
            raise BrightDatasetError(
                f"Unable to read BRIGHT sample: {sample.sample_id}"
            ) from exc

        if optical.ndim != 3 or optical.shape[0] < 3:
            raise BrightDatasetError(
                "Expected at least 3 optical bands, "
                f"got {optical.shape}"
            )

        if sar.ndim != 3 or sar.shape[0] < 1:
            raise BrightDatasetError(
                "Expected at least 1 SAR band, "
                f"got {sar.shape}"
            )

        if target.ndim != 2:
            raise BrightDatasetError(
                "Expected a 2-D target mask, "
                f"got {target.shape}"
            )

        optical = optical[:3].astype(
            np.float32,
            copy=False,
        )

        sar = sar[:1].astype(
            np.float32,
            copy=False,
        )

        target = target.astype(
            np.int64,
            copy=False,
        )

        if optical.shape[1:] != sar.shape[1:]:
            raise BrightDatasetError(
                "Optical and SAR spatial dimensions do not match: "
                f"{optical.shape[1:]} vs {sar.shape[1:]}"
            )

        if optical.shape[1:] != target.shape:
            raise BrightDatasetError(
                "Image and target spatial dimensions do not match: "
                f"{optical.shape[1:]} vs {target.shape}"
            )

        return (
            np.ascontiguousarray(optical),
            np.ascontiguousarray(sar),
            np.ascontiguousarray(target),
        )

    @staticmethod
    def _normalize_modality(
        image: np.ndarray,
    ) -> np.ndarray:
        """Apply the official BRIGHT ImageNet-style normalization."""

        mean = np.array(
            [123.675, 116.28, 103.53],
            dtype=np.float32,
        ).reshape(3, 1, 1)

        std = np.array(
            [58.395, 57.12, 57.375],
            dtype=np.float32,
        ).reshape(3, 1, 1)

        image = image.astype(
            np.float32,
            copy=False,
        )

        return np.ascontiguousarray(
            (image - mean) / std
        )

    @staticmethod
    def _pad(
        optical: np.ndarray,
        sar: np.ndarray,
        target: np.ndarray,
        size: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Pad inputs to at least the requested spatial size."""

        _, height, width = optical.shape

        padded_height = max(
            height,
            size,
        )

        padded_width = max(
            width,
            size,
        )

        optical_pad = np.zeros(
            (3, padded_height, padded_width),
            dtype=np.float32,
        )

        sar_pad = np.zeros(
            (1, padded_height, padded_width),
            dtype=np.float32,
        )

        target_pad = np.zeros(
            (padded_height, padded_width),
            dtype=np.int64,
        )

        optical_pad[
            :,
            :height,
            :width,
        ] = optical

        sar_pad[
            :,
            :height,
            :width,
        ] = sar

        target_pad[
            :height,
            :width,
        ] = target

        return (
            np.ascontiguousarray(optical_pad),
            np.ascontiguousarray(sar_pad),
            np.ascontiguousarray(target_pad),
        )

    def _random_crop(
        self,
        optical: np.ndarray,
        sar: np.ndarray,
        target: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Randomly crop all modalities using identical coordinates."""

        if self.crop_size is None:
            return (
                optical,
                sar,
                target,
            )

        size = self.crop_size

        optical, sar, target = self._pad(
            optical,
            sar,
            target,
            size,
        )

        _, height, width = optical.shape

        max_y = height - size
        max_x = width - size

        y = int(
            np.random.randint(
                0,
                max_y + 1,
            )
        )

        x = int(
            np.random.randint(
                0,
                max_x + 1,
            )
        )

        return (
            np.ascontiguousarray(
                optical[
                    :,
                    y:y + size,
                    x:x + size,
                ]
            ),
            np.ascontiguousarray(
                sar[
                    :,
                    y:y + size,
                    x:x + size,
                ]
            ),
            np.ascontiguousarray(
                target[
                    y:y + size,
                    x:x + size,
                ]
            ),
        )

    @staticmethod
    def _augment(
        optical: np.ndarray,
        sar: np.ndarray,
        target: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply synchronized spatial augmentations.

        Every operation returns contiguous arrays so they can safely
        be converted to PyTorch tensors.
        """

        # Horizontal flip.
        if np.random.random() < 0.5:
            optical = np.flip(
                optical,
                axis=2,
            ).copy()

            sar = np.flip(
                sar,
                axis=2,
            ).copy()

            target = np.flip(
                target,
                axis=1,
            ).copy()

        # Vertical flip.
        if np.random.random() < 0.5:
            optical = np.flip(
                optical,
                axis=1,
            ).copy()

            sar = np.flip(
                sar,
                axis=1,
            ).copy()

            target = np.flip(
                target,
                axis=0,
            ).copy()

        # Rotation by 90 / 180 / 270 degrees.
        if np.random.random() < 0.5:
            k = int(
                np.random.randint(
                    1,
                    4,
                )
            )

            optical = np.rot90(
                optical,
                k=k,
                axes=(1, 2),
            ).copy()

            sar = np.rot90(
                sar,
                k=k,
                axes=(1, 2),
            ).copy()

            target = np.rot90(
                target,
                k=k,
                axes=(0, 1),
            ).copy()

        return (
            np.ascontiguousarray(optical),
            np.ascontiguousarray(sar),
            np.ascontiguousarray(target),
        )

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, str]:
        """Load, augment, normalize, and return one BRIGHT sample."""

        if index < 0 or index >= len(self.samples):
            raise IndexError(
                f"BRIGHT dataset index out of range: {index}"
            )

        sample = self.samples[index]

        optical, sar, target = self._read_sample(
            sample
        )

        if self.training:
            optical, sar, target = self._random_crop(
                optical,
                sar,
                target,
            )

            optical, sar, target = self._augment(
                optical,
                sar,
                target,
            )

        elif self.crop_size is not None:
            optical, sar, target = self._pad(
                optical,
                sar,
                target,
                self.crop_size,
            )

        # Official BRIGHT representation:
        # replicate the single SAR band three times.
        sar = np.repeat(
            sar,
            repeats=3,
            axis=0,
        )

        sar = np.ascontiguousarray(
            sar
        )

        # Apply official normalization.
        optical = self._normalize_modality(
            optical
        )

        sar = self._normalize_modality(
            sar
        )

        # RGB (3) + SAR (3) = 6 channels.
        combined = np.concatenate(
            [
                optical,
                sar,
            ],
            axis=0,
        )

        combined = np.ascontiguousarray(
            combined,
            dtype=np.float32,
        )

        target = np.ascontiguousarray(
            target,
            dtype=np.int64,
        )

        if combined.shape[0] != 6:
            raise BrightDatasetError(
                f"Expected 6 input channels, "
                f"got {combined.shape}"
            )

        if combined.shape[1:] != target.shape:
            raise BrightDatasetError(
                "Input and target spatial dimensions "
                "do not match after preprocessing: "
                f"{combined.shape[1:]} vs {target.shape}"
            )

        return (
            torch.from_numpy(combined),
            torch.from_numpy(target),
            sample.sample_id,
        )


def iter_bright_samples(
    root: str | Path,
    split_file: str | Path | None = None,
    strict_split: bool = True,
) -> Iterator[BrightSample]:
    """Yield BRIGHT samples, optionally restricted to a split file."""

    yield from discover_bright_samples(
        root,
        split_file=split_file,
        strict_split=strict_split,
    )