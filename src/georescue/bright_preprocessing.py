from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio


# ImageNet normalization used by the official BRIGHT baseline.
IMAGENET_MEAN = np.array(
    [0.485, 0.456, 0.406],
    dtype=np.float32,
).reshape(3, 1, 1)

IMAGENET_STD = np.array(
    [0.229, 0.224, 0.225],
    dtype=np.float32,
).reshape(3, 1, 1)


class BrightPreprocessingError(RuntimeError):
    """Raised when BRIGHT preprocessing cannot be completed."""


def _validate_spatial_match(
    optical: np.ndarray,
    sar: np.ndarray,
    target: np.ndarray | None = None,
) -> None:
    """Validate that all BRIGHT arrays use the same H/W dimensions."""

    optical_hw = optical.shape[-2:]
    sar_hw = sar.shape[-2:]

    if optical_hw != sar_hw:
        raise BrightPreprocessingError(
            "Optical and SAR spatial dimensions do not match: "
            f"{optical_hw} vs {sar_hw}"
        )

    if target is not None:
        target_hw = target.shape[-2:]

        if optical_hw != target_hw:
            raise BrightPreprocessingError(
                "Image and target spatial dimensions do not match: "
                f"{optical_hw} vs {target_hw}"
            )


def _normalize_rgb(
    image: np.ndarray,
) -> np.ndarray:
    """Apply ImageNet normalization to a 3-channel CHW image."""

    image = image.astype(np.float32, copy=False)

    # BRIGHT DFC images are uint8 in this test release.
    # Convert [0, 255] values to [0, 1] before ImageNet normalization.
    if image.max() > 1.0:
        image = image / 255.0

    image = np.clip(
        image,
        0.0,
        1.0,
    )

    return (
        image - IMAGENET_MEAN
    ) / IMAGENET_STD


def load_bright_modalities(
    optical_path: str | Path,
    sar_path: str | Path,
    target_path: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Load one BRIGHT optical/SAR sample and optionally its target.

    Returns:
        optical: normalized RGB tensor with shape [3, H, W]
        sar: normalized 3-channel tensor with shape [3, H, W]
        target: target mask with shape [H, W] or None
    """

    optical_path = Path(optical_path)
    sar_path = Path(sar_path)

    if not optical_path.exists():
        raise FileNotFoundError(
            f"Optical raster not found: {optical_path}"
        )

    if not sar_path.exists():
        raise FileNotFoundError(
            f"SAR raster not found: {sar_path}"
        )

    try:
        with rasterio.open(optical_path) as optical_src:
            optical = optical_src.read()

        with rasterio.open(sar_path) as sar_src:
            sar = sar_src.read()

        target: np.ndarray | None = None

        if target_path is not None:
            target_path = Path(target_path)

            if not target_path.exists():
                raise FileNotFoundError(
                    f"Target raster not found: {target_path}"
                )

            with rasterio.open(target_path) as target_src:
                target = target_src.read(1)

    except rasterio.errors.RasterioIOError as exc:
        raise BrightPreprocessingError(
            "Unable to read one or more BRIGHT GeoTIFF files."
        ) from exc

    # Official BRIGHT preprocessing uses the first three pre-event bands.
    if optical.ndim != 3 or optical.shape[0] < 3:
        raise BrightPreprocessingError(
            "Expected at least 3 optical bands, "
            f"got shape {optical.shape}"
        )

    optical = optical[:3].astype(
        np.float32,
        copy=False,
    )

    # Expected BRIGHT post-event input is a single SAR channel.
    if sar.ndim != 3 or sar.shape[0] < 1:
        raise BrightPreprocessingError(
            f"Expected at least 1 SAR band, got shape {sar.shape}"
        )

    sar = sar[:1].astype(
        np.float32,
        copy=False,
    )

    # Convert single-channel SAR -> 3-channel representation.
    sar = np.repeat(
        sar,
        repeats=3,
        axis=0,
    )

    _validate_spatial_match(
        optical,
        sar,
        target,
    )

    # Apply the same ImageNet normalization convention to both
    # 3-channel modality tensors used by the official loader.
    optical = _normalize_rgb(optical)
    sar = _normalize_rgb(sar)

    if target is not None:
        target = target.astype(
            np.int64,
            copy=False,
        )

        if target.ndim != 2:
            raise BrightPreprocessingError(
                "Expected target shape [H, W], "
                f"got {target.shape}"
            )

    return optical, sar, target


def combine_modalities(
    optical: np.ndarray,
    sar: np.ndarray,
) -> np.ndarray:
    """Concatenate optical and SAR tensors into a 6-channel tensor."""

    if optical.shape[0] != 3:
        raise BrightPreprocessingError(
            f"Expected 3 optical channels, got {optical.shape}"
        )

    if sar.shape[0] != 3:
        raise BrightPreprocessingError(
            f"Expected 3 SAR channels, got {sar.shape}"
        )

    if optical.shape[1:] != sar.shape[1:]:
        raise BrightPreprocessingError(
            "Optical and SAR tensors have different spatial sizes: "
            f"{optical.shape} vs {sar.shape}"
        )

    combined = np.concatenate(
        [optical, sar],
        axis=0,
    )

    if combined.shape[0] != 6:
        raise BrightPreprocessingError(
            f"Expected 6 combined channels, got {combined.shape}"
        )

    return combined.astype(
        np.float32,
        copy=False,
    )