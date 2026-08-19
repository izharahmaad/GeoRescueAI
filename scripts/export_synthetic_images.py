from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def normalize_to_uint8(array: np.ndarray) -> np.ndarray:
    """Normalize a NumPy array to the uint8 range [0, 255]."""
    array = array.astype(np.float32)

    min_value = float(array.min())
    max_value = float(array.max())

    if max_value <= min_value:
        return np.zeros_like(array, dtype=np.uint8)

    normalized = (array - min_value) / (max_value - min_value)

    return (normalized * 255.0).clip(0, 255).astype(np.uint8)


def export_sample(sample_path: Path, output_dir: Path) -> None:
    """Export one synthetic NPZ sample into browser-friendly PNG images."""
    if not sample_path.exists():
        raise FileNotFoundError(f"Sample file not found: {sample_path}")

    data = np.load(sample_path)

    required_keys = {"optical", "sar", "target"}
    missing = required_keys.difference(data.files)

    if missing:
        raise ValueError(
            f"Missing required arrays in {sample_path}: {sorted(missing)}"
        )

    optical = data["optical"]
    sar = data["sar"]
    target = data["target"]

    # Validate optical input.
    if optical.ndim != 3 or optical.shape[0] != 3:
        raise ValueError(
            f"Expected optical shape (3,H,W), got {optical.shape}"
        )

    # Validate SAR input.
    if sar.ndim != 3 or sar.shape[0] != 1:
        raise ValueError(
            f"Expected SAR shape (1,H,W), got {sar.shape}"
        )

    # Validate target mask.
    if target.ndim != 2:
        raise ValueError(
            f"Expected target shape (H,W), got {target.shape}"
        )

    height, width = target.shape

    if optical.shape[1:] != (height, width):
        raise ValueError(
            "Optical dimensions do not match target dimensions."
        )

    if sar.shape[1:] != (height, width):
        raise ValueError(
            "SAR dimensions do not match target dimensions."
        )

    # Convert optical CHW -> HWC for RGB PNG.
    optical_image = np.transpose(optical, (1, 2, 0))
    optical_image = normalize_to_uint8(optical_image)

    # Convert single-channel SAR to grayscale PNG.
    sar_image = normalize_to_uint8(sar[0])

    # Convert target mask to binary grayscale PNG.
    target_image = (target > 0).astype(np.uint8) * 255

    output_dir.mkdir(parents=True, exist_ok=True)

    sample_name = sample_path.stem

    # Pillow infers the correct image mode from the array shape/dtype.
    Image.fromarray(optical_image).save(
        output_dir / f"{sample_name}_optical.png"
    )

    Image.fromarray(sar_image).save(
        output_dir / f"{sample_name}_sar.png"
    )

    Image.fromarray(target_image).save(
        output_dir / f"{sample_name}_target.png"
    )

    print(f"Exported demo images to: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export a synthetic GeoRescue NPZ sample "
            "to browser-friendly PNG fixtures."
        )
    )

    parser.add_argument(
        "--sample",
        default="data/synthetic/test/sample_0000.npz",
        help="Path to the synthetic NPZ sample.",
    )

    parser.add_argument(
        "--output-dir",
        default="data/synthetic/demo",
        help="Output directory for PNG fixtures.",
    )

    args = parser.parse_args()

    export_sample(
        sample_path=Path(args.sample),
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()