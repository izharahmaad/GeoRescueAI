"""Create a visual BRIGHT DFC25 inference demo from a local sample."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
from PIL import Image

from georescue.bright_dataset import BrightSegmentationDataset
from georescue.models import build_model


def normalize_to_uint8(
    array: np.ndarray,
) -> np.ndarray:
    """Convert an array to an 8-bit visualization."""
    array = np.asarray(
        array,
        dtype=np.float32,
    )

    finite = np.isfinite(array)

    if not finite.any():
        return np.zeros(
            array.shape,
            dtype=np.uint8,
        )

    values = array[finite]

    low = float(np.percentile(values, 2))
    high = float(np.percentile(values, 98))

    if high <= low:
        low = float(values.min())
        high = float(values.max())

    if high <= low:
        return np.zeros(
            array.shape,
            dtype=np.uint8,
        )

    scaled = (
        (array - low)
        / (high - low)
        * 255.0
    )

    scaled = np.clip(
        scaled,
        0,
        255,
    )

    return scaled.astype(
        np.uint8
    )


def mask_to_image(
    mask: np.ndarray,
) -> Image.Image:
    """Create a clear grayscale mask visualization."""
    mask = np.asarray(
        mask,
        dtype=np.uint8,
    )

    values = np.unique(mask)

    # Binary target/prediction.
    if len(values) <= 2:
        image = (
            (mask > 0).astype(np.uint8)
            * 255
        )

        return Image.fromarray(
            image,
            mode="L",
        )

    # Generic 4-class visualization.
    palette = np.array(
        [
            [0, 0, 0],
            [85, 85, 255],
            [85, 255, 85],
            [255, 85, 85],
        ],
        dtype=np.uint8,
    )

    clipped = np.clip(
        mask,
        0,
        len(palette) - 1,
    )

    rgb = palette[
        clipped
    ]

    return Image.fromarray(
        rgb,
        mode="RGB",
    )


def create_target_overlay(
    optical_rgb: np.ndarray,
    target: np.ndarray,
) -> Image.Image:
    """Overlay the target damage mask on optical imagery."""
    base = Image.fromarray(
        optical_rgb,
        mode="RGB",
    ).convert("RGBA")

    mask = (
        target > 0
    ).astype(np.uint8) * 150

    red = np.zeros(
        (
            target.shape[0],
            target.shape[1],
            4,
        ),
        dtype=np.uint8,
    )

    red[:, :, 0] = 255
    red[:, :, 3] = mask

    overlay = Image.fromarray(
        red,
        mode="RGBA",
    )

    return Image.alpha_composite(
        base,
        overlay,
    ).convert("RGB")


def create_prediction_overlay(
    optical_rgb: np.ndarray,
    prediction: np.ndarray,
) -> Image.Image:
    """Overlay predicted positive damage on optical imagery."""
    base = Image.fromarray(
        optical_rgb,
        mode="RGB",
    ).convert("RGBA")

    positive = (
        prediction > 0
    ).astype(np.uint8) * 150

    red = np.zeros(
        (
            prediction.shape[0],
            prediction.shape[1],
            4,
        ),
        dtype=np.uint8,
    )

    red[:, :, 0] = 255
    red[:, :, 3] = positive

    overlay = Image.fromarray(
        red,
        mode="RGBA",
    )

    return Image.alpha_composite(
        base,
        overlay,
    ).convert("RGB")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a visual BRIGHT DFC25 demo."
    )

    parser.add_argument(
        "--bright-root",
        required=True,
        help="BRIGHT root containing pre-event, post-event, and target.",
    )

    parser.add_argument(
        "--split-file",
        default="data/splits/dfc25/test_set.txt",
        help="Official BRIGHT split file.",
    )

    parser.add_argument(
        "--output-dir",
        default="outputs/predictions/bright_demo",
        help="Directory for demo outputs.",
    )

    parser.add_argument(
        "--base-channels",
        type=int,
        default=8,
        help="U-Net base channel count for CPU-friendly demo inference.",
    )

    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional trained checkpoint.",
    )

    args = parser.parse_args()

    bright_root = Path(
        args.bright_root
    )

    split_file = Path(
        args.split_file
    )

    output_dir = ROOT / Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "GeoRescue AI - BRIGHT DFC25 Demo"
    )
    print("=" * 60)

    print(
        "BRIGHT root:",
        bright_root,
    )

    print(
        "Split file:",
        split_file,
    )

    print(
        "Output directory:",
        output_dir,
    )

    dataset = BrightSegmentationDataset(
        root=bright_root,
        split_file=split_file,
        strict_split=False,
        crop_size=None,
        training=False,
    )

    if len(dataset) == 0:
        raise SystemExit(
            "No locally available BRIGHT samples were found."
        )

    image, target, sample_id = dataset[0]

    print()
    print(
        "Sample:",
        sample_id,
    )

    print(
        "Input shape:",
        tuple(image.shape),
    )

    print(
        "Target shape:",
        tuple(target.shape),
    )

    print(
        "Target classes:",
        target.unique().tolist(),
    )

    # -------------------------------------------------------------
    # Create human-readable source images.
    # -------------------------------------------------------------

    with __import__("rasterio").open(
        dataset.samples[0].optical_path
    ) as src:
        optical_raw = src.read()

    with __import__("rasterio").open(
        dataset.samples[0].sar_path
    ) as src:
        sar_raw = src.read(1)

    optical_rgb = np.transpose(
        optical_raw[:3],
        (1, 2, 0),
    )

    optical_rgb = np.clip(
        optical_rgb,
        0,
        255,
    ).astype(
        np.uint8
    )

    optical_image = Image.fromarray(
        optical_rgb,
        mode="RGB",
    )

    sar_image = Image.fromarray(
        normalize_to_uint8(
            sar_raw
        ),
        mode="L",
    )

    target_array = target.numpy()

    target_image = mask_to_image(
        target_array
    )

    target_overlay = create_target_overlay(
        optical_rgb,
        target_array,
    )

    optical_image.save(
        output_dir
        / f"{sample_id}_optical.png"
    )

    sar_image.save(
        output_dir
        / f"{sample_id}_sar.png"
    )

    target_image.save(
        output_dir
        / f"{sample_id}_target.png"
    )

    target_overlay.save(
        output_dir
        / f"{sample_id}_target_overlay.png"
    )

    # -------------------------------------------------------------
    # Build 4-class Fusion U-Net.
    #
    # If no checkpoint is supplied, this is an UNTRAINED model.
    # Therefore its prediction is a pipeline demonstration only.
    # -------------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = build_model(
        name="fusion_unet",
        optical_channels=3,
        sar_channels=3,
        num_classes=4,
        base_channels=args.base_channels,
        dropout=0.0,
    )

    model = model.to(
        device
    )

    if args.checkpoint is not None:
        checkpoint_path = Path(
            args.checkpoint
        )

        if not checkpoint_path.is_file():
            raise SystemExit(
                f"Checkpoint not found: {checkpoint_path}"
            )

        payload = torch.load(
            checkpoint_path,
            map_location=device,
        )

        state_dict = payload.get(
            "model",
            payload,
        )

        model.load_state_dict(
            state_dict
        )

        print(
            "Checkpoint loaded:",
            checkpoint_path,
        )

    else:
        print(
            "No checkpoint supplied: using an UNTRAINED model."
        )

    model.eval()

    optical = (
        image[:3]
        .unsqueeze(0)
        .to(device)
    )

    sar = (
        image[3:6]
        .unsqueeze(0)
        .to(device)
    )

    print(
        "Device:",
        device,
    )

    print(
        "Running inference..."
    )

    with torch.no_grad():
        logits = model(
            optical,
            sar,
        )

    prediction = logits.argmax(
        dim=1
    )[0].cpu().numpy()

    prediction_image = mask_to_image(
        prediction
    )

    prediction_overlay = create_prediction_overlay(
        optical_rgb,
        prediction,
    )

    prediction_image.save(
        output_dir
        / f"{sample_id}_prediction.png"
    )

    prediction_overlay.save(
        output_dir
        / f"{sample_id}_prediction_overlay.png"
    )

    metadata = {
        "sample_id": sample_id,
        "device": str(device),
        "input_shape": list(image.shape),
        "target_shape": list(target.shape),
        "target_classes": target.unique().tolist(),
        "prediction_classes": np.unique(
            prediction
        ).tolist(),
        "checkpoint": (
            str(args.checkpoint)
            if args.checkpoint
            else None
        ),
        "warning": (
            "Prediction is from an untrained model."
            if args.checkpoint is None
            else None
        ),
    }

    import json

    (
        output_dir
        / f"{sample_id}_metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "Generated files:"
    )

    for path in sorted(
        output_dir.glob(
            f"{sample_id}_*"
        )
    ):
        print(
            " ",
            path,
        )

    print()
    print(
        "BRIGHT demo completed successfully."
    )


if __name__ == "__main__":
    main()