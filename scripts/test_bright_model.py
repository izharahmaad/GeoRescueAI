from __future__ import annotations

from pathlib import Path

import torch

from georescue.bright_dataset import BrightSegmentationDataset
from georescue.models import build_model


BRIGHT_ROOT = Path(
    r"C:\Datasets\BRIGHT\local-test"
)

TEST_SPLIT = Path(
    r"data\splits\dfc25\test_set.txt"
)


def main() -> None:
    print("GeoRescue AI - BRIGHT model smoke test")
    print("=" * 60)

    dataset = BrightSegmentationDataset(
        root=BRIGHT_ROOT,
        split_file=TEST_SPLIT,
        strict_split=False,
        crop_size=None,
        training=False,
    )

    if len(dataset) == 0:
        raise SystemExit(
            "No locally available BRIGHT test samples were found."
        )

    image, target, sample_id = dataset[0]

    print("Sample ID:", sample_id)
    print("Input shape:", tuple(image.shape))
    print("Target shape:", tuple(target.shape))
    print("Input dtype:", image.dtype)
    print("Target dtype:", target.dtype)
    print("Target classes:", target.unique().tolist())

    # Official BRIGHT representation:
    # 3 optical channels + 3 replicated SAR channels.
    optical = image[:3].unsqueeze(0)
    sar = image[3:6].unsqueeze(0)

    model = build_model(
        name="fusion_unet",
        optical_channels=3,
        sar_channels=3,
        num_classes=4,
        base_channels=16,
        dropout=0.0,
    )

    model.eval()

    print()
    print("Optical batch:", tuple(optical.shape))
    print("SAR batch:", tuple(sar.shape))

    with torch.no_grad():
        logits = model(
            optical,
            sar,
        )

    print("Model output:", tuple(logits.shape))
    print("Output dtype:", logits.dtype)

    if logits.shape[0] != 1:
        raise SystemExit(
            f"Unexpected batch dimension: {logits.shape}"
        )

    if logits.shape[1] != 4:
        raise SystemExit(
            f"Expected 4 output classes, got {logits.shape[1]}"
        )

    if logits.shape[-2:] != target.shape:
        raise SystemExit(
            "Model output spatial dimensions do not match target: "
            f"{logits.shape[-2:]} vs {target.shape}"
        )

    if not torch.isfinite(logits).all():
        raise SystemExit(
            "Model produced NaN or infinite values."
        )

    prediction = logits.argmax(
        dim=1
    )

    print("Prediction shape:", tuple(prediction.shape))
    print(
        "Prediction classes:",
        torch.unique(prediction).tolist(),
    )

    print()
    print("BRIGHT model smoke test: PASS")


if __name__ == "__main__":
    main()