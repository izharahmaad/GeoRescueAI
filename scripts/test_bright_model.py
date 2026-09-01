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
    print("GeoRescue AI — BRIGHT model compatibility test")
    print("=" * 60)

    dataset = BrightSegmentationDataset(
        root=BRIGHT_ROOT,
        split_file=TEST_SPLIT,
        strict_split=False,
        crop_size=256,
        training=False,
    )

    print("Dataset samples:", len(dataset))

    image, target, sample_id = dataset[0]

    print("Sample ID:", sample_id)
    print("Input shape:", tuple(image.shape))
    print("Target shape:", tuple(target.shape))

    model = build_model(
        name="fusion_unet",
        optical_channels=3,
        sar_channels=3,
        num_classes=4,
        base_channels=16,
        dropout=0.0,
    )

    model.eval()

    # Split the 6-channel representation back into the two modality
    # tensors expected by FusionUNet.
    optical = image[:3].unsqueeze(0)
    sar = image[3:6].unsqueeze(0)

    print("Optical batch:", tuple(optical.shape))
    print("SAR batch:", tuple(sar.shape))

    with torch.no_grad():
        logits = model(
            optical,
            sar,
        )

    print("Model output:", tuple(logits.shape))
    print("Output dtype:", logits.dtype)
    print("Output finite:", bool(torch.isfinite(logits).all()))

    predictions = logits.argmax(
        dim=1
    )

    print(
        "Prediction shape:",
        tuple(predictions.shape),
    )

    print(
        "Prediction classes:",
        torch.unique(predictions).tolist(),
    )

    print()
    print("BRIGHT model compatibility: PASS")


if __name__ == "__main__":
    main()