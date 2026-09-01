from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from georescue.bright_dataset import BrightSegmentationDataset
from georescue.models import build_model


BRIGHT_ROOT = Path(
    r"C:\Datasets\BRIGHT\local-test"
)

TEST_SPLIT = Path(
    r"data\splits\dfc25\test_set.txt"
)


def main() -> None:
    print("GeoRescue AI — BRIGHT training smoke test")
    print("=" * 60)

    dataset = BrightSegmentationDataset(
        root=BRIGHT_ROOT,
        split_file=TEST_SPLIT,
        strict_split=False,
        crop_size=256,
        training=False,
    )

    if len(dataset) == 0:
        raise SystemExit(
            "No BRIGHT samples are available."
        )

    image, target, sample_id = dataset[0]

    print("Sample:", sample_id)
    print("Input:", tuple(image.shape))
    print("Target:", tuple(target.shape))

    # The current DFC25 test tile you extracted has binary labels.
    # This smoke test therefore uses a 2-class head only to validate
    # the mathematical training path. We will use the official
    # 4-class target configuration once the labeled train/holdout
    # data are available and verified.
    model = build_model(
        name="fusion_unet",
        optical_channels=3,
        sar_channels=3,
        num_classes=2,
        base_channels=16,
        dropout=0.0,
    )

    model.train()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-4,
    )

    criterion = nn.CrossEntropyLoss()

    # The loader returns [6, H, W].
    optical = image[:3].unsqueeze(0)
    sar = image[3:6].unsqueeze(0)
    target_batch = target.unsqueeze(0)

    print("Optical batch:", tuple(optical.shape))
    print("SAR batch:", tuple(sar.shape))

    optimizer.zero_grad(set_to_none=True)

    logits = model(
        optical,
        sar,
    )

    print("Logits:", tuple(logits.shape))

    loss = criterion(
        logits,
        target_batch,
    )

    print("Loss:", float(loss))

    if not torch.isfinite(loss):
        raise SystemExit(
            "ERROR: Loss is not finite."
        )

    loss.backward()

    gradient_count = 0

    for parameter in model.parameters():
        if parameter.grad is not None:
            gradient_count += 1

    if gradient_count == 0:
        raise SystemExit(
            "ERROR: No model gradients were produced."
        )

    optimizer.step()

    print("Parameters with gradients:", gradient_count)
    print()
    print("BRIGHT training smoke test: PASS")


if __name__ == "__main__":
    main()