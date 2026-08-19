import torch

from georescue.models import build_model


def test_fusion_forward() -> None:
    model = build_model("fusion_unet", 3, 1, 2, 8)
    optical = torch.randn(2, 3, 64, 64)
    sar = torch.randn(2, 1, 64, 64)
    output = model(optical, sar)
    assert output.shape == (2, 2, 64, 64)


def test_baseline_forwards() -> None:
    optical = torch.randn(1, 3, 64, 64)
    sar = torch.randn(1, 1, 64, 64)
    assert build_model("optical_unet", 3, 1, 2, 8)(optical).shape == (1, 2, 64, 64)
    assert build_model("sar_unet", 3, 1, 2, 8)(sar).shape == (1, 2, 64, 64)
