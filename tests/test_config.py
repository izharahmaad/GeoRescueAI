from georescue.config import load_config


def test_default_config() -> None:
    cfg = load_config("configs/default.yaml")
    assert cfg.data.optical_channels == 3
    assert cfg.data.sar_channels == 1
    assert cfg.model.name == "fusion_unet"
