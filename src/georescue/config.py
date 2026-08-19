"""Configuration loading and validation for GeoRescue AI."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataConfig:
    root: Path = Path("data")
    train_dir: str = "synthetic/train"
    val_dir: str = "synthetic/val"
    test_dir: str = "synthetic/test"
    optical_channels: int = 3
    sar_channels: int = 1
    num_classes: int = 2
    image_size: int = 128


@dataclass(frozen=True)
class ModelConfig:
    name: str = "fusion_unet"
    base_channels: int = 32
    dropout: float = 0.0


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 42
    epochs: int = 5
    batch_size: int = 4
    num_workers: int = 0
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    amp: bool = True
    device: str = "auto"
    checkpoint_dir: Path = Path("outputs/checkpoints")
    best_checkpoint_name: str = "best_model.pt"


@dataclass(frozen=True)
class EvaluationConfig:
    threshold: float = 0.5
    ignore_index: int = -1


@dataclass(frozen=True)
class ProjectConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)


def _path(value: str | Path) -> Path:
    return Path(value).expanduser()


def load_config(path: str | Path) -> ProjectConfig:
    """Load YAML config into validated dataclasses."""
    with Path(path).open("r", encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle) or {}

    data = DataConfig(**{
        **raw.get("data", {}),
        "root": _path(raw.get("data", {}).get("root", "data")),
    })
    model = ModelConfig(**raw.get("model", {}))
    training_raw = dict(raw.get("training", {}))
    training_raw["checkpoint_dir"] = _path(training_raw.get("checkpoint_dir", "outputs/checkpoints"))
    training = TrainingConfig(**training_raw)
    evaluation = EvaluationConfig(**raw.get("evaluation", {}))

    if data.optical_channels < 1 or data.sar_channels < 1:
        raise ValueError("optical_channels and sar_channels must be positive")
    if data.num_classes < 2:
        raise ValueError("num_classes must be at least 2")
    if model.base_channels < 4:
        raise ValueError("base_channels must be >= 4")
    if training.epochs < 1 or training.batch_size < 1:
        raise ValueError("epochs and batch_size must be >= 1")

    return ProjectConfig(data=data, model=model, training=training, evaluation=evaluation)
