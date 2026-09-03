"""Configuration loading and validation for GeoRescue AI."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataConfig:
    """Dataset-related configuration."""

    root: str
    train_dir: str
    val_dir: str
    optical_channels: int
    sar_channels: int
    num_classes: int


@dataclass(frozen=True)
class ModelConfig:
    """Model-related configuration."""

    name: str
    base_channels: int
    dropout: float


@dataclass(frozen=True)
class TrainingConfig:
    """Training-related configuration."""

    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    num_workers: int
    device: str
    amp: bool
    seed: int
    checkpoint_dir: Path
    best_checkpoint_name: str


@dataclass(frozen=True)
class AppConfig:
    """Complete GeoRescue AI configuration."""

    data: DataConfig
    model: ModelConfig
    training: TrainingConfig


def _require_mapping(
    payload: Any,
    name: str,
) -> dict[str, Any]:
    """Validate a configuration section is a mapping."""

    if not isinstance(payload, dict):
        raise ValueError(
            f"Configuration section '{name}' must be a mapping."
        )

    return payload


def load_config(
    path: str | Path,
) -> AppConfig:
    """Load and validate a YAML configuration file."""

    config_path = Path(path)

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    try:
        payload = yaml.safe_load(
            config_path.read_text(
                encoding="utf-8"
            )
        )
    except yaml.YAMLError as exc:
        raise ValueError(
            f"Invalid YAML configuration: {config_path}"
        ) from exc

    if not isinstance(payload, dict):
        raise ValueError(
            "Configuration root must be a mapping."
        )

    data_payload = _require_mapping(
        payload.get("data"),
        "data",
    )

    model_payload = _require_mapping(
        payload.get("model"),
        "model",
    )

    training_payload = _require_mapping(
        payload.get("training"),
        "training",
    )

    data = DataConfig(
        **{
            "root": str(
                data_payload["root"]
            ),
            "train_dir": str(
                data_payload.get(
                    "train_dir",
                    "train",
                )
            ),
            "val_dir": str(
                data_payload.get(
                    "val_dir",
                    "val",
                )
            ),
            "optical_channels": int(
                data_payload["optical_channels"]
            ),
            "sar_channels": int(
                data_payload["sar_channels"]
            ),
            "num_classes": int(
                data_payload["num_classes"]
            ),
        }
    )

    model = ModelConfig(
        **{
            "name": str(
                model_payload["name"]
            ),
            "base_channels": int(
                model_payload["base_channels"]
            ),
            "dropout": float(
                model_payload.get(
                    "dropout",
                    0.0,
                )
            ),
        }
    )

    training = TrainingConfig(
        **{
            "batch_size": int(
                training_payload["batch_size"]
            ),
            "epochs": int(
                training_payload["epochs"]
            ),
            "learning_rate": float(
                training_payload["learning_rate"]
            ),
            "weight_decay": float(
                training_payload["weight_decay"]
            ),
            "num_workers": int(
                training_payload.get(
                    "num_workers",
                    0,
                )
            ),
            "device": str(
                training_payload.get(
                    "device",
                    "auto",
                )
            ),
            "amp": bool(
                training_payload.get(
                    "amp",
                    True,
                )
            ),
            "seed": int(
                training_payload.get(
                    "seed",
                    42,
                )
            ),
            "checkpoint_dir": Path(
                training_payload.get(
                    "checkpoint_dir",
                    "outputs/checkpoints",
                )
            ),
            "best_checkpoint_name": str(
                training_payload.get(
                    "best_checkpoint_name",
                    "best_model.pt",
                )
            ),
        }
    )

    if data.optical_channels <= 0:
        raise ValueError(
            "data.optical_channels must be positive."
        )

    if data.sar_channels <= 0:
        raise ValueError(
            "data.sar_channels must be positive."
        )

    if data.num_classes <= 1:
        raise ValueError(
            "data.num_classes must be greater than 1."
        )

    if model.base_channels <= 0:
        raise ValueError(
            "model.base_channels must be positive."
        )

    if not 0.0 <= model.dropout < 1.0:
        raise ValueError(
            "model.dropout must be in [0, 1)."
        )

    if training.batch_size <= 0:
        raise ValueError(
            "training.batch_size must be positive."
        )

    if training.epochs <= 0:
        raise ValueError(
            "training.epochs must be positive."
        )

    if training.learning_rate <= 0:
        raise ValueError(
            "training.learning_rate must be positive."
        )

    if training.weight_decay < 0:
        raise ValueError(
            "training.weight_decay cannot be negative."
        )

    if training.num_workers < 0:
        raise ValueError(
            "training.num_workers cannot be negative."
        )

    return AppConfig(
        data=data,
        model=model,
        training=training,
    )