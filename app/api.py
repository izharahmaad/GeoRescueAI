"""FastAPI inference service for GeoRescue AI.

The service provides:
- health/status information;
- uploaded optical + SAR inference;
- a local BRIGHT DFC25 demonstration;
- the lightweight web dashboard.
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from georescue.config import load_config
from georescue.models import build_model
from georescue.train_utils import (
    load_checkpoint,
    select_device,
)


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

CONFIG_PATH = ROOT / os.getenv(
    "GEORESCUE_CONFIG",
    "configs/default.yaml",
)

CHECKPOINT_PATH = ROOT / os.getenv(
    "GEORESCUE_CHECKPOINT",
    "outputs/checkpoints/best_model.pt",
)

BRIGHT_DEMO_DIR = (
    ROOT
    / "outputs"
    / "predictions"
    / "bright_demo"
)

BRIGHT_DEMO_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ---------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------

app = FastAPI(
    title="GeoRescue AI",
    version="0.2.0",
    description=(
        "Multimodal disaster damage segmentation API "
        "with BRIGHT DFC25 demonstration support."
    ),
)


_state: dict[str, object] = {}


# ---------------------------------------------------------------------
# Model initialization
# ---------------------------------------------------------------------


def _load_state() -> None:
    """Load configuration, model, device, and optional checkpoint."""

    cfg = load_config(
        CONFIG_PATH
    )

    device = select_device(
        cfg.training.device
    )

    model = build_model(
        cfg.model.name,
        cfg.data.optical_channels,
        cfg.data.sar_channels,
        cfg.data.num_classes,
        cfg.model.base_channels,
        cfg.model.dropout,
    ).to(device)

    checkpoint_loaded = False

    if CHECKPOINT_PATH.exists():
        load_checkpoint(
            CHECKPOINT_PATH,
            model,
            device,
        )
        checkpoint_loaded = True

    model.eval()

    _state.clear()

    _state.update(
        config=cfg,
        device=device,
        model=model,
        checkpoint_loaded=checkpoint_loaded,
    )


@app.on_event("startup")
def startup() -> None:
    """Initialize model state when the FastAPI service starts."""

    try:
        _load_state()

    except Exception as exc:
        _state.clear()
        _state["startup_error"] = str(exc)


# ---------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------


@app.get("/health")
def health() -> dict[str, object]:
    """Return service and model health information."""

    return {
        "status": (
            "ok"
            if "startup_error" not in _state
            else "degraded"
        ),
        "checkpoint_exists": CHECKPOINT_PATH.exists(),
        "checkpoint_loaded": _state.get(
            "checkpoint_loaded",
            False,
        ),
        "device": str(
            _state.get(
                "device",
                "uninitialized",
            )
        ),
        "error": _state.get(
            "startup_error"
        ),
    }


# ---------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------


@app.get(
    "/",
    response_class=HTMLResponse,
)
def index() -> str:
    """Serve the GeoRescue AI web dashboard."""

    page = (
        ROOT
        / "app"
        / "static"
        / "index.html"
    )

    if not page.exists():
        raise HTTPException(
            status_code=500,
            detail="Dashboard file not found.",
        )

    return page.read_text(
        encoding="utf-8"
    )


# ---------------------------------------------------------------------
# BRIGHT demonstration metadata
#
# IMPORTANT:
# This endpoint is intentionally defined BEFORE the StaticFiles mount.
# ---------------------------------------------------------------------


@app.get("/bright-demo/info")
def bright_demo_info() -> dict[str, object]:
    """Return metadata and image URLs for the BRIGHT demonstration."""

    metadata_files = sorted(
        BRIGHT_DEMO_DIR.glob(
            "*_metadata.json"
        )
    )

    if not metadata_files:
        return {
            "available": False,
            "message": (
                "No BRIGHT demo metadata found. "
                "Run scripts/demo_bright.py first."
            ),
        }

    # Use the first generated metadata file.
    metadata_path = metadata_files[0]

    try:
        metadata = json.loads(
            metadata_path.read_text(
                encoding="utf-8"
            )
        )

    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=500,
            detail="Unable to read BRIGHT demo metadata.",
        ) from exc

    sample_id = str(
        metadata.get(
            "sample_id",
            metadata_path.stem.replace(
                "_metadata",
                "",
            ),
        )
    )

    return {
        "available": True,
        "sample_id": sample_id,
        "metadata": metadata,
        "files": {
            "optical": (
                f"/bright-demo/"
                f"{sample_id}_optical.png"
            ),
            "sar": (
                f"/bright-demo/"
                f"{sample_id}_sar.png"
            ),
            "target": (
                f"/bright-demo/"
                f"{sample_id}_target.png"
            ),
            "target_overlay": (
                f"/bright-demo/"
                f"{sample_id}_target_overlay.png"
            ),
            "prediction": (
                f"/bright-demo/"
                f"{sample_id}_prediction.png"
            ),
            "prediction_overlay": (
                f"/bright-demo/"
                f"{sample_id}_prediction_overlay.png"
            ),
        },
    }


# ---------------------------------------------------------------------
# BRIGHT static demo files
#
# IMPORTANT:
# This MUST come after /bright-demo/info.
# ---------------------------------------------------------------------


app.mount(
    "/bright-demo",
    StaticFiles(
        directory=str(BRIGHT_DEMO_DIR)
    ),
    name="bright-demo",
)


# ---------------------------------------------------------------------
# Image processing
# ---------------------------------------------------------------------


def _read_image(
    upload: UploadFile,
    channels: int,
) -> np.ndarray:
    """Read uploaded imagery into CHW float32 format."""

    raw = upload.file.read()

    if not raw:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Empty upload: "
                f"{upload.filename}"
            ),
        )

    try:
        image = Image.open(
            io.BytesIO(raw)
        )

        if channels == 3:
            image = image.convert(
                "RGB"
            )

        elif channels == 1:
            image = image.convert(
                "L"
            )

        else:
            raise ValueError(
                f"Unsupported channel count: "
                f"{channels}"
            )

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid image: "
                f"{upload.filename}"
            ),
        ) from exc

    array = np.asarray(
        image
    ).astype(
        np.float32,
        copy=False,
    )

    array /= 255.0

    if channels == 3:
        # HWC -> CHW
        array = np.moveaxis(
            array,
            -1,
            0,
        )

    else:
        # HW -> CHW
        array = array[
            None,
            ...
        ]

    return np.ascontiguousarray(
        array,
        dtype=np.float32,
    )


# ---------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------


@app.post(
    "/predict",
    response_class=Response,
    responses={
        200: {
            "description": (
                "PNG damage segmentation mask."
            ),
            "content": {
                "image/png": {},
            },
        },
        400: {
            "description": (
                "Invalid or incompatible input images."
            ),
        },
        503: {
            "description": (
                "Model is unavailable."
            ),
        },
    },
)
async def predict(
    optical: UploadFile = File(
        ...,
        description=(
            "Pre-disaster optical RGB image."
        ),
    ),
    sar: UploadFile = File(
        ...,
        description=(
            "Post-disaster SAR image."
        ),
    ),
) -> Response:
    """Run multimodal segmentation inference."""

    if "model" not in _state:
        raise HTTPException(
            status_code=503,
            detail=(
                "Model unavailable: "
                f"{_state.get('startup_error', 'not loaded')}"
            ),
        )

    cfg = _state["config"]
    device = _state["device"]
    model = _state["model"]

    optical_array = _read_image(
        optical,
        cfg.data.optical_channels,
    )

    sar_array = _read_image(
        sar,
        cfg.data.sar_channels,
    )

    if (
        optical_array.shape[1:]
        != sar_array.shape[1:]
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "Optical and SAR images must "
                "have identical spatial dimensions."
            ),
        )

    optical_tensor = torch.from_numpy(
        optical_array[None]
    ).to(device)

    sar_tensor = torch.from_numpy(
        sar_array[None]
    ).to(device)

    with torch.inference_mode():

        if hasattr(
            model,
            "optical_stem",
        ):
            logits = model(
                optical_tensor,
                sar_tensor,
            )

        elif getattr(
            model,
            "_is_sar_model",
            False,
        ):
            logits = model(
                sar_tensor
            )

        else:
            logits = model(
                optical_tensor
            )

        mask = (
            logits
            .argmax(
                dim=1
            )[0]
            .cpu()
            .numpy()
            .astype(
                np.uint8
            )
        )

    buffer = io.BytesIO()

    Image.fromarray(
        mask,
        mode="L",
    ).save(
        buffer,
        format="PNG",
    )

    return Response(
        content=buffer.getvalue(),
        media_type="image/png",
        headers={
            "Content-Disposition": (
                'inline; '
                'filename="georescue_prediction.png"'
            )
        },
    )