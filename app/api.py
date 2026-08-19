"""FastAPI inference service for GeoRescue AI.

The API loads a trained checkpoint at startup, validates uploaded optical
and SAR images, performs segmentation inference, and returns a PNG mask.
"""

from __future__ import annotations

import io
import os
from pathlib import Path
import sys

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, Response
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from georescue.config import load_config
from georescue.models import build_model
from georescue.train_utils import load_checkpoint, select_device


CONFIG_PATH = ROOT / os.getenv(
    "GEORESCUE_CONFIG",
    "configs/default.yaml",
)

CHECKPOINT_PATH = ROOT / os.getenv(
    "GEORESCUE_CHECKPOINT",
    "outputs/checkpoints/best_model.pt",
)


app = FastAPI(
    title="GeoRescue AI",
    version="0.1.0",
    description="Multimodal disaster damage segmentation API",
)

_state: dict[str, object] = {}


def _load_state() -> None:
    """Load configuration, model, device, and checkpoint into application state."""
    cfg = load_config(CONFIG_PATH)

    device = select_device(cfg.training.device)

    model = build_model(
        cfg.model.name,
        cfg.data.optical_channels,
        cfg.data.sar_channels,
        cfg.data.num_classes,
        cfg.model.base_channels,
        cfg.model.dropout,
    ).to(device)

    if CHECKPOINT_PATH.exists():
        load_checkpoint(
            CHECKPOINT_PATH,
            model,
            device,
        )

    model.eval()

    _state.update(
        config=cfg,
        device=device,
        model=model,
    )


@app.on_event("startup")
def startup() -> None:
    """Initialize the model when the FastAPI application starts."""
    try:
        _load_state()
    except Exception as exc:
        # Keep the service alive so /health can expose the startup problem.
        _state["startup_error"] = str(exc)


@app.get("/health")
def health() -> dict[str, object]:
    """Return API and model health information."""
    return {
        "status": (
            "ok"
            if "startup_error" not in _state
            else "degraded"
        ),
        "checkpoint_exists": CHECKPOINT_PATH.exists(),
        "device": str(
            _state.get(
                "device",
                "uninitialized",
            )
        ),
        "error": _state.get("startup_error"),
    }


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    """Serve the lightweight web dashboard."""
    page = ROOT / "app" / "static" / "index.html"

    if not page.exists():
        raise HTTPException(
            status_code=500,
            detail="Dashboard file not found.",
        )

    return page.read_text(encoding="utf-8")


def _read_image(
    upload: UploadFile,
    channels: int,
) -> np.ndarray:
    """Read and normalize an uploaded image into CHW float32 format."""
    raw = upload.file.read()

    if not raw:
        raise HTTPException(
            status_code=400,
            detail=f"Empty upload: {upload.filename}",
        )

    try:
        image = Image.open(io.BytesIO(raw))

        if channels == 3:
            image = image.convert("RGB")
        elif channels == 1:
            image = image.convert("L")
        else:
            raise ValueError(
                f"Unsupported channel count: {channels}"
            )

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image: {upload.filename}",
        ) from exc

    array = np.asarray(image).astype(
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
        array = array[None, ...]

    return array


@app.post(
    "/predict",
    response_class=Response,
    responses={
        200: {
            "description": "PNG damage segmentation mask.",
            "content": {
                "image/png": {},
            },
        },
        400: {
            "description": "Invalid or incompatible input images.",
        },
        503: {
            "description": "Model is unavailable.",
        },
    },
)
async def predict(
    optical: UploadFile = File(
        ...,
        description="Pre-disaster optical RGB image.",
    ),
    sar: UploadFile = File(
        ...,
        description="Post-disaster SAR image.",
    ),
) -> Response:
    """Run multimodal segmentation and return a PNG damage mask."""
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

    if optical_array.shape[1:] != sar_array.shape[1:]:
        raise HTTPException(
            status_code=400,
            detail=(
                "Optical and SAR images must have "
                "identical spatial dimensions"
            ),
        )

    optical_tensor = torch.from_numpy(
        optical_array[None]
    ).to(device)

    sar_tensor = torch.from_numpy(
        sar_array[None]
    ).to(device)

    with torch.inference_mode():
        if hasattr(model, "optical_stem"):
            logits = model(
                optical_tensor,
                sar_tensor,
            )
        elif getattr(
            model,
            "_is_sar_model",
            False,
        ):
            logits = model(sar_tensor)
        else:
            logits = model(optical_tensor)

        mask = (
            logits
            .argmax(dim=1)[0]
            .cpu()
            .numpy()
            .astype(np.uint8)
        )

    buffer = io.BytesIO()

    # Pillow infers grayscale mode from the uint8 2-D array.
    Image.fromarray(mask).save(
        buffer,
        format="PNG",
    )

    return Response(
        content=buffer.getvalue(),
        media_type="image/png",
        headers={
            "Content-Disposition": (
                'inline; filename="georescue_prediction.png"'
            )
        },
    )