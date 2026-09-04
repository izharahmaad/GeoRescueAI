
<h1 align="center">GeoRescue AI</h1>

<p align="center">
  <strong>Multimodal satellite-image damage assessment for rapid disaster-response analysis.</strong>
</p>

<p align="center">
  Optical + SAR fusion • Deep-learning segmentation • FastAPI • Interactive geospatial analysis dashboard
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white" alt="Python 3.11">
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/BRIGHT-DFC25-1F6FEB" alt="BRIGHT DFC25">
  <img src="https://img.shields.io/badge/Tests-14%20passed-2EA44F" alt="Tests">
  <img src="https://img.shields.io/badge/License-MIT-black" alt="MIT License">
</p>

---

## Overview

**GeoRescue AI** is a computer-vision research and demonstration platform for identifying disaster-related damage from satellite imagery.

The core idea is simple:

> **Combine complementary optical and Synthetic Aperture Radar (SAR) observations to produce a pixel-level damage map.**

Optical imagery provides visual information about the observed scene, while SAR supplies radar information that can remain useful when optical imagery is affected by illumination or cloud conditions. GeoRescue AI brings both modalities into a single segmentation pipeline and exposes the result through a lightweight FastAPI backend and an operator-focused analysis dashboard.

The project is designed around the **BRIGHT DFC25** disaster-damage benchmark and currently includes dataset discovery, split handling, preprocessing, multimodal model code, training utilities, inference/demo tooling, API routes, and a web interface.

> **Research/demo note:** the repository includes a real-data smoke-test path and a local BRIGHT sample workflow. A full benchmark training run requires the corresponding BRIGHT DFC25 imagery and labels to be available locally.

---

## Why GeoRescue AI?

After a major disaster, response teams need to answer questions quickly:

- Where is visible damage concentrated?
- Which areas should be inspected first?
- Can different satellite modalities reinforce each other?
- Can an analyst inspect the raw imagery and model output in one place?

GeoRescue AI focuses on the last-mile workflow between a trained segmentation model and a usable analysis interface.

### Core capabilities

| Capability | What it does |
|---|---|
| **Optical + SAR fusion** | Uses complementary satellite modalities in a shared segmentation model |
| **Pixel-level segmentation** | Produces a per-pixel class prediction rather than only an image-level label |
| **BRIGHT DFC25 support** | Provides dataset discovery, geometry validation, preprocessing, and official split handling |
| **Multiple model modes** | Optical-only U-Net, SAR-only U-Net, and multimodal Fusion U-Net |
| **Real-data smoke testing** | Verifies tensor shapes, forward pass, loss computation, and gradient flow on BRIGHT data |
| **FastAPI inference API** | Accepts optical/SAR inputs and returns a segmentation result |
| **Live analysis dashboard** | Displays imagery, overlays, model state, inference controls, logs, and export actions |
| **Reproducible split validation** | Checks official DFC25 split membership and event isolation |
| **Demo asset pipeline** | Generates/serves visual assets for a repeatable case-study workflow |

---

## System Architecture

```text
                 ┌─────────────────────────────┐
                 │      BRIGHT DFC25 Data      │
                 │  Optical + SAR + Labels     │
                 └──────────────┬──────────────┘
                                │
                                ▼
                 ┌─────────────────────────────┐
                 │ Dataset Discovery & Checks  │
                 │ • file indexing             │
                 │ • geometry validation       │
                 │ • split validation          │
                 └──────────────┬──────────────┘
                                │
                                ▼
                 ┌─────────────────────────────┐
                 │      Preprocessing          │
                 │ • optical normalization     │
                 │ • SAR preprocessing         │
                 │ • spatial alignment         │
                 │ • augmentation / cropping   │
                 └──────────────┬──────────────┘
                                │
                     ┌──────────┴──────────┐
                     │                     │
                     ▼                     ▼
              ┌──────────────┐     ┌────────────────┐
              │ Optical U-Net│     │    SAR U-Net   │
              └──────┬───────┘     └───────┬────────┘
                     │                     │
                     └──────────┬──────────┘
                                ▼
                    ┌──────────────────────┐
                    │    Fusion U-Net      │
                    │ Optical + SAR stems  │
                    │Shared encoder/decoder│
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Damage Segmentation  │
                    │   Pixel-wise Output  │
                    └──────────┬───────────┘
                               │
                ┌──────────────┴──────────────┐
                ▼                             ▼
        ┌────────────────┐          ┌────────────────────┐
        │ FastAPI Backend│          │ Analysis Dashboard │
        │ /health        │          │ imagery + overlays │
        │ /predict       │          │ controls + export  │
        │ /bright-demo/* │          │ live system state  │
        └────────────────┘          └────────────────────┘
```

---

## Model Design

GeoRescue AI currently provides three segmentation configurations.

### 1. Optical U-Net

A compact U-Net baseline operating on optical imagery.

```text
Optical RGB
    │
    ▼
Encoder → Bottleneck → Decoder
                         │
                         ▼
                  Segmentation Map
```

### 2. SAR U-Net

A matching U-Net baseline operating on SAR input.

```text
SAR
 │
 ▼
Encoder → Bottleneck → Decoder
                         │
                         ▼
                  Segmentation Map
```

### 3. Fusion U-Net

The main multimodal architecture uses **separate optical and SAR input stems**, followed by feature fusion and a shared encoder/decoder path.

```text
Optical ──► Optical Stem ──┐
                           ├──► Fusion ──► Encoder ──► Decoder ──► Mask
SAR ──────► SAR Stem ──────┘
```

This design makes the modality separation explicit while allowing the network to learn joint representations before producing the final segmentation map.

---

## Data Representation

For the BRIGHT pipeline, the multimodal input is represented as **six channels**:

```text
Channels 0–2  → Optical RGB
Channels 3–5  → SAR replicated into 3 channels
```

Example tensors used by the training pipeline:

```text
Optical : (B, 3, H, W)
SAR     : (B, 3, H, W)
Target  : (B, H, W)
```

For the current BRIGHT sample workflow, the native sample size is validated as:

```text
Input  : 1024 × 1024
Target : 1024 × 1024
```

The exact preprocessing and augmentation logic lives in the repository rather than being duplicated in this README.

---

## Dataset

### BRIGHT DFC25

GeoRescue AI is built to work with the **BRIGHT DFC25** dataset.

The repository includes utilities for:

- discovering matching optical/SAR/label files
- validating raster geometry
- checking transform and spatial-bound consistency
- reading official DFC25 split files
- handling incomplete local subsets during development
- validating train/holdout/validation/test membership
- checking unseen-event isolation for the official test events

### Official split files used by the project

```text
data/splits/dfc25/
├── train_set.txt
├── holdout_set.txt
├── val_set.txt
└── test_set.txt
```

A repository validation script is provided:

```powershell
python scripts/validate_bright_splits.py
```

The current split validation workflow checks that the final test set remains isolated from the other splits and reports the official split overlaps where applicable.

### Local development vs. full training

A small BRIGHT subset is enough to validate the software pipeline:

```text
C:\Datasets\BRIGHT\local-test
```

However, **full model training and meaningful benchmark evaluation require the complete imagery/label data for the relevant splits**.

GeoRescue AI therefore separates:

1. **engineering validation** — dataset loading, geometry checks, model forward pass, API/demo behavior
2. **research training/evaluation** — full BRIGHT data, trained checkpoints, benchmark metrics

This distinction prevents a local smoke test from being presented as a full benchmark result.

---

## Project Structure

```text
GeoRescueAI/
│
├── app/
│   ├── api.py                    # FastAPI application
│   └── static/
│       └── index.html             # Analysis dashboard
│
├── configs/
│   ├── default.yaml               # Default application/training config
│   └── bright.yaml                # BRIGHT-specific configuration
│
├── data/
│   ├── splits/
│   │   └── dfc25/
│   │       ├── train_set.txt
│   │       ├── holdout_set.txt
│   │       ├── val_set.txt
│   │       └── test_set.txt
│   └── synthetic/                 # Local smoke-test/demo data
│
├── scripts/
│   ├── train.py                   # Generic training entry point
│   ├── train_bright.py            # BRIGHT training/smoke-test entry point
│   ├── demo_bright.py             # BRIGHT visual demo workflow
│   ├── extract_bright_sample.py   # Sample extraction helper
│   ├── test_bright_model.py       # Model/data smoke tests
│   ├── test_bright_training.py    # Training-path smoke test
│   └── validate_bright_splits.py  # Official split validation
│
├── src/
│   └── georescue/
│       ├── bright_dataset.py       # Dataset loading and geometry checks
│       ├── bright_preprocessing.py # BRIGHT preprocessing
│       ├── bright_splits.py       # Split definitions/utilities
│       ├── config.py               # Configuration models
│       ├── models.py               # U-Net/Fusion U-Net
│       └── train_utils.py          # Training, evaluation, checkpoints
│
├── tests/
│   ├── test_bright_dataset.py
│   ├── test_bright_splits.py
│   └── ...
│
├── outputs/
│   ├── checkpoints/
│   └── predictions/
│
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/<YOUR_USERNAME>/GeoRescueAI.git
cd GeoRescueAI
```

### 2. Create a virtual environment

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Verify the codebase

```bash
python -m pytest -q
```

The repository's current test suite contains **14 passing tests** in the validated development state.

### 5. Run the BRIGHT real-data smoke test

Example:

```powershell
python scripts/train_bright.py `
  --bright-root "C:\Datasets\BRIGHT\local-test" `
  --split-root "data\splits\dfc25" `
  --smoke-test
```

Expected behavior includes:

```text
Optical : (1, 3, 1024, 1024)
SAR     : (1, 3, 1024, 1024)
Target  : (1, 1024, 1024)
Logits  : (1, 2, 1024, 1024)
```

The smoke test validates the real-data tensor path, model forward pass, loss calculation, and gradient flow. It is **not** a claim of trained model accuracy.

---

## Run the API

Start the FastAPI application with Uvicorn:

```powershell
python -m uvicorn app.api:app --reload
```

Default development address:

```text
http://127.0.0.1:8000
```

Open:

```text
http://127.0.0.1:8000/
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

### Health check

```http
GET /health
```

### Image inference

```http
POST /predict
```

The endpoint accepts optical and SAR image uploads and returns the generated segmentation result.

### BRIGHT demo resources

The dashboard also consumes the BRIGHT demo routes exposed by the application, including:

```text
GET /bright-demo/info
GET /bright-demo/<asset>.png
```

These routes are intended for a repeatable visual case-study workflow.

---

## Dashboard

The frontend is intentionally designed as an **analysis workstation**, not a generic admin template.

### Main workspace

The interface brings together:

- pre-event optical imagery
- post-event SAR imagery
- reference/target imagery
- segmentation and overlay views
- active-case information
- runtime/system state
- inference controls
- analysis details
- export actions

### Interaction model

The dashboard supports:

```text
FIT
100% ZOOM
GRID
RUN INFERENCE
VIEW DETAILS
EXPORT
```

Custom optical/SAR files can be submitted through the inference workflow and the returned prediction can be displayed in the analysis workspace.

### Design goals

The UI follows a restrained operational aesthetic:

- dark mission-control layout
- technical typography
- compact status indicators
- high-contrast imagery workspace
- green/cyan system accents
- minimal decorative UI
- information hierarchy centered on the imagery

A current screenshot can be placed at:

```text
docs/assets/dashboard.png
```

Then displayed here:

```markdown
![GeoRescue AI Dashboard](docs/assets/dashboard.png)
```

---

## API Example

Python example using `requests`:

```python
import requests

url = "http://127.0.0.1:8000/predict"

with open("optical.png", "rb") as optical, open("sar.png", "rb") as sar:
    response = requests.post(
        url,
        files={
            "optical": ("optical.png", optical, "image/png"),
            "sar": ("sar.png", sar, "image/png"),
        },
        timeout=120,
    )

response.raise_for_status()

with open("prediction.png", "wb") as output:
    output.write(response.content)
```

---

## BRIGHT Data Workflow

A typical development workflow looks like this:

```text
BRIGHT archive
     │
     ▼
Local dataset
     │
     ├──► Split validation
     │
     ├──► File discovery
     │
     ├──► Raster geometry checks
     │
     └──► Dataset loader
               │
               ▼
          Preprocessing
               │
               ▼
        Optical / SAR tensors
               │
               ▼
            Model
               │
               ▼
        Segmentation logits
               │
               ▼
          Prediction mask
               │
               ▼
        Dashboard / Export
```

---

## Training

The BRIGHT training entry point is:

```bash
python scripts/train_bright.py
```

A smoke test is recommended before starting a long training run:

```bash
python scripts/train_bright.py \
  --bright-root /path/to/BRIGHT \
  --split-root data/splits/dfc25 \
  --smoke-test
```

For an actual training experiment, use the full BRIGHT dataset and an experiment-specific configuration/checkpoint directory.

### Reproducibility

The training utilities include seed control and checkpoint helpers. For a research run, record:

- dataset version
- split files
- model configuration
- random seed
- crop/augmentation settings
- optimizer and learning-rate schedule
- checkpoint path
- evaluation metrics

---

## Evaluation

GeoRescue AI treats **software validation** and **model evaluation** as separate concerns.

### Engineering validation

The automated test suite checks project behavior such as:

- dataset discovery
- split handling
- preprocessing contracts
- geometry validation
- model construction
- training-path behavior

Run:

```bash
python -m pytest -q
```

### Model evaluation

For a research benchmark, metrics should be reported only after running the trained model on the appropriate held-out data.

Recommended segmentation metrics include:

| Metric | Purpose |
|---|---|
| **mIoU** | Measures region overlap across classes |
| **IoU** | Class-specific overlap |
| **Dice / F1** | Overlap-focused segmentation metric |
| **Precision** | Fraction of predicted positive pixels that are correct |
| **Recall** | Fraction of actual positive pixels recovered |
| **Pixel Accuracy** | Overall pixel classification accuracy |

> The repository should only publish measured values from actual experiments. The README intentionally does not invent benchmark accuracy numbers.

---

## Important Engineering Decisions

### Spatial consistency matters

Optical imagery, SAR imagery, and target labels must stay spatially aligned.

The BRIGHT loader validates:

- array dimensions
- geotransforms
- bounds
- spatial consistency across modalities

Floating-point transform differences introduced by raster metadata are handled with explicit numerical tolerances rather than exact string/float equality.

### Negative NumPy strides

Some NumPy augmentation operations can create negative strides. Before converting arrays to PyTorch tensors, the preprocessing path makes the data contiguous.

This prevents failures such as:

```text
ValueError: At least one stride in the given numpy array is negative
```

### Local subsets

The code supports a non-strict split mode so a small local dataset can be used for engineering tests without pretending that missing official samples are present.

---

## Current Development Status

### Implemented

- [x] BRIGHT dataset discovery
- [x] BRIGHT split-file support
- [x] Geometry validation
- [x] Optical/SAR preprocessing
- [x] Optical U-Net baseline
- [x] SAR U-Net baseline
- [x] Optical + SAR Fusion U-Net
- [x] PyTorch training utilities
- [x] Real-data smoke test
- [x] Official split validation
- [x] FastAPI backend
- [x] BRIGHT demo endpoints
- [x] Interactive analysis dashboard
- [x] Inference upload workflow
- [x] Metadata/details view
- [x] Report/export workflow
- [x] Automated tests

### Research-ready next steps

- [ ] Full BRIGHT DFC25 training run
- [ ] Formal validation/test benchmarking
- [ ] Ablation study: optical vs. SAR vs. fusion
- [ ] Hyperparameter experiments
- [ ] Stronger augmentation strategy
- [ ] Calibration / uncertainty analysis
- [ ] Model explainability
- [ ] Deployment containerization
- [ ] Production authentication and access control
- [ ] Experiment tracking

---

## Limitations

GeoRescue AI is currently a **research and demonstration system**, not an operational disaster-response platform.

Important limitations include:

1. **Model quality depends on training data and training configuration.** A working inference pipeline does not imply benchmark performance.
2. **A local smoke-test sample is not sufficient for scientific evaluation.**
3. **The dashboard is an analysis interface, not a certified emergency-response tool.**
4. **Predictions should be reviewed by domain experts before operational use.**
5. **Satellite imagery quality, acquisition timing, registration error, clouds, sensor characteristics, and scene type can affect segmentation quality.**
6. **Production deployment requires security, observability, data governance, and robust error handling beyond the development setup.**

---

## Research Reproducibility

For publication-quality experiments, keep the following with every run:

```text
Experiment/
├── config.yaml
├── split_manifest.txt
├── seed.txt
├── metrics.json
├── training_history.json
├── best_model.pt
├── final_model.pt
└── qualitative_examples/
```

This makes it possible to connect a reported result to the exact data split, configuration, and checkpoint that produced it.

---

## Citation

If you use the BRIGHT dataset, please cite the official BRIGHT dataset/paper in accordance with its documentation and license.

For this repository, add the exact paper/reference metadata used by your study here before publication:

```bibtex
@article{bright_placeholder,
  title   = {BRIGHT: ...},
  author  = {...},
  journal = {...},
  year    = {...}
}
```

Do **not** leave the placeholder citation in a published paper or final project submission; replace it with the official citation corresponding to the BRIGHT DFC25 release you used.

---

## License

This project is released under the **MIT License**.

See [`LICENSE`](LICENSE) for details.

> Dataset terms may differ from the software license. Review the BRIGHT dataset's own license and usage requirements before redistribution.

---

## Acknowledgements

GeoRescue AI builds on open-source research and engineering ecosystems including:

- **BRIGHT / DFC25** for the disaster-damage benchmark
- **PyTorch** for deep-learning model development
- **FastAPI** for the inference service
- **Python** for the project runtime and tooling

Full attribution should be preserved according to the original licenses and publication requirements of each dependency and dataset.

---

## Author

**Izhar Ahmad**

Computer Science | AI Engineering | Applied Machine Learning

GeoRescue AI is developed as a research-oriented project combining:

```text
Computer Vision
+ Remote Sensing
+ Multimodal Learning
+ Semantic Segmentation
+ Applied Machine Learning
+ Full-Stack ML Engineering
```

---

## ⭐ Why this project matters

Disaster assessment is time-sensitive.

GeoRescue AI explores how multimodal satellite observations can be converted into an interpretable, operator-facing damage-assessment workflow — from raw imagery and spatial validation all the way to segmentation output and visual analysis.

```text
Satellite Data
      ↓
Spatially Validated Inputs
      ↓
Optical + SAR Fusion
      ↓
Deep Segmentation
      ↓
Damage Map
      ↓
Human Review
      ↓
Faster Assessment
```

<p align="center">
  <strong>Built for research. Designed for real-world thinking.</strong>
</p>
