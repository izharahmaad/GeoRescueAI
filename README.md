# GeoRescue AI

**Multimodal Disaster Damage Assessment using Optical + SAR Satellite Imagery**

GeoRescue AI is a research-oriented computer-vision prototype for pixel-level disaster damage segmentation. The initial research question is whether combining pre-disaster optical imagery with post-disaster SAR imagery improves damage mapping and generalization to unseen disaster events compared with single-modality baselines.

## Research pipeline

```text
Optical RGB + SAR
      ↓
Preprocessing / alignment / normalization
      ↓
Optical-only / SAR-only / Optical+SAR models
      ↓
Pixel-level damage segmentation
      ↓
mIoU / Dice-F1 / Precision / Recall / per-class IoU
      ↓
Cross-event evaluation and error analysis
      ↓
FastAPI inference + simple dashboard
```

## Repository principles

- Keep source code, configuration, tests, documentation, and small fixtures in Git.
- Keep raw and large benchmark datasets outside Git and follow the dataset license.
- Do not invent or alter official dataset class definitions.
- Do not report numerical performance until it has been measured from a real experiment.
- Keep experiments reproducible through configuration and deterministic seeding where practical.

## Architecture

### Model modes

- `optical_unet`: optical-only baseline.
- `sar_unet`: SAR-only baseline.
- `fusion_unet`: separate optical/SAR stems followed by feature fusion and U-Net decoding.

The three modes share a common training/evaluation interface so the comparison remains controlled.

### Dataset contract

The current development loader uses `.npz` samples containing:

```text
optical:  [3, H, W] float32
sar:      [1, H, W] float32
target:   [H, W] int64
```

For a real benchmark, create a preprocessing adapter that reads the official format and maps the official labels exactly to the experiment's class IDs.

## Environment

Target: **Python 3.11**.

Windows PowerShell:

```powershell
py --version
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For a CUDA GPU, install the PyTorch build matching the target CUDA runtime using the official PyTorch installation selector rather than hard-coding an incompatible wheel into this repository.

## First smoke test

The project is intentionally designed to work without a real satellite benchmark at first:

```powershell
python scripts/make_synthetic_data.py
python scripts/train.py --config configs/default.yaml --smoke-test
python scripts/evaluate.py --config configs/default.yaml --smoke-test
```

This verifies the dataset loader, model, loss, training loop, checkpointing, and metrics before expensive real-data work begins.

## Normal training

```powershell
python scripts/train.py --config configs/default.yaml
python scripts/evaluate.py --config configs/default.yaml
```

Output artifacts are written under `outputs/` and are ignored by Git.

## Inference

Development inference accepts `.npy` / `.npz` arrays:

```powershell
python scripts/infer.py --config configs/default.yaml --optical path\\to\\optical.npy --sar path\\to\\sar.npy --output outputs\\predictions\\prediction.png
```

The API provides a browser dashboard and `/health` endpoint:

```powershell
uvicorn app.api:app --reload
```

Then open `http://127.0.0.1:8000`.

## Real-dataset integration

Before using BRIGHT, xBD-S12, or another benchmark:

1. Read the dataset's official paper, data card, and license.
2. Verify image geometry, channels, naming, and official label definitions.
3. Build a preprocessing adapter under `src/georescue/` rather than modifying benchmark data in-place.
4. Split by disaster event when studying cross-event generalization.
5. Record the split and preprocessing choices in the experiment configuration.

## Research experiment order

1. Optical-only U-Net
2. SAR-only U-Net
3. Optical + SAR fusion U-Net
4. Improved fusion model
5. Event-aware generalization
6. Error analysis and explainability

## 10-day implementation direction

| Day | Focus |
|---|---|
| 1 | Environment, repository, dataset understanding, smoke test |
| 2 | Preprocessing, modality visualization, class distribution |
| 3 | Optical baseline |
| 4 | SAR baseline |
| 5 | Fusion model |
| 6 | Training, tuning, reproducibility |
| 7 | Metrics, confusion matrix, error analysis |
| 8 | Visualization and explainability |
| 9 | FastAPI and dashboard |
| 10 | README, experiment table, demo, CV wording, Git push |

## Viva focus

Be able to explain:

- what SAR is and why it complements optical imagery;
- why segmentation is used instead of image-level classification;
- why U-Net is a sensible baseline;
- how optical and SAR features are fused;
- what mIoU and Dice measure;
- why accuracy alone can be misleading with class imbalance;
- what an ablation study is;
- why event-level testing is important for generalization;
- what evidence is required before claiming the fusion model is better.

## Status

This repository is the **engineering foundation**. It contains a runnable synthetic-data pipeline, but it does **not** contain official satellite benchmark data or real-world performance claims.
