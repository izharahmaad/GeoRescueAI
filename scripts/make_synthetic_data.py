"""Generate deterministic synthetic multimodal segmentation data for smoke tests."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def make_sample(rng: np.random.Generator, size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[0:size, 0:size]
    optical = np.zeros((3, size, size), dtype=np.float32)
    sar = np.zeros((1, size, size), dtype=np.float32)
    target = np.zeros((size, size), dtype=np.int64)

    for _ in range(3):
        cx = int(rng.integers(size // 5, 4 * size // 5))
        cy = int(rng.integers(size // 5, 4 * size // 5))
        w = int(rng.integers(max(8, size // 12), max(12, size // 4)))
        h = int(rng.integers(max(8, size // 12), max(12, size // 4)))
        x0, x1 = max(0, cx - w // 2), min(size, cx + w // 2)
        y0, y1 = max(0, cy - h // 2), min(size, cy + h // 2)
        target[y0:y1, x0:x1] = 1

    # The modalities share the same latent structure but have different noise.
    optical[0] = target * 0.75 + rng.normal(0.15, 0.08, (size, size))
    optical[1] = target * 0.35 + rng.normal(0.20, 0.08, (size, size))
    optical[2] = target * 0.20 + rng.normal(0.25, 0.08, (size, size))
    sar[0] = target * 0.65 + rng.normal(0.10, 0.12, (size, size))
    optical = np.clip(optical, 0.0, 1.0).astype(np.float32)
    sar = np.clip(sar, -0.5, 1.0).astype(np.float32)
    return optical, sar, target


def generate_split(out_dir: Path, count: int, size: int, seed: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for i in range(count):
        optical, sar, target = make_sample(rng, size)
        np.savez_compressed(out_dir / f"sample_{i:04d}.npz", optical=optical, sar=sar, target=target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/synthetic")
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--train", type=int, default=24)
    parser.add_argument("--val", type=int, default=8)
    parser.add_argument("--test", type=int, default=8)
    args = parser.parse_args()
    root = Path(args.output)
    generate_split(root / "train", args.train, args.size, 42)
    generate_split(root / "val", args.val, args.size, 43)
    generate_split(root / "test", args.test, args.size, 44)
    print(f"Synthetic dataset created under {root.resolve()}")


if __name__ == "__main__":
    main()
