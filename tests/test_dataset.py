from pathlib import Path

import numpy as np

from georescue.datasets import NPZMultiModalDataset


def test_npz_dataset(tmp_path: Path) -> None:
    optical = np.zeros((3, 32, 32), dtype=np.float32)
    sar = np.zeros((1, 32, 32), dtype=np.float32)
    target = np.zeros((32, 32), dtype=np.int64)
    np.savez_compressed(tmp_path / "sample.npz", optical=optical, sar=sar, target=target)
    dataset = NPZMultiModalDataset(tmp_path, 3, 1)
    sample = dataset[0]
    assert sample["optical"].shape == (3, 32, 32)
    assert sample["sar"].shape == (1, 32, 32)
    assert sample["target"].shape == (32, 32)
