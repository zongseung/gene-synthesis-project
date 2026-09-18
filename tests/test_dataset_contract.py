from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

from src.data.dataset import GenotypeDataset


def _write(path: Path, x, y) -> None:
    with path.open("wb") as handle:
        pickle.dump((x, y), handle)


def test_getitem_returns_channels_first(tmp_path: Path) -> None:
    # Given a (N, gene_size, K) pickle as preprocessing writes it.
    path = tmp_path / "train_data.pkl"
    x = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    _write(path, x, np.array([0, 1]))

    sample, label = GenotypeDataset(path)[1]

    # Then the model sees (K, gene_size) and the label is a scalar.
    assert tuple(sample.shape) == (4, 3)
    np.testing.assert_allclose(sample.numpy(), x[1].T)
    assert int(label) == 1


def test_mismatched_lengths_are_rejected(tmp_path: Path) -> None:
    # Given one more feature row than labels.
    path = tmp_path / "train_data.pkl"
    _write(path, np.zeros((3, 2, 2), dtype=np.float32), np.array([0, 1]))

    with pytest.raises(ValueError, match="labels"):
        GenotypeDataset(path)
