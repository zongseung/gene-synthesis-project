"""Genotype dataset for loading preprocessed Gene PCA features.

Expects pickle files containing (x_data, y_labels) tuples where:
    x_data:   (N, gene_size, K) float32 -- gene PCA features
    y_labels: (N,) int64               -- population labels (0..25)

The dataset transposes x_data to (K, gene_size) per sample in __getitem__
to match the model's expected input layout (B, C, L).
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class GenotypeDataset(Dataset):
    """Dataset for Gene PCA genotype tensors.

    Args:
        data_path: Path to a .pkl file containing (x_data, y_labels).

    Raises:
        FileNotFoundError: If data_path does not exist.
        ValueError: If the pickle does not contain a 2-element tuple
            or arrays have unexpected dimensions.
    """

    def __init__(self, data_path: str | Path) -> None:
        data_path = Path(data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        with open(data_path, "rb") as f:
            data = pickle.load(f)

        if not isinstance(data, tuple) or len(data) != 2:
            raise ValueError(
                f"Expected (x_data, y_labels) tuple, got "
                f"{type(data).__name__} with {len(data) if hasattr(data, '__len__') else '?'} elements"
            )

        x_data, y_labels = data

        # Convert to numpy arrays if needed
        if isinstance(x_data, torch.Tensor):
            x_data = x_data.numpy()
        if isinstance(y_labels, torch.Tensor):
            y_labels = y_labels.numpy()

        x_data = np.asarray(x_data, dtype=np.float32)
        y_labels = np.asarray(y_labels, dtype=np.int64)

        if x_data.ndim != 3:
            raise ValueError(
                f"Expected x_data with 3 dimensions (N, gene_size, K), "
                f"got {x_data.ndim}D with shape {x_data.shape}"
            )

        if y_labels.ndim != 1:
            raise ValueError(
                f"Expected y_labels with 1 dimension (N,), "
                f"got {y_labels.ndim}D with shape {y_labels.shape}"
            )

        if x_data.shape[0] != y_labels.shape[0]:
            raise ValueError(
                f"Sample count mismatch: x_data has {x_data.shape[0]}, "
                f"y_labels has {y_labels.shape[0]}"
            )

        # Store as tensors; transpose happens in __getitem__
        self.x_data = torch.from_numpy(x_data)       # (N, gene_size, K)
        self.y_labels = torch.from_numpy(y_labels)    # (N,)

    def __len__(self) -> int:
        return self.x_data.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (x, y) where x is (K, gene_size) float32 and y is int64 scalar."""
        # Transpose from (gene_size, K) to (K, gene_size) for model input (C, L)
        x = self.x_data[idx].permute(1, 0)  # (K, gene_size)
        y = self.y_labels[idx]
        return x, y
