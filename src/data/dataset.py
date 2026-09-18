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
        ValueError: If the arrays are not (N,G,K) features and (N,) labels
            of equal length.
    """

    def __init__(self, data_path: str | Path) -> None:
        data_path = Path(data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        with open(data_path, "rb") as f:
            x_data, y_labels = pickle.load(f)

        # (N, gene_size, K); transposed to (K, gene_size) in __getitem__.
        self.x_data = torch.from_numpy(np.asarray(x_data, dtype=np.float32))
        self.y_labels = torch.from_numpy(np.asarray(y_labels, dtype=np.int64))
        if self.x_data.ndim != 3 or len(self.x_data) != len(self.y_labels):
            raise ValueError(
                f"{data_path}: expected (N,G,K) features and (N,) labels, got "
                f"{tuple(self.x_data.shape)} and {tuple(self.y_labels.shape)}"
            )

    def __len__(self) -> int:
        return self.x_data.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (x, y) where x is (K, gene_size) float32 and y is int64 scalar."""
        # Transpose from (gene_size, K) to (K, gene_size) for model input (C, L)
        x = self.x_data[idx].permute(1, 0)  # (K, gene_size)
        y = self.y_labels[idx]
        return x, y
