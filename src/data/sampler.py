"""Population-balanced sampler with sqrt-proportional oversampling.

Minority populations receive higher sampling weight to ensure balanced
representation during training. Compatible with DDP via set_epoch().
"""

from __future__ import annotations

import math
import pickle
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import Sampler


class PopulationBalancedSampler(Sampler[int]):
    """Sqrt-proportional oversampling sampler for population-balanced training.

    Each sample's weight is proportional to ``1 / sqrt(pop_count)`` where
    ``pop_count`` is the number of samples sharing the same population label.
    This lifts minority populations (e.g., ASW with 61 samples) relative to
    majority populations (e.g., GBR with 91 samples).

    Compatible with DDP: call ``set_epoch(epoch)`` before each epoch to
    ensure different shuffling across epochs and consistent splitting
    across ranks.

    Args:
        labels: 1-D array or tensor of population labels (int, 0..num_classes-1).
        num_samples_per_epoch: Total samples to draw per epoch. If None,
            defaults to len(labels) (one pass through the dataset).
        rank: DDP rank of this process. 0 if not distributed.
        world_size: Total number of DDP processes. 1 if not distributed.
        seed: Base random seed for reproducibility.
    """

    def __init__(
        self,
        labels: np.ndarray | torch.Tensor,
        num_samples_per_epoch: int | None = None,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 20260327,
    ) -> None:
        if isinstance(labels, torch.Tensor):
            labels = labels.numpy()
        self.labels = np.asarray(labels, dtype=np.int64)
        self.num_total = len(self.labels)

        if num_samples_per_epoch is None:
            num_samples_per_epoch = self.num_total
        self.num_samples_per_epoch = num_samples_per_epoch

        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.epoch = 0

        # Compute per-sample weights: 1/sqrt(pop_count), normalized
        unique_pops, counts = np.unique(self.labels, return_counts=True)
        pop_count_map = dict(zip(unique_pops, counts))

        weights = np.zeros(self.num_total, dtype=np.float64)
        for i, label in enumerate(self.labels):
            weights[i] = 1.0 / math.sqrt(pop_count_map[label])

        # Normalize to sum to 1
        weights /= weights.sum()
        self.weights = weights

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch for deterministic shuffling (required for DDP).

        Args:
            epoch: Current epoch number.
        """
        self.epoch = epoch

    def __iter__(self) -> Iterator[int]:
        """Yield sample indices with sqrt-proportional population weighting."""
        rng = np.random.RandomState(self.seed + self.epoch)

        # Draw weighted samples
        indices = rng.choice(
            self.num_total,
            size=self.num_samples_per_epoch,
            replace=True,
            p=self.weights,
        )

        # Shard for DDP: each rank gets a disjoint subset
        indices = indices[self.rank :: self.world_size]

        return iter(indices.tolist())

    def __len__(self) -> int:
        """Number of samples this rank will yield per epoch."""
        return math.ceil(self.num_samples_per_epoch / self.world_size)
