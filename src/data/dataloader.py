"""DataLoader factory for training and evaluation.

Creates train and val DataLoaders with population-balanced sampling
for training and standard distributed sampling for validation.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, DistributedSampler

from src.data.dataset import GenotypeDataset
from src.data.sampler import PopulationBalancedSampler


def create_dataloaders(
    config: dict,
    rank: int = 0,
    world_size: int = 1,
) -> tuple[DataLoader, DataLoader]:
    """Create train and val DataLoaders from config.

    Args:
        config: Nested config dict with 'data', 'training', 'save_dir' sections.
        rank: DDP rank (0 if not distributed).
        world_size: Number of DDP processes (1 if not distributed).

    Returns:
        Tuple of (train_loader, val_loader).
    """
    data_cfg = config["data"]
    training_cfg = config["training"]
    save_dir = config.get("save_dir", "outputs/default")

    # Resolve data paths (relative to project root)
    processed_dir = Path("data/processed")
    train_path = processed_dir / "train_data.pkl"
    val_path = processed_dir / "val_data.pkl"

    # Create datasets
    train_dataset = GenotypeDataset(data_path=train_path)
    val_dataset = GenotypeDataset(data_path=val_path)

    batch_size = training_cfg["batch_size"]
    num_workers = training_cfg.get("num_workers", 4)
    seed = training_cfg.get("seed", 20260327)

    # --- Train sampler: population-balanced + DDP-aware ---
    train_sampler = PopulationBalancedSampler(
        labels=train_dataset.y_labels,
        num_samples_per_epoch=len(train_dataset),
        rank=rank,
        world_size=world_size,
        seed=seed,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # Prevent batch size mismatch in DDP
    )

    # --- Val sampler: standard distributed (no balancing) ---
    if world_size > 1:
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
        )
    else:
        val_sampler = None

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        shuffle=False if val_sampler is not None else False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
