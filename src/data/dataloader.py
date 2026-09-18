"""DataLoader factory for training and evaluation.

Creates train and val DataLoaders with population-balanced sampling
for training and standard distributed sampling for validation.
"""

from __future__ import annotations

import json
from pathlib import Path

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
    # Resolve data paths (relative to project root)
    processed_dir = Path(data_cfg.get("processed_dir", "data/processed"))
    expected_method = data_cfg.get("dim_reduction_method")
    if expected_method:
        metadata_path = Path(
            data_cfg.get(
                "preprocessing_metadata_path",
                processed_dir / "preprocessing_metadata.json",
            )
        )
        if not metadata_path.exists():
            raise ValueError(
                f"Missing preprocessing metadata for {expected_method!r}: {metadata_path}"
            )
        actual_method = json.loads(metadata_path.read_text()).get(
            "dim_reduction_method"
        )
        if actual_method != expected_method:
            raise ValueError(
                f"Preprocessing method mismatch: expected {expected_method!r}, "
                f"artifact records {actual_method!r}"
            )
    train_path = processed_dir / "train_data.pkl"
    val_path = processed_dir / "val_data.pkl"

    # Create datasets
    train_dataset = GenotypeDataset(data_path=train_path)
    val_dataset = GenotypeDataset(data_path=val_path)
    expected = (data_cfg["gene_size"], data_cfg["num_channels"])
    for dataset, path in ((train_dataset, train_path), (val_dataset, val_path)):
        if tuple(dataset.x_data.shape[1:]) != expected:
            raise ValueError(
                f"{path} has (gene_size, K)={tuple(dataset.x_data.shape[1:])}, config expects "
                f"{expected}; update data.gene_size/num_channels from the preprocessing log"
            )

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
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
