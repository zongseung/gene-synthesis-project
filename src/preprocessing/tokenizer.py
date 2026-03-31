"""Tokenization, normalization, and zero mask generation."""

from __future__ import annotations

import logging
import os
import pickle

import numpy as np
import pandas as pd
import torch

from src.preprocessing.config import GENE_SIZE_ALIGNMENT, PROCESSED_DIR

logger = logging.getLogger(__name__)


def compute_gene_size(n_genes: int) -> int:
    """Compute padded gene_size aligned to 128 (CNN downsample x3 + patch 16)."""
    gene_size = ((n_genes + GENE_SIZE_ALIGNMENT - 1) // GENE_SIZE_ALIGNMENT) * GENE_SIZE_ALIGNMENT
    logger.info(f"gene_size: {n_genes} -> {gene_size} (aligned to {GENE_SIZE_ALIGNMENT})")
    return gene_size


def tokenize_dataset(
    features_df: pd.DataFrame,
    optimal_k: int,
) -> tuple[np.ndarray, int]:
    """Convert PCA features DataFrame to tokenized tensor.

    Column format: "geneName:componentIdx"
    Groups by gene name, creating (n_samples, n_genes, K) array.
    """
    gene_groups: dict[str, list[str]] = {}
    for col in features_df.columns:
        parts = col.rsplit(":", 1)
        gene_name = parts[0]
        if gene_name not in gene_groups:
            gene_groups[gene_name] = []
        gene_groups[gene_name].append(col)

    n_samples = len(features_df)
    n_genes = len(gene_groups)
    max_components = optimal_k

    logger.info(
        f"Tokenizing: {n_samples} samples, {n_genes} genes, "
        f"{max_components} max components"
    )

    tokenized = np.zeros((n_samples, n_genes, max_components), dtype=np.float32)

    for gene_idx, (gene_name, cols) in enumerate(sorted(gene_groups.items())):
        for comp_idx, col in enumerate(sorted(cols)):
            if comp_idx < max_components:
                tokenized[:, gene_idx, comp_idx] = features_df[col].values

    logger.info(f"Tokenized shape: {tokenized.shape}")
    return tokenized, n_genes


def normalize_data(
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Normalize with fp32 statistics computed on train only."""
    xmean = x_train.mean(axis=0).astype(np.float32)
    xstd = x_train.std(axis=0).astype(np.float32)
    xstd[xstd == 0.0] += 1

    x_train_norm = np.clip((x_train - xmean) / xstd, -5.0, 5.0).astype(np.float32)
    x_val_norm = np.clip((x_val - xmean) / xstd, -5.0, 5.0).astype(np.float32)
    x_test_norm = np.clip((x_test - xmean) / xstd, -5.0, 5.0).astype(np.float32)

    stats = {"mean": xmean, "std": xstd}

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    stats_path = os.path.join(PROCESSED_DIR, "normalization_stats.pkl")
    with open(stats_path, "wb") as f:
        pickle.dump(stats, f)

    logger.info(f"Normalization stats saved: {stats_path} (shape: {xmean.shape})")
    return x_train_norm, x_val_norm, x_test_norm, stats


def generate_zero_mask(
    x_train: np.ndarray,
    gene_size: int,
    n_channels: int,
) -> np.ndarray:
    """Generate zero_mask: positions always zero across all training samples."""
    n_samples = x_train.shape[0]
    n_genes = x_train.shape[1]
    n_k = x_train.shape[2]

    padded = np.zeros((n_samples, gene_size, n_channels), dtype=np.float32)
    padded[:, :n_genes, :n_k] = x_train

    zero_mask = np.all(padded == 0, axis=0)

    mask_tensor = torch.tensor(zero_mask)
    mask_path = os.path.join(PROCESSED_DIR, "zero_mask.pt")
    torch.save(mask_tensor, mask_path)

    n_zeros = zero_mask.sum()
    total = zero_mask.size
    logger.info(
        f"Zero mask: {n_zeros}/{total} ({n_zeros / total * 100:.1f}%) "
        f"positions always zero. Saved to {mask_path}"
    )

    return zero_mask
