"""Tokenization, normalization, and zero mask generation."""

from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path
from typing import TypedDict

import numpy as np
import pandas as pd
import torch

from src.preprocessing.config import GENE_SIZE_ALIGNMENT, PROCESSED_DIR

logger = logging.getLogger(__name__)


class NormalizationStats(TypedDict):
    version: int
    mean: np.ndarray
    std: np.ndarray
    shape: tuple[int, int]
    fit_split: str
    clip: float | None


def fit_normalization_stats(
    x_train: np.ndarray,
    *,
    clip: float | None = None,
) -> NormalizationStats:
    """Fit featurewise float32 mean/std on train ``(N,G,K)`` only."""
    if x_train.ndim != 3 or x_train.shape[0] == 0:
        raise ValueError(f"Expected train tensor (N,G,K), got {x_train.shape}")
    if not np.isfinite(x_train).all():
        raise ValueError("Training tensor must contain only finite values")
    if clip is not None and (not np.isfinite(clip) or clip <= 0):
        raise ValueError(f"clip must be finite and positive, got {clip}")
    mean = x_train.mean(axis=0).astype(np.float32)
    std = x_train.std(axis=0).astype(np.float32)
    std[std == 0.0] = 1.0
    return {
        "version": 1,
        "mean": mean,
        "std": std,
        "shape": mean.shape,
        "fit_split": "train",
        "clip": clip,
    }


def apply_normalization(x: np.ndarray, stats: NormalizationStats) -> np.ndarray:
    """Apply persisted ``(G,K)`` statistics without refitting or shape coercion."""
    expected = tuple(stats["shape"])
    if x.ndim != 3 or x.shape[1:] != expected:
        raise ValueError(f"Normalization shape {expected} does not match tensor {x.shape}")
    transformed = (x - stats["mean"]) / stats["std"]
    clip = stats["clip"]
    if clip is not None:
        transformed = np.clip(transformed, -clip, clip)
    return transformed.astype(np.float32)


def invert_normalization(x: np.ndarray, stats: NormalizationStats) -> np.ndarray:
    """Invert persisted ``(G,K)`` statistics without padding or clipping."""
    expected = tuple(stats["shape"])
    if x.ndim != 3 or x.shape[1:] != expected:
        raise ValueError(f"Normalization shape {expected} does not match tensor {x.shape}")
    return (x * stats["std"] + stats["mean"]).astype(np.float32)


def load_normalization_stats(
    path: str | Path,
    *,
    expected_shape: tuple[int, int] | None = None,
) -> NormalizationStats:
    """Load current or legacy mean/std pickle and enforce the requested shape."""
    with Path(path).open("rb") as handle:
        raw = pickle.load(handle)
    mean = np.asarray(raw["mean"], dtype=np.float32)
    std = np.asarray(raw["std"], dtype=np.float32)
    if (
        mean.ndim != 2
        or std.shape != mean.shape
        or not np.isfinite(mean).all()
        or not np.isfinite(std).all()
        or np.any(std <= 0)
    ):
        raise ValueError(f"Invalid normalization statistics shapes: {mean.shape}, {std.shape}")
    shape = tuple(mean.shape)
    if expected_shape is not None and shape != expected_shape:
        raise ValueError(f"Normalization shape {shape} does not match expected {expected_shape}")
    return {
        "version": int(raw.get("version", 0)),
        "mean": mean,
        "std": std,
        "shape": shape,
        "fit_split": str(raw.get("fit_split", "train")),
        "clip": raw.get("clip", 5.0 if "version" not in raw else None),
    }


def compute_gene_size(n_genes: int) -> int:
    """Compute padded gene size aligned to the model's factor-256 stride."""
    gene_size = ((n_genes + GENE_SIZE_ALIGNMENT - 1) // GENE_SIZE_ALIGNMENT) * GENE_SIZE_ALIGNMENT
    logger.info(f"gene_size: {n_genes} -> {gene_size} (aligned to {GENE_SIZE_ALIGNMENT})")
    return gene_size


def tokenize_dataset(
    features_df: pd.DataFrame,
    optimal_k: int,
    gene_order: list[str] | None = None,
) -> tuple[np.ndarray, int]:
    """Convert feature columns to ``(N,G,K)`` in an explicit gene order.

    Column format: "geneName:componentIdx"
    """
    gene_groups: dict[str, dict[int, str]] = {}
    for col in features_df.columns:
        gene_name, component = col.rsplit(":", 1)
        if gene_name not in gene_groups:
            gene_groups[gene_name] = {}
        gene_groups[gene_name][int(component)] = col

    ordered_genes = sorted(gene_groups) if gene_order is None else gene_order
    if len(ordered_genes) != len(gene_groups) or set(ordered_genes) != set(gene_groups):
        raise ValueError("gene_order must contain every feature gene exactly once")

    n_samples = len(features_df)
    n_genes = len(gene_groups)
    max_components = optimal_k

    logger.info(
        f"Tokenizing: {n_samples} samples, {n_genes} genes, "
        f"{max_components} max components"
    )

    tokenized = np.zeros((n_samples, n_genes, max_components), dtype=np.float32)

    for gene_idx, gene_name in enumerate(ordered_genes):
        for comp_idx, col in gene_groups[gene_name].items():
            if comp_idx < max_components:
                tokenized[:, gene_idx, comp_idx] = features_df[col].values

    logger.info(f"Tokenized shape: {tokenized.shape}")
    return tokenized, n_genes


def normalize_data(
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    *,
    stats_path: str | Path | None = None,
    clip: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, NormalizationStats]:
    """Normalize with fp32 statistics computed on train only."""
    stats = fit_normalization_stats(x_train, clip=clip)
    x_train_norm = apply_normalization(x_train, stats)
    x_val_norm = apply_normalization(x_val, stats)
    x_test_norm = apply_normalization(x_test, stats)

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    output_path = Path(stats_path or os.path.join(PROCESSED_DIR, "normalization_stats.pkl"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump(stats, f)

    logger.info(f"Normalization stats saved: {output_path} (shape: {stats['shape']})")
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
