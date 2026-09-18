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


#: Population-conditional prior arms (PriorGrad ICLR 2022 / ShiftDDPMs AAAI 2023
#: "Data-Normalization"): the per-population moments move into the normalization
#: step so the diffusion model only ever learns the residual.
#:   "none"     — one global mean and std (the published baseline).
#:   "mean"     — per-population mean, pooled within-population std. Centres each
#:                population and keeps their diversity differences visible.
#:   "mean_std" — per-population mean and std. Also whitens each population, so
#:                diversity differences disappear from the model's view.
CONDITIONAL_ARMS = ("none", "mean", "mean_std")


class NormalizationStats(TypedDict):
    version: int
    mean: np.ndarray
    std: np.ndarray
    shape: tuple[int, int]
    fit_split: str
    clip: float | None
    conditional: str
    variance_floor: float


def _conditional_moments(
    x_train: np.ndarray,
    labels: np.ndarray,
    conditional: str,
    variance_floor: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-population mean plus either the pooled within-population or per-population std."""
    index = np.asarray(labels, dtype=np.int64).reshape(-1)
    if len(index) != len(x_train):
        raise ValueError(f"Got {len(index)} labels for {len(x_train)} training samples")
    if index.min() < 0:
        raise ValueError(f"Population labels must be non-negative, got {index.min()}")
    n_pops = int(index.max()) + 1
    mean = np.zeros((n_pops, *x_train.shape[1:]), dtype=np.float32)
    per_pop_std = np.zeros_like(mean)
    # Pooled within-population variance, accumulated in fp64 so the "mean" arm
    # never materializes a second copy of the residual tensor.
    pooled = np.zeros(x_train.shape[1:], dtype=np.float64)
    for pop in range(n_pops):
        rows = x_train[index == pop]
        if len(rows) == 0:
            raise ValueError(
                f"Population {pop} has no training samples; its conditional prior is undefined"
            )
        mean[pop] = rows.mean(axis=0)
        variance = rows.var(axis=0)
        per_pop_std[pop] = np.sqrt(variance)
        pooled += variance.astype(np.float64) * len(rows)
    pooled_std = np.sqrt(pooled / len(x_train)).astype(np.float32)
    if conditional == "mean":
        return mean, pooled_std
    # Variance floor on the per-population prior, as PriorGrad applies to its
    # data-dependent prior. Minority populations are numerically constant in a
    # small fraction of cells (measured on 1KG train: 7,925 of 2,555,904 cells
    # have a per-population std below 1e-3 of the pooled std, some down to
    # 2e-11). Dividing by those amplifies fp32 rounding dust into residuals of
    # order one, which is training signal made of nothing. The floor sends them
    # to zero instead.
    floor = np.float32(variance_floor) * pooled_std
    clamped = int((per_pop_std < floor).sum())
    if clamped:
        logger.info(
            f"Variance floor {variance_floor:g} x pooled std applied to {clamped}/"
            f"{per_pop_std.size} per-population cells "
            f"({clamped / per_pop_std.size:.3%}, numerically constant within a population)"
        )
    return mean, np.maximum(per_pop_std, floor)


def _select_moments(
    stats: NormalizationStats,
    n_samples: int,
    labels: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Broadcast global moments, or gather the per-population ones by label."""
    mean, std = stats["mean"], stats["std"]
    if mean.ndim == 2 and std.ndim == 2:
        return mean, std
    if labels is None:
        raise ValueError(
            f"Conditional ({stats['conditional']}) normalization statistics require "
            "per-sample population labels"
        )
    index = np.asarray(labels, dtype=np.int64).reshape(-1)
    if len(index) != n_samples:
        raise ValueError(f"Got {len(index)} labels for {n_samples} samples")
    n_pops = (mean if mean.ndim == 3 else std).shape[0]
    if index.min() < 0 or index.max() >= n_pops:
        raise ValueError(
            f"Population labels must lie in [0, {n_pops}), got "
            f"[{index.min()}, {index.max()}]"
        )
    return (
        mean[index] if mean.ndim == 3 else mean,
        std[index] if std.ndim == 3 else std,
    )


def fit_normalization_stats(
    x_train: np.ndarray,
    *,
    clip: float | None = None,
    labels: np.ndarray | None = None,
    conditional: str = "none",
    variance_floor: float = 0.01,
) -> NormalizationStats:
    """Fit featurewise float32 mean/std on train ``(N,G,K)`` only.

    With ``conditional`` in ``("mean", "mean_std")`` the moments become
    per-population ``(P,G,K)`` and ``labels`` is required. ``variance_floor``
    bounds each per-population std below at that multiple of the pooled
    within-population std, and only the ``"mean_std"`` arm has a per-population
    denominator for it to act on.
    """
    if x_train.ndim != 3 or x_train.shape[0] == 0:
        raise ValueError(f"Expected train tensor (N,G,K), got {x_train.shape}")
    if not np.isfinite(x_train).all():
        raise ValueError("Training tensor must contain only finite values")
    if clip is not None and (not np.isfinite(clip) or clip <= 0):
        raise ValueError(f"clip must be finite and positive, got {clip}")
    if conditional not in CONDITIONAL_ARMS:
        raise ValueError(
            f"Unknown conditional prior arm {conditional!r}; expected one of {CONDITIONAL_ARMS}"
        )
    if not 0 <= variance_floor < 1:
        raise ValueError(f"variance_floor must lie in [0, 1), got {variance_floor}")
    if conditional == "none":
        mean = x_train.mean(axis=0).astype(np.float32)
        std = x_train.std(axis=0).astype(np.float32)
    else:
        if labels is None:
            raise ValueError(f"conditional={conditional!r} requires population labels")
        mean, std = _conditional_moments(x_train, labels, conditional, variance_floor)
    std[std == 0.0] = 1.0
    return {
        "version": 2,
        "mean": mean,
        "std": std,
        "shape": mean.shape[-2:],
        "fit_split": "train",
        "clip": clip,
        "conditional": conditional,
        "variance_floor": float(variance_floor),
    }


def apply_normalization(
    x: np.ndarray,
    stats: NormalizationStats,
    labels: np.ndarray | None = None,
) -> np.ndarray:
    """Apply persisted ``(G,K)`` statistics without refitting or shape coercion."""
    expected = tuple(stats["shape"])
    if x.ndim != 3 or x.shape[1:] != expected:
        raise ValueError(f"Normalization shape {expected} does not match tensor {x.shape}")
    mean, std = _select_moments(stats, len(x), labels)
    transformed = (x - mean) / std
    clip = stats["clip"]
    if clip is not None:
        transformed = np.clip(transformed, -clip, clip)
    return transformed.astype(np.float32)


def invert_normalization(
    x: np.ndarray,
    stats: NormalizationStats,
    labels: np.ndarray | None = None,
) -> np.ndarray:
    """Invert persisted ``(G,K)`` statistics without padding or clipping."""
    expected = tuple(stats["shape"])
    if x.ndim != 3 or x.shape[1:] != expected:
        raise ValueError(f"Normalization shape {expected} does not match tensor {x.shape}")
    mean, std = _select_moments(stats, len(x), labels)
    return (x * std + mean).astype(np.float32)


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
    conditional = str(raw.get("conditional", "none"))
    if conditional not in CONDITIONAL_ARMS:
        raise ValueError(f"Unknown conditional prior arm {conditional!r} in {path}")
    if (
        mean.ndim not in (2, 3)
        or std.ndim not in (2, 3)
        or std.shape[-2:] != mean.shape[-2:]
        or not np.isfinite(mean).all()
        or not np.isfinite(std).all()
        or np.any(std <= 0)
    ):
        raise ValueError(f"Invalid normalization statistics shapes: {mean.shape}, {std.shape}")
    # A per-population moment without its arm recorded (or the reverse) would
    # silently normalize one way and invert another.
    if (conditional == "none") != (mean.ndim == 2):
        raise ValueError(
            f"Conditional arm {conditional!r} does not match mean shape {mean.shape} in {path}"
        )
    if (conditional == "mean_std") != (std.ndim == 3):
        raise ValueError(
            f"Conditional arm {conditional!r} does not match std shape {std.shape} in {path}"
        )
    if mean.ndim == 3 and std.ndim == 3 and std.shape[0] != mean.shape[0]:
        raise ValueError(f"Population count mismatch: {mean.shape[0]} vs {std.shape[0]}")
    shape = tuple(mean.shape[-2:])
    if expected_shape is not None and shape != expected_shape:
        raise ValueError(f"Normalization shape {shape} does not match expected {expected_shape}")
    return {
        "version": int(raw.get("version", 0)),
        "mean": mean,
        "std": std,
        "shape": shape,
        "fit_split": str(raw.get("fit_split", "train")),
        "clip": raw.get("clip", 5.0 if "version" not in raw else None),
        "conditional": conditional,
        "variance_floor": float(raw.get("variance_floor", 0.0)),
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
    labels: tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None] = (None, None, None),
    conditional: str = "none",
    variance_floor: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, NormalizationStats]:
    """Normalize with fp32 statistics computed on train only.

    ``labels`` carries the (train, val, test) population labels, required when
    ``conditional`` selects a per-population arm.
    """
    y_train, y_val, y_test = labels
    stats = fit_normalization_stats(
        x_train, clip=clip, labels=y_train, conditional=conditional,
        variance_floor=variance_floor,
    )
    x_train_norm = apply_normalization(x_train, stats, y_train)
    x_val_norm = apply_normalization(x_val, stats, y_val)
    x_test_norm = apply_normalization(x_test, stats, y_test)

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    output_path = Path(stats_path or os.path.join(PROCESSED_DIR, "normalization_stats.pkl"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump(stats, f)

    logger.info(
        f"Normalization stats saved: {output_path} (shape: {stats['shape']}, "
        f"conditional prior: {stats['conditional']}, mean: {stats['mean'].shape}, "
        f"std: {stats['std'].shape})"
    )
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
