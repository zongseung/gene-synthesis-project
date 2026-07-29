"""Build train-only population cumulants for Orthogonal Cumulant FiLM."""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np

from src.preprocessing.tokenizer import tokenize_dataset


FORMAT_VERSION = 1


def _unbiased_skew_kurtosis(
    values: np.ndarray,
    eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return bias-corrected skewness and excess kurtosis along axis 0."""
    n = values.shape[0]
    if n < 4:
        raise ValueError(f"At least 4 samples are required, got {n}")

    centered = values - values.mean(axis=0)
    squared = centered * centered
    m2 = squared.mean(axis=0)
    valid = m2 > eps

    g1 = np.zeros_like(m2)
    g2 = np.zeros_like(m2)
    g1[valid] = (
        (squared * centered).mean(axis=0)[valid] / np.power(m2[valid], 1.5)
    )
    g2[valid] = (
        (squared * squared).mean(axis=0)[valid] / np.square(m2[valid]) - 3.0
    )

    skew = np.sqrt(n * (n - 1)) / (n - 2) * g1
    kurtosis = (n - 1) / ((n - 2) * (n - 3)) * ((n + 1) * g2 + 6.0)
    return skew, kurtosis


def _group_cumulants(
    values: np.ndarray,
    labels: np.ndarray,
    n_groups: int,
    eps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shape = (n_groups, *values.shape[1:])
    skewness = np.zeros(shape, dtype=np.float64)
    kurtosis = np.zeros(shape, dtype=np.float64)
    counts = np.bincount(labels, minlength=n_groups).astype(np.int64)

    for group in range(n_groups):
        if counts[group] < 4:
            raise ValueError(
                f"Group {group} needs at least 4 samples, got {counts[group]}"
            )
        skewness[group], kurtosis[group] = _unbiased_skew_kurtosis(
            values[labels == group], eps
        )
    return skewness, kurtosis, counts


def fit_cumulant_statistics(
    x_train: np.ndarray,
    pop_labels: np.ndarray,
    superpop_labels: np.ndarray,
    *,
    n_pops: int,
    n_superpops: int,
    gene_size: int,
    prior_strength_skew: float = 50.0,
    prior_strength_kurtosis: float = 100.0,
    eps: float = 1e-6,
) -> dict[str, np.ndarray]:
    """Fit per-gene eigen-coordinates and shrunk population cumulants.

    Args:
        x_train: Normalized but unclipped `(N, genes, K)` training features.
        pop_labels: Fine population labels aligned to `x_train`.
        superpop_labels: Superpopulation labels aligned to `x_train`.
        n_pops: Number of fine populations.
        n_superpops: Number of superpopulations.
        gene_size: Padded model gene length.
        prior_strength_skew: Effective superpopulation prior size for skewness.
        prior_strength_kurtosis: Effective prior size for excess kurtosis.
        eps: Eigenvalue and variance floor.
    """
    x = np.asarray(x_train)
    pop = np.asarray(pop_labels, dtype=np.int64)
    superpop = np.asarray(superpop_labels, dtype=np.int64)

    if x.ndim != 3:
        raise ValueError(f"x_train must be 3D, got shape {x.shape}")
    if len(x) != len(pop) or len(x) != len(superpop):
        raise ValueError("Feature and label sample counts must match")
    if gene_size < x.shape[1]:
        raise ValueError(
            f"gene_size {gene_size} is smaller than {x.shape[1]} real genes"
        )
    if prior_strength_skew < 0 or prior_strength_kurtosis < 0:
        raise ValueError("Prior strengths must be non-negative")
    if not np.isfinite(x).all():
        raise ValueError("x_train contains non-finite values")

    n_samples, n_genes, n_components = x.shape
    x64 = x.astype(np.float64, copy=False)
    center = x64.mean(axis=0)
    centered = x64 - center
    covariance = np.einsum(
        "ngk,ngl->gkl", centered, centered, optimize=True
    ) / (n_samples - 1)

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    valid = eigenvalues > eps
    inv_sqrt = np.zeros_like(eigenvalues)
    sqrt = np.zeros_like(eigenvalues)
    inv_sqrt[valid] = 1.0 / np.sqrt(eigenvalues[valid])
    sqrt[valid] = np.sqrt(eigenvalues[valid])
    whitener = np.einsum(
        "gki,gi,gli->gkl", eigenvectors, inv_sqrt, eigenvectors, optimize=True
    )
    dewhitener = np.einsum(
        "gki,gi,gli->gkl", eigenvectors, sqrt, eigenvectors, optimize=True
    )
    rotated = np.einsum(
        "ngk,gkl->ngl", centered, eigenvectors, optimize=True
    )
    whitened = rotated * inv_sqrt[None, :, :]

    pop_skew, pop_kurtosis, pop_counts = _group_cumulants(
        whitened, pop, n_pops, eps
    )
    super_skew, super_kurtosis, _ = _group_cumulants(
        whitened, superpop, n_superpops, eps
    )

    pop_to_super = np.empty(n_pops, dtype=np.int64)
    for population in range(n_pops):
        parents = np.unique(superpop[pop == population])
        if len(parents) != 1:
            raise ValueError(
                f"Population {population} maps to {len(parents)} superpopulations"
            )
        pop_to_super[population] = parents[0]

    skew_weight = pop_counts / (pop_counts + prior_strength_skew)
    kurtosis_weight = pop_counts / (
        pop_counts + prior_strength_kurtosis
    )
    shrunk_skew = (
        skew_weight[:, None, None] * pop_skew
        + (1.0 - skew_weight[:, None, None]) * super_skew[pop_to_super]
    )
    shrunk_kurtosis = (
        kurtosis_weight[:, None, None] * pop_kurtosis
        + (1.0 - kurtosis_weight[:, None, None])
        * super_kurtosis[pop_to_super]
    )

    center_out = np.zeros((gene_size, n_components), dtype=np.float32)
    identity = np.eye(n_components, dtype=np.float32)
    whitener_out = np.broadcast_to(
        identity, (gene_size, n_components, n_components)
    ).copy()
    dewhitener_out = whitener_out.copy()
    rotation_out = whitener_out.copy()
    eigenvalues_out = np.zeros(
        (gene_size, n_components), dtype=np.float32
    )
    skew_out = np.zeros(
        (n_pops + 1, gene_size, n_components), dtype=np.float32
    )
    kurtosis_out = np.zeros_like(skew_out)

    center_out[:n_genes] = center.astype(np.float32)
    whitener_out[:n_genes] = whitener.astype(np.float32)
    dewhitener_out[:n_genes] = dewhitener.astype(np.float32)
    rotation_out[:n_genes] = eigenvectors.astype(np.float32)
    eigenvalues_out[:n_genes] = eigenvalues.astype(np.float32)
    skew_out[:n_pops, :n_genes] = shrunk_skew.astype(np.float32)
    kurtosis_out[:n_pops, :n_genes] = shrunk_kurtosis.astype(np.float32)

    return {
        "format_version": np.asarray(FORMAT_VERSION, dtype=np.int64),
        "n_samples": np.asarray(n_samples, dtype=np.int64),
        "n_pops": np.asarray(n_pops, dtype=np.int64),
        "n_superpops": np.asarray(n_superpops, dtype=np.int64),
        "n_genes": np.asarray(n_genes, dtype=np.int64),
        "n_components": np.asarray(n_components, dtype=np.int64),
        "gene_size": np.asarray(gene_size, dtype=np.int64),
        "prior_strength_skew": np.asarray(
            prior_strength_skew, dtype=np.float64
        ),
        "prior_strength_kurtosis": np.asarray(
            prior_strength_kurtosis, dtype=np.float64
        ),
        "center": center_out,
        "whitener": whitener_out,
        "dewhitener": dewhitener_out,
        "rotation": rotation_out,
        "eigenvalues": eigenvalues_out,
        "skewness": skew_out,
        "excess_kurtosis": kurtosis_out,
        "pop_counts": pop_counts,
        "skew_shrinkage": skew_weight.astype(np.float32),
        "kurtosis_shrinkage": kurtosis_weight.astype(np.float32),
    }


def save_cumulant_statistics(
    stats: dict[str, np.ndarray],
    output_path: str | Path,
) -> None:
    """Save a cumulant artifact as a compressed, pickle-free NPZ."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **stats)


def _infer_gene_size(processed_dir: Path, n_genes: int) -> int:
    zero_mask_path = processed_dir / "zero_mask.pt"
    if not zero_mask_path.exists():
        return n_genes

    import torch

    zero_mask = torch.load(zero_mask_path, map_location="cpu", weights_only=True)
    return int(zero_mask.shape[0])


def build_from_processed(
    processed_dir: str | Path,
    *,
    gene_size: int | None = None,
    prior_strength_skew: float = 50.0,
    prior_strength_kurtosis: float = 100.0,
    eps: float = 1e-6,
) -> dict[str, np.ndarray]:
    """Build the production artifact from existing preprocessing outputs."""
    directory = Path(processed_dir)
    with open(directory / "gene_pca_features.pkl", "rb") as handle:
        features = pickle.load(handle)
    with open(directory / "split_manifest.json") as handle:
        manifest = json.load(handle)
    with open(directory / "label_hierarchy.pkl", "rb") as handle:
        hierarchy = pickle.load(handle)
    with open(directory / "normalization_stats.pkl", "rb") as handle:
        normalization = pickle.load(handle)

    train_indices = np.asarray(manifest["train_indices"], dtype=np.int64)
    train_features = features.iloc[train_indices]
    n_components = int(normalization["mean"].shape[1])
    tokenized, n_genes = tokenize_dataset(train_features, n_components)

    mean = np.asarray(normalization["mean"][:n_genes], dtype=np.float32)
    std = np.asarray(normalization["std"][:n_genes], dtype=np.float32)
    normalized = (tokenized - mean) / std

    target_gene_size = gene_size or _infer_gene_size(directory, n_genes)
    return fit_cumulant_statistics(
        normalized,
        np.asarray(hierarchy["pop_labels"])[train_indices],
        np.asarray(hierarchy["superpop_labels"])[train_indices],
        n_pops=len(hierarchy["pop_to_idx"]),
        n_superpops=len(hierarchy["superpop_to_idx"]),
        gene_size=target_gene_size,
        prior_strength_skew=prior_strength_skew,
        prior_strength_kurtosis=prior_strength_kurtosis,
        eps=eps,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build train-only OC-FiLM cumulant statistics"
    )
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument(
        "--output", default="data/processed/cumulant_stats.npz"
    )
    parser.add_argument("--gene-size", type=int)
    parser.add_argument("--prior-strength-skew", type=float, default=50.0)
    parser.add_argument(
        "--prior-strength-kurtosis", type=float, default=100.0
    )
    parser.add_argument("--eps", type=float, default=1e-6)
    args = parser.parse_args()

    started = time.perf_counter()
    stats = build_from_processed(
        args.processed_dir,
        gene_size=args.gene_size,
        prior_strength_skew=args.prior_strength_skew,
        prior_strength_kurtosis=args.prior_strength_kurtosis,
        eps=args.eps,
    )
    save_cumulant_statistics(stats, args.output)
    elapsed = time.perf_counter() - started
    print(
        f"saved={args.output} samples={int(stats['n_samples'])} "
        f"pops={int(stats['n_pops'])} genes={int(stats['n_genes'])} "
        f"components={int(stats['n_components'])} seconds={elapsed:.3f}"
    )


if __name__ == "__main__":
    main()
