"""Hierarchical population labels, stratified split, and artifact saving."""

from __future__ import annotations

import json
import logging
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.preprocessing.config import PREPROCESS_SEED, PROCESSED_DIR

logger = logging.getLogger(__name__)


def create_hierarchical_labels(panel_path: str) -> dict:
    """Create hierarchical population labels from the panel file."""
    panel = pd.read_csv(panel_path, sep="\t")

    superpop_to_idx = {
        sp: i for i, sp in enumerate(sorted(panel["super_pop"].unique()))
    }
    pop_to_idx = {p: i for i, p in enumerate(sorted(panel["pop"].unique()))}

    pop_to_superpop = {}
    for _, row in panel.drop_duplicates(subset="pop").iterrows():
        pop_to_superpop[pop_to_idx[row["pop"]]] = superpop_to_idx[row["super_pop"]]

    idx_to_pop = {v: k for k, v in pop_to_idx.items()}
    idx_to_superpop = {v: k for k, v in superpop_to_idx.items()}

    labels = {
        "pop_to_idx": pop_to_idx,
        "idx_to_pop": idx_to_pop,
        "superpop_to_idx": superpop_to_idx,
        "idx_to_superpop": idx_to_superpop,
        "pop_to_superpop": pop_to_superpop,
        "pop_sizes": dict(panel["pop"].value_counts()),
        "pop_labels": np.array([pop_to_idx[p] for p in panel["pop"]], dtype=np.int64),
        "superpop_labels": np.array(
            [superpop_to_idx[sp] for sp in panel["super_pop"]], dtype=np.int64
        ),
    }

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    label_path = os.path.join(PROCESSED_DIR, "label_hierarchy.pkl")
    with open(label_path, "wb") as f:
        pickle.dump(labels, f)

    logger.info(
        f"Labels: {len(pop_to_idx)} populations, "
        f"{len(superpop_to_idx)} superpopulations"
    )
    logger.info(f"Saved to {label_path}")

    return labels


def compute_split_indices(
    pop_labels: np.ndarray,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = PREPROCESS_SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Population-stratified 80/10/10 index split (no data tensor touched).

    Used by the preprocessing pipeline to determine train rows *before* gene
    PCA runs, so PCA can fit on train-only matrices and avoid leakage.
    """
    idx = np.arange(len(pop_labels))
    trainval_idx, test_idx = train_test_split(
        idx,
        test_size=test_ratio,
        random_state=seed,
        shuffle=True,
        stratify=pop_labels,
    )
    trainval_labels = pop_labels[trainval_idx]
    relative_val_ratio = val_ratio / (1.0 - test_ratio)
    train_idx, val_idx = train_test_split(
        trainval_idx,
        test_size=relative_val_ratio,
        random_state=seed,
        shuffle=True,
        stratify=trainval_labels,
    )
    return train_idx, val_idx, test_idx


def split_dataset_stratified(
    tokenized: np.ndarray,
    labels: dict,
    sample_ids: list[str],
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = PREPROCESS_SEED,
    precomputed_indices: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Population-stratified train/val/test split (80/10/10).

    If `precomputed_indices=(train_idx, val_idx, test_idx)` is provided (e.g.
    the indices used to fit per-gene PCA on train-only), reuse them instead
    of resplitting — guarantees tokenized rows are consistent with the PCA
    train/val/test partition.
    """
    pop_labels = labels["pop_labels"]

    if len(tokenized) != len(pop_labels):
        raise ValueError(
            f"Sample count mismatch: tokenized={len(tokenized)}, "
            f"labels={len(pop_labels)}"
        )

    if precomputed_indices is not None:
        train_idx, val_idx, test_idx = precomputed_indices
    else:
        train_idx, val_idx, test_idx = compute_split_indices(
            pop_labels, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed,
        )

    x_train = tokenized[train_idx]
    x_val = tokenized[val_idx]
    x_test = tokenized[test_idx]
    y_train = pop_labels[train_idx]
    y_val = pop_labels[val_idx]
    y_test = pop_labels[test_idx]

    def _pop_counts(y):
        return {
            str(k): int(v)
            for k, v in pd.Series(y).value_counts().sort_index().items()
        }

    def _sample_ids_for(indices):
        return [sample_ids[i] for i in indices] if sample_ids else []

    split_manifest = {
        "seed": seed,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "n_total": int(len(train_idx) + len(val_idx) + len(test_idx)),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "train_indices": train_idx.tolist(),
        "val_indices": val_idx.tolist(),
        "test_indices": test_idx.tolist(),
        "train_sample_ids": _sample_ids_for(train_idx),
        "val_sample_ids": _sample_ids_for(val_idx),
        "test_sample_ids": _sample_ids_for(test_idx),
        "train_pop_counts": _pop_counts(y_train),
        "val_pop_counts": _pop_counts(y_val),
        "test_pop_counts": _pop_counts(y_test),
    }

    manifest_path = os.path.join(PROCESSED_DIR, "split_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(split_manifest, f, indent=2)

    logger.info(
        f"Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)} "
        f"(ratio={1-val_ratio-test_ratio:.0%}/{val_ratio:.0%}/{test_ratio:.0%}, seed={seed})"
    )

    return x_train, x_val, x_test, y_train, y_val, y_test, split_manifest


def pad_to_gene_size(data: np.ndarray, gene_size: int) -> np.ndarray:
    """Pad (N, n_genes, K) to (N, gene_size, K) with zeros."""
    n_samples, n_genes, n_k = data.shape
    if n_genes >= gene_size:
        return data[:, :gene_size, :]

    padded = np.zeros((n_samples, gene_size, n_k), dtype=data.dtype)
    padded[:, :n_genes, :] = data
    return padded


def save_all(
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    features_df: pd.DataFrame,
    gene_size: int,
) -> None:
    """Save all preprocessed artifacts to data/processed/."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    x_train_padded = pad_to_gene_size(x_train, gene_size)
    x_val_padded = pad_to_gene_size(x_val, gene_size)
    x_test_padded = pad_to_gene_size(x_test, gene_size)

    for name, x, y in [
        ("train", x_train_padded, y_train),
        ("val", x_val_padded, y_val),
        ("test", x_test_padded, y_test),
    ]:
        path = os.path.join(PROCESSED_DIR, f"{name}_data.pkl")
        with open(path, "wb") as f:
            pickle.dump((x, y), f, protocol=4)
        logger.info(f"{name.capitalize()} data: {path} {x.shape}")

    features_path = os.path.join(PROCESSED_DIR, "gene_pca_features.pkl")
    with open(features_path, "wb") as f:
        pickle.dump(features_df, f, protocol=4)
    logger.info(f"PCA features: {features_path} {features_df.shape}")
