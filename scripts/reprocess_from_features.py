#!/usr/bin/env python3
"""Re-run tokenize → pad → normalize → zero_mask → save.

Skips VCF parsing and PCA (reuses gene_pca_features.pkl).
Fixes the normalize-before-pad bug by padding first.
"""

from __future__ import annotations

import gc
import logging
import os
import pickle
import sys
import time

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _PROJECT_ROOT)

import pandas as pd

from src.preprocessing.config import PREPROCESS_SEED, PROCESSED_DIR
from src.preprocessing.labels import (
    create_hierarchical_labels,
    pad_to_gene_size,
    save_all,
    split_dataset_stratified,
)
from src.preprocessing.tokenizer import (
    compute_gene_size,
    generate_zero_mask,
    normalize_data,
    tokenize_dataset,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

PANEL_PATH = "data/integrated_call_samples_v3.20130502.ALL.panel"
FEATURES_PATH = "data/processed/gene_pca_features.pkl"


def main():
    t_start = time.time()
    logger.info("=" * 60)
    logger.info("Re-processing from gene_pca_features.pkl (pad→normalize fix)")
    logger.info("=" * 60)

    # Load cached PCA features
    logger.info(f"Loading {FEATURES_PATH}...")
    with open(FEATURES_PATH, "rb") as f:
        features_df = pickle.load(f)
    logger.info(f"Features: {features_df.shape}")

    # Detect optimal_k from column names
    sample_col = features_df.columns[0]
    gene_name = sample_col.rsplit(":", 1)[0]
    optimal_k = sum(1 for c in features_df.columns if c.startswith(f"{gene_name}:"))
    logger.info(f"Detected optimal_k = {optimal_k}")

    # Labels
    labels = create_hierarchical_labels(PANEL_PATH)
    sample_ids = list(features_df.index)

    # Handle sample count mismatch
    if len(features_df) != len(labels["pop_labels"]):
        n_min = min(len(features_df), len(labels["pop_labels"]))
        features_df = features_df.iloc[:n_min]
        labels["pop_labels"] = labels["pop_labels"][:n_min]
        labels["superpop_labels"] = labels["superpop_labels"][:n_min]
        sample_ids = sample_ids[:n_min]

    # Tokenize
    tokenized, n_genes = tokenize_dataset(features_df, optimal_k)

    # Split (80/10/10)
    x_train, x_val, x_test, y_train, y_val, y_test, _ = split_dataset_stratified(
        tokenized, labels, sample_ids,
        val_ratio=0.1, test_ratio=0.1, seed=PREPROCESS_SEED,
    )
    del tokenized
    gc.collect()

    # PAD FIRST (key fix)
    gene_size = compute_gene_size(n_genes)
    x_train = pad_to_gene_size(x_train, gene_size)
    x_val = pad_to_gene_size(x_val, gene_size)
    x_test = pad_to_gene_size(x_test, gene_size)

    # THEN normalize (stats now match gene_size)
    x_train_norm, x_val_norm, x_test_norm, _ = normalize_data(x_train, x_val, x_test)
    del x_train, x_val, x_test
    gc.collect()

    # Zero mask
    generate_zero_mask(x_train_norm, gene_size, optimal_k)

    # Save (already padded)
    save_all(
        x_train_norm, x_val_norm, x_test_norm,
        y_train, y_val, y_test, features_df, gene_size,
    )

    elapsed = time.time() - t_start
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Re-processing complete: {elapsed:.0f}s")
    logger.info(f"  gene_size: {gene_size}, K: {optimal_k}")
    logger.info(f"  normalization_stats shape: ({gene_size}, {optimal_k})")
    logger.info(f"{'=' * 60}")


if __name__ == "__main__":
    main()
