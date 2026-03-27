#!/usr/bin/env python3
"""Full preprocessing pipeline: VCF -> Gene PCA -> tokenized tensors.

OOM-safe 2-pass approach:
  Pass 1: Parse chr1,11,22 → PCA grid search → find optimal K → free
  Pass 2: Stream all 22 chr one-by-one → PCA immediately → free variants
Peak RAM ≈ 1 chromosome (~3-5GB for chr1) + accumulated PCA features (~2GB)

Usage:
    python src/preprocessing/run_pipeline.py
"""

from __future__ import annotations

import gc
import logging
import os
import sys
import time
from pathlib import Path

# Allow direct execution: python src/preprocessing/run_pipeline.py
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pandas as pd

from src.preprocessing.config import (
    MAF_THRESHOLD,
    MAX_VARIANTS_PER_GENE,
    MARGINAL_GAIN_DECAY_RATIO,
    MARGINAL_GAIN_THRESHOLD,
    PANEL_PATH,
    PCA_CANDIDATES,
    PCA_SAMPLE_GENES,
    PREPROCESS_SEED,
    PROCESSED_DIR,
    REFGENE_PATH,
    VCF_PATH,
    VCF_TBI_PATH,
)
from src.preprocessing.gene_annotation import load_refgene
from src.preprocessing.labels import create_hierarchical_labels, save_all, split_dataset_stratified
from src.preprocessing.pca import (
    analyze_pca_information_loss,
    grid_search_optimal_pca,
    stream_vcf_and_pca,
)
from src.preprocessing.tokenizer import compute_gene_size, generate_zero_mask, normalize_data, tokenize_dataset
from src.preprocessing.vcf_parser import process_one_chromosome

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def validate_input_files() -> None:
    """Check that required input files exist."""
    if not Path(VCF_PATH).exists():
        raise FileNotFoundError(f"Merged VCF not found: {VCF_PATH}")
    if not Path(VCF_TBI_PATH).exists():
        raise FileNotFoundError(
            f"Tabix index not found: {VCF_TBI_PATH}. "
            f"Run: tabix -p vcf {VCF_PATH}"
        )
    if not Path(PANEL_PATH).exists():
        raise FileNotFoundError(f"Panel file not found: {PANEL_PATH}")
    if not Path(REFGENE_PATH).exists():
        raise FileNotFoundError(f"RefGene annotation not found: {REFGENE_PATH}")

    logger.info(f"Input validated: {VCF_PATH} + .tbi + panel + refGene")


def main() -> None:
    t_start = time.time()
    logger.info("=" * 60)
    logger.info("HybridGenoDiT Preprocessing Pipeline (OOM-safe)")
    logger.info(f"  VCF: {VCF_PATH}")
    logger.info("=" * 60)

    # Step 0: Validate
    validate_input_files()

    # Load gene annotations (RefGene)
    gene_coords = load_refgene(REFGENE_PATH)

    # Step 1 (Pass 1): PCA grid search on chr1,11,22
    grid_search_chroms = [1, 11, 22]
    logger.info(f"Pass 1: PCA grid search on chr{grid_search_chroms}")

    subset_genes = {}
    sample_ids = []
    for chrom in grid_search_chroms:
        chrom_genes = gene_coords.get(str(chrom), [])
        args = (chrom, VCF_PATH, MAF_THRESHOLD, MAX_VARIANTS_PER_GENE, chrom_genes)
        _, gene_matrices, sids = process_one_chromosome(args)
        subset_genes.update(gene_matrices)
        del gene_matrices
        if not sample_ids and sids:
            sample_ids = sids

    optimal_k, _ = grid_search_optimal_pca(
        subset_genes,
        candidates=PCA_CANDIDATES,
        n_sample_genes=PCA_SAMPLE_GENES,
        marginal_threshold=MARGINAL_GAIN_THRESHOLD,
        decay_ratio=MARGINAL_GAIN_DECAY_RATIO,
    )
    del subset_genes
    gc.collect()

    # Step 2 (Pass 2): Stream all 22 chr → PCA
    logger.info(f"Pass 2: Full VCF→PCA streaming with K={optimal_k}")
    all_pca_features, sample_ids, pca_stats = stream_vcf_and_pca(
        VCF_PATH, optimal_k=optimal_k, gene_coords=gene_coords,
    )
    gc.collect()

    if not all_pca_features:
        logger.error("No PCA features extracted. Aborting.")
        sys.exit(1)

    analyze_pca_information_loss(pca_stats, optimal_k)

    # Step 3: Labels
    labels = create_hierarchical_labels(PANEL_PATH)

    features_df = pd.DataFrame(all_pca_features)
    del all_pca_features
    gc.collect()

    # Align samples
    if not (sample_ids and len(features_df) == len(labels["pop_labels"])):
        logger.warning(
            f"Sample count mismatch: features={len(features_df)}, "
            f"labels={len(labels['pop_labels'])}. Using minimum overlap."
        )
        n_min = min(len(features_df), len(labels["pop_labels"]))
        features_df = features_df.iloc[:n_min]
        labels["pop_labels"] = labels["pop_labels"][:n_min]
        labels["superpop_labels"] = labels["superpop_labels"][:n_min]
        sample_ids = sample_ids[:n_min] if sample_ids else []

    # Step 4: Tokenize
    tokenized, n_genes = tokenize_dataset(features_df, optimal_k)

    # Step 5: Split (80/10/10)
    x_train, x_val, x_test, y_train, y_val, y_test, _ = split_dataset_stratified(
        tokenized, labels, sample_ids,
        val_ratio=0.1, test_ratio=0.1, seed=PREPROCESS_SEED,
    )
    del tokenized
    gc.collect()

    # Step 6: Normalize
    x_train_norm, x_val_norm, x_test_norm, _ = normalize_data(x_train, x_val, x_test)
    del x_train, x_val, x_test
    gc.collect()

    # Step 7: Zero mask
    gene_size = compute_gene_size(n_genes)
    generate_zero_mask(x_train_norm, gene_size, optimal_k)

    # Step 8: Save
    save_all(
        x_train_norm, x_val_norm, x_test_norm,
        y_train, y_val, y_test, features_df, gene_size,
    )

    elapsed = time.time() - t_start
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Preprocessing complete: {elapsed:.0f}s ({elapsed / 60:.1f}min)")
    logger.info(f"  Genes: {n_genes}, PCA K: {optimal_k}, gene_size: {gene_size}")
    logger.info(f"  Train: {x_train_norm.shape}")
    logger.info(f"  Val:   {x_val_norm.shape}")
    logger.info(f"  Test:  {x_test_norm.shape}")
    logger.info(f"  Output: {PROCESSED_DIR}")
    logger.info(f"{'=' * 60}")

    print(f"\n--- Config values for configs/default.yaml ---")
    print(f"data.num_channels: {optimal_k}")
    print(f"data.gene_size: {gene_size}")
    print(f"---")


if __name__ == "__main__":
    main()
