#!/usr/bin/env python3
"""Full preprocessing pipeline: VCF -> Gene PCA -> tokenized tensors.

OOM-safe 2-pass approach:
  Pass 1: Labels + stratified split → K fixed to PCA_K (no VCF parsing)
  Pass 2: Stream all 22 chr one-by-one → PCA immediately → free variants
Peak RAM ≈ 1 chromosome (~3-5GB for chr1) + accumulated PCA features (~2GB)

Usage:
    python src/preprocessing/run_pipeline.py
"""

from __future__ import annotations

import gc
import json
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
    CONDITIONAL_PRIOR,
    DIM_RED_METHOD,
    GLM_FAMILY,
    PANEL_PATH,
    PCA_K,
    PREPROCESS_SEED,
    PROCESSED_DIR,
    REFGENE_PATH,
    TEST_RATIO,
    VAL_RATIO,
    VCF_PATH,
    VCF_TBI_PATH,
)
from src.preprocessing.gene_annotation import load_refgene
from src.preprocessing.labels import (
    compute_split_indices,
    create_hierarchical_labels,
    pad_to_gene_size,
    save_all,
    split_dataset_stratified,
)
from src.preprocessing.pca import stream_vcf_and_pca
from src.preprocessing.tokenizer import compute_gene_size, generate_zero_mask, normalize_data, tokenize_dataset

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

    if DIM_RED_METHOD != "glm_pca":
        raise ValueError("The production preprocessing pipeline requires glm_pca")
    logger.info("HiPoDiT Preprocessing Pipeline (OOM-safe)")
    logger.info(f"  VCF: {VCF_PATH}")
    logger.info("=" * 60)

    # Step 0: Validate
    validate_input_files()

    # Load gene annotations (RefGene)
    gene_coords = load_refgene(REFGENE_PATH)

    # Step 0.5: Labels + stratified split *before* PCA.
    # The gene PCA bases must be fit on train samples only (otherwise val/test
    # genotypes leak into every gene's loadings). Computing indices up front
    # lets us pass train_indices through the streaming PCA.
    labels = create_hierarchical_labels(PANEL_PATH)
    train_idx, val_idx, test_idx = compute_split_indices(
        labels["pop_labels"],
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        seed=PREPROCESS_SEED,
    )
    logger.info(
        f"Pre-PCA split: train={len(train_idx)}, val={len(val_idx)}, "
        f"test={len(test_idx)} (seed={PREPROCESS_SEED})"
    )

    optimal_k = PCA_K
    logger.info(
        f"Dimensionality reduction backend: {DIM_RED_METHOD} "
        f"(family {GLM_FAMILY}), K={optimal_k}"
    )

    # Step 2 (Pass 2): Stream all 22 chr → PCA (train-only fit, full transform)
    logger.info(
        f"Pass 2: Full VCF→PCA streaming with K={optimal_k} "
        f"(fit on {len(train_idx)} train rows, transform all)"
    )
    all_pca_features, sample_ids, pca_stats = stream_vcf_and_pca(
        VCF_PATH,
        optimal_k=optimal_k,
        gene_coords=gene_coords,
        train_indices=train_idx,
    )
    gc.collect()

    panel_sample_ids = pd.read_csv(PANEL_PATH, sep="\t")["sample"].astype(str).tolist()
    if sample_ids != panel_sample_ids:
        raise ValueError("VCF sample order does not exactly match panel sample order")

    if not all_pca_features:
        logger.error("No PCA features extracted. Aborting.")
        sys.exit(1)

    features_df = pd.DataFrame(all_pca_features)
    del all_pca_features
    gc.collect()

    if not (sample_ids and len(features_df) == len(labels["pop_labels"])):
        raise ValueError(
            f"Sample count mismatch: features={len(features_df)}, "
            f"labels={len(labels['pop_labels'])}"
        )

    # Step 4: Tokenize
    duplicated = pca_stats["gene"][pca_stats["gene"].duplicated()].tolist()
    if duplicated:
        # A repeated name would silently overwrite another locus's features.
        raise ValueError(f"Duplicate gene names across loci: {duplicated[:10]}")
    gene_rows = pca_stats.sort_values(["chrom", "start", "end", "gene"])
    gene_order = gene_rows["gene"].tolist()
    tokenized, n_genes = tokenize_dataset(features_df, optimal_k, gene_order=gene_order)

    # Step 5: Split (80/10/10) — reuse the indices the PCA was fit on.
    x_train, x_val, x_test, y_train, y_val, y_test, _ = split_dataset_stratified(
        tokenized, labels, sample_ids,
        val_ratio=VAL_RATIO, test_ratio=TEST_RATIO, seed=PREPROCESS_SEED,
        precomputed_indices=(train_idx, val_idx, test_idx),
    )
    del tokenized
    gc.collect()

    # Step 6: Pad to aligned gene_size BEFORE normalization
    #   so that normalization stats have the same shape as model I/O
    gene_size = compute_gene_size(n_genes)
    x_train = pad_to_gene_size(x_train, gene_size)
    x_val = pad_to_gene_size(x_val, gene_size)
    x_test = pad_to_gene_size(x_test, gene_size)

    generate_zero_mask(x_train, gene_size, optimal_k)
    x_train_norm, x_val_norm, x_test_norm, stats = normalize_data(
        x_train, x_val, x_test,
        labels=(y_train, y_val, y_test),
        conditional=CONDITIONAL_PRIOR,
    )
    del x_train, x_val, x_test
    gc.collect()

    # Step 9: Save (already padded, save_all will skip re-padding)
    save_all(
        x_train_norm, x_val_norm, x_test_norm,
        y_train, y_val, y_test, features_df, gene_size,
    )
    metadata = {
        "version": 1,
        "dim_reduction_method": DIM_RED_METHOD,
        "glm_family": GLM_FAMILY,
        "glm_projection": "fixed_decoder_likelihood",
        "glm_decoder_path": "glm_pca_decoders.pkl",
        "normalization": {
            "fit_split": stats["fit_split"],
            "shape": list(stats["shape"]),
            "clip": stats["clip"],
            "conditional_prior": stats["conditional"],
        },
        "tensor_layout": "N,G,K",
        "gene_order": gene_rows[["gene", "chrom", "start", "end"]].to_dict("records"),
        "source_sample_order": "VCF_exactly_matches_panel",
    }
    metadata_path = Path(PROCESSED_DIR) / "preprocessing_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    elapsed = time.time() - t_start
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Preprocessing complete: {elapsed:.0f}s ({elapsed / 60:.1f}min)")
    logger.info(f"  Genes: {n_genes}, PCA K: {optimal_k}, gene_size: {gene_size}")
    logger.info(f"  Conditional prior: {stats['conditional']}")
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
