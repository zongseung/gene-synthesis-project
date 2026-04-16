"""PCA grid search, per-gene PCA, and information loss analysis."""

from __future__ import annotations

import gc
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from src.preprocessing.config import (
    CHROMOSOMES,
    MAF_THRESHOLD,
    MAX_VARIANTS_PER_GENE,
    MARGINAL_GAIN_DECAY_RATIO,
    MARGINAL_GAIN_THRESHOLD,
    PCA_CANDIDATES,
    PCA_SAMPLE_GENES,
    PROCESSED_DIR,
)
from src.preprocessing.vcf_parser import process_one_chromosome

logger = logging.getLogger(__name__)


def pca_single_gene(
    gene_name: str,
    matrix: np.ndarray,
    n_components: int,
    train_indices: np.ndarray | None = None,
) -> dict | None:
    """Apply PCA to a single gene's variant matrix.

    If `train_indices` is provided, the PCA basis is fit on train rows only
    and then used to transform the full matrix (preventing val/test leakage
    into per-gene loadings). Otherwise falls back to fit_transform on the
    whole matrix (legacy behavior; leaks val/test).
    """
    n_vars = matrix.shape[1]
    n_fit_samples = (
        matrix.shape[0] if train_indices is None else int(len(train_indices))
    )
    n_comp = min(n_components, n_vars, n_fit_samples)

    if n_comp < 2:
        return None

    try:
        pca = PCA(n_components=n_comp)
        if train_indices is None:
            transformed = pca.fit_transform(matrix)
        else:
            pca.fit(matrix[train_indices])
            transformed = pca.transform(matrix)
        explained = float(np.sum(pca.explained_variance_ratio_))
        per_component = pca.explained_variance_ratio_.tolist()

        features = {}
        for k in range(n_comp):
            features[f"{gene_name}:{k}"] = transformed[:, k]

        return {
            "features": features,
            "explained_total": explained,
            "explained_per_component": per_component,
            "n_variants": n_vars,
            "actual_k": n_comp,
        }
    except Exception as e:
        logger.warning(f"PCA failed for {gene_name}: {e}")
        return None


def _evaluate_pca_k_for_gene(
    gene_name: str,
    matrix: np.ndarray,
    k: int,
    train_indices: np.ndarray | None = None,
) -> tuple[str, int, float, int]:
    """Evaluate a single (gene, K) combination for PCA explained variance.

    Uses train rows only when `train_indices` is provided, so K selection is
    not influenced by held-out val/test samples.
    """
    n_vars = matrix.shape[1]
    n_samples = (
        matrix.shape[0] if train_indices is None else int(len(train_indices))
    )
    actual_k = min(k, n_vars, n_samples)

    if actual_k < 2:
        return (gene_name, k, 0.0, 0)

    try:
        pca = PCA(n_components=actual_k)
        fit_matrix = matrix if train_indices is None else matrix[train_indices]
        pca.fit(fit_matrix)
        explained = float(np.sum(pca.explained_variance_ratio_))
        return (gene_name, k, explained, actual_k)
    except Exception as e:
        logger.warning(f"PCA eval failed for {gene_name} (k={k}): {e}")
        return (gene_name, k, 0.0, 0)


def grid_search_optimal_pca(
    gene_matrices: dict[str, np.ndarray],
    candidates: list[int] | None = None,
    marginal_threshold: float = MARGINAL_GAIN_THRESHOLD,
    decay_ratio: float = MARGINAL_GAIN_DECAY_RATIO,
    n_sample_genes: int = PCA_SAMPLE_GENES,
    train_indices: np.ndarray | None = None,
) -> tuple[int, pd.DataFrame]:
    """Grid search for optimal PCA component count K.

    Strategy: Marginal Gain Elbow detection.
    - Condition 1: marginal_gain < threshold -> select previous K
    - Condition 2: gain < previous_gain * decay_ratio -> select previous K

    `train_indices`, when provided, restricts each K evaluation to train rows
    so the chosen K is not biased by val/test signal.
    """
    if candidates is None:
        candidates = PCA_CANDIDATES

    all_genes = list(gene_matrices.keys())

    if len(all_genes) > n_sample_genes:
        np.random.seed(42)
        sample_genes = np.random.choice(all_genes, n_sample_genes, replace=False)
    else:
        sample_genes = all_genes

    logger.info(
        f"PCA grid search: K candidates={candidates}, "
        f"sample genes={len(sample_genes)}/{len(all_genes)}"
    )

    tasks = []
    for gene_name in sample_genes:
        for k in candidates:
            tasks.append((gene_name, gene_matrices[gene_name], k, train_indices))

    logger.info(f"Total tasks: {len(tasks)} ({len(sample_genes)} genes x {len(candidates)} K values)")

    results = []
    with ThreadPoolExecutor(max_workers=min(32, len(candidates))) as executor:
        futures = [executor.submit(_evaluate_pca_k_for_gene, *task) for task in tasks]
        for future in futures:
            results.append(future.result())

    results_df = pd.DataFrame(
        results, columns=["gene", "k", "explained_ratio", "actual_k"]
    )

    k_summary = {}
    for k in candidates:
        k_data = results_df[results_df["k"] == k]
        valid = k_data[k_data["actual_k"] > 0]
        if len(valid) == 0:
            continue

        mean_explained = valid["explained_ratio"].mean()
        median_explained = valid["explained_ratio"].median()
        p10_explained = valid["explained_ratio"].quantile(0.10)
        p25_explained = valid["explained_ratio"].quantile(0.25)

        k_summary[k] = {
            "mean_explained": float(mean_explained),
            "median_explained": float(median_explained),
            "p10_explained": float(p10_explained),
            "p25_explained": float(p25_explained),
            "n_valid_genes": int(len(valid)),
        }
        logger.info(
            f"  K={k:2d}: mean={mean_explained:.4f}, "
            f"median={median_explained:.4f}, p10={p10_explained:.4f}"
        )

    sorted_k = sorted(k for k in candidates if k in k_summary)
    gains = {}
    for i in range(1, len(sorted_k)):
        prev_k, curr_k = sorted_k[i - 1], sorted_k[i]
        gain = k_summary[curr_k]["mean_explained"] - k_summary[prev_k]["mean_explained"]
        gains[curr_k] = gain
        logger.info(f"  K={prev_k}->{curr_k}: marginal gain = {gain:.4f}")

    optimal_k = sorted_k[0]
    prev_gain = None
    for i in range(1, len(sorted_k)):
        curr_k = sorted_k[i]
        gain = gains[curr_k]

        if gain < marginal_threshold:
            optimal_k = sorted_k[i - 1]
            logger.info(f"  Elbow (gain {gain:.4f} < threshold {marginal_threshold}): K={optimal_k}")
            break

        if prev_gain is not None and gain < prev_gain * decay_ratio:
            optimal_k = sorted_k[i - 1]
            logger.info(f"  Elbow (gain decay {prev_gain:.4f}->{gain:.4f}): K={optimal_k}")
            break

        prev_gain = gain
        optimal_k = curr_k
    else:
        logger.warning(f"  Marginal gain did not converge. Using max K={optimal_k}")

    logger.info(f"\n{'=' * 60}")
    logger.info(f"Optimal PCA components: K={optimal_k}")
    logger.info(f"  Mean explained variance: {k_summary[optimal_k]['mean_explained']:.4f}")
    logger.info(f"{'=' * 60}\n")

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    summary_df = pd.DataFrame(k_summary).T
    summary_df.index.name = "k"
    summary_df.to_csv(os.path.join(PROCESSED_DIR, "pca_grid_search_results.csv"))
    results_df.to_csv(
        os.path.join(PROCESSED_DIR, "pca_grid_search_detail.csv"), index=False
    )

    return optimal_k, summary_df


def stream_vcf_and_pca(
    vcf_path: str,
    optimal_k: int,
    gene_coords: dict[str, list[dict]],
    chroms: list[int] | None = None,
    train_indices: np.ndarray | None = None,
) -> tuple[dict[str, np.ndarray], list[str], pd.DataFrame]:
    """Stream chromosomes sequentially: parse → PCA → free variants.

    OOM-safe: only one chromosome's variant data is in memory at a time.
    Peak memory ≈ one chromosome's variants (~3-5GB for chr1) + PCA features.

    Args:
        vcf_path: Path to merged VCF with tabix index.
        optimal_k: Number of PCA components.
        gene_coords: Per-chromosome gene boundaries from load_refgene().
        chroms: Chromosomes to process (default: all 22).
        train_indices: Optional sample-row indices to fit PCA on. When set,
            each gene's PCA basis is fit on these rows only; all 2504 samples
            are then transformed into that basis (avoids val/test leakage).
    """
    if chroms is None:
        chroms = CHROMOSOMES

    all_pca_features = {}
    all_pca_stats = []
    sample_ids = []

    logger.info(
        f"Streaming VCF→PCA: sequential, K={optimal_k}, "
        f"{len(chroms)} chromosomes (OOM-safe: 1 chr at a time)"
    )

    for i, chrom_num in enumerate(chroms):
        chrom_genes = gene_coords.get(str(chrom_num), [])
        if not chrom_genes:
            logger.warning(f"[chr{chrom_num}] No gene annotations found, skipping")
            continue

        args = (chrom_num, vcf_path, MAF_THRESHOLD, MAX_VARIANTS_PER_GENE, chrom_genes)
        _, gene_matrices, sids = process_one_chromosome(args)

        if not sample_ids and sids:
            sample_ids = sids

        n_genes_chr = len(gene_matrices)

        for gene_name in sorted(gene_matrices.keys()):
            result = pca_single_gene(
                gene_name,
                gene_matrices[gene_name],
                optimal_k,
                train_indices=train_indices,
            )
            if result is not None:
                all_pca_features.update(result["features"])
                stat_row = {
                    "gene": gene_name,
                    "n_variants": result["n_variants"],
                    "actual_k": result["actual_k"],
                    "explained_total": result["explained_total"],
                }
                for j, v in enumerate(result["explained_per_component"]):
                    stat_row[f"explained_pc{j + 1}"] = v
                all_pca_stats.append(stat_row)

        del gene_matrices
        gc.collect()

        logger.info(
            f"  [{i + 1}/{len(chroms)}] chr{chrom_num}: {n_genes_chr} genes → "
            f"PCA done, {len(all_pca_features)} total features"
        )

    pca_stats_df = pd.DataFrame(all_pca_stats)

    logger.info(
        f"Streaming VCF→PCA complete: {len(pca_stats_df)} genes, "
        f"{len(all_pca_features)} features, {len(sample_ids)} samples"
    )

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    pca_stats_df.to_csv(
        os.path.join(PROCESSED_DIR, "pca_per_gene_stats.csv"), index=False
    )

    return all_pca_features, sample_ids, pca_stats_df


def analyze_pca_information_loss(
    pca_stats_df: pd.DataFrame,
    optimal_k: int,
) -> dict:
    """Generate PCA information loss analysis for the paper."""
    stats = {}

    stats["overall"] = {
        "optimal_k": optimal_k,
        "mean_explained": float(pca_stats_df["explained_total"].mean()),
        "median_explained": float(pca_stats_df["explained_total"].median()),
        "std_explained": float(pca_stats_df["explained_total"].std()),
        "min_explained": float(pca_stats_df["explained_total"].min()),
        "max_explained": float(pca_stats_df["explained_total"].max()),
        "p5_explained": float(pca_stats_df["explained_total"].quantile(0.05)),
        "p10_explained": float(pca_stats_df["explained_total"].quantile(0.10)),
        "n_genes_total": len(pca_stats_df),
        "n_genes_above_90pct": int((pca_stats_df["explained_total"] >= 0.90).sum()),
        "n_genes_above_95pct": int((pca_stats_df["explained_total"] >= 0.95).sum()),
        "n_genes_below_70pct": int((pca_stats_df["explained_total"] < 0.70).sum()),
    }

    bins = [0, 5, 10, 20, 50, 100, 200, 500]
    pca_stats_df = pca_stats_df.copy()
    pca_stats_df["variant_bin"] = pd.cut(pca_stats_df["n_variants"], bins=bins)
    bin_stats = (
        pca_stats_df.groupby("variant_bin", observed=True)["explained_total"]
        .agg(["mean", "median", "count"])
    )
    stats["by_variant_count"] = {
        str(k): {"mean": float(v["mean"]), "median": float(v["median"]), "count": int(v["count"])}
        for k, v in bin_stats.iterrows()
    }

    threshold = pca_stats_df["explained_total"].quantile(0.05)
    worst = pca_stats_df[pca_stats_df["explained_total"] <= threshold].sort_values(
        "explained_total"
    )
    stats["worst_genes"] = worst[["gene", "n_variants", "explained_total"]].to_dict(
        "records"
    )

    analysis_path = os.path.join(PROCESSED_DIR, "pca_information_loss_analysis.json")
    with open(analysis_path, "w") as f:
        json.dump(stats, f, indent=2, default=str)
    logger.info(f"PCA information loss analysis saved to {analysis_path}")

    o = stats["overall"]
    logger.info(
        f'\nPaper summary (Methods):\n'
        f'  "Gene-level PCA with K={o["optimal_k"]} components explained '
        f'{o["mean_explained"]:.1%} (mean) / {o["median_explained"]:.1%} (median) '
        f'of per-gene variance across {o["n_genes_total"]:,} genes."\n'
    )

    return stats
