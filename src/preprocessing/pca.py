"""Per-gene PCA."""

from __future__ import annotations

import gc
import logging
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from src.preprocessing.config import (
    CHROMOSOMES,
    MAF_THRESHOLD,
    MAX_VARIANTS_PER_GENE,
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
    all_glm_decoders = {}
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

        args = (
            chrom_num,
            vcf_path,
            MAF_THRESHOLD,
            MAX_VARIANTS_PER_GENE,
            chrom_genes,
            train_indices,
        )
        _, gene_matrices, sids = process_one_chromosome(args)

        if not sample_ids and sids:
            sample_ids = sids

        n_genes_chr = len(gene_matrices)
        gene_loci = {gene["name"]: gene for gene in chrom_genes}

        for gene_name in sorted(gene_matrices.keys()):
            from src.preprocessing.config import (
                DIM_RED_METHOD, GLM_PCA_FAMILY, GLM_PCA_MAX_ITER,
            )
            from src.preprocessing.dim_reduction import reduce_single_gene
            result = reduce_single_gene(
                method=DIM_RED_METHOD,
                gene_name=gene_name,
                matrix=gene_matrices[gene_name],
                n_components=optimal_k,
                train_indices=train_indices,
                fam=GLM_PCA_FAMILY,
                max_iter=GLM_PCA_MAX_ITER,
            )
            if result is not None:
                all_pca_features.update(result["features"])
                if "loadings" in result:
                    all_glm_decoders[gene_name] = {
                        key: result[key]
                        for key in (
                            "loadings",
                            "intercept",
                            "family",
                            "link",
                            "penalty",
                            "backend",
                            "backend_version",
                            "projection",
                        )
                    }
                stat_row = {
                    "gene": gene_name,
                    "chrom": chrom_num,
                    "start": gene_loci[gene_name]["start"],
                    "end": gene_loci[gene_name]["end"],
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
    if all_glm_decoders:
        with open(os.path.join(PROCESSED_DIR, "glm_pca_decoders.pkl"), "wb") as handle:
            pickle.dump(all_glm_decoders, handle, protocol=4)

    return all_pca_features, sample_ids, pca_stats_df
