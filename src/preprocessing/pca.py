"""Per-gene dimensionality reduction over streamed VCF chromosomes.

`stream_vcf_and_pca` (the module's main export) runs Poisson GLM-PCA on every
gene of one chromosome at a time.
"""

from __future__ import annotations

import gc
import logging
import multiprocessing as mp
import os
import pickle

import numpy as np
import pandas as pd

from src.preprocessing.config import (
    CHROMOSOMES,
    GLM_PCA_MAX_ITER,
    MAF_THRESHOLD,
    MAX_VARIANTS_PER_GENE,
    PROCESSED_DIR,
)
from src.preprocessing.glm_pca import glm_pca_single_gene
from src.preprocessing.vcf_parser import process_one_chromosome, resolve_vcf_path

logger = logging.getLogger(__name__)


BLAS_THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")


def _pin_blas_threads() -> None:
    """Hold every BLAS backend to one thread per process.

    This must run in the PARENT before the pool is created. The workers use the
    ``spawn`` context, so a child imports this module — and with it numpy and
    OpenBLAS — before any pool initializer runs, and OpenBLAS reads its thread
    count at load time. Pinning from an initializer is therefore too late: it
    left 32 workers each opening 32 threads (observed load average 464 on a
    32-core box). The Poisson path never showed this because its Rust backend
    does not use BLAS; the Binomial path is numpy and scipy, so it does.
    """
    for variable in BLAS_THREAD_VARS:
        os.environ.setdefault(variable, "1")


def _reduce_one(
    task: tuple[str, np.ndarray, int, np.ndarray | None, int],
) -> tuple[str, dict | None]:
    """Picklable top-level worker: reduce one gene's variant matrix."""
    gene_name, matrix, n_components, train_indices, max_iter = task
    result = glm_pca_single_gene(
        gene_name=gene_name,
        matrix=matrix,
        n_components=n_components,
        train_indices=train_indices,
        max_iter=max_iter,
    )
    return gene_name, result


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

    # ponytail: one Pool for the whole run (not per-chromosome) to amortize
    # spin-up cost; ceiling is per-task IPC pickling of each gene's matrix,
    # fine at current gene_size but would need a shared-memory array if genes
    # grow much larger. Uses the "spawn" start method, not the Linux default
    # "fork": forking a process that already has BLAS/OpenMP background
    # threads (numpy/scipy are imported well before this point) can deadlock
    # every child on a futex forever (verified locally) because fork() only
    # copies the calling thread, leaving other threads' held locks stuck.
    ctx = mp.get_context("spawn")
    _pin_blas_threads()
    with ctx.Pool(os.cpu_count(), initializer=_pin_blas_threads) as pool:
        for i, chrom_num in enumerate(chroms):
            chrom_genes = gene_coords.get(str(chrom_num), [])
            if not chrom_genes:
                logger.warning(f"[chr{chrom_num}] No gene annotations found, skipping")
                continue

            _, gene_matrices, sids = process_one_chromosome(
                chrom_num,
                resolve_vcf_path(chrom_num, vcf_path),
                MAF_THRESHOLD,
                MAX_VARIANTS_PER_GENE,
                chrom_genes,
                train_indices,
            )

            # Fail loudly: a parser that silently returned nothing for chr2-22
            # once produced a "complete" chr1-only dataset.
            if not gene_matrices:
                raise RuntimeError(
                    f"chr{chrom_num}: parser returned 0 genes for "
                    f"{len(chrom_genes)} annotated loci"
                )
            if not sample_ids:
                sample_ids = sids
            elif sids != sample_ids:
                raise RuntimeError(f"chr{chrom_num}: sample order differs from earlier chromosomes")

            n_genes_chr = len(gene_matrices)
            gene_loci = {gene["name"]: gene for gene in chrom_genes}

            sorted_gene_names = sorted(gene_matrices.keys())
            tasks = [
                (
                    gene_name,
                    gene_matrices[gene_name],
                    optimal_k,
                    train_indices,
                    GLM_PCA_MAX_ITER,
                )
                for gene_name in sorted_gene_names
            ]

            for gene_name, result in pool.imap(_reduce_one, tasks):
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

            del gene_matrices, tasks  # tasks also references every matrix
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
