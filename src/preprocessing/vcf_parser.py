"""VCF parsing via tabix region queries on merged VCF."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

from .config import MAF_THRESHOLD, MAX_VARIANTS_PER_GENE

logger = logging.getLogger(__name__)


def process_one_chromosome(args: tuple) -> tuple[int, dict, list]:
    """Parse a single chromosome from the merged VCF via tabix region query.

    cyvcf2 returns gt_types as a numpy array in one C call — no Python loop
    over 2,504 samples. ~10x faster than pysam per-sample iteration.

    Args:
        args: (chrom_num, vcf_path, maf_threshold, max_variants)

    Returns:
        (chrom_num, gene_matrices, sample_ids)
    """
    chrom_num, vcf_path, maf_threshold, max_variants = args
    from cyvcf2 import VCF

    t0 = time.time()

    try:
        vcf = VCF(vcf_path)
        sample_ids = list(vcf.samples)
    except Exception as e:
        logger.error(f"[chr{chrom_num}] VCF open failed: {e}")
        return chrom_num, {}, []

    window_size = 100_000
    gene_variants: dict[str, list[np.ndarray]] = {}
    n_variants = 0

    for v in vcf(str(chrom_num)):
        try:
            if len(v.ALT) != 1:
                continue
            if len(v.REF) != 1 or len(v.ALT[0]) != 1:
                continue

            gt = v.gt_types
            dosage = gt.astype(np.float32)
            dosage[gt == 3] = 2.0
            dosage[gt == 2] = np.nan

            if np.all(np.isnan(dosage)):
                continue

            valid = ~np.isnan(dosage)
            n_valid = valid.sum()
            if n_valid == 0:
                continue

            af = np.nanmean(dosage[valid]) / 2.0
            maf = min(af, 1.0 - af)
            if maf < maf_threshold:
                continue

            dosage[np.isnan(dosage)] = np.nanmean(dosage[valid])

            window_start = (v.POS // window_size) * window_size
            gene_name = f"chr{chrom_num}:{window_start}"

            if gene_name not in gene_variants:
                gene_variants[gene_name] = []
            if len(gene_variants[gene_name]) < max_variants:
                gene_variants[gene_name].append(dosage)

            n_variants += 1

        except Exception:
            continue

    vcf.close()

    gene_matrices = {}
    for gene_name, variants in gene_variants.items():
        if len(variants) >= 2:
            gene_matrices[gene_name] = np.stack(variants, axis=1)

    del gene_variants
    elapsed = time.time() - t0
    logger.info(
        f"[chr{chrom_num}] {n_variants:,} variants, "
        f"{len(gene_matrices)} genes ({elapsed:.0f}s)"
    )
    return chrom_num, gene_matrices, sample_ids
