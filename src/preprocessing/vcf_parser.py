"""VCF parsing with gene boundary grouping.

Uses Rust extension (vcf_parser_rs) when available for ~5-15x speedup.
Falls back to Python/cyvcf2 implementation if Rust module is not installed.
"""

from __future__ import annotations

import bisect
import logging
import time

import numpy as np

import os

from src.preprocessing.config import (
    MAF_THRESHOLD,
    MAX_VARIANTS_PER_GENE,
    PER_CHROM_VCF_DIR,
    PER_CHROM_VCF_PATTERN,
)

logger = logging.getLogger(__name__)

# Attempt Rust import with Python fallback
try:
    from vcf_parser_rs import process_one_chromosome_rs as _process_rust
    _USE_RUST = True
    logger.info("Using Rust VCF parser (vcf_parser_rs)")
except ImportError:
    _USE_RUST = False
    logger.info("Rust VCF parser not available, using Python/cyvcf2 fallback")


def _build_gene_index(gene_list: list[dict]) -> tuple[list[int], list[dict]]:
    """Build a sorted start-position index for bisect lookup.

    Args:
        gene_list: Sorted list of {"name", "start", "end"} dicts.

    Returns:
        starts: Sorted list of gene start positions.
        genes: Corresponding gene dicts (same order as starts).
    """
    starts = [g["start"] for g in gene_list]
    return starts, gene_list


def _find_genes_for_position(
    pos: int,
    starts: list[int],
    genes: list[dict],
) -> list[str]:
    """Find all genes whose [start, end] contains the given position.

    Uses bisect for O(log n) lookup instead of linear scan.
    """
    # Find rightmost gene whose start <= pos
    idx = bisect.bisect_right(starts, pos) - 1
    if idx < 0:
        return []

    matched = []
    # Check backwards from idx (genes are sorted by start)
    for i in range(idx, -1, -1):
        g = genes[i]
        if g["start"] > pos:
            continue
        if g["end"] < pos:
            break
        matched.append(g["name"])

    return matched


def _get_per_chrom_vcf(chrom_num: int) -> str | None:
    """Return per-chromosome VCF path if it exists."""
    path = os.path.join(PER_CHROM_VCF_DIR, PER_CHROM_VCF_PATTERN.format(chrom=chrom_num))
    return path if os.path.exists(path) else None


def process_one_chromosome(args: tuple) -> tuple[int, dict, list]:
    """Dispatch to Rust (per-chromosome VCF) or Python (merged VCF + tabix)."""
    if _USE_RUST:
        chrom_num = args[0]
        per_chrom_path = _get_per_chrom_vcf(chrom_num)
        if per_chrom_path:
            # Use per-chromosome file for Rust (no full-file scan needed)
            rust_args = (args[0], per_chrom_path, args[2], args[3], args[4])
            return _process_rust(rust_args)
        # Fallback: use merged VCF (slower — scans from beginning)
        return _process_rust(args)
    return _process_one_chromosome_python(args)


def _process_one_chromosome_python(args: tuple) -> tuple[int, dict, list]:
    """Parse a single chromosome from the merged VCF via tabix region query.

    Variants are grouped by actual gene boundaries from RefGene annotation.
    Each variant is assigned to all overlapping genes (a variant in an
    overlapping region belongs to both genes).

    cyvcf2 returns gt_types as a numpy array in one C call — no Python loop
    over 2,504 samples.

    Args:
        args: (chrom_num, vcf_path, maf_threshold, max_variants, gene_list)
            gene_list: list of {"name", "start", "end"} for this chromosome,
                       sorted by start. From gene_annotation.load_refgene().

    Returns:
        (chrom_num, gene_matrices, sample_ids)
    """
    chrom_num, vcf_path, maf_threshold, max_variants, gene_list = args
    from cyvcf2 import VCF

    t0 = time.time()

    try:
        vcf = VCF(vcf_path)
        sample_ids = list(vcf.samples)
    except Exception as e:
        logger.error(f"[chr{chrom_num}] VCF open failed: {e}")
        return chrom_num, {}, []

    starts, genes = _build_gene_index(gene_list)
    gene_variants: dict[str, list[np.ndarray]] = {}
    n_variants = 0
    n_intergenic = 0

    for v in vcf(str(chrom_num)):
        try:
            # Biallelic SNP filter
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

            # Mean imputation
            dosage[np.isnan(dosage)] = np.nanmean(dosage[valid])

            # Find which gene(s) this variant belongs to
            matched_genes = _find_genes_for_position(v.POS, starts, genes)

            if not matched_genes:
                n_intergenic += 1
                continue

            for gene_name in matched_genes:
                if gene_name not in gene_variants:
                    gene_variants[gene_name] = []
                if len(gene_variants[gene_name]) < max_variants:
                    gene_variants[gene_name].append(dosage)

            n_variants += 1

        except Exception:
            continue

    vcf.close()

    # Convert to matrices (>= 2 variants for PCA)
    gene_matrices = {}
    for gene_name, variants in gene_variants.items():
        if len(variants) >= 2:
            gene_matrices[gene_name] = np.stack(variants, axis=1)

    del gene_variants
    elapsed = time.time() - t0
    logger.info(
        f"[chr{chrom_num}] {n_variants:,} genic variants, "
        f"{n_intergenic:,} intergenic skipped, "
        f"{len(gene_matrices)} genes with ≥2 variants ({elapsed:.0f}s)"
    )
    return chrom_num, gene_matrices, sample_ids
