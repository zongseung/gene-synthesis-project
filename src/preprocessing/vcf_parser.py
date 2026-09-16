"""VCF parsing with gene boundary grouping.

Thin dispatch layer over the Rust extension ``vcf_parser_rs``: it filters
biallelic SNPs by MAF, maps variants to gene boundaries and returns one dosage
matrix per gene. The extension is required — a second Python implementation of
the same rules drifted from this one once and silently produced a chr1-only
dataset, so the import failure is loud instead.

Install with ``uv pip install -e ./vcf_parser_rs`` (needs the Rust toolchain).
"""

from __future__ import annotations

import logging
import os

import numpy as np

from src.preprocessing.config import (
    PER_CHROM_VCF_DIR,
    PER_CHROM_VCF_PATTERN,
)

logger = logging.getLogger(__name__)

try:
    from vcf_parser_rs import process_one_chromosome_rs as _process_rust
except ImportError as error:  # pragma: no cover - environment problem, not logic
    raise ImportError(
        "vcf_parser_rs is not installed. Build it with: "
        "uv pip install -e ./vcf_parser_rs (requires the Rust toolchain)."
    ) from error


def resolve_vcf_path(chrom_num: int, merged_vcf_path: str) -> str:
    """Per-chromosome VCF when one exists, else the merged VCF (full scan)."""
    path = os.path.join(PER_CHROM_VCF_DIR, PER_CHROM_VCF_PATTERN.format(chrom=chrom_num))
    return path if os.path.exists(path) else merged_vcf_path


def process_one_chromosome(
    chrom_num: int,
    vcf_path: str,
    maf_threshold: float,
    max_variants: int,
    gene_list: list[dict],
    train_indices: np.ndarray | None = None,
) -> tuple[int, dict, list]:
    """Parse one chromosome of ``vcf_path`` into per-gene dosage matrices.

    ``train_indices`` rows are the only ones the MAF filter and the imputation
    mean are computed from. Call :func:`resolve_vcf_path` first to prefer a
    per-chromosome file over the merged VCF.
    """
    return _process_rust((
        chrom_num,
        vcf_path,
        maf_threshold,
        max_variants,
        gene_list,
        None if train_indices is None else list(map(int, train_indices)),
    ))
