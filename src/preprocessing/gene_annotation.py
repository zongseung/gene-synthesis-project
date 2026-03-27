"""RefGene annotation parser for gene boundary-based variant grouping."""

from __future__ import annotations

import gzip
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


def load_refgene(
    refgene_path: str,
    autosomes_only: bool = True,
) -> dict[str, list[dict]]:
    """Load RefGene annotation and return per-chromosome gene coordinates.

    For genes with multiple transcripts, uses the union (min txStart, max txEnd)
    to capture the full gene body.

    Args:
        refgene_path: Path to refGene.txt.gz
        autosomes_only: If True, only load chr1-22

    Returns:
        gene_coords: {chrom_num_str: [{"name": str, "start": int, "end": int}, ...]}
            Sorted by start position per chromosome.
            chrom_num_str uses VCF convention ("1", "2", ..., "22")
    """
    valid_chroms = {f"chr{i}" for i in range(1, 23)} if autosomes_only else None

    # Merge transcripts per gene: take union of all transcript boundaries
    gene_boundaries: dict[str, dict[str, tuple[int, int]]] = defaultdict(dict)
    # gene_boundaries[chrom][gene_name] = (min_start, max_end)

    opener = gzip.open if refgene_path.endswith(".gz") else open
    with opener(refgene_path, "rt") as f:
        for line in f:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 14:
                continue

            chrom = fields[2]       # e.g. "chr1"
            tx_start = int(fields[4])
            tx_end = int(fields[5])
            gene_name = fields[12]

            if autosomes_only and chrom not in valid_chroms:
                continue

            if gene_name in gene_boundaries[chrom]:
                prev_start, prev_end = gene_boundaries[chrom][gene_name]
                gene_boundaries[chrom][gene_name] = (
                    min(prev_start, tx_start),
                    max(prev_end, tx_end),
                )
            else:
                gene_boundaries[chrom][gene_name] = (tx_start, tx_end)

    # Convert to sorted list per chromosome, using VCF chrom convention
    gene_coords: dict[str, list[dict]] = {}
    total_genes = 0

    for chrom, genes in gene_boundaries.items():
        # "chr1" → "1"
        chrom_num = chrom.replace("chr", "")

        gene_list = [
            {"name": name, "start": start, "end": end}
            for name, (start, end) in genes.items()
        ]
        gene_list.sort(key=lambda g: g["start"])

        gene_coords[chrom_num] = gene_list
        total_genes += len(gene_list)

    logger.info(
        f"RefGene loaded: {total_genes} genes across "
        f"{len(gene_coords)} chromosomes"
    )

    return gene_coords
