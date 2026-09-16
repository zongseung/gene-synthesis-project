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

    Transcripts sharing a gene name are merged only where they overlap. Names
    such as RNU1-3 or MIR6859-1 are annotated at several distant loci; a plain
    min(txStart)/max(txEnd) union turned those into fake genes spanning up to
    132 Mb. Each non-overlapping cluster is kept as its own locus, and a name
    with more than one locus genome-wide is suffixed ``@chr{N}-{start}`` so
    feature keys stay unique across chromosomes.

    Args:
        refgene_path: Path to refGene.txt.gz
        autosomes_only: If True, only load chr1-22

    Returns:
        gene_coords: {chrom_num_str: [{"name": str, "start": int, "end": int}, ...]}
            Sorted by start position per chromosome.
            chrom_num_str uses VCF convention ("1", "2", ..., "22")
    """
    valid_chroms = {f"chr{i}" for i in range(1, 23)} if autosomes_only else None

    # transcripts[(chrom, gene_name)] = [(tx_start, tx_end), ...]
    transcripts: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)

    opener = gzip.open if refgene_path.endswith(".gz") else open
    with opener(refgene_path, "rt") as f:
        for line in f:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 14:
                continue

            chrom = fields[2]       # e.g. "chr1"
            if autosomes_only and chrom not in valid_chroms:
                continue
            transcripts[(chrom, fields[12])].append((int(fields[4]), int(fields[5])))

    # loci[gene_name] = [(chrom, start, end), ...], one entry per overlap cluster
    loci: dict[str, list[tuple[str, int, int]]] = defaultdict(list)
    for (chrom, gene_name), intervals in transcripts.items():
        intervals.sort()
        start, end = intervals[0]
        for tx_start, tx_end in intervals[1:]:
            if tx_start > end:
                loci[gene_name].append((chrom, start, end))
                start, end = tx_start, tx_end
            else:
                end = max(end, tx_end)
        loci[gene_name].append((chrom, start, end))

    gene_coords: dict[str, list[dict]] = defaultdict(list)
    for gene_name, clusters in loci.items():
        for chrom, start, end in clusters:
            chrom_num = chrom.replace("chr", "")  # "chr1" -> "1"
            name = gene_name if len(clusters) == 1 else f"{gene_name}@chr{chrom_num}-{start}"
            gene_coords[chrom_num].append({"name": name, "start": start, "end": end})

    for gene_list in gene_coords.values():
        gene_list.sort(key=lambda g: g["start"])

    n_split = sum(1 for clusters in loci.values() if len(clusters) > 1)
    logger.info(
        f"RefGene loaded: {sum(map(len, gene_coords.values()))} loci across "
        f"{len(gene_coords)} chromosomes ({n_split} names split into multiple loci)"
    )
    return dict(gene_coords)
