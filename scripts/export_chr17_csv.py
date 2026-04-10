#!/usr/bin/env python3
"""Export chromosome 17 as raw TSV directly from the 1KG VCF (parallel).

No filtering, no MAF threshold, no imputation, no gene masking — every
variant record on chr17 is streamed out as-is. Genotypes are written in
their original phased form ("0|0", "0|1", "1|0", "1|1", "./.", etc.).

Parallelization: chr17 is split into ``--workers`` position-range chunks
and each worker queries its range via tabix (``vcf("17:start-end")``).
Worker outputs are written as independent gzip members and concatenated
into the final ``.tsv.gz`` in genomic order.

Outputs::

    data/exports/chr17/chr17_raw.tsv.gz          — all variants, no annotation
    data/exports/chr17/chr17_gene_matched.tsv.gz  — variants matched to refGene genes

Columns (raw)::

    chrom, pos, id, ref, alt, qual, filter, info,
    <sample_id_1>, <sample_id_2>, ...

Columns (gene_matched)::

    gene_name, chrom, pos, id, ref, alt, qual, filter, info,
    <sample_id_1>, <sample_id_2>, ...

Usage::

    uv run python scripts/export_chr17_csv.py                    # all workers
    uv run python scripts/export_chr17_csv.py --workers 8
    uv run python scripts/export_chr17_csv.py --max-variants 10000
"""

from __future__ import annotations

import argparse
import gzip
import logging
import os
import shutil
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from pathlib import Path

# Allow direct execution from repo root
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.preprocessing.config import (
    DATA_DIR,
    PER_CHROM_VCF_DIR,
    PER_CHROM_VCF_PATTERN,
    REFGENE_PATH,
)
from src.preprocessing.gene_annotation import load_refgene

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

CHROM = "17"
EXPORT_DIR = Path(DATA_DIR) / "exports" / f"chr{CHROM}"

# GRCh37 chromosome lengths — 1KG Phase 3 uses GRCh37.
# A small overrun is harmless because tabix region queries clip to actual
# variant positions.
CHROM_LENGTHS = {
    "1": 249_250_621, "2": 243_199_373, "3": 198_022_430,
    "4": 191_154_276, "5": 180_915_260, "6": 171_115_067,
    "7": 159_138_663, "8": 146_364_022, "9": 141_213_431,
    "10": 135_534_747, "11": 135_006_516, "12": 133_851_895,
    "13": 115_169_878, "14": 107_349_540, "15": 102_531_392,
    "16": 90_354_753, "17": 81_195_210, "18": 78_077_248,
    "19": 59_128_983, "20": 63_025_520, "21": 48_129_895,
    "22": 51_304_566,
}


def _worker(task: tuple) -> tuple[int, str, int]:
    """Stream a tabix region to a temp gzip TSV (no header).

    Args:
        task: (chunk_idx, vcf_path, start, end, sep, n_samples, limit_per_chunk)

    Returns:
        (chunk_idx, temp_path, n_rows_written)
    """
    chunk_idx, vcf_path, start, end, sep, n_samples, limit = task
    from cyvcf2 import VCF

    vcf = VCF(vcf_path)
    tmp = tempfile.NamedTemporaryFile(
        prefix=f"chr17_chunk{chunk_idx:02d}_",
        suffix=".tsv.gz",
        delete=False,
        dir=tempfile.gettempdir(),
    )
    tmp.close()

    n = 0
    region = f"{CHROM}:{start}-{end}"
    with gzip.open(tmp.name, "wt", compresslevel=6) as fh:
        for v in vcf(region):
            raw = str(v).rstrip("\n").split("\t")
            if len(raw) < 9 + n_samples:
                continue
            fixed = raw[:8]
            per_sample = raw[9 : 9 + n_samples]
            gts = [s.split(":", 1)[0] for s in per_sample]
            fh.write(sep.join(fixed + gts) + "\n")
            n += 1
            if limit and n >= limit:
                break
    vcf.close()
    return chunk_idx, tmp.name, n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workers",
        type=int,
        default=min(cpu_count(), 16),
        help=f"Parallel workers (default: min(cpu, 16) = {min(cpu_count(), 16)})",
    )
    parser.add_argument(
        "--max-variants",
        type=int,
        default=0,
        help="Global cap on total variants across all chunks (0 = all).",
    )
    parser.add_argument(
        "--sep",
        default="\t",
        help="Field separator (default: TAB).",
    )
    args = parser.parse_args()

    from cyvcf2 import VCF

    vcf_path = Path(PER_CHROM_VCF_DIR) / PER_CHROM_VCF_PATTERN.format(chrom=17)
    if not vcf_path.exists():
        raise FileNotFoundError(f"chr17 VCF not found: {vcf_path}")

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = EXPORT_DIR / f"chr{CHROM}_raw.tsv.gz"
    logger.info(f"Reading:  {vcf_path}")
    logger.info(f"Writing:  {out}")

    vcf = VCF(str(vcf_path))
    sample_ids = list(vcf.samples)
    n_samples = len(sample_ids)
    vcf.close()
    logger.info(f"Samples:  {n_samples}")

    # --- Plan chunks ------------------------------------------------------
    n_workers = max(1, args.workers)
    chrom_len = CHROM_LENGTHS[CHROM]
    chunk_size = chrom_len // n_workers + 1
    limit_per_chunk = (
        (args.max_variants // n_workers + 1) if args.max_variants > 0 else 0
    )

    tasks = []
    for i in range(n_workers):
        start = i * chunk_size + 1
        end = min((i + 1) * chunk_size, chrom_len)
        if start > end:
            continue
        tasks.append(
            (i, str(vcf_path), start, end, args.sep, n_samples, limit_per_chunk)
        )
    logger.info(
        f"Parallel plan: {len(tasks)} chunks × ~{chunk_size:,} bp each, "
        f"{n_workers} workers"
    )

    # --- Run workers ------------------------------------------------------
    t0 = time.time()
    results: dict[int, tuple[str, int]] = {}
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(_worker, t): t[0] for t in tasks}
        for fut in as_completed(futs):
            idx, path, n = fut.result()
            results[idx] = (path, n)
            logger.info(
                f"  chunk {idx:02d} done: {n:,} variants "
                f"({time.time() - t0:.0f}s elapsed)"
            )

    total = sum(n for _, n in results.values())
    logger.info(f"All chunks done: {total:,} total variants in {time.time() - t0:.0f}s")

    # --- Concatenate ------------------------------------------------------
    header = (
        ["chrom", "pos", "id", "ref", "alt", "qual", "filter", "info"] + sample_ids
    )
    logger.info("Concatenating chunks (gzip multi-member)...")
    with gzip.open(out, "wt", compresslevel=6) as fh:
        fh.write(args.sep.join(header) + "\n")

    # Append each chunk as raw bytes — gzip supports concatenated members.
    with open(out, "ab") as fout:
        for idx in sorted(results.keys()):
            path, _ = results[idx]
            with open(path, "rb") as fin:
                shutil.copyfileobj(fin, fout, length=1 << 20)
            os.unlink(path)

    size_mb = out.stat().st_size / 1e6
    logger.info(
        f"Raw export done: {total:,} variants × {n_samples:,} samples, "
        f"{size_mb:.1f} MB gzip, {time.time() - t0:.0f}s total"
    )

    # --- Gene-matched export -----------------------------------------------
    logger.info("Loading refGene annotations for chr17...")
    gene_coords = load_refgene(REFGENE_PATH)
    chr17_genes = gene_coords.get(CHROM, [])
    logger.info(f"chr17 genes: {len(chr17_genes)}")

    # Build interval list sorted by start for binary search
    import bisect

    gene_starts = [g["start"] for g in chr17_genes]
    gene_ends = [g["end"] for g in chr17_genes]

    out_matched = EXPORT_DIR / f"chr{CHROM}_gene_matched.tsv.gz"
    logger.info(f"Writing gene-matched file: {out_matched}")

    t1 = time.time()
    matched_count = 0

    with gzip.open(out, "rt") as fin, gzip.open(out_matched, "wt", compresslevel=6) as fout:
        # Read and write header
        raw_header = fin.readline().rstrip("\n")
        fout.write(args.sep.join(["gene_name"] + raw_header.split(args.sep)) + "\n")

        for line in fin:
            fields = line.rstrip("\n").split(args.sep, 2)  # split only first 2 for pos
            pos = int(fields[1])

            # Find genes that overlap this position
            # A gene overlaps if gene.start <= pos <= gene.end
            right = bisect.bisect_right(gene_starts, pos)
            for i in range(right - 1, -1, -1):
                if gene_ends[i] < pos:
                    break
                if chr17_genes[i]["start"] <= pos <= chr17_genes[i]["end"]:
                    fout.write(chr17_genes[i]["name"] + args.sep + line)
                    matched_count += 1

    matched_mb = out_matched.stat().st_size / 1e6
    logger.info(
        f"Gene-matched done: {matched_count:,} variant-gene pairs, "
        f"{matched_mb:.1f} MB gzip, {time.time() - t1:.0f}s"
    )


if __name__ == "__main__":
    main()
