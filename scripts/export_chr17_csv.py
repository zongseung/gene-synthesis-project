#!/usr/bin/env python3
"""Export chromosome 17 as raw TSV directly from the 1KG VCF (parallel).

No filtering, no MAF threshold, no imputation, no gene masking — every
variant record on chr17 is streamed out as-is. Genotypes are written in
their original phased form ("0|0", "0|1", "1|0", "1|1", "./.", etc.).

Parallelization: chr17 is split into ``--workers`` position-range chunks
and each worker queries its range via tabix (``vcf("17:start-end")``).
Worker outputs are written as independent gzip members and concatenated
into the final ``.tsv.gz`` in genomic order.

Output: ``data/exports/chr17/chr17_raw.tsv.gz``

Columns::

    chrom, pos, id, ref, alt, qual, filter, info,
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
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

CHROM = "17"
EXPORT_DIR = Path(DATA_DIR) / "exports" / f"chr{CHROM}"

# GRCh37 chr17 length — 1KG Phase 3 uses GRCh37. A small overrun is harmless
# because tabix region queries clip to actual variant positions.
CHR17_LEN = 81_195_210


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
    chunk_size = CHR17_LEN // n_workers + 1
    limit_per_chunk = (
        (args.max_variants // n_workers + 1) if args.max_variants > 0 else 0
    )

    tasks = []
    for i in range(n_workers):
        start = i * chunk_size + 1
        end = min((i + 1) * chunk_size, CHR17_LEN)
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
        f"Done: {total:,} variants × {n_samples:,} samples, "
        f"{size_mb:.1f} MB gzip, {time.time() - t0:.0f}s total"
    )


if __name__ == "__main__":
    main()
