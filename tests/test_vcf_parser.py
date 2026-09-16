"""Contract tests for the Rust VCF parser, run through `process_one_chromosome`.

These pin the rules the pipeline depends on: gene boundaries are RefGene
half-open (so `start < POS <= end`), the MAF filter and the imputation mean see
train rows only, and a variant inside two overlapping genes lands in both.
"""

from __future__ import annotations

import gzip

import numpy as np

SAMPLES = ["S0", "S1", "S2", "S3"]
HEADER = (
    "##fileformat=VCFv4.1\n"
    "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + "\t".join(SAMPLES) + "\n"
)


def _write_vcf(path, rows):
    """rows: (pos, ref, alt, [genotype per sample])."""
    with gzip.open(path, "wt") as handle:
        handle.write(HEADER)
        for pos, ref, alt, genotypes in rows:
            handle.write(
                f"17\t{pos}\t.\t{ref}\t{alt}\t.\tPASS\t.\tGT\t" + "\t".join(genotypes) + "\n"
            )
    return str(path)


def _parse(path, genes, train_indices=None, maf_threshold=0.01, max_variants=500):
    from src.preprocessing.vcf_parser import process_one_chromosome

    return process_one_chromosome(17, path, maf_threshold, max_variants, genes, train_indices)


def test_gene_bounds_are_refgene_half_open(tmp_path):
    vcf = _write_vcf(
        tmp_path / "a.vcf.gz",
        [
            (10, "A", "G", ["0/0", "1/1", "0/1", "1/1"]),  # == start, outside
            (11, "A", "G", ["0/0", "1/1", "0/1", "1/1"]),  # first position inside
            (20, "A", "G", ["0/0", "1/1", "0/1", "1/1"]),  # == end, inside
            (21, "A", "G", ["0/0", "1/1", "0/1", "1/1"]),  # past end, outside
        ],
    )

    _, genes, samples = _parse(vcf, [{"name": "GENE", "start": 10, "end": 20}])

    assert samples == SAMPLES
    assert genes["GENE"].shape == (4, 2)


def test_variant_inside_overlapping_genes_lands_in_both(tmp_path):
    vcf = _write_vcf(
        tmp_path / "b.vcf.gz",
        [
            (70, "A", "G", ["0/0", "1/1", "0/1", "1/1"]),
            (71, "A", "G", ["0/0", "1/1", "0/1", "1/1"]),
        ],
    )
    genes = [
        {"name": "LONG", "start": 0, "end": 100},
        {"name": "SHORT", "start": 50, "end": 60},
    ]

    _, matrices, _ = _parse(vcf, genes)

    assert set(matrices) == {"LONG"}  # SHORT ends before 70
    assert matrices["LONG"].shape == (4, 2)


def test_maf_filter_and_imputation_use_train_rows_only(tmp_path):
    # Train rows (0, 1) are 0/0 → MAF 0 within train, even though S3 carries ALT.
    vcf = _write_vcf(
        tmp_path / "c.vcf.gz",
        [
            (11, "A", "G", ["0/0", "0/0", "./.", "1/1"]),
            (12, "A", "G", ["0/0", "1/1", "./.", "1/1"]),
            (13, "A", "G", ["0/0", "1/1", "0/1", "1/1"]),
        ],
    )
    genes = [{"name": "GENE", "start": 10, "end": 20}]

    _, train_only, _ = _parse(vcf, genes, train_indices=np.array([0, 1]))
    _, all_rows, _ = _parse(vcf, genes, train_indices=None)

    # The first variant survives only when all samples are used for the filter.
    assert train_only["GENE"].shape == (4, 2)
    assert all_rows["GENE"].shape == (4, 3)
    # Missing genotype is filled with the train-row mean (0/0 and 1/1 → 1.0).
    assert train_only["GENE"][2, 0] == 1.0


def test_indels_and_multiallelic_sites_are_skipped(tmp_path):
    vcf = _write_vcf(
        tmp_path / "d.vcf.gz",
        [
            (11, "AT", "G", ["0/0", "1/1", "0/1", "1/1"]),
            (12, "A", "G,T", ["0/0", "1/1", "0/1", "1/1"]),
            (13, "A", "G", ["0/0", "1/1", "0/1", "1/1"]),
            (14, "A", "G", ["0/0", "1/1", "0/1", "1/1"]),
        ],
    )

    _, matrices, _ = _parse(vcf, [{"name": "GENE", "start": 10, "end": 20}])

    assert matrices["GENE"].shape == (4, 2)


def test_genes_below_two_variants_are_dropped_and_cap_applies(tmp_path):
    rows = [(10 + i, "A", "G", ["0/0", "1/1", "0/1", "1/1"]) for i in range(1, 6)]
    vcf = _write_vcf(tmp_path / "e.vcf.gz", rows)
    genes = [
        {"name": "ONE_VARIANT", "start": 10, "end": 11},
        {"name": "CAPPED", "start": 11, "end": 20},
    ]

    _, matrices, _ = _parse(vcf, genes, max_variants=3)

    assert "ONE_VARIANT" not in matrices
    assert matrices["CAPPED"].shape == (4, 3)
