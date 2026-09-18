"""Decode GLM-PCA factors back to genotype dosage, and score them against real data.

Preprocessing fits, per gene, ``dosage ≈ exp(intercept + factors @ loadings.T)``
(Poisson GLM-PCA, see :mod:`src.preprocessing.glm_pca`) and stores the fitted
``loadings``/``intercept`` in ``glm_pca_decoders.pkl``. Generation produces
factors; this module runs that decoder forward to get per-variant dosage, which
is what allele-frequency and genotype-level comparisons need.

Real dosage is not stored by preprocessing, so the reference side re-parses one
chromosome from the VCF with the same gene list, MAF threshold and train rows
that preprocessing used.

    python -m src.inference.decode --syn-dir outputs/<run>/synthetic_samples --chrom 22
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from src.evaluation._io import load_synthetic
from src.preprocessing.config import (
    MAF_THRESHOLD,
    MAX_VARIANTS_PER_GENE,
    VCF_PATH,
)
from src.preprocessing.vcf_parser import process_one_chromosome, resolve_vcf_path


def decode_gene(
    factors: np.ndarray,
    loadings: np.ndarray,
    intercept: np.ndarray,
    family: str = "poi",
) -> np.ndarray:
    """Factors ``(N, K)`` → expected dosage ``(N, J)`` in the {0,1,2} range.

    The mean follows the family the representation was fit under. Poisson uses
    a log link and needs the clip, because ``exp`` runs past the two-copy
    maximum; Binomial(2, p) uses a logit link and is bounded by construction.
    """
    eta = np.asarray(intercept, dtype=np.float64) + np.asarray(
        factors, dtype=np.float64
    ) @ np.asarray(loadings, dtype=np.float64).T
    if family == "binom2":
        return 2.0 / (1.0 + np.exp(-np.clip(eta, -700.0, 700.0)))
    if family != "poi":
        raise ValueError(f"Unknown GLM-PCA family {family!r}")
    return np.clip(np.exp(eta), 0.0, 2.0)


def sample_genotypes(dosage: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Expected dosage → hard calls, one Binomial(2, dosage/2) draw per variant."""
    return rng.binomial(2, np.clip(dosage / 2.0, 0.0, 1.0)).astype(np.int8)


def _gene_rows_for_chrom(metadata_path: Path, chrom: int) -> list[tuple[int, dict]]:
    """(index in the tokenized gene axis, gene record) for one chromosome."""
    gene_order = json.loads(metadata_path.read_text(encoding="utf-8"))["gene_order"]
    return [(i, g) for i, g in enumerate(gene_order) if int(g["chrom"]) == chrom]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--syn-dir", type=Path, required=True)
    parser.add_argument("--chrom", type=int, default=22)
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=20260327)
    parser.add_argument(
        "--save-genotypes",
        action="store_true",
        help="Also write the sampled synthetic {0,1,2} calls and the real dosage "
             "for the chromosome (two int8 npz files, gene name per array).",
    )
    args = parser.parse_args()

    processed = args.processed_dir
    out_dir = args.out_dir or args.syn_dir.parent / "genotype_metrics"
    out_dir.mkdir(parents=True, exist_ok=True)

    with (processed / "glm_pca_decoders.pkl").open("rb") as handle:
        decoders = pickle.load(handle)
    train_indices = np.asarray(
        json.loads((processed / "split_manifest.json").read_text())["train_indices"]
    )
    gene_rows = _gene_rows_for_chrom(processed / "preprocessing_metadata.json", args.chrom)
    if not gene_rows:
        raise SystemExit(f"No genes for chr{args.chrom} in preprocessing_metadata.json")

    syn, _, _ = load_synthetic(
        args.syn_dir, stats_path=processed / "normalization_stats.pkl"
    )  # (N, gene_size, K), original feature scale

    _, real_matrices, _ = process_one_chromosome(
        args.chrom,
        resolve_vcf_path(args.chrom, VCF_PATH),
        MAF_THRESHOLD,
        MAX_VARIANTS_PER_GENE,
        [{"name": g["gene"], "start": g["start"], "end": g["end"]} for _, g in gene_rows],
        train_indices,
    )

    rng = np.random.default_rng(args.seed)
    rows: list[str] = ["gene,variant_index,af_real,af_syn"]
    af_real_all: list[np.ndarray] = []
    af_syn_all: list[np.ndarray] = []
    calls: dict[str, np.ndarray] = {}
    real_calls: dict[str, np.ndarray] = {}
    skipped = 0

    for gene_index, gene in gene_rows:
        name = gene["gene"]
        decoder = decoders.get(name)
        real = real_matrices.get(name)
        if decoder is None or real is None:
            skipped += 1
            continue
        loadings = np.asarray(decoder["loadings"])
        if loadings.shape[0] != real.shape[1]:
            # The VCF no longer yields the variant set the decoder was fit on.
            skipped += 1
            continue

        factors = syn[:, gene_index, : loadings.shape[1]]
        dosage = decode_gene(
            factors, loadings, decoder["intercept"], decoder.get("family", "poi")
        )
        af_syn = dosage.mean(axis=0) / 2.0
        af_real = np.asarray(real, dtype=np.float64).mean(axis=0) / 2.0

        af_real_all.append(af_real)
        af_syn_all.append(af_syn)
        rows += [f"{name},{j},{r:.6f},{s:.6f}" for j, (r, s) in enumerate(zip(af_real, af_syn))]
        if args.save_genotypes:
            calls[name] = sample_genotypes(dosage, rng)
            # The real side is re-parsed from the VCF here and nowhere else, so
            # save it too: no genotype-level real-vs-synthetic comparison
            # (AATS, privacy) is possible without it.
            real_calls[name] = np.asarray(real, dtype=np.int8)

    af_real_cat = np.concatenate(af_real_all)
    af_syn_cat = np.concatenate(af_syn_all)
    summary = {
        "chrom": args.chrom,
        "n_genes": len(af_real_all),
        "n_genes_skipped": skipped,
        "n_variants": int(af_real_cat.size),
        "n_synthetic": int(syn.shape[0]),
        "af_pearson_r": float(np.corrcoef(af_real_cat, af_syn_cat)[0, 1]),
        "af_mean_abs_error": float(np.abs(af_real_cat - af_syn_cat).mean()),
        "af_real_mean": float(af_real_cat.mean()),
        "af_syn_mean": float(af_syn_cat.mean()),
        "syn_dir": str(args.syn_dir),
    }
    (out_dir / f"allele_frequency_chr{args.chrom}.csv").write_text(
        "\n".join(rows) + "\n", encoding="utf-8"
    )
    (out_dir / f"genotype_summary_chr{args.chrom}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    if calls:
        np.savez_compressed(out_dir / f"synthetic_calls_chr{args.chrom}.npz", **calls)
        np.savez_compressed(out_dir / f"real_calls_chr{args.chrom}.npz", **real_calls)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
