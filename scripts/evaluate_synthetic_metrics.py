#!/usr/bin/env python3
"""CLI for synthetic-genotype evaluation.

This is a thin orchestration shim: it reads project-specific data
(``data/processed/test_data.pkl``, ``outputs/<run>/synthetic_samples/``,
``data/processed/label_hierarchy.pkl``), projects to PCA(2), and delegates
the numeric work to :mod:`src.evaluation`.

Numeric definitions live in:

    * :mod:`src.evaluation.dupi` — Jeong, Kim, and Im (2023) DUPI / UI / PI.
    * :mod:`src.evaluation.distribution_metrics` — Gaussian W2, MMD-RBF.
    * :mod:`src.evaluation.synthetic_pipeline` — high-level ``evaluate``.

Run ``python scripts/evaluate_synthetic_metrics.py --help`` for flags.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA


from src.evaluation import evaluate  # noqa: E402
from src.evaluation._io import (  # noqa: E402
    flatten_subsample_genes,
    load_label_hierarchy,
    load_real,
    load_synthetic,
    pop_to_superpop,
    write_csv,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-path", type=Path, default=Path("data/processed/test_data.pkl"))
    parser.add_argument("--syn-dir", type=Path, default=Path("outputs/default/synthetic_samples"))
    parser.add_argument("--hierarchy", type=Path, default=Path("data/processed/label_hierarchy.pkl"))
    parser.add_argument(
        "--stats-path",
        type=Path,
        default=Path("data/processed/normalization_stats.pkl"),
        help="Normalization stats used to restore the shared original feature scale.",
    )
    parser.add_argument(
        "--real-space",
        choices=["normalized", "original"],
        default="normalized",
        help="Space of --real-path tensors; evaluation always uses original scale.",
    )
    parser.add_argument(
        "--legacy-synthetic-space",
        choices=["normalized", "original"],
        default=None,
        help="Required only for legacy generation metadata without sample_space.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/default/evaluation_metrics"))
    parser.add_argument("--n-genes", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dupi-k", type=int, default=1)
    parser.add_argument("--tau", type=float, default=5.0)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    hierarchy = load_label_hierarchy(args.hierarchy)
    real_x, real_pop = load_real(
        args.real_path, stats_path=args.stats_path, sample_space=args.real_space,
    )
    syn_x, syn_pop, syn_names = load_synthetic(
        args.syn_dir,
        stats_path=args.stats_path,
        legacy_sample_space=args.legacy_synthetic_space,
    )
    real_flat, gene_indices = flatten_subsample_genes(real_x, args.n_genes, args.seed)
    syn_flat, _ = flatten_subsample_genes(syn_x, args.n_genes, args.seed, gene_indices)

    pca = PCA(n_components=2, random_state=args.seed)
    real_pcs = pca.fit_transform(real_flat)
    syn_pcs = pca.transform(syn_flat)
    real_sp = pop_to_superpop(real_pop, hierarchy)
    syn_sp = pop_to_superpop(syn_pop, hierarchy)
    np.save(args.out_dir / "pca_gene_indices.npy", gene_indices)

    report = evaluate(
        real_pcs=real_pcs, syn_pcs=syn_pcs, real_sp=real_sp, syn_sp=syn_sp,
        k=args.dupi_k, tau=args.tau,
    )

    summary = {
        "inputs": {
            "real_path": str(args.real_path),
            "syn_dir": str(args.syn_dir),
            "hierarchy": str(args.hierarchy),
            "stats_path": str(args.stats_path),
            "real_space": args.real_space,
            "legacy_synthetic_space": args.legacy_synthetic_space,
            "evaluation_space": "original",
            "n_genes": args.n_genes,
            "seed": args.seed,
            "dupi_k": args.dupi_k,
            "tau": args.tau,
            "metric_space": "PCA(2) fitted on real flattened subsampled genes",
        },
        "counts": {
            "n_real": int(len(real_pop)),
            "n_synthetic": int(len(syn_pop)),
            "n_features_before_pca": int(real_flat.shape[1]),
        },
        "pca": {
            "explained_variance_ratio": [float(v) for v in pca.explained_variance_ratio_],
            "components_shape": list(pca.components_.shape),
        },
        "dupi": report.dupi,
        "distribution_distances": report.distribution_distances,
        "notes": {
            "dupi_source": "Jeong, Kim, and Im (2023), Eqs. (10)-(13).",
            "dupi_interpretation": (
                "DUPI near 1 means synthetic samples are too close to real samples; "
                "DUPI near 0 means utility loss; values near the benchmark indicate balance."
            ),
        },
    }

    summary_path = args.out_dir / "summary_metrics.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(args.out_dir / "centroids.csv", report.centroid_rows)
    write_csv(args.out_dir / "class_metrics.csv", report.class_metric_rows)

    print(f"Saved summary: {summary_path}")
    print(f"Saved class metrics: {args.out_dir / 'class_metrics.csv'}")


if __name__ == "__main__":
    main()
