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
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _PROJECT_ROOT)

from src.evaluation import evaluate  # noqa: E402
from src.evaluation._io import (  # noqa: E402
    flatten_subsample_genes,
    input_fingerprint,
    load_label_hierarchy,
    load_pca_cache_meta,
    load_pca_coordinates,
    load_real,
    load_synthetic_cached,
    pca_cache_matches,
    pop_to_superpop,
    write_csv,
    write_pca_cache_meta,
    write_pca_coordinates,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-path", type=Path, default=Path("data/processed/test_data.pkl"))
    parser.add_argument("--syn-dir", type=Path, default=Path("outputs/default/synthetic_samples"))
    parser.add_argument("--hierarchy", type=Path, default=Path("data/processed/label_hierarchy.pkl"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/default/evaluation_metrics"))
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument(
        "--array-cache-mode",
        choices=["auto", "refresh", "off"],
        default="auto",
        help="Cache loaded synthetic .pt tensors into one NPZ file.",
    )
    parser.add_argument(
        "--pca-cache-mode",
        choices=["auto", "refresh", "off"],
        default="auto",
        help="Reuse cached PCA coordinates when input fingerprints match.",
    )
    parser.add_argument("--n-genes", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dupi-k", type=int, default=1)
    parser.add_argument("--tau", type=float, default=5.0)
    return parser


def _load_or_compute_pcs(
    args: argparse.Namespace,
    pca_coordinates_path: Path,
    pca_meta_path: Path,
    synthetic_array_cache: Path,
    fingerprint: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict, bool, bool]:
    """Either load PCA scores from cache or compute them from raw tensors."""
    used_pca_cache = (
        args.pca_cache_mode == "auto"
        and pca_coordinates_path.exists()
        and pca_cache_matches(pca_meta_path, fingerprint)
    )

    if used_pca_cache:
        real_pcs, syn_pcs, real_pop, syn_pop, real_sp, syn_sp, syn_names = (
            load_pca_coordinates(pca_coordinates_path)
        )
        cached_meta = load_pca_cache_meta(pca_meta_path) or {}
        pca_info = cached_meta.get("pca", {})
        print(f"Using PCA coordinate cache: {pca_coordinates_path}")
        return real_pcs, syn_pcs, real_pop, syn_pop, real_sp, syn_sp, pca_info, used_pca_cache, False

    hierarchy = load_label_hierarchy(args.hierarchy)
    real_x, real_pop = load_real(args.real_path)
    syn_x, syn_pop, syn_names, used_array_cache = load_synthetic_cached(
        args.syn_dir, synthetic_array_cache, args.array_cache_mode,
    )
    syn_x = syn_x.astype(np.float32, copy=False)

    real_flat, gene_indices = flatten_subsample_genes(real_x, args.n_genes, args.seed)
    syn_flat, _ = flatten_subsample_genes(syn_x, args.n_genes, args.seed, gene_indices)

    pca = PCA(n_components=2, random_state=args.seed)
    real_pcs = pca.fit_transform(real_flat)
    syn_pcs = pca.transform(syn_flat)
    ev = pca.explained_variance_ratio_

    real_sp = pop_to_superpop(real_pop, hierarchy)
    syn_sp = pop_to_superpop(syn_pop, hierarchy)
    pca_info = {
        "explained_variance_ratio": [float(v) for v in ev],
        "explained_variance_percent": [float(v * 100) for v in ev],
        "components_shape": list(pca.components_.shape),
        "n_features_before_pca": int(real_flat.shape[1]),
    }

    write_pca_coordinates(
        pca_coordinates_path, real_pcs, syn_pcs,
        real_pop, syn_pop, real_sp, syn_sp, syn_names,
    )
    np.save(args.out_dir / "pca_gene_indices.npy", gene_indices)
    if args.pca_cache_mode != "off":
        write_pca_cache_meta(pca_meta_path, fingerprint, pca_info)

    return real_pcs, syn_pcs, real_pop, syn_pop, real_sp, syn_sp, pca_info, used_pca_cache, used_array_cache


def main() -> None:
    args = _build_parser().parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.cache_dir or (args.out_dir / "cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    synthetic_array_cache = cache_dir / "synthetic_arrays.npz"
    pca_coordinates_path = args.out_dir / "pca_coordinates.csv"
    pca_meta_path = cache_dir / "pca_cache_meta.json"

    fingerprint = input_fingerprint(
        real_path=args.real_path,
        syn_dir=args.syn_dir,
        hierarchy=args.hierarchy,
        n_genes=args.n_genes,
        seed=args.seed,
    )

    (
        real_pcs, syn_pcs,
        real_pop, syn_pop,
        real_sp, syn_sp,
        pca_info,
        used_pca_cache, used_array_cache,
    ) = _load_or_compute_pcs(
        args, pca_coordinates_path, pca_meta_path, synthetic_array_cache, fingerprint,
    )

    report = evaluate(
        real_pcs=real_pcs, syn_pcs=syn_pcs,
        real_sp=real_sp, syn_sp=syn_sp,
        k=args.dupi_k, tau=args.tau,
    )

    summary = {
        "inputs": {
            "real_path": str(args.real_path),
            "syn_dir": str(args.syn_dir),
            "hierarchy": str(args.hierarchy),
            "n_genes": args.n_genes,
            "seed": args.seed,
            "dupi_k": args.dupi_k,
            "tau": args.tau,
            "metric_space": "PCA(2) fitted on real flattened subsampled genes",
            "array_cache_mode": args.array_cache_mode,
            "pca_cache_mode": args.pca_cache_mode,
            "array_cache_path": str(synthetic_array_cache),
            "pca_coordinates_path": str(pca_coordinates_path),
        },
        "cache": {
            "used_array_cache": used_array_cache,
            "used_pca_cache": used_pca_cache,
            "cache_dir": str(cache_dir),
        },
        "counts": {
            "n_real": int(len(real_pop)),
            "n_synthetic": int(len(syn_pop)),
            "n_features_before_pca": pca_info.get("n_features_before_pca"),
        },
        "pca": pca_info,
        "dupi": report.dupi,
        "distribution_distances": report.distribution_distances,
        "notes": {
            "dupi_source": "Jeong, Kim, and Im (2023), Eq. (8)-(11).",
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
    print(f"Saved coordinates: {pca_coordinates_path}")
    print(f"Saved class metrics: {args.out_dir / 'class_metrics.csv'}")


if __name__ == "__main__":
    main()
