#!/usr/bin/env python3
"""Re-normalize an existing processed dir with a population-conditional prior.

Implements the "Data-Normalization" shift of ShiftDDPMs (AAAI 2023), whose
forward process is PriorGrad's (ICLR 2022): subtract the conditional mean from
x_0, train the diffusion model on the residual, and add the mean back at the end
of sampling. See claudedocs/research_conditional_prior_vs_posthoc_20260917.md.

The existing splits were normalized with one global mean/std and no clipping, so
that mapping is exactly invertible. This inverts it and re-normalizes per
population, which costs minutes instead of re-running the whole VCF -> GLM-PCA
pass for every arm. Nothing about the model, the diffusion kernel or the FiLM
conditioning changes; Data-Normalization does not separate the trajectories, so
the conditioning must stay in the network.

Usage:
    .venv/bin/python scripts/make_conditional_prior_data.py \\
        --arm mean --output-dir data/processed_cond_mean
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import pickle
from pathlib import Path

import numpy as np


from src.preprocessing.tokenizer import (  # noqa: E402
    CONDITIONAL_ARMS,
    apply_normalization,
    fit_normalization_stats,
    invert_normalization,
    load_normalization_stats,
)

SPLITS = ("train_data.pkl", "val_data.pkl", "test_data.pkl")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm",
        required=True,
        choices=[arm for arm in CONDITIONAL_ARMS if arm != "none"],
        help="mean: per-population mean, pooled within-population std. "
             "mean_std: per-population mean and std.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--variance-floor",
        type=float,
        default=0.01,
        help="Lower-bound each per-population std at this multiple of the pooled "
             "within-population std. Only the mean_std arm has a per-population "
             "denominator, where it stops numerically constant cells from turning "
             "fp32 rounding dust into order-one residuals.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-3,
        help="Max allowed absolute round-trip error on the raw train tensor.",
    )
    return parser.parse_args(argv)


def load_split(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as handle:
        x, y = pickle.load(handle)
    return np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.int64)


def save_split(path: Path, x: np.ndarray, y: np.ndarray) -> None:
    with path.open("wb") as handle:
        pickle.dump((x, y), handle, protocol=4)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    source, output = args.source_dir, args.output_dir

    # Gate artifacts are frozen records; never write over an existing run.
    if output.exists():
        raise FileExistsError(f"Output directory already exists: {output}")

    source_stats_path = source / "normalization_stats.pkl"
    source_stats = load_normalization_stats(source_stats_path)
    if source_stats["conditional"] != "none":
        raise ValueError(
            f"{source_stats_path} already carries a {source_stats['conditional']!r} "
            "conditional prior; derive every arm from the unconditional baseline"
        )
    if source_stats["clip"] is not None:
        # Clipped values lost their originals, so the inverse is not the raw tensor.
        raise ValueError(
            f"{source_stats_path} was fit with clip={source_stats['clip']}; "
            "inverting it would not recover the raw features"
        )

    hierarchy_path = source / "label_hierarchy.pkl"
    with hierarchy_path.open("rb") as handle:
        n_pops = len(pickle.load(handle)["pop_to_idx"])

    raw_train, y_train = load_split(source / "train_data.pkl")
    raw_train = invert_normalization(raw_train, source_stats)
    present = set(np.unique(y_train).tolist())
    if present != set(range(n_pops)):
        raise ValueError(
            f"Train split covers {len(present)}/{n_pops} populations "
            f"(missing {sorted(set(range(n_pops)) - present)}); every population "
            "needs training rows for its own prior"
        )

    stats = fit_normalization_stats(
        raw_train, labels=y_train, conditional=args.arm, variance_floor=args.variance_floor,
    )

    # The only runnable check that matters here: the new mapping must invert.
    round_trip = float(
        np.abs(
            invert_normalization(
                apply_normalization(raw_train, stats, y_train), stats, y_train
            )
            - raw_train
        ).max()
    )
    if not round_trip < args.tolerance:
        raise ValueError(f"Round-trip error {round_trip:.3e} exceeds {args.tolerance:.3e}")

    output.mkdir(parents=True)
    save_split(output / "train_data.pkl", apply_normalization(raw_train, stats, y_train), y_train)
    del raw_train

    for name in SPLITS[1:]:
        values, labels = load_split(source / name)
        values = invert_normalization(values, source_stats)
        save_split(output / name, apply_normalization(values, stats, labels), labels)

    stats_path = output / "normalization_stats.pkl"
    with stats_path.open("wb") as handle:
        pickle.dump(stats, handle, protocol=4)

    # Everything else is unchanged, so link it rather than duplicating gigabytes.
    for entry in sorted(source.iterdir()):
        target = output / entry.name
        if entry.is_file() and not target.exists():
            os.symlink(os.path.relpath(entry.resolve(), output.resolve()), target)

    metadata_path = output / "preprocessing_metadata.json"
    if metadata_path.is_symlink():
        metadata_path.unlink()
    metadata = json.loads((source / "preprocessing_metadata.json").read_text(encoding="utf-8"))
    metadata["normalization"] = {
        "fit_split": stats["fit_split"],
        "shape": list(stats["shape"]),
        "clip": stats["clip"],
        "conditional_prior": stats["conditional"],
        "variance_floor": stats["variance_floor"],
        "n_populations": int(stats["mean"].shape[0]),
        "derived_from": str(source),
        "source_stats_sha256": hashlib.sha256(source_stats_path.read_bytes()).hexdigest(),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"arm: {stats['conditional']} (variance floor {stats['variance_floor']:g})")
    print(f"mean: {stats['mean'].shape}, std: {stats['std'].shape}")
    print(f"round-trip max abs error: {round_trip:.3e}")
    print("\n--- training overrides ---")
    print(f"data.processed_dir={output}")
    print(f"data.normalization_stats_path={output}/normalization_stats.pkl")
    print(f"data.preprocessing_metadata_path={output}/preprocessing_metadata.json")
    print("---")


if __name__ == "__main__":
    main()
