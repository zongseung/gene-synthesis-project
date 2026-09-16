#!/usr/bin/env python3
# ─── How to run ───
#   python scripts/hipodit_simulation.py --output-dir O            # plan Task 7 primary grid
#   python scripts/hipodit_simulation.py --output-dir O --stress   # + AF 0.005 / 0.995 stress rows
#   python scripts/hipodit_simulation.py --output-dir O --replicates 5 --n 500   # pilot
# Plan Task 7: calls are generated from the tilted family itself with known cohort offsets and a
# known per-SNP tau, on an oracle latent whose loci correlate through `latent_rho`; B0, B1 and T
# are fitted on 80% of the rows and scored on the rest. Nothing here reads the 1000 Genomes panel.
from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import product
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit, softmax

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.hipodit_genotype_check import provenance  # noqa: E402
from src.models.genotype_decoder import (  # noqa: E402
    GenotypeDecoder,
    call_logits,
    distance_bins,
    fit_decoder,
    nll_calls,
    sample_calls,
)

# Plan Task 7 step 1, verbatim. `cohort_ratio` is the share of cohort 0 in a two-cohort panel.
GRID = {"af": (0.05, 0.20, 0.50), "tau": (-1.0, 0.0, 1.0), "n": (500, 2000, 10000),
        "latent_rho": (0.0, 0.3, 0.7), "cohort_ratio": (0.5, 0.8)}
STRESS_AF = (0.005, 0.995)
FACTORS = tuple(GRID)
ARMS = (("B0", "none"), ("B1", "none"), ("T", "snp"))
# The true cohort effect on the logit scale, the same in every scenario; the fit must recover AF
# in spite of it, not because it is absent.
COHORT_OFFSET = (0.4, -0.4)
COLUMNS = ("scenario", "stress", *FACTORS, "replicate", "arm", "failed", "runtime_seconds",
           "nll_per_call", "af_error", "het_error", "covariance_error", "tau_bias", "tau_rmse")


def truth_decoder(snps: int, tau: float, offset: np.ndarray) -> GenotypeDecoder:
    """The data-generating arm-T decoder: a known cohort offset and one tau on every SNP."""
    positions, offsets = np.arange(1.0, snps + 1), np.array([0, snps])
    return GenotypeDecoder("T", offset, np.zeros((1, 3, 3)), distance_bins(positions, offsets, 1),
                           offsets, np.ones(len(offset)) / len(offset), 1.0, 0.0, 0.0, True,
                           tilt=np.full(snps, tau), tilt_scope="snp", lambda_tilt=1.0)


def simulate(af: float, tau: float, n: int, latent_rho: float, cohort_ratio: float, snps: int,
             rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray, GenotypeDecoder]:
    """Calls, oracle base logits and cohort labels drawn from the tilted family with known truth."""
    covariance = np.full((snps, snps), latent_rho) + (1 - latent_rho) * np.eye(snps)
    base_logits = logit(af) + rng.multivariate_normal(np.zeros(snps), covariance, size=n)
    labels = (rng.random(n) >= cohort_ratio).astype(np.int64)
    truth = truth_decoder(snps, tau, np.tile(np.array(COHORT_OFFSET)[:, None], (1, snps)))
    calls = sample_calls(truth, base_logits, labels, rng).astype(np.float64)
    return calls, base_logits, labels, truth


def replicate(task: tuple) -> list[dict]:
    """Fit every arm on one simulated replicate; a fit that raises or fails to converge is a
    failure row with NaN metrics, never a silently dropped one."""
    scenario, stress, factors, index, snps, seed = task
    import torch

    torch.set_num_threads(1)
    rng = np.random.default_rng([seed, scenario, index])
    calls, base_logits, labels, truth = simulate(**factors, snps=snps, rng=rng)
    n = len(calls)
    train, held = np.arange(int(0.8 * n)), np.arange(int(0.8 * n), n)
    truth_q = softmax(call_logits(truth, calls[held], base_logits[held], labels[held]), axis=2)
    true_af, true_het = (truth_q @ np.arange(3)).mean(axis=0) / 2, truth_q[:, :, 1].mean(axis=0)
    off_diagonal = ~np.eye(snps, dtype=bool)
    rows = []
    for arm, scope in ARMS:
        row = dict(zip(COLUMNS, (scenario, stress, *(factors[name] for name in FACTORS),
                                 index, arm, True, np.nan, *([np.nan] * 6))))
        started = time.monotonic()
        try:
            fitted = fit_decoder(arm, calls, base_logits, labels, train, np.arange(1.0, snps + 1),
                                 truth.offsets, len(COHORT_OFFSET), n_bins=1, tilt_scope=scope)
            generated = sample_calls(fitted, base_logits[held], labels[held],
                                     np.random.default_rng([seed, scenario, index, 1]))
            row.update({
                "failed": not fitted.converged, "runtime_seconds": time.monotonic() - started,
                "nll_per_call": float(np.nanmean(nll_calls(fitted, calls[held], base_logits[held],
                                                           labels[held]))),
                "af_error": float(np.abs(generated.mean(axis=0) / 2 - true_af).mean()),
                "het_error": float(np.abs((generated == 1).mean(axis=0) - true_het).mean()),
                "covariance_error": float(np.abs(np.cov(generated.T, ddof=0)
                                                 - np.cov(calls[held].T, ddof=0))[off_diagonal].mean()),
            })
            if arm == "T":
                error = fitted.tilt - factors["tau"]
                row.update({"tau_bias": float(error.mean()),
                            "tau_rmse": float(np.sqrt((error**2).mean()))})
        except Exception as error:  # noqa: BLE001 - the failure rate is a reported statistic
            row["runtime_seconds"] = time.monotonic() - started
            row["error"] = repr(error)
        rows.append(row)
    return rows


def scenarios(args: argparse.Namespace) -> list[tuple[int, bool, dict]]:
    grid = {name: getattr(args, name) for name in FACTORS}
    primary = [dict(zip(FACTORS, values)) for values in product(*(grid[name] for name in FACTORS))]
    stress = [dict(zip(FACTORS, values)) for values in
              product(STRESS_AF, *(grid[name] for name in FACTORS[1:]))] if args.stress else []
    return [(index, index >= len(primary), factors)
            for index, factors in enumerate(primary + stress)]


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    """Plan Task 7 step 4: per scenario and arm, the mean and SD of every statistic plus the
    failure rate; stress scenarios are summarised in the same table but flagged, never pooled."""
    metrics = list(COLUMNS[COLUMNS.index("runtime_seconds"):])
    grouped = rows.groupby(["scenario", "stress", *FACTORS, "arm"], sort=True)
    summary = grouped[metrics].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary["failure_rate"] = grouped["failed"].mean()
    summary["replicates"] = grouped.size()
    return summary.reset_index()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    for name, values in GRID.items():
        parser.add_argument(f"--{name.replace('_', '-')}", nargs="+", type=type(values[0]),
                            default=list(values))
    parser.add_argument("--replicates", type=int, default=200)
    parser.add_argument("--snps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--stress", action="store_true", help="add the AF 0.005/0.995 rows")
    return parser


def run(args: argparse.Namespace) -> pd.DataFrame:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.replicates < 1 or args.snps < 2 or args.jobs < 1:
        raise ValueError("--replicates and --jobs must be positive and --snps at least 2")
    started = time.monotonic()
    tasks = [(index, stress, factors, rep, args.snps, args.seed)
             for index, stress, factors in scenarios(args) for rep in range(args.replicates)]
    # ponytail: one process per replicate task; a torch fit is single-threaded here, so --jobs
    # scales linearly until the machine runs out of cores.
    with Pool(args.jobs) as pool:
        rows = pd.DataFrame([row for batch in pool.imap(replicate, tasks, chunksize=4)
                             for row in batch])
    args.output_dir.mkdir(parents=True)
    rows.to_csv(args.output_dir / "replicates.csv", index=False)
    summary = summarize(rows)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    (args.output_dir / "manifest.json").write_text(json.dumps({
        "grid": {name: getattr(args, name) for name in FACTORS}, "stress": args.stress,
        "replicates": args.replicates, "snps": args.snps, "seed": args.seed,
        "cohort_offset": COHORT_OFFSET, "arms": ARMS, "scenarios": len(tasks) // args.replicates,
        "failure_rate": float(rows["failed"].mean()), "runtime_seconds": time.monotonic() - started,
        "limitation": "Correctly specified tilted family on an oracle latent; says nothing about "
                      "the latent generator or about misspecified dependence.",
        **provenance()}, indent=2))
    return summary


if __name__ == "__main__":
    print(run(build_parser().parse_args()).to_string())
