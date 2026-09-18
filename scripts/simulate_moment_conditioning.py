#!/usr/bin/env python3
"""When is conditioning the prior on a population moment worth its estimation error?

The pipeline's conditional prior replaces one global (mean, sd) with per-population
estimates. Each conditioned moment removes real between-population signal from what
the generator must learn, and adds the sampling error of estimating that moment from
n_p training individuals. This simulation fixes a known truth and measures where the
trade turns, so the empirical finding on 1000 Genomes ("the mean helps, the variance
does not") becomes a statement about n_p rather than about one dataset.

Truth: feature g in population p is N(mu_p[g], sigma_p[g]). Three arms estimate that
from n_p draws per population and generate from the estimate:

    none      N(mu_hat_global, sd_hat_global)
    mean      N(mu_hat_p,      sd_hat_pooled)     <- pooled over all P populations
    mean_std  N(mu_hat_p,      sd_hat_p)

The generator itself is treated as exact, so what is measured is the cost of the
normalization choice alone. The estimand is the Kullback-Leibler divergence from the
true population law to the generated one, which is closed form for normals:

    KL = log(sd_hat/sigma) + (sigma^2 + (mu - mu_hat)^2) / (2 sd_hat^2) - 1/2

Defaults are calibrated to the measured 1000 Genomes panel: between-population mean
shift 0.329 pooled sd (median), diversity ratio 1.29 between the most and least
diverse population, 26 populations, n_p 49-91.

    .venv/bin/python scripts/simulate_moment_conditioning.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ARMS = ("none", "mean", "mean_std")


def normal_kl(mu: np.ndarray, sigma: np.ndarray, mu_hat: np.ndarray, sd_hat: np.ndarray) -> np.ndarray:
    """KL( N(mu, sigma^2) || N(mu_hat, sd_hat^2) ), elementwise."""
    return (
        np.log(sd_hat / sigma)
        + (sigma**2 + (mu - mu_hat) ** 2) / (2.0 * sd_hat**2)
        - 0.5
    )


def truth(
    rng: np.random.Generator, n_pops: int, n_features: int, shift: float, ratio: float
) -> tuple[np.ndarray, np.ndarray]:
    """Per-population means and sds with a target shift and diversity ratio.

    ``shift`` is the median |mu_p - mu_bar| in pooled-sd units; for a half-normal
    the median is 0.6745 sd, so the generating scale is set accordingly.
    ``ratio`` is the sd of the most diverse population over the least diverse.
    """
    mu = rng.normal(0.0, shift / 0.6745, size=(n_pops, n_features))
    mu -= mu.mean(axis=0, keepdims=True)
    spread = np.linspace(1.0 / np.sqrt(ratio), np.sqrt(ratio), n_pops)
    rng.shuffle(spread)
    sigma = np.repeat(spread[:, None], n_features, axis=1)
    return mu, sigma


def arm_estimates(draws: np.ndarray, arm: str) -> tuple[np.ndarray, np.ndarray]:
    """(mu_hat, sd_hat) broadcast to (P, G) for one arm, from ``draws`` (P, n, G)."""
    n_pops, n_per, _ = draws.shape
    per_pop_mean = draws.mean(axis=1)
    per_pop_var = draws.var(axis=1, ddof=1)
    if arm == "none":
        flat = draws.reshape(n_pops * n_per, -1)
        return (
            np.repeat(flat.mean(axis=0)[None], n_pops, axis=0),
            np.repeat(flat.std(axis=0, ddof=1)[None], n_pops, axis=0),
        )
    pooled = np.sqrt(per_pop_var.mean(axis=0))  # equal n_p, so a plain mean pools it
    if arm == "mean":
        return per_pop_mean, np.repeat(pooled[None], n_pops, axis=0)
    return per_pop_mean, np.sqrt(per_pop_var)


def run(args: argparse.Namespace) -> dict:
    sizes = [int(n) for n in args.sizes]
    curve: dict[str, list[float]] = {arm: [] for arm in ARMS}
    error: dict[str, list[float]] = {arm: [] for arm in ARMS}

    for n_per in sizes:
        replicate: dict[str, list[float]] = {arm: [] for arm in ARMS}
        for replicate_index in range(args.replicates):
            rng = np.random.default_rng(args.seed + 1000 * replicate_index + n_per)
            mu, sigma = truth(rng, args.pops, args.features, args.shift, args.ratio)
            draws = rng.normal(mu[:, None, :], sigma[:, None, :], size=(args.pops, n_per, args.features))
            for arm in ARMS:
                mu_hat, sd_hat = arm_estimates(draws, arm)
                replicate[arm].append(float(normal_kl(mu, sigma, mu_hat, sd_hat).mean()))
        for arm in ARMS:
            values = np.array(replicate[arm])
            curve[arm].append(float(values.mean()))
            error[arm].append(float(values.std(ddof=1)) if len(values) > 1 else 0.0)

    crossover = next(
        (n for n, a, b in zip(sizes, curve["mean_std"], curve["mean"]) if a < b), None
    )
    return {"sizes": sizes, "kl": curve, "kl_sd": error, "crossover_n_per_pop": crossover,
            "settings": {k: getattr(args, k) for k in
                         ("pops", "features", "shift", "ratio", "replicates", "seed")}}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pops", type=int, default=26)
    parser.add_argument("--features", type=int, default=2000)
    parser.add_argument("--shift", type=float, default=0.329,
                        help="Median |mu_p - mu_bar| in pooled-sd units (1KG measurement).")
    parser.add_argument("--ratio", type=float, default=1.29,
                        help="sd of the most diverse population over the least (1KG measurement).")
    parser.add_argument("--sizes", nargs="+",
                        default=[10, 20, 30, 49, 70, 91, 150, 300, 600, 1200])
    parser.add_argument("--replicates", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260327)
    parser.add_argument("--out", type=Path, default=Path("outputs/simulation_moment_conditioning.json"))
    args = parser.parse_args(argv)

    result = run(args)
    print(f"{args.pops} populations, {args.features} features, "
          f"shift {args.shift} sd, diversity ratio {args.ratio}, "
          f"{args.replicates} replicates\n")
    print(f"{'n per pop':>10s}" + "".join(f"{arm:>14s}" for arm in ARMS) + f"{'best':>10s}")
    for index, n_per in enumerate(result["sizes"]):
        values = [result["kl"][arm][index] for arm in ARMS]
        best = ARMS[int(np.argmin(values))]
        marker = "  <- 1KG" if 49 <= n_per <= 91 else ""
        print(f"{n_per:10d}" + "".join(f"{v:14.4f}" for v in values) + f"{best:>10s}{marker}")

    crossover = result["crossover_n_per_pop"]
    print(f"\nmean_std overtakes mean at n_p = {crossover}"
          if crossover else "\nmean_std never overtakes mean on this grid")
    print("1000 Genomes has n_p = 49-91, so the second moment is on the wrong side of it.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"saved {args.out}")


def _self_check() -> None:
    """The arms must order as the estimability argument predicts."""
    tiny = argparse.Namespace(pops=8, features=400, shift=0.329, ratio=1.29,
                              sizes=[15, 2000], replicates=3, seed=1)
    out = run(tiny)
    small, large = 0, 1
    # Conditioning on the mean beats not conditioning, at every sample size.
    assert out["kl"]["mean"][small] < out["kl"]["none"][small]
    assert out["kl"]["mean"][large] < out["kl"]["none"][large]
    # The second moment costs more than it returns when n_p is small, and pays
    # off once the per-population variance is well estimated.
    assert out["kl"]["mean_std"][small] > out["kl"]["mean"][small]
    assert out["kl"]["mean_std"][large] < out["kl"]["mean"][large]
    print("self-check ok")


if __name__ == "__main__":
    import sys

    if "--self-check" in sys.argv:
        _self_check()
    else:
        main()
