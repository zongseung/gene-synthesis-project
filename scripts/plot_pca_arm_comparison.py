#!/usr/bin/env python3
"""Compare runs on one shared PCA plane, plus per-superpopulation coverage.

Every run's evaluation fits PCA(2) on the real split only, with the same seed
and gene subsample, so the basis is identical across runs and the panels are
directly comparable. This script checks that before drawing: if the real
coordinates disagree, shared axes would be a lie and it refuses.

Row 1: real (circles) vs synthetic (crosses) per run, shared axes.
Row 2 left: coverage_at_real_nn95 per superpopulation, grouped by run.
Row 2 right: within-population spread relative to real. Coverage alone tells a
one-sided story, because it rewards landing on the real clusters and says
nothing about reproducing the diversity inside each population.

Usage::

    .venv/bin/python scripts/plot_pca_arm_comparison.py \\
        --runs baseline=outputs/20260917_wholegenome_k4 \\
               mean=outputs/20260917_cond_mean \\
               mean_std=outputs/20260917_cond_mean_std \\
        --out outputs/pca_conditional_prior_comparison.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.plot_pca import SUPERPOP_COLORS, SUPERPOP_ORDER  # noqa: E402

DEFAULT_RUNS = [
    "baseline=outputs/20260917_wholegenome_k4",
    "mean=outputs/20260917_cond_mean",
    "mean_std=outputs/20260917_cond_mean_std",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", default=DEFAULT_RUNS, metavar="LABEL=DIR")
    parser.add_argument("--out", type=Path, default=Path("outputs/pca_conditional_prior_comparison.png"))
    parser.add_argument("--tolerance", type=float, default=1e-3,
                        help="Max real-coordinate disagreement allowed between runs.")
    parser.add_argument("--dpi", type=int, default=160)
    return parser.parse_args(argv)


def load_run(run_dir: Path) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    metrics = run_dir / "evaluation_metrics"
    coords = pd.read_csv(metrics / "pca_coordinates.csv")
    summary = json.loads((metrics / "summary_metrics.json").read_text(encoding="utf-8"))
    classes = pd.read_csv(metrics / "class_metrics.csv")
    return coords, summary, classes


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    runs = []
    for spec in args.runs:
        if "=" not in spec:
            raise SystemExit(f"Expected LABEL=DIR, got {spec!r}")
        label, path = spec.split("=", 1)
        runs.append((label, Path(path)))

    loaded = [(label, *load_run(path)) for label, path in runs]

    # The shared basis is the premise of the whole figure, so verify it.
    reference = loaded[0][1].query("source == 'real'").reset_index(drop=True)
    for label, coords, _, _ in loaded[1:]:
        real = coords.query("source == 'real'").reset_index(drop=True)
        if not np.array_equal(reference.sample_id.values, real.sample_id.values):
            raise SystemExit(f"{label}: real sample ids differ from {loaded[0][0]}")
        drift = max(
            float((reference[c] - real[c]).abs().max()) for c in ("pc1", "pc2")
        )
        if drift > args.tolerance:
            raise SystemExit(
                f"{label}: real PCA coordinates differ from {loaded[0][0]} by {drift:.2e}; "
                "the runs do not share a basis and must not share axes"
            )

    all_coords = pd.concat([c for _, c, _, _ in loaded])
    pad = 0.05 * max(np.ptp(all_coords.pc1), np.ptp(all_coords.pc2))
    xlim = (all_coords.pc1.min() - pad, all_coords.pc1.max() + pad)
    ylim = (all_coords.pc2.min() - pad, all_coords.pc2.max() + pad)

    n = len(loaded)
    fig = plt.figure(figsize=(4.6 * n, 8.4))
    grid = fig.add_gridspec(2, n, height_ratios=[1.55, 1.0], hspace=0.28, wspace=0.12)

    for col, (label, coords, summary, classes) in enumerate(loaded):
        ax = fig.add_subplot(grid[0, col])
        real, syn = coords.query("source == 'real'"), coords.query("source == 'synthetic'")
        for sp in SUPERPOP_ORDER:
            s = syn[syn.superpopulation == sp]
            ax.scatter(s.pc1, s.pc2, c=SUPERPOP_COLORS[sp], s=26, alpha=0.75,
                       marker="x", linewidths=1.1)
        for sp in SUPERPOP_ORDER:
            r = real[real.superpopulation == sp]
            ax.scatter(r.pc1, r.pc2, c=SUPERPOP_COLORS[sp], s=46, alpha=0.95,
                       marker="o", edgecolors="black", linewidths=0.7)
        d = summary["dupi"]
        weights = classes.n_real.to_numpy(dtype=float)
        coverage = float(np.average(classes.coverage_at_real_nn95.to_numpy(dtype=float),
                                    weights=weights))
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_title(
            f"{label}\nDUPI {d['dupi']:.3f} (target {d['dupi_benchmark']:.3f})"
            f"   coverage {coverage:.3f}",
            fontsize=11,
        )
        ax.set_xlabel("PC1")
        if col == 0:
            ax.set_ylabel("PC2")
        else:
            ax.set_yticklabels([])
        ax.grid(alpha=0.15, linewidth=0.5)

    # Row 2 left: coverage per superpopulation, the recall-like number.
    ax = fig.add_subplot(grid[1, : max(n - 1, 1)])
    width = 0.8 / n
    offsets = np.arange(len(SUPERPOP_ORDER), dtype=float)
    for i, (label, _, _, classes) in enumerate(loaded):
        by_sp = classes.set_index("superpopulation")
        values = [float(by_sp.loc[sp, "coverage_at_real_nn95"]) for sp in SUPERPOP_ORDER]
        bars = ax.bar(offsets + (i - (n - 1) / 2) * width, values, width,
                      label=label, edgecolor="black", linewidth=0.5,
                      alpha=0.55 + 0.45 * i / max(n - 1, 1), color="#555555")
        for rect, value in zip(bars, values):
            ax.text(rect.get_x() + rect.get_width() / 2, value + 0.015, f"{value:.2f}",
                    ha="center", va="bottom", fontsize=8)
    for i, sp in enumerate(SUPERPOP_ORDER):
        ax.get_xticklabels()
        ax.axvspan(i - 0.5, i + 0.5, color=SUPERPOP_COLORS[sp], alpha=0.07)
    ax.set_xticks(offsets)
    ax.set_xticklabels(SUPERPOP_ORDER)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("coverage @ real NN 95th pct")
    ax.set_title("Fraction of real samples covered by synthetic, per superpopulation", fontsize=11)
    ax.legend(loc="lower right", fontsize=9, ncol=n)
    ax.grid(axis="y", alpha=0.2, linewidth=0.5)

    # Row 2 right: within-population spread, the cost coverage cannot see.
    ax = fig.add_subplot(grid[1, max(n - 1, 1):])
    ratios = {}
    for label, coords, _, _ in loaded:
        values = []
        for source in ("real", "synthetic"):
            group = coords[coords.source == source]
            var = group.groupby("pop_label")[["pc1", "pc2"]].var(ddof=1).sum(axis=1).dropna()
            counts = group.groupby("pop_label").size()
            values.append(float(np.sqrt(np.average(var, weights=counts[var.index]))))
        ratios[label] = values[1] / values[0]
    bars = ax.bar(range(n), [ratios[label] for label, *_ in loaded], 0.6,
                  color="#555555", edgecolor="black", linewidth=0.5)
    for i, bar in enumerate(bars):
        bar.set_alpha(0.55 + 0.45 * i / max(n - 1, 1))
    for rect, label in zip(bars, ratios):
        ax.text(rect.get_x() + rect.get_width() / 2, ratios[label] + 0.02,
                f"{ratios[label]:.2f}", ha="center", va="bottom", fontsize=9)
    ax.axhline(1.0, color="#B22222", linestyle="--", linewidth=1.2)
    ax.text(n - 0.5, 1.02, "real", color="#B22222", fontsize=8, ha="right")
    ax.set_xticks(range(n))
    ax.set_xticklabels([label for label, *_ in loaded])
    ax.set_ylim(0, 1.25)
    ax.set_ylabel("synthetic / real")
    ax.set_title("Within-population diversity, pooled over 26 populations", fontsize=11)
    ax.grid(axis="y", alpha=0.2, linewidth=0.5)

    style = [
        plt.Line2D([0], [0], marker="o", linestyle="", color="gray",
                   markeredgecolor="black", markersize=8, label="Real (held-out test)"),
        plt.Line2D([0], [0], marker="x", linestyle="", color="gray",
                   markersize=8, label="Synthetic"),
    ]
    pops = [
        plt.Line2D([0], [0], marker="s", linestyle="", markerfacecolor=SUPERPOP_COLORS[sp],
                   markeredgecolor="none", markersize=9, label=sp)
        for sp in SUPERPOP_ORDER
    ]
    fig.legend(handles=style + pops, loc="upper center", ncol=7, frameon=False,
               bbox_to_anchor=(0.5, 0.995), fontsize=9)
    fig.suptitle("Population-conditional prior on the shared real-fitted PCA plane",
                 y=0.955, fontsize=13)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    print(f"saved {args.out}")
    for label, _, summary, classes in loaded:
        by_sp = classes.set_index("superpopulation")
        cov = " ".join(f"{sp} {float(by_sp.loc[sp, 'coverage_at_real_nn95']):.2f}"
                       for sp in SUPERPOP_ORDER)
        print(f"  {label:10s} DUPI {summary['dupi']['dupi']:.3f}  "
              f"within-pop spread {ratios[label]:.2f}x real  coverage: {cov}")


if __name__ == "__main__":
    main()
