#!/usr/bin/env python3
"""Emit the manuscript as one self-contained main.tex, from the run artifacts.

Every printed number and every plotted coordinate is read from a scored run or
a simulation JSON, so the manuscript cannot drift from what was computed. The
figures are pgfplots with inline coordinates, so the document is a single file
with no external data or image dependency beyond the framework schematic.

Biometrics rarely runs more than six tables and figures combined; this document
holds two tables and three figures, one of which is an external schematic.

    .venv/bin/python scripts/make_manuscript_tex.py --out main.tex
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as st

SUPER = ["AFR", "EUR", "EAS", "SAS", "AMR"]
SEEDS = ["20260327", "20260328", "20260329", "20260330"]


def diversity_ratio(run: Path, metrics: str) -> float:
    d = pd.read_csv(run / metrics / "pca_coordinates.csv")
    out = []
    for source in ("real", "synthetic"):
        g = d[d.source == source]
        var = g.groupby("pop_label")[["pc1", "pc2"]].var(ddof=1).sum(axis=1).dropna()
        n = g.groupby("pop_label").size()
        out.append(float(np.sqrt(np.average(var, weights=n[var.index]))))
    return out[1] / out[0]


def grab(run: Path, metrics: str) -> dict:
    s = json.loads((run / metrics / "summary_metrics.json").read_text())
    g = json.loads((run / "genotype_metrics/genotype_summary_chr22.json").read_text())
    cov = {r["superpopulation"]: float(r["coverage_at_real_nn95"])
           for r in csv.DictReader((run / metrics / "class_metrics.csv").open())}
    return {"DUPI": s["dupi"]["dupi"], "UIxPI": s["dupi"]["utility_privacy_product"],
            "AF_r": g["af_pearson_r"], "AF_mae": g["af_mean_abs_error"],
            "diversity": diversity_ratio(run, metrics),
            **{f"cov_{k}": cov[k] for k in SUPER}}


def paired(prefix: str, metrics: str) -> dict[str, tuple[float, float, float, float, float]]:
    """metric -> (baseline mean, baseline sd, arm mean, arm sd, paired p)."""
    base = [grab(Path(f"{prefix}_baseline_s{s}"), metrics) for s in SEEDS]
    arm = [grab(Path(f"{prefix}_mean_s{s}"), metrics) for s in SEEDS]
    out = {}
    for key in base[0]:
        b = np.array([r[key] for r in base]); m = np.array([r[key] for r in arm])
        p = st.ttest_rel(m, b).pvalue if (m - b).std(ddof=1) > 0 else float("nan")
        out[key] = (b.mean(), b.std(ddof=1), m.mean(), m.std(ddof=1), float(p))
    return out


def pvalue(p: float) -> str:
    """A complete inline-math group, so callers never concatenate math delimiters."""
    if not np.isfinite(p):
        return "---"
    return "$p<0.0001$" if p < 1e-4 else f"$p={p:.4f}$"



COLOURS = {"AFR": "red!80!black", "EUR": "blue!70!black", "EAS": "green!55!black",
           "SAS": "violet", "AMR": "orange!90!black"}


def fmt(value: float, places: int = 3) -> str:
    return f"{value:.{places}f}"


def sd(value: float, places: int = 3) -> str:
    return f"{value:.{places}f}".lstrip("0")


def pvalue(p: float) -> str:
    """A complete inline-math group, so callers never concatenate delimiters."""
    if not np.isfinite(p):
        return "---"
    return "$p<0.0001$" if p < 1e-4 else f"$p={p:.4f}$"


def scatter_plots(run: Path, metrics: str, cap: int, seed: int) -> str:
    """Inline pgfplots coordinates for one panel, synthetic thinned to ``cap`` per group."""
    frame = pd.read_csv(run / metrics / "pca_coordinates.csv")
    rng = np.random.default_rng(seed)
    lines = []
    for source, style in (
        ("synthetic", "only marks, mark=x, mark size=1.0pt, opacity=0.5"),
        ("real", "only marks, mark=o, mark size=1.4pt, thick"),
    ):
        for sp in SUPER:
            sel = frame[(frame.source == source) & (frame.superpopulation == sp)]
            if source == "synthetic" and len(sel) > cap:
                sel = sel.iloc[rng.choice(len(sel), cap, replace=False)]
            coords = " ".join(f"({a:.2f},{b:.2f})" for a, b in zip(sel.pc1, sel.pc2))
            lines.append(f"\\addplot[{style}, {COLOURS[sp]}] coordinates {{{coords}}};")
    return "\n".join(lines)


def simulation_plots(sim: dict) -> str:
    labels = {"none": "global moments", "mean": "population mean",
              "mean_std": "population mean and variance"}
    out = []
    for arm, label in labels.items():
        coords = " ".join(f"({n},{v:.5f})"
                          for n, v in zip(sim["sizes"], sim["kl"][arm]))
        out.append(f"\\addplot coordinates {{{coords}}};\n\\addlegendentry{{{label}}}")
    return "\n".join(out)


def representation_rows() -> str:
    rows = []
    for tag, prefix in (("Poisson", "outputs/ms"), ("Binomial", "outputs/b2")):
        for arm, name in (("baseline", "global"), ("mean", "population mean")):
            r, mae = [], []
            for seed in SEEDS:
                path = Path(f"{prefix}_{arm}_s{seed}/genotype_metrics/genotype_summary_chr22.json")
                if not path.exists():
                    continue
                g = json.loads(path.read_text())
                r.append(g["af_pearson_r"]); mae.append(g["af_mean_abs_error"])
            if r:
                rows.append(f"{tag} & {name} & {len(r)} & "
                            f"{np.mean(r):.4f} & {np.mean(mae):.4f} \\\\")
    return "\n".join(rows)


def paired_rows(val: dict, test: dict) -> str:
    spec = [("EUR coverage", "cov_EUR", 3), ("EAS coverage", "cov_EAS", 3),
            ("SAS coverage", "cov_SAS", 3), ("AFR coverage", "cov_AFR", 3),
            ("AMR coverage", "cov_AMR", 3), ("DUPI", "DUPI", 3),
            ("Utility $\\times$ privacy", "UIxPI", 3),
            ("Allele-frequency $r$", "AF_r", 4),
            ("Allele-frequency MAE", "AF_mae", 4),
            ("Within-population spread", "diversity", 3)]
    rows = []
    for label, key, places in spec:
        vb, vbs, vm, vms, _ = val[key]
        tb, tbs, tm, tms, tp = test[key]
        rows.append(
            f"{label} & {fmt(vb, places)}\\,({sd(vbs, places)}) "
            f"& {fmt(vm, places)}\\,({sd(vms, places)}) "
            f"& {fmt(tb, places)}\\,({sd(tbs, places)}) "
            f"& {fmt(tm, places)}\\,({sd(tms, places)}) & {pvalue(tp)} \\\\")
    return "\n".join(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("main.tex"))
    ap.add_argument("--prefix", default="outputs/b2")
    ap.add_argument("--simulation", type=Path,
                    default=Path("outputs/simulation_moment_conditioning.json"))
    ap.add_argument("--scatter-cap", type=int, default=260,
                    help="Synthetic points drawn per superpopulation, thinned for file size.")
    args = ap.parse_args()

    val = paired(args.prefix, "evaluation_metrics_val")
    test = paired(args.prefix, "evaluation_metrics")
    sim = json.loads(args.simulation.read_text())
    template = (Path(__file__).parent / "manuscript_template.tex").read_text(encoding="utf-8")

    def T(key: str, places: int = 3) -> tuple[str, str, str]:
        b, _, m, _, p = test[key]
        return fmt(b, places), fmt(m, places), pvalue(p)

    eur_b, eur_m, eur_p = T("cov_EUR")
    eas_b, eas_m, eas_p = T("cov_EAS")
    dupi_b, dupi_m, dupi_p = T("DUPI")
    mae_b, mae_m, mae_p = T("AF_mae", 4)
    r_b, r_m, _ = T("AF_r", 4)
    div_b, div_m, div_p = T("diversity")

    poisson = json.loads(Path(
        f"outputs/ms_baseline_s{SEEDS[0]}/genotype_metrics/genotype_summary_chr22.json").read_text())
    values = {
        "PAIRED_ROWS": paired_rows(val, test),
        "REPRESENTATION_ROWS": representation_rows(),
        "SIM_PLOTS": simulation_plots(sim),
        "PLANE_BASELINE": scatter_plots(Path(f"{args.prefix}_baseline_s{SEEDS[0]}"),
                                        "evaluation_metrics_val", args.scatter_cap, 1),
        "PLANE_CONDITIONAL": scatter_plots(Path(f"{args.prefix}_mean_s{SEEDS[0]}"),
                                           "evaluation_metrics_val", args.scatter_cap, 1),
        "EUR_B": eur_b, "EUR_M": eur_m, "EUR_P": eur_p,
        "EAS_B": eas_b, "EAS_M": eas_m, "EAS_P": eas_p,
        "DUPI_B": dupi_b, "DUPI_M": dupi_m, "DUPI_P": dupi_p,
        "MAE_B": mae_b, "MAE_M": mae_m, "MAE_P": mae_p,
        "R_B": r_b, "R_M": r_m,
        "DIV_B": div_b, "DIV_M": div_m, "DIV_P": div_p,
        "POI_MAE": f"{poisson['af_mean_abs_error']:.4f}",
        "CROSSOVER": str(sim["crossover_n_per_pop"]),
        "SIM_POPS": str(sim["settings"]["pops"]),
        "SIM_FEATURES": str(sim["settings"]["features"]),
        "SIM_SHIFT": str(sim["settings"]["shift"]),
        "SIM_RATIO": str(sim["settings"]["ratio"]),
        "SIM_REPS": str(sim["settings"]["replicates"]),
        "NSEEDS": str(len(SEEDS)),
    }
    for key, body in values.items():
        token = "%%" + key + "%%"
        if token not in template:
            raise SystemExit(f"template is missing {token}")
        template = template.replace(token, body)
    leftover = [t for t in template.split("%%") if t.isupper() and t.isidentifier()]
    if leftover:
        raise SystemExit(f"unfilled placeholders: {sorted(set(leftover))}")

    args.out.write_text(template, encoding="utf-8")
    print(f"wrote {args.out} ({len(template.splitlines())} lines, "
          f"{args.out.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
