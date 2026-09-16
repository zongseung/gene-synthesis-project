#!/usr/bin/env python3
# ─── How to run ───
#   python scripts/hipodit_tables.py --multiseed-dir M --privacy-dir V --output-dir O
# Plan Task 14 step 1: every manuscript table is generated from the machine-readable gate
# outputs (summary_decoder.json, privacy_report.json); no metric is ever typed into LaTeX by hand.
from __future__ import annotations

import argparse
import json
from pathlib import Path

GATE3_ROWS = (("af_mae", "AF MAE"), ("cohort_af_mae", "Cohort AF MAE"),
              ("heterozygosity_mae", "Heterozygosity MAE"),
              ("genotype_proportion_tv", "Genotype TV"),
              ("local_dosage_covariance_mae", "Signed covariance MAE"),
              ("ld_r2_mae", r"LD $r^2$ MAE (descriptive)"),
              ("classifier_accuracy", "Population classifier accuracy"),
              ("classifier_macro_auc", "Population classifier macro AUC"))
PRIVACY_ROWS = (("exact_duplicate_rate_train", "Exact duplicates vs train"),
                ("exact_duplicate_rate_test", "Exact duplicates vs test"),
                ("near_duplicate_rate_train", "Near duplicates vs train"),
                ("nn_distance_to_train_median", "NN distance to train (median)"),
                ("nn_distance_to_train_p1", "NN distance to train (p1)"),
                ("membership_inference_auc", "Membership inference AUC"))


def tabular(header: list[str], rows: list[list[str]], caption: str, label: str) -> str:
    body = "\n".join(" & ".join(row) + r" \\" for row in rows)
    return "\n".join([r"\begin{table}[t]", r"\centering", f"\\caption{{{caption}}}",
                      f"\\label{{{label}}}", f"\\begin{{tabular}}{{l{'r' * (len(header) - 1)}}}",
                      r"\toprule", " & ".join(header) + r" \\", r"\midrule", body,
                      r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])


def gate3_table(summary: dict) -> str:
    """Baseline, proposed, paired difference, 95% CI and the number of paired runs (Task 13)."""
    seeds, metrics = summary["seed_count"], summary["metrics"]
    rows = [[label, f"{m['B0_mean']:.4f}", f"{m['T_mean']:.4f}", f"{m['difference_mean']:+.4f}",
             f"[{m['ci_low']:+.4f}, {m['ci_high']:+.4f}]", str(seeds)]
            for key, label in GATE3_ROWS if (m := metrics.get(key))]
    nll = summary["test_nll"]
    rows.insert(0, ["Per-call NLL (oracle, test)", f"{nll['b0_nll']:.4f}", f"{nll['t_nll']:.4f}",
                    f"{nll['difference']:+.4f}", f"[{nll['ci_low']:+.4f}, {nll['ci_high']:+.4f}]",
                    "individuals"])
    return tabular(["Metric", "B0", "T", r"T $-$ B0", "95\\% CI", "Paired units"], rows,
                   f"Gate 3$'$ on the locked test split, {seeds} paired seeds; "
                   "percentile bootstrap intervals.", "tab:gate3")


def privacy_table(report: dict) -> str:
    arms, diff = report["arms"], report["differences_T_minus_B0"]
    rows = [[label, f"{arms['B0']['summary'][key]['mean']:.4f}",
             f"{arms['T']['summary'][key]['mean']:.4f}", f"{diff[key]['mean']:+.4f}"]
            for key, label in PRIVACY_ROWS if key in diff]
    return tabular(["Metric", "B0", "T", r"T $-$ B0"], rows,
                   f"Gate 4 privacy and duplication, mean over {len(report['seeds'])} seeds; "
                   "empirical measurements, not a differential-privacy guarantee.", "tab:privacy")


def write_tables(multiseed_dir: Path, privacy_dir: Path, output_dir: Path) -> dict[str, Path]:
    summary = json.loads((multiseed_dir / "summary_decoder.json").read_text())
    report = json.loads((privacy_dir / "privacy_report.json").read_text())
    output_dir.mkdir(parents=True, exist_ok=False)
    written = {"gate3.tex": gate3_table(summary), "privacy.tex": privacy_table(report)}
    for name, text in written.items():
        (output_dir / name).write_text(text)
    (output_dir / "sources.json").write_text(json.dumps({
        "multiseed_dir": str(multiseed_dir), "privacy_dir": str(privacy_dir),
        "decoder_sha256": summary["decoder_sha256"], "git_commit": report.get("git_commit")},
        indent=2))
    return {name: output_dir / name for name in written}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--multiseed-dir", type=Path, required=True)
    parser.add_argument("--privacy-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    for path in write_tables(arguments.multiseed_dir, arguments.privacy_dir,
                             arguments.output_dir).values():
        print(path)
