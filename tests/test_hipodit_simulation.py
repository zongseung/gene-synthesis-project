"""Plan Task 7 simulation study and Task 14 table generation: the smallest runs that fail if
the wiring breaks."""

from __future__ import annotations

import argparse
import json

import numpy as np


def test_the_simulation_recovers_the_true_tilt_and_reports_every_arm(tmp_path) -> None:
    from scripts.hipodit_simulation import COLUMNS, run

    # Given one pilot scenario at the plan's smallest size, two replicates, one process.
    args = argparse.Namespace(output_dir=tmp_path / "sim", af=[0.2], tau=[1.0], n=[500],
                              latent_rho=[0.3], cohort_ratio=[0.8], replicates=2, snps=6,
                              seed=1, jobs=1, stress=True)
    summary = run(args)

    # Then the primary and the two stress scenarios each carry every arm with no failure,
    # the T arm's tilt estimate is signed like the truth, and the frozen columns are written.
    assert set(summary["arm"]) == {"B0", "B1", "T"} and len(summary) == 9
    assert (summary["failure_rate"] == 0).all()
    primary_t = summary[(summary["arm"] == "T") & ~summary["stress"]].iloc[0]
    assert abs(primary_t["tau_bias_mean"]) < 1.0
    assert np.isnan(summary[summary["arm"] == "B0"]["tau_bias_mean"]).all()
    rows = (tmp_path / "sim" / "replicates.csv").read_text().splitlines()
    assert rows[0].split(",")[: len(COLUMNS)] == list(COLUMNS)
    assert len(rows) == 1 + 9 * 2
    manifest = json.loads((tmp_path / "sim" / "manifest.json").read_text())
    assert manifest["scenarios"] == 3 and "git_commit" in manifest


def test_tables_are_generated_from_the_gate_outputs_only(tmp_path) -> None:
    from scripts.hipodit_tables import write_tables

    estimate = {"B0_mean": 0.02, "T_mean": 0.01, "difference_mean": -0.01,
                "difference_sd": 0.002, "ci_low": -0.012, "ci_high": -0.008}
    (tmp_path / "m").mkdir()
    (tmp_path / "m" / "summary_decoder.json").write_text(json.dumps({
        "seed_count": 10, "decoder_sha256": "ab" * 32,
        "metrics": {"af_mae": estimate, "heterozygosity_mae": estimate},
        "test_nll": {"b0_nll": 0.2, "t_nll": 0.1, "difference": -0.1, "ci_low": -0.11,
                     "ci_high": -0.09}}))
    (tmp_path / "v").mkdir()
    (tmp_path / "v" / "privacy_report.json").write_text(json.dumps({
        "seeds": [1, 2], "git_commit": "deadbeef",
        "arms": {arm: {"summary": {"membership_inference_auc": {"mean": 0.5}}}
                 for arm in ("B0", "T")},
        "differences_T_minus_B0": {"membership_inference_auc": {"mean": 0.0}}}))

    written = write_tables(tmp_path / "m", tmp_path / "v", tmp_path / "tables")

    gate3 = written["gate3.tex"].read_text()
    assert r"\begin{tabular}{lrrrrr}" in gate3 and "AF MAE & 0.0200 & 0.0100 & -0.0100" in gate3
    assert "Per-call NLL" in gate3 and "Cohort AF MAE" not in gate3
    privacy = written["privacy.tex"].read_text()
    assert "Membership inference AUC & 0.5000 & 0.5000 & +0.0000" in privacy
    assert json.loads((tmp_path / "tables" / "sources.json").read_text())["git_commit"] == "deadbeef"
