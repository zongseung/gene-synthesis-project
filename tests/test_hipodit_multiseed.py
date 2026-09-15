from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest
import numpy as np
import yaml


def completed_experiment(root: Path) -> Path:
    prepared = root / "prepared"
    prepared.mkdir()
    np.save(prepared / "feature_schedule.npy", np.array([[0, 1, 2, 0]] * 8))
    output = root / "experiment"
    output.mkdir()
    manifest = {
        "prepared_dir": str(prepared), "seeds": [11, 12],
        "parameters": {"steps": 1000, "batch_size": 64, "samples": 2,
                       "ddim_steps": 100, "device": "cpu"},
        "prepared_sha256": {}, "source_sha256": {},
        "prepared": {"split_sizes": {"train": 4, "validation": 2, "test": 2},
                     "normalization_fingerprint": "stats", "genes": 8, "components": 4},
    }
    (output / "manifest.json").write_text(json.dumps(manifest))
    for seed, schedule, af, covariance in (
        (11, "standard", 0.1, 0.2), (11, "fisher", 0.05, 0.3),
        (12, "standard", 0.3, 0.4), (12, "fisher", 0.4, 0.2),
    ):
        run = output / f"seed_{seed}" / schedule
        run.mkdir(parents=True)
        report = {
            "status": "complete", "seed": seed, "schedule": schedule, "device": "cpu",
            "steps": 1000, "ddim_steps": 100, "samples_generated": 2,
            "data_split": manifest["prepared"]["split_sizes"],
            "checks": {"finite_outputs": True}, "torch_version": "test", "cuda_version": None,
            "genotype_evaluation": {
                "samples": 2, "variants": 361,
                "synthetic": {"af_mae": af, "local_dosage_covariance_mae": covariance,
                              "local_pairs_evaluated": 100, "valid_genotype_fraction": 1.0},
                "real_latent_decoder_reference": {"af_mae": seed / 1000},
            },
        }
        (run / "diagnostic_report.json").write_text(json.dumps(report))
        config = {
            "save_dir": str(run), "model": {"name": "unchanged"},
            "data": {"processed_dir": str(prepared), "normalization_stats_sha256": "stats"},
            "training": {"batch_size": 64},
            "diffusion": {"sampling_timesteps": 100,
                          "feature_schedule": [[0] * 8, [1] * 8, [2] * 8, [0] * 8]
                          if schedule == "fisher" else None},
        }
        (run / "diagnostic_config.yaml").write_text(yaml.safe_dump(config))
        (run / "synthetic_samples").mkdir()
        (run / "synthetic_samples" / "generation_meta.json").write_text(json.dumps({
            "sample_space": "original", "stats_path": str(prepared / "normalization_stats.pkl"),
            "stats_fingerprint": "stats", "labels": [2, 3],
        }))
        with (output / "runs.jsonl").open("a") as handle:
            handle.write(json.dumps({"seed": seed, "schedule": schedule, "status": "complete",
                                     "returncode": 0}) + "\n")
    return output


def test_paired_summary_when_all_runs_complete(tmp_path: Path) -> None:
    # Given independently hand-calculated two-seed outcomes.
    output = completed_experiment(tmp_path)
    assert importlib.util.find_spec("scripts.hipodit_multiseed_summary") is not None
    from scripts.hipodit_multiseed_summary import summarize

    # When the completed experiment is aggregated.
    result = summarize(output)

    # Then observations are seeds, with sample SD and Fisher-minus-standard signs.
    af = result["metrics"]["af_mae"]
    assert af["standard"]["mean"] == pytest.approx(0.2)
    assert af["standard"]["sample_sd"] == pytest.approx(math.sqrt(0.02))
    assert af["fisher"]["mean"] == pytest.approx(0.225)
    assert af["paired_difference"]["mean"] == pytest.approx(0.025)
    assert af["paired_difference"]["sample_sd"] == pytest.approx(math.sqrt(0.01125))
    assert af["fisher_wins"] == 1
    assert result["paired_differences"][0]["af_mae"] == pytest.approx(-0.05)
    covariance = result["metrics"]["local_dosage_covariance_mae"]
    assert covariance["paired_difference"]["mean"] == pytest.approx(-0.05)
    assert covariance["fisher_wins"] == 1
    assert len(result["raw_reports"]) == 4


@pytest.mark.parametrize("damage", ["missing", "failed", "seed", "reference", "labels", "config", "nan"])
def test_summary_rejects_incomplete_or_misaligned_runs(tmp_path: Path, damage: str) -> None:
    # Given one broken required run.
    output = completed_experiment(tmp_path)
    assert importlib.util.find_spec("scripts.hipodit_multiseed_summary") is not None
    from scripts.hipodit_multiseed_summary import ExperimentError, summarize

    run = output / "seed_12" / "fisher"
    path = run / "diagnostic_report.json"
    report = json.loads(path.read_text())
    if damage == "missing":
        path.unlink()
    elif damage == "labels":
        meta = run / "synthetic_samples" / "generation_meta.json"
        value = json.loads(meta.read_text())
        value["labels"] = [3, 2]
        meta.write_text(json.dumps(value))
    elif damage == "config":
        config_path = run / "diagnostic_config.yaml"
        config = yaml.safe_load(config_path.read_text())
        config["model"]["name"] = "different"
        config_path.write_text(yaml.safe_dump(config))
    else:
        if damage == "failed":
            report["status"] = "failed"
        elif damage == "seed":
            report["seed"] = 99
        elif damage == "reference":
            report["genotype_evaluation"]["real_latent_decoder_reference"]["af_mae"] = 0.9
        elif damage == "nan":
            report["genotype_evaluation"]["synthetic"]["af_mae"] = float("nan")
        path.write_text(json.dumps(report))

    # When/Then a required run cannot be silently excluded or compared unpaired.
    with pytest.raises((ExperimentError, FileNotFoundError)):
        summarize(output)


def test_driver_records_manifest_before_failed_child(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Given prepared data and a child process that cannot start training successfully.
    assert importlib.util.find_spec("scripts.hipodit_multiseed") is not None
    from scripts.hipodit_multiseed import build_parser, run_experiment

    prepared = tmp_path / "prepared"
    prepared.mkdir()
    (prepared / "prepare_report.json").write_text(json.dumps({
        "status": "prepared", "glm_family": "binomial", "fit_split": "train",
        "split_sizes": {"validation": 251},
    }))
    output = tmp_path / "runs"
    args = build_parser().parse_args(["--prepared-dir", str(prepared), "--output-dir", str(output)])

    def failed_child(command: list[str], **kwargs: str) -> subprocess.CompletedProcess[str]:
        manifest = json.loads((output / "manifest.json").read_text())
        assert manifest["seeds"] == [20260327, 20260328, 20260329, 20260330, 20260331]
        assert manifest["parameters"]["samples"] == 251
        assert "prepare_report.json" in manifest["prepared_sha256"]
        assert command[0] == sys.executable
        return subprocess.CompletedProcess(command, 17)

    monkeypatch.setattr(subprocess, "run", failed_child)
    # When the first declared child fails.
    with pytest.raises(subprocess.CalledProcessError):
        run_experiment(args)
    # Then the failure remains visible, no summary or selective continuation is produced.
    events = [json.loads(line) for line in (output / "runs.jsonl").read_text().splitlines()]
    assert [event["status"] for event in events] == ["started", "failed"]
    assert events[-1]["returncode"] == 17
    assert not (output / "summary.json").exists()


def test_cli_refuses_overwrite(tmp_path: Path) -> None:
    # Given an existing output directory containing a user artifact.
    script = Path(__file__).parents[1] / "scripts" / "hipodit_multiseed.py"
    marker = tmp_path / "keep.txt"
    marker.write_text("keep")
    # When the CLI targets that directory.
    result = subprocess.run([sys.executable, str(script), "--output-dir", str(tmp_path)],
                            capture_output=True, text=True, check=False)
    # Then the output refusal preserves the artifact.
    assert result.returncode != 0
    assert "FileExistsError" in result.stderr
    assert marker.read_text() == "keep"


# ── Plan rev.2 §9.4 Gate 3': one training run per seed, both decoders on the same latents ──
DECODER_SEEDS = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
# Arm B0 is flat; arm T beats it on both gated metrics, by two different cohort-AF margins.
ARM_VALUES = {"": {"af_mae": 0.02, "cohort_af_mae": 0.05, "heterozygosity_mae": 0.10},
              "_T": {"af_mae": 0.0201, "heterozygosity_mae": 0.08}}


def _arm_block(values: dict) -> dict:
    return {**values, "genotype_proportion_tv": 0.3, "ld_r2_mae": 0.06,
            "ld_r2_mae_by_bin": [0.1, 0.2, 0.3], "ld_pairs_by_bin": [4, 4, 3],
            "local_dosage_covariance_mae": 0.01, "local_pairs_evaluated": 5,
            "valid_genotype_fraction": 1.0, "unique_individual_fraction": 1.0}


def completed_decoder_experiment(root: Path, seeds: list[int] = DECODER_SEEDS) -> Path:
    """A `--mode decoder` experiment over the genotype-check panel, one standard run per seed."""
    # The prepared panel and oracle fixtures are shared with the decoder's own test module.
    from test_hipodit_genotype_check import _fake_decoder_dir, _fake_prepared

    prepared, decoder_dir = root / "prepared", root / "oracle"
    _fake_prepared(prepared)
    with np.load(prepared / "dataset.npz") as data:
        arrays = dict(data)
    arrays["y"] = np.tile([3, 7, 11, 25], 4)  # every split of four then carries all four cohorts
    np.savez(prepared / "dataset.npz", **arrays)
    _fake_decoder_dir(prepared, decoder_dir)
    digest = hashlib.sha256((decoder_dir / "decoder_T.npz").read_bytes()).hexdigest()
    output = root / "experiment"
    output.mkdir()
    manifest = {
        "mode": "decoder", "prepared_dir": str(prepared), "seeds": list(seeds),
        "decoder_dir": str(decoder_dir), "decoder_sha256": digest, "eval_split": "test",
        "gate": "gate3_prime", "plan_revision": "rev2-§9",
        "parameters": {"steps": 1000, "batch_size": 64, "samples": 4,
                       "ddim_steps": 100, "device": "cpu"},
        "prepared_sha256": {}, "source_sha256": {},
        "prepared": {"split_sizes": {"train": 8, "validation": 4, "test": 4},
                     "normalization_fingerprint": "stats", "genes": 2, "components": 2},
    }
    (output / "manifest.json").write_text(json.dumps(manifest))
    rng = np.random.default_rng(7)
    for index, seed in enumerate(seeds):
        run = output / f"seed_{seed}" / "standard"
        run.mkdir(parents=True)
        arms = {"": _arm_block(ARM_VALUES[""]),
                "_T": _arm_block({**ARM_VALUES["_T"],
                                  "cohort_af_mae": 0.04 if index < len(seeds) // 2 else 0.03})}
        report = {
            "status": "complete", "seed": seed, "schedule": "standard", "device": "cpu",
            "steps": 1000, "ddim_steps": 100, "samples_generated": 4, "eval_split": "test",
            "data_split": manifest["prepared"]["split_sizes"], "decoder_dir": str(decoder_dir),
            "checks": {"finite_outputs": True}, "torch_version": "test", "cuda_version": None,
            "genotype_evaluation": {
                "samples": 4, "variants": 8, "synthetic": arms[""], "synthetic_T": arms["_T"],
                "real_latent_decoder_reference": {"af_mae": 0.011},
                "decoder_sha256": digest, "decoder_gate": {"passed": True}},
        }
        (run / "diagnostic_report.json").write_text(json.dumps(report))
        (run / "diagnostic_config.yaml").write_text(yaml.safe_dump({
            "save_dir": str(run), "model": {"name": "unchanged"},
            "data": {"processed_dir": str(prepared), "normalization_stats_sha256": "stats"},
            "training": {"batch_size": 64},
            "diffusion": {"sampling_timesteps": 100, "feature_schedule": None}}))
        (run / "synthetic_samples").mkdir()
        (run / "synthetic_samples" / "generation_meta.json").write_text(json.dumps({
            "sample_space": "original", "stats_path": str(prepared / "normalization_stats.pkl"),
            "stats_fingerprint": "stats", "labels": [3, 7, 11, 25]}))
        for suffix in ("", "_T"):
            np.save(run / f"synthetic_genotypes{suffix}.npy",
                    rng.integers(0, 3, size=(4, 8)).astype(np.int8))
        with (output / "runs.jsonl").open("a") as handle:
            handle.write(json.dumps({"seed": seed, "schedule": "standard",
                                     "status": "complete", "returncode": 0}) + "\n")
    return output


def test_decoder_summary_pairs_both_arms_of_every_seed(tmp_path: Path) -> None:
    # Given ten paired runs whose arm differences were chosen by hand.
    from scripts.hipodit_multiseed_summary import summarize_decoder, write_summary_decoder

    output = completed_decoder_experiment(tmp_path)

    # When the decoder-mode experiment is aggregated.
    result = summarize_decoder(output)

    # Then every seed contributes one T-minus-B0 difference per declared metric.
    assert result["seed_count"] == 10
    assert [row["seed"] for row in result["paired_differences"]] == DECODER_SEEDS
    heterozygosity = result["metrics"]["heterozygosity_mae"]
    assert heterozygosity["B0_mean"] == pytest.approx(0.10)
    assert heterozygosity["T_mean"] == pytest.approx(0.08)
    # An identical difference in every seed leaves the seed-resampled interval a point.
    assert heterozygosity["difference_mean"] == pytest.approx(-0.02)
    assert heterozygosity["difference_sd"] == pytest.approx(0.0)
    assert (heterozygosity["ci_low"], heterozygosity["ci_high"]) == pytest.approx((-0.02, -0.02))
    # Five seeds at -0.01 and five at -0.02 give a hand-computable mean and sample SD.
    cohort = result["metrics"]["cohort_af_mae"]
    assert cohort["difference_mean"] == pytest.approx(-0.015)
    assert cohort["difference_sd"] == pytest.approx(math.sqrt(10 * 0.005**2 / 9))
    assert -0.02 <= cohort["ci_low"] < cohort["ci_high"] <= -0.01
    # The LD bins are carried per bin, descriptively.
    assert result["metrics"]["ld_r2_mae_by_bin_2"]["B0_mean"] == pytest.approx(0.3)
    assert set(result["gate3_prime"]["descriptive_only"]) == {
        "ld_r2_mae", *(f"ld_r2_mae_by_bin_{index}" for index in range(3))}
    # The classifier is scored on the real eval block and on both synthetic arms of every seed.
    assert 0.0 <= result["classifier_real_eval"]["classifier_accuracy"] <= 1.0
    for metric in ("classifier_accuracy", "classifier_macro_auc"):
        assert metric in result["metrics"]
    # The deterministic oracle NLL resamples individuals, not seeds.
    assert result["test_nll"]["n_individuals"] == 4
    assert result["test_nll"]["resampling_unit"] == "held-out individuals"
    # And the summary is written beside, not over, the schedule-mode products.
    write_summary_decoder(output, result)
    for name in ("summary_decoder.json", "summary_decoder.csv", "paired_differences_decoder.csv"):
        assert (output / name).is_file()
    assert not (output / "summary.json").exists()
    rows = (output / "summary_decoder.csv").read_text().splitlines()
    assert rows[0].split(",")[1:] == ["seed_count", "B0_mean", "T_mean", "difference_mean",
                                      "difference_sd", "ci_low", "ci_high"]
    assert summarize_decoder(output) == result  # the seeded bootstrap is reproducible


def test_decoder_summary_rejects_a_run_that_never_decoded_arm_t(tmp_path: Path) -> None:
    # Given one run whose evaluation block carries the baseline arm only.
    from scripts.hipodit_multiseed_summary import ExperimentError, summarize_decoder

    output = completed_decoder_experiment(tmp_path)
    path = output / "seed_15" / "standard" / "diagnostic_report.json"
    report = json.loads(path.read_text())
    del report["genotype_evaluation"]["synthetic_T"]
    path.write_text(json.dumps(report))

    # When/Then the unpaired seed is refused rather than dropped from the comparison.
    with pytest.raises(ExperimentError):
        summarize_decoder(output)


@pytest.mark.parametrize("damage", ["decoder_sha256", "eval_split", "decoder_gate", "mode"])
def test_decoder_summary_rejects_runs_that_disagree_on_the_frozen_decoder(
    tmp_path: Path, damage: str,
) -> None:
    # Given one run made with a different decoder, split or gate verdict than the rest.
    from scripts.hipodit_multiseed_summary import ExperimentError, summarize_decoder

    output = completed_decoder_experiment(tmp_path)
    if damage == "mode":
        manifest = json.loads((output / "manifest.json").read_text())
        manifest["mode"] = "schedule"
        (output / "manifest.json").write_text(json.dumps(manifest))
    else:
        path = output / "seed_20" / "standard" / "diagnostic_report.json"
        report = json.loads(path.read_text())
        if damage == "decoder_sha256":
            report["genotype_evaluation"]["decoder_sha256"] = "0" * 64
        elif damage == "eval_split":
            report["eval_split"] = "val"
        else:
            report["genotype_evaluation"]["decoder_gate"]["passed"] = False
        path.write_text(json.dumps(report))

    # When/Then the mixed comparison is refused.
    with pytest.raises(ExperimentError):
        summarize_decoder(output)


def test_decoder_summary_needs_ten_seeds(tmp_path: Path) -> None:
    # Given a decoder-mode experiment declared with fewer than the pre-registered ten seeds.
    from scripts.hipodit_multiseed_summary import ExperimentError, summarize_decoder

    output = completed_decoder_experiment(tmp_path, seeds=DECODER_SEEDS[:9])

    # When/Then the under-powered paired comparison is refused.
    with pytest.raises(ExperimentError):
        summarize_decoder(output)


def _gate_metrics(**overrides: float) -> dict:
    """A passing Gate 3' metric table, before the named numbers are moved."""
    metrics = {
        "cohort_af_mae": {"difference_mean": -0.015, "ci_low": -0.02, "ci_high": -0.01,
                          "B0_mean": 0.05, "T_mean": 0.035, "difference_sd": 0.005},
        "heterozygosity_mae": {"difference_mean": -0.02, "ci_low": -0.03, "ci_high": -0.005,
                               "B0_mean": 0.10, "T_mean": 0.08, "difference_sd": 0.004},
        "af_mae": {"difference_mean": 0.0001, "ci_low": -0.001, "ci_high": 0.002,
                   "B0_mean": 0.02, "T_mean": 0.0201, "difference_sd": 0.001},
        "ld_r2_mae": {"difference_mean": -0.01, "ci_low": -0.02, "ci_high": 0.5,
                      "B0_mean": 0.06, "T_mean": 0.05, "difference_sd": 0.01},
    }
    for name, value in overrides.items():
        metric, _, field = name.rpartition("__")
        metrics[metric][field] = value
    return metrics


@pytest.mark.parametrize(("nll", "overrides", "failing"), [
    ({"difference": -0.09, "ci_high": -0.08}, {}, None),
    ({"difference": -0.09, "ci_high": 0.01}, {}, "test_nll_improved"),
    ({"difference": 0.09, "ci_high": -0.08}, {}, "test_nll_improved"),
    ({"difference": -0.09, "ci_high": -0.08}, {"cohort_af_mae__ci_high": 0.0},
     "cohort_af_mae_improved"),
    ({"difference": -0.09, "ci_high": -0.08}, {"heterozygosity_mae__difference_mean": 0.01},
     "heterozygosity_mae_improved"),
    ({"difference": -0.09, "ci_high": -0.08}, {"af_mae__T_mean": 0.0211}, "af_mae_guardrail"),
    ({"difference": -0.09, "ci_high": -0.08}, {"af_mae__T_mean": 0.021}, None),
])
def test_gate3_prime_needs_three_intervals_below_zero_inside_the_af_guardrail(
    nll: dict, overrides: dict, failing: str | None,
) -> None:
    # Given a Gate 3' table with at most one pre-registered condition moved off its boundary.
    from scripts.hipodit_multiseed_summary import _gate3_prime

    verdict = _gate3_prime(_gate_metrics(**overrides),
                           {"b0_nll": 0.2, "t_nll": 0.11, "ci_low": -0.1, **nll})

    # Then exactly the condition that was moved fails, and it alone decides the verdict.
    assert verdict["passed"] is (failing is None)
    assert [name for name in ("test_nll_improved", "cohort_af_mae_improved",
                              "heterozygosity_mae_improved", "af_mae_guardrail")
            if not verdict[name]] == ([] if failing is None else [failing])
    assert verdict["resampling_unit"]["test_nll"] == "held-out individuals"


def test_decoder_mode_declares_the_decoder_split_and_ten_seeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given a decoder-mode declaration over a prepared panel with a held-out test split.
    from scripts.hipodit_multiseed import build_parser, run_experiment
    from scripts.hipodit_multiseed_summary import ExperimentError

    prepared, decoder_dir = tmp_path / "prepared", tmp_path / "oracle"
    prepared.mkdir()
    (prepared / "prepare_report.json").write_text(json.dumps({
        "status": "prepared", "glm_family": "binomial", "fit_split": "train",
        "split_sizes": {"validation": 251, "test": 251}}))
    decoder_dir.mkdir()
    (decoder_dir / "decoder_T.npz").write_bytes(b"decoder")
    common = ["--mode", "decoder", "--prepared-dir", str(prepared),
              "--decoder-dir", str(decoder_dir), "--eval-split", "test", "--samples", "251"]

    # When fewer than the pre-registered ten seeds are declared, the run never starts.
    with pytest.raises(ExperimentError, match="seeds"):
        run_experiment(build_parser().parse_args(
            [*common, "--output-dir", str(tmp_path / "few"), "--seeds", *"123456789"]))
    assert not (tmp_path / "few").exists()

    # When ten seeds are declared, each child is a single training run carrying both arms.
    output = tmp_path / "runs"
    commands: list[list[str]] = []

    def record(command: list[str], **kwargs: str) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        return subprocess.CompletedProcess(command, 9)

    monkeypatch.setattr(subprocess, "run", record)
    with pytest.raises(subprocess.CalledProcessError):
        run_experiment(build_parser().parse_args(
            [*common, "--output-dir", str(output), "--seeds", *map(str, range(10))]))

    # Then the manifest freezes the decoder, its digest, the split and the gate it answers.
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["mode"] == "decoder"
    assert manifest["eval_split"] == "test"
    assert manifest["gate"] == "gate3_prime"
    assert manifest["plan_revision"] == "rev2-§9"
    assert manifest["decoder_dir"] == str(decoder_dir)
    assert manifest["decoder_sha256"] == hashlib.sha256(b"decoder").hexdigest()
    assert [run["seed"] for run in manifest["runs"]] == list(range(10))
    assert {run["schedule"] for run in manifest["runs"]} == {"standard"}
    assert commands[0][commands[0].index("--decoder-dir") + 1] == str(decoder_dir)
    assert commands[0][commands[0].index("--eval-split") + 1] == "test"
    assert len(commands) == 1  # the first failure stops the experiment


def test_decoder_mode_needs_a_decoder_directory(tmp_path: Path) -> None:
    # Given a decoder-mode declaration with no decoder to compare against.
    from scripts.hipodit_multiseed import build_parser, run_experiment
    from scripts.hipodit_multiseed_summary import ExperimentError

    prepared = tmp_path / "prepared"
    prepared.mkdir()
    (prepared / "prepare_report.json").write_text(json.dumps({
        "status": "prepared", "glm_family": "binomial", "fit_split": "train",
        "split_sizes": {"validation": 251, "test": 251}}))
    # When/Then the declaration is refused before any directory is created.
    with pytest.raises(ExperimentError, match="decoder"):
        run_experiment(build_parser().parse_args(
            ["--mode", "decoder", "--prepared-dir", str(prepared),
             "--output-dir", str(tmp_path / "runs"), "--seeds", *map(str, range(10))]))
