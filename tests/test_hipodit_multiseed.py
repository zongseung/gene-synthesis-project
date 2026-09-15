from __future__ import annotations

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
