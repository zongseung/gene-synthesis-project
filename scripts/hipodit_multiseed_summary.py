# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
# ─── How to run ───
# Imported by hipodit_multiseed.py using the existing training environment.
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Final, TypedDict

SCHEDULES: Final = ("standard", "fisher")
METRICS: Final = ("af_mae", "local_dosage_covariance_mae")


class ExperimentError(ValueError):
    def __init__(self, detail: str) -> None:
        self.detail = detail
        super().__init__(detail)


class Estimate(TypedDict):
    mean: float
    sample_sd: float


class MetricSummary(TypedDict):
    standard: Estimate
    fisher: Estimate
    paired_difference: Estimate
    fisher_wins: int
    ties: int


class PairedDifference(TypedDict):
    seed: int
    af_mae: float
    local_dosage_covariance_mae: float


class Summary(TypedDict):
    status: str
    seed_count: int
    metrics: dict[str, MetricSummary]
    paired_differences: list[PairedDifference]
    raw_reports: list[str]
    alignment: list[str]
    limitation: str


def verify_fingerprints(directory: Path) -> None:
    manifest = json.loads((directory / "manifest.json").read_text())
    paths = {**manifest["source_sha256"], **{
        str(Path(manifest["prepared_dir"]) / name): digest
        for name, digest in manifest["prepared_sha256"].items()
    }}
    for name, digest in paths.items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
            raise ExperimentError(f"Frozen input changed: {name}")


def summarize(directory: Path) -> Summary:
    """Reject incomplete or unpaired artifacts before calculating across-seed statistics."""
    import numpy as np
    import yaml

    verify_fingerprints(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    seeds = manifest["seeds"]
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ExperimentError("At least two distinct seeds are required for sample SD")
    parameters, prepared = manifest["parameters"], manifest["prepared"]
    expected = [(seed, schedule) for seed in seeds for schedule in SCHEDULES]
    events = [json.loads(line) for line in (directory / "runs.jsonl").read_text().splitlines()]
    completed = [(event["seed"], event["schedule"]) for event in events
                 if event["status"] == "complete" and event["returncode"] == 0]
    if completed != expected or any(event["status"] == "failed" for event in events):
        raise ExperimentError("Every declared run must finish successfully exactly once in order")
    values: dict[tuple[int, str], dict[str, float]] = {}
    raw_reports: list[str] = []
    baseline = None
    fisher_schedule = np.load(Path(manifest["prepared_dir"]) / "feature_schedule.npy").T.tolist()
    paired_reference = None
    for seed, schedule in expected:
        run = directory / f"seed_{seed}" / schedule
        report_path = run / "diagnostic_report.json"
        report = json.loads(report_path.read_text())
        required = {"status": "complete", "seed": seed, "schedule": schedule,
                    "steps": parameters["steps"], "ddim_steps": parameters["ddim_steps"],
                    "device": parameters["device"], "samples_generated": parameters["samples"],
                    "data_split": prepared["split_sizes"]}
        if any(report[key] != value for key, value in required.items()):
            raise ExperimentError(f"Report disagrees with declared experiment: {report_path}")
        if not report["checks"] or not all(value is True for value in report["checks"].values()):
            raise ExperimentError(f"Diagnostic checks failed: {report_path}")
        config = yaml.safe_load((run / "diagnostic_config.yaml").read_text())
        if (config.pop("save_dir") != str(run)
                or config["data"]["processed_dir"] != manifest["prepared_dir"]
                or config["training"]["batch_size"] != parameters["batch_size"]
                or config["diffusion"]["sampling_timesteps"] != parameters["ddim_steps"]):
            raise ExperimentError(f"Training configuration mismatch: {run}")
        feature_schedule = config["diffusion"].pop("feature_schedule")
        if (schedule == "standard") != (feature_schedule is None):
            raise ExperimentError(f"Wrong schedule configuration: {run}")
        if feature_schedule is not None:
            if feature_schedule != fisher_schedule:
                raise ExperimentError(f"Fisher feature schedule differs from prepared array: {run}")
        metadata = json.loads((run / "synthetic_samples" / "generation_meta.json").read_text())
        if (metadata["sample_space"] != "original"
                or len(metadata["labels"]) != parameters["samples"]
                or metadata["stats_fingerprint"] != prepared["normalization_fingerprint"]
                or metadata["stats_path"] != str(Path(manifest["prepared_dir"]) / "normalization_stats.pkl")):
            raise ExperimentError(f"Generation metadata mismatch: {run}")
        evaluation = report["genotype_evaluation"]
        synthetic = evaluation["synthetic"]
        if (evaluation["samples"] != parameters["samples"] or evaluation["variants"] < 1
                or synthetic["local_pairs_evaluated"] < 1 or synthetic["valid_genotype_fraction"] != 1):
            raise ExperimentError(f"Invalid genotype evaluation: {run}")
        aligned = (config, metadata, evaluation["variants"], synthetic["local_pairs_evaluated"],
                   report["torch_version"], report["cuda_version"])
        if baseline is not None and aligned != baseline:
            raise ExperimentError(f"Paired data/configuration/evaluation mismatch: {run}")
        baseline = aligned
        reference = evaluation["real_latent_decoder_reference"]
        if schedule == "fisher" and reference != paired_reference:
            raise ExperimentError(f"Same-seed decoder reference mismatch: {run}")
        paired_reference = reference
        observed: dict[str, float] = {}
        for metric in METRICS:
            value = synthetic[metric]
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ExperimentError(f"Invalid {metric}: {report_path}")
            observed[metric] = float(value)
        values[seed, schedule] = observed
        raw_reports.append(str(report_path))
    differences: list[PairedDifference] = [
        {"seed": seed,
         "af_mae": values[seed, "fisher"]["af_mae"] - values[seed, "standard"]["af_mae"],
         "local_dosage_covariance_mae": values[seed, "fisher"]["local_dosage_covariance_mae"]
         - values[seed, "standard"]["local_dosage_covariance_mae"]}
        for seed in seeds
    ]
    metrics: dict[str, MetricSummary] = {}
    for metric in METRICS:
        standard = [values[seed, "standard"][metric] for seed in seeds]
        fisher = [values[seed, "fisher"][metric] for seed in seeds]
        delta = [f - s for f, s in zip(fisher, standard)]
        metrics[metric] = {
            "standard": {"mean": mean(standard), "sample_sd": stdev(standard)},
            "fisher": {"mean": mean(fisher), "sample_sd": stdev(fisher)},
            "paired_difference": {"mean": mean(delta), "sample_sd": stdev(delta)},
            "fisher_wins": sum(value < 0 for value in delta), "ties": delta.count(0),
        }
    return {"status": "complete", "seed_count": len(seeds), "metrics": metrics,
            "paired_differences": differences, "raw_reports": raw_reports,
            "alignment": ["All declared runs complete", "Prepared/source hashes unchanged",
                          "Configs equal except output path and feature schedule",
                          "Labels, normalization, evaluation counts and runtime versions equal",
                          "Within-seed real-latent-decoder references equal"],
            "limitation": "Descriptive across-seed statistics on one frozen split; seeds are the "
            "replicates. Fisher-minus-standard < 0 is better. No significance claim. "
            "Fixed-t noise probes are not compared across schedules."}


def write_summary(directory: Path, summary: Summary) -> None:
    for name in ("summary.json", "summary.csv", "paired_differences.csv"):
        if (directory / name).exists():
            raise FileExistsError(directory / name)
    with (directory / "summary.csv").open("x", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "arm", "seed_count", "mean", "sample_sd", "fisher_wins", "ties"])
        for metric, estimates in summary["metrics"].items():
            for arm in ("standard", "fisher", "paired_difference"):
                estimate = estimates[arm]
                writer.writerow([metric, arm, summary["seed_count"], estimate["mean"],
                                 estimate["sample_sd"], estimates["fisher_wins"], estimates["ties"]])
    with (directory / "paired_differences.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["seed", *METRICS])
        writer.writeheader()
        writer.writerows(summary["paired_differences"])
    with (directory / "summary.json").open("x") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
        handle.write("\n")
