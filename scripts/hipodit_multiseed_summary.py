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
# Plan rev.2 §9.4 Gate 3': one training run per seed, decoded by both arms, on the test split.
DECODER_ARMS: Final = (("B0", ""), ("T", "_T"))
DECODER_METRICS: Final = ("af_mae", "cohort_af_mae", "genotype_proportion_tv",
                          "heterozygosity_mae", "ld_r2_mae", "local_dosage_covariance_mae")
CLASSIFIER_METRICS: Final = ("classifier_accuracy", "classifier_macro_auc")
GATE3_PRIME_CI_METRICS: Final = ("cohort_af_mae", "heterozygosity_mae")
DECODER_SEEDS_REQUIRED: Final = 10
DECODER_FIELDS: Final = ("B0_mean", "T_mean", "difference_mean", "difference_sd",
                         "ci_low", "ci_high")


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


def _declared_runs_all_completed(directory: Path, expected: list[tuple[int, str]]) -> None:
    events = [json.loads(line) for line in (directory / "runs.jsonl").read_text().splitlines()]
    completed = [(event["seed"], event["schedule"]) for event in events
                 if event["status"] == "complete" and event["returncode"] == 0]
    if completed != expected or any(event["status"] == "failed" for event in events):
        raise ExperimentError("Every declared run must finish successfully exactly once in order")


def _validated_run(directory: Path, manifest: dict, seed: int, schedule: str,
                   fisher_schedule) -> tuple[dict, dict, Path, tuple]:
    """One run's frozen contract, plus the alignment key every paired run must share."""
    import yaml

    parameters, prepared = manifest["parameters"], manifest["prepared"]
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
    if feature_schedule is not None and feature_schedule != fisher_schedule:
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
    return report, evaluation, run, aligned


def summarize(directory: Path) -> Summary:
    """Reject incomplete or unpaired artifacts before calculating across-seed statistics."""
    import numpy as np

    verify_fingerprints(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    seeds = manifest["seeds"]
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ExperimentError("At least two distinct seeds are required for sample SD")
    expected = [(seed, schedule) for seed in seeds for schedule in SCHEDULES]
    _declared_runs_all_completed(directory, expected)
    values: dict[tuple[int, str], dict[str, float]] = {}
    raw_reports: list[str] = []
    baseline = None
    fisher_schedule = np.load(Path(manifest["prepared_dir"]) / "feature_schedule.npy").T.tolist()
    paired_reference = None
    for seed, schedule in expected:
        report, evaluation, run, aligned = _validated_run(directory, manifest, seed, schedule,
                                                          fisher_schedule)
        report_path = run / "diagnostic_report.json"
        if baseline is not None and aligned != baseline:
            raise ExperimentError(f"Paired data/configuration/evaluation mismatch: {run}")
        baseline = aligned
        reference = evaluation["real_latent_decoder_reference"]
        if schedule == "fisher" and reference != paired_reference:
            raise ExperimentError(f"Same-seed decoder reference mismatch: {run}")
        paired_reference = reference
        observed: dict[str, float] = {}
        for metric in METRICS:
            value = evaluation["synthetic"][metric]
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


def _flat_arm_metrics(block: dict, scores: dict, run: Path) -> dict[str, float]:
    """One arm's declared Gate 3' metrics, LD bins unrolled and classifier scores beside."""
    flat = {**{metric: block[metric] for metric in DECODER_METRICS},
            **{f"ld_r2_mae_by_bin_{index}": value
               for index, value in enumerate(block["ld_r2_mae_by_bin"])},
            **{metric: scores[metric] for metric in CLASSIFIER_METRICS}}
    for metric, value in flat.items():
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ExperimentError(f"Invalid {metric}: {run}")
    return {metric: float(value) for metric, value in flat.items()}


def _gate3_prime(metrics: dict, nll: dict) -> dict:
    """Plan rev.2 §9.4 Gate 3': the oracle NLL and the two gated end-to-end metrics must each
    improve with a 95% CI upper bound below zero, and overall AF MAE must stay inside 105% of B0.
    LD r-squared is descriptive only (§9.1)."""
    conditions = {"test_nll_improved": nll["difference"] < 0 and nll["ci_high"] < 0}
    for metric in GATE3_PRIME_CI_METRICS:
        estimate = metrics[metric]
        conditions[f"{metric}_improved"] = (estimate["difference_mean"] < 0
                                            and estimate["ci_high"] < 0)
    conditions["af_mae_guardrail"] = (metrics["af_mae"]["T_mean"]
                                      <= 1.05 * metrics["af_mae"]["B0_mean"])
    return {
        "passed": all(conditions.values()), **conditions,
        "numbers": {"test_nll": {key: nll[key] for key in
                                 ("b0_nll", "t_nll", "difference", "ci_low", "ci_high")},
                    **{metric: metrics[metric]
                       for metric in (*GATE3_PRIME_CI_METRICS, "af_mae")}},
        "resampling_unit": {"test_nll": "held-out individuals", "paired_metrics": "seeds"},
        "descriptive_only": {name: metrics[name] for name in metrics
                             if name.startswith("ld_r2_mae")},
    }


def summarize_decoder(directory: Path) -> dict:
    """Plan rev.2 §9.4 Gate 3'. One training run per seed produces both arms from the same
    generated latents, so the pairing is exact at the latent level and the seed is the replicate."""
    import numpy as np

    from scripts.hipodit_genotype_check import (BOOTSTRAP_DRAWS, BOOTSTRAP_SEED, classifier_blocks,
                                                percentile_ci, population_classifier_scores,
                                                test_nll_comparison)

    verify_fingerprints(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("mode") != "decoder":
        raise ExperimentError("summarize_decoder needs a --mode decoder experiment")
    seeds, split = manifest["seeds"], manifest["eval_split"]
    if len(seeds) < DECODER_SEEDS_REQUIRED or len(set(seeds)) != len(seeds):
        raise ExperimentError(f"At least {DECODER_SEEDS_REQUIRED} distinct seeds are required")
    expected = [(seed, "standard") for seed in seeds]
    _declared_runs_all_completed(directory, expected)
    prepared_dir, decoder_dir = Path(manifest["prepared_dir"]), Path(manifest["decoder_dir"])
    blocks = classifier_blocks(prepared_dir, split, manifest["parameters"]["samples"])
    train_block = (blocks["train_calls"], blocks["train_labels"])
    values: dict[tuple[int, str], dict[str, float]] = {}
    raw_reports: list[str] = []
    baseline = None
    for seed, schedule in expected:
        report, evaluation, run, aligned = _validated_run(directory, manifest, seed, schedule, None)
        if "synthetic_T" not in evaluation:
            raise ExperimentError(f"Arm T was never decoded in this run: {run}")
        frozen = (evaluation["decoder_sha256"], report["eval_split"],
                  evaluation["decoder_gate"]["passed"])
        if frozen != (manifest["decoder_sha256"], split, True):
            raise ExperimentError(f"Run did not use the declared Gate 2'-passing decoder: {run}")
        if baseline is not None and aligned != baseline:
            raise ExperimentError(f"Paired data/configuration/evaluation mismatch: {run}")
        baseline = aligned
        labels = json.loads(
            (run / "synthetic_samples" / "generation_meta.json").read_text())["labels"]
        for arm, suffix in DECODER_ARMS:
            calls = np.load(run / f"synthetic_genotypes{suffix}.npy")
            values[seed, arm] = _flat_arm_metrics(
                evaluation[f"synthetic{suffix}"],
                population_classifier_scores(*train_block, calls, labels), run)
        raw_reports.append(str(run / "diagnostic_report.json"))
    names = list(values[seeds[0], "B0"])
    differences = [{"seed": seed, **{name: values[seed, "T"][name] - values[seed, "B0"][name]
                                     for name in names}} for seed in seeds]
    # One resampled seed set per replicate, shared by every metric, so the paired structure holds.
    resampled = np.random.default_rng(BOOTSTRAP_SEED).integers(
        len(seeds), size=(BOOTSTRAP_DRAWS, len(seeds)))
    metrics: dict[str, dict[str, float]] = {}
    for name in names:
        delta = np.array([difference[name] for difference in differences])
        low, high = percentile_ci(delta[resampled].mean(axis=1))
        metrics[name] = {
            "B0_mean": mean(values[seed, "B0"][name] for seed in seeds),
            "T_mean": mean(values[seed, "T"][name] for seed in seeds),
            "difference_mean": mean(delta.tolist()), "difference_sd": stdev(delta.tolist()),
            "ci_low": low, "ci_high": high}
    nll = test_nll_comparison(prepared_dir, decoder_dir, split)
    return {
        "status": "complete", "mode": "decoder", "seed_count": len(seeds), "eval_split": split,
        "decoder_dir": str(decoder_dir), "decoder_sha256": manifest["decoder_sha256"],
        "bootstrap": {"draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED, "interval": "95% percentile"},
        "metrics": metrics, "paired_differences": differences, "test_nll": nll,
        "classifier_real_eval": population_classifier_scores(
            *train_block, blocks["eval_calls"], blocks["eval_labels"]),
        "gate3_prime": _gate3_prime(metrics, nll), "raw_reports": raw_reports,
        "alignment": ["All declared runs complete", "Prepared/source hashes unchanged",
                      "Configs equal except output path",
                      "Labels, normalization, evaluation counts and runtime versions equal",
                      "Same decoder digest, eval split and Gate 2' verdict in every run"],
        "limitation": "One frozen split opened once; seeds are the replicates for the end-to-end "
        "metrics and held-out individuals for the deterministic oracle NLL. LD r-squared is "
        "descriptive, not evidence of linkage.",
    }


def write_summary_decoder(directory: Path, summary: dict) -> None:
    for name in ("summary_decoder.json", "summary_decoder.csv", "paired_differences_decoder.csv"):
        if (directory / name).exists():
            raise FileExistsError(directory / name)
    with (directory / "summary_decoder.csv").open("x", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "seed_count", *DECODER_FIELDS])
        for metric, estimates in summary["metrics"].items():
            writer.writerow([metric, summary["seed_count"],
                             *(estimates[field] for field in DECODER_FIELDS)])
    with (directory / "paired_differences_decoder.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary["paired_differences"][0]))
        writer.writeheader()
        writer.writerows(summary["paired_differences"])
    with (directory / "summary_decoder.json").open("x") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
        handle.write("\n")
