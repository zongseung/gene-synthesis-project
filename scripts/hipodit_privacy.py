#!/usr/bin/env python3
# ─── How to run ───
#   python scripts/hipodit_privacy.py --prepared-dir P --multiseed-dir M --output-dir O
# Plan §5 Phase 4 and Gate 4 (rev.2 §9.4): privacy, duplication and strata for the frozen B0 and T
# genotype arms of a finished Gate 3' run. Reads saved genotypes only; nothing is regenerated.
from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import rankdata

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.hipodit_genotype_check import (  # noqa: E402
    MAF_BIN_EDGES,
    N_COHORTS,
    _atomic_json,
    _digest,
    _variant_values,
    _write_csv,
    load_superpop_of_cohort,
    provenance,
)

# Reported arm name -> the genotype file the Gate 3' run saved for it.
ARMS = {"B0": "synthetic_genotypes.npy", "T": "synthetic_genotypes_T.npy"}
# The frozen prepared files this script consumes, hashed into the manifest.
PREPARED_FILES = ("dataset.npz", "gene_variant_map.json", "genotypes.npz", "label_hierarchy.pkl",
                  "normalization_stats.pkl")
MAF_BIN_LABELS = ("[0.01,0.05)", "[0.05,0.2)", "[0.2,0.5]")
DISTANCE_STATS = ("min", "p1", "p5", "median", "mean")
SCALAR_METRICS = ("exact_duplicate_rate_train", "exact_duplicate_rate_test",
                  "near_duplicate_rate_train", "self_duplicate_rate", "membership_inference_auc")
# Gate 4 is qualitative in plan §5 ("materially worse than B0") and rev.2 §9.4 adopts it verbatim,
# so these four constants are NOT pre-registered: they were fixed in the task brief before the
# measurement, as one operationalisation of that qualitative rule.
AUC_MARGIN = 0.02
NEAR_DUPLICATE_FACTOR = 1.5
NEAR_DUPLICATE_FLOOR = 0.01
P1_FACTOR = 0.8
GATE4_CONDITIONS = ("membership_inference_auc_margin", "exact_duplicate_guardrail",
                    "near_duplicate_guardrail", "nn_distance_p1_guardrail")
GATE4_ACTION = ("Privacy or duplication is materially worse for T: revisit sampling and "
                "regularization regardless of the utility results (plan §5 Gate 4).")
LIMITATION = ("Empirical attack and duplication measurements on one frozen split; no formal "
              "differential privacy is claimed, because training carried no privacy mechanism or "
              "budget. Synthetic rows were conditioned on the eval split's cohort labels, so the "
              "non-member set shares that composition while the member set is the whole train "
              "panel.")
NOT_ESTIMABLE = "not estimable"
COHORT_METRICS = ("af_mae", "het_mae", "nn_distance_median")
COHORT_COLUMNS = ("cohort", "population", "superpop", "n_train", "n_eval", "af_mae_B0", "af_mae_T",
                  "het_mae_B0", "het_mae_T", "nn_distance_median_B0", "nn_distance_median_T",
                  "status")
MAF_COLUMNS = ("maf_bin", "n_snps", "af_mae_B0", "af_mae_T", "het_mae_B0", "het_mae_T")
SUMMARY_COLUMNS = ("arm", "metric", "mean", "sample_sd")


def mean_abs_dosage_distance(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Mean absolute dosage difference per SNP, in 0..2, for every pair of individuals."""
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    return cdist(left, right, "cityblock") / left.shape[1]


def distance_stats(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    return {"min": float(values.min()), "p1": float(np.percentile(values, 1)),
            "p5": float(np.percentile(values, 5)), "median": float(np.median(values)),
            "mean": float(values.mean())}


def membership_inference_auc(member: np.ndarray, nonmember: np.ndarray) -> float:
    """AUC of the threshold attack that calls an individual a member when its minimum distance to
    the synthetic set is SMALL.

    The score is therefore the negated distance: the AUC is P(d_member < d_nonmember) plus half the
    ties, which is 1 - AUC(distance). 1.0 is perfect leakage, 0.5 is chance, and a sign flip would
    show up as values below 0.5 rather than above it.
    """
    member, nonmember = np.asarray(member, dtype=np.float64), np.asarray(nonmember, np.float64)
    n_member, n_nonmember = len(member), len(nonmember)
    # Ascending ranks: the smallest distance ranks first, and ties share an average rank (0.5 each).
    ranks = rankdata(np.concatenate([member, nonmember]))
    below = n_member * n_nonmember + n_member * (n_member + 1) / 2 - ranks[:n_member].sum()
    return float(below / (n_member * n_nonmember))


def arm_privacy(synthetic: np.ndarray, train: np.ndarray, test: np.ndarray,
                near_threshold: float) -> tuple[dict, np.ndarray]:
    """The Phase 4 privacy measurements for one arm, plus the per-individual nearest-train
    distances the strata table medians."""
    to_train = mean_abs_dosage_distance(synthetic, train)
    to_test = mean_abs_dosage_distance(synthetic, test)
    nearest = to_train.min(axis=1)
    metrics = {
        "nn_distance_to_train": distance_stats(nearest),
        "exact_duplicate_rate_train": float((nearest == 0).mean()),
        "exact_duplicate_rate_test": float((to_test.min(axis=1) == 0).mean()),
        "near_duplicate_rate_train": float((nearest < near_threshold).mean()),
        "self_duplicate_rate": 1.0 - len(np.unique(synthetic, axis=0)) / len(synthetic),
        # Members are the real train rows, non-members the real test rows; each is scored by its
        # own minimum distance to this arm's synthetic set.
        "membership_inference_auc": membership_inference_auc(to_train.min(axis=0),
                                                             to_test.min(axis=0)),
    }
    return metrics, nearest


def flatten(metrics: dict) -> dict:
    """One seed's metrics as a flat mean/SD-able mapping."""
    return {f"nn_distance_to_train_{stat}": metrics["nn_distance_to_train"][stat]
            for stat in DISTANCE_STATS} | {name: metrics[name] for name in SCALAR_METRICS}


def summarize(per_seed: list[dict]) -> dict:
    """Mean and sample SD across seeds, leaf by leaf."""
    return {key: {"mean": float(np.mean([seed[key] for seed in per_seed])),
                  "sample_sd": float(np.std([seed[key] for seed in per_seed], ddof=1))}
            for key in per_seed[0]}


def gate4(b0: dict, t: dict, n_synthetic: int) -> dict:
    """Plan §5 Gate 4: T must not be materially worse than B0 on privacy.

    Each brief condition is written as "T against a limit derived from B0" — algebraically the same
    as the brief's differences, and exact on the boundary instead of one ulp off it. Only the p1
    condition wants T at or above its limit; the other three want T at or below.
    """
    terms = {
        "membership_inference_auc_margin":
            (t["membership_inference_auc"], b0["membership_inference_auc"] + AUC_MARGIN),
        "exact_duplicate_guardrail":
            (t["exact_duplicate_rate_train"],
             max(b0["exact_duplicate_rate_train"], 1.0 / n_synthetic)),
        "near_duplicate_guardrail":
            (t["near_duplicate_rate_train"],
             NEAR_DUPLICATE_FACTOR * max(b0["near_duplicate_rate_train"], NEAR_DUPLICATE_FLOOR)),
        "nn_distance_p1_guardrail":
            (t["nn_distance_to_train_p1"], P1_FACTOR * b0["nn_distance_to_train_p1"]),
    }
    checks = {name: bool(value >= limit if name == "nn_distance_p1_guardrail" else value <= limit)
              for name, (value, limit) in terms.items()}
    verdict = {**checks, "passed": all(checks.values()), "n_synthetic": n_synthetic,
               "numbers": {name: {"T": value, "limit_from_B0": limit}
                           for name, (value, limit) in terms.items()},
               "membership_inference_auc_difference": (t["membership_inference_auc"]
                                                       - b0["membership_inference_auc"])}
    if not verdict["passed"]:
        verdict["action_required"] = GATE4_ACTION
    return verdict


def cohort_errors(real: np.ndarray, generated: np.ndarray, labels: np.ndarray,
                  nearest: np.ndarray) -> dict:
    """Per-cohort AF MAE, heterozygosity MAE and median nearest-train distance, eval cohorts only."""
    errors = {}
    for cohort in np.unique(labels):
        rows = labels == cohort
        errors[int(cohort)] = {
            "af_mae": float(np.abs(real[rows].mean(axis=0) - generated[rows].mean(axis=0)).mean()) / 2,
            "het_mae": float(np.abs((real[rows] == 1).mean(axis=0)
                                    - (generated[rows] == 1).mean(axis=0)).mean()),
            "nn_distance_median": float(np.median(nearest[rows])),
        }
    return errors


def maf_errors(real: np.ndarray, generated: np.ndarray, maf_bin_of_snp: np.ndarray) -> dict:
    """Pooled AF and heterozygosity MAE inside each train-MAF bin."""
    af_gap = np.abs(real.mean(axis=0) - generated.mean(axis=0)) / 2
    het_gap = np.abs((real == 1).mean(axis=0) - (generated == 1).mean(axis=0))
    return {index: {"af_mae": float(af_gap[maf_bin_of_snp == index].mean()),
                    "het_mae": float(het_gap[maf_bin_of_snp == index].mean())}
            for index in range(len(MAF_BIN_EDGES) + 1)}


def average(per_seed: list[dict]) -> dict:
    """Average same-shaped {key: {metric: value}} maps across seeds."""
    return {key: {metric: float(np.mean([seed[key][metric] for seed in per_seed]))
                  for metric in metrics} for key, metrics in per_seed[0].items()}


def cohort_rows(cohorts: dict[str, dict], train_counts: Counter, eval_counts: Counter,
                populations: list[str], superpops: list[str]) -> list[dict]:
    rows = []
    for cohort in range(N_COHORTS):
        n_eval = eval_counts.get(cohort, 0)
        row = {"cohort": cohort, "population": populations[cohort], "superpop": superpops[cohort],
               "n_train": train_counts.get(cohort, 0), "n_eval": n_eval}
        if n_eval == 0:
            # Explicit words, not a blank cell: the stratum is unmeasurable, not missing.
            rows.append({**row, **{f"{metric}_{arm}": NOT_ESTIMABLE for metric in COHORT_METRICS
                                   for arm in ARMS}, "status": NOT_ESTIMABLE})
            continue
        rows.append({**row, **{f"{metric}_{arm}": cohorts[arm][cohort][metric]
                               for metric in COHORT_METRICS for arm in ARMS},
                     "status": "low precision (n<5)" if n_eval < 5 else "ok"})
    # The panel has no cohort that is absent from train, so the unseen stratum is stated, not faked.
    rows.append({"cohort": "unseen", "status": "no unseen cohort in this panel"})
    return rows


def maf_rows(bins: dict[str, dict], maf_bin_of_snp: np.ndarray) -> list[dict]:
    return [{"maf_bin": MAF_BIN_LABELS[index],
             "n_snps": int((maf_bin_of_snp == index).sum()),
             **{f"{metric}_{arm}": bins[arm][index][metric]
                for metric in ("af_mae", "het_mae") for arm in ARMS}}
            for index in range(len(MAF_BIN_EDGES) + 1)]


def summary_rows(summaries: dict[str, dict], differences: dict) -> list[dict]:
    rows = [{"arm": arm, "metric": metric, "mean": value["mean"], "sample_sd": value["sample_sd"]}
            for arm, summary in summaries.items() for metric, value in summary.items()]
    return rows + [{"arm": "T-B0", "metric": metric, "mean": value["mean"],
                    "sample_sd": value["sample_sd"]} for metric, value in differences.items()]


def load_panel(prepared_dir: Path) -> dict:
    """Calls, labels, split indices, cohort names and MAF bins from the frozen prepared directory."""
    with np.load(prepared_dir / "genotypes.npz") as data:
        calls = data["calls"].astype(np.float64)
    if not np.isfinite(calls).all():
        raise ValueError("Privacy distances need complete calls; this panel has missing genotypes")
    with np.load(prepared_dir / "dataset.npz") as data:
        dataset = {name: data[name] for name in data.files if name != "x"}
    with (prepared_dir / "label_hierarchy.pkl").open("rb") as handle:
        hierarchy = pickle.load(handle)
    superpop_of = load_superpop_of_cohort(prepared_dir)
    with (prepared_dir / "gene_variant_map.json").open() as handle:
        maf = _variant_values(json.load(handle)["genes"], "train_maf")
    return {"calls": calls, "labels": dataset["y"].astype(np.int64), "dataset": dataset,
            "populations": [hierarchy["idx_to_pop"][cohort] for cohort in range(N_COHORTS)],
            "superpops": [hierarchy["idx_to_superpop"][int(superpop_of[cohort])]
                          for cohort in range(N_COHORTS)],
            "maf_bin_of_snp": np.searchsorted(MAF_BIN_EDGES, maf, side="right")}


def read_seed(run_dir: Path, expected: dict) -> dict:
    """One seed's frozen report, refused unless it matches the run's split and panel shape."""
    report = json.loads((run_dir / "diagnostic_report.json").read_text())
    evaluation = report["genotype_evaluation"]
    composition = {int(cohort): count for cohort, count in report["label_composition"].items()}
    found = {"eval_split": report["eval_split"], "samples": evaluation["samples"],
             "variants": evaluation["variants"], "label_composition": composition}
    if expected and found != expected:
        raise ValueError(f"{run_dir} disagrees with the first seed: {found} != {expected}")
    return found


def run(prepared_dir: Path, multiseed_dir: Path, output_dir: Path) -> dict:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    panel = load_panel(prepared_dir)
    manifest = json.loads((multiseed_dir / "manifest.json").read_text())
    # The whole privacy claim rests on these synthetic rows coming from THIS panel's train split.
    changed = [name for name, digest in manifest["prepared_sha256"].items()
               if digest != _digest(prepared_dir / name)]
    if changed:
        raise ValueError(f"{multiseed_dir} was not run on {prepared_dir}: {', '.join(changed)}")
    seeds = manifest["seeds"]
    if len(seeds) < 2:
        raise ValueError(f"Seeds are the replicates here; {len(seeds)} has no sample SD")
    calls, labels = panel["calls"], panel["labels"]
    train = calls[panel["dataset"]["train_indices"]]
    test = calls[panel["dataset"]["test_indices"]]
    # "Legitimately similar": how close a real non-member already sits to the train panel.
    reference = distance_stats(mean_abs_dosage_distance(test, train).min(axis=1))
    threshold = reference["p1"]

    shared: dict = {}
    per_seed: dict[str, list[dict]] = {arm: [] for arm in ARMS}
    cohorts: dict[str, list[dict]] = {arm: [] for arm in ARMS}
    bins: dict[str, list[dict]] = {arm: [] for arm in ARMS}
    for seed in seeds:
        run_dir = multiseed_dir / f"seed_{seed}" / "standard"
        shared = read_seed(run_dir, shared)
        if shared["eval_split"] == "train":
            raise ValueError("Members and the eval split would overlap; Phase 4 needs a held-out eval")
        eval_indices = panel["dataset"][f"{shared['eval_split']}_indices"][:shared["samples"]]
        eval_calls, eval_labels = calls[eval_indices], labels[eval_indices]
        for arm, filename in ARMS.items():
            synthetic = np.load(run_dir / filename)
            metrics, nearest = arm_privacy(synthetic, train, test, threshold)
            per_seed[arm].append({"seed": seed, **metrics})
            cohorts[arm].append(cohort_errors(eval_calls, synthetic, eval_labels, nearest))
            bins[arm].append(maf_errors(eval_calls, synthetic, panel["maf_bin_of_snp"]))

    flat = {arm: [flatten(seed) for seed in values] for arm, values in per_seed.items()}
    summaries = {arm: summarize(values) for arm, values in flat.items()}
    differences = summarize([{key: t[key] - b0[key] for key in t}
                             for b0, t in zip(flat["B0"], flat["T"])])
    verdict = gate4({key: value["mean"] for key, value in summaries["B0"].items()},
                    {key: value["mean"] for key, value in summaries["T"].items()},
                    n_synthetic=shared["samples"])

    # Read everything that can still fail before the directory exists, so a retry is not blocked
    # by a FileExistsError from the attempt that failed.
    context = phase3_context(multiseed_dir)
    output_dir.mkdir(parents=True)
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "prepared_dir": str(prepared_dir), "multiseed_dir": str(multiseed_dir),
        "prepared_sha256": {name: _digest(prepared_dir / name) for name in PREPARED_FILES},
        "source_sha256": {"scripts/hipodit_privacy.py": _digest(Path(__file__).resolve())},
        "seeds": seeds, "eval_split": shared["eval_split"],
        "n_synthetic_per_seed": shared["samples"], "n_snps": shared["variants"],
        "n_member": len(train), "n_nonmember": len(test),
        "reference_nn_distance": reference, "near_duplicate_threshold": threshold,
        "arms": {arm: {"per_seed": per_seed[arm], "summary": summaries[arm]} for arm in ARMS},
        "differences_T_minus_B0": differences,
        "context_from_phase3": context,
        "gate4": verdict, "limitation": LIMITATION, **provenance(),
    }
    _atomic_json(output_dir / "privacy_report.json", report)
    _write_csv(output_dir / "privacy_summary.csv", summary_rows(summaries, differences),
               SUMMARY_COLUMNS)
    _write_csv(output_dir / "strata_cohort.csv",
               cohort_rows({arm: average(values) for arm, values in cohorts.items()},
                           Counter(labels[panel["dataset"]["train_indices"]].tolist()),
                           Counter(shared["label_composition"]),
                           panel["populations"], panel["superpops"]), COHORT_COLUMNS)
    _write_csv(output_dir / "strata_maf.csv",
               maf_rows({arm: average(values) for arm, values in bins.items()},
                        panel["maf_bin_of_snp"]), MAF_COLUMNS)
    return report


def phase3_context(multiseed_dir: Path) -> dict:
    """Gate 3' population-classifier AUCs, carried for context only and never used in Gate 4."""
    path = multiseed_dir / "summary_decoder.json"
    if not path.is_file():
        return {"note": "no Gate 3' summary_decoder.json beside the seeds"}
    summary = json.loads(path.read_text())
    return {"classifier_macro_auc": {
                "real_eval": summary["classifier_real_eval"]["classifier_macro_auc"],
                "B0_mean": summary["metrics"]["classifier_macro_auc"]["B0_mean"],
                "T_mean": summary["metrics"]["classifier_macro_auc"]["T_mean"]},
            "used_in_gate4": False,
            "note": "Cohort separability of the synthetic arms; fidelity evidence, not a Gate 4 term"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--multiseed-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.prepared_dir.resolve(), args.multiseed_dir.resolve(),
                 args.output_dir.resolve())
    print(json.dumps(report["gate4"], sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
