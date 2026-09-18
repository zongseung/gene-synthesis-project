#!/usr/bin/env python3
# ─── How to run ───
# Imported by hipodit_rebuild_train.py for `evaluate_genotypes`; the Phase 1 Gate 1' oracle is:
#   python scripts/hipodit_genotype_check.py oracle --prepared-dir P --output-dir O [--arms t]
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import resource
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.special import expit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.genotype_decoder import (
    GenotypeDecoder,
    distance_bins,
    fit_decoder,
    nll_calls,
    sample_calls,
)


def git_commit() -> str | None:
    """HEAD's commit, read from .git without a subprocess (the drivers stub subprocess.run)."""
    # ponytail: plain .git directory only; a worktree's .git *file* needs `git rev-parse HEAD`.
    git = PROJECT_ROOT / ".git"
    if not (git / "HEAD").exists():
        return None
    head = (git / "HEAD").read_text().split()[-1]
    if (git / head).exists():
        return (git / head).read_text().strip()
    packed = (git / "packed-refs").read_text().splitlines() if (git / "packed-refs").exists() else []
    return next((line.split()[0] for line in packed if line.endswith(" " + head)),
                head if len(head) == 40 else None)


def provenance() -> dict:
    """Git commit and this process's peak RSS, for every manifest the study writes (plan Task 14)."""
    return {"git_commit": git_commit(),
            "max_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}

N_COHORTS = 26
DISTANCE_BINS = 4
PAIR_OFFSETS = (1, 2, 4)
MAF_BIN_EDGES = (0.05, 0.2)
SOURCE_FILES = ("src/models/genotype_decoder.py", "scripts/hipodit_genotype_check.py")
ABLATION_COLUMNS = ("arm", "nll_per_call", "af_mae", "cohort_af_mae", "genotype_proportion_tv",
                    "heterozygosity_mae", "ld_r2_mae", "local_dosage_covariance_mae")
STRATA_COLUMNS = ("arm", "stratum_type", "stratum", "n", "value")
# Plan rev.2 §9.3 arms: reported name -> (decoder arm, tilt scope).
TILT_ARMS = {"B0": ("B0", "none"), "B1": ("B1", "none"), "T": ("T", "snp"),
             "T_superpop": ("T", "superpop"), "T_cohort": ("T", "cohort")}
# Plan rev.2 §9.4 Gate 1' for T against B0: (metric, factor on B0, strict inequality).
GATE1_PRIME_RULES = (("nll_per_call", 1.0, True), ("heterozygosity_mae", 1.0, True),
                     ("cohort_af_mae", 1.05, False), ("af_mae", 1.05, False))
# Plan rev.2 §9.4 Gate 2' is Gate 1' without the NLL: generated latents carry no teacher-forced
# predecessor, and arm T uses none in the first place. Rules and their reported names, in order.
GATE2_PRIME_RULES = (("heterozygosity_mae", 1.0, True), ("cohort_af_mae", 1.05, False),
                     ("af_mae", 1.05, False))
GATE2_PRIME_NAMES = ("heterozygosity_improved", "cohort_af_guardrail", "af_guardrail")
# Prepared files whose bytes fix the scale and origin of the latents the decoder was fitted on.
SCALE_CHECK_FILES = ("dataset.npz", "gene_variant_map.json", "genotypes.npz",
                     "glm_pca_parameters.pkl", "normalization_stats.pkl")
REAL_FACTOR_TOLERANCE = 1e-5
DESCRIPTIVE_METRICS = ("ld_r2_mae", "ld_r2_mae_by_bin", "local_dosage_covariance_mae")
CONTROL_METRICS = ("af_mae", "heterozygosity_mae")
# Plan rev.2 §9.4 Gate 3': every bootstrap, whatever its resampling unit, is this one.
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 20260915
# ponytail: one fitted classifier per (train rows, train labels); the real train block is one
# array, so keying on its bytes costs milliseconds. Drop it if the block ever grows a gigabyte.
_CLASSIFIER_CACHE: dict[tuple[bytes, bytes], object] = {}


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _variant_values(genes: list[dict], name: str) -> np.ndarray:
    return np.array([variant[name] for gene in genes for variant in gene["variants"]],
                    dtype=np.float64)


def panel_positions(prepared_dir: Path) -> np.ndarray:
    """Reference positions in panel order, one per variant column."""
    with (Path(prepared_dir) / "gene_variant_map.json").open() as handle:
        return _variant_values(json.load(handle)["genes"], "position")


def _pairs(offsets: np.ndarray) -> list[tuple[int, int]]:
    """Within-gene column pairs at the fixed index offsets, in panel order."""
    return [(left, left + distance)
            for start, end in zip(offsets[:-1], offsets[1:])
            for distance in PAIR_OFFSETS
            for left in range(int(start), int(end) - distance)]


def _mean_or_none(values) -> float | None:
    return float(np.mean(values)) if len(values) else None


def genotype_metrics(real: np.ndarray, generated: np.ndarray, *, offsets: np.ndarray,
                     positions: np.ndarray, edges: np.ndarray, real_labels=None,
                     gen_labels=None, maf_bin_of_snp=None) -> dict:
    """Marginal, cohort and local-LD agreement between observed and generated dosages."""
    real = np.asarray(real, dtype=np.float64)
    generated = np.asarray(generated)
    offsets = np.asarray(offsets, dtype=np.int64)
    positions = np.asarray(positions, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.float64)
    valid = np.isfinite(real)
    counts = valid.sum(axis=0)
    usable = counts > 0
    real_af = np.nansum(real, axis=0) / (2 * np.maximum(counts, 1))
    generated_af = generated.mean(axis=0) / 2
    real_proportions = np.stack([(valid & (real == dosage)).sum(axis=0) / np.maximum(counts, 1)
                                 for dosage in (0, 1, 2)])
    generated_proportions = np.stack([(generated == dosage).mean(axis=0) for dosage in (0, 1, 2)])
    proportion_gap = np.abs(real_proportions - generated_proportions)

    covariance_errors, r2_errors = [], []
    binned: list[list[float]] = [[] for _ in range(len(edges) + 1)]
    skipped = 0
    # Row-paired draws (evaluate_genotypes) share the real missingness mask; pooled draws use
    # every generated row.
    paired = len(generated) == len(real)
    # ponytail: one Python pass per column pair (~3 x SNPs); vectorize only if the panel
    # outgrows a chromosome-arm gene set.
    for left, right in _pairs(offsets):
        rows = valid[:, left] & valid[:, right]
        if rows.sum() < 4:
            continue
        real_pair = np.cov(real[rows, left], real[rows, right], ddof=0)
        kept = rows if paired else slice(None)
        generated_pair = np.cov(generated[kept, left], generated[kept, right], ddof=0)
        covariance_errors.append(abs(real_pair[0, 1] - generated_pair[0, 1]))
        real_scale = real_pair[0, 0] * real_pair[1, 1]
        generated_scale = generated_pair[0, 0] * generated_pair[1, 1]
        if real_scale <= 0 or generated_scale <= 0:
            skipped += 1
            continue
        error = abs(real_pair[0, 1] ** 2 / real_scale - generated_pair[0, 1] ** 2 / generated_scale)
        r2_errors.append(error)
        binned[int(np.searchsorted(edges, positions[right] - positions[left], side="right"))].append(error)

    metrics = {
        "af_mae": float(np.abs(real_af - generated_af)[usable].mean()),
        "local_dosage_covariance_mae": _mean_or_none(covariance_errors),
        "local_pairs_evaluated": len(covariance_errors),
        "valid_genotype_fraction": float(np.isin(generated, [0, 1, 2]).mean()),
        "unique_individual_fraction": len(np.unique(generated, axis=0)) / len(generated),
        "genotype_proportion_tv": float(0.5 * proportion_gap.sum(axis=0)[usable].mean()),
        "heterozygosity_mae": float(proportion_gap[1][usable].mean()),
        "ld_r2_mae": _mean_or_none(r2_errors),
        "ld_r2_mae_by_bin": [_mean_or_none(bin_errors) for bin_errors in binned],
        "ld_pairs_by_bin": [len(bin_errors) for bin_errors in binned],
        "ld_pairs_skipped": skipped,
    }

    if real_labels is not None and gen_labels is not None:
        real_labels = np.asarray(real_labels)
        gen_labels = np.asarray(gen_labels)
        cells, by_cohort, sizes, not_estimable = [], {}, {}, []
        for cohort in sorted(np.unique(real_labels).tolist()):
            block = real[real_labels == cohort]
            sizes[int(cohort)] = len(block)
            if not (gen_labels == cohort).any():
                not_estimable.append(int(cohort))
                continue
            block_counts = np.isfinite(block).sum(axis=0)
            block_af = np.nansum(block, axis=0) / (2 * np.maximum(block_counts, 1))
            error = np.abs(block_af - generated[gen_labels == cohort].mean(axis=0) / 2)
            error = error[block_counts > 0]
            cells.append(error)
            by_cohort[int(cohort)] = float(error.mean())
        metrics["cohort_af_mae"] = _mean_or_none(np.concatenate(cells) if cells else [])
        metrics["cohort_af_mae_by_cohort"] = by_cohort
        metrics["cohort_n_real"] = sizes
        metrics["cohorts_not_estimable"] = not_estimable

    if maf_bin_of_snp is not None:
        maf_bin_of_snp = np.asarray(maf_bin_of_snp, dtype=np.int64)
        af_error = np.abs(real_af - generated_af)
        metrics["af_mae_by_maf_bin"] = [
            _mean_or_none(af_error[usable & (maf_bin_of_snp == index)])
            for index in range(len(MAF_BIN_EDGES) + 1)]
    return metrics


def _scale_check(prepared_dir: Path, decoder_dir: Path, real_factors: np.ndarray,
                 indices: np.ndarray) -> dict:
    """Plan rev.2 §9.4 Gate 2': the panel and the latent scale must be the ones the decoder saw.

    The tilt was fitted against oracle logits, so decoding generated latents is only meaningful if
    the inverse-normalization boundary still puts them on that same scale and origin.
    """
    from src.preprocessing.tokenizer import invert_normalization, load_normalization_stats

    frozen = json.loads((decoder_dir / "study_manifest.json").read_text())["prepared_sha256"]
    changed = [name for name in SCALE_CHECK_FILES
               if frozen.get(name) != _digest(prepared_dir / name)]
    if changed:
        raise ValueError(f"Prepared files changed since the decoder was fitted: {', '.join(changed)}")
    stats = load_normalization_stats(prepared_dir / "normalization_stats.pkl")
    with np.load(prepared_dir / "dataset.npz") as data:
        expected = invert_normalization(data["x"][indices], stats)
    difference = float(np.abs(np.asarray(real_factors, dtype=np.float64) - expected).max())
    if difference > REAL_FACTOR_TOLERANCE:
        raise ValueError("Real factors are off the inverse-normalized latent scale by "
                         f"{difference:.6g} > {REAL_FACTOR_TOLERANCE:g}")
    return {"prepared_hashes_match": True, "real_factor_max_abs_diff": difference}


def _load_gate1_decoder(decoder_dir: Path) -> GenotypeDecoder:
    """The arm-T decoder of an oracle run, refused unless that run passed Gate 1'."""
    verdict = json.loads((decoder_dir / "oracle_results.json").read_text())["gate1_prime"]
    if not verdict["passed"]:
        raise ValueError(f"Decoder {decoder_dir} did not pass Gate 1'; it must not decode Phase 2")
    return GenotypeDecoder.load(decoder_dir / "decoder_T.npz")


def _extremeness(real_eta: np.ndarray, synthetic_eta: np.ndarray) -> dict:
    """How far into the tails each linear predictor runs (plan rev.2 §9.4).

    The tilt was fitted on oracle logits, so generated logits that are more confident than the real
    ones would apply a heterozygosity correction calibrated somewhere else.
    """
    def tails(eta: np.ndarray) -> dict:
        absolute = np.abs(eta)
        return {"frac_abs_gt_4": float((absolute > 4).mean()),
                "frac_abs_gt_6": float((absolute > 6).mean()),
                "mean_abs": float(absolute.mean())}

    real, synthetic = tails(real_eta), tails(synthetic_eta)
    ratio = synthetic["frac_abs_gt_4"] / real["frac_abs_gt_4"] if real["frac_abs_gt_4"] else None
    return {"real": real, "synthetic": synthetic, "frac_abs_gt_4_ratio": ratio}


def gate2_prime(candidate: dict, baseline: dict) -> dict:
    """Plan rev.2 §9.4 Gate 2': on generated latents T must improve heterozygosity MAE and stay
    inside a 5% cohort and overall AF MAE margin. LD r2 is carried descriptively only."""
    checks = _gate_checks(candidate, baseline, GATE2_PRIME_RULES)
    return {
        "nll_not_applicable": True,
        **dict(zip(GATE2_PRIME_NAMES, checks)),
        "passed": all(checks),
        "metrics": {metric: {"T": candidate.get(metric), "B0": baseline.get(metric)}
                    for metric in (*(rule[0] for rule in GATE2_PRIME_RULES), "ld_r2_mae")},
    }


def evaluate_genotypes(
    prepared_dir: Path, synthetic_factors: np.ndarray, real_factors: np.ndarray, *, seed: int,
    decoder_dir: Path | None = None, split: str = "val",
) -> tuple[dict, dict[str, np.ndarray]]:
    """Decode one split's generated and real latents to calls, with arm B0 and, given a Gate
    1'-passing oracle directory, the primary arm T beside it."""
    prepared_dir = Path(prepared_dir)
    with (prepared_dir / "glm_pca_parameters.pkl").open("rb") as handle:
        parameters = pickle.load(handle)
    if any(item.get("family") != "binomial" or item.get("trials") != 2 for item in parameters):
        raise ValueError("Genotype evaluation requires a Binomial(2,p) decoder")
    with np.load(prepared_dir / "genotypes.npz") as data:
        calls, offsets = data["calls"], data["offsets"]
    with np.load(prepared_dir / "dataset.npz") as data:
        indices = data[f"{split}_indices"][:len(synthetic_factors)]
        labels = data["y"][indices].astype(np.int64)
    real = calls[indices]
    positions = panel_positions(prepared_dir)
    edges = distance_bins(positions, offsets, DISTANCE_BINS).edges
    synthetic_logits = base_logits(parameters, synthetic_factors)
    real_logits = base_logits(parameters, real_factors)

    def decode(logits: np.ndarray, draw_seed: int) -> np.ndarray:
        return np.random.default_rng(draw_seed).binomial(2, expit(logits)).astype(np.int8)

    arms = {"B0": decode(synthetic_logits, seed)}
    reconstruction = decode(real_logits, seed + 1)

    def metrics(generated: np.ndarray) -> dict:
        # Generated rows are the split's own individuals, so real and generated labels agree.
        return genotype_metrics(real, generated, offsets=offsets, positions=positions, edges=edges,
                                real_labels=labels, gen_labels=labels)

    report = {
        "synthetic": metrics(arms["B0"]),
        "real_latent_decoder_reference": metrics(reconstruction),
        "samples": len(arms["B0"]),
        "variants": arms["B0"].shape[1],
        "limitation": "One stochastic independent-binomial decoder draw; unphased local dosage covariance, not haplotype LD",
    }
    if decoder_dir is None:
        return report, arms

    decoder_dir = Path(decoder_dir)
    report["scale_check"] = _scale_check(prepared_dir, decoder_dir, real_factors, indices)
    decoder = _load_gate1_decoder(decoder_dir)
    arms["T"] = sample_calls(decoder, synthetic_logits, labels, np.random.default_rng(seed + 2))
    reconstruction_t = sample_calls(decoder, real_logits, labels, np.random.default_rng(seed + 3))
    offset = decoder.cohort_offset[labels]
    synthetic_t = metrics(arms["T"])
    # Arm T promises E[g] = 2 sigmoid(eta0 + u) exactly; one draw per individual is a loose test.
    preserved = af_preservation(expit(synthetic_logits + offset).mean(axis=0),
                                arms["T"].mean(axis=0) / 2, len(labels))
    report.update({
        "synthetic_T": synthetic_t,
        "real_latent_decoder_reference_T": metrics(reconstruction_t),
        "decoder_arm": decoder.arm, "decoder_tilt_scope": decoder.tilt_scope,
        "decoder_offset_gauge": decoder.offset_gauge,
        "decoder_pooled_af_residual": decoder.pooled_af_residual,
        "decoder_sha256": _digest(decoder_dir / "decoder_T.npz"),
        "decoder_lambda": {"lambda_u": decoder.lambda_u, "lambda_tilt": decoder.lambda_tilt},
        "extremeness": _extremeness(real_logits + offset, synthetic_logits + offset),
        "af_preservation": {
            "expected_af_vs_sampled_max_abs_diff": preserved["sampled_af_max_abs_diff"],
            "draw_noise_3sigma": preserved["draw_noise_3sigma"],
            "within_draw_noise": preserved["within_draw_noise"]},
        "decoder_gate": gate2_prime(synthetic_t, report["synthetic"]),
    })
    return report, arms


def base_logits(parameters: list[dict], factors: np.ndarray) -> np.ndarray:
    """GLM-PCA base logits in panel order, float64, for factors on the original scale."""
    return np.concatenate([factors[:, gene] @ item["loadings"].T + item["intercept"]
                           for gene, item in enumerate(parameters)], axis=1).astype(np.float64)


def load_panel(prepared_dir: Path) -> dict:
    """Everything the oracle reads from the frozen prepared directory, in panel order."""
    from src.preprocessing.tokenizer import invert_normalization, load_normalization_stats

    report = json.loads((prepared_dir / "prepare_report.json").read_text())
    with (prepared_dir / "glm_pca_parameters.pkl").open("rb") as handle:
        parameters = pickle.load(handle)
    with np.load(prepared_dir / "genotypes.npz") as data:
        calls, offsets = data["calls"].astype(np.float64), data["offsets"].astype(np.int64)
    with np.load(prepared_dir / "dataset.npz") as data:
        dataset = {name: data[name] for name in data.files}
    stats = load_normalization_stats(prepared_dir / "normalization_stats.pkl",
                                     expected_shape=(report["genes"], report["components"]))
    with (prepared_dir / "gene_variant_map.json").open() as handle:
        genes = json.load(handle)["genes"]
    maf = _variant_values(genes, "train_maf")
    return {
        "report": report, "parameters": parameters, "calls": calls, "offsets": offsets,
        "stats_fit_split": stats["fit_split"], "genes": genes,
        "positions": _variant_values(genes, "position"),
        "maf_bin_of_snp": np.searchsorted(MAF_BIN_EDGES, maf, side="right"),
        "logits": base_logits(parameters, invert_normalization(dataset["x"], stats).astype(np.float64)),
        "labels": dataset["y"].astype(np.int64),
        "train_indices": dataset["train_indices"].astype(np.int64),
        "dev_indices": dataset["val_indices"].astype(np.int64),
        "test_indices": dataset["test_indices"].astype(np.int64),
    }


def phase0_assertions(panel: dict, n_bins: int) -> list[str]:
    """Data-contract checks that must hold before any decoder sees the panel."""
    union = np.concatenate([panel["train_indices"], panel["dev_indices"], panel["test_indices"]])
    if not np.array_equal(np.sort(union), np.arange(len(panel["calls"]))):
        raise ValueError("Train, dev and test indices must be disjoint and cover every panel row")
    report = panel["report"]
    if report["fit_split"] != "train":
        raise ValueError(f"prepare_report fit_split must be 'train', found {report['fit_split']!r}")
    if report["glm_family"] != "binomial":
        raise ValueError(f"prepare_report glm_family must be 'binomial', found {report['glm_family']!r}")
    if not str(report["maf_filter"]).startswith("train-only"):
        raise ValueError(f"MAF filter must be train-only, found {report['maf_filter']!r}")
    if panel["stats_fit_split"] != "train":
        raise ValueError(f"Normalization stats must be fitted on train, found {panel['stats_fit_split']!r}")
    if any(item.get("family") != "binomial" or item.get("trials") != 2
           for item in panel["parameters"]):
        raise ValueError("Every GLM-PCA parameter block must be Binomial with two trials")
    sizes = np.cumsum([0] + [len(gene["variants"]) for gene in panel["genes"]])
    if not np.array_equal(panel["offsets"], sizes):
        raise ValueError("Genotype offsets must equal the cumulative gene sizes of the variant map")
    distance_bins(panel["positions"], panel["offsets"], n_bins)
    observed = panel["calls"][np.isfinite(panel["calls"])]
    if not np.isin(observed, [0, 1, 2]).all():
        raise ValueError("Observed calls must be diploid dosages 0, 1 or 2")
    if panel["labels"].min() < 0 or panel["labels"].max() >= N_COHORTS:
        raise ValueError(f"Cohort labels must lie inside [0, {N_COHORTS})")
    return ["splits_disjoint_and_complete", "prepare_report_fit_split_is_train",
            "prepare_report_glm_family_is_binomial", "prepare_report_maf_filter_is_train_only",
            "normalization_stats_fit_split_is_train",
            "glm_pca_parameters_are_binomial_with_two_trials",
            "offsets_match_gene_variant_map", "positions_ascend_within_every_gene",
            "observed_calls_are_diploid_dosages", "cohort_labels_within_range"]


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pilot_block(pilot_dir: Path | None) -> dict | None:
    if pilot_dir is None or not (pilot_dir / "summary.json").is_file():
        return None
    manifest = json.loads((pilot_dir / "manifest.json").read_text())
    summary = json.loads((pilot_dir / "summary.json").read_text())
    return {
        "dir": str(pilot_dir),
        "manifest_sha256": _digest(pilot_dir / "manifest.json"),
        "summary_sha256": _digest(pilot_dir / "summary.json"),
        "seeds": manifest["seeds"], "parameters": manifest["parameters"],
        "metrics": {name: {schedule: summary["metrics"][name][schedule]["mean"]
                           for schedule in ("standard", "fisher")}
                    for name in ("af_mae", "local_dosage_covariance_mae")},
    }


def study_manifest(prepared_dir: Path, panel: dict, pilot_dir: Path | None,
                   assertions: list[str]) -> dict:
    report = panel["report"]
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "prepared_dir": str(prepared_dir),
        "prepared_sha256": {str(path.relative_to(prepared_dir)): _digest(path)
                            for path in sorted(prepared_dir.rglob("*")) if path.is_file()},
        "source_sha256": {name: _digest(PROJECT_ROOT / name) for name in SOURCE_FILES},
        "split_sizes": report["split_sizes"], "seed": report["seed"],
        "components": report["components"], "glm_iterations": report["glm_iterations"],
        "maf_filter": report["maf_filter"], "snp_count": int(panel["offsets"][-1]),
        "gene_count": len(panel["genes"]),
        "latent_generator_baseline": "standard",
        "pilot": _pilot_block(pilot_dir),
        "decoder_inputs_from_test": False,
        "assertions_passed": assertions,
    }


def load_superpop_of_cohort(prepared_dir: Path) -> np.ndarray:
    """Superpopulation index of every cohort 0..25, from the prepared label hierarchy."""
    with (Path(prepared_dir) / "label_hierarchy.pkl").open("rb") as handle:
        mapping = pickle.load(handle)["pop_to_superpop"]
    if set(mapping) != set(range(N_COHORTS)):
        raise ValueError(f"pop_to_superpop must key every int cohort 0..{N_COHORTS - 1}")
    return np.array([mapping[cohort] for cohort in range(N_COHORTS)], dtype=np.int64)


def _across_seeds(values: list, reduce):
    """Reduce same-shaped per-seed metrics leaf by leaf; a leaf missing in any seed is None."""
    # ponytail: zip truncates ragged lists; only cohorts_not_estimable could be ragged, and it is
    # always empty here because generated labels tile the real dev labels.
    first = values[0]
    if isinstance(first, dict):
        return {key: _across_seeds([value[key] for value in values], reduce) for key in first}
    if isinstance(first, list):
        return [_across_seeds(list(column), reduce) for column in zip(*values)]
    return None if any(value is None for value in values) else reduce(values)


def seed_summary(per_seed: list[dict]) -> dict:
    """Leafwise mean over metric seeds, with the ddof=1 SD once there are two seeds or more."""
    def reduce(values: list) -> dict:
        spread = {"sample_sd": float(np.std(values, ddof=1))} if len(values) > 1 else {}
        return {"mean": float(np.mean(values)), **spread}
    return _across_seeds(per_seed, reduce)


def _point(result: dict) -> dict:
    """One metric dict per result: the legacy single draw or the mean over metric seeds."""
    if "sampled" in result:
        return result["sampled"]
    return _across_seeds(result["per_seed"], lambda values: float(np.mean(values)))


def af_preservation(expected_af: np.ndarray, sampled_af: np.ndarray, n_samples: int) -> dict:
    """Largest per-SNP gap between promised and sampled AF vs the largest 3-sigma draw noise."""
    # ponytail: max gap vs max noise (brief §E) cannot flag a rare SNP drifting past its own
    # 3 sigma; count per-SNP z-scores > 3 if that ever needs catching.
    expected_af = np.asarray(expected_af, dtype=np.float64)
    gap = float(np.abs(np.asarray(sampled_af, dtype=np.float64) - expected_af).max())
    noise = float((3 * np.sqrt(expected_af * (1 - expected_af) / (2 * n_samples))).max())
    return {"expected_af": expected_af.tolist(), "sampled_af_max_abs_diff": gap,
            "draw_noise_3sigma": noise, "within_draw_noise": gap <= noise, "n_samples": n_samples}


def _dev_nll(decoder, calls: np.ndarray, logits: np.ndarray, labels: np.ndarray,
             dev: np.ndarray) -> float:
    return float(np.nanmean(nll_calls(decoder, calls[dev], logits[dev], labels[dev])))


def _draw(decoder, logits: np.ndarray, labels: np.ndarray, draws: int, seed: int) -> np.ndarray:
    """Pool `draws` sequential samples per individual from one fixed generator stream."""
    rng = np.random.default_rng(seed)
    return np.concatenate([sample_calls(decoder, logits, labels, rng) for _ in range(draws)])


def _sampled_metrics(panel: dict, generated: np.ndarray, dev_labels: np.ndarray, draws: int,
                     edges: np.ndarray, cohorts: bool) -> dict:
    cohort_labels = {"real_labels": dev_labels,
                     "gen_labels": np.tile(dev_labels, draws)} if cohorts else {}
    return genotype_metrics(panel["calls"][panel["dev_indices"]], generated,
                            offsets=panel["offsets"], positions=panel["positions"], edges=edges,
                            maf_bin_of_snp=panel["maf_bin_of_snp"], **cohort_labels)


def _fit_record(decoder, calls: np.ndarray, logits: np.ndarray, labels: np.ndarray,
                dev: np.ndarray) -> dict:
    return {"nll_per_call": _dev_nll(decoder, calls, logits, labels, dev),
            "train_nll_per_call": decoder.train_nll_per_call, "converged": decoder.converged,
            "lambda_u": decoder.lambda_u, "lambda_a": decoder.lambda_a}


def _evaluate_arm(decoder, panel: dict, calls: np.ndarray, logits: np.ndarray,
                  labels: np.ndarray, args, *, columns: np.ndarray | None = None,
                  cohorts: bool = True) -> dict:
    """Legacy: dev per-call NLL plus one seed's pooled sampled metrics against real dev calls."""
    dev = panel["dev_indices"]
    generated = _draw(decoder, logits[dev], labels[dev], args.draws, args.seed)
    if columns is not None:
        restored = np.empty_like(generated)
        restored[:, columns] = generated
        generated = restored
    return {**_fit_record(decoder, calls, logits, labels, dev),
            "sampled": _sampled_metrics(panel, generated, labels[dev], args.draws,
                                        decoder.bins.edges, cohorts)}


def _evaluate_seeds(decoder, panel: dict, logits: np.ndarray, labels: np.ndarray, draws: int,
                    seeds: list[int], *, cohorts: bool = True) -> tuple[dict, np.ndarray]:
    """Dev NLL and per-seed pooled metrics, every arm on the same stream per seed, plus the AF
    sampled over all draws of all seeds."""
    dev = panel["dev_indices"]
    per_seed, dosage = [], np.zeros(logits.shape[1])
    for seed in seeds:
        generated = _draw(decoder, logits[dev], labels[dev], draws, seed)
        per_seed.append(_sampled_metrics(panel, generated, labels[dev], draws, decoder.bins.edges,
                                         cohorts))
        dosage += generated.sum(axis=0)
    result = {**_fit_record(decoder, panel["calls"], logits, labels, dev),
              "lambda_tilt": decoder.lambda_tilt, "tilt_scope": decoder.tilt_scope,
              "per_seed": per_seed, "summary": seed_summary(per_seed)}
    return result, dosage / (2 * len(dev) * draws * len(seeds))


def _fit(arm: str, panel: dict, calls: np.ndarray, logits: np.ndarray, labels: np.ndarray,
         lambda_u: float, lambda_a: float, n_bins: int, **tilt):
    return fit_decoder(arm, calls, logits, labels, panel["train_indices"], panel["positions"],
                       panel["offsets"], N_COHORTS, lambda_u=lambda_u, lambda_a=lambda_a,
                       n_bins=n_bins, **tilt)


def _gate1(arms: dict) -> dict:
    """Plan Gate 1: B3 must beat B0 on dev NLL and LD r2 without a >5% cohort AF loss."""
    nll = {arm: arms[arm]["nll_per_call"] for arm in ("B0", "B3")}
    ld = {arm: arms[arm]["sampled"]["ld_r2_mae"] for arm in ("B0", "B3")}
    cohort = {arm: arms[arm]["sampled"]["cohort_af_mae"] for arm in ("B0", "B3")}
    if None in (*ld.values(), *cohort.values()):
        return {"passed": False, "fixed_candidate": "B0",
                "reasons": ["LD r2 or cohort AF was not estimable on dev"]}
    comparisons = [
        ("nll_per_call", nll["B3"] < nll["B0"], nll["B3"], nll["B0"]),
        ("ld_r2_mae", ld["B3"] < ld["B0"], ld["B3"], ld["B0"]),
        ("cohort_af_mae", cohort["B3"] <= 1.05 * cohort["B0"], cohort["B3"], cohort["B0"]),
    ]
    passed = all(ok for _, ok, _, _ in comparisons)
    return {
        "passed": passed, "fixed_candidate": "B3" if passed else "B0",
        "reasons": [f"{name}: B3 {value:.6g} vs B0 {baseline:.6g} -> {'pass' if ok else 'fail'}"
                    for name, ok, value, baseline in comparisons],
    }


def _gate_checks(candidate: dict, baseline: dict, rules: tuple) -> list[bool]:
    """Each (metric, factor, strict) rule on flat metric dicts; not estimable fails."""
    checks = []
    for metric, factor, strict in rules:
        value, reference = candidate.get(metric), baseline.get(metric)
        checks.append(value is not None and reference is not None
                      and (value < factor * reference if strict else value <= factor * reference))
    return checks


def _gate1_prime(arms: dict) -> dict:
    """Plan rev.2 §9.4 Gate 1': T must beat B0 on dev NLL and heterozygosity MAE with at most a 5%
    cohort and overall AF MAE loss. The seed mean decides; passing seeds are counted alongside."""
    def flat(arm: str, metrics: dict) -> dict:
        return {**metrics, "nll_per_call": arms[arm]["nll_per_call"]}

    def shown(value) -> str:
        return "not estimable" if value is None else f"{value:.6g}"

    means = {arm: flat(arm, _point(arms[arm])) for arm in ("T", "B0")}
    checks = _gate_checks(means["T"], means["B0"], GATE1_PRIME_RULES)
    pass_seeds = sum(all(_gate_checks(flat("T", seed_t), flat("B0", seed_b0),
                                      GATE1_PRIME_RULES))
                     for seed_t, seed_b0 in zip(arms["T"]["per_seed"], arms["B0"]["per_seed"],
                                                strict=True))
    passed = all(checks)
    return {
        "passed": passed, "candidate": "T" if passed else "B0",
        "reasons": [f"{metric}: T {shown(means['T'].get(metric))} {'<' if strict else '<='} "
                    f"{'' if factor == 1 else f'{factor:g} x '}B0 {shown(means['B0'].get(metric))}"
                    f" -> {'pass' if ok else 'fail'}"
                    for (metric, factor, strict), ok in zip(GATE1_PRIME_RULES, checks)],
        "pass_seeds": int(pass_seeds), "metric_seeds": len(arms["T"]["per_seed"]),
        "descriptive_only": {metric: {arm: arms[arm]["summary"].get(metric) for arm in ("T", "B0")}
                             for metric in DESCRIPTIVE_METRICS},
    }


def _named_results(arms: dict, controls: dict) -> list[tuple[str, dict]]:
    return [*arms.items(), *((f"{arm}_{control}", value) for control, block in controls.items()
                             for arm, value in block.items())]


def _ablation_rows(arms: dict, controls: dict) -> list[dict]:
    rows = []
    for name, result in _named_results(arms, controls):
        point = _point(result)
        rows.append({"arm": name, "nll_per_call": result["nll_per_call"],
                     **{key: point.get(key) for key in ABLATION_COLUMNS[2:]}})
    return rows


def _strata_rows(arms: dict, controls: dict, panel: dict) -> list[dict]:
    maf_sizes = np.bincount(panel["maf_bin_of_snp"], minlength=len(MAF_BIN_EDGES) + 1)
    strata = []
    for name, result in _named_results(arms, controls):
        sampled = _point(result)
        for cohort, value in sampled.get("cohort_af_mae_by_cohort", {}).items():
            strata.append({"arm": name, "stratum_type": "cohort", "stratum": cohort,
                           "n": sampled["cohort_n_real"][cohort], "value": value})
        for index, value in enumerate(sampled.get("af_mae_by_maf_bin", [])):
            strata.append({"arm": name, "stratum_type": "maf_bin", "stratum": index,
                           "n": int(maf_sizes[index]), "value": value})
        for index, value in enumerate(sampled.get("ld_r2_mae_by_bin", [])):
            strata.append({"arm": name, "stratum_type": "distance_bin", "stratum": index,
                           "n": sampled["ld_pairs_by_bin"][index], "value": value})
    return strata


def _write_csv(path: Path, rows: list[dict], fields: tuple[str, ...]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _tilt_study(panel: dict, args: argparse.Namespace, output: Path, seeds: list[int],
                superpops: np.ndarray) -> tuple[dict, dict, dict]:
    """Plan rev.2 §9.3 arms at fixed lambdas, the AF-preservation check, the label-shuffle
    control and Gate 1'."""
    calls, logits, labels = panel["calls"], panel["logits"], panel["labels"]
    dev = panel["dev_indices"]

    def fit(name: str, fit_labels: np.ndarray):
        arm, scope = TILT_ARMS[name]
        return _fit(arm, panel, calls, logits, fit_labels, args.lambda_u, args.lambda_a,
                    args.n_bins, tilt_scope=scope, lambda_tilt=args.lambda_tilt,
                    superpop_of_cohort=superpops)

    arms = {}
    for name in TILT_ARMS:
        decoder = fit(name, labels)
        decoder.save(output / f"decoder_{name}.npz")
        arms[name], sampled_af = _evaluate_seeds(decoder, panel, logits, labels, args.draws, seeds)
        if decoder.arm == "T":
            # Arm T promises E[g] = 2 sigmoid(eta0 + u) exactly; its draws must reproduce this AF.
            expected_af = expit(logits[dev] + decoder.cohort_offset[labels[dev]]).mean(axis=0)
            arms[name]["af_preservation"] = af_preservation(
                expected_af, sampled_af, len(dev) * args.draws * len(seeds))

    shuffled = labels[np.random.default_rng(args.seed + 2).permutation(len(labels))]
    controls = {"label_shuffle": {}}
    for name in ("B1", "T"):
        # Cohort metrics are meaningless under permuted labels, so only NLL, AF and het are kept.
        result, _ = _evaluate_seeds(fit(name, shuffled), panel, logits, shuffled, args.draws, seeds,
                                    cohorts=False)
        controls["label_shuffle"][name] = {
            **{key: result[key] for key in ("nll_per_call", "train_nll_per_call", "converged")},
            "per_seed": [{key: seed[key] for key in CONTROL_METRICS}
                         for seed in result["per_seed"]],
            "summary": {key: result["summary"][key] for key in CONTROL_METRICS}}
    # Matched T-zero diagnostic (plan Task 5 step 3): T's fitted offset, the same held-out factors
    # and draw streams, only tau -> 0. Whatever separates it from T is attributable to the tilt.
    t_zero = GenotypeDecoder.load(output / "decoder_T.npz")
    t_zero = replace(t_zero, tilt=np.zeros_like(t_zero.tilt))
    controls["tilt_zero"] = {
        "T": _evaluate_seeds(t_zero, panel, logits, labels, args.draws, seeds)[0]}
    verdict = {
        "gate1_prime": _gate1_prime(arms),
        "label_control_supports_cohort": bool(
            arms["B1"]["nll_per_call"] < controls["label_shuffle"]["B1"]["nll_per_call"]),
    }
    return arms, controls, verdict


def _legacy_study(panel: dict, args: argparse.Namespace, output: Path) -> tuple[dict, dict, dict]:
    """The B0-B3 study with order/label shuffle controls and Gate 1, at fixed lambdas."""
    calls, logits, labels = panel["calls"], panel["logits"], panel["labels"]
    arms = {}
    for arm in ("B0", "B1", "B2", "B3"):
        decoder = _fit(arm, panel, calls, logits, labels, args.lambda_u, args.lambda_a, args.n_bins)
        decoder.save(output / f"decoder_{arm}.npz")
        arms[arm] = _evaluate_arm(decoder, panel, calls, logits, labels, args)

    order = np.arange(len(panel["positions"]))
    order_rng = np.random.default_rng(args.seed + 1)
    for start, end in zip(panel["offsets"][:-1], panel["offsets"][1:]):
        order[start:end] = start + order_rng.permutation(int(end - start))
    shuffled_labels = labels[np.random.default_rng(args.seed + 2).permutation(len(labels))]
    controls = {"order_shuffle": {}, "label_shuffle": {}}
    for arm in ("B2", "B3"):
        decoder = _fit(arm, panel, calls[:, order], logits[:, order], labels, args.lambda_u,
                       args.lambda_a, args.n_bins)
        controls["order_shuffle"][arm] = _evaluate_arm(
            decoder, panel, calls[:, order], logits[:, order], labels, args, columns=order)
    for arm in ("B1", "B3"):
        decoder = _fit(arm, panel, calls, logits, shuffled_labels, args.lambda_u, args.lambda_a,
                       args.n_bins)
        controls["label_shuffle"][arm] = _evaluate_arm(
            decoder, panel, calls, logits, shuffled_labels, args, cohorts=False)

    order_b3, label_b1 = controls["order_shuffle"]["B3"], controls["label_shuffle"]["B1"]
    verdict = {
        "gate1": _gate1(arms),
        "order_control_supports_ld": bool(
            arms["B3"]["nll_per_call"] < order_b3["nll_per_call"]
            and arms["B3"]["sampled"]["ld_r2_mae"] < order_b3["sampled"]["ld_r2_mae"]),
        "label_control_supports_cohort": bool(
            arms["B1"]["nll_per_call"] < label_b1["nll_per_call"]),
    }
    return arms, controls, verdict


def run_oracle(args: argparse.Namespace) -> dict:
    """Phase 0 freeze plus the Phase 1 dev oracle: Gate 1' tilt arms, or the legacy B0-B3 study."""
    if args.metric_seeds < 1 or args.draws < 1:
        raise ValueError("--metric-seeds and --draws must be positive")
    prepared_dir, output = args.prepared_dir.resolve(), args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    panel = load_panel(prepared_dir)
    assertions = phase0_assertions(panel, args.n_bins)
    tilt = args.arms == "t"
    if tilt:
        superpops = load_superpop_of_cohort(prepared_dir)
        assertions.append("label_hierarchy_maps_every_cohort")
    seeds = [args.seed + offset for offset in range(args.metric_seeds)] if tilt else [args.seed]
    settings = {"arm_set": args.arms, "lambda_u": args.lambda_u, "lambda_tilt": args.lambda_tilt,
                "lambda_a": args.lambda_a, "metric_seeds": len(seeds), "draws": args.draws}
    pilot_dir = args.pilot_dir.resolve() if args.pilot_dir else None
    _atomic_json(output / "study_manifest.json", {
        **study_manifest(prepared_dir, panel, pilot_dir, assertions), **settings, **provenance(),
        "plan_revision": "rev2-§9", "gate": "gate1_prime" if tilt else "gate1"})

    if tilt:
        arms, controls, verdict = _tilt_study(panel, args, output, seeds, superpops)
    else:
        arms, controls, verdict = _legacy_study(panel, args, output)
    bins = distance_bins(panel["positions"], panel["offsets"], args.n_bins)
    results = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "prepared_dir": str(prepared_dir), "output_dir": str(output),
        "seed": args.seed, "seeds": seeds, "n_bins": args.n_bins, **settings,
        "distance_bins": {"edges": bins.edges.tolist(),
                          "adjacent_pairs_per_bin": np.bincount(
                              bins.bin_of_snp[bins.bin_of_snp >= 0],
                              minlength=len(bins.edges) + 1).tolist()},
        "arms": arms, "negative_controls": controls, **verdict,
        "limitation": "Oracle comparison on real held-out GLM-PCA factors; no diffusion samples",
    }
    _atomic_json(output / "oracle_results.json", results)
    _write_csv(output / "decoder_ablation.csv", _ablation_rows(arms, controls), ABLATION_COLUMNS)
    _write_csv(output / "strata.csv", _strata_rows(arms, controls, panel), STRATA_COLUMNS)
    print(json.dumps(verdict, sort_keys=True, allow_nan=False), flush=True)
    return results


def percentile_ci(draws) -> tuple[float, float]:
    """95% percentile interval of an already-resampled bootstrap statistic."""
    low, high = np.percentile(np.asarray(draws, dtype=np.float64), [2.5, 97.5])
    return float(low), float(high)


def _split_indices(panel: dict, split: str) -> np.ndarray:
    return panel[f"{'dev' if split == 'val' else split}_indices"]


def test_nll_comparison(prepared_dir, decoder_dir, split: str = "test") -> dict:
    """Per-call NLL of arms B0 and T on one split's real latents, through the oracle path.

    Plan rev.2 §9.4 ruling R3: this number is deterministic and carries no diffusion seed, so the
    confidence interval for T - B0 resamples held-out individuals rather than seeds.
    """
    panel = load_panel(Path(prepared_dir))
    indices = _split_indices(panel, split)
    calls, logits, labels = (panel["calls"][indices], panel["logits"][indices],
                             panel["labels"][indices])
    decoders = {"B0": GenotypeDecoder.load(Path(decoder_dir) / "decoder_B0.npz"),
                "T": _load_gate1_decoder(Path(decoder_dir))}
    totals = {arm: np.nansum(nll_calls(decoder, calls, logits, labels), axis=1)
              for arm, decoder in decoders.items()}
    counts = np.isfinite(calls).sum(axis=1).astype(np.float64)
    per_call = {arm: float(total.sum() / counts.sum()) for arm, total in totals.items()}
    rows = np.random.default_rng(BOOTSTRAP_SEED).integers(
        len(indices), size=(BOOTSTRAP_DRAWS, len(indices)))
    gap = totals["T"] - totals["B0"]
    low, high = percentile_ci(gap[rows].sum(axis=1) / counts[rows].sum(axis=1))
    return {"b0_nll": per_call["B0"], "t_nll": per_call["T"],
            "difference": per_call["T"] - per_call["B0"], "ci_low": low, "ci_high": high,
            "n_individuals": int(len(indices)), "n_calls": int(counts.sum()),
            "split": split, "bootstrap_draws": BOOTSTRAP_DRAWS,
            "resampling_unit": "held-out individuals"}


def classifier_blocks(prepared_dir, split: str = "test", count: int | None = None) -> dict:
    """Real train dosages to fit the cohort classifier on, and the real eval block beside them."""
    panel = load_panel(Path(prepared_dir))
    train, evaluated = panel["train_indices"], _split_indices(panel, split)[:count]
    blocks = {"train_calls": panel["calls"][train], "train_labels": panel["labels"][train],
              "eval_calls": panel["calls"][evaluated], "eval_labels": panel["labels"][evaluated]}
    if not np.isfinite(np.vstack([blocks["train_calls"], blocks["eval_calls"]])).all():
        raise ValueError("The cohort classifier needs complete dosages; the panel has missing calls")
    return blocks


def population_classifier_scores(train_calls, train_labels, eval_calls, eval_labels) -> dict:
    """How readable a dosage block's cohort labels are to a classifier fitted only on real train
    rows (plan rev.2 ruling R4). Cohorts the block never conditioned on cannot enter a one-vs-rest
    AUC, so they are excluded and named."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    train_calls = np.asarray(train_calls, dtype=np.float64)
    train_labels = np.asarray(train_labels, dtype=np.int64)
    key = (train_calls.tobytes(), train_labels.tobytes())
    if key not in _CLASSIFIER_CACHE:
        # lbfgs multinomial is the default; the fit is seed-independent, hence the cache.
        _CLASSIFIER_CACHE[key] = LogisticRegression(max_iter=2000, random_state=0).fit(
            train_calls, train_labels)
    model = _CLASSIFIER_CACHE[key]
    eval_calls = np.asarray(eval_calls, dtype=np.float64)
    eval_labels = np.asarray(eval_labels, dtype=np.int64)
    unknown = np.setdiff1d(eval_labels, model.classes_)
    if unknown.size:
        raise ValueError(f"Eval cohorts absent from the fitted classifier: {unknown.tolist()}")
    kept = np.isin(model.classes_, eval_labels)
    scored = model.classes_[kept]
    probabilities = model.predict_proba(eval_calls)[:, kept]
    auc = None
    if len(scored) == 2:
        # sklearn reads a two-column score as binary, and both one-vs-rest curves share one AUC.
        auc = float(roc_auc_score(eval_labels == scored[1], probabilities[:, 1]))
    elif len(scored) > 2:
        auc = float(roc_auc_score(eval_labels,
                                  probabilities / probabilities.sum(axis=1, keepdims=True),
                                  multi_class="ovr", average="macro", labels=scored.tolist()))
    return {"classifier_accuracy": float(model.score(eval_calls, eval_labels)),
            "classifier_macro_auc": auc,
            "cohorts_excluded_from_auc": model.classes_[~kept].tolist(),
            "n_eval": int(len(eval_labels))}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Genotype decoder metrics and the Phase 1 oracle comparison")
    commands = parser.add_subparsers(dest="command", required=True)
    oracle = commands.add_parser("oracle", help="Gate 1' tilt arms or legacy B0-B3 on dev")
    oracle.set_defaults(handler=run_oracle)
    oracle.add_argument("--prepared-dir", type=Path, required=True)
    oracle.add_argument("--output-dir", type=Path, required=True)
    oracle.add_argument("--arms", choices=("t", "legacy"), default="t")
    oracle.add_argument("--pilot-dir", type=Path,
                        default=PROJECT_ROOT / "outputs/diagnostics/hipodit_multiseed_20260915")
    oracle.add_argument("--metric-seeds", type=int, default=5,
                        help="sampling seeds --seed, --seed+1, ...; legacy uses --seed only")
    oracle.add_argument("--draws", type=int, default=20)
    oracle.add_argument("--seed", type=int, default=20260915)
    oracle.add_argument("--n-bins", type=int, default=DISTANCE_BINS)
    oracle.add_argument("--lambda-u", type=float, default=1.0)
    oracle.add_argument("--lambda-tilt", type=float, default=1.0)
    oracle.add_argument("--lambda-a", type=float, default=1.0)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
