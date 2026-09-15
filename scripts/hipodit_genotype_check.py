#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
# ─── How to run ───
# Imported by hipodit_rebuild_train.py for `evaluate_genotypes`; the Phase 1 decoder
# comparison is: python scripts/hipodit_genotype_check.py oracle --prepared-dir P --output-dir O
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.special import expit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.genotype_decoder import distance_bins, fit_decoder, nll_calls, sample_calls

N_COHORTS = 26
DISTANCE_BINS = 4
PAIR_OFFSETS = (1, 2, 4)
MAF_BIN_EDGES = (0.05, 0.2)
SOURCE_FILES = ("src/models/genotype_decoder.py", "scripts/hipodit_genotype_check.py")


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def panel_positions(prepared_dir: Path) -> np.ndarray:
    """Reference positions in panel order, one per variant column."""
    with (Path(prepared_dir) / "gene_variant_map.json").open() as handle:
        genes = json.load(handle)["genes"]
    return np.array([variant["position"] for gene in genes for variant in gene["variants"]],
                    dtype=np.float64)


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
    # ponytail: one Python pass per column pair (~3 x SNPs); vectorize only if the panel
    # outgrows a chromosome-arm gene set.
    for left, right in _pairs(offsets):
        rows = valid[:, left] & valid[:, right]
        if rows.sum() < 4:
            continue
        real_pair = np.cov(real[rows, left], real[rows, right], ddof=0)
        generated_pair = np.cov(generated[:, left], generated[:, right], ddof=0)
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


def evaluate_genotypes(
    prepared_dir: Path, synthetic_factors: np.ndarray, real_factors: np.ndarray, *, seed: int,
) -> tuple[dict, np.ndarray]:
    with (prepared_dir / "glm_pca_parameters.pkl").open("rb") as handle:
        parameters = pickle.load(handle)
    if any(item.get("family") != "binomial" or item.get("trials") != 2 for item in parameters):
        raise ValueError("Genotype evaluation requires a Binomial(2,p) decoder")
    with np.load(prepared_dir / "genotypes.npz") as data:
        calls, offsets = data["calls"], data["offsets"]
    with np.load(prepared_dir / "dataset.npz") as data:
        indices = data["val_indices"][:len(synthetic_factors)]
    real = calls[indices]
    positions = panel_positions(prepared_dir)
    edges = distance_bins(positions, offsets, DISTANCE_BINS).edges

    def decode(factors: np.ndarray, draw_seed: int) -> np.ndarray:
        return np.random.default_rng(draw_seed).binomial(
            2, expit(base_logits(parameters, factors))).astype(np.int8)

    synthetic = decode(synthetic_factors, seed)
    reconstruction = decode(real_factors, seed + 1)

    def metrics(generated: np.ndarray) -> dict:
        return genotype_metrics(real, generated, offsets=offsets, positions=positions, edges=edges)

    return {
        "synthetic": metrics(synthetic),
        "real_latent_decoder_reference": metrics(reconstruction),
        "samples": len(synthetic),
        "variants": synthetic.shape[1],
        "limitation": "One stochastic independent-binomial decoder draw; unphased local dosage covariance, not haplotype LD",
    }, synthetic


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
    maf = np.array([variant["train_maf"] for gene in genes for variant in gene["variants"]])
    return {
        "report": report, "parameters": parameters, "calls": calls, "offsets": offsets,
        "stats_fit_split": stats["fit_split"], "genes": genes,
        "positions": panel_positions(prepared_dir),
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


def _dev_nll(decoder, calls: np.ndarray, logits: np.ndarray, labels: np.ndarray,
             dev: np.ndarray) -> float:
    return float(np.nanmean(nll_calls(decoder, calls[dev], logits[dev], labels[dev])))


def _draw(decoder, logits: np.ndarray, labels: np.ndarray, draws: int, seed: int) -> np.ndarray:
    """Pool `draws` sequential samples per individual from one fixed generator stream."""
    rng = np.random.default_rng(seed)
    return np.concatenate([sample_calls(decoder, logits, labels, rng) for _ in range(draws)])


def _evaluate_arm(decoder, panel: dict, calls: np.ndarray, logits: np.ndarray,
                  labels: np.ndarray, args, *, columns: np.ndarray | None = None,
                  cohorts: bool = True) -> dict:
    """Dev per-call NLL plus pooled sampled metrics against the real dev calls."""
    dev = panel["dev_indices"]
    generated = _draw(decoder, logits[dev], labels[dev], args.draws, args.seed)
    if columns is not None:
        restored = np.empty_like(generated)
        restored[:, columns] = generated
        generated = restored
    cohort_labels = {"real_labels": labels[dev],
                     "gen_labels": np.tile(labels[dev], args.draws)} if cohorts else {}
    sampled = genotype_metrics(panel["calls"][dev], generated, offsets=panel["offsets"],
                               positions=panel["positions"], edges=decoder.bins.edges,
                               maf_bin_of_snp=panel["maf_bin_of_snp"], **cohort_labels)
    return {"nll_per_call": _dev_nll(decoder, calls, logits, labels, dev),
            "train_nll_per_call": decoder.train_nll_per_call, "converged": decoder.converged,
            "lambda_u": decoder.lambda_u, "lambda_a": decoder.lambda_a, "sampled": sampled}


def _fit(arm: str, panel: dict, calls: np.ndarray, logits: np.ndarray, labels: np.ndarray,
         lambda_u: float, lambda_a: float, n_bins: int):
    return fit_decoder(arm, calls, logits, labels, panel["train_indices"], panel["positions"],
                       panel["offsets"], N_COHORTS, lambda_u=lambda_u, lambda_a=lambda_a,
                       n_bins=n_bins)


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


def _named_results(arms: dict, controls: dict) -> list[tuple[str, dict]]:
    return [*arms.items(), *((f"{arm}_{control}", value) for control, block in controls.items()
                             for arm, value in block.items())]


def _ablation_rows(arms: dict, controls: dict) -> list[dict]:
    rows = []
    for name, result in _named_results(arms, controls):
        sampled = result["sampled"]
        rows.append({"arm": name, "nll_per_call": result["nll_per_call"],
                     **{key: sampled.get(key) for key in
                        ("af_mae", "cohort_af_mae", "genotype_proportion_tv",
                         "heterozygosity_mae", "ld_r2_mae", "local_dosage_covariance_mae")}})
    return rows


def _strata_rows(arms: dict, controls: dict, panel: dict) -> list[dict]:
    maf_sizes = np.bincount(panel["maf_bin_of_snp"], minlength=len(MAF_BIN_EDGES) + 1)
    strata = []
    for name, result in _named_results(arms, controls):
        sampled = result["sampled"]
        for cohort, value in sampled.get("cohort_af_mae_by_cohort", {}).items():
            strata.append({"arm": name, "stratum_type": "cohort", "stratum": cohort,
                           "n": sampled["cohort_n_real"][cohort], "value": value})
        for index, value in enumerate(sampled.get("af_mae_by_maf_bin", [])):
            strata.append({"arm": name, "stratum_type": "maf_bin", "stratum": index,
                           "n": int(maf_sizes[index]), "value": value})
        for index, value in enumerate(sampled["ld_r2_mae_by_bin"]):
            strata.append({"arm": name, "stratum_type": "distance_bin", "stratum": index,
                           "n": sampled["ld_pairs_by_bin"][index], "value": value})
    return strata


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_oracle(args: argparse.Namespace) -> dict:
    """Phase 0 freeze plus the Phase 1 B0-B3 dev comparison on real held-out factors."""
    prepared_dir, output = args.prepared_dir.resolve(), args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    panel = load_panel(prepared_dir)
    assertions = phase0_assertions(panel, args.n_bins)
    pilot_dir = args.pilot_dir.resolve() if args.pilot_dir else None
    _atomic_json(output / "study_manifest.json",
                 study_manifest(prepared_dir, panel, pilot_dir, assertions))

    calls, logits, labels = panel["calls"], panel["logits"], panel["labels"]
    bins = distance_bins(panel["positions"], panel["offsets"], args.n_bins)
    grid = []
    for lambda_u, lambda_a in itertools.product(args.lambda_grid, repeat=2):
        decoder = _fit("B3", panel, calls, logits, labels, lambda_u, lambda_a, args.n_bins)
        grid.append({"lambda_u": lambda_u, "lambda_a": lambda_a, "converged": decoder.converged,
                     "dev_nll_per_call": _dev_nll(decoder, calls, logits, labels,
                                                  panel["dev_indices"])})
    chosen = min(grid, key=lambda entry: entry["dev_nll_per_call"])

    arms = {}
    for arm in ("B0", "B1", "B2", "B3"):
        decoder = _fit(arm, panel, calls, logits, labels, chosen["lambda_u"], chosen["lambda_a"],
                       args.n_bins)
        decoder.save(output / f"decoder_{arm}.npz")
        arms[arm] = _evaluate_arm(decoder, panel, calls, logits, labels, args)

    order = np.arange(len(panel["positions"]))
    order_rng = np.random.default_rng(args.seed + 1)
    for start, end in zip(panel["offsets"][:-1], panel["offsets"][1:]):
        order[start:end] = start + order_rng.permutation(int(end - start))
    shuffled_labels = labels[np.random.default_rng(args.seed + 2).permutation(len(labels))]
    controls = {"order_shuffle": {}, "label_shuffle": {}}
    for arm in ("B2", "B3"):
        decoder = _fit(arm, panel, calls[:, order], logits[:, order], labels, chosen["lambda_u"],
                       chosen["lambda_a"], args.n_bins)
        controls["order_shuffle"][arm] = _evaluate_arm(
            decoder, panel, calls[:, order], logits[:, order], labels, args, columns=order)
    for arm in ("B1", "B3"):
        decoder = _fit(arm, panel, calls, logits, shuffled_labels, chosen["lambda_u"],
                       chosen["lambda_a"], args.n_bins)
        controls["label_shuffle"][arm] = _evaluate_arm(
            decoder, panel, calls, logits, shuffled_labels, args, cohorts=False)

    gate1 = _gate1(arms)
    order_b3, label_b1 = controls["order_shuffle"]["B3"], controls["label_shuffle"]["B1"]
    verdict = {
        "gate1": gate1,
        "order_control_supports_ld": bool(
            arms["B3"]["nll_per_call"] < order_b3["nll_per_call"]
            and arms["B3"]["sampled"]["ld_r2_mae"] < order_b3["sampled"]["ld_r2_mae"]),
        "label_control_supports_cohort": bool(
            arms["B1"]["nll_per_call"] < label_b1["nll_per_call"]),
    }
    results = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "prepared_dir": str(prepared_dir), "output_dir": str(output),
        "seed": args.seed, "draws": args.draws, "n_bins": args.n_bins,
        "lambda_grid": grid, "chosen_lambda": {"lambda_u": chosen["lambda_u"],
                                               "lambda_a": chosen["lambda_a"]},
        "distance_bins": {"edges": bins.edges.tolist(),
                          "adjacent_pairs_per_bin": np.bincount(
                              bins.bin_of_snp[bins.bin_of_snp >= 0],
                              minlength=len(bins.edges) + 1).tolist()},
        "arms": arms, "negative_controls": controls, **verdict,
        "limitation": "Oracle comparison on real held-out GLM-PCA factors; no diffusion samples",
    }
    _atomic_json(output / "oracle_results.json", results)
    ablation = _ablation_rows(arms, controls)
    _write_csv(output / "decoder_ablation.csv", ablation, list(ablation[0]))
    _write_csv(output / "strata.csv", _strata_rows(arms, controls, panel),
               ["arm", "stratum_type", "stratum", "n", "value"])
    print(json.dumps(verdict, sort_keys=True, allow_nan=False), flush=True)
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Genotype decoder metrics and the Phase 1 oracle comparison")
    commands = parser.add_subparsers(dest="command", required=True)
    oracle = commands.add_parser("oracle", help="Compare B0-B3 decoders on the dev split")
    oracle.add_argument("--prepared-dir", type=Path, required=True)
    oracle.add_argument("--output-dir", type=Path, required=True)
    oracle.add_argument("--pilot-dir", type=Path,
                        default=PROJECT_ROOT / "outputs/diagnostics/hipodit_multiseed_20260915")
    oracle.add_argument("--draws", type=int, default=20)
    oracle.add_argument("--seed", type=int, default=20260915)
    oracle.add_argument("--n-bins", type=int, default=DISTANCE_BINS)
    oracle.add_argument("--lambda-grid", nargs="+", type=float, default=[1.0, 10.0, 100.0])
    return parser


if __name__ == "__main__":
    run_oracle(build_parser().parse_args())
