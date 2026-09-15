"""Genotype metric arithmetic, the Phase 0 data contract and the oracle study manifest."""

from __future__ import annotations

import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.special import expit

SCRIPT = Path(__file__).parents[1] / "scripts" / "hipodit_genotype_check.py"
PANEL_DIR = Path("outputs/diagnostics/hipodit_fisher_20260915_unique")

# Two genes of five and three SNPs; edges split the within-gene distances into three bins.
OFFSETS = np.array([0, 5, 8])
POSITIONS = np.array([100.0, 110.0, 140.0, 200.0, 300.0, 1000.0, 1020.0, 1120.0])
EDGES = np.array([25.0, 75.0])
REAL = np.array([
    [0, 0, 2, 0, 1, 0, 1, 2],
    [0, 1, 2, 0, 1, 1, 1, 0],
    [0, 2, 2, 1, 1, 0, 0, 2],
    [0, 0, 2, 1, 1, 1, 0, 0],
    [1, 1, 2, 2, 0, 0, 1, 2],
    [1, 2, 2, 2, 0, 1, 1, 0],
    [2, 0, 2, 1, 2, 0, 0, 2],
    [2, 1, 2, 1, 2, 1, 0, 0],
], dtype=float)
GENERATED = np.array([
    [0, 0, 2, 0, 1, 0, 1, 2],
    [1, 0, 2, 1, 1, 0, 0, 0],
    [1, 1, 2, 1, 0, 1, 1, 2],
    [2, 1, 2, 2, 2, 1, 0, 0],
], dtype=np.int8)


def _metrics(**overrides):
    from scripts.hipodit_genotype_check import genotype_metrics

    arguments = dict(offsets=OFFSETS, positions=POSITIONS, edges=EDGES)
    arguments.update(overrides)
    return genotype_metrics(REAL, GENERATED, **arguments)


def _fake_prepared(prepared: Path, *, overlap: bool = False) -> None:
    """Two genes of four SNPs over sixteen individuals, shaped like the frozen panel."""
    prepared.mkdir(parents=True)
    rng = np.random.default_rng(4)
    calls = rng.integers(0, 3, size=(16, 8)).astype(np.float32)
    np.savez(prepared / "genotypes.npz", calls=calls, offsets=np.array([0, 4, 8]))
    validation = np.arange(4) if overlap else np.arange(8, 12)
    np.savez(prepared / "dataset.npz", x=rng.normal(size=(16, 2, 2)).astype(np.float32),
             y=np.repeat([3, 7, 11, 25], 4), train_indices=np.arange(8), val_indices=validation,
             test_indices=np.arange(12, 16))
    parameters = [{"gene": f"G{gene}", "loadings": rng.normal(scale=0.5, size=(4, 2)),
                   "intercept": rng.normal(scale=0.5, size=4), "family": "binomial",
                   "trials": 2, "link": "logit"} for gene in range(2)]
    with (prepared / "glm_pca_parameters.pkl").open("wb") as handle:
        pickle.dump(parameters, handle)
    with (prepared / "normalization_stats.pkl").open("wb") as handle:
        pickle.dump({"version": 1, "mean": np.zeros((2, 2), dtype=np.float32),
                     "std": np.ones((2, 2), dtype=np.float32), "fit_split": "train",
                     "clip": None}, handle)
    spacing = (0, 30, 100, 260)
    genes = [{"gene": f"G{gene}", "start": 1000 * gene, "end": 1000 * gene + 500,
              "variants": [{"id": f"17:{1000 * gene + step}", "position": 1000 * gene + step,
                            "ref": "A", "alt": "C", "train_maf": 0.02 + 0.15 * snp}
                           for snp, step in enumerate(spacing)]} for gene in range(2)]
    (prepared / "gene_variant_map.json").write_text(json.dumps({"chromosome": 17, "genes": genes}))
    (prepared / "prepare_report.json").write_text(json.dumps({
        "status": "prepared", "fit_split": "train", "glm_family": "binomial",
        "maf_filter": "train-only >= 0.01", "components": 2, "genes": 2, "glm_iterations": 1000,
        "seed": 20260327, "split_sizes": {"train": 8, "validation": 4, "test": 4}}))
    (prepared / "preprocessing_metadata.json").write_text(json.dumps({"family": "binomial"}))
    with (prepared / "label_hierarchy.pkl").open("wb") as handle:
        pickle.dump({"pop_to_superpop": {cohort: cohort % 5 for cohort in range(26)}}, handle)


def test_metrics_reproduce_hand_computed_marginals_and_binned_ld() -> None:
    # Given eight real and four generated individuals with hand-tabulated genotype columns.
    result = _metrics()

    # Then the marginal summaries match the arithmetic done by hand on those columns.
    assert result["af_mae"] == pytest.approx(0.0390625)
    assert result["heterozygosity_mae"] == pytest.approx(0.046875)
    assert result["genotype_proportion_tv"] == pytest.approx(0.0625)
    # And every within-gene pair at index offset 1, 2 or 4 is covered once.
    assert result["local_pairs_evaluated"] == 11
    # The monomorphic third SNP leaves four pairs without an r-squared, all in bins 1 and 2.
    assert result["ld_pairs_skipped"] == 4
    assert result["ld_pairs_by_bin"] == [2, 0, 5]
    # Bin 0 holds pairs (0,1) with r-squared 1/429 real against 1/2 generated, and (5,6) with 0.
    assert result["ld_r2_mae_by_bin"][0] == pytest.approx((0.5 - 1 / 429) / 2)
    assert result["ld_r2_mae_by_bin"][1] is None
    # And the overall MAE averages pairs, not bin means.
    pooled = (2 * result["ld_r2_mae_by_bin"][0] + 5 * result["ld_r2_mae_by_bin"][2]) / 7
    assert result["ld_r2_mae"] == pytest.approx(pooled)
    assert result["ld_r2_mae"] != pytest.approx(
        (result["ld_r2_mae_by_bin"][0] + result["ld_r2_mae_by_bin"][2]) / 2)
    assert result["valid_genotype_fraction"] == 1.0
    assert result["unique_individual_fraction"] == 1.0


def test_cohort_absent_from_the_generated_rows_is_not_estimable() -> None:
    # Given a real cohort 2 that the generator never produced.
    result = _metrics(real_labels=np.array([0, 0, 0, 1, 1, 1, 2, 2]),
                      gen_labels=np.array([0, 0, 1, 1]))

    # Then it is reported as not estimable and excluded from the pooled cohort AF MAE.
    assert result["cohorts_not_estimable"] == [2]
    assert set(result["cohort_af_mae_by_cohort"]) == {0, 1}
    assert result["cohort_n_real"] == {0: 3, 1: 3, 2: 2}
    cells = np.concatenate([
        np.abs(REAL[:3].sum(0) / 6 - GENERATED[:2].mean(0) / 2),
        np.abs(REAL[3:6].sum(0) / 6 - GENERATED[2:].mean(0) / 2)])
    assert result["cohort_af_mae"] == pytest.approx(cells.mean())


def test_maf_bins_average_only_the_snps_they_contain() -> None:
    # Given a MAF bin assignment that leaves the middle bin empty.
    result = _metrics(maf_bin_of_snp=np.array([0, 0, 2, 2, 2, 2, 2, 2]))

    # Then each bin averages its own SNPs and an empty bin is not estimable.
    assert result["af_mae_by_maf_bin"][0] == pytest.approx((0.125 + 0.1875) / 2)
    assert result["af_mae_by_maf_bin"][1] is None
    assert result["af_mae_by_maf_bin"][2] == pytest.approx(0.0)


def _base_logits(prepared: Path, factors: np.ndarray) -> np.ndarray:
    from scripts.hipodit_genotype_check import base_logits

    with (prepared / "glm_pca_parameters.pkl").open("rb") as handle:
        return base_logits(pickle.load(handle), factors)


def _split_factors(prepared: Path, split: str) -> np.ndarray:
    """The inverse-normalized latents of one split, the scale `train` hands the decoder."""
    with np.load(prepared / "dataset.npz") as data:
        return data["x"][data[f"{split}_indices"]].astype(np.float64)


def _fake_decoder_dir(prepared: Path, decoder_dir: Path, *, gate1_passed: bool = True) -> None:
    """A Task 8-shaped oracle output: a fitted arm-T decoder, prepared hashes and the verdict."""
    from scripts.hipodit_genotype_check import _digest, base_logits, panel_positions
    from src.models.genotype_decoder import fit_decoder

    decoder_dir.mkdir(parents=True)
    with np.load(prepared / "genotypes.npz") as data:
        calls, offsets = data["calls"].astype(np.float64), data["offsets"]
    with np.load(prepared / "dataset.npz") as data:
        factors, labels = data["x"].astype(np.float64), data["y"]
        train_indices = data["train_indices"]
    with (prepared / "glm_pca_parameters.pkl").open("rb") as handle:
        parameters = pickle.load(handle)
    logits = base_logits(parameters, factors)
    for arm, scope in (("T", "snp"), ("B0", "none")):
        fit_decoder(arm, calls, logits, labels, train_indices, panel_positions(prepared), offsets,
                    26, tilt_scope=scope).save(decoder_dir / f"decoder_{arm}.npz")
    (decoder_dir / "study_manifest.json").write_text(json.dumps(
        {"prepared_sha256": {path.name: _digest(path) for path in sorted(prepared.iterdir())}}))
    (decoder_dir / "oracle_results.json").write_text(
        json.dumps({"gate1_prime": {"passed": gate1_passed, "candidate": "T"}}))


def test_evaluate_genotypes_keeps_its_legacy_keys(tmp_path: Path) -> None:
    # Given the frozen prepared-directory layout and factors on the GLM-PCA scale.
    from scripts.hipodit_genotype_check import evaluate_genotypes

    prepared = tmp_path / "prepared"
    _fake_prepared(prepared)
    factors = np.load(prepared / "dataset.npz")["x"][:4].astype(np.float64)

    # When the legacy entry point decodes synthetic and reconstructed factors.
    report, calls = evaluate_genotypes(prepared, factors, factors, seed=3)

    # Then the keys the multiseed summary reads survive alongside the new ones.
    for block in ("synthetic", "real_latent_decoder_reference"):
        assert {"af_mae", "local_dosage_covariance_mae", "local_pairs_evaluated",
                "valid_genotype_fraction", "unique_individual_fraction"} <= set(report[block])
        assert report[block]["valid_genotype_fraction"] == 1.0
        assert report[block]["ld_r2_mae"] is not None
    assert (report["samples"], report["variants"]) == (4, 8)
    # And without a decoder directory only the baseline binomial arm is drawn, on its own stream.
    assert set(calls) == {"B0"} and calls["B0"].shape == (4, 8)
    np.testing.assert_array_equal(calls["B0"], np.random.default_rng(3).binomial(
        2, expit(_base_logits(prepared, factors))).astype(np.int8))


def test_evaluate_genotypes_decodes_arm_t_beside_the_unchanged_binomial_draw(tmp_path: Path) -> None:
    # Given a Gate 1'-passing arm-T decoder fitted on the same prepared panel.
    from scripts.hipodit_genotype_check import evaluate_genotypes
    from src.models.genotype_decoder import GenotypeDecoder, sample_calls

    prepared = tmp_path / "prepared"
    _fake_prepared(prepared)
    decoder_dir = tmp_path / "oracle"
    _fake_decoder_dir(prepared, decoder_dir)
    # Generated latents differ from the real ones they are scored against, as they do end to end.
    real = _split_factors(prepared, "val")
    synthetic = real + 0.3

    baseline, _ = evaluate_genotypes(prepared, synthetic, real, seed=3)
    report, calls = evaluate_genotypes(prepared, synthetic, real, seed=3, decoder_dir=decoder_dir)

    # Then arm T is reported beside the legacy blocks, which are untouched.
    assert report["synthetic"] == baseline["synthetic"]
    assert report["real_latent_decoder_reference"] == baseline["real_latent_decoder_reference"]
    assert {"synthetic_T", "real_latent_decoder_reference_T", "decoder_gate", "scale_check",
            "extremeness", "af_preservation"} <= set(report)
    assert set(report["synthetic_T"]) == set(report["synthetic"])
    # And its calls are diploid dosages the decoder drew from the synthetic logits, on a stream
    # of their own that leaves the baseline draws where they were.
    assert set(calls) == {"B0", "T"}
    assert calls["T"].dtype == np.int8 and calls["T"].shape == (4, 8)
    assert set(np.unique(calls["T"]).tolist()) <= {0, 1, 2}
    with np.load(prepared / "dataset.npz") as data:
        labels = data["y"][data["val_indices"]]
    decoder = GenotypeDecoder.load(decoder_dir / "decoder_T.npz")
    np.testing.assert_array_equal(calls["T"], sample_calls(
        decoder, _base_logits(prepared, synthetic), labels, np.random.default_rng(5)))
    assert report["synthetic_T"] != report["real_latent_decoder_reference_T"]
    # And the decoder that produced them is identified in full.
    assert (report["decoder_arm"], report["decoder_tilt_scope"]) == ("T", "snp")
    assert report["decoder_offset_gauge"] == "pooled_af"
    assert report["decoder_pooled_af_residual"] < 1e-8
    assert len(report["decoder_sha256"]) == 64
    assert set(report["decoder_lambda"]) == {"lambda_u", "lambda_tilt"}
    # And the latent scale is verified against the frozen panel before anything is decoded.
    assert report["scale_check"] == {"prepared_hashes_match": True,
                                     "real_factor_max_abs_diff": 0.0}
    # And the tails of each side's cohort-offset linear predictor are recorded, not mixed up.
    assert set(report["extremeness"]) == {"real", "synthetic", "frac_abs_gt_4_ratio"}
    assert set(report["extremeness"]["real"]) == {"frac_abs_gt_4", "frac_abs_gt_6", "mean_abs"}
    offset = decoder.cohort_offset[labels]
    for side, factors in (("real", real), ("synthetic", synthetic)):
        assert report["extremeness"][side]["mean_abs"] == pytest.approx(
            np.abs(_base_logits(prepared, factors) + offset).mean())
    # And arm T's promised allele frequency is measured against its own draw noise.
    assert set(report["af_preservation"]) == {"expected_af_vs_sampled_max_abs_diff",
                                              "draw_noise_3sigma", "within_draw_noise"}
    assert report["af_preservation"]["expected_af_vs_sampled_max_abs_diff"] == pytest.approx(
        np.abs(calls["T"].mean(axis=0) / 2
               - expit(_base_logits(prepared, synthetic) + offset).mean(axis=0)).max())
    assert set(report["decoder_gate"]) == {"nll_not_applicable", "heterozygosity_improved",
                                           "cohort_af_guardrail", "af_guardrail", "passed",
                                           "metrics"}
    # And the gate scores arm T against the baseline arm of this same run.
    assert report["decoder_gate"]["metrics"]["af_mae"] == {
        "T": report["synthetic_T"]["af_mae"], "B0": report["synthetic"]["af_mae"]}


def test_evaluate_genotypes_refuses_factors_off_the_inverse_normalized_scale(tmp_path: Path) -> None:
    # Given real factors that are not what inverting the stored normalization produces.
    from scripts.hipodit_genotype_check import evaluate_genotypes

    prepared = tmp_path / "prepared"
    _fake_prepared(prepared)
    decoder_dir = tmp_path / "oracle"
    _fake_decoder_dir(prepared, decoder_dir)
    factors = _split_factors(prepared, "val")

    with pytest.raises(ValueError, match="latent scale"):
        evaluate_genotypes(prepared, factors, 2 * factors + 1, seed=3, decoder_dir=decoder_dir)


def test_evaluate_genotypes_refuses_a_panel_that_changed_since_the_decoder_was_fitted(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Given a decoder whose manifest no longer describes the genotypes it was fitted on.
    import scripts.hipodit_genotype_check as module

    prepared = tmp_path / "prepared"
    _fake_prepared(prepared)
    decoder_dir = tmp_path / "oracle"
    _fake_decoder_dir(prepared, decoder_dir)
    manifest_path = decoder_dir / "study_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["prepared_sha256"]["genotypes.npz"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))
    factors = _split_factors(prepared, "val")

    def refuse(*args, **kwargs):
        raise AssertionError("the decoder must not sample once the scale check has failed")

    monkeypatch.setattr(module, "sample_calls", refuse)

    # Then the mismatching file is named and no call is drawn.
    with pytest.raises(ValueError, match="genotypes.npz"):
        module.evaluate_genotypes(prepared, factors, factors, seed=3, decoder_dir=decoder_dir)


def test_evaluate_genotypes_refuses_a_decoder_that_failed_gate1_prime(tmp_path: Path) -> None:
    from scripts.hipodit_genotype_check import evaluate_genotypes

    prepared = tmp_path / "prepared"
    _fake_prepared(prepared)
    decoder_dir = tmp_path / "oracle"
    _fake_decoder_dir(prepared, decoder_dir, gate1_passed=False)

    with pytest.raises(ValueError, match="Gate 1'"):
        evaluate_genotypes(prepared, _split_factors(prepared, "val"), _split_factors(prepared, "val"),
                           seed=3, decoder_dir=decoder_dir)


def test_evaluate_genotypes_reads_the_requested_split(tmp_path: Path) -> None:
    # Given a panel whose validation and test rows carry different cohort labels.
    from scripts.hipodit_genotype_check import evaluate_genotypes

    prepared = tmp_path / "prepared"
    _fake_prepared(prepared)

    validation, _ = evaluate_genotypes(prepared, _split_factors(prepared, "val"),
                                       _split_factors(prepared, "val"), seed=3)
    test, _ = evaluate_genotypes(prepared, _split_factors(prepared, "test"),
                                 _split_factors(prepared, "test"), seed=3, split="test")

    # Then the cohorts scored are the ones belonging to the split that was asked for.
    assert set(validation["synthetic"]["cohort_af_mae_by_cohort"]) == {11}
    assert set(test["synthetic"]["cohort_af_mae_by_cohort"]) == {25}


@pytest.mark.parametrize(("candidate", "failing"), [
    ((0.05, 0.9, 0.9), []),
    ((0.05, 1.05, 1.0), []),
    ((0.05, 1.0500001, 1.0), ["cohort_af_guardrail"]),
    ((0.05, 1.0, 1.05), []),
    ((0.05, 1.0, 1.0500001), ["af_guardrail"]),
    ((0.1, 1.0, 1.0), ["heterozygosity_improved"]),
    ((0.05, None, 1.0), ["cohort_af_guardrail"]),
], ids=["all-improve", "cohort-at-1.05", "cohort-past-1.05", "af-at-1.05", "af-past-1.05",
        "het-tie", "cohort-not-estimable"])
def test_gate2_prime_needs_a_het_gain_inside_both_af_guardrails(candidate, failing) -> None:
    from scripts.hipodit_genotype_check import gate2_prime

    # Given B0 with heterozygosity MAE 0.1 and unit cohort and overall AF MAE on generated latents.
    names = ("heterozygosity_mae", "cohort_af_mae", "af_mae")
    gate = gate2_prime({**dict(zip(names, candidate)), "ld_r2_mae": 0.5},
                       {"heterozygosity_mae": 0.1, "cohort_af_mae": 1.0, "af_mae": 1.0,
                        "ld_r2_mae": 0.7})

    # Then T passes only when all three conditions hold; NLL never takes part.
    assert gate["nll_not_applicable"] is True
    failed = [name for name in ("heterozygosity_improved", "cohort_af_guardrail", "af_guardrail")
              if not gate[name]]
    assert failed == failing
    assert gate["passed"] is (not failing)
    # And the raw values of all four metrics are carried, LD r2 descriptively.
    assert set(gate["metrics"]) == {*names, "ld_r2_mae"}
    assert gate["metrics"]["ld_r2_mae"] == {"T": 0.5, "B0": 0.7}


def test_oracle_refuses_overlapping_splits_before_doing_any_work(tmp_path: Path) -> None:
    # Given a prepared directory whose dev rows are also training rows.
    prepared = tmp_path / "prepared"
    _fake_prepared(prepared, overlap=True)
    output = tmp_path / "oracle"

    # When the oracle runs.
    result = subprocess.run([sys.executable, str(SCRIPT), "oracle", "--prepared-dir", str(prepared),
                             "--output-dir", str(output), "--draws", "2"],
                            capture_output=True, text=True, check=False)

    # Then the split contract stops it before any artifact beyond the output directory exists.
    assert result.returncode != 0
    assert "disjoint" in result.stderr
    assert list(output.iterdir()) == []


def _oracle(prepared: Path, output: Path, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPT), "oracle", "--prepared-dir", str(prepared),
                           "--output-dir", str(output), "--draws", "2",
                           "--pilot-dir", str(output.parent / "no_pilot"), *extra],
                          capture_output=True, text=True, check=False)


def test_oracle_writes_a_hashed_study_manifest_and_the_gate1_prime_study(tmp_path: Path) -> None:
    # Given a prepared directory and no pilot experiment to link.
    from src.models.genotype_decoder import GenotypeDecoder

    prepared = tmp_path / "prepared"
    _fake_prepared(prepared)
    output = tmp_path / "oracle"

    # When the default (tilt) arm set runs over two metric seeds.
    result = _oracle(prepared, output, "--metric-seeds", "2")
    assert result.returncode == 0, result.stderr

    # Then every prepared file is fingerprinted and the frozen baseline and revision are declared.
    manifest = json.loads((output / "study_manifest.json").read_text())
    assert set(manifest["prepared_sha256"]) == {path.name for path in prepared.iterdir()}
    assert all(len(digest) == 64 for digest in manifest["prepared_sha256"].values())
    assert manifest["latent_generator_baseline"] == "standard"
    assert manifest["pilot"] is None
    assert manifest["decoder_inputs_from_test"] is False
    assert "splits_disjoint_and_complete" in manifest["assertions_passed"]
    assert (manifest["snp_count"], manifest["gene_count"]) == (8, 2)
    assert set(manifest["source_sha256"]) == {"src/models/genotype_decoder.py",
                                              "scripts/hipodit_genotype_check.py"}
    assert {key: manifest[key] for key in ("arm_set", "lambda_u", "lambda_tilt", "lambda_a",
                                           "metric_seeds", "draws", "plan_revision", "gate")} == {
        "arm_set": "t", "lambda_u": 1.0, "lambda_tilt": 1.0, "lambda_a": 1.0, "metric_seeds": 2,
        "draws": 2, "plan_revision": "rev2-§9", "gate": "gate1_prime"}

    # And each pre-registered arm carries per-seed metrics, their mean/SD and its tilt scope.
    results = json.loads((output / "oracle_results.json").read_text())
    scopes = {"B0": "none", "B1": "none", "T": "snp", "T_superpop": "superpop",
              "T_cohort": "cohort"}
    assert {arm: block["tilt_scope"] for arm, block in results["arms"].items()} == scopes
    for arm, block in results["arms"].items():
        assert len(block["per_seed"]) == 2
        assert set(block["summary"]["af_mae"]) == {"mean", "sample_sd"}
        assert block["summary"]["af_mae"]["mean"] == pytest.approx(
            np.mean([seed["af_mae"] for seed in block["per_seed"]]))
        assert GenotypeDecoder.load(output / f"decoder_{arm}.npz").tilt_scope == scopes[arm]
        # Only tilted arms carry the AF-preservation check, over 4 dev rows x 2 draws x 2 seeds.
        assert ("af_preservation" in block) == arm.startswith("T")
        if arm.startswith("T"):
            check = block["af_preservation"]
            expected = np.array(check["expected_af"])
            assert (len(expected), check["n_samples"]) == (8, 16)
            assert check["draw_noise_3sigma"] == pytest.approx(
                (3 * np.sqrt(expected * (1 - expected) / 32)).max())
            assert check["within_draw_noise"] == (
                check["sampled_af_max_abs_diff"] <= check["draw_noise_3sigma"])

    # And only the label-shuffle control runs, without cohort metrics.
    controls = results["negative_controls"]
    assert set(controls) == {"label_shuffle"} and set(controls["label_shuffle"]) == {"B1", "T"}
    assert all(set(block["summary"]) == {"af_mae", "heterozygosity_mae"}
               for block in controls["label_shuffle"].values())
    assert results["label_control_supports_cohort"] == (
        results["arms"]["B1"]["nll_per_call"] < controls["label_shuffle"]["B1"]["nll_per_call"])

    # And the Gate 1' verdict is complete and is the last line printed.
    gate = results["gate1_prime"]
    assert len(gate["reasons"]) == 4 and gate["metric_seeds"] == 2
    assert gate["candidate"] == ("T" if gate["passed"] else "B0")
    assert {"ld_r2_mae", "ld_r2_mae_by_bin"} <= set(gate["descriptive_only"])
    assert json.loads(result.stdout.strip().splitlines()[-1]) == {
        "gate1_prime": gate,
        "label_control_supports_cohort": results["label_control_supports_cohort"]}
    with (output / "decoder_ablation.csv").open() as handle:
        header, *rows = handle.read().splitlines()
    assert header == ("arm,nll_per_call,af_mae,cohort_af_mae,genotype_proportion_tv,"
                      "heterozygosity_mae,ld_r2_mae,local_dosage_covariance_mae")
    assert [row.split(",")[0] for row in rows] == [*scopes, "B1_label_shuffle", "T_label_shuffle"]
    assert (output / "strata.csv").read_text().startswith("arm,stratum_type,stratum,n,value\n")


def test_legacy_oracle_keeps_its_arms_controls_and_gate1_keys(tmp_path: Path) -> None:
    # Given the fake panel run through the legacy arm set at a fixed lambda.
    prepared = tmp_path / "prepared"
    _fake_prepared(prepared)
    output = tmp_path / "oracle"

    result = _oracle(prepared, output, "--arms", "legacy")
    assert result.returncode == 0, result.stderr

    # Then the B0-B3 study, both controls and Gate 1 keep their previous shape, with no lambda grid.
    results = json.loads((output / "oracle_results.json").read_text())
    assert set(results["arms"]) == {"B0", "B1", "B2", "B3"}
    assert all("sampled" in block and "per_seed" not in block for block in results["arms"].values())
    assert {name: set(block) for name, block in results["negative_controls"].items()} == {
        "order_shuffle": {"B2", "B3"}, "label_shuffle": {"B1", "B3"}}
    assert set(results["gate1"]) == {"passed", "fixed_candidate", "reasons"}
    assert "lambda_grid" not in results
    verdict = json.loads(result.stdout.strip().splitlines()[-1])
    assert set(verdict) == {"gate1", "order_control_supports_ld", "label_control_supports_cohort"}
    manifest = json.loads((output / "study_manifest.json").read_text())
    assert (manifest["arm_set"], manifest["gate"], manifest["metric_seeds"]) == (
        "legacy", "gate1", 1)


def test_superpop_map_must_name_every_cohort(tmp_path: Path) -> None:
    from scripts.hipodit_genotype_check import load_superpop_of_cohort

    def write(mapping: dict) -> Path:
        prepared = tmp_path / f"prepared_{len(list(tmp_path.iterdir()))}"
        prepared.mkdir()
        with (prepared / "label_hierarchy.pkl").open("wb") as handle:
            pickle.dump({"pop_to_superpop": mapping}, handle)
        return prepared

    # Given a hierarchy that names all 26 cohorts, the map is read in cohort order.
    complete = {cohort: cohort % 5 for cohort in range(26)}
    np.testing.assert_array_equal(load_superpop_of_cohort(write(complete)), np.arange(26) % 5)
    # A missing cohort or string keys are refused.
    with pytest.raises(ValueError):
        load_superpop_of_cohort(write({cohort: 0 for cohort in range(25)}))
    with pytest.raises(ValueError):
        load_superpop_of_cohort(write({str(cohort): 0 for cohort in range(26)}))


def test_generated_covariance_reuses_the_real_mask_only_when_row_counts_match() -> None:
    from scripts.hipodit_genotype_check import genotype_metrics

    # Given one within-gene pair whose last two real rows each miss a call.
    real = np.array([[0, 0], [1, 1], [2, 2], [0, 1], [np.nan, 0], [2, np.nan]])
    paired = np.array([[0, 0], [1, 1], [2, 2], [0, 1], [2, 0], [0, 2]], dtype=np.int8)
    options = dict(offsets=np.array([0, 2]), positions=np.array([100.0, 110.0]), edges=np.zeros(0))

    # When generated rows pair one-to-one with real rows, the same four rows are compared:
    # both covariances are 0.5, so the error is zero (all six generated rows would give 0.5).
    paired_error = genotype_metrics(real, paired, **options)["local_dosage_covariance_mae"]
    assert paired_error == pytest.approx(0.0)
    # When the row counts differ, every generated row is used: |0.5 - 2/3|.
    pooled_error = genotype_metrics(real, paired[:3], **options)["local_dosage_covariance_mae"]
    assert pooled_error == pytest.approx(1 / 6)


def test_seed_summary_reports_leafwise_mean_and_sample_sd() -> None:
    from scripts.hipodit_genotype_check import seed_summary

    # Given two seeds of scalar, per-bin and per-cohort metrics with bins not always estimable.
    per_seed = [{"af_mae": 1.0, "ld_r2_mae_by_bin": [2.0, None, 1.0], "by_cohort": {3: 0.5}},
                {"af_mae": 3.0, "ld_r2_mae_by_bin": [4.0, None, None], "by_cohort": {3: 1.5}}]

    summary = seed_summary(per_seed)

    # Then every numeric leaf becomes a mean and a ddof=1 SD; a leaf missing in any seed is None.
    assert summary["af_mae"] == {"mean": 2.0, "sample_sd": pytest.approx(np.sqrt(2))}
    assert summary["ld_r2_mae_by_bin"] == [{"mean": 3.0, "sample_sd": pytest.approx(np.sqrt(2))},
                                           None, None]
    assert summary["by_cohort"] == {3: {"mean": 1.0, "sample_sd": pytest.approx(np.sqrt(0.5))}}
    # And a single seed has no SD.
    assert seed_summary(per_seed[:1])["af_mae"] == {"mean": 1.0}


def test_af_preservation_compares_the_largest_gap_with_the_largest_draw_noise() -> None:
    from scripts.hipodit_genotype_check import af_preservation

    # Given 200 sampled individuals: 3 sigma is 3*sqrt(.25/400) = .075 at p=.5, .045 at p=.1.
    expected = np.array([0.5, 0.1, 0.02])
    within = af_preservation(expected, np.array([0.52, 0.1, 0.03]), 200)
    beyond = af_preservation(expected, np.array([0.5, 0.2, 0.02]), 200)

    assert within["sampled_af_max_abs_diff"] == pytest.approx(0.02)
    assert within["draw_noise_3sigma"] == pytest.approx(0.075)
    assert within["within_draw_noise"] is True
    assert within["expected_af"] == pytest.approx([0.5, 0.1, 0.02])
    # A 0.1 gap exceeds the panel-wide maximum noise even though it sits at a p=.1 SNP.
    assert beyond["sampled_af_max_abs_diff"] == pytest.approx(0.1)
    assert beyond["within_draw_noise"] is False


def _tilt_arm(nll: float, *seeds: tuple) -> dict:
    from scripts.hipodit_genotype_check import seed_summary

    names = ("heterozygosity_mae", "cohort_af_mae", "af_mae")
    per_seed = [dict(zip(names, seed)) for seed in seeds]
    return {"nll_per_call": nll, "per_seed": per_seed, "summary": seed_summary(per_seed)}


@pytest.mark.parametrize(("nll", "seed", "failing"), [
    (0.1, (0.05, 0.9, 0.9), []),
    (0.1, (0.05, 1.05, 1.0), []),
    (0.1, (0.05, 1.0500001, 1.0), ["cohort_af_mae"]),
    (0.1, (0.05, 1.0, 1.05), []),
    (0.1, (0.05, 1.0, 1.0500001), ["af_mae"]),
    (0.2, (0.05, 1.0, 1.0), ["nll_per_call"]),
    (0.1, (0.1, 1.0, 1.0), ["heterozygosity_mae"]),
    (0.1, (0.05, None, 1.0), ["cohort_af_mae"]),
], ids=["all-improve", "cohort-at-1.05", "cohort-past-1.05", "af-at-1.05", "af-past-1.05",
        "nll-tie", "het-tie", "cohort-not-estimable"])
def test_gate1_prime_needs_all_four_conditions_at_exact_boundaries(nll, seed, failing) -> None:
    from scripts.hipodit_genotype_check import _gate1_prime

    # Given B0 with NLL 0.2, heterozygosity MAE 0.1 and unit cohort and overall AF MAE.
    gate = _gate1_prime({"B0": _tilt_arm(0.2, (0.1, 1.0, 1.0)), "T": _tilt_arm(nll, seed)})

    # Then T passes only when no condition fails, and each failing condition is named.
    assert len(gate["reasons"]) == 4
    failed = [reason.split(":")[0] for reason in gate["reasons"] if reason.endswith("fail")]
    assert failed == failing
    assert gate["passed"] is (not failing)
    assert gate["candidate"] == ("B0" if failing else "T")
    assert (gate["pass_seeds"], gate["metric_seeds"]) == (0 if failing else 1, 1)


def test_gate1_prime_judges_the_seed_mean_and_counts_passing_seeds() -> None:
    from scripts.hipodit_genotype_check import _gate1_prime

    # Given a T whose second seed alone exceeds the cohort AF margin while the mean stays inside.
    arms = {"B0": _tilt_arm(0.2, (0.1, 1.0, 1.0), (0.1, 1.0, 1.0)),
            "T": _tilt_arm(0.1, (0.05, 0.9, 1.0), (0.05, 1.1, 1.0))}

    gate = _gate1_prime(arms)

    # Then the mean decides the verdict and the per-seed record shows one of two seeds passing.
    assert gate["passed"] is True and gate["candidate"] == "T"
    assert (gate["pass_seeds"], gate["metric_seeds"]) == (1, 2)


def _legacy_arm(nll, ld, cohort) -> dict:
    return {"nll_per_call": nll, "sampled": {"ld_r2_mae": ld, "cohort_af_mae": cohort}}


@pytest.mark.parametrize(("candidate", "passed"), [
    (_legacy_arm(0.1, 0.5, 1.05), True),
    (_legacy_arm(0.1, 0.5, 1.0500001), False),
    (_legacy_arm(0.2, 0.5, 1.0), False),
    (_legacy_arm(0.1, 1.0, 1.0), False),
], ids=["cohort-at-1.05", "cohort-past-1.05", "nll-tie", "ld-tie"])
def test_gate1_holds_b3_to_strict_nll_and_ld_gains_within_cohort_margin(candidate, passed) -> None:
    from scripts.hipodit_genotype_check import _gate1

    # Given B0 with NLL 0.2, LD r2 MAE 1.0 and unit cohort AF MAE.
    gate = _gate1({"B0": _legacy_arm(0.2, 1.0, 1.0), "B3": candidate})

    assert gate["passed"] is passed
    assert gate["fixed_candidate"] == ("B3" if passed else "B0")
    assert len(gate["reasons"]) == 3


def test_gate1_keeps_b0_when_ld_or_cohort_af_is_not_estimable() -> None:
    from scripts.hipodit_genotype_check import _gate1

    gate = _gate1({"B0": _legacy_arm(0.2, 1.0, 1.0), "B3": _legacy_arm(0.1, None, 1.0)})

    assert (gate["passed"], gate["fixed_candidate"]) == (False, "B0")
    assert gate["reasons"] == ["LD r2 or cohort AF was not estimable on dev"]

@pytest.mark.skipif(not PANEL_DIR.exists(), reason="prepared chr17 panel is unavailable")
def test_real_panel_distance_bin_edges_are_the_frozen_quartiles() -> None:
    # Given the frozen chr17 panel read the way the oracle reads it.
    from scripts.hipodit_genotype_check import panel_positions
    from src.models.genotype_decoder import distance_bins

    positions = panel_positions(PANEL_DIR)
    with np.load(PANEL_DIR / "genotypes.npz") as data:
        offsets = data["offsets"]

    # Then the quartile edges match the ones the decoder contract froze.
    np.testing.assert_array_equal(distance_bins(positions, offsets, 4).edges, [35, 104, 210])


def test_population_classifier_separates_cohorts_and_names_the_ones_it_cannot_score() -> None:
    # Given four cohorts whose dosage blocks are disjoint, so the cohort is readable from the row.
    from scripts.hipodit_genotype_check import population_classifier_scores

    train_calls = np.repeat(np.eye(4, dtype=np.float64) * 2, 5, axis=0)
    train_labels = np.repeat([0, 1, 2, 3], 5)

    # When the classifier fitted on those real rows scores a block holding every cohort.
    complete = population_classifier_scores(train_calls, train_labels, train_calls[::5],
                                            np.array([0, 1, 2, 3]))

    # Then it recovers each cohort and excludes none from the macro one-vs-rest AUC.
    assert complete["classifier_accuracy"] == pytest.approx(1.0)
    assert complete["classifier_macro_auc"] == pytest.approx(1.0)
    assert complete["cohorts_excluded_from_auc"] == []

    # And a block missing two cohorts is scored without them, with the exclusions reported.
    partial = population_classifier_scores(train_calls, train_labels, train_calls[::5][:2],
                                           np.array([0, 1]))
    assert partial["cohorts_excluded_from_auc"] == [2, 3]
    assert partial["classifier_macro_auc"] == pytest.approx(1.0)
    assert partial["n_eval"] == 2


def test_population_classifier_refuses_a_cohort_it_was_never_trained_on() -> None:
    # Given an evaluation block conditioned on a cohort absent from the real training rows.
    from scripts.hipodit_genotype_check import population_classifier_scores

    train_calls = np.repeat(np.eye(4, dtype=np.float64) * 2, 5, axis=0)
    # When/Then the score is refused rather than silently reported against a shifted label set.
    with pytest.raises(ValueError, match="absent"):
        population_classifier_scores(train_calls, np.repeat([0, 1, 2, 3], 5),
                                     train_calls[:1], np.array([9]))


def test_test_nll_comparison_matches_nll_calls_and_brackets_its_difference(tmp_path: Path) -> None:
    # Given a frozen panel and the Gate 1'-passing oracle decoders fitted on it.
    from scripts.hipodit_genotype_check import test_nll_comparison
    from src.models.genotype_decoder import GenotypeDecoder, nll_calls

    prepared, decoder_dir = tmp_path / "prepared", tmp_path / "oracle"
    _fake_prepared(prepared)
    _fake_decoder_dir(prepared, decoder_dir)

    # When the deterministic oracle NLL of both arms is compared on the held-out test split.
    result = test_nll_comparison(prepared, decoder_dir, split="test")

    # Then each arm reproduces the per-call NLL of its own decoder on the test rows.
    with np.load(prepared / "genotypes.npz") as data:
        calls = data["calls"].astype(np.float64)
    with np.load(prepared / "dataset.npz") as data:
        indices, labels = data["test_indices"], data["y"]
        factors = data["x"].astype(np.float64)
    logits = _base_logits(prepared, factors)
    for arm, key in (("B0", "b0_nll"), ("T", "t_nll")):
        decoder = GenotypeDecoder.load(decoder_dir / f"decoder_{arm}.npz")
        expected = np.nanmean(nll_calls(decoder, calls[indices], logits[indices], labels[indices]))
        assert result[key] == pytest.approx(float(expected))
    # And the individual-resampled interval brackets the difference it was built from.
    assert result["difference"] == pytest.approx(result["t_nll"] - result["b0_nll"])
    assert result["ci_low"] <= result["difference"] <= result["ci_high"]
    assert result["ci_low"] < result["ci_high"]
    assert (result["n_individuals"], result["n_calls"]) == (len(indices), len(indices) * 8)
    assert result["resampling_unit"] == "held-out individuals"
    # And the interval is reproducible, because the bootstrap generator is seeded.
    assert test_nll_comparison(prepared, decoder_dir, split="test") == result


def test_test_nll_comparison_refuses_a_decoder_that_failed_gate1_prime(tmp_path: Path) -> None:
    # Given an oracle directory whose Gate 1' verdict did not pass.
    from scripts.hipodit_genotype_check import test_nll_comparison

    prepared, decoder_dir = tmp_path / "prepared", tmp_path / "oracle"
    _fake_prepared(prepared)
    _fake_decoder_dir(prepared, decoder_dir, gate1_passed=False)
    # When/Then the test split is not opened with a decoder that never earned the comparison.
    with pytest.raises(ValueError, match="Gate 1'"):
        test_nll_comparison(prepared, decoder_dir, split="test")


def test_classifier_blocks_reads_the_real_train_and_eval_dosages(tmp_path: Path) -> None:
    # Given the frozen panel layout.
    from scripts.hipodit_genotype_check import classifier_blocks

    prepared = tmp_path / "prepared"
    _fake_prepared(prepared)

    # When the classifier inputs for the first two test individuals are read.
    blocks = classifier_blocks(prepared, "test", count=2)

    # Then the fitting rows are the real train block and the eval rows the requested test prefix.
    with np.load(prepared / "genotypes.npz") as data:
        calls = data["calls"].astype(np.float64)
    with np.load(prepared / "dataset.npz") as data:
        train, test, labels = data["train_indices"], data["test_indices"], data["y"]
    np.testing.assert_array_equal(blocks["train_calls"], calls[train])
    np.testing.assert_array_equal(blocks["train_labels"], labels[train])
    np.testing.assert_array_equal(blocks["eval_calls"], calls[test[:2]])
    np.testing.assert_array_equal(blocks["eval_labels"], labels[test[:2]])

    # And a panel with missing calls is refused rather than imputed behind the classifier's back.
    damaged = calls.copy()
    damaged[0, 0] = np.nan
    np.savez(prepared / "genotypes.npz", calls=damaged, offsets=np.array([0, 4, 8]))
    with pytest.raises(ValueError, match="missing calls"):
        classifier_blocks(prepared, "test")
