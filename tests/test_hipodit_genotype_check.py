"""Genotype metric arithmetic, the Phase 0 data contract and the oracle study manifest."""

from __future__ import annotations

import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

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


def test_evaluate_genotypes_keeps_its_legacy_keys(tmp_path: Path) -> None:
    # Given the frozen prepared-directory layout and factors on the GLM-PCA scale.
    from scripts.hipodit_genotype_check import evaluate_genotypes

    prepared = tmp_path / "prepared"
    _fake_prepared(prepared)
    factors = np.load(prepared / "dataset.npz")["x"][:4].astype(np.float64)

    # When the legacy entry point decodes synthetic and reconstructed factors.
    report, synthetic = evaluate_genotypes(prepared, factors, factors, seed=3)

    # Then the keys the multiseed summary reads survive alongside the new ones.
    for block in ("synthetic", "real_latent_decoder_reference"):
        assert {"af_mae", "local_dosage_covariance_mae", "local_pairs_evaluated",
                "valid_genotype_fraction", "unique_individual_fraction"} <= set(report[block])
        assert report[block]["valid_genotype_fraction"] == 1.0
        assert report[block]["ld_r2_mae"] is not None
    assert (report["samples"], report["variants"]) == (4, 8)
    assert synthetic.shape == (4, 8)


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
