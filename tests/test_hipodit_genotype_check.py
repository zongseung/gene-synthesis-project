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


def test_oracle_writes_a_hashed_study_manifest_without_a_pilot(tmp_path: Path) -> None:
    # Given a prepared directory and no pilot experiment to link.
    prepared = tmp_path / "prepared"
    _fake_prepared(prepared)
    output = tmp_path / "oracle"

    # When the oracle runs over it.
    result = subprocess.run([sys.executable, str(SCRIPT), "oracle", "--prepared-dir", str(prepared),
                             "--output-dir", str(output), "--draws", "2", "--lambda-grid", "1",
                             "--pilot-dir", str(tmp_path / "no_pilot")],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr

    # Then every prepared file is fingerprinted and the frozen baseline is declared.
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
    assert json.loads(result.stdout.strip().splitlines()[-1])["gate1"]["fixed_candidate"] in {"B0", "B3"}


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
