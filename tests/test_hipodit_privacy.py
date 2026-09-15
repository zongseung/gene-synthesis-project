"""Phase 4 privacy distances, the membership-inference direction, Gate 4 boundaries and strata."""

from __future__ import annotations

import csv
import hashlib
import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "hipodit_privacy.py"

# Two train individuals three SNPs apart; the synthetic set repeats one of them exactly.
TRAIN = np.array([[0, 0, 0], [2, 2, 2]], dtype=np.int8)
SYNTHETIC = np.array([[0, 0, 0], [1, 0, 0], [2, 2, 1], [0, 0, 0]], dtype=np.int8)
TEST = np.array([[1, 1, 1]], dtype=np.int8)
THIRD = 1.0 / 3.0


def privacy():
    from scripts import hipodit_privacy

    return hipodit_privacy


def test_distances_duplicate_rates_match_a_hand_calculation():
    # Given the hand-built panel above, whose nearest-train distances are 0, 1/3, 1/3 and 0.
    module = privacy()

    # When the arm is measured with a near-duplicate threshold strictly above the exact matches.
    metrics, nearest = module.arm_privacy(SYNTHETIC, TRAIN, TEST, near_threshold=0.5)

    assert nearest == pytest.approx([0.0, THIRD, THIRD, 0.0])
    assert metrics["nn_distance_to_train"]["min"] == pytest.approx(0.0)
    assert metrics["nn_distance_to_train"]["median"] == pytest.approx(1.0 / 6.0)
    assert metrics["nn_distance_to_train"]["mean"] == pytest.approx(1.0 / 6.0)
    # Two of four synthetic rows equal a train row exactly, and three of four rows are unique.
    assert metrics["exact_duplicate_rate_train"] == pytest.approx(0.5)
    assert metrics["near_duplicate_rate_train"] == pytest.approx(1.0)
    assert metrics["self_duplicate_rate"] == pytest.approx(0.25)
    # And the attack is wired members-to-train, non-members-to-test: the two train rows score 0 and
    # 1/3 against the synthetic set, the one test row scores 2/3, so the AUC is 1.0. Swapping the
    # two arguments would silently give 0.0.
    assert metrics["membership_inference_auc"] == pytest.approx(1.0)

    # And the near-duplicate count is strict: at a threshold of exactly 1/3 only the exact ones pass.
    strict, _ = module.arm_privacy(SYNTHETIC, TRAIN, TEST, near_threshold=THIRD)
    assert strict["near_duplicate_rate_train"] == pytest.approx(0.5)


def test_membership_inference_auc_points_the_leaking_way():
    # Given members that sit on top of the synthetic set and a non-member that does not.
    module = privacy()
    synthetic = np.array([[0, 0, 0], [2, 2, 2]], dtype=np.int8)
    members = np.array([[0, 0, 0], [2, 2, 2]], dtype=np.int8)
    nonmembers = np.array([[1, 1, 1]], dtype=np.int8)

    # When each individual is scored by its minimum distance to the synthetic set.
    leaking = module.membership_inference_auc(
        module.mean_abs_dosage_distance(members, synthetic).min(axis=1),
        module.mean_abs_dosage_distance(nonmembers, synthetic).min(axis=1))

    # Then small distance means member: perfect leakage is 1.0, never 0.0.
    assert leaking == pytest.approx(1.0)

    # And swapping the roles inverts it, so an accidental sign flip cannot read as a clean result.
    inverted = module.membership_inference_auc(
        module.mean_abs_dosage_distance(nonmembers, synthetic).min(axis=1),
        module.mean_abs_dosage_distance(members, synthetic).min(axis=1))
    assert inverted == pytest.approx(0.0)


def test_membership_inference_auc_is_near_chance_for_one_distribution():
    # Given members and non-members drawn from the same distribution as the synthetic set.
    module = privacy()
    rng = np.random.default_rng(20260915)
    synthetic = rng.integers(0, 3, size=(251, 361))
    members = rng.integers(0, 3, size=(2002, 361))
    nonmembers = rng.integers(0, 3, size=(251, 361))

    # When the same attack is run.
    auc = module.membership_inference_auc(
        module.mean_abs_dosage_distance(members, synthetic).min(axis=1),
        module.mean_abs_dosage_distance(nonmembers, synthetic).min(axis=1))

    # Then it lands on chance, the 0.5 reference the gate is read against.
    assert 0.4 < auc < 0.6


def _summary(auc: float, exact: float, near: float, p1: float) -> dict:
    return {"membership_inference_auc": auc, "exact_duplicate_rate_train": exact,
            "near_duplicate_rate_train": near, "nn_distance_to_train_p1": p1}


def test_gate4_passes_on_every_boundary_and_fails_just_past_it():
    # Given a B0 arm whose four limits are 0.5 + margin, max(0.004, 1/251), 1.5 x 0.02 and 0.8 x 0.1.
    module = privacy()
    b0 = _summary(auc=0.5, exact=0.004, near=0.02, p1=0.1)
    boundary = _summary(auc=0.5 + module.AUC_MARGIN, exact=0.004,
                        near=module.NEAR_DUPLICATE_FACTOR * 0.02, p1=module.P1_FACTOR * 0.1)

    # When arm T sits exactly on all four limits.
    verdict = module.gate4(b0, boundary, n_synthetic=251)

    # Then Gate 4 passes; the conditions are inclusive.
    assert verdict["passed"] is True
    assert all(verdict[name] for name in module.GATE4_CONDITIONS)

    # And moving any single condition past its limit fails that one and the gate.
    for name, worse in (
        ("membership_inference_auc_margin", {"auc": 0.5 + 2 * module.AUC_MARGIN}),
        ("exact_duplicate_guardrail", {"exact": 0.02}),
        ("near_duplicate_guardrail", {"near": 2 * module.NEAR_DUPLICATE_FACTOR * 0.02}),
        ("nn_distance_p1_guardrail", {"p1": 0.5 * module.P1_FACTOR * 0.1}),
    ):
        failed = module.gate4(b0, _summary(**{**{"auc": 0.5 + module.AUC_MARGIN, "exact": 0.004,
                                                "near": module.NEAR_DUPLICATE_FACTOR * 0.02,
                                                "p1": module.P1_FACTOR * 0.1}, **worse}),
                              n_synthetic=251)
        assert failed[name] is False, name
        assert failed["passed"] is False
        assert "sampling" in failed["action_required"]

    # And the exact-duplicate floor is one synthetic individual, so B0 at zero still allows 1/n.
    floor = module.gate4(_summary(0.5, 0.0, 0.02, 0.1), _summary(0.5, 1 / 251, 0.02, 0.1),
                         n_synthetic=251)
    assert floor["exact_duplicate_guardrail"] is True


def _panel(root: Path, eval_cohorts: list[int]) -> tuple[Path, Path]:
    """A miniature frozen panel and two-seed multiseed run, with only `eval_cohorts` in eval."""
    rng = np.random.default_rng(7)
    n_snps, n_train = 6, 30
    n_eval = len(eval_cohorts)
    prepared = root / "prepared"
    prepared.mkdir()
    calls = rng.integers(0, 3, size=(n_train + n_eval, n_snps)).astype(np.float32)
    np.savez(prepared / "genotypes.npz", calls=calls, offsets=np.array([0, n_snps]))
    labels = np.concatenate([np.arange(n_train) % 26, np.array(eval_cohorts)])
    np.savez(prepared / "dataset.npz", y=labels,
             train_indices=np.arange(n_train),
             val_indices=np.arange(n_train, n_train + n_eval),
             test_indices=np.arange(n_train, n_train + n_eval))
    (prepared / "gene_variant_map.json").write_text(json.dumps({"genes": [{"variants": [
        {"train_maf": maf, "position": 100.0 * index}
        for index, maf in enumerate([0.02, 0.03, 0.1, 0.15, 0.3, 0.4])]}]}))
    with (prepared / "label_hierarchy.pkl").open("wb") as handle:
        pickle.dump({"idx_to_pop": {index: f"P{index:02d}" for index in range(26)},
                     "idx_to_superpop": {index: f"S{index}" for index in range(5)},
                     "pop_to_superpop": {index: index % 5 for index in range(26)}}, handle)
    (prepared / "normalization_stats.pkl").write_bytes(b"stats")

    multiseed = root / "multiseed"
    multiseed.mkdir()
    seeds = [11, 12]
    (multiseed / "manifest.json").write_text(json.dumps({
        "seeds": seeds, "eval_split": "test",
        "prepared_sha256": {name: hashlib.sha256((prepared / name).read_bytes()).hexdigest()
                            for name in module_files()}}))
    for seed in seeds:
        run = multiseed / f"seed_{seed}" / "standard"
        run.mkdir(parents=True)
        for name in ("synthetic_genotypes.npy", "synthetic_genotypes_T.npy"):
            np.save(run / name, rng.integers(0, 3, size=(n_eval, n_snps)).astype(np.int8))
        (run / "diagnostic_report.json").write_text(json.dumps({
            "eval_split": "test",
            "label_composition": {str(cohort): eval_cohorts.count(cohort)
                                  for cohort in sorted(set(eval_cohorts))},
            "genotype_evaluation": {"samples": n_eval, "variants": n_snps}}))
    return prepared, multiseed


def _rows(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_cohorts_missing_from_eval_are_marked_not_estimable(tmp_path):
    # Given an eval split holding only cohorts 0 (n=6) and 1 (n=3).
    prepared, multiseed = _panel(tmp_path, [0] * 6 + [1] * 3)
    output = tmp_path / "out"

    # When Phase 4 runs.
    subprocess.run([sys.executable, str(SCRIPT), "--prepared-dir", str(prepared),
                    "--multiseed-dir", str(multiseed), "--output-dir", str(output)],
                   check=True, capture_output=True, text=True)

    # Then the 24 absent cohorts carry the words, not blanks or zeros, and the thin one is flagged.
    rows = {row["cohort"]: row for row in _rows(output / "strata_cohort.csv")}
    assert rows["2"]["status"] == "not estimable"
    assert rows["2"]["n_eval"] == "0"
    assert {rows["2"][column] for column in ("af_mae_B0", "af_mae_T", "het_mae_B0", "het_mae_T",
                                             "nn_distance_median_B0", "nn_distance_median_T")} == {
        "not estimable"}
    assert rows["0"]["status"] == "ok"
    assert float(rows["0"]["af_mae_B0"]) >= 0.0
    assert rows["1"]["status"] == "low precision (n<5)"
    assert float(rows["1"]["af_mae_T"]) >= 0.0
    assert rows["unseen"]["status"] == "no unseen cohort in this panel"

    # And the MAF table splits the six SNPs 2/2/2 across the three bins.
    assert [row["n_snps"] for row in _rows(output / "strata_maf.csv")] == ["2", "2", "2"]

    report = json.loads((output / "privacy_report.json").read_text())
    assert report["seeds"] == [11, 12]
    assert report["eval_split"] == "test"
    assert set(report["prepared_sha256"]) == set(module_files())
    assert "differential privacy" in report["limitation"]


def module_files() -> tuple[str, ...]:
    from scripts.hipodit_privacy import PREPARED_FILES

    return PREPARED_FILES


def test_the_near_duplicate_threshold_is_the_real_non_member_first_percentile(tmp_path):
    # Given the frozen panel and a finished two-seed run.
    module = privacy()
    prepared, multiseed = _panel(tmp_path, [0] * 6 + [1] * 3)

    # When Phase 4 measures duplication.
    report = module.run(prepared, multiseed, tmp_path / "out")

    # Then the threshold is how close a real non-member already sits to the train panel, at the
    # first percentile -- not the fifth, and not any statistic of the synthetic set itself.
    with np.load(prepared / "genotypes.npz") as data:
        calls = data["calls"].astype(np.float64)
    with np.load(prepared / "dataset.npz") as data:
        train, test = calls[data["train_indices"]], calls[data["test_indices"]]
    reference = module.distance_stats(module.mean_abs_dosage_distance(test, train).min(axis=1))
    assert reference["p1"] != reference["p5"]
    assert report["near_duplicate_threshold"] == reference["p1"]
    for arm, filename in module.ARMS.items():
        nearest = module.mean_abs_dosage_distance(
            np.load(multiseed / "seed_11" / "standard" / filename), train).min(axis=1)
        assert report["near_duplicate_threshold"] != float(np.percentile(nearest, 1))
        # And it is the threshold the reported rate was actually counted against.
        assert report["arms"][arm]["per_seed"][0]["near_duplicate_rate_train"] == pytest.approx(
            float((nearest < report["near_duplicate_threshold"]).mean()))


def test_a_malformed_phase3_summary_leaves_no_output_directory_behind(tmp_path):
    # Given a Gate 3' summary beside the seeds that Phase 4 cannot read.
    prepared, multiseed = _panel(tmp_path, [0] * 6 + [1] * 3)
    (multiseed / "summary_decoder.json").write_text(json.dumps({"metrics": {}}))
    output = tmp_path / "out"

    # When Phase 4 runs.
    result = subprocess.run([sys.executable, str(SCRIPT), "--prepared-dir", str(prepared),
                             "--multiseed-dir", str(multiseed), "--output-dir", str(output)],
                            capture_output=True, text=True)

    # Then it fails without leaving a directory that would make the retry die on FileExistsError.
    assert result.returncode != 0
    assert not output.exists()


def test_cli_refuses_a_panel_the_samples_did_not_come_from(tmp_path):
    # Given a prepared panel edited after the multiseed run recorded its hashes.
    prepared, multiseed = _panel(tmp_path, [0] * 6 + [1] * 3)
    (prepared / "normalization_stats.pkl").write_bytes(b"other stats")

    # When Phase 4 is pointed at the pair.
    result = subprocess.run([sys.executable, str(SCRIPT), "--prepared-dir", str(prepared),
                             "--multiseed-dir", str(multiseed), "--output-dir", str(tmp_path / "o")],
                            capture_output=True, text=True)

    # Then it refuses: distances against the wrong train panel would be silently plausible.
    assert result.returncode != 0
    assert "was not run on" in result.stderr
    assert "normalization_stats.pkl" in result.stderr


def test_cli_refuses_an_existing_output_directory(tmp_path):
    # Given an output directory that already holds a run.
    prepared, multiseed = _panel(tmp_path, [0] * 6 + [1] * 3)
    output = tmp_path / "out"
    output.mkdir()

    # When Phase 4 is pointed at it.
    result = subprocess.run([sys.executable, str(SCRIPT), "--prepared-dir", str(prepared),
                             "--multiseed-dir", str(multiseed), "--output-dir", str(output)],
                            capture_output=True, text=True)

    # Then it refuses rather than overwriting the frozen output.
    assert result.returncode != 0
    assert "FileExistsError" in result.stderr
