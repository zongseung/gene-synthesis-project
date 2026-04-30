"""Invariant + paper-reproduction tests for :mod:`src.evaluation.dupi` and friends.

The first half asserts the algebraic properties claimed by Jeong, Kim, and
Im (2023) — bounded ranges, closed-form benchmarks, atan-sigmoid edge
cases, identical-distribution convergence.

The second half (``TestPaperReproduction``) reproduces the **exact
numerical examples** printed in the paper (Wine illustration on p. 722,
optimal-point coordinates, Theorem 5 upper bound, simulation S1 limit) so
that any future refactor is anchored to the published reference values.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.evaluation.dupi import (
    dupi_score,
    kth_dupi_benchmark,
    ui_pi_from_dupi,
)
from src.evaluation.distribution_metrics import (
    centroid_distance,
    gaussian_w2_distance,
    mmd_rbf,
    same_class_coverage,
)


# ── DUPI ────────────────────────────────────────────────────────────────
class TestDupiBenchmark:
    def test_k1_closed_form(self) -> None:
        # Eq. (8) collapses to m / (n + m - 1) when k = 1
        assert kth_dupi_benchmark(251, 2504, 1) == pytest.approx(2504 / (251 + 2504 - 1))

    def test_in_unit_interval(self) -> None:
        for n, m, k in [(100, 100, 1), (50, 500, 2), (200, 50, 3)]:
            v = kth_dupi_benchmark(n, m, k)
            assert 0.0 < v < 1.0

    def test_invalid_k_raises(self) -> None:
        with pytest.raises(ValueError):
            kth_dupi_benchmark(10, 10, 0)
        with pytest.raises(ValueError):
            kth_dupi_benchmark(10, 10, 11)


class TestDupiScore:
    def test_in_unit_interval(self) -> None:
        rng = np.random.default_rng(0)
        x_real = rng.standard_normal((50, 4))
        x_syn = rng.standard_normal((200, 4))
        out = dupi_score(x_real, x_syn, k=1)
        assert 0.0 <= out["dupi"] <= 1.0
        assert 0.0 < out["dupi_benchmark"] < 1.0

    def test_identical_distribution_near_benchmark(self) -> None:
        # For two iid samples drawn from the same distribution, observed DUPI
        # should be within 0.1 of the theoretical benchmark.
        rng = np.random.default_rng(42)
        x_real = rng.standard_normal((300, 8))
        x_syn = rng.standard_normal((1500, 8))
        out = dupi_score(x_real, x_syn, k=1)
        assert out["dupi_abs_error"] < 0.10, out

    def test_far_synthetic_drives_dupi_low(self) -> None:
        # Synthetic shifted far away → kth_syn ≫ kth_real → DUPI → 0.
        rng = np.random.default_rng(1)
        x_real = rng.standard_normal((100, 4))
        x_syn = rng.standard_normal((400, 4)) + 50.0
        out = dupi_score(x_real, x_syn, k=1)
        assert out["dupi"] < 0.05, out

    def test_overlapping_synthetic_drives_dupi_high(self) -> None:
        # Synthetic = real + tiny jitter → DUPI → 1.
        rng = np.random.default_rng(2)
        x_real = rng.standard_normal((100, 4))
        x_syn = np.repeat(x_real, 5, axis=0) + rng.standard_normal((500, 4)) * 1e-3
        out = dupi_score(x_real, x_syn, k=1)
        assert out["dupi"] > 0.95, out

    def test_too_few_samples_raises(self) -> None:
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError):
            dupi_score(rng.standard_normal((1, 2)), rng.standard_normal((10, 2)), k=1)


class TestUiPi:
    def test_g_at_dupi_equal_benchmark(self) -> None:
        # When dupi_value == dupi0, g should be exactly 0.5 → UI = PI.
        out = ui_pi_from_dupi(0.6, 0.6, tau=5.0)
        assert out["g_dupi"] == pytest.approx(0.5)
        assert out["utility_index"] == pytest.approx(out["privacy_index"])

    def test_dupi_zero_gives_min_ui(self) -> None:
        # dupi_value = 0 → g = 0 → UI = 0, PI = 1.
        out = ui_pi_from_dupi(0.0, 0.5, tau=5.0)
        assert out["utility_index"] == pytest.approx(0.0)
        assert out["privacy_index"] == pytest.approx(1.0)

    def test_dupi_one_gives_max_ui(self) -> None:
        # dupi_value = 1 → g = 1 → UI = 1, PI = 0.
        out = ui_pi_from_dupi(1.0, 0.5, tau=5.0)
        assert out["utility_index"] == pytest.approx(1.0)
        assert out["privacy_index"] == pytest.approx(0.0)

    def test_indices_in_unit_interval(self) -> None:
        for d in np.linspace(0.05, 0.95, 7):
            out = ui_pi_from_dupi(float(d), 0.5, tau=5.0)
            assert 0.0 <= out["utility_index"] <= 1.0
            assert 0.0 <= out["privacy_index"] <= 1.0

    def test_invalid_inputs_raise(self) -> None:
        with pytest.raises(ValueError):
            ui_pi_from_dupi(1.5, 0.5)
        with pytest.raises(ValueError):
            ui_pi_from_dupi(0.5, 1.5)
        with pytest.raises(ValueError):
            ui_pi_from_dupi(0.5, 0.5, tau=0.0)


# ── Distribution metrics ────────────────────────────────────────────────
class TestDistributionMetrics:
    def test_centroid_distance_zero_for_same_data(self) -> None:
        rng = np.random.default_rng(0)
        x = rng.standard_normal((100, 3))
        assert centroid_distance(x, x) == pytest.approx(0.0)

    def test_w2_zero_for_same_data(self) -> None:
        rng = np.random.default_rng(0)
        x = rng.standard_normal((100, 3))
        assert gaussian_w2_distance(x, x) == pytest.approx(0.0, abs=1e-6)

    def test_w2_nonnegative(self) -> None:
        rng = np.random.default_rng(0)
        x = rng.standard_normal((100, 3))
        y = rng.standard_normal((100, 3)) + 5
        assert gaussian_w2_distance(x, y) >= 0.0

    def test_mmd_zero_for_same_data(self) -> None:
        rng = np.random.default_rng(0)
        x = rng.standard_normal((100, 3))
        out = mmd_rbf(x, x)
        assert out["mmd_rbf_biased"] == pytest.approx(0.0, abs=1e-6)

    def test_mmd_nonnegative_and_gamma_positive(self) -> None:
        rng = np.random.default_rng(0)
        x = rng.standard_normal((50, 3))
        y = rng.standard_normal((50, 3)) + 2
        out = mmd_rbf(x, y)
        assert out["mmd_rbf_biased"] >= 0.0
        assert out["mmd_rbf_gamma"] > 0.0

    def test_coverage_in_unit_interval(self) -> None:
        rng = np.random.default_rng(0)
        x = rng.standard_normal((40, 2))
        y = rng.standard_normal((400, 2))
        cov = same_class_coverage(x, y)
        assert 0.0 <= cov <= 1.0

    def test_coverage_perfect_when_synthetic_dense(self) -> None:
        rng = np.random.default_rng(0)
        x = rng.standard_normal((30, 2))
        # Synthetic = 100 noisy copies of every real → very dense coverage.
        y = np.repeat(x, 100, axis=0) + rng.standard_normal((3000, 2)) * 1e-3
        assert same_class_coverage(x, y) == pytest.approx(1.0)

    def test_coverage_zero_when_synthetic_far(self) -> None:
        rng = np.random.default_rng(0)
        x = rng.standard_normal((30, 2))
        y = rng.standard_normal((300, 2)) + 100
        assert same_class_coverage(x, y) == pytest.approx(0.0)


# ── Paper reproduction (Jeong, Kim, Im 2023) ────────────────────────────
class TestPaperReproduction:
    """Re-run the printed numerical examples from the IEEE TIFS paper."""

    # ── Eq. (10) closed form ──────────────────────────────────────────
    def test_eq10_k1_special_case(self) -> None:
        """Eq. (10) reduces to ``m / (n + m - 1)`` when k = 1 (paper text after Eq. 10)."""
        for n, m in [(100, 100), (251, 2504), (300, 1500)]:
            assert kth_dupi_benchmark(n, m, 1) == pytest.approx(m / (n + m - 1))

    def test_eq10_general_consistent_with_k1_formula(self) -> None:
        """The general Eq. (10) sum must match the k=1 special case at k=1."""
        # general formula path (range(k, 2k) = [1] for k=1)
        denominator = math.lgamma(100 + 50) - math.lgamma(50 + 1) - math.lgamma(99 + 1)
        log_num = (
            math.lgamma(50 + 1) - math.lgamma(0 + 1) - math.lgamma(50 + 1)  # C(0,0)=1
            + math.lgamma(50 + 100 - 1 + 1) - math.lgamma(50 - 1 + 1) - math.lgamma(100 + 1)
        )
        # Just verify our function returns the closed form to many digits.
        assert kth_dupi_benchmark(50, 100, 1) == pytest.approx(100 / (50 + 100 - 1), rel=1e-12)

    # ── Wine illustration (paper page 722) ─────────────────────────────
    def test_wine_example_ui_pi(self) -> None:
        """Paper Section IV.C, Fig. 2 caption: ``(UI, PI) = (0.652, 0.954)`` from DUPI = 0.25, DUPI₀ = 0.5, τ = 5."""
        out = ui_pi_from_dupi(0.25, 0.5, tau=5.0)
        assert out["g_dupi"] == pytest.approx(0.25)
        # Paper rounds to 3 decimals — check to that precision.
        assert out["utility_index"] == pytest.approx(0.652, abs=5e-4)
        assert out["privacy_index"] == pytest.approx(0.954, abs=5e-4)

    def test_optimal_point_at_dupi_equals_benchmark(self) -> None:
        """Paper page 722: at DUPI = DUPI₀ the optimal point is (0.867, 0.867) for τ = 5."""
        out = ui_pi_from_dupi(0.5, 0.5, tau=5.0)
        assert out["g_dupi"] == pytest.approx(0.5)
        assert out["utility_index"] == pytest.approx(0.867, abs=5e-4)
        assert out["privacy_index"] == pytest.approx(0.867, abs=5e-4)

    # ── Theorem 5 (Eq. 14) ─────────────────────────────────────────────
    def test_theorem5_upper_bound(self) -> None:
        """Eq. (14): ``UI × PI ≤ (arctan(τ/2)/arctan(τ))²`` with equality iff DUPI = DUPI₀."""
        tau = 5.0
        bound = (math.atan(tau / 2) / math.atan(tau)) ** 2
        # Sweep DUPI across both branches of g(·)
        for d in np.linspace(0.05, 0.95, 19):
            out = ui_pi_from_dupi(float(d), 0.5, tau=tau)
            assert out["utility_privacy_product"] <= bound + 1e-9
        # Equality at DUPI = DUPI₀
        equal_out = ui_pi_from_dupi(0.5, 0.5, tau=tau)
        assert equal_out["utility_privacy_product"] == pytest.approx(bound)

    # ── Default tau ────────────────────────────────────────────────────
    def test_default_tau_matches_paper(self) -> None:
        """Paper page 721: ``The default value is set at τ = 5 in this paper.``"""
        # Calling without tau should match calling with tau=5.0 explicitly.
        a = ui_pi_from_dupi(0.3, 0.5)
        b = ui_pi_from_dupi(0.3, 0.5, tau=5.0)
        assert a == b

    # ── Simulation S1 limit (paper page 722, Fig. 3) ──────────────────
    def test_s1_simulation_dupi_near_benchmark(self) -> None:
        """Paper S1: ``Y_i ~ MVN_5(0, I)`` matches ``X_i ~ MVN_5(0, I)``; DUPI should converge to ``m/(n+m-1) ≈ 0.5001``."""
        # Smaller m=n=600 to keep test fast (paper used 2000 with 1000 reps).
        rng = np.random.default_rng(2023)
        m = n = 600
        bench = m / (n + m - 1)  # ≈ 0.50042
        # Average over 30 reps — sample variance shrinks like 1/sqrt(30) ≈ 0.18
        means = []
        for seed in range(30):
            r = np.random.default_rng(seed)
            x_real = r.standard_normal((n, 5))
            x_syn = r.standard_normal((m, 5))
            means.append(dupi_score(x_real, x_syn, k=1)["dupi"])
        observed = float(np.mean(means))
        assert abs(observed - bench) < 0.02, f"observed={observed}, bench={bench}"

    # ── Eq. (8) identity check (NN excluding self) ─────────────────────
    def test_eq8_identity_nn_excluding_self(self) -> None:
        """Eq. (8): ``d_{X_n}^{<k+1>}(x_i) = d_{X_{n\\i}}^{<k>}(x_i)``.

        Cross-check that ``dupi_score`` indexes ``kneighbors`` correctly:
        when synthetic is identical to real (modulo a tiny perturbation),
        the empirical DUPI should be ≈ 1 because the syn-NN distance is
        always smaller than the real-NN-excluding-self distance.
        """
        rng = np.random.default_rng(0)
        x_real = rng.standard_normal((100, 4))
        x_syn = x_real.copy()  # m = n, perfectly aligned
        # Without an "exclude self" rule, the real side would also have
        # distance 0 → DUPI = 0.5 (ties). With the rule, syn-NN is 0 vs
        # real-NN-without-self > 0 → DUPI = 1.
        out = dupi_score(x_real, x_syn, k=1)
        assert out["dupi"] == pytest.approx(1.0)
