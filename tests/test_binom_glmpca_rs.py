"""Rust Binomial(2, p) GLM-PCA (`binom_glmpca_rs.fit_binomial`) against the Python reference."""

import time

import numpy as np
import pytest
from scipy.special import expit

binom_glmpca_rs = pytest.importorskip("binom_glmpca_rs")


def _simulate(n, j, k, seed, missing=0.02):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, k))
    v = rng.normal(size=(j, k)) * 0.7
    y = rng.binomial(2, expit(rng.normal(-0.8, 0.8, size=j) + z @ v.T)).astype(np.float64)
    y[rng.random(y.shape) < missing] = np.nan
    return y


def _probabilities(intercept, factors, loadings):
    return expit(intercept + factors @ loadings.T)


def _procrustes(target, source):
    """Rotate `source` onto `target` (the penalty is invariant to orthogonal rotations)."""
    u, _, vt = np.linalg.svd(target.T @ source)
    return source @ (u @ vt).T


def test_matches_python_reference_on_seeded_case():
    from src.preprocessing.binomial_glm_pca import fit_binomial_glm_pca

    y = _simulate(300, 40, 3, seed=11)
    train = np.arange(240)
    ref = fit_binomial_glm_pca(y, train, 3, max_iter=150)
    rs = binom_glmpca_rs.fit_binomial(y, train, 3, 150)

    assert rs["factors"].shape == (300, 3) and rs["loadings"].shape == (40, 3)
    assert rs["intercept"].shape == (40,) and rs["converged"]
    assert rs["objective_final"] < rs["objective_initial"]

    # Same penalised objective. The reference stops at L-BFGS ftol=1e-9 (relative), so it
    # sits slightly above the optimum; the Rust fit must be at least as good and within
    # that stopping tolerance of it.
    assert rs["objective_final"] <= ref.objective_final + 1e-9 * abs(ref.objective_final)
    assert abs(rs["objective_final"] - ref.objective_final) <= 1e-6 * abs(ref.objective_final)

    # Fitted probabilities are rotation-invariant and are what the decoder consumes:
    # 1e-3 absolute, which is the size of the reference's own stopping error (ftol=1e-9
    # on an objective of ~1e4 leaves coefficient errors of ~1e-3 and probability errors
    # of ~1e-4). Raw coefficients are only identified up to rotation, so loadings are
    # compared after Procrustes alignment, at the looser 1e-2.
    p_ref = _probabilities(ref.intercept, ref.factors, ref.loadings)
    p_rs = _probabilities(rs["intercept"], rs["factors"], rs["loadings"])
    prob_gap = np.abs(p_ref - p_rs).max()
    intercept_gap = np.abs(ref.intercept - rs["intercept"]).max()
    loading_gap = np.abs(_procrustes(ref.loadings, rs["loadings"]) - ref.loadings).max()
    assert prob_gap < 1e-3
    assert intercept_gap < 1e-2 and loading_gap < 1e-2
    assert prob_gap < max(intercept_gap, loading_gap)


def test_fit_is_train_only():
    y = _simulate(120, 20, 2, seed=5)
    train = np.arange(90)
    changed = y.copy()
    changed[90:] = 2 - changed[90:]
    changed[100:, :3] = np.nan

    first = binom_glmpca_rs.fit_binomial(y, train, 2, 150)
    second = binom_glmpca_rs.fit_binomial(changed, train, 2, 150)

    np.testing.assert_array_equal(first["loadings"], second["loadings"])
    np.testing.assert_array_equal(first["intercept"], second["intercept"])
    np.testing.assert_array_equal(first["factors"][train], second["factors"][train])
    assert first["objective_final"] == second["objective_final"]
    assert not np.allclose(first["factors"][90:], second["factors"][90:])
    assert np.isfinite(second["factors"]).all()


def test_rust_is_several_times_faster_on_a_realistic_gene():
    from src.preprocessing.binomial_glm_pca import fit_binomial_glm_pca

    y = _simulate(2504, 300, 4, seed=2026, missing=0.01)
    train = np.random.default_rng(2026).permutation(2504)[:2003].astype(np.int64)

    t = time.perf_counter()
    ref = fit_binomial_glm_pca(y, train, 4, max_iter=150)
    python_s = time.perf_counter() - t
    t = time.perf_counter()
    rs = binom_glmpca_rs.fit_binomial(y, train, 4, 150)
    rust_s = time.perf_counter() - t

    speedup = python_s / rust_s
    print(f"\n2003x300 K=4: python {python_s:.2f}s, rust {rust_s:.3f}s, speedup {speedup:.1f}x")
    assert rs["objective_final"] <= ref.objective_final + 1e-9 * abs(ref.objective_final)
    # The threshold is a floor, not the expected value. Measured 8.8x on dense
    # synthetic data but 3.8-4.0x on a realistic rare-variant panel (2003x300,
    # K=4, 37 monomorphic columns), where most columns are near-monomorphic and
    # the scipy reference converges faster. Three is below anything observed and
    # still proves the point; the printed number is what to watch for regressions.
    assert speedup >= 3, f"python {python_s:.2f}s vs rust {rust_s:.3f}s = {speedup:.1f}x"


def test_edge_cases():
    y = _simulate(50, 8, 2, seed=1, missing=0.0)
    train = np.arange(40)

    unobserved = y.copy()
    unobserved[:40, 2] = np.nan  # observed only in held-out rows
    with pytest.raises(ValueError, match="observed training call"):
        binom_glmpca_rs.fit_binomial(unobserved, train, 2, 50)

    with pytest.raises(ValueError, match="Components"):
        binom_glmpca_rs.fit_binomial(y, train, 8, 50)
    with pytest.raises(ValueError, match="Components"):
        binom_glmpca_rs.fit_binomial(y, np.arange(2), 2, 50)

    with pytest.raises(ValueError):
        binom_glmpca_rs.fit_binomial(np.full((50, 8), np.nan), train, 2, 50)

    bad = y.copy()
    bad[0, 0] = 0.5
    with pytest.raises(ValueError, match="0, 1, 2"):
        binom_glmpca_rs.fit_binomial(bad, train, 2, 50)

    with pytest.raises(ValueError, match="Training indices"):
        binom_glmpca_rs.fit_binomial(y, np.array([0, 0, 1]), 1, 50)

    # Monomorphic variants (all 0, all 2) must stay finite.
    mono = y.copy()
    mono[:, 0] = 0.0
    mono[:, 1] = 2.0
    fit = binom_glmpca_rs.fit_binomial(mono, train, 2, 150)
    for key in ("factors", "loadings", "intercept"):
        assert np.isfinite(fit[key]).all(), key
    assert np.isfinite(fit["objective_final"])
    p = _probabilities(fit["intercept"], fit["factors"], fit["loadings"])
    assert np.isfinite(p).all() and p[:, 0].max() < 0.05 and p[:, 1].min() > 0.95
