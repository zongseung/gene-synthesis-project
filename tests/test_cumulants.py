from __future__ import annotations

import numpy as np


def _correlated_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    base = rng.normal(size=(80, 2, 2))
    transform = np.array([[1.0, 0.8], [0.2, 1.4]])
    x = np.einsum("ngk,kl->ngl", base, transform).astype(np.float32)
    pop_labels = np.repeat([0, 1], 40)
    superpop_labels = np.zeros(80, dtype=np.int64)
    return x, pop_labels, superpop_labels


def test_fit_cumulants_zca_whitens_each_gene():
    from src.preprocessing.cumulants import fit_cumulant_statistics

    x, pop_labels, superpop_labels = _correlated_fixture()
    stats = fit_cumulant_statistics(
        x,
        pop_labels,
        superpop_labels,
        n_pops=2,
        n_superpops=1,
        gene_size=3,
        prior_strength_skew=0.0,
        prior_strength_kurtosis=0.0,
    )

    centered = x - stats["center"][:2]
    whitened = np.einsum(
        "ngk,gkl->ngl", centered, stats["whitener"][:2]
    )
    covariance = np.einsum(
        "ngk,ngl->gkl", whitened, whitened
    ) / (len(x) - 1)

    expected = np.broadcast_to(np.eye(2), covariance.shape)
    np.testing.assert_allclose(covariance, expected, atol=2e-5)


def test_fit_cumulants_uses_unbiased_skewness_and_excess_kurtosis():
    from src.preprocessing.cumulants import fit_cumulant_statistics

    x = np.array([-1.0, -1.0, 1.0, 1.0], dtype=np.float32).reshape(4, 1, 1)
    labels = np.zeros(4, dtype=np.int64)
    stats = fit_cumulant_statistics(
        x,
        labels,
        labels,
        n_pops=1,
        n_superpops=1,
        gene_size=1,
        prior_strength_skew=0.0,
        prior_strength_kurtosis=0.0,
    )

    np.testing.assert_allclose(stats["skewness"][0, 0, 0], 0.0, atol=1e-7)
    np.testing.assert_allclose(
        stats["excess_kurtosis"][0, 0, 0], -6.0, atol=1e-6
    )


def test_hierarchical_shrinkage_reduces_between_population_difference():
    from src.preprocessing.cumulants import fit_cumulant_statistics

    rng = np.random.default_rng(11)
    pop0 = rng.exponential(scale=1.0, size=(40, 1, 1))
    pop1 = -rng.exponential(scale=1.0, size=(40, 1, 1))
    x = np.concatenate([pop0, pop1]).astype(np.float32)
    pop_labels = np.repeat([0, 1], 40)
    superpop_labels = np.zeros(80, dtype=np.int64)

    raw = fit_cumulant_statistics(
        x,
        pop_labels,
        superpop_labels,
        n_pops=2,
        n_superpops=1,
        gene_size=1,
        prior_strength_skew=0.0,
        prior_strength_kurtosis=0.0,
    )
    shrunk = fit_cumulant_statistics(
        x,
        pop_labels,
        superpop_labels,
        n_pops=2,
        n_superpops=1,
        gene_size=1,
        prior_strength_skew=100.0,
        prior_strength_kurtosis=100.0,
    )

    raw_gap = abs(raw["skewness"][0, 0, 0] - raw["skewness"][1, 0, 0])
    shrunk_gap = abs(
        shrunk["skewness"][0, 0, 0] - shrunk["skewness"][1, 0, 0]
    )
    assert shrunk_gap < raw_gap


def test_artifact_padding_and_null_population_are_safe(tmp_path):
    from src.preprocessing.cumulants import (
        fit_cumulant_statistics,
        save_cumulant_statistics,
    )

    x, pop_labels, superpop_labels = _correlated_fixture()
    stats = fit_cumulant_statistics(
        x,
        pop_labels,
        superpop_labels,
        n_pops=2,
        n_superpops=1,
        gene_size=3,
    )

    assert stats["center"].shape == (3, 2)
    assert stats["whitener"].shape == (3, 2, 2)
    assert stats["skewness"].shape == (3, 3, 2)
    np.testing.assert_array_equal(stats["skewness"][-1], 0.0)
    np.testing.assert_array_equal(stats["excess_kurtosis"][-1], 0.0)
    np.testing.assert_array_equal(stats["whitener"][2], np.eye(2))
    np.testing.assert_array_equal(stats["dewhitener"][2], np.eye(2))

    output = tmp_path / "cumulants.npz"
    save_cumulant_statistics(stats, output)
    with np.load(output, allow_pickle=False) as saved:
        assert int(saved["format_version"]) == 1
        np.testing.assert_array_equal(saved["pop_counts"], [40, 40])
        assert np.isfinite(saved["excess_kurtosis"]).all()
