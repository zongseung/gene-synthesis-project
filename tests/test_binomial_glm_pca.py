"""Likelihood, missing-data and train-only contracts for the dosage decoder."""

import numpy as np
import pytest
from scipy.special import expit


def test_binomial_fit_is_train_only_and_improves_observed_likelihood():
    from src.preprocessing.binomial_glm_pca import fit_binomial_glm_pca

    # Given structured diploid calls, including missing observations.
    rng = np.random.default_rng(81)
    z = rng.normal(size=(80, 2))
    v = rng.normal(size=(12, 2))
    y = rng.binomial(2, expit(-0.7 + z @ v.T)).astype(float)
    y[0, 0] = np.nan
    train = np.arange(60)
    changed = y.copy()
    changed[60:] = 2 - changed[60:]
    # When held-out data change while the fitting rows remain fixed.
    first = fit_binomial_glm_pca(y, train, 2, max_iter=150)
    second = fit_binomial_glm_pca(changed, train, 2, max_iter=150)
    # Then the decoder and training factors are identical, with a valid fit.
    np.testing.assert_allclose(first.loadings, second.loadings)
    np.testing.assert_allclose(first.intercept, second.intercept)
    np.testing.assert_allclose(first.factors[train], second.factors[train])
    assert first.objective_final < first.objective_initial
    probabilities = first.probabilities(first.factors)
    assert np.all((probabilities > 0) & (probabilities < 1))
    assert np.isfinite(first.factors).all()


@pytest.mark.parametrize("bad", [-1.0, 0.5, 3.0, np.inf])
def test_binomial_fit_rejects_values_that_are_not_diploid_calls(bad):
    from src.preprocessing.binomial_glm_pca import fit_binomial_glm_pca

    # Given an invalid observed call; NaN alone denotes missingness.
    y = np.ones((8, 4))
    y[0, 0] = bad
    # When fitting, then the input contract rejects it.
    with pytest.raises(ValueError, match="0, 1, 2"):
        fit_binomial_glm_pca(y, np.arange(6), 2)


def test_fisher_sensitivity_includes_normalization_chain_rule():
    from src.preprocessing.binomial_glm_pca import BinomialGLMPCA

    # Given p=1/2, each SNP contributes 2p(1-p)=1/2 information.
    fit = BinomialGLMPCA(np.zeros((3, 2)), np.eye(2), np.zeros(2), 1., 0.5, True)
    # When normalized coordinates have scales two and three.
    sensitivity = fit.fisher_diagonal(np.zeros((3, 2)), np.array([2., 3.]))
    # Then mean-per-SNP information includes the squared affine scales.
    np.testing.assert_allclose(sensitivity, [1., 2.25])
