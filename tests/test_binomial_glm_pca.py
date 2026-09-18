"""Likelihood, missing-data and train-only contracts for the dosage decoder."""

import numpy as np
import pytest
from scipy.special import expit

from src.inference.decode import decode_gene
from src.preprocessing.binomial_glm_pca import BinomialGLMPCA, fit_binomial_glm_pca
from src.preprocessing.glm_pca import glm_pca_single_gene
import src.preprocessing.binomial_glm_pca as module


def test_binomial_fit_is_train_only_and_improves_observed_likelihood():

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

    # Given an invalid observed call; NaN alone denotes missingness.
    y = np.ones((8, 4))
    y[0, 0] = bad
    # When fitting, then the input contract rejects it.
    with pytest.raises(ValueError, match="0, 1, 2"):
        fit_binomial_glm_pca(y, np.arange(6), 2)


def test_fisher_sensitivity_includes_normalization_chain_rule():

    # Given p=1/2, each SNP contributes 2p(1-p)=1/2 information.
    fit = BinomialGLMPCA(np.zeros((3, 2)), np.eye(2), np.zeros(2), 1., 0.5, True)
    # When normalized coordinates have scales two and three.
    sensitivity = fit.fisher_diagonal(np.zeros((3, 2)), np.array([2., 3.]))
    # Then mean-per-SNP information includes the squared affine scales.
    np.testing.assert_allclose(sensitivity, [1., 2.25])


def test_pipeline_routes_binom2_to_the_bounded_likelihood_and_decodes_with_its_link():
    """The per-gene entry point and the genotype decoder must agree on the family."""
    rng = np.random.default_rng(5)
    z = rng.normal(size=(60, 2))
    v = rng.normal(size=(20, 2))
    calls = rng.binomial(2, expit(-0.4 + z @ v.T)).astype(float)
    train = np.arange(45)

    result = glm_pca_single_gene("GENE", calls, 2, train, family="binom2")
    assert result["family"] == "binom2" and result["link"] == "logit"
    assert len(result["features"]) == result["actual_k"] == 2
    assert all(f.shape == (60,) for f in result["features"].values())

    # The logit mean stays inside the two-copy range without needing a clip,
    # which is the whole point of the bounded likelihood.
    dosage = decode_gene(result["features"] and np.stack(
        [result["features"][f"GENE:{k}"] for k in range(2)], axis=1),
        result["loadings"], result["intercept"], "binom2")
    assert dosage.shape == (60, 20)
    assert dosage.min() > 0.0 and dosage.max() < 2.0
    # Recovered allele frequencies track the real ones.
    assert np.corrcoef(dosage.mean(axis=0), calls.mean(axis=0))[0, 1] > 0.9

    with pytest.raises(ValueError, match="Unknown GLM-PCA family"):
        decode_gene(z, v, np.zeros(20), "nb")

    # A two-variant gene falls to K=1 rather than being dropped, so the gene
    # panel stays identical to the Poisson run.
    narrow = glm_pca_single_gene("NARROW", calls[:, :2], 2, train, family="binom2")
    assert narrow is not None and narrow["actual_k"] == 1


def test_spectral_start_survives_a_driver_failure_instead_of_aborting():
    """One unlucky gene must not take down a 24k-gene preprocessing run."""
    calls = np.zeros((6, 4))
    calls[:, 0] = 1.0
    failures = []

    def always_fails(*args, **kwargs):
        failures.append(kwargs.get("lapack_driver"))
        raise np.linalg.LinAlgError("SVD did not converge")

    original = module.scipy_svd
    module.scipy_svd = always_fails
    try:
        z, v = module._spectral_start(calls, 2)
    finally:
        module.scipy_svd = original

    assert failures == ["gesdd", "gesvd"]          # both drivers tried
    assert z.shape == (6, 2) and v.shape == (4, 2)  # zero start, not an exception
    assert not z.any() and not v.any()
