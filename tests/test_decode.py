"""The GLM-PCA decoder must invert what preprocessing fit.

Fit a gene with `glm_pca_single_gene`, push its own factors back through
`decode_gene`, and check the recovered allele frequencies match the data the
fit saw. If the decoder formula or the loadings orientation drifts, this fails.
"""

from __future__ import annotations

import numpy as np

from src.inference.decode import decode_gene, sample_genotypes
from src.preprocessing.glm_pca import glm_pca_single_gene


def _binomial_gene(n_samples: int = 120, n_variants: int = 24, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(n_samples, 1))
    freq = 1.0 / (1.0 + np.exp(-(rng.normal(size=(1, n_variants)) + 0.8 * latent)))
    return rng.binomial(2, freq).astype(np.float32)


def test_decoded_allele_frequencies_track_the_fitted_gene():
    matrix = _binomial_gene()

    fit = glm_pca_single_gene("GENE", matrix, n_components=4)
    assert fit is not None

    factors = np.stack([fit["features"][f"GENE:{k}"] for k in range(fit["actual_k"])], axis=1)
    dosage = decode_gene(factors, fit["loadings"], fit["intercept"])

    assert dosage.shape == matrix.shape
    assert (dosage >= 0).all() and (dosage <= 2).all()

    af_real = matrix.mean(axis=0) / 2.0
    af_decoded = dosage.mean(axis=0) / 2.0
    assert np.corrcoef(af_real, af_decoded)[0, 1] > 0.9
    assert np.abs(af_real - af_decoded).mean() < 0.05


def test_sampled_calls_are_genotypes_centred_on_the_dosage():
    rng = np.random.default_rng(0)
    dosage = np.full((4000, 3), [0.0, 1.0, 2.0], dtype=np.float64)

    calls = sample_genotypes(dosage, rng)

    assert calls.dtype == np.int8
    assert set(np.unique(calls)).issubset({0, 1, 2})
    assert np.allclose(calls.mean(axis=0), [0.0, 1.0, 2.0], atol=0.05)
