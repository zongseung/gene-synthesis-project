"""Distribution-distance metrics for real-vs-synthetic evaluation.

Pure-numpy implementations of:

    * ``gaussian_w2_distance``     — FID-like 2-Wasserstein distance between
                                      the Gaussian fits of two point clouds.
    * ``mmd_rbf``                  — biased Maximum Mean Discrepancy with an
                                      RBF kernel and median-heuristic bandwidth.
    * ``same_class_coverage``      — fraction of real samples whose 95th
                                      percentile real-NN radius contains at
                                      least one synthetic sample.
    * ``centroid_distance``        — Euclidean distance between mean vectors.
    * ``nn_adversarial_accuracy``  — AATS, the artificial-genome literature's
                                      standard resemblance/privacy statistic.

All functions are pure (no I/O, no global state); the module depends only
on ``numpy`` and ``scikit-learn``.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors

def centroid_distance(x_real: np.ndarray, x_syn: np.ndarray) -> float:
    """Euclidean distance between the centroids of two point sets."""
    return float(np.linalg.norm(x_real.mean(axis=0) - x_syn.mean(axis=0)))


def gaussian_w2_distance(x_real: np.ndarray, x_syn: np.ndarray) -> float:
    """FID-like squared 2-Wasserstein distance between Gaussian fits.

    ``W2² = ||μ_r − μ_s||² + tr(Σ_r + Σ_s − 2 (Σ_r^{1/2} Σ_s Σ_r^{1/2})^{1/2})``

    The result is clamped at zero to absorb numerical negative trace from
    near-singular covariance matrices.
    """
    mu_r = x_real.mean(axis=0)
    mu_s = x_syn.mean(axis=0)
    cov_r = _covariance(x_real)
    cov_s = _covariance(x_syn)
    sqrt_r = _symmetric_sqrt(cov_r)
    middle = sqrt_r @ cov_s @ sqrt_r
    sqrt_middle = _symmetric_sqrt(middle)
    diff = mu_r - mu_s
    value = float(diff @ diff + np.trace(cov_r + cov_s - 2 * sqrt_middle))
    return max(value, 0.0)


def mmd_rbf(
    x_real: np.ndarray,
    x_syn: np.ndarray,
    gamma: float | None = None,
) -> dict:
    """Biased Maximum Mean Discrepancy with an RBF kernel.

    Parameters
    ----------
    x_real, x_syn : np.ndarray
        Real and synthetic point clouds in the same feature space.
    gamma : float | None, default=None
        Kernel bandwidth. ``None`` selects the median-heuristic
        ``γ = 1 / (2 · median²)`` over the pooled pairwise distance matrix.

    Returns
    -------
    dict
        ``{"mmd_rbf_biased": float, "mmd_rbf_gamma": float}``.
    """
    if gamma is None:
        combined = np.vstack([x_real, x_syn])
        d = pairwise_distances(combined, metric="euclidean")
        tri = d[np.triu_indices_from(d, k=1)]
        median = float(np.median(tri[tri > 0])) if np.any(tri > 0) else 1.0
        gamma = 1.0 / (2.0 * median * median)

    k_xx = _rbf_kernel(x_real, x_real, gamma)
    k_yy = _rbf_kernel(x_syn, x_syn, gamma)
    k_xy = _rbf_kernel(x_real, x_syn, gamma)
    value = float(k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean())
    return {"mmd_rbf_biased": max(value, 0.0), "mmd_rbf_gamma": float(gamma)}


def same_class_coverage(real_points: np.ndarray, syn_points: np.ndarray) -> float:
    """Fraction of real points covered by synthetic points within a 95th-percentile NN radius.

    For each real point, the radius is the 95th percentile of real-to-real
    nearest-neighbor distances. The point is considered "covered" if at
    least one synthetic point lies within that radius.
    """
    if len(real_points) < 2 or len(syn_points) < 1:
        return float("nan")

    real_nn = NearestNeighbors(n_neighbors=2, metric="euclidean").fit(real_points)
    real_dists, _ = real_nn.kneighbors(real_points, return_distance=True)
    radius = float(np.quantile(real_dists[:, 1], 0.95))

    syn_nn = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(syn_points)
    syn_dists, _ = syn_nn.kneighbors(real_points, return_distance=True)
    return float(np.mean(syn_dists[:, 0] <= radius))


# ── helpers ─────────────────────────────────────────────────────────────
def _covariance(x: np.ndarray) -> np.ndarray:
    if x.shape[0] <= 1:
        return np.zeros((x.shape[1], x.shape[1]), dtype=np.float64)
    return np.cov(x, rowvar=False).astype(np.float64)


def _symmetric_sqrt(mat: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eigh((mat + mat.T) / 2)
    vals = np.clip(vals, 0.0, None)
    return (vecs * np.sqrt(vals)) @ vecs.T


def _rbf_kernel(a: np.ndarray, b: np.ndarray, gamma: float) -> np.ndarray:
    sq = pairwise_distances(a, b, metric="sqeuclidean")
    return np.exp(-gamma * sq)


def nn_adversarial_accuracy(x_real: np.ndarray, x_syn: np.ndarray) -> dict:
    """Nearest-neighbour adversarial accuracy (AATS) and its two components.

    This is the statistic the artificial-genome literature reports (Yale et al.
    2020; applied to 1000 Genomes artificial genomes by Yelmen et al. 2021 and
    2023), so it is what makes our runs comparable to published models. With

        d_TT(i) = real i to its nearest *other* real sample
        d_TS(i) = real i to its nearest synthetic sample
        d_SS(j) = synthetic j to its nearest *other* synthetic sample
        d_ST(j) = synthetic j to its nearest real sample

    ``AA_truth = mean(d_TS > d_TT)``, ``AA_syn = mean(d_ST > d_SS)`` and AATS is
    their average. 0.5 means the two sets are indistinguishable, above 0.5 means
    the synthetic samples sit too far (underfitting), below 0.5 means they sit
    too close (overfitting, and the regime the literature reads as privacy
    leakage).

    ``AA_syn`` depends on how dense the synthetic set is, so compare sets of
    equal size or the number is not on the published scale.

    Parameters
    ----------
    x_real, x_syn : np.ndarray, shape (n, d) and (m, d)
        Samples in a common metric space. Needs at least 2 of each.

    Returns
    -------
    dict
        ``aats``, ``aa_truth``, ``aa_syn``, ``n_real``, ``n_synthetic`` and the
        four mean distances.
    """
    if x_real.ndim != 2 or x_syn.ndim != 2:
        raise ValueError(f"Expected 2-D arrays, got {x_real.shape} and {x_syn.shape}")
    if x_real.shape[1] != x_syn.shape[1]:
        raise ValueError(f"Feature mismatch: {x_real.shape[1]} vs {x_syn.shape[1]}")
    if len(x_real) < 2 or len(x_syn) < 2:
        raise ValueError("AATS needs at least two samples on each side")

    def _within(x: np.ndarray) -> np.ndarray:
        distances, _ = NearestNeighbors(n_neighbors=2).fit(x).kneighbors(x)
        return distances[:, 1]  # column 0 is the sample itself

    def _across(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
        distances, _ = NearestNeighbors(n_neighbors=1).fit(reference).kneighbors(query)
        return distances[:, 0]

    d_tt, d_ss = _within(x_real), _within(x_syn)
    d_ts, d_st = _across(x_real, x_syn), _across(x_syn, x_real)
    aa_truth = float(np.mean(d_ts > d_tt))
    aa_syn = float(np.mean(d_st > d_ss))
    return {
        "aats": 0.5 * (aa_truth + aa_syn),
        "aa_truth": aa_truth,
        "aa_syn": aa_syn,
        "n_real": int(len(x_real)),
        "n_synthetic": int(len(x_syn)),
        "mean_d_truth_truth": float(np.mean(d_tt)),
        "mean_d_truth_syn": float(np.mean(d_ts)),
        "mean_d_syn_syn": float(np.mean(d_ss)),
        "mean_d_syn_truth": float(np.mean(d_st)),
    }
