"""DUPI (Distance-based Utility-Privacy Index) for synthetic-data evaluation.

A faithful re-implementation of the kth-order DUPI score and its
accompanying Utility / Privacy indices proposed in

    Donghoon Jeong, Joseph H. T. Kim, and Jongho Im,
    "A New Global Measure to Simultaneously Evaluate Data Utility and
    Privacy Risk", IEEE Trans. Information Forensics and Security,
    Vol. 18, pp. 715-729, 2023. doi:10.1109/TIFS.2022.3228753

Equation map (paper → code)
---------------------------
    Eq. (8)  Identity ``d_{X_n}^{<k+1>}(x_i) = d_{X_{n\\i}}^{<k>}(x_i)``
             ↳ implemented by querying ``NearestNeighbors`` with
               ``n_neighbors = k+1`` against the fitted real set and
               taking column ``[:, k]`` (column 0 is the self-distance).

    Eq. (10) ``DUPI₀ = (1/n) Σ_i P(d_{Y_m}^{<k>}(X_i) ≤ d_{X_{n\\i}}^{<k>}(X_i))``
             closed form  ``Σ_{s=k}^{2k-1} C(s-1, k-1) C(n+m-s-1, m-k) / C(n+m-1, m)``
             ↳ :func:`kth_dupi_benchmark`. Special case ``k = 1``
               collapses to ``m / (n + m - 1)``.

    Eq. (11) ``DUPI^{<k>} = (1/n) Σ_i 1{d_{Y_m}^{<k>}(x_i) ≤ d_{X_{n\\i}}^{<k>}(x_i)}``
             ↳ :func:`dupi_score`.

    g(DUPI)  Piecewise rescaling onto ``[0, 1]`` defined on p. 721.
             ↳ inline branch inside :func:`ui_pi_from_dupi`.

    Eq. (12) ``UI(DUPI) = arctan(τ · g(DUPI)) / arctan(τ)``
    Eq. (13) ``PI(DUPI) = arctan(τ · (1 − g(DUPI))) / arctan(τ)``
             ↳ :func:`ui_pi_from_dupi`. Default ``τ = 5`` per p. 721.

    Eq. (14) ``UI · PI ≤ (arctan(τ/2) / arctan(τ))²`` with equality at
             ``DUPI = DUPI₀``. Verified by ``test_theorem5_upper_bound``.

Notes
-----
* **No-duplicates assumption** (paper §III.B(b)): each row of ``x_real``
  must be unique. This is what allows ``kneighbors`` column 0 to be
  identified with self in :func:`dupi_score`.
* The algorithm is **distribution-free** and **distance-free** (paper §III).
  It works in any metric space; we use Euclidean by default to match the
  paper's numerical experiments.
* Pure functions: no file or network I/O. Dependencies: ``numpy``,
  ``scikit-learn``. The module can be vendored verbatim.

Citation
--------
If you use this module academically, cite the paper above plus the project
README's BibTeX entry for this package.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
from sklearn.neighbors import NearestNeighbors

__all__ = [
    "dupi_score",
    "kth_dupi_benchmark",
    "ui_pi_from_dupi",
]


def kth_dupi_benchmark(n: int, m: int, k: int) -> float:
    """Theoretical DUPI value when real and synthetic share the same distribution.

    Implements Eq. (8) of Jeong, Kim, and Im (2023). For ``k = 1`` this
    collapses to ``m / (n + m - 1)``; for higher ``k`` it sums the
    binomial-coefficient contributions in log space for numerical stability.

    Parameters
    ----------
    n : int
        Number of real samples.
    m : int
        Number of synthetic samples.
    k : int
        Order of the DUPI statistic (1 ≤ k ≤ min(n − 1, m)).

    Returns
    -------
    float
        Equal-distribution benchmark value in (0, 1).
    """
    if not (1 <= k <= min(n - 1, m)):
        raise ValueError(f"k must satisfy 1 <= k <= min(n - 1, m); got {k}")
    if k == 1:
        return m / (n + m - 1)

    denominator = _log_comb(n - 1 + m, m)
    terms = [
        _log_comb(s - 1, k - 1)
        + _log_comb(n - 1 + m - s, m - k)
        - denominator
        for s in range(k, 2 * k)
    ]
    return float(math.exp(_logsumexp(terms)))


def dupi_score(x_real: np.ndarray, x_syn: np.ndarray, k: int = 1) -> dict:
    """Compute the kth-order DUPI between real and synthetic samples.

    For each real sample ``x_i`` the function compares
        d_R(x_i) = distance to its k-th nearest *real* neighbor (excluding self), and
        d_S(x_i) = distance to its k-th nearest *synthetic* neighbor.
    DUPI is then the proportion of real samples for which d_S ≤ d_R.

    Interpretation:
        * DUPI close to 1 — synthetic samples sit closer than real-real
          neighbors → potential memorization risk.
        * DUPI close to 0 — synthetic samples are far from real → utility loss.
        * DUPI close to ``kth_dupi_benchmark(n, m, k)`` — synthetic and real
          look statistically indistinguishable in the local neighborhood.

    Parameters
    ----------
    x_real : np.ndarray, shape (n, d)
        Real samples in the metric space (e.g. PCA scores).
    x_syn : np.ndarray, shape (m, d)
        Synthetic samples in the same metric space.
    k : int, default=1
        Neighborhood order.

    Returns
    -------
    dict
        Keys: ``k``, ``n_real``, ``n_synthetic``, ``dupi``,
        ``dupi_benchmark``, ``dupi_minus_benchmark``, ``dupi_abs_error``,
        ``mean_kth_real_distance``, ``mean_kth_synthetic_distance``,
        ``median_kth_real_distance``, ``median_kth_synthetic_distance``.
    """
    n = x_real.shape[0]
    m = x_syn.shape[0]
    if n <= k:
        raise ValueError(f"Need at least k+1 real samples; n={n}, k={k}")
    if m < k:
        raise ValueError(f"Need at least k synthetic samples; m={m}, k={k}")

    syn_nn = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(x_syn)
    d_syn, _ = syn_nn.kneighbors(x_real, return_distance=True)
    kth_syn = d_syn[:, k - 1]

    real_nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean").fit(x_real)
    d_real, _ = real_nn.kneighbors(x_real, return_distance=True)
    kth_real_without_self = d_real[:, k]

    value = float(np.mean(kth_syn <= kth_real_without_self))
    benchmark = kth_dupi_benchmark(n, m, k)
    return {
        "k": k,
        "n_real": n,
        "n_synthetic": m,
        "dupi": value,
        "dupi_benchmark": benchmark,
        "dupi_minus_benchmark": value - benchmark,
        "dupi_abs_error": abs(value - benchmark),
        "mean_kth_real_distance": float(np.mean(kth_real_without_self)),
        "mean_kth_synthetic_distance": float(np.mean(kth_syn)),
        "median_kth_real_distance": float(np.median(kth_real_without_self)),
        "median_kth_synthetic_distance": float(np.median(kth_syn)),
    }


def ui_pi_from_dupi(dupi_value: float, dupi0: float, tau: float = 5.0) -> dict:
    """Map a DUPI value into the (Utility Index, Privacy Index) pair.

    Implements Eqs. (10)-(11) of Jeong, Kim, and Im (2023). The DUPI value is
    first normalised by its benchmark into ``g ∈ [0, 1]``, then transformed
    through ``atan(τ · ·) / atan(τ)`` to yield UI and PI.

    Parameters
    ----------
    dupi_value : float
        Observed DUPI value in [0, 1].
    dupi0 : float
        Benchmark DUPI for the configuration (see :func:`kth_dupi_benchmark`).
    tau : float, default=5.0
        Curvature of the atan-sigmoid transform. Larger τ produces sharper
        transitions around the benchmark.

    Returns
    -------
    dict
        Keys: ``tau``, ``g_dupi``, ``utility_index``, ``privacy_index``,
        ``utility_privacy_product``.
    """
    if not 0 <= dupi_value <= 1:
        raise ValueError(f"DUPI must be in [0, 1], got {dupi_value}")
    if not 0 < dupi0 < 1:
        raise ValueError(f"DUPI benchmark must be in (0, 1), got {dupi0}")
    if tau <= 0:
        raise ValueError(f"tau must be positive, got {tau}")

    if dupi_value <= dupi0:
        g = dupi_value / (2 * dupi0)
    else:
        g = (dupi_value - dupi0) / (2 * (1 - dupi0)) + 0.5

    denom = math.atan(tau)
    ui = math.atan(tau * g) / denom
    pi = math.atan(tau * (1 - g)) / denom
    return {
        "tau": tau,
        "g_dupi": float(g),
        "utility_index": float(ui),
        "privacy_index": float(pi),
        "utility_privacy_product": float(ui * pi),
    }


# ── numerical helpers ─────────────────────────────────────────────────────
def _log_comb(n: int, r: int) -> float:
    if r < 0 or r > n:
        return -math.inf
    return math.lgamma(n + 1) - math.lgamma(r + 1) - math.lgamma(n - r + 1)


def _logsumexp(values: Iterable[float]) -> float:
    vals = list(values)
    max_val = max(vals)
    if max_val == -math.inf:
        return -math.inf
    return max_val + math.log(sum(math.exp(v - max_val) for v in vals))
