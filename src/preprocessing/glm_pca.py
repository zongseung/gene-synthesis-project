"""Per-gene GLM-PCA dimensionality reduction (Townes et al. 2019).

The default accelerated backend uses a Poisson likelihood for bounded genotype
dosage. This preserves a mean-dependent variance model, but is an approximation,
not a Binomial genotype decoder.

Reference
---------
Townes, F. W., Hicks, S. C., Aryee, M. J., & Irizarry, R. A. (2019).
"Feature selection and dimension reduction for single-cell RNA-Seq based on
a multinomial model". *Genome Biology*, 20:295.
doi:10.1186/s13059-019-1861-6

Public API mirrors :mod:`src.preprocessing.pca` so that
is called directly by :mod:`src.preprocessing.pca`:

    glm_pca_single_gene(gene_name, matrix, n_components, train_indices=None)
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_GLM_FAMILY = "poi"
DEFAULT_MAX_ITER = 100


def project_glm_factors(
    observations: np.ndarray,
    loadings: np.ndarray,
    intercept: np.ndarray,
    *,
    family: str,
    max_iter: int = DEFAULT_MAX_ITER,
    penalty: float = 1.0,
    tolerance: float = 1e-5,
) -> np.ndarray:
    """Fit held-out factors with fixed ``exp(intercept + factors @ loadings.T)``."""
    if family != "poi":
        raise ValueError(
            f"Likelihood projection is implemented for 'poi', got {family!r}"
        )

    y = np.asarray(observations, dtype=np.float64)
    v = np.asarray(loadings, dtype=np.float64)
    a = np.asarray(intercept, dtype=np.float64)
    if y.ndim != 2 or v.ndim != 2 or a.shape != (y.shape[1],):
        raise ValueError(
            f"Expected observations (N,J), loadings (J,L), intercept (J,); "
            f"got {y.shape}, {v.shape}, {a.shape}"
        )
    if v.shape[0] != y.shape[1]:
        raise ValueError(
            f"Loadings features {v.shape[0]} do not match observations {y.shape[1]}"
        )
    if not np.isfinite(y).all() or np.any(y < 0):
        raise ValueError("Poisson observations must be finite and non-negative")
    if not np.isfinite(v).all() or not np.isfinite(a).all():
        raise ValueError("Loadings and intercept must be finite")
    if (
        max_iter < 1
        or not np.isfinite(penalty)
        or penalty <= 0
        or not np.isfinite(tolerance)
        or tolerance <= 0
    ):
        raise ValueError("max_iter, penalty, and tolerance must be positive")

    factors = np.zeros((y.shape[0], v.shape[1]), dtype=np.float64)
    identity = np.eye(v.shape[1], dtype=np.float64)
    converged = False
    for _ in range(max_iter):
        eta = a + factors @ v.T
        with np.errstate(over="raise", invalid="raise"):
            try:
                mean = np.exp(eta)
            except FloatingPointError as exc:
                raise ValueError("Poisson linear predictor overflowed") from exc
        gradient = (y - mean) @ v - penalty * factors
        information = np.einsum("nj,jk,jl->nkl", mean, v, v) + penalty * identity
        step = np.linalg.solve(information, gradient[..., None])[..., 0]
        objective = float(np.sum(mean - y * eta) + 0.5 * penalty * np.sum(factors**2))
        directional_gain = float(np.sum(gradient * step))
        scale = 1.0
        accepted = False
        for _ in range(24):
            candidate = factors + scale * step
            candidate_eta = a + candidate @ v.T
            with np.errstate(over="ignore", invalid="ignore"):
                candidate_objective = float(
                    np.sum(np.exp(candidate_eta) - y * candidate_eta)
                    + 0.5 * penalty * np.sum(candidate**2)
                )
            if np.isfinite(candidate_objective) and candidate_objective <= (
                objective - 1e-4 * scale * directional_gain
            ):
                factors = candidate
                accepted = True
                break
            scale *= 0.5
        if not accepted:
            # No step size improves the objective. When the proposed Newton step
            # is already smaller than the convergence tolerance, this is a
            # numerically stationary point rather than a failure: the remaining
            # decrease sits below floating-point noise, so no backtracked step
            # can satisfy a strict Armijo decrease. An unacceptable step that is
            # still large signals a genuine breakdown and keeps raising.
            if float(np.max(np.abs(step))) < tolerance:
                converged = True
                break
            raise RuntimeError("Poisson factor projection line search failed")
        if float(np.max(np.abs(scale * step))) < tolerance:
            converged = True
            break
    if not converged:
        raise RuntimeError(f"Poisson factor projection did not converge in {max_iter} iterations")
    return factors.astype(np.float32)


# Poisson backend, published on PyPI as ``glmpca-fast`` and installed by ``uv sync``.
try:
    import glmpca_fast as _RUST_BACKEND
except ImportError:
    _RUST_BACKEND = None
    logger.warning("glmpca-fast not importable; Poisson GLM-PCA is unavailable")

# Binomial(2, p) backend, built from ``binom_glmpca_rs/``. Absent, the scipy
# reference in src.preprocessing.binomial_glm_pca still runs, just slower.
try:
    import binom_glmpca_rs as _BINOM_BACKEND
except ImportError:
    _BINOM_BACKEND = None
    logger.warning(
        "binom_glmpca_rs not importable; falling back to the slower scipy "
        "Binomial GLM-PCA reference"
    )


def _project_held_out(
    X_full: np.ndarray,
    train_indices: np.ndarray,
    loadings: np.ndarray,
    intercept: np.ndarray,
    family: str,
    max_iter: int,
) -> np.ndarray:
    n_total = X_full.shape[0]
    L = loadings.shape[1]
    out = np.zeros((n_total, L), dtype=np.float32)
    out[train_indices] = project_glm_factors(
        X_full[train_indices],
        loadings,
        intercept,
        family=family,
        max_iter=max_iter,
    )

    held = np.setdiff1d(np.arange(n_total), train_indices, assume_unique=True)
    if held.size == 0:
        return out

    out[held] = project_glm_factors(
        X_full[held],
        loadings,
        intercept,
        family=family,
        max_iter=max_iter,
    )
    return out


def glm_pca_single_gene(
    gene_name: str,
    matrix: np.ndarray,
    n_components: int,
    train_indices: np.ndarray | None = None,
    max_iter: int = DEFAULT_MAX_ITER,
    family: str | None = None,
) -> dict | None:
    """Fit a train-only Poisson GLM-PCA decoder and score every sample.

    Parameters
    ----------
    gene_name : str
    matrix : np.ndarray, shape (n_samples, n_variants)
        Dosage matrix in {0, 1, 2} (or [0, 2] after mean imputation).
    n_components : int
        Target latent dimensionality K.
    train_indices : np.ndarray | None
        If provided, GLM-PCA is fit on these rows only and held-out rows are
        projected onto the fitted basis (matches the leakage-prevention flow
        of the streaming pipeline).
    max_iter : int, default 100
        Maximum coordinate-descent iterations.

    Returns
    -------
    dict | None
        Shape that downstream pipeline code
        does not branch:
            ``features``: {f"{gene}:0": (n_samples,), ...}
            ``explained_total``: optimizer deviance reduction (≥ 0)
            ``n_variants``: int
            ``actual_k``: int

    Backend
    -------
    Requires :mod:`glmpca_fast`; unsupported families fail explicitly.
    """
    from src.preprocessing.config import GLM_FAMILY

    family = GLM_FAMILY if family is None else family
    n_vars = matrix.shape[1]
    n_fit = matrix.shape[0] if train_indices is None else int(len(train_indices))
    n_comp = min(n_components, n_vars, n_fit)
    if n_comp < 2:
        return None

    if family == "binom2":
        return _binomial_result(
            gene_name=gene_name,
            matrix=matrix,
            train_indices=train_indices,
            n_comp=min(n_comp, n_vars - 1, n_fit - 1),
            n_vars=n_vars,
            max_iter=max(150, max_iter),
        )
    if family != "poi":
        raise ValueError(
            f"Unknown GLM-PCA family {family!r}; expected 'poi' or 'binom2'"
        )

    fit_matrix = matrix if train_indices is None else matrix[train_indices]

    # ── Rust fast path ─────────────────────────────────────────────────
    if _RUST_BACKEND is not None:
        result = _RUST_BACKEND.fit_poisson(
            np.ascontiguousarray(fit_matrix, dtype=np.float32),
            L=n_comp,
            max_iter=max_iter,
            tol=1e-4,
            penalty=1.0,
            seed=42,
        )
        loadings = np.asarray(result["loadings"], dtype=np.float32)
        intercept = np.asarray(result["intercept"], dtype=np.float32)
        dev = np.asarray(result["deviance"], dtype=np.float32)
        return _build_result(
            gene_name=gene_name,
            matrix=matrix,
            train_indices=train_indices,
            loadings=loadings,
            intercept=intercept,
            family=DEFAULT_GLM_FAMILY,
            backend=str(result["backend"]),
            backend_version=str(_RUST_BACKEND.__version__),
            projection="fixed_decoder_likelihood",
            max_iter=max_iter,
            dev=dev,
            n_comp=n_comp,
            n_vars=n_vars,
        )

    raise RuntimeError("Poisson GLM-PCA requires the installed glmpca-fast backend")


def _binomial_result(
    *,
    gene_name: str,
    matrix: np.ndarray,
    train_indices: np.ndarray | None,
    n_comp: int,
    n_vars: int,
    max_iter: int,
) -> dict | None:
    """Fit Binomial(2, p) GLM-PCA, the correctly specified dosage likelihood.

    Delegates to :func:`src.preprocessing.binomial_glm_pca.fit_binomial_glm_pca`,
    which already fits on training rows only and scores held-out rows, so no
    separate projection step is needed.
    """
    # The fitter needs components < min(rows, variants), so a two-variant gene
    # drops to K=1. Keep it: dropping those 429 genes would change the panel
    # between families and break the Poisson comparison.
    if n_comp < 1:
        return None
    rows = (
        np.arange(matrix.shape[0], dtype=np.int64)
        if train_indices is None
        else np.asarray(train_indices, dtype=np.int64)
    )
    # A Binomial count must be an integer draw out of two. Fractionally imputed
    # entries are missingness, not observations, so they enter as NaN.
    calls = np.asarray(matrix, dtype=np.float64)
    integral = np.isclose(calls, np.round(calls))
    calls = np.where(integral, np.round(calls), np.nan)
    if not integral.all():
        logger.debug(
            f"{gene_name}: {(~integral).sum()} fractional entries treated as missing"
        )
    fit, backend, version = _fit_binomial(calls, rows, n_comp, max_iter)
    return _build_result(
        gene_name=gene_name,
        matrix=matrix,
        train_indices=train_indices,
        loadings=np.asarray(fit["loadings"], dtype=np.float32),
        intercept=np.asarray(fit["intercept"], dtype=np.float32),
        family="binom2",
        backend=backend,
        backend_version=version,
        projection="joint_penalized_likelihood",
        max_iter=max_iter,
        dev=np.asarray(
            [fit["objective_initial"], fit["objective_final"]], dtype=np.float32
        ),
        n_comp=n_comp,
        n_vars=n_vars,
        factors=np.asarray(fit["factors"], dtype=np.float32),
    )


def _fit_binomial(
    calls: np.ndarray, rows: np.ndarray, n_comp: int, max_iter: int
) -> tuple[dict, str, str]:
    """Fit Binomial(2, p) GLM-PCA, preferring the Rust alternating-IRLS backend.

    Measured on a realistic rare-variant panel (2003 x 300, K=4, 37 monomorphic
    columns): Rust 1.28 s against 4.84 s for the scipy L-BFGS reference, at a
    lower penalized objective and identical allele-frequency recovery. The
    Python path stays as the fallback and as the oracle the Rust is tested
    against in tests/test_binom_glmpca_rs.py.
    """
    if _BINOM_BACKEND is not None:
        return (
            _BINOM_BACKEND.fit_binomial(
                calls, np.asarray(rows, dtype=np.int64), n_comp, max_iter
            ),
            "binom_glmpca_rs",
            str(getattr(_BINOM_BACKEND, "__version__", "unknown")),
        )
    from src.preprocessing.binomial_glm_pca import fit_binomial_glm_pca

    fitted = fit_binomial_glm_pca(calls, rows, n_comp, max_iter=max_iter)
    return (
        {
            "loadings": fitted.loadings,
            "intercept": fitted.intercept,
            "factors": fitted.factors,
            "objective_initial": fitted.objective_initial,
            "objective_final": fitted.objective_final,
        },
        "binomial_glm_pca",
        "lbfgsb",
    )


def _build_result(
    *,
    gene_name: str,
    matrix: np.ndarray,
    train_indices: np.ndarray | None,
    loadings: np.ndarray,
    intercept: np.ndarray,
    family: str,
    backend: str,
    backend_version: str,
    projection: str,
    max_iter: int,
    dev: np.ndarray,
    n_comp: int,
    n_vars: int,
    factors: np.ndarray | None = None,
) -> dict:
    """Common post-processing shared by every backend.

    ``factors`` lets a fitter that already scored every row hand them over
    instead of paying for a second projection.
    """
    if dev.size >= 2 and dev[0] > 0:
        explained = float(max(0.0, 1.0 - dev[-1] / dev[0]))
    else:
        explained = 0.0

    if factors is not None:
        transformed = factors
    elif train_indices is None:
        transformed = project_glm_factors(
            matrix,
            loadings,
            intercept,
            family=family,
            max_iter=max(100, max_iter),
        )
    else:
        transformed = _project_held_out(
            X_full=matrix,
            train_indices=np.asarray(train_indices, dtype=np.int64),
            loadings=loadings,
            intercept=intercept,
            family=family,
            max_iter=max(100, max_iter),
        )

    features = {f"{gene_name}:{k}": transformed[:, k] for k in range(n_comp)}
    return {
        "features": features,
        "explained_total": explained,
        "n_variants": n_vars,
        "actual_k": n_comp,
        "loadings": loadings,
        "intercept": intercept,
        "family": family,
        "link": "log" if family in {"poi", "nb"} else "logit",
        "penalty": 1.0,
        "backend": backend,
        "backend_version": backend_version,
        "projection": projection,
    }
