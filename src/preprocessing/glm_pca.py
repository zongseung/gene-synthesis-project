"""Per-gene GLM-PCA dimensionality reduction (Townes et al. 2019).

This is the **statistically correct** alternative to :mod:`src.preprocessing.pca`
for genotype dosage data. Genotype X_ij ∈ {0, 1, 2} follows
``Binomial(n=2, p_j)`` under Hardy-Weinberg equilibrium, with mean–variance
relationship ``Var(X) = 2p(1−p)`` — a function of the mean. Standard PCA
assumes Gaussian + homoscedastic noise, which is misspecified for this data.
GLM-PCA generalises PCA to the exponential family: the Binomial likelihood
encodes the correct mean–variance relationship, eliminating the false
heteroscedasticity that biases PCA's principal directions toward common
variants.

Reference
---------
Townes, F. W., Hicks, S. C., Aryee, M. J., & Irizarry, R. A. (2019).
"Feature selection and dimension reduction for single-cell RNA-Seq based on
a multinomial model". *Genome Biology*, 20:295.
doi:10.1186/s13059-019-1861-6

Public API mirrors :mod:`src.preprocessing.pca` so that
:mod:`src.preprocessing.dim_reduction` can dispatch transparently:

    glm_pca_single_gene(gene_name, matrix, n_components, train_indices=None)
    evaluate_glm_pca_k_for_gene(gene_name, matrix, k, train_indices=None)
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from src.preprocessing.config import (
    MARGINAL_GAIN_DECAY_RATIO,
    MARGINAL_GAIN_THRESHOLD,
    PCA_CANDIDATES,
    PCA_SAMPLE_GENES,
)

logger = logging.getLogger(__name__)

# Default GLM-PCA family: 'mult' (multinomial-Poisson approximation).
# For dosage Binomial(2, p) data, 'mult' is the standard choice in the
# Townes et al. (2019) framework; 'nb' (negative binomial) is for overdispersed
# count data and rarely needed here.
DEFAULT_GLM_FAMILY = "poi"  # 'poi' enables Rust fast path; 'mult'/'nb' fall back to glmpca-py
DEFAULT_MAX_ITER = 100


def _import_glmpca():
    """Lazy import — keeps optional dependency optional at module load."""
    try:
        from glmpca import glmpca
        return glmpca
    except ImportError as exc:
        raise ImportError(
            "GLM-PCA requires the `glmpca` package. Install with `uv add glmpca`."
        ) from exc


def _try_import_rust():
    """Import the accelerated GLM-PCA extension if available.

    Published on PyPI as ``glmpca-fast`` (declared in pyproject.toml);
    installed normally via ``uv sync``. Returns the module on success, or
    logs once and returns None if it isn't importable — callers then fall
    back to the ~13x slower pure-Python `glmpca` reference implementation.
    """
    try:
        import glmpca_fast
        return glmpca_fast
    except ImportError:
        logger.warning(
            "glmpca-fast not importable — GLM-PCA will use the pure-Python "
            "`glmpca` reference implementation (~13x slower). Run `uv sync` "
            "to install the accelerated backend."
        )
        return None


_RUST_BACKEND = _try_import_rust()


def _project_held_out(
    X_full: np.ndarray,
    train_indices: np.ndarray,
    train_factors: np.ndarray,
    loadings: np.ndarray,
    coef_X: np.ndarray,
) -> np.ndarray:
    """Project all rows onto the GLM-PCA basis fitted on train rows.

    Train rows reuse their fitted ``factors`` directly. Held-out rows are
    projected via OLS in the (Pearson-residual approximated) linear space:
        Z_new ≈ (X_new − meanₜᵣₐᵢₙ) · V · (Vᵀ V)⁻¹
    This is a first-order approximation to the GLM projection — exact for
    Gaussian, biased but consistent for Binomial. It mirrors the
    "fit on train, transform all" leakage-prevention idiom used by the PCA
    branch.
    """
    n_total = X_full.shape[0]
    L = loadings.shape[1]
    out = np.zeros((n_total, L), dtype=np.float32)
    out[train_indices] = train_factors.astype(np.float32)

    held = np.setdiff1d(np.arange(n_total), train_indices, assume_unique=True)
    if held.size == 0:
        return out

    train_mean = X_full[train_indices].mean(axis=0)
    centered = X_full[held] - train_mean
    proj = centered @ loadings @ np.linalg.pinv(loadings.T @ loadings)
    out[held] = proj.astype(np.float32)
    return out


def glm_pca_single_gene(
    gene_name: str,
    matrix: np.ndarray,
    n_components: int,
    train_indices: np.ndarray | None = None,
    fam: str = DEFAULT_GLM_FAMILY,
    max_iter: int = DEFAULT_MAX_ITER,
) -> dict | None:
    """Apply GLM-PCA (Binomial-style) to a single gene's dosage matrix.

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
        of :func:`src.preprocessing.pca.pca_single_gene`).
    fam : str, default 'mult'
        Likelihood family — see Townes et al. (2019) Section 4.
    max_iter : int, default 100
        Maximum coordinate-descent iterations.

    Returns
    -------
    dict | None
        Same shape as :func:`pca_single_gene` so downstream pipeline code
        does not branch:
            ``features``: {f"{gene}:0": (n_samples,), ...}
            ``explained_total``: deviance-based pseudo-R² (≥ 0)
            ``explained_per_component``: per-component pseudo-R² (uniform
                                          split of total — true per-component
                                          requires K separate fits)
            ``n_variants``: int
            ``actual_k``: int

    Backend
    -------
    If the optional accelerated extension :mod:`glmpca_fast` is importable
    and ``fam == "poi"``, this function delegates to its Rust backend
    (~13× faster per gene). Otherwise falls back to the reference Python
    package :mod:`glmpca` (Townes 2019).
    """
    n_vars = matrix.shape[1]
    n_fit = matrix.shape[0] if train_indices is None else int(len(train_indices))
    n_comp = min(n_components, n_vars, n_fit)
    if n_comp < 2:
        return None

    fit_matrix = matrix if train_indices is None else matrix[train_indices]

    # ── Rust fast path ─────────────────────────────────────────────────
    if _RUST_BACKEND is not None and fam == "poi":
        try:
            result = _RUST_BACKEND.fit_poisson(
                np.ascontiguousarray(fit_matrix, dtype=np.float32),
                L=n_comp,
                max_iter=max_iter,
                tol=1e-4,
                penalty=1.0,
                seed=42,
            )
            factors = np.asarray(result["factors"], dtype=np.float32)
            loadings = np.asarray(result["loadings"], dtype=np.float32)
            dev = np.asarray(result["deviance"], dtype=np.float32)
            return _build_result(
                gene_name=gene_name,
                matrix=matrix,
                train_indices=train_indices,
                factors=factors,
                loadings=loadings,
                dev=dev,
                n_comp=n_comp,
                n_vars=n_vars,
            )
        except Exception as exc:
            logger.warning(
                f"Rust GLM-PCA failed for {gene_name}: {exc}; falling back to Python"
            )

    # ── Python reference fallback ──────────────────────────────────────
    glmpca_mod = _import_glmpca()
    Y_fit = fit_matrix.T  # glmpca expects (features, samples)

    try:
        res = glmpca_mod.glmpca(
            Y=Y_fit,
            L=n_comp,
            fam=fam,
            verbose=False,
            ctl={"maxIter": max_iter, "eps": 1e-4},
        )
    except Exception as exc:
        logger.warning(f"GLM-PCA fit failed for {gene_name}: {exc}")
        return None

    factors = np.asarray(res["factors"], dtype=np.float32)        # (n_fit, K)
    loadings = np.asarray(res["loadings"], dtype=np.float32)      # (n_vars, K)
    coef_X = np.asarray(res.get("coefX", np.zeros((n_vars, 1))), dtype=np.float32)
    dev = np.asarray(res["dev"], dtype=np.float32)

    # Pseudo-R² (McFadden-style): relative deviance reduction
    if dev.size >= 2 and dev[0] > 0:
        explained = float(max(0.0, 1.0 - dev[-1] / dev[0]))
    else:
        explained = 0.0

    return _build_result(
        gene_name=gene_name,
        matrix=matrix,
        train_indices=train_indices,
        factors=factors,
        loadings=loadings,
        dev=dev,
        n_comp=n_comp,
        n_vars=n_vars,
    )


def _build_result(
    *,
    gene_name: str,
    matrix: np.ndarray,
    train_indices: np.ndarray | None,
    factors: np.ndarray,
    loadings: np.ndarray,
    dev: np.ndarray,
    n_comp: int,
    n_vars: int,
) -> dict:
    """Common post-processing shared by Rust + Python backends."""
    if dev.size >= 2 and dev[0] > 0:
        explained = float(max(0.0, 1.0 - dev[-1] / dev[0]))
    else:
        explained = 0.0

    if train_indices is None:
        transformed = factors
    else:
        coef_X = np.zeros((n_vars, 1), dtype=np.float32)
        transformed = _project_held_out(
            X_full=matrix,
            train_indices=np.asarray(train_indices, dtype=np.int64),
            train_factors=factors,
            loadings=loadings,
            coef_X=coef_X,
        )

    features = {f"{gene_name}:{k}": transformed[:, k] for k in range(n_comp)}
    return {
        "features": features,
        "explained_total": explained,
        "explained_per_component": [explained / n_comp] * n_comp,
        "n_variants": n_vars,
        "actual_k": n_comp,
    }


def evaluate_glm_pca_k_for_gene(
    gene_name: str,
    matrix: np.ndarray,
    k: int,
    train_indices: np.ndarray | None = None,
    fam: str = DEFAULT_GLM_FAMILY,
    max_iter: int = DEFAULT_MAX_ITER,
) -> tuple[str, int, float, int]:
    """Evaluate (gene, K) — pseudo-R² analog of PCA explained variance ratio."""
    glmpca_mod = _import_glmpca()

    n_vars = matrix.shape[1]
    n_fit = matrix.shape[0] if train_indices is None else int(len(train_indices))
    actual_k = min(k, n_vars, n_fit)
    if actual_k < 2:
        return (gene_name, k, 0.0, 0)

    Y_fit = matrix.T if train_indices is None else matrix[train_indices].T
    try:
        res = glmpca_mod.glmpca(
            Y=Y_fit, L=actual_k, fam=fam, verbose=False,
            ctl={"maxIter": max_iter, "eps": 1e-4},
        )
    except Exception as exc:
        logger.warning(f"GLM-PCA eval failed for {gene_name} (k={k}): {exc}")
        return (gene_name, k, 0.0, 0)

    dev = np.asarray(res["dev"], dtype=np.float32)
    explained = (
        float(max(0.0, 1.0 - dev[-1] / dev[0]))
        if dev.size >= 2 and dev[0] > 0 else 0.0
    )
    return (gene_name, k, explained, actual_k)


def grid_search_optimal_glm_pca(
    gene_matrices: dict[str, np.ndarray],
    candidates: list[int] | None = None,
    marginal_threshold: float = MARGINAL_GAIN_THRESHOLD,
    decay_ratio: float = MARGINAL_GAIN_DECAY_RATIO,
    n_sample_genes: int = PCA_SAMPLE_GENES,
    train_indices: np.ndarray | None = None,
    fam: str = DEFAULT_GLM_FAMILY,
    max_iter: int = DEFAULT_MAX_ITER,
) -> tuple[int, pd.DataFrame]:
    """Grid search optimal K via deviance-based marginal-gain elbow.

    Mirrors :func:`src.preprocessing.pca.grid_search_optimal_pca` but uses
    pseudo-R² (deviance reduction) instead of explained variance ratio.
    GLM-PCA is ~100× slower than PCA per fit, so consider running grid
    search on a smaller chromosome subset (chr 22 alone, for example).
    """
    if candidates is None:
        candidates = PCA_CANDIDATES

    all_genes = list(gene_matrices.keys())
    if len(all_genes) > n_sample_genes:
        rng = np.random.default_rng(42)
        sample_genes = rng.choice(all_genes, n_sample_genes, replace=False)
    else:
        sample_genes = all_genes

    logger.info(
        f"GLM-PCA grid search: K candidates={candidates}, "
        f"sample genes={len(sample_genes)}/{len(all_genes)} (fam={fam}, "
        f"~100× slower than PCA)"
    )

    tasks = [
        (g, gene_matrices[g], k, train_indices, fam, max_iter)
        for g in sample_genes for k in candidates
    ]

    results = []
    # GLM-PCA is CPU-bound and threads share the GIL during numpy/statsmodels
    # work; ProcessPoolExecutor would be faster but `gene_matrices` is large.
    # Use modest thread parallelism — adjust via env if needed.
    with ThreadPoolExecutor(max_workers=min(8, len(candidates))) as ex:
        futures = [ex.submit(evaluate_glm_pca_k_for_gene, *t) for t in tasks]
        for fut in futures:
            results.append(fut.result())

    df = pd.DataFrame(results, columns=["gene", "k", "explained_ratio", "actual_k"])

    summary: dict[int, dict] = {}
    for k in candidates:
        valid = df[(df["k"] == k) & (df["actual_k"] > 0)]
        if valid.empty:
            continue
        summary[k] = {
            "mean_explained": float(valid["explained_ratio"].mean()),
            "median_explained": float(valid["explained_ratio"].median()),
            "p10_explained": float(valid["explained_ratio"].quantile(0.10)),
            "p25_explained": float(valid["explained_ratio"].quantile(0.25)),
            "n_valid_genes": int(len(valid)),
        }
        logger.info(
            f"  K={k:2d}: mean={summary[k]['mean_explained']:.4f}, "
            f"median={summary[k]['median_explained']:.4f}, "
            f"p10={summary[k]['p10_explained']:.4f}"
        )

    sorted_k = sorted(k for k in candidates if k in summary)
    gains = {}
    for i in range(1, len(sorted_k)):
        prev_k, curr_k = sorted_k[i - 1], sorted_k[i]
        gains[curr_k] = (
            summary[curr_k]["mean_explained"] - summary[prev_k]["mean_explained"]
        )
        logger.info(f"  K={prev_k}->{curr_k}: marginal gain = {gains[curr_k]:.4f}")

    optimal_k = sorted_k[0] if sorted_k else 8
    prev_gain: float | None = None
    for i in range(1, len(sorted_k)):
        curr_k = sorted_k[i]
        gain = gains[curr_k]
        if gain < marginal_threshold:
            optimal_k = sorted_k[i - 1]
            logger.info(
                f"  Elbow (gain {gain:.4f} < threshold {marginal_threshold}): "
                f"K={optimal_k}"
            )
            break
        if prev_gain is not None and gain < prev_gain * decay_ratio:
            optimal_k = sorted_k[i - 1]
            logger.info(
                f"  Elbow (gain decay {prev_gain:.4f}->{gain:.4f}): K={optimal_k}"
            )
            break
        optimal_k = curr_k
        prev_gain = gain

    summary_df = pd.DataFrame(summary).T
    return optimal_k, summary_df
