"""GPU-accelerated GLM-PCA via PyTorch (Townes et al. 2019, Poisson family).

This is a third backend alongside Python `glmpca-py` and Rust `glm_pca_rs`.
Provides the same per-gene API but executes on a CUDA device — ideal when
the project has an idle GPU. Per-iteration the Newton step uses batched
``torch.linalg.solve`` over all N samples (or M features), so a single
gene fits in <50 ms on an RTX A6000.

Algorithm — coordinate-block Newton (same as Rust impl):
    For each outer iter:
        η = a + V Z'                 # (M, N)
        μ = exp(η)                   # Poisson mean
        # Z update — per-sample j: Newton step in (L, L)
        z_j ← z_j + (V'·diag(μ_j)·V + λI)⁻¹ V'(y_j − μ_j)
        # V update — per-feature i: same form
        # a update — closed-form Newton

Batched matrix-solve does the per-sample/per-feature work all at once on
GPU. Designed to run gene-by-gene (avoids the variable-M padding issue);
the per-gene wall time on GPU is dominated by kernel-launch overhead
(~5–20 ms), so end-to-end speedup over CPU is moderate (~3–8×) for the
24 576-gene full run.

Citation: Townes, Hicks, Aryee, Irizarry. "Feature selection and
dimension reduction for single-cell RNA-Seq based on a multinomial
model." *Genome Biology* (2019). doi:10.1186/s13059-019-1861-6
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch

from src.preprocessing.config import (
    MARGINAL_GAIN_DECAY_RATIO,
    MARGINAL_GAIN_THRESHOLD,
    PCA_CANDIDATES,
    PCA_SAMPLE_GENES,
)

logger = logging.getLogger(__name__)


def _device() -> torch.device:
    """Resolve CUDA device from env or default cuda:0 (falls back to CPU)."""
    if not torch.cuda.is_available():
        return torch.device("cpu")
    idx = int(os.environ.get("HIPODIT_GPU", "1"))
    if idx >= torch.cuda.device_count():
        idx = 0
    return torch.device(f"cuda:{idx}")


_DEVICE = _device()
logger.info(f"GLM-PCA torch backend: device={_DEVICE}")


@torch.inference_mode()
def fit_poisson_torch(
    y_np: np.ndarray,         # (n_samples, n_variants), float32 dosage
    l: int,
    max_iter: int = 100,
    tol: float = 1e-4,
    penalty: float = 1.0,
    seed: int = 42,
    device: torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Fit Poisson GLM-PCA on GPU.

    Returns ``(factors[N, L], loadings[M, L], intercept[M], deviance, n_iter)``
    as numpy arrays so callers do not need PyTorch types.
    """
    dev = device or _DEVICE
    Y = torch.as_tensor(y_np, dtype=torch.float32, device=dev).t().contiguous()  # (M, N)
    M, N = Y.shape
    L = min(l, M, N)
    if L < 2:
        raise ValueError(f"Need at least L>=2 components; got effective L={L}")

    g = torch.Generator(device=dev).manual_seed(seed)
    Z = 0.05 * torch.randn(N, L, generator=g, device=dev, dtype=torch.float32)
    V = 0.05 * torch.randn(M, L, generator=g, device=dev, dtype=torch.float32)
    a = torch.log(Y.mean(dim=1) + 1.0).clamp(min=-10.0)  # (M,)

    eye_L = torch.eye(L, device=dev, dtype=torch.float32)
    lam = max(float(penalty), 1e-6)

    deviance = []
    prev_dev = float("inf")
    n_iter_actual = 0

    for it in range(max_iter):
        eta = a.unsqueeze(1) + V @ Z.t()      # (M, N)
        mu = eta.exp().clamp(min=1e-9, max=1e8)
        dev_val = _poisson_deviance(Y, mu)
        deviance.append(dev_val)
        n_iter_actual = it + 1

        if it > 0:
            rel = abs(prev_dev - dev_val) / (abs(prev_dev) + 1e-9)
            if rel < tol:
                break
        prev_dev = dev_val

        # ── Z update ────────────────────────────────────────────────
        resid = Y - mu                          # (M, N)
        grad_Z = resid.t() @ V - lam * Z        # (N, L)
        # H_Z[j] = V' · diag(mu[:,j]) · V + λ I, batched over j
        # einsum: 'mn,ml,mk->nlk'
        H_Z = torch.einsum("mn,ml,mk->nlk", mu, V, V) + lam * eye_L
        try:
            delta_Z = torch.linalg.solve(H_Z, grad_Z.unsqueeze(-1)).squeeze(-1)
            Z = Z + delta_Z
        except Exception as e:
            logger.warning(f"Z solve failed at iter {it}: {e}; using gradient step")
            Z = Z + 0.01 * grad_Z

        # ── V update ────────────────────────────────────────────────
        eta = a.unsqueeze(1) + V @ Z.t()
        mu = eta.exp().clamp(min=1e-9, max=1e8)
        resid = Y - mu
        grad_V = resid @ Z - lam * V            # (M, L)
        H_V = torch.einsum("mn,nl,nk->mlk", mu, Z, Z) + lam * eye_L
        try:
            delta_V = torch.linalg.solve(H_V, grad_V.unsqueeze(-1)).squeeze(-1)
            V = V + delta_V
        except Exception as e:
            logger.warning(f"V solve failed at iter {it}: {e}; using gradient step")
            V = V + 0.01 * grad_V

        # ── intercept update ────────────────────────────────────────
        eta = a.unsqueeze(1) + V @ Z.t()
        mu = eta.exp().clamp(min=1e-9, max=1e8)
        ga = (Y - mu).sum(dim=1)
        ha = mu.sum(dim=1).clamp(min=1e-3)
        a = a + ga / ha

    # Move to host
    return (
        Z.cpu().numpy().astype(np.float32),
        V.cpu().numpy().astype(np.float32),
        a.cpu().numpy().astype(np.float32),
        np.asarray(deviance, dtype=np.float32),
        n_iter_actual,
    )


def _poisson_deviance(y: torch.Tensor, mu: torch.Tensor) -> float:
    mu = mu.clamp(min=1e-9)
    # term = y log(y/μ) − (y − μ) ; if y==0 → −(−μ) = μ
    log_term = torch.where(
        y > 0,
        y * (y / mu).log() - (y - mu),
        -(y - mu),
    )
    return float((2.0 * log_term).sum().clamp_min(0.0).item())


# ── Public API mirroring src.preprocessing.glm_pca ────────────────────────
def glm_pca_torch_single_gene(
    gene_name: str,
    matrix: np.ndarray,
    n_components: int,
    train_indices: np.ndarray | None = None,
    fam: str = "poi",
    max_iter: int = 100,
) -> dict | None:
    """GPU-resident GLM-PCA for one gene; same return schema as PCA."""
    n_vars = matrix.shape[1]
    n_fit = matrix.shape[0] if train_indices is None else int(len(train_indices))
    n_comp = min(n_components, n_vars, n_fit)
    if n_comp < 2:
        return None

    fit_matrix = matrix if train_indices is None else matrix[train_indices]

    try:
        f, v, _intercept, dev, _ = fit_poisson_torch(
            np.ascontiguousarray(fit_matrix, dtype=np.float32),
            l=n_comp,
            max_iter=max_iter,
            tol=1e-4,
            penalty=1.0,
            seed=42,
        )
    except Exception as exc:
        logger.warning(f"Torch GLM-PCA failed for {gene_name}: {exc}")
        return None

    # Pseudo-R² (deviance reduction)
    explained = (
        float(max(0.0, 1.0 - dev[-1] / dev[0]))
        if len(dev) >= 2 and dev[0] > 0 else 0.0
    )

    if train_indices is None:
        transformed = f
    else:
        # Symmetric OLS projection through V for ALL rows (train + held).
        # Using the GLM-PCA Poisson-fitted f only for train rows leaves train
        # and held in different coordinate spaces; downstream PCA evaluation on
        # test then collapses because the linear-projected test sits on a much
        # narrower scale than the Poisson-scored train. Project everything via
        # (V^T V)^-1 V^T (X - μ_train) so train and held share one basis.
        ti = np.asarray(train_indices, dtype=np.int64)
        train_mean = matrix[ti].mean(axis=0)
        X_centered = matrix - train_mean
        VtV = v.T @ v + 1e-6 * np.eye(n_comp, dtype=np.float32)
        cv = X_centered @ v
        transformed = np.linalg.solve(VtV, cv.T).T.astype(np.float32)

    features = {f"{gene_name}:{k}": transformed[:, k] for k in range(n_comp)}
    return {
        "features": features,
        "explained_total": explained,
        "explained_per_component": [explained / n_comp] * n_comp,
        "n_variants": n_vars,
        "actual_k": n_comp,
    }


def evaluate_torch_k_for_gene(
    gene_name: str,
    matrix: np.ndarray,
    k: int,
    train_indices: np.ndarray | None = None,
    fam: str = "poi",
    max_iter: int = 100,
) -> tuple[str, int, float, int]:
    """Grid-search evaluation: pseudo-R² per (gene, k)."""
    n_vars = matrix.shape[1]
    n_fit = matrix.shape[0] if train_indices is None else int(len(train_indices))
    actual_k = min(k, n_vars, n_fit)
    if actual_k < 2:
        return (gene_name, k, 0.0, 0)

    fit_matrix = matrix if train_indices is None else matrix[train_indices]
    try:
        _, _, _, dev, _ = fit_poisson_torch(
            np.ascontiguousarray(fit_matrix, dtype=np.float32),
            l=actual_k,
            max_iter=max_iter,
            tol=1e-4,
            penalty=1.0,
            seed=42,
        )
    except Exception as exc:
        logger.warning(f"Torch GLM-PCA eval failed for {gene_name} (k={k}): {exc}")
        return (gene_name, k, 0.0, 0)

    explained = (
        float(max(0.0, 1.0 - dev[-1] / dev[0]))
        if len(dev) >= 2 and dev[0] > 0 else 0.0
    )
    return (gene_name, k, explained, actual_k)


def grid_search_optimal_torch(
    gene_matrices: dict[str, np.ndarray],
    candidates: list[int] | None = None,
    marginal_threshold: float = MARGINAL_GAIN_THRESHOLD,
    decay_ratio: float = MARGINAL_GAIN_DECAY_RATIO,
    n_sample_genes: int = PCA_SAMPLE_GENES,
    train_indices: np.ndarray | None = None,
    fam: str = "poi",
    max_iter: int = 100,
) -> tuple[int, pd.DataFrame]:
    """Grid search optimal K via deviance-based marginal-gain elbow on GPU.

    Identical contract to :func:`grid_search_optimal_glm_pca` — works in
    place of either backend.
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
        f"Torch GLM-PCA grid search on {_DEVICE}: K candidates={candidates}, "
        f"sample genes={len(sample_genes)}/{len(all_genes)}"
    )

    # Sequential per (gene, k) — GPU has no benefit from python-side threading
    # (kernel launches serialise on a single CUDA stream).
    results = []
    for g in sample_genes:
        for k in candidates:
            results.append(
                evaluate_torch_k_for_gene(
                    g, gene_matrices[g], k, train_indices, fam, max_iter,
                )
            )

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
            f"median={summary[k]['median_explained']:.4f}"
        )

    sorted_k = sorted(k for k in candidates if k in summary)
    optimal_k = sorted_k[0] if sorted_k else 8
    prev_gain: float | None = None
    for i in range(1, len(sorted_k)):
        prev_k, curr_k = sorted_k[i - 1], sorted_k[i]
        gain = summary[curr_k]["mean_explained"] - summary[prev_k]["mean_explained"]
        logger.info(f"  K={prev_k}->{curr_k}: marginal gain = {gain:.4f}")
        if gain < marginal_threshold:
            optimal_k = prev_k
            logger.info(f"  Elbow (gain {gain:.4f} < {marginal_threshold}): K={optimal_k}")
            break
        if prev_gain is not None and gain < prev_gain * decay_ratio:
            optimal_k = prev_k
            logger.info(f"  Elbow (gain decay {prev_gain:.4f}->{gain:.4f}): K={optimal_k}")
            break
        optimal_k = curr_k
        prev_gain = gain

    return optimal_k, pd.DataFrame(summary).T
