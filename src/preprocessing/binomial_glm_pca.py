from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.special import expit, logit

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class BinomialGLMPCA:
    factors: FloatArray
    loadings: FloatArray
    intercept: FloatArray
    objective_initial: float
    objective_final: float
    converged: bool

    def probabilities(self, factors: FloatArray) -> FloatArray:
        return expit(self.intercept + factors @ self.loadings.T)

    def fisher_diagonal(self, factors: FloatArray, scale: FloatArray) -> FloatArray:
        p = self.probabilities(factors)
        information = (2 * p * (1 - p)).mean(axis=0) @ (self.loadings**2)
        return information * scale**2 / len(self.loadings)


def _score_calls(calls: FloatArray, loadings: FloatArray, intercept: FloatArray) -> FloatArray:
    observed = np.isfinite(calls)
    y = np.nan_to_num(calls)
    shape = (len(calls), loadings.shape[1])

    def objective(flat: FloatArray) -> tuple[float, FloatArray]:
        z = flat.reshape(shape)
        eta = intercept + z @ loadings.T
        residual = (2 * expit(eta) - y) * observed
        loss = np.sum((2 * np.logaddexp(0, eta) - y * eta) * observed)
        return float(loss + 0.5 * np.sum(z**2)), (residual @ loadings + z).ravel()

    result = minimize(objective, np.zeros(np.prod(shape)), jac=True, method="L-BFGS-B",
                      options={"maxiter": 300, "ftol": 1e-12, "gtol": 1e-6})
    if not np.isfinite(result.fun) or (not result.success and np.max(np.abs(result.jac)) > 1e-4):
        raise RuntimeError(f"Binomial factor scoring failed: {result.message}")
    return result.x.reshape(shape)


def fit_binomial_glm_pca(
    calls: FloatArray, train_indices: NDArray[np.int64], components: int, *, max_iter: int = 150,
) -> BinomialGLMPCA:
    calls = np.asarray(calls, dtype=np.float64)
    train_indices = np.asarray(train_indices, dtype=np.int64)
    if calls.ndim != 2 or calls.size == 0:
        raise ValueError("Expected a nonempty samples-by-variants matrix")
    observed_values = calls[~np.isnan(calls)]
    if not np.isin(observed_values, [0, 1, 2]).all():
        raise ValueError("Observed diploid calls must be 0, 1, 2; use NaN for missingness")
    if (train_indices.ndim != 1 or len(train_indices) == 0
            or len(np.unique(train_indices)) != len(train_indices)
            or np.any(train_indices < 0) or np.any(train_indices >= len(calls))):
        raise ValueError("Training indices must be unique valid row indices")
    train = calls[train_indices]
    n, j = train.shape
    if not 1 <= components < min(n, j) or max_iter < 1:
        raise ValueError("Components must be below sample/variant counts; iterations must be positive")
    observed = np.isfinite(train)
    counts = observed.sum(axis=0)
    if np.any(counts == 0):
        raise ValueError("Every variant needs an observed training call")
    y = np.nan_to_num(train)
    intercept = logit((y.sum(axis=0) + 0.5) / (2 * counts + 1))
    centered = np.where(observed, logit((y + 0.5) / 3) - intercept, 0)
    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    z = u[:, :components] * np.sqrt(s[:components])
    v = vt[:components].T * np.sqrt(s[:components])
    split = n * components
    end = split + j * components

    def objective(flat: FloatArray) -> tuple[float, FloatArray]:
        factors = flat[:split].reshape(n, components)
        loadings = flat[split:end].reshape(j, components)
        eta = flat[end:] + factors @ loadings.T
        residual = (2 * expit(eta) - y) * observed
        nll = np.sum((2 * np.logaddexp(0, eta) - y * eta) * observed)
        penalty = 0.5 * (np.sum(factors**2) + np.sum(loadings**2))
        gradient = np.concatenate(((residual @ loadings + factors).ravel(),
                                   (residual.T @ factors + loadings).ravel(), residual.sum(axis=0)))
        return float(nll + penalty), gradient

    initial = np.concatenate((z.ravel(), v.ravel(), intercept))
    result = minimize(objective, initial, jac=True, method="L-BFGS-B",
                      options={"maxiter": max_iter, "ftol": 1e-9, "gtol": 1e-5})
    initial_loss = objective(initial)[0]
    if not np.isfinite(result.fun) or result.fun > initial_loss:
        raise RuntimeError("Binomial GLM-PCA failed to reduce its penalized likelihood")
    loadings = result.x[split:end].reshape(j, components)
    intercept = result.x[end:]
    factors = np.zeros((len(calls), components))
    factors[train_indices] = _score_calls(train, loadings, intercept)
    held = np.setdiff1d(np.arange(len(calls)), train_indices)
    if len(held):
        factors[held] = _score_calls(calls[held], loadings, intercept)
    return BinomialGLMPCA(factors, loadings, intercept, initial_loss,
                         float(result.fun), bool(result.success))


def information_schedule(information: FloatArray) -> NDArray[np.int64]:
    information = np.asarray(information, dtype=np.float64)
    if information.ndim != 2 or not np.isfinite(information).all() or np.any(information < 0):
        raise ValueError("Expected finite non-negative gene-by-factor information")
    if np.ptp(information) <= 1e-12:
        return np.ones(information.shape, dtype=np.int64)
    low, high = np.quantile(information, [1 / 3, 2 / 3])
    return np.where(information < low, 0, np.where(information > high, 2, 1))
