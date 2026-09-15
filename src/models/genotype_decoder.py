"""Cohort-calibrated local-LD categorical decoder from GLM-PCA base logits to {0, 1, 2} calls."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.special import log_softmax, softmax

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]

_ARMS = ("B0", "B1", "B2", "B3")
_DOSAGE = np.arange(3, dtype=np.float64)
_LOG_COEFFICIENT = np.log([1.0, 2.0, 1.0])


@dataclass(frozen=True)
class DistanceBins:
    edges: FloatArray
    bin_of_snp: IntArray


def distance_bins(positions: FloatArray, offsets: IntArray, n_bins: int = 4) -> DistanceBins:
    positions = np.asarray(positions, dtype=np.float64)
    offsets = np.asarray(offsets, dtype=np.int64)
    if positions.ndim != 1 or offsets.ndim != 1 or len(offsets) < 2:
        raise ValueError("Expected a 1-D reference-position vector and at least one gene offset pair")
    if offsets[0] != 0 or offsets[-1] != len(positions) or np.any(np.diff(offsets) < 0):
        raise ValueError("Gene offsets must start at 0, end at the variant count and never decrease")
    if n_bins < 1:
        raise ValueError("Distance bin count must be positive")
    spans = list(zip(offsets[:-1], offsets[1:]))
    distances = np.concatenate([np.diff(positions[start:end]) for start, end in spans])
    if np.any(distances <= 0):
        raise ValueError("Reference positions must be strictly ascending inside every gene")
    if len(distances) == 0:
        n_bins = 1
    while True:
        edges = np.quantile(distances, np.arange(1, n_bins) / n_bins) if n_bins > 1 else np.zeros(0)
        assigned = np.searchsorted(edges, distances, side="right")
        if n_bins == 1 or np.all(np.bincount(assigned, minlength=n_bins) > 0):
            break
        n_bins -= 1
    bin_of_snp = np.full(len(positions), -1, dtype=np.int64)
    successor = np.zeros(len(positions), dtype=bool)
    for start, end in spans:
        successor[start + 1:end] = True
    bin_of_snp[successor] = assigned
    return DistanceBins(np.asarray(edges, dtype=np.float64), bin_of_snp)


@dataclass(frozen=True)
class GenotypeDecoder:
    arm: str
    cohort_offset: FloatArray
    residual: FloatArray
    bins: DistanceBins
    offsets: IntArray
    cohort_weights: FloatArray
    lambda_u: float
    lambda_a: float
    train_nll_per_call: float
    converged: bool

    def save(self, path) -> None:
        np.savez(path, cohort_offset=self.cohort_offset, residual=self.residual,
                 edges=self.bins.edges, bin_of_snp=self.bins.bin_of_snp, offsets=self.offsets,
                 cohort_weights=self.cohort_weights,
                 scalars=json.dumps({"arm": self.arm, "lambda_u": self.lambda_u,
                                     "lambda_a": self.lambda_a,
                                     "train_nll_per_call": self.train_nll_per_call,
                                     "converged": self.converged}))

    @classmethod
    def load(cls, path) -> "GenotypeDecoder":
        with np.load(path) as data:
            scalars = json.loads(str(data["scalars"]))
            return cls(scalars["arm"], data["cohort_offset"], data["residual"],
                       DistanceBins(data["edges"], data["bin_of_snp"]), data["offsets"],
                       data["cohort_weights"], scalars["lambda_u"], scalars["lambda_a"],
                       scalars["train_nll_per_call"], scalars["converged"])


def _checked_calls(calls: FloatArray) -> FloatArray:
    calls = np.asarray(calls, dtype=np.float64)
    if calls.ndim != 2:
        raise ValueError("Expected a samples-by-variants call matrix")
    if not np.isin(calls[~np.isnan(calls)], [0, 1, 2]).all():
        raise ValueError("Observed diploid calls must be 0, 1, 2; use NaN for missingness")
    return calls


def _checked_labels(labels: IntArray, rows: int, n_cohorts: int) -> IntArray:
    labels = np.asarray(labels)
    if (labels.ndim != 1 or len(labels) != rows or n_cohorts < 1
            or np.any(labels < 0) or np.any(labels >= n_cohorts)):
        raise ValueError("Cohort labels must be one index per sample inside [0, n_cohorts)")
    return labels.astype(np.int64)


def _linear_predictor(decoder: GenotypeDecoder, base_logits: FloatArray,
                      labels: IntArray) -> FloatArray:
    base_logits = np.asarray(base_logits, dtype=np.float64)
    if base_logits.ndim != 2 or base_logits.shape[1] != len(decoder.bins.bin_of_snp):
        raise ValueError("Base logits must be a samples-by-variants matrix matching the decoder")
    labels = _checked_labels(labels, len(base_logits), len(decoder.cohort_offset))
    return base_logits + decoder.cohort_offset[labels]


def _predecessors(calls: FloatArray, bin_of_snp: IntArray) -> tuple[IntArray, BoolArray]:
    """Teacher-forced predecessor call per SNP; a gene start or a missing call breaks the chain."""
    previous = np.full(calls.shape, np.nan)
    previous[:, 1:] = calls[:, :-1]
    linked = (bin_of_snp >= 0)[None, :] & ~np.isnan(previous)
    return np.where(linked, np.nan_to_num(previous), 0).astype(np.int64), linked


def _categorical_logits(eta: FloatArray, residual: FloatArray, bin_of_snp: IntArray,
                        previous: IntArray, linked: BoolArray) -> FloatArray:
    # ponytail: dense (N, J, 3) logits; chunk over rows only if a panel outgrows memory.
    logits = eta[:, :, None] * _DOSAGE + _LOG_COEFFICIENT
    shift = residual[np.where(linked, bin_of_snp[None, :], 0), previous]
    return logits + np.where(linked[:, :, None], shift, 0.0)


def call_logits(decoder: GenotypeDecoder, calls: FloatArray, base_logits: FloatArray,
                labels: IntArray) -> FloatArray:
    calls = _checked_calls(calls)
    eta = _linear_predictor(decoder, base_logits, labels)
    if calls.shape != eta.shape:
        raise ValueError("Calls and base logits must have the same shape")
    previous, linked = _predecessors(calls, decoder.bins.bin_of_snp)
    return _categorical_logits(eta, decoder.residual, decoder.bins.bin_of_snp, previous, linked)


def nll_calls(decoder: GenotypeDecoder, calls: FloatArray, base_logits: FloatArray,
              labels: IntArray) -> FloatArray:
    calls = _checked_calls(calls)
    logits = call_logits(decoder, calls, base_logits, labels)
    target = np.nan_to_num(calls).astype(np.int64)[:, :, None]
    nll = -np.take_along_axis(log_softmax(logits, axis=2), target, axis=2)[:, :, 0]
    return np.where(np.isnan(calls), np.nan, nll)


def sample_calls(decoder: GenotypeDecoder, base_logits: FloatArray, labels: IntArray,
                 rng: np.random.Generator) -> NDArray[np.int8]:
    eta = _linear_predictor(decoder, base_logits, labels)
    calls = np.zeros(eta.shape, dtype=np.int8)
    previous = np.zeros(len(eta), dtype=np.int64)
    for snp, bin_index in enumerate(decoder.bins.bin_of_snp):
        logits = eta[:, snp, None] * _DOSAGE + _LOG_COEFFICIENT
        if bin_index >= 0:
            logits = logits + decoder.residual[bin_index, previous]
        cumulative = np.cumsum(softmax(logits, axis=1), axis=1)
        cumulative[:, -1] = 1.0
        previous = (rng.random((len(eta), 1)) < cumulative).argmax(axis=1)
        calls[:, snp] = previous
    return calls


def fit_decoder(arm: str, calls: FloatArray, base_logits: FloatArray, labels: IntArray,
                train_indices: IntArray, positions: FloatArray, offsets: IntArray,
                n_cohorts: int, *, lambda_u: float = 1.0, lambda_a: float = 1.0,
                n_bins: int = 4, max_iter: int = 500) -> GenotypeDecoder:
    if arm not in _ARMS:
        raise ValueError(f"Decoder arm must be one of {_ARMS}")
    calls = _checked_calls(calls)
    base_logits = np.asarray(base_logits, dtype=np.float64)
    if base_logits.shape != calls.shape:
        raise ValueError("Base logits must have the same shape as the calls")
    labels = _checked_labels(labels, len(calls), n_cohorts)
    train_indices = np.asarray(train_indices, dtype=np.int64)
    if (train_indices.ndim != 1 or len(train_indices) == 0
            or len(np.unique(train_indices)) != len(train_indices)
            or np.any(train_indices < 0) or np.any(train_indices >= len(calls))):
        raise ValueError("Training indices must be unique valid row indices")
    bins = distance_bins(positions, offsets, n_bins)
    if len(bins.bin_of_snp) != calls.shape[1]:
        raise ValueError("Reference positions must cover every variant column")

    train_calls = calls[train_indices]
    train_labels = labels[train_indices]
    weights = np.bincount(train_labels, minlength=n_cohorts) / len(train_indices)
    previous, linked = _predecessors(train_calls, bins.bin_of_snp)
    observed = ~np.isnan(train_calls)

    offset = torch.zeros((n_cohorts, calls.shape[1]), dtype=torch.float64,
                         requires_grad=arm in ("B1", "B3"))
    residual = torch.zeros((len(bins.edges) + 1, 3, 3), dtype=torch.float64,
                           requires_grad=arm in ("B2", "B3"))
    eta_base = torch.as_tensor(base_logits[train_indices])
    weight = torch.as_tensor(weights)
    label = torch.as_tensor(train_labels)
    bin_index = torch.as_tensor(np.where(linked, bins.bin_of_snp[None, :], 0))
    predecessor = torch.as_tensor(previous)
    conditioned = torch.as_tensor(linked)
    target = torch.as_tensor(np.nan_to_num(train_calls).astype(np.int64))[:, :, None]
    mask = torch.as_tensor(observed)
    dosage = torch.as_tensor(_DOSAGE)
    log_coefficient = torch.as_tensor(_LOG_COEFFICIENT)

    def penalized_nll() -> torch.Tensor:
        centered = offset - weight @ offset
        eta = eta_base + centered[label]
        logits = eta[:, :, None] * dosage + log_coefficient
        logits = logits + torch.where(conditioned[:, :, None], residual[bin_index, predecessor], 0.0)
        nll = -logits.log_softmax(2).gather(2, target)[:, :, 0]
        return (nll * mask).sum() + lambda_u * (centered**2).sum() + lambda_a * (residual**2).sum()

    parameters = [tensor for tensor in (offset, residual) if tensor.requires_grad]
    converged = True
    if parameters:
        optimizer = torch.optim.LBFGS(parameters, line_search_fn="strong_wolfe", max_iter=max_iter,
                                      tolerance_grad=1e-7, tolerance_change=1e-12, history_size=50)

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            loss = penalized_nll()
            loss.backward()
            return loss

        optimizer.step(closure)
        loss = closure()
        gradient = max(float(tensor.grad.abs().max()) for tensor in parameters)
        # Tolerance scales with the summed objective, which grows with the observed call count.
        converged = bool(torch.isfinite(loss) and gradient <= 1e-6 * max(int(observed.sum()), 1))

    with torch.no_grad():
        cohort_offset = (offset - weight @ offset).numpy()
    decoder = GenotypeDecoder(arm, cohort_offset, residual.detach().numpy(), bins,
                              np.asarray(offsets, dtype=np.int64), weights, float(lambda_u),
                              float(lambda_a), 0.0, converged)
    train_nll = np.nanmean(nll_calls(decoder, train_calls, base_logits[train_indices], train_labels))
    return replace(decoder, train_nll_per_call=float(train_nll))
