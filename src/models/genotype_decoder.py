"""Cohort-calibrated categorical decoder from GLM-PCA base logits to {0, 1, 2} calls.

Arms B0-B3 add a cohort offset and/or a local-LD chain; arm T keeps the offset, drops the chain and
tilts heterozygosity at exactly preserved expected dosage.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.special import log_softmax, softmax

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]

_ARMS = ("B0", "B1", "B2", "B3", "T")
_TILT_SCOPES = ("none", "snp", "superpop", "cohort")
_OFFSET_GAUGES = ("centered", "pooled_af")
_FITTED_OFFSET_ARMS = ("B1", "B3", "T")
# ponytail: a fixed Newton budget instead of a convergence loop; pooled_af_residual reports what it
# reached and fit_decoder marks the fit unconverged past 1e-8, which is the upgrade trigger.
_NEWTON_STEPS = 25
_AF_RESIDUAL_TOLERANCE = 1e-8
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
    tilt: FloatArray = field(default_factory=lambda: np.zeros(0))
    tilt_scope: str = "none"
    lambda_tilt: float = 0.0
    superpop_of_cohort: IntArray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    offset_gauge: str = "centered"
    pooled_af_residual: float = math.nan

    def save(self, path) -> None:
        np.savez(path, cohort_offset=self.cohort_offset, residual=self.residual,
                 edges=self.bins.edges, bin_of_snp=self.bins.bin_of_snp, offsets=self.offsets,
                 cohort_weights=self.cohort_weights, tilt=self.tilt,
                 superpop_of_cohort=self.superpop_of_cohort,
                 scalars=json.dumps({"arm": self.arm, "lambda_u": self.lambda_u,
                                     "lambda_a": self.lambda_a,
                                     "train_nll_per_call": self.train_nll_per_call,
                                     "converged": self.converged, "tilt_scope": self.tilt_scope,
                                     "lambda_tilt": self.lambda_tilt,
                                     "offset_gauge": self.offset_gauge,
                                     "pooled_af_residual": self.pooled_af_residual}))

    @classmethod
    def load(cls, path) -> "GenotypeDecoder":
        with np.load(path) as data:
            scalars = json.loads(str(data["scalars"]))
            # Files written before arm T carry no tilt fields; the dataclass defaults fill them.
            tilt = (dict(tilt=data["tilt"], tilt_scope=scalars["tilt_scope"],
                         lambda_tilt=scalars["lambda_tilt"],
                         superpop_of_cohort=data["superpop_of_cohort"]) if "tilt" in data else {})
            return cls(scalars["arm"], data["cohort_offset"], data["residual"],
                       DistanceBins(data["edges"], data["bin_of_snp"]), data["offsets"],
                       data["cohort_weights"], scalars["lambda_u"], scalars["lambda_a"],
                       scalars["train_nll_per_call"], scalars["converged"],
                       # Files written before the pooled-AF gauge carry the centered offset.
                       offset_gauge=scalars.get("offset_gauge", "centered"),
                       pooled_af_residual=scalars.get("pooled_af_residual", math.nan), **tilt)


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


def _tilted_logits(eta: torch.Tensor, tilt: torch.Tensor) -> torch.Tensor:
    """Logits (0, log 2 + tau + log x, 2 log x) whose expected dosage is exactly 2 sigmoid(eta).

    x is the positive root of (1 - p) x^2 + t (1 - 2p) x - p = 0 with t = exp(tau), solved through
    the minor probability and mirrored (x(p) = 1 / x(1 - p)) so neither tail cancels.
    """
    minor_logit = torch.where(eta > 0, -eta, eta)
    log_minor = torch.nn.functional.logsigmoid(minor_logit)
    minor = log_minor.exp()
    # ponytail: t^2 overflows past tau ~ 350; a ridge-fitted tilt never gets near it.
    spread = tilt.exp() * (1 - 2 * minor)
    log_root = (math.log(2) + log_minor
                - torch.log(spread + torch.sqrt(spread**2 + 4 * minor * (1 - minor))))
    log_x = torch.where(eta > 0, -log_root, log_root)
    return torch.stack([torch.zeros_like(log_x), math.log(2) + tilt + log_x, 2 * log_x], dim=-1)


def _tilt_group(tilt_scope: str, labels: IntArray, superpop_of_cohort: IntArray) -> IntArray:
    """Row of the (groups, J) tilt table each sample reads; a per-SNP tilt is a single row."""
    labels = np.asarray(labels, dtype=np.int64)
    if tilt_scope == "snp":
        return np.zeros_like(labels)
    return superpop_of_cohort[labels] if tilt_scope == "superpop" else labels


def _unchained_logits(decoder: GenotypeDecoder, eta: FloatArray, labels: IntArray) -> FloatArray:
    # ponytail: dense (N, J, 3) logits; chunk over rows only if a panel outgrows memory.
    if decoder.arm != "T":
        return eta[:, :, None] * _DOSAGE + _LOG_COEFFICIENT
    group = _tilt_group(decoder.tilt_scope, labels, decoder.superpop_of_cohort)
    tilt = decoder.tilt.reshape(-1, eta.shape[1])[group]
    return _tilted_logits(torch.as_tensor(eta), torch.as_tensor(tilt)).numpy()


def call_logits(decoder: GenotypeDecoder, calls: FloatArray, base_logits: FloatArray,
                labels: IntArray) -> FloatArray:
    calls = _checked_calls(calls)
    eta = _linear_predictor(decoder, base_logits, labels)
    if calls.shape != eta.shape:
        raise ValueError("Calls and base logits must have the same shape")
    logits = _unchained_logits(decoder, eta, labels)
    if decoder.arm == "T":
        return logits
    previous, linked = _predecessors(calls, decoder.bins.bin_of_snp)
    shift = decoder.residual[np.where(linked, decoder.bins.bin_of_snp[None, :], 0), previous]
    return logits + np.where(linked[:, :, None], shift, 0.0)


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
    unchained = _unchained_logits(decoder, eta, labels)
    calls = np.zeros(eta.shape, dtype=np.int8)
    previous = np.zeros(len(eta), dtype=np.int64)
    for snp, bin_index in enumerate(decoder.bins.bin_of_snp):
        logits = unchained[:, snp]
        if bin_index >= 0 and decoder.arm != "T":
            logits = logits + decoder.residual[bin_index, previous]
        cumulative = np.cumsum(softmax(logits, axis=1), axis=1)
        cumulative[:, -1] = 1.0
        previous = (rng.random((len(eta), 1)) < cumulative).argmax(axis=1)
        calls[:, snp] = previous
    return calls


def fit_decoder(arm: str, calls: FloatArray, base_logits: FloatArray, labels: IntArray,
                train_indices: IntArray, positions: FloatArray, offsets: IntArray,
                n_cohorts: int, *, lambda_u: float = 1.0, lambda_a: float = 1.0,
                n_bins: int = 4, max_iter: int = 500, tilt_scope: str = "none",
                lambda_tilt: float = 1.0, offset_gauge: str = "pooled_af",
                superpop_of_cohort: IntArray | None = None) -> GenotypeDecoder:
    """Fit the cohort offset, the local-LD residual and the tilt on the training rows.

    `offset_gauge="pooled_af"` constrains the fitted offset so that the train pooled model allele
    frequency equals the observed one per SNP; `"centered"` instead sets its train-cohort-weighted
    mean to zero per SNP. `cohort_weights` records the train cohort proportions either way, but the
    pooled-AF gauge does not use them to centre.
    """
    if arm not in _ARMS:
        raise ValueError(f"Decoder arm must be one of {_ARMS}")
    if tilt_scope not in _TILT_SCOPES:
        raise ValueError(f"Tilt scope must be one of {_TILT_SCOPES}")
    if offset_gauge not in _OFFSET_GAUGES:
        raise ValueError(f"Offset gauge must be one of {_OFFSET_GAUGES}")
    if (arm == "T") != (tilt_scope != "none"):
        raise ValueError("Arm T needs a tilt scope and arms B0-B3 take none")
    if tilt_scope == "superpop":
        superpop_of_cohort = np.asarray([] if superpop_of_cohort is None else superpop_of_cohort)
        superpops = np.unique(superpop_of_cohort)
        if (superpop_of_cohort.ndim != 1 or len(superpop_of_cohort) != n_cohorts
                or not np.array_equal(superpops, np.arange(len(superpops)))):
            raise ValueError("Superpopulations must be one contiguous 0-based index per cohort")
        superpop_of_cohort = superpop_of_cohort.astype(np.int64)
    else:
        superpop_of_cohort = np.zeros(0, dtype=np.int64)
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
    # Arms that never fit an offset leave it at zero, where the two gauges agree.
    gauge = offset_gauge if arm in _FITTED_OFFSET_ARMS else "centered"
    # Observed-call allele frequency; a SNP with no observed training call has no target to hit.
    target_af = np.nansum(train_calls, axis=0) / np.maximum(observed.sum(axis=0), 1) / 2

    offset = torch.zeros((n_cohorts, calls.shape[1]), dtype=torch.float64,
                         requires_grad=arm in _FITTED_OFFSET_ARMS)
    residual = torch.zeros((len(bins.edges) + 1, 3, 3), dtype=torch.float64,
                           requires_grad=arm in ("B2", "B3"))
    tilt_shape = {"none": (0,), "snp": (calls.shape[1],), "cohort": (n_cohorts, calls.shape[1]),
                  "superpop": (len(np.unique(superpop_of_cohort)), calls.shape[1])}[tilt_scope]
    tilt = torch.zeros(tilt_shape, dtype=torch.float64, requires_grad=arm == "T")
    group = torch.as_tensor(_tilt_group(tilt_scope, train_labels, superpop_of_cohort))
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
    frequency = torch.as_tensor(target_af)

    def pooled_af_shift(raw: torch.Tensor) -> torch.Tensor:
        """Per-SNP s solving mean_i sigmoid(eta_base + raw[c_i] - s) = target_af.

        The mean is decreasing in s, so Newton converges from zero; the iteration is unrolled in
        the graph so the fit differentiates through the constraint.
        """
        shift = torch.zeros(raw.shape[1], dtype=raw.dtype)
        for _ in range(_NEWTON_STEPS):
            p = torch.sigmoid(eta_base + raw[label] - shift)
            shift = shift + (p.mean(0) - frequency) / p.mul(1 - p).mean(0).clamp_min(1e-12)
        return shift

    def gauged_offset() -> torch.Tensor:
        """The offset the decoder reports, under the requested identifiability gauge."""
        return offset - (pooled_af_shift(offset) if gauge == "pooled_af" else weight @ offset)

    def penalized_nll() -> torch.Tensor:
        gauged = gauged_offset()
        eta = eta_base + gauged[label]
        if arm == "T":
            logits = _tilted_logits(eta, tilt.reshape(-1, eta.shape[1])[group])
        else:
            logits = eta[:, :, None] * dosage + log_coefficient
            logits = logits + torch.where(conditioned[:, :, None], residual[bin_index, predecessor], 0.0)
        nll = -logits.log_softmax(2).gather(2, target)[:, :, 0]
        # The pooled-AF constraint already fixes the per-SNP constant, so its ridge pins the raw
        # offset and picks the minimum-norm representative of the constrained fit.
        ridged = offset if gauge == "pooled_af" else gauged
        return ((nll * mask).sum() + lambda_u * (ridged**2).sum()
                + lambda_a * (residual**2).sum() + lambda_tilt * (tilt**2).sum())

    parameters = [tensor for tensor in (offset, residual, tilt) if tensor.requires_grad]
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
        # The constrained offset is what every decoding path reads, so it is what gets stored.
        gauged = gauged_offset()
        cohort_offset = gauged.numpy()
        af_residual = math.nan if gauge == "centered" else float(
            (torch.sigmoid(eta_base + gauged[label]).mean(0) - frequency).abs().max())
    converged = converged and (math.isnan(af_residual) or af_residual <= _AF_RESIDUAL_TOLERANCE)
    decoder = GenotypeDecoder(arm, cohort_offset, residual.detach().numpy(), bins,
                              np.asarray(offsets, dtype=np.int64), weights, float(lambda_u),
                              float(lambda_a), 0.0, converged, tilt=tilt.detach().numpy(),
                              tilt_scope=tilt_scope,
                              lambda_tilt=float(lambda_tilt) if arm == "T" else 0.0,
                              superpop_of_cohort=superpop_of_cohort, offset_gauge=gauge,
                              pooled_af_residual=af_residual)
    train_nll = np.nanmean(nll_calls(decoder, train_calls, base_logits[train_indices], train_labels))
    return replace(decoder, train_nll_per_call=float(train_nll))
