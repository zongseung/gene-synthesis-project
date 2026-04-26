"""High-level evaluation pipeline for synthetic genotype data.

This module ties together :mod:`src.evaluation.dupi` and
:mod:`src.evaluation.distribution_metrics` into a single ``evaluate`` entry
point that operates on already-projected PCA scores plus superpopulation
labels.

It is intentionally decoupled from project-specific I/O — readers /
caches live in :mod:`src.evaluation._io` and the CLI shim in
``scripts/evaluate_synthetic_metrics.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.neighbors import NearestNeighbors

from src.evaluation.distribution_metrics import (
    centroid_distance,
    gaussian_w2_distance,
    mmd_rbf,
    same_class_coverage,
)
from src.evaluation.dupi import dupi_score, ui_pi_from_dupi

__all__ = [
    "EvaluationReport",
    "centroid_rows",
    "class_metric_rows",
    "evaluate",
    "global_metrics",
]


@dataclass
class EvaluationReport:
    """Structured container for per-run synthetic-data evaluation results."""

    dupi: dict[str, Any]
    distribution_distances: dict[str, float]
    centroid_rows: list[dict[str, Any]] = field(default_factory=list)
    class_metric_rows: list[dict[str, Any]] = field(default_factory=list)


def evaluate(
    real_pcs: np.ndarray,
    syn_pcs: np.ndarray,
    real_sp: np.ndarray,
    syn_sp: np.ndarray,
    *,
    k: int = 1,
    tau: float = 5.0,
) -> EvaluationReport:
    """Run the full DUPI + distribution-distance pipeline.

    Parameters
    ----------
    real_pcs, syn_pcs : np.ndarray
        PCA scores (or any other shared metric space) for real and synthetic
        samples respectively.
    real_sp, syn_sp : np.ndarray of str
        Superpopulation labels aligned with the score arrays.
    k : int, default=1
        DUPI neighborhood order.
    tau : float, default=5.0
        Curvature parameter of the UI/PI atan-sigmoid mapping.

    Returns
    -------
    EvaluationReport
        Aggregated global metrics, per-superpop centroids, and per-class
        DUPI / distribution-distance rows.
    """
    g_dupi = dupi_score(real_pcs, syn_pcs, k)
    g_ui_pi = ui_pi_from_dupi(
        float(g_dupi["dupi"]), float(g_dupi["dupi_benchmark"]), tau=tau
    )
    g_mmd = mmd_rbf(real_pcs, syn_pcs)
    distances = {
        "centroid_distance": centroid_distance(real_pcs, syn_pcs),
        "gaussian_w2_distance": gaussian_w2_distance(real_pcs, syn_pcs),
        **g_mmd,
    }
    return EvaluationReport(
        dupi={**g_dupi, **g_ui_pi},
        distribution_distances=distances,
        centroid_rows=centroid_rows(real_pcs, syn_pcs, real_sp, syn_sp),
        class_metric_rows=class_metric_rows(
            real_pcs, syn_pcs, real_sp, syn_sp, k=k, tau=tau
        ),
    )


def global_metrics(
    real_pcs: np.ndarray,
    syn_pcs: np.ndarray,
    *,
    k: int = 1,
    tau: float = 5.0,
) -> dict[str, Any]:
    """Compute only the global (non-classwise) DUPI and distance metrics."""
    g_dupi = dupi_score(real_pcs, syn_pcs, k)
    g_ui_pi = ui_pi_from_dupi(
        float(g_dupi["dupi"]), float(g_dupi["dupi_benchmark"]), tau=tau
    )
    g_mmd = mmd_rbf(real_pcs, syn_pcs)
    return {
        "dupi": {**g_dupi, **g_ui_pi},
        "distribution_distances": {
            "centroid_distance": centroid_distance(real_pcs, syn_pcs),
            "gaussian_w2_distance": gaussian_w2_distance(real_pcs, syn_pcs),
            **g_mmd,
        },
    }


def centroid_rows(
    real_pcs: np.ndarray,
    syn_pcs: np.ndarray,
    real_sp: np.ndarray,
    syn_sp: np.ndarray,
) -> list[dict[str, Any]]:
    """Per-source × per-superpopulation centroid rows in PC space."""
    rows: list[dict[str, Any]] = []
    for source, pcs, labels in [
        ("real", real_pcs, real_sp),
        ("synthetic", syn_pcs, syn_sp),
    ]:
        for sp in sorted(set(labels)):
            mask = labels == sp
            center = pcs[mask].mean(axis=0)
            rows.append({
                "source": source,
                "superpopulation": sp,
                "n": int(mask.sum()),
                "pc1_centroid": float(center[0]),
                "pc2_centroid": float(center[1]),
            })
    return rows


def class_metric_rows(
    real_pcs: np.ndarray,
    syn_pcs: np.ndarray,
    real_sp: np.ndarray,
    syn_sp: np.ndarray,
    *,
    k: int,
    tau: float,
) -> list[dict[str, Any]]:
    """Per-superpopulation DUPI + distribution-distance + NN-purity rows."""
    rows: list[dict[str, Any]] = []
    all_superpops = sorted(set(real_sp) | set(syn_sp))

    syn_all_nn = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(syn_pcs)
    real_all_nn = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(real_pcs)
    _, real_to_syn_idx = syn_all_nn.kneighbors(real_pcs, return_distance=True)
    _, syn_to_real_idx = real_all_nn.kneighbors(syn_pcs, return_distance=True)

    for sp in all_superpops:
        r_mask = real_sp == sp
        s_mask = syn_sp == sp
        r = real_pcs[r_mask]
        s = syn_pcs[s_mask]
        if len(r) < 2 or len(s) < 1:
            continue

        class_dupi = dupi_score(r, s, min(k, len(r) - 1, len(s)))
        class_ui_pi = ui_pi_from_dupi(
            float(class_dupi["dupi"]),
            float(class_dupi["dupi_benchmark"]),
            tau=tau,
        )

        real_indices = np.where(r_mask)[0]
        syn_indices = np.where(s_mask)[0]
        real_to_syn_same = float(
            np.mean(syn_sp[real_to_syn_idx[real_indices, 0]] == sp)
        )
        syn_to_real_same = float(
            np.mean(real_sp[syn_to_real_idx[syn_indices, 0]] == sp)
        )

        rows.append({
            "superpopulation": sp,
            "n_real": int(len(r)),
            "n_synthetic": int(len(s)),
            "dupi": float(class_dupi["dupi"]),
            "dupi_benchmark": float(class_dupi["dupi_benchmark"]),
            "dupi_abs_error": float(class_dupi["dupi_abs_error"]),
            "utility_index": class_ui_pi["utility_index"],
            "privacy_index": class_ui_pi["privacy_index"],
            "centroid_distance": centroid_distance(r, s),
            "gaussian_w2_distance": gaussian_w2_distance(r, s),
            "mmd_rbf_biased": mmd_rbf(r, s)["mmd_rbf_biased"],
            "coverage_at_real_nn95": same_class_coverage(r, s),
            "real_to_syn_nn_same_superpop": real_to_syn_same,
            "syn_to_real_nn_same_superpop": syn_to_real_same,
        })
    return rows
