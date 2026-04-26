"""Real-vs-synthetic evaluation primitives.

Public API
----------
DUPI (Jeong, Kim, and Im 2023, IEEE TIFS):
    :func:`dupi_score`, :func:`kth_dupi_benchmark`, :func:`ui_pi_from_dupi`

Distribution-distance metrics:
    :func:`centroid_distance`, :func:`gaussian_w2_distance`,
    :func:`mmd_rbf`, :func:`same_class_coverage`

High-level pipeline:
    :class:`EvaluationReport`, :func:`evaluate`,
    :func:`global_metrics`, :func:`centroid_rows`, :func:`class_metric_rows`
"""

from src.evaluation.distribution_metrics import (
    centroid_distance,
    gaussian_w2_distance,
    mmd_rbf,
    same_class_coverage,
)
from src.evaluation.dupi import (
    dupi_score,
    kth_dupi_benchmark,
    ui_pi_from_dupi,
)
from src.evaluation.synthetic_pipeline import (
    EvaluationReport,
    centroid_rows,
    class_metric_rows,
    evaluate,
    global_metrics,
)

__all__ = [
    # DUPI
    "dupi_score",
    "kth_dupi_benchmark",
    "ui_pi_from_dupi",
    # distribution metrics
    "centroid_distance",
    "gaussian_w2_distance",
    "mmd_rbf",
    "same_class_coverage",
    # pipeline
    "EvaluationReport",
    "evaluate",
    "global_metrics",
    "centroid_rows",
    "class_metric_rows",
]
