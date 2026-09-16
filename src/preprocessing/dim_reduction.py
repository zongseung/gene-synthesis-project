"""Dispatch layer for per-gene dimensionality reduction.

Currently supports two backends:

* ``"pca"`` — Gaussian PCA (sklearn). Fast (~10 ms/gene) but misspecified
  for genotype dosage data: assumes Gaussian + homoscedastic noise, which
  ignores the Binomial(2, p) mean–variance relationship.
* ``"glm_pca"`` — Poisson GLM-PCA (Townes et al. 2019), used as an explicit
  count-model approximation for bounded dosage.

Selection is done via :data:`src.preprocessing.config.DIM_RED_METHOD`. The
dispatch returns dictionaries with the *same* schema regardless of backend
so that ``run_pipeline.py`` and downstream code do not branch.

Sweep / ablation usage::

    from src.preprocessing import dim_reduction
    res = dim_reduction.reduce_single_gene(
        method="glm_pca",
        gene_name="BRCA1",
        matrix=matrix,
        n_components=8,
    )
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)

DimRedMethod = Literal["pca", "glm_pca"]


def reduce_single_gene(
    method: DimRedMethod,
    gene_name: str,
    matrix: np.ndarray,
    n_components: int,
    train_indices: np.ndarray | None = None,
    **kwargs,
) -> dict | None:
    """Dispatch to the configured per-gene dimensionality-reduction backend."""
    if method == "pca":
        from src.preprocessing.pca import pca_single_gene
        return pca_single_gene(
            gene_name=gene_name, matrix=matrix,
            n_components=n_components, train_indices=train_indices,
        )
    if method == "glm_pca":
        from src.preprocessing.glm_pca import glm_pca_single_gene
        return glm_pca_single_gene(
            gene_name=gene_name, matrix=matrix,
            n_components=n_components, train_indices=train_indices,
            **{k: v for k, v in kwargs.items() if k in ("max_iter",)},
        )
    raise ValueError(
        f"Unknown DIM_RED_METHOD: {method!r}. "
        "Expected 'pca' | 'glm_pca'."
    )
