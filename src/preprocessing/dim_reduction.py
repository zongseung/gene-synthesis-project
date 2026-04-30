"""Dispatch layer for per-gene dimensionality reduction.

Currently supports two backends:

* ``"pca"`` — Gaussian PCA (sklearn). Fast (~10 ms/gene) but misspecified
  for genotype dosage data: assumes Gaussian + homoscedastic noise, which
  ignores the Binomial(2, p) mean–variance relationship.
* ``"glm_pca"`` — GLM-PCA with multinomial/Binomial likelihood (Townes
  et al. 2019). Statistically correct for dosage but ~100× slower per fit.

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
import pandas as pd

logger = logging.getLogger(__name__)

DimRedMethod = Literal["pca", "glm_pca", "glm_pca_torch"]


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
            **{k: v for k, v in kwargs.items() if k in ("fam", "max_iter")},
        )
    if method == "glm_pca_torch":
        from src.preprocessing.glm_pca_torch import glm_pca_torch_single_gene
        return glm_pca_torch_single_gene(
            gene_name=gene_name, matrix=matrix,
            n_components=n_components, train_indices=train_indices,
            **{k: v for k, v in kwargs.items() if k in ("fam", "max_iter")},
        )
    raise ValueError(
        f"Unknown DIM_RED_METHOD: {method!r}. "
        "Expected 'pca' | 'glm_pca' | 'glm_pca_torch'."
    )


def grid_search_optimal_k(
    method: DimRedMethod,
    gene_matrices: dict[str, np.ndarray],
    train_indices: np.ndarray | None = None,
    **kwargs,
) -> tuple[int, pd.DataFrame]:
    """Dispatch grid-search to the configured backend."""
    if method == "pca":
        from src.preprocessing.pca import grid_search_optimal_pca
        return grid_search_optimal_pca(
            gene_matrices=gene_matrices, train_indices=train_indices, **kwargs,
        )
    if method == "glm_pca":
        from src.preprocessing.glm_pca import grid_search_optimal_glm_pca
        return grid_search_optimal_glm_pca(
            gene_matrices=gene_matrices, train_indices=train_indices, **kwargs,
        )
    if method == "glm_pca_torch":
        from src.preprocessing.glm_pca_torch import grid_search_optimal_torch
        return grid_search_optimal_torch(
            gene_matrices=gene_matrices, train_indices=train_indices, **kwargs,
        )
    raise ValueError(
        f"Unknown DIM_RED_METHOD: {method!r}. "
        "Expected 'pca' | 'glm_pca' | 'glm_pca_torch'."
    )
