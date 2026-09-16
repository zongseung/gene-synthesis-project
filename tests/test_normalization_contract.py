from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def test_normalization_fits_train_only_without_default_clipping():
    from src.preprocessing.tokenizer import apply_normalization, fit_normalization_stats

    train = np.array([[[0.0]], [[2.0]]], dtype=np.float32)
    validation = np.array([[[20.0]]], dtype=np.float32)

    stats = fit_normalization_stats(train)
    transformed = apply_normalization(validation, stats)

    assert stats["fit_split"] == "train"
    assert transformed.item() == pytest.approx(19.0)
    assert stats["clip"] is None


def test_normalization_inverse_round_trip():
    from src.preprocessing.tokenizer import (
        apply_normalization,
        fit_normalization_stats,
        invert_normalization,
    )

    rng = np.random.default_rng(3)
    train = rng.normal(size=(12, 5, 2)).astype(np.float32)

    stats = fit_normalization_stats(train)
    restored = invert_normalization(apply_normalization(train, stats), stats)

    np.testing.assert_allclose(restored, train, rtol=1e-6, atol=1e-6)


def test_normalization_rejects_stats_shape_mismatch():
    from src.preprocessing.tokenizer import apply_normalization, fit_normalization_stats

    stats = fit_normalization_stats(np.zeros((3, 5, 2), dtype=np.float32))

    with pytest.raises(ValueError, match="Normalization shape"):
        apply_normalization(np.zeros((3, 4, 2), dtype=np.float32), stats)


def test_tokenizer_uses_explicit_gene_and_numeric_component_order():
    from src.preprocessing.tokenizer import tokenize_dataset

    features = pd.DataFrame(
        {
            "GENE_B:10": [10.0],
            "GENE_A:1": [1.0],
            "GENE_B:2": [2.0],
            "GENE_A:0": [0.0],
        }
    )

    tokenized, n_genes = tokenize_dataset(
        features, optimal_k=11, gene_order=["GENE_B", "GENE_A"]
    )

    assert n_genes == 2
    assert tokenized[0, 0, 2] == 2.0
    assert tokenized[0, 0, 10] == 10.0
    assert tokenized[0, 1, 0] == 0.0
    assert tokenized[0, 1, 1] == 1.0


def test_gene_size_matches_default_model_downsampling_and_patch_stride():
    from src.preprocessing.tokenizer import compute_gene_size

    assert compute_gene_size(24_482) == 24_576
    assert compute_gene_size(128) == 256
