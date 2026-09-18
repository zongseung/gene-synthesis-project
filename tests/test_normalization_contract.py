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


# ── population-conditional prior (PriorGrad / ShiftDDPMs Data-Normalization) ──

def _two_population_train():
    """Population 0 is centred at +5 with std 2; population 1 at -5 with std 0.5."""
    rng = np.random.default_rng(11)
    pop0 = rng.normal(5.0, 2.0, size=(40, 6, 2))
    pop1 = rng.normal(-5.0, 0.5, size=(40, 6, 2))
    train = np.concatenate([pop0, pop1]).astype(np.float32)
    labels = np.repeat([0, 1], 40)
    return train, labels


@pytest.mark.parametrize("arm", ["mean", "mean_std"])
def test_conditional_prior_centers_each_population_and_inverts(arm):
    from src.preprocessing.tokenizer import (
        apply_normalization,
        fit_normalization_stats,
        invert_normalization,
    )

    train, labels = _two_population_train()
    stats = fit_normalization_stats(train, labels=labels, conditional=arm)
    residual = apply_normalization(train, stats, labels)

    assert stats["conditional"] == arm
    assert stats["mean"].shape == (2, 6, 2)
    assert tuple(stats["shape"]) == (6, 2)
    # Every population's residual is centred, which is the whole point: the
    # model never sees the between-population mean shift.
    for pop in (0, 1):
        np.testing.assert_allclose(residual[labels == pop].mean(axis=0), 0.0, atol=1e-5)
    np.testing.assert_allclose(
        invert_normalization(residual, stats, labels), train, rtol=1e-5, atol=1e-4
    )


def test_mean_arm_keeps_population_diversity_that_mean_std_arm_removes():
    """The one measurable difference between the two arms."""
    from src.preprocessing.tokenizer import apply_normalization, fit_normalization_stats

    train, labels = _two_population_train()
    spread = {}
    for arm in ("mean", "mean_std"):
        stats = fit_normalization_stats(train, labels=labels, conditional=arm)
        residual = apply_normalization(train, stats, labels)
        spread[arm] = [float(residual[labels == pop].std()) for pop in (0, 1)]

    # mean arm: population 0 stays ~4x as diverse as population 1 (2.0 vs 0.5).
    assert spread["mean"][0] / spread["mean"][1] > 3.0
    # mean_std arm: both populations are whitened to unit spread.
    np.testing.assert_allclose(spread["mean_std"], [1.0, 1.0], rtol=0.05)


def test_conditional_stats_require_in_range_population_labels():
    from src.preprocessing.tokenizer import (
        apply_normalization,
        fit_normalization_stats,
        invert_normalization,
    )

    train, labels = _two_population_train()
    stats = fit_normalization_stats(train, labels=labels, conditional="mean")

    with pytest.raises(ValueError, match="population labels"):
        invert_normalization(train, stats)
    with pytest.raises(ValueError, match=r"\[0, 2\)"):
        apply_normalization(train[:2], stats, np.array([0, 7]))
    with pytest.raises(ValueError, match="2 labels for 80 samples"):
        apply_normalization(train, stats, np.array([0, 1]))
    with pytest.raises(ValueError, match="requires population labels"):
        fit_normalization_stats(train, conditional="mean")
    with pytest.raises(ValueError, match="conditional prior arm"):
        fit_normalization_stats(train, labels=labels, conditional="median")


def test_loader_rejects_moments_whose_shape_contradicts_the_recorded_arm(tmp_path):
    import pickle

    from src.preprocessing.tokenizer import load_normalization_stats

    path = tmp_path / "stats.pkl"
    with path.open("wb") as handle:
        pickle.dump(
            {
                "version": 2,
                "mean": np.zeros((3, 5, 2), dtype=np.float32),
                "std": np.ones((3, 5, 2), dtype=np.float32),
                "conditional": "none",
            },
            handle,
        )

    with pytest.raises(ValueError, match="does not match mean shape"):
        load_normalization_stats(path)


def test_variance_floor_keeps_a_within_population_constant_from_exploding():
    """Minority populations are numerically constant in some cells; without a
    floor the mean_std arm divides fp32 rounding dust by itself."""
    from src.preprocessing.tokenizer import apply_normalization, fit_normalization_stats

    train, labels = _two_population_train()
    # Population 0 is exactly constant at this coordinate, population 1 is not.
    train[labels == 0, 2, 1] = 0.7
    stats = fit_normalization_stats(train, labels=labels, conditional="mean_std")
    residual = apply_normalization(train, stats, labels)

    pooled = fit_normalization_stats(train, labels=labels, conditional="mean")["std"]
    assert stats["std"][0, 2, 1] == pytest.approx(0.01 * pooled[2, 1])
    # Unfloored, that cell's residual is fp32 dust divided by itself: order one,
    # indistinguishable from real signal in the loss. Floored, it is negligible.
    unfloored = apply_normalization(
        train,
        fit_normalization_stats(
            train, labels=labels, conditional="mean_std", variance_floor=0.0
        ),
        labels,
    )
    assert np.abs(unfloored[labels == 0, 2, 1]).max() > 0.5
    assert np.abs(residual[labels == 0, 2, 1]).max() < 1e-3
    # The floor is local: a healthy coordinate is still whitened to unit spread.
    assert residual[labels == 0, 0, 0].std() == pytest.approx(1.0, rel=0.05)

    with pytest.raises(ValueError, match="variance_floor"):
        fit_normalization_stats(train, labels=labels, conditional="mean_std", variance_floor=1.0)
