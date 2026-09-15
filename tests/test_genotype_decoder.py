"""Normalization, centering, ordering and leakage contracts for the local-LD genotype decoder."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy.special import expit, softmax
from scipy.stats import binom

from src.models.genotype_decoder import (
    GenotypeDecoder,
    call_logits,
    distance_bins,
    fit_decoder,
    nll_calls,
    sample_calls,
)

OFFSETS = np.array([0, 10, 22, 30])
COHORTS = 4
PANEL_DIR = Path("outputs/diagnostics/hipodit_fisher_20260915_unique")


def _positions(seed=3):
    rng = np.random.default_rng(seed)
    return np.concatenate([np.cumsum(rng.integers(1, 400, size=end - start))
                           for start, end in zip(OFFSETS[:-1], OFFSETS[1:])]).astype(np.float64)


def _panel(seed=7, rows=300):
    rng = np.random.default_rng(seed)
    base_logits = rng.normal(scale=0.8, size=(rows, OFFSETS[-1]))
    labels = rng.integers(0, COHORTS, size=rows)
    calls = rng.binomial(2, expit(base_logits)).astype(float)
    return calls, base_logits, labels


def _handmade(positions, n_bins=4):
    bins = distance_bins(positions, OFFSETS, n_bins)
    rng = np.random.default_rng(11)
    offset = rng.normal(scale=0.4, size=(COHORTS, OFFSETS[-1]))
    weights = np.full(COHORTS, 1 / COHORTS)
    offset -= weights @ offset
    residual = np.repeat((1.5 * np.eye(3))[None], len(bins.edges) + 1, axis=0)
    return GenotypeDecoder("B3", offset, residual, bins, OFFSETS, weights, 1.0, 1.0, 0.0, True)


def test_call_logits_normalize_and_reduce_to_the_binomial_without_a_predecessor():
    # Given a fitted B3 decoder and an unfitted B0 decoder over the same panel.
    calls, base_logits, labels = _panel()
    positions = _positions()
    train = np.arange(200)
    baseline = fit_decoder("B0", calls, base_logits, labels, train, positions, OFFSETS, COHORTS)
    fitted = fit_decoder("B3", calls, base_logits, labels, train, positions, OFFSETS, COHORTS)
    # When the teacher-forced categorical probabilities are formed.
    flat = softmax(call_logits(baseline, calls, base_logits, labels), axis=2)
    structured = softmax(call_logits(fitted, calls, base_logits, labels), axis=2)
    expected = binom.pmf(np.arange(3), 2, expit(base_logits)[:, :, None])
    calibrated = binom.pmf(np.arange(3), 2,
                           expit(base_logits + fitted.cohort_offset[labels])[:, :, None])
    starts = OFFSETS[:-1]
    # Then every call distribution is normalized, B0 is exactly Binomial, and B3 is
    # Binomial at each gene start where no predecessor conditions the draw.
    np.testing.assert_allclose(flat.sum(axis=2), 1.0, rtol=1e-12)
    np.testing.assert_allclose(structured.sum(axis=2), 1.0, rtol=1e-12)
    np.testing.assert_allclose(flat, expected, rtol=1e-10)
    np.testing.assert_allclose(structured[:, starts], calibrated[:, starts], rtol=1e-10)
    assert np.abs(fitted.residual).max() > 0


@pytest.mark.parametrize("arm", ["B1", "B3"])
def test_cohort_offsets_are_centered_on_train_cohort_proportions(arm):
    # Given an unbalanced cohort assignment over the training rows.
    calls, base_logits, _ = _panel()
    labels = np.repeat([0, 1, 2, 3], [150, 90, 45, 15])
    train = np.arange(200)
    # When the cohort offset is fitted.
    decoder = fit_decoder(arm, calls, base_logits, labels, train, _positions(), OFFSETS, COHORTS)
    # Then it is weighted-centered per SNP against the train cohort proportions.
    expected = np.bincount(labels[train], minlength=COHORTS) / len(train)
    np.testing.assert_allclose(decoder.cohort_weights, expected)
    np.testing.assert_allclose(decoder.cohort_weights @ decoder.cohort_offset, 0.0, atol=1e-8)
    assert np.abs(decoder.cohort_offset).max() > 0


def test_the_residual_chain_resets_at_every_gene_boundary():
    # Given a decoder with a non-zero local residual and a row whose calls can be flipped.
    calls, base_logits, labels = _panel()
    decoder = _handmade(_positions())
    row = int(np.flatnonzero((calls[:, 9] != 1) & (calls[:, 4] != 1))[0])
    across = calls.copy()
    across[row, 9] = 2 - calls[row, 9]
    within = calls.copy()
    within[row, 4] = 2 - calls[row, 4]
    reference = nll_calls(decoder, calls, base_logits, labels)
    # When the predecessor changes across a gene boundary versus inside a gene.
    boundary = nll_calls(decoder, across, base_logits, labels)
    interior = nll_calls(decoder, within, base_logits, labels)
    # Then only the within-gene successor reacts.
    assert boundary[row, 10] == reference[row, 10]
    assert interior[row, 5] != reference[row, 5]
    assert decoder.bins.bin_of_snp[10] == -1


def test_each_snp_is_conditioned_through_its_own_distance_bin():
    # Given a decoder whose residual table differs per distance bin.
    calls, base_logits, labels = _panel()
    bins = distance_bins(_positions(), OFFSETS, 4)
    residual = np.stack([(index + 1.0) * np.eye(3) for index in range(len(bins.edges) + 1)])
    decoder = GenotypeDecoder("B2", np.zeros((COHORTS, OFFSETS[-1])), residual, bins, OFFSETS,
                              np.full(COHORTS, 1 / COHORTS), 1.0, 1.0, 0.0, True)
    successors = np.flatnonzero(bins.bin_of_snp >= 0)
    predecessor = calls[:, successors - 1].astype(np.int64)[:, :, None]
    # When the teacher-forced logits are compared with the plain Binomial ones.
    binomial = base_logits[:, :, None] * np.arange(3) + np.log([1.0, 2.0, 1.0])
    shift = call_logits(decoder, calls, base_logits, labels)[:, successors] - binomial[:, successors]
    # Then each successor picks up its own bin's table entry for the observed predecessor.
    expected = np.broadcast_to(bins.bin_of_snp[successors] + 1.0, shift.shape[:2])
    assert len(np.unique(bins.bin_of_snp[successors])) == 4
    np.testing.assert_allclose(np.take_along_axis(shift, predecessor, axis=2)[:, :, 0], expected)
    np.testing.assert_allclose(shift.sum(axis=2), expected)


def test_the_fit_is_stationary_for_the_ridge_penalized_objective_it_reports():
    # Given a B1 decoder fitted with a ridge on the weight-centered cohort offset.
    calls, base_logits, _ = _panel()
    labels = np.repeat([0, 1, 2, 3], [150, 90, 45, 15])
    train = np.arange(200)
    decoder = fit_decoder("B1", calls, base_logits, labels, train, _positions(), OFFSETS, COHORTS,
                          lambda_u=2.0)

    def objective(offset):
        moved = replace(decoder, cohort_offset=offset)
        teacher_forced = nll_calls(moved, calls[train], base_logits[train], labels[train])
        return np.nansum(teacher_forced) + 2.0 * (offset**2).sum()

    rng = np.random.default_rng(5)
    # When the offset is nudged along weight-centered unit directions.
    derivatives = []
    for _ in range(4):
        direction = rng.normal(size=decoder.cohort_offset.shape)
        direction -= decoder.cohort_weights @ direction
        direction /= np.linalg.norm(direction)
        derivatives.append((objective(decoder.cohort_offset + 1e-2 * direction)
                            - objective(decoder.cohort_offset - 1e-2 * direction)) / 2e-2)
    # Then the reported offset is stationary, so the ridge really penalized the centered offset.
    assert np.max(np.abs(derivatives)) < 1e-3


def test_distance_bins_reject_positions_that_are_not_ascending_inside_a_gene():
    # Given positions that step backwards inside the second gene.
    positions = _positions()
    positions[15] = positions[14]
    # When binning, then the ordering contract rejects the panel.
    with pytest.raises(ValueError, match="ascending"):
        distance_bins(positions, OFFSETS, 4)


def test_fitting_ignores_every_row_outside_the_training_indices():
    # Given a panel fitted on the training rows only.
    calls, base_logits, labels = _panel()
    positions = _positions()
    train = np.arange(200)
    first = fit_decoder("B3", calls, base_logits, labels, train, positions, OFFSETS, COHORTS)
    corrupted_calls, corrupted_logits = calls.copy(), base_logits.copy()
    corrupted_calls[200:] = 2 - corrupted_calls[200:]
    corrupted_logits[200:] = -3.0 * corrupted_logits[200:]
    corrupted_labels = labels.copy()
    corrupted_labels[200:] = (corrupted_labels[200:] + 1) % COHORTS
    # When every held-out row is altered and the fit is repeated.
    second = fit_decoder("B3", corrupted_calls, corrupted_logits, corrupted_labels, train,
                         positions, OFFSETS, COHORTS)
    # Then the learned decoder and its bins are bit-identical.
    np.testing.assert_array_equal(first.cohort_offset, second.cohort_offset)
    np.testing.assert_array_equal(first.residual, second.residual)
    np.testing.assert_array_equal(first.bins.edges, second.bins.edges)
    np.testing.assert_array_equal(first.cohort_weights, second.cohort_weights)


def test_empty_quantile_bins_reduce_the_bin_count():
    # Given two genes whose adjacent distances take only two distinct values.
    positions = np.concatenate([np.arange(5), 100 + 5 * np.arange(5)]).astype(np.float64)
    offsets = np.array([0, 5, 10])
    # When a four-way quantile split is requested.
    bins = distance_bins(positions, offsets, 4)
    # Then the bin count shrinks until every bin holds adjacent pairs, and gene starts are unbinned.
    assert len(bins.edges) == 1
    assert np.array_equal(np.flatnonzero(bins.bin_of_snp < 0), [0, 5])
    np.testing.assert_array_equal(np.bincount(bins.bin_of_snp[bins.bin_of_snp >= 0]), [4, 4])


def test_genes_without_adjacent_pairs_collapse_to_a_single_bin():
    # Given a panel whose every gene holds a single SNP, so no adjacent distance exists.
    bins = distance_bins(np.array([5.0, 9.0, 2.0]), np.array([0, 1, 2, 3]), 4)
    # When it is binned, then one empty bin remains and no SNP has a predecessor.
    assert len(bins.edges) == 0
    np.testing.assert_array_equal(bins.bin_of_snp, [-1, -1, -1])


def test_the_residual_decoder_recovers_simulated_local_dependence():
    # Given calls simulated from a hand-built B3 decoder.
    positions = _positions()
    truth = _handmade(positions)
    rng = np.random.default_rng(23)
    base_logits = rng.normal(scale=0.8, size=(600, OFFSETS[-1]))
    labels = rng.integers(0, COHORTS, size=600)
    calls = sample_calls(truth, base_logits, labels, rng)
    train, dev = np.arange(400), np.arange(400, 600)
    # When B0 and B3 are fitted on the training rows.
    flat = fit_decoder("B0", calls.astype(float), base_logits, labels, train, positions, OFFSETS, COHORTS)
    structured = fit_decoder("B3", calls.astype(float), base_logits, labels, train, positions, OFFSETS, COHORTS)
    dev_flat = nll_calls(flat, calls.astype(float), base_logits, labels)[dev].mean()
    dev_structured = nll_calls(structured, calls.astype(float), base_logits, labels)[dev].mean()
    diagonal = np.einsum("bgg->b", structured.residual) / 3
    off = (structured.residual.sum(axis=(1, 2)) - np.einsum("bgg->b", structured.residual)) / 6
    # Then the sampler respects the call contract and B3 beats B0 with a positive diagonal.
    assert calls.dtype == np.int8 and calls.shape == (600, OFFSETS[-1])
    assert set(np.unique(calls)).issubset({0, 1, 2})
    assert dev_structured < dev_flat
    assert np.all(diagonal > off)
    assert structured.converged


def test_save_and_load_round_trip_every_field(tmp_path):
    # Given a fitted decoder written to disk.
    calls, base_logits, labels = _panel()
    decoder = fit_decoder("B3", calls, base_logits, labels, np.arange(200), _positions(),
                          OFFSETS, COHORTS, lambda_u=2.0, lambda_a=0.5)
    path = tmp_path / "decoder.npz"
    decoder.save(path)
    # When it is read back.
    restored = GenotypeDecoder.load(path)
    # Then every field survives the round trip.
    assert restored.arm == decoder.arm
    assert (restored.lambda_u, restored.lambda_a) == (decoder.lambda_u, decoder.lambda_a)
    assert restored.train_nll_per_call == decoder.train_nll_per_call
    assert restored.converged == decoder.converged
    np.testing.assert_array_equal(restored.cohort_offset, decoder.cohort_offset)
    np.testing.assert_array_equal(restored.residual, decoder.residual)
    np.testing.assert_array_equal(restored.cohort_weights, decoder.cohort_weights)
    np.testing.assert_array_equal(restored.offsets, decoder.offsets)
    np.testing.assert_array_equal(restored.bins.edges, decoder.bins.edges)
    np.testing.assert_array_equal(restored.bins.bin_of_snp, decoder.bins.bin_of_snp)


def test_b0_per_call_nll_matches_the_binomial_and_marks_missing_calls():
    # Given a B0 decoder and a panel with one missing call.
    calls, base_logits, labels = _panel()
    calls[0, 0] = np.nan
    calls[5, 17] = np.nan
    train = np.arange(200)
    decoder = fit_decoder("B0", calls, base_logits, labels, train, _positions(), OFFSETS, COHORTS)
    # When per-call negative log-likelihoods are evaluated.
    observed = nll_calls(decoder, calls, base_logits, labels)
    expected = -np.log(binom.pmf(np.nan_to_num(calls), 2, expit(base_logits)))
    missing = np.isnan(calls)
    # Then they equal the independent Binomial term and are NaN exactly where calls are missing.
    np.testing.assert_array_equal(np.isnan(observed), missing)
    np.testing.assert_allclose(observed[~missing], expected[~missing], rtol=1e-10)


def test_missing_predecessors_reset_the_chain_like_a_gene_boundary():
    # Given a decoder with a non-zero residual and a missing predecessor call.
    calls, base_logits, labels = _panel()
    decoder = _handmade(_positions())
    reference = nll_calls(decoder, calls, base_logits, labels)
    dropped = calls.copy()
    dropped[0, 4] = np.nan
    # When the successor's likelihood is recomputed.
    observed = nll_calls(decoder, dropped, base_logits, labels)
    expected = -np.log(binom.pmf(calls[0, 5], 2, expit(base_logits[0, 5] + decoder.cohort_offset[labels[0], 5])))
    # Then it falls back to the plain Binomial rather than reusing a stale predecessor.
    assert observed[0, 5] != reference[0, 5]
    np.testing.assert_allclose(observed[0, 5], expected, rtol=1e-10)


@pytest.mark.parametrize("bad", [
    {"arm": "B4"},
    {"calls": np.zeros((4, 4, 4))},
    {"base_logits": np.zeros((299, 30))},
    {"labels": np.full(300, 4)},
    {"train_indices": np.array([0, 0, 1])},
    {"train_indices": np.array([0, 300])},
])
def test_fit_decoder_rejects_malformed_inputs(bad):
    # Given a valid panel with one argument replaced by a malformed one.
    calls, base_logits, labels = _panel()
    arguments = dict(arm="B3", calls=calls, base_logits=base_logits, labels=labels,
                     train_indices=np.arange(200), positions=_positions(), offsets=OFFSETS,
                     n_cohorts=COHORTS)
    arguments.update(bad)
    # When fitting, then the input contract rejects it.
    with pytest.raises(ValueError):
        fit_decoder(**arguments)


@pytest.mark.parametrize("bad_call", [-1.0, 0.5, 3.0, np.inf])
def test_fit_decoder_rejects_calls_that_are_not_diploid_dosages(bad_call):
    # Given an observed call outside {0, 1, 2}; NaN alone denotes missingness.
    calls, base_logits, labels = _panel()
    calls[0, 0] = bad_call
    # When fitting, then the dosage contract rejects it.
    with pytest.raises(ValueError, match="0, 1, 2"):
        fit_decoder("B3", calls, base_logits, labels, np.arange(200), _positions(), OFFSETS, COHORTS)


@pytest.mark.skipif(not PANEL_DIR.exists(), reason="prepared chr17 panel is unavailable")
def test_real_panel_distance_bins_match_the_frozen_quartiles():
    # Given the prepared chr17 panel.
    with (PANEL_DIR / "gene_variant_map.json").open() as handle:
        genes = json.load(handle)["genes"]
    positions = np.array([variant["position"] for gene in genes for variant in gene["variants"]])
    with np.load(PANEL_DIR / "genotypes.npz") as data:
        offsets = data["offsets"]
    # When the panel is binned into quartiles of within-gene adjacent distance.
    bins = distance_bins(positions, offsets, 4)
    # Then the frozen edges and per-bin adjacent-pair counts are reproduced.
    np.testing.assert_array_equal(bins.edges, [35, 104, 210])
    np.testing.assert_array_equal(np.bincount(bins.bin_of_snp[bins.bin_of_snp >= 0]), [86, 89, 89, 89])
    np.testing.assert_array_equal(np.flatnonzero(bins.bin_of_snp < 0), offsets[:-1])
