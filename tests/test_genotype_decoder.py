"""Normalization, centering, ordering, leakage and dosage contracts for the genotype decoder."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy.special import expit, logit, softmax
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
SUPERPOPS = np.array([0, 0, 1, 1])
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


def _handmade_tilt(tilt):
    """The handmade B3 decoder re-armed as T; its residual table stays to prove arm T never reads it."""
    return replace(_handmade(_positions()), arm="T", tilt=tilt, tilt_scope="snp", lambda_tilt=1.0)


def _tilted_grid(allele, tilt):
    """Arm-T call probabilities with the allele probability varying over rows and the tilt over SNPs."""
    snps = len(tilt)
    decoder = GenotypeDecoder("T", np.zeros((1, snps)), np.zeros((1, 3, 3)),
                              distance_bins(np.arange(1.0, snps + 1), np.array([0, snps]), 1),
                              np.array([0, snps]), np.ones(1), 1.0, 0.0, 0.0, True,
                              tilt=tilt, tilt_scope="snp", lambda_tilt=1.0)
    base_logits = np.repeat(logit(allele)[:, None], snps, axis=1)
    labels = np.zeros(len(allele), dtype=np.int64)
    logits = call_logits(decoder, np.zeros(base_logits.shape), base_logits, labels)
    return softmax(logits, axis=2), expit(base_logits)


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


@pytest.mark.parametrize("arm, scope", [("B1", "none"), ("B3", "none"), ("T", "snp")])
def test_cohort_offsets_are_centered_on_train_cohort_proportions(arm, scope):
    # Given an unbalanced cohort assignment over the training rows.
    calls, base_logits, _ = _panel()
    labels = np.repeat([0, 1, 2, 3], [150, 90, 45, 15])
    train = np.arange(200)
    # When the cohort offset is fitted.
    decoder = fit_decoder(arm, calls, base_logits, labels, train, _positions(), OFFSETS, COHORTS,
                          tilt_scope=scope)
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


@pytest.mark.parametrize("arm, scope", [("B3", "none"), ("T", "snp"), ("T", "superpop")])
def test_fitting_ignores_every_row_outside_the_training_indices(arm, scope):
    # Given a panel fitted on the training rows only.
    calls, base_logits, labels = _panel()
    positions = _positions()
    train = np.arange(200)
    tilt = dict(tilt_scope=scope, superpop_of_cohort=SUPERPOPS)
    first = fit_decoder(arm, calls, base_logits, labels, train, positions, OFFSETS, COHORTS, **tilt)
    corrupted_calls, corrupted_logits = calls.copy(), base_logits.copy()
    corrupted_calls[200:] = 2 - corrupted_calls[200:]
    corrupted_logits[200:] = -3.0 * corrupted_logits[200:]
    corrupted_labels = labels.copy()
    corrupted_labels[200:] = (corrupted_labels[200:] + 1) % COHORTS
    # When every held-out row is altered and the fit is repeated.
    second = fit_decoder(arm, corrupted_calls, corrupted_logits, corrupted_labels, train,
                         positions, OFFSETS, COHORTS, **tilt)
    # Then the learned decoder and its bins are bit-identical.
    np.testing.assert_array_equal(first.cohort_offset, second.cohort_offset)
    np.testing.assert_array_equal(first.residual, second.residual)
    np.testing.assert_array_equal(first.tilt, second.tilt)
    assert first.train_nll_per_call == second.train_nll_per_call
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


@pytest.mark.parametrize("arm, scope", [("B3", "none"), ("T", "superpop")])
def test_save_and_load_round_trip_every_field(tmp_path, arm, scope):
    # Given a fitted decoder written to disk.
    calls, base_logits, labels = _panel()
    decoder = fit_decoder(arm, calls, base_logits, labels, np.arange(200), _positions(),
                          OFFSETS, COHORTS, lambda_u=2.0, lambda_a=0.5, tilt_scope=scope,
                          lambda_tilt=0.25, superpop_of_cohort=SUPERPOPS)
    path = tmp_path / "decoder.npz"
    decoder.save(path)
    # When it is read back.
    restored = GenotypeDecoder.load(path)
    # Then every field survives the round trip.
    assert restored.arm == decoder.arm
    assert (restored.lambda_u, restored.lambda_a) == (decoder.lambda_u, decoder.lambda_a)
    assert restored.train_nll_per_call == decoder.train_nll_per_call
    assert restored.converged == decoder.converged
    assert (restored.tilt_scope, restored.lambda_tilt) == (decoder.tilt_scope, decoder.lambda_tilt)
    np.testing.assert_array_equal(restored.tilt, decoder.tilt)
    np.testing.assert_array_equal(restored.superpop_of_cohort, decoder.superpop_of_cohort)
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
    {"tilt_scope": "gene"},
    {"arm": "T"},
    {"tilt_scope": "snp"},
    {"arm": "T", "tilt_scope": "superpop"},
    {"arm": "T", "tilt_scope": "superpop", "superpop_of_cohort": np.array([0, 0, 1])},
    {"arm": "T", "tilt_scope": "superpop", "superpop_of_cohort": np.array([0, -1, 1, 1])},
    {"arm": "T", "tilt_scope": "superpop", "superpop_of_cohort": np.array([0, 0, 2, 2])},
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


def test_the_tilt_keeps_expected_dosage_exact_and_is_binomial_without_a_tilt():
    # Given allele probabilities from rare to near-fixed and tilts from homozygous to heterozygous.
    allele = np.array([1e-4, 0.03, 0.3, 0.5, 0.7, 0.97, 1 - 1e-4])
    tilt = np.array([-3.0, -0.5, 0.0, 1.2, 3.0])
    # When arm T forms its call distributions.
    probabilities, p = _tilted_grid(allele, tilt)
    # Then every distribution is valid, its expected dosage is exactly 2p, and a zero tilt is Binomial.
    assert np.all(probabilities > 0)
    np.testing.assert_allclose(probabilities.sum(axis=2), 1.0, rtol=1e-12)
    np.testing.assert_allclose(probabilities @ np.arange(3), 2 * p, rtol=0, atol=1e-10)
    binomial = binom.pmf(np.arange(3), 2, p[:, tilt == 0, None])
    np.testing.assert_allclose(probabilities[:, tilt == 0], binomial, rtol=1e-10)


def test_raising_the_tilt_moves_heterozygosity_but_not_expected_dosage():
    # Given the same allele probabilities under a finely increasing tilt.
    allele = np.array([1e-4, 0.03, 0.3, 0.5, 0.7, 0.97, 1 - 1e-4])
    # When arm T forms its call distributions.
    probabilities, p = _tilted_grid(allele, np.linspace(-4.0, 4.0, 41))
    # Then the heterozygote probability rises strictly while expected dosage never moves.
    assert np.all(np.diff(probabilities[:, :, 1], axis=1) > 0)
    np.testing.assert_allclose(probabilities @ np.arange(3), 2 * p, rtol=0, atol=1e-10)


def test_arm_t_draws_and_scores_every_snp_without_its_predecessor():
    # Given an arm-T decoder that still carries the strong local residual table of B3.
    calls, base_logits, labels = _panel()
    decoder = _handmade_tilt(np.random.default_rng(13).normal(size=OFFSETS[-1]))
    cleared = replace(decoder, residual=np.zeros_like(decoder.residual))
    # When calls are drawn from one seed with and without that table, and the teacher-forced
    # logits are formed with every predecessor forced to 0, 1 and 2.
    kept = sample_calls(decoder, base_logits, labels, np.random.default_rng(0))
    dropped = sample_calls(cleared, base_logits, labels, np.random.default_rng(0))
    tilted = [call_logits(decoder, np.full(calls.shape, h), base_logits, labels) for h in range(3)]
    chained = [call_logits(replace(decoder, arm="B3"), np.full(calls.shape, h), base_logits, labels)
               for h in range(3)]
    # Then neither the draws nor the logits react to the predecessor, while the B3 chain does.
    np.testing.assert_array_equal(kept, dropped)
    np.testing.assert_array_equal(tilted[1], tilted[0])
    np.testing.assert_array_equal(tilted[2], tilted[0])
    assert not np.allclose(chained[1], chained[0])


def test_the_tilt_decoder_recovers_a_simulated_heterozygosity_tilt():
    # Given calls simulated from a hand-built arm-T decoder whose per-SNP tilt is off-centre.
    positions = _positions()
    rng = np.random.default_rng(29)
    truth = rng.normal(loc=0.6, scale=0.8, size=OFFSETS[-1])
    base_logits = rng.normal(scale=0.8, size=(1500, OFFSETS[-1]))
    labels = rng.integers(0, COHORTS, size=1500)
    calls = sample_calls(_handmade_tilt(truth), base_logits, labels, rng).astype(float)
    train, dev = np.arange(1000), np.arange(1000, 1500)
    # When arm T and the offset-only B1 are fitted on the training rows.
    tilted = fit_decoder("T", calls, base_logits, labels, train, positions, OFFSETS, COHORTS,
                         tilt_scope="snp")
    offset_only = fit_decoder("B1", calls, base_logits, labels, train, positions, OFFSETS, COHORTS)
    dev_tilted = nll_calls(tilted, calls, base_logits, labels)[dev].mean()
    dev_offset_only = nll_calls(offset_only, calls, base_logits, labels)[dev].mean()
    clear = np.abs(truth) > 0.5
    # Then the tilt comes back correlated, signed and uncentred like the truth, and T beats B1 on
    # dev calls.
    assert tilted.tilt.shape == (OFFSETS[-1],)
    assert np.corrcoef(tilted.tilt, truth)[0, 1] >= 0.8
    np.testing.assert_array_equal(np.sign(tilted.tilt[clear]), np.sign(truth[clear]))
    assert abs(tilted.tilt.mean() - truth.mean()) < 0.1
    assert dev_tilted < dev_offset_only
    assert tilted.converged


def test_arm_t_draws_keep_the_calibrated_allele_frequency_where_the_b3_chain_drifts():
    # Given a panel with allele frequencies spread away from 1/2 (where a symmetric chain cannot
    # drift), the B3 chain and an arm-T decoder sharing its cohort offset. A non-negative tilt
    # under-disperses dosage, so the Binomial draw-noise bound below is conservative for T.
    _, base_logits, labels = _panel()
    base_logits = base_logits + np.linspace(-2.5, 2.5, OFFSETS[-1])
    chained = _handmade(_positions())
    tilted = _handmade_tilt(np.abs(np.random.default_rng(31).normal(scale=1.5, size=OFFSETS[-1])))
    allele = expit(base_logits + chained.cohort_offset[labels]).mean(axis=0)
    draws = 20
    bound = 3 * np.sqrt(allele * (1 - allele) / (2 * draws * len(base_logits))) + 1e-12

    def frequency(decoder):
        rng = np.random.default_rng(37)
        return np.mean([sample_calls(decoder, base_logits, labels, rng) for _ in range(draws)],
                       axis=(0, 1)) / 2

    # When 20 synthetic panels are drawn from each decoder, then T stays inside draw noise at
    # every SNP while the B3 chain leaves it.
    assert np.all(np.abs(frequency(tilted) - allele) <= bound)
    assert np.any(np.abs(frequency(chained) - allele) > bound)


@pytest.mark.parametrize("scope, groups", [("cohort", COHORTS), ("superpop", 2)])
def test_group_tilts_hold_one_row_per_group_read_only_by_that_groups_samples(scope, groups):
    # Given a T decoder fitted with one tilt row per cohort or per superpopulation.
    calls, base_logits, labels = _panel()
    train = np.arange(200)
    arguments = (base_logits, labels, train, _positions(), OFFSETS, COHORTS)
    options = dict(tilt_scope=scope, superpop_of_cohort=SUPERPOPS)
    decoder = fit_decoder("T", calls, *arguments, **options)
    group = SUPERPOPS[labels] if scope == "superpop" else labels
    touched = SUPERPOPS[2] if scope == "superpop" else 2
    inside = group == touched
    # When the tilt row cohort 2 reads is moved, and separately only cohort 2's training calls flip.
    moved = decoder.tilt.copy()
    moved[touched] += 1.0
    before = call_logits(decoder, calls, base_logits, labels)
    after = call_logits(replace(decoder, tilt=moved), calls, base_logits, labels)
    flipped = calls.copy()
    rows = train[labels[train] == 2]
    flipped[rows] = 2 - flipped[rows]
    change = np.abs(fit_decoder("T", flipped, *arguments, **options).tilt - decoder.tilt).mean(axis=1)
    others = np.arange(groups) != touched
    # Then the table has one row per group, only that group's samples read the row, and the flip
    # moves that row far more than any other. The others still move a little: the weight-centered
    # cohort offset couples every cohort, so they cannot stay bit-identical.
    assert decoder.tilt.shape == (groups, OFFSETS[-1])
    np.testing.assert_array_equal(after[~inside], before[~inside])
    assert np.all(after[inside][:, :, 1] != before[inside][:, :, 1])
    assert np.all(change[touched] > 10 * change[others])


def test_decoders_saved_before_the_tilt_arm_still_load_and_decode_unchanged(tmp_path):
    # Given a B3 decoder written in the four-arm file layout, which has no tilt fields.
    calls, base_logits, labels = _panel()
    decoder = _handmade(_positions())
    path = tmp_path / "legacy.npz"
    np.savez(path, cohort_offset=decoder.cohort_offset, residual=decoder.residual,
             edges=decoder.bins.edges, bin_of_snp=decoder.bins.bin_of_snp, offsets=decoder.offsets,
             cohort_weights=decoder.cohort_weights,
             scalars=json.dumps({"arm": "B3", "lambda_u": 1.0, "lambda_a": 1.0,
                                 "train_nll_per_call": 0.0, "converged": True}))
    # When it is loaded.
    restored = GenotypeDecoder.load(path)
    # Then the tilt fields take their empty defaults and the B3 chain decodes as before.
    assert (restored.tilt_scope, restored.lambda_tilt) == ("none", 0.0)
    assert restored.tilt.shape == restored.superpop_of_cohort.shape == (0,)
    np.testing.assert_array_equal(nll_calls(restored, calls, base_logits, labels),
                                  nll_calls(decoder, calls, base_logits, labels))
    np.testing.assert_array_equal(
        sample_calls(restored, base_logits, labels, np.random.default_rng(0)),
        sample_calls(decoder, base_logits, labels, np.random.default_rng(0)))
