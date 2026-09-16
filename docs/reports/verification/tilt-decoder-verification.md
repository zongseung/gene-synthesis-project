# Task 7 report: HiPoDiT-T decoder arm (AF-preserving heterozygosity tilt)

**Status:** DONE_WITH_CONCERNS (one concern: test 6 can't be written the way the brief states it, see Concerns)
**Commit:** `dbb1576 feat: add AF-preserving heterozygosity-tilt decoder arm (T)`. Only the two named files are in it, with no Co-Authored-By trailer.

## What was implemented (`src/models/genotype_decoder.py`, +82 net lines, now 314)

- `_ARMS` gains `"T"`, and there's a new `_TILT_SCOPES = ("none", "snp", "superpop", "cohort")`.
- `GenotypeDecoder` has four new fields with defaults, so the existing 10-positional-arg constructors still work: `tilt=zeros(0)`, `tilt_scope="none"`, `lambda_tilt=0.0`, `superpop_of_cohort=zeros(0, int64)`.
- `save` writes `tilt` and `superpop_of_cohort` as arrays and `tilt_scope` and `lambda_tilt` in the scalars JSON. `load` fills the dataclass defaults when `"tilt"` is missing from the file, so old four-arm files still load.
- `_tilted_logits(eta, tilt)` is a single torch implementation of the brief's formula, used by both the numpy path and the LBFGS fit:
  - `log q = log σ(−|η|)`
  - `log f = log2 + log q − log(t(1−2q) + sqrt(t²(1−2q)² + 4q(1−q)))`
  - `log x = ±log f`
  - logits are `[0, log2 + τ + log x, 2 log x]`.
  - `−|η|` is written `where(η>0, −η, η)` so the gradient at exactly η=0 matches the η≤0 branch. `abs()` would give a zero gradient there.
- `_tilt_group` gives the row of the `(groups, J)` tilt table each sample reads: all zeros for `snp`, `superpop_of_cohort[labels]` for superpop, `labels` for cohort. `_unchained_logits` gives the per-call logits before any chain: the unchanged Binomial expression for B0–B3, the tilted one for T.
- `call_logits` returns early for T without computing predecessors. `sample_calls` keeps the sequential loop but never adds the residual for T. Neither function's signature changed.
- `fit_decoder` has three new keyword args: `tilt_scope="none"`, `lambda_tilt=1.0`, `superpop_of_cohort=None`.
  - T trains U (weight-centered as before) and τ, which is left uncentered. The residual stays at zero.
  - The objective gains `lambda_tilt·‖τ‖²`. τ is empty for B0–B3, so for them this adds exactly 0.0.
  - Same LBFGS setup, same `converged` definition, starting from zero.
  - `lambda_tilt` is stored as 0.0 for B0–B3. `superpop_of_cohort` is stored only for the superpop scope.
- Validation raises `ValueError` for:
  - an arm outside the 5 values
  - a scope outside the 4 values
  - arm T with scope `"none"`
  - superpop scope with `superpop_of_cohort` missing, not 1-D, length ≠ C, negative, or non-contiguous (checked as `unique == arange(len(unique))`)
- **One rule the brief doesn't state:** B0–B3 with a tilt scope other than `"none"` also raises. It uses the same one-line check (`(arm == "T") != (tilt_scope != "none")`) and enforces the brief's "B0–B3 have tilt shape (0,)".
- Two deliberate limits are marked with `# ponytail:` comments: t² overflows past τ≈350, which a ridge-fitted τ never reaches; the dense (N, J, 3) logits comment was already there.

## Tests (`tests/test_genotype_decoder.py`)

All 25 existing cases are kept. The file now has 44 cases (+19):

1. `test_the_tilt_keeps_expected_dosage_exact_and_is_binomial_without_a_tilt`: the brief's p × τ grid. All probabilities > 0, rows sum to 1, `E[G]=2p` to atol 1e-10, and τ=0 matches the Binomial pmf (rtol 1e-10).
2. `test_raising_the_tilt_moves_heterozygosity_but_not_expected_dosage`: τ over linspace(−4, 4, 41). `q(1)` rises strictly and `E[G]=2p` to atol 1e-10.
3. `test_arm_t_draws_and_scores_every_snp_without_its_predecessor`: the T decoder carries B3's strong residual table (1.5·I).
   - Samples from the same seed are identical with and without that table.
   - `call_logits` with every predecessor forced to 0/1/2 is `assert_array_equal` across the three.
   - The same decoder relabelled B3 does differ, which shows the test would catch a chain.
4. `test_the_tilt_decoder_recovers_a_simulated_heterozygosity_tilt`: τ* ~ N(0.6, 0.8) per SNP plus U; 1000 train rows, 500 dev rows. Checks Pearson ≥ 0.8, matching sign on SNPs with |τ*|>0.5, fitted mean within 0.1 of the true mean (this catches a centred τ), dev NLL of T < B1, and `converged`.
5. `test_arm_t_draws_keep_the_calibrated_allele_frequency_where_the_b3_chain_drifts`: 20 draws; the bound is `3·sqrt(p(1−p)/(2·n_draws·N)) + 1e-12`. T stays inside at every SNP; B3 goes outside it (`np.any`).
6. `test_group_tilts_hold_one_row_per_group_read_only_by_that_groups_samples[cohort|superpop]`: τ shape is `(C,J)` / `(S,J)`. Moving the tilt row cohort 2 reads leaves `call_logits` of every other group's samples `assert_array_equal`, and changes every het logit inside the group. Flipping only cohort 2's train calls moves that group's row more than 10× as much as any other row. See Concerns for why this isn't a bitwise check on the fitted rows.
7. `test_fitting_ignores_every_row_outside_the_training_indices`: now parametrized over B3 / T-snp / T-superpop. It also asserts that `tilt` and `train_nll_per_call` are bit-identical.
8. `test_decoders_saved_before_the_tilt_arm_still_load_and_decode_unchanged`: writes a file in the exact four-arm key layout. After loading, `tilt_scope == "none"`, `lambda_tilt == 0.0`, both arrays have shape (0,), and B3's `nll_calls` and `sample_calls` are bit-identical to the in-memory decoder.

Existing tests extended for the new fields and rules:
- `test_save_and_load_round_trip_every_field` now also covers T-superpop and checks all four new fields.
- `test_cohort_offsets_are_centered_on_train_cohort_proportions` adds a T-snp case (pins "T trains U").
- `test_fit_decoder_rejects_malformed_inputs` has 7 new malformed-input cases.

## Measured numbers (from this repo's code, not the prototype)

- **Dosage preservation:** max |E[G] − 2p| is 4.4e-16 on the brief's grid (τ ∈ {−3…3}) and on τ ∈ linspace(−4, 4, 41). It is 1.1e-15 even at τ ∈ linspace(−30, 30, 121). Rows sum to 1 within 2.2e-16. At τ=0 the max relative error against the Binomial pmf is 2.2e-12. Under Python 3.10 / numpy 2.1.1 / torch 2.4.1 the error is 4.4e-16.
- **Recovery (test 4):** Pearson r = **0.9952**, sign matches on 20/20 clear SNPs, fitted mean 0.582 vs true 0.590, dev NLL 0.83746 (T) vs 0.89382 (B1), converged.
- **AF drift (test 5):** T's max |AF−p| = 0.0061, worst SNP at 0.58× the bound, 0/30 SNPs outside. B3's max |AF−p| = 0.0928, up to 7.9× the bound, 24/30 SNPs outside.
- **Granularity (test 6, flip cohort 2):** mean |Δτ| per row. Cohort scope: [0.0088, 0.0095, **0.6189**, 0.0090], a 65× ratio. Superpop scope: [0.0093, **0.4509**], a 48× ratio.

## TDD evidence

RED, run after adding the tests and before any implementation:
```
.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_genotype_decoder.py
19 failed, 24 passed, 1 warning in 12.70s
```
Every failure was `TypeError: fit_decoder() got an unexpected keyword argument 'tilt_scope'`, `GenotypeDecoder.__init__() got an unexpected keyword argument 'tilt'`, or `AttributeError: ... no attribute 'tilt_scope'`. That's the expected reason: the API doesn't exist yet. The malformed-input cases fail with TypeError instead of ValueError. `{"arm": "T"}` already passed because T wasn't a valid arm yet.

First GREEN attempt: 3 failed. Both causes were test design, not implementation:
- **Test 5:** the B3 control didn't drift because `_panel` logits are centered at 0, so AF≈0.5 at every SNP, where the diagonal residual is symmetric under g↔2−g and can't move AF. Fixed by adding per-SNP intercepts (`linspace(−2.5, 2.5)`), which gives off-centre AFs like the real panel.
- **Test 6:** the brief's literal "other rows `assert_array_equal`" failed by up to 0.064. See Concerns.

GREEN:
```
.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_genotype_decoder.py
44 passed, 1 warning
.venv/bin/python -m pytest -q -p no:cacheprovider tests/
155 passed, 1 warning in 27.00s        (baseline 136 + 19 new cases; the warning is the pre-existing CUDA one)
/home/user/Envs/csdi/bin/python -c "import src.models.genotype_decoder"   -> OK
```

## Extra verification beyond the tests

- **B0–B3 bitwise unchanged.** Before editing, I fingerprinted for all four arms on a panel with a missing call: fitted `cohort_offset` and `residual`, `train_nll_per_call`, `converged`, `call_logits`, `nll_calls`, a seeded `sample_calls`, plus the `cohort_offset` and `residual` of the loaded oracle `decoder_B{0..3}.npz`. After the final change, `np.array_equal` holds on every array ("bitwise identical").
- **Oracle artifacts:** `outputs/diagnostics/hipodit_ld_oracle_20260915/decoder_B{0,1,2,3}.npz` load with `tilt_scope="none"`, `tilt.shape=(0,)` and `lambda_tilt=0.0`. Nothing under `outputs/` was touched.
- **Mutation check.** I injected 7 bugs one at a time; each is caught by at least one test:
  - no dosage root (τ on het only): 5 failures
  - T sampler reads the chain: 3
  - T logits read the chain: 1
  - soft-centred τ: 3
  - superpop routed by the wrong index: 1
  - legacy load without defaults: 1
  - U frozen for T: 2

  Two gaps turned up and were fixed before commit. The soft-centred τ was first missed, because it shrinks τ but keeps the signs; the mean-level assertion was added. Freezing U was first missed; the T case was added to the centering test.
- **Python 3.10 smoke test (csdi env, no pytest):** T fits for snp, superpop and cohort all converge, save/load round-trips with bit-identical NLL, sampling works, and the oracle files load.

## Self-review

- **Completeness:** everything in the brief is covered: fields, fit kwargs, all three scopes, the validation list, backward-compatible load, and all 8 behaviours. Test 7 is done by parametrizing the existing no-leakage test, as the brief says.
- **Quality:**
  - One torch implementation of the tilt formula is shared by the likelihood, the sampler and the fit, so the formula isn't duplicated.
  - It stays stable at p→0/1 through `logsigmoid` of the minor logit plus the mirror; |η|>745, where the minor probability underflows to 0, still gives the correct limit.
  - The sqrt argument is always ≥ t² > 0, so there are no inf gradients.
  - During review I simplified the tilt-shape lookup, used one return in the objective, and changed the superpop contiguity check to avoid `arange(max+1)`.
- **Discipline:** no new classes, registries or config objects. B0–B3 code paths keep their exact expressions.
- **Testing:** thresholds are structural claims with measured margin, not the controller's prototype numbers. Output is clean apart from the pre-existing CUDA warning.

## Concerns

1. **Test 6 can't be written the way the brief states it.** The brief says that after changing one cohort's calls, the other τ rows stay `assert_array_equal`. That can't hold with this model:
   - The brief requires U to be trained jointly and weight-centered (`w @ U = 0` per SNP). That constraint links every cohort, so changing cohort 2's calls moves every cohort's centered U row (mean |ΔU| ≈ 0.11 for untouched cohorts), and their τ rows follow (mean |Δτ| ≈ 0.009, max 0.064).
   - With U pinned (λ_u=1e8) the leak drops ~150× to ~6e-5. What's left is the tolerance of the joint LBFGS fit, which also rules out bit equality.

   The exact claim that is true is routing: a group's τ row is read only by that group's samples. The test checks that with `assert_array_equal`, plus shape checks and a >10× dominance check on the fitted rows (measured 48–65×). If the controller wants the fit-level claim exact, the only option would be fitting τ per group with U frozen, which contradicts the brief's joint (U, τ) objective.
2. **Minor, beyond the brief:** B0–B3 with a tilt scope other than `"none"` now raises `ValueError`, instead of silently ignoring the tilt.
3. **Minor:** the dataclass doesn't cross-check that a hand-built or loaded decoder's `arm` and `tilt_scope` agree. `arm` alone decides which code path runs. `fit_decoder` enforces consistency, and only fitted or saved decoders reach the downstream scripts.
