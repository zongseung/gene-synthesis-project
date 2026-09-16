# Task 12 report — pooled-AF gauge for the cohort offset

**Status:** DONE
**Commit:** `d847fe4 feat: gauge the cohort offset by the train pooled allele frequency` (two files, no
`Co-Authored-By` trailer, branch `hipodit-ld`)

## What I implemented

`src/models/genotype_decoder.py` only; every decoding path (`call_logits`, `nll_calls`,
`sample_calls`, `_linear_predictor`) is untouched because the constrained offset is what gets stored.

- `fit_decoder(..., offset_gauge: str = "pooled_af")`, values `"pooled_af"` | `"centered"`, anything
  else `ValueError`. Validated for every arm; arms that never fit an offset (`B0`, `B2`) record
  `"centered"` with a NaN residual, since at a zero offset the two gauges agree.
- `pooled_af_shift(raw)`: per-SNP `s` solving `mean_{i∈train} σ(base[i,j] + raw[c_i,j] − s_j) =
  target_af[j]` by 25 unrolled Newton steps inside the torch graph, so LBFGS differentiates through
  the constraint. `_NEWTON_STEPS = 25` is a module constant, not exposed to CLI/config.
- `target_af = nansum(train_calls) / max(observed_count, 1) / 2` — observed training calls only; the
  `max(·, 1)` keeps an all-missing SNP from raising rather than inventing a target.
- The objective uses the gauged offset for the likelihood and ridges the **raw** `Ũ` under
  `"pooled_af"` (the constraint fixes the per-SNP constant, so the ridge picks the minimum-norm
  representative) and the centered offset under `"centered"` — that branch is byte-for-byte the old
  formula.
- After the fit, `cohort_offset = Ũ − s` (under `no_grad`), `pooled_af_residual = max_j |mean_i σ(…)
  − target_af[j]|` (NaN for `"centered"`), and `converged` is ANDed with
  `residual ≤ _AF_RESIDUAL_TOLERANCE = 1e-8`.
- New dataclass fields `offset_gauge: str = "centered"` and `pooled_af_residual: float = math.nan`,
  written into the `scalars` JSON by `save` and read by `load` through `scalars.get(...)`, so `.npz`
  files without them load as centered/NaN.
- Docstring on `fit_decoder` states both gauges and that `cohort_weights` still records the train
  cohort proportions but does not centre under `"pooled_af"`.
- `# ponytail:` note on the fixed Newton budget naming its ceiling and the upgrade trigger
  (`pooled_af_residual` + the `converged` flag). Removed one duplicated `("B1","B3","T")` literal in
  favour of `_FITTED_OFFSET_ARMS`.

## Tests

All 8 brief items, plus two I added during self-review (items 9 and 10 below).

| # | Item | Where |
|---|---|---|
| 1 | constraint met, field matches the measured residual | `test_the_pooled_af_gauge_matches_the_train_allele_frequency_where_centering_drifts` |
| 2 | centered contrast (residual > 1e-10, offsets differ) | same test |
| 3 | cohort signal kept (dev NLL beats B1, cohort AF beats B0) | `test_the_pooled_af_gauge_keeps_the_cohort_signal_it_was_fitted_for` |
| 4 | train-only (incl. `pooled_af_residual`) | `test_fitting_ignores_every_row_outside_the_training_indices` — extra `("T","snp","pooled_af")` case |
| 5 | NaN handling of `target_af` | `test_the_pooled_af_target_is_taken_from_the_observed_calls_only` |
| 6 | save/load round trip + old-format defaults | `test_save_and_load_round_trip_every_field`, `test_decoders_saved_before_the_tilt_arm_still_load_and_decode_unchanged` |
| 7 | gauge validation | `test_fit_decoder_rejects_malformed_inputs` — `{"offset_gauge": "logit"}`, `{"offset_gauge": "none"}` |
| 8 | both ridges alive (arm T, `lambda_u=2.0` ≠ `lambda_tilt=0.5`) | `test_the_tilt_fit_is_stationary_for_both_ridge_penalties_it_reports` |
| 9 | the constraint is fitted through, not applied afterwards | `test_the_pooled_af_fit_is_stationary_along_its_own_constraint_surface` |
| 10 | a Newton solve that misses is not silent | `test_a_pooled_af_fit_that_misses_its_constraint_is_reported_unconverged` |

Two existing tests now pass `offset_gauge="centered"` explicitly — the weighted-centering test and
the B1 ridge-stationarity test. Both are about the centered gauge, which is no longer the default;
no assertion in either was weakened or removed.

### RED

```
$ .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_genotype_decoder.py
E       TypeError: fit_decoder() got an unexpected keyword argument 'offset_gauge'
...
17 failed, 34 passed, 1 warning in 2.64s
```

Failures were the expected three shapes: `TypeError` on the new kwarg, `AttributeError` on the two
new fields (save/load and legacy-load tests), and `DID NOT RAISE` for the two gauge-validation cases.

### GREEN

```
$ .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_genotype_decoder.py
53 passed, 1 warning in 5.0s

$ .venv/bin/python -m pytest -q -p no:cacheprovider tests/
183 passed, 1 warning in 18.59s        # baseline 174 + 9 new cases; the 1 warning is the
                                       # pre-existing CUDA-driver UserWarning

$ /home/user/Envs/csdi/bin/python -c "import src.models.genotype_decoder"
csdi import ok 3.10.12
```

Every existing artifact still loads under Python 3.10:

```
$ /home/user/Envs/csdi/bin/python -c "<load every outputs/diagnostics/*/decoder_*.npz>"
decoder_B0.npz  arm=B0  gauge=centered residual=nan     (9 files: B0-B3, T, T_cohort, T_superpop)
```

Nothing under `outputs/` was written, moved or deleted.

## Mutation checks (do the tests actually bite?)

Each mutation applied to the module, suite run, module restored from a pristine copy.

| Mutation | Result |
|---|---|
| `lambda_u * (ridged**2).sum()` → `0.0 * …` | **both** stationarity tests fail, max abs derivative 0.630 and 0.986 vs the 1e-3 bound |
| `lambda_tilt * (tilt**2).sum()` → `0.0 * …` | only `test_the_tilt_fit_is_stationary_for_both_ridge_penalties_it_reports` fails (0.238); the pre-existing B1 test stays green — exactly the Task 7 gap |
| ridge dropped **only** in the pooled path (`ridged = torch.zeros(())`) | `…stationary_along_its_own_constraint_surface` fails |
| constraint applied **post hoc** (fit with the centered gauge, shift only at the end) | `…stationary_along_its_own_constraint_surface` fails; everything else green |
| `converged` AND-guard deleted | `…reported_unconverged` fails |

The post-hoc row is why item 9 exists and why it is written as a stationarity test. My first version
of it compared the constrained fit's objective against a feasible-but-post-hoc competitor; the
post-hoc mutant **passed** that test (its own ridge-on-raw choice made it beat the competitor), so I
threw it away. The replacement nudges the reported offset along directions tangent to the constraint
surface — the normal per SNP weights each cohort by the Bernoulli variance its training rows carry —
and checks the central-difference derivative of `NLL + λ_u ‖U − mean_c U‖²` (the ridge the raw
parameterisation leaves on the constraint set, its per-SNP constant being free). A post-hoc shift is
feasible but not stationary there, so it fails.

## Measured numbers

Constraint residual, `max_j |mean_{i∈train} σ(base + U[c_i,j]) − target_af[j]|`:

| Panel | `pooled_af` | `centered` |
|---|---|---|
| test panel, 400×30, 4 cohorts, 300 train | 1.67e-16 (field) / 5.55e-16 (measured in numpy) | 4.91e-02 |
| realistic panel, 2002 train × 361 SNPs, 26 cohorts, 1% missing | 2.22e-16 (field) / 1.55e-15 (measured) | 2.25e-02 |

Fit timing, arm T with `tilt_scope="snp"` on the realistic panel (2002×361, 26 cohorts, single CPU
process, `.venv` / torch 2.13, float64):

- `offset_gauge="pooled_af"`: **6.1 s**, converged
- `offset_gauge="centered"`: 1.1 s, converged

That is ~5.5× the centered fit and comfortably inside the 22 s the prototype took, so no performance
report was warranted.

Edge probe (not a committed test): a SNP with no observed training call and a SNP monomorphic in
train, fitted under `pooled_af` — offsets stay finite (≈ −26.6, driven there by a target of 0),
residual 3.8e-12, `converged=True`, per-call NLL finite. The `max(observed_count, 1)` guard plus
`clamp_min(1e-12)` keep the degenerate columns from poisoning the summed objective.

## Files changed

- `/home/user/gene-synthesis-project/src/models/genotype_decoder.py` (+62 −8)
- `/home/user/gene-synthesis-project/tests/test_genotype_decoder.py` (+168 −7)

## Self-review findings (fixed before reporting)

1. **Item-9 test did not discriminate.** Found by running the post-hoc mutation against it; replaced
   with the constraint-surface stationarity test (see above). This is the finding that mattered.
2. **`converged=False` was an unreachable branch.** Newton with the `clamp_min` safeguard reaches
   ~1e-12 even from a target of exactly 0 or 1, so no realistic panel trips the tolerance. Added the
   `monkeypatch`-ed `_NEWTON_STEPS = 0` test so the guard is exercised rather than argued for.
3. **Duplicated arm tuple.** `("B1","B3","T")` appeared twice once I added the gauge switch; both now
   read `_FITTED_OFFSET_ARMS`.
4. **Helper bug in my own test data** (`_unbalanced` cohort sizes summing to 410 rather than 400) —
   caught immediately by the module's own label contract.

## Concerns

- **The default change reaches `scripts/hipodit_genotype_check.py`.** It calls `fit_decoder` without
  `offset_gauge`, so the oracle now fits every `B1`/`B3`/`T` arm under the pooled-AF gauge. That is
  the point of the task, but it makes the existing `outputs/diagnostics/hipodit_*_oracle_*` artifacts
  stale: the Gate 1′ study needs a rerun by the controller. I changed no file outside the two
  allowed and wrote nothing under `outputs/`.
- **Two existing tests were edited** to name `offset_gauge="centered"` (listed above). Without the
  kwarg they assert a property the new default deliberately does not have.
- The `pooled_af` fit is ~5.5× slower than the centered one, entirely from the 25 unrolled Newton
  steps per objective evaluation. 6.1 s at study size; the `# ponytail:` comment names the fixed
  budget as the ceiling if that ever matters.
