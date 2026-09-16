# HiPoDiT ponytail refactor — 2026-09-16

Behaviour-preserving deletion of abandoned scaffolding, plus two repository-safety fixes.
Findings come from two read-only audits run on this branch; every deletion below is either
provably unreachable or covered by a named existing test.

## Global Constraints

1. **No number may move.** The chr17 study produced frozen, published gate results
   (`docs/reports/hipodit_t_results_20260916.md`). Any change that can alter a metric, an RNG
   draw order, a seed, a fit, or the bytes of a saved artifact is out of scope. When in doubt,
   do not make the change; report it instead.
2. **The command surface is frozen.** Every subcommand and flag recorded in results report
   §12.1 must keep working identically.
3. **Frozen artifacts under `outputs/diagnostics/` are never regenerated or touched.**
   `outputs/training/` holds a live sweep; do not touch it either.
4. **The test suite baseline is 229 passed, 1 warning** (`.venv/bin/python -m pytest tests/ -q`).
   Every task must end at 229 passed. No task in this plan removes a test. (An earlier draft said
   Task 4 would end at 228; that was a drafting error, corrected after Task 4's implementer caught
   it. Nothing in this plan deletes a test file or a test function.)
5. Use the root `.venv` (Python 3.13, CPU) for tests. Never `uv sync` and never run anything on
   the GPU: a training sweep owns both GPUs for the duration of this work.
6. Do not reformat untouched lines. The diff should contain deletions and nothing else wherever
   possible.

## Task 1 — repository safety (do this first, it is the only task that prevents data loss)

Two defects, both verified by the controller before this plan was written.

1. `.gitignore:58` is `/docs`, so everything under `docs/` is ignored. The ten files currently
   tracked there were force-added. CLAUDE.md instructs that every decision be recorded in
   `docs/reports/...`; a newly written report is silently untracked today and is lost on a clean
   checkout. Verified: `git check-ignore -v docs/reports/newfile.md` prints `.gitignore:58:/docs`.
   Change the rule so that `docs/` is tracked. `claudedocs/` on the next line is a separate
   scratch directory and must stay ignored.
2. `isolated_runs/` is 115 GB, untracked, and matched by no ignore rule, so a single
   `git add -A` would attempt to stage it. Verified: `git check-ignore -v isolated_runs` exits
   non-zero. Add an ignore rule. While there, ignore the other untracked bulk at the repo root
   that is plainly not source: `hanmed_vlm_ko_10min/`, `ppt_final_*.pptx*`, `*.pdf` at root,
   `autoresearch/`, `paper/`, `.debug-journal.md`, `hammed_icon/`.
   Do **not** ignore `notebooks/`, `scripts/`, `src/`, `tests/`, `configs/`, `docs/`.
3. `.gitignore` has no trailing newline after `/claudedocs`; add one.

**Do not run `git add`, `git rm`, or `git commit -a` in this task.** Edit `.gitignore` only, then
commit that one file. Nothing else may enter the commit.

**Verify:** `git check-ignore -v docs/reports/newfile.md` must now exit non-zero (not ignored),
`git check-ignore -v isolated_runs` must now exit zero (ignored), and `git status --short` must
not list `isolated_runs`. Paste all three outputs.

## Task 2 — delete the dead component grid-search chain (~355 lines)

None of these are reachable. `src/preprocessing/run_pipeline.py:109` hardcodes
`optimal_k = PCA_CANDIDATES[0]`, and CLAUDE.md states the grid search does not apply to the
`glm_pca` default path.

Delete:
- `src/preprocessing/dim_reduction.py` — `grid_search_optimal_k` (the sole caller of the two
  backend grid searches; itself has zero callers repo-wide)
- `src/preprocessing/pca.py` — `grid_search_optimal_pca`, `_evaluate_pca_k_for_gene`,
  `analyze_pca_information_loss`
- `src/preprocessing/glm_pca.py` — `grid_search_optimal_glm_pca`, `evaluate_glm_pca_k_for_gene`
- constants orphaned by the above in `src/preprocessing/config.py`: `MARGINAL_GAIN_THRESHOLD`,
  `MARGINAL_GAIN_DECAY_RATIO`, `PCA_SAMPLE_GENES` — but only after grepping to confirm each has
  no other reader
- imports orphaned by the above: `json` and `ThreadPoolExecutor` in `pca.py`;
  `ThreadPoolExecutor` and `pandas` in `glm_pca.py`

Keep `PCA_CANDIDATES`: `run_pipeline.py:109` still reads it.

Before deleting, re-prove zero callers yourself with grep across `src/`, `scripts/`, `tests/`,
`notebooks/`, `configs/` and `*.md`, and paste the commands and output into your report. If any
of these turns out to have a caller, stop and report rather than deleting it.

Note for the report, do not fix: `src/preprocessing/glm_pca.py:332` calls `_import_glmpca()`,
which is not defined anywhere. It is inside the code this task deletes, so the latent `NameError`
goes away with it. Say so in your report.

**Verify:** `.venv/bin/python -m pytest tests/test_glm_pca_preprocessing.py tests/test_normalization_contract.py -q`
then the full suite at 229. Also `.venv/bin/python -c "import src.preprocessing.run_pipeline"`.

## Task 3 — delete the superseded checkpoint module and the unreachable model zero-mask path

1. Delete `src/utils/checkpoint.py` (172 lines) and its re-export in `src/utils/__init__.py`.
   It is superseded by `src/training/trainer.py` `save_checkpoint` and
   `manage_top_k_checkpoints`, which the trainer defines and calls itself. The only reference to
   the module anywhere is its own re-export line; every consumer imports `src.utils.<submodule>`
   directly. Re-prove this with grep and paste the output.
2. Delete the model-level zero-mask path in `src/models/hybrid_geno_dit.py`: the
   `register_buffer("zero_mask", None)`, `set_zero_mask`, `enforce_zeros`, and the call to
   `self.enforce_zeros(output)` in `forward`. `set_zero_mask` has zero callers, and a `None`
   buffer cannot be repopulated by a checkpoint load, so the guard is always false and the method
   is a per-forward no-op. The constraint is really enforced by
   `GaussianDiffusion._apply_zero_mask`, which the trainer and generator both wire up.

   This one touches a model the frozen runs used, so prove it is inert before you delete it:
   load an existing checkpoint from `outputs/diagnostics/hipodit_t_e2e_20260916/seed_20260327/checkpoint_diagnostic.pt`
   with `torch.load(..., map_location="cpu")` and show that `zero_mask` is not among the
   `model_state_dict` keys. Paste that output. If it *is* present, stop and report.

**Verify:** full suite at 229, plus the checkpoint-key evidence above.

## Task 4 — remaining small cleanups

Each is independent; do them in one commit.

1. `src/training/losses.py:191` — `x_t = diffusion.q_sample(x, t, noise)` is computed every step
   but consumed only inside the `if aux_cfg.get("enabled", False):` block at :229, and the
   default config has `aux_loss.enabled: false`. Move the line inside that block. `q_sample`
   draws no RNG when `noise` is passed, so no RNG ordering changes; confirm that by reading
   `src/models/diffusion.py` around the `q_sample` definition and quote the lines in your report.
2. `src/preprocessing/pca.py:286-289` — two imports sit inside the per-gene loop. Hoist them to
   module scope. Confirm no circular import by running
   `.venv/bin/python -c "import src.preprocessing.pca"`.
3. Remove the unused imports listed here, after confirming each is genuinely unused in its file:
   `src/training/trainer.py` (`shutil`, `time`, `DistributedSampler`, `load_config`),
   `src/data/sampler.py` (`pickle`, `Path`), `src/utils/config.py` (`copy`),
   `src/utils/ema.py` (`copy`), `src/utils/logger.py` (`os`),
   `src/preprocessing/config.py` (`cpu_count`), `src/preprocessing/merge_data.py` (`subprocess`),
   `src/preprocessing/vcf_parser.py` (`MAF_THRESHOLD`, `MAX_VARIANTS_PER_GENE` — both shadowed by
   function arguments; check carefully that no function body reads the module-level name).
4. Delete `src/preprocessing/__main__.py`. It has zero references, CLAUDE.md documents
   `python src/preprocessing/run_pipeline.py` instead, and it calls `main()` at import time with
   no `__name__` guard, so merely importing it would launch the 22-chromosome pipeline.
5. `src/preprocessing/merge_data.py` re-declares `VCF_DIR` and `VCF_PATTERN` verbatim from
   `src/preprocessing/config.py` (`PER_CHROM_VCF_DIR`, `PER_CHROM_VCF_PATTERN`). Import the
   constants instead of re-declaring them. Keep the module's CLI behaviour identical.

**Verify:** full suite at 229.

## Task 5 — correct the stale claims in CLAUDE.md and README

Documentation only, no code. Fix what the audits proved wrong:

1. CLAUDE.md documents `src/evaluation/run_evaluation.py`, which does not exist. The real entry
   point is `scripts/evaluate_synthetic_metrics.py`. Check its actual flags with `--help` and
   write the corrected command.
2. CLAUDE.md documents `configs/sweep.yaml` for the wandb sweep; `configs/` contains only
   `default.yaml`. Either remove the sweep command or mark the config as absent — do not invent
   a file.
3. CLAUDE.md's Key Design Principles say "`enforce_zeros` + `zero_mask` preserves biological
   constraints" in a way that implies the model enforces it. After Task 3 the only enforcement is
   in `GaussianDiffusion`. Correct the sentence to say where the constraint actually lives.
4. `README.md` documents `python src/evaluation/pca_compare.py`. Leave `pca_compare.py` itself in
   place (deleting it is not in scope), but if Task 2 or 3 changed anything it relies on, say so.
5. Three more stale spots found by the earlier tasks while working, all now wrong:
   - `README.md` around line 169 draws the model-forward diagram with an output node labelled
     `enforce_zeros` applying the zero mask. Task 3 deleted that model-level path. Correct the
     node so the diagram matches where the constraint is really applied. README lines 92 and 434
     were checked and are still accurate; leave them alone.
   - `src/preprocessing/config.py` has a comment reading `# PCA grid search` above the surviving
     candidate list, but Task 2 deleted the grid search and the list is now read only at index 0.
   - `src/preprocessing/run_pipeline.py`'s module docstring still describes a "Pass 1 ... PCA grid
     search" step that no longer exists.

**Verify:** every command you write must be one you actually ran with `--help`. Paste the output.

## Out of scope — report only, do not fix

The audits surfaced these. They are real but each needs a judgement call the refactor should not
make silently. List them in your final report; change nothing.

- `src/preprocessing/vcf_parser.py:111` — `if _USE_RUST and train_indices is None:` means the
  Rust parser is never used in the production pipeline, because `run_pipeline.py` always passes a
  non-None `train_indices`. The advertised speedup is silently off.
- `src/utils/config.py:163` — `--single_gpu` raises a bare `KeyError` for a config with no
  `distributed:` block, because `distributed` is not in `REQUIRED_KEYS`.
- `src/training/trainer.py` — checkpoints persist optimizer, scheduler and EMA state although no
  resume path exists, and `global_step` is not saved at all.
- `src/training/trainer.py` `CosineWarmupScheduler` — the first optimizer step runs at lr 0.0.
- `pyproject.toml` — `scipy` is imported in four modules but undeclared; `seaborn`, `statsmodels`
  and `glmpca` are declared but never imported; there is no `[build-system]`, which is why five
  modules each hand-roll a `sys.path.insert`.
- `src/evaluation/dupi.py` `_logsumexp` could be `scipy.special.logsumexp`, but it can shift the
  last unit in the last place of published DUPI numbers. Not worth it.

## Task 6 — the only two safe cuts in the chr17 study code

A separate audit swept the chr17 study scripts, the genotype decoder and their five test files —
5,161 lines — for dead code. It found none: zero unreferenced functions, classes, constants or
imports, and no bare `try/except` anywhere. Its verdict was that the study code is tight and that
almost every simplification there is a re-verification job against frozen artifacts. So this task
takes only the two cuts that are both zero-risk and covered by a named existing test.

1. `scripts/hipodit_multiseed_summary.py` — delete the four TypedDict declarations (`Estimate`,
   `MetricSummary`, `PairedDifference`, `Summary`) and the `TypedDict` import, replacing their use
   in annotations with plain `dict` / `list[dict]`. The file has `from __future__ import
   annotations` at the top, so every function annotation is already an unevaluated string and no
   local annotation is evaluated at runtime. Nothing introspects `__annotations__`. There is no
   mypy, pyright or ruff configuration anywhere in the repo, so these declarations are checked by
   no tool. Confirm the absence of a type-checker config yourself before deleting.
2. `scripts/hipodit_multiseed_summary.py` — `ExperimentError.__init__` stores a `detail`
   attribute that nothing reads; the class body becomes just a docstring over `ValueError`.
   `str(error)` is unchanged, which is what the tests match on. Prove `.detail` is write-only with
   grep and paste the output.

3. Two one-line comment corrections that Task 5 could not make, because Task 5 was told to change
   no Python file and these live in Python files. Both describe a grid search that Task 2 deleted:
   - `src/preprocessing/config.py` has a comment reading `# PCA grid search` above the surviving
     candidate list. That list is now read only at index 0 by `run_pipeline.py`. Reword the comment
     to say what the constant is actually for.
   - `src/preprocessing/run_pipeline.py`'s module docstring describes a "Pass 1 ... PCA grid search"
     step. Correct it to describe what Pass 1 now does. There is a second mention further down the
     same file about passing train indices "through the grid search" — fix that too if it is still
     there. Change comments and docstrings only; change no code in either file.

**Verify:** `.venv/bin/python -m pytest tests/test_hipodit_multiseed.py -q` must pass, then the
full suite at 229.

## Also report only, from the chr17 audit — do not fix

- Three near-copies of `_atomic_json` exist, but they differ: the `hipodit_genotype_check` copy
  passes `allow_nan=False` and the prepare/train copies do not. `hipodit_rebuild_train` writes a
  report whose pooled allele-frequency residual is NaN for a centered-gauge decoder, so
  consolidating onto the strict copy would turn a working run into an error. Leave all three.
- `scripts/hipodit_privacy.py` hand-rolls the membership-inference AUC where scikit-learn has
  `roc_auc_score`. The audit ran both over 200 randomized sets and they disagree in the last unit
  in the last place 64 times out of 200. That is a published Gate 4 number. Leave it.
- **Coverage gap worth filing:** no test asserts the contents of the schedule-mode `summary.csv`.
  The decoder-mode CSV header is asserted, the schedule-mode one is not.
