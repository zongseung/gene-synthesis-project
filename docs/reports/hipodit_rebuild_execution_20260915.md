# HipoDIT rebuild execution

Requested: keep GLM-PCA, review FiLM with Astra, implement and calculate with Sol, apply Ponytail, remove OC-FILM.

## Work plan

1. Completed: Astra architecture and likelihood review with primary sources. See `hipodit_rebuild_design_20260915.md`.
2. Completed: Sol implementation of corrected FiLM/AdaLN, smaller baseline, GLM scoring and normalization/metadata contracts, and removal of OC-FILM.
3. Completed: final fresh chromosome-17 pilot from real VCF, train-only preprocessing, GPU training, sampling and same-space evaluation, rerun against the final scoring implementation.
4. Completed: integrated regression checks, checkpoint reload through the public generator, canonical data loader, standard validation evaluator, and inspection of the actual artifacts.

## Existing-data observations

- VCF and panel contain 2,504 sample IDs in exactly matching order, verified by reading both inputs.
- Cached training/validation/test sizes: 2,002/251/251. Cached factors have no dimensionality-reduction provenance in DataFrame attributes.
- Cached values clipped at absolute standardized value 5: training 0.03974%, validation 0.04052%, test 0.04163% of coordinates. Inverse scaling cannot recover these tails.
- The shared project environment has a CUDA build too new for the installed driver. Existing `/home/user/Envs/csdi/bin/python` has working PyTorch 2.4.1 with CUDA 12.1. No driver or shared dependency changes are needed.

## Scope of calculations

The first fresh calculation uses a bounded chromosome-17 gene panel and a short training run. It verifies that the rebuilt pipeline executes and learns through attention; it does not establish full-genome performance, journal novelty, privacy, or genotype validity. Existing full-genome data and checkpoints are preserved.

OC-FILM-specific code/tests/configuration were removed. The old 22,829,847-byte statistics file was moved from `data/processed/cumulant_stats.npz` to `outputs/archive_oc_film_20260915/cumulant_stats.npz`, so it is absent from active preprocessing while remaining recoverable.

## Independent full-length model check

The primary agent ran the current default architecture on an actual cached training batch of shape `(2, 4, 24576)` using the existing CUDA environment. This is a dimensional/gradient check on legacy features, not a provenance or fidelity claim.

- Parameter count: 9,331,588.
- Predicted-noise shape matches the input; predictions and loss are finite (loss 0.8187583).
- All four attention residual gates receive nonzero first-step gradients: norms 3.48e-6, 5.03e-6, 3.83e-6, 3.72e-6.
- Peak allocated GPU memory: 543.2 MiB; approximately 2 seconds including data/model preparation inside the diagnostic.
- Pilot CLI help exits 0. A zero-gene request exits 2 with a positive-integer error before creating an output directory.

## Existing-sample evaluation correction

The primary agent ran `scripts/evaluate_synthetic_metrics.py` against the unchanged epoch-455 guidance-2.5 samples, with explicit legacy sample space and the original normalization statistics. Output: `outputs/20260915_rebuild_legacy_eval_gw2p5/summary_metrics.json`.

| Metric | Previous mismatched scales | Corrected common scale |
|---|---:|---:|
| Gaussian W2 | 1108.6334264 | 0.6123970 |
| MMD-RBF | 0.9594255 | 0.0396405 |
| Utility index | 0 | 0.4031391 |

Both calculations use 251 real and 780 synthetic samples, 2,000 genes and seed 42. This demonstrates an evaluation correction, not improved generation or new-model performance. Re-evaluating this historical test set is an audit of the earlier result; the fresh pilot uses validation data.

## Final fresh calculation

Authoritative output directory: `outputs/diagnostics/hipodit_rebuild_20260915_verified`.

The primary agent executed preparation from the original VCF after the final preprocessing source was frozen, then trained from scratch on the existing CUDA environment. The panel contains 32 chromosome-17 genes, four Poisson GLM-PCA factors per gene and 2,504 subjects split 2,002/251/251. The pilot adjusts spatial downsampling/patch sizes for the small panel; it retains four width-256 DiT blocks. Its architecture and full configuration are saved in `diagnostic_config.yaml`.

| Check | Observed result |
|---|---|
| Fresh preparation | 10.12 seconds; train-only MAF, imputation, GLM decoder/scoring and normalization |
| Training | 100 optimizer steps, seed 20260327, 3.85 seconds including diagnostic generation |
| Fixed training denoising probe | 1.0579865 before → 0.08256568 after, identical batch/timestep/noise |
| Attention | Final attention-gradient L2 1.7919e-4; gates open |
| Normalization round trip | Maximum absolute error 2.3842e-7 |
| Diagnostic output | 64 finite samples on original GLM-factor scale |
| Public checkpoint reload | 52 additional finite samples, two per population, through `src/inference/generator.py` |
| Canonical data loader | 2,002 training / 251 validation subjects; batch `(64,4,32)` |
| Statistics binding | Checkpoint, statistics-file SHA-256 and generation metadata agree |
| Regression suite | `82 passed, 1 warning` in 9.35 seconds |
| Syntax / patch whitespace | `compileall` and `git diff --check` exited 0 |

The ordinary evaluation CLI compared the 52 public-generator samples with 251 validation subjects: PCA(2) W2 1.798374, MMD-RBF 0.126740, utility index 0.624740. These are short-run diagnostic metrics, not improvements over the historical full-genome model: gene panel, samples, fit and evaluation split differ. The fixed denoising probe demonstrates learning on its training batch, not generalization.

Executed commands (use a new output directory to repeat preparation):

```bash
uv run --no-sync python scripts/hipodit_rebuild_check.py prepare --output-dir outputs/diagnostics/hipodit_rebuild_20260915_verified --genes 32 --components 4 --max-variants 128 --glm-iterations 30 --seed 20260327
/home/user/Envs/csdi/bin/python scripts/hipodit_rebuild_check.py train --output-dir outputs/diagnostics/hipodit_rebuild_20260915_verified --steps 100 --batch-size 64 --samples 64 --ddim-steps 20 --seed 20260327 --device cuda
uv run --no-sync python -m pytest -q
```

The existing project Python environment emits a CUDA-driver compatibility warning during CPU tests; GPU checks used the compatible existing environment without changing the driver or shared dependencies. The pilot deliberately has no EMA and no structural padding mask; public inference reports those facts and uses its trained raw weights. LSP diagnostics were unavailable; syntax checks and executable regression tests were used.

Full-autosome preprocessing/training, multiple seeds, binomial dosage decoding, LD/AF/HWE fidelity and publication-quality evaluation have not been run. Poisson reconstruction means can exceed two (observed validation maximum 5.1634), so this run is explicitly not evidence of valid discrete genotype generation. The corrected implementation and bounded calculations are complete; scientific performance claims require those subsequent experiments.
