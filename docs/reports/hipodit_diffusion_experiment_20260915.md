# GLM-PCA information and diffusion schedules

## Scope and hypothesis

Implement a runnable dosage experiment using Binomial(2,p) GLM-PCA,
train-only affine normalization, the existing HiPoDiT denoiser, and either
a standard or decoder-informed forward schedule. Missing calls are masked.
The research hypothesis is that retaining high-sensitivity latent factors
longer improves generated genotype fidelity at matched compute. It is not
an established benefit or a novelty claim.

## Mathematical contract

Stored timesteps run from 0 to T-1, after the first noising step.
For base cumulative noise h(t)=-log(alpha_bar(t)), set u(t)=(h(t)-h(0))/(h(T-1)-h(0)).
Three schedule bands use h_r(t)=h(0)+(h(T-1)-h(0))*u(t)^r, r in {0.5,1,2}.
All bands share initial and terminal signal-to-noise ratios. Larger r keeps
signal longer at intermediate times. Coordinatewise DDPM posterior and DDIM
updates must use exactly these same schedules; Min-SNR is coordinatewise.
Structural padding remains zero in forward training and reverse sampling.
Only T*3 schedule values and a K*G index map are stored, not T*K*G arrays.

Assign bands using train-only, mean-per-SNP diagonal Fisher information in
normalized coordinates: E_i,j[2*p_ij*(1-p_ij)*V_jk^2]*std_k^2.
This is a diagonal, basis-dependent approximation. It neither preserves LD
by construction nor represents a Riemannian diffusion. Quantile boundaries
are computed from training information only. Constant information uses the
ordinary schedule.

## Prior work and claim boundary

- [MuLAN, NeurIPS 2024](https://arxiv.org/abs/2312.13236): learned multivariate noise is prior art.
- [Logistic factor analysis](https://pmc.ncbi.nlm.nih.gov/articles/PMC4795615/): binomial genotype factor models are prior art.
- [DDPM](https://arxiv.org/abs/2006.11239), [DDIM](https://arxiv.org/abs/2010.02502).

The candidate contribution is the particular link between genotype-likelihood
sensitivity and the forward corruption process, together with evidence of
benefit. Establishing novelty needs comparison with generic adaptive noise
and shuffled sensitivity maps, not just a standard DDPM baseline.

## Execution checklist

- [x] Projection/schedule regressions reproduced the missing behavior (11 expected failures before implementation).
- [x] Binomial fitting/scoring, missingness and train-only contracts pass.
- [x] Standard and information schedules use one training/inference contract.
- [x] Removed confirmed unused legacy backend and duplicated loss helpers.
- [x] Ran fresh real-data preparation, both training arms and genotype checks.
- [x] Record numerical results, limitations and validation below.

## Implementation and removals

The new preparation command calls `src/preprocessing/binomial_glm_pca.py`:
SciPy L-BFGS fits a low-rank Binomial(2,p) logit likelihood plus unit ridge
penalties on factors/loadings, using only observed training calls. Scoring
uses the frozen decoder and separate training/held-out optimizer calls.
The affine normalization and Fisher schedule use training rows only.
The existing full-autosome Poisson preprocessing remains a separate legacy
path; no historical cache or checkpoint was converted or overwritten.

`src/models/noise_schedule.py` owns scalar and three-band schedules.
`src/models/diffusion.py` now shares CFG prediction and x0 reconstruction
between samplers, enforces forward/reverse padding support, weights each
coordinate with its own Min-SNR, and starts one-step DDIM at the final
training timestep. Unused schedule buffers were removed.
`src/training/losses.py` lost the unreferenced `masked_mse_loss` and
`min_snr_weight` duplicates. The unreferenced 341-line
`src/preprocessing/glm_pca_torch.py` backend was deleted. These tracked
source deletions are recoverable from Git; data/checkpoints were retained.

## Authoritative real-data pilot

Prepared data: `outputs/diagnostics/hipodit_fisher_20260915_unique`.
2,504 individuals, split 2,002/251/251; 8 chr17 genes, 4 factors per gene.
Each SNP belongs to the first retained gene in genomic order, preventing
inconsistent independent generation of the same locus in overlapping genes.
There are 361 variant columns; final verification confirmed all 361 are unique.
All 8 decoder optimizations converged within the 1,000-iteration budget.
Preparation took 6.89 seconds; normalization round-trip max error 1.19e-7.

Both arms used the same prepared data, seed 20260327, initial denoiser,
batch size 64, 100 optimizer updates, 100 DDIM steps and 64 generated people.
All 64 reference people are the corresponding validation individuals, with
the same population labels requested for synthesis. Decoder reference
draws use an independent fixed seed, identical between the two arms.

| Measurement | Standard | Fisher bands | Real latent decoder reference |
| --- | ---: | ---: | ---: |
| AF mean absolute error | 0.0444512 | 0.0392356 | 0.0177891 |
| Local dosage covariance MAE | 0.0356659 | 0.0366468 | 0.0201795 |
| Fraction in {0,1,2} | 1.0 | 1.0 | 1.0 |
| Unique individual fraction | 1.0 | 1.0 | 1.0 |

1,027 pairs were evaluated within genes at SNP-index separations 1, 2 and 4.
This measures unphased dosage covariance, not haplotype LD or LD decay.
Fisher bands reduced AF error by approximately 11.7% in this pilot but
increased covariance error by approximately 2.8%. This is mixed evidence,
not an established improvement. Standard remains the default.
Training/generation/checking took 3.89 seconds (standard) and 4.40 seconds
(Fisher), using existing PyTorch 2.4.1/CUDA 12.1.

Results and discrete genotype arrays:

- `outputs/diagnostics/hipodit_fisher_20260915_unique_standard/diagnostic_report.json`
- `outputs/diagnostics/hipodit_fisher_20260915_unique_fisher/diagnostic_report.json`
- Each run contains `synthetic_genotypes.npy`, `synthetic_samples_original.npy`,
  a checkpoint and its configuration; column identity is in the prepared
  `gene_variant_map.json` and `genotypes.npz` offsets.

Earlier `*_verified`, `*_standard_final`, and `*_fisher_final` runs retained
duplicated SNP columns from overlapping genes and are superseded by the
`*_unique*` artifacts. They remain as diagnostic history, not final results.

## Reproduce in new directories

```bash
OPENBLAS_NUM_THREADS=1 uv run --no-sync python scripts/hipodit_rebuild_check.py prepare --output-dir outputs/diagnostics/my_binomial_data --genes 8 --components 4 --max-variants 64 --glm-iterations 1000 --seed 20260327
OPENBLAS_NUM_THREADS=1 /home/user/Envs/csdi/bin/python scripts/hipodit_rebuild_check.py train --output-dir outputs/diagnostics/my_binomial_data --run-dir outputs/diagnostics/my_standard --schedule standard --steps 100 --batch-size 64 --samples 64 --ddim-steps 100 --seed 20260327 --device cuda
OPENBLAS_NUM_THREADS=1 /home/user/Envs/csdi/bin/python scripts/hipodit_rebuild_check.py train --output-dir outputs/diagnostics/my_binomial_data --run-dir outputs/diagnostics/my_fisher --schedule fisher --steps 100 --batch-size 64 --samples 64 --ddim-steps 100 --seed 20260327 --device cuda
```

The CUDA interpreter above is an existing machine-specific environment.
On another machine use a compatible PyTorch environment; no new driver or
project dependency installation was performed. The public generator reads
the band map from checkpoint configuration, including the canonical training
entry point, so generation does not silently fall back to a scalar schedule.

## Research interpretation

The prototype makes the hypothesis testable. It does not establish journal
novelty. Required follow-ups are longer fits, independent training seeds,
larger disjoint genomic panels, shuffled-information/generic adaptive-noise
controls, and matched Min-SNR ablations to isolate noising from loss weighting.
The current comparison is one small panel, one training seed, one stochastic
binomial draw per arm, no formal privacy guarantee, and no phased haplotypes.
MAF filtering excludes variants below 1%; rare-variant claims are unsupported.
The lower error from real latent factors indicates remaining generator and
decoder limitations; it is a reconstruction reference, not an unbiased
generative-performance upper bound. Fixed-t denoising probes from the two
arms have different coordinate SNRs and must not be compared as the same task.
Binomial(2,p) also assumes conditionally independent allele draws; it is not
an unrestricted three-category genotype likelihood. Heterozygosity departures
and residual inter-SNP dependence require explicit evaluation.

## Final verification

- `OPENBLAS_NUM_THREADS=1 uv run --no-sync python -m pytest -q`: 94 passed in 9.02 seconds.
  One existing CUDA-driver compatibility warning occurred in the project's
  test environment; GPU training used the compatible existing CUDA 12.1 environment.
- `uv run --no-sync python -m compileall -q src scripts tests`: exit 0.
- `git diff --check`: exit 0.
- Preparation/training CLI help succeeded; `--steps 0` was rejected with exit 2.
- Both final genotype arrays have shape (64,361), dtype int8, and only 0/1/2 values.
- The public generation CLI reloaded the final Fisher checkpoint and generated
  26 finite (4,8) latent samples, one per population, with 20 DDIM steps in 3.0 seconds.
  Output: `outputs/diagnostics/hipodit_fisher_20260915_unique_reload`.
  Expected warnings: this small pilot has no EMA weights and needs no padding mask.
  These public CLI outputs are latent tensors, not decoded genotype calls.
- Python LSP diagnostics were unavailable because no Python language server
  is installed and installation was previously declined; runtime and compile
  checks above are evidence, not a substitute claim of clean LSP diagnostics.
