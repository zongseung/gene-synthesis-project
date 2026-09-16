# HiPoDiT paired multi-seed diagnostic — 2026-09-15

## Scope

This is a prespecified diagnostic replication, not a benchmark or novelty
claim. It does not pool the earlier 100-update/64-sample pilot. The frozen
input was `outputs/diagnostics/hipodit_fisher_20260915_unique`:

- 2,504 people; population-stratified train/validation/test split
  2,002/251/251; eight chr17 genes; four factors per gene; 361 unique SNPs.
- The frozen decoder is Binomial(2,p), fitted on training rows. Input checks
  found genotype calls `(2504, 361)`, all finite and in `{0,1,2}`; the map
  had 361 mapped and 361 unique IDs.
- Key input SHA-256: `dataset.npz`
  `81bb500665192efdafd4008f32f77824e4510454f5adc914227f5eca7586f81c`;
  `genotypes.npz`
  `1eee0952d8bbbdf7e0a7cfdcf67cfbb2de5721af21ae1f46aaa73c4a01e195df`;
  `gene_variant_map.json`
  `cb7f30c8723b916605a865373f1ed3d5206d84c6a9909ac56129e925bb6c55dc`;
  decoder parameters
  `e0dc0a89cbda1245fd7451ae60de75979e6288d5cab3079b102817f912a557ff`.

The declared settings were seeds 20260327–20260331; standard and Fisher arms
per seed; 1,000 optimizer updates; batch size 64; 251 validation-labelled
samples; 100 DDIM steps; CUDA. Jobs were serial to avoid GPU contention.
Execution used the existing compatible interpreter (PyTorch 2.4.1+cu121,
CUDA 12.1, NVIDIA RTX A6000):

```bash
/home/user/Envs/csdi/bin/python scripts/hipodit_multiseed.py \
  --prepared-dir outputs/diagnostics/hipodit_fisher_20260915_unique \
  --output-dir outputs/diagnostics/hipodit_multiseed_20260915
```

## Results

Lower is better. Deltas are Fisher minus standard, so negative values favour
Fisher. Every requested seed is included.

| Seed | Standard AF MAE | Fisher AF MAE | Δ AF | Standard local covariance MAE | Fisher local covariance MAE | Δ covariance |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20260327 | 0.0264593 | 0.0211729 | -0.0052863 | 0.0233071 | 0.0201099 | -0.0031972 |
| 20260328 | 0.0230601 | 0.0227897 | -0.0002704 | 0.0207906 | 0.0205989 | -0.0001917 |
| 20260329 | 0.0252177 | 0.0277781 | 0.0025604 | 0.0234604 | 0.0252899 | 0.0018295 |
| 20260330 | 0.0229773 | 0.0256922 | 0.0027149 | 0.0205956 | 0.0211756 | 0.0005801 |
| 20260331 | 0.0227842 | 0.0241196 | 0.0013354 | 0.0201157 | 0.0203728 | 0.0002570 |

| Metric | Standard mean ± sample SD | Fisher mean ± sample SD | Paired Δ mean ± sample SD | Fisher wins |
| --- | ---: | ---: | ---: | ---: |
| AF MAE | 0.0240997 ± 0.0016499 | 0.0243105 ± 0.0025557 | 0.0002108 ± 0.0032979 | 2/5 |
| Local dosage covariance MAE | 0.0216539 ± 0.0015990 | 0.0215094 ± 0.0021496 | -0.0001445 ± 0.0018644 | 2/5 |

The repeated-seed results show no consistent Fisher advantage: its mean AF
MAE was higher and it won two of five seeds; covariance MAE was slightly
lower on average but also won only two of five seeds. This descriptive
five-seed result does not support a superiority or novelty claim; standard
remains the default.

## Execution ledger and artifacts

All ten child commands exited 0. The run started at
2026-09-15T06:30:08Z and ended at 2026-09-15T06:33:49Z; summed child runtime
was 220.684 seconds.

| Seed | Standard seconds | Fisher seconds | Status |
| ---: | ---: | ---: | --- |
| 20260327 | 22.434 | 22.565 | complete / 0 |
| 20260328 | 22.130 | 21.677 | complete / 0 |
| 20260329 | 22.086 | 22.535 | complete / 0 |
| 20260330 | 21.984 | 21.730 | complete / 0 |
| 20260331 | 21.790 | 21.754 | complete / 0 |

The complete command/start/end/status ledger is
`outputs/diagnostics/hipodit_multiseed_20260915/runs.jsonl`. Raw reports are
under `seed_<seed>/{standard,fisher}/diagnostic_report.json`; aggregate
artifacts are `manifest.json`, `summary.json`, `summary.csv`, and
`paired_differences.csv` in the same root. Their SHA-256 values are:

| Artifact | SHA-256 |
| --- | --- |
| `manifest.json` | `a18df76d177740d47a9e68bbb63dd1b8069635fd802fba0a774f162b358e8532` |
| `summary.json` | `6e3a4b48d5b4206b5eff57491988ce3b3ea2273fe749f529d1356c07b4e0477d` |
| `summary.csv` | `c537b51464bffef61d30a658fcd8bef1daf8b90a1385c855bb5c6741eba39ff1` |
| `paired_differences.csv` | `f04c2bb77b0b3a97738e1496fa9cb2833e48dd65425313575e63e73635fc79e8` |

## Independent artifact checks

- Re-running the aggregator validated the manifest's frozen prepared/source
  hashes, all declared runs, paired labels, normalization fingerprint,
  runtime versions, and same-seed real-latent decoder reference.
- Each saved `synthetic_genotypes.npy` is `(251, 361)`, `int8`, and contains
  only 0/1/2. All ten reports are `complete` and every recorded runtime check
  is true.
- Each checkpoint has its requested seed and 1,000 steps. Standard configs
  and checkpoints have no feature map; every Fisher config **and checkpoint**
  equals `feature_schedule.npy.T` exactly, including its 0/1/2 band IDs.
- The full implementation suite independently reported 104 passed with one
  pre-existing CUDA warning; compile and diff checks exited 0. The CUDA
  environment has no `pytest`, so the experiment itself was validated through
  its own runtime checks and the independent post-run inspection above.
- Python LSP diagnostics were unavailable because no Python language server is
  installed; installation had previously been declined.

## Interpretation limits

The Binomial(2,p) decoder is the genotype likelihood and yields discrete
0/1/2 calls, but it assumes conditionally independent allele draws; it is not
an unrestricted three-category genotype model. Gaussian diffusion noise is
applied in normalized GLM-PCA factor space and is not the genotype likelihood.
The legacy Poisson path is a different count-model approximation and was not
used here.

The Fisher arm changes the forward schedule and its induced coordinatewise
Min-SNR weighting together; this experiment does not isolate a causal effect
of forward noising alone. CUDA RNG was seed-controlled, but deterministic
algorithms were not requested, so these runs are reproducible seed trials,
not claims of bitwise determinism.

AF and local dosage covariance are pooled over the reused validation cohort
(with matching validation-label composition), not stratified by population.
They cover 1,027 within-gene dosage pairs at separations 1, 2, and 4; they do
not demonstrate population-wise structure, haplotype LD, LD decay, rare-
variant performance, privacy, or generalization to a new cohort/split. No
significance test is reported from five seeds.
