# HipoDIT rebuild: minimal defensible design

Architecture/statistical review, 2026-09-15. Read-only review of implementation; this report does not claim a completed rebuild or successful training. Exa returned 10 search results across FiLM, GLM-PCA and GeneticDiffusion; official DiT source and selected primary pages were fetched directly. Repeated versions of papers were treated as one source.

## Decision

Keep GLM-PCA and the existing class/time-conditioned CNN–DiT–CNN model. Correct AdaLN-Zero, reuse ordinary FiLM, remove OC-FILM, establish one preprocessing/normalization contract, and train a fresh small baseline. No new conditioning module or higher-moment objective is justified before that baseline works.

The minimal initial configuration is four DiT blocks, width 256, four heads, CNN base width 32–64, and the existing compatible channel/downsampling layout. These are conservative engineering defaults for 2,002 training individuals, not empirically selected optimal sizes. Keep K=4 provisionally; choose larger K from heldout reconstruction/deviance evidence. Retain the existing diffusion objective and start with auxiliary distribution/centroid losses disabled, so the first result has an interpretable cause. Compare higher capacity only after a valid baseline.

## Required correctness work

| Area | Required behavior | Evidence or implication |
|---|---|---|
| AdaLN-Zero | `h = (1 + gamma) * LN(x) + beta`; zero-initialize modulation output and residual gates. | [Official DiT implementation](https://github.com/facebookresearch/DiT/blob/main/models.py) uses this form. Existing `gamma * LN(x) + beta` gives zero-input attention at initialization; with zero attention biases and zero residual gate, neither branch nor gate can escape through gradient descent. The parent audit reports all 16 attention modulation/gate slices remain zero in raw and EMA epoch-455 weights. |
| FiLM | Reuse `UnifiedFiLMGenerator` and CNN modulation. | [Original FiLM paper](https://arxiv.org/abs/1709.07871) defines conditioning by featurewise affine transformations. Identity-offset gamma is a compatible parameterization; another FiLM module adds no needed capability. |
| Sample identity | Align VCF samples and panel labels by exact ID before splitting. | `run_pipeline.py` currently documents an overlap-prefix/order assumption. The parent audit verified exact ordering for present inputs; enforce that invariant rather than relying on matching row counts. |
| Train-only preprocessing | Fit filtering/imputation, GLM decoder, and normalization on training subjects; freeze them for validation/test. | `vcf_parser.py` currently estimates AF and imputation means over all observed subjects. Changing heldout rows must not change fitted training artifacts. |
| GLM scoring | Fit factors/loadings/intercepts in training; score new rows by the same likelihood with decoder fixed. | `glm_pca.py` currently uses nonlinear train factors and OLS heldout scores. `glm_pca_torch.py` uses OLS scores for all rows through GLM loadings. The latter avoids a split-specific formula but is still not GLM likelihood scoring. |
| Decoder persistence | Save loadings, intercepts, family/link, any offsets/trials, factor normalization, ordered variant IDs/alleles, and gene order. | Current returned feature dictionaries drop the decoder. Raw latent vectors cannot reconstruct dosage without it. |
| Gene order | Persist one numeric chromosome/position order for newly built features. | Alphabetic order is deterministic but does not support a genomic-locality rationale for the CNN. Cached alphabetical features can support a mechanics experiment if clearly identified; ordering changes require new metadata and fresh training. |

For family-specific scoring, freeze `eta_j = a_j + v_j^T z` (plus any explicitly modeled offsets) and minimize the negative log likelihood over `z`, with the same factor penalty used in fitting. For Poisson this is `sum(exp(eta) - y*eta) + lambda*||z||²/2`; for diploid binomial it is `sum(2*softplus(eta) - y*eta) + lambda*||z||²/2`. Use a stable finite objective and a safeguarded optimizer. Training and heldout scoring must share decoder, factor coordinate system, and regularization; an ordinary dosage-space least-squares projection does not meet this contract.

Poisson GLM-PCA remains GLM-PCA, but is a working approximation for bounded 0/1/2 dosages. For a fresh biologically motivated dosage pipeline, binomial with explicit two trials is the natural candidate; it does not establish Hardy–Weinberg validity or LD preservation by itself. Do not silently reinterpret `mult` as `Binomial(2,p)`: [official GLM-PCA documentation](https://search.r-project.org/CRAN/refmans/glmpca/html/glmpca.html) distinguishes likelihood families, observation sizes, intercepts and offsets. Preserve the actual backend semantics; cached Poisson data cannot be relabeled binomial.

## Normalization contract

Use one train-fitted per-coordinate transform `u = (z - mean_train) / scale_train`, with finite positive scales and a saved structural padding mask. Apply the same transform to real train/validation/test. The model generates `u`; either compare both real and generated samples in this space or inverse-transform both consistently. Never fit a new mean/std to synthetic samples or center each population separately for evaluation.

The existing preprocessing clips standardized values to ±5 and DDIM clips predicted clean samples to ±6. These are distinct truncations, not exact inverses. For a clean rebuild, prefer a fixed affine transform and record any needed sampling clamp explicitly; for cached data, retain its documented clipping history and measure clipped fractions. Inverse scaling cannot recover clipped tails. Save normalization and feature-order identity with the checkpoint and generated samples so mismatched artifacts fail visibly.

Structural padding and fitted constant dimensions must be recorded separately from biological constraints: a coordinate constant in the training sample is not proof it is impossible in the population. Mask only positions whose exclusion is part of the representation contract, and exclude them consistently from forward noise, loss and sampling.

## What can be concluded from the cached data

`data/processed/split_manifest.json` reports 2,002/251/251 train/validation/test subjects, with only 49–91 train subjects and 6–11 validation subjects per population. Cached tensors can test dimensions, finite losses, conditioning sensitivity, attention learning, sampler behavior and feature-distribution comparisons. They cannot validate newly corrected GLM likelihood projection, heldout dosage reconstruction, allele-frequency fidelity, LD, haplotypes or privacy without the corresponding artifacts and tests. Local VCF and RefGene files exist; this review did not establish their completeness or rerun preprocessing.

A freshly rebuilt 32-gene chromosome-17 pilot using all 2,504 subjects but only the 2,002 training subjects for fitting is an appropriate bounded first check. A 100-update/20-step-DDIM smoke run can show that this path executes and starts learning. It is neither a full-genome fit nor completed research, and cannot establish generalization or biological fidelity by its completion alone.

The reported approximately 95.1% `mean_explained` is not evidence of 95.1% information retention: current GLM code divides final deviance by deviance at random initialization, not by a fitted intercept-only null model. A real deviance-explained statistic needs an explicit null; heldout dosage reconstruction and likelihood are separate checks. Label the existing number as optimization deviance reduction, or recompute it.

## Minimal evidence before a longer run

1. At initialization each individual DiT block is identity, the attention gate receives a gradient on the first update, and attention parameters receive gradients after the gate opens. An initial zero attention-weight gradient is expected under AdaLN-Zero; requiring every parameter to update on step one is an incorrect test.
2. A fixed small batch can reduce denoising loss, class changes can affect output after updates, and conditioned/null sampling remains finite with exact padding behavior. Restart optimizer and EMA; do not resume the failed checkpoint as evidence of a correctly trained transformer.
3. GLM scoring decreases its chosen-family objective with fixed decoder, and changing heldout rows leaves train artifacts unchanged. Save/reload preserves factor scoring, ordering and normalization. Verify several heldout genes before whole-genome regeneration.
4. Evaluate real and synthetic samples in a common train-fitted feature space against a real-train versus real-validation reference. Track per-class mean/variance/coverage and train-versus-heldout nearest-neighbor distances. With so few per-class heldout subjects, uncertainty is large; bootstrap estimates and superpopulation summaries are more defensible than decisive fine-population rankings. Freeze test data until model selection ends.

## Closest prior and research hypotheses

[GeneticDiffusion](https://doi.org/10.1093/bioinformatics/btaf209), with [author code](https://github.com/TheMody/GeneDiffusion), is the close baseline: per-gene PCA compression, zero padding, conditional diffusion, and genomic evaluation. Its 1KG representation is phased haplotypes, whereas this pipeline uses diploid dosages; GLM-PCA is an explicit methodological difference. The authors report no classification-accuracy improvement from classifier-free guidance and sensitivity to reducing sampling steps. Therefore guidance above one and aggressive DDIM acceleration require validation here; paper results do not establish benefit for this model.

Hierarchical population embeddings, global attention improving inter-gene dependence, chromosome-aware ordering improving CNN locality, binomial versus Poisson GLM likelihood, and model depth are testable hypotheses. Keep the requested FiLM and GLM-PCA baseline; add complexity only after an ablation demonstrates a heldout benefit. Do not claim novelty from adding FiLM alone, genomic validity from PCA plots, or privacy from absence of exact duplicates.

## Implementation review follow-up

The landed AdaLN offset, encoder residual forwarding, smaller default, OC-FILM removal and epsilon-only validation match this design. Further review found a decoder schedule defect: the deepest block unnecessarily upsamples then crops, while the last block pads half its input because it does not upsample. The symmetric schedule is no upsample at the deepest block, then upsample at every remaining block. Sampling must enforce the structural mask at initialization and every reverse update, matching training inputs. These findings were sent to the implementation owner; this paragraph does not certify their final resolution.

This repository parameterizes guidance as `(1+w)*eps_cond - w*eps_null`; its unamplified conditional baseline is therefore `w=0`, corresponding to conventional CFG scale one.

The [accelerated backend source](https://github.com/zongseung/glmpca-fast/blob/main/src/glmpca.rs) returns raw factors/loadings/intercepts without a subsequent rotation or rescaling and uses no observation offset. Its factor score matches Poisson NLL plus `0.5*penalty*||z||²`. Its deviance history is evaluated before each iteration update, so its last stored value can differ from the returned fit's deviance at the iteration cap; recompute fit diagnostics from returned parameters.

A concrete projection check used `y=[[2]], V=[[10]], a=[-20], penalty=1`. Unprotected Newton with clipped linear predictors returned a penalized NLL of approximately `1.506e35`, versus `40` at zero factors. After the worker added Armijo backtracking on the true penalized objective, the same executed check returned `z=2.0584533`, NLL `2.743702`, and absolute score `1.36e-5` after float32 conversion. Boundary guards and whole-pipeline validation remain the implementation team's responsibility.

Final scoring decision: rescore training subjects as well as heldout subjects against the final fixed train-fitted decoder. At a short joint-fit iteration cap, the backend's stored training factors need not be optimal after its final loadings/intercept update. Use the same scorer and penalty for both groups, but separate calls for train and heldout: the current batched solver shares a line-search scale and stopping criterion, so mixing heldout rows into the training call can affect finite-iteration training results. Keep scoring iterations independent of the short decoder-fit cap. The reviewed scorer now rejects nonfinite penalty/tolerance, overflowing initial means and exhausted line searches; saved Poisson metadata includes the log link, penalty, backend/version and projection method.
