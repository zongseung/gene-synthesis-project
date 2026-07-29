# Orthogonal Cumulant FiLM Design

## Objective

Extend HiPoDiT's population-conditioned FiLM with third- and fourth-order
distribution information computed from the existing 1000 Genomes training
split. The first implementation must:

- require no new cohort or external dataset;
- run preprocessing on CPU with NumPy;
- preserve the current FiLM path for a controlled ablation;
- start as the current model at initialization;
- keep classifier-free guidance's null population unconditional; and
- avoid full third- or fourth-order tensors over all 98,304 features.

## Scope

Version 1 adds one input-space **Orthogonal Cumulant FiLM (OC-FiLM)** module
before the existing CNN encoder. Existing CNN FiLM and DiT AdaLN-Zero blocks
remain unchanged. This isolates the effect of higher-order population
statistics and keeps old checkpoints compatible when the feature is disabled.

Version 1 does not:

- reconstruct SNPs from GLM-PCA factors;
- estimate full mixed cumulant tensors;
- add an external cohort;
- alter the diffusion loss; or
- replace the existing population and superpopulation embeddings.

## Statistical construction

Let the normalized, unclipped GLM-PCA features for one gene be
\(x \in \mathbb{R}^K\), with \(K=4\). Estimate a train-only global mean
\(\mu_g\) and eigendecomposition for every gene:

\[
\Sigma_g=V_g\Lambda_gV_g^\top.
\]

The clean feature is represented in orthogonal, standardized coordinates:

\[
u_0=(x-\mu_g)V_g\Lambda_g^{-1/2}.
\]

In the whitened coordinates, estimate bias-corrected population skewness
\(\kappa_{3,c,g,k}\) and excess kurtosis \(\kappa_{4,c,g,k}\). Population
estimates are partially pooled toward their superpopulation:

\[
\hat\kappa_{r,c}
= \lambda_{r,c}\kappa_{r,c}
+ (1-\lambda_{r,c})\kappa_{r,s(c)},\qquad
\lambda_{r,c} = \frac{n_c}{n_c+\tau_r}.
\]

The default prior effective sample sizes are \(\tau_3=50\) and
\(\tau_4=100\). Fourth-order estimates shrink more because their sampling
variance is larger. These values are artifact-building arguments and must be
reported with every experiment.

For a noisy diffusion input \(x_t\), the variance of eigen-coordinate \(k\)
is:

\[
\sigma_{t,g,k}^2
=\bar\alpha_t\lambda_{g,k}+(1-\bar\alpha_t).
\]

OC-FiLM standardizes the noisy coordinate using \(\sigma_t\), and the
remaining clean-signal fraction is:

\[
r_{t,g,k}
=\sqrt{\frac{\bar\alpha_t\lambda_{g,k}}{\sigma_{t,g,k}^2}}.
\]

It then applies the first-order Edgeworth-score correction:

\[
\Delta u_t =
\frac{r_t^3\hat\kappa_3}{2}H_2(u_t)
+\frac{r_t^4\hat\kappa_4}{6}H_3(u_t),
\]

where:

\[
H_2(u)=u^2-1,\qquad H_3(u)=u^3-3u.
\]

The powers follow the attenuation of standardized cumulants under additive
Gaussian diffusion noise. The correction is mapped back through
\(V_g\operatorname{diag}(\sigma_t)\):

\[
x'_t
=x_t+\rho\,
\left(\sigma_t\odot\Delta u_t\right)V_g^\top.
\]

\(\rho\) is a bounded, learnable per-input-channel gate initialized to zero.
Therefore the enabled model initially behaves exactly like the existing
HiPoDiT model, analogous to AdaLN-Zero.

## Numerical safeguards

- Cumulants are estimated in float64 and stored as float32.
- Covariance eigenvalues below `1e-6` are treated as zero-rank directions.
- Every timestep is re-standardized with
  `alpha_bar * eigenvalue + (1 - alpha_bar)`; clean-data whitening is never
  applied directly to a noisy input.
- Hermite inputs are clipped to `[-4, 4]`.
- Orthogonal-space corrections are clipped to `[-2, 2]`.
- The learned residual gain is bounded by `0.25 * tanh(raw_gain)`.
- Padded genes use identity whitening and zero cumulants.
- The CFG null population has zero cumulants and therefore receives no
  correction.

## Artifact

The CPU builder writes `data/processed/cumulant_stats.npz` with:

- `format_version`: scalar integer;
- `center`: `(gene_size, K)`;
- `whitener`: `(gene_size, K, K)`;
- `dewhitener`: `(gene_size, K, K)`;
- `rotation`: `(gene_size, K, K)` eigenvector matrices used by the model;
- `eigenvalues`: `(gene_size, K)` used for timestep re-standardization;
- `skewness`: `(n_pops + 1, gene_size, K)`;
- `excess_kurtosis`: `(n_pops + 1, gene_size, K)`;
- `pop_counts`: `(n_pops,)`;
- `skew_shrinkage`: `(n_pops,)`;
- `kurtosis_shrinkage`: `(n_pops,)`; and
- scalar metadata for sample, population, gene, component, and prior sizes.

The builder reads:

- `gene_pca_features.pkl` for unclipped feature values;
- `split_manifest.json` for train-only row indices;
- `label_hierarchy.pkl` for aligned population and superpopulation labels;
- `normalization_stats.pkl` to express statistics in the model's normalized
  coordinate system without applying the model's `[-5, 5]` clipping.

## Model integration

`HybridCNNDiTFiLM` constructs OC-FiLM only when
`model.cumulant_modulation.enabled=true`. The module loads and validates the
artifact during model construction, then runs immediately before the existing
hierarchical population embedding and CNN encoder.

The existing conditioning path remains:

```text
population label -> hierarchical embedding -> timestep embedding
                 -> existing CNN FiLM and DiT AdaLN-Zero
```

The added path is:

```text
noisy GLM-PCA input + population label + timestep
    -> covariance eigen-coordinates
    -> timestep-specific variance standardization
    -> population skewness/kurtosis Hermite correction
    -> inverse eigen-coordinate transform
    -> zero-init bounded residual
    -> existing HiPoDiT
```

When disabled, no artifact is loaded and model behavior/state keys remain
compatible with the baseline architecture.

## Configuration

The default OC-FiLM experiment records:

```yaml
model:
  cumulant_modulation:
    enabled: true
    stats_path: data/processed/cumulant_stats.npz
    initial_gain: 0.0
    max_gain: 0.25
    value_clip: 4.0
    correction_clip: 2.0
```

The baseline ablation is produced with:

```text
model.cumulant_modulation.enabled=false
```

## Validation

Unit tests must demonstrate:

- ZCA output has identity covariance on non-degenerate directions;
- bias-corrected skewness and kurtosis recover hand-checked distributions;
- shrinkage moves a population estimate toward its superpopulation estimate;
- saved artifacts have the documented shapes and a zero null population;
- zero-initialized OC-FiLM is an exact identity;
- a nonzero gate applies the expected \(H_2/H_3\) correction;
- the null CFG label is an identity;
- invalid artifact shapes fail before training; and
- a small end-to-end HiPoDiT forward pass preserves the input shape.

Operational verification must build the real artifact on CPU, report elapsed
time and peak memory, load it through the model module, and run the complete
root test suite.

## Scientific interpretation

FiLM's constant and linear terms correspond to first- and second-order
modulation. OC-FiLM adds the next Hermite terms motivated by third- and
fourth-order cumulants. The paper claim is therefore limited to:

> Population-conditioned affine modulation can be extended with a
> train-only, hierarchically shrunk, orthogonal cumulant correction whose
> effect vanishes consistently with the diffusion noise level.

Performance and novelty claims require the planned FiLM-versus-OC-FiLM
ablation and are not implied by the implementation alone.
