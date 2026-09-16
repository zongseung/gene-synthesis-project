# HiPoDiT-LD: Cohort-calibrated local-dependence decoding for latent-diffusion synthesis of unphased genotypes

**Manuscript core draft in English: Abstract, Introduction, and Methods only.**

**Proposed article category:** Original Article, *Bioinformatics*
**Proposed category:** Genetic and Population Analysis

## Abstract

### Motivation

Individual-level genotype data are indispensable for method development and
population-genetic analysis, but their distribution is constrained by privacy
and data-governance requirements. Synthetic data are useful only if they
preserve biologically relevant structure while avoiding memorization of
training records. Existing genotype generators have demonstrated the value of
deep generative modelling, including diffusion and haplotype-copying
approaches. However, the output layer is often treated as a secondary design
choice, despite the fact that an unphased biallelic diploid genotype has the
bounded support \(\{0,1,2\}\), population-specific allele frequencies, and
local dependence across ordered SNPs.

### Results

We define HiPoDiT-LD, a latent-diffusion framework with a
\(\operatorname{Binomial}(2,p)\) GLM-PCA representation and a
cohort-calibrated local-dependence decoder. Diffusion is performed only in
continuous, normalized GLM-PCA factor space. At the observed genotype layer,
the decoder combines a cohort-specific allele-frequency offset with a
physical-distance-binned \(3\times3\) categorical residual conditioned on the
previous dosage within each gene. This construction retains valid diploid
dosage support while isolating population calibration and short-range
dependence as separately testable components. The completed five-seed
diagnostic of the pre-existing generator showed no stable advantage for a
Fisher three-band noise schedule; we therefore fix the standard Gaussian
schedule as the control for the decoder study. The substantive results of
HiPoDiT-LD will be assessed by a preregistered oracle-decoder ablation,
end-to-end paired multi-seed comparison, and empirical privacy audit.

### Availability and implementation

The final submission will archive the source code, frozen split manifest,
variant metadata, exact environment, seeds, and evaluation outputs in a public
repository. The current manuscript draft is not a release claim.

## Introduction

The scale and clinical relevance of modern human genomic resources make
individual-level genotypes valuable for developing statistical methods,
benchmarking imputation and association pipelines, and studying population
structure. At the same time, a genotype record contains information about an
individual and, indirectly, genetically related people. Synthetic genomic data
are therefore attractive as a mechanism for controlled data access, but
synthetic release is not automatically private. Utility and privacy can move
in opposite directions: models that closely reproduce high-order structure can
also expose training members to inference attacks [1].

The 1000 Genomes Project established a global public reference for human
genetic variation and remains a standard resource for population-genetic
method development [2]. At a biallelic SNP, an unphased diploid genotype is
commonly represented by alternative-allele dosage
\(G_{ij}\in\{0,1,2\}\). This support is not a generic count space. In
particular, a Poisson likelihood permits dosages greater than two and thus does
not encode the sampling support of a diploid biallelic call. A
\(\operatorname{Binomial}(2,p)\) likelihood directly represents the two allele
trials and is the basis of logistic factor models for structured genotype data
[3]. It is not, however, a complete model of linkage disequilibrium (LD),
phase, or all departures from Hardy-Weinberg equilibrium.

Generative modelling of human genotypes is now an active field. Neural network
generators have been evaluated for genome-scale data [4,5], and recent work
has used diffusion models to synthesize complete human genotypes through
low-dimensional genomic representations [6]. Haplotype-copying approaches,
including HAPNEST, provide a complementary route to large-scale genotype and
phenotype simulation [7]. Discrete denoising diffusion models also establish
that diffusion can be defined directly over finite state spaces [8]. These
studies make two consequences clear. First, using a diffusion model, a
three-category genotype output, or an ancestry label alone is not a sufficient
methodological contribution. Second, any new method must be evaluated against
appropriate baselines using real biological data, data-generating constraints,
and explicit privacy tests.

HiPoDiT was developed as a two-stage generator in which genotype matrices are
compressed into a low-dimensional latent representation and a conditional
diffusion model generates new latent factors. This decomposition is
computationally useful, but it creates three distinct possible sources of
error: representation loss in GLM-PCA, distributional error in the latent
generator, and calibration error in the genotype decoder. An independent
per-SNP Binomial decoder can produce valid \(\{0,1,2\}\) calls, but it
factorizes the genotype distribution conditional on the latent factor. Thus,
even if marginal allele frequencies are well calibrated, it need not preserve
local dosage covariance.

We address this output-layer limitation with HiPoDiT-LD. The method retains a
Binomial base distribution and adds only two regularized terms: a
cohort-specific SNP offset and a categorical residual based on the preceding
within-gene SNP and their physical separation. The intent is not to construct
a new generic diffusion process or a phased haplotype simulator. Rather, the
method tests the focused hypothesis that a cohort-calibrated,
distance-aware residual improves local unphased LD fidelity after latent
generation without degrading allele-frequency calibration or empirical privacy
relative to an independent Binomial decoder.

The study has four prespecified methodological principles. First, the
observed-data likelihood is Binomial rather than Poisson. Second, the latent
generator and the decoder are evaluated separately through an oracle-decoder
experiment before end-to-end sampling. Third, any local-dependence conclusion
must fail under a shuffled genomic-order control and any cohort-calibration
conclusion must fail under a shuffled cohort-label control. Fourth, utility
claims require paired repeated-seed confidence intervals and must be reported
alongside nearest-neighbor and membership-inference risk. These restrictions
are intended to make a modest output-layer contribution falsifiable rather
than to overstate its generality.

## Materials and methods

### Study population, genotype panel, and split

The initial study uses the public 1000 Genomes Project Phase 3 reference
panel. The frozen diagnostic panel contains \(N=2{,}504\) individuals, 26
population labels nested in five superpopulations, \(P=361\) unique SNPs, and
eight genes on chromosome 17. All selected dosage calls are finite and lie in
\(\{0,1,2\}\). Individuals are divided once by population-stratified sampling
into training (\(n_{\mathrm{tr}}=2{,}002\)), development
(\(n_{\mathrm{dev}}=251\)), and test (\(n_{\mathrm{te}}=251\)) subsets.

For every SNP, the analysis manifest retains the variant identifier, reference
and alternative alleles, reference-genome coordinate, gene assignment, and
input ordering. Before decoder fitting, SNPs are sorted in ascending coordinate
order within each gene. The split indices, filtering rules, factor rank, random
seeds, and software versions are frozen before the test set is opened. The
test set is never used to estimate allele frequencies, physical-distance bin
boundaries, normalization constants, or hyperparameters.

Let \(G\in\{0,1,2\}^{N\times P}\) be the dosage matrix and
\(M\in\{0,1\}^{N\times P}\) its observation mask, with \(M_{ij}=1\) if the
call is observed. All genotype likelihoods and reconstruction metrics are
computed only for entries with \(M_{ij}=1\). No imputed fractional dosage is
treated as a realized Binomial outcome.

### Binomial GLM-PCA

For sample \(i\), SNP \(j\), latent score \(z_i\in\mathbb R^K\), SNP intercept
\(\alpha_j\), and SNP loading \(v_j\in\mathbb R^K\), the Binomial GLM-PCA
model is

\[
\eta_{ij}=\alpha_j+v_j^{\mathsf T}z_i,\qquad
p_{ij}=\sigma(\eta_{ij}),\qquad
G_{ij}\mid z_i\sim \operatorname{Binomial}(2,p_{ij}),
\tag{1}
\]

where \(\sigma(a)=(1+\exp(-a))^{-1}\). Conditional on \(z_i\), the
log-likelihood contribution of an observed call is

\[
\log p(G_{ij}\mid z_i)=
\log {2\choose G_{ij}}+
G_{ij}\log p_{ij}+
(2-G_{ij})\log(1-p_{ij}).
\tag{2}
\]

The representation parameters are fitted on the training subset by maximizing
the masked likelihood with ridge penalties:

\[
\mathcal L_{\mathrm{GLM}}=
-\sum_{i,j:M_{ij}=1}
\log p(G_{ij}\mid z_i)
+\lambda_v\lVert V\rVert_F^2+
\lambda_z\sum_i\lVert z_i\rVert_2^2.
\tag{3}
\]

The fitted training factors are standardized coordinatewise:

\[
x_i=D^{-1}(z_i-\mu),\qquad
D=\operatorname{diag}(s_1,\ldots,s_K),
\tag{4}
\]

where \(\mu\) and \(s_k>0\) are estimated from training factors only. The
inverse map, \(z_i=\mu+Dx_i\), is applied before genotype decoding. The
parameters \((\alpha,V,\mu,D)\) are saved as part of the representation
artifact; they are never refitted on generated samples or test observations.

### Cohort-conditioned latent diffusion

The continuous normalized factor \(x_0\) is generated by a Gaussian diffusion
model conditional on a cohort label \(c\). For diffusion timestep
\(t\in\{1,\ldots,T\}\), define a fixed variance schedule
\(\beta_t\in(0,1)\), \(\alpha_t=1-\beta_t\), and
\(\bar\alpha_t=\prod_{s=1}^{t}\alpha_s\). The forward process is

\[
q(x_t\mid x_0)=
\mathcal N\!\left(
\sqrt{\bar\alpha_t}\,x_0,\,
(1-\bar\alpha_t)I
\right),
\qquad
x_t=\sqrt{\bar\alpha_t}\,x_0+
\sqrt{1-\bar\alpha_t}\,\epsilon,
\tag{5}
\]

where \(\epsilon\sim\mathcal N(0,I)\). A denoiser
\(\epsilon_\phi(x_t,t,c)\) is trained by the standard noise-prediction
objective

\[
\mathcal L_{\mathrm{diff}}(\phi)=
\mathbb E_{x_0,c,t,\epsilon}
\left[
\left\lVert
\epsilon-\epsilon_\phi(x_t,t,c)
\right\rVert_2^2
\right].
\tag{6}
\]

The reverse model is parameterized as

\[
p_\phi(x_{t-1}\mid x_t,c)=
\mathcal N\!\left(
\mu_\phi(x_t,t,c),\,
\Sigma_t
\right),
\tag{7}
\]

where \(\mu_\phi\) is calculated from the predicted noise and the fixed
schedule. Sampling starts from \(x_T\sim\mathcal N(0,I)\), iterates Equation
(7), and maps the resulting \(x_0\) through Equation (4) to obtain
\(\tilde z\). The primary decoder study fixes the existing standard schedule.
A Fisher-band schedule is retained only as a post hoc ablation because the
completed five-seed diagnostic did not show a consistent improvement over the
standard schedule.

Diffusion and decoder fitting are intentionally separate. Equation (6) is
optimized in normalized latent space, while the decoder is fitted after
inverse normalization in GLM-PCA factor space. This prevents a decoder loss
from changing the latent geometry during the first study and makes the source
of an observed improvement identifiable.

### Independent Binomial decoder

The independent decoder B0 is the reference output model:

\[
\eta^{(0)}_{ij}=\alpha_j+v_j^{\mathsf T}z_i,\qquad
q_0(G_{ij}=g\mid z_i)=
{2\choose g}
\left[p^{(0)}_{ij}\right]^g
\left[1-p^{(0)}_{ij}\right]^{2-g},
\quad g\in\{0,1,2\}.
\tag{8}
\]

Equation (8) yields valid dosage values by construction but assumes conditional
independence across SNPs given \(z_i\). It remains a mandatory control in all
experiments.

### HiPoDiT-LD decoder

HiPoDiT-LD augments the independent decoder with a cohort-specific offset and
a local categorical residual. Let \(\mathcal C\) denote the set of training
cohorts and \(u_{cj}\) the offset for cohort \(c\) at SNP \(j\). The
cohort-calibrated base probability is

\[
\eta_{ij}=\alpha_j+u_{c_i j}+v_j^{\mathsf T}z_i,\qquad
p_{ij}=\sigma(\eta_{ij}).
\tag{9}
\]

To retain an identifiable global intercept, offsets are centered using
training-cohort counts \(n_c\):

\[
\sum_{c\in\mathcal C}n_cu_{cj}=0
\qquad\text{for every }j.
\tag{10}
\]

For an ordered within-gene SNP \(j\) that is not the first SNP of its gene,
let \(r=G_{i,j-1}\) be the preceding dosage and
\(b_j=b(d_{j-1,j})\) be the bin assigned to the physical distance between SNPs
\(j-1\) and \(j\). The bin boundaries are estimated once from training-set
within-gene adjacent-SNP distances and then frozen. For \(g\in\{0,1,2\}\),
define

\[
\ell_{ijg}=
\log\!\left[
{2\choose g}p_{ij}^{g}(1-p_{ij})^{2-g}
\right]
+A_{b_j,r,g},
\tag{11}
\]

where \(A_b\in\mathbb R^{3\times3}\) is a distance-bin-specific residual
table. The decoder distribution is the categorical normalization

\[
q_\theta(G_{ij}=g\mid z_i,c_i,G_{i,j-1})=
\frac{\exp(\ell_{ijg})}
{\sum_{h=0}^{2}\exp(\ell_{ijh})}.
\tag{12}
\]

For the first SNP in each gene, \(A_{b_j,r,g}=0\), so Equation (12) reduces to
the Binomial base distribution. The chain is reset at each gene boundary. This
decoder does not model cross-gene dependence, future SNPs, or phased
haplotypes.

The decoder is trained with teacher forcing, meaning that the observed
\(G_{i,j-1}\) is supplied while fitting. Its masked penalized negative
log-likelihood is

\[
\mathcal L_{\mathrm{dec}}(\theta)=
-\sum_{i,j:M_{ij}=1}
\log q_\theta(G_{ij}\mid z_i,c_i,G_{i,j-1})
+\lambda_u\lVert U\rVert_F^2+
\lambda_A\sum_{b=1}^{B}\lVert A_b\rVert_F^2.
\tag{13}
\]

The ridge penalties shrink unstable parameters in small cohorts and sparse
distance bins toward the independent Binomial decoder. At generation time,
\(\tilde G_{i,j-1}\), rather than the real predecessor, is used in Equation
(12), and each \(\tilde G_{ij}\) is sampled from its three-category
distribution. Thus every final output is an integer in \(\{0,1,2\}\).

### Ablation design, falsification controls, and decision rule

The decoder study compares four nested models:

| Arm | \(u_{cj}\) | \(A_b\) | Interpretation |
| --- | :---: | :---: | --- |
| B0 | No | No | Independent Binomial reference |
| B1 | Yes | No | Cohort-frequency calibration only |
| B2 | No | Yes | Local-dependence residual only |
| B3 | Yes | Yes | HiPoDiT-LD |

The first experiment is an oracle-decoder evaluation. Real held-out GLM-PCA
factors are passed to B0--B3, thereby testing the decoder without generated
latent error. Two falsification controls are mandatory: a within-gene genomic
order shuffle for the local residual and a cohort-label shuffle for the cohort
offset. A mechanism-specific interpretation is allowed only if the relevant
advantage disappears under its corresponding shuffle.

Hyperparameters are selected on the development subset. B3 proceeds to
end-to-end sampling only when it improves both per-call decoder negative
log-likelihood and distance-stratified dosage \(r^2\) MAE against B0 on the
development data, while its cohort-level AF MAE does not exceed 105% of B0.
The final test evaluates B0 and B3 using at least ten paired random seeds. The
prespecified primary evidence for the method is a negative paired difference
for both test negative log-likelihood and LD \(r^2\) MAE, with 95% bootstrap
confidence-interval upper bounds below zero and no failure of the AF
calibration guardrail.

### Evaluation and privacy-risk protocol

Allele frequency at SNP \(j\) in a set \(S\) is

\[
\operatorname{AF}_{j}(S)=
\frac{1}{2|S|}
\sum_{i\in S}G_{ij},
\tag{14}
\]

and overall allele-frequency error is

\[
\operatorname{AFMAE}=
\frac{1}{P}\sum_{j=1}^{P}
\left|
\operatorname{AF}_{j}(S_{\mathrm{syn}})-
\operatorname{AF}_{j}(S_{\mathrm{real}})
\right|.
\tag{15}
\]

Local dependence is evaluated within physical-distance bins using both signed
dosage covariance and squared Pearson correlation \(r^2\). Because the inputs
are unphased dosages, these are explicitly reported as dosage-based local-LD
proxies, not as gametic or haplotype LD [9]. Secondary measures include
genotype-proportion total variation, heterozygosity error, cohort-stratified
AF MAE, and real-versus-synthetic population-classifier performance.

Privacy is evaluated empirically with exact and near-duplicate rates,
nearest-neighbor distances to training and held-out individuals, and
membership-inference attacks. Each privacy metric is reported beside, rather
than traded silently against, utility. No result from this protocol is
interpreted as a formal differential-privacy guarantee.

## References cited in this draft

1. Oprisanu B, Ganev G, De Cristofaro E. On utility and privacy in synthetic
   genomic data. *Proceedings of the Network and Distributed System Security
   Symposium*. 2022. doi:10.14722/ndss.2022.24092.
2. Auton A, Brooks LD, Durbin RM, et al.; 1000 Genomes Project Consortium. A
   global reference for human genetic variation. *Nature*. 2015;526:68--74.
   doi:10.1038/nature15393.
3. Hao W, Song M, Storey JD. Probabilistic models of genetic variation in
   structured populations applied to global human studies. *PLoS Genetics*.
   2016. https://pmc.ncbi.nlm.nih.gov/articles/PMC4795615/
4. Yelmen B, Decelle A, Ongaro L, et al. Creating artificial human genomes
   using generative neural networks. *PLOS Genetics*. 2021;17:e1009303.
5. Yelmen B, Decelle A, Boulos LL, et al. Deep convolutional and conditional
   neural networks for large-scale genomic data generation. *PLOS
   Computational Biology*. 2023;19:e1011584.
6. Kenneweg P, Dandinasivara R, Luo X, Hammer B, Schönhuth A. Generating
   synthetic genotypes using diffusion models. *Bioinformatics*.
   2025;41(Suppl 1):i484--i492. doi:10.1093/bioinformatics/btaf209.
7. Wharrie S, Yang Z, Raj V, et al. HAPNEST: efficient, large-scale generation
   and evaluation of synthetic datasets for genotypes and phenotypes.
   *Bioinformatics*. 2023;39:btad535. doi:10.1093/bioinformatics/btad535.
8. Austin J, Johnson DD, Ho J, Tarlow D, van den Berg R. Structured denoising
   diffusion models in discrete state-spaces. *NeurIPS*. 2021.
9. Rogers AR, Huff C. Linkage disequilibrium between loci with unknown phase.
   *Genetics*. 2009;182:839--844. doi:10.1534/genetics.108.093153.
