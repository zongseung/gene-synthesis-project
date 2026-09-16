# HiPoDiT-LD: Cohort-calibrated local dependence decoding for latent-diffusion synthesis of unphased genotypes

> **Internal manuscript draft, 15 September 2026. Not ready for submission.**
>
> This draft reports only results that have actually been completed. The proposed
> HiPoDiT-LD decoder has not yet undergone the prespecified oracle, end-to-end,
> or multi-seed confirmation experiments. Sections explicitly labelled
> **Pending confirmation** must be completed with frozen test-set results before
> this document is converted into a submission manuscript.

**Authors:** [To be supplied after contribution review]

**Affiliations:** [To be supplied]

**Corresponding author:** [To be supplied]

**Article type:** Original research article
**Keywords:** synthetic genotype data; population structure; binomial latent
factor model; diffusion model; linkage disequilibrium; genomic privacy

## Abstract

### Background

Sharing individual-level human genotype data is constrained by privacy and
governance requirements. A useful synthetic genotype generator must preserve
allele frequencies and population structure without simply reproducing
training individuals. For biallelic unphased genotypes, the observed dosage is
bounded to \(\{0,1,2\}\), whereas many latent generative pipelines use a
continuous reconstruction model or an independent per-site decoder. The latter
does not directly represent local dependence between adjacent variants.

### Methods

We formulate HiPoDiT-LD, a proposed extension of a GLM-PCA latent-diffusion
pipeline for unphased genotype dosage. The representation model uses a
\(\operatorname{Binomial}(2,p)\) likelihood. A standard Gaussian diffusion
model generates normalized GLM-PCA factors conditional on cohort. At decoding,
a cohort-specific SNP offset calibrates the Binomial base probability and a
distance-binned \(3\times3\) categorical residual conditions each within-gene
dosage on the preceding dosage in genomic order. The residual decoder is
trained by teacher-forced negative log-likelihood and sampled sequentially
within each gene. We prespecify oracle-decoder, end-to-end, paired multi-seed,
and privacy analyses before evaluating the new decoder on the held-out test
set.

### Results

The completed diagnostic used 2,504 individuals, 361 SNPs in eight chromosome
17 genes, and a population-stratified 2,002/251/251 train/development/test
split. Every observed call was in \(\{0,1,2\}\). In a five-seed end-to-end
diagnostic of the existing latent generator, a Fisher three-band noise schedule
did not consistently improve over the standard schedule: allele-frequency MAE
was \(0.02410\pm0.00165\) versus \(0.02431\pm0.00256\), and local dosage
covariance MAE was \(0.02165\pm0.00160\) versus
\(0.02151\pm0.00215\), respectively. These results support retaining the
standard schedule as the control; they do not establish superiority of either
schedule and do not evaluate HiPoDiT-LD.

### Conclusions

HiPoDiT-LD is a bounded-dose, cohort-calibrated local-dependence decoder whose
scientific value depends on a precommitted comparison with an independent
Binomial decoder. A publishable claim requires improvement in held-out
likelihood and distance-stratified local LD without deterioration in
cohort-level frequency calibration, diversity, or empirical privacy risk.

## Introduction

Human genomic data are unusually sensitive, yet individual-level genotypes are
valuable for method development, imputation, association analysis, and
population-genetic studies. Synthetic data can improve access, but resemblance
to source distributions is not itself a privacy guarantee. In particular,
membership-inference experiments on synthetic genomic data have shown that
utility and privacy need to be evaluated jointly rather than inferred from
visual similarity or lack of exact duplicates [1].

The 1000 Genomes Project provides a widely used reference panel with global
population structure [2]. At a biallelic SNP, an unphased diploid genotype is
naturally encoded as alternative-allele dosage \(G_{ij}\in\{0,1,2\}\). A
Binomial observation model is therefore better aligned with the support of the
data than an unbounded Poisson reconstruction model. Logistic factor analysis
provides a relevant likelihood-based representation of population structure
for such genotype data [3].

Diffusion models have been extended to discrete state spaces [4] and have
recently been applied to synthetic human genotype generation [5]. Haplotype
copying approaches such as HAPNEST offer another strong reference for
genotype-and-phenotype simulation [6]. These precedents rule out a claim that
diffusion, discrete genotype output, or local genomic structure is new in
itself. The narrow question addressed here is whether a compact decoder can
separate two error sources in an existing latent diffusion pipeline:
population-specific allele-frequency calibration and short-range dosage
dependence after dimension reduction.

We therefore propose HiPoDiT-LD: a cohort-calibrated Binomial decoder augmented
by a low-capacity, physical-distance-binned categorical residual. The method
is intentionally restricted to unphased dosage and within-gene first-order
dependence. It does not claim phased haplotype reconstruction, long-range
linkage disequilibrium (LD), rare-variant fidelity, whole-genome generation,
or formal differential privacy. The primary hypothesis is that, conditional
on a GLM-PCA latent factor and cohort label, this residual decoder can improve
held-out local LD while preserving allele-frequency calibration relative to an
independent Binomial decoder.

## Materials and methods

### Study design and data partition

The current diagnostic dataset contains 2,504 individuals from the 1000
Genomes Project Phase 3 reference panel, with labels for 26 populations nested
within five superpopulations [2]. The frozen analysis panel contains 361 unique
SNPs in eight chromosome 17 genes. Samples were divided once, with population
stratification, into training (\(n=2{,}002\)), development (\(n=251\)), and
test (\(n=251\)) subsets.

All model-selection decisions, including factor rank, regularization,
physical-distance bins, and sampling settings, must use training and
development subsets only. The test subset is reserved for the final paired
multi-seed comparison. Variant identifier, reference/alternative-allele
orientation, genomic coordinate, gene boundary, sample split, and random seed
are part of the frozen analysis manifest. Decoder training never uses a
test-derived cohort frequency, distance-bin boundary, or latent statistic.

### Genotype representation and notation

Let \(G_{ij}\) denote the alternative-allele dosage for individual \(i\) at SNP
\(j\), let \(c_i\) be the assigned population cohort, and let \(z_i\in
\mathbb R^K\) be its GLM-PCA factor. Missing calls are represented by an
explicit mask and are excluded from likelihood sums; they are not interpreted
as fractional Binomial outcomes. SNPs are sorted by reference coordinate
within each gene. The decoder is reset at a gene boundary.

### Binomial GLM-PCA representation

For each observed genotype, the GLM-PCA observation model is

\[
\eta_{ij}=\alpha_j+v_j^\top z_i,\qquad
p_{ij}=\operatorname{sigmoid}(\eta_{ij}),\qquad
G_{ij}\sim\operatorname{Binomial}(2,p_{ij}).
\]

Here \(\alpha_j\) is an SNP intercept and \(v_j\) is a loading vector. This
model retains the \(0\), \(1\), \(2\) support of an unphased diploid genotype
while providing a continuous latent representation suitable for a Gaussian
diffusion process. It is a modelling choice, not a claim that all genotype
dependence is explained by Hardy-Weinberg equilibrium or by the latent factors.

The factor score is normalized using training-only location and scale
parameters before entering the generator. The same saved parameters are
inverted before decoding generated factors. This boundary is checked because a
scale mismatch can otherwise be mistaken for a decoder error.

### Cohort-conditioned latent diffusion

Let \(x_i\) be the normalized factor representation. A conditional Gaussian
diffusion model learns to denoise \(x_i\) given timestep \(t\) and cohort
\(c_i\). The initial HiPoDiT-LD evaluation fixes the existing standard noise
schedule and generator architecture. A Fisher three-band schedule is not part
of the proposed method because the completed repeated-seed diagnostic did not
show a stable advantage (Results).

Diffusion occurs in continuous GLM-PCA factor space only. It is not a
Poisson, Binomial, or categorical diffusion over the observed genotypes. This
separation makes the contribution of the final decoder experimentally
identifiable.

### Proposed cohort-calibrated local-dependence decoder

The independent Binomial decoder, denoted B0, uses

\[
\eta^{(0)}_{ij}=\alpha_j+v_j^\top z_i,\qquad
q_0(G_{ij}=g\mid z_i)=\operatorname{Binomial}(g;2,p^{(0)}_{ij}).
\]

HiPoDiT-LD adds a cohort offset \(u_{c_i,j}\) to the base logit:

\[
\eta_{ij}=\alpha_j+u_{c_i,j}+v_j^\top z_i,\qquad
p_{ij}=\operatorname{sigmoid}(\eta_{ij}).
\]

For an SNP other than the first SNP of its gene, the final conditional
probability is

\[
q_\theta(G_{ij}=g\mid z_i,c_i,G_{i,j-1})=
\frac{\operatorname{Binomial}(g;2,p_{ij})\,
\exp\{A_{b(d_{j-1,j}),G_{i,j-1},g}\}}
{\sum_{h=0}^{2}\operatorname{Binomial}(h;2,p_{ij})\,
\exp\{A_{b(d_{j-1,j}),G_{i,j-1},h}\}},
\quad g\in\{0,1,2\}.
\]

In this expression, \(d_{j-1,j}\) is the physical distance between adjacent
SNPs and \(b(\cdot)\) maps that distance to a training-derived bin. Each
\(A_b\) is a \(3\times3\) table. The first SNP of a gene uses the Binomial base
distribution alone. The formula is deliberately limited: it does not condition
on future SNPs, other genes, inferred haplotypes, or a learned long-context
attention mechanism.

The cohort offsets satisfy a training-count-weighted centering constraint,
\(\sum_c n_cu_{cj}=0\) for every \(j\), so that \(\alpha_j\) remains the global
intercept. Offset and residual parameters use ridge shrinkage:

\[
\mathcal L(\theta)=
-\sum_{(i,j)\in\mathcal O}
\log q_\theta(G_{ij}\mid z_i,c_i,G_{i,j-1})
\lambda_u\lVert U\rVert_F^2+
\lambda_A\sum_b\lVert A_b\rVert_F^2,
\]

where \(\mathcal O\) contains observed calls. Training uses the real preceding
dosage (teacher forcing); synthetic sampling proceeds left to right in genomic
order within each gene.

### Decoder ablation and falsification tests

Four arms isolate the proposed terms:

| Arm | Cohort offset \(u_{cj}\) | Local residual \(A_b\) | Purpose |
| --- | :---: | :---: | --- |
| B0 | No | No | Independent Binomial control |
| B1 | Yes | No | Frequency-calibration effect |
| B2 | No | Yes | Local-dependence effect |
| B3 | Yes | Yes | Full proposed decoder |

The initial test is an **oracle decoder** experiment: real held-out GLM-PCA
factors are supplied to B0--B3. This identifies decoder effects without
diffusion error. Two negative controls are required. First, shuffling
within-gene SNP order should remove an authentic genomic-order local-dependence
advantage. Second, shuffling cohort labels should remove an authentic
cohort-calibration advantage. A result that survives either inappropriate
control is not interpreted as evidence for the corresponding biological
mechanism.

Only a decoder that improves both development-set per-call negative
log-likelihood and distance-stratified LD \(r^2\) MAE relative to B0, without
increasing cohort allele-frequency MAE by more than 5%, proceeds to
end-to-end generation. The final model selection is frozen before opening the
test set.

### Endpoints and statistical analysis

The primary endpoints are:

1. Per-observed-call negative log-likelihood in the oracle decoder experiment.
2. Allele-frequency mean absolute error (AF MAE), overall and within cohort.
3. Distance-stratified MAE of dosage-based \(r^2\) and signed covariance for
   within-gene SNP pairs.

Secondary endpoints are genotype-proportion total variation, heterozygosity
error, Hardy-Weinberg departure, population differentiation, visual latent
structure, real-versus-synthetic population-classifier performance, and
runtime. Dosage-based LD must be described as a proxy for local unphased
dependence; it is not phased gametic LD [7].

The final comparison uses at least ten paired random seeds for B0 and B3,
with the same split, generator checkpoint policy, synthetic sample count,
sampling steps, and seed in each pair. For each endpoint, we report the
paired mean difference and a 95% bootstrap confidence interval. The
prespecified success rule is that B3 must have a negative paired difference
for both test negative log-likelihood and LD \(r^2\) MAE, with a 95%
confidence-interval upper limit below zero, while overall and cohort AF MAE
remain at or below 105% of B0.

### Privacy-risk assessment

We will report exact and near-duplicate rates, nearest-neighbor distances from
synthetic samples to training and held-out real samples, and membership
inference using a classifier-based and a distance-based attack. These are
empirical risk measures, not a differential-privacy guarantee. This dual
utility-and-privacy evaluation follows prior synthetic-genomics work showing
that neither fidelity nor absence of direct copies establishes privacy [1].

## Results

### Completed diagnostic dataset audit

The frozen panel included 2,504 individuals, 361 mapped and unique SNPs, and
eight chromosome 17 genes. All 2,504 by 361 genotype calls used in the
diagnostic were finite and contained only the three permitted dosage values
\(\{0,1,2\}\). The analysis used four factors per gene and a
\(\operatorname{Binomial}(2,p)\) decoder fitted on training rows. These checks
justify the bounded-dose representation but do not by themselves validate the
proposed decoder.

### Completed repeated-seed diagnostic: Fisher schedule versus standard schedule

Ten end-to-end generator runs were completed: standard and Fisher three-band
schedule arms for each of five paired seeds (20260327--20260331). Each run used
1,000 optimizer updates, batch size 64, 251 synthetic samples, and 100 DDIM
steps. Lower values are better in Table 1. The delta is Fisher minus standard.

**Table 1. Completed five-seed diagnostic of the existing latent generator.**

| Seed | Standard AF MAE | Fisher AF MAE | Delta AF MAE | Standard local covariance MAE | Fisher local covariance MAE | Delta covariance MAE |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20260327 | 0.0264593 | 0.0211729 | -0.0052863 | 0.0233071 | 0.0201099 | -0.0031972 |
| 20260328 | 0.0230601 | 0.0227897 | -0.0002704 | 0.0207906 | 0.0205989 | -0.0001917 |
| 20260329 | 0.0252177 | 0.0277781 | 0.0025604 | 0.0234604 | 0.0252899 | 0.0018295 |
| 20260330 | 0.0229773 | 0.0256922 | 0.0027149 | 0.0205956 | 0.0211756 | 0.0005801 |
| 20260331 | 0.0227842 | 0.0241196 | 0.0013354 | 0.0201157 | 0.0203728 | 0.0002570 |
| Mean \(\pm\) sample SD | 0.0240997 \(\pm\) 0.0016499 | 0.0243105 \(\pm\) 0.0025557 | 0.0002108 \(\pm\) 0.0032979 | 0.0216539 \(\pm\) 0.0015990 | 0.0215094 \(\pm\) 0.0021496 | -0.0001445 \(\pm\) 0.0018644 |

Fisher achieved a lower AF MAE in two of five seeds and a lower covariance MAE
in two of five seeds. Its mean AF MAE was higher, while the mean covariance
MAE difference was small relative to its seed-to-seed variation. This
diagnostic contains no statistical significance analysis and does not support
a superiority claim for Fisher scheduling. The standard schedule is therefore
the fixed control for the decoder study.

### Pending confirmation: decoder and end-to-end results

No B0--B3 oracle decoder result, end-to-end decoder result, privacy-attack
result, or ten-seed confirmatory test result has been generated at the time of
this draft. Those values must not be inferred from Table 1, which compares
noise schedules rather than decoders.

The following results must be added after the experimental gates in the
accompanying implementation plan are passed:

| Required analysis | Required display | Condition for an affirmative claim |
| --- | --- | --- |
| Oracle B0--B3 | NLL, AF MAE, signed covariance, and \(r^2\) MAE by distance bin | B3 passes the development-set gate and both negative controls behave as expected |
| End-to-end B0 versus B3 | Paired seed scatter and cohort-stratified metrics | Oracle benefit persists after generated latents are decoded |
| Confirmatory test | Ten or more paired seeds with 95% CIs | Prespecified primary success rule is met |
| Privacy audit | Nearest-neighbor, duplicate, and membership-inference results | No material privacy regression against B0 |
| Generalization, if available | A separately harmonized external panel | External result is reported without retuning |

## Discussion

This study is designed around an elementary but consequential distinction:
the support of the observed data and the space in which generation occurs need
not be identical. The HiPoDiT generator operates on continuous GLM-PCA factors
for computational tractability, while the decoder enforces a bounded,
diploid-dose likelihood at the output. A Poisson decoder is not used because
it allocates probability to values outside \(\{0,1,2\}\). Conversely, using a
Binomial base distribution alone does not preserve all local dependence; it
models dosage through a conditional allele frequency and leaves residual LD
unexplained.

HiPoDiT-LD tests whether two low-capacity corrections address this residual
error. The cohort offset is intended to calibrate mean dosage by population.
The distance-binned table is intended to capture only the dependence of an
SNP on its immediately preceding within-gene dosage. Both are tightly
regularized, fitted from training data only, and independently ablated. This
deliberate restriction reduces the risk that a large decoder obscures whether
an apparent improvement originated in compression, diffusion, or output
calibration.

The completed five-seed diagnostic provides an important negative result for
model selection. A Fisher-band schedule did not yield a stable improvement in
AF MAE or local covariance MAE. Retaining the standard schedule keeps the
decoder comparison interpretable. It also avoids converting an inconsistent
training heuristic into a paper-level novelty claim.

The contribution should be positioned cautiously. Prior work has already
demonstrated whole-genotype diffusion [5], discrete diffusion [4], and
haplotype-based simulation [6]. Accordingly, a successful paper may claim
evidence that a cohort-calibrated local residual decoder improves a particular
unphased, limited-panel latent-diffusion setting under the stated controls. It
may not claim the first synthetic-genotype diffusion model, universal LD
preservation, phased haplotype generation, whole-genome synthesis, or formal
privacy protection.

Several limitations are structural. First, the current panel covers eight
genes on chromosome 17 and cannot support a whole-genome claim. Second,
unphased dosage cannot determine gametic phase, recombination structure, or
haplotype LD. Third, a training-only MAF filter limits conclusions about rare
variants. Fourth, the current data provide population labels rather than
disease or phenotype labels; disease prediction, GWAS utility, and
phenotype-conditioned synthesis require a separately governed cohort and
prespecified endpoints. Finally, membership-inference and nearest-neighbor
audits quantify empirical risk for the evaluated release protocol but do not
replace formal privacy accounting.

## Conclusions

We specify HiPoDiT-LD as a Binomial GLM-PCA latent-diffusion framework with a
cohort-calibrated, distance-binned local categorical decoder. The completed
diagnostic supports a standard-schedule control and rejects Fisher scheduling
as a current novelty candidate. Whether HiPoDiT-LD is a publishable method
remains conditional on the frozen decoder ablation, paired multi-seed test,
and privacy evaluation. This draft should be updated only with those measured
results, not with extrapolated claims.

## Declarations

### Ethics approval and consent to participate

The current work uses public 1000 Genomes Project reference data. The
submission version must state the applicable data-use conditions and confirm
whether any additional cohort data were used.

### Consent for publication

Not applicable for the current public reference-data analysis.

### Availability of data and materials

The diagnostic input derives from the 1000 Genomes Project reference panel
[2]. Before submission, this section must provide an accession, precise
variant-processing recipe, frozen split manifest, and a public archival link
to the final source code and evaluation artifacts.

### Competing interests

To be completed by all authors before submission.

### Funding

To be completed by all authors before submission.

### Author contributions

To be completed using a CRediT contribution statement after the study is
complete.

## References

1. Oprisanu B, Ganev G, De Cristofaro E. On utility and privacy in synthetic
   genomic data. *Proceedings of the Network and Distributed System Security
   Symposium*. 2022. doi:10.14722/ndss.2022.24092.
   https://arxiv.org/abs/2102.03314
2. Auton A, Brooks LD, Durbin RM, et al.; 1000 Genomes Project Consortium. A
   global reference for human genetic variation. *Nature*. 2015;526:68--74.
   doi:10.1038/nature15393. https://www.nature.com/articles/nature15393
3. Hao W, Song M, Storey JD. Probabilistic models of genetic variation in
   structured populations applied to global human studies. *PLoS Genetics*.
   2016. https://pmc.ncbi.nlm.nih.gov/articles/PMC4795615/
4. Austin J, Johnson DD, Ho J, Tarlow D, van den Berg R. Structured denoising
   diffusion models in discrete state-spaces. *NeurIPS*. 2021.
   https://arxiv.org/abs/2107.03006
5. Kenneweg P, Dandinasivara R, Luo X, Hammer B, Schönhuth A. Generating
   synthetic genotypes using diffusion models. *Bioinformatics*.
   2025;41(Suppl 1):i484--i492. doi:10.1093/bioinformatics/btaf209.
   https://pmc.ncbi.nlm.nih.gov/articles/PMC12261458/
6. Wharrie S, Yang Z, Raj V, et al. HAPNEST: efficient, large-scale generation
   and evaluation of synthetic datasets for genotypes and phenotypes.
   *Bioinformatics*. 2023;39:btad535. doi:10.1093/bioinformatics/btad535.
   https://doi.org/10.1093/bioinformatics/btad535
7. Rogers AR, Huff C. Linkage disequilibrium between loci with unknown phase.
   *Genetics*. 2009;182:839--844. doi:10.1534/genetics.108.093153.
