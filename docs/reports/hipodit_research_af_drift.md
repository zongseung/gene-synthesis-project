# Primary-source notes: preserving marginal allele frequency under a pairwise genotype residual

Date: 2026-09-15. Scope: literature only (no code, experiments or design decisions). Covers why a shared
pairwise genotype table on top of a frozen Binomial(2, p_ij) base can move allele frequency (AF), and which
published constructions keep per-SNP / per-cohort AF fixed. Items marked **Derivation** are algebra written for
this note, not claims taken from a source.

## Q1. Moment matching at the MLE with free intercepts

**Finding.** In a log-linear (maximum-entropy / multinomial-logit) conditional model, the maximum-likelihood fit
satisfies, for every feature f, "model expectation of f = empirical expectation of f". Berger et al. show the
maximum-entropy model under these constraints is exactly the ML fit of the exponential family. ESL gives the
logistic special case: the intercept's score equation makes fitted and observed class counts equal. Two caveats
follow from the stated form. (i) The model expectation is taken over the **empirical** context distribution
p̃(x), so the match is teacher-forced (observed predecessor genotypes), not a guarantee for free-running generation.
(ii) Equality holds at the unpenalized MLE; with a ridge λ‖δ‖² the score becomes Σ_i(y_i − μ_i) = 2λδ
(**Derivation**), so matching is only approximate.

**Exact statements.**
- Berger et al. §3.2, constraint (3) in explicit form: `Σ_{x,y} p̃(x) p(y|x) f(x,y) = Σ_{x,y} p̃(x,y) f(x,y)`.
  "A feature is a binary-valued function of (x,y); a constraint is an equation between the expected value of the
  feature function in the model and its expected value in the training data."
- §3.3, eq (4): `C ≡ { p ∈ P | p(f_i) = p̃(f_i) for i ∈ {1, 2, ..., n} }`.
- §3.5, eqs (13)–(14): L_p̃(p) = Σ_{x,y} p̃(x,y) log p(y|x), and Ψ(λ) = L_p̃(p_λ). "The model p* ∈ C with maximum
  entropy is the model in the parametric family p_λ(y|x) that maximizes the likelihood of the training sample p̃."
  Footnote 2: if the dual maximum is not attained at finite λ, p* is only the limit of such models (relevant for
  SNPs whose genotype class is absent in training).
- ESL §4.4.1, eq (4.21): score equations `∂ℓ(β)/∂β = Σ_{i=1}^N x_i (y_i − p(x_i; β)) = 0`. "Notice that since the
  first component of x_i is 1, the first score equation specifies that Σ y_i = Σ p(x_i; β); the expected number of
  class ones matches the observed number (and hence also class twos.)"

**Mapping (Derivation).** P(G_j=g | h) ∝ C(2,g)·exp(g·η_ij + A[b_j,h,g]) with η_ij = logit p_ij. A logit offset
δ_j is the canonical parameter of feature g at SNP j, so its score equation is Σ_i (G_ij − E[G_j | G_i,j−1]) = 0
(training allele count matched under observed predecessors). Indicators 1[g=k] per (j,k) match per-class counts.
An uncentered, unpenalized u_{c,j} matches per-cohort allele counts the same way.

**Sources.** Berger, A. L., Della Pietra, S. A., Della Pietra, V. J. (1996). A maximum entropy approach to natural
language processing. *Computational Linguistics* 22(1):39–71. https://aclanthology.org/J96-1002/ .
Hastie, T., Tibshirani, R., Friedman, J. (2009). *The Elements of Statistical Learning*, 2nd ed., Springer, §4.4.1,
eq (4.21), pp. 120–121. DOI 10.1007/978-0-387-84858-7; official free PDF via https://hastie.su.domains/ElemStatLearn/ .

**Confidence: high.** Both passages read in full text (Berger: scanned PDF run through pdftotext; OCR noise on tildes,
equation structure unambiguous; ESL: 12th-printing PDF). McCullagh & Nelder (1989), Agresti and Birch (1963) not opened.


> **Controller correction (2026-09-15, after measurement).** The brief handed to this agent said the
> drift runs "toward the major allele". Measured on the frozen panel it runs the other way: the fitted
> `A` *boosts* heterozygotes (class component +0.7..+0.9 on g=1, −0.5..−0.8 on g=2), B3's pooled class
> frequencies match the data exactly ([.692,.183,.125] vs real [.692,.184,.124]), and 79 % of SNPs drift
> toward the **minor** allele, i.e. toward AF 0.5. The derivative recorded in Q2 is the one that governs
> this: a het *boost* of size θ changes E[G] by +2p(1−p)(1−2p)θ, which for a minor-allele SNP (p < 0.5)
> raises AF. The sentence "which confirms your diagnosis" in Q2 refers to the superseded framing; the
> algebra stands, the direction in that clause does not.
>
> **Second correction (same measurement).** The Q3 inference "the Binomial base over-predicts heterozygotes"
> is also inherited from the superseded framing. On this panel the base *under*-predicts them: predicted het
> fraction .112 vs observed .186 on train (implied per-SNP F median −0.60, IQR −0.83..−0.24, same on dev),
> i.e. heterozygote **excess** relative to Binomial(2, p_ij). A plausible cause is that oracle z are scored
> from each individual's own calls, pushing p_ij toward 0/1 for homozygotes (mean predicted het .035 at
> observed homozygotes vs .448 at observed heterozygotes). Consequence for remedy #4: with F ≈ −0.6 and
> individual p_ij near 0 or 1, the fixed-F form violates F ≥ −min(p,1−p)/max(p,1−p) for many cells and goes
> negative. The prototype therefore uses an AF-exact exponential tilt π(g) ∝ C(2,g)·x^g·e^{θ·1[g=1]} with x
> solved in closed form from (1−p)x² + e^θ(1−2p)x − p = 0, which is valid for every θ and, by the c=(1,−2,1)
> argument in Q2, equals an F-form with an individual-specific F that stays inside the valid range.

## Q2. Inbreeding-coefficient genotype frequencies

**Finding.** Meisner & Albrechtsen write Wright's F-extended HW proportions with individual allele frequency π_is
and a per-site F_s, defined via Wright's coefficient F_s = 1 − H_O/H_E (eq 2). They allow F_s ∈ [−1, 1]:
negative means heterozygote excess, positive means homozygote excess. They note negative F can make the homozygote
probabilities negative, and they truncate and rescale in that case.

**Exact statement (eq 4, preprint).**
`P(G=g | π_is, F_s) = (1 − π_is)² + π_is(1 − π_is)F_s  (g=0);  2π_is(1 − π_is)(1 − F_s)  (g=1);  π_is² + π_is(1 − π_is)F_s  (g=2)`.
"a negative estimate indicates an excess of heterozygosity and a positive estimate indicates an excess of
homozygosity in site s. While a positive inbreeding coefficient does not change the allele frequency, a negative
inbreeding coefficient will increase the sample allele frequency as the fraction of heterozygous individuals
increases. A fixed allele frequency and a negative inbreeding coefficient can lead to negative probabilities for
the homozygous genotypes in Equation 4 [...]"

**Algebra (Derivation).**
- Normalization: (1−p)² + 2p(1−p) + p² + F·p(1−p)·(1 − 2 + 1) = 1.
- Mean: E[G] = P₁ + 2P₂ = 2p(1−p)(1−F) + 2p² + 2Fp(1−p) = 2p(1−p) + 2p² = 2p, so **E[G]/2 = p for every F**.
- Heterozygosity: P₁ = 2p(1−p)(1−F), i.e. scaled by (1−F).
- Vector form: P_F = P_HWE + F·p(1−p)·c with c = (1, −2, 1). c is the only direction (up to scale) with Σ_g c_g = 0
  and Σ_g g·c_g = 0. So on three genotype classes, any change in class mass that keeps p fixed has F form.
- Validity: P₁ ≥ 0 ⇔ F ≤ 1; P₀, P₂ ≥ 0 ⇔ F ≥ −min(p,1−p)/max(p,1−p). Heterozygote deficit (F > 0) is always valid.
  My reading of the quoted remark: AF only changes for negative F after truncation and rescaling, not under eq (4) itself.
- Contrast: multiplicative tilt P'(g) ∝ P(g)·exp(a_g) with a = (0, −ε, 0) gives dE[G]/dε|₀ = −P₁(1 − 2p). Under a
  Binomial base that is −2p(1−p)(1−2p): for p < ½ the AF moves toward 0 (the major allele wins), and the shift is zero
  only at p = ½. This matches the working diagnosis.

**Sources.** Meisner, J., Albrechtsen, A. (2019). Testing for Hardy–Weinberg equilibrium in structured populations
using genotype or low-depth next generation sequencing data. *Mol. Ecol. Resour.* 19(5):1144–1152.
DOI 10.1111/1755-0998.13019. Read the preprint: bioRxiv DOI 10.1101/468611 (v1, 2018), eqs (2)–(4), pp. 2–3.
Owning classic sources (not opened): Wright, S. (1922) Coefficients of inbreeding and relationship. *Am. Nat.*
56:330–338, DOI 10.1086/279872; Wright, S. (1949/1951) The genetical structure of populations. *Ann. Eugen.*
15:323–354, DOI 10.1111/j.1469-1809.1949.tb02451.x.

**Confidence: high** for the equation and quote (preprint PDF text). **Medium** that equation numbering and wording
are the same in the published version (not opened). The AF invariance is exact algebra.

## Q3. HWE under structure with individual-specific allele frequencies

**Finding.** LFA (Hao, Song & Storey) models x_ij ~ Binomial(2, π_ij) with logit(π_ij) = Σ_k a_ik h_kj (L = AH), all
parameters unconstrained. It picks the latent dimension d by "the best overall goodness of fit with Hardy-Weinberg
equilibrium" (d = 15 HGDP, d = 7 TGP). Hao & Storey (2019) define "structural HWE" (sHWE): H0 says every individual's
genotype is Binomial(2, π_ij). Their statistic compares observed genotype counts N(G) with Σ_j p_j(G) through a
2-df quadratic form T, with a parametric-bootstrap null when π is estimated. **Neither paper parameterizes the
alternative or says which direction (deficit or excess) structure-related deviations take.** A full-text grep of both
for heterozyg/homozyg/Wahlund/inbreed returned no hits. The direction statement, and an explicit per-site F on top of
PCA-based individual allele frequencies, come from Meisner & Albrechtsen (Q2).

**Exact statements.**
- LFA (preprint): "E[x_ij]/2 = π_ij. If an observed SNP genotype x_ij is treated as a random variable, then under
  Hardy-Weinberg Equilibrium π_ij serves to model x_ij as a Binomial parameter: x_ij ∼ Binomial(2, π_ij)."
  Model 2: "L = AH (2) where A is m × d and H is d × n [...] logit(π_ij) = Σ_{k=1}^d a_ik h_kj, where all parameters
  are free to span the real numbers R."
- Hao & Storey eq (1): "p_j(G) = E[1(x_j = G)] = (1 − π_j)² if G = 0; 2π_j(1 − π_j) if G = 1; (π_j)² if G = 2".
  "F = ½E[X]" (their F is the m × n matrix of π_ij, not an inbreeding coefficient). "T = (v − μ)ᵀ Σ⁻¹ (v − μ)", with v = (N(0), N(1)) and μ = (Σ_j p_j(0), Σ_j p_j(1)), is asymptotically
  χ² with 2 df. "When K is too small and the population structure is insufficiently modeled, the sHWE test P-values
  are skewed heavily toward zero [...] Eventually, the P-value distributions become skewed toward one, as population
  structure model is overfit to the data."
- Meisner & Albrechtsen (intro): "Extensions to HWE have been defined to incorporate an inbreeding coefficient in
  the statistical models to quantify deviations from HWE as a deficiency in observed heterozygotes. However,
  population structure will also lead to violations of the expected Hardy-Weinberg (HW) proportions by increasing
  the observed homozygosity due to the Wahlund effect, or increasing the observed heterozygosity due to recent admixture."

**Inference (not in the sources).** The Binomial base over-predicts heterozygotes. That is exactly the per-SNP
count mismatch T tests for, and it fits residual (Wahlund-type) structure that a low-rank GLM-PCA does not capture.
The published way to absorb it while holding π_ij fixed is the Q2 F parameterization.

**Sources.** Hao, W., Storey, J. D. (2019). Extending tests of Hardy–Weinberg equilibrium to structured populations.
*Genetics* 213(3):759–770. DOI 10.1534/genetics.119.302370; https://pmc.ncbi.nlm.nih.gov/articles/PMC6827367/ .
Hao, W., Song, M., Storey, J. D. (2016). Probabilistic models of genetic variation in structured populations applied
to global human studies. *Bioinformatics* 32(5):713–721. DOI 10.1093/bioinformatics/btv641; read arXiv:1312.2041.

**Confidence: high** for the parameterizations and for the absence of a direction statement in Hao & Storey (full
PMC text). **Medium** for LFA wording (arXiv preprint, not the published version).

## Q4. Marginal-preserving constructions

**(a) Latent Gaussian thresholding.** HapSim finds a covariance C with Φ(z(p_i), z(p_j); c_ij) = p_ij, i.e. the
bivariate normal CDF at the marginal-quantile thresholds equals the target joint allele frequency. It then simulates
m ~ N(0, C) and sets "s_i = 1 if m_i ≤ z(p_i) else s_i = 0". Marginals come from the thresholds, P(s_i = 1) = Φ(z(p_i)) =
p_i, whatever C is. When C has to be adjusted to be positive-definite: "The estimated allele frequencies are not
affected by this approximation, and are unbiased." LD matches only approximately ("generally in very good agreement
with its target"). HapSim thresholds phased binary haplotypes; unphased {0,1,2} data would need an ordinal
two-threshold version (**Derivation**, not in HapSim). For Genest & Nešlehová only the abstract was reachable: "the
possibility of ties that results from atoms in the probability distribution invalidates various familiar relations
that lie at the root of copula theory in the continuous case". I could not read their statement of Sklar's theorem
for discrete margins (copula non-uniqueness, margin-dependent dependence measures).

**(b) Markov/window chains fit from data.** GWAsimulator: "the conditional probabilities for the alleles at locus
d + i given the haplotype at [d + i − 4, d + i − 1] are determined based on the input phased data". "every four
consecutive SNPs are used to determine the allele at the next SNP, but the window size can be modified by the user".
"We assume all SNPs follow Hardy–Weinberg equilibrium in the general population". "the simulated data have similar
local LD patterns as the HapMap data". The method keeps short-range but not long-range LD. **The paper does not state
marginal preservation.** Derivation for a first-order chain with T_j(h→g) = n_j(h,g)/n_{j−1}(h) and start P̂(G_1),
all counted on the same N sequences: P(G_j = g) = Σ_h P̂_{j−1}(h)·n_j(h,g)/n_{j−1}(h) = Σ_h n_j(h,g)/N = P̂_j(g). By
induction every chain marginal equals the empirical marginal, because the rows of each pair table sum to the previous
SNP's counts. The same argument gives window-level marginals for window length w. It breaks if T is smoothed, tied
across SNPs, or counted on a different sample from the start distribution.
HAPGEN2 does **not** use a per-SNP empirical transition table. Each new haplotype is "an imperfect mosaic of the
haplotypes in H_R and the haplotypes that have already been simulated" (Li & Stephens copying model). The steps are
crossovers driven by genetic distance, a uniformly chosen copying haplotype, and alleles emitted with a mutation
parameter. Official doc: "It simulates haplotypes by conditioning on a reference set of population haplotypes and an
estimate of the fine-scale recombination rate across the region, so that the simulated data has the same LD patterns
as the reference data." No exact marginal guarantee is stated, and the mutation term perturbs copied alleles (inference).

**Sources.** Montana, G. (2005). HapSim. *Bioinformatics* 21(23):4309–4311. DOI 10.1093/bioinformatics/bti689.
Li, C., Li, M. (2008). GWAsimulator: a rapid whole-genome simulation program. *Bioinformatics* 24(1):140–142.
DOI 10.1093/bioinformatics/btm549. Su, Z., Marchini, J., Donnelly, P. (2011). HAPGEN2: simulation of multiple disease
SNPs. *Bioinformatics* 27(16):2304–2305. DOI 10.1093/bioinformatics/btr341; doc
https://mathgen.stats.ox.ac.uk/genetics_software/hapgen/hapgen2.html . Genest, C., Nešlehová, J. (2007). A primer on
copulas for count data. *ASTIN Bulletin* 37(2):475–515. DOI 10.2143/AST.37.2.2024077.

**Confidence.** HapSim and GWAsimulator **medium**: the quotes came through a fetch tool from the OUP pages, and
direct raw HTML was blocked (403), so exact wording is not independently verified. HAPGEN2 **medium-high** (official
documentation page plus the article). Genest & Nešlehová **low** (abstract only). The chain-marginal induction is exact algebra.

## Q5. Locally normalized chains with tied parameters (brief)

**No direct statement found** that per-position normalization with tied parameters cannot control per-position
marginals independently. Closest statement: Sutton & McCallum §6.1 on label bias in MEMMs. For the backward
recursion β_t(i) = Σ_j p(y_{t+1}=j | y_t=i, x_{t+1}) β_{t+1}(j) (eq 6.5): "Unfortunately, this sum is always 1,
regardless of the value of the current label i. What this means is that the future observations provide no
information about the current state". Also: "Label bias is not caused by a model being directed or undirected. It is
caused by the structure of the particular directed model that is used in the MEMM." This is about inference over
earlier labels, not marginal calibration, so it does not directly support the diagnosis.
**Derivation.** The generation marginal is P(G_j) = Σ_h P(G_{j−1}=h)·P(G_j | h). With A tied across a bin, the AF
shift at SNP j depends on p_ij, the predecessor distribution and A[b]. A single 3×3 table cannot zero that shift at
every SNP: to first order a tilt exp(a_g) changes P_p by P_p(g)(a_g − ā), which is AF-neutral only if it is ∝ c = (1,−2,1)
(Q2), i.e. a_g − ā ∝ c_g / P_p(g), which depends on p.
Source: Sutton, C., McCallum, A. (2012). An introduction to conditional random fields. *Foundations and Trends in
Machine Learning* 4(4):267–373, §6.1, pp. 357–358. DOI 10.1561/2200000013. The original Lafferty, McCallum & Pereira
(ICML 2001) could not be opened (404/403). **Confidence:** high that the quote is exact; low relevance.

## Remedy catalogue derived from the sources

| # | Remedy | Mechanism | AF-preservation guarantee | Extra params (J=361, 26 cohorts, 4 bins) | Support |
|---|---|---|---|---|---|
| 1 | Unpenalized per-SNP allele offset δ_j, fit jointly with A | canonical feature g per SNP; score eq Σ_i(G_ij − E[G_j \| h_i]) = 0 | On train via score equations, teacher-forced (observed predecessors); exact only if unpenalized; no guarantee for free-running generation | +361 | Berger §3.2–3.5; ESL (4.21) |
| 2 | Per-SNP per-class offsets δ_{j,1}, δ_{j,2} (class 0 as reference) | class indicators per SNP; also absorbs A's class-calibration role, leaving A to explain h-dependence | On train via score equations for per-SNP class counts (so AF and heterozygosity), teacher-forced | +722 | Berger; ESL |
| 3 | Remove centering (and ridge) from u_{c,j} | cohort×SNP intercept as a canonical feature | On train via score equations for per-cohort AF, teacher-forced (subsumes #1) | +361 df (26×361 free vs 25×361 centered); no new tensor | Berger; ESL |
| 4 | AF-neutral heterozygosity term: Binom(g;2,p) + F·p(1−p)·(1,−2,1) replacing the class-generic tilt | Wright/F parameterization | Exact by construction for the unary distribution (E[G]=2p for all valid F); the pairwise term must be constrained separately | 4 (per bin) / 361 (per SNP) / 9,386 (cohort×SNP) | Meisner & Albrechtsen eq (4); Wright 1922 (not opened) |
| 5 | Empirical transition chain T_j(h→g) = n(h,g)/n(h), start P̂(G_1) | GWAsimulator-style conditionals from counts | Exact by construction on the counting sample (induction); none if smoothed or tied; ignores individual p_ij | ≤360 pairs × 4 df = 1,440 per group; 37,440 per cohort (sparse for ~61-sample cohorts) | Li & Li 2008 (algorithm); marginal property is Derivation |
| 6 | Latent Gaussian thresholding (copula): thresholds from per-individual genotype-class probabilities, correlation per bin | margins fixed by thresholds, correlation sets dependence | Exact by construction (margins do not depend on the correlation) | 4 (per bin) or ≤360 (per adjacent pair) | Montana 2005; Genest & Nešlehová (discrete caveats, abstract only) |
| 7 | Post-hoc per-SNP offset solved on the free-running chain (3-state forward recursion per individual) so Σ_i E_chain[G_ij] = Σ_i G_ij | 1-D root per SNP, in genomic order; monotone since ∂E/∂δ_j = Σ_h P(h)·Var(G_j \| h) > 0 | Exact on train by construction for expected AF at generation time; held-out not guaranteed | +361 (or +9,386 per cohort) | None; Derivation from Q1 and Q4b |

## Sources not found / uncertainties

- Not opened: McCullagh & Nelder (1989), Agresti, Birch (1963), Wright (1922, 1949/51), Weir & Cockerham (1984),
  Song (2000), Genest & Nešlehová full text (Cambridge Core paywall; ETH copy is a JS page), and Lafferty et al. (2001)
  (404/403). McFadden (1974) PDF was reachable but is a scan with no text layer, so the known "alternative-specific
  constants reproduce aggregate shares" result could not be quoted.
- HapSim and GWAsimulator quotes came through a fetch summarizer from publisher HTML (raw HTML blocked with 403).
  Treat the exact wording as medium confidence and re-check before quoting in a manuscript.
- Meisner & Albrechtsen: read the 2018 preprint only. Their sentence that negative F "will increase the sample
  allele frequency" conflicts with eq (4), where E[G] = 2π for every F. It most likely refers to the truncated and
  rescaled distribution (my reading).
- Hao & Storey (2019) and LFA (2016) contain no heterozygote-direction statement (grep). The link from rank-limited
  structure to heterozygote over-prediction here is an inference, supported only by the Wahlund/admixture sentence in
  Meisner & Albrechtsen.
- Every "on train via score equations" guarantee (#1–#3) is a training-sample, teacher-forced equality. It does not
  bound held-out AF error (the pre-registered gate), and it does not fix the marginals of a free-running chain unless
  the fitted conditionals equal the empirical ones (Q4b). Ridge penalties break exactness in proportion to λ.
- The Q5 conclusion (a tied table cannot be AF-neutral at every SNP) is a derivation; no primary source states it.
