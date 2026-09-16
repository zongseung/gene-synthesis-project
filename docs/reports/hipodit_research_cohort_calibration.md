# Primary-source notes: cohort-specific calibration without pooled marginal drift

Date: 2026-09-15. Scope: literature only — no code, no experiments, no design decisions. Question: which
published constructions add a per-cohort correction to a frozen base model while leaving the pooled
(across-cohort) marginal untouched — exactly by construction, or exactly on the training sample.
Items tagged **Derivation** are algebra written for this note, not claims taken from a source. Several
sources were read by OCR of scanned PDFs (noted per item); OCR noise falls on symbols, not structure.

---

## Q1. Calibration / raking / iterative proportional fitting

**Finding.** Deming & Stephan pose the adjustment as a *constrained* minimisation: stay as close as
possible (weighted least squares / minimum chi-square) to the unadjusted counts **subject to** the required
marginal totals, so the margins hold exactly at the optimum by construction; their one-margin case reduces
to a pure proportionate rescaling of each row. Ireland & Kullback swap the criterion for minimum
discrimination information and prove convergence of the iterative procedure. Deville & Särndal's
calibration estimators are the survey form of the same idea, with the multiplier explicit. Existence is
*not* automatic; uniqueness comes from strict convexity of the distance, not from the constraint.

**Exact statements.**
- Deming & Stephan, p. 428 (OCR): "(2) S = Σ(m_i − n_i)²/n_i … The conditions among the m_i will arise
  from the fact that the marginal totals, after adjustment, must agree with their expected values."
  Also p. 428: "Least squares has the practical advantage of uniqueness, once the weights of the
  observations have been assigned"; and p. 430 (Case I): "(15) m_ij = n_ij(m_i./n_i.) … The adjustment is
  thus a simple proportionate one by rows, the cells in any one row all being raised or lowered by the
  proportionate adjustment in the row total." The Lagrange multipliers λ appear along the borders of
  their worked table; the iterative version is introduced later as "a simplifying iterative procedure".
- Ireland & Kullback, abstract: estimate cell probabilities of a table with known, fixed marginal
  probabilities "so as to minimize ΣΣ p_ij ln(p_ij/π_ij)"; "an iterative procedure for determining the
  estimates"; "the estimates are BAN"; "the iterative procedure is convergent."
- Särndal (2007) §4.1–4.2: "determine weights w_k to satisfy the calibration equation Σ_s w_k x_k =
  Σ_U x_k"; "Minimizing the total distance Σ_s G_k(w_k, d_k) subject to the calibration equation
  Σ_s w_k x_k = Σ_U x_k leads to w_k = d_k F(q_k x_k′λ), where λ is obtained as the solution (assuming
  one exists) of (4.1) Σ_s d_k x_k F(q_k x_k′λ) = Σ_U x_k." G_k is required "strictly convex".
  "Questions about the existence of a solution to the calibration equation are discussed in Théberge (2000)."

**Derivation (transfer to this problem).** Per-cohort allele counts and the pooled allele count are two
margins of the same cohort × SNP count table, and the pooled margin is the *sum* of the cohort margins.
So any construction that reproduces all 26 cohort counts for SNP j exactly reproduces the pooled count
for SNP j automatically. The measured conflict is therefore not a raking-style margin conflict; it exists
only because the ridge stops the cohort margins being matched exactly.

**Sources.** Deming & Stephan (1940), *Ann. Math. Statist.* 11(4):427–444, doi:10.1214/aoms/1177731829
(read: OCR of the Project Euclid scan, pp. 427–432). Ireland & Kullback (1968), *Biometrika* 55(1):179–188,
doi:10.1093/biomet/55.1.179 (abstract only). Deville & Särndal (1992), *JASA* 87(418):376–382,
doi:10.1080/01621459.1992.10475217 (**not opened** — paywalled). Särndal (2007), *Survey Methodology*
33(2):99–119, https://www150.statcan.gc.ca/n1/en/pub/12-001-x/2007002/article/10488-eng.pdf .
Csiszár (1975), *Ann. Probab.* 3(1):146–158, doi:10.1214/aop/1176996454 — abstract only ("useful existence
theorems for and characterizations of the minimizing PD", plus a convergence proof for iterative scaling).

**Confidence: high** for Deming & Stephan and Särndal (read verbatim); **medium** for Ireland & Kullback
and Csiszár (abstracts only); **low** for Deville & Särndal 1992's own wording (not opened).

---

## Q2. Constrained maximum likelihood with a moment constraint

**Finding.** Aitchison & Silvey is the general theorem: maximise the log-likelihood subject to r
restraints by solving the score equations *plus* one Lagrange multiplier per restraint, and the restraints
hold exactly at the solution. For an exponential family the multiplier's role is concrete: the constrained
optimum stays in the same family with the multiplier added to the canonical parameter — i.e. a plain
offset in logit space, one scalar per constraint. Berger et al. state both halves (constraint set, and the
resulting exponential form), and note the maximum-entropy solution under the constraints *is* the ML fit
inside that parametric family. Särndal's w_k = d_k F(q_k x_k′λ) is the same structure with F a link.

**Exact statements.**
- Aitchison & Silvey, Summary: "the estimator considered lies in the subset and is a solution of
  likelihood equations containing a Lagrangian multiplier. It is proved that, under certain conditions
  analogous to those of Cramér, these equations have a solution which gives a local maximum of the
  likelihood function." §2, eqs (2.1)–(2.2) (OCR): "(2.1) f(x, θ) + H_θ λ = 0, (2.2) h(θ) = 0, where
  f(x, θ) is the point whose ith component is ∂L(x, θ)/∂θ_i" and H_θ is the s × r matrix (∂h_i(θ)/∂θ_j).
- Berger et al. §3.1, eq (4): "C ≡ {p ∈ P | p(f_i) = p̃(f_i) for i ∈ {1, 2, …, n}}" — model expectation of
  each feature equals its empirical expectation. §3.4, eq (7): "For each feature f_i we introduce a
  parameter λ_i (a Lagrange multiplier). We define the Lagrangian Λ(p, λ) ≡ H(p) + Σ_i λ_i(p(f_i) − p̃(f_i))".
  Eq (10): "p_λ(y|x) = (1/Z_λ(x)) exp(Σ_i λ_i f_i(x, y))". "The maximum entropy model subject to the
  constraints C has the parametric form p_λ* of (10), where the parameter values λ* can be determined by
  maximizing the dual function Ψ(λ)."

**Derivation (transfer).** The pooled constraint for SNP j, Σ_i (G_ij − 2σ(α_j + v_j·z_i + u_{c_i,j} + δ_j)) = 0,
is one linear moment constraint; by (2.1) its multiplier enters the canonical parameter as the single
scalar δ_j, i.e. exactly a per-SNP logit offset. It leaves the ridge on u untouched and is a monotone 1-D
root-find in δ_j (the left side is strictly decreasing in δ_j). Equivalently: leave a per-SNP intercept
free and unpenalized in the fit — by the constraint of eq (4) with the constant feature, an unpenalized
canonical parameter forces its own sufficient statistic to match the data on the training sample.

**Sources.** Aitchison, J., Silvey, S. D. (1958). Maximum-likelihood estimation of parameters subject to
restraints. *Ann. Math. Statist.* 29(3):813–828. https://doi.org/10.1214/aoms/1177706538 (read: OCR of the
Project Euclid scan, pp. 813–815). Berger, A. L., Della Pietra, S. A., Della Pietra, V. J. (1996). A
maximum entropy approach to natural language processing. *Computational Linguistics* 22(1):39–71.
https://aclanthology.org/J96-1002/ . McCullagh & Nelder (1989) and Agresti **not opened**.

**Confidence: high** (both read verbatim; the transfer is one line of algebra, flagged as Derivation).

---

## Q3. Partial pooling / empirical Bayes instead of ridge-toward-zero

**Finding.** Efron & Morris is the primary argument for shrinking toward an estimated common mean rather
than toward zero, with the shrinkage factor estimated from the data; Gelman's multilevel formulation
generalises the target to a *group-level regression* (the superpopulation analogue: cohort effects
shrunk toward a superpop-level mean rather than toward 0), with the shrinkage variance σ_α estimated
jointly. Neither source states anything about preserving an aggregate. What can be said: shrinkage
toward the sample mean with a *common* factor preserves the unweighted mean exactly (Derivation below);
with unequal group sizes and a per-group factor it does not. In a hierarchical model the pooled fit is
protected by the free, unpenalized global term (γ₀ / the per-SNP intercept), not by the shrinkage.

**Exact statements.**
- Efron & Morris §2, eq (2.3) (OCR): "we estimate the common unknown value μ = Σ μ_i/k by X̄ = Σ X_i/k,
  shrinking all X_i toward X̄, an idea suggested by Lindley … δ_i(X) = X̄ + (1 − (k − 3)/V)(X_i − X̄) (2.3)
  with V = Σ(X_i − X̄)² and with k − 3 = (k − 1) − 2 as the appropriate constant since one parameter is
  estimated." The shrinkage is estimated from the data: "the unbiased estimates X̄ for μ and (k − 3)/V for
  1/(1 + τ²) from the marginal distribution of X".
- Gelman (2006), eq (2): "Π_j Π_i N(y_ij | α_j + βx_ij, σ_j²) · Π_j N(α_j | γ₀ + γ₁u_j, σ_α²)" — group
  intercepts are drawn around a group-level regression on a group-level predictor u_j, and the estimates
  are obtained "by shrinking toward the complete-pooling estimate"; "the deviation of the coefficients
  from the line is captured in σ_α".

**Derivation.** (1/k)Σ_i δ_i(X) = X̄ exactly in (2.3), because the target is the sample mean and the
shrinkage factor is common to all i. Replace the common factor by group-specific factors (unequal n_c:
49–91 rows here) or the unweighted target by a weighted one and the identity breaks; the shrunken
estimates then average to something between the target and the weighted data mean. So partial pooling
per se buys nothing for the pooled marginal — the exact statements in Q1/Q2 do.

**Sources.** Efron & Morris (1975), *JASA* 70(350):311–319, doi:10.1080/01621459.1975.10479864 (read by OCR
of the JSTOR scan at jhanley.biostat.mcgill.ca/bios602/MultilevelData/EfronMorrisJASA1975.pdf). Gelman
(2006), *Technometrics* 48(3):432–435, doi:10.1198/004017005000000661 (PDF:
sites.stat.columbia.edu/gelman/research/published/multi2.pdf). Gelman et al., *BDA3* ch. 5 **not opened**
(PDF exceeded the fetch size limit).

**Confidence: high** for what the two sources say; **high** that neither claims aggregate preservation
(searched both texts); **medium** that no primary source states it anywhere — absence of evidence.

---

## Q4. Genetics prior art: population frequencies on top of a global frequency

**Finding.** The two workhorse tools do *not* have a global frequency at all: STRUCTURE gives each
population an independent Dirichlet prior, ADMIXTURE has only population-specific f_kj mixed on the
probability scale. The construction the brief is after is the F-model / Balding–Nichols prior: population
frequencies are drawn *around* an ancestral frequency with a drift parameter, and Falush et al. state
E(p_klj) = p_Alj explicitly — mean-preserving by construction, on the **probability** scale, **in
expectation** over the prior (not exactly on any finite sample). The Nicholson-style Gaussian analogue is
the same idea with N(e_l, C_k·e_l(1 − e_l)), and there the truncation at 0/1 breaks exact mean preservation.

**Exact statements.**
- Falush, Stephens & Pritchard, eqs (4)–(5): "p_Al· ~ D(λ₁, λ₂, …, λ_Jl)" and "p_kl· ~ D(p_Al1(1 − F_k)/F_k,
  p_Al2(1 − F_k)/F_k, …, p_AlJl(1 − F_k)/F_k) … independently for each k and l." And: "In our
  parameterization (Equations 4 and 5) E(p_klj) = p_Alj, and Var(p_klj) = F_k p_Alj(1 − p_Alj). Hence, p_Alj
  plays a role analogous to that of p̄, and F_k plays a role like that of F_ST in the classical model."
  Credited as "related to the parameterization of Balding and Nichols (1997)", following Nicholson et al. (2002).
- Pritchard, Stephens & Donnelly, eq (4): "p_kl· ~ D(λ₁, λ₂, …, λ_Jl) independently for each k,l … We take
  λ₁ = λ₂ = · · · = λ_Jl = 1.0, which gives a uniform distribution on the allele frequencies." Genotypes:
  eq (2) "Pr(x_l^(i,a) = j | Z, P) = p_z(i)lj". No global/ancestral frequency parameter appears.
- Alexander & Lange (2011), Methods: "Given K ancestral populations, the success probability p_ij in the
  binomial distribution n_ij ~ Bin(2, p_ij) depends on the fraction q_ik of i's ancestry attributable to
  population k and on the frequency f_kj of allele 1 in population k", with q_ik ≥ 0, Σ_k q_ik = 1,
  0 ≤ f_kj ≤ 1. Mixing is on the probability scale; there is no global frequency parameter.
- Coop et al. (2010), Methods: "we follow Nicholson et al. (2002) by assuming that the population allele
  frequency in a subpopulation, x_kl, is normally distributed around an ancestral allele frequency e_l
  (0 < e_l < 1), but that the densities of x_kl above 1 and below 0 are replaced with point masses on 1 and
  0, respectively … the variance of this normal distribution is a product of a factor that is constant
  across loci multiplied by a locus-specific term: i.e., e_l(1 − e_l)."

**Transfer.** Every one of these puts the cohort deviation on the *probability* scale around a global
frequency, which is what makes E[p_cj] = p_j exact. None uses a zero-mean offset on the logit scale — the
construction that produced the measured drift. Note also that the F-model guarantee is a prior-mean
statement: the posterior/fitted p_kj need not average to p_Aj on the training sample.

**Sources.** Falush, Stephens & Pritchard (2003), *Genetics* 164(4):1567–1587, doi:10.1093/genetics/164.4.1567
(PMC1462648; full PDF read). Pritchard, Stephens & Donnelly (2000), *Genetics* 155(2):945–959,
doi:10.1093/genetics/155.2.945 (PMC1461096; full PDF read). Alexander, Novembre & Lange (2009),
*Genome Res.* 19(9):1655–1664, doi:10.1101/gr.094052.109 (**not opened** — publisher SSO); model quoted
instead from Alexander & Lange (2011), *BMC Bioinformatics* 12:246, doi:10.1186/1471-2105-12-246
(PMC3146885). Balding & Nichols (1995), *Genetica* 96:3–12, doi:10.1007/BF01441146 (**not opened**).
Nicholson et al. (2002), *JRSS-B* 64(4):695–715, doi:10.1111/1467-9868.00357 (**not opened**; restated by
Coop). Coop et al. (2010), *Genetics* 185(4):1411–1423, doi:10.1534/genetics.110.114819 .

**Confidence: high** for Falush (the E[p] = p_A statement is verbatim), Pritchard, Coop; **medium** for
ADMIXTURE's exact p_ij = Σ_k q_ik f_kj form (the 2011 sentence is verbatim, the explicit sum is not);
**low** for Balding & Nichols (1995) wording — not opened.

---

## Q5. Logit-scale vs probability-scale offsets

**Finding.** The published "calibrated adjustment" of predicted probabilities that actually pins a target
mean is Saerens et al.: rescale each posterior by the ratio of new to old prior (in the binary case a
constant logit shift) and set the new prior to the *average of the adjusted posteriors*; at the EM fixed
point the mean adjusted probability equals the target exactly, on the given sample. Elkan's Theorem 2 is
the same logit-shift algebra but its invariant is different: it preserves the class-conditional densities
P(x|j) under a change of base rate; nothing about a sample mean. For the "logit offset does not preserve
the mean probability" statement itself, the cleanest published construction is the marginalized
(marginally specified) model: specify the marginal mean model, then let the conditional model carry an
implicitly-defined offset Δ so that the conditional mean averages back to the marginal one.

**Exact statements.**
- Saerens et al., eq (4): "p̂(ω_i|x) = [p̂(ω_i)/p̂_t(ω_i)]·p̂_t(ω_i|x) / Σ_j [p̂(ω_j)/p̂_t(ω_j)]·p̂_t(ω_j|x) …
  the new a posteriori probabilities are simply the a posteriori probabilities in the conditions of the
  training set, weighted by the ratio of the new priors to the old priors". Eq (9), EM: "p̂⁽⁰⁾(ω_i) =
  p̂_t(ω_i); p̂⁽ˢ⁾(ω_i|x_k) = [p̂⁽ˢ⁾(ω_i)/p̂_t(ω_i)]p̂_t(ω_i|x_k) / Σ_j […]; p̂⁽ˢ⁺¹⁾(ω_i) = (1/N) Σ_{k=1}^N
  p̂⁽ˢ⁾(ω_i|x_k)"; "shown to maximize the likelihood of the new data."
- Elkan, Theorem 2 (OCR of the scan): "p′ = b′(p − p b) / (b − p b + b′p − b b′)", under the assumption
  "that the shift in base rate is the only change in the population … P′(x|j = 1) = P(x|j = 1) and
  P′(x|j = 0) = P(x|j = 0)". "Theorem 2 is a statement about true probabilities given different base rates."
- Comstock & Heagerty, lnMLE documentation (implementation of Heagerty 1999), §2.1: "The first model is a
  marginal logistic regression for the average response as a function of covariates: logit E(Y_ij | X_ij) =
  X_ij β (1). Serial dependence is then modeled by conditioning on a latent variable instead of other
  response variables: logit E(Y_ij | b_i, X_i) = Δ_ij + b_ij (2)", with b_i | X_i = N(0, D_i) (3).
  Δ_ij is the quantity that makes the conditional model average back to (1) — Griswold & Zeger's abstract
  calls these "the marginal constraint equations" and solves them via the implicit function theorem.

**Derivation (Jensen).** σ is strictly convex on (−∞, 0) and strictly concave on (0, ∞), so for a
non-degenerate zero-mean offset u, E[σ(η + u)] ≠ σ(η) in general, with the sign of the gap set by where η
sits; a per-SNP δ_j solved from Σ_i σ(η_ij + δ_j) = target is exactly the correction that removes it on
the training sample (Q2).

**Sources.** Saerens, Latinne & Decaestecker (2002), *Neural Computation* 14(1):21–41,
doi:10.1162/089976602753284446 (author PDF read: dipot.ulb.ac.be/dspace/bitstream/2013/68391/1/Decaestecker_NeuralComp02.pdf).
Elkan (2001), *IJCAI-01*, 973–978, https://cseweb.ucsd.edu/~elkan/rescale.pdf (read by OCR — the PDF's
embedded encoding is unreadable). Heagerty (1999), *Biometrics* 55(3):688–698,
doi:10.1111/j.0006-341X.1999.00688.x (**not opened**); Comstock & Heagerty, lnMLE documentation,
https://faculty.washington.edu/heagerty/Software/LDA/MLV/lnMLEhelp.pdf . Griswold & Zeger (2004), JHU
Biostatistics Working Paper 99, https://biostats.bepress.com/jhubiostat/paper99/ (abstract only).
Platt (1999), in *Advances in Large Margin Classifiers*, MIT Press, 61–74 — **not opened**; no quote given.

**Confidence: high** for Saerens (verbatim, author PDF); **medium-high** for Elkan (OCR; the formula's
term structure is legible but individual primes are OCR-reconstructed); **medium** for Heagerty (the model
pair is verbatim from the author's own documentation, the defining convolution equation is not — the
1999 paper was not opened); **low** for Platt (not opened).

---

## Construction catalogue

| # | Construction | What is preserved, and how | Extra parameters (26 cohorts × 361 SNPs) | Cost per fit | Source |
|---|---|---|---|---|---|
| 1 | Keep ridge on `u`, drop the centering, and refit a **free unpenalized per-SNP intercept** jointly (α_j + δ_j) | Pooled AF matched **exactly on the training sample** — the unpenalized canonical parameter's own constraint p(f) = p̃(f) | +361 | one extra coordinate in the existing penalized fit | Berger eqs (4),(7),(10); Aitchison & Silvey (2.1)–(2.2) |
| 2 | Post-hoc per-SNP logit offset δ_j solved from Σ_i σ(η_ij + u + δ_j) = Σ_i G_ij/2 | Same guarantee as #1, without refitting `u`; pooled **exact on the training sample**, per-cohort fit unchanged in shape | +361 derived scalars | 361 monotone 1-D Newton/bisection solves | Aitchison & Silvey (multiplier = offset); Saerens eq (4) |
| 3 | Saerens-style EM rescaling per SNP (prior-ratio reweight, prior := mean of adjusted probabilities) | Mean adjusted probability = target **exactly at the fixed point** on the sample | +361 (same scalars, iteratively found) | a few EM sweeps per SNP | Saerens eqs (4),(9) |
| 4 | Probability-scale cohort offsets: p_cj = p⁰_j + d_cj with Σ_c n_c d_cj = 0 | Pooled AF preserved **exactly by construction** (linear), *until* clipping to [0,1] — truncation reintroduces bias | 26 × 361 | closed-form per SNP | Nicholson-style model as stated by Coop et al. |
| 5 | F-model / Balding–Nichols prior: cohort frequency drawn around a global p_j with drift F_c | E[p_cj] = p_j **in expectation by construction**; not exact on a sample, and not a statement about the fitted values | 361 global + 26 drift (or 26 × 361 latent) | MCMC / EB per SNP | Falush eqs (4)–(5), E(p_klj) = p_Alj |
| 6 | Rake the cohort × SNP allele-count table to both margins (IPF / minimum-distance calibration) | Both margins **exact by construction**; here the pooled margin is implied by exact cohort margins (Derivation, Q1) | 26 × 361 multiplicative factors | iterative scaling, cheap; existence not guaranteed | Deming & Stephan (2),(15); Ireland & Kullback; Särndal (4.1) |
| 7 | Hierarchical/partial pooling of `u` toward a superpop mean, variances estimated | Improves per-cohort accuracy; **nothing** is preserved at pooled level unless combined with #1/#2 | 26 × 361 + 5 × 361 + variance components | EB/REML outer loop | Efron & Morris (2.3); Gelman (2006) eq (2) |
| 8 | Drop the ridge: unpenalized per-cohort MLE of `u` | Per-cohort counts exact ⇒ pooled exact by summation (Derivation) — but this is the variance blow-up the ridge was added to prevent (49–91 rows/cohort) | 26 × 361 | same fit, no penalty | Berger eq (4) with per-cohort features |

Cheapest constructions that meet the stated goal exactly on the training sample: **#2** (bolt-on, leaves
the existing `u` fit untouched) and **#1** (same guarantee, obtained by making the per-SNP intercept a
free unpenalized parameter instead of a frozen one). #7 is orthogonal — it addresses the 49-row cohorts,
not the pooled marginal.

## Sources not found / uncertainties

- **Deville & Särndal (1992) JASA** — paywalled; reported only through Särndal's own 2007 review (verbatim).
- **Ireland & Kullback (1968)**, **Csiszár (1975)** — abstracts only (OUP paywall; Project Euclid full text
  blocked by bot detection). The existence/uniqueness claims for IPF rest on those abstracts.
- **Balding & Nichols (1995)** — Springer paywall; the `p(1−F)/F` beta form appears here only as Falush et
  al. (2003) restate it in Dirichlet form. **Heagerty (1999)** — paywalled; the Δ convolution equation is
  not quoted, only the model pair from the authors' own software documentation.
- **ADMIXTURE (2009) Genome Research** — blocked by publisher SSO; equations quoted from Alexander & Lange
  (2011) instead. The explicit p_ij = Σ_k q_ik f_kj sum is *not* verbatim from any source read here.
- **Platt (1999)** — original not opened; no quote or equation is given, and the claim that its intercept
  fit pins the mean calibration probability is deliberately omitted rather than asserted.
- **BDA3 / Gelman & Hill** — the BDA3 PDF exceeded the fetch size limit; Gelman (2006) Technometrics is used
  in its place. No primary source was found that states what partial pooling does to an aggregate.
- OCR was used for Deming & Stephan, Aitchison & Silvey, Efron & Morris and Elkan (scanned or
  broken-encoding PDFs). Equation *structure* in the quotes above was legible; individual sub/superscripts
  and primes were reconstructed from context and are flagged where that matters (Elkan).
