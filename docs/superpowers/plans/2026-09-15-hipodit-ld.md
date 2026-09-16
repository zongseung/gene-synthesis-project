# HiPoDiT-LD 모델 구축 및 검증 기획서

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development`
> (recommended) or `superpowers:executing-plans` to implement this plan task-by-task.
> Implement the checklist in order. A gate that fails stops the downstream work;
> it is not a hyperparameter-tuning invitation.

> **개정 2 (2026-09-15): Gate 1 실패 후 primary model이 바뀌었다.** §3.3의 B3(local-LD residual)는
> Gate 1을 통과하지 못했고(§9.1), 아래 §1–§8은 그 판정까지의 원본 사전등록 기록으로 보존한다.
> 현재 유효한 모델·arm·gate 정의는 **§9**다. §9와 충돌하는 §1–§8 조항은 §9가 우선한다.

## 1. 목적과 현재 결정

**목적:** `{0, 1, 2}` 비위상(unphased) 유전형 dosage를 합성할 때, 현 HiPoDiT의
GLM-PCA latent diffusion은 유지하되 독립적인 Binomial 디코더를 **집단 보정
local-LD residual 디코더**로 교체할 가치가 있는지를 검증한다.

현재 확보된 5-seed 파일럿에서 Fisher 3-band diffusion schedule은 표준 schedule에
대해 일관된 이득을 보이지 않았다. AF MAE는 standard `0.02410 ± 0.00165`,
Fisher `0.02431 ± 0.00256`, local dosage-covariance MAE는 standard
`0.02165 ± 0.00160`, Fisher `0.02151 ± 0.00215`였다. 따라서 **표준 Gaussian
latent diffusion을 기준선으로 고정**하고, Fisher는 최종 부가 ablation으로만 둔다.

이번 모델의 연구 질문은 다음 하나다.

> GLM-PCA latent와 cohort label이 주어졌을 때, 물리적 SNP 거리와 직전 dosage를
> 이용하는 작은 조건부 categorical residual이 독립 Binomial 디코더보다 held-out
> local LD와 집단별 allele-frequency를 재현하는가?

## 2. 범위와 비범위

| 포함 | 제외 |
|---|---|
| `Binomial(2,p)` GLM-PCA | Poisson likelihood |
| 표준 latent Gaussian diffusion | OC-FILM 및 FILM 계열 조건화 |
| cohort-frequency offset | 위상 haplotype/장거리 LD 주장 |
| gene 내부의 genomic-order local residual | 전장유전체 합성 주장 |
| 다중 시드, train/dev/test 분리, privacy 평가 | privacy 평가 없는 익명성·DP 주장 |

현재 prepared dataset의 분석 단위는 chr17의 8 genes, 361 SNPs, 2,504 individuals이며
train/dev/test는 `2,002 / 251 / 251`이다. 모든 observed call은 유한한 `{0,1,2}`이다.
MAF filtering을 유지하는 동안 rare-variant 성능도 주장하지 않는다.

## 3. 목표 모델 명세

### 3.1 입력과 공통 latent

각 sample `i`의 dosage vector를 `G_i`, cohort를 `c_i`, GLM-PCA factor를 `z_i`라
둔다. GLM-PCA는 observed call만 사용한 Binomial(2,p) likelihood와 missing mask로
학습한다. 현재의 factor normalization과 inverse-normalization은 diffusion 앞뒤에서
그대로 사용한다.

Diffusion은 cohort-conditioned latent `z`를 생성한다. 첫 통합 실험에서는 현
`HybridCNNDiTFiLM + GaussianDiffusion`의 **standard schedule**만 사용한다. 이 문서의
신규성 후보와 학습 우선순위는 diffusion kernel이 아니라 decoder다.

### 3.2 독립 Binomial 기준 디코더 (B0)

각 SNP `j`의 base logit과 확률은

\[
\eta_{ij}^{(0)} = \alpha_j + v_j^\top z_i,\qquad
p_{ij}^{(0)} = \sigma(\eta_{ij}^{(0)}),\qquad
G_{ij}\sim \operatorname{Binomial}(2,p_{ij}^{(0)}).
\]

이는 현 구현과 비교할 기준선이며, posterior/marginal metric 모두에서 반드시 남긴다.

### 3.3 cohort 보정 local-LD residual 디코더 (B3)

cohort offset을 넣은 base logit은

\[
\eta_{ij}=\alpha_j+u_{c_i,j}+v_j^\top z_i,\qquad p_{ij}=\sigma(\eta_{ij}).
\]

각 gene 안에서 SNP는 reference-position 오름차순으로 정렬한다. 첫 SNP는 base
Binomial만 사용하고, 이후 SNP는 직전 생성 dosage와 training set으로 고정한
physical-distance bin `b(d_{j-1,j})`를 조건으로 한다.

\[
P(G_{ij}=g\mid z_i,c_i,G_{i,j-1}) \propto
\operatorname{Binomial}(g;2,p_{ij})\exp\{A_{b(d_{j-1,j}),\,G_{i,j-1},\,g}\},
\quad g\in\{0,1,2\}.
\]

`A_b`는 bin마다 `3 × 3` table 하나뿐이다. 이는 복잡한 autoregressive transformer가
아니라, biologically ordered local dependence를 Binomial base distribution에 더하는
작은 residual이다. gene 경계에서 chain을 반드시 reset한다.

학습 목적함수는 teacher forcing negative log-likelihood와 shrinkage다.

\[
\mathcal L=-\sum_{i,j\in\mathrm{observed}}\log P(G_{ij}\mid z_i,c_i,G_{i,j-1})
+\lambda_u\lVert U\rVert_F^2+\lambda_A\sum_b\lVert A_b\rVert_F^2.
\]

`u`는 training cohort 크기 가중 평균이 SNP별 0이 되도록 중심화하고 ridge penalty를
둔다. 희소 cohort에는 별도 복잡한 계층 모델 대신 이 shrinkage만 적용한다.

## 4. 실험 arm과 비교 원칙

| Arm | 식에서 추가되는 요소 | 답하는 질문 |
|---|---|---|
| B0 | 없음 | 독립 Binomial base가 기준인가? |
| B1 | `u_{c,j}` | cohort calibration만으로 충분한가? |
| B2 | `A_b` | local LD residual만으로 충분한가? |
| B3 | `u_{c,j}+A_b` | 두 구조가 상호보완적인가? |
| B3-F | B3 + Fisher schedule | decoder 이득과 schedule 이득을 분리할 수 있는가? |

`B3-F`는 B3가 통과한 뒤 한 번만 수행한다. B0--B3의 decoder 선택에는 Fisher
schedule, OC-FILM, Poisson decoder를 섞지 않는다.

## 5. 단계별 구축 계획

### Phase 0. 데이터와 기준선 동결

- [ ] Prepared artifact, split indices, SNP ID/allele/position, gene boundary, cohort mapping,
  MAF filter, GLM-PCA hyperparameter, seed 목록의 SHA-256을 `study_manifest`에 기록한다.
- [ ] 5-seed standard/Fisher 결과와 실행 명령을 결과 보고서에 연결하고, standard
  schedule을 `latent_generator_baseline`으로 명시한다.
- [ ] 모든 decoder input에서 test label, test-derived frequency, test-derived distance-bin
  edge가 쓰이지 않는지 assertion으로 확인한다.

**산출물:** 재현 가능한 manifest와 고정된 B0 reference run.

**Gate 0:** train/dev/test leakage 또는 ordering ambiguity가 하나라도 발견되면 이후
phase를 시작하지 않고 data contract부터 수정한다.

### Phase 1. Oracle decoder 비교

- [ ] Real held-out `z`를 사용해 B0, B1, B2, B3를 학습한다. 이 단계에서는 diffusion
  sample을 넣지 않는다.
- [ ] physical-distance bin edge는 train gene 내부 인접-SNP 거리의 quartile로 한 번
  산출하고 고정한다. 빈 bin이 생기면 bin 수를 줄이며, 임의의 distance function을
  새로 만들지 않는다.
- [ ] B0/B1/B2/B3의 observed-call NLL, SNP별 AF MAE, cohort별 AF MAE, genotype
  proportion, heterozygosity, LD `r²` 및 signed covariance를 dev set에서 계산한다.
- [ ] `A_b`의 효과 검증을 위해 genomic order를 gene 내부에서 shuffle한 negative
  control을 추가한다. 실제 genomic order의 이득이 없다면 LD 해석을 하지 않는다.
- [ ] cohort label을 shuffle한 negative control을 추가한다. B1/B3가 이 control에서도
  같은 이득이면 cohort effect라고 주장하지 않는다.

**산출물:** decoder ablation 표, metric별 cohort/MAF/distance-bin 상세 CSV, fixed
candidate (`B0` 또는 `B3`).

**Gate 1:** B3가 dev에서 B0보다 (a) per-call NLL, (b) distance-stratified LD `r²`
MAE 두 지표 모두 개선하지 못하거나, cohort AF MAE를 5% 넘게 악화시키면 B3를 중단한다.
그 경우 기존 독립 Binomial decoder를 유지하고 diffusion 구조를 변경하지 않는다.

### Phase 2. 합성 latent와 decoder의 결합

- [ ] Gate 1을 통과한 decoder만 standard latent diffusion output에 연결한다.
- [ ] 학습에는 teacher-forced real predecessor만 사용하고, 샘플링은 gene마다 좌→우
  genomic order로 이미 생성한 predecessor를 사용한다.
- [ ] decoder가 받는 `z`의 scale/origin이 real-oracle과 generated-latent 간에 같은지
  inverse-normalization 경계에서 검사한다.
- [ ] 전체 조건 벡터에서 cohort label이 실제 sampling prior와 일치하게 추출되는지
  기록한다. balanced-cohort sampling은 별도 연구 질문이므로 기본 sampling과 섞지
  않는다.

**산출물:** fixed-seed end-to-end synthetic genotype artifact와 B0 대비 B3 metric report.

**Gate 2:** oracle에서 얻은 LD 개선이 generated latent에서도 사라지거나 AF/heterozygosity
guardrail을 위반하면 decoder를 paper model로 올리지 않는다. 먼저 Phase 1의 decoder
regularization만 dev에서 재탐색한다.

### Phase 3. 다중 시드 확증 실험

- [ ] 개발 중에는 dev set만 이용해 regularization, number of distance bins, GLM-PCA rank를
  선택한다. test set은 이 phase에서 한 번만 연다.
- [ ] 선택된 B0와 B3를 같은 10개 이상의 seed, 같은 train split, 같은 number of synthetic
  individuals, 같은 sampling step으로 paired run 한다.
- [ ] 각 seed에서 overall 및 cohort-stratified AF MAE, genotype distribution TV,
  heterozygosity error, LD `r²` MAE by physical-distance bin, signed covariance MAE,
  real-vs-synthetic population-classifier AUC/accuracy를 저장한다.
- [ ] paired seed difference의 mean과 95% bootstrap CI를 보고한다. 단일 최고 seed나
  평균만 제시하지 않는다.

**Primary success criterion:** B3-B0의 test NLL 및 LD `r²` MAE paired difference가 모두
0보다 작고, 두 차이의 95% CI 상한이 0보다 작아야 한다. Overall 및 cohort AF MAE는 B0의
105% 이하여야 한다. 이 조건을 충족하지 않으면 성능 향상 주장을 하지 않는다.

### Phase 4. 강건성·privacy·외부 일반화 경계

- [ ] real training sample에 대한 nearest-neighbor distance, membership-inference
  classifier, exact/near duplicate rate를 B0와 B3 모두에서 계산한다.
- [ ] cohort별 최소 sample 수, unseen/rare cohort, MAF bin별 failure mode를 표로
  분리한다. data가 부족한 stratum은 결측이 아니라 `not estimable`로 표기한다.
- [ ] 독립 cohort 또는 별도 genomic panel이 확보되기 전에는 external generalization을
  주장하지 않는다. 확보되면 allele/strand/build harmonization을 먼저 검증하고, 같은
  frozen decoder로 한 번 평가한다.

**Gate 4:** privacy attack 또는 duplication이 B0보다 현저히 나빠지면, utility 결과와
무관하게 sampling/regularization을 재검토한다. formal differential privacy는 별도
학습 보장이 없으므로 주장하지 않는다.

### Phase 5. 논문화 패키지와 주장 제한

- [ ] Methods에는 dosage가 Binomial인 이유, train-only binning/centering, teacher-forcing,
  sampling order, 모든 seed와 제외 규칙을 수식과 함께 명시한다.
- [ ] Results에는 B0--B3 ablation, negative controls, seed-level scatter, cohort/MAF/
  distance stratification, privacy 결과를 함께 제시한다.
- [ ] Discussion에는 unphased dosage, chr17 8-gene panel, local first-order residual,
  external cohort 부재를 명시한다.

**허용 가능한 novelty 문장:** “HiPoDiT-LD는 cohort-calibrated Binomial base distribution에
genomic-distance-binned local categorical residual을 결합하여, unphased genotype
synthesis에서 cohort calibration과 local LD preservation을 분리 검증한다.”

**금지 문장:** “최초의 genotype diffusion”, “phased haplotype LD 보존”, “whole-genome
generation”, “Fisher schedule이 우수”, “DP/anonymous synthetic data”.

## 6. 구현 파일 책임 경계

| 책임 | 예정 파일 |
|---|---|
| Binomial GLM-PCA와 missing-mask contract | `src/preprocessing/binomial_glm_pca.py` |
| B0--B3 likelihood, teacher-forced loss, sequential sampling | `src/models/genotype_decoder.py` (new) |
| position/gene/cohort metadata를 prepared artifact에 저장 | `scripts/hipodit_rebuild_prepare.py` |
| decoder 선택과 standard latent generator 결합 | `scripts/hipodit_rebuild_train.py` |
| oracle/end-to-end metric 및 stratum report | `scripts/hipodit_genotype_check.py` |
| paired multiseed arms와 confidence interval summary | `scripts/hipodit_multiseed.py` |
| probability normalization, centering, ordering, no-leakage | `tests/test_genotype_decoder.py` (new) |

새 decoder 외에 공통 추상화, 새 configuration framework, 새 preprocessing pipeline은 만들지
않는다. 기존 GLM-PCA artifact 형식과 training CLI를 가능한 한 재사용한다.

## 7. 참고 근거

- Binomial latent-factor modelling은 genotype dosage의 `0/1/2` support와 직접 맞는다:
  [Hao, Song & Storey (2016)](https://pmc.ncbi.nlm.nih.gov/articles/PMC4795615/).
- Generic discrete diffusion 자체는 이미 확립된 방법론이므로 이번 novelty의 중심으로
  두지 않는다: [D3PM](https://arxiv.org/abs/2107.03006).
- Synthetic genotype diffusion도 이미 보고되어 있어, 단순히 diffusion을 사용했다는
  사실만으로는 기여가 되지 않는다: [GeneticDiffusion (2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12261458/).

## 8. 최종 의사결정표

| 관측 결과 | 다음 행동 | 논문 포지션 |
|---|---|---|
| Gate 1 실패 | B0 유지, diffusion 수정 중단 | decoder novelty 미채택 |
| Oracle만 개선 | Phase 2 진행 | mechanism evidence만 보류 |
| Gate 2와 3 통과 | Phase 4 및 manuscript 진행 | cohort-aware local-LD decoder |
| B3-F만 우수 | Fisher를 exploratory ablation으로 보고 | Fisher를 핵심 novelty로 주장하지 않음 |
| external panel 부재 | 내부 held-out proof-of-concept으로 한정 | generalization 주장 금지 |

## 9. 개정 2 — HiPoDiT-T (2026-09-15)

### 9.1 Gate 1 판정 기록

Phase 1 oracle 비교 결과(`outputs/diagnostics/hipodit_ld_oracle_20260915/oracle_results.json`):
per-call NLL B3 0.145612 < B0 0.197060 (통과), LD `r²` MAE B3 0.060387 < B0 0.076755 (통과),
cohort AF MAE B3 0.024127 vs B0 0.020848 = **1.157배 > 1.05배 (실패)**. §5 Gate 1과 §8 첫 행에
따라 B3를 중단하고 diffusion 구조는 변경하지 않는다. negative control은 두 개 모두 방향이
맞았다(`order_control_supports_ld: true`, `label_control_supports_cohort: true`).

원인 진단과 처방 탐색은 `docs/reports/hipodit_ld_tilt_20260915.md`에 있다. 요약: tied 3×3
residual이 genotype class mass를 옮기지만 모델에 per-SNP unary 자유도가 없어 SNP별 AF를 유지할
수 없다. 또한 dependence를 쓰지 않는 calibrated independent 모델이 B3의 LD `r²` MAE와 동등하므로,
**`r²` MAE 개선은 local LD의 증거가 아니다**(§9.4).

### 9.2 새 primary model: HiPoDiT-T

> cohort-calibrated Binomial GLM-PCA latent diffusion with AF-preserving heterozygosity tilt.

`η_ij = α_j + u_{c_i,j} + v_j^T z_i`, `p_ij = σ(η_ij)` (GLM-PCA는 동결, `u`는 decoder 단계 적합),
standard latent diffusion은 변경 없음, 디코더는

    q_ij(g) ∝ C(2,g) p_ij^g (1−p_ij)^{2−g} exp(τ_j·1[g=1] + λ_ij·g),   Σ_g g·q_ij(g) = 2 p_ij

이며 `λ_ij`는 위 보존 조건의 닫힌 해다. SNP는 **독립적으로** 샘플링한다. `A_b`, genomic-order
chain, SNP 간 autoregressive dependence는 primary model에서 쓰지 않는다(B0–B3 경로는 ablation
으로 코드에 유지). `u`의 gauge는 **pooled-AF 보존 제약**으로 고정한다(개정 2a, 2026-09-15): SNP마다 스칼라
`s_j`를 두어 `u_{c,j} ← ũ_{c,j} − s_j`로 쓰고, train 행에서

    mean_{i∈train} σ(α_j + v_j^T z_i + u_{c_i,j}) = 관측 train allele frequency_j

가 성립하도록 `s_j`를 적합 중에 매번 푼다(단조 → Newton 25회, 잔차 2.2e−16). 이전 개정에서 쓰던
"train cohort 크기 가중 평균 0" (logit 공간 중심화)은 **폐기**한다. 이유는 결과가 아니라 구조다:
그 중심화는 절편이 *함께 적합될 때*의 식별성을 위한 gauge인데 여기서 `α_j`는 동결이며, logit
공간에서 평균이 0이어도 Jensen 부등식 때문에 확률 공간의 평균(=allele frequency)은 보존되지
않는다. 측정값: train pooled AF MAE가 B0 0.00000 → centered-U 0.00259로 악화됐고, 제약을 넣으면
2.2e−16로 돌아온다. tilt가 개인별 기대 dosage를 구성적으로 보존하는 것과 같은 원리를 cohort
수준에 적용한 것이다. `τ`는 AF를 바꾸지 않으므로 아무 제약도 두지 않는다.
중심화 버전(`offset_gauge="centered"`)은 ablation으로 남긴다. `τ` 범위는 per-SNP를 기본으로 한다(dev 근거:
per-SNP 0.1077 < per-superpop 0.1107 < per-cohort 0.1229).

### 9.3 새 arm 집합 (사전등록, 이후 구조 탐색 금지)

| Arm | 구성 | 답하는 질문 |
|---|---|---|
| B0 | 독립 Binomial | 기준선 |
| B1 | `+ u_{c,j}` | cohort calibration 단독 |
| **T** | `B1 + τ_j` | **primary candidate** |
| T-sp / T-co | τ를 superpop×SNP / cohort×SNP로 | τ granularity ablation |
| B1+δ | `B1 + δ_{j,g}` | AF 보존 없는 per-SNP class 보정과의 비교 — **프로토타입 증거로만 보고**(repo 코드 재실행 대상 아님), `docs/reports/prototypes/` |
| B3+δ | `B3 + δ_{j,g}` | dependence를 더하면 얼마나 더 얻는가 — 동일하게 프로토타입 증거(exploratory) |

### 9.4a Gate 1′ 1차 실행 기록 (2026-09-15, centered gauge)

centered gauge로 실행한 Gate 1′는 (a) NLL 0.107682 < 0.197060, (b) het MAE 0.014822 < 0.074741,
(c) cohort AF MAE 0.017963 = B0의 0.858배는 통과했으나 **(d) overall AF MAE 0.004846 = 1.107배로
실패**했다(`outputs/diagnostics/hipodit_t_oracle_20260915/`). 진단 결과 원인은 tilt가 아니라 `u`의
gauge였고(§9.2), 제약을 바꾼 뒤 dev에서 네 조건이 모두 통과한다(탐색적). **이 발견은 dev 결과를
본 뒤에 이뤄졌으므로 dev 수치는 확증이 아니며, 확증은 아직 열지 않은 test split의 Gate 3′에서만
가능하다.** 1차 실행 산출물은 기록으로 보존한다.

### 9.4 새 gate

- **Gate 1′ (oracle, dev)**: T가 B0 대비 (a) per-call NLL 개선, (b) heterozygosity MAE 개선,
  (c) cohort AF MAE ≤ 1.05×B0, (d) overall AF MAE ≤ 1.05×B0. 넷 다 만족해야 통과.
  LD `r²` MAE는 **판정에서 제외**하고 기술 지표로만 보고한다(§9.1). label-shuffle control은
  `u_{c,j}` 검증용으로 유지한다. order-shuffle control은 dependence가 없으므로 해당 없음.
- **Gate 2′ (end-to-end, dev)**: generated latent에서 (a)–(d)가 유지. 추가로 inverse-normalization
  경계에서 `z`의 scale/origin 일치와 generated `p`의 극단성(`|η| > 4` 비율)을 real oracle과 비교해
  기록한다. τ가 oracle `z`에 적합되었기 때문이다.
- **Gate 3′ (test, 1회)**: seed 10개 이상 paired run(seed마다 학습 1회, 같은 generated latent에
  B0와 T 디코더를 각각 적용하므로 페어링이 latent 수준에서 정확하다). T−B0의 per-call NLL,
  cohort AF MAE, heterozygosity MAE paired difference가 모두 0보다 작고 95 % bootstrap CI
  상한이 0보다 작아야 한다. overall AF MAE는 B0의 105 % 이하. **재표본 단위를 구분한다**:
  per-call NLL은 real test latent를 쓰는 oracle 경로의 결정론적 값이므로 **test 개인** 재표본
  (2,000회, seed 고정), 나머지 end-to-end 지표는 **seed** 재표본(2,000회, seed 고정)으로 CI를
  낸다. test split은 이 단계에서 한 번만 열고, 이후 어떤 구조·하이퍼파라미터도 test 결과를 보고
  바꾸지 않는다.
- **Gate 4 (privacy)**: §5 Phase 4 그대로. tilt가 het를 올리므로 nearest-neighbour distance,
  membership inference AUC, exact/near duplicate를 B0와 T 모두에서 측정한다.

### 9.5 허용·금지 문장 (§5 Phase 5 대체)

허용: "HiPoDiT-T는 cohort-calibrated Binomial base에 allele-frequency를 보존하는 heterozygosity
tilt를 결합하여, unphased genotype synthesis에서 per-SNP genotype 분포와 cohort별 allele
frequency를 동시에 보정한다."

금지: §5의 기존 금지 문장 전부에 더해 "local LD 보존", "LD-aware diffusion", 그리고 LD `r²` MAE
개선을 linkage 증거로 제시하는 모든 서술.

### 9.6 구현 파일 책임 (§6 보강)

| 책임 | 파일 |
|---|---|
| tilt decoder (arm `T`), 기존 B0–B3 유지 | `src/models/genotype_decoder.py` |
| Gate 1′ oracle 비교, study manifest | `scripts/hipodit_genotype_check.py` |
| Gate 2′ end-to-end 결합 | `scripts/hipodit_rebuild_train.py` |
| Gate 3′ paired multiseed, bootstrap CI | `scripts/hipodit_multiseed.py`, `hipodit_multiseed_summary.py` |
| privacy·strata | Phase 4 스크립트 |
