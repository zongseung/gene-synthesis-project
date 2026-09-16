# HiPoDiT-T 최종 결과 보고서 — 2026-09-16

권위 문서는 `docs/superpowers/plans/2026-09-15-hipodit-ld.md`(원본 §1–§8 + 개정 2 §9)다. 본 문서는
Phase 5 **결과 보고서**이며 논문 원고가 아니다. 모든 수치는 §12에 나열한 산출물에서 그대로 옮겼고,
파생 비율은 계산 과정을 한 번 보인다. 같은 수치는 문서 전체에서 한 번만 적는다.

---

## 1. 요약

**HiPoDiT-T**는 동결된 Binomial(2,p) GLM-PCA latent에 standard Gaussian latent diffusion을 얹고,
디코더 단계에서만 두 가지를 더한 모델이다. 첫째 cohort×SNP logit offset `u_{c,j}`로 집단별 allele
frequency를 보정하고, 둘째 per-SNP heterozygosity tilt `τ_j`로 genotype class 분포를 보정하되 개인별
기대 dosage `E[G_ij] = 2 p_ij`를 **구성적으로** 보존한다. SNP는 서로 독립으로 샘플링하며, SNP 간
autoregressive dependence·genomic-order chain·`A_b` residual은 쓰지 않는다. diffusion kernel과
GLM-PCA는 한 줄도 바뀌지 않았다. 변경된 것은 latent → genotype 디코더뿐이다.

| Gate | 대상 | 판정 | 산출물 |
|---|---|---|---|
| Gate 1′ | oracle(real held-out `z`), dev, 5 metric seeds | **통과** (4/4 조건, 5/5 seed) | `hipodit_t_oracle_20260915_pooled/` |
| Gate 2′ | end-to-end(generated latent), dev, seed 20260327 | **통과** (적용 가능한 3/3 조건) | `hipodit_t_e2e_20260916/seed_20260327/` |
| Gate 3′ | **test split, 1회 개방**, 10 seed paired | **통과** (3 gated 지표 + guardrail) | `hipodit_t_multiseed_20260915/` |
| Gate 4 | privacy·duplication, test, 10 seed | **통과** (4/4 조건) | `hipodit_t_privacy_20260915/` |

여기까지 오는 동안 사전등록 gate가 두 번 깨졌다. 원 후보 B3(pairwise local-LD residual)는 Gate 1에서
기각되었고(§9.1), 개정 2의 T도 첫 실행에서 centered gauge 때문에 Gate 1′ 조건 (d)를 통과하지
못했다(§3.3).

**주장하는 것.** 개정 2 §9.5의 허용 문장 한 개뿐이다 — HiPoDiT-T는 cohort-calibrated Binomial base에
allele-frequency를 보존하는 heterozygosity tilt를 결합하여, unphased genotype synthesis에서 per-SNP
genotype 분포와 cohort별 allele frequency를 동시에 보정한다.
**주장하지 않는 것.** local LD 보존, LD-aware diffusion, `r²` 개선을 linkage 근거로 쓰는 서술, 우월성·
신규성·privacy·외부 일반화(§11 전체 목록).
**증거의 범위.** chr17 8 genes / 361 SNPs 패널, 한 번 개방한 251명 test split, 진단용 학습량(1,000
update). 이 패널 밖으로 확장되지 않는다.

---

## 2. 모델

### 2.1 수식 (개정 2 §9.2)

base logit과 확률:

```
η_ij = α_j + u_{c_i,j} + v_j^T z_i ,      p_ij = σ(η_ij)
```

`α_j`, `v_j`, `z_i`는 train에서 적합한 Binomial GLM-PCA 산출물로 **동결**이고, `u_{c,j}`와 `τ_j`만
디코더 단계에서 적합한다.

**(i) pooled-AF 보존 gauge.** `u`는 그 자체로 식별되지 않으므로 gauge가 필요하다. SNP마다 스칼라
`s_j`를 두어 `u_{c,j} ← ũ_{c,j} − s_j`로 쓰고, train 행에서

```
mean_{i∈train} σ(α_j + v_j^T z_i + u_{c_i,j}) = 관측 train allele frequency_j
```

가 성립하도록 `s_j`를 적합 중 매번 푼다. 좌변은 `s_j`에 대해 단조 감소하므로 0에서 출발하는 Newton
반복이 수렴하며, 구현은 고정 예산 **25회**를 목적함수 그래프 안에 펼쳐 넣어 제약을 통과하여 미분한다
(`src/models/genotype_decoder.py:_NEWTON_STEPS`). 실측 잔차
`max_j |mean_i σ(·) − target_af_j|`는 arm T에서 **2.22e−16**, arm B1에서 3.33e−16이며
(`decoder_T.npz` / `decoder_B1.npz`의 `pooled_af_residual`), 1e−8을 넘으면 적합이
`converged=False`로 표시된다.

**(ii) AF 보존 heterozygosity tilt.** 디코더의 call 분포는

```
q_ij(g) ∝ C(2,g) p_ij^g (1−p_ij)^{2−g} exp(τ_j·1[g=1] + λ_ij·g),   Σ_g g·q_ij(g) = 2 p_ij
```

이고 `λ_ij`는 이 보존 조건의 **닫힌 해**다. `x_ij = e^{λ_ij} p_ij/(1−p_ij)`, `t_j = e^{τ_j}`로 두면
조건은 `(1−p)x² + t(1−2p)x − p = 0`의 양근이며 `q ∝ (1, 2tx, x²)`이다. 수치적으로는 minor 확률
`q = min(p, 1−p)` 쪽에서 풀고 `x(p) = 1/x(1−p)` 대칭으로 되돌려 양 꼬리에서 상쇄를 피한다. 격자
검증에서 `max |E[G] − 2p|`는 **4.4e−16**(τ ∈ [−4, 4] 41점, p ∈ [1e−4, 1−1e−4]), τ ∈ [−30, 30]에서도
1.1e−15이며 행 합은 2.2e−16 이내로 1이다
(`.superpowers/sdd/2026-09-15-hipodit-ld/task-7-report.md`의 검증 기록;
회귀 테스트는 `tests/test_genotype_decoder.py::test_the_tilt_keeps_expected_dosage_exact_and_is_binomial_without_a_tilt`).

**(iii) SNP 독립 샘플링.** 각 SNP의 `q_ij`에서 독립 추출한다. predecessor 조건화가 없으므로
order-shuffle control은 해당 없고, sampling에 genomic order가 개입하지 않는다.

### 2.2 파라미터와 비용

| 구성 | 파라미터 수 | 비고 |
|---|---:|---|
| `u_{c,j}` | 9,386 | 26 cohorts × 361 SNPs |
| gauge가 고정하는 자유도 | −361 | SNP당 스칼라 `s_j` 하나 |
| `u`의 유효 자유도 | 9,025 | 9,386 − 361 |
| `τ_j` | 361 | per-SNP |
| 디코더 합계 | 9,747 | 9,386 + 361 |

적합 비용은 realistic panel(train 2,002 × 361 SNPs, 26 cohorts, float64, 단일 CPU 프로세스)에서
`offset_gauge="pooled_af"` **6.1 s**, `"centered"` 1.1 s로, 제약이 약 5.5배의 비용을 가져간다
(`.superpowers/sdd/2026-09-15-hipodit-ld/task-12-report.md`의 계측). 전량이 목적함수 평가마다
펼쳐지는 25회 Newton 단계에서 나온다. diffusion 파라미터와는 별개다.

### 2.3 두 제약은 같은 원리다

`u`의 gauge와 tilt의 `λ_ij`는 서로 다른 층위에 있지만 설계 원리가 같다. **보존해야 할 moment를
penalty에 맡기지 않고 매개화 자체에 구성적으로 강제한다.** tilt는 개인·SNP 단위에서 `E[G_ij] = 2p_ij`를
닫힌 해로 못박고, gauge는 cohort를 합산한 train pooled AF를 Newton 해로 못박는다. 둘 다 "ridge가
붙은 NLL이 알아서 moment를 맞춰 주기를 바라는" 구조가 아니다.

이 구별이 실무적으로 왜 필요한지는 측정으로 드러났다. 이전 개정이 쓰던 "train cohort 크기 가중 평균
0" 중심화는 절편이 *함께 적합될 때*의 식별성 gauge인데, 여기서 `α_j`는 동결이다. 게다가 logit 공간에서
평균이 0이어도 Jensen 부등식 때문에 확률 공간의 평균(= allele frequency)은 보존되지 않는다. 같은
패널·같은 적합에서 centered gauge의 제약 잔차는 **2.25e−02**로, 위 (i)의 2.22e−16과 14자릿수 떨어져
있다(같은 task-12 계측). `τ`는 AF를 바꾸지 않으므로 아무 제약도 두지 않는다.

문헌 근거는 다음 세 건이며, `.superpowers/sdd/2026-09-15-hipodit-ld/research-*.md`에 기록된 확신도
등급을 그대로 옮긴다.

| 출처 | 쓰이는 진술 | 노트의 확신도 |
|---|---|---|
| Aitchison, J. & Silvey, S. D. (1958) *Ann. Math. Statist.* 29(3):813–828, §2 eqs (2.1)–(2.2) | 제약 하 최대우도는 score 방정식에 제약당 Lagrange 승수 하나를 더해 풀며, 해에서 제약이 정확히 성립한다 | **high** (Project Euclid 스캔 OCR, pp. 813–815 직접 판독) |
| Berger, A. L., Della Pietra, S. A. & Della Pietra, V. J. (1996) *Comput. Linguist.* 22(1):39–71, eqs (4)/(7)/(10) | maximum-entropy 제약 집합에서 모델 기대값 = 경험 기대값이며, 해는 승수가 canonical parameter에 가산으로 들어간 지수족 형태다 | **high** (전문 판독) |
| Meisner, J. & Albrechtsen, A. (2019) *Mol. Ecol. Resour.* 19(5):1144–1152, eq (4) | F-form `P(0)=(1−π)²+π(1−π)F`, `P(1)=2π(1−π)(1−F)`, `P(2)=π²+π(1−π)F`는 모든 F에서 `E[G]/2 = π` | 식·인용문 **high**(2018 bioRxiv preprint PDF), 출판본의 식 번호·표현 일치는 **medium**(출판본 미열람) |

여기서 Aitchison & Silvey + Berger는 "승수가 곧 per-SNP logit offset"임을, 즉 `s_j` 해가 임의의 수치
트릭이 아니라 제약 하 ML의 정규 해임을 보증한다. Meisner & Albrechtsen은 **F-form 자체(eq 4)의
출처**이며, tilt는 그 형태가 기술하는 "AF를 고정한 채 class mass만 옮기는" 변형을 개인별 유효 범위
안에서 매개화한 것이다. 주의 — "그 방향이 `c = (1, −2, 1)` 하나뿐"이라는 유일성 진술은 해당 논문의
문장이 아니라 문헌 노트가 **Derivation**(노트 저자가 쓴 대수)으로 명시한 결과다. 본 보고서도 그
구분을 유지한다. 또한 이 패널의 implied F는 음수 영역이라
(`docs/reports/hipodit_ld_tilt_20260915.md` §2.1의 프로토타입 측정) 고정 F 형태는 homozygote 확률을
음수로 만들 수 있고, 그래서 F가 아니라 tilt 매개화를 쓴다. 노트가 "열람하지 못함"으로 기록한
출처(Wright 1922/1949, Balding & Nichols 1995, Deville & Särndal 1992, Platt 1999, Lafferty et al. 2001
등)는 인용하지 않는다.

---

## 3. 데이터와 사전등록

### 3.1 동결 패널

| 항목 | 값 | 출처 |
|---|---|---|
| 자료 | 1000 Genomes Phase 3, chr17 | `prepare_report.json` |
| gene 수 / SNP 수 | 8 / 361 | `gene_variant_map.json`, `genotypes.npz` |
| gene별 SNP | LINC02091 23, RPH3AL 64, LOC100506388 17, LOC105371430 53, RFLNB 40, VPS53 64, TLCD3A 54, GEMIN4 46 | `gene_variant_map.json` |
| 개인 수 | 2,504 | `prepare_report.json` |
| split | train 2,002 / validation(dev) 251 / test 251, population-stratified 80/10/10, split seed 20260327 | `split_manifest.json` |
| cohort / superpopulation | 26 / 5 | `label_hierarchy.pkl` |
| MAF filter | train-only ≥ 0.01 (실측 train 최소 MAF 0.010240, 최대 0.490010, monomorphic 0개) | `prepare_report.json`, `genotypes.npz` |
| 결측 call | 0 (`calls` 배열 `(2504, 361)`에 NaN 없음, 값은 {0,1,2}) | `genotypes.npz` |
| GLM-PCA | binomial, gene당 4 components, 1,000 iterations, train에서만 적합, 8 gene 전부 수렴 | `prepare_report.json` |
| 정규화 왕복 오차 | 1.19e−07 | `prepare_report.json` |
| latent generator baseline | standard schedule | `study_manifest.json` |

Phase 0 leakage·순서 assertion은 `study_manifest.json.assertions_passed`에 **11개**가 모두 통과로
기록되어 있다: `splits_disjoint_and_complete`, `prepare_report_fit_split_is_train`,
`prepare_report_glm_family_is_binomial`, `prepare_report_maf_filter_is_train_only`,
`normalization_stats_fit_split_is_train`, `glm_pca_parameters_are_binomial_with_two_trials`,
`offsets_match_gene_variant_map`, `positions_ascend_within_every_gene`,
`observed_calls_are_diploid_dosages`, `cohort_labels_within_range`,
`label_hierarchy_maps_every_cohort`. 같은 manifest의 `decoder_inputs_from_test`는 `false`다.

정칙화 계수는 사전에 고정했다: `lambda_u = lambda_tilt = lambda_a = 1.0`. Gate 1′에서는 dev NLL로
λ를 고르는 절차를 아예 쓰지 않았다(원본 Gate 1에서 λ를 dev NLL로 고른 것이 낙관 편향이라는 리뷰
지적을 반영한 사전 결정이다).

### 3.2 사전등록 이력

| 단계 | 시점 | 결과 |
|---|---|---|
| 원본 사전등록 (§1–§8) | 기획서 초판 | primary candidate = B3 (cohort offset + local-LD residual `A_b`) |
| Gate 1 (oracle, dev) | 2026-09-15T07:40Z | **실패** — cohort AF MAE가 B0의 1.157배(기준 1.05배). §8 첫 행에 따라 B3 중단, diffusion 구조 미변경 |
| 개정 2 (§9) | 2026-09-15 | primary model을 HiPoDiT-T로 교체, arm 집합 `{B0, B1, T}`(+ ablation `T_superpop`, `T_cohort`)로 재사전등록, gate를 Gate 1′–4로 재정의, LD `r²` MAE를 판정에서 제외 |
| Gate 1′ 1차 (centered gauge) | 2026-09-15T09:24Z | **실패** — 조건 (d) overall AF MAE 1.107배 |
| 개정 2a (gauge 교체) | 2026-09-15 | logit 공간 중심화 → pooled-AF 보존 제약. `"centered"`는 ablation으로 보존 |
| Gate 1′ 최종 | 2026-09-15T14:21Z | **통과** (4/4, 5/5 seed) |

### 3.3 Gate 1′ 1차 실행(실패) 기록

`outputs/diagnostics/hipodit_t_oracle_20260915/oracle_results.json`, centered gauge, 5 seed × 20 draws.

| 조건 | arm T | B0 기준 | 판정 |
|---|---:|---:|---|
| (a) per-call NLL | 0.107682 | 0.197060 | pass |
| (b) heterozygosity MAE | 0.014822 | 0.074741 | pass |
| (c) cohort AF MAE | 0.017963 | ≤ 1.05× 0.020928 | pass |
| (d) overall AF MAE | 0.004846 | ≤ 1.05× 0.004377 | **fail** (1.107배) |

`pass_seeds: 0`, `candidate: "B0"`. 진단 결과 원인은 tilt가 아니라 `u`의 gauge였다. train에서 pooled
AF MAE는 B0가 0.00000(동결 `α_j`가 ML score 방정식으로 데이터를 맞춤)인데 centered `u`를 더하면
0.00259로 악화됐고, tilt만 켜고 `u`를 끄면 overall AF MAE는 오히려 B0보다 나았다. train 구성
재가중으로는 아무것도 바뀌지 않아 cohort 구성 문제도 아니었다. 그래서 gauge를 §2.1 (i)로 바꿨다.

**이 수정은 dev 결과를 본 뒤에 이뤄졌다.** 따라서 Gate 1′와 Gate 2′의 dev 수치는 확증이 아니라
탐색적 결과이며, 확증의 부담 전부를 그때까지 한 번도 열지 않은 test split의 Gate 3′ 한 번에 걸었다.
Gate 3′ 이후에는 어떤 구조·하이퍼파라미터도 test 결과를 보고 바꾸지 않았다.

---

## 4. Gate 1′ — oracle 비교 (dev)

real held-out GLM-PCA factor `z`를 쓰고 diffusion sample은 넣지 않는다. 5 metric seed
(20260915–20260919) × 개인당 20 draws, dev 251명. `nll_per_call`은 표본추출이 없는 결정론적 값이다.
아래 ± 는 seed 간 sample SD.

| arm | converged | nll_per_call | train nll | af_mae | cohort_af_mae | heterozygosity_mae | genotype_proportion_tv | ld_r2_mae (기술) | local_dosage_covariance_mae |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | true | 0.197060 | 0.198752 | 0.004377 ± 4.9e−05 | 0.020928 ± 7.5e−05 | 0.074741 ± 1.5e−04 | 0.075233 ± 1.2e−04 | 0.076652 ± 2.7e−04 | 0.008446 ± 8.7e−05 |
| B1 | true | 0.193457 | 0.192387 | 0.004430 ± 7.6e−05 | 0.018969 ± 1.1e−04 | 0.075795 ± 9.4e−05 | 0.076233 ± 8.3e−05 | 0.076599 ± 2.5e−04 | 0.008259 ± 6.9e−05 |
| **T** | true | **0.107359** | 0.104362 | **0.004132 ± 3.3e−05** | **0.017544 ± 4.8e−05** | **0.014623 ± 1.6e−04** | 0.015362 ± 1.4e−04 | 0.037214 ± 1.5e−04 | 0.008129 ± 3.1e−05 |
| T_superpop | true | 0.110341 | 0.107333 | 0.004176 ± 1.8e−05 | 0.017587 ± 5.4e−05 | 0.017400 ± 1.9e−04 | 0.018091 ± 1.7e−04 | 0.040381 ± 1.1e−04 | 0.008127 ± 3.3e−05 |
| T_cohort | true | 0.122396 | 0.119155 | 0.004251 ± 3.0e−05 | 0.017869 ± 3.3e−05 | 0.027494 ± 2.0e−04 | 0.028119 ± 1.9e−04 | 0.049510 ± 1.5e−04 | 0.008162 ± 3.2e−05 |

τ granularity ablation은 per-SNP가 가장 좋다. cohort당 train 행이 49–91개뿐이라(§8) τ를 세분화하면
dev에서 도리어 나빠진다. cohort 보정은 `u_{c,j}`가 담당한다.

### 4.1 네 조건 판정

`gate1_prime.reasons` 그대로, 비율은 여기서 계산한다.

| 조건 | 판정식 | 계산 | 결과 |
|---|---|---|---|
| (a) per-call NLL 개선 | T < B0 | 0.107359 / 0.197060 = 0.545배 | pass |
| (b) heterozygosity MAE 개선 | T < B0 | 0.074741 / 0.014623 = **5.11배** 축소 | pass |
| (c) cohort AF MAE | ≤ 1.05 × B0 | 0.017544 / 0.020928 = **0.838배** | pass |
| (d) overall AF MAE | ≤ 1.05 × B0 | 0.004132 / 0.004377 = **0.944배** | pass |

`passed: true`, `metric_seeds: 5`, **`pass_seeds: 5`** — 다섯 seed 전부에서 네 조건이 동시에 성립했다.
LD `r²` MAE와 local dosage covariance MAE는 개정 2 §9.4에 따라 판정에서 제외하고 기술 지표로만 싣는다
(`descriptive_only` 블록). 거리 bin은 train gene 내부 인접 SNP 거리의 quartile로 한 번 산출해 고정했고
(edges 35 / 104 / 210 bp, bin별 인접쌍 86 / 89 / 89 / 89), dev 평가에서 LD 쌍은 bin별 104 / 148 / 202 /
573개, 합 1,027쌍이 쓰였으며 건너뛴 쌍은 0이다. 모든 arm에서 `unique_individual_fraction`과
`valid_genotype_fraction`은 1.0, `cohorts_not_estimable`은 빈 목록이다.

### 4.2 AF 보존 실측

tilt의 AF 보존은 §2.1에서 해석적으로 정확하지만, 실제 draw가 그 기대값을 재현하는지 따로 측정했다.
arm T에서 표본 AF와 모델 기대 AF의 최대 절대차는 **0.004302**로, 같은 표본 크기(251 × 20 draws × 5
seeds = 25,100)에서의 draw noise 3σ **0.006695**보다 작다(`within_draw_noise: true`). 즉 관측된 편차는
유한 표본 잡음과 구분되지 않는다.

### 4.3 label-shuffle control

cohort label을 섞어 다시 적합했다. cohort 지표는 섞인 label에서 의미가 없으므로 NLL·AF·het만 남긴다.

| arm | 실제 label NLL | shuffle NLL | shuffle af_mae | shuffle het_mae |
|---|---:|---:|---:|---:|
| B1 | (§4 표) | 0.197816 | 0.004429 ± 5.0e−05 | 0.075022 ± 9.8e−05 |
| T | (§4 표) | 0.113655 | 0.004142 ± 6.6e−05 | 0.015213 ± 1.5e−04 |

B1이 B0 대비 얻던 NLL 이득은 label을 섞으면 사라지고 오히려 B0보다 나빠진다
(`label_control_supports_cohort: true`). 즉 `u_{c,j}`는 실제 cohort 정보를 쓰고 있다. T는 섞어도 B0보다
훨씬 낮은 NLL을 유지하는데, 이는 `τ_j`가 cohort와 무관한 per-SNP 항이므로 예상되는 거동이다. 이
control은 `u`의 검증용이지 `τ`의 검증용이 아니다. order-shuffle control은 T에 dependence가 없으므로
해당 없다.

---

## 5. Gate 2′ — end-to-end (dev)

같은 학습 실행이 만든 generated latent에 동일한 frozen 디코더 두 개(B0 / T)를 각각 적용한다. seed
20260327, dev 251명, DDIM 100 step, 1,000 update, 개인당 **1 draw**. cohort label은 dev split 순서
그대로의 자연 구성이며 balanced sampling을 쓰지 않았다.

| 지표 | B0 | T | T/B0 |
|---|---:|---:|---:|
| af_mae (guardrail) | 0.026459 | 0.025052 | 0.947 |
| cohort_af_mae (guardrail) | 0.072920 | 0.069141 | 0.948 |
| heterozygosity_mae | 0.091457 | 0.063237 | — (§5.2) |
| genotype_proportion_tv | 0.093532 | 0.066603 | — |
| ld_r2_mae (기술) | 0.079166 | 0.062142 | — |
| local_dosage_covariance_mae | 0.023307 | 0.022391 | — |

판정: `heterozygosity_improved` true, `cohort_af_guardrail` true, `af_guardrail` true,
`nll_not_applicable` true → **`passed: true`**. 조건 (a) per-call NLL은 end-to-end에서 정의되지 않는다.
합성 개인에게는 채점할 실제 call이 없기 때문이며, 그래서 Gate 2′는 적용 가능한 세 조건으로만 판정된다.

### 5.1 scale·extremeness 점검

`τ`가 개인 자신의 genotype으로 scoring된 oracle `z`에 적합되었으므로, generated `z`에서 과신 정도가
달라지면 het 보정이 어긋날 수 있다. 개정 2 §9.4가 기록하라고 지정한 두 검사 결과는 다음과 같다.

| 검사 | 값 |
|---|---|
| `scale_check.prepared_hashes_match` / `real_factor_max_abs_diff` | true / **0.0** |
| same-scale 재구성 RMSE (mean / std) | 0.076998 / 0.048563 |
| `|η| > 4` 비율 — real oracle / generated | 0.653707 / 0.618865 (비율 0.946702) |
| `|η| > 6` 비율 — real / generated | 0.426030 / 0.416859 |
| mean `|η|` — real / generated | 5.458955 / 5.472499 |
| 표본 AF vs 기대 AF 최대차 / draw noise 3σ | 0.031216 / 0.066946 (`within_draw_noise: true`) |

generated `p`의 극단성은 real oracle보다 오히려 약간 낮고 평균 `|η|`는 사실상 같다. 우려했던 과신
악화는 나타나지 않았다. 다만 AF 보존 점검은 draws=1이라 허용 폭이 넓어(3σ가 §4.2의 약 10배) 검정력이
낮다.

### 5.2 oracle 대비 이득 축소

디코더의 이득은 generated latent로 넘어가면 크게 줄어든다. 이 축소는 결과와 같은 문장에서 말해야 할
사실이다.

| 비교 | oracle (real `z`, dev, 20 draws × 5 seeds) | end-to-end (generated `z`, dev, 1 draw) |
|---|---:|---:|
| heterozygosity MAE 개선 배수 (B0 ÷ T) | 5.11배 | **1.45배** |
| cohort AF MAE, arm T 절대값 | 0.017544 (§4) | 0.069141 (§5) |

즉 het 보정의 유효성은 4분의 1 이하로 떨어지고, cohort AF의 절대 오차는 약 4배로 커진다.

### 5.3 남은 오차의 지배적 원인은 디코더가 아니라 latent generator다

같은 실행에서 (1) 같은 generated latent에 디코더만 바꿨을 때의 차이와 (2) 같은 T 디코더에 latent만
real → generated로 바꿨을 때의 차이를 나란히 측정할 수 있다. real latent 기준값은 동일 보고서의
`real_latent_decoder_reference`(B0) / `real_latent_decoder_reference_T`(T)다.

| 지표 | B0 on real `z` | T on real `z` |
|---|---:|---:|
| af_mae | 0.008939 | 0.006528 |
| cohort_af_mae | 0.037783 | 0.023526 |
| heterozygosity_mae | 0.075819 | 0.018177 |

(이 값들은 draws=1·단일 seed라 §4의 oracle 표와 설정이 달라 직접 비교 대상이 아니다.)

| 지표 | (1) 디코더 효과 = generated에서 B0 − T | (2) 생성기 효과 = T에서 generated − real | (2)/(1) |
|---|---:|---:|---:|
| af_mae | 0.026459 − 0.025052 = 0.001407 | 0.025052 − 0.006528 = 0.018524 | **13.2배** |
| cohort_af_mae | 0.072920 − 0.069141 = 0.003779 | 0.069141 − 0.023526 = 0.045614 | **12.1배** |
| heterozygosity_mae | 0.091457 − 0.063237 = 0.028220 | 0.063237 − 0.018177 = 0.045061 | **1.60배** |

AF 계열에서는 latent generator가 만드는 오차가 디코더 선택이 만드는 차이의 12–13배다.
heterozygosity에서만 둘이 같은 자릿수인데, 이는 het가 디코더가 직접 조작하는 양이기 때문이다.
결론은 단순하다 — **이 파이프라인의 end-to-end 오차 예산은 디코더가 아니라 latent generator가
지배한다.** 디코더를 더 정교하게 만드는 것으로 좁힐 수 있는 여지는 위 (1)열 크기까지다.

---

## 6. Gate 3′ — 확증 (test split, 1회 개방)

test split은 이 단계에서 처음이자 마지막으로 열렸다. seed 10개(20260327–20260336), seed마다 학습 1회,
같은 generated latent에 frozen B0와 T 디코더를 각각 적용하므로 **페어링이 latent 수준에서 정확**하다.
디코더는 Gate 1′ 산출물에서 로드했으며 재적합하지 않았다(`decoder_sha256` 일치, §12). Bootstrap은
2,000 draws, percentile 95 %, bootstrap seed 20260915.

**재표본 단위를 구분한다.** per-call NLL은 real test latent를 쓰는 oracle 경로의 결정론적 값이므로
**test 개인 251명**을 재표본한다(비율-합 통계량, 유효 call 90,611 = 251 × 361). 나머지는 모두
end-to-end 지표이므로 **seed**를 재표본한다.

### 6.1 판정 지표

| 지표 | 재표본 단위 | B0 | T | T−B0 mean | SD | 95 % CI | 판정 |
|---|---|---:|---:|---:|---:|---|---|
| per-call NLL | test 개인 | 0.201223 | 0.109625 | **−0.091599** | — | [−0.095635, −0.087636] | pass |
| cohort AF MAE | seed | 0.072580 | 0.068010 | **−0.004570** | 0.000443 | [−0.004824, −0.004321] | pass |
| heterozygosity MAE | seed | 0.096375 | 0.066927 | **−0.029448** | 0.001136 | [−0.030140, −0.028800] | pass |
| overall AF MAE (guardrail) | seed | 0.024379 | 0.024214 | −0.000165 | 0.000525 | **[−0.000478, +0.000149]** | guardrail 충족 |

세 판정 지표는 평균과 95 % CI 상한이 모두 0보다 작다. `gate3_prime.passed: true`.

**overall AF MAE는 개선이라고 말할 수 없다.** guardrail 조건(T ≤ 1.05 × B0, 실측 0.024214 / 0.024379 =
0.993배)은 충족하지만 paired difference의 95 % CI가 0을 걸치고, seed별 승패도 T 5 / B0 5로 갈린다.
따라서 이 지표에 대해 본 보고서가 말하는 것은 "**악화되지 않았다**"까지이며, 그 이상은 없다.

### 6.2 seed별 승패와 기술 지표

`paired_differences_decoder.csv`의 seed별 부호를 센 결과다.

| 지표 | T가 이긴 seed 수 |
|---|---|
| cohort AF MAE | 10/10 |
| heterozygosity MAE | 10/10 |
| genotype proportion TV | 10/10 |
| LD `r²` MAE (기술) | 10/10 |
| local dosage covariance MAE | 9/10 |
| overall AF MAE | 5/10 |

| 기술 지표 (판정 제외) | B0 | T | T−B0 mean | SD | 95 % CI |
|---|---:|---:|---:|---:|---|
| genotype_proportion_tv | 0.098419 | 0.070356 | −0.028063 | 0.001083 | [−0.028690, −0.027409] |
| ld_r2_mae | 0.075894 | 0.060756 | −0.015138 | 0.001755 | [−0.016165, −0.014105] |
| ld_r2_mae bin 0 | 0.147294 | 0.114490 | −0.032804 | 0.005410 | [−0.035576, −0.029013] |
| ld_r2_mae bin 1 | 0.097688 | 0.073154 | −0.024534 | 0.003890 | [−0.026778, −0.022324] |
| ld_r2_mae bin 2 | 0.064883 | 0.054352 | −0.010531 | 0.002599 | [−0.012074, −0.008894] |
| ld_r2_mae bin 3 | 0.061204 | 0.050059 | −0.011145 | 0.002027 | [−0.012291, −0.010036] |
| local_dosage_covariance_mae | 0.022109 | 0.021381 | −0.000728 | 0.000488 | [−0.001024, −0.000443] |

LD `r²` MAE는 모든 거리 bin에서 줄었지만 **개정 2 §9.4·§9.5에 따라 판정에서 제외되고 linkage의 근거로
쓰이지 않는다.** 그 이유는 §9.3의 반례가 보여준다. summary JSON 자체도
`"LD r-squared is descriptive, not evidence of linkage"`로 기록되어 있다.

### 6.3 population classifier (fidelity 지표)

real 대 synthetic 구분이 아니라, 합성 개인의 cohort를 얼마나 맞힐 수 있는지다.

**사전등록 대비 편차.** 계획 §5 Phase 3은 "real-vs-synthetic population-classifier AUC/accuracy"를
요구한다. 이 연구가 보고하는 것은 real·synthetic을 구별하는 discriminator AUC가 **아니라**, real train
dosage에 적합한 multinomial logistic cohort 분류기를 real test와 두 합성 arm에 각각 적용한 accuracy와
macro one-vs-rest AUC다. 원 문구가 중의적이어서 이 해석으로 확정한 기록이 ledger의 R4 ruling이며, 그
확정은 측정 이전이다. 계획 문구만 읽는 독자가 discriminator AUC를 찾지 않도록 여기에 명시한다.

| 대상 | accuracy | macro AUC |
|---|---:|---:|
| real test 251명 (상한 참조) | 0.270916 | 0.896576 |
| B0 합성 | 0.125498 | 0.724419 |
| T 합성 | 0.261355 | 0.855341 |
| T−B0 (seed paired, 95 % CI) | +0.135857, [0.117131, 0.154980] | +0.130922, [0.124791, 0.136526] |

`cohorts_excluded_from_auc`는 빈 목록이고 `n_eval`은 251이다. B0의 합성 표본은 cohort 신호를 거의 잃는
반면 T의 합성 표본은 real에 훨씬 가까운 수준까지 cohort 귀속 가능성을 유지한다. **이것은 fidelity
지표로만 보고한다.** 같은 수치를 "T가 덜 익명적이다"라는 privacy 결론으로 쓰지 않는다. privacy 판정은
§7의 Gate 4 항목뿐이며, privacy report도 이 값을 `context_from_phase3`에
`used_in_gate4: false`로 격리해 담고 있다.

---

## 7. Gate 4 — privacy와 duplication

test split, 10 seed, arm당 seed마다 합성 251명. member 2,002명(train 전체), non-member 251명, SNP 361개.
near-duplicate 임계값은 real held-out 개인의 train 최근접 거리 1퍼센타일(0.031856)로 정했다.

**판정 상수의 출처.** 아래 네 상수(`AUC_MARGIN = 0.02`, exact duplicate 하한 `1/n_synthetic`,
`NEAR_DUPLICATE_FACTOR = 1.5`와 하한 `0.01`, `P1_FACTOR = 0.8`)는 **사전등록의 일부가 아니다.**
개정 2 §9.4의 Gate 4는 "§5 Phase 4 그대로"이고 §5의 Gate 4는 "B0보다 현저히 나빠지면"이라는 정성적
규칙이므로, 이 네 값은 그 정성적 규칙을 하나의 방식으로 조작화한 것이며 측정 이전에 태스크 브리프에서
고정되었다(브리프 00:55, 측정 01:08). 다른 조작화를 택했다면 다른 여유 폭이 나왔을 수 있다.

| 조건 | B0 mean (SD) | T mean (SD) | T−B0 mean (SD) | 한계 (B0에서 유도) | 판정 |
|---|---:|---:|---:|---|---|
| (a) membership inference AUC | 0.492674 (0.004681) | 0.499478 (0.008013) | +0.006805 (0.006089) | ≤ B0 + 0.02 = 0.512674 | pass |
| (b) exact duplicate rate (train) | 0.0 (0.0) | 0.0 (0.0) | 0.0 | ≤ 1/251 = 0.003984 | pass |
| (c) near duplicate rate (train) | 0.0 (0.0) | 0.0 (0.0) | 0.0 | ≤ 0.015 | pass |
| (d) NN distance p1 | 0.102909 (0.006787) | 0.093629 (0.004088) | −0.009280 (0.007782) | ≥ 0.8 × B0 = 0.082327 | pass |

`gate4.passed: true`. exact / near / self duplicate는 두 arm 10 seed 전부에서 **0건**이다. MIA AUC는
양쪽 모두 우연 수준(0.5)에서 각각 0.007 / 0.001 이내이며, 두 arm 모두 공격이 성립하지 않는다.

### 7.1 방향성 비용 — T는 train에 더 가깝다

Gate 4는 통과했지만, 통과 여부와 별개로 일관된 방향이 있다. **T의 합성 표본은 모든 분위수, 10 seed
전부, 26 cohort 전부에서 train 패널에 약 10 % 더 가깝다.**

| nearest-neighbour distance to train | B0 mean (SD) | T mean (SD) | T−B0 mean (SD) |
|---|---:|---:|---:|
| min | 0.085596 (0.010068) | 0.079501 (0.011313) | −0.006094 (0.010910) |
| p5 | 0.122992 (0.002978) | 0.110803 (0.002992) | −0.012188 (0.001937) |
| median | 0.177285 (0.002920) | **0.158726** (0.003467) | −0.018560 (0.002281) |
| mean | 0.178341 (0.002242) | 0.160408 (0.001892) | −0.017934 (0.000912) |

중앙값 비율 0.158726 / 0.177285 = 0.895, 평균 비율 0.160408 / 0.178341 = 0.899 — 두 경우 모두 약 10 %
축소다(p1은 §7의 (d)행). cohort별 중앙값도 26/26에서 T가 더 작다(§8).

이 방향성을 보정해 읽을 기준값은 real held-out 251명이 train에 대해 갖는 최근접 거리다: mean 0.106025,
median 0.105263, min **0.022161**, p5 0.047091. T에서 가장 train에 가까운 합성 개인의 거리 0.079501은
실제 held-out 개인 중 가장 가까운 사람의 거리 0.022161의 0.079501 / 0.022161 = **3.59배**다. 즉 절대
수준에서는 합성 표본이 여전히 실제 개인보다 훨씬 멀고, (d) 조건도 0.093629 / 0.082327 = 1.137, 즉
13.7 % 여유로 통과한다. 그러나 tilt 강도를 올릴 경우 가장 먼저 압박받을 지표가 이것이다.

### 7.2 한계

MIA 설정은 공격자에게 다소 불리하게, 즉 **결과에는 유리하게 잡혀 있다.** 합성 행이 eval split의
cohort label을 조건으로 생성되었으므로 non-member 집합은 합성 표본과 구성이 일치하는 반면, member
집합은 train 패널 전체다. 더 깨끗한 공격은 새 표본을 생성해야 하는데 Phase 4는 그러면 안 된다.
따라서 §7의 (a)는 "이 설정에서 우연 수준"이라는 뜻이며, MIA가 불가능함을 뜻하지 않는다.
formal differential privacy는 학습에 어떤 privacy mechanism이나 budget도 없었으므로 주장하지 않는다.

---

## 8. Strata

### 8.1 cohort 26행 (test, 10 seed 평균)

`n_train` / `n_eval`은 각 cohort의 train·평가 행 수다. 세 지표 모두 낮을수록 좋고, 마지막 두 열은
§7.1의 방향성 비용을 cohort 단위로 내린 것이다.

| cohort | pop | superpop | n_train | n_eval | af_mae B0 | af_mae T | het_mae B0 | het_mae T | NN median B0 | NN median T |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | ACB | AFR | 76 | 10 | 0.077618 | 0.075000 | 0.147867 | 0.133324 | 0.205817 | 0.188504 |
| 1 | ASW | AFR | 49 | 6 | 0.106279 | 0.098038 | 0.178947 | 0.161357 | 0.197922 | 0.183241 |
| 2 | BEB | SAS | 68 | 9 | 0.076039 | 0.068775 | 0.133549 | 0.114035 | 0.171745 | 0.150970 |
| 3 | CDX | EAS | 75 | 9 | 0.062388 | 0.057725 | 0.138443 | 0.118467 | 0.161773 | 0.136011 |
| 4 | CEU | EUR | 79 | 10 | 0.069155 | 0.068310 | 0.095097 | 0.091801 | 0.163850 | 0.150000 |
| 5 | CHB | EAS | 83 | 10 | 0.061773 | 0.056496 | 0.133573 | 0.110443 | 0.167036 | 0.145291 |
| 6 | CHS | EAS | 83 | 11 | 0.059997 | 0.055175 | 0.116167 | 0.102241 | 0.170083 | 0.149030 |
| 7 | CLM | AMR | 76 | 9 | 0.076516 | 0.072145 | 0.134749 | 0.116836 | 0.167590 | 0.152632 |
| 8 | ESN | AFR | 79 | 10 | 0.076219 | 0.070956 | 0.127895 | 0.114654 | 0.200831 | 0.184211 |
| 9 | FIN | EUR | 79 | 10 | 0.065346 | 0.061579 | 0.145263 | 0.127479 | 0.166343 | 0.151939 |
| 10 | GBR | EUR | 73 | 9 | 0.079055 | 0.074731 | 0.141120 | 0.119852 | 0.163435 | 0.144044 |
| 11 | GIH | SAS | 83 | 10 | 0.073601 | 0.072604 | 0.154958 | 0.138283 | 0.170083 | 0.148892 |
| 12 | GWD | AFR | 91 | 11 | 0.075082 | 0.072639 | 0.122060 | 0.109418 | 0.204155 | 0.185042 |
| 13 | IBS | EUR | 85 | 11 | 0.065147 | 0.060778 | 0.128481 | 0.106447 | 0.173961 | 0.154571 |
| 14 | ITU | SAS | 82 | 10 | 0.073837 | 0.070374 | 0.143518 | 0.124294 | 0.165235 | 0.148892 |
| 15 | JPT | EAS | 83 | 10 | 0.063089 | 0.055831 | 0.127950 | 0.109446 | 0.163573 | 0.144737 |
| 16 | KHV | EAS | 79 | 10 | 0.065263 | 0.059861 | 0.116011 | 0.098726 | 0.167729 | 0.147784 |
| 17 | LWK | AFR | 79 | 10 | 0.085139 | 0.079806 | 0.148670 | 0.125706 | 0.202909 | 0.181994 |
| 18 | MSL | AFR | 68 | 9 | 0.078547 | 0.072130 | 0.132902 | 0.117544 | 0.203601 | 0.181163 |
| 19 | MXL | AMR | 52 | 6 | 0.083818 | 0.080702 | 0.122761 | 0.115882 | 0.169945 | 0.148338 |
| 20 | PEL | AMR | 68 | 9 | 0.064158 | 0.060126 | 0.135149 | 0.116374 | 0.170360 | 0.154294 |
| 21 | PJL | SAS | 76 | 10 | 0.061842 | 0.057105 | 0.134543 | 0.115706 | 0.171330 | 0.154571 |
| 22 | PUR | AMR | 83 | 10 | 0.080166 | 0.076150 | 0.122382 | 0.117895 | 0.178255 | 0.157895 |
| 23 | STU | SAS | 82 | 10 | 0.068878 | 0.063906 | 0.150443 | 0.129307 | 0.170222 | 0.151524 |
| 24 | TSI | EUR | 85 | 11 | 0.066444 | 0.061861 | 0.106094 | 0.092647 | 0.173961 | 0.152078 |
| 25 | YRI | AFR | 86 | 11 | 0.071695 | 0.065462 | 0.114933 | 0.099698 | 0.203601 | 0.184765 |
| unseen | — | — | — | — | — | — | — | — | — | — |

읽히는 것은 세 가지다. 첫째, **T는 26/26 cohort에서 AF MAE와 het MAE를 모두 개선한다.** 예외가 없다.
둘째, `n_eval`이 6–11명으로 균일하게 작아 `not estimable`로 표기된 행이 하나도 없다. 셋째, `unseen`
행은 "이 패널에 unseen cohort가 없음"이라는 표시이며(`status: no unseen cohort in this panel`)
빈칸이 결측을 뜻하지 않는다. unseen / rare cohort에 대한 failure mode는 이 패널에서 측정 불가다.

### 8.2 MAF bin (test, 10 seed 평균)

| MAF bin | SNP 수 | af_mae B0 | af_mae T | 상대 변화 | het_mae B0 | het_mae T | 상대 변화 |
|---|---:|---:|---:|---:|---:|---:|---:|
| [0.01, 0.05) | 140 | 0.009230 | 0.009128 | −1.1 % | 0.018512 | 0.017225 | −7.0 % |
| [0.05, 0.2) | 122 | 0.028705 | 0.028693 | −0.04 % | 0.100274 | 0.075282 | −24.9 % |
| [0.2, 0.5] | 99 | 0.040470 | 0.040028 | −1.1 % | 0.201678 | 0.126915 | **−37.1 %** |

het 상대 변화는 (T − B0)/B0이다. 예: [0.2, 0.5]에서 (0.126915 − 0.201678)/0.201678 = −0.371.

이 패턴은 AF 보존 tilt의 **예상 거동과 정확히 일치한다**. tilt는 기대 dosage를 건드리지 않으면서
class mass만 옮기므로 AF 계열은 세 bin 모두 평평하고(±1 % 수준), het 이득은 `p(1−p)`가 큰 common SNP에
집중된다. rare bin에서 이득이 작은 것도 같은 이유다. 즉 이 표는 **설계와 기계적으로 정합한다**는
확인이며, 그 이상은 아니다. 과적합된 보정도 같은 층화 패턴을 낳으므로, 이 표를 "우연한 적합이 아니다"
라는 근거로 쓰지 않는다.

---

## 9. 음성 결과와 기각된 가설

이 절은 본 연구의 주요 기여 중 하나다. HiPoDiT-T 채택 여부와 무관하게 유효한 발견들이며, 삭제 대상이
아니다.

### 9.1 (a) B3 local-LD residual은 Gate 1에서 기각되었다

`outputs/diagnostics/hipodit_ld_oracle_20260915/oracle_results.json`, dev, seed 20260915, 20 draws,
λ = (1, 1). 원본 사전등록 §3.3의 arm 집합 전체다.

| arm | nll_per_call | af_mae | cohort_af_mae | heterozygosity_mae | genotype_proportion_tv | ld_r2_mae | local_dosage_covariance_mae |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0 | (§4와 동일한 결정론적 값) | 0.004445 | 0.020848 | 0.074764 | 0.075220 | 0.076755 | 0.008492 |
| B1 | 0.194041 | 0.005113 | 0.019264 | 0.074847 | 0.075611 | 0.076197 | 0.008445 |
| B2 (`A_b`만) | 0.150859 | 0.013294 | 0.026040 | 0.032931 | 0.035968 | 0.062697 | 0.015119 |
| B3 (`u + A_b`) | 0.145612 | 0.013704 | **0.024127** | 0.031675 | 0.035359 | 0.060387 | 0.013851 |

| Gate 1 조건 | 계산 | 결과 |
|---|---|---|
| (a) per-call NLL 개선 | 0.145612 < B0 | pass |
| (b) 거리층화 LD `r²` MAE 개선 | 0.060387 < 0.076755 | pass |
| (c) cohort AF MAE ≤ 1.05 × B0 | 0.024127 / 0.020848 = **1.157배** | **fail** |

`fixed_candidate: "B0"`. 기획서 §5 Gate 1과 §8 첫 행에 따라 B3를 중단하고 diffusion 구조는 변경하지
않았다. 음성 대조는 둘 다 방향이 맞았다(`order_control_supports_ld: true`,
`label_control_supports_cohort: true`). B3는 heterozygosity와 class TV를 크게 개선하면서도 SNP별 AF를
무너뜨렸고, 이는 구조적 이유 때문이다. genotype 3-class에서 AF를 바꾸지 않고 class mass를 옮기는
방향은 `c = (1, −2, 1)` 하나뿐인데(문헌 노트가 Derivation으로 기록한 대수이며 인용 논문의 진술이
아니다, §2.3), 곱셈형 tilt가 그 방향과 나란하려면 `p_ij`에 의존하는 조건을 만족해야 한다. 모든 SNP가 공유하는 하나의 3×3 table은 그 조건을 동시에 만족시킬 수 없고, `α_j`는
동결, `u_{c,j}`는 SNP별 gauge에 묶여 있어 이를 흡수할 unary 자유도가 남아 있지 않았다.

### 9.2 (b) fitted `A_b`는 LD가 아니라 class 보정으로 동작했다

`A_b`를 "class 성분"(predecessor `h`에 대한 평균, 중심화)과 "concordance 성분"(대각 평균 − 비대각
평균)으로 분해한 결과다. 이 분해는 프로토타입 스크립트(`docs/reports/prototypes/`)만 수행했고 repo
코드에 대응 산출물이 없으므로 **프로토타입 증거로만 인용한다**(`docs/reports/hipodit_ld_tilt_20260915.md`
§2.2).

| bin (거리) | class 성분 (g = 0 / 1 / 2) | 대각 평균 − 비대각 평균 |
|---|---|---:|
| 0 (≤ 35 bp) | −0.224 / +0.706 / −0.482 | +0.657 |
| 1 (35–104) | −0.162 / +0.892 / −0.729 | +0.408 |
| 2 (104–210) | −0.031 / +0.800 / −0.770 | +0.326 |
| 3 (> 210) | +0.064 / +0.762 / −0.826 | +0.238 |

concordance 성분이 거리와 함께 단조 감소하는 것은 LD decay와 부합한다. 그러나 절대적으로는 class
성분(het를 +0.7 ~ +0.9 밀어 올리고 g = 2를 −0.5 ~ −0.8 누름)이 지배적이다.

effect size는 repo 코드의 order-shuffle control로 직접 계산할 수 있다. gene 내부 SNP 순서를 섞고
(position과 거리 bin은 유지) 다시 적합한 arm의 NLL은 B3 0.150247, B2 0.155206이다. B3가 dependence를
쓰지 않는 쌍둥이(B1) 대비 얻은 NLL 이득 중 순서에 의존하는 몫은

```
(0.150247 − 0.145612) / (0.194041 − 0.145612) = 0.0957 → 9.6 %
```

에 불과하다. LD `r²` MAE로 같은 계산을 하면 순서 섞은 B3가 0.063765이므로
(0.063765 − 0.060387) / (0.076197 − 0.060387) = 0.214, 즉 21.4 %다. **두 지표 모두에서 `A_b` 이득의
대부분은 genomic order와 무관했다.** `A_b`는 LD가 아니라 class 보정으로 일하고 있었다.

프로토타입이 추가로 보여준 것은, per-SNP class 자유도를 먼저 준 뒤에는 `A_b`의 *잔여* 이득이 대부분
순서 의존이 된다는 점이다(NLL 66.8 %, LD `r²` 79.6 %). 즉 실제 local LD 신호는 존재하지만 작고, 앞선
모델에서는 class 보정에 가려져 있었다.

### 9.3 (c) 결정적 반례 — dependence 없이도 LD `r²` MAE는 개선된다

프로토타입의 `B1 + δ_{j,g}`(per-SNP per-class offset, SNP 간 의존성 전혀 없음, 완전 독립 샘플링)는
LD `r²` MAE가 **0.06123**이다. 같은 지표에서 B3는 repo 코드 기준 0.060387(§9.1)로, 사실상 같고 둘 다
B0의 0.076755보다 크게 낮다. T는 여기서 더 내려간다(§4). `r²`는 분모에 각 SNP의 분산이 들어가므로
**heterozygosity 보정만으로도 개선된다.**

따라서 `r²` MAE 감소를 "local LD 보존"의 증거로 제시할 수 없다. 원본 기획서 §5 Phase 1의 Gate 1 지표
(b)는 이 교란을 전제하지 않았고, 개정 2 §9.4가 LD `r²` MAE를 판정에서 제외한 근거가 바로 이것이다.
본 보고서의 §4·§5·§6에 실린 모든 `r²` 수치는 기술 지표이며 linkage 근거로 쓰이지 않는다.

(비교 주의: `B1 + δ_{j,g}` 0.06123은 프로토타입 전용 수치이고 repo 코드에 대응 arm이 없다. B3·B0·T는
repo 코드 값이다. 프로토타입은 B0/B1/B3의 dev NLL을 repo 산출물과 소수점 6자리까지 재현했으므로
계산 경로는 같지만, 두 출처를 섞은 비교라는 점을 명시한다.)

### 9.4 (d) Fisher schedule은 5 seed에서 일관된 이득이 없다

이전 파일럿(`outputs/diagnostics/hipodit_multiseed_20260915/summary.json`, seeds 20260327–20260331,
`docs/reports/hipodit_multiseed_20260915.md`)의 결과다. 이 때문에 본 연구는 standard schedule을
`latent_generator_baseline`으로 고정했다.

| 지표 | standard mean ± SD | fisher mean ± SD | paired Δ (fisher − standard) mean ± SD | fisher 승 |
|---|---:|---:|---:|---:|
| AF MAE | 0.024100 ± 0.001650 | 0.024311 ± 0.002556 | +0.000211 ± 0.003298 | 2/5 |
| local dosage covariance MAE | 0.021654 ± 0.001599 | 0.021509 ± 0.002150 | −0.000144 ± 0.001864 | 2/5 |

Fisher는 AF MAE 평균이 더 높고, covariance MAE는 평균이 근소하게 낮지만 두 지표 모두 5 seed 중 2회만
이겼다. 5 seed에서 유의성 검정은 보고하지 않는다.

---

## 10. 한계

1. **표본.** dev 251명 / test 251명, cohort당 평가 행 6–11명(§8.1). cohort별 지표의 불확실성이 크다.
2. **패널.** chr17 8 genes / 361 SNPs. 전장유전체, 다른 염색체, MAF < 0.01의 rare variant(필터로 제외),
   phased haplotype에 대해서는 아무것도 주장하지 않는다.
3. **LD 지표의 성격.** `r²`는 unphased dosage의 composite LD이며 haplotype LD가 아니다. 게다가
   §9.3의 교란 때문에 이 연구에서는 linkage 근거로 쓸 수 없다.
4. **학습량.** 1,000 update, DDIM 100 step은 **진단용** 설정이다. 논문용 학습량이 아니며, §5.3이 보인
   대로 end-to-end 오차는 이 학습량의 latent generator가 지배한다.
5. **external cohort 없음.** 독립 cohort나 별도 genomic panel이 없으므로 외부 일반화는 주장하지
   않는다. 확보되면 allele/strand/build harmonization을 먼저 검증하고 같은 frozen 디코더로 한 번만
   평가해야 한다. 이 패널에는 unseen cohort가 없어 그 failure mode도 측정되지 않았다.
6. **`τ`의 적합 대상.** `τ`는 개인 자신의 관측 call로 scoring된 oracle `z`에 적합되었다. §5.1에서
   과신 악화는 관측되지 않았지만, 이는 이 패널·이 학습량에서의 관측이지 일반 보장이 아니다.
7. **formal DP 미주장.** 학습에 privacy mechanism도 budget도 없었다(§7.2). Gate 4는 경험적 공격과
   중복 측정일 뿐이며, MIA 설정 자체가 결과에 유리하게 잡혀 있다.
8. **gauge 수정이 post-hoc이다.** pooled-AF gauge는 Gate 1′ 1차 실패를 본 뒤에 도입되었다(§3.3).
   설계 근거는 결과가 아니라 구조(§2.3)지만, dev 수치가 확증이 아니라는 사실은 변하지 않는다. 확증의
   전부는 Gate 3′ 한 번이며, 그 한 번은 판정 지표 3개에서 통과하고 guardrail 1개에서 "악화되지 않음"에
   머물렀다(§6.1).
9. **Gate 2′의 검정력.** end-to-end 판정은 단일 seed·1 draw로, AF 보존 점검의 허용 폭이 넓다(§5.1).
10. **재현성의 종류.** seed는 통제했지만 결정론적 알고리즘을 요구하지 않았으므로, 이 실행들은 seed
    시행의 재현이지 bitwise determinism의 주장이 아니다.
11. **판정 지표의 성격.** Gate 1′·Gate 3′가 판정하는 네 지표는 모두 디코더가 **직접 파라미터로 갖고
    있는** 양이다. per-call NLL은 적합 목적함수 자체이고, heterozygosity MAE는 `τ_j`가 오직 그것을
    움직이려고 존재하며, cohort AF MAE는 `u_{c,j}`가, overall AF MAE는 pooled-AF gauge가 구조적으로
    고정한다. 따라서 이 조건들은 **보정의 일반화 확인**(train에서 맞춘 보정이 held-out에서도 유지되는가)
    이지, 모델이 측정되지 않은 다른 무언가를 대가로 지불하지 않았다는 확인이 아니다. 특히 조건 (b)는 이
    모델 계열에서 tautology에 가깝다. 그 측면을 덮는 것은 기술 지표와 Gate 4뿐이며, 실제로 §7.1은 모델이
    nearest-neighbour 거리에서 약 10 %를 지불했음을 보여준다. 측정되지 않은 비용이 더 있을 수 있다.

---

## 11. 주장 범위

### 11.1 허용 문장 (개정 2 §9.5, 원문 그대로)

> "HiPoDiT-T는 cohort-calibrated Binomial base에 allele-frequency를 보존하는 heterozygosity tilt를
> 결합하여, unphased genotype synthesis에서 per-SNP genotype 분포와 cohort별 allele frequency를 동시에
> 보정한다."

이 한 문장이 본 연구가 주장하는 전부다. §1 요약과 §4–§8의 서술은 모두 이 범위 안에 있다. 원본 §5
Phase 5의 허용 문장("HiPoDiT-LD는 … local LD preservation을 분리 검증한다")은 B3 기각(§9.1)으로
무효이며 본 보고서에 쓰이지 않는다.

### 11.2 금지 문장 목록과 확인

| # | 금지 문장 | 출처 | 본 보고서에서 |
|---|---|---|---|
| 1 | "최초의 genotype diffusion" | §5 | 주장하지 않음 |
| 2 | "phased haplotype LD 보존" | §5 | 주장하지 않음 (§10.3) |
| 3 | "whole-genome generation" | §5 | 주장하지 않음 (§10.2) |
| 4 | "Fisher schedule이 우수" | §5 | 주장하지 않음 — 반대 방향의 음성 결과만 보고 (§9.4) |
| 5 | "DP / anonymous synthetic data" | §5 | 주장하지 않음 (§7.2, §10.7) |
| 6 | "local LD 보존" | §9.5 | 주장하지 않음 (§9.2, §9.3) |
| 7 | "LD-aware diffusion" | §9.5 | 주장하지 않음 — diffusion kernel은 변경되지 않았다 |
| 8 | LD `r²` MAE 개선을 linkage 증거로 제시하는 모든 서술 | §9.5 | 주장하지 않음 — `r²`는 §4·§5·§6 전부에서 기술 지표로만 표시하고, §9.3에 반례를 함께 실었다 |

추가로 다음 세 가지도 하지 않았다. (i) 우월성 주장 — B0 대비 차이는 사전등록 gate 조건의 충족 여부로만
서술하고, overall AF MAE처럼 CI가 0을 걸치는 지표는 "악화되지 않음"까지만 말했다(§6.1).
(ii) privacy 주장 — population classifier AUC는 fidelity 지표로만 쓰고 privacy 결론으로 전용하지
않았다(§6.3). (iii) 일반화 주장 — external cohort가 없으므로 내부 held-out proof-of-concept으로
한정한다(§10.5).

---

## 12. 재현

### 12.1 실행한 명령

모두 저장소 루트에서 실행했다. GPU는 NVIDIA RTX A6000, PyTorch 2.4.1+cu121 / CUDA 12.1.

```bash
# 동결 패널 준비 (prepare). --max-variants는 실행 기록에 남아 있지 않다 (아래 단락 참조).
.venv/bin/python scripts/hipodit_rebuild_check.py prepare \
  --output-dir outputs/diagnostics/hipodit_fisher_20260915_unique \
  --genes 8 --components 4 --glm-iterations 1000 --seed 20260327 \
  --max-variants 64

# Gate 1 (원본, B0-B3). 기록 당시 CLI에는 --arms가 없었고,
# 현재 HEAD에서는 --arms legacy로 동일 산출물이 재현된다(byte-for-byte 확인).
.venv/bin/python scripts/hipodit_genotype_check.py oracle \
  --prepared-dir outputs/diagnostics/hipodit_fisher_20260915_unique \
  --output-dir outputs/diagnostics/hipodit_ld_oracle_20260915

# Gate 1' 1차 (centered gauge, 실패 기록)
.venv/bin/python scripts/hipodit_genotype_check.py oracle \
  --prepared-dir outputs/diagnostics/hipodit_fisher_20260915_unique \
  --output-dir outputs/diagnostics/hipodit_t_oracle_20260915

# Gate 1' 최종 (pooled-AF gauge). 기본값: --arms t --metric-seeds 5 --draws 20
#   --seed 20260915 --n-bins 4 --lambda-u 1.0 --lambda-tilt 1.0 --lambda-a 1.0
.venv/bin/python scripts/hipodit_genotype_check.py oracle \
  --prepared-dir outputs/diagnostics/hipodit_fisher_20260915_unique \
  --output-dir outputs/diagnostics/hipodit_t_oracle_20260915_pooled

# Gate 2' (end-to-end, dev)
CUDA_VISIBLE_DEVICES=0 /home/user/Envs/csdi/bin/python scripts/hipodit_rebuild_check.py train \
  --output-dir outputs/diagnostics/hipodit_fisher_20260915_unique \
  --run-dir outputs/diagnostics/hipodit_t_e2e_20260916/seed_20260327 \
  --seed 20260327 --schedule standard --steps 1000 --batch-size 64 --samples 251 \
  --ddim-steps 100 --device cuda \
  --decoder-dir outputs/diagnostics/hipodit_t_oracle_20260915_pooled

# Gate 3' (test split, 1회)
CUDA_VISIBLE_DEVICES=0 /home/user/Envs/csdi/bin/python scripts/hipodit_multiseed.py \
  --mode decoder \
  --prepared-dir outputs/diagnostics/hipodit_fisher_20260915_unique \
  --decoder-dir outputs/diagnostics/hipodit_t_oracle_20260915_pooled \
  --eval-split test \
  --output-dir outputs/diagnostics/hipodit_t_multiseed_20260915 \
  --seeds 20260327 20260328 20260329 20260330 20260331 20260332 20260333 20260334 20260335 20260336 \
  --steps 1000 --batch-size 64 --samples 251 --ddim-steps 100 --device cuda

# Gate 4 (privacy + strata)
.venv/bin/python scripts/hipodit_privacy.py \
  --prepared-dir outputs/diagnostics/hipodit_fisher_20260915_unique \
  --multiseed-dir outputs/diagnostics/hipodit_t_multiseed_20260915 \
  --output-dir outputs/diagnostics/hipodit_t_privacy_20260915

# 구현 스위트
.venv/bin/python -m pytest -q -p no:cacheprovider tests/
```

**`--max-variants 64`는 기록된 사실이 아니라 추정값이다.** `--genes 8`, `--components 4`,
`--glm-iterations 1000`, `--seed 20260327`은 `prepare_report.json`에 그대로 들어 있지만,
`--max-variants`는 `prepare_report.json`에도 `study_manifest.json`에도 본 문서에도 기록된 적이 없다
(현재 HEAD에서는 `prepare_report.json`이 `max_variants`를 기록하도록 고쳤으나, 동결 패널은 그 이전에
만들어졌으므로 소급되지 않는다). `gene_variant_map.json`에서 직접 확인할 수 있는 것은 이것뿐이다:
8개 유전자의 variant 수가 순서대로 23 / **64** / 17 / 53 / 40 / **64** / 54 / 46이고, 최댓값 64를 두
유전자(RPH3AL, VPS53)가 정확히 공유하며, 64를 넘는 유전자는 없다. `_gene_matrix`는
`len(columns) == max_variants`에서 루프를 끊으므로 cap은 **64 이상**이어야 한다. 서로 독립인 두
유전자가 같은 값 64에서 멈춘 것은 cap이 정확히 64였다고 볼 때 가장 단순하게 설명되며, CLI 기본값
128이었다면 두 유전자가 우연히 같은 값에 걸릴 이유가 없다. 이상이 추정의 전부이고, 재현하는 쪽은
얻은 패널의 `prepared_sha256`이 `study_manifest.json`의 값과 같은지로 확인해야 한다.

### 12.2 런타임과 seed

| 단계 | 런타임 | seed |
|---|---|---|
| Gate 1 (B3) | 1 m 02 s (CPU) | 20260915 |
| Gate 1′ 1차 | 31.36 s (CPU), peak RSS 1.62 GB | 20260915–20260919 |
| Gate 1′ 최종 | manifest 생성 14:20:29Z → 결과 기록 14:21:51Z, 즉 약 82 s (CPU) | 20260915–20260919 |
| Gate 2′ | wall 22.4 s, 보고된 `runtime_seconds` 20.804 (GPU) | 20260327 |
| Gate 3′ | wall 236 s, 그중 10개 자식 프로세스 합 221.459 s (GPU) | 20260327–20260336 |
| Gate 4 | real 3.18 s / user 6.95 s / sys 0.36 s (CPU) | 위 10개 seed 재사용 |
| pytest | 24.64 s | — |

Gate 3′는 20개 ledger 이벤트, 10개 `complete`, 실패 0, seed 탈락 0, 재시도 없음이다.

### 12.3 커밋 (branch `hipodit-ld`, 오래된 것부터)

| 커밋 | 내용 |
|---|---|
| `c6b5f02` | HiPoDiT rebuild 기반(binomial GLM-PCA, rebuild/multiseed 스크립트, 테스트) |
| `e1f624d` | cohort-calibrated local-LD genotype decoder (B0–B3) |
| `0ceece5` | genotype metrics와 Phase 1 oracle decoder 비교 |
| `dbb1576` | AF 보존 heterozygosity-tilt decoder arm (T) |
| `92c3e0e` | Gate 1′ oracle을 HiPoDiT-T arm 집합에 대해 실행 |
| `d847fe4` | cohort offset을 train pooled allele frequency로 gauge |
| `31d00c8` | generated latent를 HiPoDiT-T arm으로 디코딩하고 Gate 2′ 판정 |
| `33dc42d` | held-out test split에서 HiPoDiT-T를 B0와 대조 확증 |
| `6d8833c` | Phase 4 privacy·strata 측정과 Gate 4 판정 |

각 커밋은 구현 후 별도의 독립 검증 단계를 거쳐 해당 태스크 브리프 대비 확인되었고, 그 과정에서
두 차례 문제가 발견되어 수정 후 재검증되었다. 최종 스위트는 **224 passed, 1 warning**이며, 그 하나는
`tests/test_genotype_decoder.py`의 기존 torch CUDA-driver `UserWarning`이다. §12.2의 런타임과
peak RSS는 각 실행 기록(`runs.jsonl`, 진단 보고서의 `runtime_seconds`, 태스크 실행 기록)에서 옮겼다.

### 12.4 산출물 SHA-256

본 문서 작성 시점에 `sha256sum`으로 직접 계산했다.

| 산출물 | SHA-256 |
|---|---|
| `hipodit_t_oracle_20260915_pooled/oracle_results.json` | `678fa679a3669c23a8adac0c1bdcdf09cebcabd6404db65c44e59e12e950823a` |
| `hipodit_t_oracle_20260915_pooled/study_manifest.json` | `5cc1fec2a5e7d7c2d82d1763a2e6746aa1520cfabcc0262a2bb7ef65cfc9a3ac` |
| `hipodit_t_oracle_20260915_pooled/decoder_ablation.csv` | `dbe9d763019be6a1674b4111017bcdf434bb8e113c7b4a73206eadb0f09c9321` |
| `hipodit_t_oracle_20260915_pooled/strata.csv` | `553c754abbc069564ab458197c05fa5c595b67a2d75da2ca9887bd5ec992e0ef` |
| `hipodit_t_oracle_20260915_pooled/decoder_T.npz` | `6e65544d086841b13ac6b09a90c67e6d2a23ab5a6f56bb2243e28acbd614f0c7` |
| `hipodit_t_oracle_20260915/oracle_results.json` (1차, 실패) | `ee7bfa570e282532a74b658fe3b37b014ead81bfea173ac90a7486be97ee713e` |
| `hipodit_ld_oracle_20260915/oracle_results.json` (Gate 1, 실패) | `bc8c33dee942edd4cf0337eb1c89d36f08643c62b4763fd9d79270e9a7e825c3` |
| `hipodit_t_e2e_20260916/seed_20260327/diagnostic_report.json` | `f20732b5f971a90b7a99325fbafe5a92dd968b6851dcb363e2af202a1489d6bc` |
| `hipodit_t_multiseed_20260915/summary_decoder.json` | `ff7bdb823f3252823e602a52a9209a56fa3a43c27040bd421d4d0d6a1ad71104` |
| `hipodit_t_multiseed_20260915/summary_decoder.csv` | `771e9dd6accb919b3ac79b38eec54b782b64cf03d5fb70cd7c0e6fa941d4ea86` |
| `hipodit_t_multiseed_20260915/paired_differences_decoder.csv` | `797565911f267145953ef006ae35cb2443b41164a1f789d0e584f37de1b298b7` |
| `hipodit_t_multiseed_20260915/manifest.json` | `f977163cb3a88fc05d2fbdb485c57a33fe1f9030cf11ad63d6fdd004dd2a1510` |
| `hipodit_t_multiseed_20260915/runs.jsonl` | `2470b3da2f81ce2dcffc275f40ccf09c320962da739ced42588a4ca7a1ea8136` |
| `hipodit_t_privacy_20260915/privacy_report.json` | `a4f92828fda47d71b2db091702318e50d5410e934625a3eca501ed91a67bf53f` |
| `hipodit_t_privacy_20260915/privacy_summary.csv` | `58b12ffa97d2c29d2d547ad963f8efdd1efe5611959da52dbbbf7c8712008beb` |
| `hipodit_t_privacy_20260915/strata_cohort.csv` | `c526bd39bd65239c5239911ddb6f35e76d2e30e77cd961a1712f688a32e0d227` |
| `hipodit_t_privacy_20260915/strata_maf.csv` | `e04346a144920b00116da36fdd95390aea2e99fb7d99269ec8ae6c24b972462d` |

동결 입력 패널(`outputs/diagnostics/hipodit_fisher_20260915_unique/`)의 14개 파일 SHA-256은
`study_manifest.json`과 `manifest.json`의 `prepared_sha256` 블록에 기록되어 있으며, Gate 3′의
`alignment` 항목이 모든 실행에서 이들이 변하지 않았음을 확인한다. 디코더는 Gate 1′ 산출물에서
로드되었고 재적합되지 않았다 — Gate 2′·Gate 3′가 기록한 `decoder_sha256`은 위 `decoder_T.npz`의
해시와 같다.

**`verify_fingerprints`의 현재 상태.** 이 함수는 manifest의 `source_sha256`까지 대조하므로, 실행 이후
소스가 바뀌면 실패한다.

- 파일럿 디렉터리 `outputs/diagnostics/hipodit_multiseed_20260915`에 대해서는 **지금 실패한다.** 그
  실행 뒤에 `scripts/hipodit_genotype_check.py`를 비롯한 6개 스크립트가 바뀌었기 때문이며, 따라서
  그 파일럿은 현재 HEAD에서 제자리 재요약(re-summarise)이 불가능하다. §9.4에 인용한 파일럿 수치는
  당시 기록된 산출물에서 옮긴 것이다. Gate 1′ 실행은 최종 리뷰에서 (이번 수정 이전 HEAD 기준)
  byte-for-byte 재현이 확인되었다. 이번 수정이 `src/models/genotype_decoder.py`의 수렴 판정 기준을
  바꾸었지만 적합 파라미터는 건드리지 않으며, 이 연구가 적합한 네 arm(T/snp, T/cohort, T/superpop,
  B1)이 새 기준에서도 모두 `converged: True`임을 실측으로 확인했다(여유 545–14,282배). 다만
  `study_manifest.json`은 소스 지문을 담으므로 재실행하면 그 지문 줄은 달라진다.
- Gate 3′ 디렉터리 `outputs/diagnostics/hipodit_t_multiseed_20260915`는 본 보고서 작성 시점까지는
  통과했으나, 최종 리뷰 후속 수정으로 `scripts/hipodit_multiseed_summary.py`,
  `scripts/hipodit_rebuild_prepare.py`, `src/models/genotype_decoder.py` 세 파일이 바뀌면서
  **이제 실패한다.** 산출물 자체는 위 SHA-256 그대로 손대지 않았고 `prepared_sha256`도 전부 일치하며,
  달라진 것은 소스 지문뿐이다. 즉 이 디렉터리 역시 현재 HEAD에서 제자리 재요약이 불가능하고, §6의
  수치는 재요약이 아니라 동결된 `summary_decoder.json`에서 읽어야 한다. (해당 수정 자체는
  `summary_decoder.json`이 기록하는 내용을 바꾸므로, 재요약하면 Gate 2′ per-seed 기록이 추가된
  다른 파일이 나온다. 산출물은 재생성하지 않았다.)

### 12.5 관련 문서

| 문서 | 역할 |
|---|---|
| `docs/superpowers/plans/2026-09-15-hipodit-ld.md` | 사전등록 원본(§1–§8)과 개정 2(§9) |
| `docs/reports/hipodit_ld_tilt_20260915.md` | Gate 1 실패 진단, 처방 후보 비교, 프로토타입 ablation(§4.4 부록 포함) |
| `docs/reports/hipodit_multiseed_20260915.md` | standard/Fisher 5-seed 파일럿 |
| `.superpowers/sdd/2026-09-15-hipodit-ld/research-lit-af-drift.md` | AF drift·marginal 보존 문헌 노트(확신도 등급 포함) |
| `.superpowers/sdd/2026-09-15-hipodit-ld/research-cohort-calibration.md` | 제약 하 ML·cohort calibration 문헌 노트 |
| `docs/reports/prototypes/` | 프로토타입 스크립트와 JSON(§9.2, §9.3의 프로토타입 전용 수치) |

---

## 13. 다음 단계 (결정이 아니라 선택지)

1. **패널 확장 시 gate 재적용.** 다른 염색체나 더 많은 gene으로 넓히면 Gate 1′–4를 다시 적용해야
   한다. 현재 결과는 chr17 8-gene 패널에 한정되며, 새 패널의 결과를 보고 구조를 바꾸면 그 순간
   확증이 아니라 탐색으로 되돌아간다.
2. **전장유전체에서는 SNP 단위 병렬 적합이 필요하다.** T의 목적함수는 SNP별로 완전히 분리된다.
   `τ_j`는 SNP j의 call에만, `u_{·,j}`와 gauge 스칼라 `s_j`도 SNP j에만 걸린다. 따라서 361 SNP에서
   6.1 s(§2.2)인 적합을 수백만 SNP로 그대로 늘리는 대신, SNP를 축으로 샤딩해 병렬로 적합하는 것이
   자연스러운 경로다. 현재 구현은 dense `(N, J, 3)` 텐서를 쓰므로 그 지점에서 메모리 재설계가 함께
   필요하다.
3. **논문용 학습량 재설정.** §5.3이 보인 대로 end-to-end 오차는 latent generator가 지배한다. 디코더를
   더 다듬기 전에 1,000 update 진단 설정을 논문용 학습량으로 올리고 §5.3의 (1)/(2) 분해를 다시
   측정하는 편이 정보량이 크다.
4. **marginal 보존 dependence는 별도 계획 개정 사안.** `docs/reports/hipodit_ld_tilt_20260915.md` §4.4는
   각 인접쌍의 3×3 결합분포를 Sinkhorn 사영으로 양쪽 margin에 맞추면 dependence를 넣어도 cohort AF가
   거의 움직이지 않으며, 그 구성에서는 `r²` 개선의 90.1 %가 순서 의존이 되어 §9.3의 교란이 해소된다고
   기록한다. 즉 LD를 주장하려면 그런 구성이어야 한다. 다만 이는 사전등록 arm 집합(개정 2 §9.3) 밖이고
   적합 비용도 크므로, 채택하려면 계획을 다시 개정하고 gate를 다시 정의해야 한다. 본 연구는 이
   구성을 쓰지 않았다.
5. **tilt 강도를 올릴 경우 감시할 지표.** §7.1의 nearest-neighbour 분포다. 현재 (d) 조건의 여유는
   13.7 %이고, T는 이미 모든 분위수에서 B0보다 train에 가깝다.
