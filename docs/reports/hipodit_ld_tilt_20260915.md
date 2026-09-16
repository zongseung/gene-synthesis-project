# HiPoDiT-T 모델 재정의: AF-preserving heterozygosity calibration — 2026-09-15

기획서 `docs/superpowers/plans/2026-09-15-hipodit-ld.md`의 Gate 1 실패(2026-09-15)에 대한 원인
진단, 문헌 근거, 처방 후보 비교, 그리고 primary model 재정의 기록이다. 성능 향상 주장이 아니다.
아래 수치는 모두 dev(=validation) split에서 나온 **탐색적** 결과이며, 확증은 §6의 재검증 절차를
통과한 뒤에만 가능하다.

## 1. 요약

| 항목 | 내용 |
|---|---|
| Gate 1 판정 | 실패. B3(local-LD residual)는 per-call NLL·LD r² MAE는 개선했으나 cohort AF MAE를 B0의 1.157배로 악화(기준 1.05배) |
| 실패 원인 | tied 3×3 residual이 genotype class mass를 옮기는데, 모델에 **per-SNP unary 자유도가 없어** 각 SNP의 allele frequency를 제자리에 유지할 수 없다 |
| 처방 후보 | (a) per-SNP allele offset `δ_j`, (b) per-SNP class offset `δ_{j,g}`, (c) AF 보존 heterozygosity tilt `τ_j`. dev 5 seeds에서 **세 후보 모두 Gate 1 기준 통과(5/5)** |
| 채택 | (c) 단독. **HiPoDiT-T** = cohort-calibrated Binomial GLM-PCA latent diffusion + AF-preserving heterozygosity tilt, SNP 독립 샘플링 |
| 부수 발견 (중요) | dependence를 전혀 쓰지 않은 calibrated independent 모델이 B3의 LD r² MAE와 동등(0.0612 vs 0.0605). **LD r² MAE 개선분의 대부분은 marginal calibration**이며 linkage 모델링의 증거가 아니다 |
| novelty 재정의 | "LD-aware diffusion"이 아니라 "latent genotype synthesis에서의 AF-preserving heterozygosity calibration" |

## 2. 실패 원인 진단 (측정값)

### 2.1 Binomial base는 heterozygote를 과소예측한다

frozen panel(`outputs/diagnostics/hipodit_fisher_20260915_unique`, chr17 8 genes / 361 SNPs /
2,504명, split 2,002/251/251)에서 `p_ij = σ(α_j + v_j^T z_i)`로 계산한 값이다.

| 측정 | train | dev |
|---|---:|---:|
| 예측 het 비율 − 실제 het 비율 (평균) | −0.0746 | −0.0736 |
| het를 과대예측한 SNP 비율 | 0.02 | 0.11 |
| SNP별 implied `F = 1 − het_real/het_pred` 중앙값 (IQR) | −0.600 (−0.831, −0.240) | −0.606 (−0.834, −0.271) |
| genotype class 비율 (0/1/2) 실제 | 0.692 / 0.184 / 0.124 | — |
| genotype class 비율 (0/1/2) B0 20 draws | 0.729 / 0.111 / 0.161 | — |

`F < 0`은 Binomial(2, p_ij) 대비 **heterozygote 초과**를 뜻한다. 설명 가설: oracle `z_i`는 그
개인의 관측 genotype으로 scoring되므로 homozygote에서 `p`가 0/1 쪽으로 과신된다. 실제로 train
에서 관측 homozygote 셀의 평균 예측 het는 0.035, 관측 heterozygote 셀은 0.448이다. 이 가설은
검증되지 않았고, 대안(저차원 GLM-PCA가 못 잡는 잔여 구조)과 구분되지 않는다.

### 2.2 fitted `A_b`는 대부분 class 보정, 일부만 LD

B3의 `A_b`를 "class 성분"(h에 대한 평균, 중심화)과 "concordance 성분"(대각−비대각 평균)으로
분해한 값이다.

| bin (거리) | class 성분 (g=0/1/2) | 대각 평균 − 비대각 평균 |
|---|---|---:|
| 0 (≤35 bp) | −0.224 / +0.706 / −0.482 | +0.657 |
| 1 (35–104) | −0.162 / +0.892 / −0.729 | +0.408 |
| 2 (104–210) | −0.031 / +0.800 / −0.770 | +0.326 |
| 3 (>210) | +0.064 / +0.762 / −0.826 | +0.238 |

concordance 성분이 거리와 함께 단조 감소하는 것은 LD decay와 부합한다. 그러나 절대적으로는
class 성분(het를 +0.7~0.9 밀어 올리고 g=2를 −0.5~−0.8 누름)이 지배적이다. 결과적으로 B3의
pooled class 비율은 실제와 정확히 일치(0.692/0.183/0.125)하지만, SNP별 AF는 어긋난다.

### 2.3 드리프트의 방향과 위치

| 측정 | B0 | B3 |
|---|---:|---:|
| AF MAE (dev, 20 draws) | 0.0046 | 0.0137 |
| minor-AF SNP에서 평균 AF 드리프트 | −0.0004 | +0.0060 |
| major allele 쪽으로 드리프트한 SNP 비율 | 0.47 | 0.21 |
| gene 첫 SNP(=`A` 미적용, 8개)의 AF 절대오차 | — | 0.0051 |
| chain이 걸린 SNP(353개)의 AF 절대오차 | — | 0.0139 |

드리프트는 **minor allele 쪽(AF 0.5 방향)** 으로 일어난다. het를 올리면 0→1, 2→1로 mass가
이동하고, minor-AF SNP에서는 0→1 쪽이 많아 AF가 올라간다. `A`가 적용되지 않는 gene 첫 SNP는
B0 수준을 유지하므로, 드리프트의 원인이 `A`임이 위치로도 확인된다.

### 2.4 왜 tied table로는 고칠 수 없는가

genotype 3-class에서 AF를 바꾸지 않고 class mass를 옮기는 방향은 `c = (1, −2, 1)` 하나뿐이다
(`Σ_g c_g = 0`, `Σ_g g·c_g = 0`을 동시에 만족하는 유일한 방향, §3의 F-form). 곱셈형 tilt
`exp(a_g)`가 AF 중립이 되려면 `a_g − ā ∝ c_g / P_p(g)`여야 하고, 이 조건은 `p_ij`에 의존한다.
따라서 모든 SNP에 공유되는 하나의 3×3 table은 각 SNP에서 동시에 AF 중립일 수 없다. `α_j`는
GLM-PCA에서 동결되고 `u_{c,j}`는 SNP별로 중심화되어 있으므로, 모델에는 이를 흡수할 unary
자유도가 남아 있지 않다.

## 3. 문헌 근거 (primary sources)

상세 인용·확신도는 `.superpowers/sdd/2026-09-15-hipodit-ld/research-lit-af-drift.md`에 있다.

| 주제 | 핵심 진술 | 출처 | 본 건에의 적용 |
|---|---|---|---|
| moment matching | maximum-entropy/ML 해에서는 각 feature의 모델 기대값이 학습표본 기대값과 같다 | Berger, Della Pietra & Della Pietra (1996) *Comput. Linguist.* 22(1), eq. 3/4/13/14; Hastie et al. ESL §4.4.1 eq. (4.21) | per-SNP unary 항을 넣으면 train에서 SNP별 allele/class count가 맞는다. 단 **관측된 predecessor 조건부**이고 ridge가 있으면 근사 |
| F-form | `P(0)=(1−π)²+π(1−π)F`, `P(1)=2π(1−π)(1−F)`, `P(2)=π²+π(1−π)F` | Meisner & Albrechtsen (2019) *Mol. Ecol. Resour.* 19:1144, eq. 4 (bioRxiv 10.1101/468611) | 모든 F에서 `E[G]/2 = π` (대수 확인). AF 중립 방향은 F-form이 유일 |
| F-form의 한계 | 음수 F는 homozygote 확률을 음수로 만들 수 있어 절단·재정규화가 필요 | 같은 문헌 | 본 패널은 `F ≈ −0.6`이고 `p_ij`가 0/1에 가까워 **고정 F 형태는 무효**. 그래서 tilt 매개화가 필요 |
| structured HWE | `logit(π_ij) = Σ_k a_ik h_kj`, `G_ij ~ Binomial(2, π_ij)`; sHWE 2-df count 검정 | Hao & Storey (2019) *Genetics* 213:759; Hao, Song & Storey (2016) *Bioinformatics* 32:713 | 개인별 AF 모델에서 HWE 편차를 검정하는 표준 틀. 두 논문 모두 편차의 **방향**은 진술하지 않음(전문 grep 확인) |
| marginal 보존 구성 | latent normal thresholding에서 "estimated allele frequencies are not affected by this approximation, and are unbiased" | Montana (2005) HapSim | margin을 구성적으로 고정하고 dependence만 따로 주는 접근의 선례 |
| count 기반 chain | 조건부 확률을 phased 입력의 count에서 직접 추정 | Li & Li (2008) GWAsimulator | `T(h→g) = n(h,g)/n(h)`는 경험적 marginal을 재현(귀납 증명은 위 노트의 derivation). tie·smoothing을 하면 보장 소멸 |

## 4. 처방 후보 비교 (dev, 탐색적)

프로토타입은 repo 코드가 아니라 별도 스크립트에서 실행했고, 계산 경로는 frozen panel·기존
`genotype_metrics`·기존 decoder와 동일하다(B0/B1/B3의 dev NLL이 oracle 산출물과 소수점
6자리까지 일치: 0.197060 / 0.194041 / 0.145612). 공통 설정: train 2,002명 teacher-forced 적합,
ridge λ=1, LBFGS(strong Wolfe, float64), dev 251명에 대해 개인당 20 draws, seed 5개
(20260915+0..4), 모든 arm에 동일 난수 스트림.

| arm | 추가 파라미터 | dev NLL | AF MAE | cohort AF MAE | het MAE | class TV | LD r² MAE |
|---|---|---:|---:|---:|---:|---:|---:|
| B0 (독립 Binomial) | — | 0.197060 | 0.00445 | 0.02104 | 0.07490 | 0.0754 | 0.07681 |
| B1 (`u_{c,j}`) | 9,386 | 0.194041 | 0.00506 | 0.01949 | 0.07494 | 0.0756 | 0.07614 |
| B3 (`u + A_b`) | +36 | 0.145612 | 0.01362 | 0.02426 | 0.03161 | 0.0355 | 0.06047 |
| B3 + `δ_j` | +361 | 0.133698 | 0.00474 | 0.02002 | 0.01913 | — | 0.05428 |
| B1 + `δ_{j,g}` (dependence 없음) | +722 | 0.136266 | 0.00460 | 0.02020 | 0.00952 | — | 0.06123 |
| B3 + `δ_{j,g}` | +758 | 0.127974 | 0.00473 | 0.02010 | 0.00975 | — | 0.05215 |
| **T = B1 + tilt `τ_j`** | **+361** | **0.107682** | 0.00488 | **0.01804** | 0.01490 | — | **0.03673** |

seed별 Gate 1 기준(NLL 개선 + LD r² MAE 개선 + cohort AF MAE ≤ 1.05×B0) 통과 횟수와 cohort AF
비율:

| arm | 통과 | cohort AF 비율 (seed별) |
|---|---|---|
| B3 | 0/5 | 1.144 / 1.150 / 1.162 / 1.147 / 1.163 |
| B1 | 5/5 | 0.922 / 0.929 / 0.924 / 0.926 / 0.931 |
| B3 + `δ_j` | 5/5 | 0.952 / 0.953 / 0.952 / 0.949 / 0.953 |
| B3 + `δ_{j,g}` | 5/5 | 0.956 / 0.956 / 0.956 / 0.949 / 0.961 |
| T | 5/5 | 0.852 / 0.857 / 0.858 / 0.855 / 0.866 |

B0의 cohort AF MAE는 5 seed에서 0.02104 ± 0.00013이므로, B3의 1.15배 악화는 draw noise 범위를
훨씬 넘는다. Gate 1 실패는 재현된다.

### 4.1 LD 기여 분해 (order-shuffle control)

gene 내부 SNP 열을 섞고(positions·거리 bin은 그대로) 다시 적합한 뒤, 그 arm의 "dependence 없는
쌍둥이" 대비 이득에서 순서 의존분이 차지하는 비율을 계산한다. NLL은 표본추출이 없어 결정론적이다.

| 비교 | dependence 없는 baseline | 실제 순서 | 순서 섞음 | 순서 의존 비율 |
|---|---:|---:|---:|---:|
| B3 (NLL) | B1 0.194041 | 0.145612 | 0.150247 | (0.150247−0.145612)/(0.194041−0.145612) = **9.6 %** |
| B3+`δ_{j,g}` (NLL) | 0.136266 | 0.127974 | 0.133515 | **66.8 %** |
| B3+`δ_{j,g}` (LD r² MAE) | 0.06123 | 0.05215 | 0.05938 | **79.6 %** |

두 가지가 동시에 읽힌다. 첫째, per-SNP class 자유도를 주기 전의 B3는 이득의 90 %가 genomic
order와 무관했다. 즉 `A_b`는 LD가 아니라 class 보정으로 일하고 있었다. 둘째, class 자유도를 준
뒤에는 `A_b`의 잔여 이득이 대부분 순서 의존이 된다. 즉 **실제 local LD 신호는 존재하지만 작다.**

### 4.2 결정적 반례: dependence 없이 얻는 LD r² 개선

`B1 + δ_{j,g}`는 SNP를 완전히 독립으로 생성하는데도 LD r² MAE가 0.06123으로, B3(0.06047)와
사실상 같고 B0(0.07681)보다 크게 낮다. T는 여기서 더 내려가 0.03673이다. `r²`는 분모에 각
SNP의 분산이 들어가므로 heterozygosity 보정만으로도 개선된다. 따라서 **`r²` MAE 감소를
"local LD 보존"의 증거로 제시할 수 없다.** 기획서 §5 Phase 1의 Gate 1 지표 (b)는 이 교란을
전제하지 않았다.

### 4.2.1 T의 상세 지표 (dev, 5 seeds 평균)

| 지표 | B0 | B3 | T |
|---|---:|---:|---:|
| genotype class TV | 0.0754 | 0.0353 | 0.0160 |
| signed local dosage covariance MAE | 0.00850 | 0.01379 | 0.00837 |
| LD r² MAE, bin 0/1/2/3 | 0.1718 / 0.0989 / 0.0637 / 0.0585 | 0.1351 / 0.0763 / 0.0494 / 0.0468 | 0.1032 / 0.0421 / 0.0281 / 0.0263 |
| AF MAE, MAF bin [.01,.05)/[.05,.2)/[.2,.5] | 0.0033 / 0.0042 / 0.0064 | 0.0090 / 0.0142 / 0.0194 | 0.0038 / 0.0048 / 0.0065 |

B3는 covariance MAE와 모든 MAF bin의 AF MAE를 악화시키지만, T는 covariance를 B0 수준으로
유지하면서(0.00837 vs 0.00850) class TV와 거리 bin별 r² MAE를 모두 낮춘다. 적합된 τ의
5/50/95 분위수는 0.603 / 2.651 / 3.803으로 전 SNP에서 het를 올리는 방향이며, §2.1의 `F ≈ −0.6`
(heterozygote 초과)과 부호가 일치한다.

`A_b`의 concordance 성분(대각−비대각)은 per-SNP class 자유도를 준 뒤 실제 순서에서
1.70 / 1.22 / 0.51 / 0.66(bin 0–3), 순서를 섞으면 0.82 / 0.63 / 0.36 / 0.57로 떨어진다. 순서
정보가 실재함을 보여주지만, §4.2대로 그 기여는 지표 개선분의 일부다.

### 4.3 tilt granularity

| τ 범위 | τ 파라미터 | dev NLL | cohort AF MAE | het MAE | LD r² MAE |
|---|---:|---:|---:|---:|---:|
| per-SNP | 361 | **0.107682** | 0.01804 | 0.01490 | 0.03673 |
| per-superpop × SNP | 1,805 | 0.110666 | 0.01810 | 0.01739 | 0.03994 |
| per-cohort × SNP | 9,386 | 0.122852 | 0.01841 | 0.02698 | 0.04898 |
| per-SNP, `u_{c,j}` 제거 | 361 | 0.112317 | 0.01958 | 0.01509 | 0.03804 |

τ를 세분화하면 dev에서 더 나빠진다(cohort별 train row가 49–91개뿐). 따라서 τ는 per-SNP로 두고,
cohort 보정은 `u_{c,j}`가 담당한다. `u_{c,j}` 제거 시 NLL 0.0046, cohort AF MAE 0.0015이
나빠지므로 "cohort-calibrated"라는 이름은 `u`에서 나온다.

### 4.4 (부록) dependence를 marginal 보존 방식으로 더하면

채택 모델에는 포함되지 않지만, "AF를 해치지 않으면서 dependence를 넣을 수 있는가"를 확인했다.
각 인접쌍에서 3×3 결합분포를 Sinkhorn 사영으로 양쪽 margin(`π_{i,j−1}`, `π_{i,j}`)에 맞추고,
`A_b`는 odds-ratio(결합의 gauge-불변 성분)만 담당하게 한 구성이다. 수렴하면 chain의 marginal이
목표 margin과 정확히 일치한다(독립 검증: 40 SNP 전파 후 최대 오차 1.2e−14).

| arm | dev NLL | AF MAE | cohort AF MAE | het MAE | LD r² MAE | Sinkhorn col 오차 |
|---|---:|---:|---:|---:|---:|---:|
| B1 + Sinkhorn `A_b` (tilt 없음) | 0.162501 | 0.00525 | 0.02015 | 0.07524 | 0.05691 | 1.9e−2 |
| T (채택, dependence 없음) | 0.107682 | 0.00488 | 0.01804 | 0.01490 | 0.03673 | — |
| T + Sinkhorn `A_b` | 0.101736 | 0.00488 | 0.01822 | 0.01578 | 0.02925 | 8.4e−4 |
| T + Sinkhorn `A_b`, 순서 섞음 | 0.105135 | 0.00487 | 0.01809 | 0.01526 | 0.03599 | 4.0e−7 |

읽을 점 세 가지다. 첫째, margin을 구성적으로 고정하면 dependence를 넣어도 cohort AF가 거의
움직이지 않는다(0.01804 → 0.01822). B3의 실패는 dependence 자체가 아니라 그 매개화 때문이었다.
둘째, T 위에 dependence를 더해 얻는 이득은 NLL 0.0059, LD r² MAE 0.0075이며, 그중 순서 의존분이
NLL 57.2 %, **LD r² 90.1 %**다. 즉 이 구성에서는 `r²` 개선을 linkage 근거로 쓸 수 있다(§4.2의
교란이 해소된다). 셋째, 비용은 디코더 안의 반복 사영(적합 약 44분 CPU, 50회 반복에서 column
잔차 8.4e−4)이다.

채택 모델은 이 구성을 **쓰지 않는다**. 사전등록 arm 집합(개정 2 §9.3) 밖이며, 채택하려면 계획을
다시 개정하고 Gate를 다시 정의해야 한다. 여기서는 "LD를 주장하려면 어떤 구성이어야 하는가"에
대한 근거 기록으로만 남긴다.

## 5. 채택 모델: HiPoDiT-T

### 5.1 정의

> **HiPoDiT-T**: cohort-calibrated Binomial GLM-PCA latent diffusion with AF-preserving
> heterozygosity tilt.

```
G_discrete → Binomial GLM-PCA → standard latent diffusion → cohort-calibrated heterozygosity decoder → G̃_discrete
```

1. **입력** `G_ij ∈ {0,1,2}` (unphased dosage), cohort `c_i`.
2. **Binomial GLM-PCA** (동결): `η_ij = α_j + u_{c_i,j} + v_j^T z_i`, `p_ij = σ(η_ij)`.
   `α_j`, `v_j`, `z_i`는 train으로 적합한 기존 GLM-PCA 산출물이고, `u_{c,j}`는 decoder 단계에서
   적합한다(train cohort 크기 가중 평균이 SNP별 0이 되도록 중심화 + ridge).
3. **Standard latent diffusion** (변경 없음): `x_t = √ᾱ_t x_0 + √(1−ᾱ_t) ε`,
   `L_diff = E‖ε − ε_θ(x_t, t, c)‖²`. Fisher schedule·discrete diffusion·ELBO는 쓰지 않는다.
4. **AF-preserving heterozygosity tilt**:

   `q_ij(g) ∝ C(2,g) p_ij^g (1−p_ij)^{2−g} exp(τ_j·1[g=1] + λ_ij·g)`,  `Σ_g g·q_ij(g) = 2 p_ij`

   `λ_ij`는 각 셀에서 위 보존 조건으로 결정된다. `x_ij = e^{λ_ij} p_ij/(1−p_ij)`,
   `t_j = e^{τ_j}`로 두면 조건은 `(1−p)x² + t(1−2p)x − p = 0`의 양근이고,
   `q ∝ (1, 2 t x, x²)`이다. 수치적으로는 `q = min(p, 1−p)`에서
   `log f = log2 + log q − log(t(1−2q) + √(t²(1−2q)² + 4q(1−q)))`,
   `log x = ±log f` (부호는 `p ≤ 1/2` 여부; `x(p) = 1/x(1−p)`)로 계산한다.
   이 family는 모든 τ에서 확률이 유효하고 `E[G] = 2p`가 정확히 성립한다(격자 검증 오차 < 1e−10).
   §3의 c 방향 논거에 의해 이는 **개인별로 유효 범위 안에 머무는 F-form과 동등**하다.
5. **학습 순서**: (1) train genotype으로 Binomial GLM-PCA → (2) latent 정규화 →
   (3) cohort-conditioned standard diffusion → (4) real train latent와 cohort label로
   `(u, τ)` 적합(observed call NLL + ridge, LBFGS) → (5) generated latent inverse-normalization →
   (6) SNP별 `q_ij`에서 **독립** 추출.
6. **쓰지 않는 것**: `G_{j−1}` 조건화, genomic-order Markov chain, `A_b` residual, SNP 간
   autoregressive dependence. 기존 B0–B3 경로는 ablation으로 코드에 남긴다.

### 5.2 파라미터와 비용

| 구성 | 파라미터 수 (J=361, C=26) | 비고 |
|---|---:|---|
| `u_{c,j}` | 9,386 (중심화로 유효 9,025) | 기존 B1과 동일 |
| `τ_j` | 361 | 신규 |
| 합계 | 9,747 | diffusion 파라미터 9.3 M과 별개, 적합은 CPU 수십 초 |

### 5.3 주장 범위

허용: "HiPoDiT-T는 cohort-calibrated Binomial base에 allele-frequency를 보존하는 heterozygosity
tilt를 결합하여, unphased genotype synthesis에서 per-SNP genotype 분포와 cohort별 allele
frequency를 동시에 보정한다." 이 문장은 dev 결과에 한해 성립하며 §6 통과 전에는 예비 결과다.

금지: "local LD 보존"(§4.2), "phased haplotype LD", "최초의 genotype diffusion", "whole-genome
generation", "Fisher schedule 우수", "DP/anonymous synthetic data". `r²` MAE 개선을 LD 증거로
쓰는 것도 금지한다.

## 6. 재검증 절차 (아직 통과하지 않음)

§4의 숫자는 **dev에서 모델 구조를 고르고 dev에서 평가한** post-hoc 결과다. Task 2 리뷰가 지적한
대로 λ도 dev NLL로 골랐다. 확증에는 다음이 필요하다.

1. **사전 등록**: 기획서를 개정해 arm 집합을 `{B0, B1, T}`(+ ablation `T_superpop`, `T_cohort`,
   `B1+δ_{j,g}`, `B3+δ_{j,g}`)로 고정하고, 지표·gate를 아래로 다시 쓴다. 이후 구조 탐색 금지.
2. **Gate 1′ (oracle, dev)**: T가 B0 대비 per-call NLL을 개선하고, cohort AF MAE ≤ 1.05×B0,
   heterozygosity MAE ≤ B0. LD r² MAE는 **판정 지표에서 제외**하고 기술 지표로만 보고한다(§4.2).
   dependence 없는 모델이므로 order-shuffle control은 불필요하고, label-shuffle control은
   `u_{c,j}` 검증에 유지한다.
3. **Gate 2′ (end-to-end, dev)**: generated latent에서도 개선이 유지되는지. `τ`는 개인 자신의
   genotype으로 scoring된 oracle `z`에 적합했으므로, diffusion이 만든 `z`에서는 과신 정도가 달라
   het 보정이 어긋날 수 있다. inverse-normalization 경계에서 `z`의 scale/origin 일치와
   generated `p` 분포의 극단성(예: `|η| > 4` 비율)을 real oracle과 비교해 기록한다.
4. **Gate 3′ (test, 1회)**: seed 10개 이상 paired run, test split은 이 단계에서 한 번만 연다.
   per-call NLL·cohort AF MAE·heterozygosity MAE의 paired difference와 95 % bootstrap CI.
5. **Phase 4**: tilt는 het를 올리므로 sample 다양성·중복·membership inference가 B0 대비 나빠질
   수 있다. nearest-neighbour distance, MIA AUC, exact/near duplicate를 B0와 T 모두에서 측정한다.

## 7. 한계

- dev 251명, cohort당 6–11명. cohort별 지표의 불확실성이 크다.
- chr17 8 genes / 361 SNPs 패널. 전장유전체·rare variant(MAF<0.01 제외)·phased haplotype에
  대해서는 아무 주장도 하지 않는다.
- `r²`는 unphased dosage의 composite LD이며 haplotype LD가 아니다.
- §2.1의 "oracle z 과신" 설명은 가설이다. 잔여 population structure 설명과 구분되지 않았다.
- tilt는 marginal만 보정한다. SNP 간 의존성은 전혀 모델링하지 않으므로, 생성 표본의 LD 구조는
  latent diffusion이 `z`를 통해 유도하는 만큼만 존재한다.
- §4의 모든 수치는 프로토타입 스크립트 결과다. 사전 등록 후 repo 코드(`oracle` CLI)로 재생성해야
  공식 기록이 된다.

## 8. 산출물과 재현

| 항목 | 경로 |
|---|---|
| frozen panel | `outputs/diagnostics/hipodit_fisher_20260915_unique/` |
| Gate 1 oracle 결과 | `outputs/diagnostics/hipodit_ld_oracle_20260915/{oracle_results.json,study_manifest.json,decoder_ablation.csv,strata.csv,decoder_B{0..3}.npz}` |
| 5-seed 파일럿(standard/Fisher) | `outputs/diagnostics/hipodit_multiseed_20260915/`, `docs/reports/hipodit_multiseed_20260915.md` |
| 문헌 노트 | `.superpowers/sdd/2026-09-15-hipodit-ld/research-lit-af-drift.md` |
| 프로토타입 스크립트 | `docs/reports/prototypes/af_drift_prototypes.py`, `af_tilt_variants.py` |
| decoder 구현 | `src/models/genotype_decoder.py`, `tests/test_genotype_decoder.py` |
| oracle CLI | `scripts/hipodit_genotype_check.py` |

```bash
# Gate 1 (기존, B0-B3)
.venv/bin/python scripts/hipodit_genotype_check.py oracle \
  --prepared-dir outputs/diagnostics/hipodit_fisher_20260915_unique \
  --output-dir outputs/diagnostics/hipodit_ld_oracle_20260915

# 프로토타입 (본 문서 §4)
.venv/bin/python docs/reports/prototypes/af_drift_prototypes.py OUT.json full all 24
.venv/bin/python docs/reports/prototypes/af_tilt_variants.py OUT.json full T_snp,T_superpop,T_cohort,T_snp_noU 10
```

Gate 1 oracle 실행 1분 02초(CPU), 적합 17개 모두 수렴, 선택된 λ=(1,1), 거리 bin edge
[35, 104, 210], bin별 인접쌍 [86, 89, 89, 89]. 프로토타입 실행은 arm당 14–113초(CPU).
