# HiPoDiT 연구설계: 유전형에 맞는 GLM-PCA와 집단 구조 보정 생성

작성 기준: 2026-09-15. 이 문서는 현재 코드·32유전자 파일럿과 확인한 원문 논문을 바탕으로 한 **실험 설계**다. 새로운 모델의 효과, novelty 또는 저널 게재 가능성을 입증한 결과가 아니다. 연구 목표는 현재 확보된 1000 Genomes Phase 3 자료에 맞춰 **인구집단별 이배체 유전형 충실도와 소수 집단 생성**으로 둔다. 질병/표현형 생성은 해당 라벨이 포함된 별도 cohort가 있어야 독립 연구 질문으로 다룰 수 있다. 1KG Phase 3은 2,504명·26 population·5 superpopulation 자료다. [1KG 원문](https://www.nature.com/articles/nature15393)

**Exa 재검토(2026-09-15):** 아래 cohort AF·LD 보정은 독자적인 novelty 주장으로 사용하지 않는다. [ClOneHORT](https://pmc.ncbi.nlm.nih.gov/articles/PMC11230377/)가 집단별 genotype-frequency·LD·privacy 목표로 합성 genome의 cohort 부분집합을 선택했고, [Moment Guided Diffusion](https://arxiv.org/html/2602.17211v1)은 일반적인 moment-guided 생성도 다뤘다. 이 문서의 C 단계는 성능 검증용 ablation일 뿐, 논문 기여의 기본값이 아니다. 더 강한 연구 질문은 훈련에서 제외한 population을 소수의 phased 개인으로 적응해 새 haplotype을 생성하고 독립 개인의 imputation·다양성·누출 위험을 평가하는 것이다. 이것도 [GPC](http://starai.cs.ucla.edu/papers/AnandGen226.pdf), [RESHAPE](https://pmc.ncbi.nlm.nih.gov/articles/PMC11136649/)와 구별되는 실험 및 외부 검증이 필요하다.

## 1. 주장 가능한 문제와 선행연구의 경계

단순한 “유전자별 차원 축소 + 조건부 diffusion”과 CNN/Transformer 결합은 [GeneticDiffusion](https://doi.org/10.1093/bioinformatics/btaf209)이 이미 다뤘다. Superpopulation 조건부 diffusion과 제한된 실제 데이터의 증강은 [Light PCA-DDPM](https://doi.org/10.1101/2024.10.28.620648)에 있다. [SNPgen](https://arxiv.org/html/2603.10873v1)은 3종 유전형을 출력하고 AF·LD·downstream utility·membership inference를 평가한다. [2026년 GigaScience 연구](https://doi.org/10.1093/gigascience/giag044)는 인간 자료에서 이산 유전형 생성과 LD decay 등 여러 유전학 지표를 비교했다. 계층별 조건부 역확산과 moment guidance도 각각 [branched diffusion](https://arxiv.org/abs/2212.10777), [Moment Guided Diffusion](https://arxiv.org/html/2602.17211v1)에서 제안됐다. LD의 명시적 SNP 의존성을 학습하는 접근도 [HCLT](https://pmc.ncbi.nlm.nih.gov/articles/PMC10245670/)에 있다.

따라서 연구 가설은 “첫 조건부 유전형 diffusion”이나 “LD를 처음 보존”이 아니다. 아래의 이산 디코더와 cohort 단위 보정은 **성능·오류 위치를 확인할 실험 가설**이며 독자적인 방법론 기여를 뜻하지 않는다. 논문 기여 가능성은 훈련에서 제외한 집단에 대한 소수표본 적응처럼 새로운 문제 설정에서, 기존 모델보다 phased haplotype 충실도와 독립 downstream utility를 개선했을 때 다시 판단한다.

## 2. 현재 증거와 해결해야 하는 표현 문제

현재 [Poisson GLM-PCA 코드](../../src/preprocessing/glm_pca.py)는 `family='poi'`와 빠른 `fit_poisson` 백엔드만 허용한다. [glmpca-fast 0.1.2](https://pypi.org/project/glmpca-fast/0.1.2/)도 Poisson만 지원한다. Poisson은 `0,1,2`의 상한을 가지지 않는다. [32유전자 파일럿 보고서](hipodit_rebuild_execution_20260915.md)의 검증 복원 평균 최대 5.1634는 이 한계를 실제로 드러낸다. 파일럿은 2,002/251/251 train/validation/test, K=4, 100 optimizer steps이며 attention 학습 경로와 생성 유한성을 검증했지만, 유효한 SNP 유전형이나 집단별 AF·LD 우위는 검증하지 않았다. 이전 W2/MMD 개선은 실제/합성 평가 스케일 오류 수정이므로 모델 효과에 포함하지 않는다.

차원 축소를 GLM-PCA로 유지하는 조건에서 첫 표현 후보는 **두 시행의 binomial GLM-PCA**다. SNP $j$의 실제 대립유전자 수 $g_{ij}\in\{0,1,2\}$, 잠재인자 $z_i$, 절편 $a_j$, loading $v_j$에 대해

\[
\eta_{ij}=a_j+v_j^\top z_i,\quad p_{ij}=\sigma(\eta_{ij}),\quad
q_0(g_{ij}=k\mid z_i)={2\choose k}p_{ij}^{k}(1-p_{ij})^{2-k}.
\]

Held-out scoring은 **train에 적합한** $a,V$를 고정하고 `sum_j[2 softplus(eta_ij)-g_ij eta_ij] + penalty*||z_i||²/2`를 최소화한다. [공식 GLM-PCA 문서](https://search.r-project.org/CRAN/refmans/glmpca/html/glmpca.html)는 `binom` family를 지원하지만 `sz` 기본값이 관측 열의 합이므로, 이 문제에서는 *각 개인의 시행 수 2를 명시*해야 한다. 현재 Rust Poisson 코드를 family 이름만 바꿔 재사용할 수는 없다. [Logistic Factor Analysis 원문](https://pmc.ncbi.nlm.nih.gov/articles/PMC4795615/)은 이배체 유전형의 `Binomial(2,p)` 가정을 설명하며, 이것이 구조를 조건으로 한 Hardy–Weinberg 가정과 연결된다고 명시한다. 그러므로 binomial도 HWE 편차나 SNP 간 LD를 자동 보존하지 않는다.

현재 파이프라인은 누락값을 train 평균으로 대치해 0과 2 사이의 소수 dosages를 만들 수 있다. **정수 0/1/2 호출값과 missingness mask**를 별도로 보관하거나, 관측 genotype probability가 있다면 그 확률을 이용한 기대 likelihood를 정의해야 한다. 소수 대치값을 실제 binomial 시행 결과라고 부르지 않는다. 현재 32유전자 파일럿은 정확한 변이 순서·allele mapping을 저장했지만 학습용 원래 호출값 전체를 잠재인자 아티팩트에 보관하지 않는다. [파일럿 준비 코드](../../scripts/hipodit_rebuild_prepare.py)

## 3. 생성 모델 후보: 국소 이산 디코더와 cohort 보정

GLM-PCA를 고정해 압축 손실과 생성 손실을 구별한다. 기존 HiPoDiT CNN–DiT–CNN/FiLM diffusion은 정규화된 인자 $u_i$를 생성한다. 저장된 train-only 정규화를 역변환해 $z_i$를 얻은 뒤 SNP를 복원한다. binomial $q_0$는 디코더의 *기준 분포*다. 실제 SNP 간 의존성을 모델링하는 작은 공유 residual $r_\theta$를 덧붙인다.

\[
q_\theta(g_{ij}=k\mid z_i,c_i,g_{i,<j})\propto
q_0(k\mid z_i)\exp\{r_\theta(k,\;g_{i,N^-_j},\;c_i,\;\Delta bp_j)\},
\quad k\in\{0,1,2\}.
\]

여기서 $N^-_j$는 이미 생성한 물리적으로 가까운 SNP만 포함한다. Residual은 0에서 시작하고 한 gene마다 별도 거대 모델을 만들지 않는다. 인접 좌표의 SNP 간 물리적 간격을 입력으로 주며, 학습에서는 실제 앞선 SNP를, 생성에서는 샘플링된 앞선 SNP를 쓴다. **SNP 간 조건부 의존성은 기존 연구가 있으므로 이 항 단독으로 novelty를 주장하지 않는다.** 먼저 진짜 GLM 인자를 넣고 자율 생성한 유전형이 independent binomial 디코더보다 held-out LD proxy를 복원하는지 확인한다.

그 다음 같은 population의 $N$명 샘플을 한 번에 생성한다. Train-only 목표 통계 $M_p^{train}$와 상위 집단 통계 $M_s^{train}$를 population 훈련 표본수 $n_p$와 bootstrap 추정오차에 따라 결합해 $M_p^*$를 만든다. 간단한 파일럿 형태는 $M_p^*=\alpha_p M_p^{train}+(1-\alpha_p)M_s^{train}$, $\alpha_p=n_p/(n_p+\kappa)$이며 $\kappa$는 validation에서 선택하고 test에는 고정한다. 목표는 AF와 물리적으로 가까운 SNP 쌍의 *signed dosage covariance/correlation*이다. $r^2$만 맞추면 상관의 부호를 잃고, 0-variance SNP에서는 correlation gradient가 불안정하므로 guidance에서는 공분산을 우선 사용한다.

\[
E_p(\tilde G^{1:N})=
\lambda_{AF}\sum_jw_j\big[AF_j(\tilde G)-AF_{p,j}^*\big]^2
+\lambda_{LD}\sum_{(j,k)\in\mathcal E}w_{jk}
\big[C_{jk}(\tilde G)-C_{p,jk}^*\big]^2.
\]

제안할 분포는 $q(\tilde G^{1:N}\mid\tilde Z,p)\propto\prod_iq_\theta(\tilde G_i\mid\tilde z_i,p)\exp[-E_p(\tilde G^{1:N})]$다. 이 식은 정규화 상수를 계산할 수 있다는 주장이나 정확한 역확산 샘플러의 정리가 아니다. 구현 시 **후기 역확산 단계의 soft/relaxed genotype 기대값**으로 미분 가능한 $E_p$를 근사하고 제한된 gradient step을 적용하는 후보다. 실제 `0/1/2` 샘플에서 효과를 다시 확인해야 한다. 과도한 보정은 소수 집단 다양성 감소나 훈련 개인 근접을 초래할 수 있다. [일반 moment-guidance 선행연구](https://arxiv.org/html/2602.17211v1)를 인용하고 차이를 명시한다.

## 4. 단계별 실험과 중단 기준

| 단계 | 잠그는 조건 | 실험과 다음 단계로 넘어가는 증거 |
| --- | --- | --- |
| A. 표현 감사 | 같은 훈련 split·변이 집합·gene order | 32유전자 파일럿에서 Poisson vs binomial GLM-PCA, K=4/8/16의 validation NLL·genotype calibration·AF·LD 복원 상한·계산 비용을 비교한다. Binomial solver와 fixed-decoder held-out scorer를 별도 검증한다. 실제 인자에서조차 복원이 실패하면 generator를 수정하지 않는다. |
| B. 이산 디코더 | 선택된 binomial GLM-PCA 인자와 normalization 고정 | `independent q0` vs `local qtheta`를 **실제 held-out 인자**와 **합성 인자**에 각각 적용한다. Teacher-forced validation NLL과 free-running 생성 AF·LD·diversity를 모두 본다. 실제 인자에서 개선이 없으면 AR 디코더를 버린다. |
| C. cohort guidance | B의 diffusion checkpoint·decoder·샘플수 고정 | guidance 0 vs AF만 vs AF+공분산 vs 계층적 shrinkage 포함을 비교한다. 32유전자에서 먼저 gradient 유한성·메모리·다양성·집단별 calibration을 확인한다. 개선과 다양성이 동시에 관측될 때 다중 염색체 패널로 확장한다. |
| D. 논문 수준 검증 | 실험 설정을 test 이전에 고정 | 동일 사이트·REF/ALT·sample count·sampling steps·연산 예산에서 여러 seed, 강한 외부 baseline, 독립 held-out 개인을 비교한다. Test set은 최종 평가에 한 번 사용한다. |

외부 baseline은 allele-frequency-only independent binomial, [HAPNEST](https://doi.org/10.1093/bioinformatics/btad535), [GeneticDiffusion](https://doi.org/10.1093/bioinformatics/btaf209)을 우선 검토한다. HAPNEST·GeneticDiffusion의 phased haplotype 출력은 현재 dosage 출력과 다른 정보량을 가지므로, **동일 SNP·allele의 dosage로 변환한 비교**와 원래 논문 설정의 비교를 구별한다. Light PCA-DDPM은 superpopulation baseline 후보지만 패널·표현이 다르다. SNPgen과 2026 GigaScience 논문은 선행연구와 평가 수준의 기준이다; 현재 1KG-only population 실험을 UK Biobank 질병 조건 실험과 직접 성능 순위로 비교하지 않는다. PCA는 최종 모델의 대체안이 아니라, GLM-PCA의 효과를 분리할 때만 명시적 ablation으로 둔다.

## 5. 사전 지정 평가와 해석

**Primary endpoints**는 (i) 같은 집단·같은 SNP에서의 AF absolute error, (ii) 물리적 거리 bin별 SNP dosage-correlation/공분산 및 $r^2$ decay 오차, (iii) population별 synthetic coverage/recall이다. 모든 오차는 real train–real held-out 차이와 함께 제시한다. **Secondary endpoints**는 genotype 0/1/2 validity, per-site genotype frequencies, heterozygosity/HWE departure, 집단 간 $F_{ST}$, PCA/UMAP 구조, population classifier synthetic-train→real-held-out 성능, 런타임·샘플 비용이다. [GeneticDiffusion](https://doi.org/10.1093/bioinformatics/btaf209), [Light PCA-DDPM](https://doi.org/10.1101/2024.10.28.620648), [2026 GigaScience 평가](https://doi.org/10.1093/gigascience/giag044)는 각각 분류 활용·집단 통계·genotype-frequency/LD/coverage 평가의 선행 기준이다.

`0/1/2` dosage Pearson correlation은 **gametic phase를 직접 측정한 haplotype LD가 아니다**. 현재 모델은 두 haplotype을 생성하지 않으므로 phase fidelity·recombination·local ancestry를 주장하지 않는다. [이산 유전형 평가 연구](https://doi.org/10.1093/gigascience/giag044)도 unknown phase에서 정확한 gametic LD 추정이 어렵다고 지적한다. 또한 현재 train-only MAF≥0.01 필터와 32유전자 패널로는 rare-variant 또는 full-genome 주장을 할 수 없다. 현재 1KG의 라벨은 population이고 질병 라벨이 없으므로 질병 PRS·GWAS/TSTR는 별도 phenotype cohort 없이는 primary endpoint가 아니다.

Population별 validation 개인은 매우 적어 fine-population 점추정만으로 우열을 선언하면 안 된다. 3개 이상의 독립 학습 seed와 동일 생성 수, subject-level bootstrap 신뢰구간, superpopulation pooled 결과, 26 population 개별 결과를 함께 제시한다. 각 endpoint는 **실제 인자→디코더**와 **합성 인자→디코더**를 나눠 압축·decoder·diffusion 오차의 위치를 분리한다. MMD/PCA W2만으로 AF·LD·유전형 충실도를 대체하지 않는다.

복제나 가까운 이웃 수, train-vs-heldout nearest-neighbor 거리 분포, membership inference를 다양성과 함께 감사한다. 일치하는 개인이 없다는 사실은 프라이버시 보증이 아니다. [NDSS 2022 원문](https://www.ndss-symposium.org/wp-content/uploads/2022-92-paper.pdf)은 synthetic genomic data의 utility와 membership leakage가 동시에 좋아지는 단일 접근이 없음을 보고한다. 공격 평가 결과는 **경험적 위험 평가**로 기술하며 formal differential privacy라고 주장하지 않는다.

## 6. 논문 판정 규칙

AF·dosage LD proxy·coverage가 반복적으로 개선돼도 그것만으로 cohort 보정의 방법론적 신규성이 성립하지 않는다. [ClOneHORT](https://pmc.ncbi.nlm.nih.gov/articles/PMC11230377/)와 같은 목표 통계·cohort 선택, [GPC](http://starai.cs.ucla.edu/papers/AnandGen226.pdf)의 생성·imputation, [Light PCA-DDPM](https://doi.org/10.1101/2024.10.28.620648)의 소수집단 downstream 증강을 이기는 **새 문제 설정과 외부 검증**이 필요하다. 특히 훈련에서 제외한 population을 소수의 phased 개인으로 적응하는 경우, 이미 본 집단의 AF를 맞추는 것과 달리 out-of-training 일반화를 시험할 수 있다. 이 방향도 선행연구 전체를 배제했다는 보장이 아니며, 독립 cohort의 matched SNP panel·haplotype LD·imputation·다양성·membership risk로 판단해야 한다. 다양성이나 위험이 나빠지면 AF·LD가 개선돼도 성공으로 간주하지 않는다. 현재의 32유전자 dosage 파일럿만으로는 phase fidelity, rare variant 또는 full-genome 기여를 주장할 수 없다.
