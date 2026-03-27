# FiLM + Diffusion 기반 인구군 조건부 유전형 합성 모델: 문헌 조사 및 논문 전략

**작성일**: 2026-03-27
**연구 깊이**: Deep (5-hop)
**목적**: FiLM 조건부 메커니즘을 유전형 생성 Diffusion 모델에 적용하여 SCI 논문 출판

---

## Executive Summary

FiLM(Feature-wise Linear Modulation)은 단백질 구조 예측(AlphaFold3), 단백질 서열 생성(TaxDiff), 분자 그래프 처리(GNN-FiLM, GLDM) 등 생물학 분야에서 광범위하게 활용되고 있으나, **유전형(genotype/SNP) 생성에 FiLM을 적용한 논문은 현재까지 존재하지 않는다.** 이는 명확한 연구 갭(research gap)이며, SCI 논문의 novelty로 충분한 가치를 갖는다.

또한 현재 모든 유전형 생성 모델은 표준 생성 손실(adversarial, L2 noise prediction)만 사용하고 SFS/LD/Haplotype 지표는 사후 평가에만 쓰인다. **유전학 특화 보조 손실(auxiliary loss)을 학습에 직접 통합한 선행 연구 역시 없다.** 이 두 가지를 결합하면 강력한 contribution이 된다.

**핵심 결론: FiLM을 유전형 Diffusion 모델에 적용하는 것은 가능하며, 적절한 설계를 통해 SCI급 novelty를 확보할 수 있다.**

---

## 1. FiLM의 생물학 분야 적용 현황

### 1.1 직접적 유전체 적용

| 논문 | 연도 | 학회 | 기법 | 생물학 도메인 |
|------|------|------|------|-------------|
| **Temporal FiLM** (Birnbaum et al.) | 2019 | NeurIPS | FiLM (RNN이 CNN 변조) | DNA 메틸화(6mA 사이트) 예측 |

- RNN이 장거리 의존성을 포착하여 gamma/beta를 생성하고, CNN의 활성화를 변조
- **유전체 서열에 FiLM을 직접 적용한 유일한 사례**
- 그러나 분류(classification) 태스크이며, 생성(generation)이 아님

### 1.2 분자/약물 발견 (유사 도메인)

| 논문 | 연도 | 학회 | 기법 | 도메인 |
|------|------|------|------|--------|
| **GNN-FiLM** (Brockschmidt) | 2020 | ICML | FiLM on GNN 메시지 패싱 | 분자 그래프 회귀 |
| **GLDM** (Wang et al.) | 2024 | Brief. Bioinf. | FiLMConv 인코더 + Latent Diffusion | 약물 분자 생성 (유전자 발현 프로파일 조건부) |

### 1.3 단백질 구조/서열 (FiLM/AdaLN)

| 논문 | 연도 | 학회 | 기법 | 도메인 |
|------|------|------|------|--------|
| **AlphaFold3** (Abramson et al.) | 2024 | Nature | AdaLN (FiLM 파생) | 단백질 구조 예측 |
| **TaxDiff** (Lin et al.) | 2024 | arXiv (북경대) | AdaLN-Zero (FiLM 기반) | 분류군(taxonomy) 조건부 단백질 서열 생성 |

- **TaxDiff가 가장 직접적인 선례**: 분류 체계를 조건으로 Diffusion Transformer에서 AdaLN-Zero로 서열 생성
- 우리 연구에서 "인구군(population) 조건부 유전형 생성"과 구조적으로 동일한 접근

### 1.4 핵심 발견

> **유전형(SNP/genotype) 생성에 FiLM 또는 AdaLN을 적용한 출판 논문은 현재까지 0건이다.**
> 이것이 논문의 핵심 novelty 포인트가 된다.

---

## 2. 유전형 생성 모델 현황 (2023-2026)

### 2.1 Diffusion 기반

| 논문 | 연도 | 학회 | 데이터 | 핵심 방법 |
|------|------|------|--------|----------|
| **GeneDiffusion** (Kenneweg et al.) | 2025 | ISMB/Bioinformatics | 1KG (2,504명), ALS (10,405명) | Gene PCA + 1D UNet Diffusion, one-hot 조건부 |
| **SNPgen** (Lampis et al.) | 2026 | arXiv | UK Biobank (458,724명) | GWAS 기반 변이 선택 + VAE + Latent Diffusion, classifier-free guidance |
| **GenoDiffusion** (Shi Lab) | 2024 | 학회 발표 | HLA, 전립선암 | 조건부 DDPM, classifier-free guidance |
| **DiscreteGenoGen** (Xie et al.) | 2025 | bioRxiv | 소/인간 다수 염색체 | VAE vs Diffusion vs GAN 비교 벤치마크 |

### 2.2 GAN 기반

| 논문 | 연도 | 학회 | 핵심 방법 |
|------|------|------|----------|
| **Yelmen et al.** | 2021 | PLOS Genetics | GAN + RBM, 1KG (~10K SNPs) |
| **Conv-WGAN + CRBM** (Yelmen et al.) | 2023 | PLOS Comp Bio | 컨볼루션 WGAN, 65K SNPs 확장 |
| **Genome-AC-GAN** (Ahronoviz et al.) | 2024 | bioRxiv | 보조 분류기 GAN, 소수 인구군 특화 |
| **ClOneHORT** | 2024 | bioRxiv | SFS/LD 충실도 향상, 로컬 조상 추론 0.91-0.94 |

### 2.3 기타 접근법

| 논문 | 연도 | 학회 | 핵심 방법 |
|------|------|------|----------|
| **HCLT** (Dang et al.) | 2023 | PMC | 확률적 회로, 추적 가능한 추론 |
| **HAPNEST** (Wharrie et al.) | 2023 | Bioinformatics | 하플로타입 리샘플링, coalescent 기반 |
| **Flow Matching** | - | - | **유전형에는 아직 미적용** (명확한 갭) |

### 2.4 조건부 메커니즘 비교

| 논문 | 조건부 방법 | 조건 유형 |
|------|-----------|----------|
| GeneDiffusion | One-hot 곱셈 주입 | 26개 인구군 |
| SNPgen | Classifier-free guidance + Cross-attention | 이진 질병 레이블 |
| GenoDiffusion | ResNet 블록 임베딩 + CFG | 질병 레이블 |
| Genome-AC-GAN | 보조 분류기 | 하위 인구군 |
| **제안 연구 (FiLM)** | **블록별 gamma/beta 변조** | **계층적 인구군 (pop + superpop)** |

---

## 3. 분포 매칭 손실의 문헌 현황

### 3.1 현재 상태: 평가 지표로만 사용

모든 주요 논문에서 SFS, LD, 하플로타입 다양성은 **사후 평가 지표**로만 사용:

| 지표 | 측정 방법 | 대표 논문 |
|------|----------|----------|
| **SFS/AFS** | 대립유전자 빈도 Pearson 상관 (r=0.94~0.99) | Yelmen 2021, 2023 |
| **LD** | 쌍별 r^2 감쇠 곡선 비교 (상관 0.94~0.98) | Yelmen 2021, 2023; ClOneHORT 2024 |
| **3점 상관** | SNP 삼중항 의존성 (거리 1~1024) | Yelmen 2021, 2023 |
| **하플로타입 거리** | Wasserstein 거리 (쌍별 Hamming 분포) | Yelmen 2021 |
| **k-mer 모티프** | 4-mer, 8-mer SNP 윈도우 빈도 분포 | Yelmen 2023 |
| **선택 신호** | XP-EHH (r=0.902), PBS (r=0.923) | Yelmen 2021 |

### 3.2 핵심 발견

> **미분 가능한(differentiable) SFS/LD/Haplotype 손실을 학습에 직접 사용한 논문은 현재까지 0건이다.**
> 이것이 두 번째 novelty 포인트가 된다.

### 3.3 간접적 관련 작업

- **pg-gan** (Wang et al. 2021): msprime 시뮬레이터의 출력을 CNN 판별기로 평가 → 암묵적 분포 매칭
- **CRBM의 조건부 피닝**: 겹치는 세그먼트에서 조건부 생성 → 암묵적 LD 보존
- **MMD 기반 GMMN**: Maximum Mean Discrepancy로 분포 매칭 → 유전학 특화는 아니지만 분포 수준 손실

---

## 4. 소규모 샘플 문제의 해결 전략

### 4.1 데이터 증강

| 전략 | 논문 | 핵심 | 적용 가능성 |
|------|------|------|-----------|
| **HAPNEST 하플로타입 리샘플링** | Wharrie 2023, Bioinformatics | 2,504명 참조 패널에서 100만명 합성 | **높음** - 전처리 단계에서 적용 |
| **EvoAug 진화적 증강** | Genome Biology 2023 | 돌연변이/삽입/결실/전좌 시뮬레이션 | **중간** - 유전형 데이터에 맞게 변형 필요 |
| **Simulation-on-the-fly** | Mughal 2023 | msprime로 매 에포크 새 데이터 생성 | **중간** - Diffusion 학습과 결합 복잡 |
| **하플로타입 셔플링** | 다수 | 재조합 경계에서 하플로타입 세그먼트 교환 | **높음** - 생물학적으로 타당 |

### 4.2 모델 설계 전략

| 전략 | 설명 | 근거 |
|------|------|------|
| **윈도우/유전자 수준 학습** | 전체 유전체를 한번에 학습하지 않고, 유전자/윈도우 단위로 분할 → 실효 샘플 수 100~1000배 증가 | 다수 논문에서 사용 |
| **잠재 공간 Diffusion** | VAE로 먼저 압축 후 잠재 공간에서 Diffusion → 차원 축소 | SNPgen (2026) |
| **경량 모델 설계** | PCA 공간 GAN이 전체 모델과 동등 성능 → 파라미터 줄이기가 데이터 늘리기만큼 중요 | Szatkownik 2024 (MLCB) |
| **Transfer Learning** | 대규모 집단 → 소규모 집단 전이 학습, 2~14.2% 성능 향상 | BMC Bioinformatics 2022 |

### 4.3 핵심 인사이트

> GeneDiffusion 자체가 이미 n=2,504에서 동작하는 것을 증명했다.
> 핵심은 **모델 복잡도를 데이터에 맞추는 것**이다.
> DiT급 대형 Transformer가 아니라, **기존 UNet에 FiLM만 추가하는 경량 접근**이 정답이다.

---

## 5. 논문 전략: FiLM을 살리는 구체적 방법

### 5.1 제안 논문 제목 (안)

> **"Population-Conditional Genotype Synthesis via FiLM-Modulated Diffusion with Genetics-Aware Auxiliary Losses"**

### 5.2 Contribution 정리

| # | Contribution | 선행 연구 존재 여부 |
|---|-------------|------------------|
| 1 | 유전형 생성에 FiLM 조건부 메커니즘 최초 적용 | **없음** (AlphaFold3/TaxDiff는 단백질, Temporal FiLM은 분류) |
| 2 | 계층적 인구군 임베딩 (pop + superpop) FiLM 변조 | **없음** (기존은 flat one-hot 또는 단일 레이블) |
| 3 | 미분 가능한 유전학 특화 보조 손실 (SFS/LD) 학습 통합 | **없음** (기존은 사후 평가만) |
| 4 | 소규모 코호트에서 FiLM의 파라미터 효율성 검증 | **없음** |

### 5.3 기술적 설계 — Gene PCA 표현에서도 FiLM이 작동하는 이유

**이전 리뷰의 우려**: "SFS/LD 손실은 PCA 공간에서 정의할 수 없다"

**해결책**: 두 가지 레벨의 손실 설계

#### Level 1: PCA 공간 손실 (현재 파이프라인 호환)

```
L_total = L_diffusion + lambda_1 * L_pca_dist + lambda_2 * L_pop_structure
```

- **L_pca_dist**: PCA 주성분별 분포 매칭 (MMD 또는 Wasserstein)
  - 각 유전자의 PCA 계수 분포가 실제 인구군별 분포를 재현하는지 측정
  - PCA 공간에서도 **인구군 간 분포 차이**는 보존되므로 의미 있음
- **L_pop_structure**: 인구군 구조 보존 손실
  - 생성된 샘플의 PCA projection이 실제 인구군 클러스터와 일치하는지
  - Sliced Wasserstein Distance 등 미분 가능한 분포 거리 사용

#### Level 2: SNP 공간 손실 (파이프라인 확장 시)

전처리를 변경하여 raw genotype 또는 windowed genotype을 사용할 경우:

```
L_total = L_diffusion + lambda_1 * L_sfs + lambda_2 * L_ld_local
```

- **L_sfs**: 미분 가능한 SFS 근사
  - 생성된 배치의 대립유전자 빈도 히스토그램과 실제 히스토그램의 KL divergence
  - Soft histogram (Gaussian kernel binning)으로 미분 가능하게 구현
- **L_ld_local**: 로컬 LD 보존
  - 윈도우 내 쌍별 상관 행렬의 Frobenius norm 차이
  - 계산량 제한을 위해 윈도우 크기 제한 (예: 100 SNPs)

### 5.4 FiLM이 기존 UNet에서 작동하는 구체적 구조

```
현재: x * c_emb_linear(one_hot_y) + t_emb_linear(t)       [단순 곱셈/덧셈]
제안: gamma(pop_emb, superpop_emb, t) * h + beta(...)     [블록별 FiLM 변조]
```

**계층적 인구군 임베딩**:
```
pop_emb = Embedding(26, d)        # GBR, YRI, CHB, ...
superpop_emb = Embedding(5, d)    # EUR, AFR, EAS, SAS, AMR
hierarchy_emb = MLP(concat(pop_emb, superpop_emb))

FiLM_params = FiLMGenerator(hierarchy_emb, t_emb)
  -> per-block (gamma_l, beta_l) for l = 1, ..., N_blocks
```

**TaxDiff의 선례**: 분류 체계(종 → 속 → 과)를 계층 임베딩으로 Diffusion Transformer를 변조 → 우리는 인구 체계(인구군 → 슈퍼인구군)로 동일 접근

**파라미터 효율성**: FiLM은 블록당 2d개 파라미터(gamma + beta)만 추가 → n=2,504에서도 오버피팅 위험 최소

### 5.5 실험 설계

#### 비교 대상 (Baselines)

| 모델 | 설명 |
|------|------|
| GeneDiffusion (원본) | UNet + one-hot 곱셈 조건부 |
| GeneDiffusion + CFG | Classifier-free guidance 추가 |
| GeneDiffusion + Cross-Attention | SNPgen 스타일 조건부 |
| **GeneDiffusion + FiLM (제안)** | 계층적 FiLM 변조 |
| **GeneDiffusion + FiLM + Aux Loss (제안)** | FiLM + 유전학 보조 손실 |
| Genome-AC-GAN | GAN baseline |
| HAPNEST | 비딥러닝 baseline |

#### 평가 지표

| 카테고리 | 지표 | 근거 논문 |
|----------|------|----------|
| 충실도 | 대립유전자 빈도 상관 (전체 + 저빈도) | Yelmen 2021, 2023 |
| 충실도 | LD 감쇠 곡선 (쌍별 r^2) | Yelmen 2021; ClOneHORT 2024 |
| 충실도 | 3점 상관 (고차 하플로타입) | Yelmen 2021, 2023 |
| 구조 | PCA/UMAP 인구군 클러스터 일치도 | GeneDiffusion 2025 |
| 유용성 | Recovery Rate (합성 데이터 학습 → 실제 테스트) | GeneDiffusion 2025 |
| 유용성 | 데이터 증강 효과 (5%/10%/50% 실제 + 합성) | GeneDiffusion 2025 |
| 프라이버시 | NNAA (0.5에 가까울수록 좋음) | Yelmen 2021; GeneDiffusion 2025 |
| 프라이버시 | 멤버십 추론 공격 AUC | SNPgen 2026 |
| 다양성 | k-mer 모티프 엔트로피 | Szatkownik 2024 |
| **인구군 특이성** | **인구군별 생성 품질 분산** | **새로운 지표 (FiLM의 장점을 보여주는)** |

#### 핵심 실험: FiLM의 효과를 입증하는 ablation

1. **인구군별 생성 품질**: FiLM이 소수 인구군(AMR 347명, ASW 61명)의 생성 품질을 개선하는지
2. **계층 임베딩 효과**: pop_emb만 vs pop_emb + superpop_emb
3. **블록별 변조 분석**: 어느 UNet 블록의 FiLM이 가장 중요한지 (저해상도 vs 고해상도)
4. **보조 손실 가중치**: lambda 스케줄링에 따른 SFS/LD 지표 변화

### 5.6 타겟 저널

| 저널 | IF | 적합성 | 근거 |
|------|-----|--------|------|
| **Bioinformatics** | 5.8 | 높음 | GeneDiffusion 원본 논문이 여기 게재 (ISMB) |
| **Genome Biology** | 12.3 | 높음 | EvoAug 게재, 방법론 + 생물학적 검증 조합 |
| **PLOS Computational Biology** | 4.3 | 높음 | Yelmen 2023 게재, 합성 유전체 분야 |
| **Nucleic Acids Research** | 14.9 | 중간 | 방법론이 강해야 함 |
| **Nature Communications** | 16.6 | 중간-낮음 | 광범위한 영향력 입증 필요 |
| **NeurIPS/ICML (ML venue)** | - | 중간 | 방법론적 기여 강조 시 |

---

## 6. 참고 문헌 전체 목록

### FiLM / AdaLN 관련
1. Perez et al. "FiLM: Visual Reasoning with a General Conditioning Layer" AAAI 2018. arXiv:1709.07871
2. Birnbaum et al. "Temporal FiLM: Capturing Long-Range Sequence Dependencies with Feature-Wise Modulations" NeurIPS 2019. arXiv:1909.06628
3. Brockschmidt. "GNN-FiLM: Graph Neural Networks with Feature-wise Linear Modulation" ICML 2020
4. Wang et al. "GLDM: Hit Molecule Generation with Constrained Graph Latent Diffusion Model" Briefings in Bioinformatics 25(3), 2024
5. Lin et al. "TaxDiff: Taxonomic-Guided Diffusion Model for Protein Sequence Generation" arXiv:2402.17156, 2024
6. Abramson et al. "AlphaFold3: Accurate Structure Prediction of Biomolecular Interactions" Nature, 2024
7. Peebles & Xie. "Scalable Diffusion Models with Transformers (DiT)" ICCV 2023

### 유전형 생성 (Diffusion)
8. Kenneweg et al. "Generating Synthetic Genotypes using Diffusion Models" Bioinformatics 41(Suppl 1), ISMB 2025. arXiv:2412.03278
9. Lampis et al. "SNPgen: Phenotype-Supervised Genotype Representation and Synthetic Data Generation via Latent Diffusion" arXiv:2603.10873, 2026
10. Shi Lab. "GenoDiffusion: Conditional Denoising Diffusion Probabilistic Model for Genomic Data Augmentation" 2024
11. Xie et al. "Deep Generative Models for Discrete Genotype Simulation" bioRxiv/arXiv:2508.09212, 2025

### 유전형 생성 (GAN/VAE/기타)
12. Yelmen et al. "Creating Artificial Human Genomes Using Generative Neural Networks" PLOS Genetics, 2021
13. Yelmen et al. "Deep Convolutional and Conditional Neural Networks for Large-Scale Genomic Data Generation" PLOS Comp Bio, 2023
14. Ahronoviz et al. "Genome-AC-GAN: Enhancing Synthetic Genotype Generation through Auxiliary Classification" bioRxiv, 2024
15. Dang et al. "Tractable and Expressive Generative Models of Genetic Variation Data" PMC, 2023
16. Wharrie et al. "HAPNEST: efficient, large-scale generation and evaluation of synthetic datasets" Bioinformatics, 2023
17. ClOneHORT. "Improved Fidelity in Generative Models of Synthetic Genomes" bioRxiv, 2024
18. Szatkownik et al. "Towards Creating Longer Genetic Sequences with GANs" MLCB 2024

### DNA/단백질 Diffusion
19. DaSilva et al. "DNA-Diffusion: Designing Synthetic Regulatory Elements" Nature Genetics, 2025
20. Li et al. "DiscDiff: Latent Diffusion Model for DNA Sequence Generation" arXiv:2402.06079, 2024
21. Li et al. "Absorb & Escape: Overcoming Single Model Limitations in Generating Genomic Sequences" NeurIPS 2024

### 소규모 샘플 해결
22. Lee et al. "EvoAug: improving generalization with evolution-inspired data augmentations" Genome Biology, 2023
23. Dalla-Torre et al. "Nucleotide Transformer" Nature Methods, 2024
24. Theodoris et al. "Geneformer: Transfer learning enables predictions in network biology" Nature, 2023
25. BMC Bioinformatics. "Transfer learning for genotype-phenotype prediction" 2022
26. Mughal et al. "Deep Learning in Population Genetics" Genome Biology and Evolution, 2023
27. Self-GenomeNet. "A self-supervised deep learning method for data-efficient training in genomics" Communications Biology, 2023

### 평가/프라이버시
28. Oprisanu et al. "On Utility and Privacy in Synthetic Genomic Data" NDSS
29. Burnard et al. "Generating realistic artificial human genomes using adversarial autoencoders" PMC, 2025

---

## 7. 신뢰도 평가

| 항목 | 신뢰도 | 근거 |
|------|--------|------|
| FiLM이 유전형 생성에 미적용 | **높음** (95%) | 5개 독립 검색 경로 모두 일치 |
| 유전학 보조 손실 미적용 | **높음** (90%) | 모든 주요 논문이 사후 평가만 보고 |
| n=2,504에서 FiLM이 작동 | **중간-높음** (80%) | FiLM 자체는 경량이나, 실험적 검증 필요 |
| SCI 논문 게재 가능성 | **중간-높음** (75%) | Novelty 확보, 실험 완성도에 따라 결정 |
