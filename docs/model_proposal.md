# HybridGenoDiT 모델 구성 기획서

**프로젝트명**: Population-Robust Synthetic Genotype Generation via Hybrid CNN-DiT with Hierarchical FiLM
**작성일**: 2026-03-27
**대상 데이터**: 1000 Genomes Phase 3 (2,504명, 26 인구군, chr1-22)
**목표**: SCI 논문 출판 (Bioinformatics / Genome Biology / PLOS Comp Bio 급)

---

## 1. 연구 목적 및 동기

### 1.1 문제 정의

기존 유전형 합성 모델(GeneDiffusion, Genome-AC-GAN 등)은 **인구군 조건부 생성에서 소수 인구군에 취약**하다.
26개 인구군 중 ASW(61명), MXL(64명) 등은 학습 데이터가 절대적으로 부족하여
생성 품질이 대규모 인구군(YRI 108명, GWD 113명) 대비 현저히 저하된다.

### 1.2 제안 해결책

1. **FiLM(Feature-wise Linear Modulation)** 을 유전형 Diffusion에 최초 적용
2. **계층적 인구군 임베딩** (pop + superpop)으로 소수 인구군의 정보 부족 문제 완화
3. **CNN-DiT 하이브리드** 로 근거리(LD) + 장거리(유전자 간) 패턴을 역할 분리하여 포착
4. **PCA 공간 분포 매칭 보조 손실** 로 인구군별 생성 충실도 향상

### 1.3 Novelty (선행 연구 대비)

| # | Contribution | 선행 연구 존재 |
|---|-------------|--------------|
| 1 | FiLM의 유전형 생성 최초 적용 | 없음 (AlphaFold3/TaxDiff는 단백질) |
| 2 | CNN-DiT 하이브리드 유전형 Diffusion | 없음 |
| 3 | 계층적 인구군 임베딩(pop+superpop) FiLM 변조 | 없음 |
| 4 | 소수 인구군 강건성 정량 검증 | 부분적 (Genome-AC-GAN만) |

---

## 2. 데이터 파이프라인

### 2.1 입력 데이터

```
ALL.autosomes.phase3.genotypes.vcf.gz (13.9 GB)
├── 2,504 샘플 × 22 상염색체
├── 5 슈퍼인구군 (AFR/EUR/EAS/SAS/AMR)
├── 26 세부 인구군
└── FORMAT: GT (phased genotype)
```

### 2.2 전처리 흐름 (PCA 기반, 기존 파이프라인 활용)

```
VCF (chr1-22, 13.9GB)
  → biallelic SNP 필터 + MAF > 0.01
  → refGene(hg19) 유전자 어노테이션
  → 유전자별 PCA (n_components=8)
  → 토큰화: (2504, ~21819, 8)
  → 정규화 + 제로 패딩 → (2504, 26624, 8)
  → permute → 최종 입력 텐서: (batch, 8, 26624)
```

### 2.3 레이블 체계

```
Level 1 - 슈퍼인구군 (5): AFR(661), EUR(503), EAS(504), SAS(489), AMR(347)
Level 2 - 세부 인구군 (26): GBR(91), YRI(108), CHB(103), ..., ASW(61), MXL(64)

인구군 → 슈퍼인구군 매핑:
  AFR: ACB, ASW, ESN, GWD, LWK, MSL, YRI
  EUR: CEU, FIN, GBR, IBS, TSI
  EAS: CDX, CHB, CHS, JPT, KHV
  SAS: BEB, GIH, ITU, PJL, STU
  AMR: CLM, MXL, PEL, PUR
```

---

## 3. 모델 아키텍처

### 3.1 전체 구조

```
Input (B, 8, 26624) + pop_label (B,)
         │                   │
         │              ┌────┴─────────────────┐
         │              │ Hierarchical Pop Emb  │
         │              │ pop(26,d)+superpop(5,d)│
         │              │ → fusion MLP → (B,d)  │
         │              └────┬─────────────────┘
         │                   │
         │              ┌────┴─────────────────┐
         │              │ + Timestep Embedding  │
         │              │ sinusoidal(t) → MLP   │
         │              └────┬─────────────────┘
         │                   │
         │              ┌────┴─────────────────┐
         │              │ Unified FiLM Generator│
         │              │ → CNN (γ_c, β_c)      │
         │              │ → DiT (γ_d, β_d, α_d) │
         │              └──┬────────────┬──────┘
         │                 │            │
         ▼                 ▼            │
┌─────────────────────────────┐        │
│ CNN Stem Encoder + FiLM     │        │
│                             │        │
│ (8, 26624)                  │        │
│  → FiLMConv(C)   + ↓2      │        │
│  → FiLMConv(C)   + ↓2      │        │
│  → FiLMConv(2C)  + ↓2      │        │
│  → FiLMConv(4C)            │        │
│ = (4C, 3328)               │        │
│ + skip connections 저장     │        │
└──────────┬──────────────────┘        │
           │                           │
           ▼                           │
┌─────────────────────────────┐        │
│ Patchify + Position Embed   │        │
│ (4C, 3328) → (N_tok, d)    │        │
│ patch_size=16 → 208 tokens  │        │
└──────────┬──────────────────┘        │
           │                           │
           ▼                           ▼
┌──────────────────────────────────────────┐
│ DiT Core (AdaLN-Zero = FiLM)            │
│                                          │
│  Block l (× N_blocks):                   │
│    h = γ₁·LayerNorm(x) + β₁   ← FiLM   │
│    h = MultiHeadAttention(h)             │
│    x = x + α₁·h               ← gate    │
│    h = γ₂·LayerNorm(x) + β₂   ← FiLM   │
│    h = FFN(h)                            │
│    x = x + α₂·h               ← gate    │
│                                          │
│  * α 초기값 = 0 (AdaLN-Zero)            │
│  * 학습 초기: DiT ≈ 항등함수             │
│  * 점진적으로 장거리 패턴 학습            │
└──────────┬───────────────────────────────┘
           │
           ▼
┌─────────────────────────────┐
│ Un-patchify                 │
│ (208, d) → (4C, 3328)      │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│ CNN-DiT Balance             │
│ w·DiT_out + (1-w)·CNN_out  │
│ (w는 학습 가능, 조건 의존)  │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│ CNN Decoder + FiLM + Skips  │
│                             │
│ (4C, 3328)                  │
│  → FiLMDeconv(2C) + ↑2     │
│  → FiLMDeconv(C)  + ↑2     │
│  → FiLMDeconv(C)  + ↑2     │
│  → Conv1d(8)                │
│ = (8, 26624)                │
└──────────┬──────────────────┘
           │
           ▼
Output: predicted noise ε (B, 8, 26624)
```

### 3.2 FiLM의 역할 요약

| 위치 | FiLM이 하는 일 | 유전학적 의미 |
|------|---------------|-------------|
| CNN Encoder | 근거리 필터 채널을 인구군별 증폭/감쇠 | AFR: 짧은 LD → 고주파 증폭, EUR: 긴 LD → 저주파 증폭 |
| DiT Blocks | Attention 출력 스케일링 + FFN 활성화 시프트 | 인구군별 장거리 유전자 상호작용 패턴 조절 |
| CNN Decoder | 복원 과정에서 인구군별 미세 조정 | 인구군 특이적 대립유전자 빈도 분포 복원 |

### 3.3 소수 인구군 강건성 메커니즘

```
ASW (61명) 단독 학습:
  → pop_emb(ASW) 만으로 61개 샘플에서 패턴 학습 → 불안정

ASW + 계층적 FiLM:
  → superpop_emb(AFR) 는 AFR 전체 661명에서 학습 → 안정적 기반
  → pop_emb(ASW) 는 AFR 공통 패턴 위에 ASW 미세 차이만 학습 → 경량
  → fusion(pop + superpop) → 안정적 + 특이적 조합
```

---

## 4. 손실 함수

### 4.1 주 손실: Diffusion L2

```
L_diffusion = E[||ε - ε_θ(x_t, t, y)||²]

x_t = √(ᾱ_t) · x_0 + √(1-ᾱ_t) · ε    (forward process)
ε_θ: 모델의 노이즈 예측
```

### 4.2 보조 손실: PCA 공간 분포 매칭

```
L_aux = λ₁ · L_pca_dist + λ₂ · L_pop_structure

L_pca_dist: 인구군별 PCA 계수 분포의 MMD (Maximum Mean Discrepancy)
  - 생성된 샘플 vs 실제 샘플의 유전자별 PCA 분포가 일치하는지
  - RBF 커널: k(x,y) = exp(-||x-y||² / 2σ²)

L_pop_structure: 인구군 구조 보존 손실
  - 생성 샘플의 PCA projection이 실제 인구군 클러스터와 겹치는지
  - Sliced Wasserstein Distance로 측정
```

### 4.3 Enforce Zeros

```
zero_mask: 학습 데이터에서 항상 0인 위치
생성 시 해당 위치를 0으로 강제 → 생물학적 제약 반영
```

### 4.4 전체 목적 함수

```
L_total = L_diffusion + λ₁·L_pca_dist + λ₂·L_pop_structure

λ₁, λ₂: 보조 손실 가중치 (warmup 후 활성화)
```

---

## 5. 하이퍼파라미터 랜덤 서치 구성

### 5.1 탐색 공간 요약 (10개 카테고리, 51개 파라미터)

#### Training (11개)

| 파라미터 | 유형 | 범위 |
|----------|------|------|
| lr_diffusion | log_uniform | [1e-6, 1e-2] |
| batch_size | categorical | {8, 16, 32, 64, 128} |
| epochs | int_uniform | [50, 500] |
| gradient_accumulation_steps | categorical | {1, 2, 4, 8} |
| optimizer | categorical | {adam, adamw, sgd, radam, lamb} |
| weight_decay | log_uniform | [1e-8, 1e-1] |
| gradient_clipping | uniform | [0.1, 10.0] |
| ema_decay | uniform | [0.9, 0.9999] |
| lr_scheduler | categorical | {cosine_warmup, linear_warmup, constant, cosine_annealing, one_cycle} |
| warmup_ratio | uniform | [0.01, 0.2] |
| lr_min_ratio | log_uniform | [1e-4, 0.5] |

#### Diffusion Process (7개)

| 파라미터 | 유형 | 범위 |
|----------|------|------|
| max_timesteps | categorical | {200, 300, 500, 750, 1000, 1500} |
| noise_schedule | categorical | {cosine, linear, sigmoid, sqrt} |
| prediction_target | categorical | {epsilon, x0, v_prediction} |
| sampling_timesteps | categorical | {50, 100, 200, 500} |
| guidance_type | categorical | {normal, classifier_free} |
| guidance_weight | uniform | [0.0, 10.0] |
| cfg_dropout | uniform | [0.0, 0.5] |

#### CNN Stem (9개)

| 파라미터 | 유형 | 범위 |
|----------|------|------|
| cnn_base_channels | categorical | {32, 48, 64, 96, 128, 192, 256} |
| cnn_n_blocks | int_uniform | [2, 6] |
| cnn_channel_multiplier | categorical | {(1,1,2,4), (1,2,2,4), (1,2,4,4), (1,2,4,8), ...} |
| cnn_kernel_size | categorical | {3, 5, 7, 9} |
| cnn_norm_type | categorical | {group_norm, layer_norm, instance_norm, batch_norm} |
| cnn_activation | categorical | {silu, gelu, relu, mish} |
| cnn_dropout | uniform | [0.0, 0.5] |
| cnn_use_residual | categorical | {True, False} |
| cnn_downsample_mode | categorical | {conv_stride2, avg_pool, max_pool, conv_stride4} |

#### DiT Core (9개)

| 파라미터 | 유형 | 범위 |
|----------|------|------|
| dit_d_model | categorical | {128, 192, 256, 384, 512, 768} |
| dit_n_blocks | int_uniform | [2, 12] |
| dit_n_heads | categorical | {2, 4, 6, 8, 12, 16} |
| dit_mlp_ratio | uniform | [1.0, 8.0] |
| dit_dropout | uniform | [0.0, 0.5] |
| dit_attention_dropout | uniform | [0.0, 0.3] |
| dit_use_flash_attention | categorical | {True, False} |
| patch_size | categorical | {4, 8, 16, 32, 64, 128} |
| pos_embedding_type | categorical | {learned, sinusoidal, rotary, alibi, none} |

#### FiLM / AdaLN (10개)

| 파라미터 | 유형 | 범위 |
|----------|------|------|
| film_type | categorical | {adaln_zero, film_simple, adaln, scale_only, bias_only} |
| pop_emb_dim | categorical | {64, 128, 256, 384, 512} |
| superpop_emb_dim | categorical | {32, 64, 128, 256} |
| hierarchy_fusion | categorical | {concat_mlp, add, gate, cross_attention, film_on_film} |
| timestep_emb_dim | categorical | {128, 256, 384, 512} |
| film_hidden_layers | int_uniform | [1, 4] |
| film_hidden_dim | categorical | {128, 256, 512, 768, 1024} |
| film_activation | categorical | {silu, gelu, relu, tanh} |
| film_cnn_enabled | categorical | {True, False} |
| film_shared_generator | categorical | {True, False} |

#### Balance / Aux Loss / Regularization / Data / Sampling (15개)

| 파라미터 | 유형 | 범위 |
|----------|------|------|
| balance_mode | categorical | {learned_scalar, learned_time_dependent, learned_pop_dependent, residual_add, gated_residual, dit_only, concat_proj} |
| aux_loss_enabled | categorical | {True, False} |
| lambda_pca_dist | log_uniform | [1e-5, 10.0] |
| lambda_pop_structure | log_uniform | [1e-5, 10.0] |
| aux_loss_type | categorical | {mmd_rbf, mmd_linear, sliced_wasserstein, kl_divergence, cosine_distance} |
| aux_loss_warmup_epochs | int_uniform | [0, 100] |
| mmd_kernel_bandwidth | log_uniform | [0.01, 100.0] |
| dropout_global | uniform | [0.0, 0.5] |
| label_smoothing | uniform | [0.0, 0.3] |
| stochastic_depth_rate | uniform | [0.0, 0.3] |
| input_noise_augmentation | uniform | [0.0, 0.1] |
| enforce_zeros | categorical | {True, False} |
| normalize_data | categorical | {True, False} |
| use_ddim | categorical | {True, False} |
| ddim_eta | uniform | [0.0, 1.0] |

### 5.2 제약 조건 (Constraints)

```
1. dit_d_model % dit_n_heads == 0         (어텐션 차원 호환)
2. len(cnn_channel_multiplier) == cnn_n_blocks
3. guidance_type != "classifier_free" → guidance_weight=0, cfg_dropout=0
4. aux_loss_enabled == False → lambda_pca_dist=0, lambda_pop_structure=0
5. effective_batch = batch_size * grad_accum ≤ 256
6. seq_after_cnn % patch_size == 0        (패치 크기 호환)
7. estimated_params < 50M                 (n=2504에 대한 상한)
```

### 5.3 탐색 전략

```
총 시행 횟수: 200회 (확장 가능)
방법: Random Search (균일 랜덤 + 제약 조건 후처리)
선정 기준: val_reconstruction_error (primary), recovery_rate (secondary)
조기 종료: 20 에포크 동안 val_loss 개선 없으면 중단
```

---

## 6. 파라미터 수 추정

### 6.1 기준 설정 (C=64, d=256, DiT 4블록)

| 모듈 | 파라미터 수 |
|------|-----------|
| Hierarchical Pop Embedding | ~0.07M |
| Unified FiLM Generator | ~0.5M |
| CNN Encoder (4 blocks) | ~1.2M |
| Patchify + Position Embedding | ~0.3M |
| DiT Core (4 blocks, d=256) | ~3.2M |
| CNN Decoder (4 blocks) | ~1.5M |
| **전체** | **~6.8M** |

### 6.2 탐색 범위 내 파라미터 수 범위

| 설정 | 예상 파라미터 수 | 비고 |
|------|----------------|------|
| 최소 (C=32, d=128, DiT 2블록) | ~1.5M | 극도로 경량 |
| 기준 (C=64, d=256, DiT 4블록) | ~6.8M | 권장 시작점 |
| 중간 (C=128, d=384, DiT 6블록) | ~25M | 중간 규모 |
| 최대 (C=256, d=768, DiT 12블록) | ~120M+ | 오버피팅 위험, 참고용 |

### 6.3 현재 GeneDiffusion UnetCombined과 비교

| | UnetCombined (현재) | HybridGenoDiT (기준) |
|--|--------------------|-----------------------|
| 파라미터 수 | ~10-15M (추정) | ~6.8M |
| 조건부 방식 | one-hot 곱셈 | 계층적 FiLM (AdaLN-Zero) |
| 장거리 패턴 | MLP (flat) | DiT self-attention |
| 근거리 패턴 | 1D CNN | 1D CNN + FiLM |
| 인구군 강건성 | 동일 가중치 | FiLM 인구군별 변조 |

---

## 7. 실험 설계

### 7.1 비교 모델 (Baselines)

| 모델 | 유형 | 출처 |
|------|------|------|
| GeneDiffusion (UnetCombined) | Diffusion + UNet | Kenneweg et al., ISMB 2025 |
| GeneDiffusion + CFG | + Classifier-free guidance | 자체 구현 |
| GeneDiffusion + Cross-Attn | + SNPgen 스타일 조건부 | 자체 구현 |
| Genome-AC-GAN | AC-GAN | Ahronoviz et al., 2024 |
| HAPNEST | 비딥러닝 시뮬레이터 | Wharrie et al., 2023 |
| **HybridGenoDiT (제안)** | CNN-DiT + FiLM | 본 연구 |

### 7.2 Ablation Study

| 실험 | 변경 사항 | 검증 목적 |
|------|----------|----------|
| A1. DiT 제거 (CNN-only + FiLM) | DiT 블록 제거 | DiT의 장거리 패턴 기여 |
| A2. FiLM 제거 (CNN-DiT + one-hot) | FiLM → 기존 one-hot 곱셈 | FiLM의 조건부 효과 |
| A3. 계층 제거 (pop_emb only) | superpop_emb 제거 | 계층적 임베딩의 효과 |
| A4. CNN FiLM 제거 (DiT FiLM only) | CNN 블록의 FiLM 비활성화 | CNN FiLM의 기여 |
| A5. 보조 손실 제거 | L_aux = 0 | 분포 매칭 손실의 효과 |
| A6. AdaLN → Film_simple | Zero-init 제거 | AdaLN-Zero의 학습 안정성 기여 |

### 7.3 평가 지표

| 카테고리 | 지표 | 측정 방법 |
|----------|------|----------|
| **충실도** | AF 상관 (전체) | Pearson r (실제 vs 합성 대립유전자 빈도) |
| **충실도** | AF 상관 (저빈도, MAF≤0.05) | 저빈도 변이 재현 능력 |
| **충실도** | LD 감쇠 상관 | 거리별 r² 비교 |
| **구조** | PCA 클러스터 겹침도 | Silhouette score, 시각화 |
| **유용성** | Recovery Rate | 합성 데이터 학습 → 실제 테스트 정확도 |
| **유용성** | 증강 효과 | 5%/10%/50% 실제 + 합성 혼합 시 정확도 |
| **프라이버시** | NNAA | 0.5 근접 여부 |
| **프라이버시** | 멤버십 추론 AUC | 0.5 근접 여부 |
| **다양성** | k-mer 엔트로피 | 4-mer/8-mer SNP 윈도우 분포 |
| **강건성** | 인구군별 품질 분산 | **핵심 신규 지표**: 인구군 크기 vs 품질 상관 감소 |

### 7.4 핵심 실험: 소수 인구군 강건성

```
실험 목적: FiLM + 계층적 임베딩이 소수 인구군 생성 품질을 개선하는가?

측정:
  1. 인구군별 Recovery Rate 계산
  2. 인구군 크기(n)와 Recovery Rate 간 Pearson 상관 계산
  3. FiLM 적용 전후의 상관 계수 비교
     - 높은 상관 (r→1): 소수 인구군일수록 품질 저하 → 나쁨
     - 낮은 상관 (r→0): 크기와 무관한 균일 품질 → 좋음

가설: FiLM 적용 시 |r|이 유의하게 감소
```

---

## 8. 구현 로드맵

### Phase 1: 기반 구축 (1~2주)

```
[ ] HierarchicalPopulationEmbedding 구현 (pop_to_superpop 매핑)
[ ] UnifiedFiLMGenerator 구현
[ ] FiLMConvBlock / CNNStemEncoder 구현
[ ] DiTBlock (AdaLN-Zero) / DiTCore 구현
[ ] PatchEmbed1D / Un-patchify 구현
[ ] CNNDecoder 구현
[ ] HybridCNNDiTFiLM 전체 조립
[ ] 기존 main_1k.py에 통합 (model_name="HybridGenoDiT")
[ ] 단위 테스트: forward pass shape 확인
```

### Phase 2: 학습 + 랜덤 서치 (2~4주)

```
[ ] 하이퍼파라미터 랜덤 서치 인프라 (wandb sweep 또는 자체 스크립트)
[ ] 200회 시행 중 상위 10개 설정 선별
[ ] 상위 설정으로 full 학습 (100+ 에포크)
[ ] 기준 설정(C=64, d=256, 4 blocks) 먼저 학습하여 파이프라인 검증
```

### Phase 3: 평가 + Ablation (2~3주)

```
[ ] 전체 비교 모델 학습 (baselines)
[ ] 6가지 ablation study 실행
[ ] 인구군별 강건성 분석 (핵심 실험)
[ ] 전체 평가 지표 측정 및 표/그림 생성
```

### Phase 4: 논문 작성 (2~3주)

```
[ ] 논문 초안 (Introduction, Method, Experiments, Results, Discussion)
[ ] 그림 제작 (아키텍처 다이어그램, 인구군별 결과, ablation 그래프)
[ ] 타겟 저널 선정 및 투고
```

---

## 9. 예상 리스크 및 대응

| 리스크 | 확률 | 대응 |
|--------|------|------|
| DiT 오버피팅 | 중간 | AdaLN-Zero + 조기 종료 + dropout 조합, 또는 DiT 블록 수 축소 |
| FiLM 효과 미미 | 낮음 | ablation에서 효과 없으면 "negative result" 보고, 또는 보조 손실 강화 |
| 소수 인구군 개선 미미 | 중간 | superpop 수준 FiLM만으로도 논문 기여 가능, 또는 pop별 오버샘플링 |
| 학습 불안정 | 낮음 | CNN이 기반 제공 + DiT는 Zero-gate → 점진적 기여 |
| 연산 리소스 부족 | 중간 | dit_n_blocks=2~4, d=128~256으로 축소 가능 |

---

## 10. 타겟 저널 및 일정

| 저널 | IF | 투고 목표 |
|------|-----|----------|
| **Bioinformatics** (1순위) | 5.8 | GeneDiffusion 후속, 방법론 중심 |
| **Genome Biology** (2순위) | 12.3 | 생물학적 검증 강화 시 |
| **PLOS Computational Biology** (3순위) | 4.3 | 합성 유전체 분야 |

**전체 일정 목표**: 8~12주 (구현 → 실험 → 논문 작성)
