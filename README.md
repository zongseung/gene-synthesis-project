# HybridGenoDiT

**Population-Conditional Synthetic Genotype Generation via Hybrid CNN-DiT with Hierarchical FiLM Conditioning**

1000 Genomes Phase 3 데이터(2,504 samples, 26 populations, 5 superpopulations)를 활용하여 인구군별 조건부 합성 유전형을 생성하는 Diffusion 모델.

---

## Why This Model?

### 문제: 소수 인구군의 합성 유전형 품질 저하

기존 유전형 합성 모델(GeneDiffusion, Genome-AC-GAN)은 **소수 인구군에서 치명적인 성능 저하**를 보인다.

```
1000 Genomes Phase 3 — 26개 인구군의 샘플 수 불균형

  YRI ████████████████████████████████████████████ 108
  GWD ██████████████████████████████████████████████ 113
  CLM ████████████████████████████████████████ 94
  ...
  MXL ██████████████████████████ 64        ← 소수 인구군
  ASW ████████████████████████ 61          ← 소수 인구군

  ─────────────────────────────────────────────
  최대 113 vs 최소 61 → ~1.85x 차이
  superpop 기준: AFR 661 vs AMR 347 → ~1.9x 차이
```

**근본 원인**: 절대적인 학습 데이터 부족. 61개 샘플로 인구군 특이적 대립유전자 빈도(AF), 연관 불균형(LD), 하플로타입 다양성을 학습하기에 불충분하다.

**결과**: 소수 인구군에서 생성된 합성 유전형은 AF 상관이 낮고, LD 구조가 왜곡되며, 다운스트림 분석(GWAS 보정, 임퓨테이션 패널)에 사용할 수 없다.

### 해결: FiLM 기반 계층적 인구군 조건화

HybridGenoDiT는 세 가지 핵심 메커니즘으로 이 문제를 해결한다:

#### 1. 계층적 인구군 임베딩 (Hierarchical Population Embedding)

```
기존 방식 (ASW 단독 학습):
  ASW 61 samples → pop_emb(ASW) ← 61개 gradient 신호만
  → 불안정, 고분산, 저품질 생성

제안 방식 (ASW + AFR superpopulation 공유):
  AFR 661 samples → superpop_emb(AFR) ← 661개 gradient 신호 → 안정적 기반
  ASW 61 samples  → pop_emb(ASW)      ← ASW 고유 잔차만 학습
  fusion(pop + superpop) → 안정적 + 특이적 조합

  결과: ASW 강건성 대폭 향상, AFR 내 다른 인구군과의 차별성 유지
```

#### 2. CNN-DiT 하이브리드 아키텍처

| 유전학 도메인 지식 | 모델 반영 |
|-------------------|----------|
| LD(연관 불균형)는 근거리 패턴 | **CNN**이 local LD 포착 |
| 유전자 간 상호작용은 장거리 | **DiT self-attention**이 포착 |
| 인구군 간 유전적 거리는 계층적 | 계층적 pop + superpop 임베딩 |
| 유전체의 99.9%는 공통 | 공유 백본, **FiLM**으로 0.1% 차이만 변조 |
| 특정 위치는 항상 0 (패딩/공통) | `enforce_zeros` + `zero_mask` |

#### 3. FiLM (Feature-wise Linear Modulation)

```
FiLM: output = γ · input + β

γ (scale): 인구군별 특정 유전자 영역의 중요도 조절
β (shift): 인구군별 대립유전자 빈도의 기저 수준(baseline) 이동

예시:
  AFR → γ_LD↑ (짧은 LD → 고주파 패턴 강화)
  EUR → γ_LD↓ (긴 LD → 저주파 패턴 강화)
```

### Novelty

| # | Contribution | 선행 연구 |
|---|-------------|----------|
| 1 | **FiLM의 유전형 생성 최초 적용** | 없음 (AlphaFold3/TaxDiff는 단백질 한정) |
| 2 | **CNN-DiT 하이브리드 유전형 Diffusion** | 없음 |
| 3 | **계층적 인구군 임베딩 + FiLM 변조** | 없음 |
| 4 | **소수 인구군 강건성의 정량적 검증** | 부분적 (Genome-AC-GAN만) |

---

## Model Architecture

### Default config (baseline)

| 항목 | 값 | 근거 (코드) |
|------|----|-------------|
| `in_channels (K)` | 8 | `hybrid_geno_dit.py:47-49` |
| `gene_size` | 26624 | `hybrid_geno_dit.py:50` |
| `base_channels` | 64 | `hybrid_geno_dit.py:51` |
| `channel_mult` | (1, 1, 2, 4) | `hybrid_geno_dit.py:52` |
| `n_downsamples` | 3 = `len(channel_mult) - 1` | `hybrid_geno_dit.py:76` |
| `latent_size` | 26624 / 8 = 3328 | `hybrid_geno_dit.py:77` |
| `latent_channels (4C)` | 256 | `hybrid_geno_dit.py:72` |
| `d_model` | 256 | `hybrid_geno_dit.py:54` |
| `n_dit_blocks` | 4 | `hybrid_geno_dit.py:55` |
| `n_heads` | 4 | `hybrid_geno_dit.py:56` |
| `mlp_ratio` | 4.0 | `hybrid_geno_dit.py:57` |
| `patch_size` | 16 → `n_tokens = 208` | `hybrid_geno_dit.py:59`, `dit.py:36` |
| `n_pops (+null)` | 26 (+1 = 27, CFG null) | `hybrid_geno_dit.py:60`, `conditioning.py:43` |
| `n_superpops (+null)` | 5 (+1 = 6) | `hybrid_geno_dit.py:61`, `conditioning.py:46-50` |

### 최상위 데이터 흐름 (`HybridCNNDiTFiLM.forward`)

```mermaid
flowchart TD
    subgraph INPUT["INPUT"]
        X["x : (B, 8, 26624)<br/>noisy Gene-PCA tensor"]
        T["t : (B,)<br/>diffusion timestep"]
        Y["y : (B,)<br/>pop_label ∈ [0,25] ∪ {26 = CFG null}"]
    end

    subgraph COND["CONDITIONING PATH"]
        PE["HierarchicalPopulationEmbedding<br/>pop_emb ⊕ superpop_emb → fusion MLP<br/>→ (B, 256)"]
        FG["UnifiedFiLMGenerator<br/>time_mlp(t) ⊕ pop_emb → cond_mlp<br/>→ (cnn_enc, cnn_dec, dit) FiLM params"]
    end

    subgraph ENC["CNN ENCODER  (local LD)"]
        E1["FiLMConvBlock #1<br/>8 → 64,   L: 26624 → 13312"]
        E2["FiLMConvBlock #2<br/>64 → 64,  L: 13312 → 6656"]
        E3["FiLMConvBlock #3<br/>64 → 128, L: 6656 → 3328"]
        E4["FiLMConvBlock #4 (no ↓)<br/>128 → 256, L: 3328"]
    end

    subgraph DITCORE["DiT CORE  (long-range gene interactions)"]
        P["PatchEmbed1D<br/>(B,256,3328) → reshape + Linear<br/>→ (B, 208, 256) + learned pos_emb"]
        D["DiTCore × 4 blocks<br/>AdaLN-Zero self-attn + MLP"]
        U["UnPatchify1D<br/>(B, 208, 256) → Linear + reshape<br/>→ (B, 256, 3328)"]
    end

    subgraph DEC["CNN DECODER  (reconstruction + skips)"]
        D1["FiLMDeconvBlock #1<br/>256 → 128 + skip₄"]
        D2["FiLMDeconvBlock #2<br/>128 → 64  + skip₃"]
        D3["FiLMDeconvBlock #3<br/>64  → 64  + skip₂"]
        D4["FiLMDeconvBlock #4 (no ↑)<br/>64  → 64  + skip₁"]
        FC["Conv1d 1×1<br/>64 → 8"]
    end

    subgraph OUT["OUTPUT"]
        Z["enforce_zeros<br/>output × (~zero_mask)"]
        OUTX["ε̂ : (B, 8, 26624)"]
    end

    X --> E1 --> E2 --> E3 --> E4 --> P --> D --> U --> D1 --> D2 --> D3 --> D4 --> FC --> Z --> OUTX

    T --> FG
    Y --> PE --> FG

    FG -. γ,β .-> E1
    FG -. γ,β .-> E2
    FG -. γ,β .-> E3
    FG -. γ,β .-> E4
    FG -. γ,β,α × 6 .-> D
    FG -. γ,β .-> D1
    FG -. γ,β .-> D2
    FG -. γ,β .-> D3
    FG -. γ,β .-> D4

    E1 -. skip₁ .-> D4
    E2 -. skip₂ .-> D3
    E3 -. skip₃ .-> D2
    E4 -. skip₄ .-> D1

    classDef cond fill:#fde68a,stroke:#b45309,color:#000
    classDef enc  fill:#bfdbfe,stroke:#1e40af,color:#000
    classDef dit  fill:#ddd6fe,stroke:#5b21b6,color:#000
    classDef dec  fill:#bbf7d0,stroke:#166534,color:#000
    classDef io   fill:#f3f4f6,stroke:#374151,color:#000

    class PE,FG cond
    class E1,E2,E3,E4 enc
    class P,D,U dit
    class D1,D2,D3,D4,FC dec
    class X,T,Y,Z,OUTX io
```

> 참고: 인코더는 블록 0–2가 stride-2 downsample을 수행하고 마지막 블록(#4)은 채널만 확장한다 (`cnn.py:167-173`). 디코더는 이 구조를 대칭적으로 반전해 앞 3개 블록이 ConvTranspose1d로 2배 upsample하고 마지막 블록은 길이를 유지한다 (`cnn.py:219-231`).

### Conditioning Path (`HierarchicalPopulationEmbedding` + `UnifiedFiLMGenerator`)

```mermaid
flowchart LR
    Y["pop_label y (B,)<br/>0..25 (+26 = null for CFG)"]
    T["timestep t (B,)"]

    subgraph HPE["HierarchicalPopulationEmbedding"]
        PMAP["pop_to_superpop_map<br/>(n_pops+1,) long buffer"]
        PEMB["nn.Embedding(27, 256)"]
        SEMB["nn.Embedding(6, 256)"]
        CAT1["concat → (B, 512)"]
        FUSE["Linear 512→256 · SiLU · Linear 256→256"]
        POUT["pop_emb (B, 256)"]
    end

    subgraph TME["Timestep path"]
        SINE["sinusoidal timestep_embedding<br/>dim=256, max_period=10000"]
        TMLP["Linear 256→256 · SiLU · Linear 256→256"]
        TOUT["t_emb (B, 256)"]
    end

    subgraph UFG["UnifiedFiLMGenerator"]
        CAT2["concat [pop_emb, t_emb] → (B, 512)"]
        CMLP["cond_mlp<br/>Linear 512→256 · SiLU · Linear 256→256<br/>→ cond (B, 256)"]

        subgraph CNN_ENC_FILM["cnn_enc_films (ModuleList × 4)"]
            LE1["Linear 256 → 2·64"]
            LE2["Linear 256 → 2·64"]
            LE3["Linear 256 → 2·128"]
            LE4["Linear 256 → 2·256"]
        end

        subgraph CNN_DEC_FILM["cnn_dec_films (ModuleList × 4)"]
            LD1["Linear 256 → 2·128"]
            LD2["Linear 256 → 2·64"]
            LD3["Linear 256 → 2·64"]
            LD4["Linear 256 → 2·64"]
        end

        subgraph DIT_FILM["dit_films (ModuleList × 4, AdaLN-Zero)"]
            DF["SiLU → zero_module(Linear 256 → 6·256)<br/>per block → (B, 1536)"]
        end
    end

    Y --> PEMB --> CAT1
    Y --> PMAP --> SEMB --> CAT1 --> FUSE --> POUT
    T --> SINE --> TMLP --> TOUT

    POUT --> CAT2
    TOUT --> CAT2 --> CMLP

    CMLP -. cond .-> LE1 & LE2 & LE3 & LE4
    CMLP -. cond .-> LD1 & LD2 & LD3 & LD4
    CMLP -. cond .-> DF

    LE1 -->|"chunk(2) → γ,β"| EO1["→ Encoder block 1"]
    LE2 --> EO2["→ Encoder block 2"]
    LE3 --> EO3["→ Encoder block 3"]
    LE4 --> EO4["→ Encoder block 4"]

    LD1 --> DO1["→ Decoder block 1"]
    LD2 --> DO2["→ Decoder block 2"]
    LD3 --> DO3["→ Decoder block 3"]
    LD4 --> DO4["→ Decoder block 4"]

    DF -->|"chunk(6) → γ₁,β₁,α₁,γ₂,β₂,α₂"| DITO["→ DiT blocks × 4"]

    classDef hpe fill:#fde68a,stroke:#92400e,color:#000
    classDef tme fill:#fecaca,stroke:#991b1b,color:#000
    classDef ufg fill:#ddd6fe,stroke:#5b21b6,color:#000
    class PMAP,PEMB,SEMB,CAT1,FUSE,POUT hpe
    class SINE,TMLP,TOUT tme
    class CAT2,CMLP,LE1,LE2,LE3,LE4,LD1,LD2,LD3,LD4,DF ufg
```

### 블록 내부 구조

#### `FiLMConvBlock` — 인코더 블록 (`cnn.py:18-74`)

```mermaid
flowchart TD
    X["x (B, C_in, L)"]
    G["γ (B, C_out)"]
    B["β (B, C_out)"]

    SP["skip_proj: Conv1d 1×1 (C_in≠C_out) else Identity"]
    C1["Conv1d k=3, pad=1   C_in → C_out"]
    N1["GroupNorm(min(32, C_out), C_out)"]
    FILM["FiLM: γ.unsqueeze(-1) · h + β.unsqueeze(-1)"]
    A1["SiLU"]
    C2["Conv1d k=3   C_out → C_out"]
    N2["GroupNorm"]
    A2["SiLU"]
    ADD((+))
    DS["downsample: Conv1d k=2, s=2  (if not last)  else Identity"]

    X --> C1 --> N1 --> FILM --> A1 --> C2 --> N2 --> A2 --> ADD
    X --> SP --> ADD
    G --> FILM
    B --> FILM

    ADD -->|"skip (pre-downsample)"| SKIPOUT["skip → decoder"]
    ADD --> DS --> OUT["out (B, C_out, L' = L/2 or L)"]

    classDef film fill:#fde68a,stroke:#92400e,color:#000
    class FILM film
```

#### `DiTBlock` — AdaLN-Zero 블록 (`dit.py:94-162`)

```mermaid
flowchart TD
    X["x (B, N=208, d=256)"]
    FP["film_params (B, 1536)"]
    CHK["chunk(6, dim=-1)<br/>→ γ₁,β₁,α₁,γ₂,β₂,α₂  each (B,1,256)"]

    subgraph ATTN["Self-Attention branch"]
        N1["LayerNorm(elementwise_affine=False)"]
        M1["h = γ₁·h + β₁"]
        MHA["MultiheadAttention<br/>d=256, heads=4, batch_first"]
        G1["x + α₁·h   (α₁ ≈ 0 at init)"]
    end

    subgraph MLP["FFN branch"]
        N2["LayerNorm(elementwise_affine=False)"]
        M2["h = γ₂·h + β₂"]
        F1["Linear 256 → 1024"]
        GE["GELU"]
        DR1["Dropout"]
        F2["Linear 1024 → 256"]
        DR2["Dropout"]
        G2["x + α₂·h   (α₂ ≈ 0 at init)"]
    end

    X --> N1 --> M1 --> MHA --> G1
    X -. residual .-> G1
    FP --> CHK
    CHK -. γ₁,β₁ .-> M1
    CHK -. α₁ .-> G1

    G1 --> N2 --> M2 --> F1 --> GE --> DR1 --> F2 --> DR2 --> G2
    G1 -. residual .-> G2
    CHK -. γ₂,β₂ .-> M2
    CHK -. α₂ .-> G2

    G2 --> OUT["(B, 208, 256)"]

    classDef ada fill:#ddd6fe,stroke:#5b21b6,color:#000
    class N1,M1,G1,N2,M2,G2 ada
```

`UnifiedFiLMGenerator.dit_films`가 `zero_module(Linear)`로 감싸져 있어 (`conditioning.py:131-139`) 학습 초기에 `γ=β=α=0`. LayerNorm도 `elementwise_affine=False`이므로 AdaLN-Zero의 정의에 따라 DiT는 **identity**로 시작하고, CNN 피처 위에서 점진적으로 장거리 보정을 학습한다.

#### `FiLMDeconvBlock` — 디코더 블록 (`cnn.py:77-142`)

```mermaid
flowchart TD
    X["x (B, C_in, L)"]
    SK["skip (B, skip_ch, L_skip)"]
    G["γ (B, C_out)"]
    B["β (B, C_out)"]

    UP["ConvTranspose1d k=2, s=2  (or Identity on last block)"]
    PAD["F.pad (length align)"]
    CAT["concat [h, skip] → (B, C_in+skip_ch, L_skip)"]
    SP["skip_proj: Conv1d 1×1 (concat_ch ≠ C_out)"]
    C1["Conv1d k=3   (C_in+skip_ch) → C_out"]
    N1["GroupNorm"]
    FILM["FiLM   γ · h + β"]
    A1["SiLU"]
    C2["Conv1d k=3   C_out → C_out"]
    N2["GroupNorm"]
    A2["SiLU"]
    ADD((+))

    X --> UP --> PAD --> CAT
    SK --> CAT
    CAT --> C1 --> N1 --> FILM --> A1 --> C2 --> N2 --> A2 --> ADD
    CAT --> SP --> ADD
    G --> FILM
    B --> FILM
    ADD --> OUT["out (B, C_out, L_skip)"]

    classDef film fill:#fde68a,stroke:#92400e,color:#000
    class FILM film
```

### 위치별 FiLM의 역할

| 위치 | FiLM 기능 | 유전학적 의미 |
|------|----------|-------------|
| CNN Encoder | scale/shift local filters by pop | AFR: 짧은 LD 고주파 강화 / EUR: 긴 LD 저주파 강화 |
| DiT Blocks | attention + FFN modulation + gate | 인구군별 장거리 유전자 상호작용 패턴 |
| CNN Decoder | fine-tune reconstruction by pop | 인구군별 AF 분포 복원 |

### AdaLN-Zero 학습 역학

```
학습 초기:  α ≈ 0 → DiT 출력이 0에 가까움 → 사실상 CNN만 동작
           → CNN이 먼저 local LD 구조를 안정적으로 학습

학습 중기:  α가 점진적으로 증가 → DiT가 장거리 보정 기여 시작
           → CNN의 local 표현 위에 global 패턴 추가

학습 후기:  α가 수렴 → CNN(local) + DiT(global) 최적 조합 달성
           → 인구군별 FiLM이 두 경로를 동시에 변조
```

### Diffusion Process

```
Forward (학습):
  x₀ (원본) ──noise schedule──► xₜ (노이즈 추가) ──model──► ε_pred (노이즈 예측)
  Loss = weighted_MSE(ε_pred, ε_true) × min_snr_weight(t) × (~zero_mask)

Reverse (생성):
  xₜ ~ N(0,1) ──DDIM 50 steps──► x₀ (합성 유전형)
  매 step: enforce_zeros(xₜ, zero_mask)
  생성 후: denormalize(x₀, stats) → 원래 PCA 스케일 복원
  CFG: ε = (1+w)·ε_cond - w·ε_uncond   (w=3.0)

Noise Schedule: cosine (Nichol & Dhariwal), 500 timesteps
```

### 파라미터 규모

| 모듈 | 파라미터 수 | 비고 |
|------|-----------|------|
| Hierarchical Pop Embedding | ~0.07M | pop(27) + superpop(6) + fusion MLP (null class 포함) |
| Unified FiLM Generator | ~0.5M | `time_mlp` + `cond_mlp` + per-block linear (enc 4 · dec 4 · dit 4) |
| CNN Encoder (4 blocks) | ~1.2M | stride-2 downsample × 3, 마지막 블록은 채널 확장만 |
| Patchify + Position Embedding | ~0.3M | patch_size=16 → 208 tokens |
| DiT Core (4 blocks, d=256) | ~3.2M | self-attention + FFN, AdaLN-Zero |
| CNN Decoder (4 blocks) | ~1.5M | ConvTranspose1d × 3 + skip connections + final 1×1 conv |
| **Total** | **~6.8M** | bf16: ~14MB VRAM |

---

## Evaluation Metrics

### 개요

평가는 5개 카테고리, **11개 핵심 지표**로 구성된다. 모든 지표는 Gene PCA 공간에서 직접 계산 가능하며, `ProcessPoolExecutor`로 병렬 실행한다.

```
┌─────────────────────────────────────────────────────────────────┐
│                     Evaluation Pipeline                          │
│                                                                  │
│  Real Test Data (pkl)     Synthetic Samples (.pt)               │
│       │                         │                                │
│       └────────────┬────────────┘                                │
│                    │                                             │
│    ┌───────────────┼───────────────────────────────────┐        │
│    │               │                                    │        │
│    ▼               ▼               ▼          ▼        ▼        │
│ Fidelity      Structure       Utility     Privacy   Robustness  │
│ (AF corr,     (PCA overlap,   (Recovery,  (NNAA,    (pop-size   │
│  Wasserstein)  Sliced-WD)      Augment)    DUPI,     vs quality │
│                                             MIA)      |r|)      │
│    │               │               │          │        │        │
│    └───────────────┴───────────────┴──────────┴────────┘        │
│                              │                                   │
│                    summary_metrics.json                          │
│                    per_population_metrics.csv                    │
└─────────────────────────────────────────────────────────────────┘
```

### 1. Fidelity (충실도)

생성된 데이터가 원본의 통계적 특성을 얼마나 잘 보존하는지 측정한다.

#### 1.1 AF Correlation (Allele Frequency 상관)

```
정의:
  유전자 g에 대해 real/syn의 PCA 채널별 평균값을 계산 후
  전체 유전자에 대한 Pearson 상관계수를 구한다.

  AF_real(g) = mean(real[:, :, g])   각 유전자의 평균 PCA 값
  AF_syn(g)  = mean(syn[:, :, g])

  r = Pearson(AF_real, AF_syn)

목표: r >= 0.95
해석: 인구군별 유전자 수준의 변이 빈도가 보존되었는지 판단
```

#### 1.2 Wasserstein Distance (채널별 분포 거리)

```
정의:
  PCA 성분(채널) k에 대해 real과 syn의 1D 분포 간 Wasserstein-1 거리

  W_k = W₁(real[:, k, :].flatten(), syn[:, k, :].flatten())

  최종: mean(W_k) for k = 1..K

목표: 최소화 (test set의 W_k 이하)
해석: 각 PCA 성분의 전체적 분포 형태가 보존되었는지 판단
```

### 2. Structure (구조 보존)

실제 데이터의 인구군 간 유전적 구조(클러스터링, 분화)가 보존되었는지 측정한다.

#### 2.1 PCA Overlap (Silhouette Score)

```
방법:
  1. real과 syn을 합친 후 2D PCA 수행
  2. label = {0: real, 1: synthetic}으로 Silhouette score 계산
  3. score가 0에 가까울수록 real/syn이 구분 불가 = 이상적

  S = silhouette_score(PCA_2D(concat(real, syn)), labels=[0]*n + [1]*m)

목표: S → 0 (완전히 섞임)
해석: 합성 데이터가 실제 데이터의 PCA 공간 내 분포를 재현하는지 판단
```

#### 2.2 Sliced Wasserstein Distance

```
방법:
  1. real과 syn을 2D PCA로 투영
  2. L개의 랜덤 1D 방향으로 사영(projection)
  3. 각 방향에서 1D Wasserstein 거리 계산 → 평균

  SWD = (1/L) × Σ_l W₁(proj_l(real), proj_l(syn))

목표: test set 대비 2배 이내
해석: 고차원 분포 거리를 효율적으로 근사
```

### 3. Utility (유용성)

합성 데이터가 실제 데이터를 대체하여 다운스트림 분석에 사용될 수 있는지 측정한다.

#### 3.1 Recovery Rate

```
방법:
  1. 실제 데이터로 분류기(LogisticRegression) 학습 → 실제 테스트 정확도 = Acc_real
  2. 합성 데이터로 동일 분류기 학습 → 실제 테스트 정확도 = Acc_syn
  3. Recovery Rate = Acc_syn / Acc_real

  RR = Accuracy(clf.fit(X_syn, y_syn).predict(X_test))
     / Accuracy(clf.fit(X_real, y_real).predict(X_test))

목표: RR >= 0.93
해석: 합성 데이터만으로 학습해도 실제 데이터의 93% 이상 성능 달성
```

#### 3.2 Augmentation Effect (증강 효과)

```
방법:
  혼합 비율별로 real + syn 혼합 학습 → 실제 테스트 정확도 측정

  실험 설정:
    5% real + 95% syn   → Acc_5
    50% real + 50% syn  → Acc_50
    100% real (baseline) → Acc_base

  증강 효과 = Acc_mix / Acc_base

목표: 증강 시 Acc_base 대비 개선 (특히 소수 인구군)
해석: 합성 데이터가 학습 데이터 부족 문제를 해결하는지 직접 검증
```

### 4. Privacy (프라이버시)

합성 데이터가 원본 개인의 유전 정보를 노출하지 않는지 측정한다.

#### 4.1 NNAA (Nearest Neighbor Adversarial Accuracy)

```
정의:
  각 데이터 포인트에 대해 "같은 출처(real/syn)의 가장 가까운 이웃"이
  "다른 출처의 가장 가까운 이웃"보다 가까운 비율

  Term₁ = (1/n) Σᵢ I(d(xᵢ, NN_real(xᵢ)) < d(xᵢ, NN_syn(xᵢ)))   [real → real이 더 가까움]
  Term₂ = (1/m) Σⱼ I(d(yⱼ, NN_syn(yⱼ)) < d(yⱼ, NN_real(yⱼ)))   [syn → syn이 더 가까움]
  NNAA = 0.5 × (Term₁ + Term₂)

목표: NNAA ≈ 0.5
해석:
  NNAA ≈ 0.5 → real/syn 구분 불가 = 개인 식별 불가 = 프라이버시 보호
  NNAA > 0.5 → real끼리/syn끼리 더 가까움 = 다른 분포 = 프라이버시 보호 but 유용성↓
  NNAA < 0.5 → syn이 real에 너무 가까움 = 메모리제이션 위험
```

#### 4.2 DUPI (Data Utility and Privacy Index)

> Jeong, D., Kim, J. H. T., & Im, J. (2023). *"Synthetic Data -- What, Why and How?"*
> IEEE Transactions on Information Forensics and Security.
> DOI: [10.1109/TIFS.2022.3228753](https://doi.org/10.1109/TIFS.2022.3228753)

NNAA와 달리 **이론적 벤치마크(DUPI₀)**가 존재하여 정량적 판정이 가능한 지표이다.

```
표기:
  X_n = {x₁, ..., xₙ}  원본 데이터 (n개)
  Y_m = {y₁, ..., yₘ}  합성 데이터 (m개)
  d^{<k>}_S(c)          집합 S에서 점 c까지의 k번째 최근접 이웃 거리

정의:
                    1   n
  DUPI^{<k>}  =   ─── Σ  𝟙( d^{<k>}_{Y_m}(xᵢ)  ≤  d^{<k>}_{X_n\i}(xᵢ) )
                    n  i=1

  "각 원본 점 xᵢ에 대해:
     합성 데이터의 k-NN이 원본 데이터의 k-NN보다 더 가까운 비율"
```

```
이론 벤치마크 (Theorem 4):
  X_n과 Y_m이 같은 분포에서 독립 추출된 경우:

    k=1:   DUPI₀ = m / (n + m - 1)
    n=m:   DUPI₀ = n / (2n - 1)  ≈  0.5

    일반 k: DUPI₀ = Σ_{s=k}^{2k-1} C(s-1,k-1)·C(n-1+m-s,m-k) / C(n-1+m,m)
```

```
해석:
  ┌──────────┬──────────────────────────────┬────────────────────────────┐
  │ DUPI 값  │ 의미                          │ 진단                        │
  ├──────────┼──────────────────────────────┼────────────────────────────┤
  │ ≈ 1      │ 합성이 원본에 지나치게 가까움     │ Utility↑ Privacy↓ (leakage) │
  │ ≈ DUPI₀  │ 동일 분포의 독립 샘플처럼 동작    │ ** 최적 균형 **              │
  │   (≈0.5) │                              │                            │
  │ ≈ 0      │ 합성이 원본에서 너무 멀어짐      │ Utility↓ Privacy↑ (손실)    │
  └──────────┴──────────────────────────────┴────────────────────────────┘
```

```
UI/PI 분해 (시각화용):

  리스케일링:
    DUPI ≤ DUPI₀:  g = DUPI / (2·DUPI₀)
    DUPI > DUPI₀:  g = (DUPI - DUPI₀) / (2·(1 - DUPI₀)) + 0.5

  Utility Index:  UI = arctan(τ·g) / arctan(τ)          τ=5
  Privacy Index:  PI = arctan(τ - τ·g) / arctan(τ)

  최적 조건 (Theorem 5):
    UI·PI ≤ [arctan(τ/2) / arctan(τ)]²
    등호 ⟺ DUPI = DUPI₀

    τ=5 일 때 최적 (UI₀, PI₀) ≈ (0.867, 0.867)
```

```
NNAA vs DUPI:

  공통점: 이상적 값 ≈ 0.5
  차이점:
    NNAA → 양방향 대칭 (real↔syn)         | 이론 벤치마크 없음
    DUPI → 단방향 (real→syn)              | 정확한 DUPI₀ 존재

  → NNAA: 기존 유전형 생성 논문과의 비교용
  → DUPI: 정량적 판정 및 UI/PI 시각화용
  → 본 프로젝트에서 병행 사용
```

#### 4.3 Membership Inference AUC

```
방법:
  "이 샘플이 학습에 사용되었는가?"를 추론하는 공격 모델

  1. 각 테스트 포인트의 합성 데이터까지 최소 거리 계산
  2. 학습 데이터(member)와 홀드아웃 데이터(non-member)를 구분하는
     이진 분류기 학습
  3. AUC 측정

목표: AUC ≈ 0.5 (랜덤 추측 수준)
해석: AUC > 0.6 → 학습 데이터 멤버십 추론 가능 = 프라이버시 위험
```

### 5. Robustness (강건성) -- 핵심 신규 지표

FiLM 기반 계층적 임베딩의 핵심 가설을 검증하는 지표이다.

#### 5.1 Population Size vs Quality Correlation

```
가설:
  "FiLM 없이는 인구군 크기(n)와 생성 품질 사이에 강한 양의 상관이 존재한다.
   FiLM + 계층적 임베딩 적용 후 이 상관이 약화되면 → 크기 독립적 품질 달성."

방법:
  1. 인구군별 AF 상관(quality)을 개별 계산
  2. pop_sizes = [61, 64, ..., 113]
  3. qualities = [r_ASW, r_MXL, ..., r_YRI]
  4. r = Pearson(pop_sizes, qualities)

  Robustness = |r|

목표: |r| → 0
해석:
  |r| ≈ 0 → 인구군 크기와 무관한 품질 = FiLM이 소수 인구군 보호에 성공
  |r| > 0.5 → 여전히 크기 의존적 = 추가 개선 필요
```

#### 5.2 Per-Population DUPI Gap

```
방법:
  1. 인구군별로 DUPI를 개별 계산
  2. 각 인구군의 |DUPI - DUPI₀| (gap) 측정
  3. 인구군 크기와 gap 간 상관 계산

목표: gap-size correlation → 0
해석: 모든 인구군에서 균일한 프라이버시-유용성 균형 = FiLM 강건성 확인
```

### 지표 요약 테이블

| Category | Metric | Formula | Target | 비고 |
|----------|--------|---------|--------|------|
| Fidelity | AF Correlation | Pearson(AF_real, AF_syn) | r >= 0.95 | 전체 |
| Fidelity | Wasserstein/ch | mean(W₁ per channel) | minimize | PCA 성분별 |
| Structure | PCA Overlap | Silhouette(real+syn) | S → 0 | 2D PCA |
| Structure | Sliced WD | mean(W₁ per projection) | <= 2x test | L=100 projections |
| Utility | Recovery Rate | Acc_syn / Acc_real | >= 0.93 | LogisticRegression |
| Utility | Augmentation | Acc_mix / Acc_base | > 1.0 | 5%, 50% 비율 |
| Privacy | NNAA | 0.5×(T₁+T₂) | ≈ 0.5 | 양방향, 비교용 |
| Privacy | **DUPI** | (1/n)Σ𝟙(d_syn <= d_real) | **≈ DUPI₀** | 이론 벤치마크, 판정용 |
| Privacy | MIA AUC | Binary clf AUC | ≈ 0.5 | 멤버십 추론 |
| **Robustness** | **Size-Quality \|r\|** | \|Pearson(size, quality)\| | **→ 0** | **핵심 contribution** |
| Robustness | Per-pop DUPI gap | corr(size, \|DUPI-DUPI₀\|) | → 0 | FiLM 강건성 |

---

## Data

```
data/
├── ALL.autosomes.phase3.genotypes.vcf.gz          (13.9 GB, 1KG Phase 3, chr1-22)
├── ALL.autosomes.phase3.genotypes.vcf.gz.tbi      (tabix index)
└── integrated_call_samples_v3.20130502.ALL.panel   (sample→pop→superpop mapping)
```

### 전처리 파이프라인 흐름

```
OOM-safe 2-pass approach:

Pass 1: chr1,11,22 파싱 → PCA grid search → 최적 K 결정 → 메모리 해제
Pass 2: 22 chr 순차 스트리밍 → PCA 변환 → variant 즉시 해제
Peak RAM ≈ 1 chromosome (~3-5GB for chr1) + 누적 PCA features (~2GB)

전처리 순서:
  1. VCF 파싱 (MAF >= 0.01 필터링)
  2. Gene annotation (RefGene 기반 유전자 매핑)
  3. PCA grid search (K 후보: [4,6,8,10,12,16], Marginal Gain Elbow)
  4. 전체 VCF → Gene PCA 스트리밍 변환
  5. 계층적 레이블 생성 (pop → superpop)
  6. 토큰화 + alignment 패딩 (128 단위)
  7. 정규화 (패딩 후 수행, stats shape = (gene_size, K))
  8. zero_mask 생성
  9. Stratified split (train/test) + 저장
```

### 전처리 산출물

| 파일 | Shape | 설명 |
|------|-------|------|
| `gene_pca_features.pkl` | DataFrame (2504, N_features) | 원본 PCA 피처 |
| `train_data.pkl` | (x: N×K×gene_size, y: N) | 패딩 → 정규화된 학습 데이터 |
| `test_data.pkl` | (x: N×K×gene_size, y: N) | 패딩 → 정규화된 테스트 데이터 |
| `normalization_stats.pkl` | {mean, std}: (gene_size, K) fp32 | 역정규화용 통계량 |
| `label_hierarchy.pkl` | dict (8 fields) | pop/superpop 매핑 전체 |
| `zero_mask.pt` | (gene_size, K) bool | 항상 0인 위치 마스크 |
| `split_manifest.json` | dict | 재현성 보장용 split 기록 |

---

## Quick Start

```bash
# 환경 설치
uv sync

# Phase 0.5: VCF 병합 (22 염색체 병렬)
python src/preprocessing/merge_data.py --format vcf
python src/preprocessing/merge_data.py --format pkl --maf 0.01

# Phase 1: 전처리 (VCF → Gene PCA → 토큰화)
python src/preprocessing/run_pipeline.py

# Phase 2: 모델 shape 검증
python -c "
from src.models import HybridCNNDiTFiLM
from src.utils.config import load_config
import torch

config = load_config('configs/default.yaml')
model = HybridCNNDiTFiLM(config)
x = torch.randn(2, config['data']['num_channels'], config['data']['gene_size'])
t = torch.randint(0, 500, (2,))
y = torch.randint(0, 26, (2,))
out = model(x, t, y)
print(f'Input: {x.shape} → Output: {out.shape}')
print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')
"

# Phase 3: 학습 (DDP 2-GPU, bf16)
torchrun --nproc_per_node=2 src/training/trainer.py --config configs/default.yaml

# Phase 3 (single GPU debug)
python src/training/trainer.py --config configs/default.yaml --single_gpu

# Phase 4: 추론 (인구군별 조건부 생성)
python src/inference/generator.py \
    --config configs/default.yaml \
    --model_path outputs/default/best_model.pth \
    --output_dir outputs/default/synthetic_samples

# Phase 5: 평가 (11개 지표 병렬)
python src/evaluation/run_evaluation.py \
    --config configs/default.yaml \
    --syn_dir outputs/default/synthetic_samples

# Hyperparameter sweep (wandb)
# configs/sweep.yaml을 작성한 후 실행
# wandb sweep configs/sweep.yaml --project HybridGenoDiT
# wandb agent <sweep_id>
```

---

## Project Structure

```
gene-synthesis-project/
├── configs/
│   └── default.yaml                # Canonical config (source of truth)
│
├── src/
│   ├── preprocessing/
│   │   ├── config.py               # 전처리 상수 (경로, PCA 후보, MAF 등)
│   │   ├── vcf_parser.py           # VCF 파싱 (Rust 바인딩 지원)
│   │   ├── gene_annotation.py      # RefGene 유전자 어노테이션
│   │   ├── pca.py                  # Gene PCA (grid search, Marginal Gain Elbow)
│   │   ├── tokenizer.py            # 토큰화 + alignment 패딩
│   │   ├── labels.py               # 계층적 레이블, split, 정규화, 저장
│   │   ├── merge_data.py           # VCF 병합 (22 chr 병렬)
│   │   └── run_pipeline.py         # 전처리 오케스트레이터 (OOM-safe 2-pass)
│   │
│   ├── models/
│   │   ├── hybrid_geno_dit.py      # HybridCNNDiTFiLM (전체 모델)
│   │   ├── diffusion.py            # GaussianDiffusion (cosine, DDIM, CFG)
│   │   └── modules/
│   │       ├── base.py             # timestep_embedding, zero_module
│   │       ├── conditioning.py     # HierarchicalPopEmb + UnifiedFiLMGen
│   │       ├── cnn.py              # FiLMConvBlock, CNNEncoder, CNNDecoder
│   │       └── dit.py              # PatchEmbed1D, DiTBlock, DiTCore
│   │
│   ├── training/
│   │   ├── trainer.py              # DDP + bf16 학습 루프
│   │   └── losses.py               # masked_mse, MMD, Min-SNR
│   │
│   ├── inference/
│   │   └── generator.py            # EMA 로드, DDIM 생성, 역정규화 (stats 패딩 처리)
│   │
│   ├── evaluation/
│   │   ├── run_evaluation.py       # 병렬 평가 실행기 (역정규화 포함)
│   │   └── metrics.py              # 11개 지표 구현
│   │
│   ├── data/
│   │   ├── dataset.py              # GenotypeDataset (pkl → tensor)
│   │   ├── sampler.py              # PopulationBalancedSampler (sqrt 비례)
│   │   └── dataloader.py           # DataLoader 팩토리
│   │
│   └── utils/
│       ├── config.py               # YAML 로드, CLI override, 검증
│       ├── ddp.py                  # DDP setup/cleanup
│       ├── ema.py                  # EMA (decay 0.9999)
│       ├── logger.py               # wandb 래퍼 (rank 0 only)
│       └── checkpoint.py           # .pth 저장/로드, top-k 관리
│
├── data/                            # (git 추적 안 함)
│   ├── ALL.autosomes.phase3.genotypes.vcf.gz
│   └── processed/                   # 전처리 산출물
│
├── outputs/                         # (git 추적 안 함)
│   └── default/
│       ├── best_model.pth
│       ├── synthetic_samples/
│       └── evaluation/
│
└── docs/                            # 상세 기획서 (01~10)
```

---

## Hardware Requirements

| Resource | Spec | Usage |
|----------|------|-------|
| GPU × 2 | NVIDIA RTX A6000 (48GB) | DDP 학습 (bf16) |
| RAM | 64GB+ 권장 | 전체 데이터셋 in-memory |
| Storage | 50GB+ | VCF(14GB) + 산출물 + 체크포인트 |

**VRAM 사용량** (baseline config):
```
Model parameters:   ~6.8M × 2B (bf16)  ≈  14 MB
Optimizer states:   ~6.8M × 8B (Adam)  ≈  54 MB
Activations:        batch=32            ≈  ~2 GB
──────────────────────────────────────────────────
Total per GPU:      ~2.1 GB  (49GB 중 4% 사용)
```

---

## Key Design Decisions

| 결정 | 근거 |
|------|------|
| bf16 (not fp16) | Ampere CC 8.6 네이티브 지원, exponent 8bit → GradScaler 불필요 |
| DDP (not FSDP) | 6.8M params는 단일 GPU에 충분, DDP가 단순하고 효율적 |
| cosine schedule | Diffusion에서 linear 대비 학습 안정성 우수 |
| DDIM 50-step | 500-step DDPM 대비 10x 가속, 품질 유지 |
| AdaLN-Zero | α=0 초기화 → DiT가 identity로 시작 → 안정적 학습 |
| Marginal Gain Elbow (K 선택) | threshold=0.03, decay_ratio=0.5로 데이터 적응적 |
| 패딩 → 정규화 순서 | 패딩 후 정규화하여 stats shape = (gene_size, K) 보장 |
| 역정규화 padding 처리 | stats 크기 < gene_size일 때 자동 패딩 (mean=0, std=1) |
| sqrt 비례 오버샘플링 | 균등(1:1)과 비례 사이의 균형 |
| DUPI + NNAA 병행 | DUPI: 정량적 판정, NNAA: 기존 논문 비교 |

---

## References

- Nichol, A. Q., & Dhariwal, P. (2021). Improved Denoising Diffusion Probabilistic Models. *ICML*.
- Peebles, W., & Xie, S. (2023). Scalable Diffusion Models with Transformers (DiT). *ICCV*.
- Perez, E., et al. (2018). FiLM: Visual Reasoning with a General Conditioning Layer. *AAAI*.
- Jeong, D., Kim, J. H. T., & Im, J. (2023). Synthetic Data -- What, Why and How? *IEEE TIFS*.
- Hang, T., et al. (2023). Efficient Diffusion Training via Min-SNR Weighting Strategy. *ICCV*.
- Song, J., Meng, C., & Ermon, S. (2020). Denoising Diffusion Implicit Models (DDIM). *ICLR*.
- The 1000 Genomes Project Consortium (2015). A global reference for human genetic variation. *Nature*.

---

## License

This project is for academic research purposes.
