# Hybrid CNN-DiT + FiLM: 인구군별 강건한 유전형 합성 아키텍처

**작성일**: 2026-03-27
**목적**: CNN(근거리 패턴) + DiT(장거리 상호작용) + FiLM(인구군별 변조)를 결합한 Diffusion 모델 설계

---

## 1. 설계 철학

```
"하나의 백본이 유전체의 보편적 구조를 학습하고,
 FiLM이 인구군별 '방언'을 주입한다."
```

**핵심 비유**: 인간 유전체는 99.9% 동일하다. CNN-DiT 백본이 이 공통 구조를 학습하고,
FiLM의 gamma/beta가 인구군별 0.1% 차이(대립유전자 빈도, LD 패턴, 하플로타입 다양성)를 변조한다.

**소수 인구군 강건성의 핵심**: ASW(61명), MXL(64명) 같은 소수 인구군도
같은 슈퍼인구군(AFR, AMR)의 정보를 계층 임베딩으로 공유받기 때문에
독립 학습 대비 훨씬 안정적이다.

---

## 2. 전체 아키텍처 다이어그램

```
 Input: (batch, 8, 26624)         Condition: pop_label (0-25)
         │                                    │
         ▼                                    ▼
 ┌───────────────────┐            ┌──────────────────────────┐
 │   CNN Stem        │            │  Hierarchical Population │
 │   Encoder         │            │  Embedding               │
 │                   │            │                          │
 │  Conv1d layers    │            │  pop_emb(26, d)          │
 │  + GroupNorm      │            │  superpop_emb(5, d)      │
 │  + FiLM_cnn       │            │  hierarchy = MLP(concat) │
 │  + Downsample     │            └──────────┬───────────────┘
 │                   │                       │
 │  (8, 26624)       │            ┌──────────┴───────────────┐
 │  → (C, 26624)     │            │  Timestep Embedding      │
 │  → (C, 13312)     │            │  sinusoidal(t) → MLP     │
 │  → (2C, 6656)     │            └──────────┬───────────────┘
 │  → (4C, 3328)     │                       │
 └────────┬──────────┘                       ▼
          │                       ┌──────────────────────────┐
          │  skip connections     │  FiLM Parameter Generator │
          │  stored at each       │                          │
          │  resolution           │  cond = concat(hierarchy,│
          │                       │         t_emb)           │
          ▼                       │  → MLP                   │
 ┌───────────────────┐            │  → per-block (γ_l, β_l)  │
 │   Patchify +      │            │  → per-CNN   (γ_c, β_c)  │
 │   Token Project   │            └──────────┬───────────────┘
 │                   │                       │
 │  (4C, 3328)       │                       │
 │  → patch size P   │                       │
 │  → (3328/P, d)    │                       │
 │  = (208, d)       │         ┌─────────────┤
 │    if P=16        │         │             │
 │                   │         ▼             ▼
 │  + pos_embedding  │    ┌─────────┐  ┌─────────┐
 └────────┬──────────┘    │CNN FiLM │  │DiT FiLM │
          │               │params   │  │params   │
          ▼               └────┬────┘  └────┬────┘
 ┌────────────────────────────┐│            │
 │   DiT Blocks (N=4~6)      ││            │
 │                            ││            │
 │   ┌──────────────────────┐ ││            │
 │   │ Block l:             │◄┘            │
 │   │  LayerNorm           │◄─────────────┘
 │   │  → AdaLN: γ_l·h+β_l │  (= FiLM)
 │   │  → Multi-Head Attn   │
 │   │  → AdaLN: γ_l·h+β_l │
 │   │  → FFN (MLP)         │
 │   │  → AdaLN: γ_l·h+β_l │
 │   └──────────────────────┘
 │   × N blocks              │
 └────────┬──────────────────┘
          │
          ▼
 ┌───────────────────┐
 │   Un-Patchify     │
 │   (208, d) → (4C, 3328)
 └────────┬──────────┘
          │
          ▼
 ┌───────────────────┐
 │   CNN Decoder      │
 │   (skip 연결 포함)  │
 │                    │
 │  + FiLM_cnn        │
 │  (4C, 3328)        │
 │  → (2C, 6656)      │
 │  → (C, 13312)      │
 │  → (C, 26624)      │
 │  → (8, 26624)      │
 └────────┬───────────┘
          │
          ▼
 Output: predicted noise ε (batch, 8, 26624)
```

---

## 3. 각 모듈 상세 설계

### 3.1 Hierarchical Population Embedding (계층적 인구군 임베딩)

**왜 계층적이어야 하는가?**

```
슈퍼인구군 (5)          인구군 (26)               샘플 수
─────────────────────────────────────────────────────
AFR (661)        ──→    YRI(108), LWK(99), GWD(113), MSL(85),
                        ESN(99), ACB(96), ASW(61)
EUR (503)        ──→    CEU(99), TSI(107), FIN(99), GBR(91), IBS(107)
EAS (504)        ──→    CHB(103), JPT(104), CHS(105), CDX(93), KHV(99)
SAS (489)        ──→    GIH(103), PJL(96), BEB(86), STU(102), ITU(102)
AMR (347)        ──→    MXL(64), PUR(104), CLM(94), PEL(85)
```

ASW(61명)가 단독으로 학습하면 매우 불안정하지만, AFR 슈퍼인구군의 661명 정보를 공유받으면 안정화된다.

```python
class HierarchicalPopulationEmbedding(nn.Module):
    def __init__(self, n_pops=26, n_superpops=5, d_model=256):
        super().__init__()
        # 인구군 → 슈퍼인구군 매핑 테이블
        # ASW→AFR, MXL→AMR, GBR→EUR, ...
        self.register_buffer('pop_to_superpop', torch.tensor([
            0,0,0,0,0,0,0,  # AFR 7개
            1,1,1,1,1,       # EUR 5개
            2,2,2,2,2,       # EAS 5개
            3,3,3,3,3,       # SAS 5개
            4,4,4,4          # AMR 4개
        ]))  # 실제 매핑은 label_map.pkl에 따라 조정

        self.pop_emb = nn.Embedding(n_pops, d_model)
        self.superpop_emb = nn.Embedding(n_superpops, d_model)

        # 계층 융합 MLP
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model)
        )

    def forward(self, pop_label):
        """
        pop_label: (batch,) 인구군 인덱스 0-25
        returns: (batch, d_model) 계층적 인구군 임베딩
        """
        p_emb = self.pop_emb(pop_label)                          # (B, d)
        sp_label = self.pop_to_superpop[pop_label]
        sp_emb = self.superpop_emb(sp_label)                     # (B, d)
        return self.fusion(torch.cat([p_emb, sp_emb], dim=-1))   # (B, d)
```

**강건성 메커니즘**:
- `superpop_emb`는 수백 명 단위로 학습 → 안정적 기반
- `pop_emb`는 각 인구군의 미세 차이를 잡는 **잔차(residual)** 역할
- 소수 인구군(ASW 61명)도 슈퍼인구군 정보에 기대어 합리적 생성 가능

---

### 3.2 FiLM Parameter Generator (통합 변조 파라미터 생성기)

CNN 블록과 DiT 블록 **모두**에 FiLM 파라미터를 공급하는 통합 생성기.

```python
class UnifiedFiLMGenerator(nn.Module):
    """
    인구군 임베딩 + 타임스텝 임베딩 → CNN/DiT 모든 블록의 (γ, β) 생성

    DiT에서의 AdaLN-Zero와 동일한 원리이나,
    CNN 인코더/디코더까지 확장한 것이 차별점.
    """
    def __init__(self, d_model=256, d_time=256,
                 n_cnn_blocks=4, cnn_channels=[64,64,128,256],
                 n_dit_blocks=4, d_dit=256):
        super().__init__()

        # 타임스텝 임베딩
        self.time_mlp = nn.Sequential(
            nn.Linear(d_time, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model)
        )

        # 조건 융합
        self.cond_mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model),  # pop_emb + time_emb
            nn.SiLU(),
            nn.Linear(d_model, d_model)
        )

        # CNN 블록별 FiLM 파라미터 (인코더 + 디코더)
        self.cnn_film_layers = nn.ModuleList()
        for ch in cnn_channels:
            self.cnn_film_layers.append(
                nn.Linear(d_model, ch * 2)  # γ와 β 동시 출력
            )

        self.cnn_dec_film_layers = nn.ModuleList()
        for ch in reversed(cnn_channels):
            self.cnn_dec_film_layers.append(
                nn.Linear(d_model, ch * 2)
            )

        # DiT 블록별 FiLM 파라미터 (AdaLN-Zero)
        # 각 블록에서 attn 전/후, FFN 전/후 → 3세트의 (γ, β)
        self.dit_film_layers = nn.ModuleList()
        for _ in range(n_dit_blocks):
            self.dit_film_layers.append(
                nn.Sequential(
                    nn.SiLU(),
                    zero_module(nn.Linear(d_model, d_dit * 6))
                    # 6 = (γ1,β1,α1) for attn + (γ2,β2,α2) for FFN
                    # α는 AdaLN-Zero의 gate 파라미터
                )
            )

    def forward(self, pop_emb, t):
        """
        pop_emb: (B, d_model) from HierarchicalPopulationEmbedding
        t: (B,) diffusion timestep
        """
        t_emb = self.time_mlp(timestep_embedding(t, self.d_time))  # (B, d)
        cond = self.cond_mlp(torch.cat([pop_emb, t_emb], dim=-1)) # (B, d)

        # CNN FiLM 파라미터
        cnn_enc_params = []
        for layer in self.cnn_film_layers:
            gb = layer(cond)                     # (B, ch*2)
            gamma, beta = gb.chunk(2, dim=-1)    # 각 (B, ch)
            cnn_enc_params.append((gamma, beta))

        cnn_dec_params = []
        for layer in self.cnn_dec_film_layers:
            gb = layer(cond)
            gamma, beta = gb.chunk(2, dim=-1)
            cnn_dec_params.append((gamma, beta))

        # DiT FiLM 파라미터 (AdaLN-Zero)
        dit_params = []
        for layer in self.dit_film_layers:
            params = layer(cond)                 # (B, d_dit*6)
            dit_params.append(params)

        return cnn_enc_params, cnn_dec_params, dit_params
```

---

### 3.3 CNN Stem Encoder with FiLM

```python
class FiLMConvBlock(nn.Module):
    """FiLM 변조가 포함된 1D Conv 블록"""
    def __init__(self, in_ch, out_ch, downsample=True):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(min(32, out_ch), out_ch)
        self.norm2 = nn.GroupNorm(min(32, out_ch), out_ch)
        self.downsample = nn.Conv1d(out_ch, out_ch, kernel_size=2, stride=2) if downsample else nn.Identity()
        self.skip_proj = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, gamma, beta):
        """
        x: (B, C_in, L)
        gamma, beta: (B, C_out) from FiLM generator
        """
        residual = self.skip_proj(x)

        h = self.conv1(x)
        h = self.norm1(h)
        # ── FiLM 변조 ──
        h = gamma.unsqueeze(-1) * h + beta.unsqueeze(-1)   # (B,C,1) * (B,C,L)
        h = F.silu(h)

        h = self.conv2(h)
        h = self.norm2(h)
        h = F.silu(h)

        skip = h + residual[..., :h.shape[-1]]  # skip connection 저장
        out = self.downsample(h)
        return out, skip


class CNNStemEncoder(nn.Module):
    """
    (B, 8, 26624) → (B, 4C, 3328) + skip connections

    4단계 다운샘플링: 26624 → 13312 → 6656 → 3328
    채널: 8 → C → C → 2C → 4C
    """
    def __init__(self, in_channels=8, base_channels=64):
        super().__init__()
        C = base_channels
        self.blocks = nn.ModuleList([
            FiLMConvBlock(in_channels, C, downsample=True),    # 26624→13312
            FiLMConvBlock(C, C, downsample=True),              # 13312→6656
            FiLMConvBlock(C, 2*C, downsample=True),            # 6656→3328
            FiLMConvBlock(2*C, 4*C, downsample=False),         # 3328 유지 (DiT 입력)
        ])

    def forward(self, x, cnn_film_params):
        """
        x: (B, 8, 26624)
        cnn_film_params: [(γ_0,β_0), (γ_1,β_1), ...]
        returns: features (B, 4C, 3328), skips list
        """
        skips = []
        for block, (gamma, beta) in zip(self.blocks, cnn_film_params):
            x, skip = block(x, gamma, beta)
            skips.append(skip)
        return x, skips
```

**CNN에서 FiLM이 하는 일**:
- 같은 Conv 필터가 모든 인구군에 적용되지만
- **gamma가 특정 채널을 증폭/감쇠**, **beta가 활성화를 이동**
- 예: AFR 인구군에서는 LD가 짧으므로 → 고주파 필터 채널 증폭
- 예: EAS 인구군에서는 특정 유전자 영역의 변이 패턴이 다름 → 해당 채널 시프트

---

### 3.4 Patchify + DiT Core with AdaLN-Zero (= FiLM)

```python
class PatchEmbed1D(nn.Module):
    """CNN 출력을 DiT 토큰으로 변환"""
    def __init__(self, seq_len=3328, in_channels=256, patch_size=16, d_model=256):
        super().__init__()
        self.patch_size = patch_size
        self.n_tokens = seq_len // patch_size  # 3328/16 = 208 토큰
        self.proj = nn.Linear(in_channels * patch_size, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, self.n_tokens, d_model) * 0.02)

    def forward(self, x):
        """
        x: (B, C, L)  e.g., (B, 256, 3328)
        returns: (B, 208, d_model)
        """
        B, C, L = x.shape
        x = x.reshape(B, C, self.n_tokens, self.patch_size)  # (B, C, 208, 16)
        x = x.permute(0, 2, 1, 3)                            # (B, 208, C, 16)
        x = x.reshape(B, self.n_tokens, -1)                  # (B, 208, C*16)
        x = self.proj(x)                                      # (B, 208, d)
        x = x + self.pos_emb
        return x


class DiTBlock(nn.Module):
    """
    AdaLN-Zero DiT Block (Peebles & Xie, ICCV 2023)
    = FiLM + Zero-initialization

    핵심: LayerNorm의 scale/shift를 인구군 조건에서 예측
    """
    def __init__(self, d_model=256, n_heads=4, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, int(d_model * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(d_model * mlp_ratio), d_model),
        )

    def forward(self, x, film_params):
        """
        x: (B, N, d)  토큰 시퀀스
        film_params: (B, d*6)  → (γ1,β1,α1, γ2,β2,α2)

        AdaLN-Zero:
          h = γ · LayerNorm(x) + β     ← FiLM
          out = x + α · Attn(h)        ← Zero-gate
        """
        # 파라미터 분리
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = film_params.chunk(6, dim=-1)
        # 각 (B, d) → (B, 1, d)로 확장
        gamma1 = gamma1.unsqueeze(1)
        beta1 = beta1.unsqueeze(1)
        alpha1 = alpha1.unsqueeze(1)
        gamma2 = gamma2.unsqueeze(1)
        beta2 = beta2.unsqueeze(1)
        alpha2 = alpha2.unsqueeze(1)

        # ── Self-Attention with FiLM ──
        h = self.norm1(x)
        h = gamma1 * h + beta1              # FiLM 변조
        h, _ = self.attn(h, h, h)
        x = x + alpha1 * h                  # Zero-gated residual

        # ── FFN with FiLM ──
        h = self.norm2(x)
        h = gamma2 * h + beta2              # FiLM 변조
        h = self.mlp(h)
        x = x + alpha2 * h                  # Zero-gated residual

        return x


class DiTCore(nn.Module):
    """경량 DiT: 4~6 블록, 208 토큰"""
    def __init__(self, n_blocks=4, d_model=256, n_heads=4):
        super().__init__()
        self.blocks = nn.ModuleList([
            DiTBlock(d_model, n_heads) for _ in range(n_blocks)
        ])

    def forward(self, x, dit_film_params):
        """
        x: (B, 208, d)
        dit_film_params: list of (B, d*6) for each block
        """
        for block, params in zip(self.blocks, dit_film_params):
            x = block(x, params)
        return x
```

**DiT에서 FiLM이 하는 일 (= AdaLN-Zero)**:
- **gamma**: Attention/FFN 출력의 각 차원을 인구군별로 스케일링
  - 예: EUR에서는 장거리 유전자 상관이 약함 → 특정 attention 차원 축소
  - 예: AFR에서는 유전적 다양성이 높음 → 더 넓은 차원 범위 활성화
- **beta**: 기저 활성화 수준을 인구군별로 시프트
- **alpha (Zero-gate)**: 초기에 0으로 시작 → 학습 안정성 보장
  - 학습 초기: alpha ≈ 0 → DiT가 항등 함수처럼 동작 → CNN 출력을 그대로 전달
  - 학습 진행: alpha가 점진적으로 증가 → DiT가 장거리 패턴을 보정

---

### 3.5 CNN Decoder with FiLM + Skip Connections

```python
class FiLMDeconvBlock(nn.Module):
    """FiLM 변조 + skip connection이 포함된 1D 디코더 블록"""
    def __init__(self, in_ch, out_ch, upsample=True):
        super().__init__()
        self.upsample = nn.ConvTranspose1d(in_ch, in_ch, kernel_size=2, stride=2) if upsample else nn.Identity()
        self.conv1 = nn.Conv1d(in_ch * 2, out_ch, kernel_size=3, padding=1)  # *2 for skip concat
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(min(32, out_ch), out_ch)
        self.norm2 = nn.GroupNorm(min(32, out_ch), out_ch)

    def forward(self, x, skip, gamma, beta):
        """
        x: (B, C_in, L)
        skip: (B, C_in, L*2) from encoder
        gamma, beta: (B, C_out)
        """
        x = self.upsample(x)
        x = torch.cat([x, skip[..., :x.shape[-1]]], dim=1)  # skip concat

        h = self.conv1(x)
        h = self.norm1(h)
        h = gamma.unsqueeze(-1) * h + beta.unsqueeze(-1)    # FiLM
        h = F.silu(h)
        h = self.conv2(h)
        h = self.norm2(h)
        h = F.silu(h)
        return h


class CNNDecoder(nn.Module):
    """
    (B, 4C, 3328) + skips → (B, 8, 26624)
    """
    def __init__(self, out_channels=8, base_channels=64):
        C = base_channels
        super().__init__()
        self.blocks = nn.ModuleList([
            FiLMDeconvBlock(4*C, 2*C, upsample=False),    # 3328
            FiLMDeconvBlock(2*C, C, upsample=True),        # 3328→6656
            FiLMDeconvBlock(C, C, upsample=True),          # 6656→13312
            FiLMDeconvBlock(C, C, upsample=True),          # 13312→26624
        ])
        self.final = nn.Conv1d(C, out_channels, kernel_size=1)

    def forward(self, x, skips, cnn_dec_film_params):
        for block, skip, (gamma, beta) in zip(
            self.blocks, reversed(skips), cnn_dec_film_params
        ):
            x = block(x, skip, gamma, beta)
        return self.final(x)
```

---

### 3.6 전체 모델: HybridCNNDiTFiLM

```python
class HybridCNNDiTFiLM(nn.Module):
    """
    CNN-DiT Hybrid with Unified FiLM Conditioning

    CNN  → 근거리 패턴 (LD, 하플로타입 청크)     [인구군별 FiLM]
    DiT  → 장거리 유전자 간 상호작용             [인구군별 AdaLN-Zero]
    합쳐서 → 인구군별 강건한 전체 유전형 생성
    """
    def __init__(self,
                 in_channels=8,
                 base_channels=64,    # CNN 기본 채널 (C)
                 d_model=256,         # DiT 히든 차원
                 n_dit_blocks=4,      # DiT 블록 수
                 n_heads=4,           # DiT 어텐션 헤드
                 patch_size=16,       # Patchify 크기
                 n_pops=26,
                 n_superpops=5,
                 max_timesteps=500):
        super().__init__()

        # 1. 조건부 임베딩
        self.pop_embedding = HierarchicalPopulationEmbedding(n_pops, n_superpops, d_model)

        # 2. 통합 FiLM 생성기
        cnn_channels = [base_channels, base_channels, 2*base_channels, 4*base_channels]
        self.film_gen = UnifiedFiLMGenerator(
            d_model=d_model,
            d_time=d_model,
            n_cnn_blocks=4,
            cnn_channels=cnn_channels,
            n_dit_blocks=n_dit_blocks,
            d_dit=d_model
        )

        # 3. CNN Encoder
        self.encoder = CNNStemEncoder(in_channels, base_channels)

        # 4. Patchify + DiT
        dit_input_len = 26624 // 8   # 3단계 다운샘플 후: 3328
        self.patchify = PatchEmbed1D(dit_input_len, 4*base_channels, patch_size, d_model)
        self.dit = DiTCore(n_dit_blocks, d_model, n_heads)
        self.unpatchify = nn.Linear(d_model, 4*base_channels * patch_size)

        # 5. CNN Decoder
        self.decoder = CNNDecoder(in_channels, base_channels)

        # 6. CNN-DiT 밸런싱 (GeneDiffusion의 learnable_weight_time과 유사)
        self.balance = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x, t, y=None):
        """
        x: (B, 8, 26624) 노이즈가 추가된 유전형
        t: (B,) 디퓨전 타임스텝
        y: (B,) 인구군 레이블 (0-25)
        returns: (B, 8, 26624) 예측된 노이즈
        """
        B = x.shape[0]

        # ── 조건부 임베딩 ──
        pop_emb = self.pop_embedding(y)                     # (B, d)
        cnn_enc_params, cnn_dec_params, dit_params = \
            self.film_gen(pop_emb, t)

        # ── CNN Encoder (근거리 패턴 + FiLM) ──
        features, skips = self.encoder(x, cnn_enc_params)   # (B, 4C, 3328), skips

        # ── DiT (장거리 상호작용 + AdaLN-Zero) ──
        tokens = self.patchify(features)                    # (B, 208, d)
        tokens = self.dit(tokens, dit_params)               # (B, 208, d)

        # ── Un-patchify ──
        patches = self.unpatchify(tokens)                   # (B, 208, 4C*P)
        P = self.patchify.patch_size
        C4 = features.shape[1]
        dit_out = patches.reshape(B, -1, C4, P)            # (B, 208, 4C, P)
        dit_out = dit_out.permute(0, 2, 1, 3)              # (B, 4C, 208, P)
        dit_out = dit_out.reshape(B, C4, -1)               # (B, 4C, 3328)

        # ── CNN-DiT 밸런싱 ──
        w = self.balance(pop_emb)                           # (B, 1)
        merged = (1 - w.unsqueeze(-1)) * features + w.unsqueeze(-1) * dit_out

        # ── CNN Decoder (복원 + FiLM) ──
        output = self.decoder(merged, skips, cnn_dec_params)  # (B, 8, 26624)

        return output
```

---

## 4. FiLM이 인구군별 강건성을 만드는 메커니즘

### 4.1 정보 흐름 분석

```
                    인구군 레이블 (y=ASW)
                           │
                     ┌─────┴─────┐
                     │           │
              pop_emb(ASW)  superpop_emb(AFR)
                     │           │
                     └─────┬─────┘
                      fusion MLP
                           │
                    hierarchy_emb ──────────────────────────────────────┐
                           │                                           │
                    + timestep_emb                                     │
                           │                                           │
                    ┌──────┴──────┐                                    │
                    │ CNN FiLM    │                                    │
                    │ (γ_cnn,β_cnn)                                   │
                    │             │                                    │
                    │ 근거리 패턴을│                         ┌─────────┴───┐
                    │ 인구군별로  │                         │ DiT AdaLN   │
                    │ 조절       │                         │ (γ_dit,β_dit)│
                    └─────────────┘                        │             │
                                                           │ 장거리 패턴을│
                                                           │ 인구군별로  │
                                                           │ 조절       │
                                                           └─────────────┘
```

### 4.2 인구군별 "무엇이" 달라지는가

| 유전학적 특성 | CNN FiLM이 담당 | DiT FiLM이 담당 |
|-------------|----------------|----------------|
| **LD 패턴** | AFR: 짧은 LD → 고주파 필터 증폭 (gamma↑) | - |
| **대립유전자 빈도** | 각 인구군별 MAF 분포 차이 반영 (beta shift) | - |
| **유전자 간 상관** | - | 염색체 간 이주/혼합 패턴 (gamma scaling) |
| **인구 특이적 선택** | 유전자 내 선택 신호 (예: LCT in EUR) | 다중 유전자 적응 패턴 |
| **유전적 표류** | 소규모 인구군의 높은 변이 (beta shift) | 전체 유전자 다양성 수준 조절 |

### 4.3 소수 인구군 강건성 실험 설계

```
[핵심 실험] 인구군 크기별 생성 품질 분석

      인구군          샘플수     FiLM 없음(baseline)    FiLM 적용(제안)
      ─────────────────────────────────────────────────────
      GWD (AFR)       113       ★★★★☆                 ★★★★★
      GBR (EUR)        91       ★★★☆☆                 ★★★★☆
      MXL (AMR)        64       ★★☆☆☆                 ★★★★☆  ← 핵심 개선 포인트
      ASW (AFR)        61       ★★☆☆☆                 ★★★★☆  ← 핵심 개선 포인트

FiLM의 가설: 소수 인구군일수록 계층적 임베딩의 정보 공유 효과가 크다.
검증 방법: 인구군 크기와 생성 품질(AF 상관, LD 상관) 간의 관계가
           FiLM 적용 후 약화(= 크기 의존성 감소)되는지 측정
```

---

## 5. 파라미터 수 추정 및 실현 가능성

### 5.1 파라미터 수 비교

| 모듈 | 파라미터 수 (C=64, d=256) |
|------|------------------------|
| Hierarchical Pop Embedding | 26×256 + 5×256 + MLP ≈ **0.07M** |
| Unified FiLM Generator | MLP + CNN/DiT heads ≈ **0.5M** |
| CNN Encoder (4 blocks) | Conv1d layers ≈ **1.2M** |
| Patchify + Pos Embed | Linear + buffer ≈ **0.3M** |
| DiT Core (4 blocks) | Attn + FFN ≈ **3.2M** |
| CNN Decoder (4 blocks) | Conv1d + skip ≈ **1.5M** |
| **전체** | **≈ 6.8M** |

### 5.2 현재 GeneDiffusion UnetCombined과의 비교

| | UnetCombined (현재) | Hybrid CNN-DiT-FiLM (제안) |
|--|--------------------|-----------------------------|
| 파라미터 수 | ~10-15M (추정) | ~6.8M |
| 조건부 방식 | one-hot × linear | 계층적 FiLM (AdaLN-Zero) |
| 장거리 상호작용 | MLP branch (flat) | DiT self-attention |
| 근거리 패턴 | CNN branch | CNN encoder/decoder |
| 인구군 강건성 | 없음 (동일 가중치) | FiLM 변조 (인구군별) |
| 오버피팅 위험 | 기준 | **더 낮음** (파라미터↓, 정규화↑) |

### 5.3 학습 가능성 근거

- **토큰 수 208**: ViT-Small (196 tokens)과 동일 수준 → 검증된 규모
- **DiT 4블록**: DiT-S/2의 12블록 대비 1/3 → n=2,504에 적합
- **AdaLN-Zero 초기화**: 학습 초기 DiT ≈ 항등함수 → CNN만으로 동작 → 점진적 DiT 기여 → 안정적 수렴
- **계층적 임베딩**: 소수 인구군도 슈퍼인구군 정보 공유 → 실질적으로 n=61이 아닌 n=661로 학습

---

## 6. 논문 스토리라인

### Title (안)
**"HybridGenoDiT: Population-Robust Synthetic Genotype Generation via CNN-DiT Hybrid Diffusion with Hierarchical FiLM Conditioning"**

### Abstract 구조
1. 기존 유전형 생성 모델은 인구군 조건부 생성에서 소수 인구군에 취약
2. FiLM(Feature-wise Linear Modulation)을 유전형 Diffusion에 최초 적용
3. CNN(근거리 LD/하플로타입) + DiT(장거리 유전자 상호작용) 하이브리드 아키텍처
4. 계층적 인구군 임베딩으로 소수 인구군 강건성 확보
5. 1000 Genomes Phase 3 (2,504명, 26 인구군)에서 검증

### Contribution 목록
1. **FiLM의 유전형 생성 최초 적용** (모든 선행 연구: 단백질/분자/이미지)
2. **CNN-DiT 하이브리드**: 근거리/장거리 유전 패턴의 역할 분리
3. **계층적 인구군 임베딩**: pop + superpop 계층 구조를 FiLM으로 변조
4. **소수 인구군 강건성**: ASW(61명), MXL(64명) 등에서 생성 품질 개선 정량 검증
5. (선택) **유전학 특화 보조 손실**: PCA 공간 분포 매칭

### 실험 결과 기대 (검증 필요)
- 전체 인구군 Recovery Rate: 현재 93% → 목표 95%+
- 소수 인구군 Recovery Rate: 현재 ~70-80% → 목표 90%+
- AF 상관: 현재 0.94-0.99 → 유지 또는 개선
- LD 보존: 현재 수준 → 개선 (DiT의 장거리 패턴 포착)
- NNAA: ~0.5 유지 (프라이버시 보존)
