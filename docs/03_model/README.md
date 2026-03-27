# Phase 2: 모델 아키텍처 상세 설계

---

## 입력 표현 (확정)

> **입력**: Gene PCA 텐서 `(B, K, gene_size)` + 인구군 레이블 `(B,)` — K는 전처리 그리드 서치로 자동 결정, gene_size=26624 (128의 배수 패딩)
>
> 원시 SNP/하플로타입 기반 입력은 1차 구현 범위에 포함되지 않는다. → `01_overview` "1차 구현 범위" 참조

---

## 1. 구현 순서

모듈 간 의존성에 따라 다음 순서로 구현한다:

```
[Step 2-1] 기초 모듈
    ├→ timestep_embedding()
    ├→ zero_module()
    └→ GroupNorm32

[Step 2-2] 조건부 임베딩
    ├→ HierarchicalPopulationEmbedding
    └→ UnifiedFiLMGenerator

[Step 2-3] CNN 모듈
    ├→ FiLMConvBlock
    ├→ CNNStemEncoder
    ├→ FiLMDeconvBlock
    └→ CNNDecoder

[Step 2-4] DiT 모듈
    ├→ PatchEmbed1D
    ├→ DiTBlock (AdaLN-Zero)
    ├→ DiTCore
    └→ UnPatchify

[Step 2-5] 전체 조립
    ├→ HybridCNNDiTFiLM
    └→ Shape 검증 테스트

[Step 2-6] Diffusion Process
    └→ GaussianDiffusion (기존 코드 기반 리팩토링)
```

---

## 2. Step 2-1: 기초 모듈

### 파일: `src/models/modules/base.py`

```python
import math
import torch
import torch.nn as nn

def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """
    사인파 타임스텝 임베딩 생성
    Args:
        timesteps: (B,) 정수 타임스텝
        dim: 임베딩 차원
    Returns:
        (B, dim) 임베딩 벡터
    """
    assert timesteps.dim() == 1, f"Expected 1D timesteps, got {timesteps.dim()}D"
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=timesteps.device) / half
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def zero_module(module: nn.Module) -> nn.Module:
    """모듈의 모든 파라미터를 0으로 초기화 (AdaLN-Zero용)"""
    for p in module.parameters():
        p.detach().zero_()
    return module
```

---

## 3. Step 2-2: 조건부 임베딩

### 파일: `src/models/modules/conditioning.py`

#### HierarchicalPopulationEmbedding

```python
class HierarchicalPopulationEmbedding(nn.Module):
    """
    인구군(26) + 슈퍼인구군(5) 계층적 임베딩

    도메인 근거:
    - 슈퍼인구군(AFR 661명)은 안정적 학습 기반 제공
    - 세부 인구군(ASW 61명)은 그 위에 미세 차이만 학습
    - pop_to_superpop 매핑은 전처리에서 생성된 label_hierarchy.pkl에서 로드
    """
    def __init__(self, n_pops: int = 26, n_superpops: int = 5, d_model: int = 256,
                 pop_to_superpop: dict = None):
        super().__init__()
        if pop_to_superpop is None:
            raise ValueError("pop_to_superpop mapping is required")

        mapping = torch.zeros(n_pops, dtype=torch.long)
        for pop_idx, superpop_idx in pop_to_superpop.items():
            mapping[pop_idx] = superpop_idx
        self.register_buffer('pop_to_superpop_map', mapping)

        self.pop_emb = nn.Embedding(n_pops, d_model)
        self.superpop_emb = nn.Embedding(n_superpops, d_model)
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, pop_label: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pop_label: (B,) 인구군 인덱스 0-25
        Returns:
            (B, d_model) 계층적 임베딩
        """
        assert pop_label.dim() == 1, f"Expected 1D labels, got {pop_label.shape}"
        p_emb = self.pop_emb(pop_label)
        sp_label = self.pop_to_superpop_map[pop_label]
        sp_emb = self.superpop_emb(sp_label)
        return self.fusion(torch.cat([p_emb, sp_emb], dim=-1))
```

#### UnifiedFiLMGenerator

```python
class UnifiedFiLMGenerator(nn.Module):
    """
    인구군 임베딩 + 타임스텝 → CNN/DiT 모든 블록의 FiLM 파라미터 생성

    출력:
    - CNN 인코더 블록별: (gamma, beta) 각 (B, channels)
    - CNN 디코더 블록별: (gamma, beta) 각 (B, channels)
    - DiT 블록별: (B, d_model*6) → (gamma1,beta1,alpha1, gamma2,beta2,alpha2)
    """
    def __init__(self, d_model: int = 256, d_time: int = 256,
                 cnn_channels: list = None,      # 예: [64, 64, 128, 256]
                 n_dit_blocks: int = 4, d_dit: int = 256):
        super().__init__()
        if cnn_channels is None:
            raise ValueError("cnn_channels list is required")

        self.d_time = d_time

        # 타임스텝 MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(d_time, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
        )

        # 조건 융합
        self.cond_mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
        )

        # CNN 인코더 FiLM
        self.cnn_enc_films = nn.ModuleList([
            nn.Linear(d_model, ch * 2) for ch in cnn_channels
        ])

        # CNN 디코더 FiLM
        self.cnn_dec_films = nn.ModuleList([
            nn.Linear(d_model, ch * 2) for ch in reversed(cnn_channels)
        ])

        # DiT AdaLN-Zero (6개: gamma1,beta1,alpha1 for attn + gamma2,beta2,alpha2 for FFN)
        self.dit_films = nn.ModuleList([
            nn.Sequential(nn.SiLU(), zero_module(nn.Linear(d_model, d_dit * 6)))
            for _ in range(n_dit_blocks)
        ])

    def forward(self, pop_emb: torch.Tensor, t: torch.Tensor):
        """
        Args:
            pop_emb: (B, d_model) from HierarchicalPopulationEmbedding
            t: (B,) diffusion timestep (정수)
        Returns:
            cnn_enc_params: list of (gamma, beta) per encoder block
            cnn_dec_params: list of (gamma, beta) per decoder block
            dit_params: list of (B, d_dit*6) per DiT block
        """
        t_emb = self.time_mlp(timestep_embedding(t, self.d_time))
        cond = self.cond_mlp(torch.cat([pop_emb, t_emb], dim=-1))

        cnn_enc_params = []
        for layer in self.cnn_enc_films:
            gb = layer(cond)
            gamma, beta = gb.chunk(2, dim=-1)
            cnn_enc_params.append((gamma, beta))

        cnn_dec_params = []
        for layer in self.cnn_dec_films:
            gb = layer(cond)
            gamma, beta = gb.chunk(2, dim=-1)
            cnn_dec_params.append((gamma, beta))

        dit_params = [layer(cond) for layer in self.dit_films]

        return cnn_enc_params, cnn_dec_params, dit_params
```

---

## 4. Step 2-3: CNN 모듈

### 파일: `src/models/modules/cnn.py`

```python
class FiLMConvBlock(nn.Module):
    """
    FiLM 변조 1D Conv 블록

    Conv1d → GroupNorm → FiLM(γ·h + β) → SiLU → Conv1d → GroupNorm → SiLU
    + Residual connection + Optional downsample
    """
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3,
                 downsample: bool = True):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad)
        self.norm1 = nn.GroupNorm(min(32, out_ch), out_ch)
        self.norm2 = nn.GroupNorm(min(32, out_ch), out_ch)
        self.downsample = nn.Conv1d(out_ch, out_ch, 2, stride=2) if downsample else nn.Identity()
        self.skip_proj = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor):
        """
        x: (B, C_in, L)
        gamma: (B, C_out) — FiLM scale
        beta: (B, C_out) — FiLM shift
        Returns: (out, skip) — downsampled output + skip connection
        """
        residual = self.skip_proj(x)
        h = self.conv1(x)
        h = self.norm1(h)
        h = gamma.unsqueeze(-1) * h + beta.unsqueeze(-1)  # FiLM
        h = torch.nn.functional.silu(h)
        h = self.conv2(h)
        h = self.norm2(h)
        h = torch.nn.functional.silu(h)
        skip = h + residual[..., :h.shape[-1]]
        out = self.downsample(h)
        return out, skip


class CNNStemEncoder(nn.Module):
    """
    (B, 8, 26624) → (B, 4C, L_down) + skip connections

    도메인 근거: CNN은 근거리 LD 패턴과 하플로타입 청크 구조를 포착한다.
    다운샘플링으로 시퀀스 길이를 줄여 DiT의 토큰 수를 줄인다.
    """
    def __init__(self, in_channels: int = 8, base_channels: int = 64,
                 channel_mult: tuple = (1, 1, 2, 4), kernel_size: int = 3):
        super().__init__()
        channels = [base_channels * m for m in channel_mult]
        self.blocks = nn.ModuleList()
        ch_in = in_channels
        for i, ch_out in enumerate(channels):
            downsample = (i < len(channels) - 1)  # 마지막 블록은 다운샘플 없음
            self.blocks.append(FiLMConvBlock(ch_in, ch_out, kernel_size, downsample))
            ch_in = ch_out

    def forward(self, x, cnn_film_params):
        assert len(cnn_film_params) == len(self.blocks), (
            f"FiLM params count {len(cnn_film_params)} != blocks {len(self.blocks)}"
        )
        skips = []
        for block, (gamma, beta) in zip(self.blocks, cnn_film_params):
            x, skip = block(x, gamma, beta)
            skips.append(skip)
        return x, skips
```

CNN Decoder는 대칭 구조로 `FiLMDeconvBlock` + skip concat으로 구성.

---

## 5. Step 2-4: DiT 모듈

### 파일: `src/models/modules/dit.py`

```python
class PatchEmbed1D(nn.Module):
    """CNN 출력 → DiT 토큰 변환"""
    def __init__(self, seq_len: int, in_channels: int, patch_size: int = 16,
                 d_model: int = 256):
        super().__init__()
        assert seq_len % patch_size == 0, (
            f"seq_len {seq_len} not divisible by patch_size {patch_size}"
        )
        self.patch_size = patch_size
        self.n_tokens = seq_len // patch_size
        self.proj = nn.Linear(in_channels * patch_size, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, self.n_tokens, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, L) → (B, N_tokens, d_model)"""
        B, C, L = x.shape
        x = x.reshape(B, C, self.n_tokens, self.patch_size)
        x = x.permute(0, 2, 1, 3).reshape(B, self.n_tokens, -1)
        x = self.proj(x) + self.pos_emb
        return x


class DiTBlock(nn.Module):
    """
    AdaLN-Zero DiT Block (Peebles & Xie, ICCV 2023) = FiLM + Zero-init

    핵심:
    - LayerNorm의 affine을 끄고 (elementwise_affine=False)
    - 대신 FiLM 파라미터(γ,β)를 조건에서 예측하여 주입
    - α (gate)를 0으로 초기화 → 학습 초기 DiT ≈ 항등함수
    """
    def __init__(self, d_model: int = 256, n_heads: int = 4,
                 mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, int(d_model * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(d_model * mlp_ratio), d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, film_params: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, d)
        film_params: (B, d*6) → γ1,β1,α1 (attn) + γ2,β2,α2 (FFN)
        """
        assert film_params.shape[-1] == x.shape[-1] * 6, (
            f"FiLM params dim {film_params.shape[-1]} != expected {x.shape[-1] * 6}"
        )
        g1, b1, a1, g2, b2, a2 = [p.unsqueeze(1) for p in film_params.chunk(6, dim=-1)]

        # Self-Attention + FiLM
        h = self.norm1(x)
        h = g1 * h + b1
        h, _ = self.attn(h, h, h)
        x = x + a1 * h

        # FFN + FiLM
        h = self.norm2(x)
        h = g2 * h + b2
        h = self.mlp(h)
        x = x + a2 * h

        return x


class DiTCore(nn.Module):
    """경량 DiT 코어: N개 DiTBlock 스택"""
    def __init__(self, n_blocks: int = 4, d_model: int = 256,
                 n_heads: int = 4, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.blocks = nn.ModuleList([
            DiTBlock(d_model, n_heads, mlp_ratio, dropout)
            for _ in range(n_blocks)
        ])
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, dit_params: list) -> torch.Tensor:
        assert len(dit_params) == len(self.blocks)
        for block, params in zip(self.blocks, dit_params):
            x = block(x, params)
        return self.final_norm(x)
```

---

## 6. Step 2-5: 전체 조립

### 파일: `src/models/hybrid_geno_dit.py`

```python
class HybridCNNDiTFiLM(nn.Module):
    """
    CNN-DiT Hybrid with Unified FiLM Conditioning

    도메인 매핑:
    - CNN Encoder: 근거리 LD + 하플로타입 패턴 (FiLM: 인구군별 조절)
    - DiT Core: 장거리 유전자 간 상호작용 (AdaLN-Zero: 인구군별 조절)
    - CNN Decoder: 유전형 복원 (FiLM: 인구군별 미세 조정)
    """
    def __init__(self, config: dict):
        super().__init__()
        # config에서 파라미터 추출 + 검증
        # ... (모든 config 값 검증) ...

        self.pop_embedding = HierarchicalPopulationEmbedding(...)
        self.film_gen = UnifiedFiLMGenerator(...)
        self.encoder = CNNStemEncoder(...)
        self.patchify = PatchEmbed1D(...)
        self.dit = DiTCore(...)
        self.unpatchify = nn.Linear(...)
        self.decoder = CNNDecoder(...)

    def forward(self, x, t, y):
        # shape 검증
        assert x.shape[1] == self.in_channels and x.shape[2] == self.gene_size

        # 조건부 임베딩
        pop_emb = self.pop_embedding(y)
        cnn_enc_p, cnn_dec_p, dit_p = self.film_gen(pop_emb, t)

        # CNN → DiT → CNN
        features, skips = self.encoder(x, cnn_enc_p)
        tokens = self.patchify(features)
        tokens = self.dit(tokens, dit_p)
        # un-patchify + merge + decode
        # ...
        return output
```

### Shape 검증 테스트

```python
def test_forward_pass():
    """모든 모듈의 입출력 shape 검증"""
    config = get_default_config()
    model = HybridCNNDiTFiLM(config).cuda()

    B = 4
    x = torch.randn(B, 8, 26624).cuda()
    t = torch.randint(0, 500, (B,)).cuda()
    y = torch.randint(0, 26, (B,)).cuda()

    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        out = model(x, t, y)

    assert out.shape == x.shape, f"Output shape {out.shape} != input shape {x.shape}"

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Forward pass OK. Shape: {out.shape}, Params: {n_params/1e6:.2f}M")
```

---

## 7. Step 2-6: Diffusion Process

### 파일: `src/models/diffusion.py`

기존 GeneDiffusion의 `GuassianDiffusion`을 리팩토링하여:
- bf16 호환 보장
- enforce_zeros 통합
- DDP에서 안전한 버퍼 관리
- 에러 처리 강화

```python
class GaussianDiffusion:
    def __init__(self, timesteps: int = 500, device: str = "cuda",
                 zero_mask: torch.Tensor = None, enforce_zeros: bool = True):
        if timesteps < 1:
            raise ValueError(f"timesteps must be positive, got {timesteps}")
        # ... 코사인 스케줄 계산 ...
        # zero_mask를 bf16 호환 형태로 저장
        if zero_mask is not None:
            self.zero_mask = zero_mask.bool()  # bool로 통일
        self.enforce_zeros = enforce_zeros
```

---

## 8. 파일 구조 요약

```
src/models/
├── __init__.py
├── hybrid_geno_dit.py          # HybridCNNDiTFiLM (전체 모델)
├── diffusion.py                # GaussianDiffusion (확산 프로세스)
└── modules/
    ├── __init__.py
    ├── base.py                 # timestep_embedding, zero_module
    ├── conditioning.py         # HierarchicalPopulationEmbedding, UnifiedFiLMGenerator
    ├── cnn.py                  # FiLMConvBlock, CNNStemEncoder, CNNDecoder
    └── dit.py                  # PatchEmbed1D, DiTBlock, DiTCore
```
