# Phase 6: 모델 성능 최적화 전략

- **입력 표현 전제**: Gene PCA 텐서 `(B, K, gene_size)` — `01_overview` "1차 구현 범위" 참조

---

## 전략 분류: 현재 적용 가능 vs 후속 확장

| 분류 | 전략 | 비고 |
|------|------|------|
| **현재 (PCA 기반)** | Tier 1 전체, Tier 2 전체, Tier 3 §3.1~3.4 | Gene PCA 입력에서 바로 적용 |
| **후속 확장 (원시 SNP 전제)** | Tier 3 §3.5 Latent Diffusion | VAE 별도 학습 필요, 원시 유전형 입력 전제 |

> 원시 SNP 전제 전략은 문서 내에서 **⚠️ 후속 확장** 태그로 표기한다.

---

## 개요

기본 HybridGenoDiT 구현 이후, 성능을 체계적으로 끌어올리기 위한 전략을
**데이터 → 학습 기법 → 아키텍처 → 샘플링 → 앙상블** 순서로 정리한다.
각 전략에 대해 구현 난이도, 기대 효과, 우선순위를 명시한다.

```
성능 향상 경로:

[Tier 1] 즉시 적용 (코드 수정 최소)
  ├→ EMA (Exponential Moving Average)
  ├→ 소수 인구군 오버샘플링
  ├→ DDIM / DPM-Solver++ 샘플링
  └→ Gradient Accumulation 최적화

[Tier 2] 중간 난이도 (모듈 추가)
  ├→ 유전체 특화 데이터 증강
  ├→ Classifier-Free Guidance 튜닝
  ├→ v-prediction 목표 전환
  ├→ Min-SNR 가중 손실
  └→ Stochastic Depth (Drop Path)

[Tier 3] 고급 (아키텍처 변경)
  ├→ Flash Attention 2
  ├→ Rotary Position Embedding (RoPE)
  ├→ Curriculum Learning (노이즈 스케줄)
  ├→ Self-Conditioning
  └→ Latent Diffusion 전환 (VAE 선행)
```

---

## Tier 1: 즉시 적용

### 1.1 EMA (Exponential Moving Average)

**효과**: Diffusion 모델에서 거의 필수. 학습 안정성 + 생성 품질 동시 향상.

DiT 원본, Stable Diffusion, DNA-Diffusion 등 모든 주요 Diffusion 논문이 EMA를 사용한다.
학습 중 모델 파라미터의 지수 이동 평균을 별도로 유지하고, 추론 시 EMA 모델을 사용한다.

```python
class EMAModel:
    """
    Exponential Moving Average of model parameters

    θ_ema = decay * θ_ema + (1 - decay) * θ_model
    """
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1 - self.decay
                )

    def apply_to(self, model):
        """추론 시 EMA 파라미터를 모델에 적용"""
        for name, param in model.named_parameters():
            if name in self.shadow:
                param.data.copy_(self.shadow[name])

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict
```

```python
# 학습 루프에 통합
ema = EMAModel(model.module, decay=config['ema_decay'])

for epoch in range(epochs):
    for step, (genes, labels) in enumerate(train_loader):
        # ... forward, backward, optimizer.step() ...
        ema.update(model.module)  # 매 step 후 EMA 업데이트

# 추론 시
ema.apply_to(model.module)
model.eval()
# ... 샘플 생성 ...
```

**저장 시 EMA도 함께 저장**:
```python
torch.save({
    'model_state_dict': model.module.state_dict(),
    'ema_state_dict': ema.state_dict(),
    'config': config,
}, "best_model.pth")
```

| 항목 | 값 |
|------|-----|
| 구현 난이도 | 낮음 |
| 기대 효과 | 높음 (FID 5~15% 개선이 일반적) |
| 추가 VRAM | ~모델 크기만큼 (shadow params) |
| 우선순위 | **필수** |

---

### 1.2 소수 인구군 오버샘플링

**효과**: 소수 인구군(ASW 61명, MXL 64명)의 학습 빈도를 늘려 생성 품질 향상.

```python
class PopulationBalancedSampler(DistributedSampler):
    """
    인구군별 균형 샘플링

    전략: 소수 인구군을 오버샘플링하여 배치 내 인구군 분포를 균등화
    sqrt 비례 샘플링: 원래 n_i → sqrt(n_i) 비율로 조정
    → 완전 균등(1:1)보다 현실적, 완전 비례보다 소수군 강화
    """
    def __init__(self, dataset, labels, num_replicas=None, rank=None):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)

        pop_counts = np.bincount(labels, minlength=26)
        # sqrt 비례 가중치
        weights_per_pop = 1.0 / np.sqrt(pop_counts + 1)
        sample_weights = weights_per_pop[labels]
        self.sample_weights = torch.DoubleTensor(sample_weights)

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch + self.seed)
        indices = torch.multinomial(
            self.sample_weights, len(self.dataset), replacement=True, generator=g
        )
        # DDP 분할
        indices = indices[self.rank::self.num_replicas]
        return iter(indices.tolist())
```

**세 가지 샘플링 전략 비교**:

| 전략 | ASW(61) 배치 비율 | YRI(108) 배치 비율 | 효과 |
|------|------------------|-------------------|------|
| 비례 (현재) | 2.4% | 4.3% | 소수군 학습 부족 |
| sqrt 비례 | 3.2% | 3.6% | **권장** — 균형과 현실 사이 |
| 완전 균등 | 3.8% | 3.8% | 대규모군 과소학습 위험 |

| 항목 | 값 |
|------|-----|
| 구현 난이도 | 낮음 |
| 기대 효과 | 중간 (소수 인구군 품질 10~30% 개선 가능) |
| 우선순위 | **높음** (핵심 논문 contribution과 직결) |

---

### 1.3 고속 샘플러 (DDIM / DPM-Solver++)

**효과**: 생성 품질 유지하면서 샘플링 속도 5~20배 가속.

현재 GeneDiffusion은 500 step 전체를 순차 실행한다.
DDIM이나 DPM-Solver++를 쓰면 50~100 step으로 동등 품질 달성 가능.

```python
class DDIMSampler:
    """
    DDIM (Song et al., 2020): deterministic sampling
    eta=0이면 완전 결정적, eta=1이면 DDPM과 동일
    """
    def __init__(self, diffusion, ddim_steps=50, eta=0.0):
        self.diffusion = diffusion
        self.ddim_steps = ddim_steps
        self.eta = eta

    @torch.no_grad()
    def sample(self, model, xT, y, guidance='normal', w=3.0):
        timesteps = np.linspace(0, self.diffusion.timesteps - 1,
                                self.ddim_steps, dtype=int)[::-1]
        x = xT
        for i, t in enumerate(timesteps):
            t_batch = torch.full((len(x),), t, device=x.device, dtype=torch.long)

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                if guidance == 'classifier_free':
                    eps_cond = model(x, t_batch, y)
                    eps_uncond = model(x, t_batch, torch.full_like(y, 26))  # null class
                    eps = (1 + w) * eps_cond - w * eps_uncond
                else:
                    eps = model(x, t_batch, y)

            # DDIM update
            alpha_t = self.diffusion.scalars.alpha_bar[t]
            alpha_prev = self.diffusion.scalars.alpha_bar[timesteps[i+1]] if i < len(timesteps)-1 else torch.tensor(1.0)

            pred_x0 = (x - (1 - alpha_t).sqrt() * eps) / alpha_t.sqrt()
            if self.diffusion.enforce_zeros:
                pred_x0 = self._apply_zero_mask(pred_x0)

            sigma = self.eta * ((1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev)).sqrt()
            dir_xt = (1 - alpha_prev - sigma**2).sqrt() * eps
            noise = sigma * torch.randn_like(x) if i < len(timesteps)-1 else 0

            x = alpha_prev.sqrt() * pred_x0 + dir_xt + noise

        return x
```

**DPM-Solver++** (Lu et al., 2022): 더 적은 step (20~25)으로도 고품질 생성.
라이브러리 `diffusers`의 `DPMSolverMultistepScheduler`를 직접 활용 가능.

| 샘플러 | Steps | 품질 | 속도 | 구현 난이도 |
|--------|-------|------|------|-----------|
| DDPM (현재) | 500 | 기준 | 1x | - |
| DDIM (eta=0) | 50 | ≈기준 | 10x | 낮음 |
| DPM-Solver++ | 25 | ≈기준 | 20x | 중간 (라이브러리 연동) |

| 항목 | 값 |
|------|-----|
| 기대 효과 | 속도 10~20배 가속 (품질 유지) |
| 우선순위 | **높음** (200회 HP 서치 시 시간 절약에 결정적) |

---

### 1.4 Gradient Accumulation 최적화

**효과**: 실효 배치를 키워 학습 안정성 향상.

```
현재: batch_size=32 × 2 GPU = 실효 64
최적화: batch_size=32 × grad_accum=4 × 2 GPU = 실효 256

큰 배치 = 더 안정적인 gradient 추정 = 더 안정적 학습
특히 26개 인구군으로 나뉘는 조건부 생성에서 중요
(배치 내에 다양한 인구군이 포함되어야 FiLM이 잘 학습됨)
```

```python
# gradient accumulation 적용
for micro_step in range(config['gradient_accumulation_steps']):
    genes, labels = next(train_iter)
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        pred_eps = model(xt, t, y=labels)
        loss = criterion(pred_eps, eps) / config['gradient_accumulation_steps']
    loss.backward()

optimizer.step()
optimizer.zero_grad(set_to_none=True)
```

| 항목 | 값 |
|------|-----|
| 구현 난이도 | 매우 낮음 (이미 GeneDiffusion에 패턴 존재) |
| 기대 효과 | 중간 (학습 안정성 향상) |
| 우선순위 | 높음 |

---

## Tier 2: 중간 난이도

### 2.1 유전체 특화 데이터 증강

일반 이미지 증강(flip, crop)은 유전체에 적용할 수 없다.
대신 **도메인 타당성이 있는** 증강 전략을 사용한다.

#### (a) PCA 공간 가우시안 노이즈

```python
class GeneticAugmentation:
    """PCA 공간에서의 미세 노이즈 증강"""

    def __init__(self, noise_scale=0.01):
        self.noise_scale = noise_scale

    def __call__(self, x):
        """
        x: (n_genes, 8) PCA 피처

        근거: PCA 주성분의 미세 변동은 동일 인구군 내 자연적 변이에 해당
        noise_scale은 각 성분의 std 대비 비율
        """
        noise = torch.randn_like(x) * self.noise_scale
        return x + noise
```

#### (b) 유전자 순서 셔플 (within chromosome)

```python
def shuffle_genes_within_chromosome(x, gene_to_chrom):
    """
    동일 염색체 내 유전자 순서를 셔플

    근거: PCA 표현에서는 유전자 간 위치 관계가 이미 추상화되어 있으므로,
    같은 염색체 내 유전자 순서 변경은 LD 구조를 크게 훼손하지 않음
    """
    x_aug = x.clone()
    for chrom in range(1, 23):
        idx = [i for i, c in gene_to_chrom.items() if c == chrom]
        perm = torch.randperm(len(idx))
        x_aug[idx] = x_aug[[idx[p] for p in perm]]
    return x_aug
```

#### (c) Mixup (같은 인구군 내)

```python
def population_mixup(x1, x2, alpha=0.2):
    """
    같은 인구군의 두 샘플을 보간

    근거: 동일 인구군 내 개체 간 유전적 변이는 연속적이므로
    볼록 조합(convex combination)이 생물학적으로 타당
    """
    lam = np.random.beta(alpha, alpha)
    return lam * x1 + (1 - lam) * x2
```

| 증강 방법 | 도메인 타당성 | 구현 난이도 | 기대 효과 |
|----------|-------------|-----------|----------|
| PCA 가우시안 노이즈 | 높음 | 매우 낮음 | 중간 |
| 염색체 내 셔플 | 중간 | 낮음 | 낮음~중간 |
| 인구군 내 Mixup | 높음 | 낮음 | 중간 |

---

### 2.2 Classifier-Free Guidance (CFG) 정밀 튜닝

**효과**: 인구군 조건부 생성의 품질-다양성 균형을 세밀하게 조절.

```python
# 학습 시: 일정 확률로 레이블을 null로 대체
def apply_cfg_dropout(labels, null_class=26, dropout_rate=0.1):
    """
    CFG 학습: 10% 확률로 레이블을 null_class로 대체
    → 모델이 조건부/비조건부 생성을 모두 학습
    """
    mask = torch.rand(len(labels)) < dropout_rate
    labels_dropped = labels.clone()
    labels_dropped[mask] = null_class
    return labels_dropped

# 추론 시: guidance scale w로 조건 강도 조절
def guided_prediction(model, x, t, y, null_class=26, w=3.0):
    eps_cond = model(x, t, y)
    eps_uncond = model(x, t, torch.full_like(y, null_class))
    return (1 + w) * eps_cond - w * eps_uncond
```

**인구군별 guidance weight 차별화** (고급):

```python
def adaptive_guidance_weight(pop_idx, pop_sizes, base_w=3.0):
    """
    소수 인구군 → 더 강한 guidance (조건 신호 강화)
    대규모 인구군 → 보통 guidance

    근거: 소수 인구군은 학습 데이터가 적어 비조건부 모델로 쏠리기 쉬움
    → guidance를 강화하여 인구군 특이성 보존
    """
    n = pop_sizes[pop_idx]
    median_n = np.median(list(pop_sizes.values()))
    # 소수군일수록 w↑, 대규모군일수록 w→base
    adaptive_w = base_w * (median_n / n) ** 0.3
    return min(adaptive_w, base_w * 2)  # 상한 설정
```

| 항목 | 값 |
|------|-----|
| 구현 난이도 | 낮음~중간 |
| 기대 효과 | 높음 (인구군별 품질 균형 개선) |
| 우선순위 | **높음** (논문 contribution과 직결) |

---

### 2.3 v-prediction 목표 전환

**효과**: 노이즈 수준이 높을 때(t가 클 때) 학습 안정성 향상.

```
기존: ε-prediction → 모델이 노이즈 ε를 예측
제안: v-prediction → 모델이 v = √ᾱ·ε - √(1-ᾱ)·x₀ 를 예측

v-prediction 장점:
- t=0 근처와 t=T 근처 모두에서 안정적
- Progressive Distillation(Salimans & Ho, 2022)과 호환
- Stable Diffusion 2.x 이후 기본 채택
```

```python
# v-prediction으로 전환
def compute_v_target(x0, eps, alpha_bar_t):
    """v = √ᾱ·ε - √(1-ᾱ)·x₀"""
    return alpha_bar_t.sqrt() * eps - (1 - alpha_bar_t).sqrt() * x0

def predict_x0_from_v(xt, v, alpha_bar_t):
    """v에서 x₀ 복원"""
    return alpha_bar_t.sqrt() * xt - (1 - alpha_bar_t).sqrt() * v
```

| 항목 | 값 |
|------|-----|
| 구현 난이도 | 중간 (diffusion.py 수정 필요) |
| 기대 효과 | 중간 (학습 안정성 + 품질 개선) |
| 우선순위 | 중간 |

---

### 2.4 Min-SNR 가중 손실 (Min-SNR-γ)

**효과**: 타임스텝별 손실 가중치를 자동 조정하여 학습 효율 향상.

Hang et al. (2023) "Efficient Diffusion Training via Min-SNR Weighting Strategy"

```python
def min_snr_weight(t, scalars, gamma=5.0):
    """
    Min-SNR-γ 가중치

    SNR(t) = ᾱ_t / (1 - ᾱ_t)
    weight(t) = min(SNR(t), γ) / SNR(t)

    효과: 높은 노이즈(낮은 SNR) 타임스텝의 과도한 기여를 억제
    → 학습이 특정 타임스텝에 쏠리지 않음
    """
    snr = scalars.alpha_bar[t] / (1 - scalars.alpha_bar[t])
    weight = torch.clamp(snr, max=gamma) / snr
    return weight

# 손실 함수에 적용
loss_per_sample = ((pred_eps - eps) ** 2).mean(dim=(1, 2))  # (B,)
weights = min_snr_weight(t, diffusion.scalars, gamma=5.0)   # (B,)
loss = (loss_per_sample * weights).mean()
```

| 항목 | 값 |
|------|-----|
| 구현 난이도 | 매우 낮음 (5줄 추가) |
| 기대 효과 | 중간 (학습 효율 10~20% 향상) |
| 우선순위 | 높음 |

---

### 2.5 Stochastic Depth (Drop Path)

**효과**: DiT 블록을 확률적으로 건너뛰어 정규화 + 학습 가속.

```python
class DropPath(nn.Module):
    """학습 시 확률적으로 경로를 건너뜀 (DiT 블록 수준)"""
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1 - self.drop_prob
        mask = torch.rand(x.shape[0], 1, 1, device=x.device) < keep_prob
        return x * mask / keep_prob

# DiTBlock에 적용
class DiTBlock(nn.Module):
    def __init__(self, d_model, n_heads, drop_path_rate=0.0):
        # ...
        self.drop_path = DropPath(drop_path_rate)

    def forward(self, x, film_params):
        # Self-Attention
        h = self.norm1(x)
        h = g1 * h + b1
        h, _ = self.attn(h, h, h)
        x = x + self.drop_path(a1 * h)  # drop path 적용

        # FFN
        h = self.norm2(x)
        h = g2 * h + b2
        h = self.mlp(h)
        x = x + self.drop_path(a2 * h)  # drop path 적용
        return x
```

블록별 점진적 증가 (deeper blocks → higher drop rate):
```python
drop_rates = [0.0 + i * 0.1 / (n_blocks - 1) for i in range(n_blocks)]
# 예: 4블록 → [0.0, 0.033, 0.067, 0.1]
```

---

## Tier 3: 고급

### 3.1 Flash Attention 2

**효과**: DiT의 self-attention 메모리 O(N²) → O(N), 속도 2~4배 가속.

```python
# PyTorch 2.0+ 내장 Flash Attention
class DiTBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        # nn.MultiheadAttention 대신 직접 구현
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.proj = nn.Linear(d_model, d_model)
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

    def _attention(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        q = q.transpose(1, 2)  # (B, H, N, D)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Flash Attention (PyTorch 2.0+ SDPA)
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=False
        )
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.proj(out)
```

RTX A6000 (Ampere) + PyTorch 2.0+ 에서 자동 활성화됨.

| 항목 | 값 |
|------|-----|
| 구현 난이도 | 중간 (attention 직접 구현) |
| 기대 효과 | 속도 2~4배, 메모리 50% 절감 |
| 우선순위 | 중간~높음 (토큰 수 208이면 효과 제한적, 토큰 증가 시 필수) |

---

### 3.2 Rotary Position Embedding (RoPE)

**효과**: 유전자 간 상대적 거리를 자연스럽게 인코딩.

```python
class RotaryEmbedding(nn.Module):
    """
    RoPE: 유전체에서의 의미

    유전자 A와 B의 상대적 위치(염색체 내 거리)가
    attention score에 자연스럽게 반영됨.
    → 가까운 유전자는 더 강한 상관, 먼 유전자는 약한 상관
    → LD 감쇠 패턴과 일치

    Learned position embedding과 달리 외삽(extrapolation) 가능
    → 다른 유전자 수의 데이터에도 일반화 가능
    """
    def __init__(self, dim, max_len=1024):
        super().__init__()
        freqs = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_len)
        freqs = torch.outer(t, freqs)
        self.register_buffer('cos_cached', freqs.cos())
        self.register_buffer('sin_cached', freqs.sin())

    def forward(self, x):
        seq_len = x.shape[1]
        cos = self.cos_cached[:seq_len].unsqueeze(0)
        sin = self.sin_cached[:seq_len].unsqueeze(0)
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.cat([
            x1 * cos - x2 * sin,
            x2 * cos + x1 * sin
        ], dim=-1)
```

---

### 3.3 Self-Conditioning

**효과**: 이전 step의 x₀ 예측을 다음 step의 입력에 추가 → 생성 일관성 향상.

Chen et al. (2022) "Analog Bits: Generating Discrete Data using Diffusion Models with Self-Conditioning"

```python
# 학습 시: 50% 확률로 self-conditioning 적용
def train_step_with_self_conditioning(model, x0, t, y, diffusion):
    xt, eps = diffusion.sample_from_forward_process(x0, t)

    x0_pred_prev = torch.zeros_like(x0)

    if random.random() > 0.5:
        with torch.no_grad():
            eps_pred = model(xt, t, y, x0_self_cond=torch.zeros_like(x0))
            x0_pred_prev = diffusion.predict_x0(xt, eps_pred, t)

    eps_pred = model(xt, t, y, x0_self_cond=x0_pred_prev.detach())
    loss = criterion(eps_pred, eps)
    return loss
```

```python
# 모델에 self-cond 채널 추가
class HybridCNNDiTFiLM(nn.Module):
    def __init__(self, config):
        # 입력 채널: 8 (원본) + 8 (self-cond) = 16
        in_ch = config['num_channels'] * 2 if config.get('self_conditioning') else config['num_channels']
        self.encoder = CNNStemEncoder(in_channels=in_ch, ...)

    def forward(self, x, t, y, x0_self_cond=None):
        if x0_self_cond is not None:
            x = torch.cat([x, x0_self_cond], dim=1)
        else:
            x = torch.cat([x, torch.zeros_like(x)], dim=1)
        # ... 나머지 동일 ...
```

| 항목 | 값 |
|------|-----|
| 구현 난이도 | 중간 |
| 기대 효과 | 중간~높음 (FID 5~10% 개선 보고) |
| 우선순위 | 중간 |

---

### 3.4 Curriculum Learning (노이즈 스케줄)

**효과**: 학습 초기에는 쉬운 task(낮은 노이즈)에 집중, 후기에는 어려운 task(높은 노이즈)로 확장.

```python
class CurriculumTimestepSampler:
    """
    학습 초기: t를 [0, T/2] 범위에서 샘플링 (저노이즈 위주)
    학습 후기: t를 [0, T] 전체에서 균일 샘플링

    근거: 유전형 데이터는 구조가 복잡하므로 점진적 학습이 유리
    """
    def __init__(self, max_timesteps, total_epochs):
        self.max_timesteps = max_timesteps
        self.total_epochs = total_epochs

    def sample(self, batch_size, epoch, device):
        # 선형 확장: epoch 0에서 T/4, 마지막 epoch에서 T
        progress = min(epoch / (self.total_epochs * 0.5), 1.0)
        t_max = int(self.max_timesteps * (0.25 + 0.75 * progress))
        return torch.randint(0, t_max, (batch_size,), device=device)
```

---

### 3.5 Latent Diffusion 전환 (VAE 선행) — ⚠️ 후속 확장

> **이 전략은 원시 SNP/유전형 입력을 전제로 하며, 1차 구현 범위(Gene PCA 기반)에 포함되지 않는다.**

**효과**: 차원을 대폭 축소하여 DiT 부담 감소 + 더 큰 DiT 사용 가능.

```
현재: (8, 26624) → CNN 다운 → (256, 3328) → DiT → CNN 업 → (8, 26624)
Latent: (8, 26624) → VAE 인코더 → (d, L_latent) → DiT → VAE 디코더 → (8, 26624)

SNPgen(2026)이 유전형 Latent Diffusion의 가능성을 이미 입증.
단, VAE를 별도 학습해야 하므로 2단계 파이프라인이 됨.
```

| 항목 | 값 |
|------|-----|
| 구현 난이도 | 높음 (VAE 별도 설계/학습) |
| 기대 효과 | 높음 (더 큰 DiT 사용 가능, 속도 향상) |
| 우선순위 | 낮음 (후속 연구 소재) |

---

## 종합 우선순위 로드맵

### Phase A: 기본 구현 후 즉시 적용

| # | 전략 | 기대 효과 | 추가 코드량 |
|---|------|----------|-----------|
| 1 | **EMA** | 생성 품질 5~15%↑ | ~50줄 |
| 2 | **소수 인구군 오버샘플링** | 소수군 품질 10~30%↑ | ~30줄 |
| 3 | **DDIM 샘플러** | 속도 10x↑ (품질 유지) | ~60줄 |
| 4 | **Min-SNR 가중 손실** | 학습 효율 10~20%↑ | ~5줄 |
| 5 | **Gradient Accum 최적화** | 학습 안정성↑ | 설정 변경만 |

### Phase B: HP 서치와 병행

| # | 전략 | 기대 효과 | HP 서치 파라미터 |
|---|------|----------|----------------|
| 6 | **CFG + 적응 guidance** | 인구군별 품질 균형 | guidance_weight, cfg_dropout |
| 7 | **v-prediction** | 학습 안정성↑ | prediction_target |
| 8 | **Stochastic Depth** | 정규화 + 속도↑ | stochastic_depth_rate |
| 9 | **데이터 증강** | 과적합 방지 | noise_scale, mixup_alpha |

### Phase C: 최종 모델 확정 후

| # | 전략 | 기대 효과 | 비고 |
|---|------|----------|------|
| 10 | **Flash Attention** | 속도 2~4x↑ | 토큰 수 증가 시 필수 |
| 11 | **RoPE** | 위치 일반화↑ | learned emb 대체 |
| 12 | **Self-Conditioning** | 생성 일관성↑ | 채널 수 2배 |
| 13 | **Curriculum Learning** | 학습 효율↑ | 스케줄 설계 필요 |
| 14 | **Latent Diffusion** ⚠️ | 차원 축소 | 후속 확장 (원시 SNP 전제, 1차 범위 외) |

---

## 성능 추적 체크리스트

각 최적화 전략 적용 시 다음을 wandb에 기록:

```
□ 적용 전 baseline val_rec_error
□ 적용 후 val_rec_error
□ Recovery Rate 변화
□ 인구군별 AF 상관 변화 (특히 소수군: ASW, MXL)
□ 크기-품질 상관(size_quality_corr) 변화
□ 학습 시간 변화
□ VRAM 사용량 변화
□ 샘플링 속도 변화
```
