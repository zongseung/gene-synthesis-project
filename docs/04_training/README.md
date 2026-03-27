# Phase 3: 학습 파이프라인 (DDP + bf16)

---

## 1. 학습 아키텍처

```
torchrun --nproc_per_node=2 src/train.py --config configs/default.yaml

GPU 0 (rank 0)                      GPU 1 (rank 1)
┌─────────────────────┐             ┌─────────────────────┐
│ DDP Model replica   │             │ DDP Model replica   │
│ + bf16 autocast     │             │ + bf16 autocast     │
│ + wandb logging     │◄──gradient──►│                     │
│                     │   all-reduce │                     │
└─────────────────────┘             └─────────────────────┘
         ▲                                    ▲
         │                                    │
    DistributedSampler              DistributedSampler
    (데이터 절반)                   (데이터 절반)
```

---

## 2. DDP 초기화

### 파일: `src/utils/distributed.py`

```python
import os
import torch
import torch.distributed as dist

def setup_ddp():
    """DDP 환경 초기화"""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. DDP requires GPU.")

    n_gpus = torch.cuda.device_count()
    if n_gpus < 2:
        raise RuntimeError(f"DDP requires 2+ GPUs, found {n_gpus}")

    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)

    return local_rank


def cleanup_ddp():
    """DDP 환경 정리"""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    """rank 0인지 확인 (로깅/저장은 rank 0만)"""
    return not dist.is_initialized() or dist.get_rank() == 0
```

---

## 3. 학습 루프 상세

### 파일: `src/train.py`

```python
def train(config):
    # ── DDP 초기화 ──
    local_rank = setup_ddp()
    device = torch.device(f'cuda:{local_rank}')

    # ── 데이터 로드 ──
    train_dataset = GeneticDataset1k(train=True, config=config)
    test_dataset = GeneticDataset1k(train=False, config=config)

    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    test_sampler = DistributedSampler(test_dataset, shuffle=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True,  # DDP에서 배치 크기 불일치 방지
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        sampler=test_sampler,
        num_workers=8,
        pin_memory=True,
    )

    # ── 모델 ──
    model = HybridCNNDiTFiLM(config).to(device)
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank]
    )

    # ── Diffusion ──
    zero_mask = torch.load(config['zero_mask_path'], map_location=device)
    diffusion = GaussianDiffusion(
        timesteps=config['max_timesteps'],
        device=device,
        zero_mask=zero_mask,
        enforce_zeros=config['enforce_zeros'],
    )

    # ── Optimizer (GradScaler 없이) ──
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['lr_diffusion'],
        weight_decay=config['weight_decay'],
        betas=(0.9, 0.999),
    )

    # ── EMA (성능 최적화 09 참고) ──
    ema = EMAModel(model.module, decay=config.get('ema_decay', 0.9999))
    scheduler = CosineWarmupScheduler(
        optimizer,
        warmup=config['warmup_steps'],
        max_iters=config['epochs'] * len(train_loader),
    )

    # ── 손실 함수 ──
    criterion = nn.MSELoss()

    # ── wandb (rank 0만) ──
    if is_main_process():
        wandb.init(project="HybridGenoDiT", config=config)

    # ── 학습 루프 (조기종료 없음) ──
    best_val_rec = float('inf')

    for epoch in range(config['epochs']):
        train_sampler.set_epoch(epoch)  # DDP 셔플링에 필수
        model.train()

        epoch_loss = 0.0
        epoch_steps = 0

        for step, (genes, labels) in enumerate(train_loader):
            genes = genes.to(device, non_blocking=True).permute(0, 2, 1)  # (B, 8, 26624)
            labels = labels.to(device, non_blocking=True)

            t = torch.randint(config['max_timesteps'], (len(genes),), device=device)
            xt, eps = diffusion.sample_from_forward_process(genes, t)

            optimizer.zero_grad(set_to_none=True)  # 메모리 효율

            # ── bf16 autocast (GradScaler 없이) ──
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                pred_eps = model(xt, t, y=labels)

                if config['enforce_zeros']:
                    loss = masked_mse_loss(pred_eps, eps, diffusion.not_zero_mask)
                else:
                    loss = criterion(pred_eps, eps)

            # bf16에서는 GradScaler 불필요
            loss.backward()

            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config['gradient_clipping']
            )

            # NaN 감지
            if torch.isnan(loss):
                raise RuntimeError(
                    f"NaN loss at epoch {epoch}, step {step}. "
                    f"lr={optimizer.param_groups[0]['lr']:.2e}, "
                    f"grad_norm={grad_norm:.4f}"
                )

            optimizer.step()
            scheduler.step()
            ema.update(model.module)  # EMA 업데이트 (매 step)

            epoch_loss += loss.item()
            epoch_steps += 1

            # 로깅 (rank 0만)
            if is_main_process() and step % 20 == 0:
                wandb.log({
                    'train/loss': loss.item(),
                    'train/grad_norm': grad_norm.item(),
                    'train/lr': scheduler.get_last_lr()[0],
                    'train/epoch': epoch,
                })

        # ── Validation (매 에포크) ──
        val_rec_error = validate(model, test_loader, diffusion, config, device)

        if is_main_process():
            avg_loss = epoch_loss / max(epoch_steps, 1)
            wandb.log({
                'val/reconstruction_error': val_rec_error,
                'val/epoch': epoch,
                'train/avg_loss': avg_loss,
            })

            # Best model 저장 (조기종료 없이, best만 추적)
            if val_rec_error < best_val_rec:
                best_val_rec = val_rec_error
                save_checkpoint(
                    model, optimizer, scheduler, epoch, best_val_rec, config,
                    path=f"{config['save_dir']}/best_model.pth"
                )
                print(f"[Epoch {epoch}] New best: val_rec={val_rec_error:.6f}")

            # 주기적 checkpoint
            if epoch % config.get('save_every', 20) == 0:
                save_checkpoint(
                    model, optimizer, scheduler, epoch, val_rec_error, config,
                    path=f"{config['save_dir']}/checkpoint_epoch{epoch}.pth"
                )

    cleanup_ddp()
```

---

## 4. bf16 상세 적용 가이드

### 어디에 bf16을 적용하는가

| 연산 | 정밀도 | 이유 |
|------|--------|------|
| Forward pass (Conv, Attn, FFN) | **bf16** | 속도 + 메모리 절약 |
| Loss 계산 | **bf16** (autocast 내) | forward와 동일 컨텍스트 |
| Backward pass | **bf16** (autocast 연장) | gradient도 bf16 |
| Optimizer step (Adam states) | **fp32** (자동) | Adam의 m, v는 fp32 유지 |
| 정규화 통계량 (mean, std) | **fp32** | 누적 연산 정밀도 필요 |
| 모델 저장/로드 | **fp32** | state_dict는 fp32 |

### bf16이 Diffusion 모델에서 안전한 이유

```
Diffusion 모델의 값 범위:
- 노이즈 ε ~ N(0,1) → 대부분 [-3, 3]
- 입력 x_t = √(ᾱ)·x_0 + √(1-ᾱ)·ε → 유사 범위
- 손실 MSE(pred_ε, ε) → 작은 양수

bf16 표현 범위: ±3.39 × 10^38 (fp32와 동일)
bf16 최소 정밀도: ~0.0078 (mantissa 7bit)

→ Diffusion의 값 범위에서 bf16은 충분한 정밀도 제공
→ LLM에서 검증된 bf16이 Diffusion에서도 동일하게 적용 가능
→ RTX A6000 (Ampere, CC 8.6)은 bf16 tensor core 가속 지원
```

---

## 5. 손실 함수

### 주 손실: Masked MSE

```python
def masked_mse_loss(pred, target, not_zero_mask):
    """
    zero_mask 위치를 제외한 MSE 손실
    not_zero_mask: (C, L) bool — True인 위치만 계산
    """
    mask = not_zero_mask.unsqueeze(0).expand_as(pred)  # (B, C, L)
    diff = (pred - target) ** 2
    return (diff * mask).sum() / mask.sum()
```

### 보조 손실: PCA 분포 매칭 (선택적)

```python
def mmd_rbf_loss(x_real, x_gen, sigma=1.0):
    """
    Maximum Mean Discrepancy with RBF kernel
    인구군별 PCA 분포가 일치하는지 측정
    """
    xx = torch.cdist(x_real, x_real)
    yy = torch.cdist(x_gen, x_gen)
    xy = torch.cdist(x_real, x_gen)
    k_xx = torch.exp(-xx ** 2 / (2 * sigma ** 2))
    k_yy = torch.exp(-yy ** 2 / (2 * sigma ** 2))
    k_xy = torch.exp(-xy ** 2 / (2 * sigma ** 2))
    return k_xx.mean() + k_yy.mean() - 2 * k_xy.mean()
```

---

## 6. 모델 저장 규격 (.pth)

### checkpoint 구조

```python
def save_checkpoint(model, optimizer, scheduler, epoch, val_loss, config, path):
    """DDP 호환 checkpoint 저장"""
    if not is_main_process():
        return

    # DDP wrapper 해제
    model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()

    checkpoint = {
        # 모델
        'model_state_dict': model_state,
        'config': config,

        # 학습 재개용
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch,
        'val_loss': val_loss,

        # 메타 정보
        'pytorch_version': torch.__version__,
        'cuda_version': torch.version.cuda,
    }

    torch.save(checkpoint, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, device='cuda:0'):
    """checkpoint 로드 (추론 또는 학습 재개)"""
    if not Path(path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return checkpoint.get('epoch', 0), checkpoint.get('val_loss', float('inf'))
```

### 파일 명명 규칙

```
outputs/
├── best_model.pth                    # 최종 추론 모델
├── checkpoint_epoch0.pth             # 주기적 checkpoint
├── checkpoint_epoch20.pth
├── checkpoint_epoch40.pth
└── config.yaml                       # 재현을 위한 설정 사본
```

---

## 7. 실행 방법

```bash
# 학습 (DDP, 2 GPU)
torchrun --nproc_per_node=2 src/train.py \
    --config configs/default.yaml \
    --save_dir outputs/run_001

# 단일 GPU 디버깅
python src/train.py \
    --config configs/default.yaml \
    --save_dir outputs/debug \
    --single_gpu

# wandb 오프라인
WANDB_MODE=offline torchrun --nproc_per_node=2 src/train.py ...
```
