# Phase 4: 추론 및 샘플 생성

- **입력**: `best_model.pth` (학습 산출물)
- **출력**: 인구군별 합성 유전형 `.pt` 파일 + `generation_meta.json`
- **의존 문서**: `02_preprocessing` (산출물 스키마), `04_training` (checkpoint 구조), `09_performance_optimization` §1.1 (EMA)

---

## 0. best_model.pth 내부 필드 (확정)

```python
{
    'model_state_dict': OrderedDict,    # 모델 파라미터 (DDP .module 해제 후 저장)
    'ema_state_dict': dict,             # EMA shadow 파라미터 (추론 시 이것을 사용)
    'optimizer_state_dict': OrderedDict, # 재학습용 (추론에서는 미사용)
    'scheduler_state_dict': dict,       # 재학습용
    'epoch': int,                       # 저장 시점 epoch
    'best_val_loss': float,             # best validation reconstruction error
    'config': dict,                     # 학습 시 사용된 전체 config (nested YAML 구조)
}
```

> **EMA 사용 규칙**: 추론 시 반드시 `ema_state_dict`를 모델에 로드한다. `model_state_dict`는 재학습/디버깅용이다.

## 0.5 생성 운영 규칙

| 항목 | 규칙 |
|------|------|
| 기본 생성 수 | `n_samples_per_pop=None`이면 real 데이터와 동일 수 생성 |
| 과생성 실험 | `oversample_minority=N`이면 각 인구군 최소 N개 생성 |
| generation seed | 기본 `20260327` |
| 파일 저장 dtype | float32 (`.pt`) |
| 내부 저장 범위 | run 디렉토리 내부에만 저장, 외부 공유 금지 |
| 외부 공유 기본값 | `generation_meta.json` + aggregate evaluation만 공유 |

> 합성 샘플 자체는 내부 검증용 산출물로 취급한다. 외부 공유가 필요할 경우 `10_experiment_operations`의 배포 정책을 따른다.

---

## 1. 추론 파이프라인

```
best_model.pth
     │
     ▼
[Step 1] 모델 로드 (.pth → model.eval())
     │
[Step 2] 인구군별 조건부 샘플 생성
     │     xT ~ N(0,1)
     │     for t = T, T-1, ..., 0:
     │         ε_pred = model(x_t, t, y=pop_label)    # bf16 추론
     │         x_{t-1} = reverse_step(x_t, ε_pred)
     │
[Step 3] 후처리
     │     enforce_zeros (zero_mask 적용)
     │     역정규화 (mean, std 복원)
     │
[Step 4] 저장
     │     인구군별 .pt 파일
     └→   sample_{pop}_{idx}.pt
```

---

## 2. 추론 코드

### 파일: `src/inference/generator.py`

```python
@torch.no_grad()
def generate_samples(config, model_path: str, output_dir: str,
                     n_samples_per_pop: dict = None,
                     oversample_minority: int = None,
                     seed: int = 20260327):
    """
    인구군별 합성 유전형 생성

    Args:
        config: 모델 설정 (nested dict, configs/default.yaml 구조)
        model_path: best_model.pth 경로
        output_dir: 샘플 저장 디렉토리
        n_samples_per_pop: {pop_idx: n_samples} 딕셔너리
            None이면 실제 데이터와 동일 수 생성
        oversample_minority: 소수 인구군 최소 생성 수 (지정 시 해당 수까지 과생성)
        seed: generation seed
    """
    device = torch.device('cuda:0')
    t_start = time.time()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # ── Step 1: 모델 로드 (EMA 파라미터 사용) ──
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = HybridCNNDiTFiLM(config).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # EMA 파라미터 로드 (추론 시 반드시 EMA 사용)
    if 'ema_state_dict' in checkpoint:
        for name, param in model.named_parameters():
            if name in checkpoint['ema_state_dict']:
                param.data.copy_(checkpoint['ema_state_dict'][name])
        print(f"EMA parameters loaded from {model_path}")
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"WARNING: EMA not found, using raw model weights")
    model.eval()

    print(f"Model loaded from {model_path} (epoch {checkpoint.get('epoch', '?')})")

    # ── Diffusion 프로세스 ──
    zero_mask = torch.load(config['data']['zero_mask_path'], map_location=device)
    diffusion = GaussianDiffusion(
        timesteps=config['diffusion']['max_timesteps'],
        device=device,
        zero_mask=zero_mask,
        enforce_zeros=config['data']['enforce_zeros'],
    )

    # ── Step 2: 인구군별 생성 ──
    os.makedirs(output_dir, exist_ok=True)

    if n_samples_per_pop is None:
        label_info = pickle.load(open(config['data']['label_hierarchy_path'], 'rb'))
        n_samples_per_pop = {i: n for i, (pop, n) in
                             enumerate(sorted(label_info['pop_sizes'].items()))}

    # oversample_minority: 소수 인구군을 최소 N개까지 과생성
    if oversample_minority is not None:
        n_samples_per_pop = {
            k: max(v, oversample_minority)
            for k, v in n_samples_per_pop.items()
        }

    total_generated = 0

    for pop_idx, n_samples in n_samples_per_pop.items():
        print(f"Generating {n_samples} samples for population {pop_idx}...")

        for i in range(n_samples):
            # 초기 노이즈
            xT = torch.randn(1, config['data']['num_channels'], config['data']['gene_size']).to(device)
            y = torch.tensor([pop_idx], device=device)

            # bf16 추론
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                sample = diffusion.sample_from_reverse_process(
                    model=model,
                    xT=xT,
                    timesteps=config['diffusion'].get('sampling_timesteps', None),
                    y=y,
                    guidance=config['diffusion'].get('guidance_type', 'normal'),
                    w=config['diffusion'].get('guidance_weight', 3.0),
                )

            # fp32로 변환하여 저장
            sample = sample.float().cpu()
            label = y.cpu()

            # 저장
            save_path = os.path.join(output_dir, f"sample_pop{pop_idx}_{i:04d}.pt")
            torch.save((sample.squeeze(0), label.squeeze(0)), save_path)

            total_generated += 1

        print(f"  Population {pop_idx}: {n_samples} samples saved")

    # ── Step 4: generation_meta.json 저장 ──
    import json
    from datetime import datetime

    meta = {
        "model_path": str(model_path),
        "model_epoch": checkpoint.get('epoch', None),
        "used_ema": 'ema_state_dict' in checkpoint,
        "total_samples": total_generated,
        "seed": seed,
        "oversample_minority": oversample_minority,
        "per_population": {str(k): v for k, v in n_samples_per_pop.items()},
        "generation_time_sec": time.time() - t_start,
        "config": {
            "max_timesteps": config['diffusion']['max_timesteps'],
            "guidance_type": config['diffusion'].get('guidance_type', 'normal'),
            "guidance_weight": config['diffusion'].get('guidance_weight', 3.0),
            "num_channels": config['data']['num_channels'],
            "gene_size": config['data']['gene_size'],
        },
        "timestamp": datetime.now().isoformat(),
    }
    meta_path = os.path.join(output_dir, "generation_meta.json")
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"Total: {total_generated} samples generated in {output_dir}")
    print(f"Metadata saved to {meta_path}")
```

---

## 3. 후처리: 역정규화

```python
def denormalize_samples(samples, stats_path):
    """
    생성된 샘플을 원래 스케일로 복원

    Args:
        samples: (B, K, gene_size) float32 — 모델 출력 (정규화된 상태)
        stats_path: normalization_stats.pkl 경로

    Returns:
        (B, K, gene_size) float32 — 역정규화된 텐서

    Broadcasting 규칙:
        stats['mean'] / stats['std']는 (gene_size, K) shape로 저장됨.
        모델 출력은 (B, K, gene_size) shape이므로, 전치가 필요:
        mean.T → (K, gene_size) → unsqueeze(0) → (1, K, gene_size) → broadcast with B
    """
    stats = pickle.load(open(stats_path, 'rb'))
    # (gene_size, K) → (K, gene_size) → (1, K, gene_size)
    xmean = torch.tensor(stats['mean'].T).unsqueeze(0)  # (1, K, gene_size)
    xstd = torch.tensor(stats['std'].T).unsqueeze(0)    # (1, K, gene_size)
    return samples * xstd + xmean
```

---

## 4. 실행 방법

```bash
# 기본 생성 (실제 데이터와 동일 수)
python src/inference/generator.py \
    --config configs/default.yaml \
    --model_path outputs/run_001/best_model.pth \
    --output_dir outputs/run_001/synthetic_samples

# 소수 인구군 과생성 (최소 200개까지)
python src/inference/generator.py \
    --config configs/default.yaml \
    --model_path outputs/run_001/best_model.pth \
    --output_dir outputs/run_001/oversampled \
    --oversample_minority 200
```

---

## 5. 출력 파일 형식

```
outputs/run_001/synthetic_samples/
├── sample_pop0_0000.pt    # (genome_tensor, label_tensor)
├── sample_pop0_0001.pt    #   genome: (8, 26624) float32
├── ...                    #   label: scalar int64
├── sample_pop25_0060.pt
└── generation_meta.json   # 생성 메타 정보 + seed + 생성 규칙
```

### generation_meta.json

```json
{
    "model_path": "outputs/run_001/best_model.pth",
    "model_epoch": 85,
    "used_ema": true,
    "total_samples": 2504,
    "seed": 20260327,
    "oversample_minority": null,
    "per_population": {"0": 96, "1": 61, "...": "..."},
    "config": {"max_timesteps": 500, "guidance_type": "normal", "...": "..."},
    "generation_time_sec": 1823,
    "timestamp": "2026-03-28T10:30:00"
}
```

---

## 6. 생성 산출물 사용 원칙

- `sample_pop*.pt`는 내부 평가와 증강 실험용으로만 사용한다.
- 외부 공유 기본 단위는 `evaluation/*.json`, figure, aggregate table이다.
- 개별 샘플 export가 필요하면 privacy 평가(NNAA, DUPI, membership AUC) 완료 후 별도 승인 절차를 거친다.
