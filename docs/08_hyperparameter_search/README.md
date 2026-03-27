# Phase 3.5: 하이퍼파라미터 랜덤 서치 및 추적

---

## 1. 서치 전략

```
방법:       Random Search (wandb Sweep 연동)
총 시행:    200회
GPU 할당:   시행당 DDP 2GPU (순차 실행) 또는 시행당 1GPU (2 시행 병렬)
선정 기준:  val_reconstruction_error (primary)
2차 기준:   recovery_rate (합성 학습 → 실제 테스트 정확도)
```

### 왜 Random Search인가

- Grid Search: 51개 파라미터 → 조합 폭발, 비현실적
- Bayesian: 순차 의존성 → DDP 2GPU 환경에서 병렬화 이점 없음
- Random Search: 고차원 공간에서 Grid 대비 효율적 (Bergstra & Bengio, 2012)

---

## 2. 탐색 공간 (10개 카테고리, 51개 파라미터)

### 2.1 Training (11개)

| 파라미터 | 유형 | 범위 | 기본값 |
|----------|------|------|--------|
| `lr_diffusion` | log_uniform | [1e-6, 1e-2] | 2e-4 |
| `batch_size` | categorical | {8, 16, 32, 64, 128} | 32 |
| `epochs` | int_uniform | [50, 500] | 100 |
| `gradient_accumulation_steps` | categorical | {1, 2, 4, 8} | 1 |
| `optimizer` | categorical | {adam, adamw, sgd, radam, lamb} | adamw |
| `weight_decay` | log_uniform | [1e-8, 1e-1] | 1e-4 |
| `gradient_clipping` | uniform | [0.1, 10.0] | 1.0 |
| `ema_decay` | uniform | [0.9, 0.9999] | 0.999 |
| `lr_scheduler` | categorical | {cosine_warmup, linear_warmup, constant, cosine_annealing, one_cycle} | cosine_warmup |
| `warmup_ratio` | uniform | [0.01, 0.2] | 0.05 |
| `lr_min_ratio` | log_uniform | [1e-4, 0.5] | 0.01 |

### 2.2 Diffusion Process (7개)

| 파라미터 | 유형 | 범위 | 기본값 |
|----------|------|------|--------|
| `max_timesteps` | categorical | {200, 300, 500, 750, 1000, 1500} | 500 |
| `noise_schedule` | categorical | {cosine, linear, sigmoid, sqrt} | cosine |
| `prediction_target` | categorical | {epsilon, x0, v_prediction} | epsilon |
| `sampling_timesteps` | categorical | {50, 100, 200, 500} | 500 |
| `guidance_type` | categorical | {normal, classifier_free} | normal |
| `guidance_weight` | uniform | [0.0, 10.0] | 3.0 |
| `cfg_dropout` | uniform | [0.0, 0.5] | 0.1 |

### 2.3 CNN Stem (9개)

| 파라미터 | 유형 | 범위 | 기본값 |
|----------|------|------|--------|
| `cnn_base_channels` | categorical | {32, 48, 64, 96, 128, 192, 256} | 64 |
| `cnn_n_blocks` | int_uniform | [2, 6] | 4 |
| `cnn_channel_multiplier` | categorical | {(1,1,2,4), (1,2,2,4), (1,2,4,4), (1,2,4,8), ...} | (1,1,2,4) |
| `cnn_kernel_size` | categorical | {3, 5, 7, 9} | 3 |
| `cnn_norm_type` | categorical | {group_norm, layer_norm, instance_norm, batch_norm} | group_norm |
| `cnn_activation` | categorical | {silu, gelu, relu, mish} | silu |
| `cnn_dropout` | uniform | [0.0, 0.5] | 0.0 |
| `cnn_use_residual` | categorical | {true, false} | true |
| `cnn_downsample_mode` | categorical | {conv_stride2, avg_pool, max_pool} | conv_stride2 |

### 2.4 DiT Core (9개)

| 파라미터 | 유형 | 범위 | 기본값 |
|----------|------|------|--------|
| `dit_d_model` | categorical | {128, 192, 256, 384, 512, 768} | 256 |
| `dit_n_blocks` | int_uniform | [2, 12] | 4 |
| `dit_n_heads` | categorical | {2, 4, 6, 8, 12, 16} | 4 |
| `dit_mlp_ratio` | uniform | [1.0, 8.0] | 4.0 |
| `dit_dropout` | uniform | [0.0, 0.5] | 0.0 |
| `dit_attention_dropout` | uniform | [0.0, 0.3] | 0.0 |
| `dit_use_flash_attention` | categorical | {true, false} | false |
| `patch_size` | categorical | {4, 8, 16, 32, 64, 128} | 16 |
| `pos_embedding_type` | categorical | {learned, sinusoidal, rotary, alibi, none} | learned |

### 2.5 FiLM / AdaLN (10개)

| 파라미터 | 유형 | 범위 | 기본값 |
|----------|------|------|--------|
| `film_type` | categorical | {adaln_zero, film_simple, adaln, scale_only, bias_only} | adaln_zero |
| `pop_emb_dim` | categorical | {64, 128, 256, 384, 512} | 256 |
| `superpop_emb_dim` | categorical | {32, 64, 128, 256} | 256 |
| `hierarchy_fusion` | categorical | {concat_mlp, add, gate, cross_attention, film_on_film} | concat_mlp |
| `timestep_emb_dim` | categorical | {128, 256, 384, 512} | 256 |
| `film_hidden_layers` | int_uniform | [1, 4] | 2 |
| `film_hidden_dim` | categorical | {128, 256, 512, 768, 1024} | 256 |
| `film_activation` | categorical | {silu, gelu, relu, tanh} | silu |
| `film_cnn_enabled` | categorical | {true, false} | true |
| `film_shared_generator` | categorical | {true, false} | true |

### 2.6 Balance / Aux / Regularization / Data (15개)

| 파라미터 | 유형 | 범위 | 기본값 |
|----------|------|------|--------|
| `balance_mode` | categorical | {learned_scalar, learned_time_dependent, learned_pop_dependent, residual_add, gated_residual, dit_only, concat_proj} | learned_time_dependent |
| `aux_loss_enabled` | categorical | {true, false} | false |
| `lambda_pca_dist` | log_uniform | [1e-5, 10.0] | 0.01 |
| `lambda_pop_structure` | log_uniform | [1e-5, 10.0] | 0.01 |
| `aux_loss_type` | categorical | {mmd_rbf, mmd_linear, sliced_wasserstein, kl_divergence, cosine_distance} | mmd_rbf |
| `aux_loss_warmup_epochs` | int_uniform | [0, 100] | 20 |
| `mmd_kernel_bandwidth` | log_uniform | [0.01, 100.0] | 1.0 |
| `dropout_global` | uniform | [0.0, 0.5] | 0.0 |
| `label_smoothing` | uniform | [0.0, 0.3] | 0.0 |
| `stochastic_depth_rate` | uniform | [0.0, 0.3] | 0.0 |
| `input_noise_augmentation` | uniform | [0.0, 0.1] | 0.0 |
| `enforce_zeros` | categorical | {true, false} | true |
| `normalize_data` | categorical | {true, false} | true |
| `use_ddim` | categorical | {true, false} | false |
| `ddim_eta` | uniform | [0.0, 1.0] | 0.0 |

**`pca_components`에 대한 참고**: PCA 성분 수(K)는 HP 서치가 아닌 **전처리 단계의 그리드 서치**에서 결정된다 (docs/02_preprocessing 참조). 후보 [4, 6, 8, 10, 12, 16]에서 평균 explained variance ≥ 90%인 최소 K를 자동 선택하며, 이 값이 `num_channels`로 모델 전체에 전파된다. HP 서치에서는 이미 결정된 K를 고정값으로 사용한다.|

---

## 3. 제약 조건 (자동 적용)

```python
CONSTRAINTS = {
    # 1. 어텐션 차원 호환
    "dit_d_model % dit_n_heads == 0",

    # 2. CNN 블록 수 == channel_multiplier 길이
    "len(cnn_channel_multiplier) == cnn_n_blocks",

    # 3. CFG 비활성 시 관련 파라미터 무시
    "guidance_type != 'classifier_free' → guidance_weight=0, cfg_dropout=0",

    # 4. 보조 손실 비활성 시 lambda=0
    "aux_loss_enabled == false → lambda_pca_dist=0, lambda_pop_structure=0",

    # 5. 실효 배치 ≤ 256
    "batch_size * gradient_accumulation_steps <= 256",

    # 6. 패치 크기 호환
    "seq_after_cnn % patch_size == 0",

    # 7. 모델 크기 상한 (오버피팅 방지)
    "estimated_params < 50M",
}
```

---

## 4. wandb Sweep 연동

### 파일: `configs/sweep.yaml`

```yaml
program: src/training/trainer.py
method: random
metric:
  name: val/reconstruction_error
  goal: minimize

parameters:
  lr_diffusion:
    distribution: log_uniform_values
    min: 1e-6
    max: 1e-2
  batch_size:
    values: [8, 16, 32, 64, 128]
  epochs:
    distribution: int_uniform
    min: 50
    max: 500
  optimizer:
    values: ["adam", "adamw", "radam", "lamb"]
  weight_decay:
    distribution: log_uniform_values
    min: 1e-8
    max: 1e-1
  gradient_clipping:
    distribution: uniform
    min: 0.1
    max: 10.0
  max_timesteps:
    values: [200, 300, 500, 750, 1000, 1500]
  noise_schedule:
    values: ["cosine", "linear", "sigmoid", "sqrt"]
  cnn_base_channels:
    values: [32, 48, 64, 96, 128, 192, 256]
  dit_d_model:
    values: [128, 192, 256, 384, 512, 768]
  dit_n_blocks:
    distribution: int_uniform
    min: 2
    max: 12
  dit_n_heads:
    values: [2, 4, 6, 8, 12, 16]
  dit_mlp_ratio:
    distribution: uniform
    min: 1.0
    max: 8.0
  patch_size:
    values: [4, 8, 16, 32, 64, 128]
  film_type:
    values: ["adaln_zero", "film_simple", "adaln", "scale_only", "bias_only"]
  pop_emb_dim:
    values: [64, 128, 256, 384, 512]
  hierarchy_fusion:
    values: ["concat_mlp", "add", "gate", "cross_attention"]
  balance_mode:
    values: ["learned_scalar", "learned_time_dependent", "learned_pop_dependent", "residual_add", "gated_residual"]
  # ... (전체 파라미터는 sweep_full.yaml에서)

run_cap: 200
```

### 실행

```bash
# Sweep 생성
wandb sweep configs/sweep.yaml --project HybridGenoDiT

# Agent 실행 (각 GPU 서버에서)
wandb agent <sweep_id>
```

---

## 5. 추적 지표 (wandb 로깅)

### 5.1 학습 중 실시간 추적 (매 step)

| 지표 | wandb key | 설명 |
|------|-----------|------|
| 학습 손실 | `train/loss` | MSE(pred_ε, ε) |
| Gradient norm | `train/grad_norm` | clip 전 norm |
| Learning rate | `train/lr` | 현재 lr |
| CNN-DiT 밸런스 | `train/balance_weight` | DiT 기여 비율 |
| 시간/step | `train/step_time_sec` | 벽시계 시간 |

### 5.2 Validation (매 epoch)

| 지표 | wandb key | 설명 |
|------|-----------|------|
| Val 손실 | `val/loss` | 검증 MSE |
| 복원 오차 | `val/reconstruction_error` | **primary 선정 기준** |
| 타임스텝별 오차 | `val/rec_error_histogram` | wandb.Histogram |

### 5.3 모델 메타 (시행 시작 시 1회)

| 지표 | wandb key | 설명 |
|------|-----------|------|
| 파라미터 수 | `model/total_params` | 전체 파라미터 수 |
| CNN 파라미터 | `model/cnn_params` | CNN 모듈 파라미터 |
| DiT 파라미터 | `model/dit_params` | DiT 모듈 파라미터 |
| FiLM 파라미터 | `model/film_params` | FiLM 생성기 파라미터 |
| 토큰 수 | `model/n_tokens` | DiT 입력 토큰 수 |
| VRAM 사용 | `model/vram_mb` | 피크 VRAM |

### 5.4 생성 품질 (best 모델 선정 후)

| 지표 | wandb key | 설명 |
|------|-----------|------|
| AF 상관 | `eval/af_correlation` | 대립유전자 빈도 상관 |
| LD 상관 | `eval/ld_correlation` | LD 감쇠 상관 |
| PCA Wasserstein | `eval/pca_wasserstein` | PCA 공간 분포 거리 |
| Recovery Rate (MLP) | `eval/recovery_rate_mlp` | **secondary 선정 기준** |
| Recovery Rate (CNN) | `eval/recovery_rate_cnn` | |
| NNAA | `eval/nnaa` | 프라이버시 |
| 크기-품질 상관 | `eval/size_quality_corr` | **강건성 핵심 지표** |
| 최소 인구군 AF 상관 | `eval/worst_pop_af_corr` | 최악 인구군 품질 |

---

## 6. 로깅 구현

### 파일: `src/utils/logging_utils.py`

```python
import wandb
import torch
from src.utils.distributed import is_main_process

class ExperimentLogger:
    """wandb + 콘솔 통합 로거 (rank 0만 로깅)"""

    def __init__(self, config, project="HybridGenoDiT"):
        self.enabled = is_main_process()
        if self.enabled:
            wandb.init(project=project, config=config)

    def log_step(self, metrics: dict, step: int = None):
        """학습 step 지표 로깅"""
        if not self.enabled:
            return
        wandb.log(metrics, step=step)

    def log_model_info(self, model):
        """모델 메타 정보 1회 로깅"""
        if not self.enabled:
            return

        # DDP wrapper 해제
        m = model.module if hasattr(model, 'module') else model

        total = sum(p.numel() for p in m.parameters())
        trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)

        # 모듈별 파라미터 수
        module_params = {}
        for name, module in m.named_children():
            n = sum(p.numel() for p in module.parameters())
            module_params[f"model/{name}_params"] = n

        wandb.log({
            "model/total_params": total,
            "model/trainable_params": trainable,
            **module_params,
        })

        # VRAM 추적
        if torch.cuda.is_available():
            vram = torch.cuda.max_memory_allocated() / 1e6
            wandb.log({"model/peak_vram_mb": vram})

    def log_hparams_summary(self, config):
        """하이퍼파라미터 요약 테이블 로깅"""
        if not self.enabled:
            return

        # 파생 값 추가
        config_with_derived = dict(config)
        cnn_ch = config.get('cnn_base_channels', 64)
        mult = config.get('cnn_channel_multiplier', (1,1,2,4))
        downsample_stages = len(mult) - 1
        seq_after_cnn = config.get('gene_size', 26624) // (2 ** downsample_stages)
        patch_size = config.get('patch_size', 16)
        n_tokens = seq_after_cnn // patch_size if seq_after_cnn % patch_size == 0 else -1

        config_with_derived['_seq_after_cnn'] = seq_after_cnn
        config_with_derived['_n_tokens'] = n_tokens
        config_with_derived['_cnn_output_channels'] = cnn_ch * mult[-1]
        config_with_derived['_effective_batch'] = (
            config.get('batch_size', 32) * config.get('gradient_accumulation_steps', 1)
        )

        wandb.config.update(config_with_derived, allow_val_change=True)

    def log_eval_results(self, results: dict):
        """평가 결과 로깅"""
        if not self.enabled:
            return

        flat = {}
        for category, metrics in results.items():
            if isinstance(metrics, dict):
                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                        flat[f"eval/{category}_{k}"] = v
            elif isinstance(metrics, (int, float)):
                flat[f"eval/{category}"] = metrics

        wandb.log(flat)

        # 인구군별 품질 테이블
        if 'robustness' in results and 'per_pop_quality' in results['robustness']:
            pop_data = results['robustness']['per_pop_quality']
            table = wandb.Table(columns=["pop_idx", "n_samples", "af_correlation"])
            for pop_idx, info in sorted(pop_data.items()):
                table.add_data(pop_idx, info['n_samples'], info['af_correlation'])
            wandb.log({"eval/per_pop_quality_table": table})

    def finish(self):
        if self.enabled:
            wandb.finish()
```

---

## 7. 서치 결과 분석

### 상위 시행 선별 기준

```python
def select_top_trials(sweep_id, n_top=10):
    """wandb sweep에서 상위 N개 시행 선별"""
    api = wandb.Api()
    sweep = api.sweep(f"<entity>/<project>/{sweep_id}")

    runs = sorted(
        [r for r in sweep.runs if r.state == "finished"],
        key=lambda r: r.summary.get("val/reconstruction_error", float('inf'))
    )

    top_runs = runs[:n_top]

    print(f"\n{'='*80}")
    print(f"Top {n_top} trials (by val/reconstruction_error)")
    print(f"{'='*80}")

    for i, run in enumerate(top_runs):
        s = run.summary
        c = run.config
        print(f"\n#{i+1} [{run.name}]")
        print(f"  val_rec_error: {s.get('val/reconstruction_error', '?'):.6f}")
        print(f"  lr={c.get('lr_diffusion', '?'):.2e}  "
              f"bs={c.get('batch_size', '?')}  "
              f"epochs={c.get('epochs', '?')}")
        print(f"  CNN: C={c.get('cnn_base_channels', '?')}  "
              f"DiT: d={c.get('dit_d_model', '?')} blocks={c.get('dit_n_blocks', '?')}")
        print(f"  FiLM: {c.get('film_type', '?')}  "
              f"fusion={c.get('hierarchy_fusion', '?')}")
        print(f"  params: {s.get('model/total_params', 0)/1e6:.1f}M")

    return top_runs
```

### 하이퍼파라미터 중요도 분석

```python
def analyze_hparam_importance(sweep_id):
    """파라미터별 중요도 분석 (wandb 내장 기능 + 수동 상관 분석)"""
    api = wandb.Api()
    sweep = api.sweep(f"<entity>/<project>/{sweep_id}")

    records = []
    for run in sweep.runs:
        if run.state != "finished":
            continue
        rec = dict(run.config)
        rec['val_rec_error'] = run.summary.get('val/reconstruction_error', None)
        records.append(rec)

    df = pd.DataFrame(records).dropna(subset=['val_rec_error'])

    # 수치형 파라미터와 val_rec_error 간 상관
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlations = {}
    for col in numeric_cols:
        if col == 'val_rec_error':
            continue
        corr = df[col].corr(df['val_rec_error'])
        correlations[col] = corr

    # 상관 절댓값 기준 정렬
    importance = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

    print("\nHyperparameter Importance (|correlation| with val_rec_error):")
    for name, corr in importance[:15]:
        direction = "↑ better" if corr < 0 else "↓ better"
        print(f"  {name:35s}  r={corr:+.3f}  ({direction})")

    return importance
```

---

## 8. 실행 플로우

```
[1] wandb sweep 생성
    wandb sweep configs/sweep.yaml --project HybridGenoDiT
         │
[2] 200회 랜덤 서치 실행 (DDP)
    wandb agent <sweep_id>
         │
    매 시행:
    ├→ config 샘플링 (제약 조건 자동 적용)
    ├→ 모델 생성 + shape 검증
    ├→ DDP 학습 (bf16, 조기종료 없음)
    ├→ wandb에 실시간 로깅
    └→ best_model.pth 저장
         │
[3] 상위 10개 선별
    python scripts/select_top_trials.py --sweep_id <id> --n_top 10
         │
[4] 상위 시행으로 full 평가
    for run in top_10:
    ├→ 샘플 생성 (inference)
    ├→ 전체 평가 (6개 카테고리, 병렬)
    └→ wandb에 eval 결과 로깅
         │
[5] 하이퍼파라미터 중요도 분석
    python scripts/analyze_hparam_importance.py --sweep_id <id>
         │
[6] 최종 모델 선정 + 논문 보고
```
