# HybridGenoDiT 모델 구축 기획서

**프로젝트명**: Population-Robust Synthetic Genotype Generation via Hybrid CNN-DiT with Hierarchical FiLM
**작성일**: 2026-03-27
**환경**: NVIDIA RTX A6000 x2 (49GB VRAM each), Compute Capability 8.6

---

## 문서 구조

```
docs/
├── 01_overview/              ← 현재 문서 (전체 개요, 원칙, 일정)
├── 02_preprocessing/         ← 데이터 전처리 파이프라인 (병렬화)
├── 03_model/                 ← 모델 아키텍처 상세 설계
├── 04_training/              ← 학습 파이프라인 (DDP, bf16, 손실 함수)
├── 05_inference/             ← 추론 및 샘플 생성 (.pth)
├── 06_evaluation/            ← 평가 지표 및 실험 설계
├── 07_project_structure/     ← 프로젝트 디렉토리/파일 구조
├── 08_hyperparameter_search/ ← 하이퍼파라미터 랜덤 서치 및 추적 (wandb Sweep)
├── 09_performance_optimization/ ← 성능 최적화 전략 (EMA, CFG, 증강, 샘플러 등)
└── 10_experiment_operations/ ← 재현성, run 규칙, checkpoint/배포 정책
```

---

## 1차 구현 범위 (Scope)

> **본 프로젝트의 1차 구현은 Gene PCA 표현 기반 파이프라인으로 한정한다.**

| 항목 | 1차 구현 범위 (현재) | 후속 확장 (별도 계획) |
|------|---------------------|---------------------|
| **입력 표현** | Gene PCA 텐서 `(B, K, gene_size)` | 원시 SNP / 하플로타입 |
| **손실 함수** | Masked MSE + MMD (PCA 공간) | AF 분포 손실, LD 감쇠 손실 |
| **평가 지표** | PCA 공간 분포 거리, Recovery Rate, NNAA, DUPI | SNP 단위 AF, MAF 저빈도, LD 감쇠, haplotype diversity |
| **최적화** | EMA, DDIM, CFG, Min-SNR, 오버샘플링 | 원시 SNP 전제 전략 (Latent Diffusion 등) |
| **모델** | Hybrid CNN-DiT + FiLM (Gene PCA 입력) | 원시 유전형 직접 입력 아키텍처 |

이 범위를 벗어나는 원시 SNP 기반 기법은 각 문서에서 **"후속 확장"**으로 별도 표기한다.

### Canonical Config Schema

설정 파일의 단일 기준(source of truth)은 `07_project_structure/README.md` §3에 정의된 `configs/default.yaml` 스키마이다. 모든 문서의 코드 예시는 **nested YAML 접근** 방식을 따른다.

```yaml
# 최소 합의 키 (모든 모듈이 참조하는 필수 키)
data.gene_size          # 26624 (패딩 포함 유전자 토큰 수)
data.num_channels       # K (PCA 그리드 서치로 자동 결정)
data.zero_mask_path     # "data/processed/zero_mask.pt"
data.label_hierarchy_path  # "data/processed/label_hierarchy.pkl"
diffusion.max_timesteps # 500
training.batch_size     # 32
save_dir                # "outputs/default"
```

Config 로더(`src/utils/config.py`)는 YAML nested 구조를 그대로 사용하며, flat key 변환은 수행하지 않는다. 코드에서는 `config['data']['gene_size']` 형태로 접근한다.

---

## 문서 간 Source of Truth

| 주제 | 기준 문서 | 설명 |
|------|----------|------|
| 전체 범위 및 원칙 | `01_overview` (본 문서) | 구현 범위, 핵심 원칙 |
| 전처리 산출물 스키마 | `02_preprocessing` | pkl/pt 파일 내부 구조, K 선택 규칙 |
| 모델 아키텍처 | `03_model` | 모듈 구조, shape 흐름 |
| 학습 설정 | `04_training` | 학습 루프, 손실 함수, checkpoint 구조 |
| 추론 I/O | `05_inference` | 입출력 계약, 후처리 |
| 평가 지표 정의 | `06_evaluation` | 지표 수식, 목표값 |
| Config 스키마 | `07_project_structure` §3 | canonical default.yaml |
| HP 탐색 공간 | `08_hyperparameter_search` | sweep 파라미터 |
| 최적화 전략 | `09_performance_optimization` | 우선순위별 전략 |
| 실험 운영/재현성 | `10_experiment_operations` | seed, run naming, resume, retention, 외부 공유 정책 |

---

## 재현성 및 운영 기준

구현 문서 외에도 실험 운영 계약을 고정한다. 상세 규칙은 `10_experiment_operations` 문서를 기준으로 한다.

| 항목 | 기본값 |
|------|--------|
| 전처리 seed | `20260327` |
| 학습 seed | `20260327`, `20260328`, `20260329` (최종 비교 시 3회 반복 권장) |
| split 기준 | population stratified split, test ratio 0.1 |
| split 재사용 | 한 번 생성한 `split_manifest.json`을 이후 실험에서 고정 사용 |
| run naming | `YYYYMMDD_<phase>_<tag>_<seed>` |
| checkpoint 보존 | `best_model.pth` + 주기 checkpoint 최근 3개 |
| 외부 공유 기본값 | 원본/개별 합성 샘플 비공개, aggregate 결과만 공유 |

### 보류 결정 사항

아래 항목은 1차 구현 범위에서 의도적으로 보류한다.

- PCA 역변환 기반 SNP 수준 평가
- 원시 SNP/haplotype 직접 입력 모델
- 외부 공개용 synthetic cohort 배포 정책
- 임상/민감 표현형과의 결합 실험

---

## 핵심 원칙 (모든 구현에 적용)

### P1. 병렬 처리 (Parallel Processing)

모든 전처리 및 시간 소요 작업은 병렬 처리를 필수로 적용한다.

| 작업 | 병렬화 방법 |
|------|-----------|
| VCF 파싱 (22 염색체) | `multiprocessing.Pool(22)` |
| Gene-level PCA | `joblib.Parallel` 또는 `concurrent.futures` |
| 평가 지표 계산 | 지표별 병렬 실행 |
| 하이퍼파라미터 서치 | 시행별 GPU 분배 |

### P2. 조기종료 미적용

학습 시 early stopping을 적용하지 않는다. 설정된 epoch를 끝까지 돌리고
최적 모델은 val_reconstruction_error 기준 best checkpoint으로 저장한다.

```python
# 조기종료 없이 best model 추적
if val_rec_error < best_rec_error:
    best_rec_error = val_rec_error
    torch.save(model.state_dict(), f"{save_dir}/best_model.pth")
```

### P3. bf16 부동소수점

**결론: bf16은 이 합성 모델에 적용 가능하다.**

RTX A6000 (Ampere, CC 8.6)은 bf16 하드웨어 가속을 지원한다.
bf16은 fp16 대비 동적 범위(exponent 8bit)가 넓어 gradient underflow 위험이 낮으므로
Diffusion 모델에서 안정적이다.

```
bf16 vs fp16 비교:
- fp16: exponent 5bit, mantissa 10bit → 동적 범위 좁음, GradScaler 필요
- bf16: exponent 8bit, mantissa 7bit → fp32와 동일한 동적 범위, GradScaler 불필요
```

적용 방식:
```python
# torch.autocast로 forward/backward를 bf16으로 실행
with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    pred_eps = model(xt, t, y=labels)
    loss = criterion(pred_eps, eps)

# bf16에서는 GradScaler 불필요 → 그냥 loss.backward() + optimizer.step()
loss.backward()
optimizer.step()
```

정규화(Standardization)도 bf16 정밀도에서 수행:
```python
# 데이터 정규화 시 bf16 범위 내에서 안정적
xmean = x.mean(axis=0).astype(np.float32)  # 통계량은 fp32로 계산
xstd = x.std(axis=0).astype(np.float32)
xstd[xstd == 0.0] += 1
# 실제 텐서는 bf16으로 캐스팅
x_normalized = ((x - xmean) / xstd)  # 학습 시 bf16 autocast 적용
```

### P4. DDP (Distributed Data Parallel)

2장의 RTX A6000을 모두 활용하는 DDP를 적용한다.

```python
# 실행 방법
torchrun --nproc_per_node=2 train.py

# 코드 내 DDP 구성
torch.distributed.init_process_group(backend='nccl')
local_rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(local_rank)

model = HybridCNNDiTFiLM(...).to(local_rank)
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

sampler = DistributedSampler(dataset, shuffle=True)
dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
```

### P5. GradScaler 불필요

**bf16 사용 시 GradScaler는 필요 없다.**

GradScaler는 fp16의 좁은 동적 범위에서 gradient underflow를 방지하기 위한 도구이다.
bf16은 fp32와 동일한 exponent 범위(8bit)를 가지므로 underflow 위험이 거의 없다.

```python
# fp16이면 필요:
# scaler = GradScaler()
# scaler.scale(loss).backward()
# scaler.step(optimizer)
# scaler.update()

# bf16이면 불필요:
loss.backward()
optimizer.step()
```

### P6. 도메인 주도 설계 (Domain-Driven)

유전학 도메인 지식을 모델 구조에 직접 반영한다:

| 도메인 지식 | 모델 반영 |
|------------|----------|
| LD(연관 불균형)는 근거리 패턴 | CNN이 담당 |
| 유전자 간 상호작용은 장거리 | DiT self-attention이 담당 |
| 인구군 간 유전적 거리는 계층적 | 계층적 pop+superpop 임베딩 |
| 유전체의 99.9%는 공통 | 공유 백본, FiLM으로 0.1% 차이만 변조 |
| 특정 위치는 항상 0 | enforce_zeros + zero_mask |
| 대립유전자 빈도는 인구군별 상이 | FiLM beta shift |

### P7. 에러 처리

모든 모듈에 명확한 에러 처리를 적용한다:

```python
# 패턴 1: 텐서 shape 검증
def forward(self, x, t, y):
    assert x.dim() == 3, f"Expected 3D input, got {x.dim()}D: {x.shape}"
    assert x.shape[1] == self.in_channels, (
        f"Channel mismatch: expected {self.in_channels}, got {x.shape[1]}"
    )

# 패턴 2: 데이터 무결성 검증
def load_data(path):
    if not Path(path).exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    data = pickle.load(open(path, 'rb'))
    if isinstance(data, tuple) and len(data) != 2:
        raise ValueError(f"Expected (X, y) tuple, got {len(data)} elements")
    return data

# 패턴 3: GPU/DDP 에러
def setup_ddp():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. DDP requires GPU.")
    n_gpus = torch.cuda.device_count()
    if n_gpus < 2:
        raise RuntimeError(f"DDP requires 2+ GPUs, found {n_gpus}")

# 패턴 4: 학습 중 NaN 감지
if torch.isnan(loss):
    raise RuntimeError(
        f"NaN loss at epoch {epoch}, step {step}. "
        f"lr={optimizer.param_groups[0]['lr']:.2e}, "
        f"grad_norm={grad_norm:.4f}"
    )
```

### P8. 모델 저장 형식 (.pth)

추론 모델은 `.pth` 확장자를 사용한다.

```python
# 학습 중 checkpoint 저장 (재학습 가능)
torch.save({
    'epoch': epoch,
    'model_state_dict': model.module.state_dict(),  # DDP에서는 .module
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'best_val_loss': best_val_loss,
    'config': config,
}, f"{save_dir}/checkpoint_epoch{epoch}.pth")

# 최종 추론 모델 저장 (배포용)
torch.save({
    'model_state_dict': model.module.state_dict(),
    'config': config,
}, f"{save_dir}/best_model.pth")

# 추론 시 로드
checkpoint = torch.load("best_model.pth", map_location='cuda:0')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

### P9. 재현성 우선

결과 비교가 필요한 실험은 반드시 seed와 split을 고정한다.

```python
SEED = 20260327

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
```

추가 규칙:

- train/test split은 최초 1회 생성 후 재사용
- baseline과 제안 모델은 동일 split, 동일 평가 샘플 수를 사용
- 최종 표/그림은 단일 run이 아니라 seed 반복 평균과 분산을 함께 기록

---

## 전체 파이프라인 흐름

```
[Phase 0] 환경 구축
    └→ 의존성 설치, DDP 검증, bf16 호환 확인

[Phase 1] 전처리 (병렬)                              ≈ 30분
    ├→ 22 염색체 병렬 VCF 파싱
    ├→ Gene-level PCA (joblib 병렬)
    ├→ 토큰화 + 정규화 (fp32 통계량)
    └→ Train/Test 분할 + zero_mask 생성

[Phase 2] 모델 구축                                   ≈ 1일
    ├→ HierarchicalPopulationEmbedding
    ├→ UnifiedFiLMGenerator
    ├→ CNNStemEncoder + FiLMConvBlock
    ├→ DiTCore + DiTBlock (AdaLN-Zero)
    ├→ CNNDecoder + PatchEmbed1D
    └→ HybridCNNDiTFiLM 조립 + shape 테스트

[Phase 3] 학습 (DDP + bf16)                           ≈ 3~12시간
    ├→ DDP 초기화 (2x A6000)
    ├→ GuassianDiffusion 프로세스
    ├→ bf16 autocast 학습 루프
    ├→ wandb 로깅 (rank 0만)
    └→ best_model.pth 저장

[Phase 3.5] 하이퍼파라미터 랜덤 서치 (wandb Sweep)      ≈ 수일
    ├→ 200회 Random Search (wandb sweep)
    ├→ 51개 파라미터, 10개 카테고리 탐색
    ├→ wandb에 실시간 추적 (loss, grad_norm, lr, balance, VRAM 등)
    ├→ 상위 10개 시행 선별 (val_reconstruction_error 기준)
    ├→ 하이퍼파라미터 중요도 분석
    └→ 최종 설정 확정

[Phase 4] 추론 / 샘플 생성                            ≈ 30분
    ├→ best_model.pth 로드
    ├→ 인구군별 조건부 샘플 생성
    └→ enforce_zeros 적용 후 저장

[Phase 5] 평가 (병렬)                                 ≈ 1시간
    ├→ 충실도 (AF 상관, LD 감쇠)
    ├→ 구조 (PCA 클러스터)
    ├→ 유용성 (Recovery Rate)
    ├→ 프라이버시 (NNAA)
    └→ 강건성 (인구군별 품질)

[Phase 6] Ablation + 논문 작성
```

---

## Phase 완료 기준

| Phase | 완료 기준 |
|------|-----------|
| Phase 0 | 의존성 설치, CUDA/bf16/DDP 가능 여부 확인 |
| Phase 1 | 전처리 산출물 생성 + `split_manifest.json` 저장 + shape 검증 |
| Phase 2 | 모델 shape test 통과 + 파라미터 수/VRAM 추정 기록 |
| Phase 3 | 1 epoch smoke test + checkpoint 저장/로드 확인 |
| Phase 4 | 샘플 소량 생성 + `generation_meta.json` 저장 + zero_mask 적용 검증 |
| Phase 5 | 핵심 지표 json/csv 저장 + bootstrap CI 계산 + 인구군별 결과 표 생성 |
| Phase 6 | ablation 비교표 + baseline 비교 + 논문용 figure/export 정리 |

---

## 하드웨어 리소스 계획

| 리소스 | 사양 | 활용 |
|--------|------|------|
| GPU 0 | RTX A6000 (49GB) | DDP rank 0 (학습 + 로깅) |
| GPU 1 | RTX A6000 (49GB) | DDP rank 1 (학습) |
| CPU | (시스템 기본) | 전처리 병렬화, 데이터 로딩 |
| RAM | (시스템 기본) | 전체 데이터셋 in-memory |

### VRAM 사용 추정 (기준 설정: C=64, d=256, DiT 4블록)

```
모델 파라미터:     ~6.8M × 2 bytes (bf16) ≈ 14 MB
Optimizer states:  ~6.8M × 8 bytes (Adam) ≈ 54 MB
Activations:       ~2 GB (batch=32, gradient checkpointing 없이)
────────────────────────────────────────────────
총 per-GPU:        ~2.1 GB

49 GB VRAM → 여유 충분. batch_size=64~128도 가능.
```
