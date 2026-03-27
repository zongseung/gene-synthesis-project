# Phase 0: 프로젝트 구조 및 환경

---

## 0. 구현 상태 요약

> 아래 디렉토리 구조는 **목표 구조**이다. 현재 레포 상태와의 차이를 아래 표로 정리한다.

| 모듈 | 상태 | 설명 |
|------|------|------|
| `pyproject.toml` | ✅ 존재 | 의존성 일부 미반영 (§2 참조) |
| `configs/default.yaml` | ❌ 예정 | §3에 스키마 정의됨, 파일 미생성 |
| `src/preprocessing/merge_data.py` | ✅ 존재 | VCF 병합 구현 완료 |
| `src/preprocessing/run_pipeline.py` | ❌ 예정 | 전처리 오케스트레이터 |
| `src/preprocessing/vcf_parser.py` | ❌ 예정 | VCF 병렬 파싱 |
| `src/preprocessing/gene_pca.py` | ❌ 예정 | Gene-level PCA |
| `src/models/` | ❌ 예정 | 모델 아키텍처 전체 |
| `src/training/` | ❌ 예정 | 학습 파이프라인 전체 |
| `src/inference/` | ❌ 예정 | 추론 파이프라인 전체 |
| `src/evaluation/` | ❌ 예정 | 평가 파이프라인 전체 |
| `src/utils/` | ❌ 예정 | 공통 유틸리티 전체 |
| `src/data/` | ❌ 예정 | 데이터 로더 |
| `tests/` | ❌ 예정 | 테스트 전체 |
| `scripts/` | ❌ 후속 | HP 분석 스크립트 |
| `docs/10_experiment_operations/README.md` | ❌ 예정 | 재현성/운영 정책 문서 |

---

## 1. 목표 디렉토리 구조

```
gene-synthesis-project/
├── pyproject.toml                 # 의존성 관리 (uv)
├── main.py                       # 엔트리포인트
├── README.md
│
├── configs/                       # 설정 파일
│   ├── default.yaml              # 기본 설정 (C=64, d=256, DiT 4블록)
│   ├── sweep.yaml                # wandb Sweep 설정 (랜덤 서치 탐색 공간)
│   ├── sweep_full.yaml           # 전체 51개 파라미터 sweep
│   ├── ablation_no_dit.yaml
│   ├── ablation_no_film.yaml
│   ├── ablation_no_hierarchy.yaml
│   ├── ablation_no_cnn_film.yaml
│   ├── ablation_no_aux.yaml
│   └── ablation_no_zero_init.yaml
│
├── src/
│   ├── __init__.py
│   │
│   ├── preprocessing/             # Phase 1: 전처리
│   │   ├── __init__.py
│   │   ├── vcf_parser.py         # VCF 병렬 파싱
│   │   ├── gene_pca.py           # Gene-level PCA (병렬)
│   │   ├── tokenizer.py          # 토큰화 + 정규화
│   │   ├── label_builder.py      # 계층적 레이블 생성
│   │   └── run_pipeline.py       # 전처리 실행 스크립트
│   │
│   ├── models/                    # Phase 2: 모델
│   │   ├── __init__.py
│   │   ├── hybrid_geno_dit.py    # HybridCNNDiTFiLM (전체 모델)
│   │   ├── diffusion.py          # GaussianDiffusion
│   │   └── modules/
│   │       ├── __init__.py
│   │       ├── base.py           # timestep_embedding, zero_module
│   │       ├── conditioning.py   # HierarchicalPopEmb, UnifiedFiLMGen
│   │       ├── cnn.py            # FiLMConvBlock, CNNEncoder, CNNDecoder
│   │       └── dit.py            # PatchEmbed1D, DiTBlock, DiTCore
│   │
│   ├── data/                      # 데이터 로더
│   │   ├── __init__.py
│   │   ├── dataset.py            # GeneticDataset1k (DDP 호환)
│   │   └── synthetic_dataset.py  # SynGeneticDataset (생성 샘플용)
│   │
│   ├── training/                  # Phase 3: 학습
│   │   ├── __init__.py
│   │   ├── trainer.py            # 학습 루프 (DDP + bf16)
│   │   ├── losses.py             # masked_mse, mmd_rbf, min_snr, aux losses
│   │   ├── ema.py                # EMAModel (지수 이동 평균)
│   │   ├── sampler.py            # PopulationBalancedSampler (소수군 오버샘플링)
│   │   ├── augmentation.py       # 유전체 특화 데이터 증강
│   │   └── scheduler.py          # CosineWarmupScheduler
│   │
│   ├── inference/                 # Phase 4: 추론
│   │   ├── __init__.py
│   │   ├── generator.py          # 인구군별 샘플 생성
│   │   ├── ddim_sampler.py       # DDIM 고속 샘플링
│   │   └── cfg_utils.py          # Classifier-Free Guidance (적응 weight 포함)
│   │
│   ├── evaluation/                # Phase 5: 평가
│   │   ├── __init__.py
│   │   ├── fidelity.py           # AF 상관, LD, 분포
│   │   ├── structure.py          # PCA, Wasserstein
│   │   ├── utility.py            # Recovery Rate, 증강 효과
│   │   ├── privacy.py            # evaluate_privacy() 통합 진입점
│   │   ├── privacy_dupi.py       # DUPI 전용 (Jeong et al. 2023)
│   │   ├── privacy_nnaa.py       # NNAA 전용 (기존 논문 비교용)
│   │   ├── privacy_plot.py       # UI-PI 비교 플롯
│   │   ├── robustness.py         # 인구군별 강건성 (핵심)
│   │   └── run_evaluation.py     # 전체 평가 (병렬)
│   │
│   └── utils/                     # 공통 유틸리티
│       ├── __init__.py
│       ├── distributed.py        # DDP setup/cleanup
│       ├── checkpoint.py         # .pth 저장/로드
│       ├── logging_utils.py      # ExperimentLogger (wandb + 콘솔, HP 추적)
│       └── config.py             # YAML 설정 로드/검증
│
├── scripts/                       # 분석 스크립트
│   ├── select_top_trials.py      # wandb sweep 상위 시행 선별
│   └── analyze_hparam_importance.py  # HP 중요도 상관 분석
│
├── data/                          # 데이터 (git 추적 안 함)
│   ├── raw/                      # VCF, panel 파일
│   └── processed/                # 전처리된 pkl, pt 파일
│
├── outputs/                       # 실험 결과 (git 추적 안 함)
│   ├── run_001/
│   │   ├── best_model.pth
│   │   ├── checkpoint_epoch*.pth
│   │   ├── synthetic_samples/
│   │   ├── evaluation/
│   │   └── config.yaml
│   └── ablation_*/
│
├── docs/                          # 문서
│   ├── 01_overview/
│   ├── 02_preprocessing/
│   ├── 03_model/
│   ├── 04_training/
│   ├── 05_inference/
│   ├── 06_evaluation/
│   ├── 07_project_structure/     ← 현재 문서
│   ├── 08_hyperparameter_search/
│   ├── 09_performance_optimization/
│   └── 10_experiment_operations/
│
├── notebooks/                     # 분석/시각화 노트북
│   └── ...
│
└── tests/                         # 테스트
    ├── test_model_shapes.py      # 모델 입출력 shape 검증
    ├── test_film_params.py       # FiLM 파라미터 생성 검증
    ├── test_ddp_compat.py        # DDP 호환성 검증
    └── test_bf16_stability.py    # bf16 수치 안정성 검증
```

---

## 2. 의존성

### 현재 `pyproject.toml` (실제 파일 기준)

```toml
[project]
name = "gene-synthesis-project"
version = "0.1.0"
requires-python = ">=3.13"
dependencies = [
    "ipykernel>=7.2.0",
    "matplotlib>=3.10.8",
    "numpy>=2.4.3",
    "pandas>=3.0.1",
    "pysam>=0.23.3",
    "scikit-learn>=1.8.0",
    "seaborn>=0.13.2",
    "torch>=2.11.0",
]
```

### 문서에서 요구하지만 아직 미반영된 패키지

> 아래 패키지는 구현 진행 시 `pyproject.toml`에 추가해야 한다.

| 패키지 | 용도 | 추가 시점 |
|--------|------|----------|
| `cyvcf2>=0.31` | VCF 파싱 (vcf_parser.py) | Phase 1 전처리 구현 시 |
| `joblib>=1.3` | Gene PCA 병렬화 | Phase 1 전처리 구현 시 |
| `pyyaml>=6.0` | Config 로드 | Phase 2 모델 구현 시 |
| `wandb>=0.16` | 실험 로깅 | Phase 3 학습 구현 시 |
| `scipy>=1.11` | DUPI, Wasserstein 등 평가 지표 | Phase 5 평가 구현 시 |
| `tqdm>=4.66` | 진행률 표시 | 전체 |

### GPU 환경별 참고

| 환경 | 비고 |
|------|------|
| CUDA (RTX A6000) | `torch`가 CUDA 빌드여야 함. `uv sync` 후 `python -c "import torch; print(torch.cuda.is_available())"` 검증 |
| CPU only | DDP 및 bf16 미지원. 단위 테스트용으로만 사용 |

---

## 3. 설정 파일 형식

### configs/default.yaml

```yaml
# ── 데이터 ──
data:
  gene_size: 26624
  num_channels: 8
  num_classes: 26
  num_superpops: 5
  num_samples: 2504
  test_ratio: 0.1
  split_seed: 20260327
  normalize: true
  enforce_zeros: true
  zero_mask_path: "data/processed/zero_mask.pt"
  label_hierarchy_path: "data/processed/label_hierarchy.pkl"
  split_manifest_path: "data/processed/split_manifest.json"

# ── 모델 ──
model:
  name: "HybridCNNDiTFiLM"
  # CNN
  cnn_base_channels: 64
  cnn_channel_mult: [1, 1, 2, 4]
  cnn_kernel_size: 3
  # DiT
  dit_d_model: 256
  dit_n_blocks: 4
  dit_n_heads: 4
  dit_mlp_ratio: 4.0
  dit_dropout: 0.0
  # Patch
  patch_size: 16
  # FiLM
  pop_emb_dim: 256
  superpop_emb_dim: 256
  film_type: "adaln_zero"

# ── Diffusion ──
diffusion:
  max_timesteps: 500
  noise_schedule: "cosine"
  prediction_target: "epsilon"
  guidance_type: "normal"
  guidance_weight: 3.0

# ── 학습 ──
training:
  epochs: 100                    # 조기종료 없이 전체 실행
  batch_size: 32
  lr: 2e-4
  seed: 20260327
  weight_decay: 1e-4
  optimizer: "adamw"
  gradient_clipping: 1.0
  warmup_steps: 100
  precision: "bf16"              # bf16 부동소수점
  save_every: 20                 # checkpoint 저장 주기

# ── 실험 운영 ──
experiment:
  run_name: "20260327_baseline_seed20260327"
  tags: ["baseline", "hybrid-genodit"]
  save_top_k_checkpoints: 3
  export_policy: "internal_only"

# ── DDP ──
distributed:
  backend: "nccl"
  num_gpus: 2

# ── 보조 손실 ──
aux_loss:
  enabled: false
  lambda_pca_dist: 0.01
  lambda_pop_structure: 0.01
  type: "mmd_rbf"
  warmup_epochs: 20

# ── 저장 ──
save_dir: "outputs/default"
```

---

## 4. 실행 명령어 정리

```bash
# Phase 0: 환경 설치                              ✅ 실행 가능
uv sync

# Phase 0.5: VCF 병합 (22 염색체)                  ✅ 실행 가능
python src/preprocessing/merge_data.py --format vcf
python src/preprocessing/merge_data.py --format pkl --maf 0.01

# Phase 1: 전처리 (병렬)                           ❌ 작성 예정
python src/preprocessing/run_pipeline.py

# Phase 2: 모델 테스트                             ❌ 작성 예정
python tests/test_model_shapes.py

# Phase 3: 학습 (DDP + bf16)                       ❌ 작성 예정
torchrun --nproc_per_node=2 src/training/trainer.py --config configs/default.yaml

# Phase 4: 추론                                    ❌ 작성 예정
python src/inference/generator.py \
    --config configs/default.yaml \
    --model_path outputs/default/best_model.pth

# Phase 5: 평가 (병렬)                             ❌ 작성 예정
python src/evaluation/run_evaluation.py \
    --config configs/default.yaml \
    --syn_dir outputs/default/synthetic_samples

# Ablation                                         ❌ 작성 예정
for cfg in configs/ablation_*.yaml; do
    torchrun --nproc_per_node=2 src/training/trainer.py --config $cfg
done
```

---

## 5. 에러 처리 체크리스트

| 위치 | 에러 유형 | 처리 |
|------|----------|------|
| VCF 파싱 | 파일 미존재 | `FileNotFoundError` + 해당 염색체 스킵 |
| VCF 파싱 | 변이 파싱 실패 | 개별 변이 `continue` (로그) |
| PCA | 수렴 실패 | 해당 유전자 스킵 (로그) |
| 데이터 로드 | 파일 미존재 | `FileNotFoundError` 명시적 raise |
| 데이터 로드 | shape 불일치 | `ValueError` 명시적 raise |
| 모델 forward | 입력 shape | `assert` + 메시지 |
| 모델 forward | FiLM 파라미터 수 | `assert` + 메시지 |
| DDP 초기화 | GPU 부족 | `RuntimeError` 명시적 raise |
| 학습 중 | NaN loss | `RuntimeError` + 디버그 정보 |
| 학습 중 | OOM | batch_size 줄이기 가이드 출력 |
| checkpoint 로드 | 파일 미존재 | `FileNotFoundError` |
| checkpoint 로드 | config 불일치 | `Warning` + 진행 |
| 추론 | 모델 미존재 | `FileNotFoundError` |
| bf16 | 미지원 GPU | `RuntimeError` + fp32 폴백 안내 |

---

## 6. 운영 문서 연결

프로젝트 운영 관련 기준은 아래 문서를 함께 참조한다.

| 문서 | 역할 |
|------|------|
| `01_overview` | 범위, 원칙, 완료 기준 |
| `07_project_structure` | config schema, 목표 구조, 환경 |
| `10_experiment_operations` | seed, run naming, checkpoint retention, resume, 외부 공유 정책 |
