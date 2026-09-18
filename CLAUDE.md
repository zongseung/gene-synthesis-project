# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

HiPoDiT: Population-conditional synthetic genotype generation using a Hybrid CNN-DiT diffusion model with hierarchical FiLM conditioning. Targets SCI publication using 1000 Genomes Phase 3 data (2,504 samples, 26 populations, 5 superpopulations).

## Environments

This repo has two independent virtualenvs — never merge them:

- Root `.venv` (Python 3.13.6): this project (HiPoDiT). `uv sync` / `.venv/bin/python`.
- `korean-medicine-llm/.venv` (Python 3.12): separate sub-project with a working VARCO-VISION 14B
  training stack (torch + bitsandbytes 4bit + peft). Use `korean-medicine-llm/.venv/bin/python`
  for that project; do not `uv sync` it from the root.

## Commands

```bash
# Environment
uv sync

# Preprocessing (sequential: 1 chromosome at a time → Gene PCA → tokenized tensors)
python src/preprocessing/run_pipeline.py

# VCF merging (22 chromosomes parallel)
python src/preprocessing/merge_data.py

# Training (DDP 2-GPU, bf16). 루트 .venv의 torch는 cu130이라 이 머신 드라이버(535 / CUDA 12.2)에서
# CUDA를 못 잡는다. 학습·생성은 csdi 환경으로 돌린다 (2026-09-17 확인).
PYTHONHASHSEED=0 /home/user/Envs/csdi/bin/torchrun --nproc_per_node=2 \
  src/training/trainer.py --config configs/default.yaml

# Single GPU debug
/home/user/Envs/csdi/bin/python src/training/trainer.py --config configs/default.yaml --single_gpu

# Inference (generate synthetic samples) — GPU, so csdi too
/home/user/Envs/csdi/bin/python src/inference/generator.py --config configs/default.yaml \
  --model_path outputs/run_001/best_model.pth

# Binomial(2,p) 표현으로 전처리 (러스트 백엔드, 출력 디렉터리 분리 필수)
HIPODIT_GLM_FAMILY=binom2 HIPODIT_PROCESSED_DIR=data/processed_binom2 \
  .venv/bin/python src/preprocessing/run_pipeline.py

# Population-conditional prior: re-normalize an existing processed dir (no VCF re-run)
.venv/bin/python scripts/make_conditional_prior_data.py --arm mean --output-dir data/processed_cond_mean

# 원고 생성 (산출물에서 숫자·좌표 주입, main.tex는 직접 고치지 말 것)
.venv/bin/python scripts/make_manuscript_tex.py --out main.tex

# 적률 조건화 시뮬레이션 (교차점 n_p)
.venv/bin/python scripts/simulate_moment_conditioning.py

# Evaluation (DUPI / UI / PI + distribution distances in PCA(2) space) — CPU, 루트 .venv
.venv/bin/python scripts/evaluate_synthetic_metrics.py --syn-dir outputs/run_001/synthetic_samples --out-dir outputs/run_001/evaluation_metrics

# Hyperparameter sweep (wandb) — configs/sweep.yaml is NOT in the repo (configs/ holds only
# default.yaml). Write the sweep config first, then:
# wandb sweep configs/sweep.yaml --project HiPoDiT
# wandb agent <sweep_id>

# Tests
pytest tests/
```

## Architecture

CNN-DiT hybrid diffusion model with FiLM (Feature-wise Linear Modulation) conditioning:

```
src/
├── preprocessing/        # VCF→Gene PCA→tokens (22 chr streamed one at a time, serial per-gene PCA)
├── models/
│   ├── hybrid_geno_dit.py  # Main model: CNN encoder → DiT core → CNN decoder
│   ├── diffusion.py         # GaussianDiffusion (schedule_type default cosine; configs/default.yaml uses linear, 1000 timesteps, 100 DDIM sampling steps)
│   ├── genotype_decoder.py  # latent → {0,1,2} decoders: B0–B3 ablations + arm T (see below)
│   └── modules/
│       ├── conditioning.py  # HierarchicalPopulationEmbedding + UnifiedFiLMGenerator
│       ├── cnn.py           # FiLMConvBlock, CNNStemEncoder, CNNDecoder
│       └── dit.py           # DiTBlock (AdaLN-Zero = FiLM), DiTCore, PatchEmbed1D
├── training/             # DDP trainer, EMA (loss lives in GaussianDiffusion.p_losses)
├── inference/            # DDIM sampler, CFG, population-conditional generation
├── evaluation/           # DUPI/UI/PI, distribution distances (W2, MMD-RBF), PCA(2) compare
└── utils/                # DDP setup, config loading, EMA, wandb ExperimentLogger
```

**Data flow**: `(B, C, gene_size)` → CNN encoder [FiLM] → Patchify → DiT [AdaLN-Zero] → Un-patchify → CNN decoder [FiLM] + skips → `(B, C, gene_size)`. 현재 기본 설정은 `C=4`, `gene_size=24576`.

**Conditioning path**: `pop_label(0-25)` → `HierarchicalPopulationEmbedding(pop_emb + superpop_emb)` → `+ timestep_emb` → `UnifiedFiLMGenerator` → per-block (γ, β) for CNN + (γ, β, α) for DiT

## Key Design Principles

- **bf16 everywhere, no GradScaler**: RTX A6000 (Ampere CC 8.6) supports bf16 natively. bf16 has fp32-equivalent dynamic range (8-bit exponent), so GradScaler is unnecessary. Use `torch.autocast(device_type='cuda', dtype=torch.bfloat16)`.
- **DDP on 2 GPUs**: Always `torchrun --nproc_per_node=2`. Logging/saving on rank 0 only. `DistributedSampler` with `set_epoch()`.
- **No early stopping**: Run full epochs, track best by `val_reconstruction_error`, save `best_model.pth`.
- **Optimizer**: AdamW with cosine warmup scheduler.
- **Preprocessing is sequential by design, for memory**: `stream_vcf_and_pca` walks the 22 chromosomes one at a time (`src/preprocessing/pca.py:107` logs `Streaming VCF→PCA: sequential ... (OOM-safe: 1 chr at a time)`) so peak RAM stays at roughly one chromosome, while per-gene reduction within each chromosome runs across genes in a stdlib `multiprocessing.Pool` (`src/preprocessing/pca.py`, `os.cpu_count()` workers, no `joblib`). The other parallel component is the separate merge utility `src/preprocessing/merge_data.py` (`multiprocessing.Pool`, `N_WORKERS = min(22, cpu_count())`), which is not part of the pipeline. Evaluation (`scripts/evaluate_synthetic_metrics.py`) is single-process; it caches loaded tensors and PCA coordinates instead.
- **Domain-driven**: CNN captures local LD, DiT captures long-range gene interactions, FiLM modulates per population. 생물학적 제약(항상 0인 위치)은 모델이 아니라 `GaussianDiffusion._apply_zero_mask`가 강제한다 — `enforce_zeros` 플래그 + `zero_mask` 버퍼로 q_sample·p_sample·DDIM 각 스텝과 손실에 적용된다. 모델 `forward`에는 zero-mask 경로가 없다.
- **Model saves as .pth**: `torch.save({'model_state_dict': model.module.state_dict(), 'config': config, ...}, 'best_model.pth')`.

## 인구군 조건부 prior (2026-09-17)

`claudedocs/research_conditional_prior_vs_posthoc_20260917.md` 기준. ShiftDDPMs(AAAI 2023)의
**Data-Normalization** = PriorGrad(ICLR 2022) 순방향 과정을 정규화 단계에 넣은 것이다. 인구군별 평균을
x_0에서 빼고 모델은 잔차만 학습하며, 샘플링 마지막에 역정규화로 평균을 되돌린다. **모델·diffusion
kernel·FiLM은 한 줄도 바뀌지 않았다** — Data-Normalization은 궤적을 분리하지 못하므로 조건을 계속
네트워크에 넣어야 한다(FiLM 제거 금지).

전부 `normalization_stats.pkl` 한 곳에서 처리된다. `mean`이 `(G,K)`면 기존 전역 통계,
`(P,G,K)`면 인구군별이고 `apply_normalization`/`invert_normalization`에 라벨을 넘겨야 한다.
arm은 stats 파일 자체에 `conditional` 키로 박히고, 체크포인트가 stats의 sha256을 들고 있어
학습·생성 arm 불일치는 구조적으로 불가능하다. 학습·생성 설정에 새 키는 없다.

- **arm**: `none`(기존 baseline) · `mean`(인구군 평균 + pooled within-pop std) ·
  `mean_std`(인구군 평균 + 인구군 std). 환경변수 `HIPODIT_CONDITIONAL_PRIOR`로 전처리 시 선택.
- **`variance_floor`(기본 0.01)**: `mean_std` arm 전용. 소수 인구군은 일부 셀에서 수치적으로 상수라
  per-pop std가 pooled의 1e-10까지 내려가고, 그걸로 나누면 fp32 반올림 먼지가 O(1) 잔차로 증폭된다
  (1KG train에서 8,763/2,555,904 셀 = 0.34%). PriorGrad와 같은 이유로 pooled std의 배수로 하한을 둔다.
- **기존 처리 결과 재사용**: baseline은 clip 없이 정규화됐으므로 정확히 역변환된다.
  `scripts/make_conditional_prior_data.py`가 그 성질을 이용해 VCF→GLM-PCA를 다시 돌리지 않고
  arm별 디렉터리를 6초에 만든다. 출력 디렉터리는 덮어쓰지 않고, 바뀌지 않는 산출물은 symlink한다.
- **측정된 사실**: 흡수되는 인구군 간 분산은 전체의 23.8%(평균 0.60 sd 이동). 반면 인구군 간
  다양성 차이는 이 특징 공간에서 **1.29배**(ASW 1.145 ~ CDX 0.889)에 불과하다 — 기획 문서가 가정한
  7배가 아니다. 즉 두 arm의 차이는 작을 것이고, 이득의 대부분은 두 arm이 공유하는 중심 정렬에서 온다.
- **판정(2026-09-18, Binomial 표현 4시드 쌍)**: arm은 `mean`. EUR 재현율 0.044 → 0.966,
  EAS 0.100 → 0.950 (p < 0.001), chr22 AF 오차 0.0156 → 0.0132 (p = 0.0002). 대가는 집단 내 다양성
  0.632 → 0.437 (p = 0.0010). `mean_std`를 쓰지 않는 근거는 `scripts/simulate_moment_conditioning.py`가
  낸 교차점이다. 인구군당 **학습** 개체가 91명 아래면 2차 적률의 추정 오차가 신호를 넘고, 이 패널은
  학습 기준 49~91명(중앙값 79)이다. 선택은 val에서, test는 확증으로 1회만 채점한다
  (`evaluation_metrics_val/` 대 `evaluation_metrics/`).
- **생성 설정**: `configs/default.yaml`의 `ddim_eta`는 0.0이다. 0.5에서 0.0으로 바꾸면 집단 내 다양성이
  0.58 → 0.67로 오르고 DUPI·AF는 소수점 셋째 자리에서만 움직인다. 기존 0.5의 근거였던
  `outputs/eta_sweep`은 2026-03-31의 8채널 모델이라 현재 파이프라인으로 이어지지 않는다.

## chr17 genotype-decoder study (2026-09, 현재 authoritative 실험 경로)

전장유전체 파이프라인과 **별개로 돌아가는 고정 패널 실험**이다. 위의 `src/preprocessing/run_pipeline.py`
경로가 아니라 `scripts/hipodit_*.py`를 쓴다. 결론과 판정 기준은
`docs/superpowers/plans/2026-09-15-hipodit-ld.md`(원본 §1–§8 + 개정 §9)와
`docs/reports/hipodit_t_results_20260916.md`에 있고, 코드를 바꾸기 전에 그 두 문서를 먼저 읽어야 한다.

**무엇을 바꿨나**: diffusion kernel과 GLM-PCA는 한 줄도 바뀌지 않았다. 바뀐 것은 latent → genotype
디코더뿐이다. 현재 채택 모델 **HiPoDiT-T**는 동결된 Binomial(2,p) base에 두 가지를 더한다.

- `u_{c,j}`: cohort×SNP logit offset. train pooled allele frequency를 보존하도록 SNP별 스칼라를
  적합 중에 Newton으로 푼다(`offset_gauge="pooled_af"`). logit 공간 중심화는 폐기됐다 — Jensen
  때문에 확률 공간 평균이 보존되지 않는다.
- `τ_j`: per-SNP heterozygosity tilt. 개인별 기대 dosage `E[G]=2p`를 **닫힌 해로 정확히** 보존한다.

SNP는 서로 독립으로 샘플링한다. `A_b` residual과 genomic-order chain은 primary model에 없다
(B0–B3 ablation으로만 남아 있다).

**금지 사항**: LD 주장 금지. 독립 샘플링 모델도 LD `r²` MAE를 개선하므로 `r²` 개선을 linkage
근거로 쓸 수 없다(반례는 결과 보고서 §9.3). 개정 §9.5의 허용 문장 하나를 벗어나지 말 것.

```bash
# 고정 패널 준비 (새 출력 디렉터리 필수)
uv run --no-sync python scripts/hipodit_rebuild_check.py prepare --output-dir OUT --genes 8 \
  --components 4 --max-variants 64 --glm-iterations 30 --seed 20260327

# Gate 1' oracle 비교 (real held-out latent, dev)
.venv/bin/python scripts/hipodit_genotype_check.py oracle --prepared-dir P --output-dir O

# Gate 2' end-to-end (GPU 필요)
CUDA_VISIBLE_DEVICES=0 /home/user/Envs/csdi/bin/python scripts/hipodit_rebuild_check.py train \
  --output-dir P --run-dir R --decoder-dir O --eval-split val --device cuda

# Gate 3' 확증 (test split은 1회만)
/home/user/Envs/csdi/bin/python scripts/hipodit_multiseed.py --mode decoder --decoder-dir O \
  --eval-split test --prepared-dir P --output-dir M --seeds <10개 이상>

# Gate 4 privacy
.venv/bin/python scripts/hipodit_privacy.py --prepared-dir P --multiseed-dir M --output-dir V
```

**환경 주의**: 학습·샘플링은 `/home/user/Envs/csdi/bin/python`(3.10, torch 2.4.1+cu121)으로 돌린다.
루트 `.venv`(3.13)는 CPU 전용이라 테스트와 CPU 작업 전용이다. csdi 쪽에는 pytest가 없다.

**작업 규칙**:
- 산출물 디렉터리는 덮어쓰지 않는다. 기존 디렉터리를 가리키면 `FileExistsError`로 거부된다.
- `outputs/diagnostics/*`의 gate 산출물은 동결 기록이다. 재생성하지 말 것.
- gate가 실패하면 그것으로 보고한다. 하이퍼파라미터를 고쳐 다시 돌리는 것은 기획서가 금지한다.
- 판단을 내렸으면 `docs/reports/hipodit_t_decision_log_20260916.md` 형식대로 근거와 "틀렸을 때의
  비용"을 남긴다.

## Data

```
data/
├── ALL.autosomes.phase3.genotypes.vcf.gz  (13.9 GB, 1KG Phase 3, chr1-22)
├── ALL.autosomes.phase3.genotypes.vcf.gz.tbi  (tabix index)
└── integrated_call_samples_v3.20130502.ALL.panel  (sample→pop→superpop mapping)
```

Preprocessing produces: `gene_pca_features.pkl`, `train_data.pkl`, `test_data.pkl`, `normalization_stats.pkl` (fp32), `label_hierarchy.pkl` (pop↔superpop mapping), `zero_mask.pt`.

## Important Details

- **Hierarchical labels**: 26 populations map to 5 superpopulations (AFR/EUR/EAS/SAS/AMR). The `pop_to_superpop` mapping in `label_hierarchy.pkl` is loaded by `HierarchicalPopulationEmbedding` to enable information sharing from superpop (e.g., AFR 661 samples) to minority pop (e.g., ASW 61 samples).
- **AdaLN-Zero in DiT = FiLM**: DiT blocks use `γ·LayerNorm(x) + β` with α (gate) initialized to zero. This means DiT starts as identity function and gradually learns long-range corrections on top of CNN features.
- **차원 축소**: 기본값은 `glm_pca` (`src/preprocessing/config.py`의 `DIM_RED_METHOD`, `HIPODIT_DIM_RED` 환경변수로 변경). `configs/default.yaml`은 `num_channels: 4`, `gene_size: 24576`을 쓴다. 성분 수 그리드 서치는 linear PCA 경로의 동작이며 glm_pca 기본 경로에는 적용되지 않는다. 모델 입력은 `(num_channels, gene_size)`.
- **관측 모형(family)**: `HIPODIT_GLM_FAMILY`로 고른다. 기본값은 여전히 `poi`(Poisson, 러스트 `glmpca_fast`)이지만 **`binom2`가 맞는 우도다.** 유전형은 시행 2회의 성공 횟수이고, Poisson은 dosage 2를 넘는 값에 질량을 주며 `Var = 평균`을 가정한다. 1KG chr22 실측으로 예측 평균이 2를 넘어 clip되는 비율이 3.34%, 분산이 2배 이상 틀리는 비율이 15.6%다. `binom2`로 바꾸면 chr22 대립유전자 빈도 오차가 0.0217 → 0.0156으로 줄었다(2026-09-18, 4시드). 백엔드는 러스트 `binom_glmpca_rs`를 우선 쓰고 없으면 `src/preprocessing/binomial_glm_pca.py`의 scipy 구현으로 떨어진다.
- **EMA** (Exponential Moving Average) with decay 0.999 (`configs/default.yaml: training.ema_decay`) is applied during training; EMA weights are used for inference. The 0.9999 seen in `src/utils/ema.py` is the `EMAModel` class's own default, used only when a config omits `ema_decay` — the shipped config does not.
- **Population-balanced sampling**: sqrt-proportional oversampling for minority populations.
- **wandb** logging is restricted to rank 0 in DDP. Config key: `WANDB_MODE=offline` for no-internet runs.
- 연구 기록은 `docs/reports/`와 `docs/superpowers/plans/`에 있다 (`docs/01_overview/`~`09_*` 디렉터리는 더 이상 없다).
