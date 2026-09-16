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

# Training (DDP 2-GPU, bf16)
torchrun --nproc_per_node=2 src/training/trainer.py --config configs/default.yaml

# Single GPU debug
python src/training/trainer.py --config configs/default.yaml --single_gpu

# Inference (generate synthetic samples)
python src/inference/generator.py --config configs/default.yaml --model_path outputs/run_001/best_model.pth

# Evaluation (DUPI / UI / PI + distribution distances in PCA(2) space)
python scripts/evaluate_synthetic_metrics.py --syn-dir outputs/run_001/synthetic_samples --out-dir outputs/run_001/evaluation_metrics

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
├── training/             # DDP trainer, EMA, losses (masked_mse, min_snr, mmd)
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
- **EMA** (Exponential Moving Average) with decay 0.999 (`configs/default.yaml: training.ema_decay`) is applied during training; EMA weights are used for inference. The 0.9999 seen in `src/utils/ema.py` is the `EMAModel` class's own default, used only when a config omits `ema_decay` — the shipped config does not.
- **Population-balanced sampling**: sqrt-proportional oversampling for minority populations.
- **wandb** logging is restricted to rank 0 in DDP. Config key: `WANDB_MODE=offline` for no-internet runs.
- 연구 기록은 `docs/reports/`와 `docs/superpowers/plans/`에 있다 (`docs/01_overview/`~`09_*` 디렉터리는 더 이상 없다).
