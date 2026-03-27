# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

HybridGenoDiT: Population-conditional synthetic genotype generation using a Hybrid CNN-DiT diffusion model with hierarchical FiLM conditioning. Targets SCI publication using 1000 Genomes Phase 3 data (2,504 samples, 26 populations, 5 superpopulations).

## Commands

```bash
# Environment
uv sync

# Preprocessing (parallel VCF → Gene PCA → tokenized tensors)
python src/preprocessing/run_pipeline.py

# VCF merging (22 chromosomes parallel)
python src/preprocessing/merge_data.py --format vcf
python src/preprocessing/merge_data.py --format pkl --maf 0.01

# Training (DDP 2-GPU, bf16)
torchrun --nproc_per_node=2 src/training/trainer.py --config configs/default.yaml

# Single GPU debug
python src/training/trainer.py --config configs/default.yaml --single_gpu

# Inference (generate synthetic samples)
python src/inference/generator.py --config configs/default.yaml --model_path outputs/run_001/best_model.pth

# Evaluation (parallel)
python src/evaluation/run_evaluation.py --config configs/default.yaml --syn_dir outputs/run_001/synthetic_samples

# Hyperparameter sweep
wandb sweep configs/sweep.yaml --project HybridGenoDiT
wandb agent <sweep_id>

# Tests
pytest tests/
```

## Architecture

CNN-DiT hybrid diffusion model with FiLM (Feature-wise Linear Modulation) conditioning:

```
src/
├── preprocessing/        # VCF→Gene PCA→tokens (22-chr parallel, joblib PCA)
├── models/
│   ├── hybrid_geno_dit.py  # Main model: CNN encoder → DiT core → CNN decoder
│   ├── diffusion.py         # GaussianDiffusion (cosine schedule, 500 timesteps)
│   └── modules/
│       ├── conditioning.py  # HierarchicalPopulationEmbedding + UnifiedFiLMGenerator
│       ├── cnn.py           # FiLMConvBlock, CNNStemEncoder, CNNDecoder
│       └── dit.py           # DiTBlock (AdaLN-Zero = FiLM), DiTCore, PatchEmbed1D
├── training/             # DDP trainer, EMA, losses (masked_mse, min_snr, mmd)
├── inference/            # DDIM sampler, CFG, population-conditional generation
├── evaluation/           # Fidelity, structure, utility, privacy, robustness (parallel)
└── utils/                # DDP setup, .pth checkpoint, wandb ExperimentLogger
```

**Data flow**: `(B, 8, 26624)` → CNN encoder [FiLM] → `(B, 4C, 3328)` → Patchify → `(B, 208, d)` → DiT [AdaLN-Zero] → Un-patchify → CNN decoder [FiLM] + skips → `(B, 8, 26624)`

**Conditioning path**: `pop_label(0-25)` → `HierarchicalPopulationEmbedding(pop_emb + superpop_emb)` → `+ timestep_emb` → `UnifiedFiLMGenerator` → per-block (γ, β) for CNN + (γ, β, α) for DiT

## Key Design Principles

- **bf16 everywhere, no GradScaler**: RTX A6000 (Ampere CC 8.6) supports bf16 natively. bf16 has fp32-equivalent dynamic range (8-bit exponent), so GradScaler is unnecessary. Use `torch.autocast(device_type='cuda', dtype=torch.bfloat16)`.
- **DDP on 2 GPUs**: Always `torchrun --nproc_per_node=2`. Logging/saving on rank 0 only. `DistributedSampler` with `set_epoch()`.
- **No early stopping**: Run full epochs, track best by `val_reconstruction_error`, save `best_model.pth`.
- **Optimizer**: AdamW with cosine warmup scheduler.
- **All time-intensive ops parallelized**: VCF parsing (22 workers), gene PCA (joblib), evaluation metrics (ProcessPoolExecutor).
- **Domain-driven**: CNN captures local LD, DiT captures long-range gene interactions, FiLM modulates per population, `enforce_zeros` + `zero_mask` preserves biological constraints.
- **Model saves as .pth**: `torch.save({'model_state_dict': model.module.state_dict(), 'config': config, ...}, 'best_model.pth')`.

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
- **Gene PCA**: PCA 성분 수(K)는 하드코딩이 아닌 그리드 서치로 자동 결정. 후보 [4,6,8,10,12,16]에서 500개 샘플 유전자를 멀티스레딩으로 평가하여 평균 explained variance ≥ 90%인 최소 K를 선택. 결정된 K가 `num_channels`로 모델 전체에 전파. ~21,819 genes × K PCA components → tokenized → padded → model input `(K, gene_size)`.
- **EMA** (Exponential Moving Average) with decay 0.9999 is applied during training; EMA weights are used for inference.
- **Population-balanced sampling**: sqrt-proportional oversampling for minority populations.
- **wandb** logging is restricted to rank 0 in DDP. Config key: `WANDB_MODE=offline` for no-internet runs.
- Detailed phase-by-phase documentation lives in `docs/01_overview/` through `docs/09_performance_optimization/`.
