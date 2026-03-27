#!/usr/bin/env python3
"""Population-conditional synthetic genotype generation.

Loads a trained HybridGenoDiT checkpoint (with EMA weights), generates
samples per population using DDIM sampling with optional CFG, applies
enforce_zeros and denormalization, and saves individual .pt files plus
a generation_meta.json summary.

Usage:
    python src/inference/generator.py \\
        --config configs/default.yaml \\
        --model_path outputs/run_001/best_model.pth \\
        --output_dir outputs/run_001/synthetic_samples

    python src/inference/generator.py \\
        --config configs/default.yaml \\
        --model_path outputs/run_001/best_model.pth \\
        --output_dir outputs/run_001/oversampled \\
        --oversample_minority 200
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.models import GaussianDiffusion, HybridCNNDiTFiLM
from src.utils.config import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ───────────────────────────────────────────────────────────────────
# Denormalization
# ───────────────────────────────────────────────────────────────────

def denormalize_samples(
    samples: torch.Tensor,
    stats_path: str,
) -> torch.Tensor:
    """Restore generated samples to original scale.

    Args:
        samples: (B, K, gene_size) float32 model output (normalized).
        stats_path: Path to normalization_stats.pkl.

    Returns:
        (B, K, gene_size) float32 denormalized tensor.

    Broadcasting:
        stats['mean'] and stats['std'] are (gene_size_stats, K).
        Model output is (B, K, gene_size).
        If gene_size_stats < gene_size (stats from unpadded data),
        pad stats with mean=0, std=1 for the padded region.
    """
    with open(stats_path, "rb") as f:
        stats = pickle.load(f)

    mean_np = stats["mean"]  # (gene_size_stats, K)
    std_np = stats["std"]    # (gene_size_stats, K)

    gene_size = samples.shape[2]
    stats_gene_size = mean_np.shape[0]

    # Pad stats if they were computed on unpadded data
    if stats_gene_size < gene_size:
        pad_len = gene_size - stats_gene_size
        n_k = mean_np.shape[1]
        mean_np = np.concatenate([mean_np, np.zeros((pad_len, n_k), dtype=mean_np.dtype)], axis=0)
        std_np = np.concatenate([std_np, np.ones((pad_len, n_k), dtype=std_np.dtype)], axis=0)

    # (gene_size, K) -> (K, gene_size) -> (1, K, gene_size)
    xmean = torch.tensor(mean_np.T, dtype=torch.float32).unsqueeze(0)
    xstd = torch.tensor(std_np.T, dtype=torch.float32).unsqueeze(0)

    return samples * xstd + xmean


# ───────────────────────────────────────────────────────────────────
# Generation
# ───────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_samples(
    config: dict,
    model_path: str,
    output_dir: str,
    n_samples_per_pop: dict[int, int] | None = None,
    oversample_minority: int | None = None,
    seed: int = 20260327,
) -> None:
    """Generate population-conditional synthetic genotype samples.

    Args:
        config: Nested config dict.
        model_path: Path to best_model.pth checkpoint.
        output_dir: Directory for output .pt files and metadata.
        n_samples_per_pop: {pop_idx: n_samples}. If None, matches
            the real data population sizes from label_hierarchy.pkl.
        oversample_minority: Minimum samples per population. If set,
            small populations are oversampled to at least this count.
        seed: Random seed for generation (default 20260327).
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    t_start = time.time()

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    # ── Step 1: Load model with EMA parameters ──
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Use checkpoint config if available, fall back to provided config
    model_config = checkpoint.get("config", config)
    model = HybridCNNDiTFiLM(model_config).to(device)

    # Load EMA parameters (preferred for inference)
    used_ema = False
    if "ema_state_dict" in checkpoint:
        ema_state = checkpoint["ema_state_dict"]
        shadow = ema_state.get("shadow", ema_state)
        for name, param in model.named_parameters():
            if name in shadow:
                param.data.copy_(shadow[name].to(param.data.dtype))
        used_ema = True
        logger.info(f"EMA parameters loaded from {model_path}")
    else:
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.warning("EMA not found in checkpoint, using raw model weights")

    model.eval()
    logger.info(
        f"Model loaded from {model_path} (epoch {checkpoint.get('epoch', '?')})"
    )

    # ── Diffusion process ──
    data_cfg = config["data"]
    diffusion_cfg = config["diffusion"]

    zero_mask_path = data_cfg.get("zero_mask_path", "data/processed/zero_mask.pt")
    if Path(zero_mask_path).exists():
        zero_mask = torch.load(zero_mask_path, map_location=device, weights_only=True)
        # zero_mask saved as (gene_size, K), model expects (K, gene_size)
        if zero_mask.ndim == 2 and zero_mask.shape[1] < zero_mask.shape[0]:
            zero_mask = zero_mask.T
    else:
        zero_mask = None
        logger.warning(f"Zero mask not found at {zero_mask_path}")

    diffusion = GaussianDiffusion(
        timesteps=diffusion_cfg["max_timesteps"],
        zero_mask=zero_mask,
        enforce_zeros=data_cfg.get("enforce_zeros", True),
        null_class=data_cfg.get("num_classes", 26),
    ).to(device)

    # ── Determine samples per population ──
    os.makedirs(output_dir, exist_ok=True)

    if n_samples_per_pop is None:
        label_hierarchy_path = data_cfg.get(
            "label_hierarchy_path", "data/processed/label_hierarchy.pkl"
        )
        with open(label_hierarchy_path, "rb") as f:
            label_info = pickle.load(f)

        pop_to_idx = label_info["pop_to_idx"]
        pop_sizes = label_info["pop_sizes"]
        # Map population names to indices and use their counts
        n_samples_per_pop = {}
        for pop_name, count in pop_sizes.items():
            if pop_name in pop_to_idx:
                n_samples_per_pop[pop_to_idx[pop_name]] = count

    # Apply oversampling for minority populations
    if oversample_minority is not None:
        n_samples_per_pop = {
            k: max(v, oversample_minority) for k, v in n_samples_per_pop.items()
        }

    # ── Step 2: Generate samples per population ──
    num_channels = data_cfg["num_channels"]
    gene_size = data_cfg["gene_size"]
    guidance_type = diffusion_cfg.get("guidance_type", "normal")
    guidance_weight = diffusion_cfg.get("guidance_weight", 3.0)
    ddim_steps = diffusion_cfg.get("sampling_timesteps", 50)
    batch_gen_size = 32  # Generate in batches for efficiency

    total_generated = 0
    normalization_stats_path = data_cfg.get(
        "normalization_stats_path", "data/processed/normalization_stats.pkl"
    )
    has_norm_stats = Path(normalization_stats_path).exists()

    for pop_idx in sorted(n_samples_per_pop.keys()):
        n_samples = n_samples_per_pop[pop_idx]
        logger.info(f"Generating {n_samples} samples for population {pop_idx}...")

        generated = 0
        while generated < n_samples:
            current_batch = min(batch_gen_size, n_samples - generated)
            shape = (current_batch, num_channels, gene_size)
            y = torch.full((current_batch,), pop_idx, device=device, dtype=torch.long)

            # bf16 inference
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                cfg_scale = guidance_weight if guidance_type == "classifier_free" else 0.0
                samples = diffusion.sample_ddim(
                    model=model,
                    shape=shape,
                    y=y,
                    device=device,
                    ddim_steps=ddim_steps,
                    eta=0.0,
                    guidance_scale=cfg_scale,
                )

            # Convert to fp32 for post-processing and saving
            samples = samples.float().cpu()

            # Denormalize if stats are available
            if has_norm_stats:
                samples = denormalize_samples(samples, normalization_stats_path)

            # Enforce zeros on final output
            if zero_mask is not None and data_cfg.get("enforce_zeros", True):
                # zero_mask already transposed to (K, gene_size) above
                zm = zero_mask.cpu()
                if zm.shape[0] != num_channels:
                    zm = zm.T
                samples = samples * (~zm).unsqueeze(0).float()

            # Save individual samples
            for i in range(current_batch):
                sample_tensor = samples[i]  # (K, gene_size)
                label_tensor = torch.tensor(pop_idx, dtype=torch.long)
                save_path = os.path.join(
                    output_dir, f"sample_pop{pop_idx}_{generated + i:04d}.pt"
                )
                torch.save((sample_tensor, label_tensor), save_path)

            generated += current_batch
            total_generated += current_batch

        logger.info(f"  Population {pop_idx}: {n_samples} samples saved")

    # ── Step 3: Save generation_meta.json ──
    generation_time = time.time() - t_start

    meta = {
        "model_path": str(model_path),
        "model_epoch": checkpoint.get("epoch", None),
        "used_ema": used_ema,
        "total_samples": total_generated,
        "seed": seed,
        "oversample_minority": oversample_minority,
        "per_population": {str(k): v for k, v in n_samples_per_pop.items()},
        "config": {
            "max_timesteps": diffusion_cfg["max_timesteps"],
            "ddim_steps": ddim_steps,
            "guidance_type": guidance_type,
            "guidance_weight": guidance_weight,
            "num_channels": num_channels,
            "gene_size": gene_size,
        },
        "timestamp": datetime.now().isoformat(),
        "generation_time_sec": round(generation_time, 2),
    }

    meta_path = os.path.join(output_dir, "generation_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"Total: {total_generated} samples generated in {output_dir}")
    logger.info(f"Metadata saved to {meta_path}")
    logger.info(f"Generation time: {generation_time:.1f}s")


# ───────────────────────────────────────────────────────────────────
# CLI
# ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic genotype samples with HybridGenoDiT"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML config file"
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to best_model.pth"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for synthetic samples",
    )
    parser.add_argument(
        "--oversample_minority",
        type=int,
        default=None,
        help="Minimum samples per population (oversample small pops)",
    )
    parser.add_argument(
        "--seed", type=int, default=20260327, help="Generation seed"
    )
    args = parser.parse_args()

    config = load_config(args.config)

    if args.output_dir is None:
        save_dir = config.get("save_dir", "outputs/default")
        args.output_dir = os.path.join(save_dir, "synthetic_samples")

    generate_samples(
        config=config,
        model_path=args.model_path,
        output_dir=args.output_dir,
        oversample_minority=args.oversample_minority,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
