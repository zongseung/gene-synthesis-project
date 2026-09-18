#!/usr/bin/env python3
"""Main training script for HiPoDiT diffusion model.

Supports:
- DDP training on 2 GPUs via torchrun (nccl backend)
- Single-GPU debug mode via --single_gpu flag
- bf16 autocast without GradScaler (Ampere GPU)
- EMA with shadow parameter checkpointing; decay 0.999 from
  `configs/default.yaml: training.ema_decay` (the 0.9999 in src/utils/ema.py is the
  EMAModel class default, used only when a config omits ema_decay)
- Population-balanced sqrt-proportional oversampling
- Cosine warmup scheduler
- Min-SNR-gamma loss weighting
- No early stopping: runs all epochs, saves best by val_reconstruction_error
- wandb logging on rank 0 only (called directly, no wrapper)

Usage:
    # DDP (2 GPU)
    torchrun --nproc_per_node=2 src/training/trainer.py --config configs/default.yaml

    # Single GPU debug
    python src/training/trainer.py --config configs/default.yaml --single_gpu
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import sys
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader

# Allow direct execution: torchrun src/training/trainer.py
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Project imports
from src.data.dataloader import create_dataloaders
from src.models import GaussianDiffusion, HybridCNNDiTFiLM
from src.utils.config import parse_args_with_config
from src.utils.ddp import cleanup_ddp, get_rank, get_world_size, is_main_process, setup_ddp
from src.utils.ema import EMAModel


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _bind_normalization_stats(config: dict) -> None:
    data_cfg = config["data"]
    if not data_cfg.get("normalize", False):
        return
    stats_path = Path(
        data_cfg.get("normalization_stats_path", "data/processed/normalization_stats.pkl")
    )
    if not stats_path.exists():
        raise FileNotFoundError(f"Normalization stats not found: {stats_path}")
    data_cfg["normalization_stats_sha256"] = hashlib.sha256(
        stats_path.read_bytes()
    ).hexdigest()


# ───────────────────────────────────────────────────────────────────
# Cosine warmup scheduler (LambdaLR + module-level multiplier fn)
# ───────────────────────────────────────────────────────────────────

def cosine_warmup_lr_lambda(step: int, warmup: int, max_iters: int) -> float:
    """LR multiplier: linear warmup then cosine decay to 0.

    Reproduces the old hand-rolled ``CosineWarmupScheduler(_LRScheduler)``
    exactly (verified in scratchpad: identical LR sequence, rel_tol=1e-12).
    """
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, max_iters - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def make_cosine_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup: int,
    max_iters: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Build a LambdaLR with the cosine-warmup multiplier."""
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=partial(cosine_warmup_lr_lambda, warmup=warmup, max_iters=max_iters),
    )


# ───────────────────────────────────────────────────────────────────
# Validation
# ───────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(
    model: nn.Module,
    diffusion: GaussianDiffusion,
    val_loader: DataLoader,
    device: torch.device,
) -> float:
    """Compute validation reconstruction error.

    Runs the diffusion forward process at a fixed timestep and measures
    the model's prediction error on the validation set.

    Returns:
        Mean reconstruction error (MSE) across the validation set.
    """
    model.eval()
    total_mse = 0.0
    total_samples = 0

    # Fixed timestep set for deterministic validation across epochs.
    # Evaluate at 10 evenly-spaced timesteps covering the full schedule.
    fixed_timesteps = torch.linspace(
        0, diffusion.timesteps - 1, 10, dtype=torch.long, device=device
    )

    for x, y in val_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        batch_size = x.shape[0]

        # Assign fixed timesteps deterministically (round-robin)
        t = fixed_timesteps[torch.arange(batch_size, device=device) % len(fixed_timesteps)]

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss_dict = diffusion.p_losses(
                model=model,
                x_start=x,
                t=t,
                y=y,
                use_min_snr=False,
                cfg_training=False,
            )

        total_mse += loss_dict["mse"].item() * batch_size
        total_samples += batch_size

    model.train()

    # Average across all samples (and DDP processes if applicable)
    avg_mse = total_mse / max(total_samples, 1)

    if dist.is_initialized():
        tensor = torch.tensor([total_mse, total_samples], device=device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        avg_mse = tensor[0].item() / max(tensor[1].item(), 1)

    return avg_mse


# ───────────────────────────────────────────────────────────────────
# Checkpoint saving
# ───────────────────────────────────────────────────────────────────

def save_checkpoint(
    model: nn.Module,
    ema: EMAModel,
    epoch: int,
    val_loss: float,
    config: dict,
    path: str,
) -> None:
    """Save the best model (rank 0 only).

    Inference reads ``model_state_dict`` and ``ema_state_dict``
    (src/inference/generator.py). Optimizer and scheduler state are not saved
    because no resume path exists.
    """
    if not is_main_process():
        return

    model_state = (
        model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    )
    checkpoint = {
        "model_state_dict": model_state,
        "ema_state_dict": ema.state_dict(),
        "epoch": epoch,
        "best_val_loss": val_loss,
        "config": config,
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }

    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(checkpoint, path)
    logger.info(f"Checkpoint saved: {path}")


# ───────────────────────────────────────────────────────────────────
# Main training function
# ───────────────────────────────────────────────────────────────────

def train(config: dict) -> None:
    """Full training pipeline with DDP, bf16, EMA, and wandb logging."""

    # ── Setup ──
    _bind_normalization_stats(config)
    training_cfg = config["training"]
    distributed_cfg = config.get("distributed", {})
    single_gpu = distributed_cfg.get("num_gpus", 2) == 1

    if single_gpu:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available.")
        local_rank = 0
        torch.cuda.set_device(0)
        world_size = 1
    else:
        local_rank = setup_ddp()
        world_size = get_world_size()

    device = torch.device(f"cuda:{local_rank}")

    # Seed for reproducibility (per-rank diversification for DDP)
    seed = training_cfg.get("seed", 20260327)
    rank = get_rank() if not single_gpu else 0
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)
    np.random.seed(seed + rank)

    if is_main_process():
        logger.info(f"Training config: {config}")
        logger.info(f"Device: {device}, world_size: {world_size}")

    # ── Data ──
    train_loader, val_loader = create_dataloaders(
        config, rank=rank, world_size=world_size
    )

    if is_main_process():
        logger.info(
            f"Train: {len(train_loader.dataset)} samples, "
            f"Val: {len(val_loader.dataset)} samples"
        )

    # ── Load label hierarchy (pop_to_superpop mapping for model) ──
    import pickle
    label_hier_path = config["data"].get(
        "label_hierarchy_path", "data/processed/label_hierarchy.pkl"
    )
    with open(label_hier_path, "rb") as f:
        label_hierarchy = pickle.load(f)
    config.setdefault("model", {})["pop_to_superpop"] = label_hierarchy["pop_to_superpop"]

    # ── Model ──
    model = HybridCNNDiTFiLM(config).to(device)
    if not single_gpu:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    # ── Diffusion ──
    data_cfg = config["data"]
    zero_mask_path = data_cfg.get("zero_mask_path", "data/processed/zero_mask.pt")
    if Path(zero_mask_path).exists():
        zero_mask = torch.load(zero_mask_path, map_location=device, weights_only=True)
        # zero_mask saved as (gene_size, K), model expects (K, gene_size)
        if zero_mask.ndim == 2 and zero_mask.shape[1] < zero_mask.shape[0]:
            zero_mask = zero_mask.T
    else:
        zero_mask = None
        logger.warning(f"Zero mask not found at {zero_mask_path}")

    diffusion_cfg = config["diffusion"]
    diffusion = GaussianDiffusion(
        timesteps=diffusion_cfg["max_timesteps"],
        zero_mask=zero_mask,
        enforce_zeros=data_cfg.get("enforce_zeros", True),
        min_snr_gamma=diffusion_cfg.get("min_snr_gamma", 5.0),
        null_class=data_cfg.get("num_classes", 26),
        cfg_dropout_rate=diffusion_cfg.get("cfg_dropout_rate", 0.1),
        schedule_type=diffusion_cfg.get("noise_schedule", "cosine"),
        prediction_target=diffusion_cfg.get("prediction_target", "epsilon"),
        sample_clip=diffusion_cfg.get("sample_clip", 6.0),
        feature_schedule=diffusion_cfg.get("feature_schedule"),
    ).to(device)

    # ── Optimizer: AdamW, NO GradScaler (bf16 has fp32 dynamic range) ──
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_cfg["lr"],
        weight_decay=training_cfg.get("weight_decay", 1e-4),
        betas=(0.9, 0.999),
    )

    # ── Scheduler ──
    total_steps = training_cfg["epochs"] * len(train_loader)
    scheduler = make_cosine_warmup_scheduler(
        optimizer,
        warmup=training_cfg.get("warmup_steps", 100),
        max_iters=total_steps,
    )

    # ── EMA ──
    base_model = model.module if hasattr(model, "module") else model
    ema = EMAModel(base_model, decay=training_cfg.get("ema_decay", 0.9999))

    # ── wandb (rank 0 only) ──
    exp_cfg = config.get("experiment", {})
    log_to_wandb = is_main_process()
    if log_to_wandb:
        wandb.init(
            project="HiPoDiT",
            name=exp_cfg.get("run_name", None),
            tags=exp_cfg.get("tags", None),
            reinit=True,
        )
        wandb.config.update(config, allow_val_change=True)

    # ── Save dir ──
    save_dir = config.get("save_dir", "outputs/default")
    os.makedirs(save_dir, exist_ok=True)

    # ── Training loop (no early stopping) ──
    best_val_loss = float("inf")
    global_step = 0

    if is_main_process():
        logger.info(f"Starting training for {training_cfg['epochs']} epochs")

    for epoch in range(training_cfg["epochs"]):
        # Set epoch for sampler (DDP shuffling / balanced sampling)
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        model.train()
        epoch_loss = 0.0
        epoch_steps = 0

        for step, (x, y) in enumerate(train_loader):
            x = x.to(device, non_blocking=True)  # (B, K, gene_size)
            y = y.to(device, non_blocking=True)   # (B,)

            optimizer.zero_grad(set_to_none=True)

            # ── autocast (NO GradScaler) ──
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                t = torch.randint(0, diffusion.timesteps, (x.shape[0],), device=device)
                loss_dict = diffusion.p_losses(
                    model=model,
                    x_start=x,
                    t=t,
                    y=y,
                    noise=torch.randn_like(x),
                    use_min_snr=training_cfg.get("use_min_snr", True),
                    cfg_training=diffusion_cfg.get("guidance_type", "normal") != "normal",
                )
                loss = loss_dict["loss"]

            # NaN detection before backward to avoid gradient corruption
            if torch.isnan(loss):
                raise RuntimeError(
                    f"NaN loss at epoch {epoch}, step {step}. "
                    f"lr={optimizer.param_groups[0]['lr']:.2e}"
                )

            # bf16 backward -- NO scaler needed
            loss.backward()

            # Gradient clipping
            grad_norm = nn.utils.clip_grad_norm_(
                model.parameters(),
                training_cfg.get("gradient_clipping", 1.0),
            )

            optimizer.step()
            scheduler.step()

            # EMA update (every step)
            ema.update(base_model)

            epoch_loss += loss.item()
            epoch_steps += 1
            global_step += 1

            # Logging (rank 0 only, every 20 steps)
            if is_main_process() and step % 20 == 0:
                metrics = {
                    "train/loss": loss.item(),
                    "train/mse": loss_dict["mse"].item(),
                    "train/grad_norm": (
                        grad_norm.item()
                        if isinstance(grad_norm, torch.Tensor)
                        else grad_norm
                    ),
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/epoch": epoch,
                }
                wandb.log(metrics, step=global_step)

        # ── Validation (every epoch, using EMA weights) ──
        ema.apply_shadow(base_model)
        val_rec_error = validate(model, diffusion, val_loader, device)
        ema.restore(base_model)

        if is_main_process():
            avg_loss = epoch_loss / max(epoch_steps, 1)
            wandb.log(
                {
                    "val/reconstruction_error": val_rec_error,
                    "val/epoch": epoch,
                    "train/avg_epoch_loss": avg_loss,
                },
                step=global_step,
            )
            logger.info(
                f"[Epoch {epoch}/{training_cfg['epochs']}] "
                f"avg_loss={avg_loss:.6f}, val_rec={val_rec_error:.6f}, "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )

            # Save best model
            if val_rec_error < best_val_loss:
                best_val_loss = val_rec_error
                save_checkpoint(
                    model, ema, epoch, best_val_loss, config,
                    path=os.path.join(save_dir, "best_model.pth"),
                )
                logger.info(f"  New best: val_rec={val_rec_error:.6f}")

    # ── Done ──
    if is_main_process():
        logger.info(f"Training complete. Best val_rec={best_val_loss:.6f}")

    if log_to_wandb:
        wandb.finish()

    if not single_gpu:
        cleanup_ddp()


# ───────────────────────────────────────────────────────────────────
# Entry point
# ───────────────────────────────────────────────────────────────────

def main() -> None:
    config = parse_args_with_config()
    train(config)


if __name__ == "__main__":
    main()
