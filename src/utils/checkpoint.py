"""Checkpoint save/load utilities for .pth format.

All saves happen on rank 0 only. Supports DDP model unwrapping,
EMA state persistence, and top-k checkpoint retention.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from src.utils.ddp import is_main_process

logger = logging.getLogger(__name__)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    ema: Any | None,
    epoch: int,
    val_loss: float,
    config: dict,
    save_dir: str | Path,
    filename: str | None = None,
) -> Path | None:
    """Save a training checkpoint as .pth (rank 0 only).

    Args:
        model: The model (may be wrapped in DDP).
        optimizer: Optimizer with state.
        scheduler: LR scheduler with state.
        ema: EMAModel instance or None.
        epoch: Current epoch number.
        val_loss: Current validation loss (for best-model tracking).
        config: Full config dict for reproducibility.
        save_dir: Directory to save the checkpoint.
        filename: Optional filename. Defaults to ``checkpoint_epoch{epoch}.pth``.

    Returns:
        Path to saved checkpoint, or None if not rank 0.
    """
    if not is_main_process():
        return None

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        filename = f"checkpoint_epoch{epoch}.pth"

    path = save_dir / filename

    # Unwrap DDP model
    model_state = (
        model.module.state_dict()
        if hasattr(model, "module")
        else model.state_dict()
    )

    checkpoint = {
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "ema_state_dict": ema.state_dict() if ema is not None else None,
        "epoch": epoch,
        "best_val_loss": val_loss,
        "config": config,
        "pytorch_version": torch.__version__,
        "cuda_version": getattr(torch.version, "cuda", None),
    }

    torch.save(checkpoint, path)
    logger.info(f"Checkpoint saved: {path} (epoch={epoch}, val_loss={val_loss:.6f})")

    return path


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    ema: Any | None = None,
    device: str = "cuda:0",
) -> tuple[int, float]:
    """Load a checkpoint and restore model/optimizer/scheduler/EMA state.

    Args:
        path: Path to the .pth checkpoint file.
        model: Model to load weights into (unwrapped, not DDP).
        optimizer: Optional optimizer to restore state.
        scheduler: Optional scheduler to restore state.
        ema: Optional EMAModel to restore state.
        device: Device to map tensors to.

    Returns:
        Tuple of (epoch, best_val_loss).

    Raises:
        FileNotFoundError: If checkpoint file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    if ema is not None and checkpoint.get("ema_state_dict") is not None:
        ema.load_state_dict(checkpoint["ema_state_dict"])

    epoch = checkpoint.get("epoch", 0)
    best_val_loss = checkpoint.get("best_val_loss", float("inf"))

    logger.info(
        f"Checkpoint loaded: {path} (epoch={epoch}, val_loss={best_val_loss:.6f})"
    )

    return epoch, best_val_loss


def save_top_k(save_dir: str | Path, k: int = 3) -> None:
    """Keep only the top-k checkpoints by val_loss, delete the rest.

    Parses ``best_val_loss`` from each checkpoint file. The file
    ``best_model.pth`` is always preserved regardless of k.

    Args:
        save_dir: Directory containing checkpoint .pth files.
        k: Number of checkpoints to keep (excluding best_model.pth).
    """
    if not is_main_process():
        return

    save_dir = Path(save_dir)
    checkpoint_files = sorted(save_dir.glob("checkpoint_epoch*.pth"))

    if len(checkpoint_files) <= k:
        return

    # Read val_loss from each checkpoint
    scored: list[tuple[float, Path]] = []
    for ckpt_path in checkpoint_files:
        try:
            data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            val_loss = data.get("best_val_loss", float("inf"))
            scored.append((val_loss, ckpt_path))
        except Exception as e:
            logger.warning(f"Could not read checkpoint {ckpt_path}: {e}")
            scored.append((float("inf"), ckpt_path))

    # Sort by val_loss ascending (best first), keep top-k
    scored.sort(key=lambda x: x[0])
    to_keep = {p for _, p in scored[:k]}

    for val_loss, ckpt_path in scored:
        if ckpt_path not in to_keep:
            ckpt_path.unlink()
            logger.info(f"Removed checkpoint: {ckpt_path} (val_loss={val_loss:.6f})")
