"""Distributed Data Parallel (DDP) utilities.

Environment variables LOCAL_RANK, RANK, and WORLD_SIZE are set
automatically by ``torchrun``.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist


def setup_ddp() -> int:
    """Initialize the DDP process group with NCCL backend.

    Returns:
        local_rank: The local rank of the current process.

    Raises:
        RuntimeError: If CUDA is unavailable or fewer than 2 GPUs are found.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. DDP requires GPU.")

    n_gpus = torch.cuda.device_count()
    if n_gpus < 2:
        raise RuntimeError(f"DDP requires 2+ GPUs, found {n_gpus}")

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    return local_rank


def cleanup_ddp() -> None:
    """Destroy the DDP process group if initialized."""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_rank() -> int:
    """Global rank, or 0 when no process group is initialized."""
    return dist.get_rank() if dist.is_initialized() else 0


def get_world_size() -> int:
    """Process count, or 1 when no process group is initialized."""
    return dist.get_world_size() if dist.is_initialized() else 1


def is_main_process() -> bool:
    return get_rank() == 0
