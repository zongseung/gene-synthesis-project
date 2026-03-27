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


def is_main_process() -> bool:
    """Return True if the current process is rank 0 (or DDP is not active)."""
    return not dist.is_initialized() or dist.get_rank() == 0


def get_rank() -> int:
    """Return the global rank of the current process (0 if not distributed)."""
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    """Return the total number of processes (1 if not distributed)."""
    if dist.is_initialized():
        return dist.get_world_size()
    return 1
