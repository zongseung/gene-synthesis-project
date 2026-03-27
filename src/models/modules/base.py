"""Base utilities: sinusoidal timestep embedding and zero-init helper."""

import math

import torch
import torch.nn as nn


def timestep_embedding(
    timesteps: torch.Tensor, dim: int, max_period: int = 10000
) -> torch.Tensor:
    """
    Create sinusoidal timestep embeddings.

    Args:
        timesteps: (B,) integer timesteps.
        dim: Embedding dimension.
        max_period: Controls the minimum frequency of the embeddings.

    Returns:
        (B, dim) embedding vectors.
    """
    assert timesteps.dim() == 1, f"Expected 1D timesteps, got {timesteps.dim()}D"
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(half, dtype=torch.float32, device=timesteps.device)
        / half
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat(
            [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
        )
    return embedding


def zero_module(module: nn.Module) -> nn.Module:
    """Zero-initialize all parameters of a module (for AdaLN-Zero)."""
    for p in module.parameters():
        p.detach().zero_()
    return module
