from __future__ import annotations

import math

import torch


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    t = torch.linspace(0, 1, timesteps + 1, dtype=torch.float64)
    cumulative = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    cumulative = cumulative / cumulative[0]
    return (1 - cumulative[1:] / cumulative[:-1]).clamp(0, 0.999).float()


def linear_beta_schedule(
    timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02,
) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, timesteps)


def three_band_betas(betas: torch.Tensor) -> torch.Tensor:
    cumulative = torch.cumprod(1 - betas.double(), dim=0)
    if len(betas) == 1:
        return betas[:, None].expand(-1, 3).clone()
    h = -cumulative.log()
    u = (h - h[0]) / (h[-1] - h[0])
    exponents = torch.tensor([0.5, 1., 2.], dtype=h.dtype)
    warped = h[0] + (h[-1] - h[0]) * u[:, None] ** exponents
    increments = warped - torch.cat((torch.zeros_like(warped[:1]), warped[:-1]))
    return -torch.expm1(-increments).float()
