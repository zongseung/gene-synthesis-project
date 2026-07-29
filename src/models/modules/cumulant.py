"""Orthogonal higher-cumulant input modulation for HiPoDiT."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ..diffusion import cosine_beta_schedule, linear_beta_schedule


class OrthogonalCumulantModulation(nn.Module):
    """Apply population cumulant corrections in covariance eigen-coordinates."""

    def __init__(
        self,
        *,
        n_pops: int,
        n_channels: int,
        gene_size: int,
        timesteps: int,
        schedule_type: str = "cosine",
        stats_path: str | Path | None = None,
        initial_gain: float = 0.0,
        max_gain: float = 0.25,
        value_clip: float = 4.0,
        correction_clip: float = 2.0,
    ) -> None:
        super().__init__()
        if n_pops < 1 or n_channels < 1 or gene_size < 1:
            raise ValueError("n_pops, n_channels, and gene_size must be positive")
        if timesteps < 1:
            raise ValueError("timesteps must be positive")
        if max_gain <= 0 or abs(initial_gain) >= max_gain:
            raise ValueError("initial_gain must satisfy abs(initial_gain) < max_gain")
        if value_clip <= 0 or correction_clip <= 0:
            raise ValueError("Clip values must be positive")

        self.n_pops = n_pops
        self.n_channels = n_channels
        self.gene_size = gene_size
        self.max_gain = float(max_gain)
        self.value_clip = float(value_clip)
        self.correction_clip = float(correction_clip)
        self._stats_ready = False

        identity = torch.eye(n_channels).expand(gene_size, -1, -1).clone()
        self.register_buffer("center", torch.zeros(gene_size, n_channels))
        self.register_buffer("rotation", identity)
        self.register_buffer("eigenvalues", torch.zeros(gene_size, n_channels))
        self.register_buffer(
            "skewness", torch.zeros(n_pops + 1, gene_size, n_channels)
        )
        self.register_buffer(
            "excess_kurtosis",
            torch.zeros(n_pops + 1, gene_size, n_channels),
        )

        if schedule_type == "linear":
            betas = linear_beta_schedule(timesteps)
        elif schedule_type == "cosine":
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown schedule_type: {schedule_type}")
        self.register_buffer("alphas_cumprod", torch.cumprod(1.0 - betas, dim=0))

        raw_initial = math.atanh(initial_gain / max_gain)
        self.raw_gain = nn.Parameter(
            torch.full((n_channels,), raw_initial, dtype=torch.float32)
        )

        if stats_path is not None:
            self.load_stats(stats_path)

    def load_stats(self, path: str | Path) -> None:
        """Load and validate a pickle-free cumulant NPZ artifact."""
        expected = {
            "center": (self.gene_size, self.n_channels),
            "rotation": (
                self.gene_size,
                self.n_channels,
                self.n_channels,
            ),
            "eigenvalues": (self.gene_size, self.n_channels),
            "skewness": (
                self.n_pops + 1,
                self.gene_size,
                self.n_channels,
            ),
            "excess_kurtosis": (
                self.n_pops + 1,
                self.gene_size,
                self.n_channels,
            ),
        }
        with np.load(Path(path), allow_pickle=False) as artifact:
            version = int(artifact["format_version"])
            if version != 1:
                raise ValueError(f"Unsupported cumulant format_version: {version}")

            arrays: dict[str, np.ndarray] = {}
            for name, shape in expected.items():
                array = np.asarray(artifact[name])
                if array.shape != shape:
                    raise ValueError(
                        f"{name} shape {array.shape} does not match {shape}"
                    )
                if not np.isfinite(array).all():
                    raise ValueError(f"{name} contains non-finite values")
                arrays[name] = array

        with torch.no_grad():
            for name, array in arrays.items():
                getattr(self, name).copy_(torch.from_numpy(array).float())
        self._stats_ready = True

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        if not self._stats_ready:
            raise RuntimeError("Cumulant statistics have not been loaded")
        if x.shape[1:] != (self.n_channels, self.gene_size):
            raise ValueError(
                f"Expected x shape (B, {self.n_channels}, {self.gene_size}), "
                f"got {tuple(x.shape)}"
            )
        if t.shape != (x.shape[0],) or y.shape != (x.shape[0],):
            raise ValueError("t and y must be one-dimensional and match batch size")
        device_type = x.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            features = x.float().transpose(1, 2)
            alpha_bar = self.alphas_cumprod[t].reshape(-1, 1, 1)
            sqrt_alpha = alpha_bar.sqrt()
            rotated = torch.einsum(
                "bgk,gkl->bgl",
                features - sqrt_alpha * self.center,
                self.rotation,
            )
            variance = (
                alpha_bar * self.eigenvalues
                + (1.0 - alpha_bar)
            ).clamp_min(1e-8)
            std = variance.sqrt()
            standardized = rotated / std
            bounded = standardized.clamp(-self.value_clip, self.value_clip)
            h2 = bounded.square() - 1.0
            h3 = bounded.pow(3) - 3.0 * bounded

            skewness = self.skewness[y]
            kurtosis = self.excess_kurtosis[y]
            signal_fraction = (
                alpha_bar * self.eigenvalues.clamp_min(0.0) / variance
            ).clamp(0.0, 1.0).sqrt()
            correction = (
                0.5 * signal_fraction.pow(3) * skewness * h2
                + (1.0 / 6.0)
                * signal_fraction.pow(4)
                * kurtosis
                * h3
            ).clamp(-self.correction_clip, self.correction_clip)
            correction = torch.einsum(
                "bgl,gkl->bgk", correction * std, self.rotation
            )
            gain = (
                self.max_gain * torch.tanh(self.raw_gain)
            ).reshape(1, 1, -1)
            result = features + gain * correction

        return result.transpose(1, 2).to(dtype=x.dtype)
