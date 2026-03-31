"""
Gaussian Diffusion process with cosine noise schedule, DDIM sampling,
Min-SNR loss weighting, and Classifier-Free Guidance support.

Key features:
- Cosine schedule (Nichol & Dhariwal, 2021) with 500 default timesteps.
- Min-SNR-gamma (Hang et al., 2023) for balanced per-timestep loss weighting.
- DDIM deterministic sampling for fast generation.
- CFG support: unconditional forward with null population label.
- enforce_zeros at each sampling step to preserve biological constraints.
- bf16-compatible: all buffers stored as float32, computation under autocast.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def cosine_beta_schedule(
    timesteps: int, s: float = 0.008
) -> torch.Tensor:
    """
    Cosine noise schedule (Nichol & Dhariwal, 2021).

    Returns:
        betas: (T,) noise schedule values clipped to [0, 0.999].
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(0, 0.999).float()


def linear_beta_schedule(
    timesteps: int,
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
) -> torch.Tensor:
    """
    Linear noise schedule (Ho et al., 2020).

    Args:
        timesteps: Number of diffusion steps.
        beta_start: Starting noise level.
        beta_end: Ending noise level.

    Returns:
        betas: (T,) linearly spaced noise schedule.
    """
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)


class GaussianDiffusion(nn.Module):
    """
    Gaussian diffusion process for training and sampling.

    Stores all schedule tensors as registered buffers for DDP compatibility
    and automatic device transfer.
    """

    def __init__(
        self,
        timesteps: int = 500,
        zero_mask: Optional[torch.Tensor] = None,
        enforce_zeros: bool = True,
        min_snr_gamma: float = 5.0,
        null_class: int = 26,
        cfg_dropout_rate: float = 0.1,
        schedule_type: str = "cosine",
    ):
        super().__init__()
        if timesteps < 1:
            raise ValueError(f"timesteps must be positive, got {timesteps}")

        self.timesteps = timesteps
        self.enforce_zeros_flag = enforce_zeros
        self.min_snr_gamma = min_snr_gamma
        self.null_class = null_class
        self.cfg_dropout_rate = cfg_dropout_rate

        # --- Noise schedule ---
        if schedule_type == "linear":
            betas = linear_beta_schedule(timesteps)
        elif schedule_type == "cosine":
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown schedule_type: {schedule_type}")
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # Register all schedule tensors as buffers (auto device transfer in DDP)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # Pre-computed values for q(x_t | x_0) and posterior
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", torch.rsqrt(alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod",
            torch.sqrt(1.0 / alphas_cumprod - 1),
        )

        # Posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(posterior_variance.clamp(min=1e-20)),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

        # Min-SNR weights: SNR(t) = alpha_bar_t / (1 - alpha_bar_t)
        snr = alphas_cumprod / (1.0 - alphas_cumprod)
        # min(SNR(t), gamma) / SNR(t) -- clamped for stability
        min_snr_weights = torch.clamp(snr, max=min_snr_gamma) / snr.clamp(min=1e-8)
        self.register_buffer("min_snr_weights", min_snr_weights)

        # Zero mask for biological constraints
        if zero_mask is not None:
            self.register_buffer("zero_mask", zero_mask.bool())
        else:
            self.register_buffer("zero_mask", None)

    def _extract(
        self, a: torch.Tensor, t: torch.Tensor, x_shape: tuple
    ) -> torch.Tensor:
        """Extract values from schedule tensor a at timestep indices t."""
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def _apply_zero_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Zero out positions specified by the zero mask."""
        if self.enforce_zeros_flag and self.zero_mask is not None:
            x = x * (~self.zero_mask).to(x.dtype)
        return x

    # ---------------------------------------------------------------
    # Forward diffusion (noise addition)
    # ---------------------------------------------------------------

    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Sample from q(x_t | x_0) by adding noise at timestep t.

        Args:
            x_start: (B, K, gene_size) clean data.
            t: (B,) timestep indices.
            noise: (B, K, gene_size) optional pre-sampled noise.

        Returns:
            x_t: (B, K, gene_size) noisy data at timestep t.
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alpha = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        x_t = sqrt_alpha * x_start + sqrt_one_minus_alpha * noise
        return x_t

    # ---------------------------------------------------------------
    # Training loss
    # ---------------------------------------------------------------

    def apply_cfg_dropout(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Replace population labels with null class at cfg_dropout_rate
        probability for Classifier-Free Guidance training.

        Args:
            labels: (B,) population indices.

        Returns:
            labels with some entries replaced by null_class.
        """
        mask = torch.rand(len(labels), device=labels.device) < self.cfg_dropout_rate
        labels_dropped = labels.clone()
        labels_dropped[mask] = self.null_class
        return labels_dropped

    def p_losses(
        self,
        model: nn.Module,
        x_start: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
        use_min_snr: bool = True,
        cfg_training: bool = True,
    ) -> dict[str, torch.Tensor]:
        """
        Compute training loss: MSE between predicted and actual noise,
        optionally weighted by Min-SNR-gamma.

        Args:
            model: The denoising model (HybridCNNDiTFiLM).
            x_start: (B, K, gene_size) clean data.
            t: (B,) randomly sampled timesteps.
            y: (B,) population labels.
            noise: Optional pre-sampled noise.
            use_min_snr: Whether to apply Min-SNR weighting.
            cfg_training: Whether to apply CFG label dropout.

        Returns:
            Dictionary with 'loss' (scalar) and 'mse' (unweighted for logging).
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        x_t = self.q_sample(x_start, t, noise)

        # Apply CFG label dropout during training
        if cfg_training and self.training:
            y = self.apply_cfg_dropout(y)

        # Predict noise
        pred_noise = model(x_t, t, y)

        # Per-element MSE
        mse = F.mse_loss(pred_noise, noise, reduction="none")
        reduce_dims = tuple(range(1, mse.dim()))

        # Exclude padded / always-zero positions from both loss numerator
        # and denominator so valid loci keep their original weight.
        if self.enforce_zeros_flag and self.zero_mask is not None:
            mask = (~self.zero_mask).to(mse.dtype).unsqueeze(0)  # (1, K, gene_size)
            mse = mse * mask
            valid_count = mask.sum(dim=reduce_dims).clamp(min=1)
            mse_per_sample = mse.sum(dim=reduce_dims) / valid_count
        else:
            # Mean over spatial/channel dimensions, keep batch
            mse_per_sample = mse.mean(dim=reduce_dims)

        if use_min_snr:
            weights = self._extract(self.min_snr_weights, t, (t.shape[0], 1))
            weights = weights.squeeze()
            loss = (weights * mse_per_sample).mean()
        else:
            loss = mse_per_sample.mean()

        return {
            "loss": loss,
            "mse": mse_per_sample.mean().detach(),
            "pred_noise": pred_noise,
        }

    # ---------------------------------------------------------------
    # Reverse sampling: predict x_0 from noise prediction
    # ---------------------------------------------------------------

    def _predict_x0_from_eps(
        self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor
    ) -> torch.Tensor:
        """Recover x_0 from noisy x_t and predicted noise."""
        return (
            self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    # ---------------------------------------------------------------
    # DDPM sampling
    # ---------------------------------------------------------------

    @torch.no_grad()
    def p_sample(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        guidance_scale: float = 0.0,
    ) -> torch.Tensor:
        """Single reverse step: x_t -> x_{t-1}."""
        t_batch = torch.full(
            (x_t.shape[0],), t, device=x_t.device, dtype=torch.long
        )

        # Classifier-Free Guidance
        if guidance_scale > 0.0:
            eps_cond = model(x_t, t_batch, y)
            y_null = torch.full_like(y, self.null_class)
            eps_uncond = model(x_t, t_batch, y_null)
            eps = (1.0 + guidance_scale) * eps_cond - guidance_scale * eps_uncond
        else:
            eps = model(x_t, t_batch, y)

        # Predict x_0
        pred_x0 = self._predict_x0_from_eps(x_t, t_batch, eps)
        pred_x0 = self._apply_zero_mask(pred_x0)

        # Posterior mean
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t_batch, x_t.shape) * pred_x0
            + self._extract(self.posterior_mean_coef2, t_batch, x_t.shape) * x_t
        )

        if t > 0:
            noise = torch.randn_like(x_t)
            posterior_var = self._extract(
                self.posterior_variance, t_batch, x_t.shape
            )
            x_prev = posterior_mean + torch.sqrt(posterior_var) * noise
        else:
            x_prev = posterior_mean

        return x_prev

    @torch.no_grad()
    def sample_ddpm(
        self,
        model: nn.Module,
        shape: tuple,
        y: torch.Tensor,
        device: torch.device,
        guidance_scale: float = 0.0,
    ) -> torch.Tensor:
        """
        Full DDPM reverse sampling loop.

        Args:
            model: Denoising model.
            shape: (B, K, gene_size) output shape.
            y: (B,) population labels.
            device: Target device.
            guidance_scale: CFG strength (0 = no guidance).

        Returns:
            (B, K, gene_size) generated samples.
        """
        x = torch.randn(shape, device=device)
        model.eval()

        for t in reversed(range(self.timesteps)):
            x = self.p_sample(model, x, t, y, guidance_scale)

        x = self._apply_zero_mask(x)
        return x

    # ---------------------------------------------------------------
    # DDIM sampling (fast, deterministic)
    # ---------------------------------------------------------------

    @torch.no_grad()
    def sample_ddim(
        self,
        model: nn.Module,
        shape: tuple,
        y: torch.Tensor,
        device: torch.device,
        ddim_steps: int = 50,
        eta: float = 0.0,
        guidance_scale: float = 0.0,
    ) -> torch.Tensor:
        """
        DDIM sampling (Song et al., 2020).

        Args:
            model: Denoising model.
            shape: (B, K, gene_size) output shape.
            y: (B,) population labels.
            device: Target device.
            ddim_steps: Number of DDIM steps (default 50).
            eta: Stochasticity (0 = deterministic, 1 = DDPM-equivalent).
            guidance_scale: CFG strength (0 = no guidance).

        Returns:
            (B, K, gene_size) generated samples.
        """
        model.eval()
        x = torch.randn(shape, device=device)

        # Create sub-sequence of timesteps
        timestep_seq = np.linspace(
            0, self.timesteps - 1, ddim_steps, dtype=int
        )[::-1].copy()

        for i, t in enumerate(timestep_seq):
            t_batch = torch.full(
                (shape[0],), t, device=device, dtype=torch.long
            )

            # Predict noise (with optional CFG)
            if guidance_scale > 0.0:
                eps_cond = model(x, t_batch, y)
                y_null = torch.full_like(y, self.null_class)
                eps_uncond = model(x, t_batch, y_null)
                eps = (
                    (1.0 + guidance_scale) * eps_cond
                    - guidance_scale * eps_uncond
                )
            else:
                eps = model(x, t_batch, y)

            # Current and previous alpha_bar
            alpha_t = self.alphas_cumprod[t]
            if i < len(timestep_seq) - 1:
                alpha_prev = self.alphas_cumprod[timestep_seq[i + 1]]
            else:
                alpha_prev = torch.tensor(1.0, device=device)

            # Predict x_0 (clamp alpha_t to avoid division-by-near-zero)
            alpha_t_clamped = alpha_t.clamp(min=1e-6)
            pred_x0 = (x - (1 - alpha_t).sqrt() * eps) / alpha_t_clamped.sqrt()
            pred_x0 = pred_x0.clamp(-6, 6)  # training data clipped to [-5, 5]
            pred_x0 = self._apply_zero_mask(pred_x0)

            # DDIM update
            sigma = (
                eta
                * (
                    (1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev)
                ).sqrt()
            )
            dir_xt = (1 - alpha_prev - sigma**2).clamp(min=0).sqrt() * eps
            noise = sigma * torch.randn_like(x) if i < len(timestep_seq) - 1 else 0

            x = alpha_prev.sqrt() * pred_x0 + dir_xt + noise

        x = self._apply_zero_mask(x)
        return x
