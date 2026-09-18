from __future__ import annotations

import math

import torch
from torch import nn

from src.models.noise_schedule import (
    cosine_beta_schedule,
    linear_beta_schedule,
    three_band_betas,
)


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        timesteps: int = 500,
        zero_mask: torch.Tensor | None = None,
        enforce_zeros: bool = True,
        min_snr_gamma: float = 5.0,
        null_class: int = 26,
        cfg_dropout_rate: float = 0.1,
        schedule_type: str = "cosine",
        sample_clip: float | None = 6.0,
        feature_schedule: torch.Tensor | list[list[int]] | None = None,
    ):
        super().__init__()
        if timesteps < 1:
            raise ValueError("timesteps must be positive")
        if sample_clip is not None and (not math.isfinite(sample_clip) or sample_clip <= 0):
            raise ValueError("sample_clip must be positive or null")
        if not math.isfinite(min_snr_gamma) or min_snr_gamma <= 0:
            raise ValueError("min_snr_gamma must be finite and positive")
        if not 0 <= cfg_dropout_rate <= 1:
            raise ValueError("cfg_dropout_rate must be between zero and one")
        self.timesteps = timesteps
        self.enforce_zeros_flag = enforce_zeros
        self.min_snr_gamma = min_snr_gamma
        self.null_class = null_class
        self.cfg_dropout_rate = cfg_dropout_rate
        self.sample_clip = sample_clip
        groups = None
        if feature_schedule is not None:
            groups = torch.as_tensor(feature_schedule)
            choices = torch.tensor([0, 1, 2], device=groups.device)
            if groups.ndim != 2 or groups.numel() == 0 or not torch.isin(groups, choices).all():
                raise ValueError("feature_schedule must be a (K, genes) map of 0, 1, 2")
            groups = groups.long()
        self.register_buffer("feature_schedule", groups)
        self.register_buffer("zero_mask", zero_mask.bool() if zero_mask is not None else None)
        match schedule_type:
            case "linear":
                betas = linear_beta_schedule(timesteps)
            case "cosine":
                betas = cosine_beta_schedule(timesteps)
            case _:
                raise ValueError(f"Unknown schedule_type: {schedule_type}")
        if groups is not None:
            betas = three_band_betas(betas)
        cumulative = torch.cumprod(1 - betas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", cumulative)
        self.register_buffer("sqrt_alphas_cumprod", cumulative.sqrt())
        self.register_buffer("sqrt_one_minus_alphas_cumprod", (1 - cumulative).sqrt())
        snr = cumulative / (1 - cumulative)
        self.register_buffer("min_snr_weights", (min_snr_gamma / snr).clamp(max=1))

    def _extract(self, values: torch.Tensor, t: torch.Tensor, shape: tuple) -> torch.Tensor:
        selected = values.index_select(0, t)
        if self.feature_schedule is not None:
            if tuple(self.feature_schedule.shape) != tuple(shape[1:]):
                raise ValueError("Feature schedule shape does not match model coordinates")
            return selected[:, self.feature_schedule]
        return selected.reshape(len(t), *((1,) * (len(shape) - 1)))

    def _apply_zero_mask(self, x: torch.Tensor) -> torch.Tensor:
        if self.enforce_zeros_flag and self.zero_mask is not None:
            return x.masked_fill(self.zero_mask, 0)
        return x

    def _clip_x0(self, x: torch.Tensor) -> torch.Tensor:
        return x if self.sample_clip is None else x.clamp(-self.sample_clip, self.sample_clip)

    def q_sample(
        self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)
        signal = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        amplitude = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return self._apply_zero_mask(signal * x_start + amplitude * noise)

    def apply_cfg_dropout(self, labels: torch.Tensor) -> torch.Tensor:
        dropped = labels.clone()
        dropped[torch.rand(len(labels), device=labels.device) < self.cfg_dropout_rate] = self.null_class
        return dropped

    def p_losses(
        self, model: nn.Module, x_start: torch.Tensor, t: torch.Tensor, y: torch.Tensor,
        noise: torch.Tensor | None = None, use_min_snr: bool = True, cfg_training: bool = True,
    ) -> dict[str, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise)
        if cfg_training and self.training:
            y = self.apply_cfg_dropout(y)
        predicted = model(x_t, t, y)
        squared_error = (predicted - noise).square()
        weighted = squared_error * self._extract(self.min_snr_weights, t, x_start.shape) if use_min_snr else squared_error
        if self.enforce_zeros_flag and self.zero_mask is not None:
            valid_count = (~self.zero_mask).sum().clamp(min=1) * len(x_start)
            mse = squared_error.masked_fill(self.zero_mask, 0).sum() / valid_count
            loss = weighted.masked_fill(self.zero_mask, 0).sum() / valid_count
        else:
            mse, loss = squared_error.mean(), weighted.mean()
        return {"loss": loss, "mse": mse.detach(), "pred_noise": predicted}

    def _predict_x0_from_eps(
        self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor,
    ) -> torch.Tensor:
        signal = self._extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        amplitude = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        return (x_t - amplitude * eps) / signal

    def _predict_noise(
        self, model: nn.Module, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor, guidance: float,
        *, guidance_interval: tuple[float, float] | None = None, guidance_alpha: float = 0.0,
        guide_model: nn.Module | None = None,
    ) -> torch.Tensor:
        if guidance <= 0:
            return model(x, t, y)
        if guidance_alpha < 0:
            raise ValueError("guidance_alpha must be non-negative")
        active = torch.ones_like(t, dtype=torch.bool)
        if guidance_interval is not None:
            low, high = guidance_interval
            if not 0 <= low < high <= 1:
                raise ValueError("guidance_interval must satisfy 0 <= low < high <= 1")
            fraction = t.float() / max(self.timesteps - 1, 1)
            active = (fraction >= low) & (fraction <= high)
        if not active.any():
            return model(x, t, y)
        if guide_model is None:
            # One batched forward: the conditional half and the null-class half.
            both = model(torch.cat((x, x)), torch.cat((t, t)),
                         torch.cat((y, torch.full_like(y, self.null_class))))
            conditional, guide = both.chunk(2)
        else:
            conditional, guide = model(x, t, y), guide_model(x, t, y)
        difference = conditional - guide
        if guidance_alpha > 0:
            rms = difference.flatten(1).norm(dim=1) / math.sqrt(difference[0].numel())
            difference = difference * rms.reshape(-1, *((1,) * (difference.ndim - 1))) ** guidance_alpha
        guided = conditional + guidance * difference
        return torch.where(active.reshape(-1, *((1,) * (x.ndim - 1))), guided, conditional)

    @torch.no_grad()
    def sample_ddim(
        self, model: nn.Module, shape: tuple, y: torch.Tensor, device: torch.device,
        ddim_steps: int = 50, eta: float = 0., guidance_scale: float = 0.,
        *, guidance_interval: tuple[float, float] | None = None, guidance_alpha: float = 0.0,
        guide_model: nn.Module | None = None,
    ) -> torch.Tensor:
        if not 1 <= ddim_steps <= self.timesteps or not 0 <= eta <= 1:
            raise ValueError("DDIM needs 1 <= steps <= timesteps and 0 <= eta <= 1")
        model.eval()
        if guide_model is not None:
            guide_model.eval()
        x = self._apply_zero_mask(torch.randn(shape, device=device))
        timesteps = torch.linspace(self.timesteps - 1, 0, ddim_steps).long().tolist()
        for index, t in enumerate(timesteps):
            times = torch.full((shape[0],), t, device=device, dtype=torch.long)
            eps = self._predict_noise(model, x, times, y, guidance_scale,
                                      guidance_interval=guidance_interval,
                                      guidance_alpha=guidance_alpha, guide_model=guide_model)
            alpha = self._extract(self.alphas_cumprod, times, shape)
            if index + 1 < len(timesteps):
                earlier = torch.full_like(times, timesteps[index + 1])
                previous = self._extract(self.alphas_cumprod, earlier, shape)
            else:
                previous = torch.ones_like(alpha)
            clean = self._apply_zero_mask(self._clip_x0(self._predict_x0_from_eps(x, times, eps)))
            sigma = eta * ((1 - previous) / (1 - alpha) * (1 - alpha / previous)).clamp(min=0).sqrt()
            direction = (1 - previous - sigma.square()).clamp(min=0).sqrt() * eps
            noise = sigma * torch.randn_like(x) if eta > 0 and index + 1 < len(timesteps) else 0
            x = self._apply_zero_mask(previous.sqrt() * clean + direction + noise)
        return x
