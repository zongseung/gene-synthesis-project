"""Training loss functions for HybridGenoDiT diffusion model.

Provides:
- masked_mse_loss: MSE ignoring zero_mask (padding) positions.
- mmd_loss: Maximum Mean Discrepancy with RBF kernel (auxiliary).
- min_snr_weight: Min-SNR-gamma per-timestep weighting (Hang et al., 2023).
- compute_training_loss: Orchestrator that combines noise addition, model
  forward pass, and loss computation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    zero_mask: torch.Tensor,
) -> torch.Tensor:
    """MSE loss excluding zero_mask positions (padding / biologically zero).

    Args:
        pred: (B, K, gene_size) predicted noise.
        target: (B, K, gene_size) actual noise.
        zero_mask: (gene_size, K) or (K, gene_size) bool tensor where True
            means the position is always zero and should be excluded.

    Returns:
        Scalar masked MSE loss.
    """
    # Build not_zero_mask with shape (K, gene_size) matching model layout
    if zero_mask.shape[0] != pred.shape[1]:
        # Stored as (gene_size, K) on disk, transpose to (K, gene_size)
        not_zero = ~zero_mask.T.to(pred.device)
    else:
        not_zero = ~zero_mask.to(pred.device)

    # Expand to (B, K, gene_size)
    mask = not_zero.unsqueeze(0).expand_as(pred)
    diff = (pred - target) ** 2
    return (diff * mask).sum() / mask.sum().clamp(min=1)


def mmd_loss(
    x_real: torch.Tensor,
    x_gen: torch.Tensor,
    sigma: float = 1.0,
) -> torch.Tensor:
    """Maximum Mean Discrepancy with RBF kernel.

    Used as an auxiliary loss to encourage population-level PCA distribution
    matching between real and generated features.

    Args:
        x_real: (N, D) real features.
        x_gen: (M, D) generated features.
        sigma: RBF kernel bandwidth.

    Returns:
        Scalar MMD^2 loss.
    """
    xx = torch.cdist(x_real, x_real)
    yy = torch.cdist(x_gen, x_gen)
    xy = torch.cdist(x_real, x_gen)

    k_xx = torch.exp(-xx ** 2 / (2 * sigma ** 2))
    k_yy = torch.exp(-yy ** 2 / (2 * sigma ** 2))
    k_xy = torch.exp(-xy ** 2 / (2 * sigma ** 2))

    return k_xx.mean() + k_yy.mean() - 2 * k_xy.mean()


def min_snr_weight(
    timesteps: torch.Tensor,
    alphas_cumprod: torch.Tensor,
    gamma: float = 5.0,
) -> torch.Tensor:
    """Compute Min-SNR-gamma per-timestep loss weights (Hang et al., 2023).

    SNR(t) = alpha_bar_t / (1 - alpha_bar_t)
    weight(t) = min(SNR(t), gamma) / SNR(t)

    Args:
        timesteps: (B,) timestep indices.
        alphas_cumprod: (T,) cumulative product of alphas.
        gamma: SNR clamp value (default 5.0).

    Returns:
        (B,) per-sample weights.
    """
    alpha_bar = alphas_cumprod[timesteps]
    snr = alpha_bar / (1.0 - alpha_bar).clamp(min=1e-8)
    weights = torch.clamp(snr, max=gamma) / snr.clamp(min=1e-8)
    return weights


def compute_training_loss(
    model: nn.Module,
    diffusion: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    zero_mask: torch.Tensor | None,
    config: dict,
    current_epoch: int = 0,
) -> dict[str, torch.Tensor]:
    """Orchestrate noise addition, model forward, and loss computation.

    This function wraps diffusion.p_losses with optional masked MSE and
    auxiliary MMD loss support, all under bf16 autocast.

    Args:
        model: The denoising model (HybridCNNDiTFiLM).
        diffusion: GaussianDiffusion instance.
        x: (B, K, gene_size) clean data.
        y: (B,) population labels.
        zero_mask: (gene_size, K) bool mask or None.
        config: Nested config dict.
        current_epoch: Zero-based epoch index for aux-loss warmup.

    Returns:
        Dict with 'loss' (total), 'mse' (unweighted), and optionally
        'mmd' if auxiliary loss is enabled.
    """
    device = x.device
    batch_size = x.shape[0]

    # Sample random timesteps
    t = torch.randint(0, diffusion.timesteps, (batch_size,), device=device)

    # Compute main diffusion loss via GaussianDiffusion.p_losses
    use_min_snr = config.get("training", {}).get("use_min_snr", True)
    cfg_training = config.get("diffusion", {}).get("guidance_type", "normal") != "normal"

    # Pre-sample noise so we can reuse it for aux loss (avoid double forward)
    noise = torch.randn_like(x)
    x_t = diffusion.q_sample(x, t, noise)

    loss_dict = diffusion.p_losses(
        model=model,
        x_start=x,
        t=t,
        y=y,
        noise=noise,
        use_min_snr=use_min_snr,
        cfg_training=cfg_training,
    )

    total_loss = loss_dict["loss"]
    result = {
        "loss": total_loss,
        "mse": loss_dict["mse"],
    }

    # Auxiliary MMD loss (optional)
    aux_cfg = config.get("aux_loss", {})
    if aux_cfg.get("enabled", False):
        aux_type = aux_cfg.get("type", "mmd_rbf")
        if aux_type != "mmd_rbf":
            raise ValueError(f"Unsupported aux_loss.type: {aux_type}")

        lambda_pca = aux_cfg.get("lambda_pca_dist", 0.01)
        lambda_pop = aux_cfg.get("lambda_pop_structure", 0.0)
        warmup_epochs = max(int(aux_cfg.get("warmup_epochs", 0)), 0)
        if warmup_epochs > 0:
            warmup_scale = min(1.0, float(current_epoch + 1) / warmup_epochs)
        else:
            warmup_scale = 1.0

        # Reuse pred_noise from p_losses (same noise, same x_t, no double forward)
        pred_noise = loss_dict["pred_noise"]
        pred_x0 = diffusion._predict_x0_from_eps(x_t, t, pred_noise)
        pred_x0 = pred_x0.clamp(-6, 6)

        aux_total = torch.tensor(0.0, device=device, dtype=total_loss.dtype)

        if lambda_pca > 0:
            real_flat = x.reshape(batch_size, -1).float()
            gen_flat = pred_x0.reshape(batch_size, -1).float()
            pca_mmd = mmd_loss(real_flat, gen_flat, sigma=1.0)
            result["mmd_pca"] = pca_mmd.detach()
            aux_total = aux_total + lambda_pca * pca_mmd

        if lambda_pop > 0:
            unique_pops = y.unique()
            pop_mmd_vals = []
            for pop in unique_pops:
                pop_mask = y == pop
                if pop_mask.sum() < 2:
                    continue
                real_pop = x[pop_mask].reshape(pop_mask.sum(), -1).float()
                gen_pop = pred_x0[pop_mask].reshape(pop_mask.sum(), -1).float()
                pop_mmd_vals.append(mmd_loss(real_pop, gen_pop, sigma=1.0))

            if pop_mmd_vals:
                pop_mmd = torch.stack(pop_mmd_vals).mean()
                result["mmd_pop"] = pop_mmd.detach()
                aux_total = aux_total + lambda_pop * pop_mmd

        if aux_total.item() > 0:
            result["mmd"] = aux_total.detach()
            result["aux_warmup_scale"] = torch.tensor(
                warmup_scale,
                device=device,
                dtype=total_loss.dtype,
            )
            total_loss = total_loss + warmup_scale * aux_total
            result["loss"] = total_loss

    return result
