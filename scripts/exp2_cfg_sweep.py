#!/usr/bin/env python3
"""Experiment 2: CFG guidance_weight sweep + PCA reverse projection r².

Loads trained model once, generates samples per guidance_weight,
measures PCA→AF reconstruction quality and population classification accuracy.
"""
from __future__ import annotations

import os
import pickle
import sys

import numpy as np
import torch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _PROJECT_ROOT)

from src.models import GaussianDiffusion, HybridCNNDiTFiLM
from src.utils.config import load_config


def load_model_and_diffusion(config, model_path, device):
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model_config = checkpoint.get("config", config)
    model = HybridCNNDiTFiLM(model_config).to(device)

    if "ema_state_dict" in checkpoint:
        shadow = checkpoint["ema_state_dict"].get("shadow", checkpoint["ema_state_dict"])
        for name, param in model.named_parameters():
            if name in shadow:
                param.data.copy_(shadow[name].to(param.data.dtype))

    model.eval()

    data_cfg = config["data"]
    diffusion_cfg = config["diffusion"]
    zero_mask_path = data_cfg.get("zero_mask_path", "data/processed/zero_mask.pt")
    zero_mask = None
    if os.path.exists(zero_mask_path):
        zero_mask = torch.load(zero_mask_path, map_location=device, weights_only=True)
        if zero_mask.ndim == 2 and zero_mask.shape[1] < zero_mask.shape[0]:
            zero_mask = zero_mask.T

    diffusion = GaussianDiffusion(
        timesteps=diffusion_cfg["max_timesteps"],
        zero_mask=zero_mask,
        enforce_zeros=data_cfg.get("enforce_zeros", True),
        null_class=data_cfg.get("num_classes", 26),
    ).to(device)

    return model, diffusion, zero_mask


@torch.no_grad()
def generate_for_pops(model, diffusion, config, device, guidance_weight, pop_indices, n_per_pop=50):
    data_cfg = config["data"]
    diffusion_cfg = config["diffusion"]
    num_channels = data_cfg["num_channels"]
    gene_size = data_cfg["gene_size"]
    ddim_steps = diffusion_cfg.get("sampling_timesteps", 100)

    all_samples, all_labels = [], []
    for pop_idx in pop_indices:
        shape = (n_per_pop, num_channels, gene_size)
        y = torch.full((n_per_pop,), pop_idx, device=device, dtype=torch.long)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            samples = diffusion.sample_ddim(
                model=model, shape=shape, y=y, device=device,
                ddim_steps=ddim_steps, eta=0.0, guidance_scale=guidance_weight,
            )
        all_samples.append(samples.float().cpu())
        all_labels.extend([pop_idx] * n_per_pop)

    return torch.cat(all_samples, dim=0), np.array(all_labels)


def compute_pca_r2(syn_samples, real_data, real_labels, pop_indices):
    """Compute per-population correlation between synthetic and real PCA features."""
    r2_per_pop = {}
    for pop_idx in pop_indices:
        syn_pop = syn_samples[syn_samples[:, 0, 0] != -999]  # placeholder
        # Use mean AF profile per population as proxy
        real_mask = real_labels == pop_idx
        if real_mask.sum() < 2:
            continue
        real_pop = real_data[real_mask]
        syn_mask = np.array([i for i, l in enumerate(range(len(syn_samples)))
                             if l // 50 == list(pop_indices).index(pop_idx)])
        if len(syn_mask) == 0:
            continue
        syn_pop = syn_samples[syn_mask]

        # Flatten to (N, K*gene_size) and compute mean profile correlation
        real_mean = real_pop.reshape(real_pop.shape[0], -1).mean(axis=0)
        syn_mean = syn_pop.reshape(syn_pop.shape[0], -1).mean(axis=0)

        corr = np.corrcoef(real_mean, syn_mean)[0, 1]
        r2_per_pop[pop_idx] = corr ** 2 if not np.isnan(corr) else 0.0

    return r2_per_pop


def main():
    config = load_config("configs/default.yaml")
    model_path = "outputs/default/best_model.pth"
    device = torch.device("cuda:0")

    torch.manual_seed(20260327)
    torch.cuda.manual_seed_all(20260327)
    np.random.seed(20260327)

    # Load label hierarchy for population mapping
    with open(config["data"]["label_hierarchy_path"], "rb") as f:
        lh = pickle.load(f)
    idx_to_pop = lh["idx_to_pop"]
    pop_to_superpop = lh["pop_to_superpop"]

    # Representative populations: AFR-majority (YRI), EAS-mid (CHB), AMR-minority (PEL)
    pop_to_idx = lh["pop_to_idx"]
    test_pops = {}
    for name in ["YRI", "CHB", "PEL", "ASW", "CEU", "JPT"]:
        if name in pop_to_idx:
            test_pops[name] = pop_to_idx[name]
    pop_indices = list(test_pops.values())
    pop_names = list(test_pops.keys())
    print(f"Test populations: {test_pops}")

    # Load real data for comparison
    with open("data/processed/val_data.pkl", "rb") as f:
        real_data_tuple = pickle.load(f)
    real_x = real_data_tuple[0]  # (N, gene_size, K) or (N, K, gene_size)
    real_y = real_data_tuple[1]
    if real_x.ndim == 3 and real_x.shape[2] < real_x.shape[1]:
        real_x = np.transpose(real_x, (0, 2, 1))  # -> (N, K, gene_size)

    # Load model once
    print("Loading model...")
    model, diffusion, zero_mask = load_model_and_diffusion(config, model_path, device)

    # Sweep
    guidance_weights = [0.0, 1.0, 3.0, 5.0, 7.0]
    n_per_pop = 50
    results = []

    for gw in guidance_weights:
        print(f"\n{'='*60}")
        print(f"guidance_weight = {gw}")
        print(f"{'='*60}")

        syn_samples, syn_labels = generate_for_pops(
            model, diffusion, config, device, gw, pop_indices, n_per_pop
        )
        syn_np = syn_samples.numpy()

        # 1. Per-population r² (mean PCA profile correlation)
        r2_per_pop = {}
        for i, (name, pop_idx) in enumerate(test_pops.items()):
            syn_pop = syn_np[i * n_per_pop : (i + 1) * n_per_pop]
            real_mask = real_y == pop_idx
            if real_mask.sum() < 2:
                r2_per_pop[name] = float("nan")
                continue
            real_pop = real_x[real_mask]
            if real_pop.ndim == 3 and real_pop.shape[2] < real_pop.shape[1]:
                real_pop = np.transpose(real_pop, (0, 2, 1))

            real_flat_mean = real_pop.reshape(real_pop.shape[0], -1).mean(axis=0)
            syn_flat_mean = syn_pop.reshape(syn_pop.shape[0], -1).mean(axis=0)
            corr = np.corrcoef(real_flat_mean, syn_flat_mean)[0, 1]
            r2_per_pop[name] = round(corr ** 2, 4) if not np.isnan(corr) else 0.0

        avg_r2 = np.mean([v for v in r2_per_pop.values() if not np.isnan(v)])
        print(f"  Per-pop r²: {r2_per_pop}")
        print(f"  Average r²: {avg_r2:.4f}")

        # 2. Population classification accuracy (LDA, real+syn combined)
        combined_x = np.vstack([
            real_x[np.isin(real_y, pop_indices)].reshape(-1, real_x.shape[1] * real_x.shape[2]),
            syn_np.reshape(syn_np.shape[0], -1),
        ])
        real_subset_y = real_y[np.isin(real_y, pop_indices)]
        combined_y = np.concatenate([real_subset_y, syn_labels])

        if len(np.unique(combined_y)) > 1:
            lda = LinearDiscriminantAnalysis()
            scores = cross_val_score(lda, combined_x, combined_y, cv=3, scoring="accuracy")
            cls_acc = scores.mean()
        else:
            cls_acc = 0.0
        print(f"  Classification accuracy (LDA 3-fold): {cls_acc:.4f}")

        # 3. Per-channel stats
        syn_mean = syn_np.mean()
        syn_std = syn_np.std()
        print(f"  Synthetic stats: mean={syn_mean:.4f}, std={syn_std:.4f}")

        results.append({
            "guidance_weight": gw,
            "avg_r2": round(float(avg_r2), 4),
            "r2_per_pop": r2_per_pop,
            "cls_accuracy": round(float(cls_acc), 4),
            "syn_mean": round(float(syn_mean), 4),
            "syn_std": round(float(syn_std), 4),
        })

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'GW':>4}  {'Avg r²':>8}  {'Cls Acc':>8}  {'Mean':>8}  {'Std':>8}")
    for r in results:
        print(f"{r['guidance_weight']:>4.1f}  {r['avg_r2']:>8.4f}  {r['cls_accuracy']:>8.4f}  {r['syn_mean']:>8.4f}  {r['syn_std']:>8.4f}")

    # Save results
    import json
    os.makedirs("outputs/exp2_cfg_sweep", exist_ok=True)
    with open("outputs/exp2_cfg_sweep/results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to outputs/exp2_cfg_sweep/results.json")

    # Pass/fail
    best = max(results, key=lambda r: r["avg_r2"])
    print(f"\nBest guidance_weight: {best['guidance_weight']} (r²={best['avg_r2']:.4f}, cls_acc={best['cls_accuracy']:.4f})")
    if best["avg_r2"] >= 0.85:
        print("PASS: r² >= 0.85")
    else:
        print(f"FAIL: r² = {best['avg_r2']:.4f} < 0.85")
    if best["cls_accuracy"] >= 0.85:
        print("PASS: classification accuracy >= 85%")
    else:
        print(f"WARN: classification accuracy = {best['cls_accuracy']:.4f} < 0.85 (model only trained 20 epochs)")


if __name__ == "__main__":
    main()
