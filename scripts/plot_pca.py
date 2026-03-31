#!/usr/bin/env python3
"""PCA visualization: Real vs Synthetic genotype samples (PC1 vs PC2).

Produces two figures:
  1. Superpopulation-level coloring (Real vs Synthetic side-by-side)
  2. Combined overlay plot
"""

from __future__ import annotations

import os
import pickle
import sys

import numpy as np
import torch
from pathlib import Path
from sklearn.decomposition import PCA

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _PROJECT_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ── Config ──
REAL_TEST_PATH = "data/processed/test_data.pkl"
REAL_TRAIN_PATH = "data/processed/train_data.pkl"
SYN_DIR = "outputs/default/synthetic_samples"
LABEL_HIERARCHY_PATH = "data/processed/label_hierarchy.pkl"
NORM_STATS_PATH = "data/processed/normalization_stats.pkl"
OUTPUT_DIR = "outputs/default/figures"

SUPERPOP_COLORS = {
    "AFR": "#E41A1C",
    "AMR": "#FF7F00",
    "EAS": "#4DAF4A",
    "EUR": "#377EB8",
    "SAS": "#984EA3",
}


def denormalize(x: np.ndarray, stats_path: str) -> np.ndarray:
    """Denormalize (N, gene_size, K) data using saved stats."""
    if not Path(stats_path).exists():
        return x
    with open(stats_path, "rb") as f:
        stats = pickle.load(f)
    xmean = stats["mean"]  # (gene_size_stats, K)
    xstd = stats["std"]
    gene_dim = x.shape[1]
    stats_dim = xmean.shape[0]
    if stats_dim < gene_dim:
        pad = gene_dim - stats_dim
        n_k = xmean.shape[1]
        xmean = np.concatenate([xmean, np.zeros((pad, n_k), dtype=xmean.dtype)], axis=0)
        xstd = np.concatenate([xstd, np.ones((pad, n_k), dtype=xstd.dtype)], axis=0)
    return x * xstd + xmean


def load_and_flatten(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load pkl data, denormalize, and flatten to 2D."""
    with open(path, "rb") as f:
        x, y = pickle.load(f)
    x = np.asarray(x, dtype=np.float32)
    # Denormalize to match synthetic data scale
    x = denormalize(x, NORM_STATS_PATH)
    # (N, gene_size, K) or (N, K, gene_size) -> flatten to (N, -1)
    return x.reshape(x.shape[0], -1), np.asarray(y, dtype=np.int64)


def load_synthetic(syn_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """Load synthetic .pt files."""
    pt_files = sorted(Path(syn_dir).glob("sample_pop*.pt"))
    samples, labels = [], []
    for f in pt_files:
        genome, label = torch.load(f, map_location="cpu", weights_only=True)
        samples.append(genome.numpy())
        labels.append(label.item() if label.dim() == 0 else label.numpy())
    syn_x = np.stack(samples).astype(np.float32)
    return syn_x.reshape(syn_x.shape[0], -1), np.array(labels, dtype=np.int64)


def pop_to_superpop_name(pop_idx: int, lh: dict) -> str:
    superpop_idx = lh["pop_to_superpop"][pop_idx]
    return lh["idx_to_superpop"][superpop_idx]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load label hierarchy
    with open(LABEL_HIERARCHY_PATH, "rb") as f:
        lh = pickle.load(f)

    # Load real data (train + test combined for PCA fitting)
    real_train_x, real_train_y = load_and_flatten(REAL_TRAIN_PATH)
    real_test_x, real_test_y = load_and_flatten(REAL_TEST_PATH)
    real_x = np.concatenate([real_train_x, real_test_x], axis=0)
    real_y = np.concatenate([real_train_y, real_test_y], axis=0)

    # Load synthetic data
    syn_x, syn_y = load_synthetic(SYN_DIR)

    print(f"Real: {real_x.shape}, Synthetic: {syn_x.shape}")

    # Fit PCA on real data, transform both
    pca = PCA(n_components=2, random_state=42)
    real_pcs = pca.fit_transform(real_x)
    syn_pcs = pca.transform(syn_x)

    ev = pca.explained_variance_ratio_
    print(f"PC1: {ev[0]*100:.1f}%, PC2: {ev[1]*100:.1f}%")

    # Map labels to superpop names
    real_superpops = np.array([pop_to_superpop_name(y, lh) for y in real_y])
    syn_superpops = np.array([pop_to_superpop_name(y, lh) for y in syn_y])

    # ── Figure 1: Side-by-side ──
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=150)

    for ax, pcs, spops, title in [
        (axes[0], real_pcs, real_superpops, "Real (1000 Genomes)"),
        (axes[1], syn_pcs, syn_superpops, "Synthetic (HybridGenoDiT)"),
    ]:
        for sp_name in ["AFR", "EUR", "EAS", "SAS", "AMR"]:
            mask = spops == sp_name
            ax.scatter(
                pcs[mask, 0], pcs[mask, 1],
                c=SUPERPOP_COLORS[sp_name],
                label=sp_name,
                s=8, alpha=0.6, edgecolors="none",
            )
        ax.set_xlabel(f"PC1 ({ev[0]*100:.1f}%)", fontsize=12)
        ax.set_ylabel(f"PC2 ({ev[1]*100:.1f}%)", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(fontsize=10, markerscale=3)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path1 = os.path.join(OUTPUT_DIR, "pca_real_vs_synthetic.png")
    fig.savefig(path1, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path1}")

    # ── Figure 2: Overlay ──
    fig, ax = plt.subplots(figsize=(9, 7), dpi=150)

    # Real (hollow markers)
    for sp_name in ["AFR", "EUR", "EAS", "SAS", "AMR"]:
        mask = real_superpops == sp_name
        ax.scatter(
            real_pcs[mask, 0], real_pcs[mask, 1],
            facecolors="none", edgecolors=SUPERPOP_COLORS[sp_name],
            label=f"{sp_name} (Real)",
            s=15, alpha=0.5, linewidths=0.8,
        )

    # Synthetic (filled markers)
    for sp_name in ["AFR", "EUR", "EAS", "SAS", "AMR"]:
        mask = syn_superpops == sp_name
        ax.scatter(
            syn_pcs[mask, 0], syn_pcs[mask, 1],
            c=SUPERPOP_COLORS[sp_name],
            label=f"{sp_name} (Syn)",
            s=10, alpha=0.6, marker="x", linewidths=0.8,
        )

    ax.set_xlabel(f"PC1 ({ev[0]*100:.1f}%)", fontsize=12)
    ax.set_ylabel(f"PC2 ({ev[1]*100:.1f}%)", fontsize=12)
    ax.set_title("PCA Overlay: Real vs Synthetic", fontsize=14, fontweight="bold")
    ax.legend(fontsize=8, ncol=2, markerscale=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path2 = os.path.join(OUTPUT_DIR, "pca_overlay.png")
    fig.savefig(path2, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path2}")


if __name__ == "__main__":
    main()
