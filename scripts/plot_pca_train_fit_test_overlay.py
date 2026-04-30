#!/usr/bin/env python3
"""Fit PCA on train (real) then project test (real) and synthetic through it.

Shows three panels in the SAME PC1/PC2 space so the test compression vs
train spread is directly visible.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

import sys
sys.path.insert(0, str(Path(__file__).parent))
from plot_pca import (  # type: ignore
    apply_nonzero_mask,
    load_real,
    load_synthetic,
    pop_idx_to_superpop_name,
)


SUPERPOP_COLORS = {
    "AFR": "#e74c3c",
    "EUR": "#3498db",
    "EAS": "#2ecc71",
    "SAS": "#9b59b6",
    "AMR": "#e67e22",
}


def panel(ax, pcs, sp_labels, title, xlim, ylim, alpha=0.6, marker="o", size=10):
    for sp, color in SUPERPOP_COLORS.items():
        mask = sp_labels == sp
        if mask.sum() == 0:
            continue
        ax.scatter(
            pcs[mask, 0], pcs[mask, 1],
            s=size, c=color, alpha=alpha, marker=marker,
            label=f"{sp} ({mask.sum()})",
        )
    ax.set_title(title)
    ax.set_xlabel(f"PC1")
    ax.set_ylabel(f"PC2")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--syn-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open("data/processed/label_hierarchy.pkl", "rb") as f:
        lh = pickle.load(f)
    zero_mask = torch.load(
        "data/processed/zero_mask.pt", map_location="cpu", weights_only=True
    ).numpy()

    train_x, train_y = load_real(["data/processed/train_data.pkl"], "data/processed/normalization_stats.pkl")
    test_x, test_y = load_real(["data/processed/test_data.pkl"], "data/processed/normalization_stats.pkl")
    syn_x, syn_y = load_synthetic(args.syn_dir)

    train_flat = apply_nonzero_mask(train_x, zero_mask)
    test_flat = apply_nonzero_mask(test_x, zero_mask)
    syn_flat = apply_nonzero_mask(syn_x, zero_mask)

    pca = PCA(n_components=2, random_state=42)
    train_pcs = pca.fit_transform(train_flat)
    test_pcs = pca.transform(test_flat)
    syn_pcs = pca.transform(syn_flat)
    ev = pca.explained_variance_ratio_
    print(f"PCA fit on train. PC1 {ev[0]*100:.1f}%, PC2 {ev[1]*100:.1f}%")
    print(f"Train PC range: PC1 [{train_pcs[:,0].min():.1f},{train_pcs[:,0].max():.1f}], PC2 [{train_pcs[:,1].min():.1f},{train_pcs[:,1].max():.1f}]")
    print(f"Test  PC range: PC1 [{test_pcs[:,0].min():.1f},{test_pcs[:,0].max():.1f}], PC2 [{test_pcs[:,1].min():.1f},{test_pcs[:,1].max():.1f}]")
    print(f"Syn   PC range: PC1 [{syn_pcs[:,0].min():.1f},{syn_pcs[:,0].max():.1f}], PC2 [{syn_pcs[:,1].min():.1f},{syn_pcs[:,1].max():.1f}]")

    train_sp = pop_idx_to_superpop_name(train_y, lh)
    test_sp = pop_idx_to_superpop_name(test_y, lh)
    syn_sp = pop_idx_to_superpop_name(syn_y, lh)

    all_pcs = np.vstack([train_pcs, test_pcs, syn_pcs])
    pad_x = (all_pcs[:, 0].max() - all_pcs[:, 0].min()) * 0.05
    pad_y = (all_pcs[:, 1].max() - all_pcs[:, 1].min()) * 0.05
    xlim = (all_pcs[:, 0].min() - pad_x, all_pcs[:, 0].max() + pad_x)
    ylim = (all_pcs[:, 1].min() - pad_y, all_pcs[:, 1].max() + pad_y)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    panel(axes[0], train_pcs, train_sp, f"Real Train ({len(train_pcs)})", xlim, ylim, alpha=0.4, size=8)
    panel(axes[1], test_pcs, test_sp, f"Real Test ({len(test_pcs)})", xlim, ylim, alpha=0.7, size=15)
    panel(axes[2], syn_pcs, syn_sp, f"Synthetic ({len(syn_pcs)})", xlim, ylim, alpha=0.5, marker="x", size=15)

    fig.suptitle(
        f"PCA fit on train, projected: train | test | synthetic — "
        f"PC1 {ev[0]*100:.1f}%, PC2 {ev[1]*100:.1f}%",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
