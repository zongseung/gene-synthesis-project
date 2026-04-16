"""Real vs synthetic PCA scatter comparison.

Loads real validation samples from data/processed/test_data.pkl and synthetic
samples from a directory of sample_popX_NNNN.pt files, fits PCA(2) on real,
projects both sets into that space, and saves a side-by-side scatter colored
by superpopulation.
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


def load_real(path: str) -> tuple[np.ndarray, np.ndarray]:
    with open(path, "rb") as f:
        X, y = pickle.load(f)
    # X: (N, G, C), y: (N,)
    return np.asarray(X), np.asarray(y)


def load_syn(syn_dir: str) -> tuple[np.ndarray, np.ndarray]:
    files = sorted(Path(syn_dir).glob("sample_pop*_*.pt"))
    if not files:
        raise FileNotFoundError(f"No sample_pop*_*.pt in {syn_dir}")
    xs, ys = [], []
    for f in files:
        x, y = torch.load(f, map_location="cpu", weights_only=False)
        xs.append(x.numpy())
        ys.append(int(y))
    X = np.stack(xs, axis=0)  # (N, C, G)
    X = X.transpose(0, 2, 1)  # -> (N, G, C) to match real
    return X, np.asarray(ys)


def subsample_genes(X: np.ndarray, n_genes: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    G = X.shape[1]
    if n_genes >= G:
        idx = np.arange(G)
    else:
        idx = rng.choice(G, size=n_genes, replace=False)
    return X[:, idx, :].reshape(X.shape[0], -1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--syn_dir", required=True)
    parser.add_argument("--real_path", default="data/processed/test_data.pkl")
    parser.add_argument("--hierarchy", default="data/processed/label_hierarchy.pkl")
    parser.add_argument("--out", default=None)
    parser.add_argument("--n_genes", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.out is None:
        args.out = str(Path(args.syn_dir).parent / "pca_real_vs_synthetic.png")

    print(f"[1/5] Loading real: {args.real_path}")
    X_real, y_real = load_real(args.real_path)
    print(f"      real shape: {X_real.shape}, labels: {y_real.shape}")

    print(f"[2/5] Loading synthetic: {args.syn_dir}")
    X_syn, y_syn = load_syn(args.syn_dir)
    print(f"      syn shape: {X_syn.shape}, labels: {y_syn.shape}")

    print(f"[3/5] Subsampling {args.n_genes} genes (seed={args.seed})")
    Xr = subsample_genes(X_real, args.n_genes, args.seed)
    Xs = subsample_genes(X_syn, args.n_genes, args.seed)
    print(f"      flattened: real {Xr.shape}, syn {Xs.shape}")

    print("[4/5] Fitting PCA(2) on real, projecting both")
    pca = PCA(n_components=2, random_state=args.seed)
    Zr = pca.fit_transform(Xr)
    Zs = pca.transform(Xs)
    ev = pca.explained_variance_ratio_
    print(f"      explained variance: PC1={ev[0]*100:.2f}%, PC2={ev[1]*100:.2f}%")

    print(f"[5/5] Plotting -> {args.out}")
    with open(args.hierarchy, "rb") as f:
        hier = pickle.load(f)
    pop2super = hier["pop_to_superpop"]
    super_labels = [hier["idx_to_superpop"][i] for i in range(len(hier["idx_to_superpop"]))]
    super_colors = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4", "#9467bd"]

    sr = np.array([pop2super[int(p)] for p in y_real])
    ss = np.array([pop2super[int(p)] for p in y_syn])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
    for sp, name in enumerate(super_labels):
        axes[0].scatter(Zr[sr == sp, 0], Zr[sr == sp, 1],
                        s=12, alpha=0.7, c=super_colors[sp], label=name,
                        edgecolors="none")
        axes[1].scatter(Zs[ss == sp, 0], Zs[ss == sp, 1],
                        s=12, alpha=0.7, c=super_colors[sp], label=name,
                        edgecolors="none")
    axes[0].set_title(f"Real test samples (n={len(y_real)})")
    axes[1].set_title(f"Synthetic samples (n={len(y_syn)})")
    axes[0].set_xlabel(f"PC1 ({ev[0]*100:.1f}%)")
    axes[1].set_xlabel(f"PC1 ({ev[0]*100:.1f}%)")
    axes[0].set_ylabel(f"PC2 ({ev[1]*100:.1f}%)")
    axes[0].legend(loc="best", fontsize=9, title="Superpopulation")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle("Real vs Synthetic — PCA alignment (PCA fit on real only)", fontsize=13)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
