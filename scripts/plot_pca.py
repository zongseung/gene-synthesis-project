#!/usr/bin/env python3
"""PCA visualization: Real vs Synthetic genotype samples (3-panel figure).

Produces a single figure with three panels:

    [ Real ]   [ Synthetic (GW=<gw>) ]   [ Overlay (o=Real, x=Synthetic) ]

Features are restricted to **non-zero positions** using ``zero_mask.pt`` so
PCA fits only on genomic positions that actually carry signal (padding and
always-zero positions are dropped).

Output:
    outputs/<run>/pca_real_vs_synthetic.png

Usage::

    uv run python scripts/plot_pca.py
    uv run python scripts/plot_pca.py --run-dir outputs/default --guidance-weight 7.0
    uv run python scripts/plot_pca.py --title "HiPoDiT v1 — 500ep Linear 197M"
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch
import yaml
from sklearn.decomposition import PCA

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _PROJECT_ROOT)


# ── Constants ────────────────────────────────────────────────────────────
SUPERPOP_COLORS = {
    "AFR": "#E41A1C",
    "AMR": "#FF7F00",
    "EAS": "#4DAF4A",
    "EUR": "#377EB8",
    "SAS": "#984EA3",
}
SUPERPOP_ORDER = ["AFR", "EUR", "EAS", "SAS", "AMR"]


# ── Data loading ─────────────────────────────────────────────────────────
def load_real(
    paths: list[str],
    stats_path: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Load real pkl datasets, concatenate, and denormalize.

    Real data layout in the pickle is ``(N, gene_size, K)`` — stored normalized.
    Returns (x, y) with x shape ``(N, gene_size, K)`` in the original scale.
    """
    with open(stats_path, "rb") as f:
        stats = pickle.load(f)
    mean = stats["mean"].astype(np.float32)  # (gene_size, K)
    std = stats["std"].astype(np.float32)

    xs, ys = [], []
    for p in paths:
        with open(p, "rb") as f:
            x, y = pickle.load(f)
        x = np.asarray(x, dtype=np.float32)
        if x.shape[1:] != mean.shape:
            raise ValueError(
                f"Shape mismatch in {p}: got {x.shape[1:]}, expected {mean.shape}"
            )
        xs.append(x * std + mean)
        ys.append(np.asarray(y, dtype=np.int64))
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def load_synthetic(syn_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """Load synthetic .pt files produced by ``src/inference/generator.py``.

    Each file holds ``(genome, label)`` where ``genome`` has shape
    ``(K, gene_size)`` and is already denormalized. We transpose to
    ``(gene_size, K)`` so the layout matches the real tensor.
    """
    pt_files = sorted(Path(syn_dir).glob("sample_pop*.pt"))
    if not pt_files:
        raise FileNotFoundError(f"No sample_pop*.pt under {syn_dir}")
    samples, labels = [], []
    for f in pt_files:
        genome, label = torch.load(f, map_location="cpu", weights_only=True)
        arr = genome.numpy().astype(np.float32)
        if arr.ndim == 2 and arr.shape[0] < arr.shape[1]:
            arr = arr.T  # (K, gene) -> (gene, K)
        samples.append(arr)
        labels.append(int(label.item() if label.ndim == 0 else label.numpy().item()))
    return np.stack(samples), np.asarray(labels, dtype=np.int64)


def apply_nonzero_mask(x: np.ndarray, zero_mask: np.ndarray) -> np.ndarray:
    """Flatten (N, gene_size, K) → (N, F_nonzero) using the non-zero mask."""
    flat = x.reshape(x.shape[0], -1)  # (N, gene_size * K)
    keep = ~zero_mask.reshape(-1)
    return flat[:, keep]


def pop_idx_to_superpop_name(y: np.ndarray, lh: dict) -> np.ndarray:
    pop_to_super = lh["pop_to_superpop"]
    idx_to_super = lh["idx_to_superpop"]
    return np.array([idx_to_super[pop_to_super[int(p)]] for p in y])


# ── Title helpers ────────────────────────────────────────────────────────
def infer_title(run_dir: Path, config_path: Path, checkpoint_path: Path | None) -> str:
    """Build a default suptitle from config + checkpoint metadata.

    Format: ``HiPoDiT v1 — <epochs>ep <Schedule> <params>M — PCA (non-zero features)``
    """
    spec_parts: list[str] = []
    try:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        epochs = cfg.get("training", {}).get("epochs")
        schedule = cfg.get("diffusion", {}).get("noise_schedule", "")
        if epochs:
            spec_parts.append(f"{epochs}ep")
        if schedule:
            spec_parts.append(schedule.capitalize())
    except Exception:
        pass

    if checkpoint_path and checkpoint_path.exists():
        try:
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            sd = ckpt.get("model_state_dict", ckpt)
            n_params = sum(v.numel() for v in sd.values() if torch.is_tensor(v))
            spec_parts.append(f"{n_params / 1e6:.0f}M")
        except Exception:
            pass

    spec = " ".join(spec_parts)
    head = "HiPoDiT v1"
    if spec:
        return f"{head} — {spec} — PCA (non-zero features)"
    return f"{head} — PCA (non-zero features)"


# ── Plotting ─────────────────────────────────────────────────────────────
def plot_three_panel(
    real_pcs: np.ndarray,
    real_sp: np.ndarray,
    syn_pcs: np.ndarray,
    syn_sp: np.ndarray,
    ev: np.ndarray,
    guidance_weight: float,
    suptitle: str,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=150)
    fig.suptitle(suptitle, fontsize=13, y=1.02)

    # -- Panel 1: Real --
    ax = axes[0]
    for sp in SUPERPOP_ORDER:
        mask = real_sp == sp
        if not mask.any():
            continue
        ax.scatter(
            real_pcs[mask, 0],
            real_pcs[mask, 1],
            c=SUPERPOP_COLORS[sp],
            s=12,
            alpha=0.7,
            edgecolors="none",
            label=f"{sp} ({mask.sum()})",
        )
    ax.set_title("Real Data", fontsize=13, fontweight="bold")
    ax.set_xlabel(f"PC1 ({ev[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({ev[1] * 100:.1f}%)")
    ax.legend(fontsize=9, loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # -- Panel 2: Synthetic --
    ax = axes[1]
    for sp in SUPERPOP_ORDER:
        mask = syn_sp == sp
        if not mask.any():
            continue
        ax.scatter(
            syn_pcs[mask, 0],
            syn_pcs[mask, 1],
            c=SUPERPOP_COLORS[sp],
            s=12,
            alpha=0.7,
            edgecolors="none",
            label=f"{sp} ({mask.sum()})",
        )
    ax.set_title(f"Synthetic (GW={guidance_weight})", fontsize=13, fontweight="bold")
    ax.set_xlabel(f"PC1 ({ev[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({ev[1] * 100:.1f}%)")
    ax.legend(fontsize=9, loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # -- Panel 3: Overlay --
    ax = axes[2]
    for sp in SUPERPOP_ORDER:
        m_r = real_sp == sp
        if m_r.any():
            ax.scatter(
                real_pcs[m_r, 0],
                real_pcs[m_r, 1],
                c=SUPERPOP_COLORS[sp],
                s=10,
                alpha=0.35,
                marker="o",
                edgecolors="none",
            )
        m_s = syn_sp == sp
        if m_s.any():
            ax.scatter(
                syn_pcs[m_s, 0],
                syn_pcs[m_s, 1],
                c=SUPERPOP_COLORS[sp],
                s=30,
                alpha=0.9,
                marker="x",
                linewidths=1.5,
            )
    # Legend: color swatches for pops + marker style for real/syn
    pop_handles = [
        plt.Line2D(
            [0], [0], marker="s", linestyle="",
            markerfacecolor=SUPERPOP_COLORS[sp], markeredgecolor="none",
            markersize=9, label=sp,
        )
        for sp in SUPERPOP_ORDER
    ]
    style_handles = [
        plt.Line2D(
            [0], [0], marker="o", linestyle="", color="gray",
            markerfacecolor="gray", markersize=8, alpha=0.5, label="Real",
        ),
        plt.Line2D(
            [0], [0], marker="x", linestyle="", color="black",
            markersize=9, markeredgewidth=1.5, label="Synthetic",
        ),
    ]
    ax.legend(
        handles=pop_handles + style_handles,
        fontsize=8, ncol=2, loc="best", framealpha=0.9,
    )
    ax.set_title("Overlay (o=Real, x=Synthetic)", fontsize=13, fontweight="bold")
    ax.set_xlabel(f"PC1 ({ev[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({ev[1] * 100:.1f}%)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="outputs/default", help="Run directory.")
    parser.add_argument(
        "--syn-dir",
        default=None,
        help="Synthetic samples directory (default: <run-dir>/synthetic_samples).",
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--real-splits",
        nargs="+",
        default=["test"],
        choices=["train", "val", "test"],
        help="Which real splits to include (default: test only, ~251 samples).",
    )
    parser.add_argument(
        "--guidance-weight",
        type=float,
        default=7.0,
        help="Guidance weight label for the synthetic panel title.",
    )
    parser.add_argument("--title", default=None, help="Override figure suptitle.")
    parser.add_argument(
        "--out",
        default=None,
        help="Output path (default: <run-dir>/pca_real_vs_synthetic.png).",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    syn_dir = Path(args.syn_dir or run_dir / "synthetic_samples")
    out_path = Path(args.out or run_dir / "pca_real_vs_synthetic.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # -- Load inputs --
    with open("data/processed/label_hierarchy.pkl", "rb") as f:
        lh = pickle.load(f)

    zero_mask = torch.load(
        "data/processed/zero_mask.pt", map_location="cpu", weights_only=True
    ).numpy()

    real_paths = [f"data/processed/{s}_data.pkl" for s in args.real_splits]
    real_x, real_y = load_real(
        real_paths, "data/processed/normalization_stats.pkl"
    )
    syn_x, syn_y = load_synthetic(str(syn_dir))

    print(f"Real: {real_x.shape}, Synthetic: {syn_x.shape}")
    print(f"Zero mask: {zero_mask.sum()}/{zero_mask.size} positions masked out")

    # -- Apply non-zero mask + flatten --
    real_flat = apply_nonzero_mask(real_x, zero_mask)
    syn_flat = apply_nonzero_mask(syn_x, zero_mask)
    print(f"Non-zero features: {real_flat.shape[1]}")

    # -- PCA: fit on real, transform both --
    pca = PCA(n_components=2, random_state=42)
    real_pcs = pca.fit_transform(real_flat)
    syn_pcs = pca.transform(syn_flat)
    ev = pca.explained_variance_ratio_
    print(f"PC1: {ev[0] * 100:.1f}%  PC2: {ev[1] * 100:.1f}%")

    # -- Superpop labels --
    real_sp = pop_idx_to_superpop_name(real_y, lh)
    syn_sp = pop_idx_to_superpop_name(syn_y, lh)

    # -- Title --
    suptitle = args.title or infer_title(
        run_dir=run_dir,
        config_path=Path(args.config),
        checkpoint_path=run_dir / "best_model.pth",
    )
    print(f"Title: {suptitle}")

    # -- Plot --
    plot_three_panel(
        real_pcs=real_pcs,
        real_sp=real_sp,
        syn_pcs=syn_pcs,
        syn_sp=syn_sp,
        ev=ev,
        guidance_weight=args.guidance_weight,
        suptitle=suptitle,
        out_path=out_path,
    )
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
