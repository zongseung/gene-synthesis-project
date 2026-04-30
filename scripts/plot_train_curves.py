#!/usr/bin/env python3
"""Parse trainer text log and plot loss/val curves."""
from __future__ import annotations

import argparse
import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


STEP_RE = re.compile(r"\[step (\d+)\] (.+)")
KV_RE = re.compile(r"(\w+/[\w_]+)=([\-\deE\.\+]+)")
EPOCH_RE = re.compile(r"\[Epoch (\d+)/\d+\] avg_loss=([\d\.eE\-\+]+), val_rec=([\d\.eE\-\+]+)")


def parse(log_path: str):
    series = defaultdict(list)  # key -> list of (step, value)
    epoch_stats = []
    with open(log_path) as f:
        for line in f:
            m = EPOCH_RE.search(line)
            if m:
                epoch_stats.append(
                    (int(m.group(1)), float(m.group(2)), float(m.group(3)))
                )
                continue
            m = STEP_RE.search(line)
            if m:
                step = int(m.group(1))
                for k, v in KV_RE.findall(m.group(2)):
                    try:
                        series[k].append((step, float(v)))
                    except ValueError:
                        pass
    return series, epoch_stats


def downsample(xy, max_pts=2000):
    if len(xy) <= max_pts:
        return xy
    step = len(xy) // max_pts
    return xy[::step]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    series, epoch_stats = parse(args.log)
    print(f"Parsed {len(series)} series, {len(epoch_stats)} epoch entries")

    # 1. train loss + mse vs step
    fig, ax = plt.subplots(figsize=(10, 5))
    for key, label, color in [
        ("train/loss", "total loss", "tab:blue"),
        ("train/mse", "mse (denoise)", "tab:orange"),
    ]:
        if key in series:
            xy = downsample(series[key])
            xs, ys = zip(*xy)
            ax.plot(xs, ys, label=label, alpha=0.6, lw=0.8, color=color)
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title("Training loss")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "train_loss.png"), dpi=130)
    plt.close(fig)

    # 2. val rec error per epoch + epoch avg train loss
    if epoch_stats:
        ep, tr, val = zip(*epoch_stats)
        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot(ep, tr, color="tab:blue", label="train avg loss")
        ax1.set_xlabel("epoch")
        ax1.set_ylabel("train avg loss", color="tab:blue")
        ax1.tick_params(axis="y", labelcolor="tab:blue")
        ax1.grid(alpha=0.3)

        ax2 = ax1.twinx()
        ax2.plot(ep, val, color="tab:red", label="val reconstruction error")
        ax2.set_ylabel("val rec error", color="tab:red")
        ax2.tick_params(axis="y", labelcolor="tab:red")

        best_idx = min(range(len(val)), key=lambda i: val[i])
        ax2.axvline(ep[best_idx], ls="--", color="tab:red", alpha=0.4)
        ax2.scatter([ep[best_idx]], [val[best_idx]], color="tab:red", zorder=5)
        ax2.annotate(
            f"best val={val[best_idx]:.4f} @ ep{ep[best_idx]}",
            xy=(ep[best_idx], val[best_idx]),
            xytext=(15, 15), textcoords="offset points",
            color="tab:red", fontsize=9,
        )
        plt.title("Train vs Val per epoch")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "train_vs_val.png"), dpi=130)
        plt.close(fig)

    # 3. aux losses
    aux_keys = ["train/mmd", "train/mmd_pca", "train/mmd_pop", "train/class_centroid"]
    fig, ax = plt.subplots(figsize=(10, 5))
    for key in aux_keys:
        if key in series:
            xy = downsample(series[key])
            xs, ys = zip(*xy)
            ax.plot(xs, ys, label=key.split("/")[-1], alpha=0.7, lw=0.8)
    ax.set_xlabel("step")
    ax.set_ylabel("aux loss")
    ax.set_title("Auxiliary losses")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "aux_losses.png"), dpi=130)
    plt.close(fig)

    # 4. lr + grad norm
    fig, ax1 = plt.subplots(figsize=(10, 5))
    if "train/lr" in series:
        xy = downsample(series["train/lr"])
        xs, ys = zip(*xy)
        ax1.plot(xs, ys, color="tab:purple", label="lr")
        ax1.set_yscale("log")
        ax1.set_ylabel("lr", color="tab:purple")
        ax1.tick_params(axis="y", labelcolor="tab:purple")
    ax2 = ax1.twinx()
    if "train/grad_norm" in series:
        xy = downsample(series["train/grad_norm"])
        xs, ys = zip(*xy)
        ax2.plot(xs, ys, color="tab:green", alpha=0.5, lw=0.8, label="grad norm")
        ax2.set_yscale("log")
        ax2.set_ylabel("grad norm", color="tab:green")
        ax2.tick_params(axis="y", labelcolor="tab:green")
    ax1.set_xlabel("step")
    ax1.set_title("LR schedule and gradient norm")
    ax1.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "lr_gradnorm.png"), dpi=130)
    plt.close(fig)

    print(f"Saved 4 figures to {args.out_dir}")


if __name__ == "__main__":
    main()
