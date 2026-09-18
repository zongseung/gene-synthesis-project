"""Project-specific I/O and caching for synthetic-evaluation pipelines.

This module is intentionally separate from :mod:`src.evaluation.dupi` and
:mod:`src.evaluation.distribution_metrics`, which are pure / portable.
Anything here is coupled to the ``data/processed/*.pkl`` and
``outputs/<run>/synthetic_samples/sample_pop*.pt`` conventions used by the
HiPoDiT project.
"""

from __future__ import annotations

import csv
import hashlib
import json
import pickle
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from src.preprocessing.tokenizer import invert_normalization, load_normalization_stats

SampleSpace = Literal["normalized", "original"]

__all__ = [
    "flatten_subsample_genes",
    "load_label_hierarchy",
    "load_real",
    "load_synthetic",
    "pop_to_superpop",
    "write_csv",
]


# ── filesystem fingerprints ────────────────────────────────────────────
def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ── data loaders ───────────────────────────────────────────────────────
def load_real(
    path: Path,
    *,
    stats_path: Path | None = None,
    sample_space: SampleSpace = "normalized",
) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as f:
        x, y = pickle.load(f)
    values = np.asarray(x, dtype=np.float32)
    labels = np.asarray(y, dtype=np.int64)
    if sample_space == "normalized":
        if stats_path is None:
            raise FileNotFoundError("Normalization stats are required for normalized real data")
        stats = load_normalization_stats(stats_path, expected_shape=values.shape[-2:])
        values = invert_normalization(values, stats, labels)
    return values, labels


def load_synthetic(
    syn_dir: Path,
    *,
    stats_path: Path | None = None,
    legacy_sample_space: SampleSpace | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    files = sorted(syn_dir.glob("sample_pop*_*.pt"))
    if not files:
        raise FileNotFoundError(f"No sample_pop*_*.pt files under {syn_dir}")

    meta_path = syn_dir / "generation_meta.json"
    metadata = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    sample_space = metadata.get("sample_space")
    if sample_space is None:
        if legacy_sample_space is None:
            raise ValueError(
                "Legacy synthetic sample space is unknown; pass --legacy-synthetic-space"
            )
        sample_space = legacy_sample_space
    if sample_space not in {"normalized", "original"}:
        raise ValueError(f"Unsupported synthetic sample_space: {sample_space!r}")

    expected_fingerprint = metadata.get("stats_fingerprint")
    if expected_fingerprint is not None:
        if stats_path is None:
            raise FileNotFoundError("Normalization stats are required to verify generated samples")
        if file_sha256(stats_path) != expected_fingerprint:
            raise ValueError("Normalization stats fingerprint does not match generation metadata")

    stats = load_normalization_stats(stats_path) if stats_path is not None else None
    expected_shape = tuple(stats["shape"]) if stats is not None else None

    xs: list[np.ndarray] = []
    ys: list[int] = []
    names: list[str] = []
    for file_path in files:
        genome, label = torch.load(file_path, map_location="cpu", weights_only=False)
        arr = genome.detach().cpu().numpy().astype(np.float32)
        if arr.ndim != 2:
            raise ValueError(f"{file_path} has unexpected genome shape {arr.shape}")
        if expected_shape is not None and arr.shape == expected_shape:
            pass
        elif expected_shape is not None and arr.T.shape != expected_shape:
            raise ValueError(
                f"{file_path} shape {arr.shape} does not match normalization shape {expected_shape}"
            )
        elif expected_shape is not None or arr.shape[0] < arr.shape[1]:
            arr = arr.T
        xs.append(arr)
        ys.append(int(label.item() if hasattr(label, "item") else label))
        names.append(file_path.name)

    values = np.stack(xs, axis=0)
    labels = np.asarray(ys, dtype=np.int64)
    if sample_space == "normalized":
        if stats is None:
            raise FileNotFoundError("Normalization stats are required for normalized synthetic data")
        values = invert_normalization(values, stats, labels)
    return values, labels, names


def load_label_hierarchy(path: Path) -> dict:
    with path.open("rb") as f:
        return pickle.load(f)


def pop_to_superpop(pop: np.ndarray, hierarchy: dict) -> np.ndarray:
    pop_to_super = hierarchy["pop_to_superpop"]
    idx_to_super = hierarchy["idx_to_superpop"]
    return np.array([idx_to_super[pop_to_super[int(p)]] for p in pop])


def flatten_subsample_genes(
    x: np.ndarray,
    n_genes: int,
    seed: int,
    indices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if x.ndim != 3:
        raise ValueError(f"Expected (N, G, C), got {x.shape}")
    n_available = x.shape[1]
    if indices is None:
        rng = np.random.default_rng(seed)
        if n_genes >= n_available:
            indices = np.arange(n_available)
        else:
            indices = np.sort(rng.choice(n_available, size=n_genes, replace=False))
    return x[:, indices, :].reshape(x.shape[0], -1), indices


# ── generic CSV writer ─────────────────────────────────────────────────
def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
