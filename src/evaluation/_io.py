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
from typing import Any, Literal

import numpy as np
import torch

from src.preprocessing.tokenizer import invert_normalization, load_normalization_stats

SampleSpace = Literal["normalized", "original"]

__all__ = [
    "file_stat",
    "flatten_subsample_genes",
    "input_fingerprint",
    "load_label_hierarchy",
    "load_pca_cache_meta",
    "load_pca_coordinates",
    "load_real",
    "load_synthetic",
    "load_synthetic_cached",
    "pca_cache_matches",
    "pop_to_superpop",
    "synthetic_fingerprint",
    "write_csv",
    "write_pca_cache_meta",
    "write_pca_coordinates",
]


# ── filesystem fingerprints ────────────────────────────────────────────
def file_stat(path: Path) -> dict[str, Any]:
    st = path.stat()
    return {"path": str(path), "size": st.st_size, "mtime_ns": st.st_mtime_ns}


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def synthetic_fingerprint(syn_dir: Path) -> dict[str, Any]:
    files = sorted(syn_dir.glob("sample_pop*_*.pt"))
    if not files:
        raise FileNotFoundError(f"No sample_pop*_*.pt files under {syn_dir}")
    meta_path = syn_dir / "generation_meta.json"
    return {
        "syn_dir": str(syn_dir),
        "file_count": len(files),
        "files": [
            {"name": f.name, "size": f.stat().st_size, "mtime_ns": f.stat().st_mtime_ns}
            for f in files
        ],
        "generation_meta": (
            {**file_stat(meta_path), "sha256": file_sha256(meta_path)}
            if meta_path.exists()
            else None
        ),
    }


def input_fingerprint(
    *,
    real_path: Path,
    syn_dir: Path,
    hierarchy: Path,
    stats_path: Path,
    real_space: SampleSpace,
    legacy_synthetic_space: SampleSpace | None,
    n_genes: int,
    seed: int,
) -> dict[str, Any]:
    return {
        "real": file_stat(real_path),
        "hierarchy": file_stat(hierarchy),
        "normalization_stats": {
            **file_stat(stats_path),
            "sha256": file_sha256(stats_path),
        },
        "synthetic": synthetic_fingerprint(syn_dir),
        "real_space": real_space,
        "legacy_synthetic_space": legacy_synthetic_space,
        "n_genes": n_genes,
        "seed": seed,
    }


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
    if sample_space == "normalized":
        if stats_path is None:
            raise FileNotFoundError("Normalization stats are required for normalized real data")
        stats = load_normalization_stats(stats_path, expected_shape=values.shape[-2:])
        values = invert_normalization(values, stats)
    return values, np.asarray(y, dtype=np.int64)


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
    if sample_space == "normalized":
        if stats is None:
            raise FileNotFoundError("Normalization stats are required for normalized synthetic data")
        values = invert_normalization(values, stats)
    return values, np.asarray(ys, dtype=np.int64), names


def load_synthetic_cached(
    syn_dir: Path,
    cache_path: Path,
    mode: str,
    *,
    stats_path: Path | None = None,
    legacy_sample_space: SampleSpace | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str], bool]:
    """Load synthetic tensors, optionally reusing a single-file NPZ cache."""
    if mode not in {"auto", "refresh", "off"}:
        raise ValueError(f"Unknown array cache mode: {mode}")

    current_meta = {
        **synthetic_fingerprint(syn_dir),
        "stats_sha256": file_sha256(stats_path) if stats_path is not None else None,
        "legacy_sample_space": legacy_sample_space,
    }
    if mode != "refresh" and cache_path.exists():
        try:
            with np.load(cache_path, allow_pickle=False) as data:
                cached_meta = json.loads(str(data["meta_json"].item()))
                if mode == "auto" and cached_meta == current_meta:
                    x = data["x"].astype(np.float32, copy=False)
                    pop = data["pop"].astype(np.int64, copy=False)
                    names = data["sample_names"].astype(str).tolist()
                    return x, pop, names, True
        except Exception:
            if mode != "auto":
                raise

    x, pop, names = load_synthetic(
        syn_dir,
        stats_path=stats_path,
        legacy_sample_space=legacy_sample_space,
    )
    if mode != "off":
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            cache_path,
            x=x.astype(np.float32, copy=False),
            pop=pop.astype(np.int64, copy=False),
            sample_names=np.asarray(names),
            meta_json=np.asarray(json.dumps(current_meta)),
        )
    return x, pop, names, False


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


# ── PCA-coordinate caching ─────────────────────────────────────────────
def pca_cache_matches(meta_path: Path, fingerprint: dict[str, Any]) -> bool:
    return (load_pca_cache_meta(meta_path) or {}).get("fingerprint") == fingerprint


def write_pca_cache_meta(
    meta_path: Path, fingerprint: dict[str, Any], pca_info: dict[str, Any]
) -> None:
    meta_path.write_text(
        json.dumps(
            {"fingerprint": fingerprint, "pca": pca_info},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def load_pca_cache_meta(meta_path: Path) -> dict[str, Any] | None:
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def load_pca_coordinates(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    real_pcs: list[list[float]] = []
    syn_pcs: list[list[float]] = []
    real_pop: list[int] = []
    syn_pop: list[int] = []
    real_sp: list[str] = []
    syn_sp: list[str] = []
    syn_names: list[str] = []

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pc = [float(row["pc1"]), float(row["pc2"])]
            pop = int(row["pop_label"])
            sp = row["superpopulation"]
            if row["source"] == "real":
                real_pcs.append(pc)
                real_pop.append(pop)
                real_sp.append(sp)
            elif row["source"] == "synthetic":
                syn_pcs.append(pc)
                syn_pop.append(pop)
                syn_sp.append(sp)
                syn_names.append(row["sample_id"])
            else:
                raise ValueError(f"Unexpected PCA coordinate source: {row['source']}")

    if not real_pcs or not syn_pcs:
        raise ValueError(f"PCA coordinate cache is incomplete: {path}")
    return (
        np.asarray(real_pcs, dtype=np.float64),
        np.asarray(syn_pcs, dtype=np.float64),
        np.asarray(real_pop, dtype=np.int64),
        np.asarray(syn_pop, dtype=np.int64),
        np.asarray(real_sp),
        np.asarray(syn_sp),
        syn_names,
    )


def write_pca_coordinates(
    path: Path,
    real_pcs: np.ndarray,
    syn_pcs: np.ndarray,
    real_pop: np.ndarray,
    syn_pop: np.ndarray,
    real_sp: np.ndarray,
    syn_sp: np.ndarray,
    syn_names: list[str],
) -> None:
    rows = []
    for i, (pc, pop, sp) in enumerate(zip(real_pcs, real_pop, real_sp, strict=True)):
        rows.append({
            "source": "real",
            "sample_id": f"real_{i:04d}",
            "pop_label": int(pop),
            "superpopulation": sp,
            "pc1": float(pc[0]),
            "pc2": float(pc[1]),
        })
    for name, pc, pop, sp in zip(syn_names, syn_pcs, syn_pop, syn_sp, strict=True):
        rows.append({
            "source": "synthetic",
            "sample_id": name,
            "pop_label": int(pop),
            "superpopulation": sp,
            "pc1": float(pc[0]),
            "pc2": float(pc[1]),
        })
    write_csv(path, rows)


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
