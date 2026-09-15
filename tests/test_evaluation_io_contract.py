from __future__ import annotations

import hashlib
import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from src.evaluation import _io


def _write_stats(path: Path, mean: np.ndarray, std: np.ndarray) -> None:
    with path.open("wb") as handle:
        pickle.dump({"mean": mean, "std": std}, handle)


def _write_sample(syn_dir: Path, value: np.ndarray) -> None:
    syn_dir.mkdir()
    torch.save(
        (torch.from_numpy(value.T.copy()), torch.tensor(0)),
        syn_dir / "sample_pop0_0000.pt",
    )


def test_loaders_return_real_and_synthetic_in_original_space(tmp_path: Path) -> None:
    mean = np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]], dtype=np.float32)
    std = np.array([[2.0, 4.0], [5.0, 8.0], [10.0, 12.0]], dtype=np.float32)
    stats_path = tmp_path / "normalization_stats.pkl"
    _write_stats(stats_path, mean, std)
    real_path = tmp_path / "test_data.pkl"
    with real_path.open("wb") as handle:
        pickle.dump((np.ones((1, 3, 2), dtype=np.float32), np.array([0])), handle)
    syn_dir = tmp_path / "synthetic"
    expected_syn = np.array([[11.0, 22.0], [33.0, 44.0], [55.0, 66.0]], dtype=np.float32)
    _write_sample(syn_dir, expected_syn)
    (syn_dir / "generation_meta.json").write_text(
        json.dumps({
            "sample_space": "original",
            "stats_fingerprint": hashlib.sha256(stats_path.read_bytes()).hexdigest(),
        })
    )

    real, _ = _io.load_real(real_path, stats_path=stats_path)
    synthetic, _, _ = _io.load_synthetic(syn_dir, stats_path=stats_path)

    np.testing.assert_allclose(real[0], mean + std)
    np.testing.assert_allclose(synthetic[0], expected_syn)


def test_legacy_synthetic_space_must_be_declared(tmp_path: Path) -> None:
    syn_dir = tmp_path / "synthetic"
    _write_sample(syn_dir, np.ones((2, 2), dtype=np.float32))
    (syn_dir / "generation_meta.json").write_text(json.dumps({"model_epoch": 455}))

    with pytest.raises(ValueError, match="legacy.*space"):
        _io.load_synthetic(syn_dir)


def test_generation_stats_fingerprint_must_match(tmp_path: Path) -> None:
    stats_path = tmp_path / "normalization_stats.pkl"
    _write_stats(stats_path, np.zeros((2, 2), dtype=np.float32), np.ones((2, 2), dtype=np.float32))
    syn_dir = tmp_path / "synthetic"
    _write_sample(syn_dir, np.ones((2, 2), dtype=np.float32))
    (syn_dir / "generation_meta.json").write_text(json.dumps({
        "sample_space": "original",
        "stats_fingerprint": "wrong-stats",
    }))

    with pytest.raises(ValueError, match="fingerprint"):
        _io.load_synthetic(syn_dir, stats_path=stats_path)


def test_input_fingerprint_changes_with_stats_and_generation_metadata(tmp_path: Path) -> None:
    real_path = tmp_path / "real.pkl"
    hierarchy = tmp_path / "hierarchy.pkl"
    stats_path = tmp_path / "stats.pkl"
    for path in (real_path, hierarchy, stats_path):
        path.write_bytes(b"first")
    syn_dir = tmp_path / "synthetic"
    _write_sample(syn_dir, np.ones((2, 2), dtype=np.float32))
    meta_path = syn_dir / "generation_meta.json"
    meta_path.write_text('{"sample_space":"original"}')
    first = _io.input_fingerprint(
        real_path=real_path, syn_dir=syn_dir, hierarchy=hierarchy,
        stats_path=stats_path, real_space="normalized",
        legacy_synthetic_space="original", n_genes=2, seed=1,
    )

    stats_path.write_bytes(b"second")
    second = _io.input_fingerprint(
        real_path=real_path, syn_dir=syn_dir, hierarchy=hierarchy,
        stats_path=stats_path, real_space="normalized",
        legacy_synthetic_space="original", n_genes=2, seed=1,
    )
    meta_path.write_text('{"sample_space":"normalized"}')
    third = _io.input_fingerprint(
        real_path=real_path, syn_dir=syn_dir, hierarchy=hierarchy,
        stats_path=stats_path, real_space="normalized",
        legacy_synthetic_space="original", n_genes=2, seed=1,
    )
    fourth = _io.input_fingerprint(
        real_path=real_path, syn_dir=syn_dir, hierarchy=hierarchy,
        stats_path=stats_path, real_space="original",
        legacy_synthetic_space="normalized", n_genes=2, seed=1,
    )

    assert first != second
    assert second != third
    assert third != fourth
    cache_meta = tmp_path / "pca_cache_meta.json"
    _io.write_pca_cache_meta(cache_meta, third, {})
    assert not _io.pca_cache_matches(cache_meta, fourth)


def test_evaluation_cli_help_and_legacy_error_are_user_facing(tmp_path: Path) -> None:
    script = Path(__file__).parents[1] / "scripts" / "evaluate_synthetic_metrics.py"
    help_result = subprocess.run(
        [sys.executable, str(script), "--help"], capture_output=True, text=True, check=False,
    )
    assert help_result.returncode == 0
    assert "--legacy-synthetic-space" in help_result.stdout

    real_path = tmp_path / "real.pkl"
    hierarchy = tmp_path / "hierarchy.pkl"
    stats_path = tmp_path / "stats.pkl"
    with real_path.open("wb") as handle:
        pickle.dump((np.zeros((2, 2, 2), dtype=np.float32), np.array([0, 0])), handle)
    with hierarchy.open("wb") as handle:
        pickle.dump({"pop_to_superpop": {0: 0}, "idx_to_superpop": {0: "X"}}, handle)
    _write_stats(stats_path, np.zeros((2, 2), dtype=np.float32), np.ones((2, 2), dtype=np.float32))
    syn_dir = tmp_path / "synthetic"
    _write_sample(syn_dir, np.zeros((2, 2), dtype=np.float32))
    (syn_dir / "generation_meta.json").write_text(json.dumps({"model_epoch": 455}))
    result = subprocess.run(
        [
            sys.executable, str(script), "--real-path", str(real_path),
            "--syn-dir", str(syn_dir), "--hierarchy", str(hierarchy),
            "--stats-path", str(stats_path), "--out-dir", str(tmp_path / "out"),
        ],
        capture_output=True, text=True, check=False,
    )

    assert result.returncode != 0
    assert "legacy synthetic sample space" in result.stderr.lower()


def test_pca_compare_standalone_help_loads_shared_evaluation_api() -> None:
    script = Path(__file__).parents[1] / "src" / "evaluation" / "pca_compare.py"

    result = subprocess.run(
        [sys.executable, str(script), "--help"], capture_output=True, text=True, check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--legacy_synthetic_space" in result.stdout
