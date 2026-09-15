from __future__ import annotations

import hashlib
import logging
import pickle
from pathlib import Path

import numpy as np
import pytest
import torch

from src.inference import generator


def test_checkpoint_config_is_authoritative_for_generation() -> None:
    passed = {"data": {"gene_size": 99}, "diffusion": {"max_timesteps": 99}}
    trained = {"data": {"gene_size": 8}, "diffusion": {"max_timesteps": 10}}

    resolved = generator.resolve_generation_config({"config": trained}, passed)

    assert resolved is trained


def test_passed_config_is_used_for_legacy_checkpoint_without_config() -> None:
    passed = {"data": {"gene_size": 8}, "diffusion": {"max_timesteps": 10}}

    resolved = generator.resolve_generation_config({}, passed)

    assert resolved is passed


def test_diffusion_uses_checkpoint_prediction_contract() -> None:
    data = {"num_classes": 3, "enforce_zeros": False}
    diffusion = {
        "max_timesteps": 2,
        "prediction_target": "epsilon",
        "sample_clip": 2.5,
    }

    result = generator.build_generation_diffusion(data, diffusion, None)

    assert result.sample_clip == 2.5


def test_diffusion_rejects_checkpoint_with_unsupported_prediction_target() -> None:
    data = {"num_classes": 3, "enforce_zeros": False}
    diffusion = {"max_timesteps": 2, "prediction_target": "v"}

    with pytest.raises(ValueError, match="prediction_target"):
        generator.build_generation_diffusion(data, diffusion, None)


def test_normalized_generation_requires_existing_stats(tmp_path: Path) -> None:
    data = {
        "normalize": True,
        "normalization_stats_path": str(tmp_path / "missing.pkl"),
    }

    with pytest.raises(FileNotFoundError, match="required"):
        generator.resolve_normalization_stats_path(data)


def test_generation_rejects_stats_changed_after_training(tmp_path: Path) -> None:
    stats_path = tmp_path / "stats.pkl"
    stats_path.write_bytes(b"overwritten after checkpoint")
    data = {
        "normalize": True,
        "normalization_stats_path": str(stats_path),
        "normalization_stats_sha256": hashlib.sha256(b"training stats").hexdigest(),
    }

    with pytest.raises(ValueError, match="checkpoint"):
        generator.resolve_normalization_stats_path(data)


def test_legacy_checkpoint_without_stats_fingerprint_warns(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    stats_path = tmp_path / "stats.pkl"
    stats_path.write_bytes(b"legacy stats")
    data = {"normalize": True, "normalization_stats_path": str(stats_path)}

    with caplog.at_level(logging.WARNING, logger=generator.__name__):
        result = generator.resolve_normalization_stats_path(data)

    assert result == stats_path
    assert any(record.levelno == logging.WARNING for record in caplog.records)


def test_denormalization_rejects_nonpositive_or_nonfinite_stats(tmp_path: Path) -> None:
    stats_path = tmp_path / "stats.pkl"
    with stats_path.open("wb") as handle:
        pickle.dump({
            "mean": np.zeros((2, 1), dtype=np.float32),
            "std": np.array([[1.0], [float("nan")]], dtype=np.float32),
        }, handle)

    with pytest.raises(ValueError, match="statistics"):
        generator.denormalize_samples(torch.zeros((1, 1, 2)), str(stats_path))


def test_masking_precedes_inverse_so_constant_mean_is_preserved(tmp_path: Path) -> None:
    stats_path = tmp_path / "stats.pkl"
    with stats_path.open("wb") as handle:
        pickle.dump({
            "mean": np.array([[7.0], [0.0]], dtype=np.float32),
            "std": np.ones((2, 1), dtype=np.float32),
        }, handle)
    samples = torch.tensor([[[3.0, 4.0]]])
    zero_mask = torch.tensor([[True, True]])

    result = generator.postprocess_samples(samples, zero_mask, stats_path)

    torch.testing.assert_close(result, torch.tensor([[[7.0, 0.0]]]))
