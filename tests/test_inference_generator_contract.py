from __future__ import annotations

import hashlib
import logging
import pickle
from pathlib import Path

import numpy as np
import pytest
import torch

from src.inference import generator


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


def _parse(*extra: str):
    return generator.parse_args(
        ["--config", "configs/default.yaml", "--model_path", "best_model.pth", *extra]
    )


def test_guidance_variants_default_to_the_unguided_sampler_settings() -> None:
    args = _parse()

    meta = generator.guidance_meta(
        args.guidance_interval, args.guidance_alpha, args.guide_model_path
    )

    assert meta == {
        "guidance_interval": None,
        "guidance_alpha": 0.0,
        "guide_model_path": None,
    }


def test_guidance_interval_flag_parses_into_a_pair() -> None:
    args = _parse("--guidance-interval", "0.2", "0.8")

    assert args.guidance_interval == (0.2, 0.8)


def test_generation_meta_records_the_requested_guidance_variant() -> None:
    args = _parse(
        "--guidance-interval", "0.2", "0.8",
        "--guidance-alpha", "0.5",
        "--guide-model-path", "outputs/weak/best_model.pth",
    )

    meta = generator.guidance_meta(
        args.guidance_interval, args.guidance_alpha, args.guide_model_path
    )

    assert meta == {
        "guidance_interval": [0.2, 0.8],
        "guidance_alpha": 0.5,
        "guide_model_path": "outputs/weak/best_model.pth",
    }


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
