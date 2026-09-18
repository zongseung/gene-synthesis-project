from __future__ import annotations

import hashlib
import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pytest
import torch

from src.inference import generator


def test_diffusion_uses_checkpoint_prediction_contract() -> None:
    data = {"num_classes": 3, "enforce_zeros": False}
    diffusion = {"max_timesteps": 2, "sample_clip": 2.5}

    result = generator.build_generation_diffusion(data, diffusion, None)

    assert result.sample_clip == 2.5


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
        generator.postprocess_samples(torch.zeros((1, 1, 2)), None, stats_path)


def _parse(*extra: str):
    return generator.parse_args(
        ["--config", "configs/default.yaml", "--model_path", "best_model.pth", *extra]
    )


def test_guidance_variants_default_to_the_unguided_sampler_settings() -> None:
    args = _parse()

    assert args.guidance_interval is None
    assert args.guidance_alpha == 0.0
    assert args.guide_model_path is None


def test_guidance_interval_flag_parses_into_a_pair() -> None:
    args = _parse("--guidance-interval", "0.2", "0.8")

    assert args.guidance_interval == (0.2, 0.8)


def _run_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    guidance_type: str = "classifier_free",
    guide_diffusion: dict | None = None,
    **options,
) -> tuple[dict, dict]:
    """Run generate_samples against stub weights and a stub sampler.

    ``guide_diffusion`` overrides the guide checkpoint's diffusion config, which
    otherwise matches the main checkpoint.

    Returns the kwargs the sampler received and the written generation_meta.json.
    """
    recorded: dict = {}

    class StubDiffusion:
        def to(self, device):
            return self

        def sample_ddim(self, **kwargs):
            recorded.update(kwargs)
            return torch.zeros(kwargs["shape"])

    checkpoint = {
        "model_state_dict": {},
        "config": {
            "data": {
                "num_channels": 2,
                "gene_size": 4,
                "num_classes": 3,
                "normalize": False,
                "zero_mask_path": str(tmp_path / "no_zero_mask.pt"),
            },
            "diffusion": {
                "max_timesteps": 4,
                "guidance_type": guidance_type,
                "guidance_weight": 1.5,
                "sampling_timesteps": 2,
            },
        },
    }
    guide_checkpoint = checkpoint
    if guide_diffusion is not None:
        guide_config = {
            **checkpoint["config"],
            "diffusion": {**checkpoint["config"]["diffusion"], **guide_diffusion},
        }
        guide_checkpoint = {**checkpoint, "config": guide_config}
    loads = iter((checkpoint, guide_checkpoint))
    monkeypatch.setattr(generator.torch, "load", lambda *a, **k: next(loads, guide_checkpoint))
    monkeypatch.setattr(
        generator, "load_generator_model", lambda *a: (torch.nn.Identity(), True)
    )
    monkeypatch.setattr(generator, "build_generation_diffusion", lambda *a: StubDiffusion())
    model_path = tmp_path / "best_model.pth"
    model_path.touch()
    output_dir = tmp_path / "samples"

    generator.generate_samples(
        config={},
        model_path=str(model_path),
        output_dir=str(output_dir),
        n_samples_per_pop={0: 1},
        **options,
    )

    meta = json.loads((output_dir / "generation_meta.json").read_text())
    return recorded, meta


def test_generation_threads_the_guidance_variant_into_the_sampler_and_the_meta(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    guide_path = tmp_path / "weak_model.pth"
    guide_path.touch()

    recorded, meta = _run_generation(
        tmp_path,
        monkeypatch,
        guidance_interval=(0.2, 0.8),
        guidance_alpha=0.5,
        guide_model_path=str(guide_path),
    )

    assert recorded["guidance_interval"] == (0.2, 0.8)
    assert recorded["guidance_alpha"] == 0.5
    assert isinstance(recorded["guide_model"], torch.nn.Identity)
    assert meta["config"]["guidance_interval"] == [0.2, 0.8]
    assert meta["config"]["guidance_alpha"] == 0.5
    assert meta["config"]["guide_model_path"] == str(guide_path)


def test_guide_trained_on_another_noise_schedule_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    guide_path = tmp_path / "weak_model.pth"
    guide_path.touch()

    with pytest.raises(ValueError, match="noise_schedule"):
        _run_generation(
            tmp_path,
            monkeypatch,
            guide_diffusion={"noise_schedule": "linear"},
            guide_model_path=str(guide_path),
        )


def test_unguided_run_does_not_load_the_guide_at_all(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A guide whose checkpoint would be rejected proves it is never loaded.
    recorded, _ = _run_generation(
        tmp_path,
        monkeypatch,
        guidance_type="normal",
        guide_diffusion={"noise_schedule": "linear"},
        guide_model_path=str(tmp_path / "never_read.pth"),
    )

    assert recorded["guide_model"] is None


def test_unguided_run_warns_that_the_recorded_variant_did_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger=generator.__name__):
        _, meta = _run_generation(
            tmp_path, monkeypatch, guidance_type="normal", guidance_alpha=0.5,
        )

    assert meta["config"]["guidance_alpha"] == 0.5
    assert any("Guidance variants ignored" in record.message for record in caplog.records)


def test_guided_run_does_not_warn_about_ignored_variants(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger=generator.__name__):
        _run_generation(tmp_path, monkeypatch, guidance_alpha=0.5)

    assert not any("Guidance variants ignored" in r.message for r in caplog.records)


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
