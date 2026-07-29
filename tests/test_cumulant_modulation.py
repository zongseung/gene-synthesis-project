from __future__ import annotations

import math

import numpy as np
import pytest
import torch


def _write_artifact(
    path,
    *,
    n_pops: int = 1,
    gene_size: int = 1,
    n_channels: int = 1,
    skewness: float = 2.0,
    kurtosis: float = 0.0,
    eigenvalue: float = 1.0,
) -> None:
    rotation = np.broadcast_to(
        np.eye(n_channels, dtype=np.float32),
        (gene_size, n_channels, n_channels),
    ).copy()
    eigenvalues = np.full(
        (gene_size, n_channels), eigenvalue, dtype=np.float32
    )
    whitener = rotation / math.sqrt(eigenvalue)
    dewhitener = rotation * math.sqrt(eigenvalue)
    skew = np.zeros(
        (n_pops + 1, gene_size, n_channels), dtype=np.float32
    )
    kurt = np.zeros_like(skew)
    skew[:n_pops] = skewness
    kurt[:n_pops] = kurtosis
    np.savez_compressed(
        path,
        format_version=np.asarray(1, dtype=np.int64),
        center=np.zeros((gene_size, n_channels), dtype=np.float32),
        whitener=whitener,
        dewhitener=dewhitener,
        rotation=rotation,
        eigenvalues=eigenvalues,
        skewness=skew,
        excess_kurtosis=kurt,
    )


def test_zero_initialized_modulation_is_exact_identity(tmp_path):
    from src.models.modules.cumulant import OrthogonalCumulantModulation

    artifact = tmp_path / "stats.npz"
    _write_artifact(artifact)
    layer = OrthogonalCumulantModulation(
        n_pops=1,
        n_channels=1,
        gene_size=1,
        timesteps=1,
        stats_path=artifact,
    )
    x = torch.tensor([[[2.0]]])

    result = layer(x, torch.tensor([0]), torch.tensor([0]))

    assert torch.equal(result, x)


def test_skewness_uses_second_hermite_correction(tmp_path):
    from src.models.modules.cumulant import OrthogonalCumulantModulation

    artifact = tmp_path / "stats.npz"
    _write_artifact(artifact, skewness=2.0, kurtosis=0.0)
    layer = OrthogonalCumulantModulation(
        n_pops=1,
        n_channels=1,
        gene_size=1,
        timesteps=1,
        schedule_type="linear",
        stats_path=artifact,
        max_gain=1.0,
        correction_clip=10.0,
    )
    with torch.no_grad():
        layer.raw_gain.fill_(math.atanh(0.5))

    x = torch.tensor([[[2.0]]])
    result = layer(x, torch.tensor([0]), torch.tensor([0]))

    alpha_bar = 1.0 - 1e-4
    h2 = 2.0**2 - 1.0
    correction = alpha_bar**1.5 * (2.0 / 2.0) * h2
    expected = 2.0 + 0.5 * correction
    torch.testing.assert_close(result, torch.tensor([[[expected]]]))


def test_null_population_receives_no_cumulant_correction(tmp_path):
    from src.models.modules.cumulant import OrthogonalCumulantModulation

    artifact = tmp_path / "stats.npz"
    _write_artifact(artifact, skewness=10.0, kurtosis=10.0)
    layer = OrthogonalCumulantModulation(
        n_pops=1,
        n_channels=1,
        gene_size=1,
        timesteps=1,
        stats_path=artifact,
    )
    with torch.no_grad():
        layer.raw_gain.fill_(10.0)
    x = torch.tensor([[[3.0]]])

    result = layer(x, torch.tensor([0]), torch.tensor([1]))

    assert torch.equal(result, x)


def test_timestep_standardization_accounts_for_clean_eigenvalue(tmp_path):
    from src.models.modules.cumulant import OrthogonalCumulantModulation

    artifact = tmp_path / "stats.npz"
    _write_artifact(
        artifact,
        skewness=2.0,
        kurtosis=0.0,
        eigenvalue=4.0,
    )
    layer = OrthogonalCumulantModulation(
        n_pops=1,
        n_channels=1,
        gene_size=1,
        timesteps=2,
        schedule_type="linear",
        stats_path=artifact,
        max_gain=1.0,
        correction_clip=100.0,
    )
    with torch.no_grad():
        layer.raw_gain.fill_(math.atanh(0.5))

    x = torch.tensor([[[4.0]]])
    result = layer(x, torch.tensor([1]), torch.tensor([0]))

    alpha_bar = (1.0 - 1e-4) * (1.0 - 0.02)
    variance_t = alpha_bar * 4.0 + (1.0 - alpha_bar)
    standardized = 4.0 / math.sqrt(variance_t)
    signal_fraction = math.sqrt(alpha_bar * 4.0 / variance_t)
    correction = (
        signal_fraction**3
        * (2.0 / 2.0)
        * (standardized**2 - 1.0)
    )
    expected = 4.0 + 0.5 * math.sqrt(variance_t) * correction
    torch.testing.assert_close(result, torch.tensor([[[expected]]]))


def test_correction_is_bounded(tmp_path):
    from src.models.modules.cumulant import OrthogonalCumulantModulation

    artifact = tmp_path / "stats.npz"
    _write_artifact(artifact, skewness=1e6, kurtosis=1e6)
    layer = OrthogonalCumulantModulation(
        n_pops=1,
        n_channels=1,
        gene_size=1,
        timesteps=1,
        stats_path=artifact,
        max_gain=0.25,
        correction_clip=2.0,
    )
    with torch.no_grad():
        layer.raw_gain.fill_(100.0)
    x = torch.tensor([[[100.0]]])

    result = layer(x, torch.tensor([0]), torch.tensor([0]))

    assert abs(float(result.item() - x.item())) <= 0.5


def test_artifact_shape_mismatch_fails_before_forward(tmp_path):
    from src.models.modules.cumulant import OrthogonalCumulantModulation

    artifact = tmp_path / "bad.npz"
    _write_artifact(artifact, gene_size=2)

    with pytest.raises(ValueError, match="center"):
        OrthogonalCumulantModulation(
            n_pops=1,
            n_channels=1,
            gene_size=1,
            timesteps=1,
            stats_path=artifact,
        )


def test_hybrid_model_integrates_zero_initialized_oc_film(tmp_path):
    from src.models import HybridCNNDiTFiLM

    artifact = tmp_path / "stats.npz"
    _write_artifact(artifact, gene_size=8)
    common = {
        "data": {"num_channels": 1, "gene_size": 8},
        "diffusion": {
            "max_timesteps": 4,
            "noise_schedule": "linear",
        },
        "model": {
            "base_channels": 4,
            "channel_mult": [1, 2],
            "kernel_size": 3,
            "d_model": 4,
            "n_dit_blocks": 1,
            "n_heads": 1,
            "mlp_ratio": 2.0,
            "dropout": 0.0,
            "patch_size": 2,
            "n_pops": 1,
            "n_superpops": 1,
            "pop_to_superpop": {0: 0},
        },
    }
    baseline_config = {
        **common,
        "model": {
            **common["model"],
            "cumulant_modulation": {"enabled": False},
        },
    }
    cumulant_config = {
        **common,
        "model": {
            **common["model"],
            "cumulant_modulation": {
                "enabled": True,
                "stats_path": str(artifact),
                "initial_gain": 0.0,
            },
        },
    }

    torch.manual_seed(5)
    baseline = HybridCNNDiTFiLM(baseline_config).eval()
    torch.manual_seed(5)
    cumulant = HybridCNNDiTFiLM(cumulant_config).eval()
    cumulant.load_state_dict(baseline.state_dict(), strict=False)

    x = torch.randn(1, 1, 8)
    t = torch.tensor([0])
    y = torch.tensor([0])
    with torch.no_grad():
        baseline_output = baseline(x, t, y)
        cumulant_output = cumulant(x, t, y)

    assert cumulant.cumulant_modulation is not None
    assert cumulant_output.shape == x.shape
    torch.testing.assert_close(
        cumulant_output, baseline_output, rtol=1e-6, atol=1e-6
    )
