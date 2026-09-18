import pytest
import torch
import torch.nn as nn

from src.models.diffusion import GaussianDiffusion


class ZeroNoiseModel(nn.Module):
    def forward(
        self, x: torch.Tensor, timestep: torch.Tensor, label: torch.Tensor
    ) -> torch.Tensor:
        return torch.zeros_like(x)


class RecordingZeroNoiseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.inputs = []

    def forward(
        self, x: torch.Tensor, timestep: torch.Tensor, label: torch.Tensor
    ) -> torch.Tensor:
        self.inputs.append(x.detach().clone())
        return torch.zeros_like(x)


def test_diffusion_rejects_unimplemented_prediction_target():
    with pytest.raises(ValueError, match="prediction_target"):
        GaussianDiffusion(timesteps=1, prediction_target="v")


@pytest.mark.parametrize("sample_clip", [0.0, float("inf"), float("nan")])
def test_diffusion_rejects_invalid_model_space_clip(sample_clip):
    with pytest.raises(ValueError, match="sample_clip"):
        GaussianDiffusion(timesteps=1, sample_clip=sample_clip)


def test_ddim_sampling_honors_model_space_clip():
    diffusion = GaussianDiffusion(timesteps=1, sample_clip=0.01)

    result = diffusion.sample_ddim(
        ZeroNoiseModel(),
        shape=(1, 1, 2),
        y=torch.zeros(1, dtype=torch.long),
        device=torch.device("cpu"),
        ddim_steps=1,
    )

    assert result.abs().max() <= 0.01


def test_ddim_keeps_masked_coordinates_zero_at_every_reverse_call():
    model = RecordingZeroNoiseModel()
    diffusion = GaussianDiffusion(
        timesteps=3,
        zero_mask=torch.tensor([[True, False]]),
    )

    result = diffusion.sample_ddim(
        model,
        shape=(1, 1, 2),
        y=torch.zeros(1, dtype=torch.long),
        device=torch.device("cpu"),
        ddim_steps=3,
    )

    assert all(torch.count_nonzero(x[..., 0]) == 0 for x in model.inputs)
    assert torch.count_nonzero(result[..., 0]) == 0


def test_forward_and_training_keep_padding_on_the_sampling_support():
    # Given a padded coordinate and deliberately nonzero noise there.
    model = RecordingZeroNoiseModel()
    diffusion = GaussianDiffusion(
        timesteps=1000, schedule_type="linear",
        zero_mask=torch.tensor([[True, False]]),
    )
    x = torch.zeros(2, 1, 2)
    # When the real training entry point adds noise.
    diffusion.p_losses(model, x, torch.tensor([0, 500]), torch.zeros(2).long(),
                       noise=torch.ones_like(x), cfg_training=False)
    # Then padding stays zero while valid coordinates are still diffused.
    assert torch.count_nonzero(model.inputs[0][..., 0]) == 0
    assert torch.count_nonzero(model.inputs[0][..., 1]) == 2


def test_feature_schedules_change_intermediate_noise_but_share_endpoints():
    # Given fast, ordinary and slow schedules on otherwise identical inputs.
    diffusion = GaussianDiffusion(
        timesteps=1000, schedule_type="linear",
        feature_schedule=torch.tensor([[0, 1, 2]]),
    )
    x = torch.ones(1, 1, 3)
    # When all three signals are diffused with zero supplied noise.
    first, middle, last = [diffusion.q_sample(x, torch.tensor([t]), torch.zeros_like(x))
                           for t in (0, 500, 999)]
    # Then only the intermediate signal retention differs.
    assert middle[0, 0, 0] < middle[0, 0, 1] < middle[0, 0, 2]
    torch.testing.assert_close(first, first[..., :1].expand_as(first))
    torch.testing.assert_close(last, last[..., :1].expand_as(last))


@pytest.mark.parametrize("steps", [1, 7, 100])
def test_ddim_recovers_known_clean_signal_with_featurewise_oracle(steps):
    # Given an oracle for a fixed clean signal; wrong starting times or
    # mismatched forward/reverse schedules prevent exact reconstruction.
    diffusion = GaussianDiffusion(
        timesteps=100, schedule_type="linear", sample_clip=None,
        feature_schedule=torch.tensor([[0, 1, 2]]),
    )
    clean = torch.tensor([[[0.25, -0.5, 0.75]]])

    class Oracle(nn.Module):
        def forward(self, x, t, y):
            signal = diffusion.q_sample(clean.expand_as(x), t, torch.zeros_like(x))
            amplitude = diffusion.q_sample(torch.zeros_like(x), t, torch.ones_like(x))
            return (x - signal) / amplitude

    # When sampling through the public DDIM loop.
    result = diffusion.sample_ddim(Oracle(), clean.shape, torch.zeros(1).long(),
                                   torch.device("cpu"), ddim_steps=steps)
    # Then the oracle recovers the clean signal.
    torch.testing.assert_close(result, clean, atol=2e-5, rtol=2e-5)
