import torch
import torch.nn as nn

from src.models.diffusion import GaussianDiffusion

NULL_CLASS = 26


class StubModel(nn.Module):
    """Deterministic label-dependent stand-in: x * (1 + y) + t."""

    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight
        self.calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        self.calls.append((x.clone(), t.clone(), y.clone()))
        return self.weight * (x * (1 + y.float().view(-1, 1, 1)) + t.float().view(-1, 1, 1))


def make_diffusion(timesteps: int = 11) -> GaussianDiffusion:
    return GaussianDiffusion(timesteps=timesteps, null_class=NULL_CLASS, enforce_zeros=False)


def test_alpha_zero_reproduces_standard_cfg():
    # Given the classic two-call CFG reference.
    diffusion, model = make_diffusion(), StubModel()
    x = torch.randn(2, 1, 4)
    t = torch.tensor([3, 7])
    y = torch.tensor([0, 5])
    conditional = model(x, t, y)
    unconditional = model(x, t, torch.full_like(y, NULL_CLASS))

    # When guiding with the default alpha.
    result = diffusion._predict_noise(model, x, t, y, 2.5)

    # Then the guided prediction matches (1 + w) * cond - w * uncond.
    assert torch.allclose(result, 3.5 * conditional - 2.5 * unconditional)


def test_batched_forward_matches_two_call_reference_and_runs_once():
    # Given a guided step with no interval and no guide model.
    diffusion, model = make_diffusion(), StubModel()
    x = torch.randn(3, 2, 4)
    t = torch.tensor([0, 5, 10])
    y = torch.tensor([1, 2, 3])
    reference = StubModel()
    expected = 2.0 * reference(x, t, y) - 1.0 * reference(x, t, torch.full_like(y, NULL_CLASS))

    result = diffusion._predict_noise(model, x, t, y, 1.0)

    # Then a single doubled-batch forward reproduces the two-call answer.
    assert torch.allclose(result, expected)
    assert len(model.calls) == 1
    assert len(model.calls[0][0]) == 6


def test_guidance_interval_gates_per_batch_element():
    # Given one sample inside the band and one outside it.
    diffusion, model = make_diffusion(), StubModel()
    x = torch.randn(2, 1, 4)
    t = torch.tensor([9, 1])  # fractions 0.9 and 0.1 of the horizon
    y = torch.tensor([0, 5])
    guided = diffusion._predict_noise(model, x, t, y, 2.0)
    conditional = StubModel()(x, t, y)

    result = diffusion._predict_noise(model, x, t, y, 2.0, guidance_interval=(0.5, 1.0))

    assert torch.allclose(result[0], guided[0])
    assert torch.allclose(result[1], conditional[1])


def test_whole_batch_outside_interval_skips_the_second_forward():
    # Given every sample below the band.
    diffusion, model = make_diffusion(), StubModel()
    x = torch.randn(2, 1, 4)
    t = torch.tensor([1, 2])
    y = torch.tensor([0, 5])

    result = diffusion._predict_noise(model, x, t, y, 2.0, guidance_interval=(0.5, 1.0))

    # Then only the conditional forward runs, at the original batch size.
    assert torch.allclose(result, StubModel()(x, t, y))
    assert len(model.calls) == 1
    assert len(model.calls[0][0]) == 2
    assert torch.equal(model.calls[0][2], y)


def test_power_law_scales_guidance_by_per_sample_rms():
    # Given constant per-sample inputs whose score differences are 26 and 50.
    diffusion, model = make_diffusion(), StubModel()
    x = torch.tensor([[[1.0, 1.0]], [[2.0, 2.0]]])
    t = torch.zeros(2, dtype=torch.long)
    y = torch.tensor([0, 1])

    result = diffusion._predict_noise(model, x, t, y, 2.0, guidance_alpha=0.5)

    # Then each sample's guidance term carries its own rms ** alpha factor.
    first = 1.0 + 2.0 * (1.0 - 27.0) * 26.0 ** 0.5
    second = 2.0 * 2.0 + 2.0 * (4.0 - 54.0) * 50.0 ** 0.5
    expected = torch.tensor([[[first, first]], [[second, second]]])
    assert torch.allclose(result, expected)


def test_guide_model_replaces_the_unconditional_branch_with_real_labels():
    # Given a weaker guiding model.
    diffusion, model, guide = make_diffusion(), StubModel(), StubModel(weight=0.5)
    x = torch.randn(2, 1, 4)
    t = torch.tensor([3, 4])
    y = torch.tensor([0, 5])
    conditional = StubModel()(x, t, y)
    guiding = StubModel(weight=0.5)(x, t, y)

    result = diffusion._predict_noise(model, x, t, y, 2.0, guide_model=guide)

    assert torch.allclose(result, conditional + 2.0 * (conditional - guiding))
    # Then the guide saw the real labels, and neither forward was batched.
    assert torch.equal(guide.calls[0][2], y)
    assert len(model.calls) == 1 and len(model.calls[0][0]) == 2


def test_samplers_accept_and_forward_the_guidance_options():
    # Given the public DDIM entry point with every variant enabled.
    diffusion, model, guide = make_diffusion(), StubModel(), StubModel(weight=0.5)

    result = diffusion.sample_ddim(
        model, (2, 1, 4), torch.tensor([0, 5]), torch.device("cpu"), ddim_steps=2,
        guidance_scale=1.0, guidance_interval=(0.0, 0.5), guidance_alpha=0.3,
        guide_model=guide,
    )

    assert result.shape == (2, 1, 4) and torch.isfinite(result).all()
    assert guide.calls
