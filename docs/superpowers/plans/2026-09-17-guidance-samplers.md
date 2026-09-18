# Plan: three guidance variants in the DDIM sampler

## Context

The 2026-09-17 whole-genome run (`outputs/20260917_wholegenome_k4`) generates population-conditional
samples whose PCA-space cluster centroids are displaced from the real ones. A CFG sweep
(`outputs/guidance_sweep_20260917`, weights 0.5/1/2/4) showed no single weight fixes all five
superpopulations: AMR and SAS get worse as the weight rises, EUR improves, EAS barely moves.

Literature (`claudedocs/research_cfg_population_structure_20260917.md`) points at three
sampling-time changes, all independent of each other and all without retraining:

1. Guidance applied only in a middle band of noise levels (Kynkäänniemi et al., NeurIPS 2024).
2. Power-Law CFG — the guidance term scaled by the norm of the score difference
   (Pavasovic et al., ICLR 2026).
3. Autoguidance — guide with a weaker checkpoint of the same model instead of the
   unconditional branch (Karras et al., NeurIPS 2024).

The current sampler runs two separate forward passes per step when guidance is on, which is why
generation takes 16 minutes instead of 8.5. Batching them is a prerequisite that also speeds up
every variant.

## Global Constraints

- **Default behaviour must not change.** With `guidance_scale=0` the sampler must produce exactly
  what it produces today; with `guidance_scale>0` and the new options left at their defaults
  (`guidance_interval=None`, `guidance_alpha=0.0`, `guide_model=None`) it must match standard CFG.
- Touch only `src/models/diffusion.py`, `src/inference/generator.py`, `tests/`. Another session is
  editing `scripts/hipodit_*.py` and `tests/test_hipodit_*.py` — do not touch those.
- Tests run on the CPU-only root `.venv` (`nice -n 19 .venv/bin/python -m pytest tests/ -q`).
  Keep every existing test passing; new tests must run on CPU in seconds with a tiny dummy model.
- The repo's style: no new dependencies, no new abstraction layers, small functions, Korean not
  used in code or comments.
- `GaussianDiffusion.timesteps` is the discrete count (1000 in the shipped config); DDIM sampling
  uses `sampling_timesteps` (100) evenly spaced indices into it.

## Task 1 — batched guidance, guidance interval, power-law CFG

**File:** `src/models/diffusion.py` (`_predict_noise`, `p_sample`, `ddim_sample`, `sample`)

Extend `_predict_noise` to this signature:

```python
def _predict_noise(
    self, model: nn.Module, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor, guidance: float,
    *, guidance_interval: tuple[float, float] | None = None, guidance_alpha: float = 0.0,
    guide_model: nn.Module | None = None,
) -> torch.Tensor:
```

Required behaviour:

1. **No guidance.** `guidance <= 0` → one conditional forward, returned unchanged (today's path).
2. **Interval.** `guidance_interval=(lo, hi)` with `0 <= lo < hi <= 1` gates guidance on the
   *fraction of the diffusion horizon*: guidance applies only while
   `lo <= t / (self.timesteps - 1) <= hi`, evaluated per batch element. Outside the band the
   conditional prediction is returned. `None` means always on. When the whole batch is outside the
   band, only the conditional forward runs — the second forward must be skipped, not computed and
   discarded.
3. **Batched forwards.** When the guided branch runs, the conditional and the guiding input are
   concatenated along the batch axis into a single `model(...)` call, then split. The guiding input
   is `null_class` labels for CFG, and the same `y` when `guide_model` is given.
4. **Power law.** With `guidance_alpha > 0` the guided prediction is
   `cond + guidance * diff * rms(diff) ** guidance_alpha`, where
   `diff = cond - guide` and `rms(diff)` is the per-sample root-mean-square of `diff`
   (`diff.flatten(1).norm(dim=1) / sqrt(diff[0].numel())`), broadcast back over the sample's dims.
   `guidance_alpha=0.0` reduces exactly to `cond + guidance * diff`, i.e. today's
   `(1 + guidance) * cond - guidance * guide`.
5. **Autoguidance hook.** `guide_model` replaces the unconditional branch: the guiding prediction
   is `guide_model(x, t, y)` — same labels, weaker model. It cannot be batched with the main
   model's forward, so with `guide_model` the two forwards stay separate.

`p_sample`, `ddim_sample` and `sample` gain the same three keyword arguments and pass them through
unchanged. Defaults everywhere: `guidance_interval=None`, `guidance_alpha=0.0`, `guide_model=None`.

**Tests** (new file `tests/test_guidance_variants.py`, CPU, a tiny stub model that returns a
deterministic function of its inputs so predictions are checkable):

- `guidance_alpha=0` reproduces `(1 + w) * cond - w * uncond` exactly (`torch.allclose`).
- The batched path equals a reference two-call implementation (`torch.allclose`).
- Inside the interval the guided value is returned; outside it the plain conditional is, and the
  stub model records only one forward call for the outside case.
- `guidance_alpha>0` multiplies the guidance term by the per-sample RMS raised to alpha — check
  against a hand-computed value on a 2-sample batch.
- With `guide_model`, the guiding prediction comes from the second model and the labels passed to
  it are the real `y`, not `null_class`.

## Task 2 — expose the variants through the generator CLI

**File:** `src/inference/generator.py`

Add four CLI flags and thread them into the sampling call:

- `--guidance-interval LO HI` (two floats, default `None`)
- `--guidance-alpha FLOAT` (default `0.0`)
- `--guide-model-path PATH` (default `None`) — a checkpoint loaded exactly like the main model
  (same config, `model_state_dict`, EMA weights if present), put in eval mode on the same device,
  and passed as `guide_model`
- The existing `--guidance_weight` keeps working unchanged.

`generation_meta.json` must record the three new settings alongside the existing ones so a run can
be identified later. Keep the existing keys and their spelling.

**Tests** (extend `tests/test_inference_generator_contract.py`): the metadata dict carries the new
keys with their defaults, and parsing `--guidance-interval 0.2 0.8` yields `(0.2, 0.8)`.
