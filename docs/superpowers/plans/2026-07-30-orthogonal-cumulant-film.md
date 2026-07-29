# Orthogonal Cumulant FiLM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build train-only population cumulant statistics on CPU and add an optional zero-initialized OC-FiLM input modulation to HiPoDiT.

**Architecture:** A NumPy preprocessing module converts unclipped GLM-PCA features into per-gene covariance eigen-coordinates, estimates bias-corrected skewness and excess kurtosis, and shrinks population estimates toward superpopulation estimates. A PyTorch module re-standardizes those coordinates at each diffusion timestep and applies a bounded Hermite correction before the existing CNN-DiT-FiLM backbone.

**Tech Stack:** Python 3.13, NumPy, pandas, PyTorch, pytest, YAML.

## Global Constraints

- Use only the existing 1000 Genomes train split; validation and test rows must not influence statistics.
- Use NumPy on CPU; do not add Rust or a new dependency.
- Keep existing CNN FiLM and DiT AdaLN-Zero behavior unchanged.
- OC-FiLM must be optional and zero-initialized.
- Null classifier-free guidance labels must receive zero cumulant correction.
- Existing unrelated untracked files must not be modified.

---

### Task 1: CPU cumulant artifact

**Files:**
- Create: `src/preprocessing/cumulants.py`
- Create: `tests/test_cumulants.py`

**Interfaces:**
- Produces: `fit_cumulant_statistics(x_train, pop_labels, superpop_labels, n_pops, n_superpops, gene_size, prior_strength_skew, prior_strength_kurtosis, eps) -> dict[str, np.ndarray]`
- Produces: `save_cumulant_statistics(stats, output_path) -> None`
- Produces CLI: `python -m src.preprocessing.cumulants --processed-dir data/processed`

- [x] **Step 1: Write failing ZCA and cumulant tests**

Create synthetic `(N, genes, K)` fixtures with correlated coordinates and
literal population labels. Assert identity covariance, documented artifact
shapes, finite cumulants, shrinkage direction, padding identity, eigenbasis
metadata, and null population zeros.

- [x] **Step 2: Verify the tests fail because the module is absent**

Run:

```bash
PYTHONPATH=. .venv/bin/pytest tests/test_cumulants.py -q
```

Expected: collection fails with `ModuleNotFoundError`.

- [x] **Step 3: Implement the minimal NumPy statistics**

Implement batched per-gene ZCA with `np.linalg.eigh`, bias-corrected sample
skewness and excess kurtosis, effective-prior-size shrinkage, rank handling,
padding, and compressed NPZ serialization. Reuse
`src.preprocessing.tokenizer.tokenize_dataset` in the CLI.

- [x] **Step 4: Run the focused tests**

```bash
PYTHONPATH=. .venv/bin/pytest tests/test_cumulants.py -q
```

Expected: all focused tests pass.

### Task 2: OC-FiLM PyTorch module

**Files:**
- Create: `src/models/modules/cumulant.py`
- Create: `tests/test_cumulant_modulation.py`

**Interfaces:**
- Consumes: the NPZ schema from Task 1.
- Produces: `OrthogonalCumulantModulation(nn.Module)`.
- Produces: `load_stats(path: str | Path) -> None`.
- Produces: `forward(x: Tensor, t: Tensor, y: Tensor) -> Tensor`.

- [x] **Step 1: Write failing behavioral tests**

Use a one-gene identity-whitening artifact with literal skewness and kurtosis.
Assert zero-gate identity, null-label identity, hand-derived Hermite correction,
bounded output, and artifact shape validation.

- [x] **Step 2: Verify the tests fail because the class is absent**

```bash
PYTHONPATH=. .venv/bin/pytest tests/test_cumulant_modulation.py -q
```

Expected: collection fails with `ModuleNotFoundError`.

- [x] **Step 3: Implement the minimal bounded residual layer**

Store fixed-size float32 buffers, compute diffusion alpha-cumprod from the
configured schedule, disable autocast for the cumulant math, re-standardize
each eigen-coordinate with
`alpha_bar * eigenvalue + (1 - alpha_bar)`, select population statistics by
label, apply \(H_2/H_3\), invert the rotation, clip, and apply the bounded
per-channel zero-initialized gate.

- [x] **Step 4: Run the focused tests**

```bash
PYTHONPATH=. .venv/bin/pytest tests/test_cumulant_modulation.py -q
```

Expected: all focused tests pass.

### Task 3: Model and configuration integration

**Files:**
- Modify: `src/models/hybrid_geno_dit.py`
- Modify: `src/inference/generator.py`
- Modify: `configs/default.yaml`
- Modify: `tests/test_cumulant_modulation.py`

**Interfaces:**
- Consumes: `OrthogonalCumulantModulation` from Task 2.
- Preserves: `HybridCNNDiTFiLM.forward(x, t, y) -> Tensor`.

- [x] **Step 1: Add a failing tiny-model integration test**

Construct a small model configuration with an artifact in `tmp_path`, run a
CPU forward pass, and assert output shape and zero-init equality against the
same weights with OC-FiLM disabled.

- [x] **Step 2: Verify the integration test fails**

```bash
PYTHONPATH=. .venv/bin/pytest tests/test_cumulant_modulation.py -q
```

Expected: failure because `HybridCNNDiTFiLM` does not construct OC-FiLM.

- [x] **Step 3: Integrate the optional module**

Construct OC-FiLM from `model.cumulant_modulation`, execute it before the
existing conditioning path, and keep the disabled path free of new state
keys. In inference, load `model_state_dict` before overlaying EMA parameters
so non-parameter cumulant buffers come from the checkpoint.

- [x] **Step 4: Record the experiment config**

Enable OC-FiLM in `configs/default.yaml`, point to
`data/processed/cumulant_stats.npz`, and change `run_name` and `save_dir` so
the baseline output directory cannot be overwritten.

- [x] **Step 5: Run focused and full tests**

```bash
PYTHONPATH=. .venv/bin/pytest tests/test_cumulants.py tests/test_cumulant_modulation.py -q
PYTHONPATH=. .venv/bin/pytest tests -q
```

Expected: all tests pass.

### Task 4: Build and validate the real artifact

**Files:**
- Generate, ignored: `data/processed/cumulant_stats.npz`

**Interfaces:**
- Consumes: current `data/processed` artifacts.
- Produces: a real artifact loadable by `configs/default.yaml`.

- [x] **Step 1: Build on CPU**

```bash
/usr/bin/time -v .venv/bin/python -m src.preprocessing.cumulants \
  --processed-dir data/processed \
  --output data/processed/cumulant_stats.npz
```

Expected: exit 0 with sample/population/gene/component counts and output path.

- [x] **Step 2: Validate real metadata and finite values**

Load the NPZ read-only and assert documented shapes, finite arrays, zero null
population, identity padded transforms, and counts summing to 2,002.

- [x] **Step 3: Run a real-artifact module smoke test**

Instantiate `OrthogonalCumulantModulation` with the production dimensions,
load the artifact, and run one small CPU slice through its forward method.

### Task 5: Final verification

**Files:**
- Review all files changed by Tasks 1–4.

- [x] **Step 1: Compile and test**

```bash
PYTHONPATH=. .venv/bin/python -m py_compile \
  src/preprocessing/cumulants.py \
  src/models/modules/cumulant.py \
  src/models/hybrid_geno_dit.py \
  src/inference/generator.py
PYTHONPATH=. .venv/bin/pytest tests -q
```

- [x] **Step 2: Inspect scope**

```bash
git diff --check
git status --short
git diff -- src tests configs/default.yaml
```

Confirm only OC-FiLM source, tests, and configuration changed; the generated
data artifact and local documentation remain outside tracked source.

- [x] **Step 3: Review against the design**

Check every requirement in
`docs/superpowers/specs/2026-07-30-orthogonal-cumulant-film-design.md` and
record any unimplemented or intentionally deferred item in the final handoff.
