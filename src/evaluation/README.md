# `src.evaluation` — DUPI for synthetic-data evaluation

A reference Python implementation of the **D**ata **U**tility–**P**rivacy
**I**ndex (DUPI) and its companion Utility/Privacy indices, faithful to

> Donghoon Jeong, Joseph H. T. Kim, and Jongho Im,
> "**A New Global Measure to Simultaneously Evaluate Data Utility and
> Privacy Risk**", *IEEE Trans. Information Forensics and Security*,
> Vol. 18, pp. 715–729, 2023. doi:[10.1109/TIFS.2022.3228753](https://doi.org/10.1109/TIFS.2022.3228753)

The original paper does not ship a reference implementation. To the best
of our knowledge this module is the first publicly available
implementation that reproduces the paper's printed numerical examples to
3–4 significant digits (Wine illustration, optimal point, Theorem 5
upper bound, S1 simulation). See [Tests](#tests) for the reproduction
suite.

---

## Why this module?

DUPI answers a single question: *given a real dataset and a synthetic
counterpart, are the synthetic samples too close (memorisation) or too
far (utility loss)?* It

* operates at the **dataset level**, not per-record (so it does not
  require a single-attribute privacy taxonomy like *k*-anonymity);
* is **distribution-free** — works in any metric space;
* yields **two indices in [0, 1]** (`utility_index`, `privacy_index`)
  that can be plotted on a single trade-off curve;
* admits a **closed-form benchmark** for the equal-distribution case, so
  observed values can be compared against a theoretical reference.

---

## Public API

| Symbol | What it does | Paper reference |
| --- | --- | --- |
| `dupi_score(x_real, x_syn, k=1)` | Empirical DUPI in `[0, 1]` plus diagnostics | Eq. (11) |
| `kth_dupi_benchmark(n, m, k)` | Equal-distribution benchmark `DUPI₀` | Eq. (10) |
| `ui_pi_from_dupi(dupi, dupi0, tau=5.0)` | Map DUPI → `(UI, PI, U·P)` | Eqs. (12)–(13) |
| `gaussian_w2_distance(x, y)` | FID-style Gaussian-fit Wasserstein-2 | — |
| `mmd_rbf(x, y)` | Biased MMD with RBF + median-heuristic γ | — |
| `same_class_coverage(real, syn)` | Fraction of real points covered by syn within real-NN-95th-pct radius | — |
| `centroid_distance(x, y)` | Euclidean distance between centroids | — |
| `evaluate(real_pcs, syn_pcs, real_sp, syn_sp)` | One-call pipeline → `EvaluationReport` | — |

---

## Quick start

```python
import numpy as np
from src.evaluation import dupi_score, kth_dupi_benchmark, ui_pi_from_dupi

rng = np.random.default_rng(0)
x_real = rng.standard_normal((300, 8))
x_syn  = rng.standard_normal((1500, 8))   # iid → DUPI ≈ benchmark

out  = dupi_score(x_real, x_syn, k=1)         # Eq. (11)
b    = kth_dupi_benchmark(*out["n_real":"n_synthetic"], 1)  # placeholder, see below
ui_pi = ui_pi_from_dupi(out["dupi"], out["dupi_benchmark"])   # Eqs. (12)–(13)

print(out["dupi"], out["dupi_benchmark"])
print(ui_pi["utility_index"], ui_pi["privacy_index"])
```

End-to-end pipeline that also runs distribution distances + per-class
breakdown:

```python
from src.evaluation import evaluate

report = evaluate(real_pcs, syn_pcs, real_sp, syn_sp, k=1, tau=5.0)
print(report.dupi)                    # global UI/PI/DUPI
print(report.distribution_distances)  # W2, MMD, centroid distance
print(report.class_metric_rows)       # per-superpopulation breakdown
```

---

## Reproducing the paper's printed examples

The values below are **literal** quotes from the paper. They are checked
in CI:

* **Wine illustration** (p. 722, caption of Fig. 2):
  `DUPI = 0.25, DUPI₀ = 0.5, τ = 5  →  (UI, PI) = (0.652, 0.954)`
* **Optimal point** (p. 722): `DUPI = DUPI₀, τ = 5  →  UI = PI = 0.867`,
  `U × P = 0.751`
* **Theorem 5 upper bound** (Eq. 14): `UI · PI ≤ (arctan(τ/2)/arctan(τ))²`
  with equality iff `DUPI = DUPI₀`
* **Eq. (10) special case** (k = 1): `DUPI₀ = m / (n + m − 1)`
* **Simulation S1** (p. 722): `MVN_5(0, I)` real and synthetic both at
  `m = n` should converge to the benchmark — averaged over 30 reps with
  `m = n = 600` yields DUPI within ±0.02 of `m / (2n − 1)`.

```bash
pytest tests/test_dupi.py -v          # 29 tests, all paper checks included
```

---

## Project example — running the CLI

The project ships a thin CLI shim under `scripts/evaluate_synthetic_metrics.py`
that loads the HiPoDiT 1000 Genomes data layout, projects real + synthetic
to PCA(2), and writes a JSON summary plus per-class CSVs:

```bash
python scripts/evaluate_synthetic_metrics.py \
    --syn-dir   outputs/<run>/synthetic_samples \
    --out-dir   outputs/<run>/evaluation_metrics \
    --dupi-k    1 \
    --tau       5.0
```

Outputs:

* `summary_metrics.json` — global DUPI, UI, PI, U·P, MMD, W2, centroid distance
* `class_metrics.csv` — per-superpopulation breakdown
* `centroids.csv` — real vs synthetic PC1/PC2 centroids
* `pca_coordinates.csv` — full PC1/PC2 scores (cached for re-runs)

A run on `gw_0p5` (n_real = 251, n_synthetic = 2504) produces:

| Metric | Value |
| --- | --- |
| DUPI (k = 1) | 0.530 |
| DUPI₀ benchmark | 0.909 |
| Privacy Index (τ = 5) | 0.943 |
| Utility Index | 0.706 |
| U · P | 0.666 |

---

## Module layout

```
src/evaluation/
├── __init__.py                 # public API re-exports
├── dupi.py                     # Eqs. (8), (10), (11), (12), (13)  ← citable core
├── distribution_metrics.py     # Gaussian W2, MMD-RBF, coverage
├── synthetic_pipeline.py       # high-level evaluate(), EvaluationReport
└── _io.py                      # project-specific loaders / caches
```

`dupi.py` and `distribution_metrics.py` are pure (no file or network
I/O) and depend only on `numpy` + `scikit-learn`. They can be vendored
verbatim into another project.

---

## Tests

```bash
pytest tests/test_dupi.py -v
```

* `TestDupiBenchmark` — closed-form Eq. (10), unit-interval, invalid `k`
* `TestDupiScore` — bounded range, identical-distribution convergence,
  far / overlap / too-few-samples extremes
* `TestUiPi` — Eqs. (12)–(13) edge cases, atan-sigmoid symmetry
* `TestDistributionMetrics` — W2 / MMD / coverage non-negativity, zero
  on identical input
* `TestPaperReproduction` — Wine illustration, optimal point, Theorem 5
  upper bound, default `τ`, S1 simulation, Eq. (8) self-exclusion

29 tests · runs in < 1 s.

---

## Assumptions inherited from the paper

1. **No duplicates** in `x_real` (paper §III.B(b)). The code identifies
   the column-0 nearest neighbour as the query point itself; if you have
   exact duplicates, deduplicate first or expect undefined behaviour.
2. **Common support** between real and synthetic distributions
   (§III.B(a)).
3. The provided metric is a **proper distance** in the chosen feature
   space (§III).

---

## License & citation

If you use this module, please cite **both** the original paper and this
package — see the project-root `CITATION.cff`.

The package itself inherits the project's repository license. Re-usable
files (`dupi.py`, `distribution_metrics.py`) are written so they can be
vendored into another project under the same terms.
