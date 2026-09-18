# HiPoDiT literacy refactor — 2026-09-18

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 동작을 바꾸지 않고 코드를 읽기 쉽게 만든다. 진입점 함수 분할, 경로 해킹 제거, 죽은 주석·래퍼 제거, 검증 방식 통일.

**Architecture:** 2026-09-18 두 번째 ponytail-audit(리터러시)의 12항목 중 동결 규칙에 걸리지 않고 CPU에서 검증 가능한 7개를 태스크로 만든다. 각 태스크는 TDD 순서를 따른다. 새 동작에는 먼저 실패하는 테스트를, 순수 리팩터에는 기존 green 테스트 아래에서 행동 고정(characterization) 테스트를 둔다. 어느 쪽인지 태스크마다 적었다.

**Tech Stack:** Python 3.13 (루트 `.venv`, CPU), pytest, PyTorch.

**Spec:** 이 기획서 자체. 선행: `2026-09-18-hipodit-ponytail-refactor-2.md`.

## Global Constraints

1. 숫자는 움직이지 않는다. chr17 동결 gate 결과에 영향을 줄 수 있는 변경은 범위 밖.
2. `outputs/diagnostics/`는 건드리지 않는다.
3. 테스트 기준선 `260 passed, 2 warnings`. 명령 `.venv/bin/python -m pytest tests -q`. 각 태스크는 자기가 더한 수만큼만 늘어난 숫자로 끝난다.
4. `uv sync` 금지. GPU 금지. 루트 `.venv`로만 테스트. csdi 환경은 Task 1의 설치 한 줄만 허용.
5. 커밋에 Claude co-author 트레일러를 넣지 않는다.
6. 손대지 않은 줄은 재포맷하지 않는다.

## 범위 밖 (audit 항목 중 제외, 이유)

- **`train()` 233줄 분할** — GPU 없이는 한 번도 실행해 볼 수 없다. 검증 못 하는 리팩터는 하지 않는다.
- **`fit_decoder` 16인자** — 동결 연구 코드.
- **`_atomic_json` 3벌 통합** — 두 벌은 `allow_nan` 기본, 한 벌은 `allow_nan=False`. 하나로 합치면 어느 쪽이든 동작이 바뀐다. 동결 스크립트라 그대로 둔다.
- **legacy `clip` 기본값** — `version` 키 없는 stats 파일이 `data/processed_pca_backup_*`, `processed_v1_contaminated`, `processed_glmpca_asym_legacy`에 실재한다. 지우면 그 디렉터리의 `apply_normalization` 동작이 바뀐다. 한 줄 아끼자고 건드릴 일이 아니다.

---

## Task 1 — 프로젝트를 설치 가능하게 만들고 `sys.path` 해킹 14개를 지운다

**근거:** `src`와 `scripts`는 저장소 루트를 `sys.path`에 밀어 넣는 5줄 블록으로만 import된다. 14개 파일이 같은 블록을 복사해 갖고 있다. 프로젝트는 설치돼 있지 않다(`/tmp`에서 `import src` 실패). `pyproject.toml`에 `[build-system]`이 없어 `uv sync`가 프로젝트를 가상(virtual)으로 취급하기 때문이다.

**방식:** `[build-system]`을 추가하고 `src`, `scripts`를 패키지로 선언한다. `.venv`에는 `uv pip install -e . --no-deps`, csdi에는 `pip install -e . --no-deps --ignore-requires-python`(Python 3.10이라 `requires-python` 무시 필요). 이후 `uv sync`는 자동으로 editable 설치를 유지한다.

**TDD:** 새 동작 — "루트 밖에서 `src`와 `scripts`가 import된다". 테스트 먼저, RED 확인 후 설치.

**Files:** `pyproject.toml`, `scripts/__init__.py`(신규, 빈 파일), `tests/test_project_importable.py`(신규), sys.path 블록을 가진 14개 파일, `README.md`/`CLAUDE.md` 환경 설치 절.

- [ ] Step 1 (RED): `tests/test_project_importable.py`

```python
import subprocess
import sys
from pathlib import Path


def test_src_and_scripts_import_from_any_cwd(tmp_path: Path) -> None:
    # Given an interpreter started outside the repository.
    result = subprocess.run(
        [sys.executable, "-c", "import src.models, scripts.hipodit_multiseed_summary"],
        cwd=tmp_path, capture_output=True, text=True, check=False,
    )
    # Then both packages resolve without any sys.path manipulation.
    assert result.returncode == 0, result.stderr
```

Run → FAIL with `ModuleNotFoundError: No module named 'src'`.

- [ ] Step 2 (GREEN): `pyproject.toml`에 추가

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["."]
include = ["src*", "scripts*"]
```

(기존 `[tool.setuptools.packages.find] where = ["."]` 블록을 위 블록으로 교체.) `scripts/__init__.py` 빈 파일 생성. 설치:

```bash
uv pip install -e . --no-deps --python .venv/bin/python
/home/user/Envs/csdi/bin/pip install -e . --no-deps --ignore-requires-python -q
```

Run test → PASS. `/tmp`에서 csdi로도 `import src` 확인.

- [ ] Step 3 (REFACTOR): 14개 파일에서 `sys.path.insert` 블록 제거. 파일마다 모양이 셋 중 하나다.

```python
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
```
```python
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
```
```python
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
```

`if ... sys.path.insert` 두 줄(또는 한 줄)만 지운다. `PROJECT_ROOT` 변수는 `scripts/hipodit_*`가 경로 계산에 쓰므로 남긴다. `_SCRIPT_DIR`/`_PROJECT_ROOT`가 그 뒤에 안 쓰이면 함께 지우고, `import sys`/`import os`가 고아가 되면 지운다. `scripts/plot_pca_train_fit_test_overlay.py`의 `sys.path.insert(0, str(Path(__file__).parent))`(형제 import용)도 지우고 `from plot_pca import`를 `from scripts.plot_pca import`로 바꾼다. `hipodit_rebuild_prepare/train.py`의 `from hipodit_genotype_check import`는 `hipodit_rebuild_check.py`가 `scripts/`를 cwd-path로 갖고 실행되므로 그대로 둔다 — **단, 그 두 파일은 `hipodit_multiseed`의 `source_sha256`에 들어가므로 이번 태스크에서 바꾸지 않는다** (아래 "동결 파일" 참조).

**동결 파일:** `scripts/hipodit_*.py` 7개는 `hipodit_multiseed.py`가 manifest에 sha256을 적는 파일이다. 지우는 것은 sys.path 두 줄뿐이고 실행 결과는 같지만, 다음 multiseed 실행의 `source_sha256`는 달라진다. 이것은 동결 산출물을 바꾸는 것이 아니라 새 산출물의 provenance가 새 소스를 가리키는 것이므로 허용한다.

- [ ] Step 4: 전 CLI `--help` 스모크 + 전체 테스트 → `261 passed`.
- [ ] Step 5: README `uv sync` 다음 줄과 CLAUDE.md `# Environment` 블록에 csdi 설치 한 줄 추가. 커밋.

---

## Task 2 — `load_synthetic`의 방향 판정 사슬을 편다

**근거:** `src/evaluation/_io.py:100-107`의 `if / elif / elif`는 네 경우를 한 사슬에 접었다. 읽고 나서 "언제 전치되는가"를 말하기 어렵다.

**TDD:** 순수 리팩터. 네 경우 중 테스트가 없는 것 — "어느 방향도 stats와 안 맞으면 raise" — 을 먼저 characterization 테스트로 고정(현재 green), 그 아래에서 편다.

**Files:** `src/evaluation/_io.py`, `tests/test_evaluation_io_contract.py`

- [ ] Step 1: 테스트 추가 (green 확인 — 행동 고정용)

```python
def test_synthetic_loader_rejects_a_shape_that_matches_neither_orientation(tmp_path: Path) -> None:
    stats_path = tmp_path / "stats.pkl"
    _write_stats(stats_path, np.zeros((3, 2), dtype=np.float32), np.ones((3, 2), dtype=np.float32))
    syn_dir = tmp_path / "synthetic"
    _write_sample(syn_dir, np.ones((4, 4), dtype=np.float32))
    (syn_dir / "generation_meta.json").write_text(json.dumps({"sample_space": "original"}))

    with pytest.raises(ValueError, match="normalization shape"):
        _io.load_synthetic(syn_dir, stats_path=stats_path)
```

- [ ] Step 2 (REFACTOR): 사슬을 아래로 교체

```python
        if expected_shape is None:
            if arr.shape[0] < arr.shape[1]:
                arr = arr.T  # (K, gene_size) as the generator writes it
        else:
            if arr.shape != expected_shape:
                arr = arr.T
            if arr.shape != expected_shape:
                raise ValueError(
                    f"{file_path} shape {arr.T.shape} does not match normalization shape {expected_shape}"
                )
```

- [ ] Step 3: 테스트 → `262 passed`. 커밋.

---

## Task 3 — trainer의 죽은 주석·래퍼와 ddp 중복을 정리한다

**근거:** `cosine_warmup_lr_lambda` docstring이 "verified in scratchpad"라고 적혀 있다. 스크래치패드는 없다. `make_cosine_warmup_scheduler`는 호출 1곳의 두 줄 래퍼. `src/utils/ddp.py`의 `is_main_process`/`get_rank`/`get_world_size`는 같은 `dist.is_initialized()` 분기 세 벌.

**TDD:** 순수 리팩터. `cosine_warmup_lr_lambda`는 순수 함수인데 테스트가 없다 — 분기(warmup/decay)를 가진 로직이므로 characterization 테스트를 먼저 둔다. ddp 셋은 dist 미초기화 값을 고정한다.

**Files:** `src/training/trainer.py`, `src/utils/ddp.py`, `tests/test_trainer_schedule.py`(신규), `tests/test_ddp_helpers.py`(신규)

- [ ] Step 1: 두 테스트 파일 작성 (green 확인)

```python
# tests/test_trainer_schedule.py
from src.training.trainer import cosine_warmup_lr_lambda


def test_warmup_is_linear_then_cosine_decays_to_zero():
    assert cosine_warmup_lr_lambda(0, warmup=10, max_iters=110) == 0.0
    assert cosine_warmup_lr_lambda(5, warmup=10, max_iters=110) == 0.5
    assert cosine_warmup_lr_lambda(10, warmup=10, max_iters=110) == 1.0
    assert abs(cosine_warmup_lr_lambda(60, warmup=10, max_iters=110) - 0.5) < 1e-12
    assert abs(cosine_warmup_lr_lambda(110, warmup=10, max_iters=110)) < 1e-12
```

```python
# tests/test_ddp_helpers.py
from src.utils.ddp import get_rank, get_world_size, is_main_process


def test_without_a_process_group_every_process_is_rank_zero_of_one():
    assert get_rank() == 0
    assert get_world_size() == 1
    assert is_main_process()
```

- [ ] Step 2 (REFACTOR): trainer에서 docstring의 "Reproduces the old hand-rolled ... rel_tol=1e-12" 두 줄 삭제, `make_cosine_warmup_scheduler` 삭제, 호출부를

```python
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        partial(cosine_warmup_lr_lambda,
                warmup=training_cfg.get("warmup_steps", 100), max_iters=total_steps),
    )
```

ddp.py는

```python
def get_rank() -> int:
    """Global rank, or 0 when no process group is initialized."""
    return dist.get_rank() if dist.is_initialized() else 0


def get_world_size() -> int:
    """Process count, or 1 when no process group is initialized."""
    return dist.get_world_size() if dist.is_initialized() else 1


def is_main_process() -> bool:
    return get_rank() == 0
```

- [ ] Step 3: 테스트 → `264 passed`. 커밋.

---

## Task 4 — 모델의 `assert`를 경계 검증과 내부 불변식으로 나눈다

**근거:** `src/models`에 `assert` 11개. `python -O`면 사라진다. 셋은 사용자 입력 경계(텐서 shape이 config와 맞는가, `seq_len % patch_size`), 나머지는 같은 생성자가 함께 만든 객체 간의 불변식(FiLM 파라미터 수 = 블록 수 등)이라 깨질 수 없다.

**TDD:** 경계 셋은 **행동 변경**(AssertionError → ValueError). 테스트 먼저, RED 확인. 내부 여덟은 삭제(테스트 없음 — 도달 불가).

**Files:** `src/models/hybrid_geno_dit.py`, `src/models/modules/dit.py`, `cnn.py`, `base.py`, `conditioning.py`, `tests/test_model_input_contract.py`(신규)

- [ ] Step 1 (RED)

```python
import pytest
import torch

from src.models import HybridCNNDiTFiLM
from src.models.modules.dit import PatchEmbed1D


def _model(gene_size: int = 512) -> HybridCNNDiTFiLM:
    config = {
        "data": {"num_channels": 4, "gene_size": gene_size},
        "model": {"base_channels": 8, "channel_mult": [1, 1, 2, 2, 4], "d_model": 16,
                  "n_dit_blocks": 1, "n_heads": 1, "patch_size": 16,
                  "pop_to_superpop": {i: i % 5 for i in range(26)}},
    }
    return HybridCNNDiTFiLM(config)


def test_forward_rejects_a_channel_count_that_differs_from_config():
    model = _model()
    with pytest.raises(ValueError, match="channels"):
        model(torch.randn(2, 3, 512), torch.zeros(2, dtype=torch.long), torch.zeros(2, dtype=torch.long))


def test_forward_rejects_a_gene_size_that_differs_from_config():
    model = _model()
    with pytest.raises(ValueError, match="gene_size"):
        model(torch.randn(2, 4, 256), torch.zeros(2, dtype=torch.long), torch.zeros(2, dtype=torch.long))


def test_patch_embed_rejects_a_sequence_the_patch_size_does_not_divide():
    with pytest.raises(ValueError, match="patch_size"):
        PatchEmbed1D(seq_len=10, in_channels=1, patch_size=4, d_model=8)
```

Run → 3 FAIL with `AssertionError` (not ValueError).

- [ ] Step 2 (GREEN): `hybrid_geno_dit.forward`의 assert 셋을 하나의 `ValueError`로

```python
        if x.dim() != 3 or x.shape[1] != self.in_channels or x.shape[2] != self.gene_size:
            raise ValueError(
                f"Expected (B, {self.in_channels} channels, {self.gene_size} gene_size), got {tuple(x.shape)}"
            )
```

`PatchEmbed1D.__init__`의 assert를 `if seq_len % patch_size: raise ValueError(f"seq_len {seq_len} not divisible by patch_size {patch_size}")`로. `hybrid_geno_dit.__init__:75`의 divisibility assert도 같은 이유로 `ValueError`.

- [ ] Step 3 (REFACTOR): 내부 불변식 assert 7개 삭제 — `dit.py:141`, `dit.py:199`, `cnn.py:187`, `cnn.py:250`, `base.py:23`, `conditioning.py:67`. 각각 여러 줄이면 괄호까지.

- [ ] Step 4: `grep -rn "^\s*assert " src` → 출력 없음. 테스트 → `267 passed`. 커밋.

---

## Task 5 — 테스트의 함수 안 import를 모듈 상단으로 올린다

**근거:** 8개 테스트 파일에서 37곳이 테스트 함수 안에서 `from src... import`를 한다. 파일 머리에서 의존성이 안 보인다. 한 가지 예외: `test_binomial_glm_pca.py`의 `import src.preprocessing.binomial_glm_pca as module`은 monkeypatch 대상이라 그 자리에 둬도 되지만, 상단 import와 충돌하지 않으므로 함께 올린다.

**TDD:** 테스트 코드만 바뀐다. 전체 suite가 계속 green이면 된다.

**Files:** `tests/test_binomial_glm_pca.py`, `test_binom_glmpca_rs.py`, `test_hipodit_rebuild_check.py`, `test_vcf_parser.py`, `test_dupi.py`, `test_glm_pca_preprocessing.py`, `test_hipodit_genotype_check.py`, `test_normalization_contract.py`

- [ ] Step 1: 파일마다 함수 안 `from X import a, b` 줄을 모두 모아 중복 제거 후 모듈 상단(기존 import 블록 뒤)에 두고, 함수 안의 줄을 지운다. 같은 이름을 두 함수가 다른 모듈에서 가져오는 경우는 없는지 `grep`으로 확인.
- [ ] Step 2: 테스트 → `267 passed`. 커밋.

---

## Task 6 — `src/evaluation/__init__.py` re-export를 지운다

**근거:** 53줄의 re-export와 `__all__`. 소비자는 `scripts/evaluate_synthetic_metrics.py`의 `evaluate` 하나와 `src/evaluation/README.md`의 예시 두 줄. 나머지 모듈의 `__all__` 4개도 문서 외 역할이 없다.

**TDD:** 순수 리팩터. `scripts/evaluate_synthetic_metrics.py --help` 서브프로세스 테스트가 이미 있어 import 경로를 지킨다.

**Files:** `src/evaluation/__init__.py`, `dupi.py`, `distribution_metrics.py`, `synthetic_pipeline.py`, `_io.py`, `scripts/evaluate_synthetic_metrics.py`, `src/evaluation/README.md`

- [ ] Step 1: `__init__.py`를 3줄 docstring만 남긴다.

```python
"""Real-vs-synthetic evaluation: DUPI (src.evaluation.dupi), distribution
distances (src.evaluation.distribution_metrics), the evaluate() pipeline
(src.evaluation.synthetic_pipeline) and project IO (src.evaluation._io)."""
```

- [ ] Step 2: 네 모듈의 `__all__ = [...]` 블록 삭제. 스크립트의 `from src.evaluation import evaluate`를 `from src.evaluation.synthetic_pipeline import evaluate`로. README.md의 두 예시 import 경로도 모듈 경로로.
- [ ] Step 3: 테스트 → `267 passed`. 커밋.

---

## Task 7 — `generate_samples` 261줄을 셋으로 나눈다

**근거:** 14인자·261줄. 세 단락이 이미 주석으로 나뉘어 있다: 모델·guide 로드, 인구군별 개수 결정, 생성 루프+meta. 각각을 함수로 뽑으면 본체는 화면 하나다.

**TDD:** 인구군별 개수 결정은 순수 함수가 된다 — 새 함수이므로 테스트 먼저 RED. 나머지 둘은 기존 `_run_generation` 계약 테스트 6건이 끝에서 끝까지 지킨다.

**Files:** `src/inference/generator.py`, `tests/test_inference_generator_contract.py`

- [ ] Step 1 (RED): 테스트 추가

```python
def test_samples_per_population_applies_floor_then_cap() -> None:
    counts = generator.samples_per_population(
        {0: 5, 1: 300}, oversample_minority=50, max_per_pop=100,
    )

    assert counts == {0: 50, 1: 100}


def test_samples_per_population_rejects_a_nonpositive_cap() -> None:
    with pytest.raises(ValueError, match="max_per_pop"):
        generator.samples_per_population({0: 5}, oversample_minority=None, max_per_pop=0)
```

Run → FAIL `AttributeError: module has no attribute 'samples_per_population'`.

- [ ] Step 2 (GREEN): 세 함수 추출

```python
def samples_per_population(
    counts: dict[int, int], oversample_minority: int | None, max_per_pop: int | None,
) -> dict[int, int]:
    """Apply the minority floor, then the per-population cap."""
    if oversample_minority is not None:
        counts = {k: max(v, oversample_minority) for k, v in counts.items()}
    if max_per_pop is not None:
        if max_per_pop < 1:
            raise ValueError(f"max_per_pop must be positive, got {max_per_pop}")
        counts = {k: min(v, max_per_pop) for k, v in counts.items()}
    return counts


def real_population_sizes(label_hierarchy_path: str) -> dict[int, int]:
    """{pop_idx: count} from label_hierarchy.pkl, the real data's composition."""
    with open(label_hierarchy_path, "rb") as f:
        info = pickle.load(f)
    return {info["pop_to_idx"][name]: n for name, n in info["pop_sizes"].items()
            if name in info["pop_to_idx"]}


def load_guide_model(
    guide_model_path: str, diffusion_cfg: dict, fallback_config: dict, device: torch.device,
) -> HybridCNNDiTFiLM:
    """Load the autoguidance model, refusing one trained on another noise schedule."""
    if not Path(guide_model_path).exists():
        raise FileNotFoundError(f"Guide model not found: {guide_model_path}")
    checkpoint = torch.load(guide_model_path, map_location=device, weights_only=False)
    guide_schedule = checkpoint.get("config", {}).get("diffusion", {}).get("noise_schedule", "cosine")
    if guide_schedule != diffusion_cfg.get("noise_schedule", "cosine"):
        raise ValueError(
            f"Guide model noise_schedule ({guide_schedule}) does not match the main "
            f"checkpoint ({diffusion_cfg.get('noise_schedule', 'cosine')})"
        )
    model, used_ema = load_generator_model(checkpoint, fallback_config, device)
    if not used_ema:
        logger.warning("EMA not found in guide checkpoint, using raw guide model weights")
    logger.info(f"Guide model loaded from {guide_model_path}")
    return model
```

`generate_samples` 안의 해당 블록 셋을 이 호출로 바꾼다. 로그 문장과 순서는 유지한다(계약 테스트가 `caplog`로 본다).

- [ ] Step 3: 테스트 → `269 passed`. 커밋.

---

## Self-Review

- **커버리지:** audit 12항목 중 7개 태스크(1, 5, 7, 8, 9, 10, 11, 12번 항목)가 들어갔고, 제외 4개(`train` 분할, `fit_decoder`, `_atomic_json`, legacy clip)는 이유와 함께 위에 적었다.
- **TDD 구분:** RED가 진짜 실패하는 태스크는 1, 4, 7. 나머지는 순수 리팩터로 characterization 테스트 아래에서 진행하며, 그 테스트가 즉시 green인 것은 의도된 것이다.
- **테스트 수 추적:** 260 → T1 +1 → T2 +1 → T3 +2 → T4 +3 → T5 ±0 → T6 ±0 → T7 +2 = 269.
- **동결 파일:** `scripts/hipodit_*.py` 변경은 T1의 sys.path 두 줄뿐이다.
