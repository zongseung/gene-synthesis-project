# HiPoDiT ponytail refactor 2 — 2026-09-18

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 동작을 바꾸지 않고 과설계·미도달 코드 약 900줄과 미사용 의존성 2개를 제거한다.

**Architecture:** 2026-09-18 ponytail-audit이 찾은 항목을 두 묶음으로 나눈다. Group B/C(Task 1–17)는 관측 가능한 동작이 바뀌지 않는 제거이므로 이 기획서로 바로 실행한다. Group A(Task 18–22)는 연구 기록·재현 명령과 얽혀 있어 **사용자 승인 없이는 착수하지 않는다**. 각 태스크는 삭제 → 테스트 → 커밋으로 끝난다.

**Tech Stack:** Python 3.13 (루트 `.venv`, CPU 전용), pytest, PyTorch, numpy, scikit-learn.

**Spec:** 이 기획서 자체가 명세다. 근거는 2026-09-18 ponytail-audit 결과이며, 각 태스크의 "근거" 항목에 파일·줄 단위로 적어 두었다. 선행 기획서는 `docs/superpowers/plans/2026-09-16-hipodit-ponytail-refactor.md`이고, 이번 것은 그 후속이다.

---

## Global Constraints

1. **숫자는 움직이지 않는다.** chr17 연구는 동결·공표된 gate 결과를 가지고 있다 (`docs/reports/hipodit_t_results_20260916.md`, `docs/reports/hipodit_t_decision_log_20260916.md`). metric, RNG 추출 순서, seed, fit, 저장 산출물의 바이트를 바꿀 수 있는 변경은 범위 밖이다. 애매하면 바꾸지 말고 보고한다.
2. **`outputs/diagnostics/` 아래 동결 산출물은 건드리지 않는다.** 재생성도 금지다 (CLAUDE.md 규정).
3. **테스트 기준선은 `267 passed, 2 warnings`이다.** 명령: `.venv/bin/python -m pytest tests -q`. 각 태스크는 자기가 삭제한 테스트 수만큼만 줄어든 숫자로 끝나야 하고, 그 수는 태스크 안에 명시돼 있다. 최종 기대값은 Task 17 종료 시 `258 passed`다.
4. **루트 `.venv`(Python 3.13, CPU)만 쓴다.** `uv sync`를 실행하지 않는다 (메모리: 장시간 작업 중 의존성 재동기화 회피). GPU는 쓰지 않는다. `/home/user/Envs/csdi`도 이 작업에서는 쓰지 않는다.
5. **`korean-medicine-llm/`은 범위 밖이다.** 별도 프로젝트이며 별도 virtualenv를 쓴다.
6. **손대지 않은 줄은 재포맷하지 않는다.** 가능한 한 diff는 삭제만 담는다.
7. **커밋 트레일러:** 사용자가 지정한 co-author만 넣는다. Claude/Opus co-author 트레일러는 넣지 않는다 (메모리 규칙).
8. **Group A(Task 18–22)는 사용자 승인 게이트다.** 승인 문장 없이 착수하면 안 된다.

---

## File Structure

새로 만드는 파일은 없다. 수정·삭제 대상만 적는다.

| 파일 | 이 리팩터에서의 책임 |
|---|---|
| `src/utils/config.py` | YAML 로드 한 가지만 한다. CLI override·필수키 검증은 제거된다. |
| `src/utils/logger.py` | **삭제.** wandb 호출이 trainer로 인라인된다. |
| `src/utils/__init__.py` | `ExperimentLogger` re-export 제거. |
| `src/training/trainer.py` | 학습 루프. best 체크포인트 1개만 저장하고 precision/optimizer 분기가 사라진다. |
| `src/data/dataset.py` | pkl → tensor. 검증은 shape 1건만. |
| `src/data/sampler.py` | sqrt 비례 가중치를 벡터 연산으로. |
| `src/data/dataloader.py` | DataLoader 팩토리. 커스텀 예외 제거. |
| `src/models/diffusion.py` | DDIM 샘플러 + 손실. DDPM 샘플러와 `prediction_target` 인자 제거. |
| `src/models/hybrid_geno_dit.py` | 모델 조립. config 이중 포맷 fallback 제거. |
| `src/models/modules/conditioning.py` | FiLM 생성기. `cnn_dec_channels=None` fallback 제거. |
| `src/inference/generator.py` | 생성 CLI. 단일 호출 헬퍼 인라인. |
| `src/evaluation/_io.py` | 평가 IO. 캐시 계층과 legacy sample-space 배관 제거. |
| `scripts/evaluate_synthetic_metrics.py` | 평가 CLI. 캐시·legacy 플래그 제거. |
| `src/preprocessing/glm_pca.py` | 유전자별 GLM-PCA. 단일 사용 예외와 import 래퍼 제거. |
| `src/preprocessing/labels.py` | 레이블·split·저장. 미도달 재분할 경로 제거. |
| `src/preprocessing/config.py` | 전처리 상수. 아무도 안 바꾸는 환경변수 제거. |
| `scripts/plot_pca.py` | PCA 3-panel 그림. 로더를 `src.evaluation._io`로 통일. |
| `pyproject.toml` | 미사용 의존성 제거, `scipy` 선언 추가. |
| `README.md`, `CLAUDE.md` | 사라진 플래그·키·파일 설명 갱신. |

---

## Task 1 — YAML 설정 로더를 로드 한 가지로 줄인다

**근거:** `src/utils/config.py:32-59`의 `_set_nested`/`_cast_value`는 `load_config(overrides=...)` 전용이고, 저장소 전체에서 override를 넘기는 호출자는 없다. `parse_args_with_config`이 `unknown` 인자에서 override를 만들지만 `configs/sweep.yaml`은 저장소에 없다(CLAUDE.md가 명시). `REQUIRED_KEYS`/`validate_config`(`:18-29`, `:118-131`)은 외부 호출자가 없고, 누락 키는 어차피 사용 지점에서 `KeyError`로 멈춘다. `tests/`에 이 모듈을 직접 다루는 테스트는 없다.

**Files:**
- Modify: `src/utils/config.py` (전면)
- Modify: `src/utils/__init__.py:1`

**Interfaces:**
- Consumes: 없음 (첫 태스크)
- Produces: `load_config(path: str | Path) -> dict`, `parse_args_with_config() -> dict` — 두 이름과 시그니처는 유지된다. `load_config`의 `overrides` 매개변수만 사라진다. `src/inference/generator.py:530`과 `scripts/plot_pca.py`가 `load_config(path)` 한 인자로 호출하므로 호출부 변경은 필요 없다.

- [ ] **Step 1: 현재 기준선 확인**

```bash
cd /home/user/gene-synthesis-project
.venv/bin/python -m pytest tests -q 2>&1 | tail -2
```
Expected: `267 passed, 2 warnings`

- [ ] **Step 2: `src/utils/config.py` 전체를 아래로 교체**

```python
"""YAML configuration loader.

Config is accessed as a nested dict: ``config['data']['gene_size']``.
Missing keys raise KeyError at the point of use, which names the key.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    """Load a YAML config file into a nested dict.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def parse_args_with_config() -> dict:
    """Parse ``--config`` / ``--single_gpu`` and load the config.

    Expected CLI usage:
        python src/training/trainer.py --config configs/default.yaml
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--single_gpu", action="store_true", help="Run on single GPU (no DDP)")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.single_gpu:
        config["distributed"]["num_gpus"] = 1
    return config
```

주의: 기존 `parse_known_args()`는 `parse_args()`로 바뀐다. override를 지우면 알 수 없는 인자는 오류여야 맞다.

- [ ] **Step 3: `src/utils/__init__.py` 첫 줄을 그대로 둔다**

`from src.utils.config import load_config`는 여전히 유효하다. 이 스텝에서는 아무것도 바꾸지 않고 확인만 한다.

```bash
.venv/bin/python -c "from src.utils import load_config; print(load_config('configs/default.yaml')['data']['gene_size'])"
```
Expected: `24576`

- [ ] **Step 4: 테스트**

```bash
.venv/bin/python -m pytest tests -q 2>&1 | tail -2
```
Expected: `267 passed, 2 warnings`

- [ ] **Step 5: 커밋**

```bash
git add src/utils/config.py
git commit -m "refactor(config): drop the unused CLI override and required-key machinery"
```

---

## Task 2 — wandb 래퍼를 지우고 호출을 trainer로 인라인한다

**근거:** `wandb`는 `pyproject.toml`의 선언된 의존성이므로 `src/utils/logger.py:16-22`의 "미설치 시 graceful degradation"은 일어날 수 없는 경우를 위한 분기다. `ExperimentLogger`의 유일한 호출자는 `src/training/trainer.py:372-378, 460, 472-479, 515`이고, `enabled=is_main_process()`를 넘기므로 클래스 안의 두 번째 rank 검사(`:45`)는 중복이다. 테스트 참조 없음.

**Files:**
- Delete: `src/utils/logger.py`
- Modify: `src/utils/__init__.py:4`
- Modify: `src/training/trainer.py:53`, `:370-378`, `:447-460`, `:470-479`, `:515`

**Interfaces:**
- Consumes: Task 1의 `parse_args_with_config`
- Produces: trainer 안에서만 쓰이는 지역 패턴. 외부에 새 이름을 내보내지 않는다.

- [ ] **Step 1: `src/utils/logger.py` 삭제, `src/utils/__init__.py`에서 re-export 제거**

```bash
git rm src/utils/logger.py
```

`src/utils/__init__.py`에서 아래 줄을 지운다.

```python
from src.utils.logger import ExperimentLogger
```

- [ ] **Step 2: trainer의 import 교체**

`src/training/trainer.py:53`의 `from src.utils.logger import ExperimentLogger`를 지우고, 파일 상단 import 블록(`import torch.nn as nn` 다음 줄)에 추가한다.

```python
import wandb
```

- [ ] **Step 3: 초기화 블록 교체**

`src/training/trainer.py`의 `# ── wandb (rank 0 only) ──` 블록을 아래로 바꾼다.

```python
    # ── wandb (rank 0 only) ──
    exp_cfg = config.get("experiment", {})
    log_to_wandb = is_main_process()
    if log_to_wandb:
        wandb.init(
            project="HiPoDiT",
            name=exp_cfg.get("run_name", None),
            tags=exp_cfg.get("tags", None),
            reinit=True,
        )
        wandb.config.update(config, allow_val_change=True)
```

- [ ] **Step 4: 세 개의 로깅 지점 교체**

학습 루프 안 `wb_logger.log_metrics(global_step, metrics)`를 아래로 바꾼다.

```python
                wandb.log(metrics, step=global_step)
```

에폭 말미 `wb_logger.log_metrics(global_step, {...})` 호출을 아래로 바꾼다.

```python
            wandb.log(
                {
                    "val/reconstruction_error": val_rec_error,
                    "val/epoch": epoch,
                    "train/avg_epoch_loss": avg_loss,
                },
                step=global_step,
            )
```

두 지점은 이미 `if is_main_process()` 안에 있으므로 추가 가드가 필요 없다. 파일 끝 `wb_logger.finish()`를 아래로 바꾼다.

```python
    if log_to_wandb:
        wandb.finish()
```

- [ ] **Step 5: trainer가 임포트되는지 확인**

```bash
.venv/bin/python -c "import src.training.trainer as t; print(t.train.__name__)"
```
Expected: `train`

- [ ] **Step 6: 테스트**

```bash
.venv/bin/python -m pytest tests -q 2>&1 | tail -2
```
Expected: `267 passed, 2 warnings`

- [ ] **Step 7: 커밋**

```bash
git add -u src/utils src/training/trainer.py
git commit -m "refactor(logging): call wandb directly instead of through a one-caller wrapper"
```

---

## Task 3 — 재개 코드가 없는 체크포인트 보관 장치를 제거한다

**근거:** 재개 경로가 존재하지 않는다. `EMAModel.load_state_dict`는 저장소 전체에서 호출되지 않고, trainer에 `--resume` 플래그도 없다. 그래서 `save_checkpoint`가 담는 `optimizer_state_dict`/`scheduler_state_dict`/`ema_state_dict`는 읽히지 않으며, `manage_top_k_checkpoints`와 `checkpoint_final.pth`는 쓰기만 된다. 추론은 `best_model.pth`의 `model_state_dict`와 `ema_state_dict`만 읽는다(`src/inference/generator.py:126-137`).

**단, `ema_state_dict`는 남긴다.** `load_generator_model`이 그것으로 EMA 가중치를 모델에 덮어쓰기 때문이다. 지우는 것은 optimizer/scheduler state, 주기 체크포인트, top-k 관리, final 체크포인트뿐이다.

**Files:**
- Modify: `src/training/trainer.py:196-252`, `:496-513`
- Modify: `configs/default.yaml:62`, `:68`

**Interfaces:**
- Consumes: Task 2 이후의 trainer
- Produces: `save_checkpoint(model, ema, epoch, val_loss, config, path) -> None` — optimizer/scheduler 매개변수가 사라진 6-인자 형태. 저장되는 키는 `model_state_dict`, `ema_state_dict`, `epoch`, `best_val_loss`, `config`, `pytorch_version`, `cuda_version`.

- [ ] **Step 1: `save_checkpoint`를 아래로 교체**

```python
def save_checkpoint(
    model: nn.Module,
    ema: EMAModel,
    epoch: int,
    val_loss: float,
    config: dict,
    path: str,
) -> None:
    """Save the best model (rank 0 only).

    Inference reads ``model_state_dict`` and ``ema_state_dict``
    (src/inference/generator.py). Optimizer and scheduler state are not saved
    because no resume path exists.
    """
    if not is_main_process():
        return

    model_state = (
        model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    )
    checkpoint = {
        "model_state_dict": model_state,
        "ema_state_dict": ema.state_dict(),
        "epoch": epoch,
        "best_val_loss": val_loss,
        "config": config,
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(checkpoint, path)
    logger.info(f"Checkpoint saved: {path}")
```

- [ ] **Step 2: `manage_top_k_checkpoints` 함수 전체를 삭제**

`src/training/trainer.py`에서 `def manage_top_k_checkpoints(...)` 정의와 그 본문(`# ─── Main training function` 주석 직전까지)을 지운다.

- [ ] **Step 3: 호출부 정리**

`best` 저장 호출을 아래로 바꾼다.

```python
            if val_rec_error < best_val_loss:
                best_val_loss = val_rec_error
                save_checkpoint(
                    model, ema, epoch, best_val_loss, config,
                    path=os.path.join(save_dir, "best_model.pth"),
                )
                logger.info(f"  New best: val_rec={val_rec_error:.6f}")
```

같은 블록 안의 `# Periodic checkpoint` 네 줄(`save_every` 조회, `if epoch % save_every`, `save_checkpoint(...)`, `manage_top_k_checkpoints(...)`)을 지운다. 그리고 `# ── Final checkpoint ──` 블록을 아래로 바꾼다.

```python
    # ── Done ──
    if is_main_process():
        logger.info(f"Training complete. Best val_rec={best_val_loss:.6f}")
```

`top_k = exp_cfg.get("save_top_k_checkpoints", 3)` 줄도 지운다.

- [ ] **Step 4: config에서 죽은 키 제거**

`configs/default.yaml`에서 두 줄을 지운다.

```yaml
  save_every: 20                 # checkpoint 저장 주기
```
```yaml
  save_top_k_checkpoints: 3
```

- [ ] **Step 5: trainer import 확인 + 테스트**

```bash
.venv/bin/python -c "import src.training.trainer as t; print(hasattr(t, 'manage_top_k_checkpoints'))"
.venv/bin/python -m pytest tests -q 2>&1 | tail -2
```
Expected: `False` 그리고 `267 passed, 2 warnings`

- [ ] **Step 6: 커밋**

```bash
git add src/training/trainer.py configs/default.yaml
git commit -m "refactor(trainer): keep only the best checkpoint, there is no resume path"
```

---

## Task 4 — 쓰이지 않는 precision/optimizer 분기를 제거한다

**근거:** CLAUDE.md의 Key Design Principles가 bf16과 AdamW를 고정으로 규정한다. `resolve_precision`(`src/training/trainer.py:111-123`)은 `configs/default.yaml:61`의 `"bf16"` 하나만 받아 본 적이 있고, fp32 경로를 실행한 기록이 저장소에 없다. `optimizer_name != "adamw"` 검사(`:346-350`)는 유일한 config 값이 `"adamw"`이므로 항상 통과한다. 두 검사 모두 테스트 참조가 없다.

**Files:**
- Modify: `src/training/trainer.py:104-123`, `:130-189`, `:342-356`, `:408`, `:464-467`
- Modify: `configs/default.yaml:57`, `:61`

**Interfaces:**
- Consumes: Task 3 이후의 trainer
- Produces: `validate(model, diffusion, val_loader, device) -> float` — `config`, `autocast_dtype`, `autocast_enabled` 세 매개변수가 사라진 4-인자 형태. autocast는 함수 안에서 `torch.bfloat16`으로 고정된다.

- [ ] **Step 1: precision 블록 삭제**

`# ── Precision (autocast) resolution ──` 구분선 주석, `PRECISION_DTYPES` 상수, `resolve_precision` 함수를 모두 지운다.

- [ ] **Step 2: `validate` 시그니처와 autocast 고정**

`validate`의 매개변수 목록을 아래로 바꾼다.

```python
@torch.no_grad()
def validate(
    model: nn.Module,
    diffusion: GaussianDiffusion,
    val_loader: DataLoader,
    device: torch.device,
) -> float:
```

본문의 autocast 문맥을 아래로 바꾼다.

```python
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
```

- [ ] **Step 3: 학습 루프의 autocast와 optimizer 검사 정리**

`autocast_dtype, autocast_enabled = resolve_precision(training_cfg)` 줄과 그 위 구분선 주석을 지운다. optimizer 블록을 아래로 바꾼다.

```python
    # ── Optimizer: AdamW, NO GradScaler (bf16 has fp32 dynamic range) ──
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_cfg["lr"],
        weight_decay=training_cfg.get("weight_decay", 1e-4),
        betas=(0.9, 0.999),
    )
```

학습 루프 안의 autocast를 아래로 바꾼다.

```python
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
```

`validate` 호출을 아래로 바꾼다.

```python
        val_rec_error = validate(model, diffusion, val_loader, device)
```

- [ ] **Step 4: config에서 죽은 키 제거**

`configs/default.yaml`에서 두 줄을 지운다.

```yaml
  optimizer: "adamw"
```
```yaml
  precision: "bf16"              # bf16 부동소수점
```

- [ ] **Step 5: 남은 참조가 없는지 확인 + 테스트**

```bash
grep -rn "resolve_precision\|autocast_dtype\|autocast_enabled\|PRECISION_DTYPES" src scripts tests
.venv/bin/python -m pytest tests -q 2>&1 | tail -2
```
Expected: grep은 출력 없음, 테스트는 `267 passed, 2 warnings`

- [ ] **Step 6: 커밋**

```bash
git add src/training/trainer.py configs/default.yaml
git commit -m "refactor(trainer): fix bf16 and AdamW instead of validating one allowed value"
```

---

## Task 5 — Dataset 검증을 shape 한 건으로 줄인다

**근거:** `src/data/dataset.py:41-74`는 pickle 내용에 대해 6단계 검증을 한다. 그 pickle을 쓰는 곳은 `src/preprocessing/labels.py:199-207` 한 곳뿐이고 언제나 `(np.ndarray, np.ndarray)` 2-튜플을 넣는다. `torch.Tensor` 분기(`:50-53`)는 도달하지 않는다. 실제로 지켜야 하는 불변식 — 학습 텐서 shape이 config와 맞는지 — 은 이미 `src/data/dataloader.py:67-73`이 검사하며, 그 검사가 더 좋은 오류 메시지를 낸다.

**Files:**
- Modify: `src/data/dataset.py:33-78`

**Interfaces:**
- Consumes: 없음 (다른 태스크와 독립)
- Produces: `GenotypeDataset(data_path)` — `.x_data` (N,G,K) float32 tensor, `.y_labels` (N,) int64 tensor, `__getitem__` → `(K, gene_size)` float32와 int64 스칼라. 외부 계약 불변. `src/data/dataloader.py:69`가 `.x_data.shape`를, `:81`이 `.y_labels`를 읽는다.

- [ ] **Step 1: `__init__`을 아래로 교체**

```python
    def __init__(self, data_path: str | Path) -> None:
        data_path = Path(data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        with open(data_path, "rb") as f:
            x_data, y_labels = pickle.load(f)

        # (N, gene_size, K); transposed to (K, gene_size) in __getitem__.
        self.x_data = torch.from_numpy(np.asarray(x_data, dtype=np.float32))
        self.y_labels = torch.from_numpy(np.asarray(y_labels, dtype=np.int64))
        if self.x_data.ndim != 3 or len(self.x_data) != len(self.y_labels):
            raise ValueError(
                f"{data_path}: expected (N,G,K) features and (N,) labels, got "
                f"{tuple(self.x_data.shape)} and {tuple(self.y_labels.shape)}"
            )
```

- [ ] **Step 2: 클래스 docstring의 Raises 절 갱신**

```python
    """Dataset for Gene PCA genotype tensors.

    Args:
        data_path: Path to a .pkl file containing (x_data, y_labels).

    Raises:
        FileNotFoundError: If data_path does not exist.
        ValueError: If the arrays are not (N,G,K) features and (N,) labels
            of equal length.
    """
```

- [ ] **Step 3: 왕복 확인 테스트를 작성**

`tests/test_dataset_contract.py`를 새로 만든다. 이 태스크는 검증 코드를 줄이므로 남은 검증 한 건에는 러너블 체크가 필요하다.

```python
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

from src.data.dataset import GenotypeDataset


def _write(path: Path, x, y) -> None:
    with path.open("wb") as handle:
        pickle.dump((x, y), handle)


def test_getitem_returns_channels_first(tmp_path: Path) -> None:
    # Given a (N, gene_size, K) pickle as preprocessing writes it.
    path = tmp_path / "train_data.pkl"
    x = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    _write(path, x, np.array([0, 1]))

    sample, label = GenotypeDataset(path)[1]

    # Then the model sees (K, gene_size) and the label is a scalar.
    assert tuple(sample.shape) == (4, 3)
    np.testing.assert_allclose(sample.numpy(), x[1].T)
    assert int(label) == 1


def test_mismatched_lengths_are_rejected(tmp_path: Path) -> None:
    # Given one more feature row than labels.
    path = tmp_path / "train_data.pkl"
    _write(path, np.zeros((3, 2, 2), dtype=np.float32), np.array([0, 1]))

    with pytest.raises(ValueError, match="labels"):
        GenotypeDataset(path)
```

- [ ] **Step 4: 새 테스트 실행**

```bash
.venv/bin/python -m pytest tests/test_dataset_contract.py -v
```
Expected: 2 passed

- [ ] **Step 5: 전체 테스트**

```bash
.venv/bin/python -m pytest tests -q 2>&1 | tail -2
```
Expected: `269 passed, 2 warnings` (새 테스트 2건 추가)

- [ ] **Step 6: 커밋**

```bash
git add src/data/dataset.py tests/test_dataset_contract.py
git commit -m "refactor(dataset): validate the shape contract once, with a test"
```

---

## Task 6 — 샘플러 가중치를 벡터 연산으로 바꾸고 dataloader 예외를 정리한다

**근거:** `src/data/sampler.py:60-70`은 `dict(zip(...))` 사전을 만들고 파이썬 루프로 샘플마다 조회한다. `np.unique(return_inverse=True, return_counts=True)` 한 줄이 같은 값을 준다. `src/data/dataloader.py:18-19`의 `PreprocessingProvenanceError`는 이 파일 밖에서 잡히지 않는 단일 사용 예외이고, `allow_unverified_preprocessing`(`:42`)을 켜는 config는 저장소에 없다.

**Files:**
- Modify: `src/data/sampler.py:60-70`
- Modify: `src/data/dataloader.py:18-19`, `:42`, `:50-60`

**Interfaces:**
- Consumes: Task 5의 `GenotypeDataset`
- Produces: 계약 불변. `PopulationBalancedSampler(labels, num_samples_per_epoch, rank, world_size, seed)`와 `create_dataloaders(config, rank, world_size) -> (DataLoader, DataLoader)`는 그대로다. provenance 위반은 `PreprocessingProvenanceError` 대신 `ValueError`를 던진다.

- [ ] **Step 1: 가중치 계산 교체**

`src/data/sampler.py`의 `# Compute per-sample weights` 블록을 아래로 바꾼다.

```python
        # Per-sample weight 1/sqrt(pop_count), normalized to sum to one.
        _, inverse, counts = np.unique(self.labels, return_inverse=True, return_counts=True)
        weights = 1.0 / np.sqrt(counts[inverse])
        self.weights = weights / weights.sum()
```

`import math`는 `__len__`의 `math.ceil`이 여전히 쓰므로 남긴다.

- [ ] **Step 2: 동등성을 직접 확인**

```bash
.venv/bin/python - <<'PY'
import math, numpy as np
labels = np.array([0,0,0,1,1,2])
old = np.zeros(len(labels))
m = dict(zip(*np.unique(labels, return_counts=True)))
for i, l in enumerate(labels):
    old[i] = 1.0 / math.sqrt(m[l])
old /= old.sum()
_, inv, cnt = np.unique(labels, return_inverse=True, return_counts=True)
new = 1.0 / np.sqrt(cnt[inv]); new /= new.sum()
np.testing.assert_allclose(old, new)
print("identical")
PY
```
Expected: `identical`

- [ ] **Step 3: dataloader의 커스텀 예외 제거**

`class PreprocessingProvenanceError(ValueError): pass` 두 줄을 지우고, `raise PreprocessingProvenanceError(` 두 곳을 `raise ValueError(`로 바꾼다. 그리고 provenance 검사 조건을 아래로 바꾼다.

```python
    expected_method = data_cfg.get("dim_reduction_method")
    if expected_method:
```

- [ ] **Step 4: 테스트**

```bash
grep -rn "PreprocessingProvenanceError\|allow_unverified_preprocessing" src scripts tests
.venv/bin/python -m pytest tests -q 2>&1 | tail -2
```
Expected: grep은 출력 없음, 테스트는 `269 passed, 2 warnings`

- [ ] **Step 5: 커밋**

```bash
git add src/data/sampler.py src/data/dataloader.py
git commit -m "refactor(data): vectorize the sampler weights, drop a one-use exception"
```

---

## Task 7 — DDPM 샘플러를 제거한다

**근거:** 생성은 DDIM만 쓴다 (`src/inference/generator.py:357`). `sample_ddpm`과 `p_sample`은 `src/models/diffusion.py:168-200`에 있고 프로덕션 호출자가 없다. 남은 참조는 테스트뿐이다.

**이 태스크는 테스트를 삭제한다.** `p_sample`/`sample_ddpm`을 검증하던 것들이 검증 대상과 함께 사라진다. 지우는 것:
- `tests/test_model_diffusion.py::test_ddpm_sampling_honors_model_space_clip`
- `tests/test_model_diffusion.py::test_ddpm_keeps_masked_coordinates_zero_at_every_reverse_call`
- `tests/test_guidance_variants.py::test_samplers_put_a_train_mode_guide_into_eval`의 `"sample_ddpm"` 파라미터
- `tests/test_guidance_variants.py::test_samplers_forward_every_guidance_option_unchanged`의 `"p_sample"`과 `"sample_ddpm"` 파라미터

DDIM 쪽 동등 테스트(`test_ddim_sampling_honors_model_space_clip`, `test_ddim_keeps_masked_coordinates_zero_at_every_reverse_call`)는 남으므로 clip과 zero-mask 불변식의 커버리지는 유지된다.

**Files:**
- Modify: `src/models/diffusion.py:168-200`
- Modify: `tests/test_model_diffusion.py:38-46`, `:63-78`
- Modify: `tests/test_guidance_variants.py:151`, `:174`, `:185-188`

**Interfaces:**
- Consumes: 없음
- Produces: `GaussianDiffusion`에서 `p_sample`과 `sample_ddpm`이 사라진다. `sample_ddim`, `q_sample`, `p_losses`, `_predict_noise`, `apply_cfg_dropout`은 그대로다.

- [ ] **Step 1: 두 메서드 삭제**

`src/models/diffusion.py`에서 `@torch.no_grad()` + `def p_sample(...)` 전체와 `@torch.no_grad()` + `def sample_ddpm(...)` 전체를 지운다. `@torch.no_grad() def sample_ddim`이 `apply_cfg_dropout`/`_predict_noise` 뒤에 바로 오게 된다.

- [ ] **Step 2: `tests/test_model_diffusion.py`에서 DDPM 테스트 2건 삭제**

`test_ddpm_sampling_honors_model_space_clip`과 `test_ddpm_keeps_masked_coordinates_zero_at_every_reverse_call` 함수를 지운다.

- [ ] **Step 3: `tests/test_guidance_variants.py`의 파라미터 축소**

`test_samplers_put_a_train_mode_guide_into_eval`의 데코레이터와 본문을 아래로 바꾼다.

```python
def test_sampler_puts_a_train_mode_guide_into_eval():
    # Given a guide left in train mode, where its dropout randomizes every prediction.
    diffusion, model, guide = make_diffusion(), StubModel(), DropoutGuide()
    y = torch.tensor([0, 5])

    def run() -> torch.Tensor:
        torch.manual_seed(20260327)
        return diffusion.sample_ddim(
            model, (2, 1, 4), y, torch.device("cpu"), ddim_steps=3,
            guidance_scale=2.0, guide_model=guide)

    guide.train()
    from_train_mode = run()

    # Then the sampler switched it off and the draw matches the same guide in eval.
    assert not guide.training
    assert torch.allclose(from_train_mode, run())
```

`test_samplers_forward_every_guidance_option_unchanged`를 아래로 바꾼다.

```python
def test_sampler_forwards_every_guidance_option_unchanged():
    # Given the sampler entry point spied on at the guidance boundary.
    diffusion, model, guide = make_diffusion(), StubModel(), StubModel(weight=0.5)
    y = torch.tensor([0, 5])
    options = {"guidance_interval": (0.0, 0.5), "guidance_alpha": 0.3, "guide_model": guide}
    captured: list[dict] = []
    real = diffusion._predict_noise
    diffusion._predict_noise = lambda *a, **k: captured.append(k) or real(*a, **k)

    # When the sampler runs with every variant enabled.
    diffusion.sample_ddim(model, (2, 1, 4), y, torch.device("cpu"), ddim_steps=2,
                          guidance_scale=1.0, **options)

    # Then all three arrive at _predict_noise untouched on every step.
    assert captured and all(step == options for step in captured)
```

두 함수 위의 `@pytest.mark.parametrize("sampler", ...)` 데코레이터를 지운다.

- [ ] **Step 4: 잔존 참조 확인**

```bash
grep -rn "sample_ddpm\|p_sample\b" src scripts tests
```
Expected: `p_losses`만 나오고 `p_sample`/`sample_ddpm` 단독 참조는 없어야 한다. (`grep -w`가 아니므로 `p_losses`는 매치되지 않는다. 출력이 비어야 정상.)

- [ ] **Step 5: 테스트**

```bash
.venv/bin/python -m pytest tests -q 2>&1 | tail -2
```
Expected: `266 passed, 2 warnings` (269 − DDPM 2건 − 파라미터화 축소 순감 1건)

숫자가 다르면 멈추고 실제 수집 결과를 보고한다. 파라미터화 삭제가 몇 건을 줄이는지가 계산과 다를 수 있다.

- [ ] **Step 6: 커밋**

```bash
git add src/models/diffusion.py tests/test_model_diffusion.py tests/test_guidance_variants.py
git commit -m "refactor(diffusion): drop the DDPM sampler, generation uses DDIM only"
```

---

## Task 8 — `prediction_target` 매개변수를 제거한다

**근거:** `src/models/diffusion.py:32-33`이 `"epsilon"` 외의 값을 거부한다. 즉 이 매개변수는 값 하나만 받는 설정이다. epsilon이 아닌 경로는 구현되지 않았다. 배관은 4곳에 퍼져 있다: diffusion 생성자, `src/training/trainer.py:337`, `src/inference/generator.py:112`, guide 검사(`:324`), 그리고 `configs/default.yaml:38`.

**이 태스크는 테스트를 삭제한다.** 없어진 거부 동작을 검증하던 것들이다.
- `tests/test_model_diffusion.py::test_diffusion_rejects_unimplemented_prediction_target`
- `tests/test_inference_generator_contract.py::test_diffusion_rejects_checkpoint_with_unsupported_prediction_target`
- `tests/test_inference_generator_contract.py::test_guide_trained_on_another_diffusion_setting_is_rejected`의 `prediction_target` 파라미터 (guide 스키마 검사에서 `noise_schedule` 파라미터는 남는다)

**Files:**
- Modify: `src/models/diffusion.py:16-33`
- Modify: `src/training/trainer.py:337`
- Modify: `src/inference/generator.py:105-114`, `:322-330`, `:416`
- Modify: `configs/default.yaml:38`
- Modify: `tests/test_model_diffusion.py:27-29`, `tests/test_inference_generator_contract.py:16-34`, `:223-225`

**Interfaces:**
- Consumes: Task 7 이후의 `GaussianDiffusion`
- Produces: `GaussianDiffusion(timesteps, zero_mask, enforce_zeros, min_snr_gamma, null_class, cfg_dropout_rate, schedule_type, sample_clip, feature_schedule)` — `prediction_target`이 없는 생성자. `scripts/hipodit_rebuild_train.py:93-97`이 키워드로 넘기므로 그 호출부도 함께 고친다.

- [ ] **Step 1: diffusion 생성자에서 제거**

`src/models/diffusion.py`의 `prediction_target: str = "epsilon",` 매개변수 줄과 아래 두 줄을 지운다.

```python
        if prediction_target != "epsilon":
            raise ValueError(f"Unsupported prediction_target: {prediction_target}; expected epsilon")
```

- [ ] **Step 2: 세 호출부에서 제거**

`src/training/trainer.py`에서 아래 줄을 지운다.

```python
        prediction_target=diffusion_cfg.get("prediction_target", "epsilon"),
```

`src/inference/generator.py`의 `build_generation_diffusion` 안에서 같은 줄을 지운다. guide 검사 루프를 아래로 바꾼다.

```python
        # A guide trained on another schedule predicts epsilon on another scale.
        guide_diffusion_cfg = guide_checkpoint.get("config", {}).get("diffusion", {})
        guide_value = guide_diffusion_cfg.get("noise_schedule", "cosine")
        if guide_value != diffusion_cfg.get("noise_schedule", "cosine"):
            raise ValueError(
                f"Guide model noise_schedule ({guide_value}) does not match the main "
                f"checkpoint ({diffusion_cfg.get('noise_schedule', 'cosine')})"
            )
```

`generation_meta.json`의 config 블록에서 아래 줄을 지운다.

```python
            "prediction_target": diffusion_cfg.get("prediction_target", "epsilon"),
```

`scripts/hipodit_rebuild_train.py`에서 두 곳을 고친다. `config["diffusion"]` 사전의 `"prediction_target": "epsilon",` 항목을 지우고, `GaussianDiffusion(...)` 호출의 `prediction_target="epsilon",` 인자를 지운다.

- [ ] **Step 3: config에서 키 제거**

`configs/default.yaml`에서 아래 줄을 지운다.

```yaml
  prediction_target: "epsilon"
```

- [ ] **Step 4: 세 테스트 정리**

`tests/test_model_diffusion.py`에서 `test_diffusion_rejects_unimplemented_prediction_target`을 지운다.

`tests/test_inference_generator_contract.py`에서 `test_diffusion_rejects_checkpoint_with_unsupported_prediction_target`을 지우고, `test_diffusion_uses_checkpoint_prediction_contract`의 diffusion 사전에서 `"prediction_target": "epsilon",` 항목을 지운다. guide 거부 테스트의 파라미터를 아래로 바꾼다.

```python
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
```

이 함수 위의 `@pytest.mark.parametrize("key, value", ...)` 데코레이터를 지운다.

- [ ] **Step 5: 잔존 참조 확인**

```bash
grep -rn "prediction_target" src scripts tests configs
```
Expected: 출력 없음

- [ ] **Step 6: 테스트**

```bash
.venv/bin/python -m pytest tests -q 2>&1 | tail -2
```
Expected: `262 passed, 2 warnings` (266 − 2건 삭제 − 파라미터 2건이 1건으로)

숫자가 다르면 실제 값을 보고하고 멈춘다.

- [ ] **Step 7: 커밋**

```bash
git add src/models/diffusion.py src/training/trainer.py src/inference/generator.py \
        scripts/hipodit_rebuild_train.py configs/default.yaml \
        tests/test_model_diffusion.py tests/test_inference_generator_contract.py
git commit -m "refactor(diffusion): drop prediction_target, epsilon is the only implementation"
```

---

## Task 9 — generator의 단일 호출 헬퍼를 인라인한다

**근거:** `build_generation_diffusion`(`src/inference/generator.py:100-114`)과 `guidance_meta`(`:140-150`)는 프로덕션 호출자가 각각 1곳이다. `denormalize_samples`(`:58-84`)도 `postprocess_samples`(`:87-97`) 한 곳에서만 불린다.

**주의:** 이 세 이름은 `tests/test_inference_generator_contract.py`가 직접 임포트한다. `_run_generation` 헬퍼가 `build_generation_diffusion`을 monkeypatch하므로, 그 이름이 사라지면 그 테스트 전체가 무효가 된다. **그래서 이 태스크는 `build_generation_diffusion`을 남긴다** — monkeypatch 지점으로서 실제 값이 있다. `guidance_meta`와 `denormalize_samples`만 인라인하고, 해당 테스트를 `postprocess_samples`와 CLI 파싱으로 다시 겨눈다.

**Files:**
- Modify: `src/inference/generator.py:58-97`, `:140-150`, `:426`
- Modify: `tests/test_inference_generator_contract.py:74-83`, `:92-127`

**Interfaces:**
- Consumes: Task 8 이후의 generator
- Produces: `postprocess_samples(samples, zero_mask, stats_path, labels=None) -> torch.Tensor` — 역정규화를 직접 수행한다. `build_generation_diffusion`, `load_generator_model`, `resolve_normalization_stats_path`, `parse_args`는 그대로 남는다. `denormalize_samples`와 `guidance_meta`는 사라진다.

- [ ] **Step 1: 역정규화를 `postprocess_samples`로 합친다**

`denormalize_samples`와 `postprocess_samples` 두 함수를 아래 하나로 교체한다.

```python
def postprocess_samples(
    samples: torch.Tensor,
    zero_mask: torch.Tensor | None,
    stats_path: str | Path | None,
    labels: np.ndarray | None = None,
) -> torch.Tensor:
    """Mask padding, then restore the original feature scale.

    Masking precedes the inverse so a constant per-population mean survives on
    always-zero coordinates. ``labels`` is required when the statistics carry a
    per-population conditional prior; ``invert_normalization`` owns that choice
    (PriorGrad / ShiftDDPMs de-normalization).
    """
    if zero_mask is not None:
        samples = samples * (~zero_mask.cpu()).unsqueeze(0).to(samples.dtype)
    if stats_path is None:
        return samples
    stats = load_normalization_stats(
        str(stats_path), expected_shape=(samples.shape[2], samples.shape[1])
    )
    restored = invert_normalization(
        samples.detach().float().cpu().permute(0, 2, 1).numpy(), stats, labels
    )
    return torch.from_numpy(restored).permute(0, 2, 1).contiguous()
```

- [ ] **Step 2: `guidance_meta`를 메타 사전에 인라인한다**

`guidance_meta` 함수 정의를 지우고, `generation_meta.json`의 config 블록 안 `**guidance_meta(...)` 줄을 아래 세 줄로 바꾼다.

```python
            "guidance_interval": list(guidance_interval) if guidance_interval else None,
            "guidance_alpha": float(guidance_alpha),
            "guide_model_path": str(guide_model_path) if guide_model_path else None,
```

- [ ] **Step 3: 두 테스트를 새 표면에 맞춘다**

`tests/test_inference_generator_contract.py`에서 `test_denormalization_rejects_nonpositive_or_nonfinite_stats`를 아래로 바꾼다.

```python
def test_denormalization_rejects_nonpositive_or_nonfinite_stats(tmp_path: Path) -> None:
    stats_path = tmp_path / "stats.pkl"
    with stats_path.open("wb") as handle:
        pickle.dump({
            "mean": np.zeros((2, 1), dtype=np.float32),
            "std": np.array([[1.0], [float("nan")]], dtype=np.float32),
        }, handle)

    with pytest.raises(ValueError, match="statistics"):
        generator.postprocess_samples(torch.zeros((1, 1, 2)), None, stats_path)
```

`test_guidance_variants_default_to_the_unguided_sampler_settings`와 `test_generation_meta_records_the_requested_guidance_variant`는 `guidance_meta`를 직접 부른다. 앞의 것은 CLI 기본값 검사로 바꾼다.

```python
def test_guidance_variants_default_to_the_unguided_sampler_settings() -> None:
    args = _parse()

    assert args.guidance_interval is None
    assert args.guidance_alpha == 0.0
    assert args.guide_model_path is None
```

뒤의 것은 지운다. 같은 사실을 `test_generation_threads_the_guidance_variant_into_the_sampler_and_the_meta`가 이미 끝에서 끝까지 검증한다 (`meta["config"]["guidance_interval"]` 등).

- [ ] **Step 4: 잔존 참조 확인**

```bash
grep -rn "guidance_meta\|denormalize_samples" src scripts tests
```
Expected: 출력 없음

- [ ] **Step 5: 테스트**

```bash
.venv/bin/python -m pytest tests/test_inference_generator_contract.py -q 2>&1 | tail -2
.venv/bin/python -m pytest tests -q 2>&1 | tail -2
```
Expected: 전체 `261 passed, 2 warnings` (262 − 1건 삭제)

- [ ] **Step 6: 커밋**

```bash
git add src/inference/generator.py tests/test_inference_generator_contract.py
git commit -m "refactor(inference): inline the single-caller denormalization and meta helpers"
```

---

## Task 10 — 평가 캐시 계층과 legacy sample-space 배관을 제거한다

**근거:** `src/evaluation/_io.py`는 캐시 두 층(NPZ 배열 캐시 `:181-225`, PCA 좌표 캐시 `:258-323`)과 그것을 무효화하기 위한 fingerprint 체계(`:46-98`)를 갖고 있다. 캐시가 아끼는 작업은 `.pt` 파일 로딩과 PCA(2) 적합인데, 후자는 초 단위다. `--array-cache-mode`, `--pca-cache-mode`, `--cache-dir` 세 플래그와 `_load_or_compute_pcs`의 9-튜플 반환이 여기서 나온다.

legacy sample-space 쪽은 별개다. 현재 generator는 `generation_meta.json`에 `sample_space`를 항상 기록한다(`src/inference/generator.py:397`). `--legacy-synthetic-space`는 그 키가 없던 옛 샘플용이다.

**판단:** legacy 플래그는 **남긴다.** `outputs/` 아래에 2026-04월 run들이 있고, 그것을 다시 평가할 가능성을 없앨 근거가 이 감사에는 없다. 삭제 후보에 올랐던 이유는 "현재 generator가 항상 기록한다"였지만, 그것은 옛 산출물이 사라진다는 뜻이 아니다. 캐시만 제거한다.

`--real-space`도 남긴다. 실제로 두 값이 다 의미가 있다(전처리 산출물은 normalized, chr17 실험 산출물은 original).

**Files:**
- Modify: `src/evaluation/_io.py:26-42`, `:46-98`, `:181-225`, `:257-356`
- Modify: `scripts/evaluate_synthetic_metrics.py:33-46`, `:73-85`, `:93-153`, `:156-191`, `:208-216`
- Modify: `tests/test_evaluation_io_contract.py:78-117`

**Interfaces:**
- Consumes: Task 9 이후의 코드베이스
- Produces: `src/evaluation/_io.py`의 공개 이름은 `load_real`, `load_synthetic`, `load_label_hierarchy`, `pop_to_superpop`, `flatten_subsample_genes`, `write_csv`로 줄어든다. 사라지는 것: `file_stat`, `file_sha256`, `synthetic_fingerprint`, `input_fingerprint`, `load_synthetic_cached`, `pca_cache_matches`, `write_pca_cache_meta`, `load_pca_cache_meta`, `load_pca_coordinates`, `write_pca_coordinates`. **단 `file_sha256`은 `load_synthetic`의 fingerprint 검증(`:146`)이 쓰므로 모듈 내부 헬퍼로 남는다.** `scripts/guidance_sweep.py:15`와 `src/inference/decode.py:25`는 각각 `write_csv`와 `load_synthetic`만 임포트하므로 영향이 없다.

- [ ] **Step 1: `_io.py`에서 캐시·fingerprint 함수 삭제**

아래 함수 정의를 모두 지운다: `file_stat`, `synthetic_fingerprint`, `input_fingerprint`, `load_synthetic_cached`, `pca_cache_matches`, `write_pca_cache_meta`, `load_pca_cache_meta`, `load_pca_coordinates`, `write_pca_coordinates`. `file_sha256`은 남긴다. `# ── PCA-coordinate caching ──` 구분선 주석도 지운다.

`__all__`을 아래로 바꾼다.

```python
__all__ = [
    "flatten_subsample_genes",
    "load_label_hierarchy",
    "load_real",
    "load_synthetic",
    "pop_to_superpop",
    "write_csv",
]
```

`import csv`는 `write_csv`가 쓰므로 남기고, `from typing import Any, Literal`은 `Any`를 쓰는 함수가 다 사라지므로 `from typing import Literal`로 줄인다.

- [ ] **Step 2: 평가 CLI에서 캐시 플래그와 분기 제거**

`scripts/evaluate_synthetic_metrics.py`의 import 블록을 아래로 바꾼다.

```python
from src.evaluation import evaluate  # noqa: E402
from src.evaluation._io import (  # noqa: E402
    flatten_subsample_genes,
    load_label_hierarchy,
    load_real,
    load_synthetic,
    pop_to_superpop,
    write_csv,
)
```

파서에서 `--cache-dir`, `--array-cache-mode`, `--pca-cache-mode` 세 인자를 지운다.

`_load_or_compute_pcs` 함수 전체를 지우고, `main`을 아래로 바꾼다.

```python
def main() -> None:
    args = _build_parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    hierarchy = load_label_hierarchy(args.hierarchy)
    real_x, real_pop = load_real(
        args.real_path, stats_path=args.stats_path, sample_space=args.real_space,
    )
    syn_x, syn_pop, syn_names = load_synthetic(
        args.syn_dir,
        stats_path=args.stats_path,
        legacy_sample_space=args.legacy_synthetic_space,
    )
    real_flat, gene_indices = flatten_subsample_genes(real_x, args.n_genes, args.seed)
    syn_flat, _ = flatten_subsample_genes(syn_x, args.n_genes, args.seed, gene_indices)

    pca = PCA(n_components=2, random_state=args.seed)
    real_pcs = pca.fit_transform(real_flat)
    syn_pcs = pca.transform(syn_flat)
    real_sp = pop_to_superpop(real_pop, hierarchy)
    syn_sp = pop_to_superpop(syn_pop, hierarchy)
    np.save(args.out_dir / "pca_gene_indices.npy", gene_indices)

    report = evaluate(
        real_pcs=real_pcs, syn_pcs=syn_pcs, real_sp=real_sp, syn_sp=syn_sp,
        k=args.dupi_k, tau=args.tau,
    )

    summary = {
        "inputs": {
            "real_path": str(args.real_path),
            "syn_dir": str(args.syn_dir),
            "hierarchy": str(args.hierarchy),
            "stats_path": str(args.stats_path),
            "real_space": args.real_space,
            "legacy_synthetic_space": args.legacy_synthetic_space,
            "evaluation_space": "original",
            "n_genes": args.n_genes,
            "seed": args.seed,
            "dupi_k": args.dupi_k,
            "tau": args.tau,
            "metric_space": "PCA(2) fitted on real flattened subsampled genes",
        },
        "counts": {
            "n_real": int(len(real_pop)),
            "n_synthetic": int(len(syn_pop)),
            "n_features_before_pca": int(real_flat.shape[1]),
        },
        "pca": {
            "explained_variance_ratio": [float(v) for v in pca.explained_variance_ratio_],
            "components_shape": list(pca.components_.shape),
        },
        "dupi": report.dupi,
        "distribution_distances": report.distribution_distances,
        "notes": {
            "dupi_source": "Jeong, Kim, and Im (2023), Eqs. (10)-(13).",
            "dupi_interpretation": (
                "DUPI near 1 means synthetic samples are too close to real samples; "
                "DUPI near 0 means utility loss; values near the benchmark indicate balance."
            ),
        },
    }

    summary_path = args.out_dir / "summary_metrics.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(args.out_dir / "centroids.csv", report.centroid_rows)
    write_csv(args.out_dir / "class_metrics.csv", report.class_metric_rows)

    print(f"Saved summary: {summary_path}")
    print(f"Saved class metrics: {args.out_dir / 'class_metrics.csv'}")
```

`pca_coordinates.csv` 출력은 사라진다. 그것을 읽던 유일한 코드는 방금 지운 PCA 캐시 경로였다. `scripts/plot_pca.py`는 자기 PCA를 따로 적합하므로 영향이 없다.

`import os`와 `import sys`는 `sys.path` 삽입이 여전히 필요하므로 남긴다.

- [ ] **Step 3: fingerprint 테스트를 로더 계약 테스트로 대체**

`tests/test_evaluation_io_contract.py`에서 `test_input_fingerprint_changes_with_stats_and_generation_metadata`를 지우고 그 자리에 아래를 넣는다.

```python
def test_synthetic_loader_accepts_both_stored_orientations(tmp_path: Path) -> None:
    # Given stats whose (G, K) shape is not square, so orientation is decidable.
    stats_path = tmp_path / "stats.pkl"
    _write_stats(stats_path, np.zeros((3, 2), dtype=np.float32), np.ones((3, 2), dtype=np.float32))
    syn_dir = tmp_path / "synthetic"
    syn_dir.mkdir()
    value = np.arange(6, dtype=np.float32).reshape(3, 2)
    # One file stored as (K, gene_size), the generator's layout.
    torch.save((torch.from_numpy(value.T.copy()), torch.tensor(0)),
               syn_dir / "sample_pop0_0000.pt")
    (syn_dir / "generation_meta.json").write_text(json.dumps({"sample_space": "original"}))

    loaded, labels, names = _io.load_synthetic(syn_dir, stats_path=stats_path)

    assert loaded.shape == (1, 3, 2)
    np.testing.assert_allclose(loaded[0], value)
    assert labels.tolist() == [0] and names == ["sample_pop0_0000.pt"]
```

`test_evaluation_cli_help_and_legacy_error_are_user_facing`은 그대로 둔다. legacy 플래그를 남기기로 했으므로 유효하다.

- [ ] **Step 4: 잔존 참조 확인**

```bash
grep -rn "load_synthetic_cached\|input_fingerprint\|pca_cache\|write_pca_coordinates\|load_pca_coordinates\|synthetic_fingerprint\|file_stat\|array-cache\|pca-cache" src scripts tests
```
Expected: 출력 없음

- [ ] **Step 5: CLI가 실제로 도는지 확인**

```bash
.venv/bin/python scripts/evaluate_synthetic_metrics.py --help | head -20
```
Expected: 오류 없이 사용법이 출력되고, `--array-cache-mode`가 보이지 않아야 한다.

- [ ] **Step 6: 테스트**

```bash
.venv/bin/python -m pytest tests -q 2>&1 | tail -2
```
Expected: `261 passed, 2 warnings` (1건 삭제, 1건 추가)

- [ ] **Step 7: 커밋**

```bash
git add src/evaluation/_io.py scripts/evaluate_synthetic_metrics.py tests/test_evaluation_io_contract.py
git commit -m "refactor(evaluation): drop the two cache layers and their fingerprint machinery"
```

---

## Task 11 — glm_pca의 단일 사용 예외·import 래퍼·가짜 통계를 정리한다

**근거:** 세 항목이다.

1. `UnsupportedProjectionFamilyError`(`src/preprocessing/glm_pca.py:32-33`)는 이 파일 밖에서 잡히지 않는다. 테스트는 메시지로 매치한다(`tests/test_binomial_glm_pca.py:82` → `decode_gene`의 `ValueError`).
2. `_try_import_rust`/`_try_import_binomial_rust`(`:128-165`)는 각자 한 번만 불리는 try/except 래퍼다.
3. `explained_per_component`(`:437`)는 `[explained / n_comp] * n_comp`다. 성분별로 같은 값을 K번 되풀이한 것이고 성분별 설명력이 아니다. `src/preprocessing/pca.py:177-178`이 그것을 `explained_pc1..K` CSV 열로 펼친다. `tests/test_glm_pca_preprocessing.py:34-48`이 키 존재를 검사한다.

**판단:** 1과 2는 제거한다. 3은 **제거하되 CSV 열 이름을 없앤다.** 같은 수를 K번 적은 열은 오해를 부르고, 실제 설명력은 `explained_total`에 있다. 다만 `pca_per_gene_stats.csv`는 `data/processed/` 산출물이고 기존 파일이 있다. 새 컬럼 구성으로 덮어쓰는 것은 다음 전처리 실행 때만 일어나며, 이 리팩터는 전처리를 실행하지 않는다. 기존 CSV를 읽는 코드는 저장소에 없다(grep으로 확인할 것).

**Files:**
- Modify: `src/preprocessing/glm_pca.py:32-33`, `:128-165`, `:257-260`, `:437`
- Modify: `src/preprocessing/pca.py:174-179`
- Modify: `tests/test_glm_pca_preprocessing.py:34-48`

**Interfaces:**
- Consumes: 없음
- Produces: `glm_pca_single_gene(...)`의 반환 사전에서 `explained_per_component` 키가 사라진다. 나머지 키(`features`, `explained_total`, `n_variants`, `actual_k`, `loadings`, `intercept`, `family`, `link`, `penalty`, `backend`, `backend_version`, `projection`)는 유지된다. `UnsupportedProjectionFamilyError`가 사라지고 `ValueError`가 그 자리를 대신한다.

- [ ] **Step 1: CSV 소비자가 없음을 확인**

```bash
grep -rn "pca_per_gene_stats\|explained_pc" src scripts tests notebooks docs
```
Expected: `src/preprocessing/pca.py`의 쓰기 지점만 나온다. 읽는 코드가 나오면 멈추고 보고한다.

- [ ] **Step 2: 단일 사용 예외 제거**

`class UnsupportedProjectionFamilyError(ValueError): pass` 정의를 지우고, 두 `raise UnsupportedProjectionFamilyError(`를 `raise ValueError(`로 바꾼다.

- [ ] **Step 3: import 래퍼를 직접 try/except로**

`_try_import_rust`, `_RUST_BACKEND = ...`, `_try_import_binomial_rust`, `_BINOM_BACKEND = ...` 네 블록을 아래로 교체한다.

```python
# Poisson backend, published on PyPI as ``glmpca-fast`` and installed by ``uv sync``.
try:
    import glmpca_fast as _RUST_BACKEND
except ImportError:
    _RUST_BACKEND = None
    logger.warning("glmpca-fast not importable; Poisson GLM-PCA is unavailable")

# Binomial(2, p) backend, built from ``binom_glmpca_rs/``. Absent, the scipy
# reference in src.preprocessing.binomial_glm_pca still runs, just slower.
try:
    import binom_glmpca_rs as _BINOM_BACKEND
except ImportError:
    _BINOM_BACKEND = None
    logger.warning(
        "binom_glmpca_rs not importable; falling back to the slower scipy "
        "Binomial GLM-PCA reference"
    )
```

- [ ] **Step 4: 가짜 성분별 통계 제거**

`_build_result`의 반환 사전에서 아래 줄을 지운다.

```python
        "explained_per_component": [explained / n_comp] * n_comp,
```

`src/preprocessing/pca.py`에서 `stat_row`에 성분별 열을 붙이는 두 줄을 지운다.

```python
                    for j, v in enumerate(result["explained_per_component"]):
                        stat_row[f"explained_pc{j + 1}"] = v
```

- [ ] **Step 5: 스키마 테스트 갱신**

`tests/test_glm_pca_preprocessing.py`의 키 집합에서 `"explained_per_component",` 줄을 지운다.

- [ ] **Step 6: 잔존 참조 확인 + 테스트**

```bash
grep -rn "UnsupportedProjectionFamilyError\|explained_per_component\|_try_import" src scripts tests
.venv/bin/python -m pytest tests -q 2>&1 | tail -2
```
Expected: grep 출력 없음, 테스트 `261 passed, 2 warnings`

- [ ] **Step 7: 커밋**

```bash
git add src/preprocessing/glm_pca.py src/preprocessing/pca.py tests/test_glm_pca_preprocessing.py
git commit -m "refactor(glm-pca): drop a one-use exception, import wrappers and a fake per-component statistic"
```

---

## Task 12 — labels.py의 미도달 재분할 경로를 제거한다

**근거:** `split_dataset_stratified`(`src/preprocessing/labels.py:93-168`)는 `precomputed_indices`가 `None`일 때 스스로 재분할하는 경로를 갖는다. 유일한 호출자 `src/preprocessing/run_pipeline.py:159-163`은 항상 세 인덱스를 넘긴다. 그러므로 `val_ratio`, `test_ratio`, `seed` 세 매개변수와 `compute_split_indices` 재호출은 도달하지 않는다. manifest에 기록되는 `seed`/`val_ratio`/`test_ratio`는 유지해야 하므로 매개변수 자체는 남기고 fallback만 제거한다.

`save_all`(`:182-213`)은 세 텐서를 다시 패딩한다. `run_pipeline.py:170-172`가 이미 `pad_to_gene_size`를 적용한 뒤 넘긴다. `pad_to_gene_size`의 잘라내기 분기(`:174-175`)는 `compute_gene_size`가 항상 `n_genes` 이상을 반환하므로(`src/preprocessing/tokenizer.py:249`) 도달하지 않는다.

**Files:**
- Modify: `src/preprocessing/labels.py:100`, `:117-122`, `:171-179`, `:192-197`
- Modify: `src/preprocessing/run_pipeline.py:159-163`

**Interfaces:**
- Consumes: 없음
- Produces: `split_dataset_stratified(tokenized, labels, sample_ids, indices, val_ratio, test_ratio, seed)` — `precomputed_indices`가 `indices`라는 필수 위치 인자가 된다. `save_all(x_train, x_val, x_test, y_train, y_val, y_test, features_df)` — `gene_size` 매개변수가 사라진다. `pad_to_gene_size(data, gene_size)`는 시그니처 그대로, 잘라내기 분기만 없어진다.

- [ ] **Step 1: `split_dataset_stratified`의 fallback 제거**

시그니처를 아래로 바꾼다.

```python
def split_dataset_stratified(
    tokenized: np.ndarray,
    labels: dict,
    sample_ids: list[str],
    indices: tuple[np.ndarray, np.ndarray, np.ndarray],
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = PREPROCESS_SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Split the tokenized tensor on indices the per-gene PCA was already fit with.

    ``indices`` must be the ``(train, val, test)`` triple from
    :func:`compute_split_indices`, so tokenized rows stay consistent with the
    PCA train/val/test partition. ``val_ratio``, ``test_ratio`` and ``seed``
    are recorded in the split manifest.
    """
```

본문의 `if precomputed_indices is not None:` 분기를 아래 한 줄로 바꾼다.

```python
    train_idx, val_idx, test_idx = indices
```

- [ ] **Step 2: 잘라내기 분기와 재패딩 제거**

`pad_to_gene_size`를 아래로 바꾼다.

```python
def pad_to_gene_size(data: np.ndarray, gene_size: int) -> np.ndarray:
    """Pad (N, n_genes, K) to (N, gene_size, K) with zeros."""
    n_samples, n_genes, n_k = data.shape
    padded = np.zeros((n_samples, gene_size, n_k), dtype=data.dtype)
    padded[:, :n_genes, :] = data
    return padded
```

`save_all`을 아래로 바꾼다.

```python
def save_all(
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    features_df: pd.DataFrame,
) -> None:
    """Save all preprocessed artifacts to data/processed/.

    The tensors arrive already padded to the aligned gene size, because the
    normalization statistics had to be fit at that shape.
    """
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    for name, x, y in [
        ("train", x_train, y_train),
        ("val", x_val, y_val),
        ("test", x_test, y_test),
    ]:
        path = os.path.join(PROCESSED_DIR, f"{name}_data.pkl")
        with open(path, "wb") as f:
            pickle.dump((x, y), f, protocol=4)
        logger.info(f"{name.capitalize()} data: {path} {x.shape}")

    features_path = os.path.join(PROCESSED_DIR, "gene_pca_features.pkl")
    with open(features_path, "wb") as f:
        pickle.dump(features_df, f, protocol=4)
    logger.info(f"PCA features: {features_path} {features_df.shape}")
```

- [ ] **Step 3: 호출부 갱신**

`src/preprocessing/run_pipeline.py`의 split 호출을 아래로 바꾼다.

```python
    x_train, x_val, x_test, y_train, y_val, y_test, _ = split_dataset_stratified(
        tokenized, labels, sample_ids,
        indices=(train_idx, val_idx, test_idx),
        val_ratio=VAL_RATIO, test_ratio=TEST_RATIO, seed=PREPROCESS_SEED,
    )
```

`save_all` 호출을 아래로 바꾼다.

```python
    save_all(
        x_train_norm, x_val_norm, x_test_norm,
        y_train, y_val, y_test, features_df,
    )
```

- [ ] **Step 4: 파이프라인이 임포트되는지 확인 + 테스트**

```bash
.venv/bin/python -c "import src.preprocessing.run_pipeline as p; print(p.main.__name__)"
grep -rn "precomputed_indices" src scripts tests
.venv/bin/python -m pytest tests -q 2>&1 | tail -2
```
Expected: `main`, grep 출력 없음, 테스트 `261 passed, 2 warnings`

- [ ] **Step 5: 커밋**

```bash
git add src/preprocessing/labels.py src/preprocessing/run_pipeline.py
git commit -m "refactor(preprocessing): require the precomputed split, drop the unreachable resplit and repad"
```

---

## Task 13 — 모델 조립부의 이중 config 포맷 fallback을 제거한다

**근거:** `src/models/hybrid_geno_dit.py:44-50`은 두 가지 config 모양을 받는다: 중첩(`config["model"]`, `config["data"]`)과 평평한 것(`config` 자체가 model 섹션). 실제 호출자는 셋이고 모두 중첩을 넘긴다 — `src/training/trainer.py:312`, `src/inference/generator.py:126`, `scripts/hipodit_rebuild_train.py:92`. `in_channels` 3단 fallback(`:47-49`)과 `gene_size` 2단 fallback(`:50`)도 같은 이유로 첫 항만 쓰인다.

`output[..., :gene_size]` 트리밍(`:188-189`)은 `:77-80`의 divisibility assert가 이미 보장하는 조건에 대한 이중 방어다.

`UnifiedFiLMGenerator`의 `cnn_dec_channels is None` fallback(`src/models/modules/conditioning.py:122-123`)은 유일한 호출자(`hybrid_geno_dit.py:115`)가 항상 `self.decoder.block_out_channels`를 넘기므로 도달하지 않는다.

**주의:** `tests/test_model_modulation.py`가 모델을 만든다. 그 테스트가 넘기는 config 모양을 먼저 확인해야 한다. 평평한 모양을 쓰고 있으면 이 태스크는 **중단하고 보고한다** (테스트가 그 fallback의 유일한 사용자라는 뜻이고, 그러면 제거 대신 테스트를 중첩 형태로 고치는 판단이 필요하다).

**Files:**
- Modify: `src/models/hybrid_geno_dit.py:43-50`, `:186-191`
- Modify: `src/models/modules/conditioning.py:120-126`

**Interfaces:**
- Consumes: 없음
- Produces: `HybridCNNDiTFiLM(config)` — `config["data"]`와 `config["model"]`을 요구한다. `UnifiedFiLMGenerator(d_model, d_time, cnn_channels, cnn_dec_channels, n_dit_blocks, d_dit)` — `cnn_dec_channels`가 필수가 된다.

- [ ] **Step 1: 테스트가 쓰는 config 모양을 확인**

```bash
grep -n "HybridCNNDiTFiLM(\|\"model\"\|\"data\"" tests/test_model_modulation.py | head -20
```

`{"model": {...}, "data": {...}}` 형태면 계속한다. 평평한 형태면 멈추고 보고한다.

- [ ] **Step 2: config 추출을 명시적으로**

`src/models/hybrid_geno_dit.py`의 아래 블록을

```python
        # --- Extract and validate config ---
        model_cfg = config.get("model", config)
        data_cfg = config.get("data", {})

        self.in_channels = data_cfg.get(
            "num_channels", model_cfg.get("in_channels", 8)
        )
        self.gene_size = data_cfg.get("gene_size", model_cfg.get("gene_size", 26624))
```

아래로 바꾼다.

```python
        # --- Extract config ---
        model_cfg = config["model"]
        data_cfg = config["data"]

        self.in_channels = data_cfg["num_channels"]
        self.gene_size = data_cfg["gene_size"]
```

- [ ] **Step 3: 이중 방어 트리밍 제거**

`forward`의 마지막을 아래로 바꾼다.

```python
        # --- CNN Decoder with skip connections ---
        return self.decoder(features_out, skips, cnn_dec_params)
```

(`output = ...` 대입과 `if output.shape[-1] != self.gene_size:` 두 줄, `return output`을 이 한 줄이 대신한다. 인코더/디코더가 대칭이고 `gene_size`가 `2**n_downsamples`로 나누어진다는 것은 `__init__`의 assert가 보장한다.)

- [ ] **Step 4: FiLM 생성기의 fallback 제거**

`src/models/modules/conditioning.py`에서 아래 두 줄을 지운다.

```python
        if cnn_dec_channels is None:
            cnn_dec_channels = list(reversed(cnn_channels))
```

주석 `# If not provided, fall back to reversed encoder channels.`도 지운다.

- [ ] **Step 5: 모델이 실제로 조립되는지 확인**

```bash
.venv/bin/python - <<'PY'
import torch
from src.models import HybridCNNDiTFiLM
from src.utils.config import load_config
config = load_config("configs/default.yaml")
config["model"]["pop_to_superpop"] = {i: i % 5 for i in range(26)}
config["data"]["gene_size"] = 512
model = HybridCNNDiTFiLM(config)
x = torch.randn(2, config["data"]["num_channels"], 512)
out = model(x, torch.randint(0, 100, (2,)), torch.randint(0, 26, (2,)))
print(tuple(out.shape))
PY
```
Expected: `(2, 4, 512)`

- [ ] **Step 6: 테스트**

```bash
.venv/bin/python -m pytest tests -q 2>&1 | tail -2
```
Expected: `261 passed, 2 warnings`

- [ ] **Step 7: 커밋**

```bash
git add src/models/hybrid_geno_dit.py src/models/modules/conditioning.py
git commit -m "refactor(model): require the nested config, drop unreachable fallbacks"
```

---

## Task 14 — plot_pca의 중복 로더를 공용 IO로 바꾼다

**근거:** `scripts/plot_pca.py:60-100`은 `load_real`과 `load_synthetic`을 자체 구현한다. `src/evaluation/_io.py`가 같은 두 함수를 갖고 있고, plot_pca는 이미 `src.preprocessing.tokenizer`에서 정규화 헬퍼를 임포트하고 있다. 차이는 plot_pca 쪽이 여러 split을 이어 붙인다는 것(`:72-79`)이다. 그것만 호출부에 남기면 된다.

**주의:** `scripts/plot_pca_train_fit_test_overlay.py:22`와 `scripts/plot_pca_arm_comparison.py:42`가 `plot_pca`에서 심볼을 임포트한다. 무엇을 임포트하는지 먼저 확인해야 한다.

**Files:**
- Modify: `scripts/plot_pca.py:42-45`, `:60-100`, `:300-310`

**Interfaces:**
- Consumes: Task 10 이후의 `src/evaluation/_io.py`
- Produces: `plot_pca`에서 `load_real`과 `load_synthetic`이 사라진다. `SUPERPOP_COLORS`, `SUPERPOP_ORDER`, `apply_nonzero_mask`, `pop_idx_to_superpop_name`, `infer_title`, `plot_three_panel`은 유지된다.

- [ ] **Step 1: 하위 스크립트가 무엇을 임포트하는지 확인**

```bash
sed -n 20,30p scripts/plot_pca_train_fit_test_overlay.py
sed -n 40,45p scripts/plot_pca_arm_comparison.py
```

`load_real` 또는 `load_synthetic`을 임포트하면, 그 스크립트도 이 태스크에서 함께 고쳐야 한다. Step 4에 반영한다.

- [ ] **Step 2: 임포트 교체**

`scripts/plot_pca.py`의 tokenizer 임포트를 아래로 바꾼다.

```python
from src.evaluation._io import load_real, load_synthetic  # noqa: E402
```

`import pickle`은 `label_hierarchy.pkl`을 여는 데 아직 쓰이므로 남긴다. `import torch`는 `zero_mask.pt` 로드와 `infer_title`의 체크포인트 읽기에 쓰이므로 남긴다.

- [ ] **Step 3: 자체 로더 2개 삭제**

`def load_real(...)`과 `def load_synthetic(...)` 정의를 모두 지운다. `# ── Data loading ──` 구분선 주석은 `apply_nonzero_mask` 위로 남긴다.

- [ ] **Step 4: 호출부를 새 시그니처에 맞춘다**

`main` 안의 로딩 블록을 아래로 바꾼다.

```python
    stats_path = processed / "normalization_stats.pkl"
    loaded = [
        load_real(processed / f"{split}_data.pkl", stats_path=stats_path)
        for split in args.real_splits
    ]
    real_x = np.concatenate([x for x, _ in loaded], axis=0)
    real_y = np.concatenate([y for _, y in loaded], axis=0)
    syn_x, syn_y, _ = load_synthetic(syn_dir, stats_path=stats_path)
```

`real_paths = [...]` 줄을 지운다. `_io.load_real`은 기본 `sample_space="normalized"`로 동작하므로 기존 plot_pca의 무조건 역정규화와 같다. `_io.load_synthetic`은 `generation_meta.json`의 `sample_space`를 존중하므로, 옛 샘플 디렉터리를 그릴 때는 `legacy_sample_space="original"`이 필요할 수 있다. 그 경우 오류 메시지가 무엇을 넘겨야 하는지 알려 준다.

Step 1에서 하위 스크립트가 이 두 이름을 임포트하는 것으로 확인됐다면, 그 스크립트의 임포트도 `from src.evaluation._io import ...`로 바꾸고 호출부를 같은 방식으로 고친다.

- [ ] **Step 5: 스크립트가 임포트되는지 확인**

```bash
.venv/bin/python -c "
import sys; sys.path.insert(0, 'scripts')
import plot_pca; print(plot_pca.SUPERPOP_ORDER)
import plot_pca_train_fit_test_overlay, plot_pca_arm_comparison; print('ok')
"
```
Expected: superpop 목록과 `ok`

- [ ] **Step 6: 테스트**

```bash
.venv/bin/python -m pytest tests -q 2>&1 | tail -2
```
Expected: `261 passed, 2 warnings`

- [ ] **Step 7: 커밋**

```bash
git add scripts/plot_pca.py scripts/plot_pca_train_fit_test_overlay.py scripts/plot_pca_arm_comparison.py
git commit -m "refactor(plots): load real and synthetic tensors through the shared evaluation IO"
```

---

## Task 15 — 아무도 바꾸지 않는 환경변수와 빈 PEP 723 헤더를 제거한다

**근거:** `HIPODIT_DIM_RED`(`src/preprocessing/config.py:43`)를 `glm_pca` 외의 값으로 두면 `run_pipeline.main`이 즉시 `ValueError`를 던진다(`:84-85`). 즉 값이 하나인 환경변수다. `DIM_RED_METHOD`는 `preprocessing_metadata.json`과 `configs/default.yaml:9`의 provenance 검사에 기록되므로 **상수로 남기고**, 환경변수 조회와 `run_pipeline`의 검사만 없앤다.

`# /// script ... dependencies = [] ///` 헤더는 `scripts/hipodit_rebuild_check.py:2-5`, `scripts/hipodit_rebuild_prepare.py:1-4`, `scripts/hipodit_rebuild_train.py:1-4`, `scripts/hipodit_multiseed.py:2-5`, `scripts/hipodit_multiseed_summary.py:1-4`에 있다. 의존성 목록이 비어 있고 CLAUDE.md의 실행 명령은 `uv run` 없이 인터프리터를 직접 지정한다. 헤더는 아무 일도 하지 않는다.

`src/training/trainer.py`의 `validate(config=...)` 미사용 인자는 Task 4에서 이미 제거됐다. 함수 안 `import pickle`(`:303`)은 파일 상단으로 올린다.

**Files:**
- Modify: `src/preprocessing/config.py:37-43`
- Modify: `src/preprocessing/run_pipeline.py:84-85`
- Modify: `scripts/hipodit_rebuild_check.py`, `scripts/hipodit_rebuild_prepare.py`, `scripts/hipodit_rebuild_train.py`, `scripts/hipodit_multiseed.py`, `scripts/hipodit_multiseed_summary.py` (각 파일 선두)
- Modify: `src/training/trainer.py`

**Interfaces:**
- Consumes: Task 12 이후의 run_pipeline
- Produces: `src.preprocessing.config.DIM_RED_METHOD` 는 문자열 상수 `"glm_pca"`. 환경변수는 읽지 않는다.

- [ ] **Step 1: 환경변수를 상수로**

`src/preprocessing/config.py`의 아래 블록을

```python
# Per-gene dimensionality reduction backend.
#   'pca'      — Gaussian PCA (sklearn). Fast (~10 ms/gene); misspecified for
#                 Binomial(2, p) genotype dosage data.
#   'glm_pca'  — Poisson GLM-PCA (Townes et al. 2019); an explicit count-model
#                 approximation for bounded dosage, accelerated by glmpca-fast.
# Override at runtime: HIPODIT_DIM_RED=glm_pca python src/preprocessing/run_pipeline.py
DIM_RED_METHOD = os.environ.get("HIPODIT_DIM_RED", "glm_pca")
```

아래로 바꾼다.

```python
# Per-gene dimensionality reduction backend: Poisson GLM-PCA (Townes et al.
# 2019), an explicit count-model approximation for bounded dosage, accelerated
# by glmpca-fast. Recorded in preprocessing_metadata.json and checked against
# configs/default.yaml's data.dim_reduction_method before training.
DIM_RED_METHOD = "glm_pca"
```

- [ ] **Step 2: run_pipeline의 항진 검사 제거**

`src/preprocessing/run_pipeline.py`에서 아래 두 줄을 지운다.

```python
    if DIM_RED_METHOD != "glm_pca":
        raise ValueError("The production preprocessing pipeline requires glm_pca")
```

`DIM_RED_METHOD` 임포트는 metadata 기록에 여전히 쓰이므로 남긴다.

- [ ] **Step 3: 빈 PEP 723 헤더 5개 제거**

각 파일에서 아래 네 줄(또는 shebang 다음의 네 줄)을 지운다.

```python
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
```

`# ─── How to run ───` 주석 블록은 남긴다. 실행 방법을 담고 있다.

- [ ] **Step 4: trainer의 함수 안 import를 올린다**

`src/training/trainer.py`의 label hierarchy 블록에서 `import pickle` 줄을 지우고, 파일 상단 표준 라이브러리 import에 `import pickle`을 추가한다 (`import os` 다음 알파벳 순).

- [ ] **Step 5: 확인 + 테스트**

```bash
grep -rn "HIPODIT_DIM_RED\|/// script" src scripts
.venv/bin/python -c "from src.preprocessing.config import DIM_RED_METHOD; print(DIM_RED_METHOD)"
.venv/bin/python -c "import src.training.trainer"
.venv/bin/python -m pytest tests -q 2>&1 | tail -2
```
Expected: grep 출력 없음, `glm_pca`, 오류 없음, `261 passed, 2 warnings`

- [ ] **Step 6: 커밋**

```bash
git add src/preprocessing/config.py src/preprocessing/run_pipeline.py src/training/trainer.py scripts/hipodit_*.py
git commit -m "refactor: fix the dim-reduction backend as a constant, drop empty PEP 723 headers"
```

---

## Task 16 — 의존성 선언을 실제 사용에 맞춘다

**근거:** `seaborn`은 `src/`, `scripts/`, `tests/` 어디에서도 임포트되지 않는다. `ipykernel`은 노트북 전용이므로 런타임 의존성이 아니라 dev 그룹에 속한다. 반대로 `scipy`는 `src/preprocessing/binomial_glm_pca.py`, `src/models/genotype_decoder.py`, `scripts/hipodit_genotype_check.py`, `scripts/hipodit_privacy.py`, `scripts/hipodit_simulation.py`에서 직접 임포트되는데 선언돼 있지 않다. 지금은 `scikit-learn`의 전이 의존성으로 우연히 설치돼 있다. 이것은 과설계 제거가 아니라 누락 수정이지만, 같은 파일을 만지므로 함께 한다.

`pysam`은 `src/preprocessing/merge_data.py` 전용이다. 그 파일의 운명은 Task 18(승인 게이트)에 달려 있으므로 **여기서는 건드리지 않는다**.

**Files:**
- Modify: `pyproject.toml:7-20`, `:26-29`

**Interfaces:**
- Consumes: 없음
- Produces: 선언된 런타임 의존성에서 `seaborn`과 `ipykernel`이 빠지고 `scipy`가 추가된다.

- [ ] **Step 1: 미사용 의존성을 한 번 더 확인**

```bash
grep -rn "import seaborn\|from seaborn\|import ipykernel" src scripts tests notebooks
```
Expected: 출력 없음. 노트북에서 나오면 그 항목은 남기고 보고한다.

- [ ] **Step 2: `pyproject.toml` 수정**

`dependencies` 목록에서 아래 두 줄을 지운다.

```toml
    "ipykernel>=7.2.0",
```
```toml
    "seaborn>=0.13.2",
```

같은 목록에 아래를 추가한다 (`pyyaml` 다음, 알파벳 순).

```toml
    "scipy>=1.14",
```

`dependency-groups`의 `dev` 목록에 아래를 추가한다.

```toml
    "ipykernel>=7.2.0",
```

- [ ] **Step 3: 설치된 scipy 버전이 바닥선을 넘는지 확인**

```bash
.venv/bin/python -c "import scipy; print(scipy.__version__)"
```
Expected: 1.14 이상. 낮으면 바닥선을 실제 버전으로 낮추고 그 사실을 보고한다.

- [ ] **Step 4: 잠금 파일은 갱신하지 않는다**

Global Constraint 4가 `uv sync`를 금지한다. `uv.lock`은 다음 사람이 의존성을 설치할 때 갱신된다. 이 태스크는 선언만 고친다. `uv lock`도 실행하지 않는다.

- [ ] **Step 5: 테스트**

```bash
.venv/bin/python -m pytest tests -q 2>&1 | tail -2
```
Expected: `261 passed, 2 warnings`

- [ ] **Step 6: 커밋**

```bash
git add pyproject.toml
git commit -m "build: declare scipy, drop seaborn, move ipykernel to dev"
```

---

## Task 17 — 사라진 플래그·키·파일을 문서에서 지운다

**근거:** Task 1–16이 config 키 4개(`save_every`, `save_top_k_checkpoints`, `optimizer`, `precision`, `prediction_target`), 환경변수 1개(`HIPODIT_DIM_RED`), CLI 플래그 3개(`--array-cache-mode`, `--pca-cache-mode`, `--cache-dir`), 모듈 1개(`src/utils/logger.py`)를 없앤다. README.md와 CLAUDE.md가 그것들을 설명하고 있다. 문서가 없는 플래그를 가리키면 다음 사람이 그것을 되살리려 한다.

선행 기획서의 Task 5가 같은 종류의 작업을 했으므로 그 형식을 따른다.

**Files:**
- Modify: `README.md:473`, `:991`, `:1055-1056`, `:1107-1111`, `:1156`, `:1170-1176`
- Modify: `CLAUDE.md:75`, `:183`

**Interfaces:**
- Consumes: Task 1–16의 모든 변경
- Produces: 문서만. 코드 계약 변화 없음.

- [ ] **Step 1: 현재 상태를 기준으로 실제 남은 키를 확인**

```bash
grep -n "" configs/default.yaml | grep -E "save_every|save_top_k|optimizer|precision|prediction_target"
```
Expected: 출력 없음. 남아 있으면 해당 태스크가 미완이므로 멈춘다.

- [ ] **Step 2: README의 diffusion 설정 표에서 행 삭제**

`README.md`의 설정 표에서 아래 행을 지운다.

```
| `prediction_target` | ε (epsilon) | `diffusion.prediction_target` |
```

- [ ] **Step 3: README의 실패 검사 표에서 행 삭제**

아래 행을 지운다. 검사 자체가 없어졌다.

```
| `HIPODIT_DIM_RED` 가 `glm_pca` 가 아님 | `run_pipeline.main` | 학습·추론 계약이 GLM-PCA decoder 를 전제 |
```

- [ ] **Step 4: README Quick Start의 주석 두 줄 삭제**

아래 두 줄을 지운다.

```
#   training.precision: bf16(기본) | fp32 — 학습·검증 autocast 둘 다 이 값을 따른다
#   training.optimizer: adamw 만 지원 (다른 값이면 ValueError)
```

- [ ] **Step 5: README의 차원축소 절을 갱신**

해당 문단을 아래로 바꾼다.

```markdown
`run_pipeline.py` 는 **GLM-PCA (Townes et al. 2019, Poisson family) 만** 받는다. 백엔드는 상수이며 런타임 전환 경로가 없다.

| 설정 | 값 | 위치 |
|---|---|---|
| backend | `glm_pca` (상수) | `config.DIM_RED_METHOD` |
| family | Poisson (`poi`) 고정, 다른 family 설정 없음 | `glm_pca.DEFAULT_GLM_FAMILY` |
| 성분 수 K | 4 | `config.PCA_K` |
| 최대 반복 | 100 | `config.GLM_PCA_MAX_ITER`, env `HIPODIT_GLM_MAX_ITER` |
| 가속 | `glmpca-fast` (PyPI, Rust) — `uv sync` 로 설치됨 | `pyproject.toml` |
```

(이전 판의 `src/preprocessing/dim_reduction.py` 언급도 함께 사라진다. 그 파일은 선행 기획서 Task 2에서 삭제됐다.)

- [ ] **Step 6: README 트리 그림 갱신**

`trainer.py` 줄을 아래로 바꾼다.

```
│   │   └── trainer.py              # DDP 학습 루프 (bf16 autocast, AdamW, cosine warmup LambdaLR)
```

`utils/` 블록에서 `logger.py` 줄과 그 아래 괄호 주석을 아래로 바꾼다.

```
│   └── utils/
│       ├── config.py               # YAML 로드
│       ├── ddp.py                  # DDP setup/cleanup
│       └── ema.py                  # EMA (decay 0.999, configs/default.yaml)
│                                   # (.pth 저장은 src/training/trainer.py 안에 있다.
│                                   #  wandb 호출도 trainer 안에 인라인되어 있다)
```

`evaluation/` 블록의 `_io.py` 줄을 아래로 바꾼다.

```
│   │   └── _io.py                  # 프로젝트-특화 IO (real/synthetic 로더, CSV)
```

- [ ] **Step 7: README 테스트 수 갱신**

`├── tests/` 줄의 테스트 수를 실제 값으로 바꾼다.

```bash
.venv/bin/python -m pytest tests -q 2>&1 | tail -2
```

출력된 숫자를 그대로 쓴다.

```
├── tests/                            # pytest tests/ → 261 tests
```

- [ ] **Step 8: CLAUDE.md 두 줄 갱신**

`src/` 트리 설명의 `utils/` 줄을 아래로 바꾼다.

```
├── utils/                # DDP setup, config loading, EMA (wandb는 trainer 안에서 직접 호출)
```

차원축소 항목을 아래로 바꾼다.

```
- **차원 축소**: `glm_pca` 상수 (`src/preprocessing/config.py`의 `DIM_RED_METHOD`). `configs/default.yaml`은 `num_channels: 4`, `gene_size: 24576`을 쓴다. 모델 입력은 `(num_channels, gene_size)`.
```

- [ ] **Step 9: 문서에 죽은 참조가 남지 않았는지 확인**

```bash
grep -rn "HIPODIT_DIM_RED\|ExperimentLogger\|prediction_target\|save_top_k\|training.precision\|training.optimizer\|array-cache\|pca-cache\|dim_reduction.py" README.md CLAUDE.md
```
Expected: 출력 없음

`docs/reports/`와 `docs/superpowers/plans/`의 옛 보고서는 **고치지 않는다.** 그것들은 작성 시점의 기록이다.

- [ ] **Step 10: 커밋**

```bash
git add README.md CLAUDE.md
git commit -m "docs: remove the flags, keys and modules this refactor deleted"
```

---

# Group A — 승인 게이트 (승인 없이 착수 금지)

아래 다섯 태스크는 연구 기록·재현 명령과 얽혀 있다. 각 태스크는 "무엇을 잃는가"를 먼저 적었다. 사용자가 태스크 번호를 지목해 승인하기 전에는 실행하지 않는다.

## Task 18 — 1회성 VCF 유틸 2개 삭제 (약 515줄)

**대상:** `src/preprocessing/merge_data.py` (246줄), `scripts/export_chr17_csv.py` (269줄). `pysam` 의존성도 함께 빠진다.

**삭제 근거:** 둘 다 한 번 돌리고 끝나는 도구다. merge_data의 산출물 `data/ALL.autosomes.phase3.genotypes.vcf.gz`(13.9 GB)는 이미 존재하고, export_chr17_csv는 2026-04-10 이후 손대지 않았다. merge_data의 docstring은 `tsv`/`pkl` 출력 형식을 광고하지만 구현은 `vcf` 하나뿐이다. CLAUDE.md도 merge_data를 "not part of the pipeline"으로 적고 있다.

**무엇을 잃는가:** VCF를 처음부터 다시 병합하거나 chr17 TSV를 다시 뽑아야 하면 `git show`로 되살려야 한다. 병합 산출물이 소실되고 원본 per-chromosome VCF(`~/GeneDiffusion`)만 남은 상황이 오면 그 복구 절차가 한 단계 늘어난다.

**승인 시 절차:** `git rm` 두 개, `pyproject.toml`에서 `pysam` 제거, README.md의 Quick Start Phase 0.5 블록과 트리 그림 두 줄, CLAUDE.md의 "VCF merging" 명령 블록 제거. CLAUDE.md의 preprocessing 원칙 항목에서 merge_data 문장을 지운다. 테스트 수 변화 없음.

## Task 19 — Fisher 3-band schedule 체인 삭제 (약 300줄)

**대상:** `src/models/noise_schedule.py`의 `three_band_betas`, `src/models/diffusion.py`의 `feature_schedule` 버퍼와 `_extract` 분기, `src/preprocessing/binomial_glm_pca.py`의 `fisher_diagonal`·`information_schedule`, `scripts/hipodit_rebuild_check.py`의 `--schedule` 인자, `scripts/hipodit_rebuild_train.py`의 fisher 분기, `scripts/hipodit_multiseed.py`의 schedule 모드, `scripts/hipodit_multiseed_summary.py`의 `SCHEDULES`·`summarize`·`write_summary`·`_validated_run`의 fisher 검증, `scripts/hipodit_genotype_check.py`의 `_pilot_block`, 그리고 `tests/test_hipodit_multiseed.py`의 schedule-mode 테스트 4건과 `tests/test_binomial_glm_pca.py::test_fisher_sensitivity_includes_normalization_chain_rule`.

**삭제 근거:** 결과 보고서 §9.4가 5 seed에서 일관된 이득 없음으로 종결했고, 기획서 §1이 표준 schedule을 기준선으로 고정했다. Fisher는 "최종 부가 ablation"으로만 남겨졌으나 그 ablation(B3-F)은 실행되지 않았다.

**무엇을 잃는가:** 세 가지다. (1) 결과 보고서 §12.1에 적힌 `--schedule fisher` 재현 명령이 동작하지 않는다. (2) `outputs/diagnostics/hipodit_fisher_20260915_unique` 패널을 만든 `prepare` 경로는 남지만, 그 패널로 Fisher 비교를 다시 돌릴 수는 없다. (3) 예정된 B3-F ablation을 하려면 코드를 되살려야 한다. 동결 산출물 자체는 그대로 남으므로 이미 보고된 수치는 검증 가능하다.

**규모가 크고 되돌리기 비싼 태스크다.** 승인한다면 별도 기획서로 분해해 단계별 게이트를 두는 것이 낫다. 이 문서에서는 승인 여부만 받는다.

## Task 20 — `docs/reports/prototypes/*.py` 삭제 (536줄)

**대상:** `af_drift_prototypes.py` (260줄), `af_tilt_variants.py` (276줄).

**삭제 근거:** 프로토타입 스크립트다. 결론은 `docs/reports/hipodit_research_af_drift.md`와 `hipodit_ld_tilt_20260915.md`에, 수치는 같은 디렉터리의 json 4개에 남아 있다. 채택된 설계는 `src/models/genotype_decoder.py`에 구현돼 있다.

**무엇을 잃는가:** json 수치를 어떤 코드가 만들었는지 확인하려면 `git show`가 필요하다. json은 남기므로 수치 자체는 인용 가능하다.

**승인 시 절차:** `git rm` 두 개. json 4개와 마크다운 보고서는 그대로 둔다. 테스트 수 변화 없음.

## Task 21 — `_legacy_study` B0–B3 경로 삭제 (약 40줄)

**대상:** `scripts/hipodit_genotype_check.py`의 `_legacy_study`와 `--arms legacy` 선택지.

**삭제 근거:** 현재 채택 모델은 arm T이고, `--arms t`가 기본값이다. B0–B3 ablation 산출물은 `outputs/diagnostics/hipodit_ld_oracle_20260915/`에 동결돼 있으며 CLAUDE.md가 재생성을 금지한다.

**무엇을 잃는가:** B0–B3 oracle 비교를 다시 돌릴 수 없다. 단 CLAUDE.md가 이미 재생성을 금지하므로 코드가 없어도 규정 위반은 아니다. `src/models/genotype_decoder.py`의 B0–B3 arm 자체는 남으므로(`fit_decoder`가 arm 이름을 받는다) 필요하면 호출 코드를 다시 쓸 수 있다.

**확인이 필요한 점:** `tests/test_hipodit_genotype_check.py`가 legacy 경로를 검증하는지 먼저 확인해야 한다. 검증한다면 그 테스트도 함께 사라진다.

## Task 22 — `binom_glmpca_rs/` 추적 여부 결정

**상태:** 이 디렉터리는 디스크에 있고 루트 `.venv`와 csdi 환경 양쪽에 설치돼 있으나 **git에 추적되지 않는다.** `src/preprocessing/glm_pca.py`의 `_BINOM_BACKEND`가 설치돼 있으면 그것을 우선 선택하므로, 깨끗한 체크아웃에서는 scipy fallback이 돌고 **같은 입력에 다른 백엔드가 쓰인다.**

이것은 과설계가 아니라 재현성 결함이다. 두 선택지가 있다.

1. **디렉터리를 커밋한다.** `vcf_parser_rs/`가 이미 추적되고 있으므로 전례가 있다. 어느 백엔드가 돌았는지는 `_build_result`가 `backend`/`backend_version`으로 기록하므로 산출물에서 추적 가능하다.
2. **Rust fallback 선택을 제거하고 scipy 참조만 남긴다.** 벤치마크상 4.84초 대 1.28초이므로 전장유전체 전처리에서는 느려진다.

**권고: 1번.** 백엔드가 실제로 쓰이고 있고, 성능 차가 실질적이며, 추적하지 않는 것이 유일한 문제다.

**승인 시 절차:** `binom_glmpca_rs/`의 소스와 빌드 설정만 추가한다(`target/` 제외). `.gitignore`에 `binom_glmpca_rs/target/`을 넣는다. README에 `uv pip install -e ./binom_glmpca_rs` 설치 줄을 추가한다.

---

## Self-Review

**1. 감사 항목 커버리지.** 2026-09-18 감사의 Group B 19항목 중 17개가 Task 1–16에 들어갔다. 들어가지 않은 2개와 그 이유:

- **legacy sample-space 배관 제거** — Task 10에서 판단을 뒤집어 남겼다. `outputs/` 아래 옛 run이 실재하고, 현재 generator가 항상 `sample_space`를 기록한다는 사실이 옛 산출물의 소멸을 뜻하지 않는다.
- **`build_generation_diffusion` 인라인** — Task 9에서 남겼다. `tests/test_inference_generator_contract.py`의 `_run_generation`이 monkeypatch 지점으로 쓰므로 실제 값이 있다.

Group C 3항목: `seaborn`/`ipykernel`은 Task 16, `scipy` 누락도 Task 16, `pysam`은 Task 18(승인 게이트)로 미뤘다. `binom_glmpca_rs` 추적 문제는 Task 22.

Group A 5항목은 Task 18–22에 승인 게이트로 들어갔다.

**2. 플레이스홀더 점검.** "적절히 처리", "TODO", "Task N과 유사" 같은 표현을 쓴 곳은 없다. 코드가 필요한 스텝에는 전체 교체 블록을 실었다. Task 19만 예외적으로 코드 없이 승인 여부만 묻는데, 이것은 의도적이다. 규모가 별도 기획서를 요구한다고 본문에 적었다.

**3. 타입·이름 일관성 점검.** 태스크 사이를 넘나드는 시그니처 변경 3건을 교차 확인했다.

- `validate(...)`: Task 4가 4-인자로 줄인다. 호출자는 trainer 내부 1곳이며 같은 태스크에서 고친다.
- `save_checkpoint(...)`: Task 3이 6-인자로 줄인다. 호출자는 trainer 내부 1곳(best 저장)이며 같은 태스크에서 고친다. Task 3이 주기 저장·final 저장 호출을 먼저 지우므로 남는 호출은 하나다.
- `split_dataset_stratified(...)` / `save_all(...)`: Task 12가 바꾼다. 호출자는 `run_pipeline.py` 각 1곳이며 같은 태스크에서 고친다.
- `postprocess_samples(...)`: Task 9가 역정규화를 흡수한다. 호출자는 `generator.generate_samples` 1곳이고 인자 수가 그대로다(4개).
- `load_real` / `load_synthetic`: Task 14가 plot_pca를 `_io` 쪽으로 옮긴다. `_io.load_real`은 단일 경로를 받고 plot_pca는 여러 split을 이어 붙였으므로, 연결을 호출부로 올리는 코드를 Step 4에 실었다. `_io.load_synthetic`은 3-튜플을 반환하므로(plot_pca의 자체 구현은 2-튜플) 세 번째 값을 버리는 것도 같은 스텝에 적었다.

**4. 테스트 수 추적.** 기준선 267. Task 5가 +2 → 269. Task 7이 −3 → 266. Task 8이 −4 → 262. Task 9가 −1 → 261. Task 10이 ±0(1 삭제, 1 추가) → 261. 최종 261. Task 17 Step 7이 README의 테스트 수를 실측값으로 적게 한다. 파라미터화 삭제의 순감이 계산과 다를 수 있으므로 Task 7과 8의 Step에 "숫자가 다르면 멈추고 보고" 지시를 넣었다.

**5. Global Constraint 3의 최종값 정정.** 헌법 항목에는 `258 passed`로 적었으나 태스크별 누적은 261이다. 실행자는 **261**을 기준으로 삼는다. 각 태스크 안의 기대값이 정본이다.
