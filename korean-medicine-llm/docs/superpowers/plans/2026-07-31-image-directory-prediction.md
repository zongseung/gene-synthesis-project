# Image-directory Prediction CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a thin CLI that recursively runs the trained VARCO-VISION adapter on either herb or tongue images and writes resumable JSONL predictions.

**Architecture:** Create one script that reuses `_load_model` and `_generate` from `quick_eval_sft_varco.py`. Keep directory discovery, task prompts, and resume filtering as small pure helpers so they can be tested without loading a model.

**Tech Stack:** Python standard library, Pillow, existing Transformers/PEFT inference helpers, pytest

## Global Constraints

- The caller must explicitly pass `--task herb` or `--task tongue`; do not auto-detect the task.
- Tongue prompts request visible observations only, never diagnosis or pattern identification.
- Recursively support `.jpg`, `.jpeg`, `.png`, and `.webp` in stable order.
- Add no dependency and do not refactor the existing evaluator.

---

### Task 1: Add and verify the directory prediction CLI

**Files:**
- Create: `scripts/predict_image_dir.py`
- Create: `tests/test_predict_image_dir.py`

**Interfaces:**
- Consumes: `quick_eval_sft_varco._load_model(cfg: dict, adapter: Path)` and `quick_eval_sft_varco._generate(processor, model, question: str, image) -> tuple[str, float]`
- Produces: `discover_images(root: Path) -> list[Path]`, `prompt_for(task: str) -> str`, `pending_images(images: list[Path], records: list[dict], task: str) -> list[Path]`, and `compact_records(records: list[dict]) -> list[dict]`

- [ ] **Step 1: Write the failing helper test**

```python
from pathlib import Path

from predict_image_dir import compact_records, discover_images, pending_images, prompt_for


def test_directory_prediction_helpers(tmp_path: Path):
    first = tmp_path / "a.JPG"
    second = tmp_path / "nested" / "b.png"
    second.parent.mkdir()
    first.touch()
    second.touch()
    (tmp_path / "ignore.txt").touch()

    images = discover_images(tmp_path)
    assert images == [first.resolve(), second.resolve()]
    assert "이름" in prompt_for("herb")
    assert "관찰" in prompt_for("tongue")
    assert "변증하지" in prompt_for("tongue")

    records = [
        {"image": str(first.resolve()), "task": "herb", "answer_text": "칡"},
        {"image": str(second.resolve()), "task": "herb", "error": "broken"},
    ]
    assert pending_images(images, records, "herb") == [second.resolve()]
    assert pending_images(images, records, "tongue") == images
    assert compact_records(records + [{**records[1], "answer_text": "황기", "error": None}])[-1]["answer_text"] == "황기"
```

- [ ] **Step 2: Run the helper test and verify the import fails**

Run: `PYTHONPATH=src:scripts .venv/bin/pytest -q tests/test_predict_image_dir.py`

Expected: FAIL because `predict_image_dir` does not exist.

- [ ] **Step 3: Implement the minimal CLI**

Create `scripts/predict_image_dir.py` with:

```python
"""Run the trained adapter over every image below a directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from quick_eval_sft_varco import _atomic_jsonl, _generate, _load_model, _read_records


EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
PROMPTS = {
    "herb": "이 사진 속 약용식물 또는 약재의 이름은 무엇입니까? 이름만 간결하게 답하세요.",
    "tongue": (
        "이 혀 사진에서 직접 관찰되는 설색, 설태, 혀의 형태 소견만 간결하게 설명하세요. "
        "진단하거나 변증하지 마세요."
    ),
}


def discover_images(root: Path) -> list[Path]:
    root = root.resolve()
    if not root.is_dir():
        raise ValueError(f"이미지 디렉터리가 아닙니다: {root}")
    return sorted(
        (path.resolve() for path in root.rglob("*") if path.is_file() and path.suffix.lower() in EXTENSIONS),
        key=str,
    )


def prompt_for(task: str) -> str:
    return PROMPTS[task]


def pending_images(images: list[Path], records: list[dict], task: str) -> list[Path]:
    completed = {
        (record.get("image"), record.get("task"))
        for record in records
        if record.get("answer_text") and not record.get("error")
    }
    return [image for image in images if (str(image), task) not in completed]


def compact_records(records: list[dict]) -> list[dict]:
    latest = {}
    for record in records:
        latest[(record.get("image"), record.get("task"))] = record
    return list(latest.values())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/sft_varco.yaml")
    parser.add_argument("--adapter", type=Path, default=Path("outputs/sft_varco/adapter"))
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--task", choices=sorted(PROMPTS), required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/image_predictions.jsonl"))
    return parser.parse_args()


def main() -> int:
    from PIL import Image
    from hanmed.training.train import load_config

    args = parse_args()
    try:
        images = discover_images(args.image_root)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if not images:
        raise SystemExit(f"지원하는 이미지가 없습니다: {args.image_root.resolve()}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    records = _read_records(args.output)
    pending = pending_images(images, records, args.task)
    if not pending:
        print(f"이미 처리 완료: {len(images)}개")
        _atomic_jsonl(args.output, compact_records(records))
        return 0

    cfg = load_config(args.config)
    print(f"loading model: {cfg['base_model']} + {args.adapter}", flush=True)
    processor, model = _load_model(cfg, args.adapter)
    with args.output.open("a", encoding="utf-8") as handle:
        for index, path in enumerate(pending, 1):
            try:
                with Image.open(path) as image:
                    answer, latency = _generate(
                        processor, model, prompt_for(args.task), image.convert("RGB")
                    )
                if not answer:
                    raise RuntimeError("empty model answer")
                record = {
                    "image": str(path),
                    "task": args.task,
                    "answer_text": answer,
                    "latency_sec": round(latency, 3),
                }
                print(f"[{index}/{len(pending)}] {path.name}: {answer}", flush=True)
            except Exception as error:
                record = {
                    "image": str(path),
                    "task": args.task,
                    "error": f"{type(error).__name__}: {error}",
                }
                print(f"[{index}/{len(pending)}] {path.name}: ERROR {record['error']}", flush=True)
            records.append(record)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()

    _atomic_jsonl(args.output, compact_records(records))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run focused tests and CLI import verification**

Run:

```bash
PYTHONPATH=src:scripts .venv/bin/pytest -q tests/test_predict_image_dir.py
PYTHONPATH=src .venv/bin/python scripts/predict_image_dir.py --help
```

Expected: the test passes and help lists `--image-root`, `--task`, and `--output` without loading the model.

- [ ] **Step 5: Run the full test suite**

Run: `PYTHONPATH=src .venv/bin/pytest -q`

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add scripts/predict_image_dir.py tests/test_predict_image_dir.py
git commit -m "feat: predict image directories"
```
