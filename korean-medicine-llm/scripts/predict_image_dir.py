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
        (
            path.resolve()
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in EXTENSIONS
        ),
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
    parser.add_argument(
        "--adapter", type=Path, default=Path("outputs/sft_varco/adapter")
    )
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--task", choices=sorted(PROMPTS), required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/image_predictions.jsonl")
    )
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
                print(
                    f"[{index}/{len(pending)}] {path.name}: ERROR {record['error']}",
                    flush=True,
                )
            records.append(record)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()

    _atomic_jsonl(args.output, compact_records(records))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
