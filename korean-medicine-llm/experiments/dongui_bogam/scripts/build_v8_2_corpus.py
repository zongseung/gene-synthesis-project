"""ver8.2 § 5 — phaseB_qa_v8_2_corpus.jsonl 빌드 (ver8.1 + friendly_qa_v0).

ver8.1 (formal, 34,039) + friendly_qa_v0 (~10,100) → mix corpus (~44,100).
각 row 에 `tone` 필드 부착 ("formal" | "friendly") 해서 학습/추적 분리 가능.

사용:
  .venv/bin/python experiments/dongui_bogam/scripts/build_v8_2_corpus.py
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open()]


def sync_assistant_message(row: dict) -> None:
    """학습은 messages 를 쓰므로 assistant 필드와 마지막 assistant turn 을 맞춘다."""
    assistant = row.get("assistant")
    messages = row.get("messages")
    if not assistant or not isinstance(messages, list):
        return
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            msg["content"] = assistant
            return


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path,
                    default=ROOT / "data" / "sft" / "phaseB_qa_v8_1_corpus.jsonl")
    ap.add_argument("--friendly", type=Path,
                    default=ROOT / "data" / "sft" / "friendly_qa_v0.jsonl")
    ap.add_argument("--output", type=Path,
                    default=ROOT / "data" / "sft" / "phaseB_qa_v8_2_corpus.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-shuffle", action="store_true",
                    help="섞지 않음 (디버깅 용)")
    args = ap.parse_args()

    if not args.base.exists():
        raise SystemExit(f"base 없음: {args.base}")
    if not args.friendly.exists():
        raise SystemExit(f"friendly 없음: {args.friendly}")

    base = load_jsonl(args.base)
    friendly = load_jsonl(args.friendly)
    print(f"[base]     {args.base.name:<35} {len(base):>6} rows")
    print(f"[friendly] {args.friendly.name:<35} {len(friendly):>6} rows")

    # tone 라벨 부착 (재실행 idempotent)
    for r in base:
        r.setdefault("tone", "formal")
        sync_assistant_message(r)
    for r in friendly:
        r.setdefault("tone", "friendly")
        sync_assistant_message(r)

    merged = base + friendly
    if not args.no_shuffle:
        random.Random(args.seed).shuffle(merged)
        print(f"[shuffle] seed={args.seed}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        for r in merged:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    pct = 100 * len(friendly) / max(len(merged), 1)
    print(f"\n✓ wrote {len(merged)} rows → {args.output}")
    print(f"  friendly mix ratio: {pct:.2f}%")
    print(f"  next: /sft-quality-fix {args.output.name}")


if __name__ == "__main__":
    main()
