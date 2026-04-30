"""Build ver8.2 friendly gold bootstrap rows from v8.1 SFT corpus.

This is a deterministic starter gold set for rewrite few-shot examples.
It preserves the original question/user turn and rewrites only the assistant
answer into a friendlier format with the source excerpt included verbatim.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

DEFAULT_QUOTAS = {
    "병증 설명": 45,
    "본문 설명": 25,
    "편명": 20,
    "서문": 5,
    "총목": 5,
}

FIELD_RE = re.compile(r"^(유형|항목|경로|원문 발췌):\s*(.*)$", re.M)


def parse_question(question: str) -> dict[str, str]:
    fields = {m.group(1): m.group(2).strip() for m in FIELD_RE.finditer(question)}
    if "원문 발췌" not in fields:
        m = re.search(r"원문 발췌:\s*(.+)\s*$", question or "", re.S)
        if m:
            fields["원문 발췌"] = m.group(1).strip()
    return fields


def clean_summary(answer: str) -> str:
    head = (answer or "").split("\n\n해설:", 1)[0].strip()
    for prefix in ("현대 한국어:", "정리:"):
        if head.startswith(prefix):
            head = head[len(prefix):].strip()
    head = re.sub(r"\s+", " ", head).strip()
    return head


def intro_for(subcat: str, item: str) -> str:
    item_text = f"'{item}'" if item else "이 항목"
    if subcat == "병증 설명":
        return f"이 대목은 동의보감이 {item_text}을 당시 의학 체계 안에서 어떻게 설명했는지 보여 줍니다."
    if subcat == "본문 설명":
        return f"이 기록은 동의보감 본문에서 {item_text}과 관련해 인용하거나 설명한 대목입니다."
    if subcat == "편명":
        return f"이 기록은 동의보감에서 {item_text} 항목이 어디에 놓이는지 알려 주는 표제입니다."
    if subcat == "서문":
        return "이 대목은 동의보감의 편찬·간행 맥락을 보여 주는 서문 계열 기록입니다."
    if subcat == "총목":
        return "이 기록은 동의보감의 목차와 분류 체계를 이해하는 데 쓰이는 총목 계열 항목입니다."
    return "이 대목은 동의보감 원문을 현대 한국어로 이해하기 위한 문헌 기록입니다."


def easy_explain(subcat: str, item: str, path: str) -> str:
    place = f" 경로는 {path}입니다." if path else ""
    if subcat == "병증 설명":
        return (
            f"쉽게 말하면, {item or '해당 병증'}을 증상·원인·치법의 문헌 맥락에서 정리한 자료입니다."
            f"{place} 현재 증상에 대한 진단이나 복용 지시가 아니라, 고전 본문을 이해하기 위한 설명으로 보아야 합니다."
        )
    if subcat == "본문 설명":
        return (
            f"쉽게 말하면, {item or '해당 항목'}에 대해 동의보감이 어떤 문헌 표현을 남겼는지 확인하는 자료입니다."
            f"{place} 실제 치료 판단보다는 원문 의미와 인용 맥락을 파악하는 데 초점을 두면 됩니다."
        )
    if subcat == "편명":
        return (
            f"쉽게 말하면, 이 문장은 내용 설명이라기보다 {item or '해당 항목'}의 위치를 알려 주는 제목 표지에 가깝습니다."
            f"{place} 그래서 효능이나 처방을 바로 끌어내기보다 책 안의 구조를 확인하는 자료로 읽으면 됩니다."
        )
    if subcat == "서문":
        return (
            "쉽게 말하면, 본문 처방 설명이 아니라 책이 언제, 누구의 명으로, 어떤 편찬 맥락에서 만들어졌는지를 보여 주는 기록입니다."
            f"{place} 동의보감의 성격과 간행 배경을 이해하는 자료로 읽으면 됩니다."
        )
    if subcat == "총목":
        return (
            f"쉽게 말하면, {item or '이 항목'}은 동의보감 안에서 주제들이 어떤 순서와 묶음으로 배열되는지 보여 줍니다."
            f"{place} 개별 치료법보다 책의 분류 체계를 확인하는 데 알맞은 자료입니다."
        )
    return f"쉽게 말하면, 이 기록은 동의보감 원문을 문헌적으로 풀어 읽기 위한 자료입니다.{place}"


def build_friendly_answer(row: dict) -> str:
    fields = parse_question(row.get("question", ""))
    subcat = row.get("subcat", "")
    item = fields.get("항목", "")
    path = fields.get("경로", "")
    excerpt = fields.get("원문 발췌", "")
    summary = clean_summary(row.get("assistant", ""))

    parts = [
        intro_for(subcat, item),
        f"발췌: {excerpt}" if excerpt else "",
    ]
    if summary:
        parts.append(f"원문을 풀면, {summary}")
    parts.append(easy_explain(subcat, item, path))
    return "\n\n".join(p for p in parts if p).strip()


def is_good_candidate(row: dict) -> bool:
    fields = parse_question(row.get("question", ""))
    excerpt = fields.get("원문 발췌", "")
    answer = row.get("assistant", "")
    if not excerpt or len(excerpt) < 3 or len(excerpt) > 240:
        return False
    if not answer or len(answer) < 100 or len(answer) > 900:
        return False
    if "\r" in row.get("question", "") or "\r" in answer:
        return False
    if "�" in row.get("question", "") or "�" in answer:
        return False
    return True


def sync_assistant(row: dict) -> None:
    messages = row.get("messages")
    if not isinstance(messages, list):
        return
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            msg["content"] = row["assistant"]
            return


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=ROOT / "data/sft/phaseB_qa_v8_1_corpus.jsonl")
    ap.add_argument("--output", type=Path, default=ROOT / "data/sft/friendly_gold_v0.jsonl")
    ap.add_argument("--seed", type=int, default=82)
    args = ap.parse_args()

    buckets: dict[str, list[dict]] = defaultdict(list)
    for line in args.input.open(encoding="utf-8"):
        row = json.loads(line)
        if is_good_candidate(row):
            buckets[row.get("subcat", "?")].append(row)

    rng = random.Random(args.seed)
    selected: list[dict] = []
    for subcat, quota in DEFAULT_QUOTAS.items():
        bucket = list(buckets.get(subcat, []))
        rng.shuffle(bucket)
        chosen = sorted(bucket[:quota], key=lambda r: r.get("id", ""))
        if len(chosen) < quota:
            raise SystemExit(f"not enough candidates for {subcat}: {len(chosen)} < {quota}")
        selected.extend(chosen)

    selected.sort(key=lambda r: (r.get("subcat", ""), r.get("id", "")))
    out_rows: list[dict] = []
    for idx, row in enumerate(selected, 1):
        fields = parse_question(row.get("question", ""))
        new_row = dict(row)
        new_row["id"] = f"friendly_gold_v0/{idx:03d}/{row.get('id')}"
        new_row["_source_id"] = row.get("id")
        new_row["_source_question"] = row.get("question", "")
        new_row["_source_excerpt"] = fields.get("원문 발췌", "")
        new_row["_source_assistant"] = row.get("assistant", "")
        new_row["assistant"] = build_friendly_answer(row)
        new_row["tone"] = "friendly"
        new_row["_origin"] = "bootstrap_gold_v0"
        sync_assistant(new_row)
        out_rows.append(new_row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"wrote {len(out_rows)} rows -> {args.output}")
    for subcat in DEFAULT_QUOTAS:
        print(f"  {subcat}: {sum(1 for r in out_rows if r.get('subcat') == subcat)}")


if __name__ == "__main__":
    main()
