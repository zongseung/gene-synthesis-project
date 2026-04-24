from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge and deduplicate SFT QA jsonl files.")
    p.add_argument("--inputs", nargs="+", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-jaccard", type=float, default=0.95)
    return p.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def trigram_jaccard(a: str, b: str) -> float:
    def grams(text: str) -> set[str]:
        if len(text) < 3:
            return {text}
        return {text[i : i + 3] for i in range(len(text) - 2)}

    ga, gb = grams(a), grams(b)
    return len(ga & gb) / max(len(ga | gb), 1)


def main() -> int:
    args = parse_args()
    merged: list[dict] = []
    for path in args.inputs:
        merged.extend(load_jsonl(path))

    deduped: list[dict] = []
    for row in merged:
        duplicate = False
        for kept in deduped:
            if row["question"] == kept["question"] and row["assistant"] == kept["assistant"]:
                duplicate = True
                break
            if trigram_jaccard(row["assistant"], kept["assistant"]) >= args.max_jaccard:
                if row["category"] == kept["category"]:
                    duplicate = True
                    break
        if not duplicate:
            deduped.append(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for row in deduped:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[merge_sft_qa] input={len(merged)} deduped={len(deduped)} out={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
