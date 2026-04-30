"""ver8.2 § 4.2 — ver8.1 corpus 에서 친절체 rewrite 대상 stratified sample 추출.

Plan B (10,000 rows) 기준 카테고리 비율:
  병증 설명 5,000 / 편명 3,300 / 본문 설명 1,600 / 서문 70 / 총목 30

사용:
  .venv/bin/python experiments/dongui_bogam/scripts/build_friendly_rewrite_sample.py
  # 기본: data/sft/phaseB_qa_v8_1_corpus.jsonl → data/sft/v8_1_rewrite_sample.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

# Plan B 기본 quota — 기획서 §3.3
PLAN_B_QUOTAS = {
    "병증 설명": 5000,
    "편명": 3300,
    "본문 설명": 1600,
    "서문": 70,
    "총목": 30,
}

QUOTA_ALIASES = {
    "병증": "병증 설명",
    "본문": "본문 설명",
}


def parse_quotas(spec: str) -> dict[str, int]:
    """`병증=5000,편명=3300` 형식 파싱."""
    out = {}
    for piece in spec.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "=" not in piece:
            raise ValueError(f"quota 항목은 cat=N 형식이어야 합니다: {piece}")
        k, v = piece.split("=", 1)
        key = QUOTA_ALIASES.get(k.strip(), k.strip())
        out[key] = int(v.strip())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path,
                    default=ROOT / "data" / "sft" / "phaseB_qa_v8_1_corpus.jsonl")
    ap.add_argument("--output", type=Path,
                    default=ROOT / "data" / "sft" / "v8_1_rewrite_sample.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quotas", type=str, default=None,
                    help="cat=N,cat=N 형식. 미지정 시 Plan B 기본값")
    args = ap.parse_args()

    quotas = parse_quotas(args.quotas) if args.quotas else PLAN_B_QUOTAS
    rng = random.Random(args.seed)

    if not args.input.exists():
        raise SystemExit(f"input 없음: {args.input}")

    # 카테고리별 bucket (subcat 우선, 없으면 category)
    buckets: dict[str, list[dict]] = defaultdict(list)
    for line in args.input.open():
        r = json.loads(line)
        sub = r.get("subcat") or r.get("category") or "?"
        buckets[sub].append(r)

    print(f"[input] {args.input}")
    print(f"  total: {sum(len(v) for v in buckets.values())} rows")
    print("  카테고리 분포 (top 10):")
    for k, v in sorted(buckets.items(), key=lambda x: -len(x[1]))[:10]:
        print(f"    {len(v):>6}  {k}")

    # stratified sample
    selected: list[dict] = []
    print("\n[sampling]")
    for cat, quota in quotas.items():
        bucket = buckets.get(cat, [])
        if not bucket:
            print(f"  [warn] '{cat}' — 0 rows in input")
            continue
        if len(bucket) <= quota:
            print(f"  [warn] '{cat}' has {len(bucket)} ≤ quota {quota}, taking all")
            sample = list(bucket)
        else:
            sample = rng.sample(bucket, quota)
        selected.extend(sample)
        print(f"    {len(sample):>5} from '{cat}' (bucket={len(bucket)})")

    rng.shuffle(selected)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        for r in selected:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n✓ wrote {len(selected)} rows → {args.output}")
    print("  next: rewrite_to_friendly.py 로 LLM rewrite")


if __name__ == "__main__":
    main()
