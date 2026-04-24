"""ver6 §3.1.2 원칙 3·4 기계 검증 — SFT 데이터 템플릿 동질성 감사.

기존 ver5 phaseB_qa_diverse_v3_1 의 3,000×3 고정 prefix/closing 을 수치로 공식화.
ver6 빌드 후에도 동일 스크립트로 "prefix top-1 < 15%, closing top-1 < 10%" 통과 여부 확인.

Usage:
    .venv/bin/python scripts/audit_sft_diversity.py experiments/dongui_bogam/data/sft/phaseB_qa_diverse_v3_1.jsonl
    .venv/bin/python scripts/audit_sft_diversity.py experiments/dongui_bogam/data/sft/phaseB_qa_v6_core.jsonl
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path


THRESHOLD_PREFIX_TOP1 = 0.15
THRESHOLD_CLOSING_TOP1 = 0.10
THRESHOLD_UNIQUE_PREFIX_RATIO = 0.03  # unique prefix 수 / 전체 행 수 ≥ 3%


def answer_of(d: dict) -> str:
    # messages 우선, 없으면 assistant 필드
    msgs = d.get("messages") or []
    for m in reversed(msgs):
        if m.get("role") == "assistant":
            return m.get("content") or ""
    return d.get("assistant") or d.get("response") or ""


def audit(path: Path) -> dict:
    prefixes: collections.Counter[str] = collections.Counter()
    closings: collections.Counter[str] = collections.Counter()
    total = 0
    empty = 0
    with path.open() as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                d = json.loads(ln)
            except json.JSONDecodeError:
                continue
            a = answer_of(d)
            if not a:
                empty += 1
                continue
            total += 1
            first = a.split("\n")[0][:20]
            last = a.rstrip()[-20:]
            prefixes[first] += 1
            closings[last] += 1

    if total == 0:
        return {"path": str(path), "total": 0, "verdict": "FAIL_EMPTY"}

    top_prefix, top_prefix_n = prefixes.most_common(1)[0]
    top_closing, top_closing_n = closings.most_common(1)[0]
    unique_prefix_ratio = len(prefixes) / total

    verdicts = []
    if top_prefix_n / total > THRESHOLD_PREFIX_TOP1:
        verdicts.append(
            f"FAIL_PREFIX_TOP1 ({top_prefix!r} @ {top_prefix_n}/{total}={top_prefix_n/total:.2%})"
        )
    if top_closing_n / total > THRESHOLD_CLOSING_TOP1:
        verdicts.append(
            f"FAIL_CLOSING_TOP1 ({top_closing!r} @ {top_closing_n}/{total}={top_closing_n/total:.2%})"
        )
    if unique_prefix_ratio < THRESHOLD_UNIQUE_PREFIX_RATIO:
        verdicts.append(
            f"FAIL_PREFIX_DIVERSITY (unique/total={unique_prefix_ratio:.2%} < {THRESHOLD_UNIQUE_PREFIX_RATIO:.0%})"
        )

    return {
        "path": str(path),
        "total": total,
        "empty_skipped": empty,
        "prefix_top1": {"text": top_prefix, "count": top_prefix_n, "ratio": round(top_prefix_n / total, 4)},
        "closing_top1": {"text": top_closing, "count": top_closing_n, "ratio": round(top_closing_n / total, 4)},
        "unique_prefixes": len(prefixes),
        "unique_prefix_ratio": round(unique_prefix_ratio, 4),
        "prefix_top15": prefixes.most_common(15),
        "closing_top10": closings.most_common(10),
        "verdict": "PASS" if not verdicts else "FAIL",
        "verdicts": verdicts,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", type=Path)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    results = [audit(p) for p in args.paths]
    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
        sys.exit(0)

    for r in results:
        print(f"\n=== {r['path']} ===")
        print(f"  rows: {r['total']} (empty skipped: {r.get('empty_skipped',0)})")
        print(
            f"  prefix top-1: {r['prefix_top1']['text']!r} @ "
            f"{r['prefix_top1']['count']}/{r['total']} = {r['prefix_top1']['ratio']*100:.2f}%  "
            f"(threshold ≤ {THRESHOLD_PREFIX_TOP1*100:.0f}%)"
        )
        print(
            f"  closing top-1: {r['closing_top1']['text']!r} @ "
            f"{r['closing_top1']['count']}/{r['total']} = {r['closing_top1']['ratio']*100:.2f}%  "
            f"(threshold ≤ {THRESHOLD_CLOSING_TOP1*100:.0f}%)"
        )
        print(
            f"  unique prefixes: {r['unique_prefixes']} / {r['total']} = "
            f"{r['unique_prefix_ratio']*100:.2f}%  (threshold ≥ {THRESHOLD_UNIQUE_PREFIX_RATIO*100:.0f}%)"
        )
        print(f"  verdict: {r['verdict']}")
        for v in r.get("verdicts", []):
            print(f"    - {v}")
    # non-zero exit if any FAIL
    sys.exit(0 if all(r["verdict"] == "PASS" for r in results) else 1)


if __name__ == "__main__":
    main()
