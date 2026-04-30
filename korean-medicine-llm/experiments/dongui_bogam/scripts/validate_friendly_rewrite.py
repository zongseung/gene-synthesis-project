"""ver8.2 § 4.4 — LLM rewrite 답변의 fact drift 검증.

검증 차원 (모두 통과해야 retention):
  1. **Length sanity** — rewrite / original 비율이 [min_ratio, max_ratio] 안
  2. **Source excerpt preservation** — question 의 `원문 발췌` 가 rewrite 에 보존
  3. **Hanja preservation** — 원문 발췌·원답변의 한자 token (≥2자) 이 rewrite 에 보존
  4. **Entity whitelist** — entity_whitelist_v6.yaml 토큰이 rewrite 에 보존
  5. **Citation marker** — `[N]` 출처 표시 보존

fail row 는 drop. gold 는 검증 없이 통과 (수작업이라 신뢰).

산출:
  - output (jsonl): gold + 검증 통과 rewrite 합본
  - report (json): retention rate, failure reason 통계

사용:
  .venv/bin/python experiments/dongui_bogam/scripts/validate_friendly_rewrite.py \\
      --input data/sft/friendly_rewrite_v0.jsonl \\
      --gold  data/sft/friendly_gold_v0.jsonl \\
      --output data/sft/friendly_qa_v0.jsonl \\
      --report data/sft/friendly_qa_v0.validation.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]

HANJA_RE = re.compile(r"[一-鿿]+")
CITATION_RE = re.compile(r"\[\d+\]")
EXCERPT_RE = re.compile(r"원문 발췌:\s*(.+)\s*$", re.S)


def extract_source_excerpt(question: str) -> str:
    m = EXCERPT_RE.search(question or "")
    if not m:
        return ""
    return m.group(1).strip()


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def contains_preserving_ws(haystack: str, needle: str) -> bool:
    if not needle:
        return True
    return normalize_ws(needle) in normalize_ws(haystack)


def load_whitelist(path: Path | None) -> set[str]:
    if not path or not path.exists():
        return set()
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    tokens: set[str] = set()

    def _collect(node):
        if isinstance(node, list):
            for x in node:
                _collect(x)
        elif isinstance(node, dict):
            for v in node.values():
                _collect(v)
        elif isinstance(node, str):
            if 2 <= len(node) <= 30:  # 너무 짧거나 긴 토큰 제외
                tokens.add(node)

    _collect(cfg)
    return tokens


def check_row(row: dict, whitelist: set[str], min_ratio: float,
              max_ratio: float, max_hanja_loss: int = 2,
              max_entity_loss: int = 1,
              require_excerpt: bool = True) -> tuple[bool, list[str]]:
    """row 의 rewrite 가 검증 통과인지. (passed, fail_reasons)"""
    rewritten = row.get("assistant", "")
    original = row.get("_source_assistant") or row.get("source_assistant") or ""
    source_question = row.get("_source_question") or row.get("question") or ""
    source_excerpt = (
        row.get("_source_excerpt")
        or row.get("source_excerpt")
        or extract_source_excerpt(source_question)
    )

    if not original:
        # _source_assistant 없으면 검증 불가 — fail 처리 (rewrite 출처 불명)
        return False, ["no_source_assistant"]

    reasons: list[str] = []

    # 1. length sanity
    length_ref = original
    if source_excerpt and source_excerpt not in original:
        length_ref = f"{original}\n{source_excerpt}"
    ratio = len(rewritten) / max(len(length_ref), 1)
    if ratio < min_ratio:
        reasons.append(f"too_short(ratio={ratio:.2f})")
    elif ratio > max_ratio:
        reasons.append(f"too_long(ratio={ratio:.2f})")

    # 2. source excerpt preservation
    if require_excerpt and source_excerpt and not contains_preserving_ws(rewritten, source_excerpt):
        reasons.append("source_excerpt_lost")

    # 3. hanja preservation (≥ 2자 token 만 critical)
    source_text_parts = [source_excerpt, original]
    if not source_excerpt:
        source_text_parts.insert(0, source_question)
    source_text = "\n".join(x for x in source_text_parts if x)
    orig_hanja = {h for h in HANJA_RE.findall(source_text) if len(h) >= 2}
    # rewrite 의 hanja 가 orig 의 substring 으로 들어있어도 OK
    missing_hanja = {h for h in orig_hanja if h not in rewritten}
    if len(missing_hanja) > max_hanja_loss:
        reasons.append(f"hanja_lost({len(missing_hanja)})")

    # 4. entity whitelist
    if whitelist:
        orig_entities = {e for e in whitelist if e in source_text}
        missing_entities = {e for e in orig_entities if e not in rewritten}
        if len(missing_entities) > max_entity_loss:
            reasons.append(f"entity_lost({len(missing_entities)})")

    # 5. citation marker
    if CITATION_RE.search(original) and not CITATION_RE.search(rewritten):
        reasons.append("citation_marker_lost")

    return len(reasons) == 0, reasons


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True,
                    help="rewrite 산출 jsonl (assistant 변환된)")
    ap.add_argument("--gold", type=Path, default=None,
                    help="gold seeds — 검증 없이 통과")
    ap.add_argument("--whitelist", type=Path,
                    default=ROOT / "data" / "sft" / "entity_whitelist_v6.yaml")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--min-ratio", type=float, default=1.0,
                    help="rewrite 길이 / original 의 최소 비율")
    ap.add_argument("--max-ratio", type=float, default=3.0,
                    help="최대 비율 (폭주 방지)")
    ap.add_argument("--max-hanja-loss", type=int, default=2)
    ap.add_argument("--max-entity-loss", type=int, default=1)
    ap.add_argument("--require-excerpt", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="question 의 원문 발췌가 rewrite 에 보존되어야 함")
    args = ap.parse_args()

    whitelist = load_whitelist(args.whitelist)
    print(f"[whitelist] {args.whitelist.name}: {len(whitelist)} tokens")

    if not args.input.exists():
        raise SystemExit(f"input 없음: {args.input}")

    rows_in = [json.loads(l) for l in args.input.open()]
    print(f"[input] {args.input.name}: {len(rows_in)} rows")

    passed: list[dict] = []
    failed: list[dict] = []
    failure_counts: dict[str, int] = {}
    for r in rows_in:
        ok, reasons = check_row(r, whitelist, args.min_ratio, args.max_ratio,
                                args.max_hanja_loss, args.max_entity_loss,
                                args.require_excerpt)
        if ok:
            passed.append(r)
        else:
            failed.append({"id": r.get("id"), "reasons": reasons})
            for reason in reasons:
                key = reason.split("(")[0]
                failure_counts[key] = failure_counts.get(key, 0) + 1

    rewrite_passed = len(passed)
    gold_count = 0
    if args.gold and args.gold.exists():
        gold = [json.loads(l) for l in args.gold.open()]
        for r in gold:
            r.setdefault("tone", "friendly")
            r.setdefault("_origin", "manual_gold_v0")
        passed.extend(gold)
        gold_count = len(gold)
        print(f"[gold] {args.gold.name}: {gold_count} rows (검증 없이 통과)")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        for r in passed:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    report = {
        "schema_version": "1.0",
        "input_path": str(args.input),
        "input_rows": len(rows_in),
        "gold_rows": gold_count,
        "rewrite_passed": rewrite_passed,
        "rewrite_failed": len(failed),
        "retention_rate": rewrite_passed / max(len(rows_in), 1),
        "failure_counts": failure_counts,
        "output_total": len(passed),
        "thresholds": {
            "min_ratio": args.min_ratio,
            "max_ratio": args.max_ratio,
            "max_hanja_loss": args.max_hanja_loss,
            "max_entity_loss": args.max_entity_loss,
            "require_excerpt": args.require_excerpt,
        },
        "failed_sample": failed[:20],  # 디버깅 용 sample
    }
    with args.report.open("w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n[result]")
    print(f"  rewrite passed: {rewrite_passed} / {len(rows_in)} "
          f"= {report['retention_rate']:.1%}")
    print(f"  + gold:         {gold_count}")
    print(f"  output total:   {len(passed)}")
    print(f"  failure reasons:")
    for k, v in sorted(failure_counts.items(), key=lambda x: -x[1]):
        print(f"    {v:>5}  {k}")
    print(f"\n✓ output: {args.output}")
    print(f"✓ report: {args.report}")


if __name__ == "__main__":
    main()
