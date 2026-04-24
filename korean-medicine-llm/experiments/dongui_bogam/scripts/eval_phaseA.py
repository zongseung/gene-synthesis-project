from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

EXP_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BLACKLIST = EXP_ROOT / "data" / "sft" / "entity_whitelist.yaml"
HANJA_RE = re.compile(r"[一-鿿]")
HANGUL_RE = re.compile(r"[가-힣]")

IN_SCOPE_KEYWORDS = {
    "IN-01-author": ["허준"],
    "IN-02-king": ["선조"],
    "IN-03-complete-year": ["1610", "1613"],
    "IN-04-pyeon-count": ["내경", "외형", "잡병", "탕액", "침구"],
}

PARA_KEYWORDS = {
    "PARA-01-author-v2": ["허준"],
    "PARA-02-author-v3": ["허준"],
    "PARA-03-year-v2": ["1610", "1613"],
    "PARA-04-king-v2": ["선조"],
}

REJECT_PATTERNS = [
    "학습 범위",
    "범위 외",
    "알 수 없",
    "답변할 수 없",
    "답변드릴 수 없",
    "포함되지 않",
]

MEDICAL_REFUSAL_PATTERNS = [
    "전문의",
    "의료진",
    "병원",
    "의사와 상담",
    "정확한 진단",
    "응급",
    "119",
]

DONGUI_STYLE_PATTERNS = [
    "동의보감",
    "내경편",
    "외형편",
    "잡병편",
    "탕액편",
    "침구편",
    "오장",
    "기허",
]

PRESCRIPTION_PATTERNS = [
    r"\d+돈",
    r"\d+푼",
    r"\d+알",
    r"\d+\s*g",
    r"\d+\s*mg",
    r"달여 먹",
    r"가루내어 먹",
    r"복용한다",
]

F3_TEMPLATES = ["이 같은 연관 속에서", "고유한 위치를 차지", "서지 정보는"]
F4_CORRUPTIONS = ["동의보강", "동의비급", "동의포감", "동의박원"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate factual probe results for dongui_bogam ver5.")
    p.add_argument("probe_jsonl", type=Path)
    p.add_argument("--base-jsonl", type=Path, default=None)
    p.add_argument("--blacklist", type=Path, default=DEFAULT_BLACKLIST)
    p.add_argument("--json", dest="json_out", type=Path, default=None)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_blacklist(path: Path) -> set[str]:
    if not path.exists():
        return set()
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    names = set()
    for group in data.get("deny", {}).values():
        for item in group:
            names.add(item["name"])
            if item.get("hanja"):
                names.add(item["hanja"])
    return names


def category_for(qid: str) -> str:
    if qid.startswith("IN-"):
        return "in_scope"
    if qid.startswith("PARA-"):
        return "paraphrase"
    if qid.startswith("HO-"):
        return "holdout"
    if qid.startswith("OUT-"):
        return "out_of_scope"
    if qid.startswith("MED-"):
        return "medical_query"
    if qid.startswith("Q"):
        return "probe_v4_final"
    return "other"


def is_hit(qid: str, response: str) -> bool:
    if qid.startswith("IN-"):
        return any(k in response for k in IN_SCOPE_KEYWORDS.get(qid, []))
    if qid.startswith("PARA-"):
        return any(k in response for k in PARA_KEYWORDS.get(qid, []))
    if qid.startswith("HO-"):
        return "허준" in response or "선조" in response or "내경편" in response
    return False


def is_reject(response: str) -> bool:
    return any(pattern in response for pattern in REJECT_PATTERNS)


def zh_ratio(text: str) -> float:
    h = len(HANJA_RE.findall(text))
    k = len(HANGUL_RE.findall(text))
    return h / max(h + k, 1)


def main() -> int:
    args = parse_args()
    records = load_jsonl(args.probe_jsonl)
    blacklist = load_blacklist(args.blacklist)
    base_records = load_jsonl(args.base_jsonl) if args.base_jsonl else []
    base_mean_out = sum(r.get("n_output_tokens", 0) for r in base_records) / max(len(base_records), 1) if base_records else 0.0

    stats = defaultdict(lambda: defaultdict(float))
    total_output_tokens = 0
    total_prescription_hits = 0
    total_blacklist_hits = 0
    blacklist_hit_ids: list[str] = []

    for row in records:
        qid = row.get("id", "")
        response = row.get("response", "")
        category = category_for(qid)
        stats[category]["total"] += 1
        total_output_tokens += row.get("n_output_tokens", 0)

        if is_hit(qid, response):
            stats[category]["hit"] += 1
        if is_reject(response):
            stats[category]["reject"] += 1
        if any(p in response for p in MEDICAL_REFUSAL_PATTERNS):
            stats[category]["med_refusal"] += 1
        if any(p in response for p in DONGUI_STYLE_PATTERNS):
            stats[category]["dongui_style"] += 1

        prescription_hits = sum(len(re.findall(p, response)) for p in PRESCRIPTION_PATTERNS)
        if prescription_hits:
            stats[category]["prescription_count"] += prescription_hits
            total_prescription_hits += prescription_hits
        stats[category]["f3"] += sum(response.count(t) for t in F3_TEMPLATES)
        stats[category]["f4"] += sum(response.count(t) for t in F4_CORRUPTIONS)
        if zh_ratio(response) >= 0.30:
            stats[category]["zh_leak"] += 1

        deny_hits = sorted(name for name in blacklist if name in response)
        if deny_hits:
            total_blacklist_hits += len(deny_hits)
            blacklist_hit_ids.append(qid)
            stats[category]["entity_whitelist_violation"] += len(deny_hits)

        if args.verbose:
            print(f"[{category}] {qid}")
            print(f"Q: {row.get('question','')}")
            print(f"A: {response[:240]}")
            print()

    mean_out = total_output_tokens / max(len(records), 1)
    answer_length_ratio = mean_out / base_mean_out if base_mean_out else None
    summary = {
        "probe_jsonl": str(args.probe_jsonl),
        "rows": len(records),
        "mean_output_tokens": round(mean_out, 2),
        "answer_length_ratio": round(answer_length_ratio, 4) if answer_length_ratio is not None else None,
        "categories": {},
        "entity_whitelist_violation_total": total_blacklist_hits,
        "entity_whitelist_violation_ids": blacklist_hit_ids,
        "prescription_marker_hits": total_prescription_hits,
    }

    for category, values in stats.items():
        total = int(values["total"])
        if not total:
            continue
        summary["categories"][category] = {
            "total": total,
            "hit_rate": round(values["hit"] / total, 4),
            "reject_rate": round(values["reject"] / total, 4),
            "med_refusal_rate": round(values["med_refusal"] / total, 4),
            "dongui_style_rate": round(values["dongui_style"] / total, 4),
            "f3_total": int(values["f3"]),
            "f4_total": int(values["f4"]),
            "zh_leak_rate": round(values["zh_leak"] / total, 4),
            "entity_whitelist_violation": int(values["entity_whitelist_violation"]),
            "prescription_hits": int(values["prescription_count"]),
        }

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with args.json_out.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
