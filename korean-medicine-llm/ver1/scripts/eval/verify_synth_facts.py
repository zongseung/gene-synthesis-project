"""ver4 r2 §5.4 — hanmed_synth_facts.jsonl 검증 / §6 EXP-V4-06 성공 기준 리포트.

측정 지표:
  - n_triples_total            (target ≥ 150)
  - entity_validation_pass_rate (target ≥ 0.98, else exit 1)
  - paraphrase_diversity        (카테고리 내 trigram jaccard 중앙값, target ≤ 0.3)
  - bind_density                (저자-편찬 binding regex, baseline ratio 제공 target ≥ 14)
  - n_characters_per_book       (권당 paragraph 수 + 평균 길이)
  - hanja_ratio_mean            (bilingual quality filter hint)

실패:
  entity validation pass rate < 0.98 → exit 1
  trigram jaccard median > 0.3 → warning
  bind_density ratio < 14 → warning
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.builder.extract_corpora import HANJA_RE, load_factsheet  # noqa: E402
from src.data.synth.expand_facts import (  # noqa: E402
    build_blacklist,
    validate_text,
)

# "X는 Y가 편찬" 계열 binding regex — spec §1 bind_density 정의.
import re  # noqa: E402

BIND_RE = re.compile(
    r"(허준|이제마|세종|유효통|노중례|박윤덕|이규준|이시진).{0,15}"
    r"(편찬|저술|지음|집필|완성|편집|간행)"
)


def _trigrams(text: str) -> set[str]:
    # 공백 제거 후 문자 3-gram (한국어에 적합)
    s = re.sub(r"\s+", "", text)
    return {s[i : i + 3] for i in range(len(s) - 2)} if len(s) >= 3 else set()


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _load_jsonl(path: Path) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def compute_paraphrase_diversity(records: list[dict], max_pairs_per_cat: int = 500) -> dict:
    """카테고리 × book_id 그룹 내 paragraph 쌍의 trigram jaccard 중앙값.
    큰 그룹은 max_pairs_per_cat 쌍으로 샘플링(결정성 위해 앞에서).
    """
    groups: dict[tuple[str, int], list[str]] = defaultdict(list)
    for r in records:
        key = (r.get("category", "?"), r.get("book_id"))
        groups[key].append(r.get("text", ""))

    all_jaccards: list[float] = []
    per_group: dict[str, float] = {}
    for (cat, bid), texts in groups.items():
        if len(texts) < 2:
            continue
        tris = [_trigrams(t) for t in texts]
        pair_js: list[float] = []
        count = 0
        for i in range(len(tris)):
            for j in range(i + 1, len(tris)):
                pair_js.append(_jaccard(tris[i], tris[j]))
                count += 1
                if count >= max_pairs_per_cat:
                    break
            if count >= max_pairs_per_cat:
                break
        if pair_js:
            median = statistics.median(pair_js)
            per_group[f"{cat}|book_{bid}"] = round(median, 4)
            all_jaccards.extend(pair_js)
    overall = statistics.median(all_jaccards) if all_jaccards else 0.0
    return {"overall_median": round(overall, 4), "per_group_median": per_group}


def count_bind_matches(records: list[dict]) -> int:
    return sum(len(BIND_RE.findall(r.get("text", ""))) for r in records)


def hanja_ratio(text: str) -> float:
    if not text:
        return 0.0
    return len(HANJA_RE.findall(text)) / len(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synth", type=Path, required=True)
    parser.add_argument("--factsheet", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=False, help="bind_density ratio 계산용")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "data/stats/synth_facts_verify.json",
    )
    args = parser.parse_args()

    if not args.synth.exists():
        print(f"[ERROR] synth 파일 없음: {args.synth}", file=sys.stderr)
        sys.exit(2)

    records = _load_jsonl(args.synth)
    factsheet = load_factsheet(args.factsheet)
    blacklist = build_blacklist(factsheet)

    # 1. n_triples_total
    n_triples = len(records)

    # 2. entity validation pass rate — blacklist 재검증
    pass_count = 0
    for r in records:
        ok, _ = validate_text(r.get("text", ""), blacklist)
        if ok:
            pass_count += 1
    pass_rate = pass_count / n_triples if n_triples else 0.0

    # 3. paraphrase diversity (trigram jaccard median)
    paraphrase_div = compute_paraphrase_diversity(records)

    # 4. bind_density
    synth_bind = count_bind_matches(records)
    baseline_bind = None
    bind_ratio = None
    if args.baseline and args.baseline.exists():
        baseline_recs = _load_jsonl(args.baseline)
        baseline_bind = count_bind_matches(baseline_recs)
        if baseline_bind and baseline_bind > 0:
            bind_ratio = round(synth_bind / baseline_bind, 3)
        else:
            # baseline 에 bind 매치가 0 이면 synth 자체 수치로 보고 ratio=inf
            bind_ratio = None

    # 5. per-book stats
    per_book: dict[int, dict] = defaultdict(lambda: {"n_paragraphs": 0, "chars_sum": 0})
    for r in records:
        bid = r.get("book_id")
        per_book[bid]["n_paragraphs"] += 1
        per_book[bid]["chars_sum"] += r.get("n_chars", len(r.get("text", "")))
    per_book_out = {
        str(bid): {
            "n_paragraphs": v["n_paragraphs"],
            "avg_chars": round(v["chars_sum"] / v["n_paragraphs"], 2) if v["n_paragraphs"] else 0,
        }
        for bid, v in sorted(per_book.items())
    }

    # 6. hanja ratio mean
    hj_ratios = [hanja_ratio(r.get("text", "")) for r in records] if records else [0.0]
    hanja_ratio_mean = round(sum(hj_ratios) / len(hj_ratios), 4) if hj_ratios else 0.0

    # category distribution
    cat_dist: dict[str, int] = defaultdict(int)
    for r in records:
        cat_dist[r.get("category", "?")] += 1

    report = {
        "synth_path": str(args.synth),
        "baseline_path": str(args.baseline) if args.baseline else None,
        "n_triples_total": n_triples,
        "entity_validation_pass_rate": round(pass_rate, 4),
        "entity_validation_pass_count": pass_count,
        "paraphrase_diversity": paraphrase_div,
        "bind_density": {
            "synth": synth_bind,
            "baseline": baseline_bind,
            "ratio": bind_ratio,
            "regex": BIND_RE.pattern,
        },
        "n_characters_per_book": per_book_out,
        "category_distribution": dict(cat_dist),
        "hanja_ratio_mean": hanja_ratio_mean,
        "success_criteria_exp_v4_06": {
            "n_triples_total_target": 150,
            "n_triples_total_pass": n_triples >= 150,
            "entity_pass_target": 0.98,
            "entity_pass_pass": pass_rate >= 0.98,
            "jaccard_median_target": 0.3,
            "jaccard_median_pass": paraphrase_div["overall_median"] <= 0.3,
            "bind_ratio_target": 14,
            "bind_ratio_pass": (bind_ratio is not None and bind_ratio >= 14),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # stdout report
    print("=" * 64)
    print(f"synth_facts verify — {args.synth}")
    print("=" * 64)
    print(f"n_triples_total:               {n_triples} (target ≥ 150)")
    print(f"entity_validation_pass_rate:   {pass_rate:.4f}  ({pass_count}/{n_triples}) target ≥ 0.98")
    print(f"paraphrase_diversity(median):  {paraphrase_div['overall_median']:.4f}  target ≤ 0.30")
    print(f"bind_density synth:            {synth_bind}")
    if baseline_bind is not None:
        print(f"bind_density baseline:         {baseline_bind}")
        print(f"bind_density ratio:            {bind_ratio} (target ≥ 14)")
    print(f"hanja_ratio_mean:              {hanja_ratio_mean:.4f}")
    print()
    print("per book:")
    for bid, v in per_book_out.items():
        print(f"  book_{bid}: paragraphs={v['n_paragraphs']}, avg_chars={v['avg_chars']}")
    print("category distribution:")
    for c, n in sorted(cat_dist.items()):
        print(f"  {c}: {n}")
    print()
    print(f"report: {args.output}")

    # warnings / failures
    exit_code = 0
    if pass_rate < 0.98:
        print(f"[FAIL] entity_validation_pass_rate={pass_rate:.4f} < 0.98", file=sys.stderr)
        exit_code = 1
    if paraphrase_div["overall_median"] > 0.3:
        print(
            f"[WARN] trigram jaccard median={paraphrase_div['overall_median']:.4f} > 0.30",
            file=sys.stderr,
        )
    if bind_ratio is not None and bind_ratio < 14:
        print(f"[WARN] bind_density ratio={bind_ratio} < 14", file=sys.stderr)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
