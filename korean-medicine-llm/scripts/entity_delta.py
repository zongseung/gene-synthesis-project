"""Entity frequency snapshot/diff for HanMed-LLM raw corpus (EXP-V4-00).

목적:
  resume 크롤 전/후 raw corpus에서 한국 한의학 핵심 entity (저자·왕대·연도)의
  출현 빈도를 측정하여 EXP-V4-00의 `entity_delta` 성공 기준
  (`entity_delta[허준+이제마+세종] ≥ 5`) 평가 및 서문·발문 확보 간접 검증.

사용:
  # 1. 현재 snapshot 캡처 (resume 크롤 진행 중에도 안전, 읽기만)
  .venv/bin/python scripts/entity_delta.py snapshot \
      --label pre_resume_checkpoint \
      --output data/stats/entity_snapshots/pre_resume.json

  # 2. 다른 시점 snapshot (크롤 끝난 뒤)
  .venv/bin/python scripts/entity_delta.py snapshot \
      --label post_resume \
      --output data/stats/entity_snapshots/post_resume.json

  # 3. diff 리포트 생성
  .venv/bin/python scripts/entity_delta.py diff \
      --before data/stats/entity_snapshots/pre_resume.json \
      --after  data/stats/entity_snapshots/post_resume.json \
      --output data/stats/entity_delta_v4_00.json

스캔 대상 필드:
  raw jsonl (`data/raw/mediclassics_unified/book_*/vol_*.jsonl`)의
  `original` (한자 원문) + `trans_ko` (국역) + `trans_en` (영역)의 문자열 연결.
  `cpt/` 파일은 `text` 필드. 그 외는 모두 건너뜀.

기준 entity (2개 축):
  - 핵심 한국 한의학 인물 3인 (성공 기준 판정용): 허준 / 이제마 / 세종
  - 확장 entity 리스트: 저자·왕대·타인물·연도 — 진단용
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# --- Entity dictionary ---
# 성공 기준: 이 3인의 전후 합산 delta ≥ 5
CORE_SUCCESS_ENTITIES = ["허준", "이제마", "세종"]

# 확장 entity: 진단용 (원문 hanja + 한글 둘 다 체크)
EXTENDED_ENTITIES: dict[str, list[str]] = {
    # 저자 — 한국 한의학 핵심
    "허준": ["허준", "許浚", "許氏"],
    "이제마": ["이제마", "李濟馬"],
    "유효통": ["유효통", "兪孝通"],
    "노중례": ["노중례", "盧重禮"],
    "박윤덕": ["박윤덕", "朴允德"],
    "황도연": ["황도연", "黃度淵"],
    "강명길": ["강명길", "康命吉"],
    "허임": ["허임", "許任"],
    "이시진": ["이시진", "李時珍"],  # 중국 본초강목 — 환각 누출 entity
    "장개빈": ["장개빈", "張介賓"],  # 경악전서
    # 왕대
    "세종": ["세종", "世宗"],
    "선조": ["선조", "宣祖"],
    "광해군": ["광해군", "光海君"],
    "인종": ["인종", "仁宗"],
    "영조": ["영조", "英祖"],
    "정조": ["정조", "正祖"],
    "태조": ["태조", "太祖"],
    "고종": ["고종", "高宗"],
    # 간기 연도 (4자리)
    "1433": ["1433"],
    "1610": ["1610"],
    "1613": ["1613"],
    "1894": ["1894"],
}


# --- Scan ---

def iter_records(path: Path) -> iter:
    """Yield dict records from a jsonl file, tolerating bad lines."""
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def extract_text(rec: dict) -> str:
    """Build searchable text blob from a record."""
    parts = []
    for key in ("original", "trans_ko", "trans_en", "annotation", "text"):
        val = rec.get(key)
        if isinstance(val, str):
            parts.append(val)
    return "\n".join(parts)


def count_in_blob(blob: str, variants: list[str]) -> int:
    """Total string occurrences of any variant (case-sensitive substring)."""
    total = 0
    for v in variants:
        if v:
            total += blob.count(v)
    return total


def collect_books(scan_root: Path, pattern: str) -> list[Path]:
    """Find book directories or jsonl files by pattern."""
    hits = sorted(scan_root.rglob(pattern))
    # filter log directories
    hits = [h for h in hits if "/logs/" not in str(h) and h.is_file()]
    return hits


def scan_corpus(scan_root: Path, pattern: str = "vol_*.jsonl") -> dict:
    """Scan all matched jsonl, return per-book per-entity counts."""
    files = collect_books(scan_root, pattern)
    per_book_records: dict[str, int] = defaultdict(int)
    per_book_files: dict[str, int] = defaultdict(int)
    per_entity_total: dict[str, int] = defaultdict(int)
    per_entity_per_book: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    per_entity_records_with_mention: dict[str, int] = defaultdict(int)

    t0 = time.time()
    total_records = 0

    for f in files:
        # book_id inferred from path: .../book_XXX/vol_YY.jsonl
        book_dir = f.parent.name
        per_book_files[book_dir] += 1

        for rec in iter_records(f):
            total_records += 1
            per_book_records[book_dir] += 1
            blob = extract_text(rec)
            if not blob:
                continue
            for canon, variants in EXTENDED_ENTITIES.items():
                c = count_in_blob(blob, variants)
                if c:
                    per_entity_total[canon] += c
                    per_entity_per_book[canon][book_dir] += c
                    per_entity_records_with_mention[canon] += 1

    elapsed = time.time() - t0
    return {
        "scan_root": str(scan_root),
        "pattern": pattern,
        "files_scanned": len(files),
        "total_records_scanned": total_records,
        "per_book_records": dict(per_book_records),
        "per_book_files": dict(per_book_files),
        "per_entity_mentions": dict(per_entity_total),
        "per_entity_records_with_mention": dict(per_entity_records_with_mention),
        "per_entity_per_book_mentions": {k: dict(v) for k, v in per_entity_per_book.items()},
        "scan_elapsed_sec": round(elapsed, 2),
    }


# --- Commands ---

def cmd_snapshot(args) -> int:
    scan_root = Path(args.scan_root).resolve()
    if not scan_root.exists():
        print(f"[ERROR] scan_root not found: {scan_root}", file=sys.stderr)
        return 2

    print(f"=== entity_delta snapshot ===", flush=True)
    print(f"  scan_root : {scan_root}", flush=True)
    print(f"  pattern   : {args.pattern}", flush=True)
    print(f"  label     : {args.label}", flush=True)

    data = scan_corpus(scan_root, args.pattern)
    data["run_id"] = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    data["label"] = args.label

    # summary print
    print(f"\nfiles: {data['files_scanned']} | records: {data['total_records_scanned']:,} | "
          f"elapsed: {data['scan_elapsed_sec']}s", flush=True)
    print("\n--- core success entities (허준 / 이제마 / 세종) ---", flush=True)
    for e in CORE_SUCCESS_ENTITIES:
        m = data["per_entity_mentions"].get(e, 0)
        print(f"  {e:<10} mentions={m:>6}", flush=True)

    print("\n--- extended entity report ---", flush=True)
    for e in sorted(data["per_entity_mentions"].keys(), key=lambda k: -data["per_entity_mentions"][k]):
        m = data["per_entity_mentions"][e]
        r = data["per_entity_records_with_mention"][e]
        print(f"  {e:<10} mentions={m:>6} records={r:>5}", flush=True)

    out = Path(args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\n=== snapshot saved → {out}", flush=True)
    return 0


def cmd_diff(args) -> int:
    before = json.loads(Path(args.before).read_text())
    after = json.loads(Path(args.after).read_text())

    per_entity_delta = {}
    per_entity_per_book_delta: dict[str, dict[str, int]] = {}

    all_entities = set(before.get("per_entity_mentions", {}).keys()) | set(after.get("per_entity_mentions", {}).keys())
    for e in sorted(all_entities):
        b_total = before["per_entity_mentions"].get(e, 0)
        a_total = after["per_entity_mentions"].get(e, 0)
        per_entity_delta[e] = {"before": b_total, "after": a_total, "delta": a_total - b_total}

        b_pb = before.get("per_entity_per_book_mentions", {}).get(e, {})
        a_pb = after.get("per_entity_per_book_mentions", {}).get(e, {})
        all_books = set(b_pb) | set(a_pb)
        book_deltas = {b: a_pb.get(b, 0) - b_pb.get(b, 0) for b in all_books}
        per_entity_per_book_delta[e] = {b: d for b, d in book_deltas.items() if d != 0}

    b_rec = before.get("total_records_scanned", 0)
    a_rec = after.get("total_records_scanned", 0)

    core_before = sum(before["per_entity_mentions"].get(e, 0) for e in CORE_SUCCESS_ENTITIES)
    core_after = sum(after["per_entity_mentions"].get(e, 0) for e in CORE_SUCCESS_ENTITIES)
    core_delta = core_after - core_before
    success = core_delta >= 5

    report = {
        "before": {"run_id": before.get("run_id"), "label": before.get("label"), "records": b_rec},
        "after": {"run_id": after.get("run_id"), "label": after.get("label"), "records": a_rec},
        "records_delta": a_rec - b_rec,
        "per_entity_delta": per_entity_delta,
        "per_entity_per_book_delta": per_entity_per_book_delta,
        "core_success": {
            "entities": CORE_SUCCESS_ENTITIES,
            "before_sum": core_before,
            "after_sum": core_after,
            "delta": core_delta,
            "threshold": 5,
            "met": success,
        },
    }

    print(f"=== entity_delta diff ===", flush=True)
    print(f"  before: {report['before']}", flush=True)
    print(f"  after : {report['after']}", flush=True)
    print(f"  records Δ: {report['records_delta']:+,}", flush=True)
    print("\n--- core success (허준+이제마+세종) ---", flush=True)
    cs = report["core_success"]
    marker = "✅" if cs["met"] else "❌"
    print(f"  before_sum={cs['before_sum']} after_sum={cs['after_sum']} Δ={cs['delta']:+d}  threshold≥{cs['threshold']}  {marker}", flush=True)
    print("\n--- per entity Δ (sorted by |Δ|) ---", flush=True)
    sorted_e = sorted(per_entity_delta.items(), key=lambda kv: -abs(kv[1]["delta"]))
    for e, d in sorted_e:
        if d["delta"] == 0:
            continue
        print(f"  {e:<10} {d['before']:>6} → {d['after']:>6}  Δ={d['delta']:+d}", flush=True)

    out = Path(args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n=== diff saved → {out}", flush=True)
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    sp_snap = sub.add_parser("snapshot", help="scan corpus and write entity counts to json")
    sp_snap.add_argument("--scan-root", default=str(REPO_ROOT / "data" / "raw" / "mediclassics_unified"))
    sp_snap.add_argument("--pattern", default="vol_*.jsonl")
    sp_snap.add_argument("--label", required=True, help="human-readable label (e.g. 'pre_resume', 'post_resume')")
    sp_snap.add_argument("--output", required=True)
    sp_snap.set_defaults(func=cmd_snapshot)

    sp_diff = sub.add_parser("diff", help="compare two snapshots and produce delta report")
    sp_diff.add_argument("--before", required=True)
    sp_diff.add_argument("--after", required=True)
    sp_diff.add_argument("--output", required=True)
    sp_diff.set_defaults(func=cmd_diff)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
