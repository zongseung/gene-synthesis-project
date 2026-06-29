"""core_factsheet.yaml draft 자동 생성.

출처 위계 (ver4 §2.1):
  1순위: data/raw/mediclassics_unified/book_*/vol_01.jsonl 서문·범례 regex
  2순위: data/stats/mediclassics_book_list.json raw_text 파싱
  3순위: 권별 수기 보강 (merge 시 기존 entry 보존)
  금지:  LLM 자유 생성

각 필드에 _source 태그를 붙여 추후 사람 검수 대상을 구별 가능.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import yaml

# ─────────────────────────────────────────────────────────
# 2순위 메타 파서
# ─────────────────────────────────────────────────────────

# raw_text 예시:
#   "광제비급  廣濟秘笈 ( 李景華 ) / 1790 ( 1701~1800 ) 한국어 …"
#   "사의경험방  四醫經驗方 이석간, 채득기, 박렴, 허임 ( 李碩幹, 蔡得己, 朴濂, 許任 ) / 미상 ( 1601~1700 ) …"
AUTHOR_PATTERN = re.compile(
    r"^\s*(?P<title_ko>[^ ]+)\s+(?:&nbsp;)?(?:&nbsp;)?\s*"
    r"(?P<title_zh>[一-龥ㆍ·・\s]+?)\s+"
    r"(?:(?P<author_ko>[^()/]+?)\s+)?"
    r"\(\s*(?P<author_zh>[^)]*)\s*\)\s*"
    r"/\s*(?P<year_raw>[^()]+?)\s*"
    r"\(\s*(?P<period>[^)]+?)\s*\)",
    re.UNICODE,
)

YEAR_SINGLE_RE = re.compile(r"^\s*(\d{3,4})\s*년?\s*$")
YEAR_RANGE_RE = re.compile(r"^\s*(\d{3,4})\s*[~～\-–—]\s*(\d{3,4})\s*년?\s*$")


def parse_raw_text(raw: str) -> dict:
    """raw_text에서 author/year 추출. 실패 필드는 None."""
    raw = raw.replace("&nbsp;", " ").replace("\xa0", " ")
    raw = re.sub(r"\s+", " ", raw).strip()
    m = AUTHOR_PATTERN.search(raw)
    out = {
        "author_ko": None,
        "author_hanja": None,
        "year_raw": None,
        "period_raw": None,
    }
    if not m:
        return out
    author_ko = (m.group("author_ko") or "").strip().rstrip(",").strip() or None
    author_zh = (m.group("author_zh") or "").strip() or None
    year_raw = (m.group("year_raw") or "").strip()
    period = (m.group("period") or "").strip()
    out["author_ko"] = author_ko
    out["author_hanja"] = author_zh
    out["year_raw"] = year_raw
    out["period_raw"] = period
    return out


def resolve_year(year_raw: str | None) -> tuple[int | None, int | None]:
    """(compiled_year, published_year). 단일 연도면 published만, 불명이면 둘 다 null."""
    if not year_raw:
        return None, None
    if year_raw in ("미상", "불명", "미상년"):
        return None, None
    m = YEAR_SINGLE_RE.match(year_raw)
    if m:
        y = int(m.group(1))
        return None, y
    m = YEAR_RANGE_RE.match(year_raw)
    if m:
        # range → dynasty 추론만 목적. compiled/published 둘 다 null로 두어
        # 잘못된 reign 유추를 방지한다.
        return None, None
    return None, None


def period_midpoint(period_raw: str | None) -> int | None:
    """'1601~1700' 같은 세기 범위의 중앙값 (dynasty 추론 전용)."""
    if not period_raw:
        return None
    m = YEAR_RANGE_RE.match(period_raw)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        return (a + b) // 2
    m = YEAR_SINGLE_RE.match(period_raw)
    if m:
        return int(m.group(1))
    return None


# ─────────────────────────────────────────────────────────
# Dynasty / reign 유추 (year → dynasty + Joseon reign)
# ─────────────────────────────────────────────────────────

DYNASTY_RANGES = [
    (-1000, 935, "신라", "新羅"),
    (918, 1392, "고려", "高麗"),
    (1392, 1897, "조선", "朝鮮"),
    (1897, 1910, "대한제국", "大韓帝國"),
    (1910, 1945, "일제강점기", None),
]

JOSEON_REIGNS = [
    (1392, 1398, "태조", "太祖"),
    (1398, 1400, "정종", "定宗"),
    (1400, 1418, "태종", "太宗"),
    (1418, 1450, "세종", "世宗"),
    (1450, 1452, "문종", "文宗"),
    (1452, 1455, "단종", "端宗"),
    (1455, 1468, "세조", "世祖"),
    (1468, 1469, "예종", "睿宗"),
    (1469, 1494, "성종", "成宗"),
    (1494, 1506, "연산군", "燕山君"),
    (1506, 1544, "중종", "中宗"),
    (1544, 1545, "인종", "仁宗"),
    (1545, 1567, "명종", "明宗"),
    (1567, 1608, "선조", "宣祖"),
    (1608, 1623, "광해군", "光海君"),
    (1623, 1649, "인조", "仁祖"),
    (1649, 1659, "효종", "孝宗"),
    (1659, 1674, "현종", "顯宗"),
    (1674, 1720, "숙종", "肅宗"),
    (1720, 1724, "경종", "景宗"),
    (1724, 1776, "영조", "英祖"),
    (1776, 1800, "정조", "正祖"),
    (1800, 1834, "순조", "純祖"),
    (1834, 1849, "헌종", "憲宗"),
    (1849, 1863, "철종", "哲宗"),
    (1863, 1907, "고종", "高宗"),
    (1907, 1910, "순종", "純宗"),
]


def dynasty_for_year(year: int | None) -> tuple[str | None, str | None]:
    if year is None:
        return None, None
    for start, end, ko, hanja in DYNASTY_RANGES:
        if start <= year < end:
            return ko, hanja
    return None, None


def joseon_reign_for_year(year: int | None) -> tuple[str | None, str | None]:
    if year is None:
        return None, None
    for start, end, ko, hanja in JOSEON_REIGNS:
        if start <= year < end:
            return ko, hanja
    return None, None


# ─────────────────────────────────────────────────────────
# 1순위: 원문 서문 regex
# ─────────────────────────────────────────────────────────

# 중국 연호 → 서기 매핑 (조선/명·청 연호만 최소 셋)
# preface에 "萬曆三十九年" 같은 표현이 있으면 reign 확증 가능.
ERA_MAPS = {
    "嘉靖": (1522, 1566),   # 명 세종 가정
    "萬曆": (1573, 1620),   # 명 신종 만력 → 조선 선조/광해군
    "崇禎": (1628, 1644),   # 명 의종 숭정
    "順治": (1644, 1661),   # 청 순치
    "康熙": (1662, 1722),   # 청 강희
    "雍正": (1723, 1735),   # 청 옹정
    "乾隆": (1736, 1795),   # 청 건륭
    "嘉慶": (1796, 1820),   # 청 가경
    "道光": (1821, 1850),   # 청 도광
    "咸豊": (1851, 1861),   # 청 함풍
    "同治": (1862, 1874),   # 청 동치
    "光緖": (1875, 1908),   # 청 광서
    "宣德": (1426, 1435),   # 명 선덕 → 세종 치세
    "正統": (1436, 1449),   # 명 정통
    "景泰": (1450, 1456),
    "天順": (1457, 1464),
    "成化": (1465, 1487),
    "弘治": (1488, 1505),
    "正德": (1506, 1521),
    "隆慶": (1567, 1572),
    "泰昌": (1620, 1620),
    "天啓": (1621, 1627),
    "洪武": (1368, 1398),
    "建文": (1399, 1402),
    "永樂": (1403, 1424),
    "洪熙": (1425, 1425),
    "景德": (1004, 1007),
}

ERA_RE = re.compile(r"(" + "|".join(ERA_MAPS.keys()) + r")")
JOSEON_REIGN_HANJA = {h for _, _, _, h in JOSEON_REIGNS if h}
REIGN_HANJA_RE = re.compile(r"(" + "|".join(JOSEON_REIGN_HANJA) + r")")


def scan_preface_for_reign(book_dir: Path, max_records: int = 60) -> dict:
    """vol_01.jsonl의 초반 기록에서 연호·왕대 한자 추출."""
    out = {
        "reign_hanja_hits": [],
        "era_hits": [],
        "preface_sample": None,
    }
    vol = book_dir / "vol_01.jsonl"
    if not vol.exists():
        return out
    buf = []
    with open(vol, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_records:
                break
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            o = rec.get("original") or ""
            buf.append(o)
            for m in REIGN_HANJA_RE.finditer(o):
                out["reign_hanja_hits"].append(m.group(1))
            for m in ERA_RE.finditer(o):
                out["era_hits"].append(m.group(1))
    out["preface_sample"] = (buf[0] if buf else "")[:200]
    return out


# ─────────────────────────────────────────────────────────
# 병합
# ─────────────────────────────────────────────────────────

# 중국 원전으로 조선 왕대를 매핑하면 틀리는 책들 (수기 검증)
CHINESE_ORIGIN_BOOKS = {
    100: ("명나라", "明"),    # 醫宗金鑑 — 清 건륭 7년(1742) 간행이나 조선 왕대 매핑 X
    139: ("명나라", "明"),    # 景岳全書 — 張介賓, 명말 1624
}


def build_entry(book_id: int, meta_json: dict | None, raw_dir: Path) -> dict:
    parsed = parse_raw_text(meta_json.get("raw_text", "")) if meta_json else {}
    compiled_y, published_y = resolve_year(parsed.get("year_raw"))
    period_mid = period_midpoint(parsed.get("period_raw"))
    year_for_dynasty = published_y or compiled_y or period_mid
    dyn_ko, dyn_hanja = dynasty_for_year(year_for_dynasty)

    # 중국 원전 override
    if book_id in CHINESE_ORIGIN_BOOKS:
        dyn_ko, dyn_hanja = CHINESE_ORIGIN_BOOKS[book_id]

    preface = scan_preface_for_reign(raw_dir / f"book_{book_id:03d}")

    # reign 결정 (year 우선 정책):
    #  A) year 있음 + 조선 → year로 유추 (preface hit은 consistency check)
    #  B) year 없음 + preface 연호 → 연호 중앙값으로 유추
    #  C) year 없음 + preface 왕대 한자 → 직접 채택 (false positive 감수)
    #  D) year 있음 + non-조선 → reign null
    reign_ko, reign_hanja = None, None
    reign_source = None

    if book_id in CHINESE_ORIGIN_BOOKS:
        pass  # 중국서 → reign 매핑 하지 않음
    elif year_for_dynasty is not None and dyn_ko == "조선" and period_mid != year_for_dynasty:
        # 단일 연도 기반 유추 (period 중앙값은 너무 넓어 제외)
        reign_ko, reign_hanja = joseon_reign_for_year(year_for_dynasty)
        reign_source = "year_derived"
        # consistency: preface가 다른 왕대만 많이 언급하면 confidence 저하
        if preface["reign_hanja_hits"] and reign_hanja:
            dominant = max(set(preface["reign_hanja_hits"]), key=preface["reign_hanja_hits"].count)
            if dominant != reign_hanja and preface["reign_hanja_hits"].count(dominant) >= 3:
                reign_source = "year_derived_preface_mismatch"
    elif year_for_dynasty is None and preface["era_hits"]:
        era = max(set(preface["era_hits"]), key=preface["era_hits"].count)
        era_start, era_end = ERA_MAPS[era]
        mid = (era_start + era_end) // 2
        reign_ko, reign_hanja = joseon_reign_for_year(mid)
        reign_source = f"era_{era}"
        compiled_y = mid
    elif year_for_dynasty is None and preface["reign_hanja_hits"]:
        # 마지막 수단: 서문 빈도 가장 높은 왕대
        most = max(set(preface["reign_hanja_hits"]), key=preface["reign_hanja_hits"].count)
        reign_hanja = most
        for _, _, ko, h in JOSEON_REIGNS:
            if h == most:
                reign_ko = ko
                break
        reign_source = "preface_regex_last_resort"

    entry = {
        "book_id": book_id,
        "title_ko": meta_json.get("title_ko") if meta_json else None,
        "title_hanja": meta_json.get("title_zh") if meta_json else None,
        "author": parsed.get("author_ko"),
        "author_hanja": parsed.get("author_hanja"),
        "dynasty": dyn_ko,
        "dynasty_hanja": dyn_hanja,
        "reign": reign_ko,
        "reign_hanja": reign_hanja,
        "compiled_year": compiled_y,
        "published_year": published_y,
        "genre": None,
        "topics": None,
        "related_books": None,
        "signature_items": None,
        "_source": {
            "title_ko": "kiom_meta" if meta_json and meta_json.get("title_ko") else None,
            "title_hanja": "kiom_meta" if meta_json and meta_json.get("title_zh") else None,
            "author": "kiom_meta_raw_text_regex" if parsed.get("author_ko") else None,
            "author_hanja": "kiom_meta_raw_text_regex" if parsed.get("author_hanja") else None,
            "dynasty": "year_derived" if dyn_ko else None,
            "reign": reign_source,
            "compiled_year": "kiom_meta_year_parse" if compiled_y else None,
            "published_year": "kiom_meta_year_parse" if published_y else None,
        },
        "_confidence": {
            "author": "high" if parsed.get("author_hanja") else "none",
            "reign": "high" if reign_source == "preface_regex" else ("medium" if reign_source else "none"),
            "year": "high" if published_y else ("medium" if compiled_y else "none"),
        },
        "_preface_sample": preface["preface_sample"],
        "_preface_era_hits": sorted(set(preface["era_hits"])) or None,
    }
    return entry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--book-list", type=Path, default=Path("data/stats/mediclassics_book_list.json"))
    ap.add_argument("--raw-dir", type=Path, default=Path("data/raw/mediclassics_unified"))
    ap.add_argument("--existing", type=Path, default=Path("data/facts/core_factsheet.yaml"))
    ap.add_argument("--output", type=Path, default=Path("data/facts/core_factsheet.yaml"))
    ap.add_argument("--trace", type=Path, default=Path("data/facts/_factsheet_build_trace.json"))
    args = ap.parse_args()

    # 기존 entry 보존 (수기 검증 완료된 3권)
    existing = {}
    if args.existing.exists():
        cur = yaml.safe_load(open(args.existing, encoding="utf-8")) or {}
        for e in cur.get("books", []):
            existing[e["book_id"]] = e

    # book_list 인덱스
    bl = json.load(open(args.book_list, encoding="utf-8"))
    meta_index = {b["book_id"]: b for b in bl["books"]}

    # raw crawl된 26권 ID
    crawled_ids = sorted(
        int(d.name.split("_")[1])
        for d in args.raw_dir.iterdir()
        if d.is_dir() and d.name.startswith("book_")
    )

    books_out = []
    trace = {}
    for bid in crawled_ids:
        if bid in existing:
            books_out.append(existing[bid])
            trace[bid] = "existing_preserved"
            continue
        entry = build_entry(bid, meta_index.get(bid), args.raw_dir)
        books_out.append(entry)
        trace[bid] = {
            "author_hanja": entry["author_hanja"],
            "year": entry["published_year"] or entry["compiled_year"],
            "reign": entry["reign_hanja"],
            "reign_source": entry["_source"]["reign"],
        }

    doc = {
        "_schema_version": 2,
        "_note": (
            "1순위: vol_01 서문 regex  /  2순위: kiom mediclassics_book_list  /  "
            "3순위: 수기 검증 (book_8/93/182는 기존 값 유지)  /  금지: LLM 자유생성"
        ),
        "books": sorted(books_out, key=lambda b: b["book_id"]),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        yaml.safe_dump(doc, f, allow_unicode=True, sort_keys=False, width=120)

    with open(args.trace, "w", encoding="utf-8") as f:
        json.dump(trace, f, ensure_ascii=False, indent=2)

    total = len(books_out)
    auto = sum(1 for b in books_out if "_source" in b)
    kept = total - auto
    print(f"factsheet books: {total}  (existing kept={kept}, auto-draft={auto})")
    print(f"output: {args.output}")
    print(f"trace:  {args.trace}")


if __name__ == "__main__":
    main()
