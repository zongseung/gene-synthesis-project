"""book_008 (동의보감) 전용 split 생성 — Phase A' 데이터 수집.

생성물 (data/cpt/ 직접 배치 — preprocess.py 가 {input}/{corpus}.jsonl 읽음):
  data/cpt/book008_bilingual.jsonl             # <ZH>original</ZH>\n<KO>trans_ko</KO>
  data/cpt/book008_ko_only.jsonl               # 한자 괄호 제거된 순한글 번역
  data/cpt/book008_real_facts_identity.jsonl   # 허준 언급 3건 × 20 duplicate
  data/cpt/book008_real_facts_context.jsonl    # 서문·집례·편명·역대의방·총목
  data/cpt/book008_prolog.jsonl                # factsheet 기반 1줄 prolog

스키마:
  {book_id, volume_id, content_seq, content_level, up_path_nm, text, _source, _shard}
"""
from __future__ import annotations
import json
import re
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent.parent
BDIR = ROOT / "data/raw/mediclassics_unified/book_008"
OUT_DIR = ROOT / "data/cpt"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# regex
# -----------------------------------------------------------------------------
STRIP_HANJA_PAREN = re.compile(r'\s*\([一-鿿\s·ㆍ,、]+\)')
STRIP_HANJA_BRACK = re.compile(r'\s*\[[一-鿿\s·ㆍ,、]+\]')
MULTI_SPACE = re.compile(r'[ \t]{2,}')
HANJA = re.compile(r'[一-鿿]')
HANGUL = re.compile(r'[가-힣]')


def strip_ko(text: str) -> str:
    if not text:
        return ""
    t = STRIP_HANJA_PAREN.sub('', text)
    t = STRIP_HANJA_BRACK.sub('', t)
    t = MULTI_SPACE.sub(' ', t)
    return t.strip()


def load_records():
    rows = []
    for v in sorted(BDIR.glob("vol_*.jsonl")):
        with v.open() as f:
            for line in f:
                rows.append(json.loads(line))
    return rows


# -----------------------------------------------------------------------------
# shard builders
# -----------------------------------------------------------------------------
def build_bilingual(rows):
    out = []
    for r in rows:
        orig = r.get("original") or ""
        ko = r.get("trans_ko") or ""
        if orig.strip() and ko.strip():
            text = f"<ZH>{orig.strip()}</ZH>\n<KO>{ko.strip()}</KO>"
            out.append({
                "book_id": r["book_id"], "volume_id": r["volume_id"],
                "content_seq": r["content_seq"], "content_level": r["content_level"],
                "up_path_nm": r.get("up_path_nm"),
                "text": text, "_source": "book_008/bilingual", "_shard": "bilingual",
            })
    return out


def build_ko_only(rows):
    out = []
    for r in rows:
        ko = r.get("trans_ko") or ""
        if not ko.strip():
            continue
        if r["content_level"] == "Z2":  # 목차 제외 (bilingual 에서 학습)
            continue
        stripped = strip_ko(ko)
        if not stripped:
            continue
        h = len(HANJA.findall(stripped)); k = len(HANGUL.findall(stripped))
        if h + k == 0 or h / (h + k) > 0.3:
            continue
        if len(stripped) < 4:
            continue
        out.append({
            "book_id": r["book_id"], "volume_id": r["volume_id"],
            "content_seq": r["content_seq"], "content_level": r["content_level"],
            "up_path_nm": r.get("up_path_nm"),
            "text": stripped, "_source": "book_008/ko_only_stripped", "_shard": "ko_only",
        })
    return out


PREFACE_PATH_RE = re.compile(r'(序|跋|凡例|集例|歷代醫方|總目)')
# 저자 anchor: substring 매칭 noise 최소화 위해 "허준/許浚" 만 사용.
# round_2 실측에서 "인조갑(손톱)" 같은 약재, "음양의 선조" 같은 추상 표현이
# false positive 로 잡히는 문제 확인됨.
AUTHOR_ANCHORS = ('許浚', '허준')


def build_real_facts_split(rows, duplicate_identity: int = 20):
    """identity (허준 언급) 와 context (서문·집례) 로 분리."""
    identity = []
    context = []
    seen = set()

    for r in rows:
        lvl = r["content_level"]
        up = r.get("up_path_nm") or ""
        orig = r.get("original") or ""
        ko = r.get("trans_ko") or ""
        combined = orig + " " + ko
        key = (r["volume_id"], r["content_seq"])
        if key in seen:
            continue

        text_final = strip_ko(ko) if ko.strip() else orig.strip()
        if not text_final or len(text_final) < 3:
            continue

        # A. identity: 허준/許浚 포함 → high-priority, duplicate
        if any(a in combined for a in AUTHOR_ANCHORS):
            seen.add(key)
            rec = {
                "book_id": r["book_id"], "volume_id": r["volume_id"],
                "content_seq": r["content_seq"], "content_level": lvl,
                "up_path_nm": up or None,
                "text": text_final,
                "_source": "book_008/real_facts_identity",
                "_shard": "real_facts_identity",
                "_reason": "author_heojun",
            }
            # duplicate x N for up-sampling (Allen-Zhu Part 3.1 sufficient aug.)
            for _ in range(duplicate_identity):
                identity.append(rec)
            continue

        # B. context: preface_path / OO / AA
        reason = None
        if PREFACE_PATH_RE.search(up):
            reason = "preface_path"
        elif lvl == "OO":
            reason = "title_OO"
        elif lvl == "AA":
            reason = "title_AA"

        if reason:
            seen.add(key)
            context.append({
                "book_id": r["book_id"], "volume_id": r["volume_id"],
                "content_seq": r["content_seq"], "content_level": lvl,
                "up_path_nm": up or None,
                "text": text_final,
                "_source": f"book_008/real_facts_context/{reason}",
                "_shard": "real_facts_context",
                "_reason": reason,
            })

    return identity, context


def build_prolog():
    prolog_text = (
        "[동의보감 (東醫寶鑑)] 허준 편찬. 조선 선조 대 편찬 명령(1596년), "
        "광해군 2년(1610년) 완성, 1613년 간행. 조선 최대의 종합 의서. "
        "내경편·외형편·잡병편·탕액편·침구편 5편 구성."
    )
    return [{
        "book_id": 8, "volume_id": 0, "content_seq": 0, "content_level": "PROLOG",
        "up_path_nm": None, "text": prolog_text,
        "_source": "book_008/prolog/factsheet",
        "_shard": "prolog",
    }]


def save_jsonl(items, path: Path):
    with path.open("w") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def main():
    rows = load_records()
    print(f"입력: {len(rows):,} records from book_008")

    bilingual = build_bilingual(rows)
    ko_only = build_ko_only(rows)
    # duplicate=1 : stage1 dedup 이 hash 기반이라 20× duplicate 가 3건으로
    # 압축됨을 실측. up-sampling 은 mix probability (cpt_trainer --mix) 로 처리.
    identity, context = build_real_facts_split(rows, duplicate_identity=1)
    prolog = build_prolog()

    save_jsonl(bilingual, OUT_DIR / "book008_bilingual.jsonl")
    save_jsonl(ko_only,   OUT_DIR / "book008_ko_only.jsonl")
    save_jsonl(identity,  OUT_DIR / "book008_real_facts_identity.jsonl")
    save_jsonl(context,   OUT_DIR / "book008_real_facts_context.jsonl")
    save_jsonl(prolog,    OUT_DIR / "book008_prolog.jsonl")

    def summary(name, items):
        total_chars = sum(len(it["text"]) for it in items)
        print(f"  {name:<35}: {len(items):>6,} records, {total_chars:>10,} chars")

    summary("book008_bilingual", bilingual)
    summary("book008_ko_only", ko_only)
    summary("book008_real_facts_identity", identity)
    summary("book008_real_facts_context", context)
    summary("book008_prolog", prolog)

    # identity 건수 (unique) 확인
    unique_identity = {(it["volume_id"], it["content_seq"]) for it in identity}
    print(f"\nidentity unique records: {len(unique_identity)} × 20 dup = {len(identity)} entries")
    for it in list(identity)[:3]:
        print(f"  - vol={it['volume_id']} seq={it['content_seq']} level={it['content_level']}")
        print(f"    path: {it['up_path_nm']}")
        print(f"    text: {it['text'][:150]}")

    # context reason 분포
    c = Counter(it.get("_reason", "?") for it in context)
    print(f"\ncontext reason:")
    for k, n in c.most_common():
        print(f"  {k:<20}: {n}")


if __name__ == "__main__":
    main()
