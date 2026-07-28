"""
mediclassics_unified raw record → 3개 학습 corpus 파일로 분리.
ver4 r2: book 진입 시 `data/facts/core_factsheet.yaml` 기반 long-form book_meta_prolog 1회 삽입.

입력: data/raw/mediclassics_unified/book_{NNN}/vol_{VV}.jsonl  (한 줄 = 한 record)
출력: data/cpt/
  - hanmed_bilingual.jsonl   # original AND trans_ko 모두 non-null → D2 포맷 블록
  - hanmed_zh_only.jsonl     # 한문만 (trans_ko null이거나 strip 후 빈 경우 포함)
  - hanmed_ko_only.jsonl     # 한국어만
  - hanmed_en_only.jsonl     # 영역만 (옵션)
  - corpus_stats.json        # 책별·전체 stats (record 수, char 수)

D2 bilingual block 포맷 (ver2 §04.5.2):
  <ZH>{original}</ZH>
  <KO>{trans_ko}</KO>
  <빈 줄>

Usage:
  python3 src/data/builder/extract_corpora.py \
    --input data/raw/mediclassics_unified \
    --output data/cpt \
    --factsheet data/facts/core_factsheet.yaml
"""

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

# ver4 r2 §5.5 재현성: set_global_seed 호출 (main 진입부에서 실행).
_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.utils.seed import set_global_seed  # noqa: E402


def normalize_text(s: str | None) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFC", s)
    s = re.sub(r"[ \t\u3000]+", " ", s)  # 다중 공백·전각공백 → 단일
    s = re.sub(r"\r\n?", "\n", s)        # CRLF → LF
    s = s.strip()
    return s


def is_meaningful(s: str, min_chars: int = 1) -> bool:
    return bool(s) and len(s) >= min_chars


def make_bilingual_block(original: str, trans_ko: str) -> str:
    return f"<ZH>{original}</ZH>\n<KO>{trans_ko}</KO>\n\n"


# ---------------------------------------------------------------------------
# book_meta_prolog (ver4 r2, §2.3 / §5.1)
# ---------------------------------------------------------------------------
# 한국어 token 근사: len(text) / 1.7 ≈ tokens. 340자 ≈ 200 tok, 800자 ≈ 470 tok.
# 하한 340자 미만이면 shortcut/짧은 메타줄 위험 → None 반환.
PROLOG_MIN_CHARS = 340
PROLOG_MAX_CHARS = 800

# preprocess.py Stage 1 quality_check("bilingual") 의 `hanja ≥ 0.1` 필터 통과 기준.
# 안전 마진: 임계(0.10) 보다 낮은 0.08 을 failure 기준으로 잡고, 내부 경고 후 bilingual skip.
HANJA_RE = re.compile(r"[\u3400-\u9FFF\uF900-\uFAFF]")
PROLOG_BILINGUAL_HANJA_FAIL_RATIO = 0.08


def _hanja_ratio(text: str) -> float:
    if not text:
        return 0.0
    return len(HANJA_RE.findall(text)) / len(text)


def _extract_hanja_only(s: str | None) -> str | None:
    """'허준(許浚)' → '許浚', '許浚' → '許浚', '허준' → None.

    괄호 안 한자 블록이 있으면 그 블록을, 없고 문자열 자체가 이미 한자 포함이면
    한자 토큰만 추출해 반환. 한자가 하나도 없으면 None.
    """
    if not s:
        return None
    # 괄호 안 한자: 예 `허준(許浚)` 또는 `유효통(兪孝通)·노중례(盧重禮)`
    parens = re.findall(r"\(([^)]*)\)", s)
    hanja_in_parens = [p for p in parens if HANJA_RE.search(p)]
    if hanja_in_parens:
        return "·".join(hanja_in_parens)
    # 문자열 자체가 한자를 포함하면 한자 런을 추출
    runs = re.findall(r"[\u3400-\u9FFF\uF900-\uFAFF]+", s)
    if runs:
        return "·".join(runs)
    return None


def load_factsheet(path: Path) -> dict[int, dict]:
    """core_factsheet.yaml → {book_id: fact_dict}. 없거나 파싱 실패 시 빈 dict."""
    try:
        import yaml  # PyYAML
    except ImportError:
        print("[ERROR] PyYAML 미설치. `.venv/bin/pip install pyyaml` 후 재시도.", file=sys.stderr)
        sys.exit(4)

    if not path.exists():
        print(f"[WARN] factsheet 없음: {path} → book_meta_prolog 비활성화")
        return {}

    with open(path, encoding="utf-8") as f:
        doc = yaml.safe_load(f)

    if not doc or "books" not in doc:
        print(f"[WARN] factsheet 스키마 불일치 ({path}): 'books' 키 없음 → 비활성화")
        return {}

    out: dict[int, dict] = {}
    for entry in doc["books"]:
        bid = entry.get("book_id")
        if isinstance(bid, int):
            out[bid] = entry
    return out


def _fmt_year(y) -> str | None:
    if y is None:
        return None
    try:
        return f"{int(y)}년"
    except (TypeError, ValueError):
        return None


def expand_book_meta_prolog(fact: dict) -> str | None:
    """
    권별 fact dict → 한국어 평서문 long-form paragraph (200~400 token).
    Template 기반만. LLM 호출·자유 생성 금지.
    5 섹션을 자연스러운 서술로 이어붙임 (Q&A·대화체 금지).
    필드가 null이면 해당 섹션 스킵. 최종 길이 < 340자면 None 반환.

    한자 병기 규칙 (ver4 r2.1):
      - fact sheet 에 한자 필드가 있을 때 첫 등장 `한글(漢字)` 한 번만 병기.
      - 이후 재등장은 한글로만 지칭.
      - 한자 하드코딩 금지 (fact sheet 외에서 한자 생성 금지).
      - signature_items 는 이미 한자/한문을 포함하므로 그대로 노출.
    """
    if not fact:
        return None

    title_ko = fact.get("title_ko")
    title_hanja = fact.get("title_hanja")
    author = fact.get("author")  # 이미 '허준(許浚)' 양식일 수 있음
    author_hanja = fact.get("author_hanja") or _extract_hanja_only(author)
    dynasty = fact.get("dynasty")
    reign = fact.get("reign")
    reign_hanja = fact.get("reign_hanja") or _extract_hanja_only(reign)
    compiled_year = _fmt_year(fact.get("compiled_year"))
    published_year = _fmt_year(fact.get("published_year"))
    genre = fact.get("genre")
    topics = fact.get("topics") or []
    related = fact.get("related_books") or []
    sig = fact.get("signature_items") or []

    # 저자 표시용 한글-only 축약 (병기 중복 방지). 예: '허준(許浚)' → '허준'.
    author_ko_only = re.sub(r"\s*\([^)]*\)", "", author).strip() if author else None

    # 첫 등장 병기 양식 / 재등장 양식
    if title_ko and title_hanja:
        title_full = f"『{title_ko}({title_hanja})』"
    elif title_ko:
        title_full = f"『{title_ko}』"
    elif title_hanja:
        title_full = f"『{title_hanja}』"
    else:
        return None
    title_short = f"『{title_ko}』" if title_ko else f"『{title_hanja}』"

    # 저자: 병기본(author_ko_only + author_hanja) vs 재등장(author_ko_only)
    if author_ko_only and author_hanja and author_hanja not in author_ko_only:
        author_first = f"{author_ko_only}({author_hanja})"
    elif author:
        author_first = author  # 이미 병기 포함 또는 한자만
    else:
        author_first = None
    author_rep = author_ko_only or author

    # 왕대: 병기본 vs 재등장
    if reign and reign_hanja and reign_hanja not in reign:
        reign_first = f"{reign}({reign_hanja})"
    else:
        reign_first = reign
    reign_rep = reign

    sentences: list[str] = []

    # 섹션 1: 저자·왕대·연도 (triple binding 반복) — 첫 등장에 한자 병기
    s1_parts: list[str] = []
    lead = f"{title_full}은(는)"
    if author_first and dynasty and reign_first and compiled_year:
        s1_parts.append(
            f"{lead} {dynasty} {reign_first} 때에 어의 {author_first}이(가) 편찬한 의서로, {compiled_year}에 완성되었다."
        )
    elif author_first and dynasty and reign_first:
        s1_parts.append(f"{lead} {dynasty} {reign_first} 때에 {author_first}이(가) 편찬한 의서이다.")
    elif author_first and compiled_year:
        s1_parts.append(f"{lead} {author_first}이(가) {compiled_year}에 편찬한 의서이다.")
    elif author_first:
        s1_parts.append(f"{lead} {author_first}이(가) 편찬한 의서이다.")
    elif dynasty and reign_first and compiled_year:
        s1_parts.append(f"{lead} {dynasty} {reign_first} 때 {compiled_year}에 편찬된 의서이다.")
    elif dynasty and compiled_year:
        s1_parts.append(f"{lead} {dynasty} 시대 {compiled_year}에 편찬된 의서이다.")

    # 반복 바인딩 — 재등장 시 저자·왕대는 한 번 더 한자 병기해 entity anchoring 강화
    if author_rep and compiled_year and published_year and compiled_year != published_year:
        s1_parts.append(
            f"{author_first or author_rep}은(는) {compiled_year} 편찬을 마치고 "
            f"{published_year}에 {title_full}을(를) 간행하였다."
        )
    elif author_rep and published_year:
        s1_parts.append(f"{author_first or author_rep}의 {title_full}은(는) {published_year}에 간행되었다.")
    elif author_rep and compiled_year:
        s1_parts.append(
            f"{author_first or author_rep}의 이름과 {compiled_year}이라는 편찬 시점은 이 책의 핵심 서지 정보이다."
        )

    if dynasty and reign_first and author_rep:
        s1_parts.append(
            f"즉 {dynasty} {reign_first} 대의 의학 활동을 대표하는 인물인 "
            f"{author_first or author_rep}과(와) {title_full}은(는) 함께 기억되어 왔다."
        )

    sentences.extend(s1_parts)

    # 섹션 2: 장르·편찬 배경 — title 재병기로 anchoring
    s2_parts: list[str] = []
    if genre and dynasty:
        s2_parts.append(
            f"{title_full}은(는) {dynasty} 시대 {genre}에 해당하며, "
            "당대 의학 지식을 체계적으로 집성한 성격을 가진다."
        )
    elif genre:
        s2_parts.append(
            f"{title_full}은(는) {genre}에 해당하며, 해당 분야의 지식을 체계적으로 집성한 성격을 가진다."
        )
    if reign_first and compiled_year:
        s2_parts.append(
            f"{reign_first} 대의 정치·사회적 배경 속에서 {compiled_year} 무렵 "
            f"{title_full} 편찬 작업이 이루어졌고, 이는 당시 의학 정책과 긴밀히 연결된다."
        )
    sentences.extend(s2_parts)

    # 섹션 3: 주요 주제 열거 (topics 자체에 한자 병기 포함)
    if topics:
        topic_list = ", ".join(topics)
        sentences.append(
            f"{title_full}이(가) 다루는 핵심 주제는 다음과 같다. {topic_list}. "
            "이 구성은 이 책의 내용 범위를 대표한다."
        )

    # 섹션 4: 관련 서적 (related_books 자체에 한자 병기 포함)
    if related:
        rel_list = ", ".join(related)
        sentences.append(
            f"{title_full}은(는) {rel_list} 등과 내용적·편찬적으로 연관되며, "
            "한국 한의학 문헌 계통 안에서 서로 영향을 주고받았다."
        )

    # 섹션 5: 대표 처방/본초/구절 (signature_items 자체에 한자/한문 원어 포함)
    if sig:
        # 앞 두 항목은 주서술, 나머지는 뒤 문장에 묶음
        head = sig[:2]
        tail = sig[2:]
        head_clause = "와(과) ".join(head) if len(head) > 1 else head[0]
        sentences.append(
            f"대표적인 내용으로는 {head_clause} 등이 잘 알려져 있으며, "
            f"이들은 {title_full}의 학문적 특징을 압축적으로 보여 준다."
        )
        if tail:
            tail_list = ", ".join(tail)
            sentences.append(
                f"이 외에도 {tail_list}이(가) 이 책의 대표 항목으로 함께 거론된다."
            )

    if not sentences:
        return None

    prolog = " ".join(sentences).strip()

    # 공백 정규화
    prolog = re.sub(r"\s+", " ", prolog)

    # 길이 gate
    if len(prolog) < PROLOG_MIN_CHARS:
        return None
    if len(prolog) > PROLOG_MAX_CHARS:
        # upper bound 초과 시 문장 경계에서 잘라냄 (마침표 기준)
        cut = prolog[:PROLOG_MAX_CHARS]
        last_dot = cut.rfind(".")
        if last_dot >= PROLOG_MIN_CHARS:
            prolog = cut[: last_dot + 1]
        else:
            prolog = cut
    return prolog


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="data/raw/mediclassics_unified")
    parser.add_argument("--output", type=Path, required=True, help="data/cpt")
    parser.add_argument("--min-zh-chars", type=int, default=3, help="한문 최소 길이 (필터)")
    parser.add_argument("--min-ko-chars", type=int, default=2, help="한국어 최소 길이 (필터)")
    parser.add_argument("--keep-en", action="store_true", help="영역도 별도 파일로")
    parser.add_argument(
        "--factsheet",
        type=Path,
        default=Path("data/facts/core_factsheet.yaml"),
        help="book_meta_prolog 소스 YAML (ver4 r2)",
    )
    parser.add_argument(
        "--no-prolog",
        action="store_true",
        help="book_meta_prolog 삽입 강제 스킵",
    )
    parser.add_argument("--seed", type=int, default=42, help="전역 재현성 seed (ver4 r2 §5.5)")
    args = parser.parse_args()

    # ver4 r2 §5.5: 재현성 시드 설정 (random/numpy/torch + cuBLAS 결정성)
    set_global_seed(args.seed)

    args.output.mkdir(parents=True, exist_ok=True)

    # factsheet 로드 (없거나 --no-prolog면 빈 dict)
    factsheet: dict[int, dict] = {}
    prolog_source: str | None = None
    if args.no_prolog:
        print("[INFO] --no-prolog 플래그로 book_meta_prolog 비활성화")
    else:
        factsheet = load_factsheet(args.factsheet)
        if factsheet:
            prolog_source = str(args.factsheet)
            print(f"[INFO] factsheet 로드 완료: {len(factsheet)}권, source={prolog_source}")

    out_bi = open(args.output / "hanmed_bilingual.jsonl", "w", encoding="utf-8")
    out_zh = open(args.output / "hanmed_zh_only.jsonl", "w", encoding="utf-8")
    out_ko = open(args.output / "hanmed_ko_only.jsonl", "w", encoding="utf-8")
    out_en = open(args.output / "hanmed_en_only.jsonl", "w", encoding="utf-8") if args.keep_en else None

    stats_per_book: dict[int, dict] = defaultdict(lambda: {
        "records_total": 0,
        "records_zh_only": 0,
        "records_ko_only": 0,
        "records_en_only": 0,
        "records_bilingual": 0,
        "records_filtered_short": 0,
        "chars_zh": 0,
        "chars_ko": 0,
        "chars_en": 0,
        "chars_bilingual_blocks": 0,
        "vol_files": 0,
        "has_meta_prolog": False,
        "prolog_chars": 0,
    })

    # ver4 r2.1: hanja_ratio 미달로 bilingual 파일에서 skip 된 book 기록
    stats_prolog_bilingual_skipped: list[int] = []

    book_dirs = sorted([d for d in args.input.iterdir() if d.is_dir() and d.name.startswith("book_")])
    print(f"발견된 책: {len(book_dirs)}")

    for book_dir in book_dirs:
        try:
            book_id = int(book_dir.name.split("_")[1])
        except ValueError:
            continue
        bs = stats_per_book[book_id]

        # ---- book 진입 훅: book_meta_prolog 1회 삽입 (ver4 r2 §5.1) ----
        fact = factsheet.get(book_id)
        prolog = expand_book_meta_prolog(fact) if fact else None
        if prolog:
            prolog_meta = {
                "book_id": book_id,
                "volume_id": None,
                "content_seq": -1,  # sentinel: dedup 시 주의
                "content_level": None,
                "up_path_nm": "__book_meta_prolog__",
            }
            # ver4 r2.1: bilingual quality filter (hanja ≥ 0.1) 통과 여부 자체 검증.
            # 한자 비율 부족 시 경고 후 bilingual 는 skip 하고 ko_only 만 유지.
            hr = _hanja_ratio(prolog)
            bs["prolog_hanja_ratio"] = round(hr, 4)
            if hr >= PROLOG_BILINGUAL_HANJA_FAIL_RATIO:
                out_bi.write(json.dumps({
                    **prolog_meta,
                    "text": prolog,
                    "n_chars_zh": 0,
                    "n_chars_ko": len(prolog),
                }, ensure_ascii=False) + "\n")
            else:
                print(
                    f"[WARN] book_{book_id} prolog hanja_ratio={hr:.3f} < "
                    f"{PROLOG_BILINGUAL_HANJA_FAIL_RATIO:.2f} → bilingual 파일에서 skip"
                )
                stats_prolog_bilingual_skipped.append(book_id)
            # ko_only 파일: 그대로 (한자 부족해도 한글 비율은 충분)
            out_ko.write(json.dumps({
                **prolog_meta,
                "text": prolog,
                "n_chars": len(prolog),
            }, ensure_ascii=False) + "\n")
            # zh_only에는 삽입 안 함 (한문 corpus 성격 유지)
            bs["has_meta_prolog"] = True
            bs["prolog_chars"] = len(prolog)

        for vol_file in sorted(book_dir.glob("vol_*.jsonl")):
            bs["vol_files"] += 1
            with open(vol_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    bs["records_total"] += 1

                    original = normalize_text(rec.get("original"))
                    trans_ko = normalize_text(rec.get("trans_ko"))
                    trans_en = normalize_text(rec.get("trans_en"))

                    has_zh = is_meaningful(original, args.min_zh_chars)
                    has_ko = is_meaningful(trans_ko, args.min_ko_chars)
                    has_en = is_meaningful(trans_en, 5)

                    if not (has_zh or has_ko):
                        bs["records_filtered_short"] += 1
                        continue

                    base_meta = {
                        "book_id":      rec.get("book_id"),
                        "volume_id":    rec.get("volume_id"),
                        "content_seq":  rec.get("content_seq"),
                        "content_level": rec.get("content_level"),
                        "up_path_nm":   rec.get("up_path_nm"),
                    }

                    if has_zh and has_ko:
                        block = make_bilingual_block(original, trans_ko)
                        out_bi.write(json.dumps({
                            **base_meta,
                            "text": block,
                            "n_chars_zh": len(original),
                            "n_chars_ko": len(trans_ko),
                        }, ensure_ascii=False) + "\n")
                        bs["records_bilingual"] += 1
                        bs["chars_bilingual_blocks"] += len(block)

                    if has_zh:
                        out_zh.write(json.dumps({
                            **base_meta,
                            "text": original,
                            "n_chars": len(original),
                        }, ensure_ascii=False) + "\n")
                        bs["records_zh_only"] += 1
                        bs["chars_zh"] += len(original)

                    if has_ko:
                        out_ko.write(json.dumps({
                            **base_meta,
                            "text": trans_ko,
                            "n_chars": len(trans_ko),
                        }, ensure_ascii=False) + "\n")
                        bs["records_ko_only"] += 1
                        bs["chars_ko"] += len(trans_ko)

                    if out_en and has_en:
                        out_en.write(json.dumps({
                            **base_meta,
                            "text": trans_en,
                            "n_chars": len(trans_en),
                        }, ensure_ascii=False) + "\n")
                        bs["records_en_only"] += 1
                        bs["chars_en"] += len(trans_en)

    out_bi.close()
    out_zh.close()
    out_ko.close()
    if out_en:
        out_en.close()

    totals = {
        "records_total": sum(b["records_total"] for b in stats_per_book.values()),
        "records_bilingual": sum(b["records_bilingual"] for b in stats_per_book.values()),
        "records_zh_only": sum(b["records_zh_only"] for b in stats_per_book.values()),
        "records_ko_only": sum(b["records_ko_only"] for b in stats_per_book.values()),
        "records_en_only": sum(b["records_en_only"] for b in stats_per_book.values()),
        "records_filtered_short": sum(b["records_filtered_short"] for b in stats_per_book.values()),
        "chars_zh": sum(b["chars_zh"] for b in stats_per_book.values()),
        "chars_ko": sum(b["chars_ko"] for b in stats_per_book.values()),
        "chars_en": sum(b["chars_en"] for b in stats_per_book.values()),
        "chars_bilingual_blocks": sum(b["chars_bilingual_blocks"] for b in stats_per_book.values()),
        "books_with_prolog": sum(1 for b in stats_per_book.values() if b["has_meta_prolog"]),
        "prolog_bilingual_skipped": len(stats_prolog_bilingual_skipped),
        "prolog_bilingual_skipped_book_ids": sorted(stats_prolog_bilingual_skipped),
    }
    totals["ko_coverage_pct"] = round(totals["records_ko_only"] / totals["records_total"] * 100, 2) if totals["records_total"] else 0
    totals["en_coverage_pct"] = round(totals["records_en_only"] / totals["records_total"] * 100, 2) if totals["records_total"] else 0

    stats_doc = {
        "build_date": datetime.now(timezone.utc).isoformat(),
        "prolog_source": prolog_source,
        "input_dir": str(args.input),
        "output_dir": str(args.output),
        "filters": {
            "min_zh_chars": args.min_zh_chars,
            "min_ko_chars": args.min_ko_chars,
        },
        "totals": totals,
        "per_book": dict(sorted(stats_per_book.items())),
    }
    with open(args.output / "corpus_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats_doc, f, ensure_ascii=False, indent=2)

    print()
    print("=" * 60)
    print("Build complete")
    print("=" * 60)
    print(f"  Total records:        {totals['records_total']:>10,}")
    print(f"  Bilingual:            {totals['records_bilingual']:>10,}")
    print(f"  Korean only:          {totals['records_ko_only']:>10,} ({totals['ko_coverage_pct']}%)")
    print(f"  Hanja only:           {totals['records_zh_only']:>10,}")
    print(f"  English only:         {totals['records_en_only']:>10,} ({totals['en_coverage_pct']}%)")
    print(f"  Filtered (too short): {totals['records_filtered_short']:>10,}")
    print(f"  Books w/ prolog:      {totals['books_with_prolog']:>10,}")
    print(f"  Prolog bi-skipped:    {totals['prolog_bilingual_skipped']:>10,}")
    print()
    print(f"  Chars (한문):         {totals['chars_zh']:>12,}")
    print(f"  Chars (한국어):       {totals['chars_ko']:>12,}")
    print(f"  Chars (영어):         {totals['chars_en']:>12,}")
    print(f"  Bilingual block size: {totals['chars_bilingual_blocks']:>12,}")
    print()
    print(f"  Stats: {args.output / 'corpus_stats.json'}")


if __name__ == "__main__":
    main()
