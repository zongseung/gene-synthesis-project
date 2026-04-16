"""mediclassics 161종 전체 메타데이터 수집.

상위 listing 페이지 (`https://mediclassics.kr/books/`) 의 HTML 에서
(book_id, 국역제목, 한자제목) 삼중쌍 추출해 data/stats/mediclassics_book_list.json 에 저장.

__§03.2__ 기준 메타는 API 가 아닌 HTML 렌더링 결과에서 가져온다.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import httpx

HEADERS = {
    "Authorization": "5fe23edf9dec4c718e188073e46274bd",
    "Content-Type": "application/json",
}
LIST_URL = "https://mediclassics.kr/books/"


def fetch_html() -> str:
    r = httpx.get(LIST_URL, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.text


def parse_books(html: str) -> list[dict]:
    """HTML 에서 <a href="/books/{id}"> 주변 텍스트로 한자/국역 제목 추출.

    mediclassics 카드 패턴: `<a href="/books/{id}"...>...{국역}...{한자}...</a>`
    또는 data-book-name 속성, title 속성 등 다양 — 관용적 추출.
    """
    books: dict[int, dict] = {}

    # pattern A: a href + 본문 (비탐욕)
    # id, raw_chunk (다음 </a> 까지) 수집
    for m in re.finditer(
        r'<a[^>]*href="/books/(\d+)"[^>]*>(.*?)</a>',
        html,
        flags=re.DOTALL,
    ):
        bid = int(m.group(1))
        chunk = m.group(2)
        # inner tags 제거 → 텍스트만
        text = re.sub(r"<[^>]+>", " ", chunk)
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            continue
        existing = books.get(bid, {})
        # 더 긴 텍스트 우선 (카드 전체가 더 풍부)
        if len(text) > len(existing.get("raw_text", "")):
            books[bid] = {"book_id": bid, "raw_text": text}

    # 한자 (CJK) / 한글 분리
    for bid, rec in books.items():
        raw = rec["raw_text"]
        hanja = re.findall(r"[\u3400-\u9FFF]+", raw)
        hangul = re.findall(r"[\uAC00-\uD7AF]+", raw)
        rec["hanja_tokens"] = hanja
        rec["hangul_tokens"] = hangul
        # 대표 제목: 가장 긴 한자/한글 token
        rec["title_zh"] = max(hanja, key=len) if hanja else None
        rec["title_ko"] = max(hangul, key=len) if hangul else None

    return [books[i] for i in sorted(books)]


def main() -> None:
    print(f"[fetch] GET {LIST_URL}")
    html = fetch_html()
    print(f"[fetch] html len = {len(html):,}")

    books = parse_books(html)
    print(f"[parse] books = {len(books)}")

    out_dir = Path("data/stats")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "source": LIST_URL,
        "n_books": len(books),
        "books": [
            {
                "book_id": b["book_id"],
                "title_ko": b.get("title_ko"),
                "title_zh": b.get("title_zh"),
                "raw_text": b["raw_text"],
            }
            for b in books
        ],
    }
    out_path = out_dir / "mediclassics_book_list.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[save] {out_path}")

    # 프리뷰 — 처음 20권
    print("\n=== preview (first 20) ===")
    for b in books[:20]:
        print(f"  book_{b['book_id']:3d}  {b.get('title_ko','?'):15s}  {b.get('title_zh','?')}")


if __name__ == "__main__":
    main()
