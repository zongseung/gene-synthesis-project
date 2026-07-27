"""KCI(한국학술지인용색인) 한의학 분야 초록 수집 PoC — CPT 보강용.

합법성 원칙(SCI 논문 Data Availability 대응):
  - **메타데이터 + 초록(abstract)만** 저장. 본문(full text) 절대 수집 금지.
  - 각 레코드에 source / license / DOI / KCI ID 메타 보존.
  - 페이월(DBpia·RISS 전문) 크롤링 금지. KCI OpenAPI(공공데이터, "이용허락 제한 없음")만 사용.

데이터 출처: data.go.kr 15085348 (한국연구재단 KCI 논문정보).
  KCI Open API 엔드포인트: https://open.kci.go.kr/po/openapi/openApiSearch.kci  [추정]
  - 정확한 apiCode/파라미터·서비스키 발급은 claudedocs/kci_collection_plan.md 절차 참조.
  - 키는 .env 의 DATA_GO_KR_KEY 에서 로드. 없으면 dry-run(요청만 구성·출력).

산출: jsonl 1행 = 논문 1건
  {title_ko, title_en, abstract_ko, abstract_en, authors, journal, year, doi,
   kci_id, keywords, source, license, query, collected_at}

사용:
  # 키 없을 때(요청 구성만 확인):
  PYTHONPATH=src .venv/bin/python -m hanmed_mm.data.collect_kci_abstracts --dry_run
  # 키 있을 때(소량 PoC):
  PYTHONPATH=src .venv/bin/python -m hanmed_mm.data.collect_kci_abstracts \
      --query "한의학" --max_records 100 --out data/cpt/kci_abstracts.jsonl
"""
from __future__ import annotations
import argparse, json, os, sys, time, datetime
import xml.etree.ElementTree as ET

try:
    import requests
except ImportError:
    requests = None

KCI_ENDPOINT = "https://apis.data.go.kr/B552540/KCIOpenApi/artiInfo"  # 확정(사용자 제공)
# 한의학 관련 기본 질의어(분야 한정). KCI 분류코드(예: 의약학>한의학)로의 정밀 한정은
# 키 발급 후 displayCode/분류 파라미터로 대체 권장(plan 문서 참조).
DEFAULT_QUERIES = ["한의학", "동의보감", "본초", "방제", "침구", "사상체질", "변증"]

ENV_PATH = ".env"


def load_env_key(name: str = "DATA_GO_KR_KEY") -> str | None:
    """환경변수 → .env 순으로 키 로드(dotenv 의존성 없이)."""
    v = os.environ.get(name)
    if v:
        return v.strip()
    if os.path.exists(ENV_PATH):
        with open(ENV_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, val = line.partition("=")
                if k.strip() == name:
                    return val.strip().strip('"').strip("'")
    return None


def build_request(query: str, key: str | None, page: int, rows: int) -> dict:
    """KCI OpenAPI 요청 파라미터 구성. [추정] 파라미터명은 키 발급 후 공식 문서로 확정."""
    params = {
        "serviceKey": key or "<DATA_GO_KR_KEY>",  # data.go.kr 서비스키
        "title": query,               # 제목 검색(키워드). 필요시 abstract/keyword 필드 추가
        "page": page,
        "displayCount": rows,          # 최대 100
    }
    return {"url": KCI_ENDPOINT, "params": params}


def _text(node, *tags) -> str | None:
    for t in tags:
        el = node.find(t)
        if el is not None and (el.text or "").strip():
            return el.text.strip()
    return None


def parse_records(xml_text: str, query: str) -> list[dict]:
    """KCI XML 응답 → 레코드 리스트. 태그명은 응답 스키마에 맞춰 보강 필요([추정])."""
    out = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        print(f"[parse] XML 파싱 실패: {e}", file=sys.stderr)
        return out
    # 레코드 컨테이너 후보(KCI JATS형: record/articleInfo 포함)
    records = (root.findall(".//record") or root.findall(".//articleInfo")
               or root.findall(".//item") or root.findall(".//article"))
    now = datetime.datetime.utcnow().isoformat() + "Z"
    for r in records:
        # KCI JATS: <abstract-group><abstract lang=...> 또는 <abstract>
        abstract_ko = _text(r, "abstractKo", "abstract_ko", "abstract", ".//abstract")
        abstract_en = _text(r, "abstractEn", "abstract_en")
        # 초록이 전혀 없으면 CPT 가치 없음 + 합법성(메타만) 측면에서도 스킵
        if not (abstract_ko or abstract_en):
            continue
        out.append({
            "title_ko": _text(r, "titleKo", "title_ko", "title"),
            "title_en": _text(r, "titleEn", "title_en"),
            "abstract_ko": abstract_ko,
            "abstract_en": abstract_en,
            "authors": _text(r, "authors", "author"),
            "journal": _text(r, "journalName", "journal", "pubName"),
            "year": _text(r, "pubYear", "year"),
            "doi": _text(r, "doi", "DOI"),
            "kci_id": _text(r, "articleId", "kciId", "id"),
            "keywords": _text(r, "keyword", "keywords"),
            "source": "KCI OpenAPI (data.go.kr 15085348)",
            "license": "KCI 공개 메타데이터/초록 (이용허락 제한 없음) — 본문 미수집",
            "query": query,
            "collected_at": now,
        })
    return out


def collect(query: str, key: str, out_path: str, max_records: int, rows: int,
            sleep: float, seen_ids: set) -> int:
    n = 0
    page = 1
    with open(out_path, "a", encoding="utf-8") as f:
        while n < max_records:
            req = build_request(query, key, page, rows)
            try:
                resp = requests.get(req["url"], params=req["params"], timeout=20)
            except Exception as e:
                print(f"[net] 요청 실패(query={query}, page={page}): {e}", file=sys.stderr)
                break
            if resp.status_code != 200:
                print(f"[http] {resp.status_code} (query={query}, page={page}) — 중단", file=sys.stderr)
                break
            recs = parse_records(resp.text, query)
            if not recs:
                break
            for rec in recs:
                rid = rec.get("kci_id") or rec.get("doi") or rec.get("title_ko")
                if rid in seen_ids:
                    continue
                seen_ids.add(rid)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
                if n >= max_records:
                    break
            page += 1
            time.sleep(sleep)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", default=None, help="검색어(미지정 시 기본 한의학 질의어 세트)")
    ap.add_argument("--out", default="data/cpt/kci_abstracts.jsonl")
    ap.add_argument("--max_records", type=int, default=100, help="질의어당 최대 수집 건수")
    ap.add_argument("--rows", type=int, default=50, help="페이지당 요청 건수")
    ap.add_argument("--sleep", type=float, default=0.5, help="요청 간 대기(초)")
    ap.add_argument("--dry_run", action="store_true", help="키 없이 요청 구성만 출력")
    args = ap.parse_args()

    queries = [args.query] if args.query else DEFAULT_QUERIES
    key = load_env_key()

    if args.dry_run or not key:
        if not key:
            print("[!] DATA_GO_KR_KEY 미설정(.env/환경변수) → dry-run. "
                  "키 발급 절차: claudedocs/kci_collection_plan.md")
        print("=== 구성될 요청(예시) ===")
        for q in queries:
            req = build_request(q, key, page=1, rows=args.rows)
            shown = dict(req["params"]);
            if key: shown["key"] = key[:6] + "***"
            print(f"GET {req['url']}")
            print("    params:", json.dumps(shown, ensure_ascii=False))
        print("\n[i] 실제 수집은 키 확보 + 응답 스키마 확정([추정] 태그명) 후 --dry_run 없이 실행.")
        return

    if requests is None:
        print("[!] requests 미설치", file=sys.stderr); sys.exit(1)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    # resume: 기존 출력의 id 적재
    seen = set()
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                    seen.add(d.get("kci_id") or d.get("doi") or d.get("title_ko"))
                except Exception:
                    pass

    total = 0
    for q in queries:
        c = collect(q, key, args.out, args.max_records, args.rows, args.sleep, seen)
        print(f"[collect] query='{q}' → {c}건")
        total += c
    print(f"[OK] 총 {total}건 수집(초록 보유분만), 출력: {args.out}")


if __name__ == "__main__":
    main()
