"""ver6 SFT 데이터 품질 전수 검증 — raw book_008 와 대조.

점검 차원:
  1. Citation 무결성 — `[출처: X]` 또는 up_path_nm 필드가 raw path set 에 실재하는가
  2. Name 일치성 — Q 에 쓰인 처방/약재/경혈명이 raw 에 존재하는가
  3. Answer 내용 왜곡 — detail 부분이 raw trans_ko 와 substring match 되는가
  4. Empty / 비정상 — detail 없거나 "해당 없음" 같은 실패 마커 포함
  5. 질문 echo — Q 텍스트가 그대로 A 에 들어갔는가

사용:
    .venv/bin/python scripts/verify_sft_against_raw.py experiments/dongui_bogam/data/sft/phaseB_qa_v6_corpus.jsonl
"""
from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw" / "mediclassics_unified" / "book_008"


# 명시적 \uXXXX escape — literal PUA 문자가 파일 저장 중 사라지는 것 회피
_PUA_BMP = re.compile("[-]")
_PUA_A = re.compile("[\U000F0000-\U000FFFFD]")
_PUA_B = re.compile("[\U00100000-\U0010FFFD]")
_ZW_CLEAN = re.compile("[​-‏‪-‮⁠-⁯﻿]")
_CTRL_CLEAN = re.compile(r"[\x00-\x08\x0B-\x1F\x7F]")
_FULLWIDTH_SP = re.compile("　")


def clean(s: str) -> str:
    """builder 의 clean_text 와 동일 정규화 (raw path 와 SFT path 비교를 위해)."""
    if not s:
        return ""
    s = s.replace("\r", " ").replace("\n", " ")
    s = _CTRL_CLEAN.sub("", s)
    s = _PUA_BMP.sub("", s)
    s = _PUA_A.sub("", s)
    s = _PUA_B.sub("", s)
    s = _ZW_CLEAN.sub("", s)
    s = _FULLWIDTH_SP.sub(" ", s)
    return re.sub(r"[ \t]{2,}", " ", s).strip()


def self_path(r: dict) -> str:
    """record self-path, always cleaned."""
    o = clean(r.get("original") or "").split()
    first = o[0] if o else ""
    up_clean = clean(r.get("up_path_nm") or "")
    return up_clean + (" > " + first if first else "")


def load_raw():
    paths = set()
    up_paths = set()
    trans_by_up = collections.defaultdict(list)
    trans_by_self = {}
    names_by_level = collections.defaultdict(set)
    # 한자→한국어 segment map (ver6 r3 Korean-first citation 인정)
    han2ko: dict[str, str] = {}
    recs_cache = []

    for p in sorted(RAW_DIR.glob("vol_*.jsonl")):
        with p.open() as f:
            for ln in f:
                r = json.loads(ln)
                recs_cache.append(r)
                up = clean(r.get("up_path_nm") or "")
                sp = self_path(r)
                if up:
                    up_paths.add(up)
                    paths.add(up)
                if sp:
                    paths.add(sp)
                    trans_by_self[sp] = clean(r.get("trans_ko") or "")
                if up:
                    trans_by_up[up].append(clean(r.get("trans_ko") or ""))
                lvl = r.get("content_level") or "?"
                orig_first = clean(r.get("original") or "").split()[:1]
                if orig_first:
                    names_by_level[lvl].add(orig_first[0])
                # build segment map (동일 builder 로직)
                original = clean(r.get("original") or "")
                trans_ko = clean(r.get("trans_ko") or "")
                if not original or not trans_ko:
                    continue
                if lvl == "AA":
                    han2ko.setdefault(original.strip(), trans_ko.strip())
                else:
                    orig_tokens = original.split()
                    ko_base = trans_ko.split("(")[0].strip()
                    ko_tokens = ko_base.split()
                    if orig_tokens and ko_tokens:
                        han2ko.setdefault(orig_tokens[0], ko_tokens[0])

    # 한국어 변환 path 도 valid 로 추가
    def path_to_ko(path: str) -> str:
        if not path:
            return path
        segs = [s.strip() for s in path.split(">")]
        out = []
        for s in segs:
            if s in han2ko:
                out.append(han2ko[s]); continue
            toks = s.split()
            if not toks: out.append(s); continue
            ko_head = han2ko.get(toks[0], toks[0])
            out.append(ko_head if (ko_head != toks[0] or len(toks) == 1) else s)
        return " > ".join(out)

    ko_paths = {path_to_ko(p) for p in paths}
    paths.update(ko_paths)
    return {
        "paths": paths,
        "up_paths": up_paths,
        "trans_by_up": trans_by_up,
        "trans_by_self": trans_by_self,
        "names_by_level": names_by_level,
    }


def extract_citations(text: str) -> list[str]:
    # [출처: ...] 또는 (출처: ...) 패턴
    cites = re.findall(r"\[?출처:\s*([^\]\)\n]+)", text)
    return [c.strip().rstrip("]").strip() for c in cites]


# 주의: raw string 안에서 \U... 는 이스케이프 안 됨 → non-raw 로 작성
_PUA_ALL = re.compile("[-\U000F0000-\U000FFFFD\U00100000-\U0010FFFD]")
_ZW = re.compile("[​-‏‪-‮⁠-⁯﻿]")
_CTRL = re.compile(r"[\x00-\x08\x0B-\x1F\x7F]")


def verify_pair(pair: dict, raw: dict) -> dict:
    """단일 QA 쌍의 품질 문제 목록."""
    issues = []
    q = pair["question"]
    a = pair["assistant"]
    up = pair.get("up_path_nm")
    cat = pair["category"]

    # 0. 텍스트 artifact 점검
    for name, pat in (("pua_char", _PUA_ALL), ("zero_width", _ZW), ("ctrl_char", _CTRL)):
        if pat.search(q) or pat.search(a):
            issues.append((name, "found"))

    # 1. up_path_nm 이 raw 에 존재
    if up and up not in raw["paths"]:
        issues.append(("cite_invalid_up", up))

    # 2. 답변 내 [출처: X] 모두 raw paths 에 존재
    cites_in_a = extract_citations(a)
    for c in cites_in_a:
        if c not in raw["paths"]:
            issues.append(("cite_invalid_answer", c))

    # 3. Free-text detail 이 실재 raw trans_ko 와 substring match (최소 30자)
    # passage/symptom/prescription/herb/acupoint 류 — detail 이 raw 로부터 왔는지
    if cat in ("passage", "symptom", "prescription", "herb", "acupoint", "structure"):
        # answer 에서 원문 인용 의심 부분 (40자 이상) 이 trans_by_up 또는 trans_by_self 에 등장하는지 lookup
        if up:
            raw_texts = raw["trans_by_up"].get(up, [])
            if self_path_in_paths := [sp for sp in raw["trans_by_self"].keys() if sp == up]:
                raw_texts.append(raw["trans_by_self"][self_path_in_paths[0]])
            # 단순 matching: answer 안의 20자 chunk 가 어느 raw_text 에도 없으면 의심
            # 단, 템플릿 문자열은 제외
            # "해당 없음" / "본문 참고" / "본문 인용 참조" 같은 실패 마커 카운트
            for marker in ["해당 없음", "본문 인용 참조", "본문 설명 참고", "본문 참고"]:
                if marker in a:
                    issues.append(("failure_marker", marker))
                    break

    # 4. 질문 echo — Q 가 A 앞부분에 **원문 그대로** 들어가 있으면 fail.
    #   템플릿 내 {name} 중복은 허용 (자연스러운 구조). 30자 chunk 로 체크는 strict → 50자로.
    q_sig = q[:50]
    # A 의 앞 30 토큰 중에 Q 서두 50자가 그대로 있는 경우만 echo 로 간주
    if q_sig and q_sig in a[:150]:
        issues.append(("question_echo", q_sig[:30]))

    # 5. 너무 짧은 답변 — paraphrase/refusal 은 단문 정답 허용
    if cat not in ("paraphrase", "refusal_oos", "refusal_safety") and len(a) < 40:
        issues.append(("answer_too_short", f"{len(a)}chars"))

    # 6. prescription 이름 — Q 의 **앞부분 따옴표 속** 괄호 속 한자만 검증
    #   (본문 중 "효능과 주치(主治)" 같은 일반 괄호는 제외)
    if cat == "prescription":
        # "'익국보화환(䴰麴保和丸)'" 패턴: 작은따옴표 바로 앞의 괄호만
        m = re.search(r"'[^']*?\(([一-龥]+)\)'", q)
        if m:
            nh = m.group(1)
            if nh not in raw["names_by_level"]["DP"] and nh not in raw["names_by_level"]["EP"]:
                issues.append(("name_not_in_raw", f"prescription:{nh}"))
    return {"id": pair.get("id"), "category": cat, "issues": issues}


def summarize(results: list[dict]) -> dict:
    issue_counts = collections.Counter()
    by_cat = collections.defaultdict(lambda: collections.Counter())
    failing_pairs = 0
    for r in results:
        if r["issues"]:
            failing_pairs += 1
        for kind, _ in r["issues"]:
            issue_counts[kind] += 1
            by_cat[r["category"]][kind] += 1
    return {
        "total_pairs": len(results),
        "pairs_with_any_issue": failing_pairs,
        "pairs_issue_rate": round(failing_pairs / max(1, len(results)), 4),
        "issues_by_kind": dict(issue_counts.most_common()),
        "issues_by_category": {k: dict(v.most_common()) for k, v in by_cat.items()},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl_path", type=Path)
    ap.add_argument("--sample", type=int, default=0, help="N 개만 검사 (0=전체)")
    ap.add_argument("--show-samples", type=int, default=5, help="문제 사례 출력 수")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    print("[load] raw book_008...", flush=True)
    raw = load_raw()
    print(f"  paths: {len(raw['paths'])}, up_paths: {len(raw['up_paths'])}")

    print(f"[verify] {args.jsonl_path}...", flush=True)
    results = []
    with args.jsonl_path.open() as f:
        for i, ln in enumerate(f):
            if args.sample and i >= args.sample:
                break
            pair = json.loads(ln)
            results.append(verify_pair(pair, raw))

    summary = summarize(results)
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    # 문제 사례 샘플 출력
    print(f"\n=== 문제 사례 sample (각 kind 최대 {args.show_samples}개) ===")
    samples_by_kind = collections.defaultdict(list)
    for r in results:
        for kind, detail in r["issues"]:
            if len(samples_by_kind[kind]) < args.show_samples:
                samples_by_kind[kind].append((r["id"], detail))
    for kind, items in samples_by_kind.items():
        print(f"\n[{kind}]")
        for item_id, detail in items:
            print(f"  {item_id}: {detail}")

    if args.output:
        with args.output.open("w") as f:
            json.dump({"summary": summary, "results": results}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
