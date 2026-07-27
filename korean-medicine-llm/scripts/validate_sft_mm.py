#!/usr/bin/env python3
"""한의학 멀티모달 SFT 데이터 품질 검증 스크립트.

약초 이미지→대화쌍 SFT 데이터(mm_train.jsonl)가 데이터 감사에서 발견된
4개 결함(빌더 수정 대상)을 고쳤는지 검증한다. 의존성은 표준 라이브러리 json만.
결정론적. 새 파일 생성 외 부작용 없음.

검증 항목:
  #1 tox 안전 게이트   — 독성 미검증 종을 "안전"으로 단정하지 않는가
  #2 eff abstain 존재  — 효능 abstain 답변이 존재하는가 + unlinked 종 효능 단정 금지
  #3b part 캐비엇      — 이미지 부위≠약용부위인 eff 행에 부위 캐비엇이 있는가
  #4 템플릿 다양성     — 종당 eff 답변이 100% 동일하지 않은가
  (회귀 가드) image↔text 짝짓기 불변식 보존
"""
import argparse
import collections
import json
import os
import sys


# ---------------------------------------------------------------------------
# 상수 / 사전
# ---------------------------------------------------------------------------
PART_EN_TO_KO = {
    "root": "뿌리",
    "stem": "줄기",
    "leaf": "잎",
    "flower": "꽃",
    "fruit": "열매",
    "group": "전초",
}

# 이미지 part(영문)↔호환 약용부위(한글) — 빌더(build_sft_mm._PART_COMPAT)와 동일해야 한다.
# 캐비엇 부착 기준의 단일 진실. 둘이 어긋나면 검증이 거짓 위반을 낸다.
_ALL_PARTS_KO = {"뿌리", "뿌리줄기", "줄기껍질", "수피", "줄기",
                 "잎", "꽃", "열매", "씨", "껍질", "전초"}
_PART_COMPAT = {
    "root": {"뿌리", "뿌리줄기", "줄기껍질", "수피"},
    "stem": {"줄기", "뿌리줄기", "줄기껍질", "수피", "껍질"},
    "leaf": {"잎", "전초"},
    "flower": {"꽃", "전초"},
    "fruit": {"열매", "씨", "껍질"},
    "group": _ALL_PARTS_KO,
}

# tox 답변이 "안전하다"고 단정하는 문구들
SAFE_ASSERT_PATTERNS = (
    "독초로 분류되지 않은",
    "독성이 없",
    "안전한 식물",
    "안전합니다",
    "안전한 약초",
)

# eff 답변이 효능 단정을 회피(abstain)하는 문구들
ABSTAIN_PATTERNS = (
    "효능을 단정하지 않",
    "효능을 단정할 수 없",
    "확인하지 못",
    "확인되지 않",
    "확인할 수 없",
    "단정하기 어렵",
    "고전 문헌과 연결되지 않",
)

# eff 답변이 효능을 적극적으로 단정함을 나타내는 표지
EFF_ASSERT_MARKER = "효능"

# 부위 캐비엇 표지
PART_CAVEAT_MARKER = "부위"


# ---------------------------------------------------------------------------
# 로더
# ---------------------------------------------------------------------------
def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                sys.stderr.write(f"[warn] {path}:{i} JSON 파싱 실패: {e}\n")
    return rows


def load_annotation(path):
    """species_ko -> annotation dict."""
    ann = {}
    if not os.path.exists(path):
        return ann
    for o in load_jsonl(path):
        sp = o.get("species_ko")
        if sp is not None:
            ann[sp] = o
    return ann


def last_gpt(row):
    convs = row.get("conversations") or []
    if convs and convs[-1].get("from") == "gpt":
        return convs[-1].get("value", "") or ""
    # fallback: 마지막 gpt 턴 탐색
    for c in reversed(convs):
        if c.get("from") == "gpt":
            return c.get("value", "") or ""
    return ""


# ---------------------------------------------------------------------------
# 출력 헬퍼
# ---------------------------------------------------------------------------
def verdict_line(status, name, evidence):
    print(f"[{status}] {name} — {evidence}")


# ---------------------------------------------------------------------------
# 검사 #1: tox 안전 게이트
# ---------------------------------------------------------------------------
def check_tox_safety(rows, ann):
    name = "#1 tox 안전 게이트"
    tox_rows = [r for r in rows if r.get("task") == "tox"]

    has_tox_status = any("tox_status" in a for a in ann.values())
    if not has_tox_status:
        verdict_line(
            "SKIP", name,
            f"annotation에 tox_status 필드 없음 (tox 행 {len(tox_rows)}개). "
            "독성 검증 상태를 알 수 없어 안전 단정 위반을 판정 불가.",
        )
        # 참고용: 안전 단정 문구를 포함한 tox 답변 수만 보고
        n_safe = sum(
            1 for r in tox_rows
            if any(p in last_gpt(r) for p in SAFE_ASSERT_PATTERNS)
        )
        print(f"      참고: 안전 단정 문구 포함 tox 답변 = {n_safe}/{len(tox_rows)}")
        return None  # SKIP은 통과/실패 집계에서 제외

    sp_to_status = {sp: a.get("tox_status") for sp, a in ann.items()}
    violations = []
    for r in tox_rows:
        if sp_to_status.get(r.get("species_ko")) == "unverified":
            ans = last_gpt(r)
            # abstain/유보 문구(예 "독성이 없다고 단정할 수 없으니")는 안전 단정이 아니다.
            # 부정·유보 표지가 있으면 SAFE_ASSERT 부분문자열 오탐을 배제한다.
            if any(p in ans for p in ABSTAIN_PATTERNS) or "단정할 수 없" in ans:
                continue
            if any(p in ans for p in SAFE_ASSERT_PATTERNS):
                violations.append(r)

    n = len(violations)
    status = "PASS" if n == 0 else "FAIL"
    verdict_line(status, name,
                 f"독성 미검증 종을 안전 단정한 tox 행 = {n}/{len(tox_rows)}")
    for r in violations[:3]:
        print(f"      예시: {r.get('species_ko')} | {last_gpt(r)[:60]}")
    return status == "PASS"


# ---------------------------------------------------------------------------
# 검사 #2: eff abstain 존재
# ---------------------------------------------------------------------------
def check_eff_abstain(rows, ann):
    name = "#2 eff abstain 존재"
    eff_rows = [r for r in rows if r.get("task") == "eff"]
    total = len(eff_rows)

    abstain = [
        r for r in eff_rows
        if any(p in last_gpt(r) for p in ABSTAIN_PATTERNS)
    ]
    n_abstain = len(abstain)
    ratio = (n_abstain / total * 100) if total else 0.0

    # unlinked 종에 대해 효능 단정(non-abstain + 효능 표지)한 eff 위반
    sp_to_know = {sp: a.get("knowledge_status") for sp, a in ann.items()}
    unlinked_assert = []
    for r in eff_rows:
        ks = sp_to_know.get(r.get("species_ko"))
        if ks is not None and ks != "linked":
            ans = last_gpt(r)
            is_abstain = any(p in ans for p in ABSTAIN_PATTERNS)
            if (not is_abstain) and (EFF_ASSERT_MARKER in ans):
                unlinked_assert.append(r)

    ok = (n_abstain > 0) and (len(unlinked_assert) == 0)
    status = "PASS" if ok else "FAIL"
    verdict_line(status, name,
                 f"abstain eff = {n_abstain}/{total} ({ratio:.1f}%), "
                 f"unlinked 종 효능 단정 위반 = {len(unlinked_assert)}")
    if n_abstain == 0:
        print("      상세: abstain 답변이 0개 — 모든 eff가 효능을 단정함.")
    for r in unlinked_assert[:3]:
        print(f"      위반예시(unlinked): {r.get('species_ko')} | {last_gpt(r)[:50]}")
    return status == "PASS"


# ---------------------------------------------------------------------------
# 검사 #3b: part 캐비엇
# ---------------------------------------------------------------------------
def _as_part_list(val):
    """medicinal_part_ko는 list[str]일 수도, str일 수도 있음 → 정규화."""
    if isinstance(val, list):
        return [str(x).strip() for x in val if str(x).strip()]
    if isinstance(val, str) and val.strip():
        return [val.strip()]
    return []


def _part_matches(part_en, medparts):
    """이미지 부위(영문)가 약용부위와 호환되는가 — 빌더의 _PART_COMPAT 규칙과 동일.
    호환 집합과 약용부위의 교집합이 있으면 캐비엇 불필요(일치로 간주)."""
    compat = _PART_COMPAT.get(part_en, set())
    return bool(compat & set(medparts))


def check_part_caveat(rows, ann):
    name = "#3b part 캐비엇"
    eff_rows = [r for r in rows if r.get("task") == "eff"]

    sp_to_medparts = {
        sp: _as_part_list(a.get("medicinal_part_ko"))
        for sp, a in ann.items()
    }
    # 약용부위가 비어있지 않은 종이 하나라도 있는가
    has_medpart = any(v for v in sp_to_medparts.values())
    if not has_medpart:
        verdict_line(
            "SKIP", name,
            "annotation에 비어있지 않은 medicinal_part_ko가 없음. "
            "이미지 부위↔약용부위 불일치를 판정할 기준 부재.",
        )
        return None

    mismatch_no_caveat = []
    n_checked = 0
    for r in eff_rows:
        medparts = sp_to_medparts.get(r.get("species_ko"), [])
        if not medparts:
            continue
        img_part_ko = PART_EN_TO_KO.get(r.get("part"))
        if img_part_ko is None:
            continue
        # group(군락/전체) 이미지는 특정 부위를 특정할 수 없어 캐비엇 대상 아님.
        if r.get("part") == "group":
            continue
        # abstain eff 는 효능 자체를 단정하지 않으므로 부위 캐비엇 비대상.
        if any(p in last_gpt(r) for p in ABSTAIN_PATTERNS):
            continue
        n_checked += 1
        if not _part_matches(r.get("part"), medparts):
            ans = last_gpt(r)
            if PART_CAVEAT_MARKER not in ans:
                mismatch_no_caveat.append(r)

    n = len(mismatch_no_caveat)
    status = "PASS" if n == 0 else "FAIL"
    verdict_line(status, name,
                 f"부위 불일치인데 캐비엇 없는 eff 행 = {n} "
                 f"(약용부위 기재 종의 eff {n_checked}행 검사)")
    for r in mismatch_no_caveat[:3]:
        medparts = sp_to_medparts.get(r.get("species_ko"), [])
        print(f"      예시: {r.get('species_ko')} 이미지부위={PART_EN_TO_KO.get(r.get('part'))} "
              f"약용부위={medparts}")
    return status == "PASS"


# ---------------------------------------------------------------------------
# 검사 #4: 템플릿 다양성
# ---------------------------------------------------------------------------
def check_template_diversity(rows, ann):
    name = "#4 템플릿 다양성"
    eff_rows = [r for r in rows if r.get("task") == "eff"]

    by_species = collections.defaultdict(set)
    for r in eff_rows:
        by_species[r.get("species_ko")].add(last_gpt(r))

    # 종별 distinct eff 답변 수 분포
    distinct_counts = [len(s) for s in by_species.values()]
    dist_hist = collections.Counter(distinct_counts)
    n_species = len(by_species)
    n_identical = sum(1 for c in distinct_counts if c == 1)  # 100% 동일한 종

    # 전체 eff 답변의 distinct 도입부(첫 25자)
    intros = set(last_gpt(r)[:25] for r in eff_rows)
    n_intros = len(intros)

    # 모든 eff-보유 종이 100% 동일 → FAIL
    all_identical = (n_species > 0 and n_identical == n_species)
    status = "FAIL" if all_identical else "PASS"
    verdict_line(status, name,
                 f"100% 동일 종 = {n_identical}/{n_species}, "
                 f"전체 distinct 도입부(25자) = {n_intros}")
    # 분포 출력 (distinct 답변 수 -> 종 수)
    hist_str = ", ".join(f"{k}개답변:{v}종" for k, v in sorted(dist_hist.items()))
    print(f"      종별 distinct eff 답변 분포: {hist_str}")
    if n_intros <= 2:
        print(f"      경고: 전체 eff 도입부 종류가 {n_intros}개뿐 — 다양성 부족.")
    return status == "PASS"


# ---------------------------------------------------------------------------
# 회귀 가드: image↔text 짝짓기 불변식
# ---------------------------------------------------------------------------
def check_invariants(rows):
    name = "(회귀가드) image↔text 짝짓기 불변식"
    bad_structure = 0
    bad_species_link = 0
    examples = []
    for r in rows:
        convs = r.get("conversations") or []
        ok = (
            len(convs) >= 2
            and convs[0].get("from") == "human"
            and "<image>" in (convs[0].get("value") or "")
            and convs[-1].get("from") == "gpt"
        )
        if not ok:
            bad_structure += 1
            if len(examples) < 3:
                examples.append(r.get("image"))
            continue
        # id/eff 답변에 species_ko 포함 확인.
        # 단, abstain eff 는 효능을 단정하지 않아 종명을 명시하지 않는 게 정상 → 면제.
        if r.get("task") in ("id", "eff"):
            ans = last_gpt(r)
            is_abstain_eff = r.get("task") == "eff" and any(p in ans for p in ABSTAIN_PATTERNS)
            sp = r.get("species_ko") or ""
            if sp and not is_abstain_eff and sp not in ans:
                bad_species_link += 1
                if len(examples) < 3:
                    examples.append(r.get("image"))

    n = bad_structure + bad_species_link
    status = "PASS" if n == 0 else "FAIL"
    verdict_line(status, name,
                 f"구조 위반 = {bad_structure}, species_ko 미포함(id/eff) = {bad_species_link} "
                 f"(전체 {len(rows)}행)")
    for ex in examples:
        print(f"      예시 image: {ex}")
    return status == "PASS"


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="한의학 멀티모달 SFT 데이터 품질 검증")
    ap.add_argument("--train", default="data/sft/mm_train.jsonl",
                    help="검증할 SFT JSONL (기본: data/sft/mm_train.jsonl)")
    ap.add_argument("--annotation", default="data/annotations/species_annotation.jsonl",
                    help="종 메타데이터 JSONL (기본: data/annotations/species_annotation.jsonl)")
    args = ap.parse_args()

    if not os.path.exists(args.train):
        sys.stderr.write(f"[error] train 파일 없음: {args.train}\n")
        return 2

    rows = load_jsonl(args.train)
    ann = load_annotation(args.annotation)

    print("=" * 70)
    print(f"SFT 멀티모달 데이터 검증: {args.train}")
    print(f"  행 수 = {len(rows)}, annotation 종 수 = {len(ann)}")
    print("=" * 70)

    results = []
    results.append(check_tox_safety(rows, ann))
    results.append(check_eff_abstain(rows, ann))
    results.append(check_part_caveat(rows, ann))
    results.append(check_template_diversity(rows, ann))
    # 회귀 가드는 별도 집계(항상 PASS 기대)
    inv = check_invariants(rows)

    print("=" * 70)
    graded = [r for r in results if r is not None]
    n_pass = sum(1 for r in graded if r)
    n_skip = sum(1 for r in results if r is None)
    print(f"요약: 통과 {n_pass}/4 (SKIP {n_skip}개 제외, 채점 {len(graded)}개), "
          f"회귀가드 {'PASS' if inv else 'FAIL'}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
