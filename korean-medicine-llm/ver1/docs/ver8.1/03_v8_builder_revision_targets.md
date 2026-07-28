# 03 · v8 빌더 함수별 갱신 사양 (round_1 결과 반영)

> 작성일: 2026-04-26
> ver8/00 §6 빌더 함수 매트릭스 + ver8/02 §6 builder 정보 손실 지점을 round_1 발견과 결합해 v8 빌더 (`scripts/build_sft_v8/`) 의 함수 단위 사양으로 갱신.
> ver8 의 Agent-P/A/S/M/R 분할 (ver8/00 §8) 그대로 사용. 본 문서는 각 Agent 의 입력 사양 보강.

## 0. v8 빌더 디렉토리 구조 (제안)

```
scripts/build_sft_v8/
├── __init__.py
├── helpers.py              ★ 모든 transform 헬퍼 모듈 (round_1 anchor 문제 회피)
│   ├── DOSAGE_PATTERNS    # 12 unit 정규식 (audit.py 와 동기화)
│   ├── DISCLAIMER_POOL    # 5+ phrase
│   ├── DENY_ENTITIES      # 7명 + 향후 발견분
│   ├── Q_FORMAT_ENUM
│   ├── mask_dosage()
│   ├── pick_disclaimer()
│   ├── validate_entities()
│   ├── infer_q_format()
│   ├── enforce_quote_preservation()
│   └── load_raw_record_clean()  # ver8/00 §5.1 의 그것
├── builders/
│   ├── prescription.py    # Agent-P 담당 (DP/EP/CP)
│   ├── acupoint.py        # Agent-A 담당 (DK)
│   ├── symptom.py         # Agent-S 담당 (DD)
│   ├── herb.py            # Agent-M 담당 (CH/DH)
│   ├── refusal.py         # Agent-R 담당 (refusal_v2 + pregnancy_safety)
│   ├── passage.py         # SS/ZZ
│   ├── structure.py       # AA/BB/CC/OO/Z2
│   └── niche.py           # XX/PP/TT/EE
├── assertions.py          # 빌드 종료 직전 self-assert (ver8.1 §C)
└── entry.py               # build_sft_full_corpus_v8.py 의 main
```

helper 들은 builder 들이 import 해서 사용. anchor-based 인라인 패치 X.

## 1. 함수별 v8 사양 (round_1 발견 반영)

각 sub-section 은 ver8/02 §6 의 v7 빌더 정보 손실 지점 + ver8/00 §3.2 의 v8 매핑 + round_1 발견을 함께 명시.

### 1.1 `builders/prescription.py` (Agent-P)

**v7 한계 (ver8/02 §6.1)**: `PRESCRIPTION_Q_TEMPLATES` 6 종 모두 효능/주치/역할 방향, "구성/조성 약재" 질문 0종 → 처방→조성 direct QA = 2 (실질 0).

**v8 사양 (ver8/00 §3.2.1)**: 3 variant × 6,039 = ~18,117 rows. variant: direct-efficacy / direct-composition / direct-source.

**round_1 추가 의무**:
1. `mask_dosage()` 를 모든 variant 의 answer 에 적용 (12 unit 정규식, `preserve_quoted_spans=True`)
2. `q_format` 채우기:
   - direct-efficacy → `Q_prescription_efficacy`
   - direct-composition → `Q_prescription_composition`
   - direct-source → `Q_prescription_source`
3. `validate_entities()` 로 deny 7명 + 향후 발견분 emit 단계에서 reject (mask 로 우회 X)

```python
def build_prescription_pairs(records, helpers):
    pairs = []
    for r in records:  # DP + EP + CP 6,040
        composition = helpers.extract_composition_with_dosage(r, mask=True)
        if not composition: continue
        for variant in ('efficacy', 'composition', 'source'):
            pair = build_one(r, variant, composition)
            if not helpers.validate_entities(pair): continue
            pair['q_format'] = f"Q_prescription_{variant}"
            pair['a_format'] = f"A_prescription_{variant}"
            pair['source_ids'] = [(r.volume_id, r.content_seq)]
            pairs.append(pair)
    return pairs
```

### 1.2 `builders/acupoint.py` (Agent-A)

**v7 한계 (ver8/02 §5.3)**: ACUPOINT_Q_TEMPLATES 5종 rotate `i % 5` → 396 DK 중 ~72만 INV 질문. MAJOR 15 편중. 증상→혈 inverse 부재.

**v8 사양 (ver8/00 §3.2.2)**: 4 variant × 396 = 1,584 rows. variant: meridian (inverse-1) / location / indications / symptom-to-point.

**round_1 추가 의무**:
1. `q_format` 채우기:
   - meridian → `Q_acupoint_meridian`
   - location → `Q_acupoint_location`
   - indications → `Q_acupoint_indication`
   - symptom-to-point → `Q_acupoint_symptom`
2. `MERIDIAN_SEGMENTS` 상수 (ver8/00 §3.2.2) 를 helpers.py 에 등록
3. symptom-to-point variant 의 symptom 추출은 자식 SS body 에서 명시 등장한 증상 키워드만 사용 (path leaf X)

### 1.3 `builders/symptom.py` (Agent-S)

**v7 한계 (ver8/02 §3.2)**: 증상→처방은 PRESCRIPTION[4] 의 "호소할 때" 템플릿에 묶여 context = 처방 부모 path leaf → "單方 을 호소할 때" 같은 비의미 pair 다수.

**v8 사양 (ver8/00 §3.2.4)**: 3 variant × 1,103 = 3,309 rows. variant: direct-pathology / inverse-rx / inverse-acu.

**round_1 추가 의무**:
1. inverse-rx / inverse-acu variant 의 처방·혈명은 **자식 SS/ZZ body 에서 literal substring 으로 등장한 것만** 답변에 emit. parent path token 사용 금지.
2. 답변에 dosage 가 포함되면 `mask_dosage()` 적용 (대부분 처방 inverse 답변에서 발생).
3. `q_format`:
   - direct-pathology → `Q_symptom_pathology`
   - inverse-rx → `Q_symptom_inverse_rx`
   - inverse-acu → `Q_symptom_inverse_acu`

### 1.4 `builders/herb.py` (Agent-M)

**v7 한계 (ver8/02 §4.2)**: HERB_Q_TEMPLATES 5종 중 성미 단독 질의 없음. 약재 1개당 1 row 고정 → 성미·귀경·식물학 정보 학습 신호 없음.

**v8 사양 (ver8/00 §3.2.3)**: 3 variant × 1,403 = 4,209 rows. variant: sami (성미) / indication / classification.

**round_1 추가 의무**:
1. sami variant 의 정규식 `^性[平寒溫熱凉].*?味.*?(毒|無毒).*?\.` (ver8/00 §3.2.3 표) 를 helpers.SAMI_RE 로 분리
2. indication 답변에 dosage 포함 시 mask
3. `q_format`:
   - sami → `Q_herb_sami`
   - indication → `Q_herb_indication`
   - classification → `Q_herb_classification`

### 1.5 `builders/refusal.py` (Agent-R)

**v7 한계 (ver8/02 §6.2)**: refusal 800 → 200 (75% 감소). pregnancy_safety 카테고리 없음.

**v8 사양 (ver8/00 §2.5 원칙 5 + §3.2.7)**: refusal_v2 50+ variant + pregnancy_safety 신설. PP `催生符` 1건은 pregnancy_safety 로 emit (이로써 raw 34,040 의 100% 커버).

**round_1 추가 의무**:
1. refusal_v2 의 unique answer ≥ 50 (이전 v7 의 12 → ≥ 50)
2. pregnancy_safety 답변 구조 = "동의보감 원문 조문 서술" + "현대의학 전문가 상담 권고"
3. `q_format`:
   - refusal_v2 → `Q_refusal_oos` 또는 `Q_refusal_safety`
   - pregnancy_safety → `Q_pregnancy_safety`
4. PP `vol_18 / seq_984 / 催生符` 1건은 pregnancy_safety 로 강제 매핑 (전수 커버리지 100%)

### 1.6 `builders/passage.py` (모든 Agent 공유)

**v7 한계 (ver8/02 §6.1)**: `build_passage_pairs` `target=7000` hard-cap → 15,739 SS/ZZ 폐기.

**v8 사양 (ver8/00 §3.2.5)**: cap 제거. 2 variant × 22,739 = 45,478 rows. variant: passage-summary / passage-topic.

**round_1 추가 의무**:
1. cap 제거 — `--passage-target` 인자 자체 삭제
2. `q_format`:
   - passage-summary → `Q_passage_summary`
   - passage-topic → `Q_passage_topic`
3. prefix pool ≥ 8종 (ver8/00 §2.4 원칙 4: passage 카테고리 내 prefix top-1 ≤ 8%)
4. 답변에 dosage 가 포함된 SS (특히 처방 본문) 는 mask 적용

### 1.7 `builders/structure.py`

ver8/00 §3.2.6 기준. AA/BB/CC/OO/Z2. 2 variant × 2,207 = 4,414 rows.

**round_1 추가 의무**: `q_format` = `Q_structure` 또는 `Q_meta_toc` (variant 별).

### 1.8 `builders/niche.py`

ver8/00 §3.2.7 기준. XX/PP/TT/EE/CP — v7 미활용 175건 전수 포함.

**round_1 추가 의무**:
1. CP 1건은 prescription.py 에서 합류 (ver8/00 §3.2.7 표)
2. PP 1건 (`催生符`) 은 refusal.py 의 pregnancy_safety 로 매핑
3. `q_format`: `Q_diagram_explain` / `Q_table_explain` / `Q_anatomy_diagram` / `Q_symptom_variant`

## 2. `assertions.py` — 빌드 종료 직전 self-assert (ver8.1 §C 통합)

```python
# scripts/build_sft_v8/assertions.py

from collections import Counter
import re

DOSAGE_RE = re.compile(
    r"(?<![0-9])\d+(?:\.\d+)?\s*(?:돈|푼|냥|전|알|첩|근|홉|되|말|구)"
    r"|(?<![0-9])\d+(?:\.\d+)?\s*(?:g|mg|kg|ml|L)\b"
    r"|각\s*\d+(?:\.\d+)?(?:돈|푼|냥|전|알|첩|근|홉|되|말|구|g|mg)"
)

RAW_TOTAL = 34040  # ver8/01 §1.1 검증 수치

def assert_v8_slo(qa_rows, raw_records):
    # C0. coverage
    covered = {sid for qa in qa_rows for sid in qa.get('source_ids', [])}
    raw_ids = {(r['volume_id'], r['content_seq']) for r in raw_records}
    missing = raw_ids - covered
    assert not missing, f"coverage fail: {len(missing)} records missing, sample={sorted(missing)[:5]}"
    assert len(covered) >= RAW_TOTAL, f"coverage count {len(covered)} < {RAW_TOTAL}"

    # C1. dosage_leak
    leak_rows = sum(1 for qa in qa_rows
                    if DOSAGE_RE.search(qa['messages'][-1]['content']))
    assert leak_rows <= 50, f"dosage_leak SLO violated: {leak_rows} rows leaked (cap 50)"

    # C2. literal_quote (외부 helper 사용)
    from .helpers import literal_quote_match_rate
    rate = literal_quote_match_rate(qa_rows, raw_records)
    assert rate >= 0.99, f"literal_quote regression: {rate:.4f} < 0.99"

    # C3. q_format diversity
    qf = Counter(qa.get('q_format') for qa in qa_rows)
    assert None not in qf, f"q_format null in {qf[None]} rows — emit 의무 위반"
    top_rate = max(qf.values()) / sum(qf.values())
    assert top_rate <= 0.55, f"q_format top_rate {top_rate:.3f} > 0.55"
    assert len(qf) >= 10, f"q_format unique {len(qf)} < 10"

    # C4. entity deny
    from .helpers import DENY_ENTITIES
    deny = sum(1 for qa in qa_rows for name in DENY_ENTITIES
               if name in qa['messages'][-1]['content'])
    assert deny == 0, f"deny entity {deny} occurrences in v8 corpus"

    # C5. disclaimer pool 분포
    from .helpers import DISCLAIMER_POOL
    closing_counts = Counter()
    for qa in qa_rows:
        text = qa['messages'][-1]['content']
        for d in DISCLAIMER_POOL:
            if d in text: closing_counts[d] += 1; break
    if sum(closing_counts.values()) > 0:
        max_rate = max(closing_counts.values()) / sum(closing_counts.values())
        assert max_rate <= 0.40, f"disclaimer top_rate {max_rate:.3f} > 0.40"

    return {
        "coverage": len(covered),
        "dosage_leak_rows": leak_rows,
        "literal_quote_rate": rate,
        "q_format_unique": len(qf),
        "q_format_top_rate": top_rate,
        "disclaimer_top_rate": max_rate if 'max_rate' in locals() else 0.0,
        "deny_entity_count": deny,
    }
```

`entry.py` 는 빌드 직전 다음을 호출:
```python
results = assertions.assert_v8_slo(qa_rows, raw_records)
print(json.dumps(results, indent=2))
# → phaseB_qa_v8_corpus.meta.json 에 emit
```

## 3. `entry.py` 의 `--self-test` 모드 (round_1 anchor 실패 회피)

v8 빌더 entry point 에 단일 vol 100-row sample 빌드 + assertions 즉시 실행:

```bash
python scripts/build_sft_full_corpus_v8.py --self-test
# → 100 rows 빌드 → assertions 실행 → 결과 stdout
```

이 명령이 모든 SLO 를 통과해야만 production 빌드 (`--full`) 가 진행.

## 4. v7 → v8 transition 매트릭스

ver8/00 §3.1 의 매트릭스를 round_1 결과 + helper 함수 매핑까지 포함해 보강:

| level | raw | v8 카테고리 | helper 사용 | round_1 권고 |
|---|---:|---|---|---|
| SS | 11,498 | passage + children detail | mask_dosage, infer_q_format | passage cap 제거, 8 prefix |
| ZZ | 11,241 | passage + concept_inverse | mask_dosage, infer_q_format | 동일 |
| DP | 5,273 | prescription_direct (3 variant) | mask_dosage, validate_entities | composition variant 신설 |
| EP | 766 | prescription_direct (3 variant) | 동일 | 동일 |
| CP | 1 | prescription_direct (DP 합류) | 동일 | 1건 raw 보존 |
| CC | 2,045 | structure + meta_toc | infer_q_format | 동일 |
| DD | 1,103 | symptom 3 variant | 자식 body 명시 처방·혈만 | parent leaf 사용 금지 |
| DH | 704 | herb (부위) 2 variant | SAMI_RE | 약재 부모 매핑 |
| CH | 699 | herb 3 variant (sami 추가) | SAMI_RE, mask_dosage | 성미 추출 정규식 신설 |
| DK | 396 | acupoint 4 variant | MERIDIAN_SEGMENTS | symptom-to-point 신설 |
| BB | 109 | structure + meta_toc | — | — |
| XX | 102 | diagram_explain + 단방 bridge | — | "凡 N 種" 카운트 QA |
| TT | 26 | table_explain (3 variant) | — | 운기·五腧五行 markdown 변환 |
| Z2 | 25 | meta_toc + concept_inverse | — | — |
| AA | 23 | structure + meta_toc | — | — |
| PP | 19 | anatomy_diagram (1건은 pregnancy_safety) | — | `催生符` → safety_refusal |
| OO | 5 | preface_direct | — | — |
| EE | 5 | symptom_variant | — | — |
| **합계** | **34,040** | — | — | **100% 커버, ≥ 76,788 rows 출력** |

## 5. v7 빌더 (`scripts/build_sft_full_corpus.py`) 와의 분리

- v7 빌더는 **유지** (regression test 용)
- v8 빌더는 신규 디렉토리 `scripts/build_sft_v8/` 에 작성
- 두 빌더 산출물은 별개 파일로 저장 → 비교 학습 가능

→ ver8 §A.1 의 Day 0 체크리스트 대로 진행. round_2 와 v8 빌더 작성은 병렬 가능.

## 6. v8 빌더 작성 우선순위 (round_1 결과로 갱신)

ver8/00 §9 Day-by-day 일정에 round_1 결과 반영:

| Day | 작업 | round_1 으로 인한 변경점 |
|---|---|---|
| Day 0 | 디렉토리 + helpers.py 골격 | helpers.py 가 round_1 발견 4종 강제 흡수 (§C.1~§C.4) |
| Day 1 | passage.py + structure.py (대량) | passage cap 제거, 8 prefix |
| Day 2 | prescription.py + herb.py | composition variant + sami_re 신설 |
| Day 3 | acupoint.py + symptom.py | symptom-to-point + 자식 body 스캔 |
| Day 4 | refusal.py + niche.py | pregnancy_safety + PP `催生符` 1건 binding |
| Day 5 | assertions.py + entry.py + --self-test | round_1 §C 4개 self-assert 통합 |
| Day 6 | full build + audit + 비교 | round_2 fixed corpus 와 v8 corpus eval probe 비교 |

총 7일 (ver8 와 동일). round_1 발견은 helpers.py 와 assertions.py 가 흡수하므로 일정 영향 없음.
