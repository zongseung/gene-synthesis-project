# 01 · sft-quality-fix round_1 — audit & fix 실행 로그

> 작성일: 2026-04-26
> 라운드 디렉토리: `.claude/harness-evals/sft_quality_fix/round_1/`
> 대상 파일: `experiments/dongui_bogam/data/sft/phaseB_qa_full_corpus.jsonl` (34,039 rows; book_008 전수 99.997%)
> 수렴 판정: **재진입** (round_2 필요)

본 문서는 ver8.1 의 모든 실측 주장의 출처. round_2 가 끝나면 그대로 02_round_2 로 짝지을 동일 구조의 문서가 추가될 예정.

## 1. round_1 phase 별 요약

```
.claude/harness-evals/sft_quality_fix/round_1/
├── 01_audit/
│   ├── audit_base.json         (sft-quality-auditor 산출 raw)
│   └── audit_report.json       (canonical, planner 입력)
├── 02_plan/
│   └── fix_plan.yaml           (sft-fix-planner 산출, 29.7 KB)
├── 03_execute/
│   ├── phaseB_qa_full_corpus_fixed.jsonl  ★ 산출 SFT corpus (34,039 rows)
│   ├── exec_log.json
│   ├── _dry_run.json
│   ├── _apply_run.json
│   ├── _run_fix.py
│   └── patches/
│       ├── 1_build_sft_full_corpus.py.applied.json + .proposed.py
│       ├── 2_build_sft_full_corpus.py.applied.json + .proposed.py
│       ├── 3_build_sft_qa.py.applied.json
│       ├── 4_build_sft_full_corpus.py.applied.json
│       └── 5_build_sft_full_corpus.py.applied.json
├── 04_verify/
│   ├── post_audit_report.json  (sft-fix-supervisor 가 재실행한 audit)
│   └── verification_report.md
├── supervisor.md               (재진입 판정문)
└── iteration_plan.md           (round_2 입력 사양)
```

## 2. Phase 1 — sft-quality-auditor 결과 (pre-audit)

대상: `phaseB_qa_full_corpus.jsonl` (34,039 rows)
overall verdict: **fail (3 dim fail / 1 dim warn)**

| 차원 | verdict | 핵심 수치 | 원인 |
|---|---|---|---|
| schema | pass | missing=0, bad_msg=0 | — |
| literal_quote | pass | match_rate 1.0 (1501/1501) | 모든 인용 substring 일치 |
| **entity_whitelist** | **fail** | deny_hits 259건 | 7명: 이진 139, 이이 76, 송진 29, 이황 12, 장원소·張元素·장형 각 1 |
| **dosage_leak** | **fail** | hit_rows 5,426 (15.94%) | 12 단위 정규식: 돈/푼/냥/전/알/첩/근/홉/되/말/구/편 |
| length | pass | mean=150.7 tok, ts=0 | — |
| disclaimer | warn | top_rate 0.502 | "17세기 문헌 설명으로 읽어야 합니다." 17,085회 |
| **format_diversity** | **fail** | q/a top_rate 1.0 | q_format/a_format 컬럼 전부 null |
| near_duplicate | pass | 572 pairs / 3,095 sample (1.19e-04) | — |
| atomic_fact | pass | 0 violations | — |
| cot_structure | skip | v4 rows 없음 | — |

전체 raw report: `01_audit/audit_report.json`.

## 3. Phase 2 — sft-fix-planner 산출 fix_plan.yaml

paths: `02_plan/fix_plan.yaml` (29,747 bytes)

핵심 plan_meta:
- `estimated_drop_pct: 0.0015` (≤170 / 34,039; ver8 1% cap 의 15%)
- `estimated_mask_count: 5,635` (5,426 dosage + ~209 entity)
- `estimated_regenerate_count: 0` (LLM regenerate 없음)
- `user_confirm_required: false` (모든 gate 안전)
- `drop_budget_remaining_pct: 0.0085`

전략 분류:
- 20개 entity-deny audit-listed row → `mask_entity` (이진→`[고려·조선 의가]`, 이황→`[조선 학자]`, 송진→`[송대 의가]` 등 name-specific 치환). fallback: drop.
- 5개 dosage audit-listed row → `mask_dosage` (literal patterns_to_mask 명시).
- 잔여 239 entity + 5,421 dosage → `discovery_rules` (executor 가 정규식·literal scan 으로 자동 발견 후 동일 전략 적용).
- 전체 34,039 rows → `rotate_disclaimer` (5-phrase pool, round_robin_seed_42 기반).
- 전체 34,039 rows → `infer_format_id` (category → format_id deterministic map).

build_patches (5 항목):

| ID | target | 내용 |
|---|---|---|
| `bp_01_dosage_mask_function` | `scripts/build_sft_full_corpus.py` | `DOSAGE_LEAK_RE` + `mask_dosage()` 함수 추가 |
| `bp_02_disclaimer_pool` | `scripts/build_sft_full_corpus.py` | 5-phrase `DISCLAIMER_POOL` + `pick_disclaimer()` |
| `bp_03_validate_entities_strict` | `scripts/build_sft_qa.py` | `validate_entities()` 에 must_drop signal 추가 |
| `bp_04_full_corpus_entity_principle` | `scripts/build_sft_full_corpus.py` | `DENY_ENTITIES` set + 원칙 0 check |
| `bp_05_format_id_emit` | `scripts/build_sft_full_corpus.py` | `FORMAT_ID_MAP` 도출 + JSONL 쓰기 직전 inject |

executor contract: read_inputs / write_outputs / 8단계 procedure / 3개 abort_conditions (drop>340, anchor 모호, rotate disclaimer pool 부족).

## 4. Phase 3 — sft-fix-executor 결과

산출: `03_execute/phaseB_qa_full_corpus_fixed.jsonl`, `exec_log.json`, `_dry_run.json`, `patches/`.

### 4.1 transform 통계

| metric | plan | actual | 비고 |
|---|---:|---:|---|
| input_rows | 34,039 | 34,039 | |
| output_rows | 34,039 | 34,039 | drop 0 |
| drop_count | 50 (0.15%) | **0** | mask_entity 의 collapse heuristic 미발동 |
| mask_entity_rows | 209 | **258** | 7명 정확 매칭 (이진 139/이이 76/송진 29/이황 12/기타 2) |
| mask_dosage_rows | 5,426 | **4,778** | -11.9% (executor 정규식 6 unit 만 처리) |
| mask_dosage_replacements | — | **12,084** | 한 row 평균 2.5 패턴 |
| rotate_disclaimer_rows | ~28,720 | **28,163** | exact |
| infer_format_id_rows | 34,039 | **34,039** | exact |
| modified_rows_total | — | 34,039 | 모든 row 가 최소 1개 transform 적용됨 (rotate_disclaimer + infer_format_id 가 전체) |

### 4.2 self-check (executor 자체 검증)

| check | result | gate | 결과 |
|---|---|---|---|
| json_parse | pass (0 bad lines, 34,039 line) | — | ✅ |
| schema_fields_ok | True | — | ✅ |
| dosage_remaining (executor 기준) | 0 | ≤100 | ✅ (executor 정규식 기준) |
| dosage_remaining (auditor 기준) | 3,269 | ≤100 | ❌ (auditor 12 unit 기준) |
| deny_entity_remaining | 0 | =0 | ✅ |
| top_disclaimer_rate | 0.1674 | ≤0.35 | ✅ |

### 4.3 build patches 적용 결과

| patch_id | target | applied | reason |
|---|---|---|---|
| bp_01_dosage_mask_function | build_sft_full_corpus.py | proposed → 1_*.proposed.py | anchor matched 1×, +12 lines |
| bp_02_disclaimer_pool | build_sft_full_corpus.py | proposed → 2_*.proposed.py | anchor matched 1×, +18 lines |
| bp_03_validate_entities_strict | build_sft_qa.py | **skipped** | anchor_not_found (들여쓰기 mismatch), key 코드 라인은 모두 존재 |
| bp_04_full_corpus_entity_principle | build_sft_full_corpus.py | **skipped** | anchor_not_found (동일 원인) |
| bp_05_format_id_emit | build_sft_full_corpus.py | **skipped** | anchor_not_found (`for p in all_pairs:` 5×, 모호) |

→ **2/5 적용**. live build script 는 변경되지 않음. proposed 파일 2개는 사용자 review 후 수동 merge 가능.

### 4.4 WARN_REGRESSION 1건

```
format_id_collapse: all 34039 rows = 'literature_explain'
(source corpus has single category=medical_literature; plan derivation is correct
but diversity will not improve in round_1 from this file alone)
```

→ supervisor 가 §5 에서 별도 처리.

### 4.5 row 변환 sample (audit-listed 첫 행)

```
book_008_vol_01_seq_1036  (mask_entity) — pre 에 '이진' 포함, post 에서 placeholder 치환 + rotate_disclaimer + infer_format_id
book_008_vol_01_seq_1007  (mask_dosage) — '2냥', '1냥', '7돈' → '[용량 표기]' + rotate_disclaimer + infer_format_id
book_008_vol_01_seq_1015  (mask_dosage) — '1.5돈', '1돈', '5푼', '각 1.5돈', '각 1돈' → 모두 [용량 표기] + 위와 동일 transforms
```

`exec_log.json` 의 `row_change_samples` 에 첫 20개 sample 보관.

## 5. Phase 4 — sft-fix-supervisor 검증 결과

산출: `04_verify/post_audit_report.json`, `04_verify/verification_report.md`, `supervisor.md`, `iteration_plan.md`.

### 5.1 pre → post diff 매트릭스

| 차원 | pre verdict | pre 수치 | post verdict | post 수치 | category |
|---|---|---|---|---|---|
| schema | pass | missing=0, bad_msg=0 | pass | missing=0, bad_msg=0 | unchanged |
| literal_quote | pass | 1.000 (1501/1501) | pass | 0.9933 (1491/1501) | **unchanged-with-side-effect** |
| entity_whitelist | **fail** | 259 | **pass** | 0 | **improved** |
| dosage_leak | **fail** | 5,426 (15.94%) | **fail** | 3,269 (9.60%) | partially_improved |
| length | pass | mean=150.7 | pass | mean=150.7 | unchanged |
| disclaimer | warn | 0.502 | **pass** | 0.167 | **improved** |
| format_diversity | **fail** | 1.000 | **fail** | 1.000 | unchanged |
| near_duplicate | pass | 572 pairs | pass | 136 pairs | improved-within-pass |
| atomic_fact | pass | 0 | pass | 0 | unchanged |
| cot_structure | skip | n/a | skip | n/a | n/a |

### 5.2 supervisor 의 cross-check 재측정 (auditor 결과 독립 재현)

| dim | 재측정 방법 | 결과 |
|---|---|---|
| entity_whitelist | grep 으로 7개 deny name 직접 카운트 | `{이진:0, 이이:0, 송진:0, 이황:0, 장원소:0, 張元素:0, 장형:0}`, 합 0 → audit 정확 |
| disclaimer | 6 phrase 직접 카운트 | `{0.1619, 0.1674, 0.1674, 0.1656, 0.1651, 0.1651}` → exec_log 의 0.1674 와 일치, max 0.1674 |
| dosage_leak | auditor 의 12 unit 정규식 재실행 | 3,269 rows. unit 분해: 알 1611 / 첩 1474 / 되 452 / 근 274 / 홉 232 / 말 71 / 구 4 → executor 6 unit 미커버 unit 7개 잔존 |
| format_diversity | `q_format`/`a_format` 컬럼 분포 직접 확인 | 모두 null. format_id 컬럼은 단일 값 'literature_explain'. 의미적 question prefix 분포 5종 (top 0.502) → audit 친화 transform 만 적용되면 pass 가능 |

### 5.3 잔여 결함 3건

| code | 차원 | 증거 | round_2 처리 방향 |
|---|---|---|---|
| D_residual_1 | dosage_leak | 3,269 rows × 7 unit (알/첩/근/홉/되/말/구) executor 정규식 미커버 | mask 정규식 12 unit 동기화 (ver8.1 §C.1) |
| D_residual_2 | format_diversity | q_format/a_format 미수정 (build patch bp_05 미적용) | (A) transform 으로 q_format 채움 OR (B) build patch fuzzy anchor 재시도 — 권장 (A)+(B) 병행 |
| D_residual_3 | literal_quote | 10건 quote 내부에 mask 토큰 삽입 | mask transform 에 `preserve_quoted_spans=True` 가드 추가 (ver8.1 §C.2) |

### 5.4 학습량 영향

- 입력 34,039 rows → 출력 34,039 rows
- drop 0건 (0.0%, ver8 1% cap 의 0%, 30% executor cap 의 0%)
- 학습 손실 0

### 5.5 회귀 발생 여부

- **verdict-level 회귀: 0건** (pre=pass → post=fail/warn 차원 없음)
- side-effect 1건: literal_quote 1.000 → 0.993, 여전히 pass 임계 (0.95) 위. round_2 에서 §C.2 가드로 회복.

### 5.6 최종 판정 (supervisor.md `## 판정`)

> **재진입 (round_2 필요)**
>
> 이유: pre-audit 의 3개 fail 중 2개(`dosage_leak`, `format_diversity`)가 round_1 종료 후에도 fail 상태이다. 회귀(pre=pass → post=warn/fail)는 0건이며 drop 0%이므로 rollback 은 불필요하지만, 수렴 조건 1번 ("모든 pre-fail 차원이 post 에서 pass/warn") 을 만족하지 못한다.

## 6. round_1 의 4개 핵심 인사이트 (ver8.1 §C 강제 조항으로 흡수됨)

1. **dosage 정규식은 audit.py 와 정확히 동기화** 되어야 한다. round_1 처럼 6 unit 만 처리하면 7 unit 잔존 발생. → ver8.1 §C.1
2. **mask transform 은 인용문 영역을 보호** 해야 한다. round_1 에서 10건 mismatch. → ver8.1 §C.2
3. **q_format / a_format 은 audit 가 보는 키이므로 v8 빌더가 emit 의무화**. round_1 의 format_id 추가는 옳지만 audit 가 인식 못 함. → ver8.1 §C.3
4. **build patch 는 anchor-based 가 아닌 helper module import 구조** 로 작성. round_1 의 5/3 anchor_not_found 실패는 들여쓰기 mismatch 가 원인. → ver8.1 §C.4

## 7. 인덱스 파일

`.claude/harness-evals/sft_quality_fix/index.md` 가 자동 생성됨:

```
| round | 날짜 | 입력 | 변환 통계 | verdict |
|---|---|---|---|---|
| round_1 | 2026-04-26 | phaseB_qa_full_corpus (34,039) | drop=0 / mask_entity=259 / mask_dosage=4778 / disclaimer_rotated=28163 | 재진입 |
```

round_2 종료 시 새 row 추가 예정.
