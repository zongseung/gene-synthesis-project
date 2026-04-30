# 04 · sft-quality-fix round_2 — 수렴 보고서

> 작성일: 2026-04-26
> 라운드 디렉토리: `.claude/harness-evals/sft_quality_fix/round_2/`
> 최종 보고서: `.claude/harness-evals/sft_quality_fix/final_report.md`
> **수렴 판정**, 학습 데이터 production 진입 완료.

본 문서는 ver8.1 시리즈의 마지막 라운드 trail. round_1 의 잔여 결함이 round_2 에서 어떻게 해소되었고, 어떤 산출물이 학습 input 으로 채택되었는지 정리.

## 1. round_2 가 시작된 이유 (round_1 잔여 결함 3종)

| code | 차원 | round_1 post 수치 | round_2 SLO |
|---|---|---|---|
| D_residual_1 | dosage_leak | 3,269 rows (9.60%) | ≤ 50 rows |
| D_residual_2 | format_diversity | q_top_rate 1.0 (q_format=null) | q_top_rate ≤ 0.55, ≥ 5 unique |
| D_residual_3 | literal_quote | 0.9933 (10 quote-mask 충돌) | ≥ 0.99 (회귀 0) |

## 2. round_2 fix_plan.yaml 핵심

산출: `.claude/harness-evals/sft_quality_fix/round_2/02_plan/fix_plan.yaml`

### 2.1 file_transforms (2종, deterministic)

**mask_dosage_v2** — round_1 v1 의 6 unit 을 12 unit + SI + 합산 표현으로 확장:

```python
patterns = [
    r"(?<![0-9])\d+(?:\.\d+)?\s*(?:돈|푼|냥|전|알|첩|근|홉|되|말|구)",  # 11 hangul
    r"(?<![0-9])\d+(?:\.\d+)?\s*(?:g|mg|kg|ml|L)\b",                  # SI
    r"각\s*\d+(?:\.\d+)?(?:돈|푼|냥|전|알|첩|근|홉|되|말|구|g|mg)",     # ge_aggregate
    r"\d+편(?![가-힣])",                                              # pyeon count only
]
preserve_quoted_spans = True   # 큰따옴표 / 컬리쿼트 영역 보호
replacement_token = "[용량 표기]"
```

**infer_q_format_from_prefix** — user 첫 발화 prefix 5종 → q_format/a_format enum:

| prefix (12자) | q_format | a_format | rows |
|---|---|---|---:|
| `다음 동의보감 병증 관` | Q_symptom_record | A_symptom_summary | 17,085 |
| `다음 동의보감 표제를 ` | Q_pyeon_label | A_pyeon_summary | 11,078 |
| `다음 동의보감 본문 기` | Q_passage_record | A_passage_summary | 5,319 |
| `다음 동의보감 서문 대` | Q_preface | A_preface_summary | 465 |
| `다음 동의보감 총목·목` | Q_index | A_index_summary | 92 |
| (fallback) | Q_other | A_other | 0 |

### 2.2 build_patches (skip)

`build_patches: []` — round_1 의 bp_03/04/05 anchor_not_found 3건은 manual review 단계로 위임. 학습 input 결정과 직교 (transform 으로 이미 해결).

### 2.3 plan_meta

| field | 값 |
|---|---|
| round | 2 |
| base_input | `round_1/03_execute/phaseB_qa_full_corpus_fixed.jsonl` (34,039 rows) |
| estimated_drop_pct | 0.0 |
| estimated_mask_count | ~3,200 |
| estimated_regenerate_count | 0 |
| user_confirm_required | false |
| drop_budget_remaining_pct | 0.01 |
| verify_gates | 11개 (5 fix + 6 regression) |

## 3. round_2 executor 결과

산출: `.claude/harness-evals/sft_quality_fix/round_2/03_execute/`
- `phaseB_qa_full_corpus_fixed_r2.jsonl` (34,039 rows, 92 MB, SHA256 274c3f9b…)
- `exec_log.json`
- `_dry_run.json`
- `_apply_round2.py` (deterministic, 재현 가능)

### 3.1 transform 통계

| transform | rows touched | substitutions | by_pattern_id |
|---|---:|---:|---|
| mask_dosage_v2 | 3,248 | 4,064 | hangul_units_11: 4,016 / ge_aggregate: 48 / si_units: 0 / pyeon_count_only: 0 |
| infer_q_format_from_prefix | 34,039 | — | (위 표 참조) |

`quoted_span_skips = 43` — round_1 v1 mask 가 인용문 안에서 잘못 치환했던 dosage 토큰을 round_2 가 정확히 보호.

### 3.2 self-check (executor 자체)

| check | 결과 |
|---|---|
| json_parse | 34,039 / 34,039 OK, 0 bad |
| drop_count | 0 |
| dosage residual (auditor 12 unit) | **22 rows / 43 matches** (모두 인용문 안) |
| q_top_rate | 1.0000 → **0.5019** |
| q_format unique | 5 |
| literal_quote 회복 | round_1 의 10 quote-mask mismatch 가 이번 mask 에서는 발생 안 함 (`gate_literal_quote_guard 100%`) |
| _fix_round_history | `[1, 2]` (round_1 메타 누적) |

### 3.3 spec deviation

- plan 은 `messages[0].content` 를 prefix lookup 에 사용하라고 했으나, 본 corpus 의 messages[0] 은 system turn → executor 가 first `role=user` message 로 walk. 100% prefix 매칭 (fallback 0건) → harmless deviation, exec_log 에 기록됨.

## 4. round_2 supervisor 검증 결과

산출: `round_2/04_verify/`, `round_2/supervisor.md`, `final_report.md`

### 4.1 pre → post diff 매트릭스 (round_2 기준)

| 차원 | pre (= round_1 post) | post (= round_2 post) | 결과 |
|---|---|---|---|
| schema | pass | pass | unchanged |
| literal_quote | pass (0.9933) | pass (0.9933) | guard 효과로 회귀 0 |
| entity_whitelist | pass (0) | pass (0) | unchanged ✅ |
| **dosage_leak** | **fail (3,269 / 9.60%)** | **warn (22 / 0.06%)** | **fail→warn** (99.3% 감소) |
| length | pass | pass | unchanged |
| disclaimer | pass (0.167) | pass (0.165) | unchanged |
| **format_diversity** | **fail (1.0)** | **pass (0.5019, 5 unique)** | **fail→pass** |
| near_duplicate | pass (136 pairs) | pass (136 pairs) | unchanged |
| atomic_fact | pass (0) | pass (0) | unchanged |
| cot_structure | skip | skip | n/a |

### 4.2 verify gates (11개 모두 통과)

| gate | 조건 | 측정 | 결과 |
|---|---|---|---|
| gate_dosage_leak | hit_rows ≤ 50 | 22 | ✅ pass |
| gate_format_q_top | q_top_rate < 0.55 | 0.5019 | ✅ pass |
| gate_format_q_unique | unique_count ≥ 5 | 5 | ✅ pass |
| gate_literal_quote_guard | residual dosage 모두 인용 안 | 43/43 | ✅ pass |
| gate_entity_whitelist | deny_hits = 0 | 0 | ✅ pass |
| gate_disclaimer | max_rate ≤ 0.20 | 0.165 | ✅ pass |
| gate_near_duplicate | pair_rate ≤ 5e-4 | 2.8e-5 | ✅ pass |
| gate_length | mean 80~250 tok | 150.7 | ✅ pass |
| gate_drop | drop_pct = 0 | 0.0 | ✅ pass |
| gate_schema | missing_fields = 0 | 0 | ✅ pass |
| gate_atomic_fact | violations = 0 | 0 | ✅ pass |

### 4.3 cross-check (auditor 결과 독립 재현)

supervisor 가 `_crosscheck.py` 로 executor 의 모든 주장을 독립 재측정:
- dosage residual: 22 rows / 43 matches / 43 inside_quotes / 0 outside_quotes ✓
- q_format unique: 5 ✓
- q_top_rate: 0.5019 ✓
- entity deny: 0 (7 names 직접 grep) ✓
- literal_quote: 0.9933 ✓ (round_1 의 0.9933 그대로 유지, 회귀 0)

### 4.4 수렴 5조건 평가

| # | 조건 | 평가 |
|---|---|---|
| 1 | 모든 pre-fail → post pass/warn | ✅ (entity:fail→pass, dosage:fail→warn, format:fail→pass) |
| 2 | 회귀 (pre=pass → post=fail/warn) 0건 | ✅ |
| 3 | drop ≤ 30% | ✅ (0%) |
| 4 | build_patches idempotent | ✅ (적용 0건이라 idempotency N/A) |
| 5 | 잔여 warn 정당화 | ✅ (dosage 22 rows 모두 고전 인용 안, ver5 06_safety §3.2 정책 X) |

→ **수렴**.

## 5. 학습 데이터 production 진입

### 5.1 production 파일

```
experiments/dongui_bogam/data/sft/
├── phaseB_qa_v8_1_corpus.jsonl       ★ 학습 input (34,039 rows, 92 MB)
├── phaseB_qa_v8_1_corpus.stats.json  ★ 메타·통계
├── README.md                          ★ 학습 input 가이드
└── (기타 historical 파일 — 직접 사용 X)
```

SHA256 검증:
```
274c3f9b30e8ee9aad232b680a71603868da9fb5170d9d0da338443bbc021af7  phaseB_qa_v8_1_corpus.jsonl
274c3f9b30e8ee9aad232b680a71603868da9fb5170d9d0da338443bbc021af7  round_2/03_execute/phaseB_qa_full_corpus_fixed_r2.jsonl
```

### 5.2 학습 config 변경

`train_files` 를 `phaseB_qa_v8_1_corpus.jsonl` 로 교체. 행 수·schema 호환성 유지.

### 5.3 inference 시 안전 정책

학습 데이터에 quoted citation 안 dosage 22 rows 잔존 → inference 단계에서 `audit.py` 의 12-unit DOSAGE_PATTERNS 로 모델 출력 한 번 더 필터 권고 (defense-in-depth).

## 6. round_1 + round_2 누적 회고

### 6.1 차원별 3-단계 매트릭스 (v7 → round_1 post → round_2 post)

| 차원 | v7 base | round_1 post | round_2 post | 누적 변화 |
|---|---|---|---|---|
| schema | pass | pass | pass | — |
| literal_quote | 1.0000 | 0.9933 | 0.9933 | -0.0067 (mask 부산물, SLO 위) |
| entity_whitelist | fail (259) | pass (0) | pass (0) | **fail → pass** |
| dosage_leak | fail (5,426 / 15.94%) | fail (3,269 / 9.60%) | warn (22 / 0.06%) | **fail → warn** (99.6% ↓) |
| length | pass | pass | pass | — |
| disclaimer | warn (0.502) | pass (0.167) | pass (0.165) | **warn → pass** |
| format_diversity | fail (1.0) | fail (1.0) | pass (0.5019, 5 unique) | **fail → pass** |
| near_duplicate | 572 pairs | 136 pairs | 136 pairs | -76% (pass 내) |
| atomic_fact | pass | pass | pass | — |
| cot_structure | skip | skip | skip | — |

### 6.2 누적 transform 통계

| metric | round_1 | round_2 | 누적 |
|---|---:|---:|---:|
| input_rows | 34,039 | 34,039 | 34,039 |
| drop_count | 0 | 0 | **0** |
| mask_entity_rows | 258 | 0 | 258 |
| mask_dosage_rows (touched) | 4,778 | 3,248 | ~7,000 (overlap) |
| mask_dosage_substitutions | 12,084 | 4,064 | 16,148 |
| rotate_disclaimer_rows | 28,163 | 0 | 28,163 |
| infer_q_format_rows | 0 | 34,039 | 34,039 |
| infer_format_id_rows | 34,039 | 0 | 34,039 |
| _fix_actions per row (max) | 3 | 5 | 5 |

### 6.3 raw book_008 커버리지

- raw 34,040 records → SFT 34,039 rows = **99.997%**
- 미포함 1건: `vol_18 / seq_984 / PP / 催生符` (trans_ko=`"\r\n"` 빈 부적 레코드)
- v8 빌더 (`scripts/build_sft_v8/`) 가 작성될 때 ver8/00 §3.2.7 대로 `pregnancy_safety` refusal 1 row 로 emit 시 100% 도달.

## 7. ver8.1 시리즈 마무리

| 문서 | 상태 |
|---|---|
| README.md | ✅ |
| 00_data_construction_plan.md | ✅ (round_2 결과 미반영 → §A 표 갱신 권장 — 본 문서 §5 가 그 역할) |
| 01_round_1_audit_and_fix_log.md | ✅ |
| 02_round_2_backlog.md | ✅ (round_2 완료 후 historical reference 가 됨) |
| 03_v8_builder_revision_targets.md | ✅ |
| **04_round_2_log_and_convergence.md** | ✅ (본 문서) |

ver8.1 시리즈는 본 04 문서로 완결. 다음 버전 (ver8.2 또는 ver9) 은 학습 후 환각 비교 보고서 또는 v8 빌더 산출물 보고서가 될 예정.

## 8. 다음 단계 (학습 외 트랙)

### 8.1 build script manual merge (deferred build_patches)

round_1 의 5개 build_patches 중 2개 (`bp_01_dosage_mask_function`, `bp_02_disclaimer_pool`) 가 `*.proposed.py` 로 작성됨. 다음 빌드 사이클을 위해 manual merge 권고:

```bash
# 검토 (예시)
diff scripts/build_sft_full_corpus.py \
     .claude/harness-evals/sft_quality_fix/round_1/03_execute/patches/1_build_sft_full_corpus.py.proposed.py | less

# 적용 시:
cp .claude/harness-evals/sft_quality_fix/round_1/03_execute/patches/2_build_sft_full_corpus.py.proposed.py \
   scripts/build_sft_full_corpus.py
```

bp_03/04/05 는 anchor_not_found 로 proposed.py 도 생성되지 않음 → fix_plan.yaml 의 anchor_old/anchor_new 를 직접 fuzzy match 후 수동 작성.

### 8.2 v8 builder (장기)

ver8.1/03_v8_builder_revision_targets.md 가 청사진. `scripts/build_sft_v8/helpers.py` 에 round_2 의 mask_dosage_v2 / infer_q_format_from_prefix 를 그대로 흡수하면 v8 corpus (76,788 rows 목표) 는 처음부터 audit pass.

### 8.3 학습 후 eval probe

학습 완료 후 ver5 의 Q1~Q19 probe 실행 → 환각 baseline 측정 → round_2 가 처리한 4 차원 (entity / dosage / disclaimer / format) 이 모델 행동에 정착했는지 검증.

ver8.1 산출물의 환각 측정 결과는 ver8.2 또는 ver9 의 1쪽 보고서로 별도 작성 예정.
