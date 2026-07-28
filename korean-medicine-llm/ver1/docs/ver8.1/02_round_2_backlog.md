# 02 · sft-quality-fix round_2 백로그

> 작성일: 2026-04-26
> 입력: round_1 supervisor.md `## 다음 라운드 지시` + iteration_plan.md
> 목표 디렉토리: `.claude/harness-evals/sft_quality_fix/round_2/`

본 문서는 round_2 가 시작되면 그대로 planner / executor / supervisor 가 참조해야 할 작업 목록. round_1 의 supervisor 지시를 분해해 plan_yaml row 단위로 변환했다.

## 0. round_2 가 풀어야 할 잔여 결함 3종

| code | 차원 | round_1 post 수치 | round_2 SLO |
|---|---|---|---|
| D_residual_1 | dosage_leak | 3,269 rows (9.60%) | ≤ 50 rows (≤ 0.15%, audit pass) |
| D_residual_2 | format_diversity | q_top_rate 1.0 (q_format=null) | q_top_rate ≤ 0.55, ≥ 5 unique q_format |
| D_residual_3 | literal_quote | match_rate 0.9933 (10 mismatch) | match_rate ≥ 0.99 (회귀 0) |

## 1. round_2 fix_plan.yaml 골격

round_2 planner 는 다음 yaml 조각을 베이스로 작성한다 (iteration_plan.md 에서 발췌·확장):

### 1.1 dosage_leak v2 (최우선)

```yaml
file_transforms:
  - id: mask_dosage_v2
    description: "audit.py 의 12-unit DOSAGE_PATTERNS 와 정확히 동기화"
    pattern_pool:
      # 한글 단위 11종 — round_1 미커버 7종 추가
      - regex: "(?<![0-9])\\d+(?:\\.\\d+)?\\s*(?:돈|푼|냥|전|알|첩|근|홉|되|말|구)"
        replacement: "[용량 표기]"
      # SI 단위
      - regex: "(?<![0-9])\\d+(?:\\.\\d+)?\\s*(?:g|mg|kg|ml|L)\\b"
        replacement: "[용량 표기]"
      # 합산 표현 ("각 N단위", "각 N돈씩")
      - regex: "각\\s*\\d+(?:\\.\\d+)?(?:돈|푼|냥|전|알|첩|근|홉|되|말|구|g|mg)"
        replacement: "각 [용량 표기]"
    # '편' 단위 처리 — 분류 라벨 (내경편/외형편) 과 카운트 (1편/2편) 충돌 회피
    pattern_pool_caveats:
      - guard: "negative lookahead `\\d+편(?![가-힣])`"
        rationale: "'1편의 약재' 는 매칭, '내경편' 은 매칭 안 함"
    preserve_quoted_spans: true   # D_residual_3 가드
    estimated_rows_affected: 3200
    acceptance_gate:
      - "post-audit dosage_leak.hit_rows <= 50"
```

### 1.2 format_diversity 해결 (병행 2-track)

#### A-track — transform-only (즉시 audit 통과)

```yaml
file_transforms:
  - id: infer_q_format_from_prefix
    description: "question prefix 5종 → q_format/a_format 컬럼 채움. 의미적 신호를 audit 가 보는 키로 매핑"
    field_targets: ["q_format", "a_format"]
    prefix_map:
      "다음 동의보감 병증 관":  {q_format: "Q_symptom_record",  a_format: "A_symptom_summary"}
      "다음 동의보감 표제를 ":  {q_format: "Q_pyeon_label",     a_format: "A_pyeon_summary"}
      "다음 동의보감 본문 기":  {q_format: "Q_passage_record",  a_format: "A_passage_summary"}
      "다음 동의보감 서문 대":  {q_format: "Q_preface",         a_format: "A_preface_summary"}
      "다음 동의보감 총목·목":  {q_format: "Q_index",           a_format: "A_index_summary"}
    fallback:
      q_format: "Q_other"
      a_format: "A_other"
    estimated_rows_affected: 34039
    acceptance_gate:
      - "post-audit format_diversity.q_top_rate < 0.55"
      - "post-audit format_diversity.q_format unique_count >= 5"
```

#### B-track — build script 영구 fix

```yaml
build_patches:
  - id: bp_05_format_id_emit_v2
    target: scripts/build_sft_full_corpus.py
    strategy: "round_1 patches/2_build_sft_full_corpus.py.proposed.py 를 reference 로 anchor 재추출 (whitespace ignored fuzzy match)"
    reason: "round_1 anchor_not_found 의 원인은 들여쓰기 차이. proposed.py 는 정상 동작."
    applied_via: "manual review (사용자에게 diff 제시 후 승인)"
    rollback_safe: true
    coordination_note: "A-track 적용으로 audit pass 후, B-track 은 다음 빌드의 정상 emit 보장용. 두 track 충돌 없음 (transform 은 row 단위, patch 는 build 단위)."

  - id: bp_03_validate_entities_strict_v2
    target: scripts/build_sft_qa.py
    strategy: "fuzzy anchor (whitespace ignored)"
    rollback_safe: true

  - id: bp_04_full_corpus_entity_principle_v2
    target: scripts/build_sft_full_corpus.py
    strategy: "fuzzy anchor (whitespace ignored)"
    rollback_safe: true
```

### 1.3 literal_quote 회귀 가드

```yaml
verify_gates:
  - "literal_quote.match_rate >= 0.99"   # round_1 가 0.993 → 회귀 시 즉시 abort
  - "near_duplicate.pair_rate <= 5e-4"   # 현재 2.8e-05
  - "disclaimer.max_rate <= 0.20"        # 현재 0.167
  - "entity_whitelist.deny_hits_total == 0"
```

§1.1 의 `preserve_quoted_spans: true` 가 있으면 자연 충족 예상.

### 1.4 plan_meta

```yaml
plan_meta:
  round: 2
  base_input: ".../round_1/03_execute/phaseB_qa_full_corpus_fixed.jsonl"
  estimated_drop_pct: 0.0       # mask 만, drop 없음
  estimated_mask_count: 3200    # dosage v2 추가만
  estimated_regenerate_count: 0
  user_confirm_required: false
  drop_budget_remaining_pct: 0.01   # round_1 0% 사용 후 cap 1% 그대로
```

## 2. round_2 expected outcome

| 차원 | round_1 post | round_2 target |
|---|---|---|
| schema | pass | pass |
| literal_quote | 0.993 | ≥ 0.99 |
| entity_whitelist | 0 | 0 (유지) |
| **dosage_leak** | **3,269 (fail)** | **≤ 50 (pass)** |
| length | mean 150.7 | mean ≈ 150 (변동 없음) |
| disclaimer | 0.167 | ≤ 0.20 |
| **format_diversity** | **1.0 (fail)** | **≤ 0.55 (pass)** |
| near_duplicate | 136 pairs | ≤ 5e-4 |
| atomic_fact | 0 | 0 |
| cot_structure | skip | skip |

→ 모든 차원 pass / pass-with-skip → **round_2 supervisor 가 수렴 판정 → final_report.md 작성** 예상.

## 3. round_2 진행 전 사용자가 결정해야 할 항목

### Q1. round_1 의 proposed.py 2개를 live build script 에 merge 하시겠습니까?

- 파일: `round_1/03_execute/patches/1_build_sft_full_corpus.py.proposed.py`, `2_build_sft_full_corpus.py.proposed.py`
- 변경 분량: 1번 +12 lines (DOSAGE_LEAK_RE + mask_dosage()), 2번 +18 lines (5-phrase DISCLAIMER_POOL + pick_disclaimer())
- 검토 방법:
  ```bash
  diff scripts/build_sft_full_corpus.py \
       .claude/harness-evals/sft_quality_fix/round_1/03_execute/patches/1_build_sft_full_corpus.py.proposed.py
  ```
- 권장: ✅ merge. 다음 빌드부터 같은 결함 자동 방지.

### Q2. dosage `편` 단위 처리 정책 확정

- ver8 §3.2.7 의 XX `凡四十一種.` ("총 41 종" 종 표기) 는 mask 안 함이 맞음 (분류 정보).
- "1편의 약재" / "2편" 같은 카운트는 mask?
- 권장: "X편(?!\[가-힣]\)" 으로 mask (label 과 충돌 회피).

### Q3. round_2 가 끝나면 학습을 시작하시겠습니까?

- round_2 산출물 `phaseB_qa_full_corpus_fixed_r2.jsonl` 이 ver8 §7.2 SLO 모두 만족 시 → 학습 input 채택 권고.
- 그렇지 않으면 round_3 (최대) 필요. round_3 은 환각 데이터 패턴이 더 미세할 때만 의미가 있어 보통 불필요.

## 4. round_2 실행 명령

본 백로그를 입력으로 사용해 round_2 실행:

```bash
# 사용자 프롬프트 형식:
# /sft-quality-fix
# round_1 의 supervisor.md 와 iteration_plan.md 그리고 ver8.1/02_round_2_backlog.md 를 입력으로
# round_2 를 시작해. base_input 은 round_1/03_execute/phaseB_qa_full_corpus_fixed.jsonl.
```

또는 manual:
```bash
mkdir -p .claude/harness-evals/sft_quality_fix/round_2/{01_audit,02_plan,03_execute/patches,04_verify}
# then invoke sft-fix-planner with audit_json = round_1/04_verify/post_audit_report.json
# (round_2 는 audit 단계 skip 가능 — round_1 의 post_audit 이 round_2 의 pre_audit)
```

## 5. round_3 가 필요할 가능성

낮음. round_2 의 transform 은 모두 정규식 기반 deterministic 이라 fail 시나리오는 적음.

발생 가능 시나리오:
- A. `편` 정규식이 "외형편/내경편" 등 분류 라벨까지 mask → format_diversity 회귀 → round_3 에서 negative lookahead 강화
- B. `preserve_quoted_spans` 구현이 nested quote 처리 실패 → round_3 에서 규칙 정밀화

이 두 시나리오 모두 round_2 verify_gates 의 `literal_quote ≥ 0.99` 와 `q_format unique_count ≥ 5` 로 자동 검출됨.

## 6. v8 빌더 트랙과의 관계

ver8 §8 의 Agent-P/A/S/M/R 분할에 따른 `scripts/build_sft_v8/` 구축은 **본 round_2 와 병렬 진행 가능**. 두 작업은 별개 파일을 건드리므로 충돌 없음:
- round_2 → `phaseB_qa_full_corpus_fixed_r2.jsonl` (현 v7 빌더 산출물의 정제판)
- v8 빌더 → `phaseB_qa_v8_corpus.jsonl` (신규 76,788 rows, 14 매핑 방향)

학습 input 채택 우선순위:
1. 즉시 (1~2일 내): `phaseB_qa_full_corpus_fixed_r2.jsonl`
2. 중기 (1~2주): `phaseB_qa_v8_corpus.jsonl`
3. 두 input 으로 학습한 모델의 환각 비교 → ver8.1 보고서 후속 (또는 ver9)
