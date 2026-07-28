# 동의보감 SFT 데이터 v8.1 데이터 구축 계획 (갱신본)

> 작성일: 2026-04-26
> ver8 의 `00_data_construction_plan.md` 를 round_1 실측 결과로 갱신.
> ver8 의 §1 (배경)·§2 (6 원칙)·§4 (신규 카테고리)·§5 (전처리 choke point)·§8 (Agent 분할) 본문은 그대로 유효 — 변경된 §3 / §6 / §7 만 본 문서가 다룬다.
> 변경 사유는 모두 round_1 실측이며, file:line 또는 harness-evals 산출물로 추적 가능.

---

## A. 구현 상태 (2026-04-26 기준 — IMPLEMENTATION STATUS 갱신)

ver8 §"구현 상태" 표를 round_1 직후 상태로 대체:

| 항목 | ver8 (04-24) | ver8.1 (04-26) | 경로 / 비고 |
|---|---|---|---|
| 기획서 (00) | ✅ 작성 완료 | ✅ → 본 문서로 supersede | `docs/ver8.1/00_data_construction_plan.md` |
| raw schema 분석 (01) | ✅ | ✅ 유효 (변경 없음) | `docs/ver8/01_raw_data_schema.md` |
| v7 gap 분석 (02) | ✅ | ✅ 유효 | `docs/ver8/02_v7_gap_analysis.md` |
| v7 빌더 (참고) | ✅ | ✅ | `scripts/build_sft_full_corpus.py` |
| v7 post-processor | ✅ | ✅ | `scripts/augment_sft_v7.py` |
| **v8 빌더 디렉토리** | ❌ | ❌ 여전히 미생성 | `scripts/build_sft_v8/` (ver8/00 §A.1 항목, 본 round 미진행) |
| **v8 빌더 entry point** | ❌ | ❌ 여전히 미작성 | `scripts/build_sft_full_corpus_v8.py` |
| **v8 corpus 산출물** | ❌ | ❌ 여전히 미생성 | `experiments/dongui_bogam/data/sft/phaseB_qa_v8_corpus.jsonl` |
| **v7 corpus audit** | ❌ | ✅ **완료 (round_1 pre-audit)** | `.claude/harness-evals/sft_quality_fix/round_1/01_audit/audit_report.json` (대상: phaseB_qa_full_corpus.jsonl) |
| **v7 corpus quality fix (round_1)** | — | ✅ **완료, verdict 재진입** | `phaseB_qa_full_corpus_fixed.jsonl` 산출 |
| **v7 corpus quality fix (round_2)** | — | ✅ **완료, verdict 수렴** | `phaseB_qa_full_corpus_fixed_r2.jsonl` 산출, 04_round_2_log_and_convergence.md 참조 |
| **production 학습 input 진입** | — | ✅ **완료** | `experiments/dongui_bogam/data/sft/phaseB_qa_v8_1_corpus.jsonl` (34,039 rows, SHA256 274c3f9b…) |
| **build_sft_full_corpus.py 5개 patches** | — | 2/5 proposed (수동 review 대기), 3/5 anchor_not_found | `round_1/03_execute/patches/` |
| **v8 adapter 학습 결과** | ❌ | ❌ | `experiments/dongui_bogam/outputs_ver8_gemma_v1/` |

따라서 ver8.1 의 §6 expected rows 와 §7.2 pass 기준 중 **round_1 이 측정한 차원만 실측** 이며, 나머지는 ver8 의 설계 목표치를 그대로 유지한다 (변동 없음).

---

## B. ver8 §6 "expected rows" 와 round_1 실측치의 비교

ver8 은 v8 빌더가 만들 76,788 rows 를 목표로 했으나 round_1 은 v7 빌더 산출물 (`phaseB_qa_full_corpus.jsonl`, 34,039 rows) 을 audit 했다. 따라서 직접 비교는 v7 vs ver8 목표 비교다.

| 항목 | v7 실측 (현 corpus) | ver8 목표 | gap (v8 - v7) |
|---|---:|---:|---:|
| 입력 raw 레코드 | 34,040 | 34,040 | 0 |
| QA rows 출력 | 34,039 | 76,788 | +42,749 |
| 커버리지 (raw → QA) | 99.997% (1건 PP `催生符` 미포함, ver8 §3.2.7 safety 처리 대상) | 100% | +1 record (=safety refusal 1 row) |
| 처방→조성 direct QA | 2 (실질 0) | ~6,039 (DP+EP × 1 variant) | +6,037 |
| 증상→처방 inverse QA | 1,997 (의미 부정확) | ~1,103 (DD × 1 variant, 자식 본문 명시 처방만) | -894 (그러나 의미 정확도 ↑) |
| 약재→성미 direct QA | 0 (HERB Q_TEMPLATES 누락) | ~1,403 (CH+DH 1 variant) | +1,403 |
| passage rows | 7,000 (`build_passage_pairs` hard-cap) | ~45,478 (cap 제거 + 2 variant) | +38,478 |
| TT/PP/EE/CP rows | 0 | 51 + parents | +~100 |

→ ver8.1 은 ver8 목표 76,788 rows 를 그대로 유지. 그러나 round_1 의 dosage_leak 잔존 (9.6%) 과 format_diversity 미해결 발견은 **v8 빌더 사양에 추가 강제 조항** 으로 들어가야 한다 (다음 §C 참고).

---

## C. v8 빌더 사양에 새로 추가되는 강제 조항 (round_1 발견 반영)

ver8 §3.2 의 각 레벨별 매핑 표에 다음 4개 강제 조항을 **v8 빌더 entry point 가 빌드 종료 직전에 self-assert** 한다.

### C.1 dosage_leak 12-unit 마스킹 (audit 동기화)

ver8 은 §2 원칙 5 (safety) 에서 "용량 직접 generation 금지" 만 언급. round_1 은 다음을 추가 발견:

- v7 corpus 의 `assistant` 본문에 12 단위 (`돈/푼/냥/전/알/첩/근/홉/되/말/구/편`) 의 dosage 패턴이 5,426행 (15.94%) 에 산재.
- round_1 mask 는 `(돈/푼/냥/전/g/mg)` 6 단위만 적용 → 잔여 3,269행 (9.60%) 미처리.

→ **v8.1 강제 조항 §C.1**: v8 빌더는 모든 `assistant` 본문에 대해 audit.py 의 `DOSAGE_PATTERNS` (12 단위) 와 동일한 정규식으로 mask 를 적용해야 한다. 빌드 종료 직전 self-assert:

```python
# build_sft_full_corpus_v8.py 말미
import re
DOSAGE_RE = re.compile(r"\d+(?:\.\d+)?\s*(?:돈|푼|냥|전|알|첩|근|홉|되|말|구)|\d+(?:\.\d+)?\s*(?:g|mg|kg|ml|L)\b")
leak_rows = sum(1 for qa in qa_rows if DOSAGE_RE.search(qa['messages'][-1]['content']))
LEAK_HARD_CAP = 50  # = audit.py FAIL 기준 (>50)
assert leak_rows <= LEAK_HARD_CAP, f"dosage_leak v8 SLO violated: {leak_rows} rows leaked (cap={LEAK_HARD_CAP})"
```

`편` 단위는 `내경편/외형편` 같은 분류 라벨과 충돌하므로 v8 빌더는 negative lookahead `\d+편(?!(?:\s|$|[^가-힣]))` 또는 어휘 화이트리스트 (`{1편, 2편, 3편}` 같은 카운트만 매칭) 를 사용한다.

### C.2 인용문 영역 mask 보호 (literal_quote 회귀 가드)

round_1 mask 가 인용문 (큰따옴표 `"..."` 영역) 안의 패턴까지 치환해 literal_quote.match_rate 가 1.000 → 0.993 으로 저하 (10건 mismatch). 여전히 pass 임계 (0.95) 위지만 정책 위반.

→ **v8.1 강제 조항 §C.2**: v8 빌더의 모든 mask transform 은 `preserve_quoted_spans=True` 모드를 기본값으로. 인용문 영역 안의 dosage 패턴은 mask 하지 않거나, mask 시 인용문 외부에서만 치환. 인용문 내 dosage 가 safety 위험 행이면 그 row 는 통째로 drop (record 화 X).

빌드 후 self-assert:
```python
assert literal_quote_match_rate(qa_rows, raw_records) >= 0.99, "literal_quote regression"
```

### C.3 q_format / a_format 컬럼 강제 emit (format_diversity)

round_1 발견: v7 빌더는 `q_format`/`a_format` 키를 emit 하지 않으므로 audit 가 100% `unknown` 으로 보고 → fail. round_1 executor 가 추가한 `format_id` 는 audit 가 보지 않는 키.

→ **v8.1 강제 조항 §C.3**: v8 빌더는 모든 row 에 `q_format` 과 `a_format` 두 키를 반드시 채워 emit. ver8 §3 의 18 레벨별 매핑 카테고리와 1:1 대응되는 enum 값으로:

```python
Q_FORMATS = Literal[
    "Q_passage_summary", "Q_passage_topic",
    "Q_prescription_efficacy", "Q_prescription_composition", "Q_prescription_source",
    "Q_acupoint_meridian", "Q_acupoint_location", "Q_acupoint_indication", "Q_acupoint_symptom",
    "Q_herb_sami", "Q_herb_indication", "Q_herb_classification",
    "Q_symptom_pathology", "Q_symptom_inverse_rx", "Q_symptom_inverse_acu",
    "Q_structure", "Q_meta_toc",
    "Q_diagram_explain", "Q_table_explain",
    "Q_refusal_oos", "Q_refusal_safety", "Q_pregnancy_safety",
    "Q_paraphrase",
]
```

빌드 종료 시 self-assert:
```python
from collections import Counter
qf = Counter(qa['q_format'] for qa in qa_rows)
top_rate = max(qf.values()) / sum(qf.values())
assert top_rate <= 0.55, f"q_format top_rate={top_rate:.3f} violates v8 SLO (≤0.55)"
assert len(qf) >= 10, f"q_format diversity insufficient: {len(qf)} unique"
```

audit 측에서도 `q_format`/`a_format` 가 enum 값을 가질 때 카운트하도록 audit.py 를 호환 (이는 audit 측 이슈이므로 ver8.1 이 의무는 아니지만 round_2 에서 같이 고친다).

### C.4 build_sft_v8 의 patches_applied self-validation

ver8 §A.1 가 명시한 `scripts/build_sft_v8/` 디렉토리는 round_1 시점에 미생성. round_1 의 5개 build patches 중 3개가 anchor_not_found 로 실패한 사실로 보아, v8 빌더는 처음부터 다음을 만족하도록 작성:

- 모든 helper (`mask_dosage`, `pick_disclaimer`, `validate_entities`, `infer_format_id`) 를 한 모듈 (`scripts/build_sft_v8/helpers.py`) 에 모아 명시적 import 로 사용. 인라인 anchor 패치 가정 X.
- entry point (`scripts/build_sft_full_corpus_v8.py`) 가 `--self-test` 플래그로 단일 vol 100 row sample 빌드 + 4개 self-assert 를 즉시 실행할 수 있어야 한다.
- 빌드 메타 (`phaseB_qa_v8_corpus.meta.json`) 에 다음을 emit: `helper_versions`, `dosage_pattern_set`, `disclaimer_pool`, `q_format_enum_size`, `assert_results: {coverage, dosage_leak, literal_quote, q_format_diversity}`.

---

## D. ver8 §7.2 "pass 기준" 갱신 — 측정 가능 형식

ver8 §7.2 의 pass 기준을 audit 측 verdict 에 직접 매핑한 형태로 다시 쓴다 (round_2 supervisor 가 그대로 복사 가능):

| ver8 §7.2 항목 | audit 차원 | v8 SLO | v7 base | round_1 post | **round_2 post (final)** | 비고 |
|---|---|---|---:|---:|---:|---|
| 전수 커버리지 | (외부 assert) | 100.000% (34,040/34,040) | 99.997% (1 PP record 누락) | 99.997% | **99.997%** | v8 빌더 신설 시 PP `催生符` 1건 pregnancy_safety 로 emit → 100% |
| 스키마 무결성 | schema | pass (missing=0, bad_msg=0) | pass | pass | **pass** | ✅ |
| 원문 인용 무결 | literal_quote | match_rate ≥ 0.99 | 1.000 | 0.9933 | **0.9933** | round_2 quoted-span guard 로 회귀 0 |
| 인물 화이트리스트 | entity_whitelist | deny_hits = 0 | 259 (FAIL) | 0 ✅ | **0** ✅ | round_1 에서 해결, round_2 유지 |
| 용량 누출 | dosage_leak | hit_rows ≤ 50 | 5,426 (FAIL) | 3,269 (FAIL) | **22 (WARN)** | round_2 12-unit mask + quoted-span guard. 잔여 22 모두 인용문 안 attribution |
| 답변 길이 | length | mean 80~250 tok, too_short=0 | mean=150.7 | mean=150.7 | **mean=150.7** | ✅ |
| 면책 다양성 | disclaimer | top_rate ≤ 0.40 | 0.502 (WARN) | 0.167 ✅ | **0.165** ✅ | round_1 6-phrase pool 균등 분포 |
| 포맷 다양성 | format_diversity | q_top_rate ≤ 0.55 | 1.0 (FAIL) | 1.0 (FAIL) | **0.5019, 5 unique** ✅ | round_2 infer_q_format_from_prefix 로 해결 |
| 근사 중복 | near_duplicate | pair_rate ≤ 5e-4 | 1.19e-04 | 2.8e-05 | **2.8e-05** | ✅ |
| 원자 사실 | atomic_fact | violations=0 | 0 | 0 | **0** | ✅ |
| CoT 구조 | cot_structure | (v4 rows 없으면 skip) | skip | skip | **skip** | n/a |

**round_2 결론**: 0 fail / 1 warn (dosage_leak 22 rows, 모두 고전 인용 안). 학습 input 으로 채택 가능 → **`phaseB_qa_v8_1_corpus.jsonl` 로 production 진입 완료**. v8 빌더 (`scripts/build_sft_v8/`) 가 작성될 때 §C 강제 조항으로 같은 SLO 자동 보장.

---

## E. 학습 input 결정 (단기 / 중기 / 장기)

### 단기 (v8 빌더 미생성 상태에서 즉시 학습 시작 가능 path)

| 옵션 | 산출 파일 | 장점 | 단점 / 잔여 위험 |
|---|---|---|---|
| **A** | `experiments/dongui_bogam/data/sft/phaseB_qa_full_corpus.jsonl` (원본) | 빌드 재실행 불필요 | entity_whitelist 259 deny / dosage_leak 5,426 / disclaimer 50.19% — **safety/style collapse 재현 가능성 100%** |
| **B (권장)** | `.claude/harness-evals/sft_quality_fix/round_1/03_execute/phaseB_qa_full_corpus_fixed.jsonl` | round_1 fix 반영 (entity 0, disclaimer 16.7%) | dosage_leak 9.6% 잔존, q_format=null. 학습 데이터로는 사용 가능하나 환각·collapse 측정 시 dosage 변수 통제 필요 |
| C | round_2 산출물 (미생성) | 모든 차원 SLO 만족 | round_2 미실행 — 02_round_2_backlog 가 진행되어야 함 |
| D | v8 corpus (미생성) | ver8 76,788 rows + 14 매핑 방향 inverse + niche level 흡수 | v8 빌더 미작성, ETA 3-5 일 |

→ **즉시 학습 권고**: 옵션 B (round_1 fixed) 로 baseline 학습 1회 완료 + eval probe 실행 → round_2 결과 나오면 옵션 C 로 재학습 비교. v8 빌더는 별도 트랙으로 진행.

### 중기 (round_2 완료 후, ~2~3 일)

옵션 C 로 학습. round_2 의 expected outcome (iteration_plan.md §"round_2 expected outcome") 에 따르면:
- dosage_leak ≤ 50 (현 3,269)
- format_diversity q_top_rate ≤ 0.55 (현 1.0)
- 다른 차원 모두 유지

이 수준이면 ver8 §7.2 의 모든 pass 기준 만족 → **production 학습 input 으로 채택 가능**.

### 장기 (v8 빌더 완료 후, ~1~2 주)

옵션 D 로 학습. 76,788 rows 가 14 매핑 방향 inverse 를 포함하므로 hanmed_cpt round_3 final_report 에서 보고된 환각 패턴 (Q2/Q4/Q9/Q10) 의 근본 원인 7가지 중 6가지가 데이터 레벨에서 해소.

---

## F. round_2 / v8 빌더로의 transition checklist

ver8.1 채택 후 다음 순서로 진행:

1. ✅ ver8.1 작성 (본 문서) — **현재 단계**
2. ⏭ round_2 실행 (02_round_2_backlog.md 참고)
3. ⏭ round_2 supervisor 가 수렴 판정 → `phaseB_qa_full_corpus_fixed_r2.jsonl` 학습 input 채택
4. ⏭ baseline SFT 학습 1회 + eval probe Q1~Q19 측정 → 환각 baseline 기록
5. ⏭ ver8/00 §A.1 의 `scripts/build_sft_v8/` 디렉토리 신규 작성 시작 (Agent-P/A/S/M/R 분할 — ver8/00 §8 그대로 유효)
6. ⏭ v8 빌더 산출물에 ver8.1 §C 4개 강제 조항 self-assert 통합
7. ⏭ v8 corpus 학습 + eval probe 비교 → 환각 비교 보고서 작성

---

## 부록 G. ver8 본문에서 변경되지 않은 부분

본 문서가 명시적으로 갱신하지 않은 ver8 의 다음 항목은 **그대로 유효**:

- ver8/00 §1 (배경 및 문제 의식) — v6 → v7 실패 요약, 환각 근본 원인 랭킹
- ver8/00 §2 (설계 원칙 6 조) — 원칙 0~5 모두 유효
- ver8/00 §3 (raw → QA 매핑 18 레벨) — §3.1 매트릭스 + §3.2.1~3.2.7 상세 모두 유효
- ver8/00 §4 (신규 카테고리 — inverse / cross / meta)
- ver8/00 §5 (전처리 choke point `load_raw_record_clean`)
- ver8/00 §8 (Agent 분할 — Agent-P/A/S/M/R)
- ver8/00 §9 (Day-by-day 구축 일정)
- ver8/01 (raw schema forensic, 34,040 전수)
- ver8/02 (v7 gap 분석 14 매핑 방향)

ver8.1 은 위 항목을 **재작성하지 않고** ver8 본문을 그대로 인용한다 (§ 참조로). 두 문서를 함께 읽어야 v8 전체 사양이 완성된다.
