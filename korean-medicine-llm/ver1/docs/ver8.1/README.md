# 동의보감 SFT 데이터 v8.1 — 기획서 갱신본

> 작성일: 2026-04-26
> 버전: v8.1 (ver8 의 점진 갱신; ver8 의 § 와 일대일 대응)
> 대응 산출물: `experiments/dongui_bogam/data/sft/phaseB_qa_v8_corpus.jsonl` (목표) / `phaseB_qa_full_corpus_fixed.jsonl` (round_1 잠정)
> 근거: `.claude/harness-evals/sft_quality_fix/round_1/`, `docs/ver8/{00,01,02}.md`

## 왜 ver8.1 이 필요한가

ver8 (2026-04-24) 은 **설계 기획서** 였다 (00_data_construction_plan.md §"구현 상태" 가 모든 산출물을 ❌ 미생성으로 표기). 2026-04-26 에 `phaseB_qa_full_corpus.jsonl` (34,039 rows, book_008 raw 34,040 중 1건만 `\r\n` 부적 PP 레코드를 제외하고 사실상 전수 커버) 에 대해 **sft-quality-fix 하네스 round_1 을 실행**, ver8 §6 의 expected rows 와 §7.2 pass 기준이 **현재 빌더 (`scripts/build_sft_full_corpus.py`)** 산출물에서 어떻게 실측되는지 처음으로 측정할 수 있게 되었다.

ver8.1 은 ver8 기획서에서 다음 두 종류 항목을 **갱신** 한다:

1. **검증된 사실로 대체**: ver8 의 "기대치 / 설계 목표치" 를 round_1 실측치로 대체.
2. **실측 발견된 잔여 결함과 수정 절차 명시**: ver8 에 없던 dosage_leak unit 패턴 미스매치, format_diversity 의 audit-emit field 단절, build patch anchor_not_found 같은 운영 단계 이슈를 정리하고 round_2 백로그로 넘긴다.

ver8 의 §1 (배경)·§2 (6 원칙)·§3 (raw → QA 매핑) 자체 설계 의도는 그대로 유효하므로 ver8.1 에서 재기술하지 않는다. ver8 → ver8.1 의 차이가 의미 있는 부분만 본 폴더 문서가 다룬다.

## 문서 인덱스

| 파일 | 역할 | ver8 대응 |
|---|---|---|
| [00_data_construction_plan.md](./00_data_construction_plan.md) | 갱신된 v8.1 데이터 구축 계획 — 구현 상태, 변경 이유, 잔여 작업 | ver8/00 supersede |
| [01_round_1_audit_and_fix_log.md](./01_round_1_audit_and_fix_log.md) | sft-quality-fix round_1 의 4 phase 산출물 인용·검증 | 신규 (ver8 에 없음) |
| [02_round_2_backlog.md](./02_round_2_backlog.md) | round_1 supervisor 가 지정한 round_2 작업 항목 (round_2 완료 후 historical reference) | 신규 |
| [03_v8_builder_revision_targets.md](./03_v8_builder_revision_targets.md) | ver8 §6 빌더 함수 별 갱신 사양 (round_1 결과 반영) | ver8/00 §6 supersede |
| [04_round_2_log_and_convergence.md](./04_round_2_log_and_convergence.md) | **★ round_2 수렴 보고서 — 학습 데이터 production 진입 완료** | 신규 (시리즈 마감) |

## 1쪽 요약 (시간 없을 때 이것만 읽어도 됨) — **2026-04-26 round_2 수렴 후 갱신**

- **데이터 커버리지**: book_008 raw 34,040 레코드 중 34,039 개가 production 학습 input `phaseB_qa_v8_1_corpus.jsonl` 에 매핑됨 (99.997%). 미포함 1건은 `vol_18 / seq_984 / PP / 催生符` (빈 부적 레코드) — v8 빌더 신설 시 `pregnancy_safety` 1 row 로 emit 하여 100% 도달.
- **품질 (sft-quality-auditor 10 차원, round_2 post)**:
  - **PASS**: schema, literal_quote (0.9933), entity_whitelist (0 deny), length, disclaimer (0.165), format_diversity (q_top 0.5019, 5 unique), near_duplicate (136 pairs), atomic_fact, cot_structure(skip)
  - **WARN**: dosage_leak (22 rows / 0.06%, 모두 고전 인용문 안의 attribution — safety 정책 수용 가능)
  - **FAIL**: 0건
  - **회귀 0 / drop 0** (누적)
- **production 학습 input**:
  - 경로: `experiments/dongui_bogam/data/sft/phaseB_qa_v8_1_corpus.jsonl` (34,039 rows, 92 MB)
  - SHA256: `274c3f9b30e8ee9aad232b680a71603868da9fb5170d9d0da338443bbc021af7`
  - **즉시 학습 가능** (config 의 train_files 만 교체)
- **빌드 스크립트 patch**: 5개 중 2개 (`bp_01_dosage_mask`, `bp_02_disclaimer_pool`) `*.proposed.py` 로 작성, 3개 (`bp_03~05`) 는 들여쓰기 mismatch — **다음 빌드 사이클** manual merge 권고. 학습 input 결정과 직교 (이미 transform 으로 해결).
- **다음 단계**: 학습 시작 → eval probe 측정 → ver8.2 비교 보고서 작성. v8 빌더 (`scripts/build_sft_v8/`) 는 ver8.1/03 청사진으로 별도 트랙 진행.

## 추적 가능성

| 인용 출처 | 본 ver8.1 가 인용하는 핵심 수치 |
|---|---|
| `docs/ver8/00_data_construction_plan.md` | 6 원칙, §3.1 매핑 매트릭스 18 레벨, §6 expected rows 76,788, §7.2 pass 기준 |
| `docs/ver8/01_raw_data_schema.md` | raw 34,040 레코드, 18 content_level, §1.2 field schema, §2.1~2.18 레벨별 샘플 |
| `docs/ver8/02_v7_gap_analysis.md` | §7 14 매핑 방향 gap 표, §6 builder 함수별 정보 손실 매트릭스 |
| `.claude/harness-evals/sft_quality_fix/round_1/01_audit/audit_report.json` | pre-audit 10 차원 verdict |
| `.claude/harness-evals/sft_quality_fix/round_1/03_execute/exec_log.json` | round_1 transform 통계, patches_applied/skipped 내역 |
| `.claude/harness-evals/sft_quality_fix/round_1/04_verify/post_audit_report.json` | post-audit 10 차원 verdict |
| `.claude/harness-evals/sft_quality_fix/round_1/04_verify/verification_report.md` | pre→post diff, cross-check 재측정 |
| `.claude/harness-evals/sft_quality_fix/round_1/supervisor.md` | 재진입 판정 + round_2 지시 |
| `.claude/harness-evals/sft_quality_fix/round_1/iteration_plan.md` | round_2 planner 입력 사양 |

ver8.1 의 모든 주장은 위 11개 파일 중 하나로 근거 추적 가능해야 한다. 실측 외 가설은 § 안에 "🔮 가정" 표시.
