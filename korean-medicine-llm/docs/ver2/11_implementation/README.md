# 11. Implementation Plan — Core 14 → Stage 1 CPT → Demo v0 (R3.5 정합)

> 이 섹션은 **기획(§01~§10) 을 코드로 옮기는 작업 지도**. 수집 완료된 Core 14 데이터가 Bllossom-8B primary 파이프라인에서 어떻게 처리되고, M2 에서 어떤 스크립트가 어떤 순서로 구현되는지를 한 곳에서 본다.

## 섹션 인덱스

| # | 파일 | 용도 |
|---|---|---|
| 11.1 | [pipeline_data_flow.md](pipeline_data_flow.md) | 단계 0~5 데이터 플로우, Bllossom 실측 기반 cap 재산정, 각 단계 입출력 schema |
| 11.2 | [work_order.md](work_order.md) | M2 구현 순서 7개, 각 작업 spec (LoC·의존·gate) |
| 11.3 | [current_state.md](current_state.md) | 수집·코드·asset 현황 스냅샷 (실측 테이블) |

## 한 줄 요약

**Core 14 HanMed unique = 2.72M tokens (Bllossom)** → epoch 3~5 × 40% mix → **total cap 20~34M**. 단일 A6000 × 2 DDP 에서 pilot 1회 ~50분. Stage 1 CPT adapter 후 `hanmed chat` REPL 데모 가능.

## M2 진입 조건 (blocker 요약)

| # | 조건 | 상태 | 해결 경로 |
|---|---|---|---|
| G-DATA | Core 14 수집본 확보 | ✅ 완료 | — |
| G-DATA+ | Core 25 확장 (권장) | ❌ 중단 | `mediclassics_orchestrator.py --books 7,44,46,47,49,54,60,70,71,94,139,183` |
| G-EVAL | `eval/hanmed_eval_v0/{T1,T2,T4,T5}.jsonl` + hashes | 🟡 T1 1 샘플 placeholder only | 전문가 curation (§05) |
| G-SCRIPT | `tokenizer_extend.py`, `build_corpus_manifest.py`, `cpt_trainer.py` | ❌ 미구현 | §11.2 work_order |
| G-DRIFT | preprocess.py B1/B3/B4 + M1~M6 수정 | ❌ 미적용 | §11.2 work_order |

## 관련 문서

- 모델: [`04_model_strategy/base_model_and_training.md`](../04_model_strategy/base_model_and_training.md) — Bllossom primary 전환 (§4.2)
- 파이프라인: [`04_model_strategy/preprocessing_and_cpt_spec.md`](../04_model_strategy/preprocessing_and_cpt_spec.md) — §C Stage 1 CPT 상세
- 데모: [`10_demo_cli/README.md`](../10_demo_cli/README.md) — hanmed CLI v0
- 평가: [`05_evaluation/hanmed_eval.md`](../05_evaluation/hanmed_eval.md) — T1~T5
- 하네스 이력: `.claude/harness-evals/hanmed-cpt-spec/` — R1~R3.5 검증 로그

## 변경 이력

| 버전 | 날짜 | 변경 |
|---|---|---|
| R3.5 | 2026-04-16 | 11_implementation 섹션 신설. ver2 기획 R1~R3.5 수렴 반영 |
