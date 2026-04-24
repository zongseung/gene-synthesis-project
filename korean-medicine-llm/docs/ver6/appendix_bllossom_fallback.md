# ver6 부록 · Bllossom Fallback 경로 (Path A)

- 발동 조건: `00_halluc_repetition_fix_plan.md` §10 참조
- 상태: **대기** (Gemma 경로 실패 시에만 활성화)

---

## A. 구 경로 유지 근거

ver5 Bllossom-8B (CPT + SFT) 파이프라인은 이미 작동 확인됨:
- `outputs/cpt_bllossom_ver5_v3_1/adapter` — SFT 완주 (step 2282, 1 epoch)
- `outputs/hanmed_merged_ver5_v3_1/` — merge 완료 (16GB)
- 임상 답변 품질은 낮으나 **인프라는 작동**. 데이터 재설계로 활용 가능.

## B. Gemma 실패 시 복원 순서

1. `docker compose -f docker-compose.gemma.yml down`
2. `docker compose -f docker-compose.phaseA.merged.yml up -d` (구 서빙 복원)
3. `experiments/dongui_bogam/src/training/sft_trainer.py` 에서 Gemma 패치 revert (git checkout) — bllossom preset 으로 돌림
4. 4,500쌍 ver6 데이터 (clinical + basic + paraphrase + refusal) 를 **Bllossom 에 재학습**
5. LoRA target 축소: `embed_tokens`, `lm_head` 제외
6. CPT 재학습 고려: `book008_real_facts_identity` 를 3 unique → 30 unique × 10 paraphrase 로 확장

## C. 주 수정 포인트 (Gemma 경로와 공통)

- **데이터 4원칙** (§3.1.2 본편) 은 Bllossom 경로에도 동일 적용.
- `entity_whitelist_clinical.yaml` · `build_sft_clinical.py` · `audit_sft_diversity.py` 는 base 와 무관하게 재사용.
- chat template: Bllossom 은 Llama-3 ChatML (`<|start_header_id|>assistant<|end_header_id|>\n\n`) — 기존 `response_template_ids` 유지.

## D. 예상 성능 (Gemma 대비)

| 지표 | Gemma+ver6 예상 | Bllossom+ver6 fallback 예상 |
|---|---|---|
| 임상 변증 hit | ≥ 80% | 50~70% |
| 반복 loop | 0 | 데이터 원칙으로 억제, 잔류 가능 |
| 사실 Q1~Q4 | 3/4 | 3/4 (Bllossom 은 CPT merged 로 Q1 맞췄던 경력) |

## E. 참고 파일

- 구 기획서: `docs/ver5/02_sft_design.md` · `docs/ver5/03_data_pipeline.md`
- round_1 분석: `.claude/harness-evals/phaseB_sft_plan/round_1/iteration_plan.md`
- 구 실험 결과: `outputs/probes/probe_v4_final.jsonl`, `outputs/probes/phaseA_eval.jsonl`
