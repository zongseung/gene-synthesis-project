# ver4 — Expository Knowledge-Injection CPT (SFT·RAG-free)

**상태**: r2 (2026-04-20, 데이터 수집 감사 반영) · EXP-V4-00 resume 크롤 완료 (2026-04-20 13:47, 120,235 records / 26권 DONE)
**계기**:
- ver2.2 R3.5 adapter의 factual probe 환각률 **3/4 (75%)** 실증
- ver2 README의 "Core 14 수집 완료 ✅"는 사실이 아니며 **핵심 5권(동의보감·향약집성방 포함)이 크롤 중단 상태**로 저장되어 있음이 실측됨 (2026-04-20)

**제약**:
- SFT 금지 (답변 길이 collapse 우려)
- RAG는 측정용만 (배포 후보 아님)
- → **CPT 단일 paradigm**

---

## 문서 인덱스

| # | 파일 | 내용 |
|---|---|---|
| 01 | [`01_validation_report.md`](01_validation_report.md) | 하네스 round_1 검증 — 환각 증거 / 코퍼스 실측 / 전처리 결함 / 학습 버그 / "메타 prefix" 처방 실패 예측 |
| 02 | [`02_plan_v4.md`](02_plan_v4.md) | ver4 r1 기획서 — P-A+ (Expository KI-CPT) 단일 경로, EXP-V4-01/02/03/05/06 |
| 03 | [`03_serving_and_cli/`](03_serving_and_cli/README.md) | EXP-V4-03 이후 M2 — vLLM + Docker 서빙, `hanmed` 쉘 엔트리 · Remote backend 구현 기획 |
| 04 | [`04_dead_code_audit.md`](04_dead_code_audit.md) | 저장소 전체 dead code / legacy / one-shot 스크립트 정리 후보. Tier 1~5 삭제 계획 |

## 하네스 원본 산출물 (round_1)

- `.claude/harness-evals/hanmed_cpt/round_1/generator.md`
- `.claude/harness-evals/hanmed_cpt/round_1/discriminator.md`
- `.claude/harness-evals/hanmed-cpt-spec/round_3/reviewer.md`
- `.claude/harness-evals/hanmed_cpt/round_1/iteration_plan.md`

## 한눈 요약

**무엇이 문제였나**:
1. 코퍼스에 "책↔저자↔왕대" triple 명시 없음 (`extract_corpora.py`가 서지 메타 탈락)
2. entity 빈도 14:1 비대칭 (이시진 101 vs 허준 7 vs 이제마 2) → Chinese prior 항상 승리
3. book 경계 무시 greedy pack → prefix 주입 처방도 shortcut learning으로 실패
4. 학습 코드 5개 버그 (modules_to_save 누락 / epoch=1 / best_model 없음 / save-eval 불일치 / LR 바닥)
5. contamination hash gate 포맷 mismatch

**무엇을 바꾸나 (ver4 r1 확정)**:
- ❌ 폐기: ver2 전체 / 1차 진단의 block prefix 주입 / ver4 r0의 SFT(P-B)·RAG-deployment(P-C)
- ✅ **P-A+ (Expository Knowledge-Injection CPT)** 단일 채택:
  - `data/facts/core_factsheet.yaml` — 사람 검증 fact sheet (권당 10~15 triple × N권 = 10N~15N, N은 EXP-00 이후 완주 검증 권 수)
  - `src/data/synth/expand_facts.py` — template × paraphrase ×4, entity validation
  - 책당 75 paragraph × 4 paraphrase × 300 tok = 권당 ~0.09M → **N=21 기준 ~1.89M 합성 tokens** (N은 EXP-00 이후 완주 검증 권 수, 상세 표는 02_plan §2.1)
  - mix: synth_facts 25% / bilingual 35% / zh 15% / ko 25% (N=21 기준, N에 따라 ±5% 조정)
  - 학습 버그 5건 전수 수정
- ✅ RAG는 **측정 전용** — 현재 raw corpus로 달성 가능한 T1 상한 파악용

**SFT를 버린 이유**:
- instruction QA tuning이 답변 길이를 collapse시킴. 한의학 고전 해제·해설체 출력과 어긋남.
- ver4는 새 지표 `answer_length_ratio` (0.8 ~ 1.2)를 도입해 이 조건을 수치로 강제.

**무엇으로 성공을 판정하나**:
- 주 지표: `T1_acc ≥ 70%`, `T1_paraphrase ≥ 50%`, `answer_length_ratio ∈ [0.8, 1.2]`, `forgetting_rate ≤ 5%p`
- 부지표: `bind_density × ≥ 14` (Chinese prior 역전), `retrieval_recall@3 ≥ 70%` (raw corpus 충분성)
- `eval_loss`는 secondary (factual recall과 비상관 실증됨)

**다음 한 걸음** (우선순위, r2):
0. **EXP-V4-00 (데이터 수집 감사 + resume 크롤, 8~16h + 2h)** — ✅ **resume 크롤 완료** (5권 DONE, +10,922 records, 총 120,235). 🔲 post_resume entity snapshot + diff 측정은 미완
1. EXP-V4-02 (T1 eval set 30문항, 4h) — EXP-00과 병렬 가능
2. EXP-V4-01 (base baseline + 답변 길이 측정, 1h)
3. EXP-V4-05 (RAG upper bound, 5h) — raw corpus 충분성 게이트
4. **EXP-V4-06 (fact sheet + 합성 코퍼스 파이프라인, 2~3d)** — P-A+의 선결 조건
5. **EXP-V4-03 (P-A+ 재학습, 12~16h)** — 최종 판정 실험

## 구현 진행 상태 (2026-04-20 기준)

### 🟢 크롤 데이터 무관 — 즉시 가능

| # | 파일 | 상태 | 비고 |
|---|---|---|---|
| 1 | `scripts/probe_factual.py` | ✅ **완료** | base/adapter 두 모드, CLI, jsonl I/O, EXP-V4-01에 그대로 사용 가능 |
| 2 | `scripts/probe_answer_length.py` | 🔲 | §1.1 프로토콜, 40 프롬프트, base·adapter 양쪽 token 집계 |
| 3 | `scripts/audit_collection.py` | 🔲 | `data/stats/book_completeness.json` 출력, 크롤 중에도 실행 가능 |
| 4 | `scripts/entity_delta.py` | ✅ **완료** | snapshot/diff 서브커맨드. checkpoint_01 캡처 완료 (raw 기준 허준 43, 이제마 4, 세종 9, 이시진 932) |
| 5 | `scripts/rebuild_manifests.py` | 🔲 | DONE 된 book_024/139 즉시 치유 |
| 6 | `src/training/cpt_trainer.py` | ✅ **완료** | 5 클러스터 전부: `modules_to_save=["embed_tokens","lm_head"]` / `num_train_epochs=3` / `load_best_model_at_end+metric_for_best_model="eval_loss"` / `save_steps=eval_steps=50+save_strategy="steps"` / `lr_scheduler_type="cosine_with_min_lr"+min_lr_rate=0.1` |
| 7 | `src/utils/seed.py` | ✅ **완료** | `CUBLAS_WORKSPACE_CONFIG=:4096:8` (module import 시점) + `torch.use_deterministic_algorithms(True,warn_only)` + cudnn deterministic. `extract_corpora.py` / `mediclassics_orchestrator.py` / `expand_facts.py` caller 추가 |

### 🟡 크롤/선행 산출물 필요

| # | 파일 | 상태 | 선결 |
|---|---|---|---|
| 8 | `src/data/synth/expand_facts.py` | ✅ **완료** | Template × paraphrase×4 × entity validation. **N=26 (실측)**: 1,791 records / entity pass 100% / bind_density 113× (target ≥14×의 8배) |
| 9 | `scripts/verify_synth_facts.py` | ✅ **완료** | n_triples / entity_pass_rate / trigram jaccard / bind_density(synth vs ko_only baseline) / hanja_ratio_mean. stdout + `data/stats/synth_facts_verify.json`. ⚠ jaccard 계산 경로에 조기반환 버그 (N=26에서 None 반환) — 경미, 별도 수정 |
| 10 | `src/data/builder/extract_corpora.py` (book_meta_prolog) | ✅ **완료** | `load_factsheet` + `expand_book_meta_prolog` (한자 병기, hanja_ratio≥0.10 gate, bilingual fallback). CLI `--factsheet / --no-prolog`. smoke: books_with_prolog=3, prolog_bilingual_skipped=0 |
| 11 | `src/data/builder/preprocess.py` (book 경계 + hash 정규화 + synth 통합) | ✅ **완료** | Stage 2 `book_boundary_flushes` + `sequences_per_book` + `unique_books` assertion. `extract_zh_for_hash()` 로 bilingual `<ZH>…</ZH>` 정규화 해시 (ko_only skip). `CORPORA_KINDS`에 `hanmed_synth_facts: ko_only` 등록 — N=26 synth 1,791 rec → 266 packed seqs (544K tok), book 경계 invariant(`flushes=24, unique_books=25`) 통과 |

### 🔴 사람 작업 (자동화 불가)

| # | 파일 | 상태 | 공수 |
|---|---|---|---|
| 12 | `data/facts/core_factsheet.yaml` | ✅ **완료 (N=26)** | `scripts/build_factsheet_draft.py` 자동 추출 (KIOM `raw_text` regex + vol_01 서문 연호 regex + 연도→왕대 유추) + 수기 보완 (파싱 실패 5권 복구 · 12권 genre/topics/signature_items · book_60 title 교정 · 중국원전 override). 충족도: author 26/26 · reign 16/26(조선권만) · year 22/26 · genre 17/26 · **4/4 충족 16권**. 기존 #12 "수기 12~18h"는 대체됨 |
| 13 | `eval/hanmed_eval_v0/T1_factual.jsonl` | 🔲 | 4h (30문항 + κ≥0.9 이중 라벨) |

## 라운드 기록

| 라운드 | 일자 | 에이전트 | 결론 |
|---|---|---|---|
| round_1 | 2026-04-20 | generator + discriminator + reviewer + iteration-planner | 진단 완료. 실험 없이 재실행 무의미 → 종료하고 ver4 r0 작성 |
| r0 → r1 | 2026-04-20 | (사용자 제약 반영) | SFT 제거, RAG를 측정용으로 강등, P-A+ 단일 paradigm 확정 |
| r1 → r2 | 2026-04-20 | (데이터 수집 실측) | Core 14 중 4권 + Core 25 중 1권 크롤 중단 확인. EXP-V4-00 (resume 크롤) 선결 단계 신규 추가. 서문·발문 누락이 환각의 후속 원인 가설로 추가 |
| 구현 진행 | 2026-04-20 | (코드 수정) | EXP-V4-00 resume 크롤 DONE (13:47, +10,922 rec). #10/#11 구현 완료 (extract_corpora prolog + preprocess book 경계/hash 정규화). #12 샘플 3권 초안 |
| 구현 진행 ② | 2026-04-20 | (3-agent 병렬) | #6 cpt_trainer 5버그 수정 / #7 seed determinism / #8 expand_facts + #9 verify_synth_facts 구축. N=3 synth smoke 488 records, entity pass 100%, bind_density 54×. 남은 차단: #2 probe_answer_length, #12 factsheet 확장 (human), #13 T1 eval (human), EXP-V4-00 post_resume entity snapshot/diff |
| 구현 진행 ③ | 2026-04-20 | (factsheet 자동화 + 수기) | `scripts/build_factsheet_draft.py` 신규로 **N=3 → N=26** 확장 (KIOM `raw_text` regex + vol_01 서문 연호 regex + 연도→왕대 유추). 수기 보완: 파싱 실패 5권 복구 + 12권 genre·topics·signature_items + book_60 title 교정(금리산인→宜彙) + book_100/139 중국원전 override. **synth records 488 → 1,791 (3.7×), bind_density 54× → 113×**. #12 원래 "수기 12~18h" 분류가 자동화로 대체 |
| 구현 진행 ④ | 2026-04-20 | (synth 파이프라인 통합) | `preprocess.py CORPORA_KINDS`에 `hanmed_synth_facts: ko_only` 1줄 추가. Stage 1/2 smoke 통과 (1,791 kept 100%, 266 packed seqs, 544K tok). book 경계 assertion invariant 통과. **EXP-V4-03 재학습 즉시 착수 가능 상태**. 남은 잔여(non-blocker): #2 probe_answer_length(EXP-01 판정 시), #13 T1 eval (human), EXP-V4-00 post_resume entity snapshot/diff (측정용), verify_synth_facts jaccard 버그 (cosmetic) |

## 관련 산출물 경로

- 실패한 adapter: `outputs/cpt_bllossom/adapter` (checkpoint-156, 2.4 GB, 보관)
- 학습 로그: `outputs/cpt_bllossom/train.log`
- trainer state: `outputs/cpt_bllossom/checkpoint-156/trainer_state.json`
- 현 코퍼스: `data/cpt_processed/corpus_v1.json`
- probe 스크립트: `scripts/probe_factual.py` ✅ (2026-04-20 완료, `/tmp/hanmed_probe.py` 영구화)

## ver2 대비 변경점

| 항목 | ver2 | ver4 r1 |
|---|---|---|
| paradigm | CPT | CPT (단일 유지) |
| 주 지표 | `eval_loss` | `T1_acc`, `T1_hallu`, `T1_paraphrase`, `answer_length_ratio` |
| SFT 계획 | M2 검토 | **금지** (길이 collapse) |
| RAG 계획 | — | **측정용만** (배포 X) |
| 환각 처방 | mix 조정 | fact sheet + N-parametrized 합성 long-form 코퍼스 (~1~2M tok) |
| 메타 주입 | — | book당 1회 200~400 tok prolog (짧은 메타줄 X) |
| 평가셋 | M2 로드맵 | EXP-V4-02로 **선결** |
| 학습 버그 수정 | — | 5건 전수 수정 강제 |
| contamination gate | bilingual block 해시 | 원문 정규화 해시 + heldout 사전등록 ≥ 20 |

## r0 → r1 → r2 변경점

| 항목 | r0 | r1 | r2 |
|---|---|---|---|
| paradigm 수 | 3 (P-A / P-B / P-C) | 1 (P-A+) | 1 (P-A+), 단 선결 단계 추가 |
| SFT | EXP-V4-04 채택 후보 | **기각** | 동일 (기각 유지) |
| RAG | 배포 후보 중 하나 | 측정 전용 | 동일 |
| 합성 코퍼스 규모 | prolog 1줄/권 (<0.1M) | long-form ~12M tokens (계산 오류) | **수정**: N-parametrized (N=14 → 1.26M / N=21 → 1.89M / N=25 → 2.25M) |
| 신규 지표 | — | `answer_length_ratio`, `bind_density` | + `entity_delta`, `post_resume_complete_books` |
| 신규 실험 | — | EXP-V4-06 | + **EXP-V4-00 (데이터 수집 감사 + resume 크롤)** |
| fact sheet | — | `data/facts/core14_factsheet.yaml` 수기 curation | `core_factsheet.yaml` (수집 완주 검증 후 scope 결정) |
| 데이터 상태 | "Core 14 완료 ✅" 가정 | 동일 | **실측 결과 5권 미완료 확인, 재크롤 선결** |
| Fact 출처 위계 | 명시 없음 | 명시 없음 | 1. 원문 서문·발문 / 2. KIOM 메타 / 3. 백과 교차검증 / 금지: LLM 자유생성 |
