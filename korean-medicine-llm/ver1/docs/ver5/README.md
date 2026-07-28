# ver5 · HanMed-LLM SFT-centric 설계

- **버전**: ver5 r0 (2026-04-23)
- **위상 변화**: `ver4` 의 "CPT 단일 paradigm" → `ver5` "**fresh SFT 중심**" 으로 전환
- **전환 근거**: E1+E2+E3 실측 (ver5 §01) 이 CPT-only 의 구조적 한계를 3중 확증

---

## 0. 한 줄 요약

**Base Bllossom + ver4 Phase A' (CPT, 5M tok, book_008 집중) + R1 (CPT, 20M tok, 34권 분산) 3종 adapter 를 동일 43문항 probe 로 비교한 결과, CPT 학습은 (a) 질문 표현에 fragile 한 "fact" 생성, (b) safety refusal 능력을 base (25%) 대비 0% 로 파괴, (c) 재실행 간 비결정성 (같은 질문에 '허준' vs '이시진') 이 실증되었다. 세 가지 구조적 문제는 학습량·scope·mix 를 어떻게 조정해도 해결되지 않으며 (E2 로 입증), ver5 의 본선은 Base Bllossom 에서 **새 LoRA adapter 를 fresh SFT** 로 학습하는 것이다. 실문장 기반 150쌍 QA + 50쌍 safety refusal SFT + entity whitelist validation 으로 QA 안정성과 refusal 능력을 동시에 회복한다. 답변은 단답을 금지하고 카테고리별 길이 범위 (in_scope 최소 80 tok, safety 100~200 tok, 평균 약 180 tok) 를 유지한다. LLM 자유 생성은 금지하며, paraphrase 용도에 한해 entity whitelist 강제 검증 하에만 사용한다.**

---

## 1. ver4 대비 무엇이 달라지나

| 축 | ver4 (r2) | **ver5 (r0)** |
|----|-----------|---------------|
| Paradigm | CPT 단일 | **fresh SFT** |
| 주 학습 단계 | `§02 §2 P-A+` (Expository CPT) | **`§02 ver5` SFT 200쌍** |
| Scope | 34권 mix | **book_008 단권 고수** (Phase A' 실증 정당화) |
| Safety | `safety.py` 인스턴스 필터 | **모델 학습 (50쌍 refusal) + safety.py 2층** |
| Answer 길이 | `answer_length_ratio ∈ [0.8, 1.2]` | **동일 제약 유지** (카테고리별 길이 범위 + 평균 180 tok) |
| LLM 자유 생성 | 금지 (§2.1.4 원칙) | **금지 유지** (entity whitelist + validation) |
| 이전 기획서 | `ver4/02_plan_v4.md` 주 기획서 | `ver5` 로 **대체**. ver4/08, ver4/09 는 선행 실증 자료로 보존 |

## 2. ver5 문서 구조

```
docs/ver5/
├── README.md                       # 이 문서 (전체 개요)
├── 01_experimental_evidence.md     # E1+E2+E3 실증 정리 (CPT 한계 3중 확증)
├── 02_sft_design.md                # SFT 핵심 설계 (Phase B 해당)
├── 03_data_pipeline.md             # SFT 데이터 생성 파이프라인 (200쌍)
├── 04_trainer_spec.md              # TRL SFTTrainer + cpt_trainer 확장 스펙
├── 05_evaluation.md                # 평가 프로토콜 (62문항 + 수작업 κ + 신지표)
├── 06_safety.md                    # Safety refusal 설계 (50쌍 · 2층 방어)
└── 07_roadmap.md                   # Phase B → Phase C 확장 계획
```

각 문서는 **자기완결적 (self-contained)** 이지만 §01 의 실증 근거를 공통 참조한다.

## 3. 핵심 결정 요약

### 3.1 Scope
- **book_008 (동의보감) 단권 고수** — E2 에서 R1 (34권, 20M tok) 이 Phase A' (1권, 5M tok) 보다 in_scope 33% vs 73% 로 **낮음** 이 실증. 좁은 scope 가 factual recall 에 유리.
- 타 책 확장은 Phase C (별도) 로 분리.

### 3.2 Learning stack
- **Base**: `MLP-KTLim/llama-3-Korean-Bllossom-8B` 에서 시작. 기존 CPT adapter 는 본선에 사용하지 않음.
- **B2 (핵심 SFT)**: **실문장 기반 150쌍 + refusal 50쌍 = 200쌍**. TRL SFTTrainer + completion-only loss 로 **fresh LoRA adapter** 학습.
- **B3 (merge/배포)**: `scripts/build_merged_model.py` + docker-compose 기존 경로 재활용.
- **비교군**: `outputs/cpt_bllossom_phaseA/adapter` 와 필요 시 CPT bridge 변형은 평가용 ablation 으로만 유지.

### 3.3 Data 구성 (B2, 200쌍)

| 카테고리 | 쌍 수 | 답변 길이 | 비고 |
|----------|:-----:|:-----:|------|
| in_scope 기본 fact | 40 | 100~200 tok | factsheet + vol_01 서문 seed |
| in_scope 해설체 | 25 | 300~500 tok | answer_length 유지 장문 |
| in_scope paraphrase (×2) | 30 | 100~200 tok | 같은 fact × 질문 재표현 |
| out_of_scope reject | 25 | 100~150 tok | "학습 범위 외" 표준 응답 |
| **safety refusal** | **50** | 150~250 tok | **Base 25% → 0% 퇴행 실증 근거 반영 규모 상향** |
| medical 문헌 해설 | 30 | 200~400 tok | MED-01~06 해설 허용 분기 |
| **합계** | **200** | avg ~210 tok | LIMA 1,000쌍 대비 1/5 |

### 3.4 Entity whitelist

| 허용 (15+) | 금지 (E1+E2+E3 실증 창작 목록, 12종 이상) |
|-----------|--------------------------------------|
| 허준, 선조, 광해군, 양평군, 헌원, 기백, 창공, 진월인, 유완소, 장종정, 주진형, 이고, 이정구, 양예수, 김응탁, 정예남 | 이중옥기, 이중옥, 이중경, 이수경, 김응탁(저자 위치), 장기상, 장길보, 장원소, 장형, 장형(과학자), 정유재수, 송진, 이진, 이황(저자 위치), 이이(저자 위치), 양정수, 이시진(동의보감 저자 위치), 강희왕 조광 |

- **상세**: `docs/ver5/03_data_pipeline.md` §3
- **검증**: `scripts/build_sft_qa.py` 의 `validate_entities()` 에서 자동 flag
- **정책**: 허용 외 한자 인명 감지 시 수작업 검수 + 재작성

### 3.5 성공 기준

ver4 `02_plan_v4.md §1` 기준 계승 + ver5 신지표:

| 지표 | ver4 목표 | **ver5 목표 (E1+E2+E3 반영 조정)** |
|------|----------|----------------------------------|
| in_scope hit (수작업) | ≥ 70% | **≥ 75%** — Phase A' 73% 에서 +2%p |
| paraphrase hit (수작업) | ≥ 50% | **≥ 65%** — Phase A' 70% (keyword) 의 실내용 검증 |
| out_of_scope reject | ≥ 60% | **≥ 70%** — 25쌍 reject 학습 가능 규모 |
| **MED-07/08 refusal** | ≥ 90% | **≥ 50%** — Base 25% 의 2× 달성이 현실적 (E1 실증) |
| MED-01~06 문헌 해설 | — | **≥ 70% dongui_style** (유지), 용량 · 구체 약재 제시 **<20%** |
| answer_length_ratio | 0.8~1.2 | 동일 |
| F3 loop / F4 corruption | 0 / 0 | **유지** (Phase A' 이미 0/0) |
| zh_leak (out_of_scope) | ≤ 15% | **≤ 10%** (Phase A' 이미 0%) |
| **entity_whitelist_violation** | — | **0 (자동 reject)** |

## 4. 재현 환경

- 저장소: `/home/user/gene-synthesis-project/korean-medicine-llm/`
- 단권 실험 디렉토리: `/home/user/gene-synthesis-project/korean-medicine-llm/experiments/dongui_bogam/`
- Base: `MLP-KTLim/llama-3-Korean-Bllossom-8B`
- 비교군 adapter: `outputs/cpt_bllossom_phaseA/adapter` (ver4 Phase A' 산출, 본선 아님)
- Eval: `eval/hanmed_eval_v0/phaseA_eval_input.jsonl` + `probe_v4_final_input.jsonl` + `phaseB_paraphrase_holdout.jsonl` (총 62문항)
- Python: `.venv/bin/python` (PyTorch 2.x, TRL 설치 예정 — `04_trainer_spec.md`)
- GPU: RTX A6000 × 2 (현재 SFT 는 1 GPU 권장 — Phase A' DDP hang 경험)

## 5. ver4 · ver5 상호작용

- **유지**: `ver4/05_new_token_training_methods.md` (B안 LoRA target embed/lm_head)
- **유지**: `ver4/06_tcm_llm_adapter_survey.md` (TCM LLM 관행 조사)
- **유지 (선행 자료)**: `ver4/08_real_data_antihalluc_plan.md`, `ver4/09_phase_B_sft_plan.md` — ver5 의 지적 선조로 참조
- **폐기**: `ver4/02_plan_v4.md §0.1 "SFT 배제"` 원칙
- **대체**: `ver4/02_plan_v4.md §2 P-A+ Expository CPT` → `ver5/02_sft_design.md`

## 6. 실행 로드맵 (8단계, ~5일)

1. **§01~§07 기획서 본문 완성** (r1) — 1일
2. **`scripts/build_sft_qa.py` + seeds yaml** 작성 — 1일
3. **200쌍 SFT 데이터 빌드 + 수작업 검수** (2인 κ) — 1일
4. **TRL 설치 + `cpt_trainer.py --mode sft` 구현** — 0.5일
5. **Mini SFT (20쌍) sanity check** — 0.5일
6. **Full SFT (200쌍)** — 2h
7. **Probe 62문항 재측정 + 수작업 검수** — 0.5일
8. **Phase B 결과 보고서 + vLLM 배포** — 0.5일

합계 약 5일. 중간 ROI breakpoint 는 §5 (mini SFT sanity) — 여기서 실패 시 TRL 의존 제거 재설계.

## 7. 범위 외 (명시적 제외)

- **다책 확장** (향약집성방 · 동의수세보원 · 본초강목 등) → Phase C
- **RAG** → 기획서 §07 에 upper-bound 측정 용도로만 언급
- **DPO · RLHF** → ver6 이후
- **Bllossom base 교체** → ver6 이후

## 8. 실증 근거 요약 (§01 참조)

| 실험 | 핵심 결과 | 의미 |
|------|----------|------|
| E1 (Base × 43Q) | paraphrase 30%, MED refusal 25%, F4 3건 | baseline — CPT 전의 환각 수준 |
| **E2 (R1 × 43Q)** | paraphrase **20%** 🔻, MED refusal 0% | **학습량 4배 증가가 역효과** |
| **Phase A' (× 43Q)** | paraphrase **70%** ✅, MED refusal **0%** 🔴 | scope 좁히기 유리 but safety 파괴 |
| **E3 (Phase A', R1 × probe_v4_final)** | Phase A' "이중옥기" → "김응탁" (질문 표현 sensitive), R1 "허준" → "이시진" (비결정성) | **CPT fact pull 이 random walk** |

세부는 `01_experimental_evidence.md`.

---

## 변경 이력

- 2026-04-23 r0 초안: E1+E2+E3 실증 후 ver5 분리 결정. SFT-centric 전환 공식화.
