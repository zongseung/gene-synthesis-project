# HanMed-LLM Proposal — **ver2.2 (R3.2 primary 전환)**

한의학 고전 + 한국어 특화 LLM 도메인 adapter 구축 기획서.

- 작성일: 2026-04-16 (ver2 초기) · 2026-04-16 ver2.1 패치 · 2026-04-16 **R3.2 primary 전환**
- 버전: **ver2.2** (ver1 → ver2 → ver2.1 → harness 3 round → **R3.2 tokenizer 실측 기반 primary 전환**)
- 상태: R3.2 반영, **Bllossom-8B primary 승격 / Solar 기각** (M2 착수 가능)
- 이전: `../proposal_v0_draft.md` (v0), `../01_overview/` ~ `../09_roadmap/` (ver1)

## 한 줄 요약 (R3.2)

**Bllossom-8B (Llama-3 한국어)** 위에 **bf16 LoRA 로 Stage 1 CPT (next-token prediction, 자기지도학습) + Stage 2 SFT**. 한문↔국역 병렬 데이터는 **bilingual block concatenation** 포맷으로 CPT 에 편입. 총 CPT 예산 **20M~60M tokens cap** (R3.2 Bllossom 실측 기반; Core 14 2.72M unique × epoch 3~5 × 40% 믹스). 데이터는 한국한의학연구원 **mediclassics.kr** (Core 14 수집 완료, Core 25 확장 대기).

**Base 모델 선택 근거** (`scripts/tokenizer_compare.py` 실측): Bllossom-8B 한문 tok/char 1.040, 한글 0.745, byte_fallback 0% — Solar-10.7B (1.533 / 1.254 / **53%**) 를 모든 메트릭에서 압도. ver2.0~2.1 의 "Solar primary + 150~250M cap" 전제는 18.16M 자 (161종 외삽) + Solar byte-fallback 미실측에 기인한 이중 과대추정이었음.

## 섹션 구성

| # | 섹션 | 파일 | 작성 |
|---|---|---|---|
| 01 | Overview | [01_overview/overview.md](01_overview/overview.md) | Agent A |
| 02 | Data Source | [02_data_source/data_verification.md](02_data_source/data_verification.md) | Agent A |
| 03 | Data Pipeline | [03_data_pipeline/acquisition.md](03_data_pipeline/acquisition.md) | Agent A |
| 04 | Model Strategy | [04_model_strategy/base_model_and_training.md](04_model_strategy/base_model_and_training.md) | Agent A |
| 04a | Preprocessing · Tokenizer · CPT spec | [04_model_strategy/preprocessing_and_cpt_spec.md](04_model_strategy/preprocessing_and_cpt_spec.md) | Agent A |
| 05 | Evaluation | [05_evaluation/hanmed_eval.md](05_evaluation/hanmed_eval.md) | Agent B |
| 10 | Demo CLI (`hanmed chat`) | [10_demo_cli/README.md](10_demo_cli/README.md) | R3.3 세분화 (9+1 파일) |
| 11 | Implementation Plan (M2 작업 지도) | [11_implementation/README.md](11_implementation/README.md) | R3.5 신규 (pipeline + work_order + current_state) |
| 06 | Infrastructure | [06_infrastructure/gpu_framework.md](06_infrastructure/gpu_framework.md) | Agent B |
| 07 | License · Ethics | [07_license_ethics/license_ethics.md](07_license_ethics/license_ethics.md) | Agent B |
| 08 | Risks | [08_risks/risk_register.md](08_risks/risk_register.md) | Agent B |
| 09 | Roadmap | [09_roadmap/milestones.md](09_roadmap/milestones.md) | Agent B |

총 9개 섹션 / **1,265 라인**.

## ver1 → ver2 핵심 변경

### BLOCKER 3건 해결

| # | ver1 문제 | ver2 해결 위치 |
|---|---|---|
| **B1** | Stage 1 CPT objective가 self-supervised / causal LM 임을 9개 문서 어디에도 명시 안 함 | **§04.5 서두 박스 + §01.3 기술 요약 표** — "Objective: next-token prediction (causal LM), 자기지도학습. Loss = cross-entropy over all non-pad tokens." |
| **B2** | 한문↔국역 병렬 데이터의 CPT 학습 포맷 미정의 | **§04.5.2 신규** — `<ZH>…</ZH>\n<KO>…</KO>\n\n` bilingual block concatenation, 2048 greedy pack (블록 경계 보존), BOS/EOS 규칙, tag-inclusive causal loss, 예시 2블록 포함 |
| **B3** | 토큰 예산 이중계상 (영역 누수 + 병렬 재사용) | **§02.5 재산정** (HanMed unique 32M~43M, 영역 제외 별도 라인) + **§04.5.3 폐기 선언** (ver1 수치 명시 폐기) + **150M~250M cap** (HanMed 40% 믹스 × 1.5~3 epoch) |

### MAJOR/MINOR 반영

| 항목 | 반영 위치 |
|---|---|
| Solar Apache-2.0 variant 불확실성 → 조건부 서술, M0 24시간 확증 | §04.2.2, §07.2, §08 R6, §09 M0 최상단 |
| chrF를 monitoring 지표로 강등, 전문가 선호가 primary | §05.3.1 |
| Wiki-ko replay 근거 부족 → T5 general-ko regression 신설 (**ver2.1: 20→100문항 확대**, E5 granularity 확보) | §05.2, §05.6, §08 M4 gate |
| DUS LoRA 복제 layer 독립 → adapter ×2 리스크 | §04.3 주석, §08 R14 |
| Eval contamination 훅 (held-out 30문장 hash filter) | §03.4.2, §05.7, §06.5, §08 R15 |
| 한자 tokens/char 1.3 magic number → data-driven median+0.2 | §04.4.2, §09 M2 |
| 체크포인트 tie-break (Stage별 기준) | §06.6 |
| ChatML prompt format 전 단계 통일 | §05.9, §06.9 |
| 전문가 계약 (NDA, 보수, 소유권, COI) | §07.9 + 별도 섹션 |
| SFT 합성 데이터 검수 rubric placeholder | §07.10 |
| DVC 외부 CSP 금지 (KIOM 재배포 해석) | §06.5, §07 |
| M0 체크리스트 재정렬 (KIOM → Solar 라이선스 → 전문가 LoI) | §09 M0 |
| M3 pilot ablation 2건 (Wiki-ko 20/30/50, DUS LoRA 독립/공유) | §09 M3 |

## 공통 결정 로그 (D1~D13)

두 에이전트가 drift 없이 동일 결정을 따르도록 사용한 source of truth. 요약:

- **D1** Stage 1 CPT = causal LM / 자기지도학습
- **D2** 병렬 = bilingual block concat (`<ZH>…</ZH>\n<KO>…</KO>\n\n`)
- **D3** HanMed unique 32M~43M (영역 제외), CPT cap 150M~250M
- **D4** 데이터 믹스: HanMed 40% (원문 25 / 국역 10 / 병렬 5), Wiki-ko 30%, CBETA 20% (내부), 예비 10%
- **D5** Stage 0 tokenizer: data-driven (median + 0.2)
- **D6** Base: **Bllossom-8B primary (R3.2 승격, tokenizer 실측 근거)**, Qwen2.5-7B backup 1, Mistral-Nemo backup 2, Solar 기각
- **D7** Solar DUS LoRA 복제 layer → adapter ×2 가능성 (R14)
- **D8** Data packing (greedy 2048, EOS between docs) + loss masking (기본 동일, Wiki-ko 0.5× ablation) + contamination hash hook
- **D9** ChatML prompt format 전 단계 통일
- **D10** T5 General-ko regression 평가 태스크 (drop ≤ 3%p)
- **D11** 체크포인트 tie-break: Stage 1 val_loss / Stage 2 T1 chrF / Stage 3 전문가 선호
- **D12** DVC remote = 로컬 NFS / 기관 내부만
- **D13** 전문가 계약 + SFT 검수 rubric placeholder

## 교차 일관성 검증 결과 (수동 grep)

| 항목 | 검증 |
|---|---|
| HanMed unique 32M~43M | §01, §02, §04, §08, §09 — 일치 ✅ |
| CPT cap 150M~250M | §01, §04, §08, §09 — 일치 ✅ |
| bilingual / `<ZH>` / `<KO>` | §01, §02, §03, §04, §05, §06 — 일치 ✅ |
| causal LM / 자기지도 | §01, §04 — 정본에 명시 ✅ |
| Solar Apache 불확실성 | §04, §07, §08, §09 — 일관 ✅ |
| M0 Solar 24시간 확증 | §07, §08, §09 — 일관 ✅ |

## ver2 → ver2.1 패치 (2026-04-16 적용)

ver2 doc-discriminator 감사에서 **APPROVE_WITH_CHANGES** 판정. B1/B2/B3 BLOCKER는 전부 PASS, 4개 MAJOR + 1개 MINOR 에이전트 drift를 in-place 패치로 해결:

| # | 문제 | 패치 |
|---|---|---|
| **M1** | Contamination hash 알고리즘 drift — §03.4.2=SHA256, §05.7=SHA-1 → 교집합 영구 공집합, 검사 무력화 | **SHA256 통일** (§03.4.2, §05.7) |
| **M2** | Contamination held-out 파일 경로 drift — §03=`ver2/05_evaluation/eval_holdout.jsonl`, §05=`eval/hanmed_eval_v0/{T1,T2,T5}.jsonl`. §03 훅은 T2/T5 필터 누락 | **`eval/hanmed_eval_v0/{T1,T2,T5}.jsonl` + `eval/hashes/heldout_{T1,T2,T5}.txt` 로 통일, 필터 범위 `data/cpt/*.jsonl` 전체로 확장** (§03.4.2, §05.7) |
| **M3** | §06.6 Stage 2 승격 primary = T1 chrF, 그러나 §05.3.1은 chrF를 monitoring으로 강등 → best adapter와 exit gate primary 엇갈림 | **Stage 2 primary = 전문가 선호 승률 (lag 수용), tentative best = chrF 2단계 프로토콜** (§06.6, §09 M4) |
| **M4** | E5 measurement bug — T5 20문항 × 1문항=5%p vs 목표 drop ≤ 3%p, 측정 granularity 아래 | **T5 20→100문항 확대 (KLUE-YNAT stratified 100)**, 1%p granularity 확보 (§05.2, §05.3.5, §05.5.1, §01.4, §08 M2 gate, §09 M1/M2) |
| **m5** (MINOR) | §04.2.2 Solar DUS base 사실 오기 "Llama-2 기반" | **"Mistral-7B 기반" (Upstage 2023 SOLAR paper)** 로 교정 (§04.2.2) |

패치 부가: §01 L68 "100샘플" (pre-existing drift) → "30샘플" (§05.5.2 T1 30개와 정합).

**총 변경**: 20건 edit, 8 파일 (§01, §03, §04, §05, §06, §08, §09, README).

## 다음 실행

1. (선택) **ver2.1 재감사** — 만약 패치 검증이 필요하면 doc-discriminator 1회 더
2. **M0 착수** — KIOM 이메일 + Solar 라이선스 24시간 확증 + 전문가 LoI + 데이터 탐색
3. 내경편 권1 파일 수동 다운로드 → `mediclassics_parsing_spec.md` §13 열린 질문 해소

## 참고

- `../proposal_v0_draft.md` — v0 draft + 초기 discriminator 리포트
- `../01_overview/` ~ `../09_roadmap/` — ver1 (REJECT 판정)
- mediclassics.kr (데이터 1차 소스)
- info.mediclassics.kr/apps/dist-texts/ (배포 서비스 안내)
