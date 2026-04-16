# 10. Demo CLI — `hanmed` 인터랙티브 터미널 (ver2.2 R3.5 정합성 보강)

> **R3.5 정합성 보강 (2026-04-16)**: R3.4 이후 문서군 내부 drift 를 추가 정리. (1) `v0` 범위에서 `hanmed serve`/`--remote` 를 명확히 v1 예약으로 분리, (2) cold-start 와 exit criterion 충돌 해소, (3) 세션 path·timestamp·model pinning 재현성 보강, (4) `hanmed eval` 옵션/threshold SSoT 정렬. 원본 단일 스펙은 `_archive_v1_single.md` 로 보존.

## 섹션 인덱스

| # | 파일 | 요약 |
|---|---|---|
| 10.1 | [scope_and_goals.md](scope_and_goals.md) | Scope, non-goals, **RAG 제외 명시**, Claude Code / Ollama 등 비교 |
| 10.2 | [architecture.md](architecture.md) | 전체 데이터 플로우, 모듈 분리 |
| 10.3 | [adapter_paths.md](adapter_paths.md) | **CPT-only vs CPT+SFT 두 경로** 비교 (사용자 지적 해소) |
| 10.4 | [inference_backend.md](inference_backend.md) | vLLM / transformers / llama.cpp 트레이드오프 |
| 10.5 | [prompt_and_safety.md](prompt_and_safety.md) | ChatML 템플릿, 2-layer safety, §05 T4 refusal |
| 10.6 | [session_management.md](session_management.md) | save/load 스키마, 히스토리 sliding window |
| 10.7 | [deployment.md](deployment.md) | **로컬 / 기관 서버 / 클라우드 3-way** 트레이드오프 (신규) |
| 10.8 | [packaging.md](packaging.md) | pyproject.toml, pip install, adapter 배포 |
| 10.9 | [milestones_and_exit.md](milestones_and_exit.md) | D1~D6 마일스톤, exit criteria |
| 10.10 | [config_constants.md](config_constants.md) | **SSoT** — 상수/CLI 옵션/환경변수 전수 (R3.4) |
| 10.11 | [cli_visual_identity.md](cli_visual_identity.md) | **ASCII 배너 / 캐릭터 / naming** — CLI 미감 설계 |
| Archive | [_archive_v1_single.md](_archive_v1_single.md) | R3.2 단일 스펙 (legacy, 참조용) |

## 한 줄 요약

**Bllossom-8B + HanMed-CPT LoRA (Stage 1) 로컬 추론** 을 터미널 REPL 로 감싸는 데모. SFT 는 옵션. RAG 는 사용하지 않는다. 배포는 **로컬 primary / 기관 내부 서버 secondary / 클라우드는 KIOM 승인 조건부**.

## R3.4 추가 수정 (2026-04-16)

R3.3 에 대한 harness 검증 후 **APPROVE_WITH_CHANGES** 판정. 5개 patch 적용:

| # | 변경 | 영향 |
|---|---|---|
| I1 | T4 "30개" → **20개** (§05 실측 정합) + paraphrase 30 held-out 추가 | 5 파일 |
| I2 | regex 리터럴 `N` → `\d+` 교정 (`(몇\|N) ?회` 등), pseudo-code 주석 추가 | prompt_and_safety + inference_backend |
| I3 | `config_constants.md` SSoT 신설 (상수/CLI 옵션/env/timezone) + ver2/README `10_demo_cli/demo_cli_spec.md` 404 링크 수정 | 신규 + ver2 README |
| I4 | §C.3 → §C.5 인용 정정, **Null-result fallback 순서 역전** (knowledge → RAG, format → SFT), P-CPT chat template 주장 완화 + M2 H1 gate | adapter_paths + scope_and_goals |
| I5 | Safety anti-contamination 프로토콜 (author separation + paraphrase held-out + 한문 jailbreak 분리) | prompt_and_safety |

## R3.5 추가 수정 (2026-04-16)

R3.4 반영 후 남은 문서 간 불일치를 정리:

| # | 변경 | 영향 |
|---|---|---|
| I6 | `v0` 배포 범위 재정렬 — 기관 서버는 **S1(SSH + REPL)** 만 v0, `hanmed serve` / `--remote` 는 v1 예약 | scope_and_goals, deployment, config_constants |
| I7 | 성능/exit 기준 분리 — **REPL 진입 시간**, **cold first-token**, **warm first-token** 를 분리 정의 | scope_and_goals, inference_backend, milestones_and_exit |
| I8 | 세션 재현성 강화 — 저장 path 를 XDG 기준으로 통일, timestamp UTC 저장, `model_revision`/`system_prompt_sha256`/atomic write 추가 | architecture, session_management, config_constants |
| I9 | `hanmed eval` 명세 정합 — `--paraphrase`, `--hanmun`, 개별 threshold 옵션을 SSoT 와 safety 문서에 동시 반영 | prompt_and_safety, config_constants |
| I10 | release pinning 강화 — `base_revision = "main"` 금지, immutable revision / adapter version 분리 | packaging |

## R3.3 핵심 변경

### 사용자 지적 (1): RAG 제거

**이전 (R3.2 단일 spec)**: §10.14 열린 결정 #3 "RAG (mediclassics 원문 검색 후 인용) — v2 범위"
**변경**: **전면 제거** (10.1 Out-of-scope 에 명시).

**이유**:
- ver2 논문 기여 축 = `(a) 병렬 CPT 레시피, (b) 평가 벤치, (c) bilingual block 효과`
- RAG 는 4번째 기여 축을 추가 → 논문이 comparison paper 가 됨 (memory: 금지)
- CPT 모델 자체의 knowledge injection 품질이 논문 기여의 핵심 → RAG 로 보완하면 CPT 실패를 덮음
- 구현 복잡도 (indexing / retrieval / rerank) 대비 v0 scope 에 불필요

### 사용자 지적 (2): DAPT/CPT 중심인데 왜 SFT?

**이전 (단일 spec)**: 데모가 Stage 2 SFT adapter 에 의존 (10.9 Adapter 배포)
**변경**: **10.3 adapter_paths.md 신설** — 2가지 경로 분리.

| 경로 | 사용 모델 | 기여 축과의 관계 | 권장 상황 |
|---|---|---|---|
| **P-CPT** | Bllossom-8B + Stage 1 CPT LoRA | ver2 논문 primary 기여 검증 | **ver2 논문 데모 (기본)** |
| P-SFT | Bllossom-8B + Stage 1 CPT merged + Stage 2 SFT LoRA | 사용자 체감 UX 향상 | v1 (post-paper) 또는 SFT 데이터 준비 완료 시 |

**이유**:
- Stage 1 CPT = DAPT = self-supervised (§04.5 D1). instruction following 은 **base Bllossom-8B Llama-3 chat template 의 내장 capability** 에 의존 — CPT 가 이를 망가뜨리지 않도록 §05 T5 regression gate (drop ≤ 3%p) 로 보호
- SFT 는 사용자 체감 품질은 올리지만 논문 기여 축 분산 — v1 이후로 분리
- **CPT-only 데모가 논문 주장의 직접 검증**: "CPT 후 zero-shot T1 번역 / T2 QA 가 baseline 대비 향상됐는지" 가 핵심

자세한 분석은 `adapter_paths.md` 참조.

## 배포 3-way (10.7 deployment.md)

| 경로 | 환경 | 사용자 | KIOM 라이선스 | 비용 |
|---|---|---|---|---|
| **L-로컬** | 개발자 자신의 A6000 | 1인 | ✅ OK (§07 범위) | 0 (보유 장비) |
| **S-기관 내부 서버** | Gachon GPU 서버 | 연구실 멤버 | ✅ OK (§07 비상업) | 0 (기관) |
| **C-클라우드** | AWS g5 / RunPod | 외부 사용자 | **❌ KIOM 승인 필수** (§07) | $1~2/h |

v0 는 L/S 만. C 는 KIOM 서면 승인 후 v1 이후.

## 의존성

- Stage 1 CPT adapter (`outputs/cpt_bllossom/adapter/`) — §04a §D G0~G9 gates green 후
- (옵션) Stage 2 SFT adapter (`outputs/sft_bllossom/adapter/`) — §04.6 기준

## 관련 기획 문서

- 모델: `../04_model_strategy/base_model_and_training.md` (§4.2 Bllossom primary)
- 전처리: `../04_model_strategy/preprocessing_and_cpt_spec.md` (§C Stage 1 CPT)
- 평가: `../05_evaluation/hanmed_eval.md` (T1~T5 refusal / regression)
- 라이선스: `../07_license_ethics/license_ethics.md`
