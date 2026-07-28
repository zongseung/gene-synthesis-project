# HanMed-LLM Proposal — **ver3 (M2/M3 공식화 패치)**

> ver2.2 R3.5 정본을 상속, 이번 세션의 **Core 14 pilot 실측** + **Core 25 확장 진행** + **`hanmed_cli` v0 smoke** 결과를 반영해 **M2/M3 단계를 공식화**한다. 새 비교 축을 도입하지 않고 pilot → 본 run 의 **선형 확장**을 이어붙이는 문서.

- 작성일: 2026-04-16
- 버전: **ver3** (ver2 → ver2.1 → ver2.2 R3.2~R3.5 → **ver3 pilot-ingest**)
- 이전 정본: [`../ver2/README.md`](../ver2/README.md) — 수정하지 않음
- 상태: pilot 완료 (2h 53m, eval_loss 2.065 → 1.887), Core 25 확장 중, CLI v0 smoke green — M2/M3 계획 수립

## 한 줄 요약

**Core 14 실측 cap 20.4M tokens / 156 steps / 2h 53m 28s / eval_loss 2.065 → 1.887 (ppl 6.60)** pilot 은 **under-training regime 에서도 한의학 register 획득이 가능**함을 보였다 (CLI 1-shot smoke Q1/Q2 한의학 체례 성공 · Q3 format following 약함 · Q4 pre-safety 완벽). ver3 는 이 지점에서 **(a) 데이터 수집을 Tier 1~3 으로 확장**, **(b) 백본 재학습 Stage 1 본 run + Stage 2 SFT 도입**, **(c) v0 로컬 REPL 을 v1 vLLM serve + Docker + 클라우드로 승격**하는 두 가지 궤도를 분리 기획한다.

## 문서 구성

| # | 파일 | 목적 |
|---|---|---|
| 1 | [`data_and_backbone_retrain.md`](data_and_backbone_retrain.md) | **데이터 수집 Tier 1~3 확장 + 평가셋 curation + Wiki-ko replay + Stage 1 본 run + Stage 2 SFT** 공식 계획 (M2 → M3 경로) |
| 2 | [`deployment.md`](deployment.md) | **v0 L-로컬 REPL → v1 vLLM serve + Docker → v1 C-클라우드** 배포 스테이지 계획 (M2 → M3 → M4 병행 경로) |

## 이번 세션 pilot 실측 snapshot

| 항목 | 값 | 출처 |
|---|---|---|
| run_name | `cpt_bllossom_3e_20M` | `outputs/cpt_bllossom/train_manifest.json` |
| cap_tokens | 20,400,000 | 실측 manifest |
| epoch_variant | 3 | 실측 manifest |
| total_steps | 156 | 실측 manifest (effective_tokens_per_step 131,072) |
| 소요 시간 | **2h 53m 28s** | `train.log` (A6000 DDP 2-GPU) |
| eval_loss (step 50 → 100 → 156) | **2.065 → 1.913 → 1.887** (ppl 6.60) | `train.log` |
| train_loss 범위 | step10 2.807 → step150 1.859 (평균 2.097) | `train.log` |
| LoRA trainable | **83.9M / 8.11B (1.03%)** | CPT trainer 표준 출력 |
| base | Bllossom-8B (R3.2 primary) | manifest |
| tokenizer | `data/tokenizer/hanmed_bllossom_ext` (vocab 128,260) | manifest |
| Mix 실측 (post-normalize) | bilingual 0.125 / zh 0.625 / ko 0.25 | manifest |
| adapter 산출 | `outputs/cpt_bllossom/adapter/` | `adapter_model.safetensors` + `chat_template.jinja` 확인 |

**해석 (ver3 문서 1 §1 에서 정량 재인용)**:
- eval_loss **-0.178** (50 → 156 step, −8.6%): under-training regime 에서도 도메인 적응 신호 확인
- CLI 1-shot smoke: Q1(인삼 성미) 한의학 register 획득, Q4(증상 호소) pre-safety regex 완벽 refusal → **register 학습 ✅**
- Q3 "사물탕 약재 4개" 34자 즉시 종료 → **format following 취약 → Stage 2 SFT 도입 근거**
- chat_template 보존 (`chat_template.jinja` 유지) → **P-CPT 경로 유효**

## 수집 현황 snapshot

| 출처 | 값 | 비고 |
|---|---|---|
| `data/cpt/corpus_stats.json` | chars_zh 1,203,407 · chars_ko 1,969,632 · records 25,059 | **Core 14 완료** |
| `data/cpt_processed/corpus_v1.json` | HanMed unique **2.72M tokens** (Bllossom 기준) · 3 corpus packed 2,745 seq | Stage 2 pack 완료 |
| `data/stats/mediclassics_book_list.json` | **161종 전체 메타** (A 10 / B 39 / C 33 / D 16 / E 20 / F 8 / G 5 / ? 30) | 카테고리 분류 완료 |
| Core 25 확장 | 11권 수집 중 (`7,44,46,47,49,54,60,70,71,94,139,183`) | 쿼터 문제로 진행 중 |

## M2 / M3 단계 overview

| 단계 | 문서 1 (데이터·백본) | 문서 2 (배포) |
|---|---|---|
| **M2** (현재) | Tier 1 Core 25 완료 + 평가셋 T1~T5 curation + Wiki-ko 수집 + corpus_v2 manifest | v0 L-로컬 검증 완료 (유지) · vLLM backend 스켈레톤 → 실가동 + `hanmed serve` 서브커맨드 |
| **M3** | Stage 1 본 run (cap 60M / 150M candidate) + Stage 2 SFT curation 개시 + rank ablation 16/32/64 | Docker 패키징 + 기관 HTTP S2 + CI/CD (GHCR) |
| **M4** | §E null-result 대응 ablation (20M / 60M / 200M 3-way) + SFT pilot + 전체 평가 T1~T5 | C-클라우드 (RunPod A6000) 파일럿 + KIOM 승인 병행 |

## 관련 ver2 문서 cross-reference

| ver3 문서 | 참조하는 ver2 섹션 |
|---|---|
| `data_and_backbone_retrain.md` §1 | `ver2/04_model_strategy/preprocessing_and_cpt_spec.md` §C.4.3 (cap 20~60M) · §E ablation |
| `data_and_backbone_retrain.md` §2 | `ver2/03_data_pipeline/acquisition.md` §3.5 (보조 코퍼스 라이선스) |
| `data_and_backbone_retrain.md` §3 | `ver2/05_evaluation/hanmed_eval.md` §5.2~§5.6 (T1~T5 exit criteria) |
| `data_and_backbone_retrain.md` §4 | `ver2/11_implementation/work_order.md` W0 (Core 25 재개) |
| `data_and_backbone_retrain.md` §5~§6 | `ver2/04_model_strategy/base_model_and_training.md` §4.5 (Stage 1) · §4.6 (Stage 2 SFT) |
| `data_and_backbone_retrain.md` §7 | `ver2/04_model_strategy/preprocessing_and_cpt_spec.md` §E (ablation null-result 대응) |
| `deployment.md` §1 | `ver2/10_demo_cli/deployment.md` §10.7 (L/S1/S2/C 매트릭스) |
| `deployment.md` §2 | `ver2/10_demo_cli/inference_backend.md` §10.4.1 (vLLM primary) |
| `deployment.md` §3 | `ver2/10_demo_cli/prompt_and_safety.md` §10.5.4 (safety 2-layer) · §10.5.5 (T4 평가) |
| `deployment.md` §4 | `ver2/10_demo_cli/packaging.md` §10.8.3 (Docker 초안) |
| `deployment.md` §5 | `ver2/07_license_ethics/license_ethics.md` §7.1 (KIOM 라이선스 경로) |
| `deployment.md` §7 | `ver2/10_demo_cli/packaging.md` §10.8.5 (CI/CD v1) |
| `deployment.md` §8 | `ver2/10_demo_cli/milestones_and_exit.md` §10.9.3~.4 (v0→v1→v2) |

## ver2.2 → ver3 전환 요지

1. **수치 재확정**: Core 14 HanMed unique **2.72M tokens** + cap 20.4M pilot 정상 수렴 → ver2.2 R3.2 의 "cap 20~60M" 범위를 실측 lower-bound 로 고정.
2. **Stage 2 SFT 복권**: ver2.2 §10.3 에서 "P-CPT primary, P-SFT v1 이후" 로 분리했으나, pilot Q3 format 취약이 실측으로 드러나 **M3 에서 SFT 를 "옵션" 이 아닌 "본 run 직후 curation 개시" 로 공식화**. 논문 기여 축은 여전히 CPT primary, SFT 는 format 보조 adapter.
3. **배포 v1 조기화**: `hanmed_cli` v0 smoke 가 한문 register + pre-safety 까지 입증 → vLLM + Docker 패키징을 M3 에 착수, KIOM 승인 트랙과 **병렬** 진행.
4. **Tier 기반 데이터 확장**: ver2 "Core 25" 단일 목표에서 **Tier 1 (Core 25 필수) / Tier 2 (B 경험의방 + Wiki-ko replay) / Tier 3 (C 고전 + D 본초 + E 전문)** 3-tier 로 분할해 M2/M3/M4 경로 분리.

## 다음 실행

- 문서 1 §4 에 따라 **Core 25 크롤 완료** (현재 진행 중) → `corpus_stats.json` 재생성 → `corpus_v2.json` manifest
- 문서 1 §3 에 따라 **T1~T5 평가셋 curation** 착수 (M2 exit gate)
- 문서 2 §2 에 따라 **vLLM backend 활성화** (pilot adapter 이미 존재 → 즉시 가능)

## 참조

- `../ver2/README.md` — ver2.2 R3.5 정본 (수정 금지)
- `../proposal_v0_draft.md` — v0 draft (역사)
- `outputs/cpt_bllossom/train_manifest.json` — pilot 실측 정본
- `outputs/cpt_bllossom/train.log` — 156 steps full log
- `data/cpt_processed/corpus_v1.json` — Stage 2 pack 산출물 (SHA-256 pin)
