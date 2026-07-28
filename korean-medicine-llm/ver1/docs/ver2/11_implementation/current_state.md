# 11.3 Current State — 실측 스냅샷 (2026-04-16)

> 코드·데이터·문서 현황을 한 테이블로. M2 착수 시 참조.

## 1. 데이터

### 1.1 Raw 수집

| 경로 | 상태 | 크기 |
|---|---|---|
| `data/raw/mediclassics_unified/` | ✅ Core 14 | 14 책 디렉토리 |
| `data/raw/mediclassics_unified/orchestrator.log` | ✅ | 수집 로그 |
| Core 25 확장 (11권 추가) | ❌ 중단 | W0 에서 재개 가능 |

### 1.2 전처리 산출

| 파일 | records | 용도 | 상태 |
|---|---|---|---|
| `data/cpt/hanmed_bilingual.jsonl` | 21,043 | Stage 1 5% | ✅ |
| `data/cpt/hanmed_zh_only.jsonl` | 23,338 | Stage 1 25% | ✅ |
| `data/cpt/hanmed_ko_only.jsonl` | 22,597 | Stage 1 10% | ✅ |
| `data/cpt/hanmed_en_only.jsonl` | 8,137 | v1 CPT 제외 | ✅ |
| `data/cpt/corpus_stats.json` | — | 통계 | ✅ |
| `data/cpt_processed/*_clean.jsonl` | — | Stage 1 결과 | ❌ W3 실행 후 |
| `data/cpt_processed/*_packed_2048.jsonl` | — | Stage 2 결과 | ❌ W3 실행 후 |
| `data/cpt_processed/corpus_v1.json` | — | SHA 핀 manifest | ❌ W4 구현 후 |
| `data/tokenizer/hanmed_bllossom_ext/` | — | special token 추가본 | ❌ W1 구현 후 |
| `data/dict/hanmed_terms.jsonl` | — | NER seed (옵션) | ❌ |
| `data/replay/wiki_ko_*.jsonl` | — | 30% replay | ❌ |
| `data/replay/cbeta_*.jsonl` | — | 20% replay (내부) | ❌ |
| `data/replay/aihub_*.jsonl` | — | 10% replay (내부) | ❌ |

### 1.3 Character / Token 실측

| 지표 | 값 |
|---|---|
| chars_zh (Core 14) | 1,203,407 |
| chars_ko (Core 14) | 1,969,632 |
| chars_en (Core 14) | 1,726,112 |
| ko_coverage | 90.18 % |
| en_coverage | 32.47 % |
| **Bllossom tok/char (zh)** | **1.040** |
| **Bllossom tok/char (ko)** | **0.745** |
| **Bllossom byte_fallback** | **0 %** |
| HanMed unique (Core 14 × Bllossom) | **2.72 M tokens** |
| HanMed unique (Core 25 predict × Bllossom) | 3.8 ~ 4.5 M |

### 1.4 Eval asset

| 파일 | 상태 | 비고 |
|---|---|---|
| `eval/hashes/heldout_T1.txt` | ✅ (R3 placeholder 1 sample) | positive-control 용 |
| `eval/hanmed_eval_v0/T1.jsonl` | ❌ | 전문가 curation |
| `eval/hanmed_eval_v0/T2.jsonl` | ❌ | QA 30 |
| `eval/hanmed_eval_v0/T4.jsonl` | ❌ | safety 20 |
| `eval/hanmed_eval_v0/T4_paraphrase.jsonl` | ❌ | held-out 30 (R3.4) |
| `eval/hanmed_eval_v0/T4_hanmun.jsonl` | ❌ | 한문 10 (R3.4) |
| `eval/hanmed_eval_v0/T5.jsonl` | ❌ | KLUE-YNAT 100 |

## 2. 코드

### 2.1 실존 스크립트

| 파일 | LoC | 역할 | 현재 drift |
|---|---|---|---|
| `src/data/crawler/mediclassics_orchestrator.py` | — | multi-process 크롤 | ✅ 정상 |
| `src/data/builder/extract_corpora.py` | 215 | raw → bilingual/zh/ko/en | ✅ 실행됨 |
| `src/data/builder/preprocess.py` | 381 | Stage 1 + Stage 2 | ❌ B1/B3/B4 + M1/M3/M4/M6 |
| `src/training/smoke_cpt.py` | 188 | 파이프라인 smoke | ❌ Qwen 하드코딩 (Bllossom 전환 필요), device_map="auto" DDP 충돌 |

### 2.2 미구현 스크립트 (M2 blocker)

| 파일 | 예상 LoC | 우선순위 | 의존 |
|---|---|---|---|
| `src/data/builder/tokenizer_extend.py` | 60 | W1 | — |
| `src/data/builder/build_corpus_manifest.py` | 120 | W4 | W3 실행 |
| `src/training/cpt_trainer.py` | 400 | **W5 (핵심)** | — |
| `src/utils/seed.py` | 20 | (M1 fix 함께) | — |
| `src/hanmed_cli/` 패키지 | ~800 | W7 | W6 |

### 2.3 테스트

| 파일 | 상태 |
|---|---|
| `tests/` 디렉토리 자체 | ❌ 미생성 |
| T1~T5 pytest (§04a §F.4) | ❌ |
| `test_chatml_template.py` (E6, R3.4 강화) | ❌ |
| `test_safety_pre_patterns.py` (T4 20 + para 30) | ❌ |

## 3. 문서 (ver2)

### 3.1 섹션별 현황

| # | 파일 | 버전 | 마지막 수정 |
|---|---|---|---|
| 01 | overview | ver2.2 (Bllossom primary) | 2026-04-16 |
| 02 | data_source | ver2 | 2026-04-16 |
| 03 | data_pipeline | ver2.2 (실측 반영) | 2026-04-16 |
| 04 | model_strategy | ver2.2 R3.2 | 2026-04-16 |
| 04a | preprocessing_and_cpt_spec | ver2.2 **R3.4 확정** | 2026-04-16 |
| 05 | evaluation | ver2 | — |
| 06 | infrastructure | ver2 | — |
| 07 | license_ethics | ver2 | — |
| 08 | risks | ver2 | — |
| 09 | roadmap | ver2 | — |
| 10 | demo_cli | ver2.2 **R3.5 정합성 보강** (9+1 파일) | 2026-04-16 |
| **11** | **implementation (본 섹션)** | **R3.5 신규** | 2026-04-16 |

### 3.2 Harness 검증 이력

| Round | 대상 | 판정 |
|---|---|---|
| R1 | preprocessing_and_cpt_spec 초안 | REJECT_AND_REGENERATE |
| R2 | R2 개정본 | REJECT_AND_REGENERATE |
| R3 / R3.1 | R3 + minor edits | APPROVE |
| R3.2 | base model 전환 (Solar → Bllossom) | 실측 근거 확보 |
| R3.3 | demo_cli 단일 → 9파일 분할 | APPROVE_WITH_CHANGES |
| R3.4 | demo_cli patch 5건 | APPROVE |
| R3.5 | demo_cli 정합성 보강 + 11 implementation 신설 | **현재** |

로그: `.claude/harness-evals/hanmed-cpt-spec/`

## 4. 환경

### 4.1 HW

| 항목 | 값 |
|---|---|
| GPU | RTX A6000 48GB × (2대 가정, 단일 사용자) |
| CUDA | 12.x |
| OS | Ubuntu 22.04 / Linux 6.8 |
| Python | 3.10.x (`.venv`) |

### 4.2 주요 의존성 (`pyproject.toml`)

| 패키지 | 버전 |
|---|---|
| torch | 2.5.1+cu121 |
| transformers | 5.5.4 (또는 4.57.0 env 기본) |
| peft | (사용 예정) |
| vllm | (미설치, M2 W7) |
| click, prompt_toolkit, rich | (미설치, M2 W7) |

### 4.3 장시간 학습 주의 (memory)

- `uv run` 대신 **`.venv/bin/python`** 사용 — 멀티 시간 학습 중 deps resync 방지
- wandb offline mode (`WANDB_MODE=offline`)
- DDP: `torchrun --nproc_per_node=2`

## 5. 다음 세션 우선 작업

| # | 작업 | 선택지 |
|---|---|---|
| 1 | W1 `tokenizer_extend.py` 구현 | ✅ 저위험 즉시 |
| 2 | W2 `preprocess.py` drift fix | ✅ 저위험 즉시 |
| 3 | W3 Stage 1+2 실행 (packed jsonl 확보) | W1, W2 후 |
| 4 | W5 `cpt_trainer.py` 설계·구현 | 병렬 가능 (W3 와 무관) |
| 5 | W0 Core 25 크롤 재개 | 병렬 백그라운드 |

사용자가 어느 지점부터 코드로 옮길지 지정 시 해당 파일 작성 시작.
