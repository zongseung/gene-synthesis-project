# 11.1 Pipeline Data Flow — Core 14 → CPT → Demo

> 단계 0~5. 각 단계의 입출력 스키마, 현재 상태, 실측 수치, 코드 경로를 단일 테이블로.

## 전체 개요

```
[0] data/raw/mediclassics_unified/book_NNN/vol_VV.jsonl   ✅ 14권 완료
     │  extract_corpora.py::normalize_text + make_bilingual_block
     ▼
[1] data/cpt/hanmed_{bilingual,zh_only,ko_only,en_only}.jsonl  ✅ 완료
     │  preprocess.py::stage1_clean (dedup + quality + contamination)
     ▼
[2] data/cpt_processed/{name}_clean.jsonl                  ❌ 미생성
     │  preprocess.py::stage2_pack  (Bllossom tokenizer + pack 2048)
     ▼
[3] data/cpt_processed/{name}_packed_2048.jsonl            ❌ 미생성
     │  cpt_trainer.py (DDP + LoRA r=32 + bf16 CPT)   ← M2 핵심 blocker
     ▼
[4] outputs/cpt_bllossom/adapter/                          ❌ 학습 후
     │  hanmed_cli (vLLM + LoRA serving + REPL)
     ▼
[5] 사용자 REPL (`hanmed chat`)                            ❌ v0 target
```

## 단계 0 — Raw 수집

**상태**: ✅ Core 14 완료. Core 25 확장 대기 (11권 추가 가능).

**입력 없음** (API 크롤).

**출력**: `data/raw/mediclassics_unified/book_{NNN}/vol_{VV}.jsonl`

```json
{"book_id": 8, "volume_id": 1, "content_seq": 138,
 "content_level": "ZZ", "up_path_nm": "內景篇卷之一 > 身形 > 形氣之始",
 "original": "乾鑿度云 …", "trans_ko": "《건착도》에 …",
 "trans_en": "...", "annotation": null, "index_num": 1}
```

**코드**: `src/data/crawler/mediclassics_orchestrator.py` (이미 동작)

**실측 수치**:
- 14권 = 25,059 records
- chars_zh = 1,203,407 / chars_ko = 1,969,632 / chars_en = 1,726,112
- ko_coverage = 90.18%, en_coverage = 32.47%

## 단계 1 — Block 추출 (`extract_corpora.py`)

**상태**: ✅ 완료 (실행됨, 산출물 존재).

**입력**: `data/raw/mediclassics_unified/`

**처리**: NFC 정규화 + 공백 정규화 + bilingual block wrap + 길이 필터 (zh ≥ 3자, ko ≥ 2자)

**출력**: `data/cpt/*.jsonl` + `corpus_stats.json`

```json
{"book_id": 8, "volume_id": 1, "content_seq": 138,
 "text": "<ZH>乾鑿度云 …</ZH>\n<KO>《건착도》에 …</KO>\n\n",
 "n_chars_zh": 58, "n_chars_ko": 61}
```

**산출 (실측)**:
| 파일 | records | 용도 |
|---|---|---|
| `hanmed_bilingual.jsonl` | **21,043** | Stage 1 CPT 병렬 5% |
| `hanmed_zh_only.jsonl` | 23,338 | Stage 1 CPT 한문 25% |
| `hanmed_ko_only.jsonl` | 22,597 | Stage 1 CPT 국역 10% |
| `hanmed_en_only.jsonl` | 8,137 | Stage 1 제외 (§01.6 scope 밖) |

## 단계 2 — Clean (`preprocess.py::stage1_clean`)

**상태**: ❌ 미실행. eval/hashes placeholder 는 있음 (T1 1 샘플 commit).

**입력**: `data/cpt/*.jsonl`

**처리**: F1 dedup (SHA-1) + F2 길이 (5~50000) + F3 repeat run (> 0.5) + F4 언어비율 + F5 contamination (SHA-256)

**출력**: `data/cpt_processed/{name}_clean.jsonl` + `preprocess_stats.json`

```json
{"book_id": 8, ..., "text_hash_sha1": "3f2a…",
 "text": "<ZH>…</ZH>\n<KO>…</KO>"}
```

**실행 명령**:
```bash
.venv/bin/python src/data/builder/preprocess.py --stage 1 \
  --input data/cpt --output data/cpt_processed \
  --corpora hanmed_bilingual,hanmed_zh_only,hanmed_ko_only \
  --eval-hash-dir eval/hashes
```

**현재 drift** (§04a §F.3 — M2 수정 대상):
- B1: docstring `0.3` vs code `0.5` (line 12 / line 123)
- B4: contamination hash dir 미존재 시 **silent skip** (line 321-325). M2 에서 `sys.exit(2)` 로 hard-fail

**예상 산출**: dedup 로 ~5~10% 감소 예상 → ~60,000 records.

## 단계 3 — Pack (`preprocess.py::stage2_pack`)

**상태**: ❌ 미실행. **special token 추가 단계 선행 필요** (§11.2 work_order W1).

**입력**: `data/cpt_processed/*_clean.jsonl`

**처리**: Bllossom tokenizer 로 encode → greedy pack up to 2048 tokens → EOS 삽입 → BOS 1회 → pad

**출력**: `data/cpt_processed/{name}_packed_2048.jsonl`

```json
{"input_ids": [128000, 128256, 119455, ..., 128009, pad, pad]}
```
- length = 2048 (고정)
- 128000 = `<|begin_of_text|>` (BOS)
- 128256~128259 = `<ZH>`, `</ZH>`, `<KO>`, `</KO>` (tokenizer_extend 후)
- 128009 = `<|eot_id|>` (Llama-3 EOS)

**실행 명령**:
```bash
.venv/bin/python src/data/builder/preprocess.py --stage 2 \
  --input data/cpt --output data/cpt_processed \
  --tokenizer data/tokenizer/hanmed_bllossom_ext \
  --seq-len 2048
```

**현재 drift**:
- B3: `preprocess.py:315` default seq_len=1024 vs spec 2048 → CLI override 필수
- **special token 미등록 시 tag 가 3 subword 로 쪼개짐** (Bllossom 실측 `<ZH>` → `['<', 'ZH', '>']`)

**실측 토큰 예산 (Bllossom tok/char)**:

| corpus | chars | tok/char | tokens (unique) |
|---|---|---|---|
| hanmed_zh | 1.20M | 1.040 | **1.25M** |
| hanmed_ko | 1.97M | 0.745 | **1.47M** |
| bilingual (zh+ko 합) | — | — | (중복 — zh+ko 에 포함됨) |
| **HanMed unique 합 (Core 14)** | | | **~2.72M tokens** |

**Packed sequence 추정**:
- avg record ≈ 50 tokens (record당 한문 + 국역 합)
- 60,000 records × 50 tok = ~3M tokens → seq 2048 pack → **~1,500 packed sequences**

## 단계 4 — Stage 1 CPT 학습 (`cpt_trainer.py`)

**상태**: ❌ **미구현 (§F.1 P4, M2 핵심 blocker, ~400 LoC).**

**입력**: `data/cpt_processed/{name}_packed_2048.jsonl` + Wiki-ko replay / CBETA / AI Hub (M2 추가 수집)

**처리**: Bllossom-8B + LoRA r=32 + bf16 + DDP × 2. causal LM next-token prediction.

**출력**: `outputs/cpt_bllossom/adapter/` (LoRA weights + tokenizer + config + manifest)

### Core 14 실측 기반 학습 parameter

| 항목 | 값 | 근거 |
|---|---|---|
| base | Bllossom-8B | §4.2 R3.2 |
| LoRA r / α | 32 / 64 | §04a §C.5 |
| target_modules | q, k, v, o, gate_proj, up_proj, down_proj | §04a §C.5 |
| precision | bf16 | §04a §C.5 |
| micro_batch × grad_accum | 2 × 16 = 32 | §04a §C.5 |
| seq_len | 2048 | §04a §C.3 |
| LR | 1e-4 | §04a §C.5 |
| wd | 0.0 | §04a §C.5 |
| scheduler | cosine + warmup | §04a §C.5 |

### Cap 시나리오 (Bllossom 실측 기반 · R3.4)

| 시나리오 | HanMed unique | epoch | HanMed train tok | total cap (40% mix) |
|---|---|---|---|---|
| **Core 14 R3a** | 2.72M | **3** | 8.16M | **20.4M** |
| Core 14 R3b | 2.72M | 5 | 13.6M | 34M |
| Core 25 (C) R3a | 4.05M | 3 | 12.15M | 30.4M |
| Core 25 (C) R3b | 4.05M | 5 | 20.25M | 50.6M |

### Warmup · Total steps 산출

```
total_steps = ceil(cap_tokens / (micro_bs × grad_accum × seq_len × N_GPU))
            = ceil(cap_tokens / 65,536)              # single GPU
            = ceil(cap_tokens / 131,072)             # DDP 2-GPU
warmup_steps = int(total_steps × 0.05)
```

예:
- Core 14 R3a (20.4M), DDP 2 → **total 156 steps / warmup 8**
- Core 14 R3a (20.4M), single → total 311 steps / warmup 16
- Core 25 R3b (50.6M), DDP 2 → total 386 steps / warmup 19

### 예상 학습 시간 (A6000 × 2 DDP bf16)

- step 당 ~20 s (seq 2048 × effective batch 32 × Bllossom 8B bf16 + grad ckpt)
- Core 14 R3a DDP: 156 × 20 ≈ **~52 분**
- Core 25 R3b DDP: 386 × 20 ≈ **~130 분**

## 단계 5 — Demo CLI (`hanmed chat`)

**상태**: ❌ 미구현 (§10, v0 target).

**입력**: `outputs/cpt_bllossom/adapter/` + Bllossom-8B base

**처리**: vLLM LoRA serving + ChatML chat template + safety layer + streaming

**출력**: 터미널 REPL 응답

실행:
```bash
hanmed chat --adapter outputs/cpt_bllossom/adapter
```

상세는 [§10 demo CLI README](../10_demo_cli/README.md).

## 데이터 유실 · 의미 보존 체크 포인트

| 단계 | 체크 항목 | 현재 상태 |
|---|---|---|
| 1 → 2 | `\n\n` block 경계 보존 (WS_RE drift M6) | ❌ preprocess.py:74 `\s+` 가 개행 압축 — M2 수정 필요 |
| 2 → 3 | special token tokenize 단일 token 보장 | ❌ tokenizer_extend 선행 필요 |
| 2 → 3 | Stage 2 packed seq 전부 `len == seq_len` | ❌ assert 없음 — §04a §F M3 |
| 2 → 3 | contamination drop > 0.5% 시 실패 | ❌ silent skip — B4 |
| 3 → 4 | corpus_v1.json + SHA-256 pin | ❌ manifest 스크립트 미구현 |
| 4 → 5 | chat template preservation | ❌ H1 실측 gate M2 — §10 E6 |

## 요약 표

| 단계 | 코드 | 입력 | 출력 | 상태 |
|---|---|---|---|---|
| 0 | `mediclassics_orchestrator.py` | API | raw jsonl | ✅ Core 14 |
| 1 | `extract_corpora.py` | raw | `data/cpt/*.jsonl` | ✅ |
| 2 | `preprocess.py --stage 1` | `data/cpt` | `*_clean.jsonl` | ❌ 실행 대기 |
| pre-3 | `tokenizer_extend.py` | Bllossom base | `data/tokenizer/hanmed_bllossom_ext` | ❌ 미구현 |
| 3 | `preprocess.py --stage 2` | clean + tokenizer | `*_packed_2048.jsonl` | ❌ 실행 대기 |
| pre-4 | `build_corpus_manifest.py` | packed | `corpus_v1.json` (SHA pin) | ❌ 미구현 |
| 4 | `cpt_trainer.py` | packed + replay | `outputs/cpt_bllossom/adapter/` | ❌ **미구현 (M2 핵심)** |
| 5 | `hanmed chat` | adapter | REPL 응답 | ❌ v0 target |
