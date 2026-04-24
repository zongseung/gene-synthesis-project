<p align="center">
  <img src="../hammed_icon/HanMed_1.png" alt="HanMed mascot — turtle apothecary" width="280">
</p>

<h1 align="center">HanMed-LLM</h1>

<p align="center">
  <em>한의학 고전 해제 도우미 · Llama-3-Korean-Bllossom-8B 기반 LoRA CPT + vLLM 서빙 + 터미널 CLI</em>
</p>

---

동의보감·향약집성방·의방유취 등 조선 의서를 학습해 저자·왕대·편찬 정보·주요 처방에 대한 **서지/해제 질문**에 답한다. 임상 진단·처방 권고 용도 아님.

---

## 0. 현재 상태 & 버전 히스토리

| 버전 | 날짜 | Paradigm | Base / Adapter | 데이터 scope | 상태 |
|---|---|---|---|---|---|
| **v0.1 (ver4 P-A+)** | 2026-04 초 | CPT (LoRA r=32) | Bllossom-8B + 34권 mix | 20.4 M tok cap, 156 steps | 과거 운영 (`hanmed-p-a-plus` merged) |
| **Phase A'** (ver4 §08) | 2026-04 중 | CPT 단권 | Bllossom-8B + book_008 | 5 M tok cap | 비교군 adapter (`cpt_bllossom_phaseA`) |
| **ver5 v3.1 (SFT)** | **2026-04-22 현재 운영** | **Fresh SFT** (TRL, LoRA r=32) | **Bllossom-8B** + `phaseB_qa_diverse_v3_1.jsonl` | 21,475쌍 (train 18,254 / val 3,221), 1 epoch, LR 2e-5 | **서빙 중** (`hanmed_merged_ver5_v3_1`, Llama arch / vocab 128,260) |
| ver6 r2 (Gemma) | 2026-04-23 계획 | SFT on Gemma-3 12B-IT | `models/gemma-3-12b-it` (로컬 23 GB) + book_008 | 19,023쌍 (cov 89.72%) | zero-probe 실측 완료, **본선 전환 전 상태** (실험 adapter: `experiments/dongui_bogam/outputs_ver6_gemma_v1/`) |

전환 근거는 각 기획서에 정리돼 있다:

- ver4 → ver5: **CPT 한계 3중 확증** (질문 표현 fragility / safety refusal 0% / 재실행 비결정성) — [`docs/ver5/01_experimental_evidence.md`](docs/ver5/01_experimental_evidence.md)
- ver5 → ver6: **SFT 환각(F1) · 반복(F3) 공동 원인** (답변 템플릿 42% 동질성 · 임상 QA 0건 · 허위 citation · LoRA `embed_tokens` 포함) — [`docs/ver6/00_halluc_repetition_fix_plan.md`](docs/ver6/00_halluc_repetition_fix_plan.md)

## 1. 모델 개요

| 구분 | ver4 (v0.1) | **ver5 v3.1 (current)** | ver6 r2 (planned) |
|---|---|---|---|
| 제품명 | DONGUI | DONGUI | DONGUI |
| Shell 명령 | `hanmed` | `hanmed` | `hanmed` |
| Base | [Bllossom-8B](https://huggingface.co/MLP-KTLim/llama-3-Korean-Bllossom-8B) | **Bllossom-8B (동일)** | [`google/gemma-3-12b-it`](https://huggingface.co/google/gemma-3-12b-it) ※ 현재 전환 **미완료** |
| Adapter | LoRA r=32, α=64, dropout 0.05 · 7 proj | LoRA r=32 · 7 proj (embed 제외) | LoRA r=32, α=64 · target TBD |
| Tokenizer 확장 | 128 256 → **128 260** (`<ZH>/</ZH>/<KO>/</KO>`) | 동일 (ver4 와 호환) | Gemma 262 144 vocab (미확장) |
| Precision | bf16 | bf16 | bf16 |
| Objective | Causal LM next-token (CPT) | TRL SFT (completion-only loss) | TRL SFT (완료 loss) |
| Context (base/서빙) | 8 192 / **4 096** | 8 192 / **4 096** | 8 192 / 4 096 |
| Served model name | `hanmed-p-a-plus` | `hanmed-sft-ver5-v3-1` | `hanmed-gemma-ver6` |
| Merged weight 경로 | `outputs/hanmed_merged_v0.1` | `outputs/hanmed_merged_ver5_v3_1` | TBD |

## 2. 모델 작동 흐름

### 2.1 전체 파이프라인 (ver4 CPT / ver5 SFT 공통)

```
[mediclassics.kr 26권 1KG 원문]
           │
           ▼  크롤 (rate-limited 병렬)
[data/raw/mediclassics_unified/book_*/vol_*.jsonl]
    한문 + 국역 + (영역) 3중 병렬 레코드
           │
           │────────────────────────┬──────────────────────────┐
           ▼                        ▼                          ▼
  ver4 CPT 경로            ver5 SFT 경로 (현 운영)       ver6 경로 (계획)
           │                        │                          │
  extract_corpora          build_sft_qa/diverse            동일 book_008
  + prolog 삽입            → phaseB_qa_diverse_v3_1         + Gemma tokenizer
                           (21,475쌍 = 18,254 + 3,221)
           │                        │                          │
  [data/cpt/{bi,zh,          [data/sft/                   [data/sft/
    ko,synth}.jsonl]           book008_full_sft.jsonl]     gemma_*.jsonl]
           │                        │                          │
  preprocess (clean/pack)    audit_sft_diversity +        audit_sft_diversity
  book 경계 assertion        augment_sft_v7 (refusal)     (동일)
           │                        │                          │
  cpt_trainer (DDP 2-GPU)    sft_trainer (TRL,            sft_trainer
   ver4 §2.1 mix             completion-only loss)         (Gemma preset)
   (synth25/bi35/zh15/ko25)  3 epoch, LR 2e-5
           │                        │                          │
  [outputs/cpt_bllossom/]    [outputs/cpt_bllossom_       [outputs/ver6_*/]
                              ver5_v3_1/adapter]
           │                        │                          │
           └─────────┬──────────────┴──────────────────────────┘
                     ▼  scripts/build_merged_model.py (peft.merge_and_unload)
           [outputs/hanmed_merged_*/]   ← merged HF safetensors + ext tokenizer
                     │
                     ▼  docker/docker-compose{.phaseA,.merged,.gemma}.yml (vLLM)
                     │
           [http://localhost:8000/v1/completions]
                     │
                     ▼  httpx SSE stream
                     │
                  [hanmed CLI]
```

### 2.2 단계별 역할

| 단계 | 스크립트 | 입력 | 출력 | 핵심 역할 |
|---|---|---|---|---|
| 수집 | `src/data/crawler/mediclassics_orchestrator.py` | book_id 목록 | `data/raw/mediclassics_unified/` | 권별 병렬 크롤, content_seq resume |
| 추출 | `src/data/builder/extract_corpora.py` | raw jsonl | `data/cpt/*.jsonl` | 3중 병렬 → corpus 분리 + 권당 1회 `book_meta_prolog` 삽입 (한자 병기, hanja ratio ≥ 0.10 gate) |
| 합성 | `src/data/synth/expand_facts.py` | `data/facts/core_factsheet.yaml` | `data/cpt/hanmed_synth_facts.jsonl` | 26권 fact sheet → template × paraphrase×4 → 1 791 paragraph (entity validation 100%) |
| 정제 | `src/data/builder/preprocess.py` (Stage 1) | cpt jsonl | `*_clean.jsonl` | SHA-1 dedup / 품질 필터 / `<ZH>…</ZH>` 정규화 contamination gate |
| 패킹 | `src/data/builder/preprocess.py` (Stage 2) | clean jsonl | `*_packed_2048.jsonl` | Bllossom ext tokenizer → greedy pack, **book 경계 assertion** 으로 cross-book blend 차단 |
| 학습 | `src/training/cpt_trainer.py` | packed jsonl 4 corpus | LoRA adapter | DDP 2-GPU, interleave 비중 (synth 25 / bi 35 / zh 15 / ko 25), cosine_with_min_lr, best_model tracking |
| 병합 | `scripts/build_merged_model.py` | best adapter | merged HF model | `peft.merge_and_unload` + ext tokenizer 통합 저장 |
| 서빙 | `docker/docker-compose.yml` | merged model | OpenAI API endpoint | vLLM 0.7.0, bf16, max_num_seqs 16, gpu_util 0.85 |
| 클라이언트 | `src/hanmed_cli/` | stdin | 스트림 응답 | Click CLI + Rich splash + httpx SSE |

### 2.3 Prolog 주입 (§2 — P-A+ 핵심)

책별 첫 블록 앞에 **200~400 token long-form 서문** 1회 삽입. Fact sheet 값(저자·왕대·연도·장르·주요 처방)만 치환해 자유 생성 금지. 예:

```
『東醫寶鑑』은 조선의 어의 허준(許浚)이 선조(宣祖)의 명을 받아 1596년 착수하여
1610년 완성하고 1613년(광해군 5년)에 간행한 종합 의서이다. …
```

→ entity binding 빈도를 raw corpus 대비 약 **113 ×** 증폭 (허준 43 → 928, 이제마 3 → 628, 이시진 932 → 0).

## 3. 리소스 요구사항

### 3.1 학습 (one-time)

| 항목 | 요구 |
|---|---|
| GPU | **NVIDIA A6000 48 GB × 2** (DDP) / A100 40 GB × 2 도 가능 |
| GPU VRAM (per card) | ~48 GB 사용 (bf16 + LoRA r=32 + optimizer + activation checkpointing) |
| CPU | 8 코어 이상 (DataLoader num_workers=4) |
| 시스템 RAM | ≥ 32 GB |
| 디스크 | ≥ 50 GB (raw 110 MB + cpt 180 MB + cpt_processed 380 MB + model 16 GB × 2 체크포인트) |
| 학습 시간 | **~3 시간** (20.4M tokens cap, 156 steps, ~73 s/step) |
| Throughput | ~65 K tokens / step · ~ 890 tokens/s aggregate |

### 3.2 서빙 (정상 운영)

| 항목 | 요구 |
|---|---|
| GPU | **1장**, VRAM ≥ 20 GB (Bllossom-8B bf16 + KV cache 4 096 × 16 seqs) |
| 권장 | RTX 4090 24 GB / A6000 48 GB / A100 40 GB |
| CPU | 2 코어 |
| 시스템 RAM | ≥ 16 GB |
| 디스크 | ~20 GB (merged model 16 GB + docker image ~4 GB) |
| Docker + nvidia-container-toolkit | 필수 |
| 지연 | TTFT ≤ 1.5 s · Streaming ~50 tok/s per request |

### 3.3 클라이언트 (CLI)

| 항목 | 요구 |
|---|---|
| Python | ≥ 3.10 |
| 의존성 | `click>=8.1 rich>=13.7 prompt_toolkit>=3.0 httpx>=0.27` |
| 네트워크 | 서빙 엔드포인트(`HANMED_ENDPOINT`) 도달 가능 |
| 터미널 | 24-bit truecolor 지원 권장 (xterm-256color + `COLORTERM=truecolor`) |

## 4. 데이터 스펙

### 4.1 코퍼스 규모

| 항목 | 값 |
|---|---|
| Source | KIOM mediclassics.kr 한의학고전DB |
| 책 수 | **26권** (한국 한의학 핵심 Core 14 + 확장 12권) |
| Raw records | **182 978** (한문 + 국역 + 영역 3중) |
| 한국어 번역률 | 80.1% (146 629 records) |
| 학습 corpus | bilingual 114 973 / zh-only 140 925 / ko-only 111 425 / synth 1 791 |
| Packed sequences | 9 077 / 4 802 / 4 749 / 266 (seq_len 2 048) |
| 총 토큰 | ~38.7 M (mix 가중 샘플링 후 20.4 M 학습 투입) |

### 4.2 Factsheet (P-A+ knowledge injection seed)

`data/facts/core_factsheet.yaml` — 26 권 수기 검증 fact sheet

| 필드 | 충족도 | 비고 |
|---|---|---|
| `author_hanja` | 26 / 26 | 모든 권 저자 한자 확보 |
| `reign_hanja` | 16 / 26 | 조선 왕대 (중국·일제강점기 10권 제외) |
| `published_year` | 20 / 26 | 연도 확인된 권만 |
| `genre` | 17 / 26 | 장르 수기 분류 |
| `topics` / `signature_items` | 17 / 9 | 주요 권만 상세 |

### 4.3 데이터 파일 포맷

**raw** (mediclassics_unified):
```json
{"book_id": 8, "volume_id": 1, "content_seq": 138,
 "original": "乾鑿度云 …", "trans_ko": "《건착도》에 …", "trans_en": "…"}
```

**bilingual block** (학습용):
```
<ZH>乾鑿度云 …</ZH>
<KO>《건착도》에 …</KO>
```

**packed** (tokenizer 적용 후):
```json
{"input_ids": [128000, 128256, 119455, ..., 128009, 128001, 128001]}
// 128000 = BOS, 128009 = EOS, 128001 = pad
// 128256 = <ZH>, 128257 = </ZH>, 128258 = <KO>, 128259 = </KO>
```

## 5. 학습 프로토콜

| 하이퍼파라미터 | 값 |
|---|---|
| Optimizer | AdamW (β₁ 0.9, β₂ 0.95, weight_decay 0.0) |
| LR | 1 × 10⁻⁴ |
| LR scheduler | `cosine_with_min_lr` (min_lr_rate 0.1) |
| Warmup | total_steps × 5% (= 7 steps @ cap 20.4M) |
| Effective batch | micro 2 × grad_accum 16 × world 2 = **64 sequences / step** |
| Tokens per step | 131 072 |
| `num_train_epochs` | 3 (cap 20.4M · 1 epoch ≈ 38M tokens 이므로 실효 ~0.54 epoch) |
| Val split | 2% per-corpus, seed=42 |
| Eval interval | 50 steps (save_steps = eval_steps) |
| `load_best_model_at_end` | True, `metric_for_best_model = eval_loss` |
| `modules_to_save` | `["embed_tokens", "lm_head"]` (vocab resize 반영) |

Mix (ver4 §2.1 P-A+ 설계):

| corpus | 비중 | packed seqs |
|---|---|---|
| `hanmed_synth_facts` | **25%** | 266 (synth oversample ~9.4×) |
| `hanmed_bilingual` | **35%** | 9 077 |
| `hanmed_zh_only` | **15%** | 4 802 |
| `hanmed_ko_only` | **25%** | 4 749 |

## 6. 추론 / 서빙 프로토콜

### 6.1 OpenAI API 호환 엔드포인트

- 기본 URL: `http://localhost:8000/v1`
- 지원 엔드포인트: `/models`, `/completions` (stream SSE), `/health`, `/tokenize`
- 모델명 파라미터: `hanmed-p-a-plus`
- vLLM 옵션: `--dtype bfloat16 --max-model-len 4096 --max-num-seqs 16 --gpu-memory-utilization 0.85`

### 6.2 Prompt 포맷 (Llama-3 ChatML)

```
<|start_header_id|>system<|end_header_id|>

<system prompt>
<|eot_id|><|start_header_id|>user<|end_header_id|>

<user message>
<|eot_id|><|start_header_id|>assistant<|end_header_id|>

```

클라이언트(`hanmed_cli.conversation.Conversation`)가 tokenizer chat template 로 자동 생성 → prompt-based `/v1/completions` 로 전송 (chat template 이중 적용 회피).

### 6.3 샘플링 기본값

| 인자 | 기본 |
|---|---|
| `temperature` | 0.2 |
| `top_p` | 0.9 |
| `max_new_tokens` | 512 |
| `repetition_penalty` | 1.05 |

## 7. CLI 사용

```bash
# 설치 (editable)
.venv/bin/python -m ensurepip --upgrade
.venv/bin/pip install --no-deps -e .

# 기본 실행 (vLLM 서버 기동 전제)
hanmed

# splash 만 (백엔드 없이)
hanmed --splash-only

# 원격 엔드포인트 override
HANMED_ENDPOINT=https://api.example.com/v1 hanmed

# 로컬 backend (디버그용)
hanmed chat --backend transformers --adapter outputs/cpt_bllossom/best_model
```

REPL 슬래시 명령: `/help /exit /reset /save <n> /load <n> /temp <f> /max <n> /tokens`

## 8. 서빙 배포

### 8.1 Docker Compose 변형 5종

용도별로 별개의 compose 파일을 둔다 (서빙 경로/모델/포트만 다르며 Dockerfile 은 공통 `Dockerfile.vllm`).

| 파일 | 모델 경로 | 모드 | 용도 |
|---|---|---|---|
| `docker-compose.yml` | `outputs/adapter_current` (symlink) | **LoRA direct** | 매 이터레이션 어댑터 hot-swap (`ln -sfn cpt_bllossom_R{n}/adapter ..`) |
| `docker-compose.merged.yml` | `outputs/hanmed_merged_v0.1` | merged | ver4 P-A+ 정식 배포 (selectable model name `hanmed-p-a-plus`) |
| `docker-compose.phaseA.yml` | `outputs/cpt_bllossom_phaseA/adapter` | LoRA direct | Phase A' 실험군 빠른 검증 |
| `docker-compose.phaseA.merged.yml` | `outputs/hanmed_merged_phaseA` | merged | Phase A' 안정 서빙 |
| `docker-compose.gemma.yml` | `models/gemma-3-12b-it` | 무학습 base | ver6 zero-training probe (`google/gemma-3-12b-it`) |

> **현재 서빙**: ver5 SFT v3.1 merged (`outputs/hanmed_merged_ver5_v3_1`) — `docker-compose.merged.yml` 을 `--model` override 해 기동하거나, 전용 compose 를 `ver5_v3_1` 경로로 패치해 사용한다.

### 8.2 배포 절차 (ver5 예시)

```bash
# 1. adapter → merged 모델
PYTHONHASHSEED=0 .venv/bin/python scripts/build_merged_model.py \
  --adapter outputs/cpt_bllossom_ver5_v3_1/adapter \
  --output  outputs/hanmed_merged_ver5_v3_1

# 2. Docker Compose 기동 (merged)
cd docker && docker compose -f docker-compose.merged.yml up -d --build

# 3. 헬스 확인
curl -sf http://localhost:8000/health
curl -s http://localhost:8000/v1/models | jq '.data[0]'

# 4. 샘플 질의
curl -s http://localhost:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"hanmed-sft-ver5-v3-1","prompt":"동의보감 저자는 ","max_tokens":64,"temperature":0}' \
  | jq -r '.choices[0].text'
```

ver4 CPT 배포 스펙: [`docs/ver4/03_serving_and_cli/`](docs/ver4/03_serving_and_cli/) · ver6 Gemma 전환 계획: [`docs/ver6/00_halluc_repetition_fix_plan.md`](docs/ver6/00_halluc_repetition_fix_plan.md)

## 9. 재현성

| 항목 | 보장 방식 |
|---|---|
| PYTHONHASHSEED | 모든 데이터 스크립트 `PYTHONHASHSEED=0` prefix 필수 (진입부 `set_global_seed` 가 assert) |
| CUBLAS 결정성 | `CUBLAS_WORKSPACE_CONFIG=:4096:8` (seed.py 가 import 시점 세팅) |
| PyTorch 결정성 | `torch.use_deterministic_algorithms(True, warn_only=True)` + cudnn deterministic |
| 데이터 해시 | corpus_stats.json / preprocess_stats.json 에 source paths + counts 기록 |
| 학습 config | `train_manifest.json` 로 학습 완료 시점에 모든 HP 기록 |
| 체크포인트 | `save_total_limit=2` + best model tracking → last + best 2 개 유지 |

## 10. 디렉토리 구조

```
korean-medicine-llm/
├── README.md                # 이 파일 (모델 스펙)
├── pyproject.toml           # hanmed CLI 엔트리 포인트
├── uv.lock                  # uv 환경 lockfile
│
├── docs/                    # 상세 기획서 (버전별 · self-contained)
│   ├── 01_overview ~ 09_roadmap/   # 주제별 원자 문서 (r0)
│   ├── ver2/, ver3/                # 초기 라운드 기록 (역사)
│   ├── ver4/                       # CPT P-A+ (v0.1 과거 운영)
│   │   ├── 01_validation_report.md
│   │   ├── 02_plan_v4.md
│   │   ├── 03_serving_and_cli/
│   │   ├── 04_dead_code_audit.md
│   │   └── 08_real_data_antihalluc_plan.md  # Phase A' 기획
│   ├── ver5/                       # SFT 전환 (현 운영)
│   │   ├── README.md
│   │   ├── 01_experimental_evidence.md  # CPT 한계 3중 확증
│   │   ├── 02_sft_design.md
│   │   ├── 03_data_pipeline.md
│   │   ├── 04_trainer_spec.md           # TRL SFTTrainer
│   │   ├── 05_evaluation.md
│   │   ├── 06_safety.md
│   │   ├── 07_roadmap.md
│   │   ├── 08_sft_build_plan.md
│   │   └── 09_v4_complex_reasoning_plan.md
│   ├── ver6/                       # Gemma-3 12B 전환 계획
│   │   ├── 00_halluc_repetition_fix_plan.md
│   │   └── appendix_bllossom_fallback.md
│   ├── 10_cli_visual_identity/     # 거북 mascot spec + png2ascii 툴
│   └── research_hanmed_cpt_methodology_20260421.md
│
├── src/
│   ├── data/
│   │   ├── crawler/mediclassics_orchestrator.py
│   │   ├── builder/{extract_corpora, preprocess, tokenizer_extend, build_wiki_ko}.py
│   │   └── synth/expand_facts.py
│   ├── training/
│   │   ├── cpt_trainer.py          # ver4 DDP CPT
│   │   └── sft_trainer.py          # ver5+ TRL SFTTrainer (Bllossom/Gemma preset)
│   ├── hanmed_cli/
│   │   ├── main.py chat.py render.py conversation.py
│   │   ├── safety.py session.py config.py
│   │   ├── inference/{base, transformers_backend, remote_openai}.py
│   │   └── prompts/{branding.py, system_v0.1.md, turtle_24col.ansi}
│   └── utils/seed.py               # 결정성 보장
│
├── scripts/                        # 각 단계별 operator CLI
│   ├── (build) build_sft_qa.py / build_sft_full_corpus.py / build_sft_diverse.py /
│   │          build_sft_clinical.py / build_sft_complex.py / build_book008_splits.py
│   ├── (audit) audit_sft_diversity.py / augment_sft_v7.py / _v7_refusal_variants.py
│   ├── (verify) verify_synth_facts.py / verify_sft_against_raw.py /
│   │            verify_packed_content.py / tokenizer_verify.py / tokenizer_compare.py
│   ├── (probe) probe_factual.py / probe_adapter.py / probe_ver6_quick.py /
│   │           probe_ver7_{quick,data_grounded,pregnancy,prescription}.py /
│   │           gemma_zero_probe.py / gemma_zero_probe_transformers.py
│   ├── (deploy) build_merged_model.py / deploy_phaseA.sh / cli_phaseA.sh /
│   │            cli_mock.py / cli_oneshot_smoke.py
│   ├── (meta)   fetch_book_metadata.py / build_factsheet_draft.py /
│   │            classify_books.py / entity_delta.py
│
├── data/
│   ├── raw/mediclassics_unified/   # 26권 크롤 결과
│   ├── cpt/                        # ver4 extract_corpora 산출
│   ├── cpt_processed/              # ver4 preprocess Stage 1+2
│   ├── sft/                        # ver5 SFT QA 코퍼스
│   │   ├── book008_full_sft.jsonl          # 34,039쌍 (book_008 full raw → Q/A)
│   │   ├── book008_full_sft_sample.jsonl   # 20쌍 (smoke)
│   │   ├── phaseB_qa_full_corpus.jsonl     # Phase B 전체 corpus
│   │   ├── phaseB_qa_diverse_v3.jsonl      # v3 diverse
│   │   ├── phaseB_qa_diverse_v3_1.jsonl    # ★ ver5 v3.1 실제 학습 입력 (21,475쌍)
│   │   ├── phaseB_qa_complex_v4.jsonl      # complex reasoning 증강
│   │   ├── complex_seeds.yaml              # complex QA 템플릿 시드
│   │   ├── entity_whitelist{,_v6}.yaml     # audit_sft_diversity 검증용
│   │   └── *.stats.json / *.validation.json
│   ├── facts/core_factsheet.yaml   # 26권 fact sheet
│   ├── tokenizer/hanmed_bllossom_ext/   # 128 260 vocab
│   └── stats/                      # 통계 + factsheet build trace
│
├── eval/
│   ├── README.md
│   ├── hashes/                     # contamination 해시 레지스트리
│   └── hanmed_eval_v0/             # probe 입력 번들
│       ├── phaseA_eval_input.jsonl        # 43쌍 (Phase A' 검증용)
│       ├── phaseB_complex_probe.jsonl     # 23쌍 (ver5 complex reasoning)
│       ├── probe_v4_final_input.jsonl     # 4쌍 (ver4 final gate)
│       └── T1_content.jsonl               # 10쌍 (factual recall)
│
├── experiments/
│   └── dongui_bogam/               # book_008 단권 실험 (symlink 집합)
│       ├── README.md
│       ├── raw/ cpt/ cpt_processed/ scripts/ src/ docs/ harness/ logs/
│       ├── outputs/                        # → outputs/cpt_bllossom_phaseA
│       ├── outputs_ver5_book008_full/      # ver5 v1 산출
│       ├── outputs_ver5_book008_full_smoke/# ver5 smoke 산출
│       ├── outputs_ver6_gemma_v1/          # ver6 Gemma 첫 산출
│       └── outputs_ver7_gemma_patched/     # ver6 후속 패치
│
├── docker/                         # vLLM 서빙 (compose 5종 → §8.1 참조)
│   ├── Dockerfile.vllm
│   ├── docker-compose.yml                     # LoRA direct hot-swap
│   ├── docker-compose.merged.yml              # ver4 P-A+ merged
│   ├── docker-compose.phaseA.yml              # Phase A' LoRA direct
│   ├── docker-compose.phaseA.merged.yml       # Phase A' merged
│   └── docker-compose.gemma.yml               # ver6 Gemma-3 12B probe
│
├── models/                         # HF 로컬 weights cache (gitignored)
│   └── gemma-3-12b-it/             # ver6 base (23 GB, 5 shards)
│
└── outputs/                        # 학습 산출 (gitignored)
    ├── cpt_bllossom/                   # ver4 v0.1 adapter + checkpoints
    ├── cpt_bllossom_phaseA/            # Phase A' adapter
    ├── cpt_bllossom_R1/                # ver4 R1 재학습
    ├── cpt_bllossom.synth_run/         # synth 실험
    ├── cpt_bllossom_ver5{,_v2,_v3,_v3_1,_v4_sanity}/  # ver5 SFT 이터레이션
    ├── hanmed_merged_v0.1(.synth)/     # ver4 merged
    ├── hanmed_merged_R1/               # ver4 R1 merged
    ├── hanmed_merged_ver5_v3_1/        # ★ 현 서빙 merged
    ├── adapter_current/                # LoRA direct hot-swap symlink
    └── probes/                         # probe 실행 로그
```

## 11. 라이선스

| 구성 | 라이선스 | 조건 |
|---|---|---|
| mediclassics 데이터 | KIOM 비상업 무료 이용 | 출처 표기 = "한의학고전DB (mediclassics.kr)". 상업 이용은 `kiombook@kiom.re.kr` 서면 문의 |
| Bllossom-8B base | Llama 3 Community License | 상업 이용 조건부 허용 |
| HanMed adapter | 연구용 (기본) | 가공물 공개는 KIOM 사전 승인 |
| 본 저장소 코드 | TBD | 연구·교육 목적 |

## 12. 면책

이 모델은 **한의학 고전 문헌 해제 도우미**이며 임상 진단·처방·의료 조언 도구가 아닙니다. 생성된 응답은 원문 해제 보조일 뿐 의학적 판단의 근거가 될 수 없습니다. 자격 있는 한의사·의사와 상담하십시오.

## 13. 문서 인덱스

### 현재 운영 (ver5 SFT)
- 개요: [`docs/ver5/README.md`](docs/ver5/README.md)
- CPT 한계 실증: [`docs/ver5/01_experimental_evidence.md`](docs/ver5/01_experimental_evidence.md)
- SFT 설계: [`docs/ver5/02_sft_design.md`](docs/ver5/02_sft_design.md)
- SFT 데이터 파이프라인: [`docs/ver5/03_data_pipeline.md`](docs/ver5/03_data_pipeline.md)
- TRL Trainer 스펙: [`docs/ver5/04_trainer_spec.md`](docs/ver5/04_trainer_spec.md)
- 평가 프로토콜: [`docs/ver5/05_evaluation.md`](docs/ver5/05_evaluation.md)
- Safety refusal 설계: [`docs/ver5/06_safety.md`](docs/ver5/06_safety.md)
- Phase C 로드맵: [`docs/ver5/07_roadmap.md`](docs/ver5/07_roadmap.md)
- SFT 구축 플랜: [`docs/ver5/08_sft_build_plan.md`](docs/ver5/08_sft_build_plan.md) · Complex reasoning: [`docs/ver5/09_v4_complex_reasoning_plan.md`](docs/ver5/09_v4_complex_reasoning_plan.md)

### 차기 계획 (ver6 Gemma)
- 환각·반복 공동 해소 기획: [`docs/ver6/00_halluc_repetition_fix_plan.md`](docs/ver6/00_halluc_repetition_fix_plan.md)
- Bllossom fallback 부록: [`docs/ver6/appendix_bllossom_fallback.md`](docs/ver6/appendix_bllossom_fallback.md)

### 과거 라운드 (참고)
- ver4 P-A+ CPT: [`docs/ver4/README.md`](docs/ver4/README.md) · 검증 [`01_validation_report.md`](docs/ver4/01_validation_report.md) · 학습 [`02_plan_v4.md`](docs/ver4/02_plan_v4.md) · 서빙 [`03_serving_and_cli/README.md`](docs/ver4/03_serving_and_cli/README.md) · dead-code [`04_dead_code_audit.md`](docs/ver4/04_dead_code_audit.md) · Phase A' [`08_real_data_antihalluc_plan.md`](docs/ver4/08_real_data_antihalluc_plan.md)
- CPT 방법론 연구노트: [`docs/research_hanmed_cpt_methodology_20260421.md`](docs/research_hanmed_cpt_methodology_20260421.md)
- 이전 라운드: `docs/ver2/`, `docs/ver3/`

### CLI & 시각 아이덴티티
- Claude Code 스타일: [`docs/10_cli_visual_identity/03_claude_code_style.md`](docs/10_cli_visual_identity/03_claude_code_style.md)
- 거북 mascot draft: [`docs/10_cli_visual_identity/01_turtle_apothecary_draft.md`](docs/10_cli_visual_identity/01_turtle_apothecary_draft.md) · 채팅박스 레이아웃: [`02_chatbox_layout.md`](docs/10_cli_visual_identity/02_chatbox_layout.md)

### 실험
- book_008 단권 실험 (Phase A'/B): [`experiments/dongui_bogam/README.md`](experiments/dongui_bogam/README.md)
