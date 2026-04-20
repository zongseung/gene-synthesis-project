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

## 1. 모델 개요

| 구분 | 값 |
|---|---|
| 제품명 | DONGUI (Shell 명령: `hanmed`) |
| 버전 | v0.1 (ver4 P-A+ CPT) |
| Base | [MLP-KTLim/llama-3-Korean-Bllossom-8B](https://huggingface.co/MLP-KTLim/llama-3-Korean-Bllossom-8B) (Llama-3 8B 한국어 추가학습본) |
| Adapter | LoRA r=32, α=64, dropout 0.05 · 7 projection (q/k/v/o + gate/up/down) |
| Extended tokenizer | Bllossom vocab 128 256 → **128 260** (+ `<ZH>/</ZH>/<KO>/</KO>`) |
| Precision | bf16 (GradScaler 미사용) |
| Context window | 8 192 tokens (base) / 4 096 serving (latency 최적) |
| Objective | Causal LM next-token prediction (self-supervised DAPT + P-A+ knowledge injection) |
| Served model name | `hanmed-p-a-plus` (OpenAI API compatible) |

## 2. 모델 작동 흐름

### 2.1 전체 파이프라인

```
[mediclassics.kr 26권 1KG 원문]
           │
           ▼  크롤 (rate-limited 병렬)
[data/raw/mediclassics_unified/book_*/vol_*.jsonl]
    한문 + 국역 + (영역) 3중 병렬 레코드
           │
           ▼  extract_corpora (prolog 삽입 · 투명 메타)
[data/cpt/{bilingual, zh_only, ko_only, synth_facts}.jsonl]
           │
           ▼  preprocess Stage 1 (clean) + Stage 2 (pack)
[data/cpt_processed/*_packed_2048.jsonl]
    book 경계 고정 · <ZH>…</ZH> 해시 gate · BOS/EOS pack
           │
           ▼  cpt_trainer (DDP 2-GPU, bf16, LoRA)
[outputs/cpt_bllossom/best_model/]   ← P-A+ CPT adapter
           │
           ▼  build_merged_model (peft.merge_and_unload)
[outputs/hanmed_merged_v0.1/]        ← merged HF safetensors
           │
           ▼  docker compose (vLLM OpenAI server)
[http://localhost:8000/v1/completions]
           │
           ▼  httpx SSE stream
[hanmed CLI]   ← 사용자 터미널
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

```bash
# 1. adapter → merged 모델
.venv/bin/python scripts/build_merged_model.py \
  --adapter outputs/cpt_bllossom/best_model \
  --output outputs/hanmed_merged_v0.1

# 2. Docker Compose 기동
cd docker && docker compose up -d --build

# 3. 헬스 확인
curl -sf http://localhost:8000/health
curl -s http://localhost:8000/v1/models | jq '.data[0]'

# 4. 샘플 질의
curl -s http://localhost:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"hanmed-p-a-plus","prompt":"동의보감 저자는 ","max_tokens":64,"temperature":0}' \
  | jq -r '.choices[0].text'
```

세부 배포 스펙: [`docs/ver4/03_serving_and_cli/`](docs/ver4/03_serving_and_cli/)

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
├── docs/                    # 상세 기획서
│   ├── ver4/                # 현 운영 버전 (P-A+ CPT)
│   │   ├── 01_validation_report.md
│   │   ├── 02_plan_v4.md
│   │   ├── 03_serving_and_cli/   # vLLM + Docker + hanmed entry
│   │   └── 04_dead_code_audit.md
│   ├── 10_cli_visual_identity/   # CLI 디자인 spec (거북 mascot v3)
│   └── ver2, ver3/          # 이전 라운드 기록 (역사)
├── src/
│   ├── data/
│   │   ├── crawler/mediclassics_orchestrator.py
│   │   ├── builder/{extract_corpora, preprocess, tokenizer_extend}.py
│   │   └── synth/expand_facts.py
│   ├── training/cpt_trainer.py     # DDP CPT 트레이너
│   ├── hanmed_cli/
│   │   ├── main.py chat.py render.py conversation.py
│   │   ├── safety.py session.py config.py
│   │   ├── inference/{base, transformers_backend, remote_openai}.py
│   │   └── prompts/{branding.py, system_v0.1.md, turtle_24col.ansi}
│   └── utils/seed.py               # 결정성 보장
├── scripts/
│   ├── build_factsheet_draft.py    # KIOM raw_text → factsheet
│   ├── build_merged_model.py       # adapter → merged HF model
│   ├── entity_delta.py             # 저자 빈도 snapshot/diff
│   ├── probe_factual.py            # T1 factual probe (base vs adapter)
│   └── verify_synth_facts.py       # synth corpus 검증
├── data/
│   ├── raw/mediclassics_unified/   # 26권 크롤 결과
│   ├── cpt/                        # extract_corpora 산출
│   ├── cpt_processed/              # preprocess Stage 1+2
│   ├── facts/core_factsheet.yaml   # 26권 fact sheet
│   ├── tokenizer/hanmed_bllossom_ext/
│   └── stats/                      # 통계 + factsheet build trace
├── eval/hashes/                    # contamination 해시 레지스트리
├── docker/                         # vLLM 서빙
│   ├── Dockerfile.vllm
│   └── docker-compose.yml
└── outputs/cpt_bllossom/           # 학습 산출 (adapter + checkpoints)
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

- 현 운영 버전: [`docs/ver4/README.md`](docs/ver4/README.md)
- 검증 보고: [`docs/ver4/01_validation_report.md`](docs/ver4/01_validation_report.md)
- 학습 기획: [`docs/ver4/02_plan_v4.md`](docs/ver4/02_plan_v4.md)
- 서빙 배포: [`docs/ver4/03_serving_and_cli/README.md`](docs/ver4/03_serving_and_cli/README.md)
- CLI 디자인: [`docs/10_cli_visual_identity/03_claude_code_style.md`](docs/10_cli_visual_identity/03_claude_code_style.md)
- 이전 라운드: `docs/ver2/`, `docs/ver3/`
