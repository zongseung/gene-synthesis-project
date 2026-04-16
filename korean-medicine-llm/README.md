# HanMed-LLM

한의학 고전 + 한국어 도메인 특화 LLM — **Bllossom-8B 위 bf16 LoRA CPT** (옵션 SFT) 프로젝트.

데이터: 한국한의학연구원 [mediclassics.kr](https://mediclassics.kr) — 한문 원문 + 국역 + 영역 3중 병렬.

상세 기획서: [`docs/ver2/README.md`](docs/ver2/README.md) — **ver2.2 R3.5** (Bllossom primary 전환 + demo CLI 분할 정합성 보강 완료).

---

## 1. 현재 상태 (2026-04-16)

| 단계 | 상태 |
|---|---|
| Core 14 수집 | ✅ 14권 (chars_zh 1.20M / chars_ko 1.97M, ko coverage 90.2%) |
| Core 25 확장 | ⏳ 대기 (11권 추가, [§W0](docs/ver2/11_implementation/work_order.md)) |
| Stage 1 clean + Stage 2 pack | ✅ 19,609 + 19,802 + 19,186 → packed 2,745 seqs (seq_len 2048) |
| Tokenizer 확장 (§A.4) | ✅ Bllossom vocab 128,260 (+ `<ZH>/</ZH>/<KO>/</KO>`) |
| corpus_v1.json (SHA-256 핀) | ✅ [`data/cpt_processed/corpus_v1.json`](data/cpt_processed/corpus_v1.json) |
| **Stage 1 CPT pilot** | 🔥 진행 중 (cap 20.4M tokens, DDP 2-GPU, ~156 steps) |
| Demo CLI (`hanmed chat`) | ✅ v0 skeleton (transformers backend), 실제 추론은 adapter 완성 후 |
| Core 25 / 평가셋 / SFT | 🧭 M2 로드맵 |

## 2. 환경

- Python 3.10+, Linux + CUDA 12.x
- GPU: **RTX A6000 48 GB × 2** (DDP), bf16 LoRA 기준 1장으로도 동작
- 주요 패키지: `torch 2.5.1+cu121`, `transformers`, `peft==0.13.2`, `wandb`, `click`, `prompt_toolkit`, `rich`
- **주의 (재현성)**: 장시간 학습은 `.venv/bin/python` 사용 (`uv run` 금지 — mid-session dep resync 회피)

```bash
uv sync
# 필요 시 개별 설치
uv pip install --python .venv/bin/python peft==0.13.2 wandb prompt_toolkit
```

## 3. 파이프라인 — 수집 → 전처리 → 학습 → 데모

`docs/ver2/11_implementation/pipeline_data_flow.md` 가 정본. 아래는 재현 순서.

### 3.1 데이터 수집 (Core 14 이미 완료)

```bash
# Core 14 (현재 저장소가 이 상태)
python3 src/data/crawler/mediclassics_orchestrator.py \
  --output data/raw/mediclassics_unified \
  --books 8,56,69,86,93,182,291,1,4,9,24,38,59,100

# Core 25 확장 (선택, 11권 추가 — 수시간 소요)
python3 src/data/crawler/mediclassics_orchestrator.py \
  --output data/raw/mediclassics_unified \
  --books 7,44,46,47,49,54,60,70,71,94,139,183
```

Resume 자동 (중단해도 `max(content_seq)+1` 부터 이어감).

### 3.2 Tokenizer 확장 (W1)

```bash
PYTHONHASHSEED=0 .venv/bin/python src/data/builder/tokenizer_extend.py
# → data/tokenizer/hanmed_bllossom_ext/  (vocab 128,260, ids 128256~128259)
```

### 3.3 Raw → bilingual/zh/ko 분리 (§03.3)

```bash
PYTHONHASHSEED=0 .venv/bin/python src/data/builder/extract_corpora.py \
  --input data/raw/mediclassics_unified --output data/cpt
# → data/cpt/hanmed_{bilingual,zh_only,ko_only,en_only}.jsonl + corpus_stats.json
```

### 3.4 Stage 1 clean + Stage 2 pack (W2/W3)

```bash
# Stage 1: dedup / quality / contamination (§D G1, G5)
PYTHONHASHSEED=0 .venv/bin/python src/data/builder/preprocess.py --stage 1 \
  --input data/cpt --output data/cpt_processed \
  --corpora hanmed_bilingual,hanmed_zh_only,hanmed_ko_only \
  --eval-hash-dir eval/hashes

# Stage 2: Bllossom tokenizer + pack (BOS/EOS/pad, seq_len 2048)
PYTHONHASHSEED=0 .venv/bin/python src/data/builder/preprocess.py --stage 2 \
  --input data/cpt --output data/cpt_processed \
  --tokenizer data/tokenizer/hanmed_bllossom_ext --seq-len 2048
```

검증:
```bash
PYTHONHASHSEED=0 .venv/bin/python scripts/tokenizer_verify.py       # V1~V5 (special tokens, round-trip)
PYTHONHASHSEED=0 .venv/bin/python scripts/verify_packed_content.py  # C1~C7 (BOS, pad, tag pair, len=2048)
```

### 3.5 corpus manifest (W4 — SHA-256 pin)

```bash
PYTHONHASHSEED=0 .venv/bin/python src/data/builder/build_corpus_manifest.py
# → data/cpt_processed/corpus_v1.json (§B.3)
```

### 3.6 Stage 1 CPT 학습 (W5/W6)

```bash
# DDP 2-GPU, cap 20.4M tokens (Core 14 R3a, epoch≈3)
PYTHONHASHSEED=0 WANDB_MODE=offline \
.venv/bin/torchrun --nproc_per_node=2 src/training/cpt_trainer.py \
  --cap-tokens 20400000 --epoch-variant 3 \
  2>&1 | tee outputs/cpt_bllossom/train.log

# dry-run (모델 로드 없이 파이프라인만)
PYTHONHASHSEED=0 .venv/bin/python src/training/cpt_trainer.py --dry-run \
  --cap-tokens 20400000 --epoch-variant 3
```

§C.5 default HP (LoRA r=32/α=64, LR 1e-4, cosine warmup 5%, bf16, micro 2×accum 16, seq 2048).
자동 계산: `total_steps = ⌈cap / (2·16·2048·N_GPU)⌉`, `warmup = ⌊total·0.05⌋`.
val split 2% per-corpus (§C.7), eval 매 50 steps.

### 3.7 Demo CLI (`hanmed chat`)

```bash
# base 만 (adapter 완성 전)
PYTHONPATH=src .venv/bin/python -m hanmed_cli.main chat --backend transformers

# P-CPT adapter 사용 (학습 완료 후)
PYTHONPATH=src .venv/bin/python -m hanmed_cli.main chat \
  --adapter outputs/cpt_bllossom/adapter --mode cpt

# 세션 관리
PYTHONPATH=src .venv/bin/python -m hanmed_cli.main sessions list
```

REPL slash 명령: `/help /exit /reset /save <n> /load <n> /temp <f> /max <n> /tokens`.

세부 스펙: [`docs/ver2/10_demo_cli/README.md`](docs/ver2/10_demo_cli/README.md).

## 4. 디렉토리

```
korean-medicine-llm/
├── README.md                      # 이 파일
├── pyproject.toml
├── docs/ver2/                     # 기획서 (ver2.2 R3.5)
│   ├── README.md                  # 섹션 01~11 인덱스
│   ├── 01_overview/
│   ├── 02_data_source/
│   ├── 03_data_pipeline/
│   ├── 04_model_strategy/         # §04a preprocessing_and_cpt_spec (APPROVE)
│   ├── 05_evaluation/
│   ├── 07_license_ethics/
│   ├── 10_demo_cli/               # 11 파일 (§10)
│   └── 11_implementation/         # pipeline + work_order + current_state
├── src/
│   ├── data/
│   │   ├── crawler/mediclassics_orchestrator.py   # multi-process 크롤
│   │   └── builder/
│   │       ├── extract_corpora.py                 # raw → bilingual/zh/ko
│   │       ├── preprocess.py                      # Stage 1 + Stage 2
│   │       ├── tokenizer_extend.py                # special token 4개 추가
│   │       └── build_corpus_manifest.py           # SHA-256 pin
│   ├── training/
│   │   ├── cpt_trainer.py                         # DDP CPT 학습 (W5)
│   │   └── smoke_cpt.py                           # 파이프라인 smoke
│   ├── hanmed_cli/                                # v0 데모 CLI (§10)
│   │   ├── main.py conversation.py safety.py
│   │   ├── session.py chat.py render.py config.py
│   │   ├── inference/{base,transformers_backend,vllm_backend}.py
│   │   └── prompts/system_v0.1.md
│   └── utils/seed.py                              # PYTHONHASHSEED assert + 전역 seed
├── scripts/
│   ├── tokenizer_compare.py           # 7 tokenizer probe
│   ├── tokenizer_probe_bllossom.py    # Bllossom 세부
│   ├── tokenizer_verify.py            # V1~V5 검증
│   └── verify_packed_content.py       # packed jsonl C1~C7 검증
├── data/
│   ├── raw/mediclassics_unified/      # 크롤 결과 (Core 14)
│   ├── cpt/                           # extract_corpora 산출
│   ├── cpt_processed/                 # preprocess Stage 1+2 + corpus_v1.json
│   ├── tokenizer/hanmed_bllossom_ext/ # 확장 tokenizer
│   └── stats/tokenizer_verify.json    # 검증 리포트
├── eval/
│   ├── README.md
│   └── hashes/heldout_T1.txt          # contamination placeholder (R3)
└── outputs/cpt_bllossom/              # 학습 산출
    └── adapter/                       # (학습 완료 후 LoRA weights)
```

## 5. 모델 · 학습 요지

| 항목 | 값 | 근거 |
|---|---|---|
| Base | **MLP-KTLim/llama-3-Korean-Bllossom-8B** | §4.2 R3.2 (tokenizer 실측 1위, byte_fallback 0%) |
| Adapter | LoRA r=32, α=64, dropout 0.05 | §C.5 |
| Target modules | q/k/v/o + gate/up/down (7개) | §C.5 |
| Precision | bf16 | §C.5 (GradScaler 불필요) |
| Seq length | 2048 | §C.3 |
| Objective | causal LM next-token prediction (self-supervised, DAPT) | §C.1 |
| Mix (v0) | HanMed bilingual 5 + zh 25 + ko 10 = 40% (HanMed only — replay 는 M2) | §C.2 |
| cap tokens (Core 14 R3a) | 20.4M | §C.4.3 |
| Scheduler | cosine w/ warmup = int(total_steps × 0.05) | §C.5 |
| Val split | 2% per-corpus, seed=42 | §C.7 |

## 6. 데이터 파일 포맷

**raw** (`data/raw/.../vol_*.jsonl`):
```json
{"book_id": 8, "volume_id": 1, "content_seq": 138, "content_level": "ZZ",
 "original": "乾鑿度云 …", "trans_ko": "《건착도》에 …", "trans_en": "…"}
```

**cpt block** (`data/cpt/hanmed_bilingual.jsonl`):
```json
{"book_id": 8, "text": "<ZH>乾鑿度云 …</ZH>\n<KO>《건착도》에 …</KO>\n\n",
 "n_chars_zh": 58, "n_chars_ko": 61}
```

**packed** (`data/cpt_processed/hanmed_*_packed_2048.jsonl`):
```json
{"input_ids": [128000, 128256, 119455, ..., 128009, 128001, 128001]}
// 128000 = BOS <|begin_of_text|>, 128009 = EOS <|eot_id|>, 128001 = pad, 128256 = <ZH> ...
```

## 7. 라이선스

- **mediclassics 데이터**: KIOM 비상업 무료 이용 (§07). 상업 이용은 `kiombook@kiom.re.kr` 서면 문의. 출처 표기 의무 = "한의학고전DB (mediclassics.kr)".
- **본 코드**: 연구용. 가공물·adapter 공개는 KIOM 사전 승인 후.
- **Bllossom-8B base**: Llama 3 Community License (조건부 상업 가능).

## 8. 다음 작업 (M2 로드맵)

| # | 작업 | 상태 |
|---|---|---|
| W0 | Core 25 확장 크롤 (Wiki-ko/CBETA/AIHub replay 수집 포함) | ⏳ |
| W6 | CPT pilot 완주 + loss·ppl 리포트 | 🔥 |
| — | §E ablation (cap 20M vs 60M vs 200M) | 🧭 |
| — | §05 HanMed-Eval v0 curation (T1~T5, T4 redteam 20 + paraphrase 30 + 한문 10) | 🧭 |
| — | hanmed_cli 실제 adapter 로 REPL 테스트 | 🧭 |

세부 계획: [`docs/ver2/11_implementation/work_order.md`](docs/ver2/11_implementation/work_order.md).

## 9. 트러블슈팅

| 증상 | 원인 | 대응 |
|---|---|---|
| `AttributeError: torch.float8_e8m0fnu` | peft ≥ 0.15 + torch < 2.6 조합 | `uv pip install peft==0.13.2` |
| `WandbCallback requires wandb` | report_to=["wandb"] 인데 미설치 | `uv pip install wandb` + `WANDB_MODE=offline` |
| `PYTHONHASHSEED must be '0'` | 재현성 guard | 명령 앞에 `PYTHONHASHSEED=0` prefix |
| `contamination hash dir not found` (exit 2) | §D G1 gate | `eval/hashes/` 디렉토리 생성 또는 `--allow-missing-eval` |
| packed jsonl len ≠ 2048 | Stage 2 재실행 누락 | `rm data/cpt_processed/*_packed_*.jsonl` 후 §3.4 재실행 |
| OOM (학습 중) | 2nd process 가 GPU 점유 | `nvidia-smi` 확인, 다른 python 프로세스 종료 |
| 크롤러 HTTP 405 무한반복 | 책당 quota 소진 | `--pause 90~120` 로 상향 |

진행 중 프로세스 일괄 중지:
```bash
pkill -f mediclassics_orchestrator   # 크롤러
pkill -f cpt_trainer                 # 학습
```

## 10. 문서 인덱스

- [전체 기획서 ver2.2 R3.5](docs/ver2/README.md)
- [모델 전략 §04](docs/ver2/04_model_strategy/base_model_and_training.md) · [전처리/CPT §04a](docs/ver2/04_model_strategy/preprocessing_and_cpt_spec.md)
- [데모 CLI §10 (11 파일)](docs/ver2/10_demo_cli/README.md)
- [구현 계획 §11](docs/ver2/11_implementation/README.md) — pipeline / work_order / current_state
- [harness 검증 로그](/.claude/harness-evals/hanmed-cpt-spec/) — R1~R3.5
