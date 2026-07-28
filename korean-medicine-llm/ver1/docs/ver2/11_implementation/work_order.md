# 11.2 Work Order — M2 구현 순서 7개

> 각 작업에 목적·입출력·의존·LoC 추정·exit gate 를 1:1 명세. 순서대로 진행하면 `hanmed chat` v0 데모까지 도달.

## 작업 DAG

```
W1 tokenizer_extend.py  ─┐
                          ├─► W3 Stage 1+2 실제 실행  ─► W6 CPT pilot  ─► W7 hanmed chat
W2 preprocess drift fix ─┤                              ▲
                          │                              │
                          └─► W4 build_corpus_manifest ──┤
                                                         │
                          W5 cpt_trainer.py ─────────────┘
                          (병렬)

W0 Core 25 크롤 재개 (병렬, optional)
```

## W0 — Core 25 크롤 재개 (optional, 병렬)

**목적**: HanMed unique 를 2.72M → ~4M 으로 확장해 cap 여유 확보.
**의존**: 없음.
**LoC**: 0 (기존 스크립트 재실행).
**예상 시간**: ~수시간 (서버 rate limit, 13 권 × ~250 req/60s pause).

```bash
.venv/bin/python src/data/crawler/mediclassics_orchestrator.py \
  --output data/raw/mediclassics_unified \
  --books 7,44,46,47,49,54,60,70,71,94,139,183 \
  --delay 0.5 --concurrency 2 --pause 60
```

**Exit**: 11권 추가 수집 + `corpus_stats.json` chars_zh ≥ 1.7M.

**Note**: Core 14 만으로도 pilot 가능하므로 W0 는 병렬 진행 또는 생략 가능.

---

## W1 — `tokenizer_extend.py` 구현

**목적**: Bllossom tokenizer 에 `<ZH> </ZH> <KO> </KO>` 4개 special token 추가 + 저장. Stage 2 pack 이 tag 를 single token 으로 encode 하게.
**의존**: 없음.
**LoC**: ~60.
**위치**: `src/data/builder/tokenizer_extend.py`

**기능**:
```python
# 핵심 로직
tok = AutoTokenizer.from_pretrained("MLP-KTLim/llama-3-Korean-Bllossom-8B")
added = tok.add_special_tokens({
    "additional_special_tokens": ["<ZH>", "</ZH>", "<KO>", "</KO>"]
})
tok.save_pretrained("data/tokenizer/hanmed_bllossom_ext")
# 검증: 128256~128259 id 할당 확인
```

**CLI**:
```bash
.venv/bin/python src/data/builder/tokenizer_extend.py \
  --base MLP-KTLim/llama-3-Korean-Bllossom-8B \
  --output data/tokenizer/hanmed_bllossom_ext \
  --specials "<ZH>,</ZH>,<KO>,</KO>"
```

**Exit**:
- `data/tokenizer/hanmed_bllossom_ext/tokenizer.json` 존재
- `tok.encode("<ZH>") == [128256]` (1 token)
- vocab size 128,260

---

## W2 — `preprocess.py` drift 수정 (B1/B3/B4 + M6)

**목적**: R3 리뷰어 지적 4건 중 즉시 가능한 것 해소.
**의존**: 없음.
**LoC**: ~30 edit.
**대상**: `src/data/builder/preprocess.py`

**수정 항목**:

| # | 파일:라인 | 변경 |
|---|---|---|
| B1 | `preprocess.py:12` docstring | "0.3" → "0.5" (실제 임계 일치) |
| B3 | `preprocess.py:315` | `default=1024` → `default=2048` |
| B4 | `preprocess.py:321-325` | eval_dir 부재 시 `sys.exit(2)`, drop > 0.5% 시 `sys.exit(3)`. `--allow-missing-eval` 플래그로만 skip 허용 |
| M6 | `preprocess.py:67` | `WS_RE = r"\s+"` → `r"[ \t\u3000]+"` 로 좁히고 `\n\n` 최대 2개 보존 별도 rule |

**Exit**:
- `.venv/bin/python -c "from src.data.builder.preprocess import normalize; assert normalize('<ZH>x</ZH>\n<KO>y</KO>\n\n').count('\n') >= 2"` 통과
- `eval/hashes/` 삭제 후 preprocess 실행 → exit code 2
- seq_len 미지정 실행 → 2048 default

---

## W3 — Stage 1 + Stage 2 실행

**목적**: `data/cpt/` → `data/cpt_processed/*_packed_2048.jsonl` 산출.
**의존**: W1 (tokenizer), W2 (drift fix).
**LoC**: 0 (기존 코드 실행).
**예상 시간**: Stage 1 ~5분, Stage 2 ~15분 (Bllossom tokenizer load 포함).

```bash
# Stage 1
.venv/bin/python src/data/builder/preprocess.py --stage 1 \
  --input data/cpt --output data/cpt_processed \
  --corpora hanmed_bilingual,hanmed_zh_only,hanmed_ko_only \
  --eval-hash-dir eval/hashes

# Stage 2
.venv/bin/python src/data/builder/preprocess.py --stage 2 \
  --input data/cpt --output data/cpt_processed \
  --tokenizer data/tokenizer/hanmed_bllossom_ext \
  --seq-len 2048
```

**Exit**:
- `data/cpt_processed/hanmed_{bilingual,zh_only,ko_only}_clean.jsonl` 생성
- `data/cpt_processed/hanmed_{bilingual,zh_only,ko_only}_packed_2048.jsonl` 생성
- 각 packed jsonl 의 모든 line `len(input_ids) == 2048`
- `preprocess_stats.json` 의 contamination drop_ratio < 0.5%
- 실측 packed_sequences 합계가 Bllossom 추정 (~1,500) 과 ±20% 범위

---

## W4 — `build_corpus_manifest.py` 구현

**목적**: `data/cpt_processed/corpus_v1.json` 생성 (SHA-256 핀 + git SHA + tokenizer 정보). 학습 config 에 핀 포함용.
**의존**: W3 (packed 산출).
**LoC**: ~120.
**위치**: `src/data/builder/build_corpus_manifest.py`

**출력 포맷**: `preprocessing_and_cpt_spec.md §B.3` 참조.

```bash
.venv/bin/python src/data/builder/build_corpus_manifest.py \
  --processed-dir data/cpt_processed \
  --tokenizer data/tokenizer/hanmed_bllossom_ext \
  --seq-len 2048 \
  --output data/cpt_processed/corpus_v1.json
```

**Exit**:
- `corpus_v1.json` 존재 + 모든 `*_sha256` 필드 64 자리 hex
- `git_sha` = 현 HEAD
- `tokenizer_extended = true`

---

## W5 — `cpt_trainer.py` 구현 (**M2 핵심**)

**목적**: Bllossom-8B + LoRA r=32 + bf16 DDP Stage 1 CPT 학습 스크립트.
**의존**: 없음 (코드만 — 실행은 W3/W4 완료 후).
**LoC**: **~400**.
**위치**: `src/training/cpt_trainer.py`

### 구조

```
cpt_trainer.py
├── argparse (cap_tokens, epoch_variant, lora_rank, …)
├── set_global_seed(42) + PYTHONHASHSEED assert
├── config load (corpus_v1.json pin + base model + adapter target)
├── DDP init (torch.distributed + local_rank 기반)
├── Bllossom-8B + bf16 load (from extended tokenizer path)
├── LoRA wrap (peft.LoraConfig, target_modules 7개)
├── Dataset load (packed jsonl → input_ids)
├── interleave_datasets(probabilities=[...], seed=42)
├── TrainingArguments:
│   - bf16=True, gradient_checkpointing=True
│   - warmup_steps = int(total_steps × 0.05)
│   - total_steps = cap_tokens / (effective_batch × seq_len)
│   - cosine scheduler
├── wandb init (rank 0 only, WANDB_MODE=offline)
├── Trainer.train()
├── save adapter + optimizer state + trainer state + corpus_v1.json 사본
└── sanity: chat template preservation probe (200 prompt, ΔEOT-rate)  # E6
```

### CLI

```bash
torchrun --nproc_per_node=2 src/training/cpt_trainer.py \
  --corpus data/cpt_processed/corpus_v1.json \
  --base MLP-KTLim/llama-3-Korean-Bllossom-8B \
  --tokenizer data/tokenizer/hanmed_bllossom_ext \
  --output outputs/cpt_bllossom \
  --cap-tokens 20_000_000 \
  --epoch-variant 3 \
  --lora-rank 32 --lora-alpha 64 \
  --lr 1e-4 --warmup-ratio 0.05 \
  --mix "hanmed_bilingual:0.05,hanmed_zh_only:0.25,hanmed_ko_only:0.10,wiki_ko:0.30,cbeta:0.20,aihub:0.10"
```

**Exit** (smoke 기준 — 본 pilot 전):
- `outputs/cpt_bllossom/adapter/` 존재
- `train.log` 에 final loss 기록 + grad_norm 안정
- GPU peak mem < 40 GB (DDP 2-GPU 각각)
- E6 probe: ΔEOT-rate < 2%p vs base (3 seed)

---

## W6 — Stage 1 CPT pilot 실행

**목적**: Core 14 R3a (cap 20M) 첫 pilot. §E ablation 의 "20M run" 역할.
**의존**: W3, W4, W5.
**LoC**: 0.
**예상 시간**: DDP 2-GPU 기준 **~50분** (20M / 131K ≈ 156 steps × 20s).

```bash
# 첫 pilot
torchrun --nproc_per_node=2 src/training/cpt_trainer.py \
  --cap-tokens 20_000_000 --epoch-variant 3 \
  --output outputs/cpt_bllossom_r3a_20M \
  [나머지 인자 W5 참조]

# log 모니터
tail -f outputs/cpt_bllossom_r3a_20M/train.log
```

**Exit (§10 E6 + §04a §C.8)**:
- val_loss 수렴 (초기 100 steps 내 grad_norm < 10)
- T5 KLUE-YNAT drop ≤ 3%p vs base
- T1 chrF delta vs no-CPT baseline ≥ +0.5 (null-result 기각 기준)
- E6 chat template: ΔEOT-rate < 2%p
- contamination drop_ratio ≤ 0.5% (corpus_v1.json 확인)

**실패 시 분기** (§10.9.6 / §04a §C.4.3):
- T2 factual null → RAG fallback 검토
- T4 format null → SFT 추가 검토
- grad explode → LR 5e-5 재시도

---

## W7 — `hanmed_cli/` 패키지 + `chat.py` REPL

**목적**: §10 demo CLI v0. P-CPT 경로로 터미널 질의응답.
**의존**: W6 (adapter 산출).
**LoC**: ~800 (9 모듈 분산, §10 architecture 참조).
**위치**: `src/hanmed_cli/`

### 구현 순서 (§10 마일스톤 D1~D6)

| D | 작업 | 파일 | LoC |
|---|---|---|---|
| D1 | vLLM backend + Bllossom 로드 + 단발 generate | `inference/vllm_backend.py`, `inference/base.py` | ~150 |
| D2 | REPL loop + ChatML history + streaming | `chat.py`, `conversation.py`, `main.py` | ~250 |
| D3 | Safety 2-layer + T4 20 + paraphrase 30 + 한문 10 | `safety.py`, `tests/` | ~200 |
| D4 | Session save/load + Rich render + slash cmd | `session.py`, `render.py` | ~150 |
| D5 | Transformers fallback backend | `inference/transformers_backend.py` | ~100 |
| D6 | pyproject.toml + immutable revision pin | `pyproject.toml` | ~50 |

**Exit (§10.9.2)**:
- E1 REPL < 5s, cold FT < 30s, warm FT < 5s
- E2 safety 99% core / 95% paraphrase / 90% 한문
- E3 8K context sliding preserve system
- E4 session round-trip
- E5 peak GPU mem < 30 GB
- E6 chat template preservation (H1 실측 gate)
- E7 한문 출력 UTF-8 터미널 깨짐 없음

---

## 작업 순서 요약

| 순서 | 작업 | LoC | 예상 시간 | 선행 |
|---|---|---|---|---|
| 1 | W1 tokenizer_extend | 60 | 30분 | — |
| 2 | W2 preprocess drift fix | 30 edits | 30분 | — |
| 3 | W3 Stage 1+2 실행 | 0 | 20분 | W1, W2 |
| 4 | W4 build_corpus_manifest | 120 | 1시간 | W3 |
| 5 | W5 cpt_trainer.py | 400 | 1~2일 | — (W3/W4 와 병렬) |
| 6 | W6 CPT pilot 실행 | 0 | 50분 | W3, W4, W5 |
| 7 | W7 hanmed_cli | 800 | 2~3일 | W6 |

총 ~4~6 일 + pilot 실행 시간.

**W0 (Core 25 크롤 재개)** 는 전체 기간 동안 백그라운드로 병렬 진행 가능 (서버 rate limit 이 병목).

## 바로 착수 추천

사용자 결정 필요:
1. **W1 + W2 병렬 시작** (가장 저위험, 즉시 가능)
2. **W5 cpt_trainer.py 설계 먼저** (M2 핵심이므로 blocker)
3. **W0 크롤 재개 먼저** (데이터 확장이 우선이면)

어느 것부터 코드로 옮길지 지정하시면 해당 파일 작성을 시작합니다.
