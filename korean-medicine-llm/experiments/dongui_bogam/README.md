# experiments/dongui_bogam

동의보감 (東醫寶鑑, book_008) 단일 책 기반 환각 해결 실험 (Phase A').

본 디렉토리는 **symlink 집합**이며 실제 파일은 저장소 원래 경로에 있습니다.
원본을 건드리지 않고 파일만 한 곳에 모은 구조입니다.

## 배경

- 기획서: [`docs/ver4/08_real_data_antihalluc_plan.md`](../../docs/ver4/08_real_data_antihalluc_plan.md)
- 진단 근거: `.claude/harness-evals/hanmed_cpt/round_2/supervisor.md`
- 합성 데이터 (`hanmed_synth_facts.jsonl`) 는 **학습에서 전량 제외**.
  원본 텍스트만 사용 (Pillar 1+2 실데이터 경로).

## 구조

```
experiments/dongui_bogam/
├── raw/                                # → data/raw/mediclassics_unified/book_008
│   └── vol_01~23.jsonl                 # 23권 × 총 34,040 records
├── cpt/                                # build_book008_splits.py 산출 raw shards
│   ├── bilingual.jsonl                 # <ZH>원문</ZH><KO>번역</KO> (34,039 records)
│   ├── ko_only.jsonl                   # 한자 괄호 제거한 순한글 (30,127)
│   ├── real_facts_identity.jsonl       # 허준 언급 3건
│   ├── real_facts_context.jsonl        # 서문·집례·편명 548건
│   └── prolog.jsonl                    # factsheet 기반 1줄 (수동)
├── cpt_processed/                      # preprocess.py stage1+2 산출 (all symlinks)
│   ├── *_clean.jsonl                   # stage1 dedup/품질 필터 후
│   └── *_packed_2048.jsonl             # stage2 pack (record_sep=none 적용)
├── scripts/                            # 학습·평가·검증 스크립트 (symlinks)
│   ├── build_book008_splits.py         # book_008 → 5 shard 분할
│   ├── build_factsheet_draft.py        # factsheet YAML 자동 생성
│   ├── probe_factual.py                # 환각 probe (no_repeat_ngram_size=6 적용)
│   ├── probe_adapter.py                # adapter load + generate
│   ├── verify_packed_content.py        # packed 파일 무결성
│   ├── entity_delta.py                 # entity 빈도 snapshot/diff
│   ├── tokenizer_compare.py            # Bllossom tokenizer 분해 분석
│   ├── tokenizer_probe_bllossom.py     # tokenizer 개별 probe
│   ├── tokenizer_probe_quick.py
│   ├── tokenizer_verify.py
│   ├── build_merged_model.py           # adapter + base → merged FP16 model
│   ├── fetch_book_metadata.py          # KIOM metadata crawl
│   ├── cli_mock.py / cli_oneshot_smoke.py  # CLI 검증
│   └── (verify_synth_facts.py, classify_books.py 제외 — 타 작업 영역)
├── src/                                # 핵심 전처리·학습 모듈 (symlinks)
│   ├── data/builder/
│   │   ├── extract_corpora.py          # raw → hanmed_{bi,zh,ko}.jsonl
│   │   ├── preprocess.py               # stage1 clean + stage2 pack (--record-sep 추가)
│   │   ├── build_wiki_ko.py            # wiki_ko replay corpus 빌더
│   │   └── tokenizer_extend.py         # 4 special token (<ZH>/<KO>/<JA>/<EN>)
│   └── training/cpt_trainer.py         # CPT trainer (LoRA + CORPUS_PATHS)
├── docs/                               # 기획서·진단 문서 (symlinks)
│   ├── 08_real_data_antihalluc_plan.md # 본 실험의 기획서 (ver4 §08)
│   ├── 02_plan_v4.md                   # ver4 마스터 플랜
│   ├── 05_new_token_training_methods.md # special token 학습 방법론
│   └── 07_R1_probe_results.md          # R1 probe 결과
├── harness/                            # 환각 진단 라운드 보고 (symlinks)
│   ├── round_1/ (generator/discriminator/iteration_plan)
│   └── round_2/ (generator/discriminator/supervisor + _workspace)
├── logs/
│   ├── train_ddp_failed.log            # 2-GPU DDP hang 로그 (NCCL timeout)
│   └── train_singlegpu.log             # 단일 GPU 재학습 로그 (현재 실행 중)
├── outputs/                            # → outputs/cpt_bllossom_phaseA (학습 adapter)
└── core_factsheet.yaml                 # → data/facts/core_factsheet.yaml (book_id=8)
```

모든 파일이 symlink 이므로 원본 수정은 이 폴더에서도 자동 반영됩니다.

## Phase A' 학습 설정

| 항목 | 값 |
|------|----|
| Base | `MLP-KTLim/llama-3-Korean-Bllossom-8B` |
| Mode | LoRA r=32, α=64, target=q/k/v/o/gate/up/down + embed_tokens + lm_head |
| Trainable params | 92,356,864 (1.14%) |
| Mix | ko_only 0.30 / bilingual 0.30 / real_facts_context 0.15 / real_facts_identity 0.10 / prolog 0.05 / wiki_ko 0.10 |
| Cap tokens | 5M |
| Record separator | `none` (EOS-as-separator 수정 적용) |
| GPU | 단일 (DDP 호환성 이슈로 2-GPU 포기) |

## 적용된 수정사항 (round_2 진단 기반)

| 수정 | 파일 | 변경 |
|------|------|------|
| EOS-as-separator 제거 | `src/data/builder/preprocess.py` | `--record-sep none` 옵션 추가 → record 사이 `<|eot_id|>` 삽입 안 함 |
| Probe hotfix | `scripts/probe_factual.py` | `no_repeat_ngram_size=6` 추가 (F3 loop 완화) |
| book008 corpus 등록 | `src/training/cpt_trainer.py` | `CORPUS_PATHS` 에 5개 shard 추가 |
| 한자 괄호 제거 | `scripts/build_book008_splits.py` | ko_only split 에만 적용. bilingual 은 원본 유지 |

## 생성 명령 재현

```bash
# 1. book_008 을 shard 로 분리
PYTHONHASHSEED=0 .venv/bin/python scripts/build_book008_splits.py

# 2. preprocess stage1 + stage2 (record_sep=none)
PYTHONHASHSEED=0 .venv/bin/python -m src.data.builder.preprocess \
  --input data/cpt \
  --output data/cpt_processed \
  --corpora book008_bilingual,book008_ko_only,book008_real_facts_identity,book008_real_facts_context,book008_prolog \
  --stage all \
  --record-sep none

# 3. 단일 GPU 학습
PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline \
  .venv/bin/python -m src.training.cpt_trainer \
  --output outputs/cpt_bllossom_phaseA \
  --mix "book008_ko_only:0.30,book008_bilingual:0.30,book008_real_facts_context:0.15,book008_real_facts_identity:0.10,book008_prolog:0.05,wiki_ko:0.10" \
  --cap-tokens 5000000 \
  --epoch-variant 3 \
  --seed 42

# 4. 학습 완료 후 probe
.venv/bin/python scripts/probe_factual.py \
  --adapter outputs/cpt_bllossom_phaseA/adapter \
  --probe_set outputs/probes/probe_v4_content_v2_input.jsonl \
  --output outputs/probes/phaseA_eval.jsonl
```

## 평가 계획 (학습 완료 후)

1. `outputs/probes/phaseA_eval.jsonl` 생성 — round_2 동일 10문항
2. `lora_embedding_A` norm 측정 — round_2 의 "embed LoRA 잠듦" 해소 확인
3. F3 loop / F4 글자 변형 / F2 wrong-entity 빈도 비교표

## vLLM / Docker 배포

### 빠른 배포 — LoRA direct (권장)

adapter 파일만으로 base 위에 동적 로드. merge 빌드 불필요.

```bash
# 학습 완료 + outputs/cpt_bllossom_phaseA/adapter/ 존재 확인 후
scripts/deploy_phaseA.sh direct
```

- 컨테이너: `hanmed_vllm_phaseA`
- 이미지: `hanmed-llm:phaseA`
- compose 파일: `docker/docker-compose.phaseA.yml`
- Base: `MLP-KTLim/llama-3-Korean-Bllossom-8B` (HF cache 사용)
- Adapter: `outputs/cpt_bllossom_phaseA/adapter/` 볼륨 마운트
- OpenAI 호환 endpoint: `http://localhost:8000/v1/{models,chat/completions}`

### 완전 merged 배포

adapter 를 base 에 merge 하여 단일 HF 모델로 저장 후 로드. 성능 안정적.

```bash
scripts/deploy_phaseA.sh merged              # merge + up
scripts/deploy_phaseA.sh merged --skip-build # 이미 merged 있으면 skip
```

- 컨테이너: `hanmed_vllm_phaseA_merged`
- compose: `docker/docker-compose.phaseA.merged.yml`
- Model 경로: `outputs/hanmed_merged_phaseA/` (약 16 GB)

### 공통 명령

```bash
scripts/deploy_phaseA.sh smoke      # "동의보감 편찬자?" 샘플 질문
scripts/deploy_phaseA.sh down       # 컨테이너 전체 중지

# 수동 health / API
curl http://localhost:8000/health
curl http://localhost:8000/v1/models
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "dongui-bogam",
    "messages": [{"role":"user","content":"동의보감 편찬자는?"}],
    "max_tokens": 300, "temperature": 0
  }'
```

### 어댑터 교체 (재학습 시)

LoRA direct 모드는 adapter 경로만 바꾸고 restart:
```bash
export HANMED_ADAPTER_DIR=../outputs/cpt_bllossom_phaseA_v2/adapter
cd docker && docker compose -f docker-compose.phaseA.yml restart
```

### 주의

- 학습이 GPU 를 점유 중이면 vLLM 컨테이너 기동 실패. 학습 완료 후 실행.
- `--max-lora-rank=32` 가 고정 (현재 LoRA config 와 일치).
- `lora-extra-vocab-size=256` 은 확장된 4개 special token (`<ZH>/<KO>/<JA>/<EN>`) 수용.

## CLI 연동

`src/hanmed_cli/` 는 이미 **vLLM OpenAI 호환 API** 와 연동된 터미널 REPL.
Phase A' 모델에 맞게 environment variable 을 자동 설정하는 wrapper 스크립트 제공.

### 기동

```bash
# direct mode vLLM 기동 후
scripts/deploy_phaseA.sh direct

# 별도 터미널에서 CLI
scripts/cli_phaseA.sh direct
# → endpoint: http://localhost:8000/v1, model: dongui-bogam
# → splash 출력 후 REPL 진입

# merged mode 인 경우
scripts/cli_phaseA.sh merged
# → model: hanmed-phaseA
```

### CLI 구조 (이미 존재하는 모듈)

`src/hanmed_cli/` — Click + prompt_toolkit 기반 REPL
```
hanmed_cli/
├── main.py                     # Click entry (bare `hanmed` = splash + remote REPL)
├── chat.py                     # REPL loop (pre-safety → conversation → backend → post-safety)
├── conversation.py             # chat_template 렌더, sliding window
├── config.py                   # Defaults SSoT
├── session.py                  # 대화 저장/복원
├── safety.py                   # pre/post-check (refusal, footer)
├── render.py                   # 배너, 스트리밍 출력
├── inference/
│   ├── base.py                 # Backend ABC
│   ├── vllm_backend.py         # 로컬 vLLM python API
│   ├── transformers_backend.py # 로컬 HF (디버그용)
│   └── remote_openai.py        # ← docker vLLM 연결 (default)
└── prompts/
    ├── system_v0.1.md          # system prompt
    └── turtle_24col.ansi       # 거북이 마스코트 ANSI 아트
```

### Phase A' 용 독립 분리

`experiments/dongui_bogam/cli/hanmed_cli/` → 원본 심볼릭 링크.
`experiments/dongui_bogam/scripts/cli.sh` → `scripts/cli_phaseA.sh` 심볼릭.
CLI 코드 자체는 공유 (단일 버전), 실험별로 env var 로 endpoint/model 만 달리 지정.

### 환경변수

| 변수 | 기본값 (direct) | 기본값 (merged) | 설명 |
|------|----------------|-----------------|------|
| `HANMED_ENDPOINT` | `http://localhost:8000/v1` | (동일) | vLLM OpenAI 호환 API |
| `HANMED_MODEL` | `dongui-bogam` | `hanmed-phaseA` | served model name (docker-compose 의 `--served-model-name` / `--lora-modules` 에 등록된 이름) |

### Slash 명령 (REPL 내)

```
/help            명령 목록
/save <name>     세션 저장
/load <name>     세션 로드
/temp 0.3        sampling temperature 조정
/max 512         max_new_tokens 조정
/tokens          현재 context token 수
/reset           대화 초기화
/exit            종료
```

### 로컬 디버그 (vLLM 없이)

컨테이너 띄우지 않고 adapter 를 직접 로드해 볼 때:

```bash
.venv/bin/python -m hanmed_cli.main chat \
  --backend transformers \
  --adapter outputs/cpt_bllossom_phaseA/adapter
```
단 8B 모델을 CPU/GPU 에 직접 로드하므로 무거움. vLLM 배포가 권장.

## 주의

- `outputs/` symlink 는 학습 중이라 아직 adapter 가 없을 수 있음. 학습 완료 후 `outputs/adapter/` 또는 `outputs/checkpoint-NNN/` 생성됨.
- 원본 파일을 이동하지 마세요. symlink 가 끊어집니다.
- 향약집성방 등 다른 책은 `experiments/{book_name}/` 로 같은 패턴 생성 예정.
