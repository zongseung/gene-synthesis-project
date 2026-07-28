# HanMed-VLM (ver2)

VARCO-VISION-2.0-14B(LlavaOnevision: SigLIP + Qwen3) 기반 LoRA SFT로 만드는 한의학 멀티모달 모델. 약용식물 사진을 입력받아 **종 식별 · 독성 판별 · 문헌 근거 효능 서술**을 수행하고, 근거가 없으면 명시적으로 유보(abstain)한다. 핵심 요구는 정확도가 아니라 **환각 억제** — 독초를 안전하다고 답하는 것이 최악의 실패 모드다.

이전 세대(Gemma-3-12B 텍스트+RAG)는 `ver1/`에 보존되어 있다 — [ver1/README.md](ver1/README.md).

---

## Quick Start

### 환경

이 서브프로젝트는 `korean-medicine-llm/.venv`(Python 3.12)에 전용 가상환경을 갖는다. 저장소 루트의 `.venv`(3.13.6)는 다른 프로젝트(HiPoDiT) 것이며 `hanja`와 완전한 `transformers`가 없다.

```bash
cd korean-medicine-llm
# 항상 .venv/bin/python 사용. uv run 금지(장시간 작업 중 의존성 재동기화 위험).
PYTHONHASHSEED=0 PYTHONPATH=src .venv/bin/python -m hanmed.training.train --config configs/sft_text.yaml --dry_run
```

주의:
- **`uv sync`로 이 venv를 재구성할 수 없다** — 현재 `pyproject.toml`(`requires-python>=3.10`)과 `uv.lock`이 numpy 버전 제약과 충돌해 `uv sync`가 즉시 resolution 실패한다(실측, 2026-07-29). 기존 `.venv`를 그대로 쓸 것. 의존성을 바꿀 때는 `uv pip install --dry-run`으로 먼저 확인한다.
- `torchrun`도 반드시 `.venv/bin/torchrun` + `PYTHONHASHSEED=0` prefix. system `torchrun`은 다른 pyarrow/datasets 버전이라 크래시한다.
- `vllm` 설치 금지 — torch 2.6.0+cu124 → 2.11.0 교체를 강제해 4bit 경로를 깬다.

### 학습 — 2단 SFT, 트레이너 하나

**파이프라인 순서는 LLM → VLM이다.** 두 단계 모두 같은 트레이너 `src/hanmed/training/train.py` 하나로 돈다 — 코드가 다른 게 아니라, config가 어떤 키를 설정하느냐로 단계가 갈린다. `text_train`만 있으면 텍스트 전용(1차), `tongue_train`/`herb_train`이 있으면 멀티모달(2차)이다. 이것이 이 코드베이스에서 가장 흔히 오해되는 부분이다.

```bash
# 1차 — 텍스트 SFT (이미지 0장, 고전 번역문으로 도메인 언어 적응)
PYTHONHASHSEED=0 PYTHONPATH=src .venv/bin/python -m hanmed.training.train \
    --config configs/sft_text.yaml --dry_run     # 모델 로드 없이 데이터셋/콜레이트만 검증
PYTHONHASHSEED=0 PYTHONPATH=src .venv/bin/python -m hanmed.training.train \
    --config configs/sft_text.yaml               # 실제 학습

# 2차 — 멀티모달 SFT (설진+약초 통합, 1차 replay 8.8% 포함)
PYTHONHASHSEED=0 PYTHONPATH=src .venv/bin/python -m hanmed.training.train \
    --config configs/sft_varco.yaml --dry_run
PYTHONHASHSEED=0 PYTHONPATH=src .venv/bin/python -m hanmed.training.train \
    --config configs/sft_varco.yaml --max_steps 2 --max_samples 16   # smoke (모델 로드+forward+LoRA)
PYTHONHASHSEED=0 PYTHONPATH=src .venv/bin/python -m hanmed.training.train \
    --config configs/sft_varco.yaml               # 실제 학습
```

두 `--dry_run` 커맨드 모두 실측 확인됨 — 1차는 `train=79526`, 2차는 `train=44951`(tongue=4769, herb=36240, text=3942) 데이터셋을 로드하고 콜레이트까지 통과한다.

### 평가

```bash
# 벤치 빌드 (필요 시 재생성)
PYTHONHASHSEED=0 PYTHONPATH=src .venv/bin/python -m hanmed.bench.build_bench --out data/eval/hanmed_bench

# 채점 — 골드 기반 모의예측으로 채점기 자체를 검증
PYTHONHASHSEED=0 PYTHONPATH=src .venv/bin/python -m hanmed.bench.run_eval --demo --track track1
# 실제 예측 채점
PYTHONHASHSEED=0 PYTHONPATH=src .venv/bin/python -m hanmed.bench.run_eval --pred <예측.jsonl> --track all
```

`run_eval.py --demo --track track1` 실행 확인됨(P/R/F1 정상 산출).

---

## 모듈 지도 (`src/hanmed/`)

| 모듈 | 담당 |
|---|---|
| `corpus/` | 원천 zip → label_index parquet 보완, 의방유취 등 한문 원문 → 한국어 번역(GPT API, temperature 0 직역) |
| `knowledge/` | 종 단위 지식 카드(`annotate.py`) · 근거 사전 SQLite 빌드/조회(`build_ontology.py`) · 학명 크로스워크(`crosswalk.py`) · 한자 독음 링크 확장(`link_species.py`) |
| `stage1_llm/` | 1차 텍스트 SFT 렌더러(`build.py`) — 고전 번역문을 instruct 포맷으로 감싼다. 답변은 원문 verbatim, 창작 0 |
| `stage2_vlm/` | 2차 멀티모달 SFT 빌더 — 약초(`build_herb.py`) · 설진(`build_tongue.py`) · WebDataset 샤딩(`shard.py`) · 이미지 스테이징(`stage_images.py`); `_ablation/`에 CPT 경로(기본 미사용, ablation 보존용) |
| `training/` | 단일 트레이너 `train.py` — config 키로 1차/2차 분기, LoRA(rank64 rslora), 균형 샘플러, 4bit/bf16 겸용 |
| `shared/` | 라벨 인덱스(`label_index.py`) · 부위 어휘 단일 진실(`parts.py`) · tar 샤드 랜덤 액세스 리더(`shard_image_reader.py`) |
| `gate/` | 추론 시점 검증 — 1단(KB 조회) 구현됨(`verify.py`); 2단(NLI 함의)·3단(불확실성 탐지)은 미구현 |
| `bench/` | 평가셋 빌더(`build_bench.py`) · 채점기(`run_eval.py`) · 동결 SigLIP linear probe(`siglip_probe.py`) |

---

## 데이터

- **종 지식 카드** `data/annotations/species_annotation.jsonl` — 206종. `knowledge_status` 분포(근거 사전 실측): `linked` 80 · `ambiguous` 43 · `unlinked` 83. `tox_status` 분포: `unverified` 104 · `safe_documented` 65 · `toxic` 37.
- **이미지** 원천 라벨 인덱스 기준 651,415장(151/612 데이터셋 합산, `data/shards/herb_shard_index.json`).
- **1차 SFT** `data/sft/text_train.jsonl` 79,526행 / `text_val.jsonl` 813행 / `text_replay.jsonl`(2차용) 3,942행.
- **2차 SFT** 설진 `data/sft/tongue_sft/` train 4,769 · val 556 · test 556. 약초 `data/sft/mm_train_resolved.jsonl` 36,240행 / `mm_val_resolved.jsonl` 4,716행(이미지가 실제로 샤드에 있는 행만; `mm_train.jsonl` 51,640행 중 이미지 미확보분은 제외됨).
- **근거 사전** `data/ontology.sqlite` — `fact` 7,353건(전부 (book_id, vol, seq) 출처 보유), `species_herb` 링크 81건.

---

## 벤치마크 — `data/eval/hanmed_bench`

5트랙, 총 4,848문항(`manifest.json` 실측). 원래 6트랙 설계였으나 **track5(KISTI 색상 baseline)는 실채점 기능이 없어 삭제**했다 — `score_track5`가 예측을 보지 않고 상수만 반환했고 gold 값이 전부 `[추정]` placeholder였다. 삭제 근거는 `claudedocs/hanmed_benchmark_design.md`.

| 트랙 | n | 측정 |
|---|---|---|
| track1_tongue_byeonjeung | 556 | 설진 다중라벨 P/R/F1(카테고리별) + 변증 부분문자열 recall |
| track2_donguibogam_citation | 544 (골드 인용 937) | 동의보감 인용 precision/recall(글자단위+복합키). 골드 인용 전량이 원문에 부분문자열로 grounding 검증됨(grounding_rate=1.0) |
| track3_abstain | 68 | 가짜 약초명·가짜 설진 소견·위험 요청 등에 대한 보류 정확도, 적정/부적정 보류율(over-refusal 포함) |
| track4_herb_toxic_id | 90 (독초 30 / 비독초 60) | 종 top-1 정확도 + 독초 per-class P/R. text-only(학명 질문)라 이미지 능력은 측정하지 않음 |
| track6_herb_image | 3,590 (148종) | 종 top-1 + 독성 라벨 정확도 + abstain 판정, probe_type(species_id/toxicity/efficacy_abstain/answerable_control)별 개별 보고. 독성 다수결 baseline 59.17% |

---

## 안전 설계

이 프로젝트가 존재하는 이유가 이 절이다: **검증되지 않은 것을 안전하다고 말하면 안 된다.**

### 3값 이산 라벨, 폴백 없음

`tox_status`는 `toxic` / `safe_documented` / `unverified` 셋 중 하나이며 확률이 아니다. `unverified`를 "안전"으로 읽는 것이 이 프로젝트에서 가장 흔한 오독이다. `stage2_vlm/build_herb.py`의 `render_T2`는 이 3값을 그대로 분기하며 **폴백이 없다** — `tox_status`가 없거나 인식 불가한 값이면 `ValueError`로 죽는다. 이전 폴백은 `is_poisonous`(bool)를 봤는데 이 라벨이 "미검증"과 "무독 확인"을 구분하지 못해, `unverified` 8,085행 전부가 "독초로 분류되지 않습니다"라는 안전 단정으로 렌더되고 있었다 — fail-open 결함이었고 지금은 제거됐다.

### 게이트 1단은 독성 판정을 단독으로 책임진다

`src/hanmed/gate/verify.py`의 `verify_claim`은 독성 술어(`독성`)를 만나면 고전 본문 대조 없이 `species.tox_status` 라벨만 답한다. 동의보감이 龍葵(까마중)를 무독이라 적어도, 현대 주석이 `toxic`이면 `toxic`이다. 최악의 실패 모드(독초를 안전하다고 답함)와 직결되므로 여기엔 확률 판정도 문헌 대조도 개입시키지 않는다. 효능·주치 등 다른 술어는 KB 조회로 `supported`/`unsupported`/`no_knowledge`를 매기고, `no_knowledge`(링크 없음)면 그 주장 자체를 유보로 치환한다.

### 자동 링크는 기본적으로 신뢰하지 않는다

한자 독음 대조로 종↔표제어 링크를 80→107종으로 자동 확장했으나, 27종(현재 29종) 전수 검사에서 오류율 26%가 나왔다 — 서양등골나물(국화과 독초)이 메밀의 표제어(蕎麥)에 연결되는 식이었다. 독성 충돌 필터로는 이 오류가 안 걸린다(전부 `tox_status=unverified`). [ADR-0001](docs/adr/0001-toxicity-conflict-holds-auto-links.md)·[ADR-0002](docs/adr/0002-reading-links-are-recorded-not-served.md)에 따라 `link_grade=reading`은 `candidate`와 동일하게 취급되어 **기록만 하고 게이트가 쓰지 않는다**. 사람이 확인한 `exact_hanja` 링크만 게이트가 근거로 낸다. 독성 충돌 표시 종은 8건이며 사람 검토용으로 유지된다.

### 답변 게이트

`gate.verify.gate_answer`는 판정을 문장으로 바꾼다 — `no_knowledge`는 "문헌 기록이 없어 답변을 유보합니다", `supported`는 근거(권·seq)를 붙이고, `unsupported`는 "문헌 근거를 확인하지 못했습니다"로 유보한다. 2단(NLI 함의 판정)이 아직 없어 `unsupported`도 잠정적으로 유보 처리한다.

---

## 알려진 한계

- **62/206종에 학습 이미지가 없다.** 612 데이터셋이 부분 샤딩(121종 중 30종만 로컬 확보)이라, 종 지식 카드는 206종에 있지만 실제 이미지가 붙은 종은 144종뿐이다(`mm_train_resolved.jsonl` ∪ `mm_val_resolved.jsonl` 기준 실측). `stage2_vlm/build_herb.py`의 `write_resolved`가 이 필터링을 수행한다.
- **괄호가 든 종명 4종이 경로 라운드트립 불일치로 드롭된다.** `당백출(큰꽃삽주)` · `망강남(석결명)` · `지리강활(개당귀)` · `파(실파)` — SFT 빌더(`build_herb.py`)는 `species_annotation.jsonl`의 원본 종명(괄호 포함)으로 논리 경로를 만드는데, 샤드 인덱스는 `shard.py:59`의 `re.sub(r"[^0-9A-Za-z가-힣_-]", "_", s)`가 괄호를 밑줄로 바꿔 저장한다(`064_당백출(큰꽃삽주)` vs 인덱스의 `064_당백출_큰꽃삽주`). 두 경로가 문자열로 어긋나 이 4종은 `mm_*_resolved.jsonl`에서 전량 제외된다.
- **게이트 2단·3단은 미구현이다.** 1단(KB 조회)만 동작하며, `verify.py`의 `unsupported` 처리는 "2단 미구현이라 아직 유보한다"는 잠정 조치다. 3단(불확실성 탐지)은 `bench/siglip_probe.py`의 동결 SigLIP top-k 마진을 근거로 설계만 되어 있다(`docs/adr/0003`).
- **`uv sync`로 이 venv를 재구성할 수 없다.** 위 Quick Start 참고.

---

## 문서

- [`CONTEXT.md`](CONTEXT.md) — 용어집(종/표제어/링크, knowledge_status, tox_status, 게이트 판정값 등)
- [`claudedocs/vlm_plan/`](claudedocs/vlm_plan/) — 설계 전체. [`00_README.md`](claudedocs/vlm_plan/00_README.md)가 문서 지도, [`02_decisions.md`](claudedocs/vlm_plan/02_decisions.md)가 확정 결정과 근거
- [`docs/adr/`](docs/adr/) — 결정 기록(ADR-0001 독성 충돌, ADR-0002 독음 링크, ADR-0003 별도 분류기 폐기)

이전 세대(Gemma-3-12B 텍스트+RAG, archived)는 [`ver1/README.md`](ver1/README.md). 저장소 루트는 별개 프로젝트(HiPoDiT, 유전체 합성)이며 [`../README.md`](../README.md)에 있다.
