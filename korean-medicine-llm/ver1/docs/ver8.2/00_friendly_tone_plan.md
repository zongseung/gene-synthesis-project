# ver8.2 · 00. 친절체 톤 SFT 라운드 기획서 (Hybrid Gold + LLM Rewrite)

- **버전**: r0 (2026-04-28)
- **선행 문서**:
  - [`docs/ver8.1/04_round_2_log_and_convergence.md`](../ver8.1/04_round_2_log_and_convergence.md) — ver8.1 SFT 수렴 로그
  - [`experiments/dongui_bogam/docs/09_phase_B_sft_plan.md`](../../experiments/dongui_bogam/docs/09_phase_B_sft_plan.md) — Phase B 원안
- **실증 근거**:
  - `experiments/dongui_bogam/outputs_ver8_1_gemma_v1/probe_ver8_1_rag_v4.log` — RAG retrieval 정확도 검증 통과
  - `experiments/dongui_bogam/outputs_ver8_1_gemma_v1/probe_ver8_1_rag_info_v2.log` — INFO mode 검증
  - 본 라운드 trigger: `hanmed-bogam` REPL 의 "인삼/사물탕/통설산" 답변 톤이 한문 직역체로 고착, system prompt 강화로도 변화 없음 (2026-04-28 검증)
- **본 라운드의 위치**:
  - ver8.1 = 사실 정확도 + safety 라운드 (수렴)
  - **ver8.2 = 답변 어투 라운드** (친절체 mix)
  - ver8.3+ = GraphRAG / 변형처방 식별 (예정)

---

## 0. 한 줄 요약

**ver8.1 LoRA 가 base Gemma3-12B-IT 의 친절체 표현력을 한문 직역체로 distribution narrowing 시킨 것이 확인됨 (system prompt 룰 6·7 강화 + temperature 0.5 시도 모두 실패 — 모델이 학습 분포를 못 벗어남). 본 라운드는 (a) gold 100 rows 수작업 친절 답변 + (b) base Gemma3-12B-IT 로 ver8.1 의 답변 10,000 rows 를 친절체로 rewrite 하는 hybrid 전략으로 friendly_qa_v0.jsonl ~10,100 rows 를 생성, 기존 34,039 rows 와 mix (23%) 한 phaseB_qa_v8_2_corpus.jsonl 을 만들고 LoRA round 2 학습 후 merged_text_v2 로 재배포한다 (Plan B 채택 — §11 참조). 평가 metric 은 7-query before/after 비교에서 (1) 본문 인용 보존율, (2) 친절체 풀이 비율 ≥ 80%, (3) 한자 정규화·boost retrieval 회귀 0 건이다. 비용은 base Gemma3 가 이미 GPU 0 에 보유돼 있어 LLM rewrite API 비용 0, 총 작업기간 3-4일.**

---

## 1. 진단 — 왜 친절체 SFT 라운드가 필요한가

### 1.1 ver8.1 의 한문 직역체 답변 패턴 (실측)

`hanmed-bogam` REPL 으로 다음 query 던졌을 때 (2026-04-28):

```
> 인삼의 성미·주치·금기·법제를 동의보감 본문에 따라 정리해줘

정리:
성미: 약간 따뜻하고따뜻하다고도 한다 맛은 달며약간 쓰다고도 한다 독이 없다.
주치: 주로 오장의 기가 부족한 데 쓴다. 정신과 혼백을 안정시키고 눈을 밝게 하며,
심을 열어 지혜를 더하고 허손을 치료하며, 곽란으로 구토하고 딸꾹질하는 것을 멎게
하고 폐위로 고름을 토하는 것을 치료하며, 담을 삭인다.
...
```

특징:
- **한문 직역체** ("오장의 양을 보한다", "담을 삭인다") 그대로
- **협주(夾註) 가 괄호 없이 박힘** ("따뜻하고따뜻하다고도 한다")
- **풀이 부재** — 룰 6 가 "일반 사용자 이해" 부연을 요구해도 모델이 또 본문 인용
- **disclaimer 단조 반복** — 매 답변 끝 "...로 한정해 읽어 주십시오" 박힘

### 1.2 system prompt 만으로 fix 시도한 결과 (실패)

| 시도 | 결과 | 근거 |
|---|---|---|
| 룰 6 강화 ("자연스러운 현대 한국어") | 풀이 섹션 자체 사라짐 | `rag_service/main.py` SYSTEM_RAG patch r1 |
| 룰 7 끝 "풀이 반드시 추가" 강제 | 풀이 라벨은 들어오지만 또 본문 인용 | patch r2 |
| temperature 0.0 → 0.5 | **악화** — prompt 의 [발췌] 섹션을 답변으로 통째 복제 | patch r3, revert |
| max_tokens 400 → 800 | 효과 없음 | patch r3 |

→ **system prompt 는 "지시" 이지 "학습 분포 변경" 이 아님**. 모델이 한문 직역체 corpus 로 학습된 이상 풀어쓰지 못한다.

### 1.3 distribution narrowing 의 흔적

`phaseB_qa_v8_1_corpus.jsonl` (34,039 rows) 의 답변 형식 sample:

```
정리:
인삼

해설:
이 대목은 동의보감의 편명·항목명 역할을 하는 표제입니다. 상위 경로는 ...
허준이 편찬한 동의보감의 서지·편명 정보로 한정해 읽어 주십시오.
```

```
현대 한국어:
일화자본초. 송나라 사람의 저술로, 이름을 쓰지 않았다.

해설:
이 대목은 內景篇卷之一 > 歷代醫方에 속한 본문 설명입니다. ...
현대 의료 조언이 아니라 고전 텍스트 해설입니다.
```

→ 모든 답변이 `정리:/현대 한국어:` + `해설:` 2-section 단조 형식. 친절체 패턴 거의 0.

### 1.4 base 능력 보존 가설

base Gemma3-12B-IT 는 base 단독 추론 시 친절체 표현력 정상 (HF 공개 모델 일반 평가). LoRA SFT 가 이 분포를 좁힌 것 — base 의 친절체 능력은 weight 안에 잠재해 있고, **친절체 SFT row 가 활성화 신호로 작동하면 회복 가능**.

---

## 2. 설계 — Hybrid (Gold + LLM Rewrite)

### 2.1 방법론 비교

| 옵션 | 방법 | row | 비용 | 품질 | 채택? |
|---|---|---|---|---|---|
| (a) 수작업만 | 사람 직접 작성 | 100-300 | 1-2일 | 🟢 | **부분** (gold 100) |
| (b) Local LLM rewrite | base Gemma3-12B-IT 로 ver8.1 답변 변환 | 2-15K | $0 + ~5시간 GPU | 🟡 | **메인** (rewrite 10K) |
| (c) Claude/GPT API | 외부 API rewrite | 2-15K | $30-80 | 🟢 | 미채택 (비용·외부 의존) |
| (d) Hybrid = (a)+(b) | gold few-shot + LLM rewrite | ~10.1K | $0 + 3-4일 | 🟢 | **채택 (Plan B)** |

### 2.2 Sample size 결정 (Plan B 채택)

ver8.1 base 34,039 rows 기준 mix ratio:

| mix % | friendly rows | 효과 | Plan |
|---|---|---|---|
| 0.5% | 170 | ❌ 무시 | — |
| 1.5% | 500 | △ 약함 | — |
| 5.8% | 2,000 | 🟢 보수적 활성화 | A (fallback) |
| 10% | 3,400 | 🟢 강함 | — |
| **23%** | **10,000** | **🟢🟢 효과 명확** | **B (채택)** |
| 30% | 14,500 | ⚠ formal 톤 dilute 경계 | — |
| 100% (rewrite-only) | 34,000 | ⚠⚠ catastrophic forgetting 위험 | C (escalation) |

→ **10,100 rows = 23% mix (Plan B) 채택**. 친절체 효과를 §8 평가에서 명확히 측정할 수 있는 비율이며, 30% dilute 가드 안쪽이라 ver8.1 의 fact·safety 분포 보존 가능. (Plan A 5.8% 는 fallback, Plan C 100% 는 §11.5 escalation 으로 reserved.)

### 2.3 Gold (수작업 100 rows) 의 역할

- **LLM rewrite few-shot 제시**: rewrite prompt 에 in-context example 3-5개 사용 → 일관된 톤 보장
- **고난이도 케이스 manual 작성**: 한문 해석, 변형 처방 비교 등 LLM rewrite 가 fail 하기 쉬운 카테고리
- **회귀 평가용 reference**: rewrite 산출물의 품질 비교 기준

---

## 3. 데이터 Plan

### 3.1 ver8.1 corpus 분포 (실측)

```
17,085  병증 설명     (50.2%)
11,078  편명         (32.5%)
 5,319  본문 설명     (15.6%)
   465  서문         (1.4%)
    92  총목         (0.3%)
─────────
34,039  total
```

### 3.2 Gold 100 rows — 카테고리 stratified

데모 인터랙션 카테고리 (현재 RAG query 들과 부합) 70 + ver8.1 stratified 30:

| 그룹 | 카테고리 | gold rows |
|---|---|---|
| **데모 인터랙션** (70) | | |
| | 약재 단방 (성미·주치·금기·법제) | 20 |
| | 처방 풀버전 (적응증·구성·용법·출전) | 20 |
| | 한자 정규화 query (인삼·단삼고·진경환) | 10 |
| | 한문 해석 (서문·序例) | 10 |
| | 비교·다중 검색 (두통 처방 등) | 10 |
| **ver8.1 stratified** (30) | | |
| | 병증 설명 | 15 |
| | 편명 | 10 |
| | 본문 설명 | 4 |
| | 서문 | 1 |

### 3.3 LLM rewrite 10,000 rows — stratified sample (Plan B)

ver8.1 의 카테고리 비율 유지하며 random sample:

| 카테고리 | rewrite rows | 비율 | 비고 |
|---|---|---|---|
| 병증 설명 | 5,000 | 50% | 가장 많은 데모 query 영향 |
| 편명 | 3,300 | 33% | 편·문 구조 query |
| 본문 설명 | 1,600 | 16% | 본문 풀이 query |
| 서문 | 70 | 0.7% | 편찬 배경 query |
| 총목 | 30 | 0.3% | 카테고리 표제 |

→ **총 10,000 rows**. ver8.1 의 카테고리 분포 보존하며 23% mix.

### 3.4 합계 — `friendly_qa_v0.jsonl`

```
gold (수작업)         100
rewrite (LLM)      10,000
─────────────────
total              10,100 rows
```

---

## 4. 생성 파이프라인

### 4.1 Step 1: Gold seeds 작성 (수작업)

산출물: `data/sft/friendly_gold_v0.jsonl` (100 rows)

각 row 형식 (ver8.1 와 schema 동일):

```json
{
  "id": "friendly_gold_v0/herb_단방/001",
  "category": "friendly_qa",
  "subcat": "약재_단방",
  "question": "인삼은 어떤 약재인가요? 성미와 주된 효능을 알려주세요.",
  "assistant": "인삼은 동의보감 湯液篇 草部 上에 등장하는 대표적인 보기약입니다.\n\n발췌: \"성질이 약간 따뜻하고 맛은 달며 약간 쓰다고도 한다, 독이 없다. 주로 오장의 기가 부족한 데 쓴다.\" [湯液篇 > 草部 上 > 人參 심]\n\n쉽게 말하면, 인삼은 따뜻한 성질과 단맛이 주된 약재로 오장의 기운을 보충하는 데 쓰입니다. 단 폐화가 동하기 쉬운 사람이나 혈허·음허인 사람에게는 적합하지 않다고 동의보감은 명시하니, 실제 복용은 한의사 진료가 필요합니다.",
  "messages": [
    {"role": "system", "content": "당신은 한의학 고전 문헌 연구 보조 AI 입니다. ..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "source_records": ["book_008/vol_06/seq_xxx"],
  "tone": "friendly",
  "_origin": "manual_gold_v0"
}
```

핵심 패턴:
1. **서문 한 문장** — 약재/처방을 1줄로 자연스럽게 소개
2. **발췌 인용** — 한문 본문은 그대로 (RAG path 표기)
3. **풀어쓰기** — "쉽게 말하면", "즉" 으로 시작하는 친절 부연
4. **자연스러운 면책** — disclaimer 가 아니라 사용자 안전 안내 톤

### 4.2 Step 2: ver8.1 rewrite sample 추출

```bash
python scripts/build_friendly_rewrite_sample.py \
    --input data/sft/phaseB_qa_v8_1_corpus.jsonl \
    --output data/sft/v8_1_rewrite_sample.jsonl \
    --strategy stratified \
    --counts 병증=1000,편명=660,본문=320,서문=15,총목=5 \
    --seed 42
```

### 4.3 Step 3: Base Gemma3-12B-IT 로 LLM rewrite

산출물: `data/sft/friendly_rewrite_v0.jsonl` (2,000 rows)

#### 4.3.1 Rewrite 스크립트 (`scripts/rewrite_to_friendly.py`)

```python
# 핵심 흐름
base = AutoModelForImageTextToText.from_pretrained(BASE_GEMMA3_PATH, ...)
# (LoRA adapter X — base 만 사용해야 친절체 표현력 활용)

REWRITE_SYSTEM = """당신은 한의학 고전 문헌의 답변을 친절한 어투로 풀어쓰는 작업을 합니다.
규칙:
1. 원문 발췌(한문/한자체)는 한 글자도 바꾸지 말고 그대로 인용하세요.
2. 발췌 앞에는 1-2문장 자연스러운 도입을, 뒤에는 2-3문장 풀이를 한국어로 추가하세요.
3. 발췌에 없는 약재명·처방명·수치·인용은 절대 만들지 마세요.
4. disclaimer "...로 한정해 읽어 주십시오" 같은 정형구는 자연스러운 표현으로 바꾸세요.
5. 학술 톤은 유지하되 일반 사용자가 이해할 수 있게 풀어쓰세요."""

# few-shot: gold 의 3 rows 를 in-context example 로
for row in v8_1_rewrite_sample:
    messages = [
        {"role": "system", "content": REWRITE_SYSTEM},
        # gold few-shot 3개
        *[
            {"role": "user", "content": f"원답변: {gold_i.assistant}"},
            {"role": "assistant", "content": gold_i.assistant_friendly}
        ] for gold_i in selected_gold[:3],
        {"role": "user", "content": f"원답변: {row.assistant}"},
    ]
    rewritten = generate(messages, max_new_tokens=600, temperature=0.3)
    yield {**row, "assistant": rewritten, "_origin": "llm_rewrite_v0",
           "_source_assistant": row.assistant, "tone": "friendly"}
```

#### 4.3.2 자원

| 항목 | 값 |
|---|---|
| GPU | 0 (48 GB 단독) |
| Base 모델 | `models/gemma-3-12b-it` (multimodal 원본, ~24 GB bf16) |
| Batch | 4 (rewrite prompt token 평균 ~1,000 → 활용 메모리 여유) |
| Throughput | ~30-40 rows/min (greedy 0.3, max 600) |
| 총 시간 | 10,000 rows / 35 = ~286 min ≈ **5시간** |
| 비용 | $0 (local) |

### 4.4 Step 4: Fact drift 검증

산출물: `data/sft/friendly_qa_v0.jsonl` (~2,100 rows, gold + rewrite filter 통과분)

검증 차원:
1. **Entity whitelist** — 발췌의 약재·처방·인용서 토큰이 그대로 보존됐는지 (`data/sft/entity_whitelist_v6.yaml` 활용)
2. **Numeric preservation** — 용량 단위 (돈/푼/냥/g/mg) 가 변하지 않았는지
3. **Hanja preservation** — 한자 본문이 한글로 풀어진 부분 없는지 (`[一-龥]` regex)
4. **Length sanity** — rewrite 답변이 원본의 1.0~3.0배 길이 안 (너무 짧거나 폭주 X)

검증 fail row 는 drop. 예상 retention rate: 90-93% (rewrite 10,000 → ~9,000-9,300, Plan B 의 retention 보수 92% 가정).

```bash
python scripts/validate_friendly_rewrite.py \
    --input data/sft/friendly_rewrite_v0.jsonl \
    --whitelist data/sft/entity_whitelist_v6.yaml \
    --gold data/sft/friendly_gold_v0.jsonl \
    --output data/sft/friendly_qa_v0.jsonl \
    --report data/sft/friendly_qa_v0.validation.json
```

---

## 5. ver8.2 mix corpus 빌드

### 5.1 빌드

```bash
python scripts/build_v8_2_corpus.py \
    --base data/sft/phaseB_qa_v8_1_corpus.jsonl \
    --friendly data/sft/friendly_qa_v0.jsonl \
    --output data/sft/phaseB_qa_v8_2_corpus.jsonl \
    --shuffle --seed 42
```

기대 산출:
```
phaseB_qa_v8_2_corpus.jsonl
  ├─ ver8.1 (formal)            34,039 rows  (77%)
  └─ friendly_qa_v0 (friendly) ~10,100 rows  (23%)
─────────────────────────────────────────
                               ~44,100 rows
```

### 5.2 `/sft-quality-fix` 정제

`/sft-quality-fix data/sft/phaseB_qa_v8_2_corpus.jsonl` 호출.

기대 결함:
- **near-duplicate** — friendly_rewrite 와 원본 ver8.1 답변이 question 동일 → near-dup hit. 처리: `_origin` 필드로 둘 다 보존 (rewrite 가 의도적 paraphrase)
- **format diversity** — 친절체 추가로 다양성 ↑ (개선 방향)
- **disclaimer 반복** — gold/rewrite 는 disclaimer 단조구 안 쓰니 ver8.1 만 hit. 영향 없음.

라운드 1 수렴 기대. 단 plan 단계에서 `_origin: llm_rewrite_v0` 행은 dedup 대상 제외 가드 명시 필요.

---

## 6. LoRA round 2 학습

### 6.1 학습 파라미터 (ver8.1 와 동일 + Plan B 조정)

| 항목 | ver8.1 | ver8.2 (Plan B) | 변경 사유 |
|---|---|---|---|
| base | gemma-3-12b-it | 동일 | — |
| rank r | 16 | 16 | — |
| lora_alpha | 32 | 32 | — |
| lora_dropout | 0.05 | 0.05 | — |
| target_modules | q/k/v/o + gate/up/down | 동일 | — |
| epochs | 3 | **2** | mix corpus 가 44K rows 라 학습량 충분 |
| lr | 2e-4 | **1.2e-4** | mix 23% 라 더 낮춰 distribution 안정 (Plan A 의 1.5e-4 보다 ↓) |
| batch_size | 16 (effective) | 16 | — |
| max_seq_len | 2048 | 2048 | — |
| warmup_ratio | 0.03 | **0.05** | mix 라 warmup 살짝 길게 — 친절체 분포 적응 |

### 6.2 mix 학습 안정성 보장

- **shuffle**: 친절체 row 가 한 epoch 안에 고르게 분산
- **mixed loss check**: friendly rows 의 loss 가 정상 하강 모니터링 (collapse 안 되게)
- **eval split**: 친절체에서 100 rows 별도 holdout → eval loss 추적

### 6.3 산출물

```
experiments/dongui_bogam/outputs_ver8_2_gemma_v1/
├── adapter/
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── ...
└── trainer_state.json
```

---

## 7. 배포

### 7.1 Merge + text-only 추출

```bash
# 1) merge
PYTHONHASHSEED=0 .venv/bin/python \
    experiments/dongui_bogam/scripts/build_merged_model_ver8_1.py \
    --adapter experiments/dongui_bogam/outputs_ver8_2_gemma_v1/adapter \
    --output  experiments/dongui_bogam/outputs_ver8_2_gemma_v1/merged \
    --device cuda

# 2) text-only 추출 (multimodal handler 우회)
.venv/bin/python experiments/dongui_bogam/scripts/extract_text_only_merged.py
# (스크립트 SRC/DST 인자 추가 — 기존 v1 hardcode 분리 필요)
```

### 7.2 vLLM 재배포

```bash
cd experiments/dongui_bogam
HANMED_MERGED_DIR=../outputs_ver8_2_gemma_v1/merged_text \
    docker compose -f docker/compose.ver8_1.yml up -d --force-recreate hanmed_vllm_ver8_1
```

→ RAG sidecar (`hanmed_rag`) 는 그대로 유지. vLLM 만 swap.

---

## 8. 평가

### 8.1 Before/After 비교 query set (7개)

`experiments/dongui_bogam/eval/friendly_tone_qaset.yaml` (신규):

| # | Query | 측정 항목 |
|---|---|---|
| 1 | 인삼(人蔘)의 성미와 귀경에 대해 알려줘 | 친절체 도입·풀이 유무 |
| 2 | 사물탕에 대해 알려줘 | 룰 7 + 풀이 친절체 |
| 3 | 유옹(乳癰)에 쓰는 단삼고(丹蔘膏)의 조성과 적응증 | 한자 정규화 회귀 0 |
| 4 | 두통에 동의보감에서 등장하는 처방 | 다중 처방 + 친절 안내 |
| 5 | 동의보감 5편의 큰 구성 | 편 개념 친절 설명 |
| 6 | 醫者雅言軒岐 ... 본문 의미 | 한문 해석 + 풀이 |
| 7 | 제가 심리적 불안 동반 소화기... 처방 부탁 | safety REFUSED 회귀 0 |

### 8.2 정량 metric

| 차원 | 측정 | 목표 |
|---|---|---|
| **친절체 풀이 비율** | 답변에 "풀이/쉽게 말하면/즉/자연스러운 한국어 부연" 등장 row% | ≥ 80% |
| **본문 인용 보존율** | retrieve된 발췌의 한자 본문이 답변에 그대로 포함된 row% | ≥ 95% |
| **Disclaimer 정형구** | "...로 한정해 읽어 주십시오" 박힌 row% | ≤ 30% (ver8.1 ~95%) |
| **Hanja 정규화 회귀** | Q1·D2·D4 의 boost rank 1.0 sim | 100% (회귀 0) |
| **Safety REFUSED 회귀** | Q7 같은 진료성 query mode | 100% REFUSED (회귀 0) |
| **Fact drift** | entity whitelist 통과율 | ≥ 99% |

### 8.3 정성 평가 (사용자)

`hanmed-bogam` REPL 으로 7-query 던지고 답변 톤 직접 비교 (before vs after). 발표/시연 데모 직접 활용.

---

## 9. 일정 / 리소스

### 9.1 Timeline (3-4일, Plan B)

| Day | Step | 시간 |
|---|---|---|
| **Day 1** | Gold seeds 100 rows 수작업 | 6-8 시간 |
| | Rewrite sample 추출 + script 작성 | 1 시간 |
| **Day 2** | LLM rewrite 실행 (GPU 0, 10K rows) | **5 시간** |
| | Fact drift 검증 + retention 확인 | 1-2 시간 |
| **Day 3** | ver8.2 mix corpus 빌드 + `/sft-quality-fix` | 1-2 시간 |
| | LoRA round 2 학습 (44K rows × 2 epoch) | **3-5 시간** |
| **Day 4** | Merge + text-only 추출 + 배포 | 1 시간 |
| | 7-query 평가 + report 작성 | 2-3 시간 |

### 9.2 자원

| 항목 | 사용량 |
|---|---|
| GPU 0 (48GB) | rewrite 5시간 + 학습 3-5시간 + merge 0.5시간 = **~9 시간 점유** |
| 디스크 | merged_text_v2 ~22 GB + intermediate corpus ~500 MB |
| API 비용 | **$0** (local 만) |
| 사람 시간 | gold 작성 6-8h + 평가 2-3h + 산발 3-4h = ~13 시간 |

---

## 10. 위험 요소

| 위험 | 가능성 | 대응 |
|---|---|---|
| LLM rewrite 의 fact drift | 중 | entity whitelist 검증 + length sanity (Step 4) |
| 친절체 mix 가 fact 정확도 dilute | 저 | mix 5.8% 보수적 비율, lr 낮춤 |
| `/sft-quality-fix` 의 near-dup 가 rewrite 행 drop | 중 | `_origin: llm_rewrite_v0` 행 dedup 제외 가드 |
| 학습 후 한자 정규화 회귀 | 저 | 평가 Q1·D2·D4 회귀 0 명시 — fail 시 mix 비율 조정 |
| 학습 후 safety REFUSED 회귀 | 저 | 평가 Q7 회귀 0 — fail 시 safety_refusal seeds 보강 라운드 |
| base Gemma3 가 friendly text 도 한문체로 출력 | 중 | gold few-shot 3개 + REWRITE_SYSTEM 명시 + temperature 0.3 |
| rewrite 답변이 너무 추상적 | 저 | gold few-shot 의 풀이 패턴 강제 ("쉽게 말하면", "즉") |

---

## 11. Scaling 시나리오 — Sample Mix vs Full Rewrite

§3 의 2,000 rows (5.8% mix) 는 보수적 sweet spot 채택값이지만, 라운드 사이즈를 확대 또는 축소할 수 있다. 본 절은 3-tier 시나리오 (A/B/C) 의 효과·자원·위험 예측을 정리하고, escalation 경로를 명시한다.

### 11.1 시나리오 비교

| 항목 | **Plan A**<br>Sample Mix (fallback) | **Plan B (채택)**<br>Heavy Mix | **Plan C**<br>Full Rewrite |
|---|---|---|---|
| friendly rows (rewrite + gold) | 2,100 | 10,100 | 34,100 |
| ver8.2 corpus 총 rows | ~36,100 | ~44,000 | **~34,100 (rewrite 만, ver8.1 미사용)** 또는 ~68,000 (concat) |
| friendly 비율 | **5.8%** | **23%** | **100% (rewrite-only)** 또는 50% (concat) |
| GPU rewrite 시간 (Gemma3-12B-IT) | ~1 시간 | ~5 시간 | ~**17 시간** |
| Retention rate 가정 (entity whitelist 통과) | 95% | 92% | 88% (drift 누적) |
| 비용 | $0 | $0 | $0 |
| 학습 epoch (제안) | 2 | 2 | **1.5** (over-fit 방지) |
| 학습 시간 (round 2) | ~2-4 시간 | ~3-5 시간 | ~5-8 시간 |
| 사람 시간 (gold 작성) | 6-8 시간 | 6-8 시간 (동일) | 6-8 시간 (동일) |
| **총 작업기간** | **2-3일** | **3-4일** | **5-7일** |

### 11.2 효과 예측

| 평가 metric (§8 정의) | Plan A | Plan B | Plan C |
|---|---|---|---|
| 친절체 풀이 비율 | 50-70% | 75-90% | **95-100%** |
| 본문 인용 보존율 | 95-98% | 90-95% | 80-90% (드리프트 ↑) |
| Disclaimer 정형구 박힘 | 60-70% (ver8.1 분포 우세) | 30-50% | **5-15%** |
| 한자 정규화 회귀 | 0% | 0-2% | 5-10% (rewrite drift) |
| Safety REFUSED 회귀 | 0% | 0-3% | 3-8% (safety_refusal corpus 비율 dilute) |
| Fact drift (entity whitelist fail) | ≤1% | 2-5% | 5-12% |
| **사용자 체감 톤 변화** | △ 약한 개선 | 🟢 명확한 친절체 | 🟢🟢 매우 친절 |

### 11.3 위험 비교

| 위험 | Plan A | Plan B | Plan C |
|---|---|---|---|
| Fact drift 누적 | 낮음 | 중간 | **높음** — 17시간 rewrite, retention 88% |
| ver8.1 의 안전 정책 (REFUSED, REFUSAL_TEMPLATE) dilute | 거의 없음 | 가능 | **명백** — safety_refusal 비율 ↓ |
| 한자 본문 보존 실패 | 거의 없음 | 가능 | 가능 (rewrite 가 "풀어쓰기" 강제로 한자 → 한글) |
| catastrophic forgetting (formal 톤 사라짐) | 없음 | 부분 | **명백** — formal 톤 학습 분포 0 |
| 학습 over-fit (mix 비율 30% 초과) | 없음 | 경계선 | 강함 (단 rewrite-only 면 ver8.1 base 0) |
| 라운드 시간 초과 | 없음 | 낮음 | 높음 (5-7일) |

### 11.4 시나리오별 권장 조건

**Plan A 채택 조건** (현재 기본):
- ver8.1 의 사실 정확도·safety 정책을 보존하는 것이 최우선
- 친절체 효과는 "약간 부드러워지는 것" 정도면 충분
- 작업기간 2-3일 안에 완료 필요
- 첫 친절체 라운드 (실험적)

**Plan B 로 escalation 조건** (Plan A 평가 후 부족 시):
- §8 의 "친절체 풀이 비율" 이 70% 미만으로 측정됨
- 한자 정규화·safety REFUSED 회귀 0건 유지
- 2-3일 추가 시간 확보 가능
- B 의 비율 23% 가 §3.4 "30% mix 초과 시 dilute" 가드 안쪽

**Plan C 채택 조건** (특수 상황):
- 데모/발표용 100% 친절체 필요 (학술 정확도 < 톤 우선)
- ver8.1 의 safety 정책은 별도 layer (RAG sidecar safety) 에서 충분히 보장된다고 판단
- 한자 보존 실패는 RAG 의 retrieval 발췌 인용으로 보완 가능
- 5-7일 작업기간 + GPU 17시간 점유 허용

### 11.5 Escalation 경로

```
Plan A (5.8% mix, 2K rows)
   │
   │ §8 평가 후 "친절체 비율 < 70%" 또는 "사용자 체감 부족"
   ↓
Plan B (23% mix, 10K rows)
   │
   │ §8 평가 후 "Plan B 도 부족" — 일반 사용자 대상 demo 필요
   ↓
Plan C (100% rewrite, 34K rows)
   │
   │ Plan C 결과의 fact drift 측정. 회귀 0 보증 안 되면 rollback
   ↓
[rollback to Plan B] 또는 [Plan B + 추가 fact 검증 라운드]
```

각 escalation 단계에서 **이전 라운드의 산출물 (gold, rewrite, mix corpus, 학습 adapter) 보존**. 다음 단계는 추가 데이터 생성만, 처음부터 새로 안 짠다.

### 11.6 비용·효율 요약

| 시나리오 | "친절체 효과 1단위" 당 자원 |
|---|---|
| Plan A | 효과 1 → GPU 1h + 사람 8h |
| Plan B | 효과 2 → GPU 5h + 사람 8h (5h/1.7배) |
| Plan C | 효과 3 → GPU 17h + 사람 8h (3.4배) |

→ **marginal return diminishing** — Plan B 가 cost-effectiveness sweet spot. Plan C 는 "극대화" 가 명시적으로 필요할 때만 채택.

### 11.7 본 라운드 결정

본 기획서의 default 는 **Plan B** (10,100 rows, 23% mix — §3 참조). Plan B 의 §8 평가 결과에 따라:
- §8 metric 모두 충족 + 정성 평가 OK → 라운드 수렴, ver8.3 (GraphRAG) 진행
- §8 친절체 비율 < 80% → ver8.2.1 escalation = Plan C 부분 적용 (예: 친절체 비율 30%)
- §8 fact drift > 5% 또는 safety 회귀 > 0 → rollback to Plan A (5.8%) 재학습

Plan A (5.8%, fallback) / Plan C (100%, escalation) 는 §11 reference 로 보유.

---

## 12. 후속 라운드

본 라운드 (ver8.2) 수렴 후:

| 라운드 | 주제 | 우선순위 |
|---|---|---|
| **ver8.3** | GraphRAG path-only KG (변형처방·다중 hop) | 높음 |
| ver8.4 | 약재 사전 NER (KIOM 600종) → (처방)→(약재) edge | 중 |
| ver8.5 | 추가 친절체 mix 비율 조정 (5.8% → 10% 효과 검증) | 중 |
| ver9.0 | base 모델 swap (Gemma3-27B 등) | 저 |

---

## 13. 산출물 목록

본 라운드 종료 시 다음 파일 존재:

```
data/sft/
├── friendly_gold_v0.jsonl                        # 100 rows 수작업
├── friendly_rewrite_v0.jsonl                     # 2,000 rows LLM rewrite raw
├── friendly_qa_v0.jsonl                          # 검증 통과분 ~2,100
├── friendly_qa_v0.validation.json                # entity whitelist · length 통과 통계
├── v8_1_rewrite_sample.jsonl                     # 추출 sample
└── phaseB_qa_v8_2_corpus.jsonl                   # mix corpus ~36K

scripts/
├── build_friendly_rewrite_sample.py              # ver8.1 stratified 추출
├── rewrite_to_friendly.py                        # base Gemma3 rewrite
├── validate_friendly_rewrite.py                  # fact drift 검증
└── build_v8_2_corpus.py                          # mix + shuffle

experiments/dongui_bogam/
├── outputs_ver8_2_gemma_v1/
│   ├── adapter/
│   ├── merged/
│   └── merged_text/
└── eval/
    └── friendly_tone_qaset.yaml                  # 7-query 평가셋

docs/ver8.2/
├── 00_friendly_tone_plan.md                      # 본 문서
├── 01_gold_seeds_design.md                       # 카테고리 70+30 자세히 (작성 예정)
├── 02_rewrite_pipeline_log.md                    # rewrite 실행 로그 (작성 예정)
├── 03_validation_report.md                       # fact drift 보고 (작성 예정)
├── 04_round_2_training_log.md                    # LoRA 학습 로그 (작성 예정)
└── 05_evaluation_report.md                       # before/after 7-query (작성 예정)
```

---

## 14. 즉시 시작 — Step 1 (Gold seeds 카테고리 정의)

**Day 1 오전**: `01_gold_seeds_design.md` 작성. 카테고리 70 + 30 의 sub-카테고리 정의 + 각 카테고리 1 예시.

이후 사용자 + AI 협업으로 100 rows 수작업 (1 row 당 5-8분 → 8-13시간).

---

**Decision point**: 본 기획서가 사용자 의도와 부합하면 `01_gold_seeds_design.md` 작성 시작. 의문 있으면 §2 (방법론), §3 (sample size), §6 (학습 파라미터) 중 어디 조정할지 우선 결정.
