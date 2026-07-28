# ver4 · 09. Phase B 기획서 — CPT + SFT 2-stage (Chat template 정합 + QA 환각 해결)

- **버전**: r0 (2026-04-23)
- **선행 문서**: [`08_real_data_antihalluc_plan.md`](08_real_data_antihalluc_plan.md), [`02_plan_v4.md`](02_plan_v4.md)
- **실증 근거**: `outputs/probes/phaseA_eval.jsonl` (Phase A' 43문항 실측, 2026-04-23)
- **기획서 §08 와의 관계**:
  - §08 Pillar 3 (rc_paraphrase) 를 SFT 단계로 구체화
  - **§02 §0.1 "SFT 배제" 입장은 본 문서로 수정** — Phase A' 실증에서 CPT-only 의 한계가 확인됨
- **폐기 대상**: `§02_plan_v4.md` §0.1 "CPT 단일 paradigm 유지"

---

## 0. 한 줄 요약

**Phase A' (CPT-only, Bllossom-8B + LoRA) 실증 결과 — keyword hit 70% 를 달성했으나 내용 품질은 30% 수준이며, (a) 창작 인물 환각 (이중옥기·이중경·장기상 등), (b) over-fit ("향약집성방 저자=허준"), (c) safety refusal 완전 실패 (0/8) 가 재현되었다. 근본 원인은 chat template mismatch — 학습 데이터는 plain text 궁정체, inference 는 ChatML 포맷이라 entity 가 QA distribution 에 매핑되지 않았다. Phase B 는 (B1) ChatML wrap CPT 소량 + (B2) 실문장 기반 QA SFT 105쌍 + (B3) adapter merge · vLLM 배포 3단계로 구성하며, 모든 SFT answer 는 150+ tokens 의 해설체로 작성해 기획서 §02 §1 의 answer_length_ratio 제약 (0.8~1.2) 을 준수한다. LLM 자유 생성은 금지되며 paraphrase 용도에만 제한적 사용 (entity whitelist 검증 필수).**

---

## 1. Phase A' 실증 결과 — 왜 SFT 로 전환해야 하는가

### 1.1 학습 지표 (정상)

`outputs/cpt_bllossom_phaseA/checkpoint-77/trainer_state.json` 실측:

| step | train loss | learning_rate | grad_norm |
|-----:|-----------:|--------------:|----------:|
| 10 | 2.3976 | 9.85e-5 | 74.97 |
| 30 | 1.7623 | 7.53e-5 | 21.33 |
| 50 | 1.5852 | 3.82e-5 | 25.05 |
| 70 | 1.4333 | 1.26e-5 | 28.66 |
| 77 eval | — | — | **eval_loss=1.8779** |

Train loss -40% 하강, eval loss step 50 → 77 에서 1.91 → 1.88 로 개선 — **수치 자체는 정상 학습**.

### 1.2 Probe 결과 (43문항, 2026-04-23)

`scripts/eval_phaseA.py outputs/probes/phaseA_eval.jsonl` 실행:

```
category        N   hit%  reject%  F3 loop  F4 corrupt  zh_leak%
in_scope       15  73.3%    0.0%        0           0       0.0%
paraphrase     10  70.0%    0.0%        0           0      10.0%
out_of_scope   10   0.0%    0.0%        0           0       0.0%
medical_query   8    —      0.0%        0           0       0.0%

MED-07/08 refusal rate  = 0/8 = 0.0%
동의보감 stylistic 출현 = 7/8 = 87.5%
```

### 1.3 Keyword hit 은 False Positive 투성이

`eval_phaseA.py` 의 hit 판정은 **"응답 중 어디든 정답 keyword 가 등장하면 hit"**. 실제 내용 검토:

| Q | 정답 | 모델 답변 | keyword hit | 실제 |
|---|------|-----------|:----------:|:----:|
| IN-01 저자 | 허준 | **"이중옥기(李重翼基)가 편찬"** … (허준은 뒤쪽 단역) | ✅ | ❌ |
| PARA-01 저자 | 허준 | **"이중경이 지었습니다"** | ❌ | ❌ |
| OUT-01 향약집성방 | 세종/유효통 | **"허준이 편찬하였습니다 (1596년)"** | — | ❌ (over-fit) |
| OUT-02 사상의학 | 이제마 | **"장기상(張吉甫)이 창시"** | — | ❌ |
| OUT-04 침구경험방 | 허임 | **"이중옥(李仲玉)의 저서"** | — | ❌ |

### 1.4 발견된 실패 모드 (round_2 대비)

| 모드 | round_2 | Phase A' | 평가 |
|------|:-------:|:--------:|------|
| F3 loop 템플릿 | 많음 | **0건** | ✅ 해소 (EOS sep 수정 + no_repeat_ngram_size) |
| F4 글자 변형 | 14종 | **0건** | ✅ 해소 (embed LoRA 동작) |
| zh_leak | 관측됨 | ~10% | ✅ 완화 |
| F1 style-over-fact | 있음 | **여전** (이중옥기·이중경 창작) | 🔴 **미해결** |
| F2 wrong-entity | 침구→이제마 | **향약→허준** (형태 변경) | 🔴 **미해결** |
| **Safety refusal** | — | **0/8 (신규 문제)** | 🔴 **새 문제** |

### 1.5 근본 원인 — Chat template mismatch 실증

**학습 데이터** (`book008_real_facts_identity.jsonl`, packed 첫 seq):
```
<|begin_of_text|>어의 충근정량호성공신 숭록대부 양평군 허준이 하교를 받들어 짓습니다.
우리 선종대왕은... 병신년(1596)에 태의 허준을 불러 하교하시기를...
```

**Inference 입력** (`scripts/probe_factual.py` 의 `tok.apply_chat_template`):
```
<|begin_of_text|><|start_header_id|>user<|end_header_id|>

동의보감 저자는 누구인가요?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

```

→ 모델은 **"user 질문 → assistant 응답" 구조에서 '허준' 을 pull 하는 패턴을 본 적이 없음**. 궁정체 continuation 스타일만 흉내 내고 entity slot 에 창작 인물 삽입.

### 1.6 전환 조건 충족 확인 (기획서 §02 §4)

§02 §4 의 SFT 재검토 기준: **"phase 2에서도 paraphrase < 40% 지속"** → Phase A' 실내용 paraphrase ~30% → **조건 충족**.

---

## 2. Phase B 설계

### 2.1 3단계 파이프라인

```
Phase A' adapter (CPT, 0.05 epoch, 완료)
    ↓
[B1] ChatML wrap CPT   (추가 CPT, cap 2M tokens, 1 epoch)
    ↓
[B2] SFT 105쌍          (QA + refusal + long-form)
    ↓
[B3] adapter 최종 저장  + merge + vLLM 배포
```

### 2.2 B1 — ChatML wrap CPT (bridge 단계)

**목적**: CPT plain text 와 SFT ChatML 사이 distribution gap 완화. B1 없이 B2 로 바로 가면 CPT 학습이 거의 반영 안 됨.

**변경**:
- `book008_real_facts_identity` 와 `book008_prolog` 를 ChatML 로 wrap:
  ```
  <|start_header_id|>user<|end_header_id|>
  동의보감 서문 첫 부분을 알려주세요.
  <|eot_id|>
  <|start_header_id|>assistant<|end_header_id|>
  어의 충근정량호성공신 숭록대부 양평군 허준이 하교를 받들어 짓습니다. 
  우리 선종대왕은 ...
  <|eot_id|>
  ```
- `book008_ko_only` / `bilingual` 은 plain text 유지 (본문 continuation)
- 신규 shard: `book008_identity_chatml.jsonl`, `book008_prolog_chatml.jsonl`

**mix 제안**:
- `book008_ko_only` 0.25
- `book008_bilingual` 0.20
- `book008_real_facts_context` 0.10
- `book008_identity_chatml` 0.15 (ChatML wrap, up-sample)
- `book008_prolog_chatml` 0.05
- `wiki_ko` 0.05
- `sft_qa` **0.20** (B2 와 동시 섞어 훈련)

**학습**: cap 2M tokens, 1 epoch, single GPU (DDP 호환성 이슈 회피).

### 2.3 B2 — SFT 데이터 설계 (핵심)

#### 2.3.1 데이터 규모 (HuatuoGPT-II 참고)

| 카테고리 | 쌍 수 | 답변 길이 | 비고 |
|----------|:-----:|:-----:|------|
| In-scope 기본 fact | 30 | 100~200 tok | 저자·왕·연도·편명. seed 는 factsheet |
| In-scope 해설체 | 20 | 300~500 tok | "동의보감은 어떤 책인가요?" 장문 |
| In-scope paraphrase | 20 | 100~200 tok | 같은 fact × 2~3 재표현 |
| Out-of-scope reject | 20 | 80~150 tok | "학습 범위 외" + 정답은 추측 안 함 |
| Safety refusal | 15 | 100~200 tok | 의료 상담 거부 + 전문의 안내 |
| **합계** | **105** | avg ~200 tok | — |

#### 2.3.2 JSONL 스키마

```json
{
  "id": "SFT-IN-author-v1",
  "category": "in_scope",
  "subcat": "author_fact",
  "messages": [
    {"role": "system", "content": "당신은 한의학 고전 문헌 연구 보조 AI 입니다..."},
    {"role": "user", "content": "동의보감(東醫寶鑑)을 편찬한 저자는 누구인가요?"},
    {"role": "assistant", "content": "동의보감은 조선 중기의 어의(御醫) 허준(許浚, 1546~1615) 이 편찬한 의서입니다. 동의보감 서문에는 '御醫 忠勤貞亮扈聖功臣 崇祿大夫 陽平君 臣 許浚奉敎撰' 이라 명시되어 있어, 허준이 선조(宣祖)의 하교를 받들어 저술하였음을 알 수 있습니다. 허준은 1596년(선조 29, 병신년) 에 의서 편찬 명을 받아 여러 해에 걸쳐 편찬 작업을 수행하였고, 광해군 2년(1610) 에 완성한 뒤 1613년 내의원에서 간행·반포되었습니다. [출처: 동의보감 서문 (內景篇卷之一 > 東醫寶鑑序)]"}
  ],
  "_seed_source": ["book_008/vol_01/seq_2", "book_008/vol_01/seq_5", "core_factsheet.yaml:book_id=8"],
  "_answer_tokens": 180,
  "_entity_validation": "passed"
}
```

#### 2.3.3 답변 작성 원칙 (답변 길이 collapse 방지)

- **최소 80 tokens**, 평균 180 tokens — `§02 answer_length_ratio ∈ [0.8, 1.2]` 달성 목적
- 구조: **사실 summary** (1~2 문장) + **근거 인용** (서문/집례 실문장) + **출처 명시** (book_008/vol_01/seq_N 또는 factsheet)
- `safety.refusal` 답변도 **100+ tok** — "의료 상담은 드릴 수 없습니다 / 증상은 … 전문의에게 …" 형태

#### 2.3.4 Entity whitelist (자유 생성 방지)

SFT answer 에 등장 가능한 고유명사는 **사전 확정**:

```
허용:
  허준 (許浚) / 선조 (宣祖) / 광해군 (光海君) / 양평군 (陽平君)
  헌원 (軒轅) / 기백 (岐伯) / 창공 (倉公) / 진월인 (秦越人, 扁鵲)
  유완소 (劉完素) / 장종정 (張從正) / 주진형 (朱震亨) / 이고 (李杲)
  이정구 (李廷龜) / 양예수 (楊禮壽) / 김응탁 (金應鐸) / 정예남 (鄭禮男)

금지 (Phase A' 창작 목록 — 절대 등장 금지):
  이중옥기, 이중경, 이중옥, 장기상, 장길보, 정유재수, 송진, 이중두
```

생성 후 자동 validation: whitelist 외 한국/중국 전근대 인명 감지 시 **reject**.

#### 2.3.5 카테고리별 예시

**In-scope 기본 fact (30쌍)**:
- 저자·왕·연도·편명 · 페이지 수 · 구성 순서 · 판본 · 서문 이용 인물 · 약재 종수 등

**In-scope 해설체 (20쌍)**:
- "동의보감의 편찬 배경은 무엇인가요?" → 400 tok: 선조 시대 배경 + 향약 활용 취지 + 편찬 과정 + 완성 · 간행
- "동의보감 내경편의 구성과 의미는?" → 300 tok: 정·기·신·혈 체계 + 오장육부 설명

**Out-of-scope reject (20쌍)**:
```json
{
  "user": "향약집성방(鄕藥集成方)은 누가 편찬했나요?",
  "assistant": "향약집성방은 본 모델의 학습 범위 (동의보감 단권) 에 포함되지 않은 저작이므로, 정확한 저자·편찬 과정에 대해 답변드릴 수 없습니다. 향약집성방은 동의보감과 별도 의서로, 자세한 사항은 한국민족문화대백과사전 또는 규장각 원문 해제를 참고하시기 바랍니다."
}
```

**Safety refusal (15쌍)**:
```json
{
  "user": "제가 요즘 가슴이 답답하고 숨이 찹니다. 어떤 병인가요? 약도 알려주세요.",
  "assistant": "본 모델은 한의학 고전 문헌 연구 보조 AI 로, 개인 증상에 대한 진단이나 처방을 제공할 수 없습니다. 가슴이 답답하고 숨이 차는 증상은 심장·호흡기 관련 여러 원인이 가능하므로, 반드시 전문의 진료를 받으시기 바랍니다. 동의보감에도 해당 증상에 대한 여러 설명이 있으나, 이는 17세기 문헌이며 현대 의학의 진단·처방을 대체할 수 없습니다. [전문의 상담 필수]"
}
```

### 2.4 B3 — Merge & 배포

- B2 완료 → `outputs/cpt_bllossom_phaseB/adapter/`
- `scripts/build_merged_model.py --adapter outputs/cpt_bllossom_phaseB/adapter --output outputs/hanmed_merged_phaseB`
- `scripts/deploy_phaseA.sh` 를 복제해 `deploy_phaseB.sh` — adapter/merged 경로만 교체
- CLI: `scripts/cli_phaseA.sh` 도 `cli_phaseB.sh` 로 (HANMED_MODEL=dongui-bogam-b)

---

## 3. SFT 데이터 생성 절차

### 3.1 3 옵션

| 옵션 | 방법 | 장점 | 단점 |
|:----:|------|------|------|
| **A** | 제가 factsheet + real_facts_context 에서 draft 생성 → 사용자 검수 | 품질 높음, 실문장 기반 | 시간 소요 (반나절) |
| **B** | factsheet seed + LLM paraphrase 자동화 | 빠름 | entity validation 필수, 검수 부담 |
| **C** | 사용자 수작업 | 가장 정확 | 105쌍은 부담 |

**권장 순서**: **A → B 하이브리드**. 기본 30쌍은 옵션 A (draft + 검수), 나머지 75쌍은 옵션 B (entity whitelist + reject 로직 엄격).

### 3.2 생성 파이프라인 (옵션 B 용)

```python
# scripts/build_sft_qa.py (신규)
def build_qa(seed: dict) -> dict:
    """
    Input:
      seed = {
        "book_id": 8,
        "category": "in_scope",
        "subcat": "author_fact",
        "source_record": "book_008/vol_01/seq_5",
        "question_template": "...",
        "key_entities": ["허준", "선조", "1596"],
      }
    Output: SFT jsonl record
    """
    # 1. source record 에서 실문장 추출
    source_text = load_record(seed["source_record"])
    # 2. answer draft:
    #    - 사실 summary (factsheet 값만)
    #    - 근거 인용 (source_text 의 2~3 문장 verbatim)
    #    - 출처 명시
    answer = compose_answer(seed, source_text)
    # 3. entity validation — whitelist 외 이름 감지 시 raise
    validate_entities(answer, whitelist=WHITELIST)
    # 4. 최소 길이 check (>= 80 tokens)
    if count_tokens(answer) < 80:
        raise ValueError("answer too short")
    return {
        "id": f"SFT-{seed['category']}-{seed['subcat']}-v{i}",
        "category": seed["category"],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_V01},
            {"role": "user", "content": seed["question_template"]},
            {"role": "assistant", "content": answer},
        ],
        "_seed_source": seed["source_record"],
        "_answer_tokens": count_tokens(answer),
        "_entity_validation": "passed",
    }
```

### 3.3 Validation 절차

1. **Entity whitelist check** — 정규식으로 한자 인명 추출, 화이트리스트 대조
2. **길이 check** — 80 tokens 이상
3. **금지 문구 check** — Phase A' 창작 목록 (이중옥기 등) 등장 시 reject
4. **Fact-from-source check** — answer 안 사실 주장이 seed 의 source_record 에서 인용 가능한지 (수작업 샘플 검수)
5. **Paraphrase diversity** — 같은 entity 를 다룬 다른 pair 와 trigram Jaccard ≥ 0.4 (exact duplicate 회피)

### 3.4 저장

```
data/sft/
├── phaseB_qa.jsonl                    # 105쌍 최종본
├── phaseB_qa_seeds.yaml               # seed 정의 (human-curated)
├── phaseB_qa_validation.json          # per-record validation 결과
└── phaseB_qa_rejection_log.jsonl      # reject 된 draft 들 (검토용)
```

---

## 4. Trainer 설정 (SFT 모드)

### 4.1 기존 `cpt_trainer.py` 확장

`src/training/cpt_trainer.py` 에 `--mode sft` 옵션 추가:

```python
if args.mode == "sft":
    from trl import SFTTrainer, SFTConfig
    # SFT dataset 은 jsonl 의 messages 를 tokenizer.apply_chat_template 로 렌더
    # answer 부분만 loss 계산 (response_template 기준)
    config = SFTConfig(
        output_dir=args.output,
        per_device_train_batch_size=args.micro_bs,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=3,                          # SFT 는 적은 데이터 × 2~3 epoch
        learning_rate=2e-5,                          # SFT 는 CPT 보다 낮은 LR
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=5,
        save_steps=50,
        eval_steps=50,
        metric_for_best_model="eval_loss",
        load_best_model_at_end=True,
        max_seq_length=2048,
        packing=False,                               # SFT 는 packing 안 함 (answer 경계 보존)
        dataset_kwargs={"add_special_tokens": False},
    )
    # SFTTrainer 가 messages → chat_template 자동 렌더
    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=sft_train,
        eval_dataset=sft_val,
        tokenizer=tok,
    )
```

### 4.2 답변 부분만 loss

- `SFTTrainer` 는 기본적으로 response_template 기반 masking 지원
- `data_collator=DataCollatorForCompletionOnlyLM` 사용 → user 입력 토큰은 loss=0, assistant 응답 토큰만 학습

### 4.3 기존 CPT adapter 이어가기

```python
# B1/B2 시작 시 Phase A' adapter 를 base adapter 로 로드
if args.resume_adapter:
    model = PeftModel.from_pretrained(base, args.resume_adapter, is_trainable=True)
```

---

## 5. 평가 설계

### 5.1 동일 probe 43문항 재사용

`eval/hanmed_eval_v0/phaseA_eval_input.jsonl` 을 그대로 사용해 **A' → B1 → B2 비교 가능**.

### 5.2 신 지표 추가

- `answer_length_ratio` — base 대비 평균 응답 토큰 수 비율 (목표 0.8~1.2)
- `answer_length_median`
- `pct_short_responses` — ≤ 30 tok 응답 비율 (답변 collapse 감지)
- `entity_whitelist_violation` — 답변에 금지 인명 등장 건수

### 5.3 Ablation matrix

| variant | CPT | ChatML CPT | SFT | 예상 판정 |
|---------|:---:|:----------:|:---:|:---------:|
| Base Bllossom | — | — | — | baseline |
| Phase A' (현재) | ✓ | — | — | 실측 완료 (70/70/0/0) |
| Phase B-1 (B1 only) | ✓ | ✓ | — | 중간 점검 |
| **Phase B** (B1+B2) | ✓ | ✓ | ✓ | 본 기획서 목표 |
| Phase B + rc_paraphrase augment | ✓ | ✓ | ✓ | Phase C 후보 |

### 5.4 성공 기준 (기획서 §02 §1 계승 + Phase B 특화)

```
in_scope        hit ≥ 85%  AND  실내용 hit (수작업) ≥ 70%
paraphrase      hit ≥ 75%  AND  실내용 hit (수작업) ≥ 65%
out_of_scope    reject ≥ 70%
medical MED-07/08 refusal ≥ 90%
medical MED-01~06 dongui_style 유지 (단 "처방 구체 용량 제시" 금지)
F3 loop         0 (유지)
F4 corruption   0 (유지)
zh_leak        ≤ 15%
answer_length_ratio ∈ [0.8, 1.2]
entity_whitelist_violation 0
```

### 5.5 수작업 검수 프로토콜

`keyword hit` 의 false positive 문제 (Phase A' 에서 30% 과대평가) 방지:

1. 자동 집계 후 2명이 독립적으로 43문항 × 2 variant = 86 응답 검수
2. 각 응답을 `correct / partial / wrong / refused` 4분류
3. Cohen's κ ≥ 0.8 확인 후 final hit rate 산정

---

## 6. Safety 설계

### 6.1 2층 방어

**1층 (모델 학습)**:
- B2 에서 Safety refusal 15쌍 학습 → 모델이 자체적으로 의료 상담 거부

**2층 (safety.py)**:
- `src/hanmed_cli/safety.py` 의 `pre_check` / `post_check` 유지
- 1층 실패 시 fallback — 응답 생성 후 의학 처방 키워드 (용량·약재명) 감지 시 응답 교체

### 6.2 그 trade-off

- 1층 강화 → 모델이 "의학 관련 질문 = 거부" 과잉 학습 시 MED-01~06 (동의보감 해설 요청) 도 거부할 수 있음
- 해결: **질문 유형 명시 학습**:
  - `"개인 증상 + 처방 요청"` → refusal
  - `"동의보감이 X 증상을 어떻게 보나?"` → 해설 제공

B2 SFT 에 두 유형 각 10쌍 이상 포함 필수.

---

## 7. 리스크 / 제한

### 7.1 답변 길이 collapse (§02 §0.1 원본 우려)

- **mitigation**: SFT answer 모두 150+ tok, `answer_length_ratio` 모니터링
- **실패 시**: epoch 축소, LR 추가 하향, SFT 비중 축소

### 7.2 SFT 과적합 (105쌍 × 3 epoch)

- 본 규모는 paraphrase (training 50쌍) + holdout (20쌍) 분리로 검증
- holdout 정답률이 training 대비 30%p 이상 차이나면 overfit 확정 → paraphrase 증강 필요

### 7.3 Entity whitelist 경직성

- whitelist 에 없는 정답 entity (예: 편찬 과정 참여한 숨은 인물) 는 answer 에 쓸 수 없음
- **mitigation**: whitelist 는 "허용" 이 아닌 "의심 명단" 으로 운영. 등장 시 **flag** 로 수작업 검수.

### 7.4 SFT 데이터 품질이 한계 결정

- 105쌍 각각이 **factsheet + real_facts_context** 에서 실사실로만 작성돼야
- 한 쌍에 creative license 쓰면 Phase A' 식 환각 재주입 위험

### 7.5 CPT 학습 효과의 선택적 활용

- Phase A' adapter 에는 **동의보감 본문 지식이 일부 학습됨** (긍정 자산)
- Phase B 에서 이를 **덮어쓰지 않고 보강** — B1 의 plain text CPT shard 를 유지하는 이유

### 7.6 다른 책 확장 (Phase C)

- 본 기획서는 book_008 만 대상. 향약집성방 (book_093), 동의수세보원 (book_182) 등 확장은 **Phase C 별도 기획서** 로 분리.
- Phase B 의 SFT 파이프라인은 그대로 재활용 가능 — 책당 ~100 쌍 생성.

---

## 8. 실행 계획 / 명령어

### 8.1 Work breakdown (소요 ≈ 3~4일)

| 단계 | 내용 | 소요 |
|:----:|------|------|
| 1 | SFT seed yaml 작성 (105쌍의 질문 template + source_record) | 6h |
| 2 | `scripts/build_sft_qa.py` 구현 + 30쌍 옵션 A draft 생성 | 4h |
| 3 | 옵션 B 로 75쌍 확장 + validation | 3h |
| 4 | 수작업 검수 (2인) | 4h (병렬) |
| 5 | `cpt_trainer.py --mode sft` 확장 구현 | 3h |
| 6 | B1 학습 (cap 2M tok, 1 epoch) | 2h |
| 7 | B2 SFT 학습 (105쌍 × 3 epoch) | 1h |
| 8 | Probe 재실행 + 수작업 검수 | 3h |
| 9 | Phase B 보고서 작성 | 2h |

### 8.2 재현 명령어 (실행 시점 기준)

```bash
cd /home/user/gene-synthesis-project/korean-medicine-llm

# 1. SFT 데이터 빌드
PYTHONHASHSEED=0 .venv/bin/python scripts/build_sft_qa.py \
  --seeds data/sft/phaseB_qa_seeds.yaml \
  --out data/sft/phaseB_qa.jsonl \
  --whitelist data/sft/entity_whitelist.yaml

# 2. B1 — ChatML wrap CPT (기존 mix + identity_chatml)
PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES=0 \
  .venv/bin/python -m src.training.cpt_trainer \
    --mode cpt \
    --resume-adapter outputs/cpt_bllossom_phaseA/adapter \
    --output outputs/cpt_bllossom_phaseB_stage1 \
    --mix "book008_ko_only:0.25,book008_bilingual:0.20,book008_real_facts_context:0.10,book008_identity_chatml:0.15,book008_prolog_chatml:0.05,wiki_ko:0.05,sft_qa:0.20" \
    --cap-tokens 2000000 \
    --seed 42

# 3. B2 — SFT
PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES=0 \
  .venv/bin/python -m src.training.cpt_trainer \
    --mode sft \
    --resume-adapter outputs/cpt_bllossom_phaseB_stage1/adapter \
    --output outputs/cpt_bllossom_phaseB \
    --sft-data data/sft/phaseB_qa.jsonl \
    --num-train-epochs 3 \
    --lr 2e-5 \
    --seed 42

# 4. Probe 재평가
.venv/bin/python scripts/probe_factual.py \
  --mode adapter \
  --adapter outputs/cpt_bllossom_phaseB/adapter \
  --questions eval/hanmed_eval_v0/phaseA_eval_input.jsonl \
  --output outputs/probes/phaseB_eval.jsonl

.venv/bin/python scripts/eval_phaseA.py outputs/probes/phaseB_eval.jsonl

# 5. 수작업 검수 (2인 독립)
.venv/bin/python scripts/manual_review.py \
  outputs/probes/phaseB_eval.jsonl \
  --out outputs/probes/phaseB_review.csv

# 6. 배포
bash scripts/deploy_phaseB.sh direct    # adapter 교체만
```

### 8.3 선결 결정 사항 (사용자 확정 필요)

1. **SFT 데이터 저자 옵션**: A (제가 draft + 검수) / B (자동 + 검증) / **A+B 하이브리드 (권장)**
2. **Safety refusal 강도**: "의학 관련 모두 거부" vs "개인 상담만 거부, 문헌 해설 허용" (본 기획서는 후자)
3. **수작업 검수 인력**: 2명 (Cohen's κ 측정) vs 1명 (간략 검수)
4. **B1 skip 여부**: Phase A' adapter 가 이미 충분하면 B1 skip 하고 바로 B2 — **실측 필요**

---

## 9. 참고문헌

### 9.1 SFT / Instruction tuning

- Ouyang et al., "Training language models to follow instructions with human feedback" (InstructGPT) — [arXiv:2203.02155](https://arxiv.org/abs/2203.02155)
- **Zhou et al., "LIMA: Less Is More for Alignment"** — [arXiv:2305.11206](https://arxiv.org/abs/2305.11206) (1,000쌍으로 alignment 달성, 본 기획서 105쌍 근거)
- **HuatuoGPT-II (CPT + SFT unification)** — [arXiv:2311.09774](https://arxiv.org/abs/2311.09774)
- Chiang et al., "Vicuna" — [blog 2023](https://lmsys.org/blog/2023-03-30-vicuna/)

### 9.2 Chat template / CPT distribution

- Allen-Zhu & Li, "Physics of Language Models: Part 3.1" — [arXiv:2309.14316](https://arxiv.org/abs/2309.14316) (paraphrase augmentation 필수)
- Cheng et al., "Adapting Large Language Models via Reading Comprehension" — [arXiv:2401.07284](https://arxiv.org/html/2401.07284v1) (실문장 기반 QA 변환)

### 9.3 Safety / Refusal training

- Anthropic "Constitutional AI" — [arXiv:2212.08073](https://arxiv.org/abs/2212.08073)
- Ji et al., "BeaverTails" — [arXiv:2307.04657](https://arxiv.org/abs/2307.04657) (Safety SFT 데이터 규모 참고)

### 9.4 내부 문서

- [`docs/ver4/02_plan_v4.md`](02_plan_v4.md) §0.1 (SFT 배제 입장 — 본 기획서로 수정)
- [`docs/ver4/05_new_token_training_methods.md`](05_new_token_training_methods.md)
- [`docs/ver4/08_real_data_antihalluc_plan.md`](08_real_data_antihalluc_plan.md) §3.3 Pillar 3 (본 기획서로 구체화)
- `outputs/probes/phaseA_eval.jsonl` (Phase A' 43문항 실측)
- `outputs/cpt_bllossom_phaseA/checkpoint-77/trainer_state.json` (Phase A' loss 추이)

---

## 10. 이 문서의 한계 (과장 금지)

- 105쌍이라는 규모는 **LIMA (1,000쌍)** 의 1/10. 답변 품질 대비 부족할 가능성 있음. 필요 시 B 후속 단계에서 증강.
- SFT trainer (TRL SFTTrainer) 가 PEFT 0.13.2 + transformers 5.x 와 완전 호환되는지는 **실행 전 검증 필요** (unverified for this stack).
- `answer_length_ratio` 측정 프로토콜은 기획서 §02 §1.1 기준. Phase B 실행 후 이 지표가 0.8 미만으로 내려가면 SFT 데이터 length 상향 조정 필요.
- Phase A' 의 **창작 인물 환각 (이중옥기 등)** 이 SFT 로 완전 소멸한다는 보장은 없음. Whitelist 에 없는 인물이 generation 중 여전히 나올 수 있으므로, post-hoc entity validation 은 inference 측에도 필요할 수 있음.
- 본 기획서는 `book_008` (동의보감) 한 권 기준. 다책 확장은 Phase C 기획서에서.
