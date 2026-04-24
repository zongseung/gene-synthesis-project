# ver4 · 08. 실데이터 기반 환각 해결 기획서 (synth_facts 폐기 + real-text pivot)

- **버전**: r0 (2026-04-22)
- **선행 문서**: [`02_plan_v4.md`](02_plan_v4.md), [`07_R1_probe_results.md`](07_R1_probe_results.md)
- **진단 근거**: [`round_1/discriminator.md`](../../.claude/harness-evals/hanmed_cpt/round_1/discriminator.md), [`round_2/supervisor.md`](../../.claude/harness-evals/hanmed_cpt/round_2/supervisor.md)
- **폐기 대상 (본 문서로 대체)**:
  - `02_plan_v4.md` §2.1 "long-form expository paragraph 합성" — **LLM 생성 paraphrase** 경로 폐기
  - EXP-V4-06 "fact sheet + synthetic corpus pipeline" — synth_facts 코퍼스 산출 파트 폐기 (fact sheet 자체는 ground-truth 앵커로 유지)
- **유지**: `02_plan_v4.md` §2.1 의 fact sheet curation (`data/facts/core_factsheet.yaml`), `02_plan_v4.md` §5.1~5.3 의 전처리·trainer 버그 수정 (별도 작업으로 병행)

---

## 0. 한 줄 요약

**`hanmed_synth_facts.jsonl` (LLM 생성 합성 코퍼스) 은 round_2 에서 관찰된 F3 loop 환각의 직접 재료임이 실측으로 확인됨. 본 기획서는 (a) 합성 코퍼스 전량 학습 제외, (b) `data/raw/mediclassics_unified/` 의 실제 서문·발문·저자 언급 섹션을 자동 추출해 학습 mix 에 up-sample, (c) 실문장만 paraphrase 하는 reading-comprehension 스타일 augmentation 으로 entity binding 을 회복시킨다. 외부 근거: Allen-Zhu "Physics of LM" 3.1/3.3 (augmentation 필요 + junk ratio 가 capacity 파괴), HuatuoGPT-II (real-data QA unification), BianCang (general replay), arXiv:2401.07284 (Extended-Text Reading Comprehension).**

---

## 1. 문제 재정의

### 1.1 round_2 환각 실측 (요약)

probe 10문항 (`outputs/probes/probe_v4_content_v2.jsonl`, 2026-04-20) 에서 관찰된 실패 모드:

| 모드 | 설명 | 증거 |
|------|------|------|
| **F1 style-over-fact** | 수사만 흉내, 사실 오답 | C5, C9 (처방 구성 대부분 오답) |
| **F2 wrong-entity** | 책↔저자↔왕 잘못 바인딩 | Q2 "사상의학 창시자=장원소", C7 "침구경험방 저자=이제마" |
| **F3 repetition-loop** | 동일 템플릿 문장 반복 | C2 (579 tok), C3 (600 tok) loop |
| **F4 token-corruption** | 책 제목 글자 변형 | "동의보강/비급/포감/박원/봉급" 등 14종, 전 코퍼스 grep 0회 → 모델 창작 |

### 1.2 round_2 에서 확정된 3대 근본 원인

1. **EOS-as-separator 학습** — `preprocess.py:293,396` 이 `<|eot_id|>` 를 record 구분자로 삽입. 실측 EOS density = synth 0.31% / bi 0.63% / ko 1.17% / zh 1.47% (mix 가중 0.98%). 모델이 EOS 를 "종결" 이 아닌 "문단 전환" 으로 학습.
2. **Chat template mismatch** — CPT 는 plain text, inference 는 ChatML. 훈련 코퍼스에 `<|start_header_id|>` (128006) **0회**. inference 가 OOD 로 진입.
3. **Embed LoRA 잠듦** — `adapter_model.safetensors` 실측 `lora_embedding_A` norm = **0.267** (zeros init 잔존). 새 special token 4개 (`<ZH>/<KO>/<JA>/<EN>`) 학습 사실상 실패.

### 1.3 합성 코퍼스가 환각의 **추가** 주원인임을 입증하는 증거

`data/cpt/hanmed_synth_facts.jsonl` (1,791 records) 를 직접 열어본 결과, 모든 record 에 아래 **4~6개 footer 문장이 96~98% 빈도로 반복**되어 있음:

| freq | 문장 (book_8 동의보감 기준) |
|-----:|------|
| 98% | "이 같은 서지 정보는 『동의보감』이(가) 종합의서의 범주에 놓이는 까닭을 뒷받침한다" |
| 96% | "『동의보감』은(는) 조선 시대 의학 문헌사 안에서 고유한 위치를 차지한다" |
| 95% | "선조(宣祖) 대의 편찬 환경은 『동의보감』의 성격을 이해하는 또 다른 단서가 된다" |
| 77% | "이러한 연관 속에서 허준과(와) 『동의보감』은(는) 한국 한의학사에서 함께 기억되어 왔다" |

→ round_2 환각 응답 C2·C3·C7 의 반복 템플릿과 **동일한 문장**. 즉 F3 loop 환각은 synth_facts 의 footer 템플릿을 그대로 재생산한 결과로 판단됨.

본 프로젝트 `data/facts/core_factsheet.yaml` 주석 `_note: '1순위: vol_01 서문 regex / 2순위: kiom book_list / 3순위: 수기 검증 / 금지: LLM 자유생성'` 은 **이미 올바른 원칙을 명시**하고 있었으나 `src/data/synth/expand_facts.py` 가 이를 준수하지 못함.

### 1.4 그럼에도 augmentation 을 **완전히 버릴 수 없는** 이유

Allen-Zhu & Li, "Physics of Language Models: Part 3.1" ([arXiv:2309.14316](https://arxiv.org/abs/2309.14316), ICML 2024):

> "For knowledge to be reliably extracted, it must be **sufficiently augmented** (e.g., through paraphrasing, sentence shuffling, translations) during pretraining. Without such augmentation, knowledge may be memorized but not extractable, leading to 0% accuracy, regardless of subsequent instruction fine-tuning."

→ fact 1회 등장만으로는 CPT 학습량 부족. 반드시 augmentation 필요. **문제는 augmentation 자체가 아니라 "junk paraphrase" 임**.

Allen-Zhu & Li, Part 3.3 ([arXiv:2404.05405](https://arxiv.org/abs/2404.05405), ICLR 2025):

> "Junk data significantly reduces model capacity — **a 1:7 ratio of useful-to-junk training tokens causes capacity for useful knowledge to lose by a factor of 20x, even when useful knowledge is exposed 100 times**."
> "Prepending training data with **domain names** significantly increases a model's knowledge capacity."

→ 현재 synth_facts 는 footer 반복 구조상 "junk with embedded useful fact" 패턴. capacity 관점에서 real-text 대비 오히려 학습 효율을 **낮춤**.

---

## 2. 외부 증거 종합 (본 기획서의 이론적 지지대)

### 2.1 Physics of Language Models (Allen-Zhu & Li)

| 논문 | 핵심 처방 | 본 기획서 적용 |
|------|-----------|---------------|
| [Part 3.1 (2309.14316)](https://arxiv.org/abs/2309.14316) | Knowledge 는 **paraphrase augmentation 없이는 추출 불가능** | 실문장 기반 paraphrase 는 유지. LLM 자유 생성만 차단. |
| [Part 3.2 (2309.14402)](https://arxiv.org/abs/2309.14402) | Knowledge **manipulation** (역질문·비교) 능력은 inverse-search augmentation 필요 | 현 단계에서는 직접 추출까지 목표. manipulation 은 phase 2. |
| [Part 3.3 (2404.05405)](https://arxiv.org/abs/2404.05405) | **Domain prefix** 추가 시 capacity ↑ / **junk ratio 1:7 이면 capacity 20× 감소** | 각 record 에 `[book_title]` prefix 1회 삽입. synth_facts junk 제거. |

### 2.2 도메인 적응 / 의학 LLM

| 논문 / 모델 | 기법 | 본 기획서 적용 |
|-------------|------|---------------|
| [HuatuoGPT-II (arXiv:2311.09774)](https://arxiv.org/abs/2311.09774) | raw corpus → **LLM 으로 QA 재포장** 후 학습, CMExam +14% | QA 포맷 unification 은 **선택적**. 우선은 real-text augment 으로 입증 후 도입. |
| [BianCang (arXiv:2411.11027)](https://arxiv.org/abs/2411.11027) | TCM mix 에 **general replay 16%** 포함, TCMSD +60pp | `wiki_ko` 를 15~20% 편성. 이미 `data/cpt/wiki_ko.jsonl` 확보 완료. |
| [Extended-Text Reading Comprehension (arXiv:2401.07284)](https://arxiv.org/html/2401.07284v1) | **기존 도메인 텍스트 자체를** reading-comprehension style 로 재포장 | real preface 문장 → cloze / Q-A 변형. LLM 자유 생성 대체 경로. |

### 2.3 지식 획득 동역학

| 논문 | 함의 |
|------|------|
| [How Do LLMs Acquire Factual Knowledge (NeurIPS 2024, 2406.11813)](https://arxiv.org/html/2406.11813) | Factual knowledge 는 **micro-acquisition + forgetting 반복**. **low-freq 는 더 자주 노출시켜야 함**. |
| [Law of Knowledge Overshadowing (2502.16143)](https://arxiv.org/html/2502.16143v1) | High-freq entity 가 low-freq 를 overshadow → 현상황 "이시진 1795 : 허준 64" 의 직접 설명. **up-sampling 필수**. |
| [Bring Your Own Knowledge (2502.12598)](https://arxiv.org/html/2502.12598) | Knowledge expansion 방법론 survey. "data augmentation + up-sampling" 이 domain-specific knowledge injection 의 표준. |

### 2.4 고전 문헌 NLP

| 논문 | 함의 |
|------|------|
| [Cross-language classic entities (npj Heritage Science 2025)](https://www.nature.com/articles/s40494-025-01624-y) | 고전 한문/국역 병치 코퍼스에서 엔티티 인식 — mediclassics 와 동형. preface/colophon 기반 entity 추출이 유효함을 보고. |

### 2.5 기존 저장소 문서와의 관계

| 문서 | 본 기획서와의 관계 |
|------|-------------------|
| [`02_plan_v4.md`](02_plan_v4.md) §2.1 | **대체** — synth LLM 생성 → real-text augment |
| [`02_plan_v4.md`](02_plan_v4.md) §5.1~5.3 | **유지** — 전처리·trainer 버그 수정은 독립 트랙으로 병행 |
| [`05_new_token_training_methods.md`](05_new_token_training_methods.md) | **유지** — `trainable_token_indices` 경로로 embed LoRA 활성화 (round_2 D-NEW-1 대응) |
| [`06_tcm_llm_adapter_survey.md`](06_tcm_llm_adapter_survey.md) | **유지** — TCM LLM 관행 대조표 |
| [`research_hanmed_cpt_methodology_20260421.md`](../research_hanmed_cpt_methodology_20260421.md) §2.1.1~2.1.4 | **유지** — mix ratio 연구 결과 그대로 사용 |

---

## 3. 실데이터 기반 환각 해결 전략 — 4 pillars

### Pillar 1. 실서문·발문·저자 언급 직접 추출 (real-fact extractor)

**원칙**: `data/raw/mediclassics_unified/book_*/vol_01.jsonl` 의 첫 10~30 records (보통 content_level=AA/XX/OO/ZZ, up_path_nm 에 "序/跋/凡例" 포함) 는 **편찬 경위·저자·왕대** 를 그대로 담고 있음. 이를 **있는 그대로** 추출해 학습 데이터로 재투입.

**증거 샘플** (`data/raw/mediclassics_unified/book_008/vol_01.jsonl`):

| seq | level | trans_ko (실측) |
|-----|-------|----------------|
| 2 | XX | 어의 충근정량호성공신 숭록대부 양평군 **허준**이 하교를 받들어 짓습니다 (원문: `臣 許浚奉敎撰`) |
| 5 | ZZ | 우리 **선종대왕**은 … 병신년(1596)에 태의(太醫) 허준을 불러 하교하시기를 … |
| 6 | ZZ | **양평군 허준**은 일찍이 선종대왕 때에 의서를 지으라는 명을 특별히 받들어 … 이제 책을 편찬하여 진상하였다 |

→ `(저자: 허준, 왕: 선조, 연도: 1596~1613)` triple 이 **실제 동의보감 서문에 그대로** 존재.

**신규 스크립트**: `src/data/builder/extract_real_facts.py`

```
기능:
  1. mediclassics_unified/book_NNN/vol_01.jsonl 스캔
  2. 서지 섹션 식별:
     - content_level ∈ {AA, XX, OO, ZZ} AND content_seq ≤ 30
     - up_path_nm 에 {"序", "凡例", "跋", "目錄"} 포함
     - 또는 `core_factsheet.yaml` 의 author_hanja 가 trans_ko/original 에 등장
  3. 식별된 record 들을 별도 파일로 저장: data/cpt/hanmed_real_facts.jsonl
     스키마: {book_id, volume_id, content_seq, content_level, up_path_nm, text, _source: "real_preface"}
  4. 각 record 앞에 [book_title_ko (book_title_hanja)] prefix 1회 삽입
     (Allen-Zhu Part 3.3 "domain prefix" 증거 기반)
```

**기대 산출**: 34 books × 평균 10~30 records = 340~1020 real-fact records. 기존 bilingual 블록과 별도 shard 로 보존 → mix 비율 독립 조정 가능.

---

### Pillar 2. Fact sheet 기반 "최소 주입 1회 평서문"

**원칙**: `data/facts/core_factsheet.yaml` 의 curated triple 을 **권당 1줄 평서문 prolog** 로 주입. LLM 자유 생성 없이 **고정 template + factsheet 값 치환** 만 허용. 이는 `02_plan_v4.md` §5.1 의 book_meta_prolog 개념을 **template-lock** 으로 제한하는 버전.

**고정 template 예시**:

```
[{title_ko} ({title_hanja})] {author} 편찬, {dynasty} {reign} 대, {published_year}년 간행.
```

**검증 절차**:
- factsheet 의 `_confidence` 필드가 `high` 인 필드만 prolog 에 포함
- `none/low` 필드는 드롭 (불확실한 fact 주입 금지)
- 생성된 prolog 문자열은 `data/stats/book_meta_prolog.jsonl` 에 저장 후 **eye-review 필수**

**예상 규모**: book 당 1줄 × ~40자 × 34 books ≈ 1,400자. 학습 비중은 미미하나 Allen-Zhu Part 3.3 "domain prefix = capacity boost" 의 직접 구현.

---

### Pillar 3. Reading-comprehension 스타일 augmentation (실문장 paraphrase **만**)

**원칙**: LLM 자유 생성 금지. 대신 **추출된 실문장** 만 입력으로 받아 paraphrase. 출력에 factsheet 외 entity 가 등장하면 자동 reject.

**신규 스크립트**: `src/data/synth/rc_paraphrase.py` (기존 `expand_facts.py` 대체)

```
입력:
  - hanmed_real_facts.jsonl (Pillar 1 산출)
  - core_factsheet.yaml (entity whitelist)

처리:
  1. 각 real-fact record 마다 paraphrase ×2~3 요청
     - prompt: "다음 문장을 의미 보존하며 문체만 바꿔 다시 쓰세요. 원문에 없는 인물·연도·책명을 추가하지 마세요."
     - 구체 input 은 한 record (예: book_008 seq=5 본문 150자)
  2. 출력 entity validation:
     - `core_factsheet.yaml` 의 전 book author_hanja/author_ko 집합 외 인명 등장 시 reject
     - 원문에 없는 연도 (4자리 숫자) 등장 시 reject
     - "이 같은 연관 속에서" 등 footer 키워드 등장 시 reject (블랙리스트)
  3. 통과한 paraphrase 만 `data/cpt/hanmed_rc_paraphrase.jsonl` 에 기록
     - paraphrase 마다 base_record_id 와 entity_validation_pass 플래그 포함

출력:
  - real-fact 1개 → paraphrase 2~3개 (품질 검증 통과한 것만)
  - 예상 규모: 340~1020 real-fact × paraphrase 2 ≈ 700~2000 records
```

**Allen-Zhu Part 3.1 정합성**: paraphrase 개수 2~3 은 동 논문 실험의 "augmentation sufficient" 최소값. sentence shuffling 도 추가 증강 고려 (§phase C.2).

---

### Pillar 4. General replay 와 mix 재설계

**기존 `data/cpt/wiki_ko.jsonl` (33MB, 2026-04-21 확보) 활용**. `cpt_trainer.py:99` 에 이미 경로 예약됨. packed 도 완료 (`data/cpt_processed/wiki_ko_packed_2048.jsonl`).

[`research_hanmed_cpt_methodology_20260421.md`](../research_hanmed_cpt_methodology_20260421.md) §2.1.2 BianCang 실측 (general 16% → TCMSD +60pp) 과 §2.1.4 GeRe (1K 샘플만으로 +6.5pp F1) 의 기준선:

**신 mix 제안** (synth_facts 제거, real-fact / prolog / rc_paraphrase / wiki_ko 통합):

| corpus | 현 v5 비율 | 신 제안 | 비고 |
|--------|-----------|---------|------|
| hanmed_bilingual | 0.45 | **0.30** | 하향. zh 편중 완화. |
| hanmed_zh_only | 0.20 | **0.10** | 하향. KO 앵커 강화. |
| hanmed_ko_only | 0.35 | **0.30** | 소폭 하향. |
| **hanmed_real_facts** | — | **0.10** | 신규 (Pillar 1). up-sample 5~10×. |
| **hanmed_rc_paraphrase** | — | **0.05** | 신규 (Pillar 3). |
| **wiki_ko** | — | **0.15** | 신규 general replay. BianCang 16% 에 준함. |
| hanmed_synth_facts | 0.00 (v5 미포함 추정) | **0.00** | **전량 제외** (本 기획서). |

KO 앵커 = ko_only + real_facts + rc_paraphrase + wiki_ko = 0.60 (research doc §0 기준 0.50 이상 권장).
ZH 순수 노출 = zh_only = 0.10.

---

## 4. 구체 구현 계획 (Phase A ~ D)

### Phase A (D+0) — "safe rollback": synth 제거만

**목표**: 환각이 synth_facts 에 의존한다는 가설을 **가장 싸게 검증**.

1. `cpt_trainer.py:99,107` 에서 `hanmed_synth_facts` 경로 제거
2. mix: bi 0.45 / zh 0.15 / ko 0.25 / wiki_ko 0.15 (synth 자리에 wiki 투입)
3. 기존 전처리 산출물 재사용 (재 preprocess 불필요)
4. LoRA config 변경:
   - `modules_to_save` 제거
   - `trainable_token_indices={"embed_tokens":[128256..128259],"lm_head":[128256..128259]}` 추가 ([`05_new_token_training_methods.md`](05_new_token_training_methods.md) A 안)
5. `preprocess.py:293,396` 의 EOS-as-separator 는 Phase A 에서는 **건드리지 않음** — inference 쪽 hotfix 로만 대응 (`probe_factual.py` 에 `no_repeat_ngram_size=6`, `repetition_penalty=1.3`)

**소요**: 코드 1h + 재학습 3~4h + probe 1h = 약 **6h**.

**기대**: F3 loop 템플릿 반복 빈도 50%+ 감소. F2 "이제마 남발" 빈도 감소. F4 글자 변형 부분 감소.

**판정 기준**: round_2 probe 10문항 재실행 → "이 같은 연관 속에서" substring 출현율이 > 50% 감소하면 "synth 가설 확정".

---

### Phase B (D+1~3) — real-fact extractor

**선결**: Phase A 판정 완료.

1. `src/data/builder/extract_real_facts.py` 구현 (Pillar 1 사양)
2. 34 books 순회해 `data/cpt/hanmed_real_facts.jsonl` 생성
3. 샘플링 50 records eye-review — entity 오기, OCR 노이즈 확인
4. book_meta_prolog 생성 (Pillar 2): `data/stats/book_meta_prolog.jsonl` (34 줄)
5. `preprocess.py` 로 packed 생성: `data/cpt_processed/hanmed_real_facts_packed_2048.jsonl`
6. mix 재조정: 위 Pillar 4 신 mix 비율
7. 재학습 + probe

**지표 (신규)**:
- `bind_density_real` = "허준" 오른쪽 50자 내 "동의보감" 동시 출현 비율
  - baseline (현 v5 bilingual 기준): 측정 후 기록
  - 목표: × ≥ 5 개선 (round_1 discriminator 의 14:1 불균형을 3:1 로 완화)
- `F4_corruption_count` = probe 응답에서 책 제목 글자 변형 건수

**성공 기준**: `T1_acc ≥ 50%` AND `bind_density_real × ≥ 5` AND `F4_corruption_count = 0`.

**소요**: 스크립트 6h + preprocess 1h + 학습 4h + 평가 2h = 약 **13h** (1~2일).

---

### Phase C (D+3~5) — reading-comprehension paraphrase

**선결**: Phase B `T1_acc` 가 50~70% 범위에서 정체 시 진입 (개선 여지 있는 경우).

1. `src/data/synth/rc_paraphrase.py` 구현 (Pillar 3 사양)
2. LLM API 선택:
   - 옵션 1: Claude/GPT API (외부 key 필요, per-record 비용 ~$0.001 × 1000 = $1)
   - 옵션 2: 로컬 Bllossom-8B chat 모드 자체-paraphrase (API 비용 0, 품질 낮음)
   - 옵션 3: `transformers` 로 Qwen2.5-7B-Instruct (오픈) 로컬 (중간)
   - **초안 선택**: 옵션 1 (품질 우선, 비용 미미)
3. 생성 → entity validation → 통과본만 `data/cpt/hanmed_rc_paraphrase.jsonl`
4. 통과율·다양성 통계 (`data/stats/rc_paraphrase_verify.json`):
   - entity_validation_pass_rate ≥ 95% 목표
   - trigram jaccard 중앙값 ≤ 0.5 (synth_facts 는 0.50 이었음, rc 는 원문 제약으로 더 낮아야)
5. packed 생성 및 mix 투입
6. 재학습 + probe

**지표**: `T1_paraphrase` (holdout 재표현 문항 정답률) ≥ 50%.

**실패 해석**:
- `T1_paraphrase < 30%` → paraphrase 다양성 부족. prompt 다변화 또는 sentence shuffling (Allen-Zhu Part 3.1 의 "sentence order augmentation") 도입.
- entity_validation_pass_rate < 80% → LLM 이 자유 생성하려 함. prompt 재설계 (현재 지시문 강화).

**소요**: 스크립트 + API 호출 + 검증 = 약 **16h** (2~3일).

---

### Phase D (D+5~7) — ablation 재학습 + 최종 평가

**목표**: 각 pillar 기여도 정량화 → SCI 논문 ablation table 재료.

| 변형 | real_facts | rc_paraphrase | wiki_ko | prolog | 예상 T1 |
|------|:---------:|:-------------:|:-------:|:------:|:-------:|
| A0 (baseline) | — | — | — | — | 25% (round_2 실측 1/4) |
| A1 (Phase A) | — | — | 0.15 | — | 30~40% |
| A2 (Phase B) | 0.10 | — | 0.15 | **O** | 50~60% |
| A3 (Phase C) | 0.10 | 0.05 | 0.15 | **O** | 60~70% |
| A3+RAG | — | — | — | — | T1 상한 측정만 |

**A0** 은 기존 v5 그대로. **A1/A2/A3** 는 신 adapter. 각각 동일 random seed + 동일 T1_factual 30문항으로 비교.

**성공 기준** (`02_plan_v4.md` §1 지표 재사용):
- A3 에서 `T1_acc ≥ 70%` AND `T1_paraphrase ≥ 50%` AND `answer_length_ratio ∈ [0.8, 1.2]` AND `forgetting_rate ≤ 5%p`

**실패 시 전환**:
- A3 `T1_acc < 50%` 이고 A3+RAG `recall@3 ≥ 70%` 면 raw corpus fact 자체 부족 → Core 확장 크롤 (book_139/056 remaining vols) 필요 → `02_plan_v4.md` EXP-V4-00 resume 로 복귀.

---

## 5. 합성 데이터 마이그레이션

### 5.1 `hanmed_synth_facts.jsonl` 처리

- **학습 제외** (Phase A 부터).
- **파일 자체는 삭제하지 않음** — 향후 비교/검증에 필요할 수 있음. `data/cpt/hanmed_synth_facts.jsonl.deprecated` 로 rename 하거나 그대로 두되 `cpt_trainer.py` 의 `CORPUS_PATHS` 에서 제거.
- `data/cpt_processed/hanmed_synth_facts_*.jsonl` 은 mix 미참여 상태로 유지 (disk 손실 방지).

### 5.2 `data/facts/core_factsheet.yaml`

- **유지** — ground-truth 앵커 역할. Pillar 2 (prolog) 와 Pillar 3 (entity whitelist) 의 입력.
- 현 `core_factsheet.yaml.bak_n3` 는 이전 버전 백업이므로 유지.

### 5.3 `src/data/synth/expand_facts.py`

- **사용 금지** — 자유 생성 경로. `__init__.py` 에서 export 제거.
- 코드는 남기되 상단에 deprecation 주석 삽입 (reference 용).

### 5.4 `scripts/verify_synth_facts.py`

- **부분 유지** — diversity 측정 로직은 `rc_paraphrase` 검증에도 재사용 가능. `scripts/verify_rc_paraphrase.py` 로 복제 후 입력 경로 변경.

---

## 6. 성공 기준 / 실패 해석

### 6.1 종료 조건 (phase 1)

`02_plan_v4.md` §4 기준 계승 + 신 지표 추가:

```
T1_acc ≥ 70%
AND T1_paraphrase ≥ 50%
AND answer_length_ratio ∈ [0.8, 1.2]
AND forgetting_rate ≤ 5%p
AND bind_density_real × ≥ 5
AND F4_corruption_count = 0
AND "이 같은 연관 속에서" substring count (per 100 responses) ≤ 5
```

### 6.2 실패 시 단계적 전환

| 지표 실패 | 해석 | 다음 액션 |
|-----------|------|----------|
| F4_corruption > 0 | byte-fallback + embed LoRA 미학습 | `trainable_token_indices` 수준을 LoRA r=32 → full-matrix `modules_to_save` 로 교체 (DDP 호환성 재확인) |
| bind_density × < 3 | real-fact 추출량 부족 | vol_01 외 vol_02~03 앞부분도 포함. 또는 Core 확장 크롤 |
| T1_paraphrase < 30% | augmentation 부족 | Phase C 필수화, sentence shuffling 추가 |
| answer_length < 0.7 | 짧은 paraphrase 가 답변 길이 collapse 유발 | rc_paraphrase 의 min_tokens=150 강제 |
| forgetting > 10%p | KLUE 드롭 | wiki_ko 0.15 → 0.25 증량 또는 learning rate 하향 |

### 6.3 ablation 재료 (SCI 논문 Table)

| Method | T1_acc | T1_paraphrase | answer_length | forgetting | bind_density_real |
|--------|:------:|:-------------:|:-------------:|:----------:|:-----------------:|
| Bllossom-8B base | (A0) | | | — | |
| + synth_facts (v5) | (old) | | | | |
| + wiki replay only (A1) | | | | | |
| + real_facts + prolog (A2) | | | | | |
| + rc_paraphrase (A3) | | | | | |

---

## 7. 리스크 / 제한

### 7.1 실데이터 자체가 부족할 리스크

- 동의보감 book_008 은 서문 풍부하나 `book_056 의방유취` 등 일부는 `02_plan_v4.md` EXP-V4-00 에서 **본문 5% 미만**만 수집됨. 해당 책은 vol_01 서문조차 없을 수 있음.
- **대응**: Phase B 의 `extract_real_facts.py` 에서 "책당 real-fact record 수" 를 집계. 5 records 미만인 책은 `data/stats/book_factuality.json` 에 `low_coverage` 표시. 학습 mix 에서 해당 책을 **제외**하거나 외부 출처 (한국민족문화대백과) 보조 수집 검토.

### 7.2 reading-comprehension paraphrase 가 여전히 환각을 재주입할 리스크

- LLM 이 "의미 보존" 지시를 무시하고 footer-like 문장을 자동 추가할 가능성.
- **대응**: entity blacklist 에 round_2 에서 관찰된 junk 문장 (`"이 같은 연관 속에서"`, `"한국 한의학사에서 함께 기억되어 왔다"`, `"고유한 위치를 차지한다"`) 명시. 해당 substring 포함 시 reject.
- **추가 대응**: LLM 선택시 **temperature=0.3**, **system prompt 에 "금지 문구 list"** 명시. 샘플 50개 수동 검토 후 bulk 생성.

### 7.3 Allen-Zhu paraphrase ratio 대비 소규모

- 논문 실험은 paraphrase ×5 이상을 사용. 본 계획은 ×2~3.
- **대응**: Phase C 초기에 ×2 로 시작해 `T1_paraphrase` 가 50% 미만이면 ×5 로 증량. API 비용 ~$5 선 (low).

### 7.4 34 books 의 대부분이 중국 의서

- mediclassics catalog 에 본초강목·상한론 등 중국 의서 다수 포함. 한국 entity (허준/이제마/세종) binding 보강 목표 대비 scope mismatch 가능.
- **대응**: `extract_real_facts.py` 에 `core_factsheet.yaml:dynasty == "조선"` 필터 옵션 추가. 조선 의서 우선 학습 mix 구성.

### 7.5 외부 API 의존 (Phase C)

- Claude/GPT API 호출이 실패 / 비용 초과 / rate limit 가능.
- **대응**: 로컬 Qwen2.5-7B fallback 경로 준비. 품질 저하 시 엔티티 검증 통과율 기준으로 자동 폐기.

---

## 8. 즉시 실행 가능한 파일 경로 / 명령

### 8.1 Phase A 명령어 (가장 먼저)

```bash
cd /home/user/gene-synthesis-project/korean-medicine-llm

# 1. cpt_trainer.py 수정 (synth 제거 + wiki 편성 + trainable_token_indices)
#    대상 라인: src/training/cpt_trainer.py:99,107,457
#    수정 후 diff 를 먼저 검토.

# 2. inference probe hotfix — round_2 supervisor A1 액션
#    대상: scripts/probe_factual.py — no_repeat_ngram_size=6, repetition_penalty=1.3,
#          eos_token_id=[128009, 128001] 추가.

# 3. 재학습 (2 GPU)
PYTHONHASHSEED=0 .venv/bin/torchrun --nproc_per_node=2 \
  src/training/cpt_trainer.py \
  --output_dir outputs/cpt_bllossom_phaseA \
  --mix_name phaseA

# 4. 재평가
.venv/bin/python scripts/probe_adapter.py \
  --adapter outputs/cpt_bllossom_phaseA/adapter \
  --probe_set outputs/probes/probe_v4_content_v2_input.jsonl \
  --output outputs/probes/probe_phaseA.jsonl
```

### 8.2 Phase B 신규 파일

- `src/data/builder/extract_real_facts.py` — 신규
- `src/data/builder/__init__.py` — export 추가
- `data/cpt/hanmed_real_facts.jsonl` — 생성물 (gitignore)
- `data/cpt_processed/hanmed_real_facts_clean.jsonl` — preprocess 산출
- `data/cpt_processed/hanmed_real_facts_packed_2048.jsonl` — packed
- `data/stats/book_meta_prolog.jsonl` — prolog 목록
- `src/training/cpt_trainer.py:CORPUS_PATHS` — 신 corpus 등록

### 8.3 Phase C 신규 파일

- `src/data/synth/rc_paraphrase.py` — 신규 (expand_facts.py 대체)
- `scripts/verify_rc_paraphrase.py` — verify_synth_facts.py 기반 복제
- `data/cpt/hanmed_rc_paraphrase.jsonl` — 생성물
- `data/stats/rc_paraphrase_verify.json` — 검증 리포트

---

## 9. 참고문헌

### 9.1 외부 (peer-reviewed / arXiv)

**Knowledge injection / acquisition**
- Allen-Zhu & Li, "Physics of Language Models: Part 3.1, Knowledge Storage and Extraction" — [arXiv:2309.14316](https://arxiv.org/abs/2309.14316) (ICML 2024)
- Allen-Zhu & Li, "Physics of Language Models: Part 3.2, Knowledge Manipulation" — [arXiv:2309.14402](https://arxiv.org/abs/2309.14402)
- Allen-Zhu & Li, "Physics of Language Models: Part 3.3, Knowledge Capacity Scaling Laws" — [arXiv:2404.05405](https://arxiv.org/abs/2404.05405) (ICLR 2025)
- Chang et al., "How Do Large Language Models Acquire Factual Knowledge During Pretraining?" — [arXiv:2406.11813](https://arxiv.org/html/2406.11813) (NeurIPS 2024)
- Zhang et al., "Law of Knowledge Overshadowing: Towards Understanding, Predicting, and Preventing LLM Hallucination" — [arXiv:2502.16143](https://arxiv.org/html/2502.16143v1)
- Ou et al., "How Do LLMs Acquire New Knowledge? A Knowledge Circuits Perspective on Continual Pre-Training" — [arXiv:2502.11196](https://arxiv.org/html/2502.11196v1)
- Survey, "Bring Your Own Knowledge: A Survey of Methods for LLM Knowledge Expansion" — [arXiv:2502.12598](https://arxiv.org/html/2502.12598)
- "Comparing Knowledge Injection Methods for LLMs in a Low-Resource Regime" — [arXiv:2508.06178](https://arxiv.org/pdf/2508.06178)
- Ovadia et al., "Fine-Tuning or Retrieval? Comparing Knowledge Injection in LLMs" — [Semantic Scholar](https://www.semanticscholar.org/paper/b512451d431df9e411bea4c99f7135d010275445)

**Domain adaptation / CPT**
- Cheng et al., "Adapting Large Language Models via Reading Comprehension" (Extended-Text Reading Comprehension) — [arXiv:2401.07284](https://arxiv.org/html/2401.07284v1)
- BenTsao / Huatuo-Llama-Med-Chinese — [arXiv:2304.06975](https://arxiv.org/abs/2304.06975)
- HuatuoGPT-II — [arXiv:2311.09774](https://arxiv.org/abs/2311.09774)
- BianCang — [arXiv:2411.11027](https://arxiv.org/abs/2411.11027)
- Chinese-LLaMA-Alpaca — [arXiv:2304.08177](https://arxiv.org/abs/2304.08177)
- Chao et al., "Examining Forgetting in Continual Pretraining of Aligned LMs" — [arXiv:2401.03129](https://arxiv.org/abs/2401.03129)

**Classical text NLP**
- "Automatic recognition of cross-language classic entities based on large language models" — [npj Heritage Science 2025](https://www.nature.com/articles/s40494-025-01624-y)

**PEFT / trainable tokens**
- PEFT Troubleshooting, "Extending the vocabulary" — [HF docs](https://huggingface.co/docs/peft/en/developer_guides/troubleshooting)
- PEFT Trainable Tokens reference — [HF docs](https://huggingface.co/docs/peft/en/package_reference/trainable_tokens)

### 9.2 내부 (저장소)

- [`docs/ver4/02_plan_v4.md`](02_plan_v4.md) — ver4 마스터 플랜
- [`docs/ver4/05_new_token_training_methods.md`](05_new_token_training_methods.md) — 신 special token 학습 방법론
- [`docs/ver4/06_tcm_llm_adapter_survey.md`](06_tcm_llm_adapter_survey.md) — TCM LLM adapter 관행 조사
- [`docs/ver4/07_R1_probe_results.md`](07_R1_probe_results.md) — R1 probe 결과
- [`docs/research_hanmed_cpt_methodology_20260421.md`](../research_hanmed_cpt_methodology_20260421.md) — CPT 실패 모드 방법론 조사
- [`.claude/harness-evals/hanmed_cpt/round_1/`](../../.claude/harness-evals/hanmed_cpt/round_1/) — round_1 생성자·판별자·iteration plan
- [`.claude/harness-evals/hanmed_cpt/round_2/`](../../.claude/harness-evals/hanmed_cpt/round_2/) — round_2 generator·discriminator·supervisor
