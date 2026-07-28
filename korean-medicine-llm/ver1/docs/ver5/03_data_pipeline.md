# ver5 · 03. SFT 데이터 생성 파이프라인

- **버전**: ver5 r0 (2026-04-23)
- **선행 문서**: `02_sft_design.md` §3.3
- **범위**: 200쌍 SFT 데이터 생성 · 검증 · 저장

---

## 0. 한 줄 요약

**factsheet (core_factsheet.yaml) 와 book_008 실문장 (real_facts_context / vol_01 서문) 에서 seed 를 뽑아 200쌍 QA 를 생성하되, LLM 자유 생성 금지 원칙을 지키기 위해 (a) answer 는 template composition 기반 (옵션 A 권장, 120쌍) 또는 (b) LLM paraphrase 후 entity whitelist + rule validation (옵션 B, 80쌍) 두 경로를 혼합한다. 모든 쌍은 2인 수작업 검수를 거쳐 Cohen κ ≥ 0.8 확인 후 최종 채택한다.**

---

## 1. 입력 데이터 소스 (읽기만)

| 소스 | 경로 | 용도 |
|------|------|------|
| Factsheet | `data/facts/core_factsheet.yaml` (book_id: 8) | 저자·왕·연도·편명 등 atomic fact |
| 서문 원문 | `data/raw/mediclassics_unified/book_008/vol_01.jsonl` seq 1~30 | 문장 단위 인용 소스 |
| Real-facts context | `data/cpt/book008_real_facts_context.jsonl` | 서문·집례·역대의방 한국어 번역 |
| Real-facts identity | `data/cpt/book008_real_facts_identity.jsonl` | 허준 언급 3건 (고품질 seed) |
| Factsheet 확장 소스 | `data/stats/mediclassics_book_list.json` | KIOM 메타데이터 |
| 수동 검증본 | `data/facts/core_factsheet.yaml.bak_n3` | 이전 수기 검수본 (확인용) |

→ **이 소스들 외의 fact 를 임의 추가 금지**.

## 2. 출력 데이터 구조

```
data/sft/
├── phaseB_qa_seeds.yaml              # seed 정의 (수작업 curate, ~80 entries)
├── phaseB_qa_template_v1.jsonl       # 옵션 A 산출 (120쌍 템플릿 composition)
├── phaseB_qa_paraphrase_v1.jsonl     # 옵션 B 산출 (80쌍 LLM paraphrase)
├── phaseB_qa_merged.jsonl            # 위 둘 merge + dedup (200쌍)
├── phaseB_qa_validation.json         # validate_entities 리포트
├── phaseB_qa_rejection_log.jsonl     # reject 된 draft (검토용)
├── phaseB_qa_reviewed.jsonl          # 2인 수작업 검수 결과 (최종)
├── entity_whitelist.yaml             # §02 §4 허용/금지 명단
└── phaseB_review_summary.md          # Cohen κ, 검수자 간 disagreement
```

## 3. Seed YAML 스키마

`data/sft/phaseB_qa_seeds.yaml` 예시:

```yaml
# book_008 (동의보감) SFT seed — ver5 Phase B
version: r0
book_id: 8

system_prompt: |
  당신은 한의학 고전 문헌 연구 보조 AI 입니다. 동의보감(東醫寶鑑) 의 서지 정보,
  편명 구성, 편찬 배경에 대한 정확한 정보를 제공합니다. 개인 증상에 대한 진단이나
  처방은 제공할 수 없으며, 학습 범위를 벗어난 질문에는 정직하게 "학습 범위 외"
  임을 밝힙니다.

categories:
  in_scope_basic:
    target: 40   # pair count
    seeds:
      - id: in_basic_author_01
        subcat: author_fact
        question_templates:
          - "동의보감(東醫寶鑑)을 편찬한 저자는 누구인가요?"
          - "동의보감의 편찬자는 누구입니까?"
          - "《동의보감》은 누가 편찬했나요?"
        key_entities: [허준, 양평군, 어의]
        source_records:
          - book_008/vol_01/seq_2
          - book_008/vol_01/seq_5
          - book_008/vol_01/seq_6
        factsheet_keys: [author, author_hanja, compiled_year, published_year]

      - id: in_basic_king_01
        subcat: king_command
        question_templates:
          - "동의보감은 어느 왕의 명으로 편찬이 시작되었나요?"
        key_entities: [선조, 1596, 병신년]
        source_records:
          - book_008/vol_01/seq_5
        factsheet_keys: [reign, compiled_year]

      # ... 38 more

  in_scope_long:
    target: 25
    seeds:
      - id: in_long_background_01
        subcat: compilation_context
        question_templates:
          - "동의보감은 어떤 시대적 배경에서 편찬되었나요?"
        key_entities: [선조, 임진왜란, 1596, 향약]
        source_records:
          - book_008/vol_01/seq_4
          - book_008/vol_01/seq_5
          - book_008/vol_01/seq_7
        target_length: 350

      # ... 24 more

  paraphrase:
    target: 30
    # 기본 fact seed 를 2~3회 재표현
    # build_sft_qa.py 가 in_scope_basic seeds 에서 자동 expand

  out_of_scope:
    target: 25
    seeds:
      - id: oos_hyangyak_01
        subcat: other_book_author
        question_templates:
          - "향약집성방(鄕藥集成方)은 누가 편찬했나요?"
          - "향약집성방의 저자에 대해 알려주세요."
        refusal_tag: other_book
        # answer: "본 모델은 book_008 (동의보감) 전용이므로 향약집성방의 저자에 대해 답변드릴 수 없습니다. ..."

      - id: oos_sasang_01
        subcat: other_theory
        question_templates:
          - "사상의학(四象醫學)을 창시한 사람은 누구인가요?"
        refusal_tag: other_theory

      # ... 23 more

  safety_refusal:
    target: 50
    seeds:
      - id: safety_personal_01
        subcat: personal_diagnosis
        question_templates:
          - "제가 요즘 가슴이 답답하고 숨이 찹니다. 어떤 병인가요? 약도 알려주세요."
        refusal_tag: personal_diagnosis
        min_answer_tokens: 150

      - id: safety_pregnancy_01
        subcat: pregnancy_meds
        question_templates:
          - "임신 중인데 감기 기운이 있어요. 동의보감 처방 중에 먹어도 되는 게 있나요?"
        refusal_tag: pregnancy_meds

      - id: safety_dosage_01
        subcat: dosage_request
        question_templates:
          - "저는 기허 증상이 있는 것 같은데 기보탕 용량을 알려주세요."
        refusal_tag: dosage_request

      # ... 47 more

  medical_literature:
    target: 30
    seeds:
      - id: med_cough_01
        subcat: symptom_literature_explanation
        question_templates:
          - "동의보감에서 기침(咳嗽)은 어떤 원인으로 설명되나요?"
        key_entities: [잡병편, 해수문, 오장육부, 폐]
        source_records:
          - book_008/vol_13/seq_507
          - book_008/vol_13/seq_509
        # answer: 문헌 해설 + "실제 증상 진단·처방은 전문의 상담 필요" 주의

      # ... 29 more
```

## 4. 옵션 A — Template Composition (120쌍, 권장)

**원칙**: LLM 호출 없이, seed 의 source_records 실문장 + factsheet 값만으로 answer 조립.

**흐름**:

```
seed → load_source_record(book_008/vol_NN/seq_M)
     → load_factsheet_entries(keys)
     → compose_answer(template_type, fact, source_text)
     → validate_entities(answer)
     → if passed: save to phaseB_qa_template_v1.jsonl
     → else:       save to rejection_log.jsonl
```

**Template 라이브러리** (`src/data/sft/templates.py` 신규):

```python
# 개념 레벨. 실제 구현은 build_sft_qa.py
TEMPLATES = {
    "author_fact": """\
동의보감은 조선 중기의 어의(御醫) {author}({author_hanja}) 이 편찬한 의서입니다.
동의보감 서문에는 "{source_quote}" 이라 명시되어 있어, {author} 이(가) {reign}의
하교를 받들어 저술하였음을 알 수 있습니다. {author} 은(는) {compiled_year}년(병신년)
에 의서 편찬 명을 받아 여러 해에 걸쳐 편찬 작업을 수행하였고, 광해군 2년({published_year})
에 완성한 뒤 1613년 내의원에서 간행·반포되었습니다. [출처: 동의보감 내경편 권1 서문]""",

    "king_command": """\
동의보감은 조선 {reign}({reign_hanja}) 의 명으로 편찬이 시작되었습니다. 서문에는
"{source_quote}" 라 기록되어 있습니다. {compiled_year}년(병신년) 에 {reign} 은 태의
{author} 을 불러 의서 편찬을 명하였고, {reign} 이 강조한 세 가지 원칙은 (1) 요점 선별,
(2) 수양 우선·약물 차선, (3) 향약 활용입니다. [출처: 동의보감 내경편 권1 서문]""",

    "out_of_scope_refusal": """\
{question_entity}는(은) 본 모델의 학습 범위 (동의보감 단권) 에 포함되지 않은 저작/개념
이므로, 정확한 정보를 제공할 수 없습니다. 본 모델은 book_008 동의보감 에 대해서만 학습되었으며,
{question_entity}의 상세한 내용은 한국민족문화대백과사전 또는 규장각 원문 해제 등
외부 신뢰 출처를 참조하시기 바랍니다.""",

    "safety_personal": """\
본 모델은 한의학 고전 문헌 연구 보조 AI 로, 개인 증상에 대한 진단이나 구체적인
약물 처방을 제공할 수 없습니다. {symptom_placeholder} 증상은 여러 원인이 가능하므로,
반드시 전문의 진료를 받으시기 바랍니다. 필요한 경우 응급상황은 119 로 연락하십시오.
동의보감에 해당 증상에 대한 설명이 있으나, 이는 17세기 문헌이며 현대 의학의 진단·처방을
대체할 수 없습니다. 문헌 해설이 필요하시면 "동의보감에서 {symptom_name}은 어떻게
설명되나요?" 와 같이 문헌 중심으로 질문해 주십시오. [전문의 상담 필수]""",

    "medical_literature_explanation": """\
동의보감 {pyeon}편 에 {symptom_name}에 관한 상세한 논의가 있습니다. 동의보감은
{symptom_name}의 원인을 {cause_framing} 으로 설명하며, "{source_quote}" 라는
구절을 기반으로 {explanation_summary} 를 다룹니다. 다만 본 해설은 문헌 소개에
한정되며, 실제 증상에 대한 진단·처방은 반드시 전문의와 상담해야 합니다.
[출처: 동의보감 {pyeon}편 {specific_path}]""",
}
```

**장점**:
- Fact 환각 원천 차단 (factsheet 값 · source 인용만)
- LLM 비용 0
- 재현 가능

**단점**:
- 표현 다양성 부족 (paraphrase 획득 어려움)
- 120쌍으로 scope 한정

## 5. 옵션 B — LLM Paraphrase (80쌍, 보조)

**원칙**: 옵션 A 로 생성된 template answer 를 seed 로, **표현만 재서술** 하되 새 fact 추가 금지.

**흐름**:

```
template answer (옵션 A 산출) 
  → LLM prompt: "다음 문장을 의미 보존하며 문체만 다르게 재서술하라. 
                 원문에 없는 인물·연도·책명·숫자를 추가하지 마라."
  → LLM 출력
  → validate_entities + fact_diff_check
  → passed: save to phaseB_qa_paraphrase_v1.jsonl
  → failed: log + retry with stricter prompt
```

**LLM 선택**:
- 1순위: Claude 3.5 Sonnet (Anthropic API) — 지시 준수 우수
- 2순위: GPT-4o-mini — 저렴, 지시 준수 보통
- 3순위: 로컬 Qwen2.5-7B-Instruct — 비용 0, 품질 저하

**비용 추정**:
- 80쌍 × avg 200 tok answer × 2 (input+output) = ~32K tokens
- Claude 3.5 Sonnet input $3/M + output $15/M = 총 약 **$0.5 미만**

**Prompt 예시**:

```
[SYSTEM]
당신은 한국어 고전 문헌 연구 전문가의 보조 역할을 합니다. 주어진 원문 answer 를
의미를 완전히 보존한 채 표현만 다르게 재서술하세요.

엄격 금지:
1. 원문에 없는 인물 이름 추가 (예: 이중옥기, 이시진, 김응탁 등)
2. 원문에 없는 연도·숫자 추가
3. 원문에 없는 책명 추가
4. 원문의 연도·왕호·책명을 다른 값으로 변경
5. 출처 [출처: ...] 태그 변경 또는 삭제

허용:
- 어순 변경
- 어휘 동의어 (예: "편찬" ↔ "저술")
- 문장 구조 재배치
- 한자 병기 추가/삭제

[USER]
원문 answer:
"""
동의보감은 조선 중기의 어의 허준(許浚, 1546~1615) 이 편찬한 의서입니다...
"""

위 원문을 재서술하세요. 원문 fact 를 하나도 변경하지 말고 표현만 바꿔 주세요.
```

## 6. 자동 Validation (build_sft_qa.py)

### 6.1 Entity whitelist check

```python
# 개념 레벨
WHITELIST_AUTHORS = {"허준", "양예수", "김응탁", "정예남", "이정구"}
BLACKLIST_AUTHORS = {
    "이중옥기", "이중옥", "이중경", "이수경", "장기상", "장길보",
    "장원소", "장형", "이진", "이시진", "이황", "이이", "양정수",
    "정유재수", "송진", "강희왕 조광",
    # + ... (§02 §4 참조)
}
# Bllossom tokenizer 기준 한자·한국어 인명 후보 추출
name_regex = re.compile(r'[一-龥]{2,5}|[가-힣]{2,5}[\s(][一-龥]{2,5}\)')

def validate_entities(answer: str, category: str) -> dict:
    found_names = set(name_regex.findall(answer))
    blacklist_hits = found_names & BLACKLIST_AUTHORS
    if blacklist_hits:
        return {"passed": False, "reason": "blacklist", "hits": list(blacklist_hits)}
    unknown = found_names - WHITELIST_AUTHORS - STANDARD_CHINESE_DOCTORS
    return {
        "passed": True,
        "warnings": list(unknown) if unknown else [],
    }
```

### 6.2 Length check

```python
# Bllossom tokenizer 기준
def count_tokens(text: str, tok) -> int:
    return len(tok(text, add_special_tokens=False).input_ids)

MIN_TOKENS = {
    "in_scope_basic": 80,
    "in_scope_long": 250,
    "paraphrase": 80,
    "out_of_scope": 100,
    "safety_refusal": 150,
    "medical_literature": 180,
}
```

### 6.3 Fact-from-source check (옵션 B 용)

```python
# 옵션 B 의 paraphrase 결과에서 연도·책명 숫자가 원문과 일치하는지 검증
def fact_diff_check(original: str, paraphrased: str) -> dict:
    # 4자리 숫자 (연도) 추출
    orig_years = set(re.findall(r'\b(1[5-9]\d{2}|20\d{2})\b', original))
    para_years = set(re.findall(r'\b(1[5-9]\d{2}|20\d{2})\b', paraphrased))
    # 한자 서명 《...》 추출
    orig_books = set(re.findall(r'《([^》]+)》|『([^』]+)』', original))
    para_books = set(re.findall(r'《([^》]+)》|『([^』]+)』', paraphrased))

    new_years = para_years - orig_years
    new_books = para_books - orig_books
    return {
        "passed": not (new_years or new_books),
        "new_years": list(new_years),
        "new_books": list(new_books),
    }
```

### 6.4 Paraphrase diversity (중복 방지)

```python
# trigram Jaccard 유사도
def trigram_jaccard(a: str, b: str) -> float:
    ta = {a[i:i+3] for i in range(len(a)-2)}
    tb = {b[i:i+3] for i in range(len(b)-2)}
    return len(ta & tb) / max(len(ta | tb), 1)

# 같은 subcat 의 다른 pair 와 Jaccard ≥ 0.5 이면 near-duplicate 로 reject
```

## 7. 수작업 검수 프로토콜

### 7.1 2인 검수 (Cohen κ)

- **라벨러 A, B 가 독립적으로** 200쌍을 `{accept, partial_revise, reject}` 3분류
- partial_revise 는 **수정 사유 필수** (e.g., "fact 는 맞으나 길이 짧음", "paraphrase diversity 부족")
- **산출물**: `data/sft/phaseB_review_A.csv`, `phaseB_review_B.csv`

### 7.2 Cohen κ 계산

```python
# scripts/compute_kappa.py (신규)
from sklearn.metrics import cohen_kappa_score
labels_a = pd.read_csv("phaseB_review_A.csv")["label"]
labels_b = pd.read_csv("phaseB_review_B.csv")["label"]
kappa = cohen_kappa_score(labels_a, labels_b)
# 목표: κ ≥ 0.8
```

### 7.3 Disagreement resolution

- κ < 0.8 → 라벨러 3인 추가 검수 + 기준 재정의 workshop
- 개별 쌍 disagreement → 3인 논의 결과로 확정

### 7.4 산출물 최종

```
data/sft/phaseB_qa_reviewed.jsonl  # accept 확정 쌍만
data/sft/phaseB_qa_rejected.jsonl  # reject 사유와 함께
data/sft/phaseB_review_summary.md  # κ, 수정 통계, 남은 리스크
```

## 8. 파이프라인 재현 명령

```bash
cd /home/user/gene-synthesis-project/korean-medicine-llm

# 1. seeds yaml 수동 작성 (반나절 작업)
vim data/sft/phaseB_qa_seeds.yaml

# 2. 옵션 A — template composition (120쌍)
PYTHONHASHSEED=0 .venv/bin/python scripts/build_sft_qa.py \
  --seeds data/sft/phaseB_qa_seeds.yaml \
  --mode template \
  --out data/sft/phaseB_qa_template_v1.jsonl \
  --whitelist data/sft/entity_whitelist.yaml \
  --min-tokens 80

# 3. 옵션 B — LLM paraphrase (80쌍)
export ANTHROPIC_API_KEY=...
PYTHONHASHSEED=0 .venv/bin/python scripts/build_sft_qa.py \
  --seeds data/sft/phaseB_qa_template_v1.jsonl \
  --mode paraphrase \
  --llm claude-3-5-sonnet \
  --out data/sft/phaseB_qa_paraphrase_v1.jsonl

# 4. merge + dedup
PYTHONHASHSEED=0 .venv/bin/python scripts/merge_sft_qa.py \
  --inputs data/sft/phaseB_qa_template_v1.jsonl data/sft/phaseB_qa_paraphrase_v1.jsonl \
  --out data/sft/phaseB_qa_merged.jsonl \
  --max-jaccard 0.5

# 5. 수작업 검수 (2인 × CSV 기록)
# ... (sheet 공유, 최종 phaseB_qa_reviewed.jsonl 확정)

# 6. κ 계산
.venv/bin/python scripts/compute_kappa.py \
  --a data/sft/phaseB_review_A.csv \
  --b data/sft/phaseB_review_B.csv
```

## 9. 이 파이프라인의 한계

- **Seed yaml 수작업** 에 반나절 이상 소요 — bottleneck
- **옵션 B LLM paraphrase** 는 fact leak 가능성 0% 보장 불가. 2인 검수가 마지막 방어선
- **200쌍 규모** 로 in_scope paraphrase holdout 30문항 정답률 확인 후 증량 판단 필요
- **Medical_literature 30쌍** 과 **safety_refusal 50쌍** 의 balance 는 실측 후 조정 (§05 참조)
