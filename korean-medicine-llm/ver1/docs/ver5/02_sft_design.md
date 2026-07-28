# ver5 · 02. SFT 핵심 설계

- **버전**: ver5 r0 (2026-04-23)
- **선행 문서**: `01_experimental_evidence.md` (SFT 전환 근거)
- **후속 문서**: `03_data_pipeline.md` (데이터), `04_trainer_spec.md` (구현), `05_evaluation.md` (평가)
- **범위**: book_008 (동의보감) 단권. 다책은 `07_roadmap.md` Phase C.

---

## 0. 한 줄 요약

**ver5 본선은 Base Bllossom 위에 실문장 기반 150쌍 + 50쌍 refusal SFT (B2, 핵심) 로 새 LoRA adapter 를 fresh 학습해 QA 안정성 · safety refusal · entity 환각 세 가지 문제를 동시에 해결하는 것이다. Phase A' CPT adapter 와 ChatML wrap CPT 는 비교군으로만 남긴다. 모든 answer 는 단답을 금지하고 카테고리별 길이 범위 (in_scope 최소 80 tok, safety 100~200 tok, 평균 약 180 tok) 를 유지하며, LLM 자유 생성은 금지하고 entity whitelist + rule-based validation 에 통과한 draft 만 학습에 투입한다.**

---

## 1. Stack 개요

```
┌─────────────────────────────────────────────┐
│ Base: MLP-KTLim/llama-3-Korean-Bllossom-8B   │   (변경 없음)
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│ B2 (핵심 SFT): 200쌍 (150 + 50 refusal)       │   TRL SFTTrainer
│  • completion-only loss                        │   3 epoch, LR 2e-5
│  • fresh LoRA adapter                          │   Base 에서 새로 시작
│  • category별 answer 길이 제약                 │   data: 04_trainer_spec
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│ B3 (배포): adapter merge → vLLM               │   scripts/build_merged_model.py
│  • outputs/cpt_bllossom_ver5/adapter          │   docker-compose.phaseA.yml 재활용
│  • vLLM LoRA direct 또는 merged              │
└─────────────────────────────────────────────┘
```

비교군:
- `V1`: `outputs/cpt_bllossom_phaseA/adapter` (기존 Phase A')
- 선택 ablation: ChatML wrap CPT 또는 `--resume-adapter` 재개 실험

## 2. 설계 원칙 (4가지)

### 2.1 선행 CPT 실험은 교훈으로만 사용

| 선행 실험이 보여준 것 | fresh SFT 에 반영 | 비고 |
|----------------------|:----------------:|------|
| 동의보감 한국어 본문 fluency 가 필요 | ✅ | answer template / data 품질로 확보 |
| F4 corruption, zh_leak 억제 필요 | ✅ | eval 지표로 유지 |
| Factual recall 은 CPT-only 로 불안정 | ✅ | **SFT 가 QA 매핑 안정화** |
| Safety refusal 은 CPT-only 로 붕괴 | ✅ | **refusal 50쌍 + safety.py 로 회복** |
| 장문 덤프/길이 불균형 통제 필요 | ✅ | **길이 제약 학습** |

### 2.2 실문장 기반 (LLM 자유 생성 금지)

- 모든 SFT answer 의 **fact 주장** 은 `data/facts/core_factsheet.yaml` 또는 `data/raw/mediclassics_unified/book_008/vol_*.jsonl` 의 실문장에 근거
- LLM (GPT-4 등) 은 **paraphrase 용도에만** 허용 (표현만 변경, 새 fact 추가 불허)
- 생성 후 `validate_entities()` 자동 검증

### 2.3 Scope 고수

- book_008 단권. 향약집성방 · 동의수세보원 같은 다른 조선 의서는 Phase C (별도)
- Out-of-scope 질문에는 **"학습 범위 외입니다" 표준 reject** 학습 (25쌍)

### 2.4 Answer 길이 제약

- 모든 in_scope · paraphrase answer **최소 80 tok, 평균 180 tok**
- safety refusal 도 **100~200 tok** (단답 금지)
- 목적: `answer_length_ratio ∈ [0.8, 1.2]` 달성

## 3. 단계 상세

### 3.1 Mainline — fresh LoRA from base

- **시작점**: Base Bllossom 에 `get_peft_model()` 로 새 LoRA adapter 부착
- **정책**: `--resume-adapter` 없는 fresh 학습을 본선으로 정의
- **이유**: CPT-only 경로에서 확인된 creative 환각 prior 와 safety 붕괴를 본선에서 재사용하지 않기 위함

### 3.2 비교군 — Phase A' / ChatML wrap CPT (선택 ablation)

**목적**: fresh SFT 본선과 비교하기 위한 legacy 경로 보존. 본선 필수 아님.

**신규 데이터**:
- `data/cpt/book008_identity_chatml.jsonl`
- `data/cpt/book008_prolog_chatml.jsonl`

각각 Phase A' 의 identity/prolog shard 를 다음 포맷으로 wrap:

```
<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{sampled_question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{original_plain_text_from_phaseA}<|eot_id|>
```

- `{sampled_question}`: seed question 템플릿 (예: "동의보감 서문을 알려주세요.")
- `{original_plain_text_from_phaseA}`: 기존 `book008_real_facts_identity.jsonl` 의 실제 텍스트

**학습**: cap 2M tok, 1 epoch, LR 5e-5, single GPU.

**선택 이유**: fresh SFT 가 충분하면 skip. 수행 시에도 **비교군** 으로만 해석 (§05 참조).

### 3.3 B2 — 핵심 SFT 200쌍

**데이터 구성**:

| 카테고리 | 쌍 수 | 답변 길이 (tok) | 비고 |
|----------|:-----:|:-----:|------|
| in_scope 기본 fact | 40 | 100~200 | 저자·왕·연도·편명 등 factsheet seed |
| in_scope 해설체 장문 | 25 | 300~500 | "동의보감 편찬 배경은?" 같은 탐구형 |
| in_scope paraphrase (×2) | 30 | 100~200 | 같은 fact 의 재표현. query robustness 획득 |
| out_of_scope reject | 25 | 100~150 | 타 책 질문 → 표준 reject |
| **safety refusal** | **50** | 150~250 | 개인 증상 · 약물 상담 거부 + 전문의 안내 |
| medical 문헌 해설 | 30 | 200~400 | MED-01~06 형태 (해설 허용 분기) |
| **합계** | **200** | avg ~210 | — |

**Answer 작성 규칙**:

1. **Fact 주장** 은 반드시 factsheet · real_facts_context · vol_01 서문에서 인용
2. **인용문** 은 `"..." ([출처: book_008/vol_01/seq_N])` 형태로 명시
3. **Entity whitelist** 통과해야 함 (자동 validation)
4. **카테고리별 최소 길이 준수** — in_scope/paraphrase 최소 80 tok, safety 최소 100 tok, 그보다 짧으면 reject

**예시 (in_scope basic fact)**:
```yaml
id: SFT-IN-author-v1
category: in_scope
subcat: author_fact
user: "동의보감(東醫寶鑑)을 편찬한 저자는 누구인가요?"
assistant: |
  동의보감은 조선 중기의 어의(御醫) 허준(許浚, 1546~1615) 이 편찬한
  의서입니다. 동의보감 서문에는 "御醫 忠勤貞亮扈聖功臣 崇祿大夫 陽平君
  臣 許浚奉敎撰" 이라 명시되어 있어, 허준이 선조(宣祖) 의 하교를
  받들어 저술하였음을 알 수 있습니다. 허준은 1596년(선조 29, 병신년)
  에 의서 편찬 명을 받아 여러 해에 걸쳐 편찬 작업을 수행하였고,
  광해군 2년(1610) 에 완성한 뒤 1613년 내의원에서 간행·반포되었습니다.
  [출처: 동의보감 내경편 권1 서문]
_seed_source:
  - book_008/vol_01/seq_2
  - book_008/vol_01/seq_5
  - core_factsheet.yaml:book_id=8
_answer_tokens: 195
_entity_whitelist_check: passed
```

**예시 (safety refusal)**:
```yaml
id: SFT-SAFETY-personal-01
category: safety_refusal
subcat: personal_diagnosis
user: "제가 요즘 가슴이 답답하고 숨이 찹니다. 어떤 병인가요? 약도 알려주세요."
assistant: |
  본 모델은 한의학 고전 문헌 연구 보조 AI 로, 개인 증상에 대한 진단이나
  구체적인 약물 처방을 제공할 수 없습니다. 가슴이 답답하고 숨이 차는
  증상은 심장·호흡기 관련 여러 원인이 가능하므로, 반드시 전문의 진료를
  받으시기 바랍니다. 필요한 경우 응급상황은 119 로 연락하십시오.
  동의보감에 해당 증상에 대한 설명이 있으나, 이는 17세기 문헌이며
  현대 의학의 진단·처방을 대체할 수 없습니다. 문헌 해설이 필요하시면
  "동의보감에서 기허는 어떻게 설명되나요?" 와 같이 문헌 중심으로 질문해
  주십시오. [전문의 상담 필수]
_answer_tokens: 175
_safety_tag: personal_diagnosis
```

**예시 (medical 문헌 해설)**:
```yaml
id: SFT-MED-cough-01
category: medical_literature
subcat: symptom_explanation
user: "동의보감에서 기침은 어떻게 설명되나요?"
assistant: |
  동의보감 잡병편 권5 에 기침에 관한 상세한 논의가 있습니다. 동의보감은
  기침의 원인을 오장육부 전체 기능과 연결해 설명하며, "오장육부가 모두
  기침하게 하니 폐만 기침하게 하는 것이 아닙니다" 라는 황제내경 소문의
  구절을 인용합니다. 구체적으로 피모와 폐의 상합 관계, 한기의 침입 경로,
  계절별 병사의 작용을 다루며 폐해·심해·간해·비해·신해 등 오장해로
  구분합니다. 다만 본 해설은 문헌 소개에 한정되며, 실제 증상에 대한
  진단·처방은 반드시 전문의와 상담해야 합니다.
  [출처: 동의보감 잡병편 권5 해수문]
_seed_source:
  - book_008/vol_13/seq_507
  - book_008/vol_13/seq_509
_answer_tokens: 220
_entity_whitelist_check: passed
```

### 3.4 B3 — 배포

- `scripts/build_merged_model.py --adapter outputs/cpt_bllossom_ver5/adapter --output outputs/hanmed_merged_ver5`
- `docker-compose.phaseA.yml` 의 `HANMED_ADAPTER_DIR` 를 `../outputs/cpt_bllossom_ver5/adapter` 로 교체
- CLI wrapper (`scripts/cli_phaseA.sh`) 도 동일한 env var 로 재사용

## 4. Entity whitelist (허용 / 금지)

### 4.1 허용 (fact answer 에 등장 가능)

- **저자/편찬자**: 허준(許浚), 양예수(楊禮壽), 김응탁(金應鐸), 정예남(鄭禮男), 이정구(李廷龜)
- **왕 · 왕족**: 선조(宣祖), 광해군(光海君)
- **관호 · 공신호**: 어의(御醫), 양평군(陽平君), 충근정량호성공신, 숭록대부
- **의학 시조 (서문 인용)**: 헌원(軒轅, 황제), 기백(岐伯)
- **참고 중국 의가 (서문 인용)**: 창공(倉公), 진월인(秦越人, 扁鵲), 유완소(劉完素), 장종정(張從正), 주진형(朱震亨), 이고(李杲)
- **필요 시 추가**: factsheet yaml 에 추가 후 whitelist 자동 재생성

### 4.2 금지 (E1+E2+E3 실증 창작 목록)

다음 entity 가 **저자 / 편찬자 / 창시자 위치에** 등장 시 **자동 reject**:

| 금지 entity | 기원 | 실제 의미 |
|-------------|------|----------|
| 이중옥기(李重翼基) | Phase A' 창작 | 존재하지 않는 인물 |
| 이중옥(李仲玉) | Phase A' 창작 | 존재하지 않는 인물 |
| 이중경 | Phase A' 창작 | 존재하지 않는 인물 |
| 이수경 | Base 창작 | 존재하지 않는 인물 |
| 장기상(張吉甫) | Phase A' 창작 | 존재하지 않는 인물 |
| 장길보 | Phase A' 창작 | 존재하지 않는 인물 |
| 장원소(張元素) | R1 창작 | 금원4대가 중 1인, **사상의학 창시자 아님** |
| 장형(張衡) | Base 창작 | 중국 후한 천문학자, **의학자 아님** |
| 이진(李珍) | R1 창작 | 존재하지 않는 인물 |
| 이시진(李時珍) — 동의보감 저자 위치 | R1 창작 | 본초강목 저자 (중국), **동의보감과 무관** |
| 이황(李滉) — 저자 위치 | Base 창작 | 조선 유학자, 의학자 아님 |
| 이이(李瀷) — 저자 위치 | Base 창작 | 조선 유학자, 의학자 아님 |
| 양정수(楊挺壽) | Base 창작 | 존재하지 않는 인물 |
| 정유재수 | Phase A' 창작 | "정유재란" 을 인물로 오인 |
| 송진(宋進) | Phase A' 창작 | 존재하지 않는 인물 |
| 김응탁 — 동의보감 주저자 위치 | Phase A' (E3) 창작 | 실존 인물이나 **허준의 보조자**. 주저자 아님 |
| 강희왕 조광 | Phase A' (E3) 창작 | 청나라 황제, **동의보감과 무관** |

- **집계 갱신**: 매 실험 라운드마다 새 창작 entity 발견되면 yaml 에 append

### 4.3 Validation 로직

```python
# 개념 레벨 — 상세 구현은 04_trainer_spec.md §3
def validate_entities(answer: str, whitelist: set[str], blacklist: set[str]) -> dict:
    result = {"passed": True, "violations": []}
    # 1. 한자·한국어 인명 패턴 추출
    names = extract_names(answer)  # regex 기반
    # 2. blacklist 감지 시 무조건 fail
    for n in names:
        if n in blacklist:
            result["passed"] = False
            result["violations"].append(("blacklist", n))
    # 3. whitelist 외 인명 → warning (수작업 검수 대상)
    for n in names:
        if n not in whitelist and n not in blacklist:
            result["violations"].append(("unknown", n))
    return result
```

## 5. 성공 기준 / 실패 해석

### 5.1 목표 지표 (B2 완료 후 probe)

```
in_scope hit (수작업)        ≥ 75%      (Phase A' 73% 에서 소폭 개선)
paraphrase hit (수작업)      ≥ 65%      (Phase A' keyword 70% 의 실내용 회복)
out_of_scope reject          ≥ 70%      (현재 0% → 25쌍 학습으로)
MED-07/08 refusal            ≥ 50%      (Base 25% → 2× 달성이 현실적)
MED-01~06 dongui_style       ≥ 70%      (유지)
MED-01~06 구체 용량 제시     ≤ 20%      (safety 대 해설 균형)
answer_length_ratio          0.8~1.2
F3 loop                      0 (유지)
F4 corruption                0 (유지)
zh_leak (out_of_scope)       ≤ 10%
entity_whitelist_violation   0
```

### 5.2 실패 해석

| 관찰 | 해석 | 대응 |
|------|------|------|
| in_scope hit < 60% | SFT 에 core factsheet 커버리지 부족 | seeds yaml 재작성, 수작업 검수 부담 증가 |
| paraphrase hit ≫ in_scope | 암기 (training overfit) | paraphrase 증강, holdout 분리 재학습 |
| out_of_scope reject < 50% | 25쌍 부족 | 45쌍으로 증량 |
| MED refusal < 30% | 50쌍 refusal 도 부족 | 100쌍 + MED-01~06 분기 학습 재설계 |
| MED-01~06 구체 용량 ≥ 30% | safety 과소학습 | refusal 학습 시 "용량 금지" 명시 pair 추가 |
| answer_length_ratio < 0.7 | SFT answer collapse | epoch 축소, LR 하향 |
| entity_whitelist_violation > 0 | 데이터 fan-out 통제 실패 | seeds 재검수, LLM paraphrase prompt 강화 |

## 6. Ablation Matrix

| Variant | Phase A' CPT | ChatML wrap CPT | Fresh B2 (SFT) | 목적 |
|---------|:------------:|:----------------:|:---------------:|------|
| V0 Base | — | — | — | E1 baseline (기준) |
| V1 Phase A' | ✓ | — | — | 기존 CPT 비교군 |
| V2 Fresh B2 | — | — | ✓ | **ver5 본 설계** |
| V3 Bridge + B2 | — | ✓ | ✓ | 선택 ablation (bridge 필요성 검증) |
| V4 Resume + B2 | ✓ | — | ✓ | 선택 ablation (seed resume 영향 분리) |

→ 본선 판정은 V2 기준. V3/V4 는 필요 시에만 추가한다.

## 7. 리스크 / 제한

### 7.1 200쌍이 부족할 가능성

- LIMA: 1,000쌍 (Alignment). HuatuoGPT-II: 수만 쌍 medical QA.
- **우리 규모는 LIMA 의 1/5** — book_008 좁은 scope 이라 정당화 가능하나 실측 필요
- **Mitigation**: B2 후 paraphrase holdout 정답률이 35% 미만이면 `300쌍 증강 → 재학습`

### 7.2 Safety 과잉 거부 (false reject)

- 50쌍 refusal 이 강력 학습되어 MED-01~06 문헌 해설까지 거부할 위험
- **Mitigation**: refusal 50쌍과 medical 해설 30쌍을 **쌍별 대비 학습** (§03)
- **실패 시**: 문헌 해설 쌍 수 +20 으로 증강

### 7.3 LLM paraphrase 의존성

- `03_data_pipeline` 에서 paraphrase 생성에 GPT-4 API 사용 (옵션 B)
- **환각 재주입 위험**: validate_entities 가 못 잡는 사실 오류 가능
- **Mitigation**: **2인 수작업 검수** + 샘플링 10% 재측정

### 7.4 Entity whitelist 의 경직성

- 허용 목록에 없는 정답 entity 를 쓸 수 없음
- **Mitigation**: whitelist 는 **"의심 명단"** 으로 운영. 등장 시 flag → 수작업 확인 후 허용 추가

### 7.5 Phase A' resume 경로의 리스크

- Phase A' 의 "creative 환각 prior" 가 SFT 후에도 잔존할 가능성
- **정책 변경**: ver5 본선은 이 경로를 채택하지 않음. 필요 시 V4 resume ablation 으로만 측정

## 8. 다음 문서

- `03_data_pipeline.md` — seeds yaml 스키마, `build_sft_qa.py` 구현, LLM paraphrase 호출 규칙
- `04_trainer_spec.md` — TRL SFTTrainer 호환성, `cpt_trainer.py --mode sft` 확장, response_template 명시
- `05_evaluation.md` — 62문항 프로토콜, Cohen κ, `eval_phaseA.py` 확장
- `06_safety.md` — 50쌍 refusal + safety.py 2층 방어
- `07_roadmap.md` — Phase B → Phase C 다책 확장
