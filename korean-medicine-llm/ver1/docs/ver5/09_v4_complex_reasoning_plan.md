# ver5 · 09. v4 복합 추론 SFT 기획서 (Complex-Reasoning SFT)

- **버전**: ver5 v4 r0 (2026-04-23)
- **선행**: `08_sft_build_plan.md` (기초 SFT) · v3 (`phaseB_qa_diverse_v3.jsonl` 18,782쌍) 학습 완료
- **대상**: `outputs/cpt_bllossom_ver5_v3/adapter` 위에 **2차 SFT** 로 복합 추론 능력 주입
- **출력**: `outputs/cpt_bllossom_ver5_v4/adapter`

---

## 0. 한 줄 요약

**v3 는 "1 record → 1 fact" 형 단일 질문에서 자연어 QA 를 확보했지만, 복합 질문 (multi-hop · compare · list · conditional) 에서는 학습 시그널 부재로 여전히 약점을 보인다. v4 는 기존 34,040 record 를 재활용하되 **질문 생성 단계에서 2~3 record 를 조합** 하여 약 1,000~1,500 쌍의 복합 추론 데이터를 추가하고, 답변은 chain-of-thought (CoT) 스타일 로 단계 추론 구조를 학습시킨다. 학습은 v3 adapter 를 resume 해 연속 SFT 로 진행하며, 복합 probe 20문항을 신규 작성해 v3↔v4 델타를 수작업 검수한다.**

---

## 1. 배경 — v3 의 관측된 약점

### 1.1 v3 의 데이터 구조 한계

- 모든 학습 쌍이 **1 record = 1 fact** 구조
- multi-record 연결·비교·조건부 답변은 학습 시그널 0
- `source_records` 필드가 단일 ref 만 (`[book_008/vol_01/seq_5]`)

### 1.2 실제 사용 시나리오에서 실패할 복합 질문 유형

| 유형 | 예시 질문 | 실패 원인 |
|------|----------|----------|
| Multi-hop | "허준이 편찬을 시작한 왕은 누구고 완성 시에는 왕이 누구였나요?" | vol_01/seq_5 + seq_6 연결 학습 없음 |
| Compare | "내경편과 외형편은 각각 무엇을 다루나요?" | 2 편 총목 동시 인용 학습 없음 |
| Contrast | "기허와 허로의 차이는?" | 2 병증 대조 학습 없음 |
| List | "잡병편에 포함된 병증을 모두 알려주세요" | 총목 records 를 하나의 답변에 나열 학습 없음 |
| Conditional | "기침에 가래가 섞인 경우 동의보감은 어느 유형으로 분류하나요?" | 조건 분기 reasoning 학습 없음 |
| Temporal | "편찬 시작과 완성 사이에 어떤 일이 있었나요?" | 시간 순서 연결 reasoning 없음 |

### 1.3 v3 학습 끝난 후 실측 필요 (선행 조건)

- v3 SFT 완료 후 **복합 probe 20문항** 으로 실제 실패 패턴 기록
- 기록된 실패를 seed 설계에 직접 반영

## 2. v4 목표 지표

| 지표 | v3 기준 (예상) | **v4 목표** |
|------|:---:|:---:|
| 단일 서지 QA (수작업) | ≥ 75% | **유지 (≥ 75%)** |
| 자연어 병증 QA | ≥ 65% | **유지 (≥ 65%)** |
| **Multi-hop (2-hop)** | < 20% | **≥ 55%** |
| **Compare/contrast** | < 20% | **≥ 50%** |
| **List 류** | < 30% | **≥ 60%** |
| **Conditional** | < 15% | **≥ 40%** |
| safety refusal | ≥ 50% | **유지** |
| entity 환각 | 0 (blacklist) | **유지** |
| answer_length_ratio | 0.8~1.2 | **유지** |

v3 성취는 유지하면서 복합 구간만 선택적으로 끌어올린다.

## 3. 4개 Seed 유형 상세

### 3.1 Multi-hop seed (500쌍)

**정의**: 2~3 개 record 를 연결해 **하나의 답** 을 도출. answer 에 각 record 의 literal quote 와 추론 연결 문장을 모두 포함.

**구성 축** (record 조합 규칙):

| 조합 유형 | record 패턴 | 쌍 수 |
|----------|------------|:---:|
| 서문 chain | vol_01/seq_5 + seq_6 (편찬 시작→완성) | 80 |
| 서문 + 인명 | seq_2 (허준 서명) + seq_5 (편찬 인원) | 60 |
| 편 + 편 순서 | seq_8 (5편 구성) + seq_14/18/34 (편별 세부) | 80 |
| 병증 + 위치 | 병증 설명 + 해당 편 총목 | 150 |
| 인용 연쇄 | 같은 병증의 2~3 source records (병인→증상→분류) | 130 |

**템플릿 예시 (multi_hop_regents)**:

```
Q: 동의보감 편찬이 시작된 때와 완성된 때의 왕은 각각 누구였나요?

A: 동의보감 편찬의 시작과 완성 시 왕은 다릅니다.

① 편찬 시작 시점의 왕은 선조(宣祖) 입니다. 서문에 "{quote_seq5}" 라고 
기록되어 있어, 1596년 병신년에 선조가 태의 허준을 불러 편찬을 명한 것을 
확인할 수 있습니다.

② 완성 시점의 왕은 광해군(光海君) 입니다. 서문은 "{quote_seq5_ending}" 
라 기술하여, 1610년 경술년에 허준이 책을 완성해 진상하였고 이때 이미 
선조는 승하(1608) 한 뒤 광해군이 즉위한 상태였음을 알려 줍니다.

③ 이어지는 서문 기록 "{quote_seq6}" 에 따르면 광해군은 완성된 책을 
받아 내의원에서 간행·반포하도록 명하였습니다.

즉, 동의보감은 선조 → 광해군 두 왕대에 걸쳐 편찬된 책입니다.
[출처: 동의보감 내경편 권1 서문]
```

- 핵심 요소: 각 hop 마다 **번호 표시 (①②③) + literal quote + 중간 추론 문장**
- 모델이 "multi-record 를 이렇게 연결한다" 는 구조를 학습

### 3.2 Compare/Contrast seed (300쌍)

**정의**: 같은 level 의 두 개체를 대조.

| 대조 축 | 예시 | 쌍 수 |
|--------|------|:---:|
| 편 × 편 | "내경편 vs 외형편", "잡병편 vs 탕액편" | 80 |
| 병증 × 병증 | "기허 vs 허로", "기침 vs 천식" | 80 |
| 개념 × 개념 | "정 vs 기", "혈 vs 영" | 60 |
| 인물 × 인물 | "허준 vs 양예수 역할" | 30 |
| 원인 × 원인 | "한사(寒邪) vs 열사(熱邪) 기침 원인" | 50 |

**템플릿 예시 (compare_pyeons)**:

```
Q: 동의보감 내경편과 외형편은 각각 어떤 내용을 다루나요?

A: 두 편은 대상 영역이 뚜렷이 다릅니다.

**내경편 (內景篇)** 은 인체 내부 요소를 다룹니다. 
총목에 "{quote_seq14}" 라고 기록되어 있듯, 신형·정·기·신·혈 같은 몸의
근본 구성과 "{quote_seq18}" 의 오장육부(五臟六腑) 체계가 내경편의 중심
주제입니다.

**외형편 (外形篇)** 은 인체 외부 부위의 병을 다룹니다. 
총목은 "{quote_seq24}" 처럼 머리·얼굴·눈에서 시작해 "{quote_seq26}" 의
귀·코·입·치아·인후 등 각 부위별 증상을 분류합니다.

**정리**: 내경편은 "몸 안" 의 원리·장부 체계, 외형편은 "몸 밖" 의 부위별
질환을 대상으로 한 분류입니다.
[출처: 동의보감 내경편 권1 총목]
```

- **굵은 글씨 표제 + 2-column 구조 + 정리 한 줄** 로 비교 scaffold 를 학습

### 3.3 List seed (200쌍)

**정의**: "…의 종류는?" / "…에 속한 항목은?" 에 대해 총목 record 들을 하나의 답에 **리스트로 통합**.

| list 대상 | 예시 질문 | 쌍 수 |
|----------|----------|:---:|
| 편의 권 구성 | "내경편은 몇 권이고 권별 주제는?" | 40 |
| 권의 하위 문 | "잡병편 권5 에 속한 문(門)들은?" | 60 |
| 병증 분류 | "잡병편 풍문에 속한 병명은?" | 60 |
| 편의 대주제 | "외형편이 다루는 신체 부위는?" | 40 |

**템플릿 예시 (list_pyeon_volumes)**:

```
Q: 동의보감 잡병편에는 어떤 주제들이 포함되나요?

A: 잡병편은 외감·내상·기타 증후를 포괄적으로 다루는 편으로, 총목에
따르면 다음과 같이 구성됩니다.

1) 천지운기·심병·변증·진맥·용약·토·한·하 — "{quote_seq32}"
2) 풍·한(상) — "{quote_seq34}"  
3) 한(하)·서·습·조·화 — "{quote_seq36}"
4) 내상·허로 — "{quote_seq38}"
5) 곽란·구토·해수 — "{quote_seq40}"
… (이하 잡병편6~10 생략 혹은 계속)

즉 잡병편은 외감(풍·한·서·습·조·화) → 내상·허로 → 각종 급성·만성
증후 → 응급·부인·소아 순으로 분류됩니다.
[출처: 동의보감 내경편 권1 잡병편 총목]
```

- **번호 매기기 + 각 항목 literal quote + 구조 요약** 으로 list 능력 학습

### 3.4 Chain-of-Thought (CoT) 답변 구조

**정의**: 복합 질문에서 답변을 "단계 추론" 형태로 구성. §3.1~3.3 의 모든 답변에 기본 적용.

**CoT 4-단계 뼈대**:

```
1. 질문 분해       : "이 질문은 …와 …를 묻고 있습니다."
2. 각 파트별 근거  : "먼저 …에 대해 동의보감은 '…' 라 기록합니다."
                     "다음으로 …는 '…' 라 기술되어 있습니다."
3. 연결·추론       : "두 기록을 합치면 …라는 결론이 나옵니다."
4. 최종 답         : "따라서 …입니다."
```

**예시** (Conditional 질문):

```
Q: 가래가 섞인 기침은 동의보감에서 어떻게 분류되나요?

A: 이 질문에 답하려면 동의보감이 기침을 어떻게 세분하는지 먼저 보아야
합니다.

**Step 1** — 기침의 기본 분류: 동의보감 잡병편 해수문은 기침을 세 가지로
나눕니다. "{quote_vol13_seq479}" 에 따르면 ① 해(咳) 는 소리만 있고
가래가 없는 것, ② 수(嗽) 는 소리는 없고 가래만 있는 것, ③ 해수(咳嗽) 는
가래와 소리가 모두 있는 것입니다.

**Step 2** — 질문 상황 매핑: 질문은 "가래가 섞인 기침" 입니다. 가래가
있으므로 ①(해) 는 아닙니다. 소리 여부는 명시되지 않았으므로, 소리가
없으면 ②(수), 소리도 함께 있으면 ③(해수) 에 해당합니다.

**Step 3** — 결론: 가래만 있는 경우에는 **수(嗽)**, 가래와 소리가 모두
있는 경우에는 **해수(咳嗽)** 로 분류됩니다. 다만 본 답은 17세기 문헌의
분류 기술을 소개한 것이며, 실제 진단·치료는 전문의와 상담해 주십시오.

[출처: 동의보감 잡병편 권5 해수문 (vol_13)]
```

- **Step 표시 + literal quote + 분기 논리** 를 명시적으로 보여 줌
- 모델이 "조건부 질문에서는 이렇게 단계적으로 접근" 하는 패턴을 학습

## 4. 데이터 파이프라인

### 4.1 스크립트 확장

```
scripts/build_sft_diverse.py           # v3 (기존)
scripts/build_sft_complex.py           # v4 신규 — multi_hop/compare/list/cot 생성
data/sft/complex_seeds.yaml            # v4 신규 seed yaml (curate)
data/sft/phaseB_qa_complex_v4.jsonl    # v4 산출 (~1,000~1,500쌍)
data/sft/phaseB_qa_v4_merged.jsonl     # v3 + v4 merge (optional)
```

### 4.2 complex_seeds.yaml 스키마

```yaml
version: v4
system_prompt: <v3 와 동일>

multi_hop:
  - id: mh_regents_transition
    subtype: multi_hop_temporal
    question_templates:
      - "동의보감 편찬이 시작된 때와 완성된 때의 왕은 각각 누구였나요?"
      - "동의보감은 어느 왕의 명으로 시작해서 어느 왕 때 완성되었나요?"
    source_records:
      - { ref: book_008/vol_01/seq_5, role: start }
      - { ref: book_008/vol_01/seq_6, role: complete }
    template_id: multi_hop_regents
    min_length_tokens: 250

compare:
  - id: cmp_naegyeong_vs_oeheyong
    subtype: compare_pyeons
    question_templates:
      - "동의보감 내경편과 외형편은 각각 무엇을 다루나요?"
    source_records:
      - { ref: book_008/vol_01/seq_14, role: left_inner }
      - { ref: book_008/vol_01/seq_18, role: left_inner }
      - { ref: book_008/vol_01/seq_24, role: right_outer }
      - { ref: book_008/vol_01/seq_26, role: right_outer }
    template_id: compare_pyeons
    min_length_tokens: 300

list:
  - id: list_jabbyeong_topics
    subtype: list_pyeon_topics
    question_templates:
      - "동의보감 잡병편에는 어떤 주제들이 포함되나요?"
    source_records:
      - { ref: book_008/vol_01/seq_32, role: part }
      - { ref: book_008/vol_01/seq_34, role: part }
      - { ref: book_008/vol_01/seq_36, role: part }
      - { ref: book_008/vol_01/seq_38, role: part }
      - { ref: book_008/vol_01/seq_40, role: part }
    template_id: list_pyeon_topics
    min_length_tokens: 280

conditional:
  - id: cond_cough_with_phlegm
    subtype: conditional_symptom_classify
    question_templates:
      - "가래가 섞인 기침은 동의보감에서 어떻게 분류되나요?"
    source_records:
      - { ref: book_008/vol_13/seq_479, role: definition }
    template_id: cot_symptom_classify
    min_length_tokens: 300
```

### 4.3 build_sft_complex.py 주요 로직

```python
def expand_multi_hop(seed, records, ...):
    """각 source_records[i].ref 의 quote 를 얻어 template 에 주입.
    CoT 구조로 답변 조립."""
    for qt in seed.question_templates:
        slots = {
            "quote_start": fetch_quote("start"),
            "quote_complete": fetch_quote("complete"),
            ...,
        }
        answer = render(seed.template_id, slots)
        emit({
            "q_format": f"F_complex_{seed.subtype}",
            "a_format": "A_cot",
            "messages": ...
        })

def validate_complex(answer, seed):
    # 1. 각 source_records quote 가 answer 에 literal 존재
    # 2. Step/①② 표식 존재 (CoT 구조 확인)
    # 3. min_length_tokens 충족
    # 4. entity_whitelist pass
    # 5. atomic_fact_check
```

### 4.4 자동 생성 범위

| 유형 | 수작업 seed | 자동 확장 (질문 templates × 2~3) | 합계 목표 |
|------|:---:|:---:|:---:|
| multi_hop | 150 | ×2 | **300** |
| compare | 100 | ×2 | **200** |
| list | 80 | ×2 | **160** |
| conditional | 50 | ×3 | **150** |
| CoT 재작성 (기존 v3 일부) | - | 기존 답변 rewrite | **200** |
| **합계** | **380 seed** | — | **~1,000~1,200** |

v3 의 18,782 과 합치면 약 20,000쌍. 크기 과함 우려 시 v3 일부 (중복성 높은 병증) 를 subsample.

## 5. 학습 구성

### 5.1 2차 SFT (v3 adapter resume)

```bash
cd experiments/dongui_bogam
PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES=0 \
  ../../.venv/bin/python scripts/train_sft.py \
    --data data/sft/phaseB_qa_complex_v4.jsonl \
    --resume-adapter ../../outputs/cpt_bllossom_ver5_v3/adapter \
    --output ../../outputs/cpt_bllossom_ver5_v4 \
    --epochs 2 --lr 1e-5 \
    --micro-bs 2 --grad-accum 4 \
    --lora-rank 32 --lora-alpha 64 \
    --max-seq-length 1536
```

**파라미터 근거**:
- **resume**: v3 가 이미 자연어 포맷·서지 QA 를 확보했으므로 그 위에 쌓음 (fresh 로 다시 시작하면 v3 학습 낭비)
- **lr 1e-5** (v3 의 2e-5 보다 절반): 기존 능력 보존 + 복합 레이어만 미세 조정
- **epochs 2**: 복합 데이터가 작으므로 2회 노출 (total ~300 step)
- **max_seq_length 1536** (v3 의 1024 보다 증가): CoT 답변이 길어서 1024 로 잘릴 위험

### 5.2 Fresh 재학습 대안 (backup)

v3 adapter resume 이 예상 밖 부작용 (catastrophic forgetting, overfit) 을 보이면:
- v3 dataset + v4 dataset 을 **merge** 해 fresh LoRA 로 1회 학습
- train mix 비율: v3 80% / v4 20% (stratified sampling)

## 6. 평가

### 6.1 복합 probe 20문항 신규 작성

`eval/hanmed_eval_v0/phaseB_complex_probe.jsonl`:

| ID | 유형 | 질문 |
|----|------|------|
| CX-01 | multi_hop | "편찬 시작과 완성 시 왕은?" |
| CX-02 | multi_hop | "허준과 이정구는 각각 무엇을 담당했나요?" |
| CX-03~05 | multi_hop | … 3개 더 |
| CX-06 | compare | "내경편과 외형편 차이는?" |
| CX-07 | compare | "기허와 허로 차이는?" |
| CX-08~10 | compare | … 3개 |
| CX-11 | list | "잡병편에 포함된 주제들은?" |
| CX-12 | list | "내경편 권별 주제는?" |
| CX-13~15 | list | … 3개 |
| CX-16 | conditional | "가래 있는 기침은?" |
| CX-17 | conditional | "임신 중 기침은 어떻게 분류?" |
| CX-18~20 | conditional | … 3개 |

### 6.2 수작업 검수 기준

기존 v3 probe 기준 + 복합 전용 추가 항목:

| 항목 | 합격 기준 |
|------|---------|
| 각 hop 의 fact 정확성 | 모든 hop 이 정확해야 correct |
| hop 간 연결 논리 | "따라서" / "즉" 류 명시적 연결 |
| CoT 구조 (Step/번호) | 복합 질문에서 Step 표식 존재 |
| literal quote | 각 source_records 의 quote 가 answer 에 포함 |

### 6.3 v3 ↔ v4 델타 비교

```
V3 probe 20 complex 응답 → 수작업 검수 → correct rate X%
V4 probe 20 complex 응답 → 수작업 검수 → correct rate Y%
Δ = Y - X

목표 Δ:
  multi_hop  : +35%p
  compare    : +30%p  
  list       : +30%p
  conditional: +25%p
```

### 6.4 regression 방지 체크

v3 probe 43문항 (단일 QA) 에서 v4 가 v3 대비 3%p 이상 떨어지면 overfit. resume lr 하향 또는 epoch 축소.

## 7. 일정 (3일)

| Day | 작업 | 산출물 |
|:---:|------|------|
| 1 | `complex_seeds.yaml` 380 seed curate + `build_sft_complex.py` 작성 | seed yaml + script |
| 1 | `phaseB_complex_probe.jsonl` 20문항 curate | probe jsonl |
| 2 | v4 dataset 생성 + 품질 검증 (quote literal, CoT 구조, entity) | `phaseB_qa_complex_v4.jsonl` (~1,000쌍) |
| 2 | v4 학습 (resume) | `outputs/cpt_bllossom_ver5_v4/adapter` |
| 3 | v3/v4 복합 probe 수작업 검수 + 델타 집계 | 보고서 |

## 8. 리스크

| 리스크 | 가능성 | 대응 |
|-------|:---:|------|
| resume 시 catastrophic forgetting (v3 단일 QA 능력 저하) | 중 | lr 1e-5 + regression probe 43 병행 |
| CoT 길이 증가로 max_seq_length 2048 초과 | 중 | max_seq 1536 확대, 초과 샘플 reject |
| multi_hop 의 hop 간 논리 오류 | 중 | template literal quote 강제 + hop 번호 표식 검증 |
| compare 쌍이 자동 생성 시 "차이 없음" 같은 빈 답변 | 저 | 수작업 검수로 reject, template 구조가 비교 prompt 강제 |
| 1,000쌍이 너무 적어 효과 미미 | 중 | 수작업 검수 후 부족하면 2,000쌍으로 증량, epoch 3 확장 |
| CoT 가 실제 사용자 질문에는 과함 (장황) | 저 | 단일 질문에는 v3 포맷 유지, CoT 는 복합 전용 |

## 9. 구현 체크리스트

- [ ] v3 smoke test 완료 — 복합 probe 20문항 실측 기록
- [ ] `data/sft/complex_seeds.yaml` 380 seed 작성
- [ ] `scripts/build_sft_complex.py` 구현 (multi_hop/compare/list/cot 분기)
- [ ] template literal quote validator 추가 (각 role 별 quote 가 answer 에 literal 존재)
- [ ] CoT 구조 validator (Step/①②/번호 패턴 검출)
- [ ] `eval/hanmed_eval_v0/phaseB_complex_probe.jsonl` 20문항 작성
- [ ] v4 dataset 생성 (~1,000쌍) + 품질 리포트
- [ ] train_sft.py 에 `--resume-adapter` 플래그 확인 / 필요시 추가
- [ ] v4 학습 실행 (2 epoch, lr 1e-5)
- [ ] v4 adapter 로 복합 probe 20 + 기존 probe 43 generate
- [ ] 수작업 검수 결과표 작성 (v3 vs v4 delta)

## 10. 범위 외 (명시적 제외)

- **Multi-turn 대화** — v5 이후
- **DPO** (복합 답변 쌍 선호 학습) — v5 이후
- **자동 seed 생성 (LLM 호출)** — 본 v4 는 수작업 curate 만. 환각 위험 차단.
- **편명 재포함** — v3 에서 제외한 path 라벨은 v4 에서도 제외 유지

## 11. 이 기획서의 자기 한계

- 복합 능력 평가 기준은 수작업 검수에 전적으로 의존 — 자동화 미구현
- 380 seed curate 가 가장 큰 병목 (예상 1일 오전~오후) 
- 1,000~1,500 쌍 규모는 단일 GPU 2 epoch 로 ~2시간 학습 가능하나, 복합 추론에 충분한 노출인지는 실측 필요
- v3 adapter resume 경로에서 LoRA 가 어떻게 추가 업데이트 되는지 PEFT 버전 세부 동작 확인 필요 (peft 0.13.2 기준)

---

## 변경 이력

- 2026-04-23 r0: v3 훈련 중 복합 질문 약점 진단에서 파생. v3 완료 후 실측 결과를 반영해 r1 업데이트 예정.
