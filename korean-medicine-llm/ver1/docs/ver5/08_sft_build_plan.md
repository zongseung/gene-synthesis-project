# ver5 · 08. SFT 제작 실행 기획서 (Build Plan)

- **버전**: ver5 r1 (2026-04-23)
- **선행 문서**: `02_sft_design.md` (설계), `03_data_pipeline.md` (파이프라인), `04_trainer_spec.md` (trainer)
- **후속 문서**: 없음 — 본 문서 실행 결과를 `PHASE_B_BUILD_REPORT.md` 로 따로 남긴다.
- **목적**: ver5 r0 의 200쌍 SFT 설계를 **실제로 실행 가능한 수준** 으로 세분화. 지금은 _문서만 있고 코드·데이터는 아직 없다_. 본 기획서는 "무엇을, 어떤 순서로, 어떤 파일에 작성하는가" 를 확정한다.

---

## 0. 한 줄 요약

**`book_008/vol_01~23.jsonl` 에 23권 · 34,040 records 가 `trans_ko` 한국어 번역본까지 전부 있으므로 SFT answer 의 fact 는 "factsheet 주입" 이 아니라 **`trans_ko` 를 literal substring 으로 quote** 하는 방식으로 환각 위험 없이 구성할 수 있다. 특히 vol_01/seq_5 한 record 에 "선조 1596 하교·허준+보조자 5인·정유재란 중단·광해군 1610 완성·25권" 8개 atomic fact 가 모두 문장으로 담겨 있어 in_scope_basic seed 절반을 단일 record 인용으로 해결한다. 본 기획서는 (Day 1) trans_ko 고밀도 record 인덱싱 + seeds yaml (quote_span 지정형) curate, (Day 2) template 라이브러리 + `build_sft_qa.py` 옵션 A (quote 존재 assert + atomic fact check) 구현 및 120쌍 생성, (Day 3) Claude paraphrase 80쌍 + fact_diff, (Day 4) 2인 수작업 검수 및 κ, (Day 5) mini-20쌍 sanity SFT, (Day 6) 200쌍 full SFT, (Day 7) 62문항 probe + 보고서 의 7일 스케줄을 확정한다. 핵심 원칙 변경은 (a) primary source 를 factsheet → `trans_ko` literal 로 뒤집고, (b) seed 스키마 `source_records` 를 `{ref, quote_span, lang}` 로 확장해 build 가 각 quote 의 literal 존재를 강제 검증하며, (c) ChiMed 2.0 의 classical→modern 자동변환은 KIOM 번역본 존재로 **불필요**, (d) factsheet 는 hanja 병기/숫자 정규화 용도로만 남긴다는 네 가지다.**

---

## 1. 현재 상태 (2026-04-23 실측)

### 1.1 book_008 데이터 인벤토리 — 이미 다 있다

`data/raw/mediclassics_unified/book_008/` 실측:

| 항목 | 값 |
|------|----|
| 총 volume 수 | **23** (vol_01.jsonl ~ vol_23.jsonl) |
| 총 record 수 | **34,040** |
| 각 record 필드 | `book_id`, `volume_id`, `content_seq`, `content_level`, `up_path_nm`, `original` (한문), **`trans_ko` (한국어 번역)**, `trans_en` (영역), `annotation`, `index_num` |
| content_level | `AA` 편 헤더 / `OO` 서문 표지 / `XX` 저자 서명 / `ZZ` 본문 |

→ **한국어 번역본이 34,040 문장 완비**. 즉 SFT answer 의 fact 는 **factsheet 에 의존하지 않고 `trans_ko` 를 literal quote** 하는 것만으로 환각 위험 없이 채울 수 있다. ChiMed 2.0 의 "classical → modern 자동 변환" 은 **우리에겐 불필요** (이미 KIOM 번역본 존재).

**vol_01 서문의 fact 밀도 예시** (seq 5 trans_ko 한 줄):

> "병신년(1596)에 태의(太醫) 허준을 불러 하교하시기를… 허준이 물러나와 유의(儒醫) 정작(鄭碏)ㆍ태의 양예수(楊禮壽)ㆍ김응탁(金應鐸)ㆍ이명원(李命源)ㆍ정예남(鄭禮男)과 관청을 설치하고 책을 편찬하여… 정유재란을 만나 여러 의사들이 뿔뿔이 흩어져 일이 마침내 중단되었습니다. 그 후 선종대왕이 다시 허준에게 하교하여 홀로 책을 편찬하게 하시고 대궐에서 소장하고 있는 의서 오백권을 내어주어 고증하게 하셨는데… 성상(聖上)이 즉위한 지 삼년이 된 경술년(1610)에 허준이 비로소 작업을 마치고 진상하면서 《동의보감》이라고 이름을 붙였으니 모두 25권입니다."

→ 이 한 record 로 **저자 · 편찬 보조자 · 왕(선조→광해군) · 편찬 시작(1596) · 중단 사건(정유재란) · 완성 연도(1610) · 권수(25) · 책 이름 확정** 8개 atomic fact 가 동시 커버.

seq 2 (저자 서명) 는 `trans_ko` **그대로**:
> "어의 충근정량호성공신 숭록대부 양평군 허준이 하교를 받들어 짓습니다."
→ 어의·공신호·양평군·저자명 4 fact.

### 1.2 나머지 이미 있는 자원

| 자원 | 경로 | 비고 |
|------|------|------|
| factsheet (보조) | `data/facts/core_factsheet.yaml` | book_id=8 entry 확인 필요. **primary 는 아님** — trans_ko literal 이 우선 |
| real_facts_identity | `data/cpt/book008_real_facts_identity.jsonl` | vol_01 seq 2 추출본. 참고용 |
| 평가 probe 43문항 | `eval/hanmed_eval_v0/phaseA_eval_input.jsonl` | SFT 후 재사용 |
| probe_v4_final 4문항 | `eval/hanmed_eval_v0/probe_v4_final_input.jsonl` | 재측정 기준 |
| entity blacklist 목록 | `docs/ver5/02_sft_design.md §4.2` | yaml 아직 아님 (본 기획에서 yaml 화) |
| Trainer skeleton | `src/training/cpt_trainer.py` | `--mode sft` 분기는 아직 없음 |

### 1.3 데이터 소스 우선순위 (확정)

```
1순위 (primary)   : book_008/vol_NN/content_seq=M 의 trans_ko literal
                    → answer 의 fact 문장을 "따옴표 + 그대로" 로 인용
2순위 (normalize) : factsheet book_id=8 의 정규화 값
                    → 숫자·이름 병기·한자 표기 일관화 (예: "許浚" hanja 병기)
3순위 (shape)     : template body (§4.2)
                    → 인용문을 감싸는 해설 문장. LLM 자유 생성 X
금지                : 위 3개 외 어떤 출처도 answer fact 로 사용 금지
```

### 1.4 아직 없는 것 (본 기획으로 생성)

| 산출물 | 경로 | 책임 섹션 |
|--------|------|----------|
| seeds yaml | `data/sft/phaseB_qa_seeds.yaml` | §3 |
| entity whitelist yaml | `data/sft/entity_whitelist.yaml` | §4.1 |
| template 라이브러리 | `src/data/sft/templates.py` | §4.2 |
| SFT 빌더 | `scripts/build_sft_qa.py` | §4.3 |
| merge/dedup | `scripts/merge_sft_qa.py` | §4.4 |
| κ 계산 | `scripts/compute_kappa.py` | §5.3 |
| trainer `--mode sft` 분기 | `src/training/cpt_trainer.py` (확장) | `04_trainer_spec.md` 참조 |
| probe holdout 15문항 | `eval/hanmed_eval_v0/phaseB_paraphrase_holdout.jsonl` | §7 |
| 검수자 CSV prep | `scripts/prep_manual_review.py` | §5.2 |

### 1.5 리스크 유발 전제

- **factsheet 의존도 크게 낮춰짐** — trans_ko literal 이 primary 이므로 factsheet book_id=8 누락은 blocking 이 아니다. 정규화용으로만 사용.
- TRL 미설치 → `04_trainer_spec.md §1.2` 명령 선행.
- 검수자 2명 확보 실패 → §5.4 의 1인 대체 플랜 발동.
- vol_01 seq 2~9 외 **본문 (vol_02~23) 에서 medical_literature 30쌍 seed 발굴** 필요 — 한 번 훑어야 함 (Day 1 포함).

## 2. 연구 반영 (2025 업데이트)

ver5 r0 가 작성된 후 공개된/재확인된 자료를 본 기획에 어떻게 녹이는지:

| 자료 | 핵심 시사점 | 본 기획 반영 |
|------|-----------|-------------|
| LIMA (Zhou 2023 → 2025 재평가) | "knowledge 는 pretrain 에, SFT 는 style 을 surface" | 200쌍으로 **fact 를 새로 주입** 하지 않음. 이미 Base 에 존재하는 "허준·선조" 지식을 QA 매핑으로 surface 한다는 설계 관점 유지 |
| HuatuoGPT-II (Chen 2024) | knowledge-graph 샘플링 + template QA + back-translation 로 style 통일 | 우리는 KG 없이 **factsheet + 서문 인용** 기반 template 로 대체. back-translation 대신 **Claude paraphrase (옵션 B, 80쌍)** 로 style 통일 |
| ChiMed 2.0 (2025) | 고전 한문 → 현대 중국어 자동 변환으로 comprehension 향상 | **우리는 불필요** — KIOM mediclassics 가 `trans_ko` 한국어 번역을 34,040 records 전부 제공. "漢文→韓文" 자동 변환은 오히려 환각 유발 위험이 커서 금지. `original` (한문) 은 보존 인용 slot 에만 literal 노출 |
| KoMeP (2025, PMC12086433) | atomic-fact 단위 Perplexity-API 기반 RAG verifier | 우리는 외부 API 의존 피하고 `fact_diff_check` 를 **atomic-fact list 기반** 으로 강화 (§4.3.4) |
| TRL v0.11+ (2025) | chat_template 내 `{% generation %}` 로 assistant-only masking 가능 | 먼저 Bllossom tokenizer 의 chat_template 이 generation 키워드를 지원하는지 확인 (§6.1). 지원하면 사용, 아니면 `DataCollatorForCompletionOnlyLM` fallback |
| MinHashLSH dedup 가이드 | shingle k=5~7 권장 | ver5 r0 의 trigram (k=3) Jaccard 0.5 는 **k=5 Jaccard 0.4** 로 상향 (§4.3.5) |

## 3. Seeds YAML 설계 확정

### 3.0 선결 — 서문 fact-rich record 인덱싱

primary source 가 trans_ko literal 이므로, 저자·왕·연도·편 구성 같은 고밀도 서문 record 의 좌표를 먼저 고정:

| 좌표 (book_008/volNN/seqM) | content_level | up_path_nm | trans_ko 핵심 fact |
|-----|:----:|----|----|
| vol_01/seq_2 | XX | 內景篇卷之一 | 어의/충근정량호성공신/숭록대부/양평군/허준 |
| vol_01/seq_3 | OO | 內景篇卷之一 > 東醫寶鑑序 | 동의보감 서문 (표지) |
| vol_01/seq_4 | ZZ | 東醫寶鑑序 | 헌원·기백·창공·진월인·유완소·장종정·주진형·이고 (의가 역사) |
| vol_01/seq_5 | ZZ | 東醫寶鑑序 | **선조 1596 명령 → 허준+양예수+김응탁+정예남+이명원+정작 → 정유재란 중단 → 1610 완성 25권** (최고밀도) |
| vol_01/seq_6 | ZZ | 東醫寶鑑序 | 광해군 가납 + 이정구 서문 명 |
| vol_01/seq_7~9 | ZZ | 東醫寶鑑序 | 책의 구조 (내경/외형/잡병/…) + 중화위육의 다스림 |
| vol_01/seq_10 | ZZ | 東醫寶鑑序 | 1611 신해년 이정구 서문 찬 |

→ **vol_01/seq_5 하나만으로 in_scope_basic 20 seed 중 절반을 literal quote 로 커버**. 나머지는 `seq_2 / seq_6 / seq_7 / seq_10` 등에서 조합.

#### 3.0.1 실행: seed-rich record 추출 스크립트

```bash
.venv/bin/python - <<'PY'
import json, sys
seqs_to_extract = {
    1: [2, 3, 4, 5, 6, 7, 8, 9, 10],   # 서문 핵심
}
for vol, seqs in seqs_to_extract.items():
    path = f"data/raw/mediclassics_unified/book_008/vol_{vol:02d}.jsonl"
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if r["content_seq"] in seqs:
            print(f"=== vol_{vol:02d}/seq_{r['content_seq']} ({r['content_level']}) ===")
            print(f"path: {r['up_path_nm']}")
            print(f"trans_ko: {r['trans_ko'][:200]}...")
            print()
PY
```

factsheet book_id=8 확인은 **선택** (optional normalization 용).

#### 3.0.2 본문 (vol_02~23) medical_literature seed 발굴

```bash
# 예: 기침 = 해수문 찾기
grep -l "해수문\|기침\|咳嗽" data/raw/mediclassics_unified/book_008/vol_*.jsonl
# 예: 기허 찾기
grep -l "기허\|氣虛" data/raw/mediclassics_unified/book_008/vol_*.jsonl
```

→ 발견된 vol/seq 를 medical_literature seed 30개 `source_records` 로 직접 고정.

### 3.1 카테고리별 seed 수 (ver5 r0 계승)

| 카테고리 | 최종 쌍 수 | seed 수 (원본) | 확장 비율 |
|----------|:---------:|:-------------:|:--------:|
| in_scope_basic | 40 | 20 | ×2 (question_templates) |
| in_scope_long | 25 | 25 | ×1 |
| paraphrase | 30 | (in_scope_basic 재사용) | ×1.5 |
| out_of_scope | 25 | 12 | ×2 |
| safety_refusal | 50 | 25 | ×2 |
| medical_literature | 30 | 15 | ×2 |
| **합계** | **200** | **97** seeds | — |

→ seeds yaml 총 **97개 항목** 수작업 curate. 실문장 seed 97개는 2026-04-23 기준 1일 이내 작업 가능.

### 3.2 단일 seed 스키마 (확정 · trans_ko 중심)

```yaml
- id: <category>_<subcat>_<NN>           # 예: in_basic_author_01
  category: in_scope_basic               # 6개 enum
  subcat: author_fact                    # 카테고리별 세분화 key (free-form)
  question_templates:                    # 1개 이상. paraphrase 는 2~3개 권장
    - "동의보감(東醫寶鑑)을 편찬한 저자는 누구인가요?"
  key_entities: [허준, 양평군, 어의]      # answer 에 반드시 등장할 whitelist entity
  forbidden_entities: []                 # 이 seed 에서 추가 금지할 entity (optional)

  # ★ primary source — book_008/vol_NN/seq_M 으로 실문장 좌표
  source_records:
    - ref: book_008/vol_01/seq_2
      quote_span: "어의 충근정량호성공신 숭록대부 양평군 허준이 하교를 받들어 짓습니다."
      lang: ko              # ko = trans_ko literal, hanja = original literal
    - ref: book_008/vol_01/seq_5
      quote_span: "병신년(1596)에 태의(太醫) 허준을 불러 하교하시기를"
      lang: ko

  # ★ 보조 정규화 (optional) — hanja 병기, 숫자 표기 일관화만 사용
  factsheet_keys:
    - author_hanja          # "許浚" 병기용
    - reign_hanja           # "宣祖" 병기용

  template_id: author_fact                # §4.2 TEMPLATES 매칭
  target_length_tokens: 180               # Bllossom tokenizer 기준 목표
  min_length_tokens: 100                  # 이하는 reject
  refusal_tag: null                       # safety/out_of_scope 만 사용
  _notes: "필수 fact: 허준·선조·1596·1610·1613"  # curator 메모 (optional)
```

**핵심 변경** (ver5 r0 대비):
- `source_records` 가 `str` 리스트 → `{ref, quote_span, lang}` dict 리스트로 확장. `quote_span` 은 해당 record 의 `trans_ko` 또는 `original` 에서 **정확한 substring** 이어야 한다 (build 단계에서 자동 assert).
- `factsheet_keys` 는 역할 축소 — "hanja 병기 / 숫자 표기 정규화" 용으로만.
- build_sft_qa.py 가 각 quote_span 을 해당 jsonl 에서 **literal 존재 여부** 를 강제 검증 → 철자 실수·오인용 0.

### 3.3 safety_refusal seed 특이사항

```yaml
- id: safety_personal_01
  category: safety_refusal
  subcat: personal_diagnosis
  question_templates:
    - "제가 요즘 가슴이 답답하고 숨이 찹니다. 어떤 병인가요? 약도 알려주세요."
  key_entities: []
  source_records: []                      # safety 는 원문 인용 없음
  factsheet_keys: []
  template_id: safety_personal
  target_length_tokens: 180
  min_length_tokens: 150
  refusal_tag: personal_diagnosis
  paired_with: med_qi_deficiency_01       # §6 대비 쌍 링크 (optional)
  _safety_variables:                      # template slot fill
    symptom_summary: "가슴이 답답하고 숨이 차는"
    possible_causes: "심장·호흡기 관련"
    appropriate_specialist: "내과 또는 순환기내과 전문의"
    emergency_note: "증상이 심하거나 갑자기 발생한 경우 119 에 연락하십시오."
    related_literature: "기허·심·폐 관련"
    symptom_name: "기허로 인한 호흡 곤란"
```

### 3.4 paraphrase 자동 확장 규칙

`in_scope_basic` 20개 seed 중 **15개** 를 `question_templates` 가 2~3개인 것으로 작성.
- 2 templates 로 expand 시: 각 seed → 2쌍 (원본 + paraphrase_1) = 30쌍
- 3 templates 로 expand 시: 각 seed → 3쌍 중 랜덤 2쌍 선택

`build_sft_qa.py --mode template` 내부에서 자동 처리.

## 4. 빌더 파이프라인 설계

### 4.1 entity_whitelist.yaml

```yaml
# data/sft/entity_whitelist.yaml
version: r0
generated_from: docs/ver5/02_sft_design.md §4

allow:
  authors:
    - { name: 허준, hanja: 許浚, role: "동의보감 주저자" }
    - { name: 양예수, hanja: 楊禮壽, role: "편찬 보조" }
    - { name: 김응탁, hanja: 金應鐸, role: "편찬 보조" }
    - { name: 정예남, hanja: 鄭禮男, role: "편찬 보조" }
    - { name: 이정구, hanja: 李廷龜, role: "서문 찬" }
  kings:
    - { name: 선조, hanja: 宣祖 }
    - { name: 광해군, hanja: 光海君 }
  titles:
    - { name: 어의, hanja: 御醫 }
    - { name: 양평군, hanja: 陽平君 }
    - "충근정량호성공신"
    - { name: 숭록대부, hanja: 崇祿大夫 }
  classical_doctors_ref_only:
    - { name: 헌원, hanja: 軒轅, ctx: "서문 인용 黃帝" }
    - { name: 기백, hanja: 岐伯, ctx: "서문 인용" }
    - { name: 창공, hanja: 倉公, ctx: "서문 인용" }
    - { name: 진월인, hanja: 秦越人, ctx: "扁鵲, 서문 인용" }
    - { name: 유완소, hanja: 劉完素 }
    - { name: 장종정, hanja: 張從正 }
    - { name: 주진형, hanja: 朱震亨 }
    - { name: 이고, hanja: 李杲 }

deny:
  # E1+E2+E3 실측 창작 entity (저자/편찬자/창시자 위치에서만 금지)
  - { name: 이중옥기, origin: "Phase A' 창작", meaning: "존재하지 않음" }
  - { name: 이중옥, origin: "Phase A' 창작" }
  - { name: 이중경, origin: "Phase A' 창작" }
  - { name: 이수경, origin: "Base 창작" }
  - { name: 장기상, hanja: 張吉甫, origin: "Phase A' 창작" }
  - { name: 장길보, origin: "Phase A' 창작" }
  - { name: 장원소, hanja: 張元素, origin: "R1 창작", note: "실존 금원4대가 but 동의보감/사상의학 무관" }
  - { name: 장형, hanja: 張衡, origin: "Base 창작", note: "후한 천문학자, 의학자 아님" }
  - { name: 이진, hanja: 李珍, origin: "R1 창작" }
  - { name: 이시진, hanja: 李時珍, origin: "R1 창작", note: "본초강목 저자, 동의보감과 무관 — 동의보감 저자 위치 금지" }
  - { name: 이황, hanja: 李滉, origin: "Base 창작", note: "조선 유학자, 저자 위치 금지" }
  - { name: 이이, hanja: 李瀷, origin: "Base 창작", note: "조선 유학자, 저자 위치 금지" }
  - { name: 양정수, hanja: 楊挺壽, origin: "Base 창작" }
  - { name: 정유재수, origin: "Phase A' 창작", note: "정유재란 오인" }
  - { name: 송진, hanja: 宋進, origin: "Phase A' 창작" }
  - { name: "김응탁(주저자위치)", origin: "E3 실측", note: "실존 보조자인데 주저자 위치 환각 금지" }
  - { name: 강희왕 조광, origin: "E3 창작", note: "청나라 황제 오인" }
```

- **spec**: blacklist 의 `김응탁` 은 `allow.authors` 에도 있다. **컨텍스트 의존 금지** 라서 yaml 은 `김응탁(주저자위치)` 같이 `(role_constraint)` 를 명시하되, validator 는 문자열 검출 후 context heuristic 적용 (§4.3.3).

### 4.2 src/data/sft/templates.py — 고정 문장 라이브러리

설계 원칙:
1. **한문 인용** 은 template 내 `{source_quote}` slot 으로만 삽입 (build 가 source_records 에서 literal 복사)
2. fact 값 (연도·이름) 은 `{factsheet.<key>}` 치환만 허용. 자유 생성 없음
3. 각 template 출력은 **항상 `[출처: ...]` 태그로 종료**

```python
# src/data/sft/templates.py (신규)
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class Template:
    id: str
    category: str
    body: str                  # Python str.format compatible
    required_keys: List[str]   # body 에서 반드시 치환될 key
    citation_tag: str          # 말미 [출처: ...]

TEMPLATES: Dict[str, Template] = {
    "author_fact": Template(
        id="author_fact",
        category="in_scope_basic",
        # ★ 규칙: source_quote_* 는 반드시 trans_ko/original 에서 literal.
        #         build 단계에서 각 quote 가 해당 record 에 존재하는지 assert.
        body=(
            "동의보감은 조선 중기의 어의(御醫) 허준({author_hanja}) 이 편찬한 의서입니다.\n\n"
            "동의보감 서문에는 저자 서명이 다음과 같이 기록되어 있습니다:\n"
            "\"{source_quote_author_line}\" "
            "(동의보감 {up_path_nm_1} seq_{seq_1})\n\n"
            "또한 서문에는 편찬 경위가 다음과 같이 명시되어 있습니다:\n"
            "\"{source_quote_compilation}\" "
            "(동의보감 {up_path_nm_2} seq_{seq_2})\n\n"
            "이처럼 허준은 {reign_hanja_short}의 하교를 받들어 1596년(병신년) 편찬을 "
            "시작하였고, 광해군 2년(1610) 에 완성하여 1613년 내의원에서 간행·반포되었습니다."
        ),
        required_keys=[
            "author_hanja",                  # 例 "許浚"
            "source_quote_author_line",      # vol_01/seq_2 trans_ko literal
            "up_path_nm_1", "seq_1",
            "source_quote_compilation",      # vol_01/seq_5 trans_ko literal (일부 substring)
            "up_path_nm_2", "seq_2",
            "reign_hanja_short",             # 例 "선조(宣祖)"
        ],
        citation_tag="[출처: 동의보감 내경편 권1 서문 (KIOM mediclassics)]",
    ),

    "king_command": Template(...),           # 유사 구조
    "pyeon_structure": Template(...),        # 5편 구성 설명
    "compilation_period": Template(...),     # 1596~1610
    "signature_principles": Template(...),   # 3원칙 (요점/수양/향약)

    "out_of_scope_refusal": Template(
        id="out_of_scope_refusal",
        category="out_of_scope",
        body=(
            "{question_entity}(은)는 본 모델의 학습 범위(동의보감 단권)에 "
            "포함되지 않은 저작/개념이므로, 정확한 정보를 제공할 수 없습니다. "
            "본 모델은 book_008 동의보감에 대해서만 학습되었으며, "
            "{question_entity}의 상세한 내용은 한국민족문화대백과사전 또는 "
            "규장각 원문 해제 등 외부 신뢰 출처를 참조하시기 바랍니다. "
            "동의보감 관련 질문으로 바꿔 주시면 답변드릴 수 있습니다."
        ),
        required_keys=["question_entity"],
        citation_tag="",
    ),

    "safety_personal": Template(
        id="safety_personal",
        category="safety_refusal",
        body=(
            "본 모델은 한의학 고전 문헌 연구 보조 AI 로, 개인 증상에 대한 "
            "진단이나 구체적인 약물 처방을 제공할 수 없습니다.\n\n"
            "{symptom_summary} 증상은 {possible_causes} 등 여러 원인이 가능하므로, "
            "반드시 {appropriate_specialist} 진료를 받으시기 바랍니다. "
            "{emergency_note}\n\n"
            "동의보감에 {related_literature} 에 대한 설명이 있으나, 이는 "
            "17세기 문헌이며 현대 의학의 진단·처방을 대체할 수 없습니다. "
            "문헌 해설이 필요하시면 \"동의보감에서 {symptom_name}은 어떻게 "
            "설명되나요?\" 와 같이 문헌 중심으로 질문해 주십시오."
        ),
        required_keys=["symptom_summary", "possible_causes",
                       "appropriate_specialist", "emergency_note",
                       "related_literature", "symptom_name"],
        citation_tag="[전문의 상담 필수]",
    ),

    "safety_emergency": Template(...),       # 119 우선
    "medical_literature": Template(...),     # 문헌 해설

    # 총 12~15개 template 예상
}

def render(template_id: str, slots: Dict[str, str]) -> str:
    t = TEMPLATES[template_id]
    missing = [k for k in t.required_keys if k not in slots]
    if missing:
        raise ValueError(f"template {template_id}: missing keys {missing}")
    return t.body.format(**slots) + ("\n" + t.citation_tag if t.citation_tag else "")
```

### 4.3 scripts/build_sft_qa.py (핵심 신규 파일)

#### 4.3.1 argparse

```python
p = argparse.ArgumentParser()
p.add_argument("--seeds", type=Path, required=True)       # phaseB_qa_seeds.yaml
p.add_argument("--mode", choices=["template", "paraphrase"], required=True)
p.add_argument("--categories", nargs="*", default=None)   # 필터
p.add_argument("--limit", type=int, default=None)         # sanity 용
p.add_argument("--whitelist", type=Path, required=True)   # entity_whitelist.yaml
p.add_argument("--factsheet", type=Path,
               default="data/facts/core_factsheet.yaml")
p.add_argument("--raw-dir", type=Path,
               default="data/raw/mediclassics_unified/book_008")
p.add_argument("--tokenizer", type=Path,
               default="data/tokenizer/hanmed_bllossom_ext")
p.add_argument("--out", type=Path, required=True)
p.add_argument("--min-tokens", type=int, default=80)
p.add_argument("--llm", choices=["claude-3-5-sonnet", "gpt-4o-mini"],
               default="claude-3-5-sonnet")
p.add_argument("--api-key-env", default="ANTHROPIC_API_KEY")
p.add_argument("--seed", type=int, default=42)
p.add_argument("--dry-run", action="store_true")
```

#### 4.3.2 옵션 A — Template composition flow (trans_ko literal 중심)

```
load seeds.yaml
load factsheet.yaml               # hanja 병기 등 보조
load entity_whitelist.yaml
load tokenizer

# 1회만: book_008/vol_NN.jsonl 을 전부 메모리 dict 로 적재
records = {}  # key: (vol_id, seq) -> record dict with trans_ko, original, up_path_nm
for vol in range(1, 24):
    for line in open(f"{raw_dir}/vol_{vol:02d}.jsonl"):
        r = json.loads(line)
        records[(r["volume_id"], r["content_seq"])] = r

for seed in seeds[category in --categories or all]:
    # 1. source_records 의 각 quote_span 이 해당 record 에 literal 로 존재하는지 assert
    for sr in seed.source_records:
        vol_id, seq = parse_ref(sr["ref"])   # book_008/vol_01/seq_5 -> (1, 5)
        rec = records[(vol_id, seq)]
        field = rec["trans_ko"] if sr["lang"] == "ko" else rec["original"]
        assert sr["quote_span"] in field, \
            f"seed {seed.id}: quote_span not in {sr['ref']}"

    # 2. template slot 채우기
    for qt in seed.question_templates[:expansion_rule(seed)]:
        slots = compose_slots(seed, factsheet, records)
        #   source_quote_*     ← records[(vol, seq)][trans_ko or original] literal
        #   author_hanja       ← factsheet["books"][8]["author_hanja"]
        #   up_path_nm_1/2     ← records 에서 자동 조회
        answer = templates.render(seed.template_id, slots)

        # 3. validation
        v_ent = validate_entities(answer, whitelist)
        v_len = validate_length(answer, seed.min_length_tokens, tokenizer)
        v_quote = validate_quotes_in_answer(answer, seed.source_records)
        v_atomic = atomic_fact_check(answer)    # §4.3.4

        if all([v_ent.passed, v_len.passed, v_quote.passed, v_atomic.passed]):
            emit({
                "id": f"{seed.id}__q{i}",
                "category": seed.category,
                "subcat": seed.subcat,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": qt},
                    {"role": "assistant", "content": answer},
                ],
                "_source": {
                    "template_id": seed.template_id,
                    "seed_id": seed.id,
                    "source_records": [sr["ref"] for sr in seed.source_records],
                    "factsheet_keys": seed.factsheet_keys,
                    "_answer_tokens": v_len.tokens,
                },
            })
        else:
            log_rejection(seed, v_ent, v_len, v_quote, v_atomic)
```

**강제 사항**:
1. `source_records[i].quote_span` 이 해당 record 에 substring 으로 **존재하지 않으면 hard fail**. seed curator 의 오타 · 오기억 차단.
2. template 이 렌더한 answer 내부에도 **quote_span 이 그대로 포함** 되어야 한다 (`validate_quotes_in_answer`). template 의 `{source_quote_*}` slot 이 fact 를 손실 없이 실었는지 확인.
3. atomic_fact_check (§4.3.4) 가 연도·권수·편명에 대해 독립 검증.

#### 4.3.3 validate_entities (context-aware)

ver5 r0 의 pure set membership 을 **context-sensitive** 로 확장:

```python
AUTHOR_CONTEXT_PATTERNS = [
    r"(?P<name>[가-힣]{2,4})(?:이|가|은|는|께서)?\s*(?:편찬|저술|지었|쓰|편집|편저)",
    r"저자(?:는|가)?\s*(?P<name>[가-힣]{2,4})",
    r"편찬자(?:는|가)?\s*(?P<name>[가-힣]{2,4})",
    # 한자 병기 포함
    r"(?P<name>[一-龥]{2,4})\s*\(?[가-힣]{0,4}\)?\s*(?:편찬|저술)",
]

def validate_entities(answer: str, whitelist: Whitelist) -> Result:
    # 1. 문자열 literal 매치 (무조건 fail)
    for deny in whitelist.deny_strict:
        if deny.name in answer:
            return fail("deny_strict_literal", deny.name)

    # 2. context-dependent: "김응탁(주저자위치)" → 저자 context 에서만 fail
    for author_pat in AUTHOR_CONTEXT_PATTERNS:
        for m in re.finditer(author_pat, answer):
            candidate = m.group("name")
            if candidate in whitelist.deny_in_author_role:
                return fail("deny_in_author_role", candidate)
            if candidate not in whitelist.allow_authors_set:
                return warn("unknown_author_candidate", candidate)

    # 3. classical_doctors_ref_only 는 저자 context 에서만 금지
    # ...

    return pass_()
```

#### 4.3.4 atomic fact check (KoMeP 2025 반영)

옵션 B paraphrase 후 의무 실행:

```python
ATOMIC_FACTS_BOOK008 = {
    "author_ko": "허준",
    "author_hanja": "許浚",
    "reign": "선조",
    "compiled_year": "1596",
    "published_year": "1613",
    "publish_complete_year": "1610",
    "pyeon_count": "5",
    "pyeon_names": ["내경편", "외형편", "잡병편", "탕액편", "침구편"],
}

def atomic_fact_check(answer: str) -> dict:
    violations = []
    # (a) 허준 등장했는데 저자 hanja 가 다른 한자면 fail
    if "허준" in answer:
        hanja_hits = re.findall(r"許[^\s]{0,3}", answer)
        if hanja_hits and not any("許浚" in h for h in hanja_hits):
            violations.append(("author_hanja_mismatch", hanja_hits))
    # (b) 편찬 완료 연도가 1610 이외 값으로 등장?
    years = re.findall(r"\b(1[5-9]\d{2})\b", answer)
    bad_years = [y for y in years
                 if y not in {"1596", "1597", "1610", "1613", "1615"}]
    if bad_years:
        violations.append(("unknown_year", bad_years))
    # (c) pyeon 수가 "5" 가 아닌 값으로 등장?
    for bad in ["4편", "6편", "7편", "3편"]:
        if bad in answer:
            violations.append(("wrong_pyeon_count", bad))
    return {"passed": not violations, "violations": violations}
```

#### 4.3.5 near-duplicate filter (k=5 shingle + Jaccard 0.4)

MinHashLSH 가이드(2025 dedup 가이드) 의 **k=5~7 권장** 을 반영해 k=3 → k=5 로 상향, 임계 0.5 → 0.4 로 엄격화:

```python
def shingles(text: str, k: int = 5) -> set[str]:
    text = re.sub(r"\s+", "", text)
    return {text[i:i+k] for i in range(len(text) - k + 1)}

def jaccard(a: set, b: set) -> float:
    return len(a & b) / max(len(a | b), 1)

def is_near_duplicate(candidate: str, existing: list[set], threshold: float = 0.4):
    c_sh = shingles(candidate)
    for e_sh in existing:
        if jaccard(c_sh, e_sh) >= threshold:
            return True
    return False
```

200쌍 규모에서 O(N²) 비교는 40,000 쌍 비교 = 수초 내 완료. MinHashLSH 는 Phase C 확장 시 도입.

### 4.4 scripts/merge_sft_qa.py

```
load phaseB_qa_template_v1.jsonl (120)
load phaseB_qa_paraphrase_v1.jsonl (80)
merge → 200
run near-duplicate filter (k=5, thr 0.4)
  intra-group (paraphrase ↔ template 서로 기본 fact 공유해서 overlap 예상)
  → paraphrase 중 제거, 부족 시 옵션 A 추가 seed 에서 보충
run final atomic_fact_check on all 200
shuffle (seed=42)
save phaseB_qa_merged.jsonl
save phaseB_qa_validation_report.json
```

## 5. 수작업 검수 (Cohen κ)

### 5.1 prep_manual_review.py

```
load phaseB_qa_merged.jsonl
for each example:
    row = {
        "id": ex["id"],
        "category": ex["category"],
        "user": ex["messages"][1]["content"],
        "assistant": ex["messages"][2]["content"],
        "label": "",      # accept / partial_revise / reject
        "reason": "",     # free-text
        "revised_assistant": "",  # partial_revise 시 사용
    }
write data/review/phaseB_reviewer_A.csv
write data/review/phaseB_reviewer_B.csv   # 동일 순서 (앵커 공정성)
```

### 5.2 검수 기준 (라벨러 워크샵용 1 pager)

```
accept         : 사실 정확 + 길이 기준 충족 + 문체 자연스러움
partial_revise : 사실 정확하나 수정 필요 (길이 부족, paraphrase 표현 어색, …)
                 → revised_assistant 필수
reject         : 사실 오류 / entity 환각 / safety 위반 / 길이 극단
```

→ `accept = 1`, `partial_revise = 2`, `reject = 3` 로 인코딩 후 κ 계산.

### 5.3 compute_kappa.py

```python
from sklearn.metrics import cohen_kappa_score
import pandas as pd
a = pd.read_csv(args.a)["label"]
b = pd.read_csv(args.b)["label"]
k = cohen_kappa_score(a, b)
# 목표 ≥ 0.8
# disagreement 세부 쌍 덤프
disagree = (a != b)
```

### 5.4 검수자 1명 축소 플랜 (fallback)

- κ 측정 불가 명시 (reviewers_count=1)
- 자동 지표로 **interval 검수** 수행: 50쌍마다 샘플링 5쌍 자체 재검수 → consistency rate 로 κ 대체
- 최종 보고서에 "단일 검수자 편향 존재 가능" 명시

## 6. TRL integration 상세 (ver5 r0 04 보완)

### 6.1 Bllossom chat_template 의 {% generation %} 지원 확인

TRL 최신(v0.11+) 은 chat_template 에 `{% generation %} … {% endgeneration %}` 블록이 있으면 자동으로 그 구간만 label 로 간주. Bllossom 기본 template 은 Llama-3 계열이라 **generation 키워드가 없을** 가능성 높음 → 선 확인.

```bash
.venv/bin/python - <<'PY'
from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained("data/tokenizer/hanmed_bllossom_ext",
                                   trust_remote_code=True)
tmpl = t.chat_template or ""
print("has_generation_tag:", "{% generation %}" in tmpl)
print("has_endgeneration_tag:", "{% endgeneration %}" in tmpl)
print("--- template ---")
print(tmpl[:600])
PY
```

- `has_generation_tag == True` → TRL 최신 경로 (`assistant_only_loss=True`) 사용
- `has_generation_tag == False` → **`DataCollatorForCompletionOnlyLM` fallback** (`04_trainer_spec.md §2.3`)

### 6.2 fallback 인 경우 response_template ids 확정

```bash
.venv/bin/python - <<'PY'
from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained("data/tokenizer/hanmed_bllossom_ext",
                                   trust_remote_code=True)
tmpl = "<|start_header_id|>assistant<|end_header_id|>\n\n"
ids = t(tmpl, add_special_tokens=False).input_ids
print("response_template_ids =", ids)
# 예상: [128006, 78191, 128007, 271]  (Llama-3 계열)
PY
```

→ 결과를 `cpt_trainer.py` 의 `run_sft()` 에 상수로 하드코딩.

### 6.3 packing=False, max_seq_length=2048

200쌍 × max 500 tok answer + user + system ≈ 800 tok 정도. 2048 여유. packing 금지는 sample boundary 보존 위함.

## 7. Holdout 15문항 설계

### 7.1 생성 원칙

- SFT seeds yaml 의 `question_templates` 어느 것과도 trigram Jaccard ≤ 0.3
- expected fact 는 SFT 학습 fact 와 동일 (overfit 검증)
- 형식: `{id, category=holdout_paraphrase, question, expected}` (existing probe 와 동일 JSONL schema)

### 7.2 15문항 스켈레톤

| HO-ID | 주제 | 질문 예시 | expected |
|-------|------|----------|----------|
| HO-01 | author | "『동의보감』을 누구에게 지어달라 했는지 알려주세요" | 허준 |
| HO-02 | pyeon | "동의보감이 어떤 편들로 나뉘는지 알려줄 수 있나요?" | 5편 — 내경·외형·잡병·탕액·침구 |
| HO-03 | period | "동의보감 편찬은 몇 년에 시작해서 몇 년에 끝났나요" | 1596 명령, 1610 완성 |
| HO-04 | reign | "어느 임금 때 편찬이 시작되었는지 알려주세요" | 선조 |
| HO-05 | publish | "동의보감은 언제 책으로 나왔나요?" | 1613 |
| HO-06 | country | "동의보감은 어느 나라에서 만들어진 의서인가요?" | 조선 |
| HO-07 | book_type | "동의보감은 어떤 종류의 책으로 분류되나요?" | 의서(종합 의서) |
| HO-08 | title_mean | "'동의보감' 이라는 제목은 무슨 의미인가요?" | 동의(동쪽 의학)의 보배로운 거울 |
| HO-09 | helpers | "허준 혼자 집필한 건가요? 함께 한 사람은 없나요?" | 양예수·김응탁·정예남·이정구 |
| HO-10 | num_volumes | "동의보감은 총 몇 권으로 구성되어 있나요?" | 25권 (편 5, 권 총 25) |
| HO-11 | three_principles | "허준이 책을 쓸 때 선조가 강조한 원칙은?" | 요점/수양 우선/향약 활용 |
| HO-12 | yangpyeong | "허준의 작위는 무엇이었나요?" | 양평군 |
| HO-13 | language | "동의보감은 원래 어떤 글자로 쓰였나요?" | 한문(漢文) |
| HO-14 | kiom | "현재 동의보감을 연구하는 국내 기관은 어디인가요?" | 한국한의학연구원(KIOM) |
| HO-15 | unesco | "동의보감은 유네스코에서 어떤 지정을 받았나요?" | 세계기록유산(2009) |

→ `eval/hanmed_eval_v0/phaseB_paraphrase_holdout.jsonl` 로 저장.

## 8. 7일 실행 스케줄

| Day | 작업 | 산출물 | 책임 섹션 |
|:---:|------|--------|---------|
| **1** | §3.0.1 vol_01 서문 9 seq 덤프 + §3.0.2 본문 medical_literature grep | stdout 로그 → seed curate 가이드 | §3.0 |
| **1** | seeds yaml 97 entries curate (trans_ko literal quote_span 포함) | `data/sft/phaseB_qa_seeds.yaml` | §3 |
| **1** | entity_whitelist yaml 작성 | `data/sft/entity_whitelist.yaml` | §4.1 |
| **2** | `src/data/sft/templates.py` 12~15 template 작성 (slot 이름 고정) | 템플릿 라이브러리 | §4.2 |
| **2** | `scripts/build_sft_qa.py` 옵션 A 구현 (record 메모리 로드 + quote assert) + 120쌍 생성 | `phaseB_qa_template_v1.jsonl` | §4.3 |
| **3** | 옵션 B Claude paraphrase 80쌍 + fact_diff_check + atomic_fact_check | `phaseB_qa_paraphrase_v1.jsonl` | §4.3.4 |
| **3** | `scripts/merge_sft_qa.py` → 200쌍 merge + dedup | `phaseB_qa_merged.jsonl` | §4.4 |
| **4** | 2인 수작업 검수 + κ 측정 | `phaseB_qa_reviewed.jsonl`, `phaseB_review_summary.md` | §5 |
| **4** | holdout 15문항 yaml 작성 | `phaseB_paraphrase_holdout.jsonl` | §7 |
| **5** | `cpt_trainer.py --mode sft` 확장 + TRL 호환 확인 + mini sanity (20쌍) | `outputs/sft_sanity_20/` | `04_trainer_spec.md` §6 |
| **6** | **Full SFT 200쌍** (3 epoch, 1 GPU) | `outputs/cpt_bllossom_ver5/adapter/` | §6 |
| **7** | probe 62문항 + 수작업 검수 + Phase B 보고서 | `outputs/probes/V2_eval.jsonl`, `PHASE_B_BUILD_REPORT.md` | `05_evaluation.md` |

총 **7일** (ver5 r0 README 의 5일 추정보다 +2일 — 검수 + TRL 호환 현실 반영).

## 9. 실패 시 gate / 재진입

```
Day 2 gate: 옵션 A 120쌍 중 validation 통과 < 80 (67%)
  → template required_keys 혹은 factsheet 누락 점검 → seeds yaml 재작성

Day 3 gate: 옵션 B paraphrase fact_diff_check reject > 25 (31%)
  → Claude prompt 강화 (금지 entity 명시), GPT-4o-mini fallback

Day 4 gate: Cohen κ < 0.6
  → 검수 기준 재정의 workshop, 3인 추가 검수

Day 5 gate: mini-20쌍 sanity eval_loss NaN 또는 response_template mismatch
  → TRL fallback (§6.2) 또는 generation-tag 경로 재검토

Day 6 gate: 학습 중 OOM / NCCL hang
  → single GPU + micro_bs 1 + grad_accum 16

Day 7 gate: in_scope 수작업 correct < 60%
  → seeds 재커레이트 + 300쌍 재학습 (ver5 r0 §7.1 mitigation)
  → 동시에 resume + B2 ablation (V4) 로 baseline 확인
```

## 10. Experiments 디렉토리 수정 사항 확정

`experiments/dongui_bogam/` 은 현재 symlink farm. 본 기획 관련 추가:

| 대상 | 변경 |
|------|------|
| `experiments/dongui_bogam/scripts/build_sft_qa.py` | 신규 symlink → `../../../scripts/build_sft_qa.py` |
| `experiments/dongui_bogam/scripts/merge_sft_qa.py` | 신규 symlink |
| `experiments/dongui_bogam/scripts/compute_kappa.py` | 신규 symlink |
| `experiments/dongui_bogam/scripts/prep_manual_review.py` | 신규 symlink |
| `experiments/dongui_bogam/src/data/sft/` | 신규 디렉토리 (templates.py) |
| `experiments/dongui_bogam/docs/ver5/` | 이미 존재 — 본 문서 자동 가시 |
| `experiments/dongui_bogam/eval/phaseB_paraphrase_holdout.jsonl` | 신규 (실파일, symlink 아님) |

→ **experiments 는 읽기 전용 symlink 중심** 이므로 실제 코드·데이터는 korean-medicine-llm 본체에 작성 후 link.

## 11. 열린 질문 (사전 결정 필요)

1. **검수자 2명 확보 가능한가?** 불가 시 §5.4 대체.
2. **Claude 3.5 Sonnet API key 사용 가능한가?** 불가 시 Qwen2.5-7B 로컬로 paraphrase — 품질 하락 감수.
3. ~~**book_id=8 entry 가 factsheet 에 이미 확정?**~~ → **해결됨**. trans_ko literal 이 primary, factsheet 는 hanja 병기 등 정규화 용도로만 쓰므로 blocking 아님.
4. **Bllossom chat_template 이 {% generation %} 포함?** §6.1 확인 후 코드 분기 결정.
5. **200쌍 학습 예산은 1 GPU × 2h 로 충분한가?** `04_trainer_spec.md §4.1` 32 steps 추정 근거 — 실측 필요.
6. **medical_literature 30쌍 seed 의 `source_records` 확보** — vol_02~23 에서 해수문·기허·풍·허로 등 대표 병증 record 를 Day 1 오전에 grep 해 실제 좌표를 고정. 못 찾으면 해당 쌍은 in_scope 로 재분류.

## 12. 본 기획서의 자기 한계

- ver5 r0 02/03 에서 중복되는 부분은 의도적 ("한 번 더 읽어도 이해 되도록"). 디자인 변경은 없고 실행 순서·책임 파일을 **확정** 하는 것이 본 문서의 가치.
- template 개수 "12~15" 는 상한/하한 — 구체 숫자는 seeds yaml 작성 후 확정.
- 최신 TRL minor version 변동에 취약. Day 5 mini sanity 실패 시 fallback plan 있음.
- Claude paraphrase 비용 $0.5 미만 추정 — 실제 청구 시 청구서 기록 필요.

---

## 변경 이력

- 2026-04-23 r1: 본 문서 초안 (ver5 r0 대비 실행 가능성 강화). 2025 연구 반영 + 현상태 gap + 7일 스케줄.
- 2026-04-23 r1.1: book_008 데이터 인벤토리 실측 반영 (23 vols × 34,040 records × trans_ko 완비). primary source 를 factsheet → trans_ko literal 로 확정. seed 스키마 `source_records` 에 `quote_span` 필드 추가. template `author_fact` 예시를 literal-quote 기반으로 재작성. ChiMed 2.0 자동변환 불필요 결정. §3.0 을 "factsheet 확정" 에서 "서문 record 좌표 고정" 으로 교체.
