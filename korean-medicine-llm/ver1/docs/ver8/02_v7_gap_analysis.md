# v7 SFT Corpus vs. 동의보감 book_008 — 전수 Gap 분석

작성일: 2026-04-24
작성자: hanmed_cpt/round_3 Agent 2
대상: `experiments/dongui_bogam/data/sft/phaseB_qa_v7_corpus.jsonl` (17,733 rows)
원본: `data/raw/mediclassics_unified/book_008/vol_{01..23}.jsonl` (34,040 records)
분석 코드: `/tmp/v7_gap_analysis.py`, `/tmp/v7_forensic_v2.py` (실측 Python, 전수 스캔)

---

## 1. 요약

v7 17,733 rows 는 raw 34,040 레코드 중 **이름 엔티티 기준 99%+** 를 커버(처방·경혈·증론 100%, 약재 92~95%) 하지만, **정보 방향 (direction) 기준으로는 단방향 편향** 이 극단적이다. 세 가지 핵심 gap —

1. **처방명 → 조성(구성 약재/용량) 직접 QA = 전체 5,978 row 중 2건 (0.03%)**.
   builder 는 `{효능·주치}` 질문 템플릿 6종만 정의하고, 답변에는 children SS 본문을 그대로 붙이므로 "돈/냥/푼" 용량 키워드는 67.3% assistant 에 우연히 포함되나, **"X 처방의 구성 약재는?" 같은 명시적 composition-inverse 질문 신호는 0에 가까움**.
2. **경혈→경맥 역매핑(INV)은 250 row로 MAJOR 15개 혈 편중 — 15대 외 DK 374 혈은 1 row 밖에 없음**. 합곡 31 / 태충 26 / 족삼리 22 rows 는 표면적으로 존재하지만, *Q 와 A 모두 같은 템플릿으로 반복* 되어 실제 다양성 낮음.
3. **증상→경혈, 증상→처방 역매핑이 builder 에서 0**. DD 1,103 증상은 "증상→해설(병리)" 한 방향만, `symptom` 카테고리 어느 답변도 "~에는 X 처방/혈을 써야" 형태의 formula/acupoint 이름을 우선 배치하지 않음. `symptom→rx` 관련 rows 는 대부분 `prescription` 카테고리의 template 5 (`"한 환자가 '{context}' 을 호소할 때 … '{name}'"`, 968 row) 에 묶여 있고, context 가 **처방 부모 path 의 마지막 token 을 기계적으로 사용** (`"단방" 을 호소할 때 `총백`"` 같은 비의미 pair 다수 발생).

즉 v7 은 이름→본문 lookup 사전에 매우 가깝고, 실제 임상 reasoning 방향 — 증상/증후/체질 → 처방/혈·약 — 은 학습 신호가 거의 주어지지 않았다.

---

## 2. 레코드-레벨 커버리지 (content_level 별)

raw book_008 의 content_level 분포 (총 34,040):

| content_level | 의미 | raw records |
|---|---|---:|
| SS | 본문 (standard section) | 11,498 |
| ZZ | 본문 (보완) | 11,241 |
| DP | 처방명 (prescription main) | 5,273 |
| CC | 중주제 (chapter sub-topic) | 2,045 |
| DD | 증론 (disease/symptom) | 1,103 |
| EP | 처방 (extra prescription) | 766 |
| DH | 약재 (herb main) | 704 |
| CH | 약재 (sub-class) | 699 |
| DK | 경혈 (acupoint) | 396 |
| BB | 대문 (chapter) | 109 |
| XX | 도해·부 설명 | 102 |
| TT | 운기표 등 표 | 26 |
| Z2 | 총목·목차 | 25 |
| AA | 편명 최상위 | 23 |
| PP | 인체 부위 목록 | 19 |
| OO | 서문 | 5 |
| EE | 부인 항목 부록 | 5 |
| CP | 一方 (특수) | 1 |

### 2.1 엔티티 이름 (first-token han + ko) 커버리지

각 level 별 unique 엔티티가 v7 의 (a) question 또는 (b) assistant 또는 (c) up_path_nm 에 한 번이라도 등장하는지 전수 substring 검사:

| level | raw recs | unique ents | Q hits | A hits | up_path hits | all_missing | 커버리지 |
|---|---:|---:|---:|---:|---:|---:|---:|
| DP | 5,273 | 3,820 | 3,820 | 3,820 | 3,770 | 0 | 100.0% |
| EP | 766 | 728 | 728 | 728 | 725 | 0 | 100.0% |
| CH | 699 | 699 | 615 | 641 | 608 | 58 | 91.7% |
| DH | 704 | 449 | 420 | 425 | 411 | 24 | 94.7% |
| DK | 396 | 389 | 389 | 389 | 388 | 0 | 100.0% |
| DD | 1,103 | 1,005 | 1,005 | 1,003 | 751 | 0 | 100.0% |
| CC | 2,045 | 1,821 | 1,803 | 1,775 | 748 | 15 | 99.2% |
| BB | 109 | 106 | 106 | 106 | 104 | 0 | 100.0% |
| AA | 23 | 23 | 23 | 23 | 23 | 0 | 100.0% |
| OO | 5 | 5 | 5 | 5 | 5 | 0 | 100.0% |
| Z2 | 25 | 25 | 24 | 24 | 24 | 1 | 96.0% |
| XX | 102 | 57 | 56 | 56 | 0 | 1 | 98.2% |

### 2.2 한자 vs 한국어 이름 (Q 에 등장한 엔티티 수)

| level | 한자 han in Q | 한국어 ko in Q | 차이 |
|---|---:|---:|---:|
| DP | 2,696 | 3,819 | 한국어 dominant (한자 단독 등장 71%) |
| EP | 500 | 728 | 동일 경향 |
| CH | 495 | 584 | — |
| DH | 211 | 420 | 한자 mention 50% 미만 |
| DK | 355 | 389 | 거의 parity |
| DD | 825 | 1,005 | — |

즉 한자 이름은 Q 에서 상대적으로 적게 노출 (Bllossom 토크나이저가 한자를 byte-fallback 처리하는 문제와 결합해 한자-only 질의 시 성능 저하). 특히 **XX 레벨(도해) 은 한자 mention = 0** — builder `build_diagram_pairs` 가 up_path 의 한국어 leaf 만 사용하기 때문.

### 2.3 실제 "누락" 레코드

`all_missing = 0` 이라는 것은 han/ko 중 하나라도 v7 어딘가 등장한다는 매우 약한 조건. **엄격 기준 (direct-subject QA 개수 기준)은 §4, §6 에 별도 제시.**

주목:
- **CH 58건 완전 누락** — tokenizer 분리 후 첫 토큰이 "白", "紅", "靑" 같은 1자 색깔 한자이거나 한국어 first token 이 "양" 같은 조사와 충돌. `build_sft_full_corpus.py:397 GENERIC_PREFIX_BLOCKLIST` 에 의한 first-wins aliasing 차단 부작용.
- **DH 24건 누락** — 복합 약재명 (예 `白鵝肉`, 분류체계 상 sub-entry) 의 한국어 alias 가 build 단계에서 mapping 실패.
- **CC 15건 누락** — 중복 이름 (`"風"`, `"痰"` 등 단일 한자) 은 gate 2 에 의해 mapping 제외되어 Q 에 오르지 못함.

---

## 3. QA 포맷 클러스터링 (category × question prefix 40자)

v7 17,733 rows 카테고리 분포:

| category | n | subcat 분리 |
|---|---:|---|
| passage | 7,000 | SS 3,695 / ZZ 3,305 |
| prescription | 5,978 | DP 5,228 / EP 750 |
| structure | 1,932 | CC 1,883 / AA 23 / Z2 17 / BB 5 / OO 4 |
| symptom | 1,103 | DD 1,103 |
| herb | 650 | DH 584 / CH 66 |
| acupoint | 646 | DK 396 / INV 250 |
| mun_inscope | 146 | IN 146 (v7 augment A6) |
| refusal_oos | 100 | OOS 100 (v7 augment A3) |
| refusal_safety | 100 | SFY 100 (v7 augment A3) |
| paraphrase | 75 | basic_fact 75 |
| diagram | 3 | XX 3 |

### 3.1 Top-15 question-prefix clusters (전체 60 중)

| rank | category | 건수 | question 접두 40자 |
|---:|---|---:|---|
| 1 | passage | **1,400** | `이 대목은 동의보감 어떤 주제에 대한 설명인지, 요지를 짚어 주세요.\n발` |
| 2 | passage | 20 | `다음 동의보감 본문 대목의 핵심 의미를 설명해 주세요.\n발췌: 성질이 평` |
| 3 | passage | 19 | `다음 동의보감 본문 대목의 핵심 의미를 설명해 주세요.\n발췌: 《내경》에` |
| 4 | passage | 18 | `다음 동의보감 본문 대목의 핵심 의미를 설명해 주세요.\n발췌: 성질이 차` |
| 5 | structure | 16 | `동의보감 편성 상 침구법(鍼灸法) 이 놓이는 자리와 역할을 정리해 주세요` |
| 6 | passage | 16 | `다음 동의보감 본문 대목의 핵심 의미를 설명해 주세요.\n발췌: 여러 가지` |
| 7 | passage | 13 | `다음 동의보감 본문 대목의 핵심 의미를 설명해 주세요.\n발췌: 성질이 약` |
| 8 | passage | 12 | `다음 동의보감 본문 대목의 핵심 의미를 설명해 주세요.\n발췌: 성질이 따` |
| 9 | structure | 12 | `맥법(脉法) 의 위치를 동의보감 편·장 계층에서 설명해 주세요.` |
| 10 | structure | 11 | `동의보감 침구법(鍼灸法) 은 어떤 구조 단위이며 상위 · 하위에 어떤 내` |

Passage cluster #1 이 1,400 rows (전체의 7.9%) 로 하나의 쿼리 prefix 가 지배 — **원칙 3 (unique_prefix_ratio ≥ 3%)에 대한 보더라인**. 이 밀도로 인해 probe Q3 `"이 대목은 동의보감 어떤 주제에 대한 설명인지"` 같은 질문에 passage 스타일 답변이 관성적으로 생성됨.

### 3.2 매핑 방향별 실측 row 수

| QA 템플릿 실체 | 실측 rows | 템플릿 위치 |
|---|---:|---|
| 처방명 direct QA: `"'X' 처방의 효능과 주치"` | 1,000 | `PRESCRIPTION_Q_TEMPLATES[0]` (build_sft_full_corpus.py:76) |
| 처방명 direct QA: `"'X' 은 어느 편에 나오는 처방"` | 1,011 | [1] (L77) |
| 처방명 direct QA: `"'X' 처방의 출전과 활용"` | 971 | [2] (L78) |
| 처방명 direct QA: `"편에 나오는 처방 'X' 의 역할"` | 1,027 | [3] (L79) |
| 증상→처방: `"한 환자가 '{context}' 을 호소할 때 … 'X'"` | **968** | [4] (L80) |
| 처방명 direct QA: `"'X' 은 동의보감 대표 처방 중 하나"` | 1,001 | [5] (L81) |
| **처방명 → 조성/구성 QA** (`"구성"` OR `"조성"` + `"처방"`) | **2** | builder 없음 (passage 에 우연 등장) |
| 경혈 inverse: `"'X' 은(는) 어느 경맥에 속"` / INV template A | 99 (Q 문자열) / 250 (subcat=INV) | augment_sft_v7.py:685 `template_a_q` |
| 경혈 direct: `"경혈 'X' 의 위치와 주치"` | 72 | `ACUPOINT_Q_TEMPLATES[0]` (L111) |
| 경혈 direct: `"X 혈의 출전과 임상 적용"` | 90 | [4] (L115) |
| 약재 direct: `"'X' 의 성질·주치"` / `"성질과 주치"` | 0 Q (`"성미"` 0) | `HERB_Q_TEMPLATES[2]` (L96) 실제로는 `성질과 주치` 로 템플릿화됨 |
| 약재 direct 전체 (any of 5 template) | 520 | 650 herb row 중 520 매치 |
| 증상 → 처방 (Q 에 `증상` AND `처방`) | 1,029 | 사실상 위 `호소할 때` 및 `한 환자가` 항목 합산 |
| 증상 → 경혈 (symptom → acupoint 이름 답변) | 112 (매우 느슨한 추정) | 없음 (mun_inscope 에 간접 경혈 등장) |

---

## 4. 정보 손실 케이스 포렌식 (10 + 10 + 10)

forensic 상세는 `/tmp/v7_forensic_v2.json`. subject-QA count 는 `category` 와 이름 literal 이 question 에 등장하는 row 수.

### 4.1 처방 (DP) 샘플 10

| # | 처방명 | raw_body_len | v7 direct-subject Q | 손실 정보 (raw 에는 있으나 v7 answer 에 없거나 shell) |
|---:|---|---:|---:|---|
| 1 | 가미온담탕 (加味溫膽湯) | 197 | 4 | raw 에는 반하 3.5돈, 진피 2.2돈, 죽여·지실 각 1.5돈… 11개 약재 정량 정확히 명시. v7 첫 답변은 **다른 유사 처방(향부자 2.4돈 등)의 데이터를 그대로 인용** — children 경로 mismatch 로 엉뚱한 body 주입. |
| 2 | 천문동 (天門冬) — DP 단방 | 81 | 68 | 본문 1문장 정보 (혼백 안정·건망 치료 + 2돈 용법) 는 반영, 그러나 68 row 중 다수는 타 처방 (태평환 등) answer 가 `천문동` 키워드를 포함해 우연히 매칭 — 즉 "천문동 약재" 정체성은 탕액편 CH 와 DP 단방 2중 등록된 상태. |
| 3 | 청대산 (靑黛散) | 130 | 3 | raw: 황련·황백 각 3돈, 청대·마아초·주사 각 6푼 … 6개 약재. v7 answer 포함하지만 closing 이 `구체 조성 · 용량은 본문을 참고해 주세요` shell — 본문에 이미 조성이 있는데 shell 로 덮음. |
| 4 | 익비환 (益脾丸) | 155 | 2 | 갈화 2냥·소두화·초두구 각 1냥 등 명시. v7 answer 본문 그대로 포함. ✅ 유지. |
| 5 | 축비음 (縮脾飮) | 129 | 3 | 사인 1.5돈·초과·오매육 각 1돈 … 명시. answer 유지. |
| 6 | 포황초 (蒲黃草) | 85 | 2 | raw 정보 그대로 복원. ✅ |
| 7 | 산약 (山藥) — DP 단방 | 74 | 87 | 87 중 첫 답변이 `한산약(寒疝藥)` — **이름 충돌로 인한 wrong-entity 우선 매칭**. `산약` literal 이 들어있는 모든 row 가 모임. direct-subject QA 는 사실상 1~2건에 불과. |
| 8 | 첨과자 (甛瓜子) — DP 단방 | 91 | 2 | raw 정보 반영 ✅ |
| 9 | 지마유 (脂麻油) — DP 단방 | 202 | 8 | raw 의 '서문백 머리카락 토함' 에피소드 등 긴 설화까지 있으나 v7 answer 는 다른 경로 (내경편 권3) 의 `지마유` children 200자 요약으로 치환 — **한 처방명이 여러 편·경로에 중복 등장 시 builder 가 첫 경로 body 를 덮어씀**. |
| 10 | 견우자 (牽牛子) — DP 단방 | 81 | 19 | 19 중 첫 매칭이 `잡병편 권6 > 부종` 본문 — direct-subject 가 아님. 실제 `내경편 권4 > 소변 > 단방 > 牽牛子` 은 1 row 만 (passage 로 등록). |

**처방 10건 소견**: 10 중 **8건** 은 본문(dosage 포함)을 answer 에 복사하나, 2건은 shell 로 마무리. 더 심각한 구조 문제는 **중복 이름 collision** — 같은 `산약` / `견우자` / `지마유` 가 탕액편 CH + 여러 단방 DP 로 최대 20+ 경로에 등장, 각각 children body 가 다른데 builder 는 하나만 씀. 따라서 `"산약 이란?"` 질문에 항상 동일 방향의 답변만 학습.

### 4.2 약재 (CH) 샘플 10

| # | 약재명 | raw_body_len | v7 subject | 손실 정보 |
|---:|---|---:|---:|---|
| 1 | 수중석자 (水中石子) | 98 | 1 | raw 정보 유지 ✅ |
| 2 | 지모 (知母) | 495 | 138 | **치명적**: raw 에 성미 (차고 평이·쓰고 달다·독 없음)·주치 (골증노열·신기허손·소갈·학질·황달)·식물학적 기술·채취법·주의 (신장 허함에 금) 까지 495자 있으나, v7 첫 매칭은 가감사백산 처방 답변 (지모 1돈 등 포함) — **약재 자체의 성미·귀경 정보는 138 rows 중 몇 개뿐, 대부분은 지모가 약재로 들어간 타 처방 body**. 약재→성미 학습 신호 희박. |
| 3 | 우리자 (牛李子) | 307 | 3 | raw: 성질·맛·주치·식물 기술·열매 묘사·대체명 (서리자). v7 passage 에는 다른 `용규` 본문에 우리자가 비유로 나옴. Subject-QA 는 1건 실제. |
| 4 | 박초 (朴硝) | 401 | 25 | raw: 성질 아주 차고 쓰고 짬, 오장병·적취 설명 + 제조법 `박초 → 초석 → 1회 제련`. v7 에서는 평위산 관련 처방 answer (`박초 5돈` 용법). **약재 정체성 보다 처방 내 활용 사례 위주**. |
| 5 | 박로 (博勞) | 74 | 1 | 한 줄 raw 그대로 반영 ✅ |
| 6 | 육천기 (六天氣) | 474 | 2 | 모두 passage 로 흡수, direct herb QA 없음 — **긴 본문은 passage 에 smoosh 되어 약재 entity 정체가 분산**. |
| 7 | 호박 (琥珀) | 323 | 47 | 47 중 호박정지환 등 호박 함유 처방이 대부분. 약재 자체 성미·오림 치료는 몇 개만. |
| 8 | 석회 (石灰) | 425 | 13 | 13 중 석회 처방 (쇠붙이 상) 이 대표 — 약재 자체 `저양·개소·악창·나병` 등 다수 주치는 passage 로 단 1건. |
| 9 | 마편초 (馬鞭草) | 213 | 4 | passage 포맷으로 reasonably 반영. |
| 10 | 은조어 (銀條魚) | 89 | 1 | 한 줄 유지. |

**약재 10건 소견**: CH builder 는 children SS 를 수집하지만, **한 약재가 수십 개 처방의 원재료로 등장할 경우 "처방 내 쓰임새" rows 가 수십 배 많아, 약재→성미/귀경 direct QA 가 소수 (HERB_Q_TEMPLATES 상 650 row 전체 중 성미 질문 0건)**. 약재 식별자는 학습되지만 성미·귀경·식물학 외관 같은 지식 체계적 학습 신호 없음.

### 4.3 경혈 (DK) 샘플 10

| # | 경혈명 | raw_body_len | v7 subject | 손실 정보 |
|---:|---|---:|---:|---|
| 1 | 오추 (五樞二穴) | 61 | 1 | raw 그대로 answer. ✅ |
| 2 | 곤륜 (崑崙二穴) | 170 | 4 | ACUPOINT A_TEMPLATE 3 으로 저장 + augment INV 가 MAJOR list 포함. ✅ |
| 3 | 규음 (竅陰二穴) | 255 | 3 | ✅ direct subject QA 정상. |
| 4 | 소충 (少衝二穴) | 95 | 5 | 5 row 중 첫 매칭은 **`내경편 권1 > 기 > 침구법` 이라는 잡병 본문** (기 관련 혈 나열에 소충이 포함). **경혈 자체 subject QA 는 1~2건만**. |
| 5 | 질변 (秩邊二穴) | 183 | 2 | A_TEMPLATE 3 으로 저장. ✅ |
| 6 | 곡택 (曲澤二穴) | 146 | 1 | direct ✅ |
| 7 | 중려내수 (中膂內腧二穴) | 113 | 3 | 3 건 중 2건 이 잡병 passage 우연 매칭. |
| 8 | 하요 (下腰一穴) | 78 | 2 | ✅ |
| 9 | 상렴 (上廉二穴) | 101 | 3 | 3건 모두 `장부·금침혈` 개론 body. 경혈 direct info 는 1건만. |
| 10 | 천용 (天容二穴) | 64 | 1 | ✅ |

**경혈 10건 소견**: DK builder 는 SS body 를 대부분 정상 반영. 그러나 **MAJOR 15개 혈이 아닌 경우 INV (경맥 귀속) QA 는 1건뿐** → `합곡은 어느 경맥?` 같은 질문이 MAJOR_ACUPOINTS 에 포함된 혈만 robust 하게 학습. 상렴·소충 같은 혈은 경맥 질문 신호가 사실상 없음.

---

## 5. 학습 신호 방향 분석 (probe 결과 매트릭스)

### 5.1 probe 잘 함 (5 유형)

| probe | 관련 키워드 rows (Q+A any) | 메커니즘 |
|---|---:|---|
| Q1 담 증상 | **3,142** | symptom `담음(痰飮)` 카테고리 DD + 관련 처방 (`이진탕` 등) 수천 row 에 "담" 키워드 다량 노출. |
| Q4 풍 원인 | 31 (엄격 기준) ~ 수천 (느슨) | `풍문` mun_inscope 146 + passage 수백 row. "중풍·풍한" 조문 풍부. |
| Q9 당귀 엔티티 보존 | **1,475** | 당귀는 수백 처방 (사물탕·당귀보혈탕…) 의 원료 → answer body 에 반복 등장. 식별자 robust. |
| Q10 족삼리 경맥 | **22** | augment_sft_v7.py:743 `priority_kos = {"족삼리"}` 로 INV 3 템플릿 × 중복 9 row. "족양명위경" 귀속 학습됨. |
| Q15 숙지황 편 | **444** | 숙지황 함유 보약·허로 처방 수백 건 → 편 맥락 (`잡병편 권4 > 허로`) 다회 등장. |

### 5.2 probe 잘 못 함 (6 유형)

| probe | 관련 rows | 원인 |
|---|---:|---|
| Q2/Q11/Q12/Q14/D1~D5 처방 조성 | **composition 직접 Q = 2** | `PRESCRIPTION_Q_TEMPLATES` 6종 중 "{name} 의 구성/조성/약재?" 형태 **없음**. 답변에 dosage 포함되어도, Q↔A 쌍의 intent 는 "효능/주치/역할" 이므로 reward 방향 다름. 토큰 레벨로 약재 이름이 answer 에 있어도 "이 처방의 구성이 뭐냐?" 프롬프트에 답하도록 학습되지 않음. |
| Q17 합곡 경맥 | 31 (대부분 MAJOR INV) | 합곡은 MAJOR 에 있으나 INV 3 템플릿 = 3 row + DK direct 1 row = 4 subject row. MAJOR 내에서도 밀도 최저 그룹 (족삼리 9 row 대비). |
| Q18 태충 경맥 | 26 | 동일. MAJOR 에 있지만 priority 아님. |
| Q19 정기신 편 오분류 | **1** | `정기신` 은 raw 의 `精`/`氣`/`神` 세 BB 의 compound. builder 가 단일 엔티티로 다루지 않음. 오직 structure QA template 으로만 간접 학습. |
| P1~P7 구체 병증 | 가변 | 병증→처방 역매핑이 builder 에서 한 template (PRESCRIPTION[4] "호소할 때") 에 모두 묶여 있고, **context = 처방 부모 path 의 마지막 token** 이라 `"單方" 을 호소할 때` 같은 비의미 조합 대량 생성. |
| 증상 → 경혈 | **~112 (추정)** | builder 0 — symptom 카테고리 답변은 "병리·치법 개요" 로 정리하며 경혈 이름을 체계적으로 주입하지 않음. |

### 5.3 구체 수치

| 유형 | v7 rows | 판정 |
|---|---:|---|
| composition_q_rows (구성/조성 + 처방) | 2 | 치명적 공백 |
| dosage_answer_rows (돈/냥/푼 포함 answer) | 6,786 | 포함되어 있으나 intent mismatch |
| sym_to_rx (증상+처방 둘 다 Q 에) | 1,997 | 표면만, context 매칭 문제 |
| acu_to_meridian (INV-subcat + pattern) | 253 | MAJOR 편중 |
| 합곡/태충 mentions | 31 / 26 | MAJOR 내 최저 |
| 정기신 mentions | 1 | 학습 불가 |

---

## 6. builder 로직 정보 손실 지점 (file:line)

### 6.1 `scripts/build_sft_full_corpus.py`

| 함수 | 사용 필드 | 버리는 필드 | 구조적 손실 |
|---|---|---|---|
| `build_structure_pairs` (L553) | `original`, `trans_ko`, `up_path_nm`, children SS/ZZ `trans_ko` | `trans_en`, `annotation`, `index_num`, `footnote` | AA/BB/CC/OO/Z2 = 2,207 레코드. children body 요약 200자 제한 (L602). |
| `build_prescription_pairs` (L639) | children SS/ZZ `trans_ko` (max 400자) | **처방 레코드 자체의 `original`·`trans_ko` 본문**, 사용된 `detail` 의 말미 `《출전》` 주석 | **처방명 → 조성 QA template 부재** (L75-82, Q_TEMPLATES 6종 모두 효능/주치/역할/출전 방향). `detail[:200]` 절단으로 약재 ≥10개 처방은 뒷부분 소실. |
| `build_herb_pairs` (L709) | children SS/ZZ | `footnote`, CH 자체 긴 본문 (경우에 따라 400자 이상) | 한 약재 한 쌍 1 row 로 고정 (variant 1종) → 성미·귀경 vs 주치 vs 효능 방향성 혼합. `_is_shell_fallback` (L520) 로 shell skip 하나 meaningful body 가 400자 넘으면 잘림. |
| `build_acupoint_pairs` (L762) | children SS/ZZ (max 300자) | DK 자체 본문, alternate name `(別名)`, `《출전》` 다수 | ACUPOINT_Q_TEMPLATES 5종 중 "어느 경맥?" inverse (L112) 존재하나 **실제 rotate index `i % 5` 로 1/5 확률 — DK 396 레코드 중 약 72 건만 해당**. 실측: `acu_어느경맥` 질문 99건. |
| `build_symptom_pairs` (L805) | children SS/ZZ | DD 자체 본문의 선두 요약, 상위 경로 병리 분류 | 증상→처방/혈 방향 inverse QA **없음**. SYMPTOM_Q_TEMPLATES 5종 모두 "병리·분류·호소 시 의미" 해설 방향. |
| `build_passage_pairs` (L884) | 7,000 random sample (L892 `cand[:target]`) | **나머지 15,739 SS/ZZ 레코드 전폐기** | 최대 가치 본문 loss. `--passage-target 7000` 하드 캡. 선별 기준 `len(trans_ko) >= 80` (L890) 이 유일. |
| `build_diagram_pairs` (L846) | XX 의 `trans_ko` 만 | 단방 (83건 제외, L858), XX 의 `original` 한자 | XX 57 uniq 중 3 row 만 생성 (category `diagram` = 3). 98% XX 신호 손실. |
| `build_paraphrase_pairs` (L918) | 하드코드 3 seed × 5 Q × 5 A | raw 전체 | paraphrase 는 기본 사실 3 주제에 국한, raw 와 무관. |

**구조적 정보 손실 누적:**
- TT 26 (운기 표), PP 19 (인체 부위), EE 5, CP 1 → **모두 0 row 변환**. builder 라우팅 미구현.
- XX 102 → 3 row (diagram). 97% 버려짐.
- SS+ZZ 22,739 → 7,000 (passage) + children 요약에 일부 흡수. 나머지 ~14,000+ 본문 직접 QA 생성 안 됨.

### 6.2 `scripts/augment_sft_v7.py` (post-processing)

| 함수 | 기여 | 손실 |
|---|---|---|
| `stage1_filter_shell_answers` (L181) | CROSS_REF_RE (L77) 로 "X문에 나온다" 패턴 매칭 → 80% 이상 shell 은 drop, 아니면 trim | drop 된 row 자체 손실 (workspace 로만 보관). v6 → v7 에서 204 row 감소 (17,937 → 17,733). |
| `_rebuild_refusal_oos` / `_safety` (L255, L306) | 다양성 확보된 refusal 각 100 row | v6 대비 refusal 총량 800 → 200 (75% 감소) — 범위-외 질문 robust 감소 가능성. |
| `_build_mun_inscope_qa` (L410) | "-문" surface 를 쿼리에 쓰는 146 row | MUN_TARGETS 20종 중 평균 7~8 row/종. 밀도 부족. |
| `_build_acupoint_inverse_qa` (L552) | INV 250 row (MAJOR 15 × ~3 templates + non-major breadth) | non-major 374 혈 각 1 row. MAJOR 만 3~9 rotation (priority 족삼리 9 row). |
| `validate_diversity` (L792) | prefix_top1 ≤ 15%, closing_top1 ≤ 10% 감사 | `mun_inscope` 의 한 prefix 가 편중 시 violation 로깅 (실제 violation 여부는 rows 적어 기각) |

**augment_sft_v7 의 핵심 공백**:
- 처방→조성 inverse QA 신규 생성 함수 **없음**.
- 증상→경혈 inverse QA 신규 생성 함수 **없음**.
- 약재→성미/귀경 직접 QA template **없음**.
- raw 의 TT/PP/EE/CP 레벨 추가 활용 **없음**.

---

## 7. 정량 Gap 테이블 (14 매핑 방향)

"필요 rows" 산정: raw unique 엔티티 × variant 배수 (각 방향마다 2~4 variant 가 다양성 원칙 3 충족에 필요; 본 분석은 conservatively 2× 사용).

| # | 매핑 방향 | raw 근거 엔티티 | 필요 rows (2× 추정) | v7 실제 rows | Gap | 구조적 원인 |
|---:|---|---:|---:|---:|---:|---|
| 1 | passage → 해석 | 22,739 (SS+ZZ) | 45,478 | 7,000 | **-38,478** | `build_passage_pairs` hard-cap `target=7000` (L884). 나머지 본문은 children summary 로 일부만 흡수 |
| 2 | 처방명 → 조성 (약재·용량) | 6,039 (DP+EP) | 12,078 | **2** (실질 0) | **-12,076** | `PRESCRIPTION_Q_TEMPLATES` (L75) 6종 중 구성 질문 없음. composition inverse QA 빌더 부재 |
| 3 | 처방명 → 주치 | 6,039 | 12,078 | 5,007 | -7,071 | DP+EP 5,978 row 의 Q/A 가 주치 방향. 다만 EP 일부 (771) 과 DP 일부 (771) 는 shell. cover ratio ~83% |
| 4 | 증상 → 처방 (inverse) | 1,103 (DD) | 2,206 | 1,997* | +실질-1,300 | `*` 는 Q 에 증상·처방 둘 다 등장한 row 인데 대부분 `PRESCRIPTION[4]` "호소할 때" 템플릿. context = path leaf. `單方`·`여러 가지` 같은 비의미 context 대량 |
| 5 | 경혈명 → 경맥 | 396 (DK) | 792 | 250 (INV subcat) | -542 | `_build_acupoint_inverse_qa` MAJOR 15 × 3 + non-major 1 → 편중 |
| 6 | 경혈명 → 위치 | 396 | 792 | 148 | -644 | ACUPOINT Q_TEMPLATE rotate 5종 중 "위치와 주치" 는 1 variant |
| 7 | 경혈명 → 주치 | 396 | 792 | 162 | -630 | DK builder SS body 를 detail 로 묶되 "주치" 질의 pair 는 한정 |
| 8 | 증상 → 경혈 (inverse) | 1,103 | 2,206 | ~112 (느슨) | -2,094 | builder **없음**. symptom 카테고리에서 혈 이름 출현은 우연 |
| 9 | 약재명 → 성미 | 1,403 (CH+DH unique names 2,292) | 2,806 | 0~521 (성미 token 포함 Q 521, 그러나 Q 자체 문구는 "성질과 주치") | ≈ -2,285 | HERB Q_TEMPLATE 5종 중 성미 단독 질의 없음. `[2]` "성질과 주치" 한 종 |
| 10 | 약재명 → 효능 | 1,403 | 2,806 | ~277 | -2,529 | HERB_Q_TEMPLATES 5종 중 "대표 효능" 1종 |
| 11 | 약재명 → 편·부 (분류) | 1,403 | 2,806 | ~277 | -2,529 | HERB_Q_TEMPLATES[3] "약재 분류" 한 종 |
| 12 | 편명 → 수록 주제 | 2,207 (AA+BB+CC+OO+Z2) | 4,414 | 1,932 | -2,482 | STRUCTURE Q 5종 rotate. CC 중 15 완전 누락 |
| 13 | 개념어 → 편 소속 | 2,207 | 4,414 | 146 | **-4,268** | `augment_sft_v7` mun_inscope MUN_TARGETS 20종 × 평균 7 row |
| 14 | passage → up_path | 22,739 | 22,739 (1×) | 7,000 | -15,739 | passage 카테고리 모든 answer 에 `[출처:]` cite 있음; passage_target 자체가 한계 |
| (bonus 15) | XX 도해 → 설명 | 57 uniq XX | 114 | 3 | -111 | `build_diagram_pairs` 3 row 만 생성 (subcat 통계 XX=3) |
| (bonus 16) | TT/PP/EE/CP → 설명 | 51 (raw records) | 100 | 0 | -100 | builder 라우팅 부재 |

### 7.1 우선순위 제안 (ver8 대비)

1. **매핑 #2 처방→조성** (gap -12,076). 템플릿 추가: `"{name} 의 구성 약재와 용량을 나열해 주세요"` → answer 는 children SS 본문 + 명시적 **"구성:" 헤더**. 현재 dosage 는 answer 에 있으나 Q/A intent 정렬이 안 돼서 학습 안 되는 문제 해결.
2. **매핑 #8 증상→경혈**, **#4 증상→처방** (inverse) 체계화. DD 레코드의 parent path 를 쓰지 말고, children body 안에서 **명시적으로 등장하는 처방명/혈명 스캔** 후 inverse QA 생성.
3. **매핑 #9~11 약재 성미/귀경/부**. CH/DH 본문 첫 한두 문장은 전형적으로 "성질이 X 하고 맛은 Y 하며 독이 Z" 패턴 → regex 로 추출해 성미 QA 고정 template 생성 가능.
4. **매핑 #1 passage** 7,000 → 최소 14,000 확대. children body 요약 의존도 낮추고 passage direct QA 를 현재의 2×.
5. **매핑 #5~7 경혈 INV** non-major 374 혈 각 2 row (INV + 위치) 확대.
6. **bonus TT/PP/EE** : 운기표·인체부위 QA 신규 빌더 (minor 이나 공백 채우기).

---

## 부록 A. shell-answer 잔존 집계

| 패턴 | v7 answer count |
|---|---:|
| `본문을 참고` | 991 |
| `본문에는 본문 참고` (self-ref) | 0 ← v7 의 A5 fix 효과 |
| `본문 참조` | 0 |
| `원문 참조` | 41 |
| `본문 인용 참조` | 0 |
| `구체 조성 · 용량은 본문` | 965 |
| `에 나온다` | 37 |

processing stage 1 (CROSS_REF_RE) 이 대부분 제거했으나, prescription template `"구체 조성 · 용량은 본문을 참고해 주세요"` 는 (build_sft_full_corpus.py:688, L87 A_TEMPLATE 3 tail) **정상 템플릿으로 남아 965 row 유지** — 이 문구가 Q 가 "조성" 을 물을 때 answer 가 "본문 참고" 로 회피하는 **shell pattern** 의 핵심.

## 부록 B. answer prefix top-3 per category (다양성 근거)

| category | top-1 prefix | top-1 count |
|---|---|---:|
| acupoint | `족삼리 혈 해설: 삼리(三里) 혈은 …` | 3 |
| prescription | `동의보감 잡병편 권11 > 소아 > 두창 > 두창 > …` | 4 |
| passage | `**해설**: 성질이 평(平)하고 맛은 달며 …` | 5 |
| structure | `동의보감 잡병편 권7 > 옹저 > 옹저(옹저) …` | 5 |
| symptom | `동의보감 잡병편 권10 > 부인 > 소아 > 소아(小兒 …` | 6 |
| paraphrase | `동의보감은 허준이 편찬한 의서입니다. …` | 5 |
| refusal_oos | `본 모델의 인용 검증은 동의보감 원문에 한해 가능합니다 …` | 3 |
| refusal_safety | `본 모델은 동의보감 조문의 '계열' 을 설명할 수 있으 …` | 3 |

category 단위 prefix top-1 비율은 모두 1% 미만 — 원칙 3 은 만족. 그러나 Q-prefix 기준 `이 대목은 동의보감 어떤 주제…` 1,400 row 는 passage 카테고리 내 20% 수준으로, passage 내부 Q 다양성 하한에 근접.

## 부록 C. 분석 재현

```bash
# 레코드·엔티티·QA 수치 일체 재현
python3 /tmp/v7_gap_analysis.py

# subject-level 포렌식 (DP/CH/DK 각 10)
python3 /tmp/v7_forensic_v2.py

# 중간 산출물
# /tmp/v7cov.json          - content_level 별 coverage 표
# /tmp/v7_clusters.json    - category × prefix cluster top-60
# /tmp/v7_intents.json     - intent 별 Q count
# /tmp/v7_forensic_v2.json - 10+10+10 case 상세 raw vs v7
# /tmp/v7_signal.json      - probe-유형별 rows
# /tmp/v7_shell.json       - shell-answer 잔존 패턴
# /tmp/v7_gap_table.json   - 14 mapping direction 테이블
# /tmp/v7_prefix.json      - category 별 answer prefix top
```

분석 근거 레코드 수 (raw): 34,040. v7 rows: 17,733. 두 수치 모두 파일 line count 와 정확히 일치 검증.
