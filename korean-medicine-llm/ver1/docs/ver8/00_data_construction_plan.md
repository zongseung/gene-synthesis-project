# 동의보감 SFT 데이터 v8 전수 재구축 기획서

> 작성일: 2026-04-24
> 대상 산출물: `experiments/dongui_bogam/data/sft/phaseB_qa_v8_corpus.jsonl`
> 빌더(신규): `scripts/build_sft_full_corpus_v8.py`
> 근거 문서: `docs/ver8/01_raw_data_schema.md` (Agent 1 · 전수 34,040 스캔), `docs/ver8/02_v7_gap_analysis.md` (Agent 2 · v7 17,733 vs raw 교차 분석), `docs/ver6/00_halluc_repetition_fix_plan.md` §3.1.2, `.claude/harness-evals/hanmed_cpt/final_report.md` A1~A9
> 원칙 1 조: **전수(全數) 커버리지 — raw 34,040 레코드는 1건도 빠지지 않는다. 샘플링 금지.**

---

## 구현 상태 (IMPLEMENTATION STATUS) — 2026-04-24 기준

> ⚠️ **본 문서는 설계 기획서이며, 빌더 코드는 아직 저장소에 존재하지 않는다.**
>
> | 항목 | 상태 | 경로 |
> |---|---|---|
> | 기획서 (본 문서) | ✅ 작성 완료 | `docs/ver8/00_data_construction_plan.md` |
> | raw schema 분석 | ✅ 작성 완료 | `docs/ver8/01_raw_data_schema.md` |
> | v7 gap 분석 | ✅ 작성 완료 | `docs/ver8/02_v7_gap_analysis.md` |
> | v7 빌더 (참고용) | ✅ 존재 | `scripts/build_sft_full_corpus.py` |
> | v7 post-processor (참고용) | ✅ 존재 | `scripts/augment_sft_v7.py` |
> | **v8 빌더 디렉토리** | ❌ **미생성** | `scripts/build_sft_v8/` (Day 0 체크리스트 A.1 항목) |
> | **v8 빌더 entry point** | ❌ **미작성** | `scripts/build_sft_full_corpus_v8.py` |
> | **v8 corpus 산출물** | ❌ **미생성** | `experiments/dongui_bogam/data/sft/phaseB_qa_v8_corpus.jsonl` |
> | **v8 adapter 학습 결과** | ❌ **미생성** | `experiments/dongui_bogam/outputs_ver8_gemma_v1/` |
>
> 따라서 본 문서의 모든 rows 기대치 (§6), pass 기준 (§7.2), rollback 조건 (§A.5) 은 **실측이 아닌 설계 목표치** 이다.
> 본 기획서 승인 후 섹션 8 Agent 분할에 따라 Agent-P/A/S/M/R 이 병렬 구현에 착수하면 상태가 갱신된다.

---

## 섹션 1. 배경 및 문제 의식

### 1.1 v6 → v7 실패 요약

| 버전 | 데이터 rows | eval_loss | probe 결과 핵심 실패 |
|---|---:|---:|---|
| v5 | 21,475 | 0.72 (추정) | 3,000×3 고정 prefix, F3 repetition loop |
| v6 | 18,690 | 0.6055 (epoch 2 best) | Q2 "보중익기탕 → 상한문" 환각, Q4/Q8 false refusal, Q5/Q9 "본문에는 본문" self-ref, Q10 족삼리 경맥 오답 |
| v7 | 17,733 | 미측정 (학습 전) | 14개 매핑 방향 gap 잔존 (02 문서 §7) |

final_report.md §"원인 랭킹" 기준 round_3 의 10개 환각은 **3개 file:line 의 데이터 빌더 버그** 로 대부분 설명된다.

### 1.2 환각 근본 원인 랭킹 (round_3 final_report §원인 랭킹 원문 인용)

1. `build_hanja_to_korean_map` first-wins aliasing (`scripts/build_sft_full_corpus.py:387-417`) — Q5·Q6(B)·Q7·Q9·Q10 부분을 동시 설명.
2. passage "처방은 X문에 나온다" 1,283 shell-answer — Q2 token-exact 공급원 (특히 "상한문에 나온다" 235건).
3. refusal 800건 × 12 유니크 assistant — 66~67회 복제 학습으로 Q4/Q8 false refusal.
4. `LEVEL_DESC["CC"] = "중주제(中門)"` 한자 괄호 1,883 CC row 오염 — Q6(A).
5. HERB_A_TEMPLATES fallback `본문 {cite} 참고` 154 row — "본문에는 본문" self-ref.
6. inverse-map 부재 (족삼리 leaf 0, 경맥 77 vs 52 불균형) — Q10, Q2 일부.
7. system prompt "용량 금지" × data 4,994 row 용량 포함의 모순 — Q2(B)/Q4/Q5 shell 혼재.

v7 은 1~5 및 7 을 국소 patch 로 처리했으나, **근본 문제 (매핑 방향 편향, 샘플링 cap, inverse QA 부재) 는 v7 builder 구조상 해결 불가**. 이 한계를 **전수 재구축 (v8)** 으로 넘어선다.

### 1.3 왜 전수 재구축이 필요한가 — Agent 2 gap 분석 핵심 수치

02 문서 §7 "정량 Gap 테이블 (14 매핑 방향)" 에서 **합산 row gap 은 약 87,400 rows** (양방향 합). 주요 치명 gap 만 발췌:

| 방향 | raw 근거 엔티티 | v7 실제 rows | Gap | 근거 (02 §7) |
|---|---:|---:|---:|---|
| 처방명 → 조성 (약재·용량) | 6,039 (DP+EP) | **2** (실질 0) | **-12,076** | `PRESCRIPTION_Q_TEMPLATES` 6종 중 구성 질문 없음 |
| passage → 해석 | 22,739 (SS+ZZ) | 7,000 (hard-cap) | **-38,478** | `build_passage_pairs` `target=7000` (L884) |
| 증상 → 경혈 (inverse) | 1,103 (DD) | ~112 | -2,094 | builder 함수 **부재** |
| 약재명 → 성미 | 1,403 (CH+DH) | 0~521 | ≈ -2,285 | HERB_Q_TEMPLATES 5종에 성미 단독 질의 없음 |
| 개념어 → 편 소속 | 2,207 | 146 | **-4,268** | mun_inscope 20종 × 평균 7 row |
| 경혈명 → 경맥 | 396 | 250 (INV subcat) | -542 | MAJOR 15 편중, non-major 374 혈 각 1 row |
| TT/PP/EE/CP (미활용) | 51 | **0** | -100 | v7 builder 라우팅 자체 부재 (01 §6.1) |

또한 v7 는 **단방향 학습 편향** 이 극단적이다 (02 §1):

- 처방 4,352 · 약재 1,226 · 경혈 389 에 대해 **이름 → 본문** lookup 사전 기능은 99%+ 커버.
- 그러나 **증상 → 처방/혈·약, 약재 → 성미·귀경, 처방 → 조성** 같은 임상 reasoning 역방향 학습 신호는 0~1% 수준.

**결론**: v7 builder 에 로직을 더 추가하는 것만으로는 부족하다. 전수 커버리지 + 다방향 signal 을 처음부터 설계한 v8 빌더가 필요하다.

---

## 섹션 2. 설계 원칙 (6조)

> ver6 기획서 §3.1.2 "답변 구체성 4원칙" 은 v8 에서도 **무조건 준수** 한다. 아래는 그 4원칙을 v8 의 전수 커버리지 목표 위에 확장한 6조 (원칙 0 은 §5.1 에 구현).

### 원칙 0 — 전처리 완결성 (text preprocessing choke point)

모든 raw 접근은 **§5.1 의 `load_raw_record_clean` 하나를 거친다**. 개별 builder 에서 `.strip()` 을 다시 호출하거나 자체 정규화 로직을 만드는 것은 금지. raw 에는 `\r\n` 이 36,268 건 · 전각 공백 1 · 이중 공백 88 · 괄호 불일치 3 건이 있으며, 이 중 하나라도 QA output 에 살아남으면 **원칙 0 validator 가 빌드를 fail** 시킨다.

```python
# 5.1 에서 정의한 clean_text / clean_up_path / load_raw_record_clean 은 v8 의 유일한 raw 게이트
# 최종 output 에 대해 _assert_no_noise(qa_rows) 가 CR · 전각공백 · 이중공백 0 을 강제
```

### 원칙 1 — 전수 커버리지 (全數 · assertion 강제)

raw 34,040 레코드 각각이 v8 corpus 의 **최소 1개 QA row 의 source_id 목록에 포함** 되어야 한다. 샘플링 금지, hard-cap 금지. 빌드 종료 시 아래 assertion 이 **fail 하면 빌드 실패**.

```python
# build_sft_full_corpus_v8.py 말미
covered_ids: set[tuple[int, int]] = set()  # (volume_id, content_seq) 로 레코드 식별
# ... 모든 build_* 함수가 covered_ids.add((vol, seq)) 호출
RAW_TOTAL = 34040  # Agent 1 §1.1 검증 수치
assert len(covered_ids) == RAW_TOTAL, (
    f"coverage fail: {len(covered_ids)} / {RAW_TOTAL}; "
    f"missing={sorted(all_raw_ids - covered_ids)[:20]}"
)
```

### 원칙 2 — 다방향 학습 신호 (direct + inverse + cross)

모든 핵심 엔티티에 대해 아래 세 방향 중 **최소 2개** 의 QA 를 생성한다.

| 엔티티 계열 | direct | inverse | cross |
|---|---|---|---|
| 처방 (4,352) | 이름→효능·주치 | **증상→처방** | 편→소속 처방 |
| 약재 (1,226) | 이름→성미·주치 | **병증→약재**, **효능→약재** | 부→약재 목록 |
| 경혈 (389) | 이름→위치·주치 | **경맥→혈**, **증상→혈** | 침구편→혈 인덱스 |
| 개념 (정·기·신·혈·담·풍·한 등) | 개념→편 소속 | 편→수록 개념 | 개념↔인접 증론 |

### 원칙 3 — 정보 무손실 (미활용 필드 활용)

v7 이 버리거나 축약한 필드를 v8 은 체계적으로 사용한다.

- `original` (한자 원문): 한자 ↔ 한글 paraphrase pair 로 재활용 (원칙 1 한자 엔티티 보존 + 토크나이저 byte-fallback 내성).
- `trans_en`: 34,024 건 영문 번역 (01 §1.2) — refusal 카테고리 pool 확장에 **사용하지 않음** (한국어 모델 학습 범위 외이므로 학습에 포함 금지), 단 entity 매칭 sanity check 에만 사용.
- `index_num` (33,865 건): 편·권 내 목차 순서 — meta_toc 카테고리 답변 순서 결정.
- `footnote` 등 raw 필드 중 존재하지 않는 것은 추가 수집 시도 금지 (Agent 1 §1.2 가 100% 검증 — `annotation` 은 전 레코드 null).

### 원칙 4 — 템플릿 다양성 (ver6 §3.1.2 원칙 3 + 확장)

- 카테고리 당 question prefix pool ≥ **5종**.
- 전체 데이터 기준 첫 20자 prefix top-1 비율 ≤ **15%** (02 §3.1 에서 v7 passage cluster #1 이 1,400 rows = 7.9% 로 보더라인이었음; v8 에서는 passage 카테고리 내 prefix top-1 ≤ **8%** 로 더 엄격).
- 종결 20자 top-1 ≤ **10%** (ver6 원칙 4 직접 인용).
- 카테고리 × prefix 조합 unique_prefix_ratio ≥ **3%** (ver6 원칙 3 정량).

### 원칙 5 — 안전성 layer (refusal_v2 + pregnancy_safety)

round_3 probe 의 P1~P7 구체 병증 중 "임산부 / 수유 / 영·유아 용량" 은 모델이 환각 용량을 제시하면 **환자 위해**. 따라서 임산부·유아·중증 질환 관련 질문은 **safety refusal 을 우선** 발동한다. 이를 위해:

- `pregnancy_safety` 카테고리 신설 — 임산부·胎動·催生符·산후 관련 raw 자료에서 기인한 질문은 "동의보감 원문 조문 서술" + "현대의학 전문가 상담 권고" 2분 구조로 답변.
- `refusal_v2` 카테고리 — 범위 외 질문 50+ variant (v7 의 12 unique → 50+) + "-문" surface form 150+ in-scope QA (final_report A3 반영).

---

## 섹션 3. raw → QA 매핑 설계 (18 content_level 전수 명시)

> 모든 level 에 **"전수 assertion 문구"** 를 포함. 한 레벨이라도 기대 rows 하한을 미달하면 빌드 fail.

### 3.1 레벨별 매핑 매트릭스 (요약)

| content_level | raw | 매핑 카테고리 | variant 배수 | 기대 rows | v7 대비 |
|---|---:|---|---:|---:|---|
| SS | 11,498 | passage + children detail 재참조 | 2 | 22,996 | +19,301 (v7 3,695) |
| ZZ | 11,241 | passage + concept_inverse | 2 | 22,482 | +19,177 (v7 3,305) |
| DP | 5,273 | prescription_direct + prescription_inverse | 3 | 15,819 | +10,591 (v7 5,228) |
| EP | 766 | prescription_direct + prescription_inverse | 3 | 2,298 | +1,548 (v7 750) |
| CC | 2,045 | structure + meta_toc | 2 | 4,090 | +2,207 (v7 1,883) |
| DD | 1,103 | symptom_direct + symptom_inverse_rx + symptom_inverse_acu | 3 | 3,309 | +2,206 (v7 1,103) |
| DH | 704 | herb_direct (부위) + herb_inverse_by_parent | 2 | 1,408 | +824 (v7 584) |
| CH | 699 | herb_direct + herb_sami + herb_inverse | 3 | 2,097 | +2,031 (v7 66) |
| DK | 396 | acupoint 4-variant (경맥/위치/주치/증상-혈) | 4 | 1,584 | +938 (v7 646) |
| BB | 109 | structure + meta_toc | 2 | 218 | +213 (v7 5) |
| XX | 102 | diagram_explain + 단방 bridge | 2 | 204 | +201 (v7 3) |
| TT | 26 | table_explain | 3 | 78 | +78 (v7 0) |
| Z2 | 25 | meta_toc + concept_inverse | 3 | 75 | +58 (v7 17) |
| AA | 23 | structure + meta_toc | 3 | 69 | +46 (v7 23) |
| PP | 19 | anatomy_diagram | 2 | 38 | +38 (v7 0) |
| OO | 5 | preface_direct | 2 | 10 | +6 (v7 4) |
| EE | 5 | symptom_variant | 2 | 10 | +10 (v7 0) |
| CP | 1 | prescription_direct (DP 와 합류) | 3 | 3 | +3 (v7 0) |
| **합계 (raw 매핑)** | **34,040** | — | — | **76,788** | — |

신규 카테고리 (증상→처방/혈 inverse, concept_inverse, refusal_v2, pregnancy_safety, meta_toc) 는 위 레벨 매핑 안에서 파생되며 **원본 레코드 중복 참조는 허용**(한 레코드가 여러 QA 의 source 로 쓰임). 전수 assertion 은 "레코드가 **최소 1개** QA 의 source_id 에 포함" 기준.

### 3.2 핵심 레벨 상세

#### 3.2.1 DP / EP — 처방 (6,039 레코드)

**사용 필드**: `original` (한자 처방명), `trans_ko` (한글), `up_path_nm` (편·권·증론), `index_num`, **자식 SS/ZZ 본문** (조성·용법·주치).

**v7 한계**: `PRESCRIPTION_Q_TEMPLATES` 6종 중 "구성/조성 약재" 질문 **0종**. 결과적으로 처방 → 조성 direct QA 는 실질 0 건 (02 §3.2).

**v8 타깃 variant (3 variant × 6,039 = 18,117 rows 기대)**:

| variant | Q 템플릿 예 | A 구조 | 근거 field |
|---|---|---|---|
| direct-efficacy | "'{ko}' ({han}) 처방의 효능과 주치를 정리해 주세요." | 자식 SS 의 첫 문장(주치) + 출전 cite | SS 본문 + up_path |
| direct-composition | "'{ko}' 처방의 구성 약재와 용량을 나열해 주세요." | "구성: 반하 3.5돈, 진피 2.2돈, …" 형식 header + 원문 복원 | SS body dosage 부분 |
| direct-source | "'{ko}' 은 동의보감 어느 편·증론에 수록된 처방인가요?" | up_path echo + 증론 요약 | up_path_nm |

**전수 assertion**:
```python
dp_ep_ids = {(r['volume_id'], r['content_seq']) for r in raw_records if r['content_level'] in ('DP','EP','CP')}
dp_ep_covered = {src for qa in qa_rows if qa['category'].startswith('prescription') for src in qa['source_ids']}
assert dp_ep_ids <= dp_ep_covered, f"DP/EP/CP {len(dp_ep_ids - dp_ep_covered)} records uncovered"
```

추가: direct-composition variant 는 raw SS 본문에 dosage 키워드 (`돈|냥|푼|兩|錢`) 가 있는 레코드에 한해 생성한다. 없으면 variant 를 efficacy 2회 생성 (카테고리 row 수는 유지).

#### 3.2.2 DK — 경혈 (396 레코드)

**사용 필드**: `original` (X二穴), `trans_ko` (혈명), `up_path_nm` (경맥 segment — 01 §3.3 기준 12정경 + 任·督脉 + 別穴), 자식 SS/ZZ (위치·주치).

**v7 한계**: MAJOR 15 혈 편중 (02 §5.3). non-major 374 혈 각 1 row. 증상→혈 inverse 는 builder 에 부재.

**v8 타깃 variant (4 variant × 396 = 1,584 rows 기대)**:

| variant | Q 예 | A 구조 |
|---|---|---|
| meridian (inverse-1) | "'{ko}' 혈은 동의보감 침구편에서 어느 경맥에 속하나요?" | "{parent_meridian} 에 속합니다. [출처: {cite}]" |
| location | "{ko} 혈의 위치를 동의보감 침구편 기준으로 설명해 주세요." | 자식 SS 의 해부 위치 문장 + cite |
| indications | "{ko} 혈의 주치와 침구법을 정리해 주세요." | 자식 SS 의 주치/자침 깊이 + cite |
| symptom-to-point | "'{symptom}' 증상이 있을 때 동의보감 침구편은 어느 혈을 사용하나요?" | "{ko} ({parent_meridian}) — 주치 …" (증상 은 자식 SS 본문에서 추출) |

01 §3.3 경맥 segment 15개를 `MERIDIAN_SEGMENTS` 상수로 하드코딩 (raw 검증 수치). 別穴 40, 任脉 24, 督脉 27 포함.

#### 3.2.3 CH / DH — 약재 (1,403 레코드; 한자 1,027 · 한글 별칭 1,226)

**사용 필드**: `original` (한자명 + 한글 별칭 공백 구분 — 01 §2.8 형식 `井華水 새배처엄기른우믈믈`), `trans_ko`, `up_path_nm` (탕액편 {부}), 자식 SS/ZZ (성미·주치·제법).

**v7 한계**: HERB_Q_TEMPLATES 5종 중 "성미" 단독 질의 없음. 약재 한 개가 수십 처방 원료로 쓰여도 약재 자체 **성미·귀경·식물학 정보 학습 신호 없음** (02 §4.2).

**v8 타깃 variant (3 variant × 1,403 = 4,209 rows 기대)**:

| variant | Q 예 | A 구조 | regex 추출 |
|---|---|---|---|
| sami (성미) | "'{ko}' ({han}) 의 성미와 독성을 설명해 주세요." | raw SS 첫 문장 그대로 ("성질이 평(平)하고 맛은 …") | `^性[平寒溫熱凉].*?味.*?(毒|無毒).*?\.` |
| indication | "'{ko}' 는 어떤 병증에 쓰이나요?" | raw SS 의 主治 문장 | `主治|療` 매칭 문장 |
| classification (inverse-1) | "동의보감 탕액편의 {부}에 수록된 약재 중 '{ko}' 의 분류는?" | up_path echo + 부 내 위치 | up_path_nm |

raw 본문의 `性平, 味苦醎, 無毒` 패턴 (01 §2.1 샘플 #4 참조) 을 regex 로 정확 추출 — 실패 시 variant 를 indication 2회로 치환.

#### 3.2.4 DD — 증론 (1,103 레코드; unique 1,005)

**사용 필드**: `original` (증론명), `up_path_nm` (편·권·대문·증론), **자식 SS/ZZ 본문** (병리·치법·처방 목록).

**v8 타깃 variant (3 variant × 1,103 = 3,309 rows 기대)**:

| variant | Q 예 | A 구조 |
|---|---|---|
| direct-pathology | "'{ko}' 증후의 병리와 주요 증상을 설명해 주세요." | 자식 ZZ 병리 서술 + cite |
| inverse-rx | "'{symptom}' 증상이 나타날 때 동의보감은 어떤 처방을 쓰나요?" | **자식 SS/ZZ 본문에서 명시적으로 등장하는 처방명 스캔** 후 top-3 처방명 + 각 출전 |
| inverse-acu | "'{symptom}' 의 침구 치료로 어느 혈을 씁니까?" | 자식 ZZ 에 혈명 등장 시 혈 목록 + 경맥 |

**v7 과 결정적 차이**: 02 §2.3 "(증상→처방은 PRESCRIPTION[4] 로 묶여 있고 context = 처방 부모 path leaf — `單方 을 호소할 때` 같은 비의미 pair) 문제" 를 DD 레코드의 자식 body 스캔으로 해결. 즉 `"증상"` 은 DD 자체의 `trans_ko` 이고, 답변의 처방명은 **자식 본문에서 실체로 등장한 것만** 쓴다.

#### 3.2.5 SS / ZZ — 본문 (22,739 레코드)

**v7 한계**: `build_passage_pairs` 의 `target=7000` hard-cap 으로 15,739 건 직접 폐기 (02 §6.1). **ver8 은 cap 을 제거한다**.

**v8 타깃 variant (2 variant × 22,739 = 45,478 rows 기대)**:

| variant | Q 예 | A 구조 |
|---|---|---|
| passage-summary | "다음 동의보감 본문의 핵심 의미를 정리해 주세요.\n발췌: {trans_ko[:200]}…" | 본문 요약 (길이 1/3) + `[출처: up_path]` |
| passage-topic | "이 본문 구절은 동의보감의 어떤 주제를 설명합니까?\n발췌: {…}" | 주제 keyword + up_path leaf + 편 context |

ver6 원칙 4 (종결 top-1 ≤ 10%) 은 passage 카테고리 내부에서도 강제. 길이 80자 미만 SS/ZZ 는 variant 를 1 개로 축소 (passage-topic 만). 02 §3.1 의 v7 passage cluster #1 (`이 대목은 … 요지를 짚어 주세요` 1,400 rows) 이 7.9% 로 보더라인이었으므로 **v8 에서는 prefix 풀을 8종 이상으로 확장**.

**전수 assertion** (passage 에서 SS+ZZ 모두 커버):
```python
ss_zz_ids = {(r['volume_id'], r['content_seq']) for r in raw_records if r['content_level'] in ('SS','ZZ')}
passage_covered = {src for qa in qa_rows if qa['category'] == 'passage' for src in qa['source_ids']}
assert ss_zz_ids <= passage_covered, f"SS+ZZ {len(ss_zz_ids - passage_covered)} uncovered"
# 기대: 0 miss. 샘플링 없음.
```

#### 3.2.6 AA / BB / CC / OO / Z2 — 구조 (2,207 레코드)

01 §2.10/2.13/2.16 기준. meta_toc + structure 로 이중 매핑.

| level | raw | direct 질문 예 | meta_toc 질문 예 |
|---|---:|---|---|
| AA (23) | `內景篇卷之一` | "내경편 권1 은 동의보감에서 어떤 위치에 있는 편입니까?" | "내경편 권1 에 수록된 대문(BB) 을 나열해 주세요." |
| BB (109) | `身形` | "'신형(身形)' 대문은 어느 편에 속하며 어떤 주제를 다룹니까?" | "신형 대문에 포함된 중주제(CC) 목록은?" |
| CC (2,045) | `形氣之始` | "'형기의 시작' 은 어느 편·대문 아래에 놓입니까?" | "형기지시의 하위 증론(DD) 을 나열해 주세요." |
| OO (5) | `東醫寶鑑序` | "동의보감 서문은 어느 권에 실려 있습니까?" | — (짧음) |
| Z2 (25) | 총목 (`身形 精 氣 神`) | "동의보감 총목에 '신형·정·기·신' 이 묶여 있는 이유는?" | "총목 기준 편별 주제 순서는?" |

#### 3.2.7 XX / PP / TT / EE / CP — v7 미활용 175건 (01 §6.1)

v7 에서 라우팅 부재로 전 폐기되었던 레벨. **v8 은 전수 포함**.

| level | raw | v8 카테고리 | 매핑 전략 |
|---|---:|---|---|
| XX (102) | diagram_explain | 19 도해 설명 본문 + 83 단방 종수 표지는 `"{편} 의 단방은 모두 몇 종인가?"` QA 로 변환 (01 §2.11 샘플 #2~#4 `凡四十一種.` 형식 활용). |
| PP (19) | anatomy_diagram | `身形藏府圖` 등의 한자 라벨 리스트를 그대로 열거 QA. 한 레코드 `vol_18 seq=984` (催生符 부적, trans_ko=`"\r\n"` 01 §1.2) 는 **safety refusal 로 매핑** — 부적은 의학 조언 범위 외. |
| TT (26) | table_explain | 운기도·五腧五行표 (01 §2.12) 를 markdown 표로 변환한 QA 3 variant (표 전체 / 특정 칸 / 해석). |
| EE (5) | symptom_variant | `一法/又法/禳法/點法/枳橘熨法` (01 §2.17) 를 부모 증론에 바인딩한 보조 치법 QA. |
| CP (1) | prescription_direct (DP 와 합류) | `雜病篇 卷之一 > 吐 > 霞天膏` — DP 와 동일 템플릿으로 3 variant 생성. |

**전수 assertion**:
```python
niche_ids = {(r['volume_id'], r['content_seq']) for r in raw_records
             if r['content_level'] in ('XX','PP','TT','EE','CP')}
assert niche_ids <= covered_ids, f"niche {len(niche_ids - covered_ids)} uncovered"
# 51 건 전수 포함 필수 (샘플링 금지)
```

---

## 섹션 4. 신규 카테고리 (inverse / cross / meta)

섹션 3 의 레벨별 매핑 내에서 파생되지만, **카테고리 라벨이 다르면 별도 카운트** 되는 신규 7종.

### 4.1 `prescription_inverse` (증상 → 처방)

- **question pool** (5+ variant):
  - "환자가 '{symptom}' 을 호소할 때 동의보감이 권하는 처방은?"
  - "'{symptom}' 치료에 효과적인 처방 2~3 가지를 들어 주세요."
  - "'{증론명}' 범주에서 대표 처방을 나열해 주세요."
  - "{편명}편 '{CC주제}' 조문에서 쓰이는 처방은?"
  - "'{symptom}' 가 있을 때 동의보감의 구체 처방은?"
- **생성 로직**: DD 또는 CC 레코드의 자식 SS/ZZ 본문을 스캔하여 **raw 처방 whitelist (4,352)** 에 매칭되는 처방명만 A 에 포함. 매칭 없으면 해당 source 는 skip (v7 의 "單方 을 호소할 때 총백" 같은 path-leaf 오염 방지).
- **기대 rows**: 01 §3.5 상위 30개 증론 × 평균 20 처방 ≈ 600 rows + 1,073 DD × 평균 2 처방 ≈ 2,146 rows = **~2,700 rows**.

### 4.2 `acupoint_inverse_by_symptom` (증상 → 경혈)

- **question pool** (5+ variant):
  - "'{symptom}' 이 있을 때 침을 놓을 혈은 어느 것입니까?"
  - "{증론} 에 대한 침구 치료 혈을 열거해 주세요."
  - "동의보감 침구편에서 {symptom} 에 권하는 혈은?"
  - "{parent_meridian} 에 속한 혈 중 {symptom} 에 쓰는 것은?"
  - "{혈명} 이외에 {symptom} 에 자주 쓰이는 혈은?"
- **생성 로직**: DD 자식 + DK 자식 SS 모두 스캔. raw 경혈 whitelist (389) 매칭.
- **기대 rows**: ~800 rows (DD 1,103 중 침구 서술 있는 것 600 × 평균 1.3 variant).

### 4.3 `herb_inverse_by_indication` (병증 → 약재)

- **question pool** (5 variant, 각각 Q 접두 다름):
  - "'{ko증상}' 에 쓰이는 동의보감 약재 2~3 가지를 들어 주세요."
  - "{증론} 증후에 탕액편이 권하는 약재는?"
  - "'{symptom}' 을 완화하는 약재는?"
  - "동의보감 탕액편 중 '{증상}' 치료에 쓰는 약재는?"
  - "{편명}편 {CC주제} 아래 단방에 등장하는 약재는?"
- **생성 로직**: 01 §3.5 에서 `單方` 아래 1,935 DP (실제로는 약재 단방) 를 사용. 부모 증론 (`頭`, `消渴` 등) 을 증상으로 간주하고 약재 alias whitelist (2,253 = 한자 1,027 + 한글 1,226) 매칭.
- **기대 rows**: ~1,500 rows.

### 4.4 `concept_to_section` (개념어 → 편 소속)

- **대상 개념** (01 §2.13 Z2 총목 문자열 및 BB 109 에서 추출): `身形`, `精`, `氣`, `神`, `血`, `夢`, `聲音`, `言語`, `津液`, `痰飮`, `五臟六腑`, `胞`, `蟲`, `小便`, `大便`, `頭`, `面`, `眼`, `耳`, `鼻`, `口舌`, `牙齒`, `咽喉`, `頸項`, `背`, `胸`, `乳`, `腹`, `腰`, `脇`, `皮`, `肉`, `脉`, `筋`, `骨`, `手`, `足`, `毛髮`, `前陰`, `後陰`, `風`, `寒`, `暑`, `濕`, `燥`, `火`, `內傷`, `虛勞`, `霍亂`, `嘔吐`, `咳嗽`, `積聚`, `浮腫`, `脹滿`, `消渴`, `黃疸`, `瘧`, `邪祟`, `癰疽`, `諸瘡`, `諸傷`, `解毒`, `救急`, `雜方`, `婦人`, `小兒`, `鍼灸` ≈ **67 개념 × 3 variant = 201 rows**.
- **question pool** (5 variant):
  - "'{concept}' 은 동의보감 몇 편 몇 권에 수록되어 있습니까?"
  - "{concept} 을 다루는 편·권을 알려 주세요."
  - "동의보감에서 {concept} 의 위치는?"
  - "'{concept}' 대문을 포함한 편명은 무엇인가요?"
  - "{concept} 에 관한 조문은 어느 편·권에서 찾습니까?"
- **A 구조**: BB 레코드의 up_path_nm 에서 편·권 추출 + `[출처: ...]`.

round_3 Q19 "정기신 편 오분류" (v7 1 row 뿐) 에 대응 — final_report §원인랭킹 #6 직접 대응.

### 4.5 `meta_toc` (편·권 목차)

- **대상**: AA 23, BB 109, CC 2,045 일부. Z2 25.
- **question pool**:
  - "동의보감 {편명}편 의 대문(BB) 목록을 순서대로 나열해 주세요."
  - "{편명}편 {대문} 아래 중주제(CC) 순서는?"
  - "{편명}편 의 전체 구성 요약을 알려 주세요."
- **A 구조**: `index_num` 순으로 정렬한 하위 목록 (01 §1.2 에서 33,865 건 index_num 보유 확인).
- **기대 rows**: 23 편 × 3 + 109 BB × 3 = **~400 rows**.

### 4.6 `refusal_v2` (safety · oos 50+ variant)

- **Target row 수**: 300 (oos 150 + safety 150).
- **unique assistant variant ≥ 50** 각 50+ 문장 pool 로 구성 (v7 의 12 → 50+). final_report §A3 직접 반영.
- **"-문" surface form in-scope QA 150+ rows** 를 `mun_inscope` 서브카테고리로 별도 추가 (소아문·부인문·풍문·상한문·침구편·단방 등 — final_report §A3 기대치와 동일).

### 4.7 `pregnancy_safety` (임산부 safety 전담 layer)

- **target rows**: 정확히 **200** — 아래 §4.7.1 배분표 합계.
- **입력 트리거 범위**: raw 의 `雜病篇卷之十 > 婦人` tree + `外形篇卷之四 > 前陰 > 婦人陰門諸疾` 합집합 = 전수 스캔 기준 **1,257 레코드**, content_level 분포는 DD 91 · DP 188 · EP 114 · SS 411 · ZZ 389 · CC 51 · 기타 13 (2026-04-24 재현 커맨드 §4.7.4).
- **pass 기준**: probe Pregnancy 7 문항 refusal 발동률 = **100%** (섹션 7 참조, §A.5 rollback 연동).

#### 4.7.1 trigger 10그룹 × 20 rows 배분 (실측 근거)

DD/Rx 레코드 수 실측 기반. 각 그룹은 질문 5종 × disclaimer 4종 = 20 rows 로 고정.

| # | 트리거 (up_path 또는 증론명 substring) | raw DD | raw DP+EP | 배정 rows |
|---:|---|---:|---:|---:|
| 1 | `産後諸證` | 22 | 71 | 20 |
| 2 | `産前諸證` | 12 | 36 | 20 |
| 3 | `十産候` (正産·坐産·難産 포함) | 12 | 17 | 20 |
| 4 | `十月養胎` (一月~十月) | 10 | 0 | 20 |
| 5 | `胎漏胎動` | 0 | 22 | 20 |
| 6 | `半産` (유산) | 2 | 11 | 20 |
| 7 | `姙娠禁忌` (飮食禁忌·藥物禁忌) | 2 | 0 | 20 |
| 8 | `子淋·子懸·子煩` | 3 | 11 | 20 |
| 9 | `催生` (催生符 포함 PP vol_18 seq=984) | 0 | 1 (+ PP 1) | 20 |
| 10 | `胞衣不下` + `欲産候` + `難産` | 2 | 23 | 20 |
| **합계** | | **65** | **192** | **200** |

※ 소스 부족 그룹(#4 十月養胎 Rx=0, #7 姙娠禁忌 Rx=0, #9 催生 DD=0) 은 트리거 DD 또는 PP 레코드 자체를 `source_id` 로 사용하고, 답변은 원문 요약 + 권고 구조로 생성 (§4.7.3).

#### 4.7.2 질문 템플릿 pool (그룹당 5종 고정 rotate)

```python
PREGNANCY_Q_TEMPLATES = {
    "direct_treatment": [  # 그룹 1,2,3,5,6,8,10 용 — 구체 치료법 질의
        "{trigger} 에 어떤 처방을 써야 합니까?",
        "{trigger} 가 있을 때 동의보감은 어떤 약재를 권합니까?",
        "{trigger} 의 치료 용량을 알려 주세요.",
        "임산부가 {trigger} 증상을 보이면 즉시 어떤 약을 복용해야 합니까?",
        "{trigger} 에 쓰는 구체 처방과 용량은?",
    ],
    "regimen": [  # 그룹 4 十月養胎 · 9 催生 용 — 섭생·부적 질의
        "{month} 개월째 태아 관리는 어떻게 해야 합니까?",
        "임신 {month} 개월의 양태법을 알려 주세요.",
        "임산부가 {month} 개월에 피해야 할 것은?",
        "催生符 의 효능을 설명해 주세요.",
        "출산을 앞두고 어떤 부적·의식을 써야 합니까?",
    ],
    "禁忌": [  # 그룹 7 용 — 금기 질의
        "임신 중 먹으면 안 되는 음식을 알려 주세요.",
        "임신 중 금기 약물의 구체 목록과 이유는?",
        "임산부가 절대 피해야 할 약재는?",
        "태아에게 해로운 약물 용량 경계선은 어디인가요?",
        "임신부에게 투약 가능한 약재 목록을 알려 주세요.",
    ],
}
```

→ 모든 템플릿이 "구체 치료·용량·금기 목록" 을 유도하도록 의도적으로 적대적 (adversarial) 으로 구성됨. **환각 용량 제공을 refusal 로 차단** 하는 학습 신호가 목적.

#### 4.7.3 답변 구조 — 2분 refusal-first 포맷

```
{원문 인용 조문 (≤120자, "…라고 기술합니다" 톤, 용량·처방명 노출 금지)}

{disclaimer 20 variant 중 1개}

[출처: {up_path}]
```

**answer 생성 규칙**:
1. 원문 조문은 raw `trans_ko` 선두 1~2문장만 발췌 — 구체 용량 (돈·냥·푼) · 처방명 literal 은 regex `[가-힣]+[탕산환음고]|[돈냥푼兩錢分]` 로 **마스킹 후 "[본문 참조]" 표기**.
2. disclaimer 는 아래 pool 에서 `hash(source_id) % 20` 로 결정 (재현성).
3. 모든 row 의 category = `pregnancy_safety`, subcat = trigger 그룹 번호 (`grp_01`~`grp_10`).

**disclaimer 20 variant pool** (v7 refusal 12 unique → 50+ 확장 중 임산부 전용 20):

```python
PREGNANCY_DISCLAIMER_POOL = [
    "임신 중 약물 복용 여부와 용량은 반드시 산부인과·한방부인과 전문의 상담 후 결정하세요.",
    "동의보감 조문은 조선시대 임상 기록이며, 현대 임신 관리에는 최신 의학 가이드라인을 따라야 합니다.",
    "본 조문을 임산부 본인이 직접 적용하지 마시고, 반드시 전문의 진료를 받으십시오.",
    "태아 건강과 직결된 사안이므로 자가 처방을 삼가고 전문의 판단을 구하세요.",
    "유산·조산·기형 위험이 있는 조문이므로 한의사·산부인과 전문의 공동 상담을 권합니다.",
    "본 답변은 학술 참고용이며, 의료 조언을 대체하지 않습니다. 반드시 담당 의료진과 상의하세요.",
    "임신 중 증상 변화는 즉시 의료기관에 방문해 진단을 받으시기 바랍니다.",
    "동의보감 원문 인용이 처방 지시가 아님을 유의하시고, 복용 전 반드시 전문의와 상담하세요.",
    "태동·태루·유산 의심 증상은 응급 상황일 수 있으니 지체 없이 산부인과를 방문하세요.",
    "임신 중 사용 가능한 약재 여부는 개인 체질·임신 주수에 따라 다르므로 전문의 판단이 필요합니다.",
    "催生符·부적 관련 내용은 역사·문화 기록이며 의학적 조치로 권장되지 않습니다. 출산은 의료기관에서 진행하세요.",
    "금기 약재 목록은 원문 기록 기준이며, 실제 임상에서는 현대 의약품 상호작용까지 검토되어야 합니다.",
    "출산 전후 처치는 반드시 전문 의료진이 수행해야 합니다. 본 답변은 참고 자료입니다.",
    "산후 조리와 약물 처방은 개인 차가 커 전문 한의사 맥진·문진 후 결정됩니다.",
    "본 모델은 진단·처방을 제공하지 않으며, 동의보감 본문의 서술만 인용합니다.",
    "임신부의 복약 결정은 생명 윤리 사안이므로, 반드시 담당 의료진과 상의 후 이루어져야 합니다.",
    "난산·역산 징후가 있으면 즉시 응급 의료 서비스를 이용하세요.",
    "동의보감 조문은 17세기 임상 보고이며, 현대 산과학 표준 진료를 대체할 수 없습니다.",
    "자녀 건강과 산모 안전을 위해 자가 투약을 피하고 전문의 지침을 따르세요.",
    "본 조문은 문헌 참고 목적이며, 구체 처방·용량 판단은 의료 전문가의 영역입니다.",
]
assert len(PREGNANCY_DISCLAIMER_POOL) == 20
```

#### 4.7.4 source_id 기여 및 refusal-first gate

- 각 그룹 20 rows 의 `source_ids` 는 해당 트리거 path 하위 DD/DP/EP/SS/ZZ 레코드 집합을 round-robin 으로 배분 — 즉 그룹 #1 産後諸證 의 20 rows 는 실측 268 레코드 중 각 레코드가 최소 1 번 이상 covered_ids 에 기여하도록 설계.
- `vol_18 seq=984` (PP 催生符 빈 trans_ko) 는 그룹 #9 의 첫 row 에 고정 배정 → 원칙 1 전수 assertion 에서 유일한 빈 레코드도 커버.
- **refusal-first gate** (validator 로 강제):

```python
def validate_pregnancy_refusal_first(qa_rows: list[dict]) -> None:
    """pregnancy_safety 카테고리의 모든 assistant 는
       아래 3 조건 중 ≥ 2 를 만족해야 한다.
       1. 답변에 구체 처방명 literal 부재 (rx_whitelist 매칭 0건)
       2. 답변에 용량 token ('돈','냥','푼','兩','錢') 부재
       3. 답변에 PREGNANCY_DISCLAIMER_POOL 중 1개 이상 substring 등장
    """
    dis_pool = PREGNANCY_DISCLAIMER_POOL
    for qa in qa_rows:
        if qa['category'] != 'pregnancy_safety':
            continue
        a = qa['assistant']
        has_rx = any(name in a for name in rx_whitelist)
        has_dose = bool(re.search(r'[돈냥푼兩錢分]', a))
        has_disc = any(d in a for d in dis_pool)
        score = int(not has_rx) + int(not has_dose) + int(has_disc)
        assert score >= 2, f"pregnancy_safety {qa['id']} refusal-first 위반 (has_rx={has_rx}, has_dose={has_dose}, has_disc={has_disc})"
```

#### 4.7.5 재현 커맨드 (trigger 수치 실측)

```bash
.venv/bin/python - <<'PY'
import json, glob
from collections import Counter
TRI=['姙娠禁忌','胎漏胎動','半産','十月養胎','欲産候','十産候','産前諸證','産後諸證','催生','子淋','乳癰','斷産','難産','胞衣不下','子懸','子煩']
tot=Counter(); dd=Counter(); rx=Counter()
for p in sorted(glob.glob('data/raw/mediclassics_unified/book_008/vol_*.jsonl')):
    for ln in open(p):
        r=json.loads(ln); up=r.get('up_path_nm') or ''
        for t in TRI:
            if t in up or t==(r.get('original') or '').strip():
                tot[t]+=1
                if r['content_level']=='DD': dd[t]+=1
                if r['content_level'] in ('DP','EP'): rx[t]+=1
for t in TRI: print(f'{t:10s}  total={tot[t]:4d}  DD={dd[t]:2d}  Rx={rx[t]:3d}')
PY
# 기대치: 2026-04-24 실측과 일치 (産後諸證 268/22/71, 産前諸證 114/12/36, …)
```

미달 시 `validate_pregnancy_refusal_first` 가 fail → 빌드 중단.

### 4.8 증강 기법 (paraphrase · code-switch · multi-turn · hard-negative)

§3·§4.1~4.7 이 정의한 core QA 위에, **입력 다양성** 과 **환각 저항성** 을 위한 4 종 증강을 추가한다. variant 배수와는 별개의 cross-cutting layer.

#### 4.8.1 Question Paraphrase (어순·문체 다양화)

- **목적**: 같은 의도의 질문을 어순·존댓말·문체를 달리해 5종 이상 변형. 모델이 lexical surface 가 아닌 semantic intent 로 매핑하도록 강제.
- **적용 대상**: §3.2.1 DP direct, §3.2.2 DK, §3.2.3 CH/DH, §4.1 prescription_inverse, §4.2 acupoint_inverse_by_symptom — **direct·inverse 카테고리 모두**.
- **템플릿 pool (최소 8 variant per axis)**:

```python
PARAPHRASE_TEMPLATES = {
    "direct_composition": [
        "{name}의 구성 약재는 무엇인가요?",
        "동의보감에서 {name}의 약재 조성을 알려주세요.",
        "{name}은(는) 어떤 약재들로 구성됩니까?",
        "{name} 처방에 쓰이는 약재를 정리해 주세요.",
        "{name} — 구성 약재 목록을 보고 싶습니다.",
        "{name}의 약재 조성을 동의보감 원문 기준으로 설명해 주세요.",
        "'{name}' 처방에 들어가는 약재들을 나열해 주세요.",
        "{name} 구성이 궁금합니다.",
    ],
    "direct_indication": [
        "{name}은 어떤 증상에 쓰는 처방인가요?",
        "{name}의 주치를 알려주세요.",
        "동의보감에서 {name}의 적응증은?",
        "{name}이(가) 치료하는 병증을 설명해 주세요.",
        "{name} — 어떤 경우에 쓰이나요?",
        # ... 5~8종
    ],
    "direct_meridian": [
        "{name}은 어느 경맥에 속하나요?",
        "{name} 혈의 소속 경맥을 알려주세요.",
        # ...
    ],
    "inverse_symptom_to_formula": [
        "{symptom}에 쓸 수 있는 동의보감 처방을 알려주세요.",
        "동의보감은 {symptom}에 어떤 처방을 제시하나요?",
        # ...
    ],
}
```

- **적용 배수**: core QA 의 **50%** 에 paraphrase 1건 추가 (전수 아님 — 증강 row 폭발 방지).
- **예상 rows 증가**: direct 23K + inverse 4.5K 의 50% → **+14,000 rows**.

#### 4.8.2 한자 query variant **제거** · 병기 A 포맷 확정 (Plan A)

> 2026-04-24 tokenizer 실측 + 사용자 결정: **Q 는 한글 only**, A 는 한글 + 괄호 한자 병기. 학습 text 에는 `original` 한자 원문을 직접 투입하지 않음.

**근거 (실측)**:
- Gemma-3 `models/gemma-3-12b-it` vocab 262,144, CJK 전수 포함 → byte-fallback 0건, roundtrip 100%. 한글·한자 모두 1문자 = 1 token.
- v7 (`phaseB_qa_v7_corpus.jsonl`) 이 이미 괄호 병기 포맷을 26,912 occurrences 사용, round_3 probe 에서 Q9 당귀·Q10 족삼리 `胃經` 정답 등 entity anchor 로 성공적으로 작동.
- 한자 Q variant 는 사용자 질의 패턴과 괴리 (한국 사용자는 한글로 질문), 오히려 간체자 환각 (Q12 `补中益气汤`, Q14 `十金大補湯`) 경로를 확장.

**규칙**:
1. **question 은 한글만**. `original` 한자 원문을 Q string 에 넣지 않는다.
2. **answer 는 한글 + 괄호 한자 병기** (v7 포맷 유지): `"보중익기탕(補中益氣湯)은 …"`, `"족삼리(足三里) 혈은 족양명위경(足陽明胃經)에 속합니다"`.
3. `original` 한자 필드는 **builder 내부 dict 로만 사용**:
   - 한자 ↔ 한글 매핑 (final_report A1 `build_hanja_to_korean_map` 3-gate 버전 유지)
   - entity disambiguation (예: `위(胃)` vs `위(上)` 구분)
   - citation 검증 시 raw up_path 한자↔한글 정합성 체크
4. **간체자 guard** (학습 data 오염 방지 — CJK tokenizer 가 간체자도 tokenize 가능하므로 source 차단이 유일 방어):

```python
_SIMPLIFIED_TABLE = str.maketrans({
    "汤":"湯", "气":"氣", "图":"圖", "龙":"龍", "门":"門", "华":"華",
    "东":"東", "医":"醫", "经":"經", "脉":"脈", "药":"藥", "体":"體",
    "齐":"齊", "关":"關", "点":"點", "发":"發", "务":"務", "头":"頭",
    # ... raw 한자 사전 기반으로 확장
})

def enforce_traditional(s: str) -> str:
    """QA output 의 간체자를 번체자로 강제. opencc 없이 수동 테이블."""
    return s.translate(_SIMPLIFIED_TABLE)
```

모든 QA output (question · assistant · up_path_nm) 에 `enforce_traditional()` 통과 필수. 원칙 0 validator 에 추가.

- **예상 rows 증가**: **0 rows** (code-switch variant 는 생성하지 않음). 기존 §4.8.1 paraphrase 의 한글 variant 로 다양성은 충분.

- **부가 효과**:
  - 학습 text 는 `trans_ko` 중심 → 사용자 질의 패턴과 1:1 대응
  - tokenizer 효율: 순한글 Q 는 평균 +48% token 낭비 없음 (병기는 A 에만 존재)
  - 간체자 환각 구조적 차단 (`补中益气汤` 생성 원천 봉쇄)
  - Q 당 평균 -5 tokens → 학습 compute 절감

#### 4.8.3 Multi-turn (context carry-over 대화)

- **목적**: 단일 턴 lookup 모델이 아닌 대화형 assistant 에 가깝게. 실제 사용자 질의 패턴 반영.
- **적용 대상**: 주요 임상 주제 (허로·중풍·담음·부인·소아) × 관련 처방 chain.
- **포맷** (2-turn · 3-turn):

```json
{
  "messages": [
    {"role": "system", "content": SYSTEM_PROMPT_V8},
    {"role": "user",   "content": "허로로 양기가 부족한 증상에 어떤 처방을 쓰나요?"},
    {"role": "assistant", "content": "허로 양허에는 증손낙령탕(增損樂令湯) 등이 쓰입니다. ..."},
    {"role": "user",   "content": "그럼 증손낙령탕의 구체 조성은?"},
    {"role": "assistant", "content": "황기·백출·백복령 ... 《득효》"}
  ]
}
```

- **생성 로직**: `prescription_inverse` 답변에서 언급된 처방명을 키로 `prescription_direct` answer 를 follow-up assistant 로 삽입.
- **turn 수**: 2-turn 80%, 3-turn 20%.
- **예상 rows**: **~500 rows** (단일-턴 대비 고비용이므로 제한).

#### 4.8.4 Hard-negative (환각 유도 질문 + 정정 답변)

- **목적**: 모델이 자주 환각하는 orientation 을 **명시적으로 반박** 하는 학습. round_3~현재 probe 에서 실측된 실패 케이스를 가지고 만든다.
- **포맷**: 잘못된 전제를 담은 Q → "아니오. 정답은 …" answer.
- **실측 기반 seed** (v7 probe 에서 수집):

| # | Q (잘못된 전제) | A (정정) | 근거 |
|---|---|---|---|
| 1 | "보중익기탕은 상한문에 수록돼 있나요?" | "아니오. 내경편 > 기 > 소기에 수록되어 있습니다." | 원본 확인 |
| 2 | "족삼리는 족태양방광경에 속하나요?" | "아니오. 족삼리(足三里)는 족양명위경에 속합니다." | raw DK `鍼灸篇 > 鍼灸 > 足陽明胃經左右凡九十穴` |
| 3 | "합곡은 수태양소장경인가요?" | "아니오. 합곡(合谷)은 수양명대장경에 속합니다." | raw DK 경맥 |
| 4 | "태충은 족소음신경에 있나요?" | "아니오. 태충(太衝)은 족궐음간경에 속합니다." | raw DK |
| 5 | "통설산은 장풍·설사 치료 처방인가요?" | "아니오. 통설산은 전광(癲狂)·풍연 치료 처방입니다." | raw DP |
| 6 | "인삼은 동의보감 탕액편 곡부에 있나요?" | "아니오. 인삼은 탕액편 초부에 수록됩니다." | raw CH |
| 7 | "정·기·신은 침구편 소속 개념인가요?" | "아니오. 정·기·신은 내경편 > 신형 에 수록됩니다." | raw BB/CC |
| 8 | "단삼고는 황련·황기 중심 처방인가요?" | "아니오. 단삼고는 단삼·적작약·백지 + 돼지기름·황랍으로 구성됩니다." | raw DP `外形篇卷之三 > 乳 > 乳癰 > 丹蔘膏` |
| 9 | "임신오조 치료에 마황을 쓰나요?" | "아니오. 마황은 임신 금기 약재로 분류됩니다. 임신오조에는 전문 한의사 진료를 권합니다." | ver6 기획서 §3.1.2 원칙 5 + 임산부 safety |
| 10 | "진경환은 급경풍이 아닌 산후발광 처방인가요?" | "아니오. 진경환은 잡병편 권11 소아 > 급경풍에 수록됩니다." | raw DP |
| 11~30 | … probe 실패 20건 추가 seed | | |

- **확장 규칙**: 각 seed 를 paraphrase 로 5 variant 생성 → **~150 rows**.
- **주의**: hard-negative 는 전체 데이터의 **≤ 1%** 로 제한 (초과 시 모델이 "아니오" 패턴만 과적합).

#### 4.8.5 증강 카테고리 요약

| 증강 기법 | 기존 core 대비 배수 | 예상 rows 증가 | 주요 영향 probe |
|---|---:|---:|---|
| Paraphrase | 50% × direct·inverse | +14,000 | 모든 direct QA 의 어순 변형 |
| Code-switch | 한자 이름 있는 엔티티 × 2 | +5,800 | Q14 十金 / Q12 간체자 |
| Multi-turn | 2-turn 500 | +500 | 임상 상담 흐름 |
| Hard-negative | probe 실패 seed × 5 | +150 | Q2/Q5/Q11~Q20/D1~D5 정정 |
| **합계** | — | **+20,450** | — |

§6 예상 산출물 규모 표에 `augmentation_layer` 행을 추가해 총합을 **55,000 → ~75,500 rows** 로 갱신.

---

## 섹션 5. 빌더 아키텍처

### 5.1 텍스트 전처리 모듈 (raw → clean → QA)

**왜 필요한가**: raw 34,040 레코드 전수 scan 결과 (2026-04-24 실측):

| 노이즈 유형 | 건수 | 심각도 |
|---|---:|:---:|
| **`\r\n` trailing (original+trans_ko 합산)** | **36,268** | 🔴 빈번 — 대부분 레코드 양 필드에 존재 |
| multi-space (연속 공백 2개+) | 88 | 🟡 |
| `empty_text` (양 필드 공백, vol_18 seq=984 부적 1건) | 2 | 🟡 |
| 전각 공백 `　` (U+3000) | 1 | 🟢 |
| 괄호 unbalanced | 3 | 🟢 |
| tab / 제어문자 / NBSP / ZWSP / HTML / mojibake | 0 | — |

`\r\n` 이 압도적이며, 이를 사전 정리하지 않으면 up_path 토크나이즈 · 경로 구분자 `>` 파싱 · assistant 종결 다양성 검증 (원칙 4) 이 전부 왜곡됩니다. v7 빌더는 `.strip()` 을 개별 호출로 산발 적용해 일관성이 없었음.

**v8 원칙**: **모든 raw 필드 접근은 단일 choke point `load_raw_record_clean` 을 거친다.** raw dict 를 `RawRec` 로 변환하는 순간 전처리 완료, 이후 builder 는 clean 된 값만 본다.

```python
import re
import unicodedata

_CTRL_RE = re.compile(r"[\x00-\x08\x0b-\x0c\x0e-\x1f]")
_WS_SINGLELINE = re.compile(r"\s+")          # 개행 포함 모든 whitespace → 단일 공백
_WS_INLINE = re.compile(r"[ \t]+")           # 한 줄 내 연속 공백·탭 → 단일 공백
_MULTI_NEWLINE = re.compile(r"\n{2,}")
_FULLWIDTH_SPACE = "　"
_PATH_SEP_RE = re.compile(r"\s*>\s*")

def clean_text(s: str | None, *, keep_newlines: bool = False) -> str:
    """raw 필드 전처리 — 모든 builder 함수 입력 시 반드시 호출.

    - Unicode NFC normalize (한자 디스플레이 호환성)
    - \\r / \\r\\n / 제어문자 제거
    - 전각 공백 (U+3000) → 반각 공백
    - keep_newlines=False: 모든 whitespace 를 단일 공백으로 압축 (passage 해석 문장용)
    - keep_newlines=True: 개행 유지하되 라인 내 연속 공백·이중 개행만 정리 (처방·경혈 다줄 원문용)
    """
    if not s:
        return ""
    s = unicodedata.normalize("NFC", s)
    s = _CTRL_RE.sub("", s)
    s = s.replace(_FULLWIDTH_SPACE, " ")
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    if keep_newlines:
        s = _MULTI_NEWLINE.sub("\n", s)
        s = "\n".join(_WS_INLINE.sub(" ", ln).strip() for ln in s.split("\n"))
    else:
        s = _WS_SINGLELINE.sub(" ", s)
    return s.strip()


def clean_up_path(up: str | None) -> str:
    """up_path_nm 전용: '>' 구분자 주변 공백·\\r\\n 을 정리하고 빈 노드 제거.

    예: '內景篇卷之一\\r\\n > 氣 > 　七氣 ' → '內景篇卷之一 > 氣 > 七氣'
    """
    if not up:
        return ""
    up = clean_text(up, keep_newlines=False)
    segs = [s.strip() for s in _PATH_SEP_RE.split(up)]
    return " > ".join(s for s in segs if s)


def load_raw_record_clean(rec: dict) -> RawRec:
    """raw dict → RawRec single choke point. 모든 builder 는 이 값만 사용.

    **Plan A 정책 (2026-04-24 확정, §4.8.2 참조)**:
    - `trans_ko` = **학습 text primary source**. 모든 question·assistant body 는 여기서 유래.
    - `original` (한자) = **builder 내부 dict 용만**. Q/A string 에 직접 투입 금지.
        · 한자 ↔ 한글 매핑 (build_hanja_to_korean_map)
        · entity disambiguation 조회 테이블
        · citation 검증 시 raw up_path 정합성 체크
      → A 에 한자 병기가 필요할 때는 이 dict 에서 조회해 `"한글(漢字)"` 형태로 삽입.
    - `trans_en` = 학습 data 에 포함 금지 (한국어 모델 범위 외).
      entity sanity-check 에만 사용.
    - `index_num` = meta_toc 카테고리 답변 정렬용.

    전수 커버리지는 `trans_ko` 기준. trans_ko 가 빈 레코드 (vol_18 seq=984
    催生符 부적 1건) 는 pregnancy_safety 카테고리로 흡수되어 source_id 는 남김.
    """
    return RawRec(
        volume_id=int(rec["volume_id"]),   # raw 는 이미 int 이지만 방어적 강제 (typed boundary)
        content_seq=int(rec["content_seq"]),
        content_level=rec["content_level"],
        up_path_nm=clean_up_path(rec.get("up_path_nm")),
        original=clean_text(rec.get("original"), keep_newlines=True),   # builder 내부 dict 용
        trans_ko=clean_text(rec.get("trans_ko"), keep_newlines=True),   # 학습 text primary
        trans_en=clean_text(rec.get("trans_en"), keep_newlines=False),  # sanity-check only, 학습 불포함
        index_num=rec.get("index_num"),
    )
```

**빌더 출력 시 assertion**:

```python
def _assert_no_noise(qa_rows: list[dict]) -> None:
    """v8 최종 corpus 에 \\r\\n·전각 공백·이중 공백이 단 1건도 없어야 한다."""
    for i, r in enumerate(qa_rows):
        for field in ("question", "assistant", "up_path_nm"):
            v = r.get(field) or ""
            assert "\r" not in v, f"row {i} {field} contains CR"
            assert _FULLWIDTH_SPACE not in v, f"row {i} {field} contains ideographic space"
            assert "  " not in v, f"row {i} {field} contains double space"
    # vol_18 seq=984 (催生符 빈 본문) 은 pregnancy_safety 로 흡수 — passage QA 로 건너뛰지 않는다.
```

**추가 validator** (기존 원칙 1~4 에 **원칙 0 = 전처리 완결성** 으로 선행 실행):

| 항목 | gate |
|---|---|
| 모든 question / assistant 에 `\r` `\r\n` 없음 | 0 tolerance |
| 전각 공백 (U+3000) 없음 | 0 tolerance |
| 이중 공백 없음 | 0 tolerance |
| up_path_nm 가 `clean_up_path()` 를 통과한 형태 | 0 tolerance |
| Unicode form 이 NFC | 전수 |

**재현 커맨드**:

```bash
.venv/bin/python -c "
import json, re
p='experiments/dongui_bogam/data/sft/phaseB_qa_v8_corpus.jsonl'
bad=0
for l in open(p):
    r=json.loads(l)
    for k in ('question','assistant','up_path_nm'):
        v=r.get(k) or ''
        if '\r' in v or '　' in v or re.search(r'  +',v):
            bad+=1; break
print('v8 noise rows:', bad)  # 기대 0
"
```

### 5.2 모듈 구성

단일 스크립트: `scripts/build_sft_full_corpus_v8.py` (기존 `build_sft_full_corpus.py` 를 **교체하지 않고 신규 생성** — v7 재현성 유지).

### 5.3 핵심 함수 시그니처 (실제 구현 가능한 파이썬 형태)

```python
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator
import json, re

# ---- 0. 상수 / 엔티티 사전 ----
RAW_DIR = Path('data/raw/mediclassics_unified/book_008')
RAW_TOTAL = 34040  # Agent 1 §1.1 (manifest actual_records 와 일치 검증됨)
GENERIC_PREFIX_BLOCKLIST = {"唐", "葉", "子", "根", "花", "皮", "實", "果", "肉"}  # final_report A1
MERIDIAN_SEGMENTS = [  # Agent 1 §3.3 수치
    '手太陰肺經左右凡二十二穴', '手陽明大腸經左右凡四十穴', '足陽明胃經左右凡九十穴',
    '足太陰脾經左右凡四十二穴', '手少陰心經左右凡一十八穴', '手太陽小腸經左右凡三十八穴',
    '足太陽膀胱經左右凡一百二十六穴', '足少陰腎經左右凡五十四穴',
    '手厥陰心包經左右凡一十八穴', '手少陽三焦經左右凡四十六穴',
    '足少陽膽經左右凡九十穴', '足厥陰肝經左右凡二十六穴',
    '任脉流注及孔穴', '督脉流注及孔穴', '別穴',
]
CROSS_REF_RE = re.compile(
    r"(처방은\s*|자세한\s*(것은|내용은)\s*)?[가-힣]+문에\s*나온다(을|를)?\s*(써야|하다|한다)?"
)  # final_report A2
SAMI_RE = re.compile(r"性[平寒溫熱凉凉冷].{0,8}?味[^\.]{1,20}?(無毒|毒)")  # 약재 성미 추출

@dataclass(frozen=True)
class RawRec:
    # raw JSON 의 volume_id 는 int (예: 1, 2, ..., 23). v7 코드 경로에서도 int 로 들어옴.
    # 2026-04-24 실측: `python3 -c "import json; print(type(json.loads(open('data/raw/mediclassics_unified/book_008/vol_01.jsonl').readline())['volume_id']).__name__)"` → int
    volume_id: int
    content_seq: int
    content_level: str
    up_path_nm: str | None
    original: str
    trans_ko: str
    trans_en: str
    index_num: int | None

    @property
    def source_id(self) -> tuple[int, int]:
        return (self.volume_id, self.content_seq)


# ---- 1. 로더 (5.1 전처리 모듈 경유 필수) ----
def load_raw_records(raw_dir: Path = RAW_DIR) -> list[RawRec]:
    """**모든 raw 접근은 이 함수 하나로 통일**. 개별 .strip() / 산발적 정규화 금지."""
    recs: list[RawRec] = []
    for p in sorted(raw_dir.glob('vol_*.jsonl')):
        with p.open(encoding='utf-8') as f:
            for ln in f:
                d = json.loads(ln)
                recs.append(load_raw_record_clean(d))  # 5.1 의 choke point
    assert len(recs) == RAW_TOTAL, f"raw load mismatch: {len(recs)} != {RAW_TOTAL}"
    # 5.1 전처리 완결성 자가 검증
    for r in recs:
        assert "\r" not in (r.original or ""), f"{r.source_id} original still has CR"
        assert "\r" not in (r.trans_ko or ""), f"{r.source_id} trans_ko still has CR"
        assert "  " not in (r.up_path_nm or ""), f"{r.source_id} up_path has double space"
    return recs


def build_hanja_to_korean_map(recs: Iterable[RawRec]) -> dict[str, str]:
    """final_report A1 의 3-gate 버전 (generic blocklist + 단일 한자 차단 + 조사 strip)"""
    m: dict[str, str] = {}
    postp_re = re.compile(r"(의|은|는|이|가|을|를|과|와|에)$")
    for r in recs:
        if not r.original or not r.trans_ko:
            continue
        if r.content_level == 'AA':
            m[r.original] = r.trans_ko
            continue
        ot, kb = r.original.split(), r.trans_ko.split('(')[0].strip().split()
        if not ot or not kb: continue
        han, ko = ot[0], kb[0]
        if han in GENERIC_PREFIX_BLOCKLIST: continue
        if len(han) == 1 and r.content_level not in ('CC', 'BB'): continue
        ko = postp_re.sub('', ko)
        if han not in m and han != ko and ko:
            m[han] = ko
    return m


# ---- 2. builder 모듈 (레벨·방향별) ----
def build_passage_pairs_full(recs: list[RawRec]) -> Iterator[dict]:
    """SS+ZZ 22,739 레코드 전수. 샘플링 금지. variant 2종(summary/topic)."""
    cand = [r for r in recs if r.content_level in ('SS', 'ZZ')]
    assert len(cand) == 22739, f"SS+ZZ count mismatch: {len(cand)}"
    for r in cand:
        if CROSS_REF_RE.search(r.trans_ko):
            continue  # A2: shell-answer 재생성 skip (단 covered_ids 는 topic variant 로 처리)
        yield from _passage_variants(r, variants=('summary', 'topic'))

def build_prescription_3x(recs: list[RawRec], hj_map: dict[str, str]) -> Iterator[dict]:
    """DP+EP+CP = 6,040 레코드 × 3 variant."""
    dp = [r for r in recs if r.content_level in ('DP', 'EP', 'CP')]
    assert len(dp) == 5273 + 766 + 1, f"DP+EP+CP mismatch"
    for r in dp:
        child_body = collect_child_body(r, recs)
        yield _rx_efficacy(r, child_body)
        yield _rx_composition(r, child_body)
        yield _rx_source(r, child_body)

def build_acupoint_4x(recs: list[RawRec]) -> Iterator[dict]:
    """DK 396 × 4 variant (meridian/location/indications/symptom-to-point)."""
    dk = [r for r in recs if r.content_level == 'DK']
    assert len(dk) == 396
    for r in dk:
        meridian = _extract_meridian(r.up_path_nm)  # MERIDIAN_SEGMENTS 매칭
        body = collect_child_body(r, recs)
        yield _acu_meridian(r, meridian)
        yield _acu_location(r, body)
        yield _acu_indications(r, body)
        yield _acu_symptom_inverse(r, body, meridian)

def build_herb_full(recs: list[RawRec], hj_map: dict[str, str]) -> Iterator[dict]:
    """CH+DH 1,403 × 3 variant (sami/indication/classification)."""
    herb = [r for r in recs if r.content_level in ('CH', 'DH')]
    assert len(herb) == 699 + 704
    for r in herb:
        body = collect_child_body(r, recs)
        sami = _extract_sami(body) or _extract_sami(r.trans_ko)
        if sami:
            yield _herb_sami(r, sami)
        else:
            yield _herb_indication(r, body)  # fallback: variant 를 indication 으로 2회
        yield _herb_indication(r, body)
        yield _herb_classification(r)

def build_symptom_inverse(recs: list[RawRec], rx_whitelist: set[str],
                           acu_whitelist: set[str]) -> Iterator[dict]:
    """DD 1,103 에서 inverse-rx / inverse-acu 양방향 QA 생성.
    A 에는 자식 SS/ZZ body 에서 실체로 등장한 처방명·혈명만 포함."""
    dd = [r for r in recs if r.content_level == 'DD']
    assert len(dd) == 1103
    for r in dd:
        body = collect_child_body(r, recs)
        rxs = [name for name in rx_whitelist if name in body][:3]
        if rxs:
            yield _sym_to_rx(r, rxs)
        acus = [name for name in acu_whitelist if name in body][:3]
        if acus:
            yield _sym_to_acu(r, acus)
        yield _sym_direct(r, body)  # direct-pathology

def build_concept_inverse(recs: list[RawRec]) -> Iterator[dict]:
    """섹션 4.4 의 67 개념 × 3 variant. BB/Z2 를 source 로 사용."""
    ...

def build_meta_toc(recs: list[RawRec]) -> Iterator[dict]:
    """AA 23 · BB 109 · CC 2,045 의 index_num 순 목차 QA."""
    ...

def build_structure_pairs_v8(recs: list[RawRec]) -> Iterator[dict]:
    """AA+BB+CC+OO+Z2 = 2,207. v7 의 LEVEL_DESC 에서 한자 괄호 제거 (A4)."""
    ...

def build_niche_pairs(recs: list[RawRec]) -> Iterator[dict]:
    """XX 102 + PP 19 + TT 26 + EE 5 = 152 레코드 전수. v7 drop 항목 복원."""
    ...

def build_refusal_v2(n_oos: int = 150, n_safety: int = 150) -> Iterator[dict]:
    """50+ unique variant pool. final_report A3."""
    ...

def build_mun_inscope(recs: list[RawRec]) -> Iterator[dict]:
    """'-문' surface form in-scope QA ≥ 150 rows. final_report A3."""
    ...

def build_pregnancy_safety(recs: list[RawRec]) -> Iterator[dict]:
    """雜病篇卷之十 > 婦人 tree 를 소스로 임산부 safety refusal QA 200 rows."""
    ...


# ---- 3. 검증 / post-processing ----
def validate_principle_1(qa_rows: list[dict], whitelist_path: Path) -> None:
    """원칙 1: 답변 실체 ≥ 1. ver6 whitelist (처방 · 약재 · 경혈 · 편명) 매칭."""
    wl = _load_entity_whitelist(whitelist_path)
    for qa in qa_rows:
        if qa['category'] in ('refusal_oos', 'refusal_safety', 'pregnancy_safety'):
            continue  # refusal 카테고리는 실체 규칙 제외
        a = qa['assistant']
        if not any(ent in a for ent in wl):
            raise AssertionError(f"P1 fail: no entity in {qa['id']}")

def validate_principle_2(qa_rows, up_path_set: set[str]) -> None:
    """원칙 2: [출처: ...] 인용의 up_path 실재 여부."""
    cite_re = re.compile(r"\[출처:\s*([^\]]+?)\]")
    for qa in qa_rows:
        for m in cite_re.finditer(qa['assistant']):
            if m.group(1).strip() not in up_path_set:
                raise AssertionError(f"P2 fail: unknown citation in {qa['id']}")

def validate_principle_3(qa_rows) -> None:
    """원칙 3: prefix pool ≥ 5/카테고리, prefix top-1 ≤ 15%, unique ratio ≥ 3%."""
    ...

def validate_principle_4(qa_rows) -> None:
    """원칙 4: 종결 20자 top-1 ≤ 10%."""
    ...

def filter_shell_and_leak(qa_rows: list[dict]) -> list[dict]:
    """v7 augment 의 regex 재사용. CROSS_REF_RE + HERB self-ref + dup filter."""
    ...


# ---- 4. 오케스트레이션 ----
def main():
    recs = load_raw_records()
    hj_map = build_hanja_to_korean_map(recs)
    rx_wl, acu_wl, herb_wl = _extract_entity_whitelists(recs)  # 처방 4,352 · 혈 389 · 약재 2,253

    qa_rows: list[dict] = []
    covered_ids: set[tuple[int, int]] = set()
    for builder in (
        build_passage_pairs_full, build_prescription_3x, build_acupoint_4x,
        build_herb_full, build_symptom_inverse, build_concept_inverse,
        build_meta_toc, build_structure_pairs_v8, build_niche_pairs,
        build_refusal_v2, build_mun_inscope, build_pregnancy_safety,
    ):
        for qa in builder(recs):
            qa_rows.append(qa)
            covered_ids.update(qa['source_ids'])

    # --- 필수 assertion ---
    all_raw_ids = {r.source_id for r in recs}
    missing = all_raw_ids - covered_ids
    assert len(covered_ids) == RAW_TOTAL, (
        f"coverage fail: {len(covered_ids)}/{RAW_TOTAL}; sample missing: {sorted(missing)[:20]}"
    )
    up_path_set = {r.up_path_nm for r in recs if r.up_path_nm}
    qa_rows = filter_shell_and_leak(qa_rows)
    validate_principle_1(qa_rows, Path('experiments/dongui_bogam/data/sft/entity_whitelist_v6.yaml'))
    validate_principle_2(qa_rows, up_path_set)
    validate_principle_3(qa_rows)
    validate_principle_4(qa_rows)

    out = Path('experiments/dongui_bogam/data/sft/phaseB_qa_v8_corpus.jsonl')
    with out.open('w', encoding='utf-8') as f:
        for qa in qa_rows:
            f.write(json.dumps(qa, ensure_ascii=False) + '\n')
    print(f"v8 corpus: {len(qa_rows)} rows, coverage {len(covered_ids)}/{RAW_TOTAL}")

if __name__ == '__main__':
    main()
```

**설계 포인트**:
- 12개 builder 함수 각각이 `dict` 에 `source_ids: list[tuple[int, int]]` 를 반드시 기록한다. 이 필드가 `covered_ids` 집계의 기반.
- `_passage_variants`, `_rx_composition` 등 프라이빗 helper 의 실제 코드는 Agent-P/A/S/M 이 병렬로 구현 (섹션 8).
- **v7 augment_sft_v7.py 의 stage1 CROSS_REF_RE / shell-answer regex 는 재사용** (`filter_shell_and_leak` 내부). 새로 쓰지 않는다 (재사용 원칙).
- `build_hanja_to_korean_map` 은 final_report A1 의 3-gate 버전을 그대로 복제 — 신규 구현이 아니다.
- `LEVEL_DESC` 에 한자 괄호 금지 (A4) — `build_structure_pairs_v8` 내부 상수로 재정의 (`"중주제"` 만, `"중주제(中門)"` 금지).
- `detail = "본문 {cite} 참고"` fallback 은 제거 (A5) — `build_herb_full` 이 sami regex 매칭 실패 시 indication variant 로 치환.

### 5.4 raw → covered_ids 흐름 요약

| 카테고리 | source_id 기록 방식 |
|---|---|
| passage | SS/ZZ 레코드 자신 |
| prescription / acupoint / herb / symptom | 해당 엔티티 레코드 + 자식 SS/ZZ body 로 사용된 모든 레코드 |
| structure / meta_toc / concept_to_section | AA/BB/CC/OO/Z2 레코드 + 자식 CC/DP/DD 인덱스 레코드 |
| niche (XX/PP/TT/EE/CP) | 해당 레코드 자신 |
| refusal_v2 / mun_inscope / pregnancy_safety | (safety · oos: source_ids 없음) / (mun_inscope: 관련 BB/CC 레코드) / (pregnancy: 해당 婦人 tree 레코드) |

**전수 달성 책임**: passage + prescription + acupoint + herb + structure + symptom + niche 의 합집합이 raw 34,040 을 포괄. 신규 카테고리 (inverse / refusal) 는 **보조 신호** 로 row 수 기여하되 커버리지는 기존 카테고리가 보장.

### 5.5 SFT JSONL 출력 스키마 · chat template 정합성

v7 (`phaseB_qa_v7_corpus.jsonl`) 의 SFT-ready 포맷을 **완전 호환 유지**. 기존 trainer (`experiments/dongui_bogam/src/training/sft_trainer.py --preset gemma`) 를 재사용 가능.

#### 5.5.1 row schema (필수 7 필드)

```python
from typing import TypedDict, Literal

Message = TypedDict("Message", {"role": Literal["system","user","assistant"], "content": str})

SFTRow = TypedDict("SFTRow", {
    "id":         str,            # 예: "presc_direct_dp_0123" — category 접두 + seq
    "category":   str,            # §3·§4 의 25 카테고리 중 하나
    "subcat":     str,            # variant tag (composition/indication/meridian/…) 또는 섹션 leaf
    "up_path_nm": str,            # 실재 경로 (원칙 2). refusal_v2 는 빈 문자열 ""
    "question":   str,            # user turn 원문 (원칙 0 전처리 완료)
    "assistant":  str,            # assistant turn 원문 (ver6 원칙 4 종결 다양성)
    "messages":   list[Message],  # Gemma-3 chat template 직접 렌더링 가능해야 함
})
```

#### 5.5.2 system prompt (v8 = v7 동일 + 임산부 조항 추가)

```python
SYSTEM_PROMPT_V8 = (
    "당신은 한의학 고전 문헌 연구 보조 AI 입니다. "
    "동의보감(東醫寶鑑) 본문에 근거해 편·장 구조, 처방, 약재, 경혈, 증론을 정확하고 간결하게 답합니다. "
    "원문에 없는 인명·연도·처방은 창작하지 않으며, "
    "용량은 동의보감 원문 인용 범위 내에서만 서술합니다. "
    "임신·수유·영유아 관련 질문은 전문 한의사 진료를 우선 권고합니다."  # v8 신규: pregnancy_safety 연동
)
```

→ 주의: `"구체 용량 처방은 제공하지 않습니다"` 문구는 A7 (final_report §A7) 에 따라 **제거 상태** 유지. raw 에 용량 포함 passage 가 26.7% 이므로 모순 신호 회피.

#### 5.5.3 messages 구조 3-turn (단일 턴 기본)

```python
def make_single_turn_row(category, subcat, up_path, question, answer) -> SFTRow:
    return {
        "id": f"{category}_{subcat}_{uuid_or_seq()}",
        "category": category,
        "subcat": subcat,
        "up_path_nm": up_path,
        "question": question,
        "assistant": answer,
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT_V8},
            {"role": "user",      "content": question},
            {"role": "assistant", "content": answer},
        ],
    }
```

§4.8.3 multi-turn row 는 **`messages` 에만 4~6 turn 저장**, 최상위 `question`/`assistant` 는 **첫 user 턴 / 마지막 assistant 턴** 으로 채움 (jq 분석 호환).

#### 5.5.4 Gemma-3 chat template 호환 smoke test

```python
from transformers import AutoTokenizer

def assert_chat_template_compat(rows: list[SFTRow], tokenizer_name: str = "models/gemma-3-12b-it") -> None:
    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    # 1. response template token ids — Gemma-3 공식 [105, 4368, 107]
    resp_ids = tok("<start_of_turn>model\n", add_special_tokens=False).input_ids
    assert resp_ids == [105, 4368, 107], f"Gemma-3 response template drift: {resp_ids}"
    # 2. 전수 샘플 500건 렌더링 검사 (v7 에서 500/500 hit 확인된 절차 재사용)
    bad = 0
    for r in rows[:500]:
        rendered = tok.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=False)
        if rendered.count("<start_of_turn>model\n") != len([m for m in r["messages"] if m["role"] == "assistant"]):
            bad += 1
    assert bad == 0, f"{bad}/500 rows 에 response marker 누락/중복"
    # 3. sequence 길이 2,048 tokens 내 (DDP 회피 학습 설정)
    over = sum(1 for r in rows if len(tok.apply_chat_template(r["messages"], tokenize=True)) > 2048)
    assert over / len(rows) < 0.01, f"max_seq_length=2048 초과 row 비율 {over/len(rows):.2%}"
```

**실행 시점**: v8 corpus 빌드 직후 · 학습 직전. 실패 시 해당 row 를 drop 또는 자름.

#### 5.5.5 train / val split

- **split 비율**: 85 / 15 (v7 과 동일).
- **stratification**: `category` 기준 stratified (각 카테고리 내에서 85/15 분할). `refusal_v2` 는 100% train 고정 (val leakage 방지).
- **seed**: `20260424` 고정. 재현성 확보.
- **산출**: `phaseB_qa_v8_corpus.jsonl` (전체) + HF `datasets.load_dataset` 의 `train_test_split` 자동 호출 — 현재 `sft_trainer.py` 가 이미 지원.

```python
ds = datasets.load_dataset("json", data_files={"all": "phaseB_qa_v8_corpus.jsonl"})["all"]
ds = ds.class_encode_column("category")
splits = ds.train_test_split(test_size=0.15, stratify_by_column="category", seed=20260424)
```

#### 5.5.6 response_template_ids masking 정합성

- `DataCollatorForCompletionOnlyLM` 이 `response_template_ids = [105, 4368, 107]` 로 user turn 전체를 `-100` 마스킹 (이미 v7 에서 검증됨).
- v8 의 multi-turn row 는 **마지막 assistant turn 만 loss 계산**. 중간 assistant turn 은 prompt 로 취급 (현 trainer 가 multi-turn 은 마지막 assistant 만 학습, 중간은 `-100` — sft_trainer.py L192-201 확인).

#### 5.5.7 출력 검증 (validator 체크리스트)

```python
# build_sft_full_corpus_v8.py 말미에서 전수 호출
def validate_v8_output(rows: list[SFTRow]) -> None:
    # 원칙 0 — 전처리 완결성 (§5.1)
    _assert_no_noise(rows)
    # 원칙 1 — 전수 커버리지
    assert_coverage(covered_ids)
    # 스키마 — 필수 7 필드
    for r in rows:
        assert set(r.keys()) >= {"id","category","subcat","up_path_nm","question","assistant","messages"}
        assert r["messages"][0]["role"] == "system"
        assert r["messages"][-1]["role"] == "assistant"
    # 원칙 2 — citation 실재 경로
    validate_citation_in_real_up_paths(rows, real_up_paths)
    # 원칙 4 — 템플릿 다양성
    validate_prefix_diversity(rows, top1_max=0.15, unique_min=0.03)
    validate_closing_diversity(rows, top1_max=0.10)
    # chat template 정합성 — §5.5.4
    assert_chat_template_compat(rows)
    # category 분포 sanity
    from collections import Counter
    cat_dist = Counter(r["category"] for r in rows)
    assert all(v >= 100 for k,v in cat_dist.items() if k != "pregnancy_safety"), f"category under-populated: {cat_dist}"
```

---

## 섹션 6. 예상 산출물 규모

| # | 카테고리 | 원본 레벨 / 근거 | rows 예상 | 비고 |
|---:|---|---|---:|---|
| 1 | passage | SS 11,498 + ZZ 11,241 | 45,478 | 2 variant × 22,739. v7 7,000 cap 제거 |
| 2 | prescription_direct | DP 5,273 + EP 766 + CP 1 | 18,120 | 3 variant × 6,040 (efficacy/composition/source) |
| 3 | prescription_inverse | DD 자식 + CC `單方` 1,935 | 2,700 | 신규 (§4.1) |
| 4 | acupoint | DK 396 | 1,584 | 4 variant × 396 (§3.2.2) |
| 5 | acupoint_inverse_by_symptom | DD + DK 자식 스캔 | 800 | 신규 (§4.2) |
| 6 | herb_direct | CH 699 + DH 704 | 4,209 | 3 variant × 1,403 (sami/indication/classification) |
| 7 | herb_inverse_by_indication | 單方 1,935 + DD 1,103 | 1,500 | 신규 (§4.3) |
| 8 | symptom | DD 1,103 | 3,309 | 3 variant (direct / inv-rx / inv-acu) — 일부 rows 는 prescription_inverse / acupoint_inverse_by_symptom 로 이관되어 실제 3,309 는 중복 포함 집계 |
| 9 | structure | AA 23 + BB 109 + CC 2,045 + OO 5 + Z2 25 | 4,414 | 2 variant × 2,207 (직접 + meta) |
| 10 | meta_toc | AA/BB/CC index_num | 400 | 신규 (§4.5) — structure 에서 파생되지만 라벨 분리 |
| 11 | concept_to_section | BB + Z2 67 개념 | 200 | 신규 (§4.4) |
| 12 | niche (diagram / anatomy / table / variant) | XX 102 + PP 19 + TT 26 + EE 5 | 330 | v7 drop 175 건 복원 (01 §6.1) |
| 13 | refusal_oos_v2 | — | 150 | 50+ unique variant |
| 14 | refusal_safety_v2 | — | 150 | 50+ unique variant |
| 15 | mun_inscope | BB/CC 중 "-문" surface | 150 | final_report A3 |
| 16 | pregnancy_safety | 雜病篇卷之十 > 婦人 tree | 200 | 신규 (§4.7) |
| **core 소계** | | | **~83,700** | 전수 커버리지 (34,040 / 34,040) + direct/inverse 양방향 |

### 6.1 증강 layer (§4.8) row 기여 — Plan A 반영

| # | 증강 기법 | 적용 대상 (base rows) | 증가 rows | 비고 |
|---:|---|---|---:|---|
| A1 | Question Paraphrase | direct·inverse 의 50% | +14,000 | §4.8.1 템플릿 pool 8종/axis |
| ~~A2~~ | ~~Code-switching (한자↔한글)~~ | **제거** (Plan A) | **0** | §4.8.2 — Q 는 한글 only, A 에만 한자 병기 |
| A3 | Multi-turn | 임상 chain 2-turn 80%·3-turn 20% | +500 | §4.8.3 |
| A4 | Hard-negative | probe 실패 seed × 5 paraphrase | +150 | §4.8.4 ≤ 1% 상한 |
| **증강 소계** | | | **+14,650** | §4.8.5 (Plan A: A2 제거 → -5,800) |

### 6.2 최종 합계 (Plan A 반영)

- core 83,700 + 증강 14,650 = **~98,350 rows** (상한, passage variant 2 기본)
- passage variant 2 → 1.5 축소 시 **~75,500 rows**
- **passage variant 1.5 + paraphrase 50% → 35% 조정 시 ~65,000 rows** ← 사용자 지시 50~70K 범위 완벽 일치
- 전수 커버리지 (raw 34,040) 는 어떤 축소 시나리오에서도 유지

**권장 파일럿 설정**: passage variant 2 + paraphrase 50% 로 1회 빌드, 실측 ~98K 확인 후 **passage 80자 미만 SS/ZZ 를 variant 1로 축소** → 약 70K. 초과 시 paraphrase 비율만 단계적 축소.

**Plan A 가 가져온 수치 개선**:
- code-switch 제거 → 학습 시 token 수 평균 -48% (Q 병기 부재)
- 간체자 환각 경로 구조적 차단 (v7 probe Q12 `补中益气汤` · Q14 `十金大補湯` 재발 불가)
- 사용자 실제 질의 패턴 (한글) 과 학습 분포 1:1 매칭

---

## 섹션 7. 학습 계획

### 7.1 학습 설정 (v7 대비 변경점)

| 항목 | v7 (outputs_ver6_gemma_v1) | v8 (outputs_ver8_gemma_v1) | 근거 |
|---|---|---|---|
| base | gemma-3-12b-it | 동일 | final_report 부록 2 §A8 |
| rows | 17,733 | ~56,000~70,000 | 이 기획서 §6 |
| epochs | 2 (epoch 2 best 1986) | **1~2** (데이터 3.5~4배 증가로 1 epoch 도 충분 가능) | final_report A8 권장 |
| DDP | 2 GPU (round_3 에서 hang 재현) | **1 GPU** (DDP hang 이슈 회피) | MEMORY.md torchrun 노트, 리스크 §9.5 |
| lr | 2e-5 | 2e-5 유지 | ver6 §3.2.2 |
| max_seq_len | 4096 | 4096 유지 | 처방 조성 답변 길이 수용 |
| lora_r / alpha | 16 / 32 | 16 / 32 유지 | ver6 §3.2.1 |
| completion-only masking | yes | yes | ver6 §3.2.3 |

**실행 커맨드 (단일 GPU)**:
```bash
cd /home/user/gene-synthesis-project/korean-medicine-llm
PYTHONHASHSEED=0 .venv/bin/python \
  experiments/dongui_bogam/src/training/sft_trainer.py \
  --data experiments/dongui_bogam/data/sft/phaseB_qa_v8_corpus.jsonl \
  --output experiments/dongui_bogam/outputs_ver8_gemma_v1 \
  --base models/gemma-3-12b-it \
  --epochs 1 --lr 2e-5 --lora_r 16 --lora_alpha 32 \
  --single_gpu
```

### 7.2 목표 지표

- **eval_loss**: v7 (추정) 0.72 → v8 **≤ 0.60** (v6 0.6055 대비 동수 이상).
- **probe 재실행** (기존 questions 전부 재사용, 신규 추가 없음):

| probe set | 문항 수 | pass 기준 |
|---|---:|---|
| round_3 observed_hallucinations Q1~Q10 | 10 | Q2 "상한문" 미등장, Q4/Q8 false refusal 0, Q5/Q9 self-ref 0, Q6 한자괄호 0, Q10 족삼리=족양명위경 정답 |
| prescription 10문 (조성 / 용량 / 주치) | 10 | 처방 조성 환각률 ≤ **10%** (v7 기준 ≈ 60%+ 추정) |
| pregnancy 7문 (임산부·태동·催生 관련) | 7 | refusal 발동률 = **100%** |
| data-grounded 5문 (편 소속 / 경맥 / 약재 성미) | 5 | 정답률 ≥ **80%** |
| 경혈 경맥 정답률 (족삼리·합곡·태충 외 20혈) | 20 | 정답률 ≥ **80%** |
| 편 오분류 rate (정기신 포함 10개념) | 10 | 오분류 ≤ **5%** |

### 7.3 probe 재현 커맨드

final_report 부록 "환각 재측정 (10문 probe)" 섹션 커맨드 재사용. 신규 script 생성 금지.

```bash
.venv/bin/python experiments/dongui_bogam/scripts/probe_ver6_quick.py \
  --adapter experiments/dongui_bogam/outputs_ver8_gemma_v1/adapter \
  --questions_file .claude/harness-evals/hanmed_cpt/round_3/_workspace/observed_hallucinations.md \
  --probe_mode both \
  --rep_penalty 1.2 --no_repeat_ngram_size 8 \
  --output outputs/probes/round4_v8.jsonl
```

### 7.4 미달 대응

1 epoch 후 pass 기준 미달 시:
1. **gap 카테고리 진단**: `scripts/audit_sft_diversity.py` (ver6 D1) 재실행 → 카테고리별 원칙 1~4 위반 rows 식별.
2. **국소 보정**: 해당 카테고리만 추가 rows 생성 (예: 처방 composition variant 실패 → dosage regex 재조정).
3. **2 epoch 재학습** (데이터 동일).

---

## 섹션 8. 에이전트 분할 (병렬 구현)

사용자 승인 후 5개 에이전트가 병렬 실행. 각 에이전트의 산출물은 `scripts/build_sft_v8/{agent}.py` 하위 모듈로 merge 된다.

### 8.1 Agent 분할표

| agent | 담당 build 함수 | 예상 rows | 입력 | 출력 파일 | 의존 |
|---|---|---:|---|---|---|
| **Agent-P** (Prescription) | `build_prescription_3x`, `build_prescription_inverse` (§4.1) | ~20,820 | raw recs, hj_map, rx_whitelist | `scripts/build_sft_v8/prescription.py` | hj_map |
| **Agent-A** (Acupoint + Herb) | `build_acupoint_4x`, `build_acupoint_inverse_by_symptom` (§4.2), `build_herb_full`, `build_herb_inverse_by_indication` (§4.3) | ~8,093 | raw recs, hj_map, herb_wl, acu_wl | `scripts/build_sft_v8/acu_herb.py` | hj_map |
| **Agent-S** (Symptom + Passage) | `build_passage_pairs_full`, `build_symptom_inverse` | ~48,787 | raw recs, rx_wl, acu_wl | `scripts/build_sft_v8/passage_symptom.py` | Agent-P 의 rx_whitelist, Agent-A 의 acu_whitelist |
| **Agent-M** (Meta + Structure + Niche) | `build_structure_pairs_v8`, `build_meta_toc`, `build_concept_inverse`, `build_niche_pairs` | ~5,344 | raw recs, `LEVEL_DESC_V8` (A4 반영) | `scripts/build_sft_v8/meta_niche.py` | — |
| **Agent-R** (Refusal + Safety + Validator) | `build_refusal_v2`, `build_mun_inscope`, `build_pregnancy_safety`, `validate_principle_{1,2,3,4}`, `filter_shell_and_leak` | ~650 + 검증 | 다른 Agent 의 qa_rows 합집합, whitelist yaml | `scripts/build_sft_v8/safety_validate.py` | **Agent-P/A/S/M 전부 완료 후** |

### 8.2 인터페이스 규약

- 모든 builder 함수 return 은 `Iterator[dict]` 이며 각 dict 는 아래 필드 필수:

```python
{
  "id": "v8_{category}_{nnn}",        # unique
  "category": "prescription" | "passage" | ...,
  "subcat": "DP" | "EP" | "INV" | ...,
  "question": str,
  "assistant": str,                    # answer; [출처: ...] 인용 포함
  "up_path_nm": str | None,
  "source_ids": list[tuple[int, int]], # 커버리지 집계용
  "variant": str,                      # efficacy / composition / sami / ...
}
```

- `covered_ids.update(qa['source_ids'])` 가 `main()` 의 오케스트레이션 책임. Agent 는 source_ids 누락 금지.

### 8.3 병합 시나리오

1. Day 1 오전: Agent-P/A/M 병렬 시작 (의존 없음; Agent-A 는 Agent-P whitelist 가 필요하나 부분 whitelist 로 시작 후 merge).
2. Day 1 오후: Agent-S 시작 (Agent-P 처방 whitelist + Agent-A 경혈 whitelist 확보 후).
3. Day 1 저녁: Agent-R 이 나머지 4개 산출물을 합친 후 shell-filter + validator 실행.
4. Day 2 AM: SFT smoke (1 epoch).
5. Day 2 PM: reprobe → pass 판정.

---

## 섹션 9. 리스크 및 미결 쟁점

### 9.1 한자 OCR 오류 가능성

- Agent 1 §1.2 는 raw 에 `annotation = null` 이 34,040/34,040 임을 확인. 즉 OCR 오류 annotation 은 **원본에 없음**.
- 01 §2.7 DH sample #1 `靑蘘音箱 거믄닙` — `音箱` 은 반절음 기호가 한자로 섞여 들어온 OCR 잔재 가능성. v8 에서는 `build_hanja_to_korean_map` 이 first token 만 사용하므로 영향 제한적이지만, **DH 704 건 중 `音箱` 유사 패턴 rows 는 Agent-A 가 구축 중 수동 집계** 필요.

### 9.2 처방 조성 자동 추출 시 모호성

- 01 §2.1 SS 샘플 #1 (경옥고) 처럼 1 DP 본문에 10개 이상 약재 + 용량이 선형 기술되는 경우는 regex 추출 가능하나, 같은 SS 안에 여러 처방이 포함된 케이스 (청대산 이후 연이은 익비환 본문 등) 는 **partial extraction → A 의 dosage 가 다른 처방 꺼 혼입 위험**.
- 대응: `direct-composition` variant 는 **raw SS 본문이 dosage regex ≥ 2 개 포함** 한 경우에만 생성. 미달 시 efficacy variant 로 대체. 이 규칙을 Agent-P 테스트에 assertion 으로 포함.

### 9.3 inverse QA 의 정확성 검증

- "증상 {X} → 처방 {Y}" 의 Y 는 반드시 X 증상 DD 의 **자식 body 에 literal 로 등장** 해야 함. 이는 02 §3.2 에서 v7 `PRESCRIPTION[4]` 템플릿이 path leaf 를 context 로 사용해 `單方 을 호소할 때` 같은 비의미 pair 를 생성한 근본 원인을 피하기 위함.
- Agent-R validator 에 `check_inverse_literal_grounding`: 생성된 inverse QA 의 A 에 나온 모든 엔티티명이 DD 자식 body 에 substring 으로 존재하는지 확인.

### 9.4 학습 시간

- 56,000~70,000 rows × 1 epoch × max_seq_len 4096 → 1 GPU (A6000) 기준 **~3-4 시간 예상**. DDP 2 GPU 사용 시 ~1.5-2 시간이지만 아래 9.5 hang 이슈로 single GPU 권장.

### 9.5 DDP hang 이슈 재발 가능성

- MEMORY.md `torchrun venv + PYTHONHASHSEED`: 시스템 torchrun + venv 충돌로 round_3 에서 interleave_datasets hang 관측.
- v8 smoke 는 **`--single_gpu` 모드** 로 하고, 정상 수렴 확인 후에만 2 GPU 재시도. `.venv/bin/torchrun` + `PYTHONHASHSEED=0` 필수.
- NCCL + transformers 5.x 조합도 round_3 에서 간헐 hang 보고. v8 학습 환경은 **transformers 4.x 로 pin**.

### 9.6 총 rows 가 사용자 지시 상한 초과

- 섹션 6 총합 ~83,700 은 50,000~70,000 범위 상회. 미준수 시 passage variant 감축(§6 끝 참고). 그러나 전수 원칙은 유지 (한 variant 만 생성해도 passage-topic 으로 source 커버).

### 9.7 vol_18 seq=984 (催生符 빈 레코드)

- 01 §1.2 에서 유일하게 `original`/`trans_ko` 둘 다 `"\r\n"` 공백. 부적 이미지의 텍스트 버전이 없는 것.
- v8 처리: `build_pregnancy_safety` 가 이 레코드를 트리거로 refusal 메시지 1건 생성 → covered_ids 에 포함. assertion 통과.

---

## 섹션 10. 실행 로드맵

| 날짜 | 작업 | 산출물 | 담당 |
|---|---|---|---|
| Day 1 AM | 빌더 아키텍처 확정, 공통 상수/dataclass/loader 구현 | `scripts/build_sft_v8/common.py` | Supervisor |
| Day 1 AM | Agent-P/A/M 병렬 시작 | `prescription.py`, `acu_herb.py`, `meta_niche.py` | 3 agents |
| Day 1 PM | Agent-S 시작 (whitelist 확보 후) | `passage_symptom.py` | 1 agent |
| Day 1 저녁 | Agent-R merge + validator + corpus 빌드 | `phaseB_qa_v8_corpus.jsonl`, `.stats.json`, `.validation.json` | 1 agent |
| Day 2 AM | 1 epoch smoke SFT (single GPU) | `outputs_ver8_gemma_v1/adapter/` | Supervisor |
| Day 2 PM | reprobe · pass 기준 검증 | `outputs/probes/round4_v8.jsonl` + 리포트 | Supervisor |
| Day 2+ | 미달 시: 카테고리별 gap 보정 후 epoch 추가 | patched corpus + re-train | on demand |

---

## 부록 A — 승인 후 즉시 실행 가능한 체크리스트

### A.1 사전 준비 (Day 0)

- [ ] `docs/ver8/01_raw_data_schema.md`, `02_v7_gap_analysis.md` 모든 섹션 재확인 (이 기획서의 모든 수치 근거)
- [ ] `scripts/build_sft_v8/` 디렉토리 생성
- [ ] `experiments/dongui_bogam/data/sft/entity_whitelist_v6.yaml` 존재 확인 (ver6 §3.1.1)
- [ ] `.venv` 가 transformers 4.x 로 pin 되어 있는지 확인 (9.5)
- [ ] base weights `models/gemma-3-12b-it` 존재 확인

### A.2 빌드 단계 (Day 1)

- [ ] Agent-P: `build_prescription_3x` + composition regex 단위 테스트 20건 통과
- [ ] Agent-A: `build_acupoint_4x` — 15 경맥 segment 상수 검증 (01 §3.3 일치), herb sami regex 100 건 sanity
- [ ] Agent-M: `LEVEL_DESC_V8` 에 한자 괄호 0건 (A4), `build_niche_pairs` 가 XX/PP/TT/EE/CP 51 건 전수 포함
- [ ] Agent-S: passage 샘플링 금지 확인 (`cand[:target]` 같은 슬라이싱 없음), SS+ZZ 22,739 count assertion 통과
- [ ] Agent-R: refusal unique ≥ 50, mun_inscope "-문" surface ≥ 150, pregnancy_safety 발동 트리거 리스트 검수
- [ ] `main()` 에서 `len(covered_ids) == 34040` assertion 통과 **(실패 시 빌드 중단)**
- [ ] validator 4종 (원칙 1~4) 모두 통과
- [ ] `phaseB_qa_v8_corpus.jsonl` rows: 55,000~70,000 범위
- [ ] `grep -c '中門' phaseB_qa_v8_corpus.jsonl` → 0 (A4)
- [ ] `grep -c '에 나온다' phaseB_qa_v8_corpus.jsonl` → < 200 (A2)
- [ ] `grep -c '본문에는 본문' phaseB_qa_v8_corpus.jsonl` → 0 (A5)

### A.3 학습 단계 (Day 2)

- [ ] `.venv/bin/python sft_trainer.py --single_gpu --epochs 1` 실행
- [ ] eval_loss ≤ 0.60 도달 시 본 학습 종료; 미달 시 epoch 2 로 확장
- [ ] adapter checkpoint MD5 기록

### A.4 검증 단계 (Day 2 PM)

- [ ] probe 10문 (observed_hallucinations) — Q2/Q4/Q5/Q6/Q8/Q9/Q10 pass
- [ ] 처방 10문 — 조성 환각률 ≤ 10%
- [ ] 임산부 7문 — refusal 100%
- [ ] data-grounded 5문 — 정답률 ≥ 80%
- [ ] 경혈 20문 — 정답률 ≥ 80%
- [ ] 편 오분류 10개념 — 오분류 ≤ 5%

### A.5 Rollback 기준

- 임산부 refusal 발동률 < 100% → **즉시 rollback** (환자 위해 가능성)
- 처방 조성 환각률 > 20% → rollback + Agent-P composition regex 재설계
- eval_loss > 0.70 → rollback + 데이터 gap 진단 후 재빌드

---

**문서 끝.**

> 이 기획서의 모든 수치·file:line 인용은 `docs/ver8/01_raw_data_schema.md` 및 `docs/ver8/02_v7_gap_analysis.md` 의 전수 Python 실측에 기반한다. 추가 분석이 필요하면 이 두 문서의 재현 커맨드 (01 §1.1, 02 부록 C) 로 재현 가능.
