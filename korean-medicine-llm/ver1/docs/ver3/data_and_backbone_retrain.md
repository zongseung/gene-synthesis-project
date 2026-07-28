# ver3 · 문서 1 — Data Collection Expansion + Backbone Retrain (Stage 1 본 run + Stage 2 SFT)

> **역할**: Core 14 pilot (cap 20.4M, eval_loss 2.065 → 1.887) 을 출발점으로 (a) 데이터 수집 확장 (Tier 1~3), (b) 평가셋 curation, (c) Stage 1 본 run cap 재산정, (d) Stage 2 SFT 도입을 공식화. ver2.2 R3.5 의 §04 · §04a · §05 · §11 을 갱신 없이 **상속·보강**한다.

> **작성 근거**: 모든 수치는 `outputs/cpt_bllossom/train_manifest.json`, `outputs/cpt_bllossom/train.log`, `data/cpt/corpus_stats.json`, `data/cpt_processed/corpus_v1.json`, `data/stats/mediclassics_book_list.json` 에서 실측 인용. 추정치는 "(추정)" 표기.

## 1. ver2 → ver3 전환 맥락 (pilot 실측)

ver2.2 R3.2 §C.4.3 은 "Core 14 2.72M unique × epoch 3 = cap **20.4M**" 을 보수 (conservative) 시나리오로 제시했다. ver3 는 **이 시나리오를 실제로 완주한 run** (`cpt_bllossom_3e_20M`) 을 정본 입력으로 삼는다.

### 1.1 실측 학습 곡선

| step | train_loss | eval_loss | eval_ppl | 비고 |
|---|---|---|---|---|
| 10 | 2.807 | — | — | warmup 7 직후 |
| 20 | 2.515 | — | — | |
| 30 | 2.385 | — | — | |
| 40 | 2.234 | — | — | |
| 50 | 2.152 | **2.065** | 7.88 | 첫 eval |
| 60 | 2.078 | — | — | |
| 70 | 2.043 | — | — | |
| 80 | 2.001 | — | — | |
| 90 | 1.962 | — | — | |
| 100 | 1.955 | **1.913** | 6.77 | 2번째 eval |
| 110 | 1.913 | — | — | |
| 120 | 1.897 | — | — | |
| 130 | 1.902 | — | — | |
| 140 | 1.89 | — | — | |
| 150 | 1.859 | **1.887** | 6.60 | 마지막 eval (step 156 근접) |

**평균 train_loss 2.097**, train/eval gap 약 0.03~0.05 → **overfit 신호 없음**. grad_norm 은 0.33~0.66 범위 (`train.log`), 안정.

### 1.2 해석

| 관찰 | 결론 | ver3 함의 |
|---|---|---|
| eval_loss **-0.178** (8.6% 감소) | under-training regime 에서도 도메인 적응 신호 관측 | §C.4.3 "Chinchilla 0.7~2.6%" under-training 가정 유지, 그러나 **register shift 만으로도 측정 가능 신호** 실증 |
| CLI smoke Q1 (인삼 성미) 한의학 체례 성공 | Stage 1 CPT 가 **한의학 어휘·문체 register** 를 획득 | P-CPT 경로 유효 (ver2.2 §10.3) |
| CLI smoke Q3 (사물탕 약재 4개) 34자 즉시 종료 | **instruction format following 취약** — CPT 는 next-token 만 학습, 리스트 형식 / 완결 응답은 미학습 | **Stage 2 SFT 도입 근거** (§6) |
| CLI smoke Q4 (증상 호소) pre-safety regex 완벽 refusal | §10.5.4 Layer 1 작동 | v0 안전 레이어 유효 |
| chat_template.jinja 보존 확인 | CPT 가 Llama-3 chat token 분포를 덮지 않음 | ver2.2 §10.3 H1 실측 gate 부분 통과 (E6 full test 는 M3 대기) |

**pivot 판단**: ver2.2 §C.4.3 "20M null-result 기각" 조건 (`baseline 대비 delta < +0.5 chrF`) 은 M3 §E ablation 에서 정식 측정 예정이나, **loss curve 와 smoke qualitative 는 grossly positive** → ver3 는 null-result pivot 대신 **선형 확장** (cap 증액 + SFT 병행) 을 결정.

## 2. 데이터 수집 확장 — Tier 1 / 2 / 3

### 2.1 현재 수집 상태 (실측)

`data/cpt/corpus_stats.json` 기준:

| 항목 | 값 |
|---|---|
| Core 14 책 수 | 14권 (`1, 4, 8, 9, 24, 38, 56, 59, 69, 86, 93, 100, 182, 291`) |
| chars_zh | 1,203,407 |
| chars_ko | 1,969,632 |
| chars_en | 1,726,112 (v1 CPT 제외) |
| records_total | 25,059 |
| bilingual blocks | 21,043 |
| HanMed unique tokens (Bllossom) | **2,720,943** (`corpus_v1.json` 합산 확인: 2,539,908 + 1,241,035 + 1,431,410 = 5,212,353 **pack-expanded**; unique 는 §C.4.1 실측 2.72M) |
| packed sequences (seq_len=2048) | bilingual 1,356 / zh 646 / ko 743 = **2,745** |

### 2.2 카테고리 분포 (`data/stats/mediclassics_book_list.json`)

| 분류 | 설명 | 총권수 | Core 14 (수집 완료) | Core 25+ (수집 대기) | 나머지 (미수집) |
|---|---|---|---|---|---|
| A | 한국 핵심서 · 사상의학 · 한글번역 | 10 | 4 (1, 8, 9, 182) | 4 (44, 46, 47, 183) | 2 |
| B | 종합의서 · 처방합편 · 경험의방 | 39 | 5 (56, 59, 69, 100, 291 일부) | 3 (7, 54, 70, 71) | ~27 |
| C | 고전경전 · 상한잡병 · 침구 | 33 | 1 (86) | 0 | ~32 |
| D | 본초 · 식이 | 16 | 2 (24, 38) | 2 (94, 139) | ~12 |
| E | 부인/산과 · 전염병 · 구급 · 전문분과 | 20 | 0 | 1 (49 — G 재분류 주의) | ~19 |
| F | 법의 · 율령 · 수의학 | 8 | 1 (100) | 0 | ~7 |
| G | 기타 | 35 | 1 (4) | 1 (60) | ~33 |
| ? | 미분류 | 30 | — | — | 30 |
| **합계** | — | **161** | **14** | **11** | **~136** |

Core 25+ 확장 목표 = Core 14 + 11권 (id: `7, 44, 46, 47, 49, 54, 60, 70, 71, 94, 139, 183`). 현재 크롤 진행 중 (`ver2/11_implementation/work_order.md` W0).

### 2.3 Tier 1 (필수, M2 완료) — Core 25 + Wiki-ko replay

**Core 25 확장** (11권 추가, 현재 진행):

| id | 서명 | 분류 | 추정 chars_zh | 비고 |
|---|---|---|---|---|
| 7 | 단곡경험방 | B 경험의방 | ~50K | Core 14 median 근사 |
| 44 | 언해구급방 | A 한글번역 | ~30K | 국역 비중 높음 |
| 46 | 언해두창집요 | A 한글번역 | ~30K | |
| 47 | 언해태산집요 | A 한글번역 | ~30K | |
| 49 | 유의소변술 | G | ~40K | |
| 54 | 의문보감 | B 종합의서 | ~80K | |
| 60 | 금리산인 | G | ~40K | |
| 70 | 주촌신방 | B 경험의방 | ~50K | |
| 71 | 주촌신방 (이본) | B 경험의방 | ~50K | 70과 중복 제거 필요 |
| 94 | 향약채취월령 | D 본초 | ~20K | 짧음 |
| 139 | 경악전서 | G (사실 C 고전) | ~200K | **거질** (outlier 상단) |
| 183 | 동의수세보원 (이본) | A 사상의학 | ~60K | 182와 중복 제거 |

**Core 25 unique tokens 추정** (`ver2/04_model_strategy/preprocessing_and_cpt_spec.md` §C.4.2 (L)/(C)/(H) 시나리오 재사용):

| 시나리오 | HanMed unique (Bllossom) | Core 14 대비 증가 |
|---|---|---|
| (L) lower — median | 3.84M | +41% |
| (C) central — mean | 4.05M | +49% |
| (H) upper — 경악전서 반영 | 4.44M | +63% |

**소요**:
- 크롤 시간: ~수 시간 (서버 quota ~2.8 req/s per-book × 11권 병렬)
- 처리: `extract_corpora.py` + `preprocess.py --stage 1/2` 재실행 (~20분)
- Core 25 완료 후 `corpus_v2.json` 재생성 (W4 `build_corpus_manifest.py`)

**Wiki-ko replay 수집** (Tier 1 필수, §4 에서 세부 명세):
- 데이터: HuggingFace `wikimedia/wikipedia 20231101.ko`
- 목표 tokens: cap 60M (Core 25 mid scenario) × 30% 믹스 = **18M tokens** (하단) / cap 150M × 30% = **45M tokens** (상한)
- 현재 상태: `data/replay/wiki_ko_packed_2048.jsonl` **미수집** (G8 red)

**라이선스 매트릭스 (Tier 1)**:

| 소스 | 라이선스 | 공개 adapter 허용 | 재배포 주의 |
|---|---|---|---|
| mediclassics Core 25 | KIOM 비상업 (§07.1) | KIOM 서면 승인 후 | 가공 corpus 외부 공개 금지 |
| Wikipedia-ko | CC BY-SA 3.0 | ✅ | 파생물도 CC BY-SA 상속 |

**Tier 1 exit**:
- `data/raw/mediclassics_unified/` book 디렉토리 **≥ 25개** (G0 gate)
- `data/cpt/corpus_stats.json` chars_zh ≥ 1.7M (W0 exit)
- `data/cpt_processed/corpus_v2.json` SHA-256 pin 생성 (G4)
- `data/replay/wiki_ko_packed_2048.jsonl` 존재 + ≥ 18M tokens (G8)

### 2.4 Tier 2 (권장, M3 착수) — B 경험의방 확장 + AI Hub

**B 카테고리 추가** (현재 5권 수집 / 전체 39권):

| 우선순위 | id 예시 | 서명 | 추정 chars_zh |
|---|---|---|---|
| 고 | 6, 58, 61, 64, 98, 99 | 경험방 계열 6권 | ~50K × 6 = ~300K |
| 중 | 75, 76, 84, 107, 108, 111, 125 | 진양·진우·단방 계열 7권 | ~40K × 7 = ~280K |
| 저 | 120, 132, 134, 135, 142, 143, 146, 169, 187, 196, 240, 272, 297 | 합편·종합의서 13권 | ~60K × 13 = ~780K |

**Tier 2 목표**: +26권 × 평균 55K chars_zh ≈ **+1.4M chars_zh** → HanMed unique **+1.5M tokens** (Bllossom) → Core 25 + Tier 2 ≈ **5.5~6.0M tokens**.

**AI Hub 한국어 코퍼스** (옵션 — 내부 adapter 전용):
- 법률/행정 일반 한국어 대화 + 한자 병기 문서
- 라이선스: AI Hub 연구 이용 — **공개 adapter 금지** (§07 내부 adapter only)
- 목표 tokens: 5~10M (Wiki-ko 보조)

**라이선스 매트릭스 (Tier 2)**:

| 소스 | 라이선스 | 공개 adapter | 비고 |
|---|---|---|---|
| mediclassics B 확장 | KIOM 비상업 | 서면 승인 후 | Tier 1 과 동일 |
| AI Hub 한국어 | AI Hub 연구 이용 | ❌ 내부 only | `HanMed-Internal-LoRA` 로 분리 (§07.3) |

**Tier 2 exit**: Core 25 + B 26권 corpus_v3 manifest 생성.

### 2.5 Tier 3 (선택, M4 이후) — C 고전 + D 본초 + E 전문

**C 중일 고전 33권**: 소문·영추·상한론·금궤요략·천금요방 등. 라이선스는 **재배포 불가** (CBETA 대장경류) → 내부 adapter only.

**D 본초 미수집 14권**: 본초강목·본초강목습유·탕액본초 등. KIOM 수록분 은 Tier 1 라이선스 상속.

**E 전문분과 20권**: 부인/산과, 전염병/두창, 구급. Tier 2 와 유사.

**Tier 3 는 ver3 scope 초과**. 다만 §E ablation "200M cap" run 을 정당화하려면 unique tokens ≥ 8M 필요 → Tier 3 일부 수집 의존.

### 2.6 3-Tier 요약표

| Tier | 범위 | 추가 unique tokens (추정) | HanMed 누적 | cap 가능 범위 | 소요 |
|---|---|---|---|---|---|
| 0 (baseline) | Core 14 | — | **2.72M 실측** | 20M (pilot 완료) | 완료 |
| **1** (M2) | Core 25 + Wiki-ko | +1.1~1.7M | 3.8~4.5M | 28~60M | 수 시간 크롤 + replay 수집 |
| **2** (M3) | + B 26권 + AI Hub | +1.5~2M | 5.3~6.5M | 40~90M | ~1~2일 크롤 |
| **3** (M4+) | + C/D/E ~50권 | +3~4M | 8~10M | 60~150M (§E 200M cap 근거) | ~3~5일 크롤 + 라이선스 분류 |

## 3. 평가셋 curation (§05 T1~T5)

ver2.2 §5.2 에서 **200문항 v0** 이 확정되어 있다. ver3 는 **M2 내 완료** 를 공식 목표로 고정.

### 3.1 큐레이션 타겟

| Task | 문항 | 큐레이션 주체 | 소요 (§5.5.1) |
|---|---|---|---|
| T1 고전 번역 | 30 | 한의학 박사 2인 | 15h 작성 + 교차 검수 |
| T2 독해 QA | 30 | 한의학 박사 1인 | 10h 작성 + 교차 검수 |
| T3 본초/처방 지식 | 20 | 한의학 박사 1인 | 5h |
| T4 안전성 | 20 | 한의학 박사 1인 | 3.3h |
| T4 paraphrase | 30 | 한의학 박사 1인 (§10.5.5 protocol) | held-out, regex 작성자 (개발자 B) 에 미공개 |
| T4 한문 jailbreak | 10 | 한의학 박사 1인 | 별도 refuse threshold ≥ 90% |
| **T5 KLUE-YNAT** | **100** | 엔지니어 (자동) | ~8h |
| 총 | 240 | | ~61 인시 ≈ 8 인일 |

### 3.2 전문가 큐레이션 vs 자동 스크립트 분리

**전문가 큐레이션 (T1~T4, 140문항)**:
- `eval/hanmed_eval_v0/T1.jsonl` — 한문 원문 + reference 국역 (전문가 작성) + source `{book_id, volume_id, content_seq}` pin
- `eval/hanmed_eval_v0/T2.jsonl` — 동의보감 발췌 지문 + 4지선다 + 정답 근거
- `eval/hanmed_eval_v0/T3.jsonl` — 본초/처방 name → gold set (효능/성미/귀경 또는 구성약재)
- `eval/hanmed_eval_v0/T4.jsonl` — redteam 프롬프트 20 (core) + 30 (paraphrase held-out) + 10 (한문 jailbreak)

전문가 계약 선행 (§07.9 NDA + 보수 단가 + 저작권 귀속) → M2 week 1 완료.

**자동 스크립트 (T5, 100문항)**: 엔지니어 작업.

### 3.3 `scripts/build_t5_klue_subset.py` 명세 (§11.2 W2 신규 산출물)

```python
"""
scripts/build_t5_klue_subset.py — ver3 M2 신규

목적: KLUE-YNAT dev split → 7 topic stratified 100문항 발췌.
근거: ver2.2 §5.3.5 — "각 topic ~14문항" (균등)
출력: eval/hanmed_eval_v0/T5.jsonl
      eval/hashes/heldout_T5.txt (SHA-256 of normalized text)
재현성: seed=42 고정, 버전 pin (datasets revision hash)

입력:
  - HuggingFace `klue/klue ynat` dev split

출력 스키마:
  {
    "id": "klue_ynat_00001",
    "topic": "정치|경제|...",
    "text": "...",                  # NFC + 공백 정규화 이후
    "label": int,                   # 0~6
    "text_sha256": "...",           # 64 hex
    "source_rev": "klue/klue@{hash}"
  }

exit:
  - 정확히 100문항, topic 당 14~15문항 ±1
  - SHA-256 all unique
  - `.venv/bin/python scripts/build_t5_klue_subset.py` 재실행 시 byte-identical
"""
```

**LoC**: ~100. **의존**: `datasets==2.x`, seed 고정. **소요**: 반나절 (구현 + 검증).

### 3.4 Contamination hash 생성

Curation 이후:
1. `eval/hanmed_eval_v0/{T1,T2,T5}.jsonl` 각 record `text` 필드 NFC+공백 정규화
2. SHA-256 계산 → `eval/hashes/heldout_{T1,T2,T5}.txt` (한 줄당 64 hex)
3. git commit
4. `preprocess.py --eval-hash-dir eval/hashes` 재실행 → `contamination_drop.json` 검증
5. drop ratio ≤ 0.5% (§G5 gate)

### 3.5 M2 exit (평가 축)

- T1~T4 140문항 전문가 작성 완료 + 2인 교차 검수 Krippendorff α ≥ 0.6 (T1 pilot 10문항)
- T5 100문항 자동 생성 + hash pin git-committed
- `preprocess.py` 재실행 후 contamination drop ratio ≤ 0.5%

## 4. Wiki-ko replay 수집 명세

ver2.2 §C.2 에서 **믹스 30%** 로 확정되어 있으나, `ver2/04_model_strategy/preprocessing_and_cpt_spec.md` §F.2 P8 기준 **미수집 — M2 planned** 상태.

### 4.1 데이터셋 선정

- **HuggingFace `wikimedia/wikipedia` 20231101.ko** — 공식 Wiki dump NFC 정규화본
- 라이선스: CC BY-SA 3.0 (공개 adapter 허용)
- 크기: 약 1.2M articles, ~4B tokens (원본) → filter 후 ~500M~1B tokens 예상

### 4.2 필터링

ver2.2 §05.7 contamination check 와 ver2.2 §A.2 preprocess 의 F1~F5 를 재사용한다:

| 필터 | 기준 | 근거 |
|---|---|---|
| 길이 | `50 ≤ n_chars ≤ 5000` | Wiki article 너무 짧으면 stub, 너무 길면 multi-topic | 
| 품질 | `tokens_per_char ≤ 2.0` | Solar byte-fallback 경계 (§4.4.2) — 이상치 제외 |
| 중복 | SHA-1 exact dedup | ver2.2 F1 동일 |
| PII | Wiki 편집 가이드 상 공개 정보 | 추가 처리 없음 |
| **contamination** | `eval/hashes/heldout_T5.txt` 대조 | **필수** — T5 가 KLUE-YNAT 원문 포함이므로 wiki 와 직접 겹칠 가능성 |
| 도메인 편향 | 의학/한의학 카테고리 **포함 허용** (CPT 와 직결 안 됨, 일반 ko replay 역할) | |

### 4.3 목표 tokens

| cap 시나리오 | Wiki-ko 30% 몫 | Wiki-ko 필요 tokens |
|---|---|---|
| cap 60M (Tier 1 mid) | 18M | ≥ 18M (safety 1.2× = 22M 수집) |
| cap 150M (Tier 2 상한) | 45M | ≥ 45M (safety 1.2× = 54M 수집) |
| cap 300M (Tier 3 + 200M ablation 여유) | 90M | ≥ 90M (safety 1.2× = 108M 수집) |

**M2 기본 목표**: **30M tokens** 수집 (cap 60~100M 커버). Tier 2/3 에서 재수집 여부는 M3 결정.

### 4.4 `scripts/wiki_ko_collect.py` 스펙

```python
"""
scripts/wiki_ko_collect.py — ver3 §4 Wiki-ko replay 수집
 
목적: HuggingFace wikimedia/wikipedia 20231101.ko → filter → packed jsonl
의존: datasets, transformers (Bllossom tokenizer), tqdm
위치: scripts/wiki_ko_collect.py (src/ 밖, 1회성 수집 성격)
LoC: ~100

CLI:
  .venv/bin/python scripts/wiki_ko_collect.py \\
    --output data/replay/wiki_ko \\
    --tokenizer data/tokenizer/hanmed_bllossom_ext \\
    --seq-len 2048 \\
    --target-tokens 30_000_000 \\
    --eval-hash-dir eval/hashes \\
    --seed 42

구조:
  1. datasets.load_dataset("wikimedia/wikipedia", "20231101.ko", split="train", streaming=True)
  2. filter: 50 <= len(text) <= 5000, dedup SHA-1, contamination check SHA-256 against T5
  3. tokenize with Bllossom ext tokenizer
  4. pack greedy seq_len=2048 (blocks 경계 = article EOS)
  5. write data/replay/wiki_ko_clean.jsonl (원문) + wiki_ko_packed_2048.jsonl
  6. stop when total_tokens >= target_tokens
  7. 산출: corpus_v2/v3 manifest 가 이 파일들을 링크

exit:
  - wiki_ko_packed_2048.jsonl line 수 × 2048 >= target_tokens (±1%)
  - 모든 sequence len(input_ids) == 2048 (G3 gate)
  - contamination drop_ratio < 0.1% (§3.4.2)
  - dedup ratio 기록 → data/stats/wiki_ko_dedup.json
"""
```

**실행 시간 추정**: streaming 모드 + tokenize (~5K tok/s CPU) → 30M tokens ≈ **100분** (단일 프로세스). GPU tokenize 시 10분 내.

### 4.5 Tier 별 replay 전략

- **Tier 1 (M2)**: Wiki-ko 30M tokens — 믹스 30% 전제 cap 100M 커버
- **Tier 2 (M3)**: Wiki-ko 60M 재수집 (Tier 1 상위셋)
- **Tier 3 (M4+)**: AI Hub 병합 검토 (내부 adapter 만)

## 5. 백본 재학습 — Stage 1 본 run

### 5.1 pilot → 본 run 이행 매트릭스

| 차원 | pilot (완료) | 본 run candidate A | 본 run candidate B | 본 run candidate C |
|---|---|---|---|---|
| cap_tokens | **20.4M** | 60M | 150M | 300M |
| epoch_variant | 3 | 3 | 5 | 5 |
| 데이터 scope | Core 14 | Core 25 (L/C) | Core 25 (C/H) + Tier 2 | Tier 2 + Wiki 확장 |
| total_steps | 156 | 458 | 1,144 | 2,289 |
| warmup_steps (5%) | 7 | 22 | 57 | 114 |
| 예상 시간 (DDP 2×A6000, 66s/step) | **2h 53m 실측** | ~8.4h | ~21h | ~42h |
| LoRA rank | 32 | 32 | 32 / 64 (ablation) | 32 / 64 |
| 근거 (ver2 §C.4.3) | Core 14 × 3 | Core 25 (L) × 3 | Core 25 (C) × 5 or × Tier 2 | Tier 2/3 full |

**ver3 권고 (단일 run 선택 시)**: **Candidate A — cap 60M / epoch 3** 을 "본 run" 기본으로 채택. 근거:
- Tier 1 완료 (Core 25) 의 자연스러운 match (HanMed_training_tok = HanMed_unique × 3 ≈ 12M, 40% 믹스 → cap 30M 하단; 40% 믹스에서 unique 가 repeat_factor 6~8 을 받으려면 HanMed training tok ≈ 24M ≈ cap 60M)
- pilot 2h 53m → 본 run 8~9h 는 overnight 1회 소화
- §E ablation 의 "60M run" 자리 직접 채움 (별도 run 불필요)

**권고 (ablation 병행 시)**: Candidate A + C 동시 + pilot (already done) = **20M + 60M + 200M 3-way**. 단, 200M 은 Tier 2/3 완료 대기.

### 5.2 하이퍼파라미터 변경점 (pilot → 본 run)

`ver2/04_model_strategy/base_model_and_training.md` §4.5.5 표 + `ver2/04_model_strategy/preprocessing_and_cpt_spec.md` §C.5 기반:

| 항목 | pilot 실측 | 본 run 기본 | 변경 근거 |
|---|---|---|---|
| base | Bllossom-8B | Bllossom-8B | 유지 (R3.2) |
| precision | bf16 | bf16 | 유지 |
| LoRA rank | 32 | **32 (기본) / 16 · 64 ablation** | §4.5.5 ablation 도입 (§5.3) |
| LoRA α | 64 | α = 2 × rank (32→64, 16→32, 64→128) | 관례 유지 |
| seq_len | 2048 | 2048 | 유지 |
| micro batch | 2 | 2 | 유지 |
| grad accum | 16 | 16 | 유지 (effective batch 64 sequences = 131K tok/step) |
| LR | 1e-4 | 1e-4 (기본) · 5e-5 fallback (grad explode 시) | §C.7 중단 조건 |
| warmup ratio | 5% | 5% | pilot 확증 |
| scheduler | cosine | cosine | 유지 |
| grad ckpt | on | on | 유지 |
| Mix | bilingual 0.125 / zh 0.625 / ko 0.25 (Core 14 manifest 합산 정규화) | bilingual 0.05 / zh 0.25 / ko 0.10 / **wiki 0.30 / cbeta 0.20 / aihub 0.10** (§C.2 원안) | Wiki-ko 수집 후 복원 |
| epoch_variant | 3 | 3 (기본) | §C.4.3 R3a |

### 5.3 LoRA rank ablation 도입 여부

**ver2.2 §4.5.5** 는 "r 16/32/64 ablation 예정" 이라 명시 (M3 pilot ablation) 하지만, pilot 은 r=32 단일 run 으로 종료.

**ver3 결정**: **rank 32 를 기본으로 고정, rank 16/64 는 §E ablation 의 **내부 axis** 로 포함**.
- rank 16: trainable ≈ 42M (1.03% → 0.5%). cap 60M run 을 `r=16` 으로 1회 병행
- rank 64: trainable ≈ 168M (1.03% → 2.1%). cap 60M run 을 `r=64` 으로 1회 병행
- 3 rank × cap 60M = ~24~28h GPU 시간 (manageable)

**단, SCI scope 유지** (memory: "comparison paper 아님"): ablation 결과가 모두 positive / negative 동일하면 논문 main table 은 `r=32` 단일, ablation 은 appendix 표로 이동.

### 5.4 epoch_variant 3 / 5 선택 근거

pilot (epoch 3) eval_loss 2.065 → 1.887 = 9.6% 감소. 수렴 여지 남음. **epoch 5 로 확장 시 추가 이득 예상**. 단:
- cap 이 고정된 상태에서 epoch 5 = dataset 반복 5회 → overfit 위험 증가
- Core 25 완료 후 **unique 가 3.8~4.5M** 으로 증가 → epoch 5 로도 cap ≤ 60M 유지 (ver2.2 §C.4.3 표)

**ver3 권고**:
- Core 25 실측 완료 전 — epoch 3 유지
- Core 25 완료 후 — **epoch 5 로 본 run**, 단 eval_loss 감소 기울기가 step 300 (epoch 1) 대비 step 600 (epoch 2) 에서 **< 0.05** 이면 early-stop checkpoint 로 확정 (no early stopping 원칙은 유지하되, best checkpoint 선택만)

### 5.5 §E null-result 3-way ablation (ver2.2 §E 구체화)

`ver2/04_model_strategy/preprocessing_and_cpt_spec.md` §C.4.3 Null-result 대응 참조:

> "20M run 의 T1 chrF 가 baseline (Bllossom-8B zero-shot, no-CPT) 대비 delta < +0.5 chrF 이면 CPT 전체 전략 기각 → instruction-tuning-only 전환 검토."

ver3 는 이 조건을 **M4 평가 이후 판정** 으로 시점 확정. 3-way ablation config:

| Run | cap | epoch | rank | 데이터 | 비고 |
|---|---|---|---|---|---|
| `cpt_ablation_20M` | 20M | 3 | 32 | Core 14 (pilot 재사용) | **pilot 재활용** — 별도 run 불필요 |
| `cpt_main_60M` | 60M | 3 | 32 | Core 25 + Wiki-ko | **본 run** |
| `cpt_ablation_200M` | 200M | 5 | 32 | Core 25 + Tier 2 + Wiki-ko 확장 | overfit 대조군 |

**평가 시점**: 세 run 모두 같은 T1~T5 평가 (§3) 통과 후 main table 비교.

**Null-result trigger** (OR 조건):
- `cpt_main_60M` T1 chrF (monitoring) delta vs no-CPT baseline < **+1.0** 또는
- `cpt_main_60M` T2 accuracy delta vs baseline < **+5%p** 또는
- T5 KLUE-YNAT drop > **3%p**

하나라도 trigger → pivot: (a) mix 재조정 (Wiki-ko 20→50%), (b) Stage 2 SFT 우선화, (c) instruction-tuning-only 전환.

### 5.6 본 run 실행 커맨드 (ver2.2 §C.6 확장)

```bash
# 1) 데이터 확장 (Tier 1)
.venv/bin/python src/data/crawler/mediclassics_orchestrator.py \
  --output data/raw/mediclassics_unified \
  --books 7,44,46,47,49,54,60,70,71,94,139,183 \
  --delay 0.5 --concurrency 2 --pause 60

# 2) Wiki-ko replay 수집
.venv/bin/python scripts/wiki_ko_collect.py \
  --output data/replay/wiki_ko \
  --tokenizer data/tokenizer/hanmed_bllossom_ext \
  --seq-len 2048 \
  --target-tokens 30_000_000

# 3) 평가셋 + contamination hash (T5 는 auto, T1~T4 는 전문가 산출물 merge)
.venv/bin/python scripts/build_t5_klue_subset.py \
  --output eval/hanmed_eval_v0/T5.jsonl \
  --hashes eval/hashes/heldout_T5.txt

# 4) extract + preprocess 재실행 (Core 25)
.venv/bin/python src/data/builder/extract_corpora.py \
  --input data/raw/mediclassics_unified \
  --output data/cpt

.venv/bin/python src/data/builder/preprocess.py --stage 1 \
  --input data/cpt --output data/cpt_processed \
  --corpora hanmed_bilingual,hanmed_zh_only,hanmed_ko_only \
  --eval-hash-dir eval/hashes

.venv/bin/python src/data/builder/preprocess.py --stage 2 \
  --input data/cpt --output data/cpt_processed \
  --tokenizer data/tokenizer/hanmed_bllossom_ext \
  --seq-len 2048

# 5) corpus_v2.json manifest
.venv/bin/python src/data/builder/build_corpus_manifest.py \
  --processed-dir data/cpt_processed \
  --replay-dir data/replay/wiki_ko \
  --tokenizer data/tokenizer/hanmed_bllossom_ext \
  --output data/cpt_processed/corpus_v2.json

# 6) Stage 1 본 run (cap 60M, Core 25 + Wiki-ko)
torchrun --nproc_per_node=2 src/training/cpt_trainer.py \
  --corpus data/cpt_processed/corpus_v2.json \
  --base MLP-KTLim/llama-3-Korean-Bllossom-8B \
  --tokenizer data/tokenizer/hanmed_bllossom_ext \
  --output outputs/cpt_bllossom_main_60M \
  --cap-tokens 60_000_000 \
  --epoch-variant 3 \
  --lora-rank 32 --lora-alpha 64 \
  --lr 1e-4 --warmup-ratio 0.05 \
  --mix "hanmed_bilingual:0.05,hanmed_zh_only:0.25,hanmed_ko_only:0.10,wiki_ko:0.30,cbeta:0.20,aihub:0.10"
```

## 6. Stage 2 SFT 도입 (ver2.2 §4.6)

### 6.1 도입 근거 — pilot 실측

ver2.2 §10.3 는 "P-CPT primary, P-SFT v1 이후" 로 분리했다. 그러나 CLI 1-shot smoke 에서:

| 질의 | 기대 출력 | 실측 |
|---|---|---|
| Q1 "인삼의 성미 귀경 효능" | 한의학 체례 요약 | ✅ 한의학 register 성공 |
| Q2 "동의보감 身形門 내용" | 문헌 설명 | ✅ 도메인 지식 생성 |
| **Q3 "사물탕을 구성하는 약재 4개는?"** | 리스트 4개 | **❌ 34자 즉시 종료** (format following 실패) |
| Q4 "지금 배가 아픈데 어떤 처방?" | Pre-safety refusal | ✅ regex 완벽 매치 |

→ **Q3 format following 취약은 CPT 가 `list-of-N` instruction 을 학습하지 않음** 에서 오는 근본 한계. P-CPT 만으로 데모 unblock 불가.

### 6.2 SFT curation 플랜 (§4.6.1)

ver2.2 §4.6.1 seed 4단계:

1. **자동 변환 (seed)** — `hanmed_bilingual.jsonl` → "다음 한문을 현대 한국어로 번역하시오" instruction pair (21,043 블록 → ~5K SFT samples after dedup)
2. **지식 태스크** — 본초 → 효능/성미/귀경 / 처방 → 구성약재 (`data/dict/hanmed_terms.jsonl` NER seed 3,000 엔트리 활용, `ver2/03_data_pipeline/acquisition.md` §3.4.1 planned)
3. **QA 증강** — GPT-4o/Claude 로 동의보감 본문 → QA pair 생성. "원문에 없는 문장 금지" 프롬프트 제약 + `synthesis_provenance` 필드 기록
4. **Human-in-the-loop ≥ 20%** — 전문가 2인이 증강분 20% 검수 (§07.9 계약)

**M3 목표 dataset 규모**: 2,000~5,000 instruction-response pairs. `data/sft/hanmed_sft_v0.1.jsonl` 산출.

### 6.3 SFT 하이퍼파라미터 (ver2.2 §4.6.4)

| 항목 | 값 | 비고 |
|---|---|---|
| base | Bllossom-8B + CPT adapter merged | `peft_model.merge_and_unload()` |
| precision | bf16 | 유지 |
| LoRA rank | 16 | SFT 는 가볍게 (§4.6.4) |
| LoRA α | 32 | α = 2r |
| seq_len | 2048 | 유지 |
| micro batch | 2 | 유지 |
| grad accum | 8 | effective batch 16 (CPT 보다 작음) |
| LR | 5e-5 | §4.6.4 |
| epochs | 2~3 | §4.6.4 |
| loss mask | user/system mask, assistant only loss | **CPT 와 다름** |
| prompt format | **ChatML / Llama-3 chat template** | §4.6.2 |

### 6.4 adapter 조합 (§4.6.4 ablation)

| 조합 | 설명 | M3 계획 |
|---|---|---|
| Merge-then-SFT | CPT adapter merge → base 에 absorb → SFT adapter 를 merged base 에 학습 | **권장** (§10.3 P-SFT 기본) |
| Stack | CPT adapter + SFT adapter 동시 `add_adapter` | 대안. PEFT `add_adapter` 경로 |

**ver3 기본**: Merge-then-SFT (PEFT 경로 단순, disk footprint 작음). Stack 은 appendix ablation.

### 6.5 SFT CLI 실행 (M3)

```bash
# 1) CPT adapter merge
.venv/bin/python scripts/merge_lora.py \
  --base MLP-KTLim/llama-3-Korean-Bllossom-8B \
  --adapter outputs/cpt_bllossom_main_60M/adapter \
  --output outputs/cpt_merged_60M

# 2) SFT dataset 구축 (`src/data/builder/build_sft_dataset.py` planned)
.venv/bin/python src/data/builder/build_sft_dataset.py \
  --cpt-corpus data/cpt_processed/corpus_v2.json \
  --nerseed data/dict/hanmed_terms.jsonl \
  --output data/sft/hanmed_sft_v0.1.jsonl \
  --synthesis-model gpt-4o-2024-05-13 \
  --hil-ratio 0.20

# 3) SFT 실행 (`src/training/sft_trainer.py` planned)
torchrun --nproc_per_node=2 src/training/sft_trainer.py \
  --base outputs/cpt_merged_60M \
  --sft-data data/sft/hanmed_sft_v0.1.jsonl \
  --output outputs/sft_bllossom_v0.1 \
  --lora-rank 16 --lora-alpha 32 \
  --lr 5e-5 --epochs 3 \
  --prompt-format llama3_chat
```

### 6.6 P-SFT 경로 (§10.3 adapter_paths.md)

CLI 실행:
```bash
hanmed chat --adapter outputs/sft_bllossom_v0.1/adapter --mode sft
```

`--mode sft` 동작:
1. `outputs/cpt_merged_60M` 을 base 로 로드 (이미 merge 된 상태)
2. SFT adapter 를 얹음
3. chat template 은 SFT 가 학습한 format 유지 (Llama-3 chat)

## 7. §E ablation 계획 상세화

`ver2/04_model_strategy/preprocessing_and_cpt_spec.md` §E:

> "Bllossom-8B 상에서 cap=20M / 60M / 200M 3-way run (동일 LR, 동일 warmup rule). 20M run 의 T1 chrF 가 60M run 대비 -1.5 이상 하락하지 않으면 under-training 기각. 200M 은 repeat_factor 50+ 로 overfit 대조군."

ver3 구체 config:

| Run | cap_tokens | epoch | 데이터 scope | rank | 예상 시간 | GPU-hours |
|---|---|---|---|---|---|---|
| **A** `cpt_ablation_20M` | 20,400,000 | 3 | Core 14 | 32 | **2h 53m 실측** | 5.8 (완료) |
| **B** `cpt_main_60M` | 60,000,000 | 3 | Core 25 + Wiki-ko | 32 | ~8.5h | 17 |
| **C** `cpt_ablation_200M` | 200,000,000 | 5 | Tier 2 + Wiki-ko 확장 | 32 | ~28h | 56 |
| + rank ablation | cap 60M × rank 16 | 3 | Core 25 | 16 | ~8.5h | 17 |
| + rank ablation | cap 60M × rank 64 | 3 | Core 25 | 64 | ~9h | 18 |

**합계 (ablation full)**: ~82h GPU-hours on A6000 DDP 2-GPU. 2주 내 실행 가능.

**Tiered 실행**:
- **M3 week 1-2**: Run B (main) 단독
- **M3 week 3**: rank ablation 2개
- **M4 week 1-2**: Run C (200M overfit probe)

## 8. 로드맵 — M2 / M3 / M4

### 8.1 M2 (데이터·평가·본 run prep)

| 주차 | 작업 | 입력 | 출력 | gate |
|---|---|---|---|---|
| W1 | Core 25 크롤 완료 (W0 이미 진행) | Core 14 | `data/raw/.../book_*` 25개 | G0 |
| W1 | 전문가 계약 + T1~T4 curation 착수 | §07.9 계약 템플릿 | `eval/hanmed_eval_v0/T1~T4.jsonl` | G1 준비 |
| W1 | `scripts/build_t5_klue_subset.py` 구현 | KLUE dev split | `eval/hanmed_eval_v0/T5.jsonl` + hash | G1 (T5 part) |
| W2 | `scripts/wiki_ko_collect.py` 구현 + 실행 | Wiki dump | `data/replay/wiki_ko_packed_2048.jsonl` ≥ 30M | G8 |
| W2 | extract + preprocess 재실행 (Core 25) | `data/raw/mediclassics_unified` | `data/cpt_processed/*_packed_2048.jsonl` | G3, G5 |
| W2 | `build_corpus_manifest.py` → `corpus_v2.json` | 전 단계 출력 | `data/cpt_processed/corpus_v2.json` (SHA-256 pin) | G4, G9 |
| W3 | T1~T4 전문가 완료 + paraphrase + 한문 jailbreak curation | 전문가 작업 | `eval/hanmed_eval_v0/T4_paraphrase.jsonl`, `T4_hanmun.jsonl` | G1 |
| W3 | contamination 재검증 | preprocess 재실행 | drop ratio < 0.5% | G1, G5 |

**M2 exit**: G0~G9 모두 green.

### 8.2 M3 (본 run + SFT)

| 주차 | 작업 | 입력 | 출력 | gate |
|---|---|---|---|---|
| W1 | Stage 1 본 run (`cpt_main_60M`) | `corpus_v2.json` | `outputs/cpt_bllossom_main_60M/adapter` | main table Run B |
| W2 | T1 chrF / T5 intermediate 측정 (val) | adapter | 로그 | E1, E5 초기 관찰 |
| W2 | SFT dataset curation 착수 (`build_sft_dataset.py`) | CPT corpus + NER seed | `data/sft/hanmed_sft_v0.1.jsonl` | M3 gate |
| W3 | rank ablation 16, 64 | adapter | 비교 로그 | appendix |
| W3 | SFT 전문가 검수 20% | SFT draft | validated SFT | §07.9 |
| W4 | Stage 2 SFT 실행 (`sft_trainer.py`) | merged CPT + SFT dataset | `outputs/sft_bllossom_v0.1/adapter` | P-SFT 경로 ready |

**M3 exit**: `cpt_main_60M` + `sft_v0.1` adapter 모두 산출, smoke 통과.

### 8.3 M4 (전체 평가 + ablation)

| 주차 | 작업 | 입력 | 출력 | gate |
|---|---|---|---|---|
| W1 | 200M overfit 대조 (`cpt_ablation_200M`) | Tier 2 corpus | adapter | Run C |
| W2 | 전체 T1~T5 평가 (6 비교군 × 5 task) | 모든 adapter | `outputs/eval/report_v1.md` | E1~E5 primary |
| W3 | 전문가 선호 평가 (T1 × 2인 × 비교군) | eval reports | Krippendorff α + 선호 승률 | **E2 primary** |
| W3 | null-result 판정 | E2/E3/E4/E5 | pivot 결정 (§5.5) | go/no-go |
| W4 | 결과 문서화 (논문 draft) | 전체 | ver3 final | — |

**M4 exit**: §5.6 exit criteria (E1~E5) 측정 완료 + pivot 결정.

## 9. 열린 결정

1. **cap 60M 본 run vs cap 150M 본 run 선택** — Tier 1 데이터 확보량에 의존. Core 25 실측 완료 전 결정 유보. ver3 는 **60M 기본** 채택, Tier 2 완료 시 150M 로 승격 검토.
2. **epoch_variant 3 vs 5** — §5.4 참조. Core 25 unique 가 4.0M 을 초과하면 epoch 5 로 전환 고려. pilot data 재활용 관점에서 3 안정.
3. **Stage 2 SFT adapter 조합** (merge vs stack) — §6.4. ver3 는 merge 기본, stack 은 appendix.
4. **AI Hub 한국어 replay 포함 여부** — Tier 2. 공개 adapter 제외 확정이므로 `HanMed-Internal-LoRA` 로만. 포함 시 이득이 실측 < 1%p 이면 미포함 논문 채택.
5. **200M run 실행 시점** — M4 week 1 vs M4 week 3. Tier 2/3 데이터 확보 진행도에 의존.
6. **CBETA 한문 20% 믹스 유지 여부** — 수집 진행 안 된 상태. ver3 M3 는 **CBETA 0%** 로 본 run, M4 Tier 2/3 에서 복원 고려. ver2.2 §C.2 의 "CBETA 20%" 는 `HanMed-Internal-LoRA` 에만 적용.
7. **LoRA rank ablation 우선순위** — rank 16 (compute 절약) vs rank 64 (capacity 증가). 둘 중 하나만 한다면 rank 16 (trainable 42M, A6000 여유 확보 + 공개 adapter size 감소).
8. **SFT synthetic QA 증강 모델 선택** — GPT-4o vs Claude Sonnet 4.6. ver2.2 §07.4.1 "경쟁 LLM 학습 금지" 조항 조건부 통과. ver3 는 **GPT-4o primary, Claude backup**, 양쪽 `synthesis_provenance` 태그 구분 기록.

## 10. 부록 A — pilot run 원시 metric (train.log 발췌)

```
start: 2026-04-16T22:14:44+09:00
world_size: 2 (DDP)
total_steps: 156
warmup_steps: 7
effective_tokens_per_step: 131,072

step10  loss=2.807  grad_norm=0.551  lr=9.996e-05  epoch=0.0597
step20  loss=2.515  grad_norm=0.382  lr=9.841e-05  epoch=0.1194
step30  loss=2.385  grad_norm=0.328  lr=9.472e-05  epoch=0.1791
step40  loss=2.234  grad_norm=0.398  lr=8.904e-05  epoch=0.2388
step50  loss=2.152  grad_norm=0.443  lr=8.164e-05  epoch=0.2985
        eval_loss=2.065  eval_runtime=14.15s  samples/s=3.96
step60  loss=2.078  grad_norm=0.463  lr=7.284e-05  epoch=0.3582
step70  loss=2.043  grad_norm=0.494  lr=6.303e-05  epoch=0.4179
step80  loss=2.001  grad_norm=0.661  lr=5.263e-05  epoch=0.4776
step90  loss=1.962  grad_norm=0.472  lr=4.213e-05  epoch=0.5373
step100 loss=1.955  grad_norm=0.520  lr=3.197e-05  epoch=0.597
        eval_loss=1.913  eval_runtime=14.15s
step110 loss=1.913  grad_norm=0.459  lr=2.261e-05  epoch=0.6567
step120 loss=1.897  grad_norm=0.466  lr=1.446e-05  epoch=0.7164
step130 loss=1.902  grad_norm=0.465  lr=7.886e-06  epoch=0.7761
step140 loss=1.89   grad_norm=0.434  lr=3.178e-06  epoch=0.8358
step150 loss=1.859  grad_norm=0.416  lr=5.436e-07  epoch=0.8955
        eval_loss=1.887  eval_runtime=14.16s (epoch 0.9313)
end:    2026-04-16T~01:08  (duration 2h 53m 28s)
```

## 11. 부록 B — corpus_v1.json pin snapshot

```json
{
  "version": "corpus_v1",
  "crawl_scope": "core_14",
  "tokenizer": "MLP-KTLim/llama-3-Korean-Bllossom-8B",
  "tokenizer_extended": true,
  "tokenizer_ext_vocab_size": 128260,
  "seq_len": 2048,
  "corpora": {
    "hanmed_bilingual": { "n_blocks_clean": 19609, "n_packed_seqs": 1356, "total_tokens": 2539908 },
    "hanmed_zh_only":   { "n_blocks_clean": 19802, "n_packed_seqs":  646, "total_tokens": 1241035 },
    "hanmed_ko_only":   { "n_blocks_clean": 19186, "n_packed_seqs":  743, "total_tokens": 1431410 }
  },
  "contamination": { "hash_algo": "sha256", "hashes_loaded": 7 },
  "git_sha": "c558761e4b7265f624e06802ab7d6489d4074008"
}
```

**주의**: `total_tokens` 합 5.21M 은 pack 후 (EOS + pad 포함) 수치. unique 2.72M 과 구분 — corpus 간 seq 포맷 차이로 pack_expanded > unique.

## 12. ver2 cross-reference (필수 읽기)

이 문서를 읽는 후속 agent 는 반드시 다음 ver2 섹션과 대조:

- `ver2/01_overview/overview.md` — 프로젝트 정의
- `ver2/03_data_pipeline/acquisition.md` **§3.5** — 보조 코퍼스 라이선스 매트릭스
- `ver2/03_data_pipeline/acquisition.md` **§3.6** — Core 14 / Core 25 / 확장 계획
- `ver2/04_model_strategy/base_model_and_training.md` **§4.2.2** — Bllossom primary 근거
- `ver2/04_model_strategy/base_model_and_training.md` **§4.5** — Stage 1 CPT objective + 믹스
- `ver2/04_model_strategy/base_model_and_training.md` **§4.6** — Stage 2 SFT
- `ver2/04_model_strategy/preprocessing_and_cpt_spec.md` **§C.4.3** — cap 20M~60M 정당화
- `ver2/04_model_strategy/preprocessing_and_cpt_spec.md` **§E** — ablation null-result
- `ver2/05_evaluation/hanmed_eval.md` **§5.2~§5.6** — T1~T5 exit
- `ver2/07_license_ethics/license_ethics.md` **§7.1** — KIOM 라이선스
- `ver2/07_license_ethics/license_ethics.md` **§7.9** — 전문가 계약
- `ver2/11_implementation/work_order.md` **W0** — Core 25 재개 명령
