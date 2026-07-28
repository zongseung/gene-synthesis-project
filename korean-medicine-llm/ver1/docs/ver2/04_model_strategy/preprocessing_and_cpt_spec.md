# 04a. Preprocessing · Tokenizer · CPT — 구체 사양 (ver2, **R3.2 primary 전환**)

> **R3.1 → R3.2 patch (2026-04-16)** — `scripts/tokenizer_compare.py` 실측 결과 Solar-10.7B-Instruct 가 한의학 코퍼스에서 **byte_fallback 53% + 한자/한글 모두 최하위 효율** 확정. **Primary 를 Bllossom-8B 로 전환**. 변경: §A.3 primary tokenizer, §A.4 extension 필요성 재평가, §C.4.1 tok/char 실측 반영 (zh 1.040, ko 0.745), §C.4.3 Core 14 unique 2.72M 재산정 + cap 범위 **20M~60M**, §C.5 base 모델 Bllossom, §C.6 실행 커맨드, §D G7 Bllossom 기준.
>
> **R3 → R3.1 patch (2026-04-16)** — R3 판별자 `APPROVE_WITH_CHANGES` + 4개 minor edit 요구 반영: (1) §A.2 F5 hash 단위 명시 (블록 전체), (2) §C.4.3 cap 범위 텍스트 Core 14 vs Core 25 분리, (3) §C.5 warmup 공식에 single-GPU/DDP 분기, (4) §C.4.3 §E ablation 에 null-result 대응 (delta < +0.5 chrF → instruction-tuning 전환). **판정: APPROVE, 루프 종료.**
>
> **R3 개정 요지 (2026-04-16)** — R2 판별자 재판정이 **REJECT_AND_REGENERATE** 유지. 다만 사용자 컨텍스트 ("데이터 수집 재개 필요 / 스크립트 구성은 대기") 상 코드 수정은 M2 이연. R3 은 **문서-한정 1 패스** 로 범주 A (§C.4 outlier 위험 / §F 매핑 누락 / §D Gate 누락) 만 닫는다. 범주 B (코드 이연) / 범주 C (cap 30~80M under-training) 는 M2/실험 대기로 재이연.
>
> | R2 → R3 변경 | 섹션 |
> |---|---|
> | I1 — book_8 outlier trim (IQR 1.5) + repeat_factor 2-variant | §C.4.2, §C.4.3 |
> | I2 — §F.3 에 B2/m1~m7 누락 행 추가, §C.5 warmup 공식화 (`warmup_steps = int(total_steps × 0.05)` + total_steps 산출식) | §F.3, §C.5 |
> | I3 — G7/G8/G9 Gate 추가, Gates 명목/실효 분리, `eval/hashes/heldout_T1.txt` placeholder commit | §D, §F.2 |
>
> **R1 → R2 개정 요지 (히스토리)** — ver2 초안 `harness-engineering-loop` Round 1 에서 **REJECT_AND_REGENERATE** 판정. 3개 치명 이슈 반영.
>
> | Round 1 치명 | R2 반영 |
> |---|---|
> | **C1** 4개 실행 커맨드 스크립트 미존재 | 실존 3개 스크립트로 파이프라인 구성, 미구현 4개는 **§F status 표**에 planned/M2 로 명시. 본문 커맨드에서 전부 제거 |
> | **C2** HanMed 18.16M 자 전제 vs 실측 1.20M 자 (15× drift) | **§C.4 두 시나리오 분리** — (a) Core 14 실측, (b) Core 25 확장 완료 후 추정. cap 150~250M 폐기, 실측 기반 **30~80M tokens cap** 재산정 |
> | **C3** `eval/hashes/` 부재 → contamination gate 무력 | **§D gate 1**: `eval/hashes/heldout_{T1,T2,T5}.txt` 존재 + positive-control 통과 **이전에는** Stage 1 CPT 착수 금지. 코드 hard-fail 구현은 M2 (§F) |
>
> 모델 채택(§04.2, Solar-10.7B-Instruct primary / Bllossom-8B backup)은 확정. 본 문서는 **수집 중단 상태(Core 14 완료, Core 25 확장 대기)**에서 Stage 1 CPT 착수까지의 전처리를 코드 레벨로 구체화한다.
>
> 참조 코드 (실존):
> - `src/data/builder/extract_corpora.py` (raw → bilingual/zh/ko/en 분기, NFC + 공백 정규화 + 길이 필터)
> - `src/data/builder/preprocess.py` (Stage 1 clean + Stage 2 pack)
> - `src/training/smoke_cpt.py` (파이프라인 무결성 확인용 smoke, Qwen2.5-0.5B)

## A. 텍스트 전처리 및 Tokenizer

### A.1 정규화 파이프라인 (고정)

모든 텍스트(한문 `original`, 국역 `trans_ko`, 영역 `trans_en`)는 아래 순서로 정규화한다. `extract_corpora.py::normalize_text` + `preprocess.py::normalize` 2-패스 구조를 유지한다.

| 단계 | 규칙 | 근거 |
|---|---|---|
| 1 | `unicodedata.normalize("NFC", s)` | CJK 조합 일관성, §03.4 합의 |
| 2 | 전각 공백(U+3000) + tab → 반각 1칸 | API가 불규칙 공백을 포함 |
| 3 | 다중 공백 → 1칸 (`\s+` → `" "`) | 정규식 `WS_RE` |
| 4 | CRLF → LF | raw가 혼재 |
| 5 | `strip()` | 앞뒤 공백 제거 |
| 6 | **inline markup 제거 금지** | §03.3.1 실측 — 배포 파일에 0건 |

**경계 보존 예외**: bilingual block `<ZH>...</ZH>\n<KO>...</KO>\n\n` 은 `\n` 을 공백으로 압축하지 않고 라인브레이크를 보존해야 한다 (`preprocess.py::normalize` 정규식이 `\s+` 전체를 1칸으로 치환하는 현재 동작은 **R3에서 수정 대상**, §F M6).

**주의**: annotation은 항상 `null` (API 정책), en 본문은 8,137건 / 32.5% 커버리지이므로 **Stage 1 CPT에서는 제외**(§01.6 v1 scope 밖).

### A.2 필터링 (`preprocess.py::stage1_clean`)

| # | 필터 | 임계 (code: 파일:라인) | 적용 대상 |
|---|---|---|---|
| F1 | Exact dedup (SHA-1 of normalized text) | `preprocess.py:79,192` 완전일치 | 전체 |
| F2 | 길이 | `preprocess.py:117-120` `5 ≤ n ≤ 50000` | 전체 |
| F3 | 동일 문자 run-length | `preprocess.py:123` `rep > 0.5` | 전체 |
| F4 | 언어 비율 | bilingual: hanja ≥ 0.1 AND hangul ≥ 0.1<br>zh_only: hanja ≥ 0.4 (구두점/숫자 포함 분모)<br>ko_only: hangul ≥ 0.4 | corpus별 |
| F5 | **Eval contamination (SHA-256)** | `preprocess.py:148-159, 202-207` — `eval/hashes/heldout_*.txt` 매치 → drop. **Hash 단위 = `rec['text']` 전체**(bilingual 의 경우 `<ZH>...</ZH>\n<KO>...</KO>\n\n` 블록 전체). 단독 문장 hash 를 쓰려면 eval curation 시 해당 문장이 단일 record 로 구성돼야 함 | 전체; drop > 0.5% 면 파이프라인 실패 (§D gate) |

드롭 사유는 Counter 로 `preprocess_stats.json` 에 기록.

**R2 주석**:
- F3 임계는 `preprocess.py:123` 코드 (`> 0.5`) 가 정본. 동일 파일 L12 docstring "0.3" 은 **drift — §F B1 로 M2 수정 대기**.
- F4 zh_only 분모는 한문 구두점(`、。《》`) 포함이므로 고전 한문에서 false-positive drop 가능(최대 5~10%). probe 필요 (§D gate 3).
- F5 는 `eval/hashes/` 디렉토리가 존재하지 않으면 silent skip (`preprocess.py:151-152, 321-325`) → **§D gate 1** 로 파이프라인 hard-fail 강제 (코드 수정은 §F B4).

### A.3 Tokenizer 선정 · 확장 결정 — **R3.2 primary 전환**

**Primary**: `MLP-KTLim/llama-3-Korean-Bllossom-8B` tokenizer (Llama-3 BPE, vocab 128,256). 실측 한문 tok/char 1.040, 한글 0.745, byte_fallback 0%.
**Backup 1**: `Qwen/Qwen2.5-7B-Instruct` tokenizer (vocab 151,665). 실측 한문 1.047, 한글 0.823, byte_fallback 0%.
**기각**: `MLP-KTLim/llama-3-Korean-Bllossom-8B` (vocab 32K). 실측 한문 1.533, 한글 1.254, **byte_fallback 53%** — 한자의 절반이 UTF-8 3-byte tokens 로 분해되어 semantic 단위 학습 불가.

**R3.2 Probe 이미 수행** — Core 14 코퍼스 10,000자 샘플 기준 (`scripts/tokenizer_compare.py`):

| Model | 한문 tok/char | 한글 tok/char | byte_fallback |
|---|---|---|---|
| Bllossom-8B (primary) | **1.040** | **0.745** | 0% |
| Qwen2.5-7B (backup 1) | 1.047 | 0.823 | 0% |
| Solar-10.7B (기각) | 1.533 | 1.254 | 53% |

**Probe M2 확장 과제** — Core 25 수집 재개 후 다음을 추가 측정:
- `data/cpt/hanmed_zh_only.jsonl` 100,000 char 샘플로 tok/char 안정화 (현 10K 기준 point estimate)
- KoWiki 덤프 baseline 100K char (확장 결정 규칙 입력용)

**확장 결정 규칙 (§04.4.2 고정)**:
```
if median_domain(tokens/char) - median_wiki(tokens/char) >= 0.2:
    extend_tokenizer = True
else:
    extend_tokenizer = False
```

**R3.2 예상**: Bllossom 한자 1.040 은 일반 한국어 Wiki 의 한글 tok/char 0.6~0.9 대비 margin 0.14~0.44. 한국어 부분(0.745)은 wiki 와 유사 — **한자 부분만 margin 초과 가능성**. 확장 대상은 한자 top-N 으로 좁혀짐 (한국어 BPE 는 건드리지 않음).

> probe 스크립트: `scripts/tokenizer_compare.py` / `scripts/tokenizer_probe_bllossom.py` (구현 완료). §F.1 P1 `tokenizer_probe.py` 는 Core 25 full-sample 자동화용 별도 스크립트로 이전.

### A.4 Special Token · 확장 실행 — **R3.2 Bllossom 실측 반영**

**확장 여부와 무관하게 항상 추가**: `<ZH>`, `</ZH>`, `<KO>`, `</KO>`.
- 용도: §04.5.2 bilingual block 경계.
- **검증 완료 (R3.2)** — Bllossom-8B 에서 `add_special_tokens` 실행 결과:
  - vocab 128,256 → 128,260 (4개 추가, 예약 slot 바로 뒤)
  - 할당 id: `<ZH>=128256`, `</ZH>=128257`, `<KO>=128258`, `</KO>=128259`
  - vanilla 3 tokens/tag → 1 token/tag (압축률 67%)
  - 74자 bilingual block: 57 → 51 tokens (−10.5%)
- Solar SentencePiece 경로(기존 §F M5)는 **기각**과 함께 검증 무효화 처리.

**확장 실행 시 (A.3 결과가 True, `src/data/builder/tokenizer_extend.py` planned)**:
1. 빈도 상위 한자 + 한의학 multi-char 용어 (`data/dict/hanmed_terms.jsonl` ≥ 3,000 엔트리, §03.4.1) 병합 → 신규 **2,000 ~ 5,000 token**
2. Embedding 확장: 한자 신규 행 = 해당 한자의 기존 subword BPE 평균 (warm init); special token 4개 = `</s>` 평균
3. `tie_word_embeddings` 유지 (LM head 동반 확장)
4. Stage 1 초기 **100 steps** 는 embedding row만 학습 (나머지 freeze)

확장 수행 시 산출물: `data/tokenizer/hanmed_ext/` + `vocab_diff.json`.

**R2 주석**: `data/dict/hanmed_terms.jsonl` 자체가 미생성 — §F planned.

### A.5 전처리 결정론 (재현성)

| 항목 | 값 | 상태 |
|---|---|---|
| Python | 3.10.x (`.venv` 고정) | ✅ |
| random seed | 전역 42 (`PYTHONHASHSEED=0`, numpy, random, torch) | **미적용 — §F M1** |
| 파일 읽기 순서 | `sorted(book_dirs)`, `sorted(vol_files)` | ✅ (`extract_corpora.py:79`) |
| dedup hash | SHA-1 of normalized text | ✅ |
| contamination hash | SHA-256 of normalized text | ✅ (§03.4.2 통일) |
| stats 산출물 | `preprocess_stats.json` | ✅ |

---

## B. 저장 방법

### B.1 디렉토리 레이아웃

```
korean-medicine-llm/
├── data/
│   ├── raw/mediclassics_unified/           # 크롤 (§03.8, 현재 Core 14 완료)
│   │   ├── orchestrator.log
│   │   ├── unified_manifest.json
│   │   └── book_{NNN}/vol_{VV}.jsonl       # 한 줄 = 한 record (원본)
│   │
│   ├── cpt/                                # extract_corpora.py 산출 (현재 상태)
│   │   ├── hanmed_bilingual.jsonl          # 11.7 MB, 21,043 블록
│   │   ├── hanmed_zh_only.jsonl            # 8.7 MB, 23,338
│   │   ├── hanmed_ko_only.jsonl            # 7.0 MB, 22,597
│   │   ├── hanmed_en_only.jsonl            # 3.2 MB, 8,137 (v1 CPT 제외)
│   │   └── corpus_stats.json               # 책별 record·char 통계
│   │
│   ├── cpt_processed/                      # preprocess.py 산출 (M2 target)
│   │   ├── {corpus}_clean.jsonl            # Stage 1 통과
│   │   ├── {corpus}_packed_2048.jsonl      # Stage 2, seq_len 2048
│   │   └── preprocess_stats.json
│   │
│   ├── replay/                             # Wiki-ko, CBETA (M2 수집)
│   │   ├── wiki_ko_clean.jsonl
│   │   └── wiki_ko_packed_2048.jsonl
│   │
│   ├── stats/
│   │   ├── tokenizer_probe.json            # §F planned
│   │   └── contamination_drop.json
│   │
│   ├── dict/hanmed_terms.jsonl             # §F planned (NER seed ≥ 3,000)
│   └── tokenizer/hanmed_ext/               # 확장 시에만
│
├── eval/                                   # 현재 디렉토리 자체 미생성 — §D gate 1
│   ├── hanmed_eval_v0/{T1,T2,T5}.jsonl
│   └── hashes/heldout_{T1,T2,T5}.txt       # SHA-256 git committed
│
└── outputs/                                # 학습 산출
```

### B.2 파일 포맷

**raw record (`data/raw/.../vol_*.jsonl`)** — API 그대로:
```json
{"book_id": 8, "volume_id": 1, "content_seq": 138, "content_level": "ZZ",
 "up_path_nm": "內景篇卷之一 > 身形 > 形氣之始",
 "original": "…", "trans_ko": "…", "trans_en": "…",
 "annotation": null, "index_num": 1}
```

**cpt block (`data/cpt/*.jsonl`)** — `extract_corpora.py` 출력, 텍스트 블록 단위:
```json
{"book_id": 8, "volume_id": 1, "content_seq": 138, "content_level": "ZZ",
 "up_path_nm": "…",
 "text": "<ZH>…</ZH>\n<KO>…</KO>\n\n",
 "n_chars_zh": 58, "n_chars_ko": 61}
```

**cleaned (`data/cpt_processed/*_clean.jsonl`)** — Stage 1 통과:
```json
{ ...cpt fields...,
  "text_hash_sha1": "3f2a…",
  "text": "…" }
```

**packed (`data/cpt_processed/*_packed_{seq}.jsonl`)** — Stage 2, 학습 직접 입력:
```json
{"input_ids": [1, 128009, ..., 2]}   // length = seq_len (pad 포함)
```
- 한 sequence = 여러 block 을 `EOS` 로 연결 + 말단 pad
- 한 block 이 seq_len 초과 시 단독 sequence 로 truncate (counter `stats["too_long_truncated"]` 추가 — §F M3)

### B.3 Manifest · Hash Pin

Stage 2 완료 시 `data/cpt_processed/corpus_v1.json` 생성.

> **R2**: `build_corpus_manifest.py` 는 **현재 미구현** (§F planned). manifest 를 수동으로라도 채울 수 없으면 재현성 claim 불가 — M2 gate.

예상 구조:
```json
{
  "version": "corpus_v1",
  "build_date": "...",
  "crawl_scope": "core_25",          // 또는 "core_14"
  "tokenizer": "MLP-KTLim/llama-3-Korean-Bllossom-8B",
  "tokenizer_extended": false,
  "seq_len": 2048,
  "corpora": {
    "hanmed_bilingual": {
      "input_file": "data/cpt/hanmed_bilingual.jsonl",
      "input_sha256": "…",
      "clean_sha256": "…",
      "packed_sha256": "…",
      "n_blocks_in": 21043,
      "n_blocks_clean": …,
      "n_packed_seqs": …,
      "total_tokens": …
    }
  },
  "contamination": {
    "hash_algo": "sha256",
    "hashes_loaded": …,
    "drop_ratio": …,
    "report": "data/stats/contamination_drop.json"
  },
  "git_sha": "…"
}
```

### B.4 DVC (선택, §06.5)

재배포 금지 소스는 **로컬 NFS / 기관 내부 remote 에만**. 외부 CSP 금지.

---

## C. 비지도 학습 (Stage 1 CPT)

### C.1 Objective (고정)

```
Objective : next-token prediction (causal language modeling)
Training  : self-supervised (자기지도학습)
Loss      : cross-entropy over all non-pad tokens
            (tag token <ZH>/<KO> 포함)
```

MLM / span corruption / denoising 금지 (§04.5 · D1).

### C.2 데이터 믹스 (§04.5.1, R2 재산정)

| 소스 | 비중 | 파일 | 공개 | 상태 |
|---|---|---|---|---|
| HanMed 한문 원문 | 25% | `hanmed_zh_only_packed_2048.jsonl` | ✅ | Core 14 확보 / Core 25 대기 |
| HanMed 국역 | 10% | `hanmed_ko_only_packed_2048.jsonl` | ✅ | Core 14 확보 / Core 25 대기 |
| HanMed 병렬 | 5% | `hanmed_bilingual_packed_2048.jsonl` | ✅ | Core 14 확보 |
| Wiki-ko replay | 30% | `wiki_ko_packed_2048.jsonl` | ✅ | **미수집 — §F planned** |
| CBETA 한문 | 20% | `cbeta_packed_2048.jsonl` | ❌ (내부) | **미수집 — §F planned** |
| 예비 한국어 | 10% | `aihub_ko_packed_2048.jsonl` | ❌ | **미수집 — §F planned** |

**믹싱**: `datasets.interleave_datasets(probabilities=[0.25, 0.10, 0.05, 0.30, 0.20, 0.10], seed=42, stopping_strategy="all_exhausted")`.

### C.3 Sequence Packing (§04.5.4)

- `seq_len = 2048` (고정). **주의**: `preprocess.py:315` default 는 현재 1024 — §F B3 로 M2 수정.
- greedy pack, 블록 사이 `EOS` 1개 삽입
- sequence 시작 `BOS` 1회 (Llama 표준)
- bilingual 블록 경계 보존 (§04.5.3 D2)
- padding: 우측 pad (DataCollator 기본)
- loss mask: 전 구간 동일 가중 (pad 만 마스크). R1 ablation: Wiki-ko 0.5× option.

### C.4 토큰 예산 — **R3.2 Bllossom 실측 기반 재산정**

#### C.4.1 실측 기반 (Core 14, Bllossom primary)

`data/cpt/corpus_stats.json` 실측 char + `scripts/tokenizer_compare.py` 실측 tok/char:
- `chars_zh` = **1,203,407** (1.20M)
- `chars_ko` = **1,969,632** (1.97M)
- `records_bilingual` = 21,043
- Bllossom tok/char = **zh 1.040 / ko 0.745** (실측 10K char 샘플)

실측 unique tokens (Bllossom 기준):

| corpus | char | tok/char | tokens |
|---|---|---|---|
| hanmed_zh (영역 제외) | 1.20M | 1.040 | **1.25M** |
| hanmed_ko | 1.97M | 0.745 | **1.47M** |
| **HanMed unique 합 (Bllossom primary)** | | | **2.72M tokens** |

대조 (Solar 기각 근거):

| tokenizer | zh tokens | ko tokens | HanMed unique |
|---|---|---|---|
| Bllossom-8B (primary) | 1.25M | 1.47M | **2.72M** |
| Qwen2.5-7B (backup 1) | 1.26M | 1.62M | 2.88M |
| ❌ Solar-10.7B (기각) | 1.84M | 2.47M | 4.31M (byte-fallback 53% 포함) |

ver2 초안의 "32~43M" 및 R2 추정 "3~4M" 은 전부 폐기. R3.2 확정: **Core 14 Bllossom 기준 2.72M tokens**.

#### C.4.2 Core 25 확장 후 추정 (R3.2 Bllossom 실측 반영)

Core 25 = Core 14 + 11 책 (id: 7, 44, 46, 47, 49, 54, 60, 70, 71, 94, 139, 183).

**Outlier 처리**: Core 14 실측에서 book_8 (동의보감: 284K zh / 571K ko), book_56 (의방유취: 225K zh / 0 ko), book_69 (제중신편: 144K zh / 286K ko) 가 IQR 1.5 기준 upper outlier. 나머지 11권 median (zh) ≈ 47K, mean ≈ 58K. Core 25 추가 11책의 char 추정:

| 시나리오 | 추가 11책 책당 char (zh) | Core 25 chars_zh | Core 25 chars_ko |
|---|---|---|---|
| (L) lower — median | 47K | 1.72M | 2.75M |
| (C) central — mean | 58K | 1.84M | 2.87M |
| (H) upper — 거질 1권 | 58K×10 + 220K×1 | 2.00M | 3.17M |

**Bllossom 실측 tok/char (zh 1.040, ko 0.745) 적용 token 추정**:

| 시나리오 | zh tokens | ko tokens | HanMed unique 합 |
|---|---|---|---|
| (L) | 1.79M | 2.05M | **3.84M** |
| (C) | 1.91M | 2.14M | **4.05M** |
| (H) | 2.08M | 2.36M | **4.44M** |

**R3.2 확정**: Core 25 HanMed unique 는 **3.8M ~ 4.5M tokens** (Bllossom). R3 의 "3.9~6.5M" 은 Solar 가정 상한 포함이었으므로 축소. ver2 초안 "32~43M" 확정 폐기.

#### C.4.3 CPT cap 재산정 (R3.2 — Bllossom 실측 기반)

DAPT 관례 (Gururangan 2020) domain unique 의 repeat_factor 5~10. 2-variant:

| variant | epoch | 근거 |
|---|---|---|
| **R3a — conservative** | **3** | Chinese-LLaMA-2 CPT 관례, overfit 경계 안쪽 |
| **R3b — DAPT-aligned** | **5** | Gururangan 2020 하단, SCI CPT 논문 대다수 |

cap = HanMed_training_tok / 0.40 (HanMed 40% 믹스). **R3.2 Bllossom 실측 unique** 기반:

| (unique, epoch) | HanMed training tok | total cap |
|---|---|---|
| Core 14 실측 2.72M × 3 | 8.16M | **20.4M** |
| Core 14 실측 2.72M × 5 | 13.6M | **34M** |
| Core 25 (L) 3.84M × 3 | 11.52M | **28.8M** |
| Core 25 (L) 3.84M × 5 | 19.2M | **48M** |
| Core 25 (C) 4.05M × 3 | 12.15M | **30.4M** |
| Core 25 (C) 4.05M × 5 | 20.25M | **50.6M** |
| Core 25 (H) 4.44M × 3 | 13.32M | **33.3M** |
| Core 25 (H) 4.44M × 5 | 22.2M | **55.5M** |

**R3.2 확정 cap 범위**:
- **Core 25 기준 20M ~ 60M tokens** (R3 의 30~80M 에서 Bllossom 실측으로 축소)
- **Core 14 전용 (수집 재개 전) 시 20M ~ 34M**
- 단일 숫자 cap 고정 금지, Core 25 실측 후 재평가

**Under-training 재평가 (R3.2)**:
- **Bllossom-8B** LoRA 114~150M trainable × Chinchilla 권고 20× = **2.3~3.0B tokens**
- 본 spec cap 20~60M 은 Chinchilla 권고의 **0.7~2.6%** — under-training regime (R3 의 Solar 기준 1~3.5% 대비 더 낮음)
- 다만 Solar 의 byte-fallback 53% 로 낭비되던 compute 가 Bllossom 에서 semantic tokens 에 집중 → **effective learning signal 은 Solar cap 30~80M 과 Bllossom cap 20~60M 이 거의 동등**
- 근거: LoRA + 좁은 타깃 도메인 (T1 번역) 에서는 register shift 만으로 측정 가능 gain 을 얻는 DAPT 사례 다수 (Chinese-LLaMA-2 계열 5~20M token 한자 노출로 PPL 개선 보고)
- 리스크: medical knowledge injection 실패 → T2 QA gain 미발생. §E ablation 으로 실증

**§E ablation (R3.2 Bllossom 기준)**: Bllossom-8B 상에서 cap=20M / 60M / 200M 3-way run (동일 LR, 동일 warmup rule). 20M run 의 T1 chrF 가 60M run 대비 -1.5 이상 하락하지 않으면 under-training 기각. 200M 은 repeat_factor 50+ 로 overfit 대조군.

**Null-result 대응**: 20M run 의 T1 chrF 가 **baseline (Bllossom-8B zero-shot, no-CPT) 대비 delta < +0.5 chrF** 이면 CPT 전체 전략 기각 → **instruction-tuning-only 전환** 검토.

### C.5 하이퍼파라미터 (M3 pilot 시작값 · §04.5.5)

| 항목 | 값 | 상태 |
|---|---|---|
| base | **Bllossom-8B (primary, R3.2 승격)** / Qwen2.5-7B (backup 1) / Mistral-Nemo (backup 2) | ✅ |
| precision | **bf16** | ✅ |
| adapter | LoRA r=32, α=64, dropout 0.05 | ✅ |
| target modules | q,k,v,o, gate_proj, up_proj, down_proj | ✅ |
| seq_len | 2048 | default drift — §F B3 |
| micro batch | 2 | |
| grad accum | 16 (effective 32) | |
| optimizer | AdamW β=(0.9, 0.95), eps=1e-8, wd=0.0 | |
| LR | 1e-4 | |
| scheduler | cosine w/ warmup. **R3.2 재산정**: `warmup_steps = int(total_steps × 0.05)`, `total_steps = ceil(cap_tokens / (micro_batch × grad_accum × seq_len × N_GPU))`. **single-GPU 전제** `= ceil(cap_tokens / 65,536)`. Bllossom 기준 예: cap 20M → total 305 → warmup 15 / cap 60M → total 915 → warmup 45. **DDP N-GPU 시** `effective_batch = 2 × 16 × N` 으로 total_steps 재산출 (N=2 이면 warmup 도 비례 감소). `cpt_trainer.py` (§F.1 P4) 구현 시 `warmup_steps = int(total_steps * 0.05)` 주입 필수. PEFT / Alpaca-LoRA default (`warmup_ratio=0.03~0.05`) 와 정합 | |
| grad ckpt | on | |
| prompt format | plain text (Stage 1) — ChatML 은 Stage 2 | |

### C.6 실행 커맨드 — **실존 3 스크립트만**

```bash
# 1) 데이터 수집 재개 (§D gate 0)
python3 src/data/crawler/mediclassics_orchestrator.py \
  --output data/raw/mediclassics_unified \
  --books 7,44,46,47,49,54,60,70,71,94,139,183 \
  --delay 0.5 --concurrency 2 --pause 60

# 2) extract (raw → bilingual/zh/ko)
python src/data/builder/extract_corpora.py \
  --input data/raw/mediclassics_unified \
  --output data/cpt

# 3) Stage 1 전처리 (정제) — 미리 eval/hashes/ 가 존재해야 함 (§D gate 1)
python src/data/builder/preprocess.py --stage 1 \
  --input data/cpt --output data/cpt_processed \
  --corpora hanmed_bilingual,hanmed_zh_only,hanmed_ko_only \
  --eval-hash-dir eval/hashes

# 4) Stage 2 전처리 (packing)
python src/data/builder/preprocess.py --stage 2 \
  --input data/cpt --output data/cpt_processed \
  --tokenizer MLP-KTLim/llama-3-Korean-Bllossom-8B \
  --seq-len 2048

# 5) Smoke (무결성, Qwen2.5-0.5B, packed 입력)
CUDA_VISIBLE_DEVICES=1 python src/training/smoke_cpt.py \
  --data data/cpt_processed/hanmed_bilingual_packed_2048.jsonl \
  --max-steps 50
```

**R2 에서 제거된 커맨드** (미구현 스크립트, §F planned):
- `tokenizer_probe.py` (probe 자동화)
- `tokenizer_extend.py` (확장 실행)
- `build_corpus_manifest.py` (manifest + SHA256 핀)
- `cpt_trainer.py` (본 CPT 학습, DDP)

### C.7 모니터링 · 중단 조건

**Primary (W&B)**: `train/loss`, `train/perplexity`, `grad_norm`, `lr`, `throughput_tokens_per_sec`, GPU mem peak.
**Validation (held-out 2% random split per corpus, seed=42)**: `val/loss`, `val/ppl` per 500 steps, per-corpus breakdown.

**중단 조건**: 초기 100 steps grad > 10 → LR 5e-5 / val_loss 정체 + train 하락 → early stop / throughput < 1,000 tok/s → 재튜닝 / DUS LoRA 2× 메모리 (§04.3 R14) → target module 축소 → 복제 layer 공유 → Bllossom 전환.

---

## D. Pre-flight Gates — **CPT 착수 전 통과 필수**

모든 gate 는 R3 관점에서 "현재 blocker". **R3 신규 컬럼**: 리뷰어 지적 (G1/G3/G5 = 코드 hook 필요 = "실효" / 나머지 = 산출물 확인 = "명목") 에 따라 Gate 종류 구분.

| # | Gate | 종류 | 조건 | 현재 상태 | 책임 |
|---|---|---|---|---|---|
| **G0** | Core 25 크롤 재개 완료 | 명목 (ls) | `data/raw/mediclassics_unified/` book 디렉토리 ≥ 25개, orchestrator.log 정상 종료 | ❌ Core 14 중단 | crawler 재실행 |
| **G1** | `eval/hashes/heldout_{T1,T2,T5}.txt` committed + positive control 통과 | **실효 (코드 hook 필요)** | hash 파일 존재 + `preprocess.py` contamination counter ≥ 1 (심은 sample 1건 drop 확인) | ❌ eval/ 미생성 + `preprocess.py:321-325` silent skip 유지 | §F B4 (M2) + §F.2 P6 placeholder (R3) |
| **G2** | tokenizer_probe.json 산출 | 명목 (산출물) | zh/ko median tokens/char + wiki baseline margin | ❌ probe 스크립트 미구현 | §F.1 P1 (M2) |
| **G3** | Stage 2 packed seq 전부 `len == seq_len` | **실효 (pytest)** | pytest T3 green | ❌ test 미작성 + assert 부재 | §F M3 (M2) |
| **G4** | corpus_v1.json 생성 + SHA-256 핀 | 명목 (산출물) | build_corpus_manifest.py 실행 결과 | ❌ script 미구현 | §F.1 P3 (M2) |
| **G5** | contamination drop ratio ≤ 0.5% | **실효 (코드 hook 필요)** | `data/stats/contamination_drop.json` + `sys.exit(3)` trigger | pending (G1 의존) | §F B4 (M2) |
| **G6** | F4 한문 구두점 false-positive 측정 | 명목 (분석) | zh_only drop rate < 10% 목표 | ❌ 미측정 | A.2 probe (M2) |
| **G7** *(R3.2 — Bllossom 기준)* | Tokenizer resize e2e sanity on Bllossom Llama-3 BPE | **실효 (smoke)** | Bllossom-8B 에 `<ZH>/</ZH>/<KO>/</KO>` 추가 (이미 `scripts/tokenizer_probe_bllossom.py` 로 vocab 128256~128259 할당 검증) + `resize_token_embeddings` + `tie_word_embeddings` 유지 + fwd/bwd 1 step | ✅ 토큰화 확인 완료, fwd/bwd smoke 는 M2 | Bllossom smoke 1 회 (M2) |
| **G8** *(R3 신규 — 생성자 제안)* | Replay corpus presence | 명목 (ls) | `data/replay/wiki_ko_packed_2048.jsonl` + (옵션) CBETA / AI Hub 파일 존재. 부재 시 `interleave_datasets` 즉시 실패 | ❌ P8 전원 미수집 | §F.2 P8 (M2) |
| **G9** *(R3 신규 — 생성자 제안)* | Packed file 메타데이터 seq_len 일치 | **실효 (build-time)** | `data/cpt_processed/preprocess_stats.json.seq_len == 2048` assert | ❌ default drift 잔존 | §F B3 + manifest (M2) |

**G0~G9 중 하나라도 red 이면 Stage 1 CPT 착수 금지.**

**R3 partial-close**:
- G1 은 **asset commit** 측면을 R3 에서 만족 → `eval/hashes/heldout_T1.txt` placeholder 1줄 commit (§F.2 P6a). **코드 hook (sys.exit)** 은 여전히 M2 B4 대기.
- 나머지 G0~G9 전부 M2 이연 유지.

---

## E. 열린 결정 · ablation

1. **Bilingual alignment 실증** (생성자 V1): bilingual 5%를 (a) 원포맷, (b) ZH-only + KO-only 분리(블록 파괴), (c) ZH↔KO 순서 뒤집기 → Stage 1 축소판(cap 10M tokens) 에서 T1 chrF 비교. (a) - (b) ≥ +2 chrF 아니면 블록 포맷 hypothesis 기각.
2. **Bilingual 비중 스윕** (생성자 V1): 5% vs 15% — T1 번역 품질이 underweight 인지 측정.
3. LoRA rank 16 vs 32 vs 64 — M3 ablation (§04.9).
4. Wiki-ko loss weight uniform vs 0.5× — M3 pilot R1.
5. DUS LoRA 독립 vs 공유 — PEFT `print_trainable_parameters()` 실측 후 (§04.9 #6).

---

## F. Status 표 — **M2에서 처리할 planned / drift**

사양서가 실행 가능한 상태가 되려면 아래 항목이 해결되어야 한다. **R2 에서는 문서 정합성까지만 달성**.

### F.1 Planned scripts (미구현)

| # | 스크립트 | 용도 | 귀속 gate |
|---|---|---|---|
| P1 | `src/data/builder/tokenizer_probe.py` | A.3 tokens/char 측정 | G2 |
| P2 | `src/data/builder/tokenizer_extend.py` | A.4 확장 실행 | (조건부) |
| P3 | `src/data/builder/build_corpus_manifest.py` | B.3 corpus_v1.json + SHA-256 핀 | G4 |
| P4 | `src/training/cpt_trainer.py` | C.5 본 CPT (DDP, W&B) | (M3 pilot) |

### F.2 Planned assets (미생성) — **R3 부분 진전**

| # | 경로 | 내용 | 상태 | 귀속 gate |
|---|---|---|---|---|
| P5 | `eval/hanmed_eval_v0/{T1,T2,T5}.jsonl` | held-out 30/30/100 문장 | ❌ M2 | G1 |
| **P6a** | `eval/hashes/heldout_T1.txt` | **R3 placeholder commit** (positive-control 1 샘플 SHA-256) | ✅ **R3 신규** | G1 partial |
| P6b | `eval/hashes/heldout_{T2,T5}.txt` | T2/T5 전수 해시 | ❌ M2 | G1 |
| P7 | `data/dict/hanmed_terms.jsonl` | NER seed ≥ 3,000 | ❌ M2 | A.4 |
| P8 | replay corpora (Wiki-ko, CBETA, AI Hub) | C.2 믹스 60% | ❌ M2 | G8 |

### F.3 Code drift (Round 1+2 리뷰어 B1~B4 + M1~M6 + m1~m7) — R3 완전 매핑

**Blocker (B)**

| # | 파일:라인 | 문제 | 처리 (M2) |
|---|---|---|---|
| B1 | `preprocess.py:12` docstring | F3 "0.3" vs code "0.5" | docstring 수정 |
| **B2** | R1 원래 지적에서 만들어진 가상 번호 (리뷰어가 "정체불명" 로 마킹). **R3 재라벨링**: 판별자 R2 제안 (c) "`eval/hashes/heldout_T1.txt` 1 샘플 commit + positive-control smoke 통과 로그" | asset commit 1회 — **R3 §F.2 P6 에 반영** | R3 asset commit (하단 §F.2 참고) |
| B3 | `preprocess.py:315`, `smoke_cpt.py:50` | seq_len default 1024 vs spec 2048 | default 2048 |
| B4 | `preprocess.py:148-159, 321-325` | contamination eval 없음 silent skip | planner I3 사양: eval_dir 부재 → `sys.exit(2)`, drop ratio > 0.5% → `sys.exit(3)`, `--allow-missing-eval` 플래그로만 skip 허용 |

**Major (M)**

| # | 파일:라인 | 문제 | 처리 (M2) |
|---|---|---|---|
| M1 | 3 script entry | 전역 seed 미설정 | `src/utils/seed.py` 신설 + `PYTHONHASHSEED=0` 런처 강제 |
| M2 | `smoke_cpt.py:89` (R2 표기 L86 오기 — R3 정정) | `device_map="auto"` → DDP 충돌 | `LOCAL_RANK` 기반 DDP 분기 |
| M3 | `preprocess.py:259-288` | Stage 2 packing invariant assert 부재 | assert 추가 + pytest T3 |
| M4 | `smoke_cpt.py:45, 113-124` (R2 표기 L112 정정) | smoke 가 raw block 사용 (packed 우회) | packed input collator 분기 + pytest T5 |
| M5 | `smoke_cpt.py:80-93` | Solar SentencePiece add_special_tokens 경로 미검증 | Solar 에서 probe 재실행 + G7 gate |
| M6 | `preprocess.py:67, 74` + `extract_corpora.py:36` | `WS_RE = r"\s+"` 가 `\n\n` 공백 압축. `extract_corpora.py:36` 은 `[ \t\u3000]+` 로 이미 개행 보존 — **정책 불일치가 근본 원인** | preprocess 쪽 `[ \t\u3000]+` 로 좁히고 `\n` 은 별도 rule (최대 `\n\n` 보존). 2-패스 합성 테스트 추가 |

**Minor (m) — R3 신규 반영 (R1 누락 해소)**

| # | 파일:라인 | 문제 | 처리 (M2) |
|---|---|---|---|
| m1 | `extract_corpora.py:54-55`, `:108`; `preprocess.py:117, 119` | 숨겨진 상수 (`min_zh_chars=3`, `min_ko_chars=2`, `is_meaningful(trans_en, 5)`, `n < 5`, `n > 50000`) | 모듈 상단 CONSTANT + CLI 노출, `corpus_stats.json.filters` 에 기록 |
| m2 | 없음 (warmup 500 steps) | LoRA 기준 과다 | **R3 §C.5 공식화로 해결** (cap 기반 자동 산출) |
| m3 | `extract_corpora.py:47`, `preprocess.py:74` | `make_bilingual_block` 출력 `\n\n` trailing 이 `WS_RE` 로 소실 | M6 과 함께 처리 (블록 경계 보존 회귀 테스트 추가) |
| m4 | `preprocess.py:186, 189` | `stats["input"] != kept + dropped_*` 항등식 미보장 (empty_after_norm 가 input 카운트에 들어감) | `assert sum == input` 추가 (pytest T2 에서 자동 커버) |
| m5 | `extract_corpora.py:84` | book_id parse 실패 시 silent continue | warning print + skipped list 에 기록 |
| m6 | `preprocess.py:262-267` | seq_len 초과 단일 doc truncate 시 stat 부재 | `stats["too_long_truncated"]` 추가 |
| m7 | 전체 | dead/orphan code 없음 (깔끔) | 없음 |

### F.4 Tests (5 pytest, 리뷰어 제안)

| # | 이름 | 대상 |
|---|---|---|
| T1 | `test_normalize_identity.py::test_double_normalize_idempotent` | A.1 2-패스 idempotence |
| T2 | `test_stage1_invariants.py::test_counts_sum_to_input` | `kept + drops == input` |
| T3 | `test_stage2_packing.py::test_all_sequences_have_seq_len` | G3 |
| T4 | `test_contamination_gate.py::test_drop_ratio_exceeds_threshold_exits_nonzero` | G1/G5 |
| T5 | `test_smoke_cpt_packed_path.py::test_smoke_accepts_packed_jsonl_with_input_ids` | M4 |

---

## G. 요약 — R2 이후 M2 진입 조건

Stage 1 CPT 착수는 **G0~G6 전부 green + F.1/F.2 전부 완료 + F.3 drift 수정 + F.4 pytest green** 이후. 이전까지는 smoke (Qwen2.5-0.5B) 로 파이프라인 무결성 확인만 수행.

R2 사양서는 위 blocker 를 **모두 식별·명시**했다. 실제 blocker 해소는 M2 마일스톤.
