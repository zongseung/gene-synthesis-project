# 03. Data Pipeline — 획득·파싱·정제 (ver2.2 — 실측 반영)

> **ver2.2 변경**: 실제 API 호출·rate-limit 측정·multi-process orchestrator 아키텍처 반영. 초기 가정(Playwright 메타데이터, 단일 파일 다운로드, 1 req/sec)은 폐기.

## 3.1 원칙

1. KIOM 비상업 이용 범위 준수
2. **실측 sustained rate ~2.8 req/s per-book** (서버 quota 기반 — §3.2.5)
3. 모든 수집 레코드에 `book_id`, `volume_id`, `content_seq`, `crawl_date` 기록
4. 가공 코퍼스의 **공개 배포는 KIOM 서면 승인 후에만**
5. 재배포 불가 소스는 별도 디렉토리에 격리 (상세는 `ver2/07_license_ethics/`)
6. **Eval contamination 방지**: held-out 평가 문장은 CPT 입력에서 사전 필터링 (§3.4.2)

## 3.2 Acquisition (실측 확정)

### 3.2.1 데이터 출처 (verified)

| 엔드포인트 | 용도 | 비고 |
|---|---|---|
| `https://mediclassics.kr/books/{id}` (HTML) | 책 메타 (서명·권 목록) | 정규식으로 `href="/books/{id}/volume/{N}"` 추출 |
| `https://mediclassics.kr/api/books/8/volumes/` (JSON) | 동의보감 권 목록·content_total | **id=8만 200 OK, 나머지 403** |
| `https://mediclassics.kr/books/{id}/volume/{v}/content/{seq}` (JSON) | **본문 + 한문/국역/영역 + 계층경로** | **161종 모두 200 OK** |

인증 헤더 (배포 앱 JS에 hardcoded — 공개):
```
Authorization: 5fe23edf9dec4c718e188073e46274bd
Content-Type: application/json
```

**Playwright 불필요**: HTML/API 모두 정적·정규식·JSON 파싱으로 충분.

### 3.2.2 단일 record 응답 스키마 (실측)

```json
{
  "book_id": 8,
  "volume_id": 1,
  "content_seq": 138,
  "content_level": "Z",
  "content_level_depth": "Z",
  "up_path_nm": "內景篇卷之一 > 身形 > 形氣之始",
  "original": "乾鑿度云, 天形出乎乾, 有太易太初太始太素 …",
  "trans_2": "《건착도》에, 하늘[天]의 형(形)은 건(乾)에서 나오니 …",
  "trans_1": "In the Book of Changes Heavenly-Qian Chisel Measurement …",
  "annotation": [],
  "index_num": 1,
  "lang_code_2": "BOOK_15_02",
  "lang_code_1": "BOOK_15_01"
}
```

**3중 병렬 (한문 / 한국어 / 영어)** 자동 정렬 — 별도 alignment 불필요.

### 3.2.3 권 boundary 감지

`content_seq`가 `content_total`을 넘으면 서버는 **HTTP 200 + empty body** 응답.
→ 파일 크기 0인 응답을 sentinel로 사용. `EMPTY_RUN_THRESHOLD = 3` (3연속 빈 응답 시 권 종료로 판정).

### 3.2.4 Multi-process Orchestrator 아키텍처

**파일**: `src/data/crawler/mediclassics_orchestrator.py` (446 라인)

```
ProcessPoolExecutor(max_workers = min(cpu_count, num_books))
├─ Worker 1 (book_id=8)   ─── asyncio + httpx (concurrency=2)
├─ Worker 2 (book_id=56)  ─── asyncio + httpx
├─ ...
└─ Worker N (book_id=...) ─── asyncio + httpx
```

- **책당 1 process**: per-book rate limit이 분리되어 있으므로 책간 독립 (§3.2.5)
- **process 내부 asyncio**: I/O-bound이므로 GIL 무관. concurrency=2면 sustained quota 거의 채움.
- **multiprocessing이 의미 있는 이유**: 책별 독립 quota를 동시에 소진. CPU 가속 목적 아님.
- **N의 상한**: `min(cpu_count, num_books)`. 32 코어 머신에서 책 7권이면 7 워커가 최적 (책 14권이면 14 워커).
- **같은 책에 워커 여러 개 붙이는 건 무의미**: 같은 quota를 더 빨리 소진 → 60s pause 같음.

### 3.2.5 Rate Limit 실측 (book 4 광제비급, 2026-04-16)

```
실험 A: 단일 권에 30 burst 동시 요청 → 30/30 OK, 3.76s
실험 B: 5권 × 6 burst = 30 동시 요청 → 30/30 OK, 3.70s (per-volume 격리 아님)
실험 C: 단일 권에 300 sustained (concurrency=10) → 240 OK + 60 rate-limit (HTTP 405), 첫 405 = seq 241
```

**결론**:
- Rate limit은 **per-book + cumulative** (per-IP / per-volume 아님)
- 책당 quota ≈ **240 successful requests** → **HTTP 405 Method Not Allowed** 반환 (실제 의미는 rate limit)
- pause 권장 = **60s** (이후 다음 batch 가능)
- **Sustained throughput 상한 ≈ 2.8 req/s per book** = 240 req / (27s active + 60s pause)
- 책간 독립 → 7 책 병렬이면 **최대 ~20 req/s 합산**

### 3.2.6 Bulletproof Retry 정책

| 에러 | 분류 | 정책 |
|---|---|---|
| HTTP 405 / 429 / 503 | rate limit | **최대 30회 재시도**, sleep = `pause × min(attempt, 5)` (60→300s 점증) |
| HTTP 5xx (서버 일시 장애) | network | 최대 5회, exponential backoff (`2^n`, cap 60s) |
| `httpx.ReadTimeout` / `ConnectError` / `RemoteProtocolError` / `PoolTimeout` | network | 최대 5회, exponential backoff |
| `JSONDecodeError` | network | 최대 5회 (응답 깨짐 대응) |
| HTTP 200 + empty body, HTTP 404 | sentinel | 권 끝 → 정상 종료 |

전체 retry 로직은 `mediclassics_orchestrator.py::fetch_content` 참조.

### 3.2.7 KIOM 사전 통지

- 자동 다운로드 시작 전 KIOM에 **"연구 목적 자동 다운로드 ≤ 3 req/s per-book"** 이메일 통지
- ToS가 자동 수집을 금지하면 수기 다운로드로 전환

## 3.3 Parsing — 배포 파일 / API 응답

### 3.3.1 Markup 역공학 — 완료

**결과**: 배포 파일·API 응답 **모두 inline 태그 제거된 평문**. 공식 markup 스펙(`info.mediclassics.kr/document/document/markup-syntax-for-classics`)의 `[xx/..]`, `{..}`, `#..#`는 **웹 뷰어 전용**이며 배포 파일에는 0건 (실측: 동의보감 내경편 권1 1254 record 전수 grep).

상세는 `02_data_source/mediclassics_parsing_spec.md` 참조.

### 3.3.2 API 응답 → 정제 JSONL (실측 스키마)

크롤러가 저장하는 record (`extract_record` 후):
```json
{
  "book_id": 8,
  "volume_id": 1,
  "content_seq": 138,
  "content_level": "ZZ",
  "up_path_nm": "內景篇卷之一 > 身形 > 形氣之始",
  "original": "乾鑿度云, …",
  "trans_ko": "《건착도》에, …",
  "trans_en": "In the Book of Changes …",
  "annotation": null,
  "index_num": 1
}
```

- `content_level`은 `content_level + content_level_depth` 2자 문자열 (AA, BB, ZZ, DP, SS 등)
- `up_path_nm`이 **계층 경로 그대로 제공** → SectionStack trie 재구성 불필요
- `annotation`은 항상 빈 배열 (배포 정책상 제거됨)
- `trans_2 → trans_ko`, `trans_1 → trans_en` 으로 키 이름 정규화

### 3.3.3 Bilingual block 빌드 (D2 포맷 변환)

```
data/raw/mediclassics_unified/book_{id}/vol_{v}.jsonl
        │
        │ build_bilingual_blocks.py  (trans_ko 비어있지 않은 record만)
        ▼
data/cpt/hanmed_bilingual.jsonl      # §04.5.2 D2 포맷
```

각 record를 단일 텍스트 블록으로 변환:
```
<ZH>{original}</ZH>
<KO>{trans_ko}</KO>

```

원문 단독(`trans_ko == null`)과 국역 단독은 별도 파일(`hanmed_zh_only.jsonl`, `hanmed_ko_only.jsonl`)로 분리 저장.

### 3.3.4 파서 검증

- 무작위 **50 record 수기 검수** → 정렬 정확도 목표 ≥ **95%** (실측 — API가 이미 정렬해서 거의 100% 예상)
- 미달 시 해당 서적은 `data/parsed/_excluded.txt` 에 기록, 파서 수정

## 3.4 Cleaning

| 단계 | 처리 |
|---|---|
| 문자 정규화 | API 응답 이미 CJK Unified → 추가 작업 없음 |
| 공백·전각 | 전각 공백 → 반각, NFC 정규화 |
| 중복 제거 | SimHash 문장 수준, threshold 실험으로 튜닝 |
| 길이 필터 | 한문 ≥ 3자, 국역 ≥ 5어절 |
| PII | 고전 문헌만 → 해당 없음 |
| **Eval contamination 필터** | §3.4.2 참조 |

### 3.4.1 도메인 사전 (NER seed)
- 본초명, 병명, 혈자리, 처방명을 수기 + 규칙 기반으로 ≥ **3,000 엔트리** (초기)
- DP/SS prefix가 처방 표제·본문 자동 추출 신호로 사용 가능 (`mediclassics_parsing_spec.md` §4.2)
- 저장: `data/dict/hanmed_terms.jsonl`

### 3.4.2 Eval contamination check (ver2.1 path · hash 통일)

**목적**: HanMed-Eval v0의 held-out 문장이 CPT 학습 입력에 섞여 들어가 평가 지표가 부풀려지는 것을 방지.

**훅 위치**: `build_bilingual_blocks.py` 출력 단계 및 **`data/cpt/*.jsonl` 전체** (bilingual / plain-zh / plain-ko / Wiki-ko replay / CBETA / 보조 코퍼스 포함) 최종 빌드 단계.

**절차**:
1. **소스**: `eval/hanmed_eval_v0/{T1,T2,T5}.jsonl` — T1 held-out 한문 30문장 + T2 독해 QA 지문 원문 30문장 + T5 KLUE-YNAT 원문 100문장
2. 각 문장을 NFC 정규화 + 공백 정규화 후 **SHA256** 계산 → `eval/hashes/heldout_{T1,T2,T5}.txt` (git committed)
3. CPT 입력 JSONL(`data/cpt/*.jsonl`) 빌드 시, 각 블록·문단을 문장 단위로 쪼개 동일한 정규화 적용 후 hash 대조
4. 매치되는 문장이 포함된 블록은 **드롭**, 드롭 카운트를 `data/stats/contamination_drop.json`에 기록
5. 드롭 > 0.5% 이면 파이프라인 실패 처리 (의심스러운 중복 — 수기 확인)

이 훅은 `ver2/06_infrastructure/gpu_framework.md` §6.5 재현성 파이프라인과 `ver2/05_evaluation/hanmed_eval.md` §5.7 평가 정의에서 모두 참조된다. **해시 함수(SHA256) · 소스 경로 · hash 파일 경로** 는 세 문서에서 **완전 일치**해야 한다 (ver2.1 drift 해결).

## 3.5 보조 코퍼스 (옵션)

CPT에서 일반 언어 능력 유지를 위해 replay 데이터 필요. 라이선스별 분리 보관, 학습 시 "공개용 adapter"와 "내부 전체 adapter" 분리 (상세는 `ver2/07_license_ethics/`).

| 소스 | 유형 | 라이선스 | 공개 adapter | 내부 adapter |
|---|---|---|---|---|
| mediclassics (한문+국역+영역) | 도메인 | KIOM 비상업 | ✅ 승인 후 | ✅ |
| Wikipedia-ko | 일반 한국어 | CC BY-SA 3.0 | ✅ | ✅ |
| AI Hub 일반 한국어 | 일반 한국어 | AI Hub 연구 이용 | ❌ | ✅ |
| 모두의말뭉치 | 일반 한국어 | 국립국어원 이용약관 | ❌ | ✅ |
| CBETA 대장경 | 한문 | 비상업/학술 | ❌ | ✅ (옵션) |
| CTP (Chinese Text Project) | 한문 | 학술 이용 | ❌ | ✅ (옵션) |

## 3.6 수집 대상 및 진행 (현재 — Core 14)

mediclassics 161종 중 한국 한의학 핵심 25종 식별. 그 중 **Core 14** 진행:

| 그룹 | book_ids | 분류 |
|---|---|---|
| **Core 7** (1차) | 8, 56, 69, 86, 93, 182, 291 | 동의보감 / 의방유취 / 제중신편 / 침구경험방 / 향약집성방 / 사상신축본 / 방약합편 |
| **+7 확장** (2차) | 1, 4, 9, 24, 38, 59, 100 | 사의경험방 / 광제비급 / 동의사상신편 / 본초정화 / 식료찬요 / 의종손익 / 외과심법요결 |

**확장 후보 (Core 21 / 30)**: 7, 44, 46, 47, 49, 54, 60, 70, 71, 94, 139, 183 (한글 의서 + 종합)

## 3.7 Exit Criteria

| # | 조건 | 미달 시 대응 |
|---|---|---|
| E1 | 메타데이터 25종 중 ≥ 14종 수집 성공 | scope를 성공 서적만으로 제한 |
| E2 | 파서 정렬 정확도 ≥ 95% on validation set | 해당 서적 제외 |
| E3 | 최종 파싱 서적 수 ≥ 7종 | 스코프를 동의보감 단일 서적 심화로 축소 |
| E4 | KIOM 자동 다운로드 통지 응답 ≤ 2주 | 수기 다운로드 + 연구자 개인 계정 경로로 전환 |
| E5 | Eval contamination drop 비율 ≤ 0.5% | 수기 확인, held-out 재구성 |
| E6 | 책당 ko_coverage ≥ 90% | 국역 누락 서적 별도 표기 |

## 3.8 산출물 디렉토리 (현재)

```
data/raw/mediclassics_unified/
├── orchestrator.log                    # 통합 로그
├── unified_manifest.json               # 14권 통합 통계 (총 record, KO/EN coverage)
├── logs/
│   ├── book_001.log
│   ├── book_004.log
│   └── ... (책별 상세 로그)
├── book_001/  사의경험방
│   ├── manifest.json
│   └── vol_01.jsonl
├── book_008/  동의보감
│   ├── manifest.json
│   ├── vol_01.jsonl
│   ├── vol_02.jsonl
│   └── ... vol_23.jsonl
└── book_{004, 009, 024, 038, 056, 059, 069, 086, 093, 100, 182, 291}/
```

## 3.9 실행 명령

```bash
# Core 7 (또는 임의 책 set)
python3 src/data/crawler/mediclassics_orchestrator.py \
  --output data/raw/mediclassics_unified \
  --books 8,56,69,86,93,182,291 \
  --delay 0.5 --concurrency 2 --pause 60 \
  --rl-max-retries 30 --net-max-retries 5

# 확장 (별도 process로 동시 실행 가능, 같은 output 디렉토리)
python3 src/data/crawler/mediclassics_orchestrator.py \
  --output data/raw/mediclassics_unified \
  --books 1,4,9,24,38,59,100

# 진행 모니터
tail -f data/raw/mediclassics_unified/orchestrator.log
find data/raw/mediclassics_unified -name "*.jsonl" | xargs wc -l | tail -1
```

자세한 사용·재현 가이드: `korean-medicine-llm/README.md`.
