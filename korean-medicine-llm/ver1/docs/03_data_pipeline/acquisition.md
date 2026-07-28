# 03. Data Pipeline — 획득·파싱·정제

## 3.1 원칙

1. KIOM 비상업 이용 범위 준수
2. rate limit 1 req/sec, 연구용 User-Agent 명시
3. 모든 다운로드에 `downloaded_at`, `source_url`, `sha256` 기록
4. 가공 코퍼스의 **공개 배포는 KIOM 서면 승인 후에만**
5. 재배포 불가 소스는 별도 디렉토리에 격리, 공개용 adapter 학습에 미사용

## 3.2 Acquisition

### 3.2.1 메타데이터 수집
- 대상: `info.mediclassics.kr/contents/database/list` (JS 렌더 필요)
- 방법: **Playwright headless Chromium**
- 출력: `data/meta/books.csv`
  - 컬럼: `book_id, title, author, era, volumes, chars_hanja, has_ko, has_en, download_url`
- 수기 검증: 10개 행 수기 대조, 불일치 시 셀렉터 재조정

### 3.2.2 원문 다운로드
- 대상: 배포 서비스의 개별 파일 URL
- 방법: `httpx` async client + `asyncio.Semaphore(1)` (rate limit)
- 저장: `data/raw/mediclassics/{book_id}.txt`
- 매니페스트: `data/raw/manifest.jsonl` — 한 줄에 한 파일 (sha256, downloaded_at, source_url, size_bytes)
- 재다운로드 정책: 버전 차이 확인 시에만, 기본은 idempotent

### 3.2.3 KIOM 사전 통지
- 자동 다운로드 시작 전 KIOM에 **"연구 목적 자동 다운로드 1 req/sec"** 를 이메일 통지
- ToS가 자동 수집을 금지하면 수기 다운로드로 전환

## 3.3 Parsing

### 3.3.1 Markup 역공학
1. 대표 서적 1종(예: 동의보감 내경편 권1) 선정
2. 공식 문서(`info.mediclassics.kr/document` 가 있다면 그 페이지) 확인
3. 없으면 수기 분석 → 태그 인벤토리 작성
4. 결과는 `docs/03_data_pipeline/markup_spec.md` 에 문서화 (지금은 placeholder)

### 3.3.2 파서 구현

초기 구현:
- Python, 정규식 기반
- 태그 수가 많아지거나 중첩이 복잡해지면 `lark` grammar로 승격

출력 스키마 (JSONL, 한 줄 = 한 레코드):
```json
{
  "book_id": "dongui_bogam",
  "chapter": "내경편",
  "volume": 1,
  "section_id": "001",
  "orig_zh": "東醫寶鑑者 …",
  "trans_ko": "동의보감이란 …",
  "trans_en": null,
  "notes": [
    {"type": "author_note", "text": "…"}
  ],
  "provenance": {
    "file": "dongui_bogam.txt",
    "line_range": [120, 145]
  }
}
```

스키마 버전: `schema_version: 1` — 스키마 변경 시 증가 + 재파싱 강제.

### 3.3.3 파서 검증
- 무작위 **50 레코드 수기 검수** → 정렬 정확도 목표 ≥ **95%**
- 미달 시 해당 서적은 `data/parsed/_excluded.txt` 에 기록, 파서 수정 2차 시도
- 2차도 미달 시 해당 서적 제외하고 진행

## 3.4 Cleaning

| 단계 | 처리 |
|---|---|
| 문자 정규화 | 이미 CJK Unified → 추가 작업 없음 |
| 공백·전각 | 전각 공백 → 반각, NFC 정규화 |
| 중복 제거 | SimHash 문장 수준, threshold 실험으로 튜닝 |
| 길이 필터 | 한문 ≥ 3자, 국역 ≥ 5어절 |
| PII | 고전 문헌만 → 해당 없음 (현대 코퍼스 섞을 시 재검토) |

### 3.4.1 도메인 사전 (NER seed)
- 본초명, 병명, 혈자리, 처방명을 수기 + 규칙 기반으로 ≥ **3,000 엔트리** (초기)
- 저장: `data/dict/hanmed_terms.jsonl`
- 용도: (a) tokenizer extension 후보, (b) 평가 T3 정답 sets, (c) RAG 단계에서 index

## 3.5 보조 코퍼스 (옵션)

CPT에서 일반 언어 능력 유지를 위해 replay 데이터가 필요. **라이선스 교차오염** 때문에 소스별로 구분 보관하고, 학습 시 "공개용 adapter"와 "내부 전체 adapter"를 분리 실험한다.

| 소스 | 유형 | 라이선스 | 공개용 adapter 사용 | 내부 adapter 사용 |
|---|---|---|---|---|
| mediclassics (한문+국역) | 도메인 | KIOM 비상업 (서면 승인 시 변경 가능) | ✅ 승인 후 | ✅ |
| Wikipedia-ko | 일반 한국어 | CC BY-SA 3.0 | ✅ | ✅ |
| AI Hub 일반 한국어 | 일반 한국어 | AI Hub 연구 이용 | ❌ (재배포 불가) | ✅ |
| 모두의말뭉치 | 일반 한국어 | 국립국어원 이용약관 | ❌ | ✅ |
| CBETA 대장경 | 한문 | 비상업/학술 | ❌ | ✅ (옵션) |
| CTP (Chinese Text Project) | 한문 | 학술 이용 | ❌ | ✅ (옵션) |

### 3.5.1 라이선스 교차오염 원칙

- adapter weights는 **학습에 사용된 가장 엄격한 라이선스**에 묶인다
- 공개 의도가 있는 `HanMed-Public-LoRA` 는 ✅ 열의 소스만 사용
- 내부 실험용 `HanMed-Internal-LoRA` 는 전부 사용, 논문 실험 표에는 결과 비교 목적으로만 포함

## 3.6 Exit Criteria

| # | 조건 | 미달 시 대응 |
|---|---|---|
| E1 | 메타데이터 161종 중 ≥ 120종 수집 성공 | scope를 성공 서적만으로 제한 |
| E2 | 파서 정렬 정확도 ≥ 95% on validation set | 해당 서적 제외, 1회 파서 수정 시도 |
| E3 | 최종 파싱 서적 수 ≥ 80종 | 스코프를 **동의보감 단일 서적 심화**로 축소 |
| E4 | 2주 내 KIOM 자동 다운로드 통지 응답 없음 | 수기 다운로드 + 연구자 개인 계정 경로로 전환 |

## 3.7 산출물 요약

```
data/
├── raw/
│   ├── mediclassics/{book_id}.txt
│   └── manifest.jsonl
├── meta/books.csv
├── parsed/
│   ├── mediclassics/{book_id}.jsonl
│   └── _excluded.txt
├── stats/
│   ├── corpus_v1.json
│   └── parse_report.json
└── dict/hanmed_terms.jsonl
```
