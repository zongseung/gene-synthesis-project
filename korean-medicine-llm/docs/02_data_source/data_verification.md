# 02. Data Source — mediclassics.kr 검증

## 2.1 소스 개요

한국한의학연구원(KIOM) 운영 **한의학고전DB** — https://mediclassics.kr  
2007년부터 한문 고전을 국·영문으로 번역하는 KIOM 사업의 공개 결과물. 회원가입 없이 열람·다운로드 가능하며, 고문헌 텍스트 가공용 웹앱(Mediclassics Tools)도 함께 제공된다.

## 2.2 확인된 사실 (검증일: 2026-04-16)

검증 방법: `WebFetch` on mediclassics.kr 및 info.mediclassics.kr 하위 페이지 + WebSearch 교차 확인.

| 항목 | 값 | 출처 |
|---|---|---|
| 수록 서적 수 | **161종** | mediclassics.kr 메인 |
| 원문 총량 | **약 18,162,000자** (한자) | mediclassics.kr 메인 |
| 인코딩 | UTF-8 | info.mediclassics.kr/apps/dist-texts/ |
| 한자 정규화 | CJK Compatibility → CJK Unified (통합) | 배포 서비스 안내 |
| 마크업 | "simple markup syntax" — 별도 문서 참조 | 배포 서비스 안내 |
| 이미지·교감 기록 | 배포 파일에서 **제외** (이미지 내 텍스트만 유지) | 배포 서비스 안내 |
| 비상업 이용 | 제한 없음 | 배포 서비스 안내 |
| 상업 이용 | `kiombook@kiom.re.kr` 문의 필수 | 배포 서비스 안내 |
| 출처 표기 | "한의학고전DB (mediclassics.kr)" 의무 | 배포 서비스 안내 |
| 벌크 API | **없음** — 개별 다운로드만 | 배포 서비스 안내 |
| 회원가입 | 불필요 (열람·다운로드 모두) | mediclassics.kr 전반 |

## 2.3 미확인 항목 (M0~M1 내 해소 필요)

| 항목 | 방법 | 담당 |
|---|---|---|
| 161종 국역 제공 서적 정확 목록 | Playwright로 `info.mediclassics.kr/contents/database/list` 렌더 | M0 |
| 161종 영역 제공 서적 목록 | 위와 동일 | M0 |
| 서적별 글자수 | 위와 동일 | M0 |
| markup 문법 공식 문서 URL | info.mediclassics.kr/document 확인 | M0 |
| 정렬 단위 (문/절/권) | 대표 서적 수기 검토 | M1 |
| 국역/영역의 저작권 주체 (KIOM vs 개별 역자) | KIOM 이메일 문의 시 확인 | M0~M2 |

미해소 항목은 §03 파이프라인의 파서 설계 및 §07 라이선스 해석에 영향을 준다.

## 2.4 2차 출처 (참고용, 단독 근거로 사용 금지)

| 출처 | 연도 | 내용 | 해석 |
|---|---|---|---|
| 헬로디디 "한의학연, '한의학 고전 DB' 웹 서비스 실시" | 2014~2015 | "38종 한의학 고전 국역 완료" | 당시 시점 수치. 현재(161종, 2026)는 상향 가능성 — 2.3 실측 결과로 대체 |
| 나무위키 "동의보감" | - | 동의보감 국·영역 PDF 무료 배포 | 구체 수치 인용은 금지, context 용도만 |

## 2.5 코퍼스 규모 — 범위 추정 (commitment 아님)

**중요**: 아래 숫자는 **Stage 0에서 토크나이저 확정 후 실측**으로 교체된다. 본 추정은 상·하한 범위로만 사용하며, 학습 예산 결정 근거로 쓸 수 없다.

### 2.5.1 원문 (한문)
- 문자 수: 약 18.16M 자 (고정)
- Solar tokenizer(Llama BPE 기반) 한자 평균 tokens/char: 추정 1.2 ~ 2.5
- → **~22M ~ ~45M tokens**

### 2.5.2 국역 (한국어)
- 커버 서적 비율 (2014 기사 기준 38/161 ≈ 23.6%) — 상한 산정용
- 평균 한자↔한국어 비율 1:1.5 어절 → 국역 분량 대략 (18.16M × 0.236 × 1.5) ≈ 6.4M 어절
- Solar tokenizer 한국어 tokens/어절 ≈ 1.3~1.5
- → **~3M ~ ~10M tokens** (커버 비율·번역 밀도 변동 포함)

### 2.5.3 영역
- 동의보감 위주 → 대략 1M~3M tokens

### 2.5.4 총 raw 코퍼스 (범위)

| 구성 | 하한 | 상한 |
|---|---|---|
| 한문 원문 | 22M | 45M |
| 국역 | 3M | 10M |
| 영역 | 1M | 3M |
| **합계 tokens** | **~26M** | **~58M** |

이 범위는 **from-scratch 학습에는 명백히 부족**하고, LoRA CPT에는 **충분**하다. §04에서 상세 예산.

### 2.5.5 실측 동결 절차
1. M2에서 Solar tokenizer로 실측
2. `data/stats/corpus_v1.json` 스냅샷 저장 (git committed)
3. 이후 모든 학습 예산 계산은 이 스냅샷만 참조
4. 스키마 변경 시 `schema_version` 증가 → 재학습 플래그

## 2.6 저장 레이아웃

```
korean-medicine-llm/
├── data/
│   ├── raw/mediclassics/{book_id}.txt        # 원본 다운로드
│   ├── raw/manifest.jsonl                    # SHA256, download_at, source_url
│   ├── meta/books.csv                        # 2.3 Playwright 결과
│   ├── parsed/mediclassics/{book_id}.jsonl   # 03장 파서 출력
│   └── stats/corpus_v1.json                  # 2.5.5 동결 스냅샷
```

## 2.7 열린 질문

- KIOM이 161종 전체에 대해 **이본(異本)** 관리를 어떻게 하는지 — 여러 판본이 있을 경우 파일이 단일인지 분리인지
- 교감 기록이 "제외"된다는 것이 원문 해석에 영향을 주는 주석까지 빠진다는 의미인지, 아니면 판본 대조 표기만 제외되는지
- 이 두 질문은 M0 KIOM 이메일에 포함
