# 02. Data Source — mediclassics.kr 검증 (ver2)

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
| **markup에서 저자주·이본주기 분리 필요 여부** | KIOM 이메일 + markup_spec 역공학 | M0~M1 (§03.3.1) |

마지막 행은 ver2 신규 질문이다. Bilingual block 빌드(§03.3.2) 시 원문·국역에 저자주가 섞여 들어가면 alignment 품질이 떨어지므로, KIOM 측에 "markup tag 중 저자주(`<note>`류)와 이본주기가 본문과 구분되어 제공되는지"를 명시적으로 확인한다.

미해소 항목은 §03 파이프라인의 파서 설계 및 §07 라이선스 해석에 영향을 준다.

## 2.4 2차 출처 (참고용, 단독 근거로 사용 금지)

| 출처 | 연도 | 내용 | 해석 |
|---|---|---|---|
| 헬로디디 "한의학연, '한의학 고전 DB' 웹 서비스 실시" | 2014~2015 | "38종 한의학 고전 국역 완료" | 당시 시점 수치. 현재 커버리지의 하한 참고용 (대략 24%) |
| 나무위키 "동의보감" | - | 동의보감 국·영역 PDF 무료 배포 | 구체 수치 인용 금지, context 용도만 |

## 2.5 코퍼스 규모 — 재산정 (BLOCKER B3 해결)

**중요 변경**: ver1 §2.5.4의 "총 raw 26M~58M tokens" 수치는 **이중계상/영역 포함 때문에 폐기**한다. ver2는 (a) **HanMed unique** (영역 제외)와 (b) **영역(별도)**을 분리 표기한다. 병렬 데이터는 원문+국역의 **재포맷**이지 새로운 토큰이 아니므로 unique 합계에 가산하지 않는다.

실측 전까지 아래 숫자는 **범위**로만 사용하며, 학습 예산 결정은 `data/stats/corpus_v1.json` 동결 이후에만 확정한다.

### 2.5.1 한문 원문
- 문자 수: 약 18.16M 자 (고정)
- Solar tokenizer(Llama BPE 기반) 한자 평균 tokens/char: **1.5 ~ 1.8** (공통 결정 로그 D3)
- → **≈ 27M ~ 33M tokens**

### 2.5.2 국역 (한국어)
- 커버 서적 비율 ≈ 24% (2014 기준 38/161의 하한, M0에서 상향 가능)
- 한자 : 한국어 어절 ≈ 1 : 1.5, 한국어 어절당 tokens ≈ 1.3
- → (18.16M × 0.24 × 1.5 × 1.3) ≈ **≈ 5M ~ 10M tokens** (커버·번역 밀도 변동 반영 상한)

### 2.5.3 영역 (별도 — v1 CPT scope 제외)
- 동의보감 중심 일부 서적 → **≈ 1M ~ 3M tokens**
- **v1 CPT 학습 예산에는 포함하지 않는다**. 별도 ablation 또는 후속 태스크 데이터로만 사용.

### 2.5.4 HanMed unique tokens (ver2 확정 범위)

| 구성 | 하한 | 상한 | CPT 포함 |
|---|---|---|---|
| 한문 원문 | 27M | 33M | ✅ |
| 국역 | 5M | 10M | ✅ |
| 병렬 블록 (§04.5.2 재포맷) | 재활용 | 재활용 | ✅ (태그 토큰만 추가 ~ 무시 가능) |
| **HanMed unique 합계 (영역 제외)** | **≈ 32M** | **≈ 43M** | ✅ |
| 영역 (별도) | 1M | 3M | ❌ (v1 scope 제외) |

이 범위는 **from-scratch 학습에는 명백히 부족**하며, LoRA CPT에는 **충분**하다. ver2 §04.5.3의 CPT 예산 cap **150M~250M**은 이 unique 범위에 HanMed 1.5~3 epoch과 믹스 40%를 곱해 산출된다.

### 2.5.5 실측 동결 절차
1. M2에서 Solar tokenizer(또는 확장 tokenizer)로 실측
2. `data/stats/corpus_v1.json` 스냅샷 저장 (git committed)
   - 필드: `hanja_tokens`, `ko_tokens`, `bilingual_block_tag_tokens`, `tokens_per_char_hanja`, `tokens_per_eojeol_ko`, `schema_version`, `sha256_of_raw`
3. 이후 모든 학습 예산 계산은 이 스냅샷만 참조
4. 스키마 변경 시 `schema_version` 증가 → 재학습 플래그

Agent B 기준으로 이 수치(32M~43M unique)는 `ver2/08_risks/risk_register.md` **A2 assumption**에서 동일 수치로 참조되어야 한다.

## 2.6 저장 레이아웃

```
korean-medicine-llm/
├── data/
│   ├── raw/mediclassics/{book_id}.txt           # 원본 다운로드
│   ├── raw/manifest.jsonl                       # SHA256, download_at, source_url
│   ├── meta/books.csv                           # 2.3 Playwright 결과
│   ├── parsed/mediclassics/{book_id}.jsonl      # §03 파서 출력
│   ├── cpt/hanmed_bilingual.jsonl               # §03.3.2 bilingual block
│   └── stats/corpus_v1.json                     # 2.5.5 동결 스냅샷
```

## 2.7 열린 질문

- KIOM이 161종 전체에 대해 **이본(異本)** 관리를 어떻게 하는지 — 여러 판본이 있을 경우 파일이 단일인지 분리인지
- 교감 기록이 "제외"된다는 것이 원문 해석에 영향을 주는 주석까지 빠진다는 의미인지, 아니면 판본 대조 표기만 제외되는지
- **markup에서 저자주·이본주기가 본문 태그와 분리되어 있는지** (§2.3 신규 항목과 동일)

위 세 질문은 M0 KIOM 이메일에 포함.
