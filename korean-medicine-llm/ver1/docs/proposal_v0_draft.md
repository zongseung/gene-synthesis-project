# 한의학 Foundation LLM 기획서 v0.1

작성일: 2026-04-16
상태: 초안 / 타당성 검토용
작성: HanMed-LLM 프로젝트 (가칭)

> ⚠️ 본 문서는 아이디어 단계의 타당성 기획서이며, 실제 데이터 사용·모델 학습 전에 KIOM(한국한의학연구원)과의 라이선스 확인이 선행되어야 합니다.

---

## 0. Executive Summary

**목표**: 한의학 고전 및 현대 지식을 다루는 한국어·한문 특화 LLM을 구축한다.

**핵심 데이터**: 한국한의학연구원(KIOM)의 **한의학고전DB (mediclassics.kr)**. 161종 고의서, 약 **1,816만 자** 규모의 한문 원문, 일부 서적의 국역/영역 병렬 데이터가 UTF-8 텍스트(단순 markup)로 다운로드 가능.

**현실적 스코프 재정의**: 1,800만 자 수준의 원문 코퍼스는 **from-scratch foundation 학습에는 부족**하다. 따라서 본 기획의 실질적 타겟은 **(A) 강력한 일반 LLM 위에 continued pretraining + SFT를 얹는 "도메인 특화 모델"** 이다. "Foundation"은 "한의학 도메인 내에서 downstream 태스크의 기반(base)이 되는 모델"이라는 의미로 사용한다.

**산출물**:
1. HanMed-Corpus v1: 정제된 한문/국역/영역 병렬 말뭉치 + 통계 리포트
2. HanMed-Base: continued pretraining 된 도메인 base 모델
3. HanMed-Chat: SFT + preference tuning 된 대화형 모델
4. HanMed-Eval: 한의학 지식·독해·번역 평가 벤치마크
5. 논문 1편 (SCI) — 핵심 기여는 병렬 한문 고전 도메인 적응 레시피 + 한의학 평가 벤치마크.

---

## 1. 배경 및 문제 정의

### 1.1 왜 한의학 LLM인가
- 일반 LLM(GPT-4, Claude, Qwen 등)은 한의학 고전 한문 독해와 임상 지식 통합에서 도메인 커버리지가 낮다.
- 한의학 고전은 **한문 원문 + 한국어 번역 + (일부) 영어 번역**이 병렬로 존재하는 매우 드문 코퍼스다. 이는 기계 번역·용어 정렬·교차언어 독해 학습에 강한 신호를 준다.
- 국내 한의대/임상/연구 현장에서 고전 독해, 처방 해설, 본초 정보 검색을 도와줄 수 있는 도구가 필요하다.

### 1.2 목표 태스크
1. 한문 고전 원문 → 현대 한국어 번역 (문맥·용어 보존)
2. 한문/국역 문장의 요지 요약, 처방·본초 추출
3. 동의보감·침구경험방 등 특정 서적 기반 QA
4. 처방(方劑) 구성 약재 분해, 효능·주치 설명
5. (장기) 한의학 현대 임상·연구 문헌 혼합 QA

비목표: 한의학 처방의 실제 임상 적용, 진단 보조. 본 모델은 **문헌 연구 보조**가 1차 목적이며, 임상 의사결정 도구로 홍보하지 않는다(규제·안전 이슈).

---

## 2. 데이터 소스 검증

### 2.1 mediclassics.kr 확인 결과 (2026-04-16, WebFetch/WebSearch 기준)

| 항목 | 확인 내용 | 출처 |
|---|---|---|
| 수록 서적 수 | **161종** | mediclassics.kr 메인 페이지 |
| 한문 원문 분량 | 약 **18,162K자** (≈ 1,816만 자) | mediclassics.kr 메인 |
| 인코딩 | UTF-8 | 배포 서비스 안내 |
| 한자 처리 | CJK Compatibility → CJK Unified로 통합 정규화 | 배포 서비스 안내 |
| 마크업 | "simple markup syntax" — 별도 문법 문서 제공 | 배포 서비스 안내 |
| 이미지/교감 기록 | **제외**됨 (이미지 내 텍스트만 유지) | 배포 서비스 안내 |
| 비상업 이용 | **제한 없음** | 배포 서비스 안내 |
| 상업 이용 | kiombook@kiom.re.kr 이메일 문의 필수 | 배포 서비스 안내 |
| 출처 표기 | "한의학고전DB (mediclassics.kr)" 명시 의무 | 배포 서비스 안내 |
| 벌크 다운로드 API | **없음** — 개별 다운로드만 제공 | 배포 서비스 안내 |
| 국역 | 제공(총 38종 국역이라는 언론 보도, DB 상 일부 서적) | Hellodd 기사 |
| 영역 | 동의보감 국·영역 모두 제공, 기타 일부 | KIOM 발표 / 나무위키 |

### 2.2 아직 검증되지 않은 사항 (❗)
- 161종 중 **국역이 있는 서적의 정확한 목록과 각 서적의 글자 수**
- 영역본 제공 서적의 정확한 목록 (동의보감 외)
- markup 문법의 구체 정의 (주석, 단락, 저자주, 이본주기 등)
- 정렬 단위: 문장/절/권/책 중 어느 단위로 국역-원문이 대응되는가
- 서적별 저작권 상태 (원 저작은 public domain이나 국역/영역 저작권은 KIOM 또는 역자)

**Action item**: 기획 확정 전 "contents/database/list" 페이지를 headless browser(Playwright)로 렌더링하여 서적별 메타데이터를 직접 확보한다.

### 2.3 코퍼스 규모 감각 체크
- 원문 한자 약 1.82e7 자. 한자 문자 ≈ 1 token (Qwen 계열 기준) 가정 시 **≈ 1.8e7 tokens**.
- 국역이 전체 161종 중 약 1/4 커버된다고 보수적으로 가정하면 한국어 번역 약 500만~1,000만 어절 수준, **≈ 1e7 ~ 2e7 Korean tokens**.
- 합산 기대치: **3e7 ~ 4e7 tokens**.

**해석**:
- Chinchilla-optimal 기준으로는 7B 모델 from-scratch에 **3~4 orders of magnitude 부족** (140B tokens 필요).
- 따라서 from-scratch 학습은 불가. **continued pretraining (CPT) + SFT** 가 유일한 현실적 경로.
- CPT에는 수천만 tokens도 **충분히 의미 있는 규모** (예: BloombergGPT 도메인 추가, MedPaLM 도메인 적응 사례 참고).

---

## 3. 데이터 확보·가공 파이프라인

### 3.1 획득 (acquisition)
1. **개별 다운로드 스크립트**: mediclassics.kr 배포 페이지 URL 패턴을 파악 후, 비상업 이용 범위 내에서 rate-limit(예: 1 req/sec) 하에 순차 다운로드. ToS 위반 방지를 위해 KIOM에 사전 통지.
2. **메타데이터 크롤링**: `info.mediclassics.kr/contents/database/list` 를 Playwright로 렌더링하여 서적별 (서명, 저자, 시대, 권수, 글자수, 국역/영역 여부) 수집.
3. **수동 다운로드 fallback**: 자동 다운로드 시 robots.txt/ToS 저촉 시, 연구자 권한으로 수동 다운로드 후 raw 보관.

### 3.2 파싱 (parsing)
- **HanMed markup parser**: KIOM 배포 파일의 마크업 문법 문서를 읽고 (i) 단락, (ii) 주석, (iii) 원문/국역 정렬 태그를 구조화 JSON으로 변환.
- 출력 단위: `{book_id, chapter, section, orig_zh, trans_ko, trans_en?, notes[]}` 레코드.
- 검증: 무작위 50개 레코드를 수기 검수, 정렬 정확도 ≥ 95% 확인.

### 3.3 정제·라벨링
- 한자 정규화: 이미 CJK Unified로 통합되어 있음 → 추가 정규화 최소.
- 고유명사 사전: 본초명·병명·혈자리·처방명 개체명 리스트 구축 (≥ 5,000 entries) — downstream NER·retrieval에 활용.
- 중복 제거: SimHash 기반 문장 중복 제거.
- 품질 필터: 한자 비율, 문장 길이, 국역 누락 여부 체크.

### 3.4 보조 코퍼스 (옵션)
외부 코퍼스로 코퍼스 규모를 1~2 orders 늘리는 게 필수.
- **한국어 일반**: AI Hub 의료·법률 제외 일반 코퍼스, KLUE, 모두의말뭉치
- **한문 일반**: Kanseki Repository, CTP (Chinese Text Project, 비상업/학술), 고려대장경(CBETA 유사)
- **생의학**: PubMed abstracts (CC0), K-MMLU 의료 문항
- 각 소스는 **독립적으로 라이선스 체크** 후 편입.

---

## 4. 모델 전략

### 4.1 base 모델 후보 비교

| 후보 | 장점 | 단점 |
|---|---|---|
| **Qwen2.5-7B / 14B** | 한문·중국어 고전 tokenizer 커버리지 최상, 다국어, 상업 라이선스 OK | 한국어는 모국어 수준 아님 |
| **EXAONE-3.5 (LG)** | 한국어 최상, 국내 레퍼런스 | 한자 고전 토큰 분해가 비효율적일 수 있음, 라이선스 제약 |
| **Llama-3.1-8B** | 생태계, 도구 풍부 | 한문·한국어 둘 다 약함 |
| **Gemma-2-9B** | 경량, 토크나이저 준수 | 고전 한문 약함 |
| **HyperCLOVA-X seed** | 한국어 최고 | 공개 weights 제한, 한문 미지수 |

**1차 선택**: **Qwen2.5-7B-Instruct** (또는 14B).
이유: (1) 한문 tokenization이 본 프로젝트에서 가장 중요한 축, (2) 상업 이용 가능 Apache-2.0 계열 라이선스, (3) 한국어도 acceptable 수준.

**대안 실험**: EXAONE 또는 Llama-3-Korean variant 1개와 A/B.

### 4.2 학습 단계
1. **Stage 0 — Tokenizer 확장(옵션)**: 한의학 고유명사 약 3,000개를 BPE에 추가 학습(tokenizer extension). 사전 평가에서 평균 token/entity ≥ 3 인 경우에만 진행.
2. **Stage 1 — Continued Pretraining (CPT)**:
   - 데이터: HanMed-Corpus 한문 + 국역 병렬 + 보조 코퍼스 (약 1~5 B tokens 혼합, replay 10~30%).
   - Objective: next-token prediction.
   - Mix ratio: 한의학 고전(한문+국역) 30%, 일반 한국어 40%, 일반 한문 20%, 일반 영어 10%.
   - 학습량: 1~3 epoch over 한의학 part (≈ 2~6 B tokens 총 학습량).
   - Precision: bf16, ZeRO-2 또는 FSDP, RTX A6000 ×2 또는 A100 cluster.
3. **Stage 2 — SFT**:
   - Seed 데이터: 고전 → 현대 한국어 번역쌍 (이미 DB에 존재), 처방 설명, 본초 정보, Q&A.
   - 합성 데이터: GPT-4/Claude로 번역·해설 데이터 증강 (human-in-the-loop 검수). 단, **합성데이터를 최종 모델 출력 주장에 사용할 때는 저자성을 명시**.
4. **Stage 3 — Preference tuning(옵션)**: DPO로 번역 품질·한의학 사실성 개선.

### 4.3 평가-중심 설계
학습 전에 평가셋 먼저. "Evaluation first, then train."

---

## 5. 평가 계획 (HanMed-Eval v0)

현재 공개된 한의학 전용 LLM 벤치마크는 없음. 직접 구축한다.

### 5.1 태스크 구성
1. **고전 번역 (translation)**: 한문 → 한국어 BLEU / chrF / COMET / 전문가 선호도.
2. **고전 독해 QA**: 동의보감 권별 MCQ (저자 직접 큐레이션 500 문항 목표).
3. **본초·처방 지식**: 본초 → 효능·성미·귀경 매핑, 처방 → 구성 약재 분해.
4. **임상 문헌 요약**: 현대 한의학 논문 abstract 요약.
5. **안전성**: 임상 의사결정 유도 프롬프트에 대해 적절한 거부/면책 응답 비율.

### 5.2 비교군
- Qwen2.5-7B/14B-Instruct (base)
- GPT-4o / Claude Sonnet (closed)
- EXAONE-3.5
- 본 프로젝트 HanMed-Chat

### 5.3 측정 지표
- 번역: BLEU, chrF, COMET (한국어 COMET 모델 사용), 전문가 5점 평가 (n=3 평가자, 100 샘플)
- QA: accuracy, token-F1
- 안전성: 부적절 임상 조언 비율 (목표 < 1%)

---

## 6. 시스템·인프라

- GPU: 최소 A6000 ×2 (보유), SFT까지 커버. CPT 전체는 A100 40GB ×8 수준 권장.
- 학습 프레임워크: **torchtune** 또는 **Llama-Factory** (CPT + SFT + DPO one-stop).
- Experiment tracking: wandb (rank 0 only).
- Data: 로컬 NVMe, 버전 관리 DVC.
- 재현: 데이터 해시 + config snapshot + git commit SHA.

---

## 7. 라이선스 및 법적 검토

### 7.1 mediclassics.kr
- **비상업 연구 및 논문 publication**은 허용 범위로 판단. 다만 (i) 출처 표기 의무, (ii) 재배포 비권장 — 즉 **가공된 데이터를 공개 배포하려면 사전 문의 필수**.
- 논문에 사용하려면: 논문 methods에서 정확한 출처, 파일 버전, 다운로드 일자 기재.
- 모델 weights 공개(Hugging Face 등)를 원할 경우 **상업 이용으로 간주될 수 있으므로 kiombook@kiom.re.kr 사전 승인**을 받은 뒤 진행. 이메일 문의 → 양해각서(MOU) 체결까지 2~8주 예상. 기획 초기에 착수해야 일정이 밀리지 않는다.

### 7.2 합성 데이터
- GPT-4/Claude 출력으로 SFT 데이터를 만들 경우 각 API의 ToS 확인 필요. 일반적으로 **자체 모델 학습에 사용 가능**하지만 조건이 있으므로 재확인.

### 7.3 의료 규제
- 한의학 처방·치료 권고는 의료법 관할. 모델을 임상 의사결정 지원 도구로 광고하지 않는다. 논문 claim도 "문헌 연구 보조"로 한정.

### 7.4 합성 고전 생성의 위험
- LLM이 원문에 없는 내용을 생성해 "고전 인용"으로 제시하는 hallucination은 학술 신뢰 훼손 가능 → **인용 근거 제시** 기능을 core requirement로 둔다(retrieval-augmented answer).

---

## 8. 리스크 레지스터

| # | 리스크 | 영향 | 대응 |
|---|---|---|---|
| R1 | 코퍼스 규모 부족 (CPT 효과 미미) | 중 | 보조 코퍼스 확대, 평가 기반 mix ratio 튜닝 |
| R2 | mediclassics 상업 이용 제약 → weights 미공개 강요 | 중 | KIOM 사전 협의, MOU 체결 옵션 |
| R3 | markup 문법이 복잡/불완전 → 파싱 실패 | 중 | 파서 개발에 1~2주 버퍼, 샘플 수기 검증 |
| R4 | Hallucination으로 거짓 한문 인용 생성 | 고 | RAG 필수, 인용 근거 UI, 평가에 factuality 포함 |
| R5 | 한의학 평가 벤치마크 부재 → 비교 어려움 | 중 | HanMed-Eval 자체 구축, 전문가 2~3인 검수 |
| R6 | 임상 오용 위험 | 고 | 안전성 평가·disclaimer·사용 범위 문서화 |
| R7 | from-scratch "foundation" 주장 오해 | 중 | 논문·문서에서 일관되게 "continued pretraining" 명시 |
| R8 | GPU 자원 부족 (14B 이상 모델) | 중 | 7B에서 먼저 성공 후 scale-up |

---

## 9. 로드맵 (가안, 총 6개월)

| 월 | 마일스톤 |
|---|---|
| M1 | KIOM 사전 문의, 데이터 크롤링, markup parser, corpus v0.1 |
| M2 | Corpus v1 (정렬/정제), HanMed-Eval v0 (100 문항) |
| M3 | Stage 1 CPT 예비 실험 (Qwen2.5-7B), 평가 세팅 |
| M4 | 본 CPT + SFT 1차, 중간 평가, 병렬 한-한문 번역 베이스라인 |
| M5 | DPO, 최종 평가, ablation (mix ratio, tokenizer extension) |
| M6 | 논문 작성, HanMed-Chat v1 data card·model card 공개 |

**Critical path**: M1 KIOM 라이선스 승인 → 이게 밀리면 전체 일정이 밀림.

---

## 10. 열린 질문 (Decision log 필요)

1. 14B vs 7B: 평가 개선 폭 대비 GPU 비용.
2. Tokenizer 확장 실제 이득: 한의학 고유명사가 이미 한자 단위로 잘 쪼개지면 확장 불필요.
3. RAG 기본 탑재 여부: 학습된 지식만으로 갈 것인가, retrieval을 fact-grounding으로 쓸 것인가.
4. 한문 → 영어 번역까지 지원할 것인가(영역본 데이터 활용).
5. 모델 공개 범위: weights / LoRA adapters / API only 중 어느 것?
6. 평가에 포함될 전문가 검수 규모(비용/시간).

---

## 11. 다음 실행 항목 (1주 내)

- [ ] KIOM `kiombook@kiom.re.kr` 에 연구 목적·범위·공개 계획 포함한 문의 메일 발송.
- [ ] `info.mediclassics.kr/contents/database/list` 수동 스냅샷 확보(Playwright).
- [ ] mediclassics 배포 파일 1종(예: 동의보감 내경편 권1) 다운로드 → markup 구조 역공학.
- [ ] 7B 모델 하드웨어 요구량 benchmark (A6000 ×2 기준 throughput).
- [ ] HanMed-Eval v0 설계 회의 (3개 태스크 정의, 전문가 패널 초안).

---

## Appendix A. 출처

- mediclassics.kr 메인 페이지 (161 서적, 18,162K 원문 표기)
- info.mediclassics.kr/apps/dist-texts/index.html (배포 서비스 안내, 라이선스)
- info.mediclassics.kr/contents/database/list (서적 목록 — Playwright 필요)
- Hellodd, "한의학연, '한의학 고전 DB' 웹 서비스 실시" (38종 국역 언급)
- 나무위키 "동의보감" (국영역 무료 배포 언급, 2차 출처)
