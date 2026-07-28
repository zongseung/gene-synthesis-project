# 07. License · Ethics · Safety

## 7.1 mediclassics.kr 데이터 라이선스

### 7.1.1 명시된 조건
- **비상업 이용**: 제한 없음
- **재배포**: 원칙적 가능하나 "지속 업데이트되므로 비권장"
- **상업 이용**: `kiombook@kiom.re.kr` 서면 문의 필수
- **출처 표기 의무**: "한의학고전DB (mediclassics.kr)" — 논문·데이터카드·모델카드·UI 전부

### 7.1.2 보수 해석 (권장)

아래는 문구가 모호한 부분의 **보수적** 해석이며, KIOM 공식 회신으로 확정될 때까지 본 해석을 따른다.

| 시나리오 | 판단 |
|---|---|
| 논문 publication (결과 보고) | ✅ 가능. 단 methods에 file version, download date, git SHA 명시 |
| 학습 코드·파서 공개 | ✅ 가능 (KIOM 데이터 포함 안 함) |
| 가공된 corpus를 HF Datasets에 공개 | ❌ **KIOM 공식 승인 전까지 금지** (재배포로 해석 가능) |
| adapter weights 공개 (HF Models) | ❌ **KIOM 공식 승인 전까지 금지** (학습 데이터로 KIOM corpus 포함) |
| 상업 API·앱 형태 서비스 | ❌ 명백히 상업 이용 — 반드시 KIOM 승인 선행 |

**원칙**: "불확실하면 공개하지 않는다. 공개는 승인 후에만."

### 7.1.3 KIOM 문의 일정 (현실 재추정)

v0 draft의 "2~8주"는 낙관적이었다. 국가출연연구소 데이터 이용허락의 일반적 경로:

| 단계 | 기간 |
|---|---|
| 1차 문의 → 담당자 응답 | 1~4주 |
| 자료 제출 (연구계획서, 이용 범위, 공개 계획) | 1~2주 |
| 법무·계약 검토 | 2~6주 |
| MOU 또는 이용허락서 체결 | 2~6주 |
| **총** | **2~6개월** |

**→ 이것이 critical path** (§09). 프로젝트 시작 전 (M0) 즉시 이메일 발송.

## 7.2 Base 모델 라이선스

| 모델 | 라이선스 | 상업 이용 | 연구 | adapter 재배포 |
|---|---|---|---|---|
| Solar-10.7B-Instruct-v1.0 | cc-by-nc-4.0 (원본) / Apache-2.0 variant 존재 | NC 버전 ❌ / Apache ✅ | ✅ | **base 라이선스 상속** |
| Llama-3.1-Korean-Bllossom-8B | Llama 3 Community License | 조건부 ✅ (MAU 제한) | ✅ | 상속 |
| EXAONE-3.5-7.8B | EXAONE AI Model License | ❌ 상업 제한 | ✅ | 상속 |
| Qwen2.5-7B | Apache-2.0 | ✅ | ✅ | ✅ |
| Qwen2.5-14B | Qwen License (MAU 100M 제한) | 조건부 | ✅ | 상속 |

**주의**:
- 우리가 배포하는 LoRA adapter는 base 모델 라이선스를 **상속**한다.
- Primary base 선정 시 Solar **Apache-2.0 variant** 를 명확히 사용 — 아니면 연구용으로만 가능.
- M0 체크리스트: Solar 버전을 HF page에서 확인하고 README에 SHA 기록.

## 7.3 보조 코퍼스 Cross-Contamination

§03.5 표 참고. 핵심 원칙:

> **adapter weights는 학습에 사용된 가장 엄격한 라이선스에 묶인다.**

분리 학습 전략:
- `HanMed-Public-LoRA`: **재배포 가능** 소스만 (mediclassics 승인분 + Wikipedia-ko + Solar Apache variant)
- `HanMed-Internal-LoRA`: 전부 사용 — 논문 실험표의 "ablation with auxiliary" 용도

공개 배포는 `Public` 버전만, `Internal` 은 논문 내 수치 비교에만 사용.

## 7.4 합성 SFT 데이터 (GPT-4 / Claude)

### 7.4.1 API ToS 상태 (2026-04 기준, 변경 가능)
- **OpenAI**: 출력으로 "경쟁 LLM 학습" 금지 조항 존재
- **Anthropic**: 유사 제한
- → 본 프로젝트는 "한의학 도메인 특화 어시스턴트"로 포지셔닝, 경쟁 일반 목적 LLM이 아님을 문서에 명시

### 7.4.2 안전 경로
- 전문가가 직접 작성한 seed SFT 데이터가 **최소 50%**
- 합성 증강은 보조만, 전부 `synthesis_provenance` 태그
- 논문·model card에 합성 비율 투명 공개
- 상용 배포 시 해당 합성 데이터는 별도 재라이선싱 필요

## 7.5 IRB / 윤리심의

### 7.5.1 v1 범위 — mediclassics 고전만
- 고전 문헌 = **비식별 공개 텍스트**, 인간 대상 연구 아님
- → **IRB 면제** 경로로 판단 (기관 IRB 사무국 확인 필요)

### 7.5.2 향후 v2 확장 시
- 현대 한의학 임상 논문 / 환자 데이터 포함 시 → **IRB 필수**
- 본 기획은 v1에서 해당 시나리오 배제

### 7.5.3 기관 확인
- 연구자 소속 기관 IRB 사무국에 **면제 신청**으로 공식 기록 남김 (M0)

## 7.6 의료 규제 및 Safety

### 7.6.1 규제 위치
- 한국 의료법상 "진단·처방"은 한의사 면허 행위
- 본 모델이 이를 대체하는 것처럼 광고하면 의료법 저촉 가능

### 7.6.2 Disclaimer (필수 문구)
```
본 모델은 한의학 고문헌 연구 보조 도구이며,
진단·치료·처방의 의학적 판단을 대체하지 않습니다.
임상 적용은 한의사 등 면허 의료인의 판단 하에만 이루어져야 합니다.
```
- 모델 카드 intended use
- README
- 평가 UI
- 논문 abstract 및 conclusion
위 모든 곳에 명시.

### 7.6.3 T4 평가 연계
- §05.3.4 T4 안전성 평가에서 refuse rate ≥ 99% 목표
- 미달 시 SFT 데이터에 "임상 결정 거부 + 면책" 응답 추가 학습

## 7.7 Hallucination 책임

### 7.7.1 위험
- "원문에 없는 문장을 고전 인용으로 생성" → **학술 신뢰 훼손**, 잠재적 명예훼손 위험

### 7.7.2 Mitigation
1. SFT에 "인용 시 근거 section_id 출력" 태스크 포함
2. 평가 T1에서 전문가가 근거 원문 매칭 여부 확인
3. 장기 (v2): RAG (retrieval-augmented generation) 탑재 — 매 답변에 `[출처: 동의보감 내경편 권1 §012]` 링크

## 7.8 데이터 주체·개인정보

- 고전 저자(수백 년 전 사망) → 개인정보 해당 없음
- 현대 KIOM 역자의 **저작인접권**은 존재 → KIOM이 저작권자 또는 그 수임자로 판단
- 학습·배포에 별도 개인정보 처리는 없음

## 7.9 체크리스트 (M0 ~ M2)

- [ ] KIOM 이메일 발송 (`kiombook@kiom.re.kr`) — 연구목적, 범위, 공개계획, 자동 다운로드 통지 포함
- [ ] Solar base HF page에서 정확한 라이선스 버전 확인, README 기록
- [ ] 연구자 소속 기관 IRB 면제 신청
- [ ] OpenAI/Anthropic API ToS 최신 버전 재확인
- [ ] Disclaimer 문구 3개국어 준비 (국문, 영문)
- [ ] 보조 코퍼스별 라이선스 파일을 `data/licenses/{source}.txt` 로 commit
