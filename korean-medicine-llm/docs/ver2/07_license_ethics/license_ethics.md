# 07. License · Ethics · Safety (ver2)

> ver1 대비 변경: Solar Apache-2.0 variant 서술을 **불확실 조건부**로 약화(D6), §7.9 체크리스트에 전문가 계약 템플릿·SFT 검수 rubric·prompt format 표준 placeholder 추가, 전문가 계약 세부 문단 신설.

## 7.1 mediclassics.kr 데이터 라이선스

### 7.1.1 명시된 조건
- **비상업 이용**: 제한 없음
- **재배포**: 원칙적 가능하나 "지속 업데이트되므로 비권장"
- **상업 이용**: `kiombook@kiom.re.kr` 서면 문의 필수
- **출처 표기 의무**: "한의학고전DB (mediclassics.kr)" — 논문·데이터카드·모델카드·UI 전부

### 7.1.2 보수 해석 (권장)

KIOM 공식 회신 전까지 아래 해석을 따른다.

| 시나리오 | 판단 |
|---|---|
| 논문 publication | ✅ 가능. file version, download date, git SHA 명시 |
| 학습 코드·파서 공개 | ✅ 가능 (KIOM 데이터 포함 안 함) |
| 가공 corpus를 HF Datasets에 공개 | ❌ 승인 전 금지 (재배포로 해석 가능) |
| adapter weights 공개 (HF Models) | ❌ 승인 전 금지 |
| 상업 API·앱 서비스 | ❌ 반드시 KIOM 승인 선행 |
| **DVC remote를 외부 클라우드(S3/GCS)에 두기** | ❌ **금지** — 해외 CSP 저장은 재배포로 해석될 여지 (§06.5, D12) |

**원칙**: "불확실하면 공개하지 않는다."

### 7.1.3 KIOM 문의 일정 (현실)

국가출연연구소 데이터 이용허락의 일반 경로:

| 단계 | 기간 |
|---|---|
| 1차 문의 → 담당자 응답 | 1~4주 |
| 자료 제출 (연구계획서, 이용 범위, 공개 계획) | 1~2주 |
| 법무·계약 검토 | 2~6주 |
| MOU 또는 이용허락서 체결 | 2~6주 |
| **총** | **2~6개월** |

**→ 이것이 critical path** (§09 M0 최상단 항목).

## 7.2 Base 모델 라이선스

| 모델 | 라이선스 | 상업 | 연구 | adapter 재배포 |
|---|---|---|---|---|
| **Solar-10.7B-Instruct-v1.0** | **cc-by-nc-4.0 확실 / Apache-2.0 variant 존재 불확실** — M0 HF 페이지 라이선스 문자열 직접 확인 필수 | NC ❌ / Apache ✅(확증 시) | ✅ | 상속 |
| Llama-3.1-Korean-Bllossom-8B | Llama 3 Community License | 조건부 ✅ (MAU 제한) | ✅ | 상속 |
| EXAONE-3.5-7.8B | EXAONE AI Model License | ❌ 상업 제한 | ✅ | 상속 |
| Qwen2.5-7B | Apache-2.0 | ✅ | ✅ | ✅ |
| Qwen2.5-14B | Qwen License (MAU 100M 제한) | 조건부 | ✅ | 상속 |

**주의 (D6)**:
- 우리가 배포하는 LoRA adapter는 base 라이선스를 **상속**한다.
- Solar의 Apache-2.0 variant 존재 여부는 **현 시점에서 불확실**. M0 체크리스트 최상단에 Solar 라이선스 확증 배치 (§09 M0).
- **실패 시(Apache variant 없음 또는 조건 부적합) 즉시 Bllossom-8B로 전환**. 추가 fallback은 Qwen2.5-14B-Instruct(한자 우수).
- Base 확증 전까지 본 문서의 Solar 관련 서술은 **조건부**로 읽을 것.

## 7.3 보조 코퍼스 Cross-Contamination

> **adapter weights는 학습에 사용된 가장 엄격한 라이선스에 묶인다.**

분리 학습 전략:
- `HanMed-Public-LoRA`: 재배포 가능 소스만 (mediclassics 승인분 + Wikipedia-ko + base Apache variant)
- `HanMed-Internal-LoRA`: 전부 사용 — 논문 ablation 표에서만 수치 비교, 외부 배포 금지

공개는 `Public` 버전만.

## 7.4 합성 SFT 데이터 (GPT-4 / Claude)

### 7.4.1 API ToS 상태 (2026-04 기준)
- OpenAI: "경쟁 LLM 학습" 금지 조항
- Anthropic: 유사 제한
- → 본 프로젝트는 "한의학 도메인 특화 어시스턴트" 포지셔닝, 경쟁 일반 목적 LLM 아님을 문서에 명시

### 7.4.2 안전 경로
- 전문가 직접 작성 seed SFT 데이터가 **최소 50%**
- 합성 증강은 보조만, 전부 `synthesis_provenance` 태그
- **SFT 검수 rubric** 적용 — 허용/반려 기준·반려 시 재생성·inter-reviewer agreement 목표는 `docs/rubric/sft_review_rubric.md` 참조 (§04는 Agent A 담당, 본 문서는 placeholder만)
- 논문·model card에 합성 비율 투명 공개

## 7.5 IRB / 윤리심의

- v1 범위 (mediclassics 고전) = **비식별 공개 텍스트**, 인간 대상 연구 아님 → **IRB 면제** 경로
- 기관 IRB 사무국에 면제 신청으로 공식 기록 남김 (M0)
- v2 확장 (현대 임상/환자) 시 IRB 필수 — 본 기획 범위 외

## 7.6 의료 규제 및 Safety

### 7.6.1 규제 위치
한국 의료법상 "진단·처방"은 한의사 면허 행위. 본 모델이 이를 대체한다고 광고하면 의료법 저촉.

### 7.6.2 필수 Disclaimer
```
본 모델은 한의학 고문헌 연구 보조 도구이며,
진단·치료·처방의 의학적 판단을 대체하지 않습니다.
임상 적용은 한의사 등 면허 의료인의 판단 하에만 이루어져야 합니다.
```
모델카드, README, 평가 UI, 논문 abstract·conclusion 전부 명시.

### 7.6.3 T4 연계
§05.3.4 T4 refuse rate ≥ 99%. 미달 시 SFT 안전성 데이터 추가.

## 7.7 Hallucination 책임

- "원문에 없는 문장을 고전 인용으로 생성" → 학술 신뢰 훼손, 잠재적 명예훼손 리스크
- Mitigation: (1) SFT에 근거 `section_id` 출력 태스크, (2) T1 평가에서 전문가가 근거 원문 매칭, (3) v2 RAG 탑재

## 7.8 데이터 주체·개인정보

- 고전 저자 → 개인정보 해당 없음
- KIOM 역자 저작인접권 → KIOM이 저작권자 또는 수임자
- 학습·배포에 별도 개인정보 처리 없음

## 7.9 전문가 계약 (신규)

전문가 2인(한의학 박사)을 섭외하여 평가 큐레이션과 선호도 평가에 참여시킨다. 섭외는 LoI → 계약 → 업무 순으로 진행하며, 계약 템플릿은 M0에서 준비한다.

계약 포함 조항:
- **NDA**: KIOM corpus 및 학습 전 snapshot에 대한 비공개 의무
- **보수 단가**: 시간당 단가 (학계 표준 자문료), 실제 투입 시간 기준 정산
- **데이터 소유권**: 전문가가 작성한 eval 문항·reference 번역·rubric 평가는 **프로젝트에 귀속**, 전문가는 공동저자 자격 유지
- **평가 결과 저작권**: 선호도 평가 집계 결과는 논문·model card에 사용 가능하되 개별 평가자 익명화
- **COI(이해충돌) 선언**: SCI 제출 시 의무. 경쟁 LLM 프로젝트 참여·KIOM 직접 고용 관계 등을 사전 선언

## 7.10 체크리스트 (M0 ~ M2)

- [ ] **KIOM 이메일 발송** (`kiombook@kiom.re.kr`) — 연구목적, 범위, 공개계획, 자동 다운로드 통지 포함 [§09 M0 최우선]
- [ ] **Solar base HF page 라이선스 문자열 직접 확인** — Apache-2.0 variant 존재 여부 확증. 실패 시 즉시 Bllossom 전환 결정 [§09 M0 최우선]
- [ ] 연구자 소속기관 IRB 면제 신청
- [ ] OpenAI/Anthropic API ToS 최신 스냅샷 저장
- [ ] Disclaimer 국/영문 준비
- [ ] 보조 코퍼스 라이선스 파일 `data/licenses/{source}.txt` commit
- [ ] **전문가 계약 템플릿 작성** — NDA, 보수 단가, 데이터 소유권, 평가 저작권, COI 조항 포함
- [ ] **SFT 합성 데이터 검수 rubric 초안** — placeholder: `docs/rubric/sft_review_rubric.md` (§04 연계)
- [ ] **Prompt format 표준 문서** — placeholder: `docs/standards/prompt_format.md` (§06.9 연계)
- [ ] DVC remote 위치 확정 — 로컬 NFS 또는 기관 내부 스토리지 (§06.5, §07.1.2)
