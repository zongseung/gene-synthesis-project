# 08. Risk Register & Exit Gates

## 8.1 Risk Table

영향도: 고/중/낮 — 실패 시 프로젝트에 미치는 충격.  
확률: 고/중/낮 — 현재 시점에서 발생 가능성.

| # | 리스크 | 영향 | 확률 | 대응 | 담당 마일스톤 |
|---|---|---|---|---|---|
| R1 | KIOM 승인 지연 (>3개월) | 고 | 중 | M0 즉시 문의, adapter 공개 일정 flex, 내부 실험은 병렬 진행 | M0 |
| R2 | markup 파싱 정확도 < 95% | 중 | 중 | 파서 2주 버퍼, 실패 서적 exclude 리스트, grammar 승격 | M1~M2 |
| R3 | CPT 효과 미미 (base 대비 < +5 chrF) | 중 | 중 | 믹스 재튜닝 → rank 상향(32→64) → Stage 0 확장 재시도 | M4 |
| R4 | Hallucinated classical citation | **고** | **고** | 근거 section_id 출력 SFT, 평가 T1에 근거 매칭 포함, 장기 RAG | 전 단계 |
| R5 | 전문가 평가 인력 미확보 | 중 | 중 | 대학 협력처 사전 섭외, v0는 100문항으로 소형화 | M1 |
| R6 | base 모델 라이선스 (Solar NC 이슈) | 중 | 중 | Apache-2.0 variant 확인, Bllossom backup 준비 | M0 |
| R7 | 보조 코퍼스 cross-contamination | 중 | 중 | Public/Internal adapter 이원 학습 | M3 |
| R8 | 한자 tokenizer 비효율 → 학습 비용 증가 | 중 | 낮 | Stage 0 확장, seq_len 조정 | M2 |
| R9 | 국역 커버리지가 예상보다 낮음 (<20%) | 중 | 낮 | 동의보감 단일 서적 심화로 scope 축소 | M2 |
| R10 | GPU 점유 충돌 (gene-synthesis와 공유) | 낮 | 중 | 캘린더 blocking, 야간/주말 우선 | 전 단계 |
| R11 | 합성 SFT 데이터 ToS 변경 | 낮 | 낮 | seed 직접 작성 비율 ≥ 50% 유지, ToS 변경 시 합성분 배제 | M4 |
| R12 | 전문가 평가 inter-rater disagreement (α<0.4) | 중 | 낮 | Calibration 10샘플, rubric 재작성, 3인 증원 | M5 |
| R13 | 논문 reviewer: "기여 불충분" | 중 | 중 | 평가 벤치 자체 기여 강조, 비교군 확대 | M6 |

## 8.2 Exit Gates (마일스톤별 go/no-go)

### M2 gate — 데이터·평가 준비
**필수 조건**:
- Corpus parse 정확도 ≥ 95% on validation set
- 파싱 성공 서적 수 ≥ 80 (161 중)
- HanMed-Eval v0 100문항 완성 + 2인 교차 검수
- Solar tokenizer 실측 완료, Stage 0 결정

**미달 시**:
- 파싱: 스코프를 **동의보감 단일 서적 심화**로 축소
- Eval: v0를 60문항으로 축소 (T1/T2 각 20, T3/T4 각 10)
- Tokenizer: 확장 스킵하고 진행

### M4 gate — CPT + SFT 결과
**필수 조건** (§05.6):
- T1 chrF: base 대비 +5점 이상
- T1 전문가 선호 승률 ≥ 55%
- T2 accuracy: base 대비 +10%p 이상
- T4 refuse rate ≥ 99%

**미달 시 조치**:
- 1개 미달: 해당 태스크 타겟 데이터 증강 후 재학습 1회
- 2개 이상 미달: M4~M5 피벗 회의 → 아래 4.2.2 피벗 옵션

### M5 gate — 최종 eval
**필수 조건**:
- 위 4개 지표 중 **3개 이상 충족**
- Ablation 결과 재현성 확인 (seed 2개)

**미달 시**:
- 논문 방향을 **negative result + 한의학 LLM 평가 벤치마크 기여**로 전환
- 학습 레시피 실패 원인 분석 publish

### 8.2.1 피벗 옵션 (M4 gate 실패 시)
1. **Scope 축소**: 동의보감 단일 서적 + 번역·QA 2개 태스크만
2. **Base 전환**: Solar → Bllossom 또는 EXAONE
3. **LoRA rank 상향**: 32 → 64 또는 128, target modules 확대
4. **Full fine-tune 평가**: A6000 2장 FSDP + ZeRO-3 offload 1 epoch pilot
5. **Negative result publish**: 벤치마크 + 레시피 실패 분석

## 8.3 Risk Log 운영

- 주간 업데이트 별도 파일: `docs/08_risks/risk_log.md`
- 신규 리스크 발견 시 본 테이블에 추가 + log에 날짜별 기록
- 해결된 리스크는 "resolved" 표기, 삭제하지 않음 (history 보존)

## 8.4 Assumption Register

프로젝트가 깔고 가는 **가정**들 — 깨지면 피벗이 필요한 항목.

| # | 가정 | 깨질 경우 |
|---|---|---|
| A1 | mediclassics 개별 다운로드가 자동화 허용 범위 | 수기 다운로드 전환 |
| A2 | 국역 서적 수 ≥ 38종 (2014 기사 기준) | scope 축소 |
| A3 | Solar-10.7B Apache-2.0 variant 존재 | Bllossom 또는 EXAONE 전환 |
| A4 | 전문가 2인 섭외 가능 | v0 문항 수 축소 |
| A5 | GPU 2장이 8주 이상 점유 가능 | 임차 클러스터 사용 |
| A6 | KIOM이 연구 공개에 협력적 | 내부용으로만 scope 제한 |

각 가정은 M0~M1 사이에 **명시적으로 검증**해야 함 → §09 체크리스트 반영.
