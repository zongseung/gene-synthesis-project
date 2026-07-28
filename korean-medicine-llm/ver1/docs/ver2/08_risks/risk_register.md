# 08. Risk Register & Exit Gates (ver2)

> ver1 대비 변경: R3에 토큰 예산(32~43M unique, 150~250M cap) 근거 반영, R6 대응 강화, **R14 (DUS LoRA 독립), R15 (Eval contamination) 신규**, §8.2 exit gate에 T5 general-ko 추가, §8.4 A3 정정.

## 8.1 Risk Table

영향도: 고/중/낮 / 확률: 고/중/낮.

| # | 리스크 | 영향 | 확률 | 대응 | 담당 |
|---|---|---|---|---|---|
| R1 | KIOM 승인 지연 (>3개월) | 고 | 중 | M0 즉시 문의, adapter 공개 일정 flex, 내부 실험 병렬 진행 | M0 |
| R2 | markup 파싱 정확도 < 95% | 중 | 중 | 파서 2주 버퍼, 실패 서적 exclude, grammar 승격 | M1~M2 |
| R3 | **CPT 효과 미미** (T1 chrF < +3, 전문가 승률 < 55%) | 중 | 중 | HanMed unique 32~43M tokens 기준 믹스 재튜닝, rank 32→64 상향, 재학습 1회. Cap 150~250M 내 재실행 (§04.5 D3) | M4 |
| R4 | Hallucinated classical citation | **고** | **고** | 근거 section_id 출력 SFT, T1 근거 매칭, 장기 RAG | 전 단계 |
| R5 | 전문가 평가 인력 미확보 | 중 | 중 | 대학 협력처 사전 섭외, v0 200문항 유지(T5 자동채점 100 포함), 필요 시 100으로 축소 | M1 |
| R6 | **Solar Apache-2.0 variant 불확실 / 확증 실패** | 중 | **중** | **M0 24시간 내 HF 페이지 라이선스 문자열 직접 확인, 불확실 시 즉시 Bllossom-8B 전환** (D6, §07.2) | M0 |
| R7 | 보조 코퍼스 cross-contamination | 중 | 중 | Public/Internal adapter 이원 학습 | M3 |
| R8 | 한자 tokenizer 비효율 → 학습 비용 증가 | 중 | 낮 | Stage 0 data-driven 결정 (D5), seq_len 조정 | M2 |
| R9 | 국역 커버리지 예상보다 낮음 (<20%) | 중 | 낮 | 동의보감 단일 서적 심화로 scope 축소 | M2 |
| R10 | GPU 점유 충돌 (gene-synthesis와 공유) | 낮 | 중 | 캘린더 blocking, 야간/주말 우선 | 전 단계 |
| R11 | 합성 SFT ToS 변경 | 낮 | 낮 | seed 직접 작성 ≥ 50% 유지, ToS 변경 시 합성분 배제 | M4 |
| R12 | 전문가 inter-rater α<0.4 | 중 | 낮 | Calibration 10샘플, rubric 재작성, 3인 증원 | M5 |
| R13 | 논문 reviewer: "기여 불충분" | 중 | 중 | 평가 벤치 자체 기여 강조, 비교군 확대, T5 general-ko 결과 포함 | M6 |
| **R14** | **Solar DUS 복제 layer → PEFT 독립 LoRA 할당 → adapter 크기·메모리 ~2×** (D7) | 중 | 중 | **M3 pilot에서 peak mem·adapter param 실측. 필요 시 복제 layer LoRA 공유 정책 ablation (shared vs independent) 실행. 공유로 성능 drop 시 grad ckpt on + bs 축소로 대응** | M3 |
| **R15** | **Eval set contamination — held-out 30문장이 CPT 학습 셋에 유출** | 고 | 낮 | **M2에서 hash filter 훅 자동화 (§05.7, §06.5 D8). 학습 데이터 prep 시 `eval/hashes/heldout_*.txt` 와 교집합 검사, 발견 시 빌드 fail** | M2 |

## 8.2 Exit Gates

### M2 gate — 데이터·평가 준비
**필수 조건**:
- Corpus parse 정확도 ≥ 95% on validation set
- 파싱 성공 서적 수 ≥ 80 (161 중)
- HanMed-Eval v0 **200문항** 완성 + 2인 교차 검수 (T1~T4 수기 큐레이션, T5 100은 KLUE-YNAT stratified 자동 발췌)
- Solar tokenizer **실측 완료, Stage 0 data-driven 결정** (D5)
- **Eval contamination hash filter 훅 동작 확인** — 더미 학습 셋으로 `build fail` 경로 테스트 통과 (R15)

**미달 시**:
- 파싱: 스코프 → 동의보감 단일 서적 심화
- Eval: v0를 100문항으로 축소 (T1/T2 각 20, T3/T4 각 10, T5 40 — granularity 2.5%p, E5 3%p 경계)
- Tokenizer: data-driven 기본값으로 진행

### M4 gate — CPT + SFT 결과
**필수 조건** (§05.6):
- E2: T1 전문가 선호 승률 ≥ 55% (**primary**)
- E3: T2 accuracy base 대비 +10%p
- E4: T4 refuse rate ≥ 99%
- **E5: T5 general-ko Δ accuracy drop ≤ 3%p** (신규)
- E1 (T1 chrF +3점): monitoring only, 블로킹 아님

**미달 시**:
- 1개 미달: 해당 태스크 타겟 데이터 증강 후 재학습 1회. E5 미달 시 Wiki-ko replay 비중 30→50% 상향
- 2개 이상 미달: M4~M5 피벗 회의 → §8.2.1 피벗 옵션

### M5 gate — 최종 eval
**필수**: 위 primary 4개(E2~E5) 중 **3개 이상 충족** + seed 2개 재현성 확인.
**미달 시**: 논문 방향을 평가 벤치 기여 + negative result로 전환.

### 8.2.1 피벗 옵션 (M4 gate 실패)
1. Scope 축소 (동의보감 1종 + 태스크 2개)
2. Base 전환 (Solar → Bllossom 또는 EXAONE)
3. LoRA rank 상향 (32 → 64/128, target modules 확대)
4. DUS LoRA 공유 정책 전환 (R14)
5. Replay 비중 상향 (R1 E5 대응)
6. Full fine-tune 1 epoch pilot (FSDP + ZeRO-3)
7. Negative result publish (벤치 + 레시피 실패 분석)

## 8.3 Risk Log 운영

- 주간 업데이트: `docs/08_risks/risk_log.md`
- 신규 리스크 발견 시 본 테이블에 추가 + log 날짜별 기록
- 해결된 리스크는 "resolved" 표기 유지

## 8.4 Assumption Register

| # | 가정 | 깨질 경우 |
|---|---|---|
| A1 | mediclassics 개별 다운로드가 자동화 허용 범위 | 수기 다운로드 전환 |
| A2 | **HanMed unique 토큰 32M~43M** (영역 제외), **국역 서적 수 ≥ 38종** (§02.5, §04.5 D3) | scope 축소, epoch 수 상향 |
| **A3** | **Solar-10.7B Apache-2.0 Instruct variant 존재 (M0 확증 대상)** | **즉시 Bllossom-8B 전환** (§07.2, R6) |
| A4 | 전문가 2인 섭외 가능 (계약 템플릿 §07.9) | v0 문항 수 축소 200→100 |
| A5 | GPU 2장이 8주 이상 점유 가능 | 임차 클러스터 사용 |
| A6 | KIOM이 연구 공개에 협력적 | 내부용으로만 scope 제한 |
| **A7** | **CPT 총 학습량 150M~250M tokens cap 내에서 HanMed epoch 1.5~3 유의한 효과 산출** (D3) | 3 epoch 이상 또는 Stage 0 확장 병행 |
| **A8** | **DUS 복제 layer에 대해 PEFT LoRA 공유 또는 독립 중 한 쪽이 최소 한 개 이상 동작** (R14) | rank 축소 또는 grad ckpt on + bs 축소 |

각 가정은 M0~M3 사이 **명시적으로 검증** — §09 체크리스트 반영.
