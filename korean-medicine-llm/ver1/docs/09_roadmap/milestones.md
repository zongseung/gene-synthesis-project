# 09. Roadmap — 6 Months

## 9.1 원칙

1. **Evaluation first**: HanMed-Eval v0가 Stage 1 CPT보다 먼저 완료
2. **KIOM licensing = critical path**: M0 즉시 착수, 모든 마일스톤과 병렬 진행
3. 각 마일스톤은 **exit gate**를 가지며, 미달 시 즉시 피벗 회의
4. 모든 내부 실험은 KIOM 승인 여부와 무관하게 진행, **공개만 승인 후**

## 9.2 마일스톤 요약

| 월 | 이름 | 핵심 산출물 | Gate |
|---|---|---|---|
| M0 | Pre-kickoff | KIOM 메일, 서적 목록, 1종 역공학, GPU 벤치 | — |
| M1 | 데이터 기반 | 파서 v0.1, Corpus v0.1, Eval v0 태스크 설계, 전문가 LoI | — |
| M2 | Eval 완성 | **HanMed-Eval v0 100문항**, Corpus v1, tokenizer 실측 | **G2** |
| M3 | CPT pilot | 5% 데이터 pilot, 믹스·하이퍼 1차, memory/throughput 실측 | — |
| M4 | CPT + SFT | Full CPT run, SFT, 중간 평가 | **G4** |
| M5 | 확장·ablation | (옵션) DPO, ablation (rank/mix/tokenizer), 전 비교군 eval | **G5** |
| M6 | 논문·릴리스 | 논문 draft, model/data card, 재현 레시피 공개 | — |

## 9.3 마일스톤 상세

### M0 — Week 0 (pre-kickoff, 1주)
병렬 4개 트랙:

1. **라이선스 트랙** (critical path 시작)
   - [ ] KIOM 이메일 초안 작성 → `kiombook@kiom.re.kr` 발송
   - [ ] 메일에 포함: 연구 목적, 이용 범위, publication·weights 공개 계획, 자동 다운로드 1 req/sec 통지, 교감기록 범위 질문
   - [ ] 소속기관 IRB 면제 신청
   - [ ] OpenAI/Anthropic API ToS 스냅샷 저장

2. **데이터 탐색 트랙**
   - [ ] Playwright로 `info.mediclassics.kr/contents/database/list` 렌더 → `data/meta/books.csv`
   - [ ] 대표 서적 1종 (동의보감 내경편 권1) 수기 다운로드
   - [ ] markup 태그 인벤토리 수기 작성 → `markup_spec.md`

3. **인프라 트랙**
   - [ ] Solar-10.7B HF page에서 정확한 라이선스 variant 확인
   - [ ] HF 모델 캐시 (offline 대비)
   - [ ] A6000 1장에서 Solar-10.7B bf16 LoRA **dummy run 500 steps** → peak mem, tok/s 기록
   - [ ] Llama-Factory vs torchtune 각 1 epoch → 프레임워크 확정

4. **인력 트랙**
   - [ ] 한의학 박사 2인 섭외 (LoI)
   - [ ] 평가 rubric 초안 공유

### M1 — Month 1: 데이터 기반
- [ ] 파서 v0.1 — 파이썬 정규식 기반, 대표 5종 통과
- [ ] 파서 검증 셋 50 레코드 수기 검수 → 정확도 보고
- [ ] Corpus v0.1 (처음 50종)
- [ ] HanMed-Eval v0 태스크 **설계만** (스키마, rubric, 프롬프트 템플릿)
- [ ] 보조 코퍼스 수집 시작 (Wiki-ko, CBETA)

### M2 — Month 2: 평가 먼저
- [ ] Parser v1 (확장), 161 중 가능한 전부 처리
- [ ] Corpus v1 동결 → `corpus_v1.json` 스냅샷
- [ ] **HanMed-Eval v0 100문항 큐레이션 + 2인 교차 검수**
- [ ] Solar tokenizer 실측 → Stage 0 확장 결정
- [ ] **G2 gate 판정**

### M3 — Month 3: CPT pilot
- [ ] Stage 1 CPT LoRA pilot
  - 데이터: Corpus v1의 5% 샘플 + replay 5%
  - Steps: 1,000
  - 목적: 하이퍼 검증, 메모리 실측, 학습 안정성
- [ ] 결과 분석 → 믹스·rank·lr 1차 튜닝
- [ ] 최종 config 동결 (`configs/stage1_cpt.yaml`)

### M4 — Month 4: CPT + SFT 본 run
- [ ] Stage 1 CPT 전체 — 목표 100M~300M tokens total
- [ ] 중간 eval (step 50%, 100%)
- [ ] Stage 2 SFT — seed + 합성(검수 ≥ 20%)
- [ ] HanMed-Eval v0 전 지표 측정
- [ ] **G4 gate 판정**

### M5 — Month 5: 확장 · ablation
- [ ] (옵션) Stage 3 DPO
- [ ] Ablation:
  - LoRA rank 16 / 32 / 64
  - 믹스 비중 (HanMed 30/50/70%)
  - Stage 0 확장 유/무
  - Solar vs Bllossom
- [ ] 비교군 전체 평가 (Solar base, Bllossom, Qwen2.5-14B, GPT-4o, Claude)
- [ ] Inter-rater agreement 확인
- [ ] **G5 gate 판정**

### M6 — Month 6: 논문·릴리스
- [ ] 논문 draft (SCI target)
- [ ] Model card + data card
- [ ] **KIOM 승인 여부에 따라**:
  - ✅ 승인: `HanMed-Public-LoRA` adapter 공개 (HF)
  - ❌ 미승인: 논문만, 코드·재현 레시피만 공개
- [ ] 재현 스크립트 `scripts/train.sh` + README 최종화

## 9.4 Critical Path 시각화

```
M0 KIOM email ────────────────────────────────────────────┐
     │ (2~6 months background)                            │
     └───────────────────────────────────────→ M6 decision ▼
                                                    공개 여부
M0 data explore ─→ M1 parser ─→ M2 corpus+eval ─→ M3 pilot ─→ M4 CPT+SFT ─→ M5 ablation ─→ M6 paper
                                       ▲
                                       │ evaluation first 강제
                                       │
M0 infra/expert ─────────→ M1 섭외 완료 ┘
```

## 9.5 일정 리스크와 Contingency

| 상황 | Contingency |
|---|---|
| KIOM 회신 6개월 초과 | 논문은 예정 제출, adapter 공개만 postpone |
| M2 gate 미달 (파서) | 스코프 → 동의보감 1종 심화 |
| M4 gate 미달 (CPT 효과) | 1회 재학습 → 2차 미달 시 피벗 회의 |
| M5 gate 미달 (최종) | 논문 방향 전환 — 평가 벤치 + negative result |
| GPU 충돌 (gene-synthesis 과부하) | 임차 클러스터 (A100 ×8) 단기 임차 |
| 전문가 이탈 | v0 축소 (100→60문항) |

## 9.6 주간 운영

- 매주 월요일: 주간 goal 설정 → `docs/09_roadmap/weekly/{YYYY-WW}.md`
- 매주 금요일: 결과 요약 + risk_log 업데이트
- 월말: 마일스톤 gate 판정 회의
