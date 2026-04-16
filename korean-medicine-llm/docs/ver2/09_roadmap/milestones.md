# 09. Roadmap — 6 Months (ver2)

> ver1 대비 변경: M0 최상단 3줄 재정렬(KIOM 메일 → Solar 라이선스 확증 → 전문가 LoI+계약 템플릿), M2 gate 에 contamination 훅 검증 추가, M3 pilot에 **Wiki-ko replay / DUS LoRA 공유** 두 ablation 명시, M4 본 run 목표를 **150M~250M tokens** cap 으로 재기술, M4 gate 에 E5 (T5 general-ko drop ≤ 3%p) 추가, §9.5 contingency 에 Solar fallback 1주 전환 추가.

## 9.1 원칙

1. **Evaluation first**: HanMed-Eval v0 완료 후 Stage 1 CPT 착수
2. **KIOM licensing = critical path**: M0 즉시 착수, 모든 마일스톤과 병렬
3. **Base 라이선스 확증 = M0 blocking**: Solar Apache variant 불확실 → 24시간 내 확증 또는 Bllossom 전환 (D6, §07.2)
4. 각 마일스톤은 exit gate를 가지며 미달 시 피벗 회의
5. 내부 실험은 KIOM 승인 여부와 무관 진행, **공개만 승인 후**

## 9.2 마일스톤 요약

| 월 | 이름 | 핵심 산출물 | Gate |
|---|---|---|---|
| M0 | Pre-kickoff | KIOM 메일, Solar 라이선스 확증, 전문가 LoI+계약, 1종 역공학, GPU 벤치 | — |
| M1 | 데이터 기반 | 파서 v0.1, Corpus v0.1, Eval v0 태스크 설계 (T1~T5) | — |
| M2 | Eval 완성 | **HanMed-Eval v0 200문항** (T5=100), Corpus v1, tokenizer data-driven, **contamination hash 훅** | **G2** |
| M3 | CPT pilot | 5% 데이터 pilot, **Wiki-ko replay / DUS LoRA ablation**, memory 실측 | — |
| M4 | CPT + SFT | **Full CPT 150M~250M tokens**, SFT, 중간 평가 | **G4** |
| M5 | 확장·ablation | (옵션) DPO, rank / mix / Stage 0 ablation, 비교군 eval | **G5** |
| M6 | 논문·릴리스 | 논문 draft, model/data card, 재현 레시피 공개 | — |

## 9.3 마일스톤 상세

### M0 — Week 0 (pre-kickoff, 1주)

**최우선 3줄 (순서대로)**:
1. [ ] **KIOM `kiombook@kiom.re.kr` 이메일 발송** — 연구 목적, 이용 범위, publication·weights 공개 계획, 자동 다운로드 1 req/sec 통지, 교감기록 범위 질문. **Critical path 시작점** (§07.1.3).
2. [ ] **Solar-10.7B-Instruct HF 페이지 라이선스 문자열 직접 확인** — Apache-2.0 variant 존재 여부 확증. **24시간 내 미확정 시 즉시 Bllossom-8B로 전환 결정**, 후속 config 템플릿도 Bllossom 기준으로 초기화 (D6, §07.2, R6).
3. [ ] **전문가 2인 LoI + 계약 템플릿 초안** — NDA, 보수 단가, 데이터 소유권, 평가 저작권, COI (§07.9).

**병렬 트랙** (위 3줄 진행 중 병행):

1. 라이선스 트랙
   - [ ] 소속기관 IRB 면제 신청
   - [ ] OpenAI/Anthropic API ToS 스냅샷 저장
   - [ ] DVC remote 위치 확정 (로컬 NFS / 기관 스토리지, §06.5)
2. 데이터 탐색 트랙
   - [ ] Playwright로 `info.mediclassics.kr/contents/database/list` 렌더 → `data/meta/books.csv`
   - [ ] 대표 서적 1종 (동의보감 내경편 권1) 수기 다운로드
   - [ ] markup 태그 인벤토리 → `markup_spec.md`
3. 인프라 트랙
   - [ ] HF 모델 캐시 (Solar + Bllossom 둘 다)
   - [ ] A6000 1장 bf16 LoRA **dummy 500 steps** → peak mem, tok/s, adapter param 수 기록
   - [ ] Llama-Factory vs torchtune 각 1 epoch → 프레임워크 확정 (ChatML template 지원 확인, §06.9)
   - [ ] `configs/prompt_template.py` 스켈레톤 작성 (ChatML, D9)
4. 인력 트랙
   - [ ] rubric 초안 공유: `eval/rubric/t1_translation.md`

### M1 — Month 1: 데이터 기반
- [ ] 파서 v0.1 — 정규식 기반, 대표 5종 통과
- [ ] 파서 검증 셋 50 레코드 수기 검수 → 정확도 보고
- [ ] Corpus v0.1 (처음 50종)
- [ ] HanMed-Eval v0 태스크 **설계만** (T1~T5 스키마, rubric, 프롬프트 템플릿)
- [ ] **T5 공공 벤치 확정** — **KLUE-YNAT 100문항 (stratified)** 로 ver2.1 확정. 선정 스크립트 `scripts/build_t5_klue_subset.py` 작성 (7 topic 균등, seed 고정, dev split)
- [ ] 보조 코퍼스 수집 시작 (Wiki-ko, CBETA)

### M2 — Month 2: 평가 먼저
- [ ] Parser v1, 161 중 가능한 전부 처리
- [ ] Corpus v1 동결 → `corpus_v1.json` 스냅샷
- [ ] **HanMed-Eval v0 200문항 큐레이션** — T1~T4 100 수기 + 2인 교차 검수, T5 100 자동 스크립트(KLUE-YNAT)
- [ ] Solar tokenizer 실측 → **data-driven Stage 0 결정** (median + 0.2 margin, D5)
- [ ] **Eval contamination hash filter 훅 구현·테스트** — `eval/hashes/heldout_*.txt` 생성, dummy 학습 셋으로 build-fail 경로 검증 (R15, §05.7, §06.5)
- [ ] HanMed unique token 수 실측 → A2 (32~43M) 범위 검증
- [ ] **G2 gate 판정**

### M3 — Month 3: CPT pilot

Pilot 목적: 하이퍼 검증, 메모리·throughput 실측, **두 개의 ablation 동시 수행**.

- [ ] Stage 1 CPT LoRA pilot (Corpus v1의 5% 샘플 + replay)
- [ ] **Ablation 1 — Wiki-ko replay 비중** (R1 대응, D4)
  - run A: 20% / run B: 30% / run C: 50%
  - 측정: val_loss, T5 general-ko Δaccuracy
  - 목표: E5(drop ≤ 3%p)를 충족하는 최소 비중 선정
- [ ] **Ablation 2 — DUS LoRA 공유 vs 독립** (R14, D7)
  - run D: PEFT 기본(복제 layer 독립) / run E: 복제 layer LoRA weight 공유
  - 측정: peak mem, adapter param 수, val_loss 동등성
  - 목표: 메모리 2× 회피 가능 여부 판정
- [ ] 결과 분석 → 믹스·rank·lr 1차 튜닝
- [ ] 최종 config 동결 (`configs/stage1_cpt.yaml`) — replay 비중, LoRA 공유 정책 반영

### M4 — Month 4: CPT + SFT 본 run
- [ ] Stage 1 CPT 본 run — **목표 150M~250M tokens total** (D3 cap). HanMed unique 32~43M × epoch 1.5~3 + 보조 믹스
- [ ] 중간 eval (step 50%, 100%) — T1 chrF, T5 Δaccuracy 포함
- [ ] Stage 2 SFT — seed + 합성(검수 ≥ 20%, rubric §07.4.2 / §07.9)
- [ ] HanMed-Eval v0 전 지표 측정 (T1~T5)
- [ ] 체크포인트 승격 — Stage 1 val_loss, **Stage 2 전문가 선호 승률 (lag 수용) + tentative best = T1 chrF** (§06.6 D11, ver2.1 역전)
- [ ] **G4 gate 판정** — E2 전문가 승률 / E3 T2 / E4 T4 refuse / **E5 T5 general-ko drop ≤ 3%p**

### M5 — Month 5: 확장·ablation
- [ ] (옵션) Stage 3 DPO — 전문가 선호 승률 primary (D11)
- [ ] Ablation:
  - LoRA rank 16 / 32 / 64
  - 믹스 비중 (HanMed 30/50/70%)
  - Stage 0 확장 유/무
  - Solar vs Bllossom (base 전환 경로가 살아있다면)
- [ ] 비교군 전체 평가 — ChatML 통일 (§05.9)
- [ ] Inter-rater agreement 확인
- [ ] **G5 gate 판정**

### M6 — Month 6: 논문·릴리스
- [ ] 논문 draft (SCI target)
- [ ] Model card + data card (T5 general-ko 결과 포함)
- [ ] **KIOM 승인 여부에 따라**:
  - 승인: `HanMed-Public-LoRA` adapter 공개 (HF)
  - 미승인: 논문만, 코드·재현 레시피만 공개
- [ ] 재현 스크립트 `scripts/train.sh` + README 최종화

## 9.4 Critical Path

```
M0 KIOM email ───────────────────────────────────────────────┐
M0 Solar license check (24h) ─→ [OK] Solar / [FAIL] Bllossom │
     │ (2~6 months background)                               │
     └─────────────────────────────────────→ M6 decision ▼
                                                 공개 여부
M0 data explore ─→ M1 parser ─→ M2 corpus+eval+hash ─→ M3 pilot+ablation ─→ M4 CPT+SFT ─→ M5 ablation ─→ M6 paper
                                       ▲
                                       │ evaluation first
M0 infra / expert LoI+계약 ─→ M1 섭외 완료 ┘
```

## 9.5 일정 리스크와 Contingency

| 상황 | Contingency |
|---|---|
| KIOM 회신 6개월 초과 | 논문 예정 제출, adapter 공개만 postpone |
| **Solar 라이선스 확증 실패** | **1주 내 Bllossom-8B 전환 완료** — HF 캐시, config 템플릿, dummy run, 라이선스 문서 갱신 (R6) |
| M2 gate 미달 (파서) | 스코프 → 동의보감 1종 심화 |
| M2 contamination 훅 실패 | 수기 hash 검사 → 빌드 fail 확인 후 진행 |
| M3 DUS LoRA 독립만 동작, 메모리 2× | grad ckpt on, batch size 1, seq 2048 유지 (R14) |
| M4 E5 미달 (general-ko drop > 3%p) | Wiki-ko replay 30→50% 상향, 재학습 1회 |
| M4 CPT 효과 미미 | 1회 재학습 → 2차 미달 시 피벗 회의 (§08.2.1) |
| M5 gate 미달 | 논문 방향 전환 (negative result + 벤치 기여) |
| GPU 충돌 | 임차 클러스터 (A100 ×8) 단기 임차 |
| 전문가 이탈 | v0 120→60 축소 |

## 9.6 주간 운영

- 매주 월: 주간 goal → `docs/09_roadmap/weekly/{YYYY-WW}.md`
- 매주 금: 결과 요약 + `docs/08_risks/risk_log.md` 업데이트
- 월말: 마일스톤 gate 판정 회의
