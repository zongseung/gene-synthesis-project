# HanMed-LLM Proposal (ver1)

한의학 고전 + 한국어 특화 LLM 도메인 adapter 구축 기획서.

- 작성일: 2026-04-16
- 버전: **ver1**
- 상태: 타당성 기획 (KIOM 라이선스 승인 대기 전)

## 한 줄 요약

Solar-10.7B-Instruct 위에 **bf16 LoRA로 CPT + SFT** 하여 한의학 고전 번역·독해·지식 QA를 수행하는 도메인 특화 모델을 구축한다. 데이터는 한국한의학연구원 **mediclassics.kr** (161종, 약 1,816만 자)를 중심으로 사용한다.

## 섹션 구성

| # | 섹션 | 파일 | 핵심 |
|---|---|---|---|
| 01 | Overview | [01_overview/overview.md](01_overview/overview.md) | 프로젝트 정의, 기술 요약, 성공 기준, 비목표 |
| 02 | Data Source | [02_data_source/data_verification.md](02_data_source/data_verification.md) | mediclassics 검증, 미확인 to-do, 코퍼스 규모 범위 |
| 03 | Data Pipeline | [03_data_pipeline/acquisition.md](03_data_pipeline/acquisition.md) | 획득·파싱·정제, 보조 코퍼스 라이선스 매트릭스 |
| 04 | Model Strategy | [04_model_strategy/base_model_and_training.md](04_model_strategy/base_model_and_training.md) | Solar-10.7B + bf16 LoRA, Stage 0/1/2/3, 토큰 예산 |
| 05 | Evaluation | [05_evaluation/hanmed_eval.md](05_evaluation/hanmed_eval.md) | HanMed-Eval v0 100문항, chrF 중심, 전문가 rubric |
| 06 | Infrastructure | [06_infrastructure/gpu_framework.md](06_infrastructure/gpu_framework.md) | GPU 예산, Llama-Factory, 재현성 정책 |
| 07 | License · Ethics | [07_license_ethics/license_ethics.md](07_license_ethics/license_ethics.md) | KIOM 현실 일정, base 라이선스, IRB, safety |
| 08 | Risks | [08_risks/risk_register.md](08_risks/risk_register.md) | Risk table, exit gates, assumption register |
| 09 | Roadmap | [09_roadmap/milestones.md](09_roadmap/milestones.md) | M0~M6, critical path, contingency |

## ver0 → ver1 변경 요약

| 항목 | v0 draft | **ver1** |
|---|---|---|
| Base 모델 | Qwen2.5-7B/14B | **Solar-10.7B-Instruct** (Korean-first) |
| Precision / 학습법 | 미정 | **bf16 LoRA** 확정 |
| CPT 토큰 예산 | 1~5B tokens (수학 모순) | **100M~300M tokens** (BLOCKER 해결) |
| Tokenizer 추정 | "한자 ≈ 1 token 가정" | **M2 실측 후 결정**, 범위로만 표기 |
| HanMed-Eval 규모 | 500문항 (인력 미추정) | **100문항**, v1 확장은 M5+ |
| 번역 지표 | BLEU / COMET / 인간 | **chrF + 인간 선호**, COMET은 참고만 |
| KIOM 일정 | 2~8주 | **2~6개월**, critical path |
| IRB · Exit · Data version | 누락 | **§07, §08, §06 추가** |
| Base 라이선스 | Qwen 14B 오기재 | base별 정확 기재 + 상속 원칙 명시 |
| 보조 코퍼스 cross-contamination | 미언급 | **Public/Internal adapter 이원 학습** |
| 로드맵 순서 | parser 먼저 → eval | **eval first 강제**, M2 gate에 반영 |

자세한 디스크리미네이터 지적과 대응은 `proposal_v0_draft.md` 참고.

## 다음 실행 항목 (M0 우선순위)

1. **KIOM `kiombook@kiom.re.kr` 메일 발송** — critical path
2. Playwright로 서적 목록 스냅샷
3. 대표 서적 1종 다운로드 + markup 역공학
4. Solar-10.7B A6000 bf16 LoRA dummy run (500 steps) → tok/s·peak mem 실측
5. 한의학 박사 2인 섭외 LoI

## 열린 결정 (필요 시 논의)

- Solar-10.7B Apache variant 사용 확정 vs Bllossom-8B 전환
- LoRA rank 16 / 32 / 64 — M3 pilot ablation
- CPT adapter merge vs stack (SFT 이후) — M4 ablation
- Stage 3 DPO 수행 여부 — M5 gate
- 한문 → 영어 번역 포함 여부 — v1 scope 밖, 데이터 있으면 ablation
- 모델 공개 범위 — weights / LoRA / API only

## 참고

- mediclassics.kr (데이터 1차 소스)
- info.mediclassics.kr/apps/dist-texts/ (배포 서비스 안내)
- info.mediclassics.kr/contents/database/list (서적 목록, JS 렌더)
- `proposal_v0_draft.md` (이전 버전 + 디스크리미네이터 피드백 원문)
