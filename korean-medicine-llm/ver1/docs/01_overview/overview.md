# 01. Overview — HanMed-LLM ver1

- 작성일: 2026-04-16
- 버전: ver1 (discriminator 피드백 반영 + bf16 LoRA 전략 확정)
- 이전: `../proposal_v0_draft.md`

## 1.1 프로젝트 정의

**HanMed-LLM**: 한의학 고전과 현대 한국어를 교차 이해하는 한국어·한자 도메인 특화 LLM. 최종 사용자가 한국어로 질의하고 한국어로 답을 받는 **문헌 연구 보조 도구**.

## 1.2 "Foundation"의 재정의

1,800만 자 규모 코퍼스는 from-scratch pretraining에 불충분하다. 본 기획의 실질 전략은 **한국어 특화 base LLM + bf16 LoRA 기반 continued pretraining(CPT) + SFT** 이다. 논문·문서·모델 카드 전체에서 "Foundation"이라는 단어는 "한의학 downstream 태스크의 도메인 adapter 묶음"이라는 의미로만 사용한다.

## 1.3 기술 요약

| 축 | 선택 |
|---|---|
| Base (primary) | **Solar-10.7B-Instruct-v1.0** (Upstage, DUS) |
| Base (backup) | Llama-3.1-Korean-Bllossom-8B |
| Precision | **bf16** (RTX A6000 native, GradScaler 불필요) |
| Adaptation | **LoRA** (rank 32, target q/k/v/o + mlp) |
| Stages | Stage 0 tokenizer ext(조건부) → Stage 1 CPT → Stage 2 SFT → (옵션) Stage 3 DPO |
| Framework | Llama-Factory (1차), torchtune (대안) |
| GPU | RTX A6000 48GB × 1~2 (bf16 LoRA는 1장으로 충분) |
| 데이터 | mediclassics.kr 161종 + Wiki-ko + (옵션) CBETA 한문 |

### 왜 한국어 특화 base인가
- 출력 언어가 한국어이므로 decoder-side 유창성이 최우선.
- 한자 토크나이징 비효율은 Stage 0 BPE extension으로 보완 가능.
- Qwen2.5(한자 강함, 한국어 보통) 대비 최종 사용자 경험이 더 좋다.

### 왜 bf16 LoRA인가
- A6000 48GB에서 10.7B 모델 full fine-tune은 메모리 불가 (~100GB 필요).
- QLoRA(4-bit base)는 학습 속도·품질 소폭 열위, bf16 LoRA가 명확히 더 낫다.
- LoRA adapter는 수백 MB → 배포·버전관리·라이선스 분리에 유리.

## 1.4 산출물

1. **HanMed-Corpus v1**: 정제·정렬된 mediclassics 코퍼스 (JSONL)
2. **HanMed-LoRA-CPT**: Solar-10.7B 위 CPT adapter
3. **HanMed-LoRA-SFT**: 이어진 SFT adapter
4. **HanMed-Eval v0**: 100문항 평가 벤치마크
5. **논문 1편 (SCI target)**: 기여 = (a) 병렬 한문-한국어 도메인 적응 레시피, (b) 한의학 평가 벤치
6. (옵션) Model card / data card / adapter weights 공개 — **KIOM 승인 전제**

## 1.5 성공 기준 (high-level)

| 기준 | 목표 | 측정 |
|---|---|---|
| 고전 번역 품질 | base 대비 전문가 선호 승률 ≥ 55% | Likert 5점 × 2인 × 100샘플 |
| QA 정확도 | base 대비 +10%p | HanMed-Eval v0 T2 |
| 안전성 | 임상 의사결정 유도 거부율 ≥ 99% | T4 redteam 프롬프트 |
| 재현성 | 단일 `scripts/train.sh` + config로 재현 | seed 고정, DVC 데이터 |

세부 exit criteria는 §05.6, §08.2 참고.

## 1.6 비목표 (non-goals)

- 임상 진단/처방 의사결정 지원 도구 아님
- 환자 상담/실시간 처방 서비스 아님
- From-scratch pretraining 아님
- 한의학 외 일반 의학 coverage 목표 아님
- 한문 → 영어 번역은 v1 scope 밖 (데이터 있으면 ablation 언급 수준)

## 1.7 문서 구조

| 섹션 | 내용 |
|---|---|
| 02 data_source | mediclassics.kr 검증, 코퍼스 규모 범위 |
| 03 data_pipeline | 획득·파싱·정제·보조 코퍼스 |
| 04 model_strategy | base 선택, bf16 LoRA stage 설계 |
| 05 evaluation | HanMed-Eval v0 태스크·지표·exit |
| 06 infrastructure | GPU, 프레임워크, 재현성 |
| 07 license_ethics | KIOM, 모델 라이선스, IRB, safety |
| 08 risks | risk register, exit gate |
| 09 roadmap | M0~M6 마일스톤, critical path |
