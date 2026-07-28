# 01. Overview — HanMed-LLM ver2

- 작성일: 2026-04-16
- 버전: ver2 (doc-discriminator REJECT_AND_REGENERATE 피드백 반영)
- 이전: `../../01_overview/overview.md` (ver1)

## 1.0 ver1 → ver2 변경 요약

ver1은 세 가지 BLOCKER로 reject 되었다. ver2는 이들을 정면으로 해결한다.

| BLOCKER | ver1 상태 | ver2 해결 |
|---|---|---|
| **B1** Stage 1 objective 부재 | 9개 문서 전체에 "next-token prediction / causal LM / 자기지도학습" 단어 0건 | §04.5 서두에 박스로 명시, 본 §01.3 표에도 반영 |
| **B2** 병렬 데이터 포맷 미정 | "병렬 10%"만 기재, 포맷 결정 없음 | `<ZH>…</ZH>\n<KO>…</KO>` bilingual block concatenation 확정 (§04.5.2) |
| **B3** 토큰 예산 이중계상 | "26M~58M raw"에 영역 포함 + "병렬 10%"가 한문+국역 재사용 → 이중계상 | §02.5 재산정, HanMed unique = 32M~43M (영역 제외), CPT cap 150M~250M (§04.5.3) |

이외 경미한 변경: Solar Apache-2.0 variant를 "M0 검증 대기"로 약화, DUS LoRA 2× 메모리 리스크 명시.

## 1.1 프로젝트 정의

**HanMed-LLM**: 한의학 고전과 현대 한국어를 교차 이해하는 한국어·한자 도메인 특화 LLM. 최종 사용자가 한국어로 질의하고 한국어로 답을 받는 **문헌 연구 보조 도구**.

## 1.2 "Foundation"의 재정의

mediclassics.kr의 ~18.16M 한자 코퍼스는 from-scratch pretraining에 불충분하다. 본 기획의 실질 전략은 **한국어 특화 base LLM + bf16 LoRA 기반 continued pretraining(CPT) + SFT**이다. 논문·문서·모델 카드 전체에서 "Foundation"이라는 단어는 "한의학 downstream 태스크의 도메인 adapter 묶음"이라는 의미로만 사용한다.

## 1.3 기술 요약

| 축 | 선택 |
|---|---|
| Base (primary) | **Llama-3.1-Korean-Bllossom-8B** (R3.2 승격 — tokenizer 실측) |
| Base (backup 1) | Qwen2.5-7B-Instruct (한자 특화) |
| Base (backup 2) | Mistral-Nemo-Instruct (Apache 2.0) |
| Base 기각 | Solar-10.7B-Instruct — byte_fallback 53% (tokenizer 실측) |
| **Base 모델 라이선스** | Llama 3 Community (조건부 상업 가능) |
| Precision | **bf16** (RTX A6000 native, GradScaler 불필요) |
| Adaptation | **LoRA** (rank 32, target q/k/v/o + mlp) |
| **Stage 1 Objective** | **Causal LM next-token prediction (자기지도학습 / self-supervised)** |
| **병렬 데이터 포맷** | **Bilingual block concatenation** `<ZH>…</ZH>\n<KO>…</KO>` (§04.5.2 참조) |
| Stages | Stage 0 tokenizer ext(조건부) → Stage 1 CPT → Stage 2 SFT → (옵션) Stage 3 DPO |
| CPT 토큰 예산 | **150M ~ 250M tokens (cap)**, HanMed 1.5~3 epoch (§04.5.3) |
| Framework | Llama-Factory (1차), torchtune (대안) |
| GPU | RTX A6000 48GB × 1~2 (bf16 LoRA는 1장으로 충분, DUS LoRA 2× 리스크 M3 실측) |
| 데이터 | mediclassics.kr 161종 + Wiki-ko + (내부 옵션) CBETA 한문·AI Hub |

### 왜 Bllossom-8B primary 인가 (R3.2)

`scripts/tokenizer_compare.py` 실측 (Core 14 10K char 샘플 × 7 후보 tokenizer):

- **Bllossom-8B**: vocab 128K / 한문 tok/char 1.040 / 한글 0.745 / **byte_fallback 0%** → 전 후보 중 1위
- Solar-10.7B (기존 primary): vocab 32K / 한문 1.533 / 한글 1.254 / **byte_fallback 53%** → 한자 절반이 UTF-8 3-byte tokens 로 분해되어 semantic 학습 불가 → **기각**
- 같은 compute 에서 학습되는 meaningful tokens 이 Bllossom 이 Solar 대비 약 **2×**
- 8B → A6000 48GB 에 더 여유, throughput Solar 10.7B 대비 ~2×
- Llama 3 Community License (조건부 상업 가능)

### 왜 bf16 LoRA인가
- A6000 48GB에서 10.7B 모델 full fine-tune은 메모리 불가 (~100GB 필요).
- QLoRA(4-bit base)는 학습 속도·품질 소폭 열위, bf16 LoRA가 명확히 더 낫다.
- LoRA adapter는 수백 MB → 배포·버전관리·라이선스 분리에 유리.

## 1.4 산출물

1. **HanMed-Corpus v1**: 정제·정렬된 mediclassics 코퍼스 (JSONL, bilingual block 포함)
2. **HanMed-LoRA-CPT**: base 위 CPT adapter
3. **HanMed-LoRA-SFT**: 이어진 SFT adapter
4. **HanMed-Eval v0**: **200문항** 평가 벤치마크 (T1 번역 30 / T2 독해 30 / T3 지식 20 / T4 안전성 20 / T5 general-ko KLUE-YNAT 100)
5. **논문 1편 (SCI target)**: 기여 = (a) 병렬 한문-한국어 도메인 적응 레시피, (b) 한의학 평가 벤치, (c) bilingual block concatenation 형식의 효과 분석
6. (옵션) Model card / data card / adapter weights 공개 — **KIOM 승인 전제**

## 1.5 성공 기준 (high-level)

| 기준 | 목표 | 측정 |
|---|---|---|
| 고전 번역 품질 | base 대비 전문가 선호 승률 ≥ 55% | Likert 5점 × 2인 × 30샘플 (T1 held-out, §05.3.1) |
| QA 정확도 | base 대비 +10%p | HanMed-Eval v0 T2 |
| 안전성 | 임상 의사결정 유도 거부율 ≥ 99% | T4 redteam 프롬프트 |
| 재현성 | 단일 `scripts/train.sh` + config로 재현 | seed 고정, `corpus_v1.json` hash pin |

세부 exit criteria는 `ver2/05_evaluation/hanmed_eval.md` §5.6, `ver2/08_risks/risk_register.md` §8.2 참고.

## 1.6 비목표 (non-goals)

- 임상 진단/처방 의사결정 지원 도구 아님
- 환자 상담/실시간 처방 서비스 아님
- From-scratch pretraining 아님
- 한의학 외 일반 의학 coverage 목표 아님
- 한문 → 영어 번역은 v1 scope 밖 (데이터 있으면 ablation 언급 수준)
- 영역(英譯) 코퍼스는 v1 CPT scope에서 **제외** (별도 태스크로만 선택 사용, §02.5 참조)

## 1.7 문서 구조

| 섹션 | 내용 |
|---|---|
| 02 data_source | mediclassics.kr 검증, 코퍼스 규모 재산정 |
| 03 data_pipeline | 획득·파싱·정제·bilingual block 빌드·eval contamination 훅 |
| 04 model_strategy | base 선택, bf16 LoRA stage 설계, objective/포맷/예산 확정 |
| 05 evaluation | HanMed-Eval v0 태스크·지표·exit |
| 06 infrastructure | GPU, 프레임워크, 재현성 |
| 07 license_ethics | KIOM, 모델 라이선스, IRB, safety |
| 08 risks | risk register (R1 replay ablation, R14 DUS LoRA mem 2× 포함), exit gate |
| 09 roadmap | M0~M6 마일스톤, critical path |
