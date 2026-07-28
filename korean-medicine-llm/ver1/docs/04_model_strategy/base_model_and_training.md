# 04. Model Strategy — Base + bf16 LoRA

## 4.1 설계 원칙

1. **한국어 출력 품질이 1순위** — 최종 사용자 체험 기준.
2. **한자 토크나이징 비효율은 Stage 0에서 해결 가능**.
3. **bf16 LoRA**만 사용. full fine-tune 안 함. QLoRA 안 함.
4. **A6000 48GB 단일 GPU에서 돌아야 함**. DDP는 throughput 확장용.
5. **Evaluation first** — HanMed-Eval v0 완성 전에는 Stage 1 학습 시작 금지.

## 4.2 Base 모델 선정

### 4.2.1 후보 비교

| 후보 | 파라미터 | 한국어 | 한자 tok. | 라이선스 | 비고 |
|---|---|---|---|---|---|
| **Solar-10.7B-Instruct** | 10.7B | ★★★ | ★ | cc-by-nc-4.0 / Apache-2.0 variant | **Primary** |
| Llama-3.1-Korean-Bllossom-8B | 8B | ★★ | ★ | Llama 3 Community | **Backup** |
| EXAONE-3.5-7.8B | 7.8B | ★★★ | ★ | EXAONE AI Model License | 상업 제한 |
| Qwen2.5-14B-Instruct | 14B | ★★ | ★★★ | Qwen License (MAU 100M 제한) | 한국어 약 |

### 4.2.2 Primary 선택: **Solar-10.7B-Instruct-v1.0**

- 한국어 유창성 강함 (Upstage 자체 튜닝)
- 10.7B는 A6000 48GB bf16 LoRA에 여유롭게 맞음
- DUS(Depth-Up-Scaled) 구조 — Llama-2 기반, LoRA 호환성 검증됨
- Apache-2.0 variant 존재 (배포 시점 확인 필수, §07)

### 4.2.3 Backup: **Llama-3.1-Korean-Bllossom-8B**

- 1차 실패 시 또는 라이선스 문제 발생 시 전환
- 8B로 더 가벼움, throughput 2배

## 4.3 메모리 예산 (bf16 LoRA, A6000 48GB)

| 항목 | 크기 |
|---|---|
| base weights (bf16) | 21.4 GB |
| LoRA adapters (r=32, q/k/v/o/gate/up/down) | ~0.3 GB |
| gradients (LoRA only) | ~0.3 GB |
| AdamW states (LoRA only) | ~0.6 GB |
| activations (seq 2048, micro bs 2, grad ckpt on) | ~10~14 GB |
| CUDA context · workspace | ~2 GB |
| **합계 (예상)** | **~35 GB** → ✅ A6000 48GB에 여유 |

실측은 M3 pilot run에서 수행, 이 표 업데이트.

**DDP 2장 사용 시**: throughput 2배, 메모리 예산은 동일 (데이터 병렬).

## 4.4 Stage 0 — Tokenizer 확장 (조건부)

### 4.4.1 측정
1. mediclassics 샘플 100만 자에 대해 Solar tokenizer로 `tokens/char` 계산
2. 태스크별 breakdown:
   - 한자 평균 tokens/char
   - 한국어 평균 tokens/어절

### 4.4.2 확장 결정 규칙

| 조건 | 조치 |
|---|---|
| 한자 tokens/char ≤ 1.3 | 확장 **스킵** |
| 한자 tokens/char > 1.3 | 확장 실행 |

### 4.4.3 확장 절차 (실행 시)
1. 빈도 상위 한자 + 한의학 multi-char 용어 (§03.4.1 사전 활용) → **2,000 ~ 5,000** 토큰 추가
2. Embedding matrix 확장: 신규 행은 기존 한자 임베딩 평균으로 초기화 (warm init)
3. LM head tie-weights 유지
4. Stage 1 초기 100 steps는 embedding row만 warmup (나머지 frozen)

## 4.5 Stage 1 — Continued Pretraining (LoRA)

### 4.5.1 데이터 믹스 (v1)

| 소스 | 비중 | 이유 |
|---|---|---|
| HanMed 한문 원문 | 25% | 도메인 핵심 |
| HanMed 국역 | 15% | 번역 신호 |
| HanMed 한문↔국역 병렬 | 10% | 교차언어 정렬 |
| Wiki-ko replay | 30% | 일반 한국어 능력 유지 (재배포 OK) |
| CBETA 한문 (옵션, 내부만) | 20% | 한문 일반화 |

### 4.5.2 토큰 예산 (BLOCKER 해결 — v0 수학 모순 제거)

전제: HanMed raw token 범위는 26M ~ 58M (§02.5.4).

- HanMed 학습 목표: **2 ~ 3 epoch over HanMed 부분**
  - = 52M ~ 174M tokens HanMed 학습량
- 믹스 비중 HanMed 50% (25+15+10) → 총 학습량 = 2 × HanMed 학습량
  - = **104M ~ 348M tokens total**
- **목표 최대 학습량**: **약 200M ~ 300M tokens**

이 수치는 v0 draft의 "1~5B tokens"를 폐기한 결과다. 실측 후 M3 pilot에서 재조정.

### 4.5.3 하이퍼파라미터 (초안)

| 항목 | 값 | 비고 |
|---|---|---|
| LoRA rank | 32 | r 16/32/64 ablation 예정 |
| LoRA alpha | 64 | alpha = 2 × rank 관례 |
| target modules | q,k,v,o, gate_proj, up_proj, down_proj | 표준 구성 |
| LoRA dropout | 0.05 | |
| learning rate | 1e-4 | LoRA 표준 |
| scheduler | cosine with warmup 500 steps | |
| weight decay | 0.0 (LoRA) | |
| seq length | 2048 | |
| micro batch | 2 | A6000 기준 |
| grad accum | 16 | effective batch 32 |
| optimizer | AdamW (β1=0.9, β2=0.95, eps=1e-8) | |
| precision | **bf16** | no GradScaler |
| grad ckpt | on | 메모리 절감 |
| shuffle | full shuffle within shard | |

## 4.6 Stage 2 — SFT

### 4.6.1 데이터 구성
1. **Seed** (자동 생성 가능): HanMed 한문↔국역 병렬 → "다음 한문을 현대 한국어로 번역하시오" 인스트럭션 변환
2. **지식 태스크**: 본초 → 효능/성미/귀경, 처방 → 구성약재 (사전 기반)
3. **QA 증강**: GPT-4o/Claude로 해설 QA 증강 — 단, **원문에 없는 문장을 고전 인용으로 생성 금지** 프롬프트 제약
4. **human-in-the-loop 검수** ≥ 20% (증강 데이터)

### 4.6.2 합성 데이터 정책
- 합성 응답 원본(model, prompt, date)을 JSONL에 `synthesis_provenance` 필드로 기록
- 논문·model card에 합성 비율 명시

### 4.6.3 하이퍼파라미터

| 항목 | 값 |
|---|---|
| LoRA rank | 16 (SFT는 가볍게) |
| learning rate | 5e-5 |
| epochs | 2~3 |
| seq length | 2048 |
| micro batch | 2 |
| effective batch | 16 |

CPT adapter를 merge하지 않고 **두 adapter를 stack** (PEFT supports this), 또는 CPT adapter merge 후 SFT adapter 추가 — ablation으로 비교.

## 4.7 Stage 3 — DPO (옵션)

- 전문가 선호 쌍 50~100개 수집 (번역 품질 중심)
- DPO fine-tune with β=0.1
- 효과 미달 시 skip, 논문에 negative result로 보고

## 4.8 재현성 고정

| 항목 | 방법 |
|---|---|
| random seed | torch/numpy/random 모두 고정 (ENV `PYTHONHASHSEED=0`) |
| config snapshot | YAML 전체를 output dir에 복사 |
| git SHA | `scripts/train.sh` 시작 시 로그 |
| data stats | `corpus_v1.json` hash가 config에 포함 |
| 체크포인트 | adapter + optimizer state + trainer state |

## 4.9 열린 결정

1. LoRA rank 16 vs 32 vs 64 — M3 pilot ablation
2. Tokenizer extension 유무 — M2 실측 후
3. CPT adapter merge vs stack — M4 ablation
4. DPO 수행 여부 — M5 go/no-go
5. Solar vs Bllossom 비교 실험 규모 — M3 pilot 후 결정
