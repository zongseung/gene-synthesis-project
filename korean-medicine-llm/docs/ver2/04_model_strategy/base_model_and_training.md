# 04. Model Strategy — Base + bf16 LoRA (ver2)

## 4.1 설계 원칙

1. **한국어 출력 품질이 1순위** — 최종 사용자 체험 기준.
2. **한자 토크나이징 비효율은 Stage 0에서 해결 가능**.
3. **Stage 1 CPT는 causal LM next-token prediction 기반 자기지도학습(self-supervised)**. 다른 objective(MLM, span corruption) 사용하지 않음.
4. **bf16 LoRA**만 사용. full fine-tune 안 함. QLoRA 안 함.
5. **A6000 48GB 단일 GPU에서 돌아야 함**. DDP는 throughput 확장용.
6. **Evaluation first** — HanMed-Eval v0 held-out set 확정 전에는 Stage 1 학습 시작 금지.

## 4.2 Base 모델 선정 — **R3.2 primary 전환 (2026-04-16)**

**전환 근거**: `scripts/tokenizer_compare.py` 실측 결과 (mediclassics 10K char × 7 후보):

| Model | vocab | 한문 tok/char | 한글 tok/char | byte_fallback | Core 14 unique |
|---|---|---|---|---|---|
| 🥇 **Bllossom-8B (Llama-3 ko)** | **128,256** | **1.040** | **0.745** | **0%** | **2.72M tokens** |
| 🥈 Qwen2.5-7B-Instruct | 152K | 1.047 | 0.823 | 0% | 2.88M |
| 🥉 Mistral-Nemo-Instruct | 131K | 1.194 | 0.690 | 0% | 2.79M |
| Qwen2.5-14B-Instruct | 152K | 1.047 | 0.823 | 0% | 2.88M |
| EXAONE-3.5-7.8B | 102K | 1.262 | 0.665 | 0% | 2.82M |
| ❌ **Solar-10.7B-Instruct (기존 primary)** | **32,000** | **1.533** | **1.254** | **53%** | 4.31M |
| ❌ DeepSeek-V2-Lite | 100K | 1.233 | 1.712 | 0% | 4.85M |

**Solar 기각 사유**:
- vocab 32K 가 한자·한글 커버 부족 → **UTF-8 byte fallback 53%** (한자 절반 이상이 `<0xE4><0xB9><0xBE>` 류 3-byte tokens 로 분해)
- 한자/한국어 모두 효율 최하 (한문 +47%, 한글 +68% 대비 Bllossom)
- 같은 compute 에서 학습되는 semantic tokens 이 Bllossom 대비 약 **절반**
- §4.4.2 data-driven extension 규칙 (`median - wiki ≥ 0.2`) 에 거의 확정적으로 trigger 되지만, 2,000~5,000 token 추가해도 Bllossom 대비 열위 유지 예상

### 4.2.1 후보 비교 (실측 반영)

| 후보 | 파라미터 | 한국어 | 한자 tok | 라이선스 | 비고 |
|---|---|---|---|---|---|
| **Llama-3.1-Korean-Bllossom-8B** | 8B | ★★★ | ★★★ | Llama 3 Community (조건부 상업 가능) | **Primary (R3.2 승격)** |
| **Qwen2.5-7B-Instruct** | 7B | ★★ | ★★★ | Qwen License (MAU 100M 제한) | **Backup 1 (한자 특화 백업)** |
| Mistral-Nemo-Instruct | 12B | ★★ | ★★ | Apache 2.0 | Backup 2 (multilingual, 안전한 라이선스) |
| ❌ Solar-10.7B-Instruct | 10.7B | ★★★ | ★ byte-fallback | cc-by-nc-4.0 | **기각** (tokenizer 비효율) |

### 4.2.2 Primary — Llama-3.1-Korean-Bllossom-8B

- 한국어 유창성 강함 (MLP-KTLim Llama-3 기반 한국어 tuning)
- **Tokenizer 효율 전 후보 중 1위** — byte_fallback 0%, 한자 1.040 tok/char, 한글 0.745 tok/char
- 8B → A6000 48GB bf16 LoRA 에 여유, throughput Solar 10.7B 대비 ~2×
- Llama 3 Community License — 조건부 상업 가능, 도메인 adapter 배포 제약 적음
- Llama-3 vocab 128K + 예약 slot (128K~128256) 바로 뒤에 `<ZH>/</ZH>/<KO>/</KO>` 4개 special token 깔끔하게 할당 (128256~128259)
- **§04a §A.3 tokenizer extension 은 Bllossom 에서는 거의 불필요 예상** (margin < 0.2)

### 4.2.3 Backup 1 — Qwen2.5-7B-Instruct

- Bllossom 과 tokenizer 효율 동급 (한문 1.047, 한글 0.823)
- vocab 152K — 한자 coverage 탁월
- 한국어 출력 품질은 Stage 2 SFT 에서 검증 필요
- Qwen License MAU 100M 제약 — 연구용으로는 문제 없으나 상업 배포 시 재검토

### 4.2.4 Backup 2 — Mistral-Nemo-Instruct (R3.2 신설)

- Apache 2.0 — 라이선스 가장 자유
- multilingual BPE vocab 131K, 한문 1.194 / 한글 0.690 (한글 최상위)
- 12B → A6000 에서 여유 감소, micro batch 축소 필요

### 4.2.5 기각 — Solar-10.7B-Instruct (R3.2 demote)

- 실측에서 tokenizer 효율 최하위 확정
- DUS 구조의 LoRA 2× 메모리 리스크 (R14) 와 무관하게, byte-level fallback 자체로 한의학 CPT 에 부적합
- Apache-2.0 variant 검증도 중단 (M0 최상단 과제 제거)

## 4.3 메모리 예산 (bf16 LoRA, A6000 48GB)

| 항목 | grad ckpt **on** | grad ckpt **off** |
|---|---|---|
| base weights (bf16, 10.7B) | 21.4 GB | 21.4 GB |
| LoRA adapters (r=32, q/k/v/o/gate/up/down) | ~0.3 GB | ~0.3 GB |
| gradients (LoRA only) | ~0.3 GB | ~0.3 GB |
| AdamW states (LoRA only) | ~0.6 GB | ~0.6 GB |
| activations (seq 2048, micro bs 2) | **~5 ~ 10 GB** | **~10 ~ 14 GB** |
| CUDA context · workspace | ~2 GB | ~2 GB |
| **합계 (추정)** | **~30 ~ 35 GB** | **~35 ~ 39 GB** |

A6000 48GB에 여유. 실측은 M3 pilot run에서 수행, 이 표 업데이트.

**DUS LoRA 리스크 주석 (ver2 신규)**: Solar의 DUS 구조는 layer를 복제·re-scale하므로 PEFT 기본값에서 복제 layer마다 **독립 LoRA adapter**가 생성될 수 있다. 이 경우 adapter 크기와 optimizer state가 **최대 2×**로 증가, 활성화에도 간접 영향이 있다. M3 pilot에서 실측 필수. 관련 리스크는 `ver2/08_risks/risk_register.md` **R14**로 등록 요청. 실측이 2×에 근접하면 (a) target module 축소, (b) 복제 layer에 LoRA 공유 패치, (c) Bllossom-8B로 전환 중 하나를 선택.

## 4.4 Stage 0 — Tokenizer 확장 (조건부, data-driven)

### 4.4.1 측정
1. mediclassics 샘플 **100만 자** (M2)에 대해 Solar/Bllossom tokenizer로 `tokens/char` 측정
2. 태스크별 breakdown:
   - 한자 평균 tokens/char (median, p90)
   - 한국어 평균 tokens/어절 (median)
3. 결과는 `data/stats/tokenizer_probe.json` 에 저장

### 4.4.2 확장 결정 규칙 (ver2, magic number 제거)

ver1의 "한자 tokens/char ≤ 1.3"는 **임의 값**이므로 폐기. 대신 **실측 median + 고정 margin** 방식:

| 조건 | 조치 |
|---|---|
| `median(tokens/char) < median_baseline + 0.2` | 확장 **스킵** |
| `median(tokens/char) ≥ median_baseline + 0.2` | 확장 실행 |

여기서 `median_baseline`은 동일 tokenizer가 일반 한국어 Wiki 샘플에서 보이는 `tokens/char` 중앙값이다. 즉 "도메인 코퍼스가 일반 한국어 대비 0.2 이상 더 조각나면 확장"이라는 상대 기준.

### 4.4.3 확장 절차 (실행 시)
1. 빈도 상위 한자 + 한의학 multi-char 용어 (§03.4.1 사전 활용) → **2,000 ~ 5,000** 토큰 추가
2. **Special token 동시 추가**: `<ZH>`, `</ZH>`, `<KO>`, `</KO>` (§4.5.2 병렬 포맷용)
3. Embedding matrix 확장: 신규 행은 기존 한자 임베딩 평균으로 초기화 (warm init); special token은 `</s>` 임베딩 평균으로 초기화
4. LM head tie-weights 유지
5. Stage 1 초기 100 steps는 embedding row만 warmup (나머지 frozen)

Tokenizer 확장이 **스킵**되더라도 `<ZH>`·`<KO>` special token 4개는 별도로 추가한다 (병렬 포맷 필수 조건).

## 4.5 Stage 1 — Continued Pretraining (LoRA)

> **Objective**: **Next-token prediction (causal language modeling). Self-supervised (자기지도학습).**  
> Loss는 모든 non-pad token에 대한 **cross-entropy**. Wiki-ko replay를 0.5× down-weight하는 옵션은 §4.5.4 ablation (R1)에서만 비교한다. MLM, span corruption, denoising 등 다른 objective는 사용하지 않는다 (decoder-only 아키텍처와 부적합).

### 4.5.1 데이터 믹스 (ver2 확정)

| 소스 | 비중 | 형태 | 공개 adapter |
|---|---|---|---|
| HanMed 한문 원문 | **25%** | plain causal LM | ✅ |
| HanMed 국역 | **10%** | plain causal LM | ✅ |
| HanMed 병렬 (bilingual block) | **5%** | §4.5.2 D2 포맷 | ✅ |
| Wiki-ko (재배포 가능) | **30%** | plain causal LM (replay) | ✅ |
| CBETA 한문 | **20%** | plain causal LM | ❌ (내부 adapter만) |
| 예비 한국어 일반 (AI Hub 등) | **10%** | plain causal LM | ❌ (내부 adapter만) |

HanMed 총합 = 25 + 10 + 5 = **40%**.

**R1 ablation**: Wiki-ko 비중을 **20% / 30% / 50%** 로 M3 pilot에서 스윕하여 도메인 획득 ↔ 일반 능력 보존 trade-off 측정.

### 4.5.2 병렬 데이터 포맷 — Bilingual Block Concatenation (ver2 신규, B2 해결)

한문↔국역 병렬 5%는 다음 **단일 포맷**으로만 학습에 주입한다. Instruction tuning 포맷이나 span corruption은 사용하지 않는다.

**블록 단위 포맷**:
```
<ZH>{한문 원문}</ZH>
<KO>{국역}</KO>

```

- 블록 내부 구분자: `\n` (ZH 닫기와 `<KO>` 사이)
- 블록 사이 구분자: `\n\n` (double newline)
- `<ZH>`, `<KO>`는 §4.4.3에서 추가한 **special token**

**예시 2블록 (2048 seq packing)**:
```
<ZH>東醫寶鑑者 我國醫學之大全也</ZH>
<KO>동의보감은 우리나라 의학의 집대성이다.</KO>

<ZH>人蔘 味甘微苦 性微溫 補元氣</ZH>
<KO>인삼은 맛이 달고 약간 쓰며 성질이 약간 따뜻하여 원기를 보한다.</KO>

```

**패킹 규칙**:
- Greedy pack **최대 2048 tokens** per sequence
- **블록 경계를 자르지 않는다** — 블록 전체가 한 sequence에 들어가지 못하면 다음 sequence로 이월
- Sequence 시작에 **BOS 1회** (Llama 표준), 끝에 **EOS 1회**
- BOS/EOS는 블록 내부에 들어가지 않음

**Loss 범위**:
- 전 구간 **causal LM cross-entropy**
- 태그 토큰(`<ZH>`, `</ZH>`, `<KO>`, `</KO>`)도 **loss에 포함** — 모델이 언어 전환 신호를 학습하도록 함
- Masking 없음 (instruction 스타일 masking과 혼동 금지)

**왜 이 형식인가 (3줄 정당화)**:
1. Decoder-only causal LM과 100% 호환 — 별도 objective 필요 없음
2. 블록 내부에서 한자→한국어 전환이 한 context window에 보장되므로 **교차언어 정렬 신호가 attention으로 자동 학습**됨
3. Instruction 포맷은 Stage 2 SFT에서 다시 도입되므로, Stage 1에서는 "순수 병렬" 신호만 주는 것이 CPT 목적과 부합

### 4.5.3 토큰 예산 (B3 해결 — 이중계상 폐기)

**폐기 선언**: ver1 draft의 "1 ~ 5B tokens" 및 수정본 "104M ~ 348M total"은 (a) 영역 포함, (b) 병렬 10% 이중계상의 결함이 있어 **ver2에서 폐기**한다.

**ver2 재산정 (from §02.5)**:
- HanMed unique (영역 제외) = **32M ~ 43M tokens** (`ver2/02_data_source/data_verification.md` §2.5.4)
- HanMed 목표 epoch: **1.5 ~ 3** → HanMed training tokens = **~48M ~ ~130M**
- HanMed 믹스 비중 **40%** → total training tokens = HanMed_tok / 0.40
  - 하한: 48M / 0.40 ≈ **120M**
  - 상한: 130M / 0.40 ≈ **325M**
- **CPT 예산 cap: 150M ~ 250M tokens** (고정 상한)
  - 하한 120M는 120M으로 내리지 않고 안전마진 포함 **150M**에서 시작
  - 상한 325M는 디스크·시간·평가 현실성 고려 **250M**에서 컷
  - 상한 도달 시 HanMed epoch을 재계산, 필요 시 2.5 epoch에서 중단

Agent B `ver2/09_roadmap/milestones.md` M4 타겟과 `ver2/08_risks/risk_register.md` A2 assumption은 이 **150M~250M / HanMed 32M~43M unique** 수치와 정확히 일치해야 한다.

### 4.5.4 Data packing & loss masking (ver2 신규)

- **Packing**: greedy pack up to **2048 tokens**
- **문서 경계**: 문서(plain LM) 사이에 **EOS** 삽입
- **Sequence 경계**: 각 sequence 시작에 **BOS 1회**
- **Loss masking 기본**: 전 구간 동일 가중 (pad 제외)
- **R1 option**: Wiki-ko replay에 한해 **0.5× down-weight** — M3 pilot에서 (a) uniform, (b) 0.5× Wiki-ko 두 조건만 비교
- **Eval contamination hook**: `ver2/03_data_pipeline/acquisition.md` §3.4.2 훅을 CPT 빌드 파이프라인이 **통과한 이후에만** 학습 시작. Drop 비율 > 0.5% 면 파이프라인 실패 처리 (`contamination_drop.json` 확인).

### 4.5.5 하이퍼파라미터 (초안)

| 항목 | 값 | 비고 |
|---|---|---|
| LoRA rank | 32 | r 16/32/64 ablation 예정 |
| LoRA alpha | 64 | alpha = 2 × rank 관례 |
| target modules | q,k,v,o, gate_proj, up_proj, down_proj | 표준 구성 (DUS 복제 layer 정책은 §4.9) |
| LoRA dropout | 0.05 | |
| learning rate | 1e-4 | LoRA 표준 |
| scheduler | cosine with warmup 500 steps | |
| weight decay | 0.0 (LoRA) | |
| seq length | 2048 | §4.5.4 packing |
| micro batch | 2 | A6000 기준 |
| grad accum | 16 | effective batch 32 |
| optimizer | AdamW (β1=0.9, β2=0.95, eps=1e-8) | |
| precision | **bf16** | no GradScaler |
| grad ckpt | on (§4.3) | |
| prompt format | **ChatML (Solar default)** — Stage 1은 no-system plain text, Stage 2에서 ChatML 활성화 | |
| shuffle | full shuffle within shard | |

## 4.6 Stage 2 — SFT

### 4.6.1 데이터 구성
1. **Seed** (자동 생성 가능): `hanmed_bilingual.jsonl` → "다음 한문을 현대 한국어로 번역하시오" 인스트럭션 변환
2. **지식 태스크**: 본초 → 효능/성미/귀경, 처방 → 구성약재 (사전 기반)
3. **QA 증강**: GPT-4o/Claude로 해설 QA 증강 — 단, **원문에 없는 문장을 고전 인용으로 생성 금지** 프롬프트 제약
4. **human-in-the-loop 검수** ≥ 20% (증강 데이터)

### 4.6.2 Prompt format
- **ChatML** (Solar default). `<|im_start|>system ... <|im_end|>` / `<|im_start|>user ... <|im_end|>` / `<|im_start|>assistant ...`
- Loss mask: user/system turn은 mask, assistant turn만 loss 계산 (표준 SFT)
- Bllossom backup 선택 시 Llama-3 chat template로 자동 전환

### 4.6.3 합성 데이터 정책
- 합성 응답 원본(model, prompt, date)을 JSONL에 `synthesis_provenance` 필드로 기록
- 논문·model card에 합성 비율 명시

### 4.6.4 하이퍼파라미터

| 항목 | 값 |
|---|---|
| LoRA rank | 16 (SFT는 가볍게) |
| learning rate | 5e-5 |
| epochs | 2~3 |
| seq length | 2048 |
| micro batch | 2 |
| effective batch | 16 |

**Adapter 조합 ablation**: CPT adapter를 (a) merge 후 SFT adapter 추가, (b) 두 adapter **stack** (PEFT `add_adapter`) 두 방식을 M4에서 비교.

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
| data hash | `corpus_v1.json` 의 `sha256_of_raw` 가 config에 **핀**으로 포함 |
| bilingual block 빌드 SHA | `hanmed_bilingual.jsonl` sha256 기록 |
| **eval contamination hook** | §4.5.4 — 실행 로그 `contamination_drop.json` 아티팩트로 저장 |
| 체크포인트 | adapter + optimizer state + trainer state |

## 4.9 열린 결정

1. LoRA rank 16 vs 32 vs 64 — M3 pilot ablation
2. Tokenizer extension 실행 여부 — M2 실측 후 (§4.4.2)
3. CPT adapter merge vs stack — M4 ablation (§4.6.4)
4. DPO 수행 여부 — M5 go/no-go
5. Solar vs Bllossom 비교 실험 규모 — M3 pilot 후 결정
6. **DUS LoRA 독립 vs 공유 정책** (R14) — Solar 선택 시 M3에서 adapter 크기/메모리 실측 후 (a) 독립 유지, (b) 복제 layer 공유 패치, (c) Bllossom 전환 중 선택
7. **Wiki-ko replay loss 가중** — M3 pilot에서 uniform vs 0.5× 비교 (R1)
