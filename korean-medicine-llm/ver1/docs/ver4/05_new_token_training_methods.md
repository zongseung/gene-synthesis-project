# 05. 신규 special token 학습 방법론 (DDP 호환)

> 작성: 2026-04-21 (ver4 §5.3 보강)
> 대상 배경: HanMed-LLM Stage 1 CPT, Bllossom-8B (Llama-3 계열, `tie_word_embeddings=False`),
> tokenizer 128,256 → 128,260 (특수 토큰 4개 추가), 2×A6000 DDP, PEFT 0.13.2.
> 현 문제: `LoraConfig(modules_to_save=["embed_tokens","lm_head"])` + DDP 조합이
> `_verify_params_across_processes` AllGather 에서 hang (PEFT Issue #899 계열).

---

## 1. 문제 정의 (요약)

1. `resize_token_embeddings(128260)` 로 `embed_tokens` / `lm_head` 에 4행 추가 (현재 코드: `mean_resizing=False` → 0-init).
2. 새 4토큰은 **`hanmed_bilingual` 코퍼스에서만** record 당 ~9회 등장 (전체 mix 의 45% 비중). 즉 학습 반드시 필요.
3. 현재 해결책 `modules_to_save=["embed_tokens","lm_head"]` 은 PEFT 의 `ModulesToSaveWrapper(original_module + modules_to_save.default)` 로 wrap → rank 간 `model.parameters()` 순회 불일치 → DDP init 시 param count AllGather 에서 무한 대기.
4. 필요 조건:
   - (a) 새 4토큰의 embedding / lm_head 행이 업데이트 될 것
   - (b) DDP (2 GPU) 호환
   - (c) 새 토큰의 표현력이 bilingual corpus 학습 품질을 지탱할 수 있을 것
   - (d) 구현 난이도 낮고 재현 가능할 것

---

## 2. 방법론 비교

각 항목: **설명 → 장단점 → 구현 난이도 → HanMed 적용 시 주의점 → 근거**.

### A. Trainable Tokens (PEFT ≥0.13 의 `trainable_token_indices`) — **권장 1순위**

**설명.** PEFT 가 공식 지원하는 vocab extension 전용 기능. `LoraConfig.trainable_token_indices={"embed_tokens":[128256,128257,128258,128259]}` 로 **지정한 token id의 행만** 학습 가능한 `TrainableTokensLayer` 로 감싼다. 원본 embedding 행 128,256개는 frozen, 새 4행만 업데이트. Tied weight 자동 처리 (Transformers convention 따를 때).

**장점.**
- 설계 의도(“새 4토큰만 학습”)에 **정확히** 일치.
- Trainable params 극소화 (4 × 4096 × 2 ≈ 32K ≪ 현재 1.05B).
- PEFT 공식 vocab-expansion 권장 1순위 경로 (troubleshooting 문서의 "Extending the vocabulary" 섹션).
- `ModulesToSaveWrapper` 사용 안 함 → DDP 호환성 이슈 회피.
- VRAM/디스크 저장 둘 다 감소 (gemma-2-2b 기준 full-FT 대비 ~4GiB 절약 사례).

**단점 / 주의.**
- PEFT 공식 문서상 **"FSDP/DeepSpeed 완전 지원 미보장"** 명시. DDP 에 대해선 명시적 불가 문구 없음 → 표준 DDP 에서는 동작이 합리적으로 기대되나, 우리 환경(PEFT 0.13.2)에서 **실행해 봐야 확정** (unverified for this exact stack).
- PEFT 0.16.0 에 `modules_to_save` 와 `trainable_token_indices` 가 같은 layer 에 걸리면 regression (Issue #2653). **우리는 0.13.2 + `modules_to_save` 제거 조합**이라 무관.
- Llama-3 8B 는 `tie_word_embeddings=False` (untied). 즉 `embed_tokens` 와 `lm_head` 는 **독립**으로 학습돼야 한다. PEFT 의 "tied weight 자동 처리"는 untied 에서는 작동 안 함 → **`trainable_token_indices` 에 `lm_head` 도 같이 지정 필요**.
- `init_weights=True` (default) 는 “현재 embedding 값으로 초기화 → 학습 전 no-op”. 우리 코드는 `mean_resizing=False` 로 이미 0-init → 새 행은 0에서 출발, LoRA 와 달리 **선형 델타가 아니라 행 자체가 직접 학습**되므로 표현력 제한 없음.

**구현 난이도.** 낮음 — LoraConfig 에 한 줄 추가, `modules_to_save` 제거.

```python
lora_config = LoraConfig(
    r=32, lora_alpha=64,
    target_modules=list(LORA_TARGET_MODULES),   # 7개 attn/MLP 그대로
    lora_dropout=0.05, bias="none",
    task_type=TaskType.CAUSAL_LM,
    trainable_token_indices={
        "embed_tokens": [128256, 128257, 128258, 128259],
        "lm_head":      [128256, 128257, 128258, 128259],   # untied 모델이므로 명시
    },
    # modules_to_save 제거
)
```

**근거.**
- PEFT Troubleshooting — "Extending the vocabulary" (trainable tokens 를 **most parameter-effective** 로 표기)
- PEFT Trainable Tokens 공식 reference
- PEFT Issue #2653 (0.16.0 regression, 0.13.2 영향 없음)
- Llama docs — `tie_word_embeddings` 기본 false for 8B

---

### B. LoRA on embed_tokens / lm_head (하이브리드) — **권장 2순위 (A fallback)**

**설명.** `target_modules = [..., "embed_tokens", "lm_head"]` 로 확장. PEFT ≥0.8 은 `nn.Embedding` / tied `lm_head` 에 LoRA delta 지원. `modules_to_save` 제거.

**장점.**
- `ModulesToSaveWrapper` 회피 → DDP 호환.
- 표현력 조정 가능 (rank 32 면 새 4행 × hidden 4096 = 저-rank 근사로 충분).
- PEFT 공식 2순위 권장.
- Chinese-LLaMA (Cui et al. 2023, arXiv:2304.08177) Stage 2 가 이 구조와 유사: "LoRA + embeddings, LM heads, newly added LoRA parameters are trained".

**단점 / 주의.**
- **기존 128,256행도 LoRA delta 적용 받음** → 원본 한국어 능력이 CPT 중에 일부 드리프트할 가능성. `rank=32` 면 영향은 제한적이지만 A 대비 원본 보존성 낮음.
- 저장/로딩 시 `save_embedding_layers` 플래그 관리 필요.
- 새 4토큰의 초기 표현이 LoRA delta (두 개 저-rank 행렬 곱) 로만 구성 → rank=32 가 0-init 행 위에 얹히는 표현력이 A (직접 행 업데이트) 대비 이론적으로 작음. 그러나 token 당 학습량(record 당 ~9회 × bilingual 45% 비중 × 6만 records) 고려하면 실측 차이는 작을 가능성 높음.

**구현 난이도.** 낮음.

**근거.**
- PEFT Troubleshooting — "Using an adapter, e.g. LoRA" 경로
- Chinese-LLaMA-Alpaca 논문 (arXiv:2304.08177) §3.2

---

### C. Full-FT + `modules_to_save` + **단일 GPU** (현 코드, DDP 포기) — 참고

**설명.** 기존 `modules_to_save=["embed_tokens","lm_head"]` 유지, DDP 대신 `nproc_per_node=1` 로 실행.

**장점.**
- 설계 문서 §5.3 (1) 에 적힌 "full-train" 의도 그대로.
- 새 4토큰 + 기존 128,256행 **둘 다** full 업데이트.

**단점 / 주의.**
- Wall-clock 대략 **2배** 증가 (현 구성 기준 ~1.5시간 → ~3시간).
- Trainable 1.13B, VRAM ~36GB (A6000 한 장 내).
- 새 4행 학습에 쏠림 없이 기존 vocab row 전체가 업데이트되어 **한국어 base 능력 드리프트 위험** (원본 Bllossom 성능 일부 손상 가능).
- "급할 때 우회" 용도 이외 장점 없음.

**구현 난이도.** 매우 낮음 (이미 구현됨, torchrun nproc=1 만 변경).

**근거.**
- 현 `cpt_trainer.py` 구현

---

### D. Full-FT + `modules_to_save` + FSDP / DeepSpeed ZeRO-3 — **비권장 (위험)**

**설명.** DDP 대신 FSDP 또는 DeepSpeed ZeRO-3 로 분산 전환.

**장점.**
- 이론상 param sharding 으로 `_verify_params_across_processes` 이슈 우회 가능.

**단점 / 주의.**
- PEFT 공식 문서: **"Passing the `modules_to_save` config parameter to PEFT is untested at present when using FSDP"** (명시적 경고).
- DeepSpeed ZeRO-3 는 `exclude_frozen_parameters` 관련 별도 패치 필요 (HF Transformers Issue #27874).
- 설정 복잡도 급증, 재현성 저하. 2-GPU 규모에는 과분.

**구현 난이도.** 중~상. Bllossom-8B 규모에 FSDP 이점 거의 없음.

**근거.**
- PEFT FSDP guide ("untested with modules_to_save")
- HF Transformers Issue #27874

---

### E. Chinese-LLaMA 2-Stage Training (Cui et al. 2023)

**설명.** Stage 1: transformer 전부 freeze, embedding/lm_head 만 학습. Stage 2: LoRA + embedding/lm_head trainable 같이.

**장점.**
- 새 토큰의 initial 표현을 안정적으로 정착시킨 뒤 전체 학습.
- 대규모 vocab 확장(20K token 추가) 시 실효성 입증 (Chinese-LLaMA-2).

**단점 / 주의.**
- HanMed-LLM 은 **4토큰 추가**라 Stage 분리의 이득 미미.
- Stage 1 이 별도 run → 스케줄/파이프라인 복잡도 증가.
- Stage 2 역시 `modules_to_save` 를 쓰면 같은 DDP 이슈 재발.

**구현 난이도.** 상 (2-stage 파이프라인).

**근거.** arXiv:2304.08177 §3.2.

---

### F. Gradient masking 기반 선택 학습 (GMT / Spectrum / TokenTune / TokenSeek) — 2024~2025

**설명.** `embed_tokens.weight.register_hook()` 또는 optimizer step 직전에 mask 적용해서 **특정 row 만 gradient 통과**시키는 저수준 custom. 넓게는 GMT (arXiv:2406.15330), Spectrum (HF 블로그), TokenTune (EMNLP 2024), TokenSeek (arXiv:2601.19739), TS-PEFT (arXiv:2511.16147) 이 동일 아이디어군.

**장점.**
- 완전한 통제 (“128256~128259 행만 gradient 받음”).
- DDP 호환 (PEFT wrap 없음).

**단점 / 주의.**
- 직접 구현: hook 등록 시점, optimizer 의 stateful update (AdamW 의 m/v) 도 masking 필요, FSDP 전환 시 호환성 재점검 필요.
- 위 논문들은 **모두 다른 문제**(전체 모델의 token-level informativeness 선별 등) 를 푼다. "신규 vocab row 학습"에 딱 맞는 논문은 없음 (정확히는 PEFT trainable_tokens 가 그 자리를 차지).
- PEFT A 가 같은 효과를 **지원 하에** 달성하므로 custom 채택 이유 약함.

**구현 난이도.** 중~상.

**근거.** 위 arXiv 들. 단 직접적 vocab-expansion fit 은 부족. 참고용.

---

### G. Embedding-only PEFT (IA³, Prompt Tuning 등) — **부적합**

**설명.** IA³ 는 key/value/ffn activation 에 scalar rescaling, Prompt Tuning 은 prefix token 삽입. embedding matrix 행을 직접 업데이트하지 않음.

**단점.**
- 새 토큰의 0-init 행은 그대로 남아 의미 없는 표현 유지.
- HanMed 문제 해결 불가.

**결론.** 이 방향은 우리 문제에 **맞지 않음**. 목록 완전성을 위해만 기재.

---

## 3. HanMed-LLM 권장 순위

| 순위 | 방법 | 이유 요약 |
|---|---|---|
| **1** | **A. Trainable Tokens** | 설계 의도(“새 4토큰만 학습”) 정확 일치, DDP 호환 (PEFT wrap 이슈 없음), 구현 1줄, 원본 128256행 완전 보존, PEFT 공식 권장 1순위 |
| 2 | B. LoRA on embed/lm_head | A 실행 중 이슈 발견 시 fallback. Chinese-LLaMA precedent. 기존 행도 미세 드리프트 허용 시 채택 |
| 3 | C. 단일 GPU | 의도 완전 유지해야 하고 A/B 둘 다 실패할 때. 시간 2배 |
| 4 | E. Chinese-LLaMA 2-stage | 새 토큰 수가 수천 이상으로 늘어날 때만 검토 |
| 5 | F. Gradient masking | A 가 없었다면 이것이 1순위였을 것. 지금은 대체재 |
| 제외 | D. FSDP + modules_to_save | PEFT 공식 untested. 2-GPU 규모에 과투자 |
| 제외 | G. IA³ / Prompt Tuning | 문제와 맞지 않음 |

### 실행 계획 (A 채택 기준)

1. `cpt_trainer.py` `build_model()` 의 `LoraConfig` 수정:
   - `modules_to_save` 제거.
   - `trainable_token_indices={"embed_tokens":[128256..128259], "lm_head":[128256..128259]}` 추가.
   - `resize_token_embeddings(..., mean_resizing=False)` 유지 (0-init 으로 학습 의존).
2. `TrainingArguments.ddp_find_unused_parameters=False` 로 원복 가능 (unused param 발생 안 함).
3. 첫 step 로그까지 관찰. VRAM 예상치 큰 폭 감소 (~18GB base + activation + AdamW (90M trainable 기준 ~1GB)) → 여유 충분.
4. 학습 수렴 후 새 4토큰 embedding norm 변화 로그 출력해 실제 업데이트 여부 검증 (예: `embed_tokens.weight[128256:128260].norm()` 비교).

### 검증 (A 가 맞다는 보증)

- A 가 우리 스택(peft 0.13.2, torch 2.5.1+cu121, accelerate 1.13.0, transformers 5.5.4)에서 DDP 로 돈다는 직접 실측은 아직 없음 (`unverified for this exact stack`).
- 1회 DDP 테스트 실행으로 (1) `[4/4] Starting Trainer.train()…` 이후 **첫 logging step 로그가 찍히는지** (2) VRAM 이 90M trainable 반영해 작아졌는지 확인 → 둘 다 OK 면 검증 종료.
- 실패 시 B 로 pivot.

### 품질 회귀 감시 포인트

- 기존 방식 대비 A 는 원본 128,256행이 frozen → base 한국어 능력 **완전 보존**이 오히려 이득일 가능성 (CPT 목적이 한의학 도메인 얹기이지 base 자체 개선이 아니므로).
- 평가(`eval/hanmed_eval_v0/`) 에서 domain-specific quality 가 기존 full-FT 대비 저하되면 B 로 재조정.

---

## 4. 참고 문헌 / 출처

### PEFT 공식
- [PEFT Troubleshooting — "Extending the vocabulary"](https://huggingface.co/docs/peft/en/developer_guides/troubleshooting) (trainable_tokens / LoRA / full-FT 세 경로 비교)
- [PEFT Trainable Tokens 공식 reference](https://huggingface.co/docs/peft/en/package_reference/trainable_tokens) (`TrainableTokensConfig`, FSDP 미지원 경고)
- [PEFT LoRA dev guide](https://huggingface.co/docs/peft/main/en/developer_guides/lora) ("Efficiently train tokens alongside LoRA")

### PEFT 관련 이슈
- [Issue #899 — modules_to_save + DDP + grad ckpt 비호환](https://github.com/huggingface/peft/issues/899) (ver4 §5.3 DDP hang 의 근거)
- [Issue #2653 — trainable_token_indices 0.16.0 regression](https://github.com/huggingface/peft/issues/2653) (0.13.2 영향 없음)
- [Issue #1750 — tied weight embed/lm_head 처리 질의](https://github.com/huggingface/peft/issues/1750) (Llama-3 8B 는 untied 이므로 영향 없음)
- [HF Transformers Issue #27874 — PEFT + DeepSpeed ZeRO-3 state dict](https://github.com/huggingface/transformers/issues/27874)

### 논문
- Cui et al. 2023, **"Efficient and Effective Text Encoding for Chinese LLaMA and Alpaca"** — arXiv:2304.08177 (vocab extension 2-stage 정석)
- Liu et al. 2024, **"DoRA: Weight-Decomposed Low-Rank Adaptation"** — arXiv:2402.09353 (참고, embedding 적용 여부 unverified)
- Hayou et al. 2024, **"LoRA+: Efficient Low Rank Adaptation of Large Models"** — arXiv:2402.12354 (LR 분리; embedding 적용 unverified)
- Kopiczko et al. 2024 (ICLR), **"VeRA: Vector-based Random Matrix Adaptation"** — arXiv:2310.11454 (공유 random matrix; embedding 대상 아님)
- GMT — Liao et al. 2024, **"Gradient-Mask Tuning Elevates the Upper Limits of LLM Performance"** — arXiv:2406.15330 (gradient masking 일반)
- TS-PEFT 2025 — arXiv:2511.16147 (Token-Selective PEFT, 신규)
- Spectrum (HF 블로그) — 선택적 fine-tuning 개관

### 모델 / 토크나이저
- [MLP-KTLim/llama-3-Korean-Bllossom-8B](https://huggingface.co/MLP-KTLim/llama-3-Korean-Bllossom-8B) — base
- [Llama docs (tie_word_embeddings default)](https://huggingface.co/docs/transformers/main/model_doc/llama)

### 분산 학습 경로
- [PEFT FSDP guide — modules_to_save untested 경고](https://huggingface.co/docs/peft/en/accelerate/fsdp)
- [PEFT DeepSpeed guide](https://github.com/huggingface/peft/blob/main/docs/source/accelerate/deepspeed.md)

---

## 5. 메모

- 본 문서는 "이전 작동했던 git 코드로 되돌리자"는 즉시 대응의 정식 대체안을 정리한 것. 이전 코드(3b6eff3, `modules_to_save` 없음)로 돌릴 경우 DDP 는 복구되지만 **새 4토큰이 0-init 상태로 학습 안 되는 문제가 잠복**. A 채택 시 이 함정도 해소된다.
- A 실행 후 새 토큰 norm 변화 로깅은 §11.2 W5 재현성 manifest 에 포함 권장.
- `transformers==5.5.4` 는 본 저장소 설치 기록 기준 값(내부 개발 버전 가능성). PEFT 0.13.2 와의 완전한 호환은 실행 시 확인 필요 (`unverified`).
