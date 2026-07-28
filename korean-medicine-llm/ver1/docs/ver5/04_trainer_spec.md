# ver5 · 04. Trainer 구현 스펙

- **버전**: ver5 r0 (2026-04-23)
- **대상 파일**: `src/training/cpt_trainer.py` 확장 (`--mode sft` 분기)
- **선결 상태**: ver4/09 리뷰어가 지적한 **TRL 미설치 + SFT flag 전무** 상태. 본 문서가 구체 스펙.

---

## 0. 한 줄 요약

**`cpt_trainer.py` 를 `--mode {cpt,sft}` 로 분기 확장하되 두 분기의 공통 경로 (model/adapter 로드, LoRA config, TrainingArguments 골격) 를 재사용하고, SFT 분기에서는 Base Bllossom 위에 fresh LoRA adapter 를 붙여 TRL `SFTTrainer + DataCollatorForCompletionOnlyLM` 을 사용한다. `--resume-adapter` 는 본선이 아니라 선택 ablation 용으로만 남기며, response_template 은 Bllossom chat_template 의 `<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n` boundary 를 명시 지정한다.**

---

## 1. 의존성 선결

### 1.1 TRL 설치 여부 확인

```bash
.venv/bin/python -c "import trl; print(trl.__version__)"
# → 미설치 시 ModuleNotFoundError
```

### 1.2 설치 명령

```bash
# transformers 5.x 호환 TRL 버전 확인
.venv/bin/pip install 'trl>=0.11.0,<0.13.0'
# 또는 구체 버전 핀
.venv/bin/pip install trl==0.11.4
```

### 1.3 호환성 매트릭스 (sanity 필수)

| 패키지 | 현재 버전 | ver5 필요 | 비고 |
|--------|----------|----------|------|
| transformers | (`pip show transformers`) | ≥ 4.45 (chat_template support) | 대개 OK |
| peft | 0.13.2 | 0.13.x | `is_trainable=True` API 있음 |
| trl | **미설치** | **0.11.x** | `SFTTrainer + SFTConfig` |
| datasets | (확인) | ≥ 2.19 | messages 컬럼 지원 |
| torch | 2.x | ≥ 2.1 | bf16 |

**호환성 실패 시 fallback**: TRL 제거 후 `src/training/sft_collator.py` 자체 구현 (response template boundary masking). `§5 fallback` 참조.

## 2. `cpt_trainer.py` 확장 스펙

### 2.1 argparse 신규 플래그

```python
# 기존 plus:
p.add_argument("--mode", choices=["cpt", "sft"], default="cpt",
               help="cpt (기존) / sft (ver5).")
p.add_argument("--resume-adapter", type=str, default=None,
               help="선택 ablation: 기존 adapter 경로를 이어학습할 때만 사용")
p.add_argument("--sft-data", type=str, default=None,
               help="SFT jsonl 경로 (mode=sft 필수)")
p.add_argument("--sft-val-split", type=float, default=0.15,
               help="SFT train/val split 비율")
p.add_argument("--sft-epochs", type=int, default=3,
               help="SFT num_train_epochs")
p.add_argument("--sft-lr", type=float, default=2e-5,
               help="SFT learning rate (CPT lr 보다 낮게)")
```

### 2.2 모드별 분기

```python
def main():
    args = parse_args()
    if args.mode == "cpt":
        run_cpt(args)       # 기존 경로
    elif args.mode == "sft":
        run_sft(args)       # 신규
```

### 2.3 `run_sft()` 구현 스켈레톤

```python
def run_sft(args):
    # 1. Base + tokenizer 로드 (기존 build_model 재사용)
    tok = load_tokenizer(args.tokenizer)
    model = load_base_model(args.base, dtype=torch.bfloat16)
    model.resize_token_embeddings(len(tok), mean_resizing=False)  # 128260

    # 2. 본선은 fresh LoRA, resume-adapter 는 선택 ablation
    if args.resume_adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(
            model, args.resume_adapter, is_trainable=True
        )
    else:
        # fresh LoRA
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, lora_cfg)

    # 3. SFT dataset 로드
    from datasets import load_dataset
    ds = load_dataset("json", data_files=args.sft_data, split="train")
    # {"id", "category", "messages": [...], ...}
    split = ds.train_test_split(test_size=args.sft_val_split, seed=args.seed)
    train_ds, val_ds = split["train"], split["test"]

    # 4. Response template 명시 (Bllossom chat_template 기준)
    response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    # DataCollator 는 string 대신 token id 시퀀스 사용 권장 (tokenizer 의존)
    response_template_ids = tok(response_template, add_special_tokens=False).input_ids

    from trl import SFTTrainer, SFTConfig
    from trl.trainer.utils import DataCollatorForCompletionOnlyLM
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template_ids,
        tokenizer=tok,
    )

    # 5. SFTConfig
    sft_config = SFTConfig(
        output_dir=args.output,
        per_device_train_batch_size=args.micro_bs,        # 1 or 2
        per_device_eval_batch_size=args.micro_bs,
        gradient_accumulation_steps=args.grad_accum,      # 8 or 16
        num_train_epochs=args.sft_epochs,                 # 3
        learning_rate=args.sft_lr,                        # 2e-5
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=5,
        save_steps=50,
        eval_steps=50,
        evaluation_strategy="steps",
        metric_for_best_model="eval_loss",
        load_best_model_at_end=True,
        greater_is_better=False,
        max_seq_length=2048,
        packing=False,                                    # SFT 는 packing 금지
        dataset_kwargs={
            "skip_prepare_dataset": False,
            "add_special_tokens": False,  # double BOS 방지
        },
        seed=args.seed,
        report_to=["wandb"] if get_rank() == 0 else "none",
    )

    # 6. SFTTrainer
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tok,
        data_collator=collator,
    )

    # 7. 학습
    trainer.train()

    # 8. 저장
    model.save_pretrained(Path(args.output) / "adapter")
    tok.save_pretrained(Path(args.output) / "adapter")
```

### 2.4 Dataset format 요구

`--sft-data` jsonl 의 각 line:

```json
{
  "id": "SFT-IN-author-v1",
  "category": "in_scope",
  "messages": [
    {"role": "system", "content": "당신은 한의학 고전..."},
    {"role": "user", "content": "동의보감 저자는?"},
    {"role": "assistant", "content": "허준(許浚)이 편찬... [출처: ...]"}
  ]
}
```

SFTTrainer 가 `messages` 를 `tokenizer.apply_chat_template` 로 자동 렌더. `dataset_text_field` 는 미지정 (messages 우선).

## 3. Response Template 명시 — 중요

### 3.1 왜 id 시퀀스 사용?

- `DataCollatorForCompletionOnlyLM` 의 `response_template` 을 **문자열** 로 주면 tokenizer 가 경계를 tokenize 할 때 surrounding token 영향으로 정확한 match 실패 빈번
- **token id 시퀀스** 로 주면 exact match 보장

### 3.2 Bllossom chat_template 확인

```bash
.venv/bin/python -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('data/tokenizer/hanmed_bllossom_ext', trust_remote_code=True)
template = '<|start_header_id|>assistant<|end_header_id|>\n\n'
ids = tok(template, add_special_tokens=False).input_ids
print('template:', repr(template))
print('ids:', ids)
print('decoded back:', repr(tok.decode(ids)))
"
```

예상 출력 (Llama-3 계열 기준):
```
template: '<|start_header_id|>assistant<|end_header_id|>\n\n'
ids: [128006, 78191, 128007, 271]   # (정확한 id 는 확인 필요)
decoded back: '<|start_header_id|>assistant<|end_header_id|>\n\n'
```

### 3.3 경계 mismatch 경고

`SFTTrainer` 가 학습 시점에 다음 경고를 낼 수 있음:
```
UserWarning: Could not find response key `[128006, 78191, 128007, 271]` in the following instance: ...
```

발생 시 해결:
1. `add_special_tokens=False` 확인
2. `tok.apply_chat_template(messages, add_generation_prompt=False)` 로 pre-render 했을 때 실제 id 시퀀스 비교
3. `instruction_template` 도 명시 (`<|start_header_id|>user<|end_header_id|>\n\n` id 시퀀스)

## 4. Hyperparameter 선정 근거

| Param | 값 | 근거 |
|-------|----|------|
| `num_train_epochs` | 3 | LIMA 논문 § 4 (3~5 권장), 200쌍 규모 적합 |
| `learning_rate` | 2e-5 | HuatuoGPT-II SFT stage 참고, Phase A' CPT 1e-4 보다 5× 낮게 |
| `lr_scheduler_type` | cosine | warm-up 후 안정 수렴 |
| `warmup_ratio` | 0.05 | 200쌍 × 3 epoch = ~38 step, warmup 2 step |
| `per_device_train_batch_size` | 1 or 2 | A6000 48GB + bf16 + LoRA + grad ckpt = 2 가능 |
| `gradient_accumulation_steps` | 8 | effective batch 16 (small, SFT 적합) |
| `max_seq_length` | 2048 | messages 렌더 후 최대 길이 감안, 대부분 <1500 |
| `packing` | False | SFT 는 sample boundary 보존 필수 |
| `evaluation_strategy` | steps | eval_steps=50 으로 2~3회 측정 |
| `metric_for_best_model` | eval_loss | SFT holdout loss 기준 best checkpoint 선택 |

### 4.1 Effective training step 계산

- train 170쌍 (15% val split) × 3 epoch = 510 samples
- effective batch 16 → **~32 steps total**
- logging_steps=5 → 6회 log
- save_steps=50 → 1회 save (마지막) = best checkpoint

규모 작지만 LIMA 와 유사한 profile. 문제는 **small dataset 에서 eval_loss 변동성** 크므로 `§05 evaluation` 의 probe 검수가 최종 판정.

## 5. Fallback — TRL 없이 직접 구현

TRL 설치 실패 시 자체 collator:

```python
# src/training/sft_collator.py (신규)
from transformers import DataCollatorForLanguageModeling
import torch

class CompletionOnlyCollator(DataCollatorForLanguageModeling):
    """response_template 이후 토큰만 loss 계산."""
    def __init__(self, tokenizer, response_template_ids, **kwargs):
        super().__init__(tokenizer=tokenizer, mlm=False, **kwargs)
        self.response_template_ids = response_template_ids

    def torch_call(self, examples):
        batch = super().torch_call(examples)
        # labels 에서 response_template 이전은 -100 (loss 제외)
        for i, input_ids in enumerate(batch["input_ids"]):
            # find response_template_ids subsequence
            start = _find_subsequence(input_ids.tolist(), self.response_template_ids)
            if start is None:
                batch["labels"][i] = torch.tensor([-100] * len(input_ids))
            else:
                # response 시작 전까지 mask
                batch["labels"][i, :start + len(self.response_template_ids)] = -100
        return batch

def _find_subsequence(seq, sub):
    for i in range(len(seq) - len(sub) + 1):
        if seq[i:i+len(sub)] == sub:
            return i
    return None
```

**train dataset 전처리** (`tokenizer.apply_chat_template` 로 pre-tokenize):

```python
def render_sft_sample(example, tokenizer):
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    # tokenize
    enc = tokenizer(text, truncation=True, max_length=2048, padding=False,
                    add_special_tokens=False)
    enc["labels"] = enc["input_ids"].copy()
    return enc

train_ds = train_ds.map(lambda x: render_sft_sample(x, tok), remove_columns=["messages"])
```

이 방식은 TRL 의존 없지만 구현 · 검증 공수 크므로 **1순위는 TRL 설치**.

## 6. Dry-run 검증 (Mini sanity)

### 6.1 Mini SFT 20쌍으로 분기 동작 확인

```bash
# 20쌍 mini 생성 (옵션 A 의 in_scope_basic 만)
PYTHONHASHSEED=0 .venv/bin/python scripts/build_sft_qa.py \
  --seeds data/sft/phaseB_qa_seeds.yaml \
  --mode template \
  --categories in_scope_basic \
  --limit 20 \
  --out data/sft/sanity_20.jsonl

# cpt_trainer.py --mode sft 실행 (dry-run)
PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES=0 \
  .venv/bin/python -m src.training.cpt_trainer \
    --mode sft \
    --output outputs/sft_sanity_20 \
    --sft-data data/sft/sanity_20.jsonl \
    --sft-epochs 2 \
    --sft-lr 2e-5 \
    --micro-bs 1 --grad-accum 8 \
    --seed 42 --dry-run
```

### 6.2 Dry-run 통과 확인 포인트

- [ ] argparse 에러 없음
- [ ] TRL import OK
- [ ] apply_chat_template 렌더 결과 길이 정상
- [ ] response_template_ids 매칭 성공 (warning 없음)
- [ ] DataCollator 가 label masking 정상 (user 부분 -100, assistant 부분 token id)
- [ ] Step 1 이 GPU 상에서 성공 (OOM 없음)
- [ ] eval_loss 계산 가능

### 6.3 실패 패턴별 대응

| 패턴 | 원인 | 대응 |
|------|------|------|
| `ModuleNotFoundError: trl` | TRL 미설치 | `.venv/bin/pip install trl==0.11.4` 또는 `§5 fallback` |
| `Could not find response key` | response_template tokenize mismatch | id 시퀀스 재확인, `add_special_tokens=False` |
| `CUDA OOM` | micro_bs 2 + grad_ckpt 조합 | micro_bs 1 로 |
| `is_trainable=True` AttributeError | PEFT 버전 | peft 0.13.2 확인 |
| NCCL hang (DDP 재시도 시) | §05 B 안 LoRA on embed/lm_head 이슈 | single GPU 로 |

## 7. DDP 호환성

- **Phase A' 에서 DDP 2-GPU 실패 이력** (NCCL timeout, `target_modules` 에 embed_tokens 포함 시)
- ver5 SFT 도 **single GPU 권장** (micro_bs 1~2 × grad_accum 8 = effective 8~16)
- 향후 DDP 시도 시 `05_new_token_training_methods.md` A 안 (`trainable_token_indices`) 재검토 필요

## 8. Resume 정책

- ver5 본선은 **resume 없이 fresh LoRA adapter** 로 시작
- `--resume-adapter` 는 Phase A' 또는 기타 legacy adapter 를 비교군으로 측정할 때만 사용
- ver5 SFT 결과는 **독립 adapter** 로 저장: `outputs/cpt_bllossom_ver5/adapter/`
- Merged model: `outputs/hanmed_merged_ver5/`
- 기존 `outputs/cpt_bllossom_phaseA/` 는 보존 (ablation 비교용)

## 9. 이 스펙의 한계

- TRL 0.11.x API 는 minor release 마다 변경 사항 있음. **설치 후 SFTConfig · SFTTrainer signature 재확인 필수**
- `response_template` ids 는 실제 Bllossom tokenizer 기준으로 최종 확인 (§3.2 명령 실행)
- DataCollatorForCompletionOnlyLM 가 PEFT wrapper 모델과 완전 호환되는지 mini SFT 까지 **실측 필요**
- 본 문서는 코드 골격만 제공. 실제 `cpt_trainer.py` diff 는 §6 dry-run 후 별도 PR 로.
