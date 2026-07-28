"""
Smoke test — bf16 LoRA CPT (causal LM, self-supervised) on bilingual corpus.

목적: 파이프라인 무결성 검증. 학습 효과 검증 아님.
- base 모델은 작은 것으로 빠르게 (Qwen2.5-0.5B)
- ~50 steps만 돌려서 loss 감소·메모리 안정성 확인

Usage:
  CUDA_VISIBLE_DEVICES=1 python src/training/smoke_cpt.py
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


def load_bilingual_jsonl(path: Path, max_records: int | None = None) -> Dataset:
    rows = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_records and i >= max_records:
                break
            r = json.loads(line)
            rows.append({"text": r["text"]})
    return Dataset.from_list(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/cpt/hanmed_bilingual.jsonl"))
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--output-dir", type=Path, default=Path("outputs/smoke_cpt"))
    ap.add_argument("--max-records", type=int, default=2000, help="smoke test 용 records")
    ap.add_argument("--max-steps", type=int, default=50)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--micro-bs", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("HanMed-LLM smoke CPT (causal LM, bf16 LoRA)")
    print("=" * 60)
    print(f"  base model:   {args.base_model}")
    print(f"  data:         {args.data}")
    print(f"  records used: {args.max_records}")
    print(f"  max steps:    {args.max_steps}")
    print(f"  seq len:      {args.seq_len}")
    print(f"  micro bs:     {args.micro_bs} × grad_accum {args.grad_accum} = {args.micro_bs * args.grad_accum}")
    print(f"  LoRA:         r={args.lora_r}, alpha={args.lora_alpha}")
    print(f"  GPU:          {torch.cuda.get_device_name(0)} (CUDA_VISIBLE_DEVICES masked)")
    print()

    # 1) Tokenizer
    print("[1/5] Loading tokenizer …")
    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # bilingual block special tokens
    new_specials = ["<ZH>", "</ZH>", "<KO>", "</KO>"]
    added = tok.add_special_tokens({"additional_special_tokens": new_specials})
    print(f"  added {added} special tokens (bilingual block tags)")

    # 2) Model + LoRA
    print("[2/5] Loading base model in bf16 …")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    if added > 0:
        model.resize_token_embeddings(len(tok))

    print("[3/5] Wrapping with LoRA …")
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    # 3) Data
    print("[4/5] Loading bilingual data …")
    raw = load_bilingual_jsonl(args.data, max_records=args.max_records)
    print(f"  loaded {len(raw)} records")

    def tokenize_fn(batch):
        return tok(
            batch["text"],
            truncation=True,
            max_length=args.seq_len,
            padding=False,
        )

    tokenized = raw.map(tokenize_fn, batched=True, remove_columns=["text"])
    print(f"  tokenized: avg len = {sum(len(x) for x in tokenized['input_ids']) / len(tokenized):.0f} tokens/record")

    collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)

    # 4) Trainer
    print("[5/5] Starting training …")
    targs = TrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.micro_bs,
        gradient_accumulation_steps=args.grad_accum,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        warmup_steps=10,
        bf16=True,
        logging_steps=5,
        save_steps=args.max_steps,
        save_total_limit=1,
        report_to="none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        weight_decay=0.0,
        remove_unused_columns=False,
        dataloader_num_workers=2,
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=tokenized,
        data_collator=collator,
    )

    t0 = time.time()
    result = trainer.train()
    elapsed = time.time() - t0

    print()
    print("=" * 60)
    print("Smoke test DONE")
    print("=" * 60)
    print(f"  steps:        {result.global_step}")
    print(f"  final loss:   {result.training_loss:.4f}")
    print(f"  elapsed:      {elapsed:.1f}s")
    print(f"  throughput:   {result.global_step * args.micro_bs * args.grad_accum / elapsed:.2f} samples/s")
    print(f"  peak GPU mem: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    # Save adapter
    adapter_dir = args.output_dir / "final_adapter"
    trainer.save_model(str(adapter_dir))
    tok.save_pretrained(str(adapter_dir))
    print(f"  adapter saved to {adapter_dir}")

    # Sanity: tokenize back special tokens
    test = "<ZH>東醫寶鑑</ZH>\n<KO>동의보감</KO>"
    enc = tok.encode(test)
    print(f"  sanity tokenize check: {len(enc)} tokens for '{test[:30]}...'")


if __name__ == "__main__":
    main()
