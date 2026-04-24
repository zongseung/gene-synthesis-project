"""ver5 Phase B SFT trainer — dongui_bogam experiment.

data/sft/phaseB_qa_template_v1.jsonl (16쌍 MVP) 위에서
Base Bllossom-8B + fresh LoRA 를 학습.

TRL 미설치 환경 대응: HF Trainer + custom completion-only collator
(기획서 docs/ver5/04_trainer_spec.md §5 fallback 경로).

실행:
    cd experiments/dongui_bogam
    PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES=0 \\
        ../../.venv/bin/python scripts/train_sft.py \\
        --data data/sft/phaseB_qa_template_v1.jsonl \\
        --output ../../outputs/cpt_bllossom_ver5 \\
        --epochs 3 --lr 2e-5
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

# ──────────────── paths ────────────────
_SCRIPT = Path(__file__)
EXP_ROOT = _SCRIPT.parent.parent           # experiments/dongui_bogam
ROOT = EXP_ROOT.parents[1]                 # korean-medicine-llm

DEFAULT_BASE = "MLP-KTLim/llama-3-Korean-Bllossom-8B"
DEFAULT_TOKENIZER = ROOT / "data" / "tokenizer" / "hanmed_bllossom_ext"

LORA_TARGETS = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)

# Bllossom Llama-3 계열 response template — chat_template 기준
# "<|start_header_id|>assistant<|end_header_id|>\n\n" → [128006, 78191, 128007, 271]
RESPONSE_TEMPLATE_IDS = [128006, 78191, 128007, 271]


# ──────────────── dataset ────────────────

def load_sft_dataset(path: Path, tokenizer, max_length: int) -> list[dict]:
    """jsonl → tokenized samples with completion-only labels."""
    rows: list[dict] = []
    skipped = 0
    for line in path.open(encoding="utf-8"):
        ex = json.loads(line)
        msgs = ex["messages"]
        # full tokenize (system+user+assistant)
        text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False
        )
        enc = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding=False,
            add_special_tokens=False,
        )
        input_ids = enc["input_ids"]
        # completion-only: assistant 구간만 label, 나머지 -100
        labels = [-100] * len(input_ids)
        start = _find_subseq(input_ids, RESPONSE_TEMPLATE_IDS)
        if start is None:
            skipped += 1
            continue
        content_start = start + len(RESPONSE_TEMPLATE_IDS)
        for i in range(content_start, len(input_ids)):
            labels[i] = input_ids[i]
        rows.append({
            "input_ids": input_ids,
            "attention_mask": enc["attention_mask"],
            "labels": labels,
            "_id": ex["id"],
            "_category": ex["category"],
        })
    if skipped:
        print(f"[warn] {skipped} samples skipped (response template not found)")
    return rows


def _find_subseq(seq: list[int], sub: list[int]) -> int | None:
    n = len(sub)
    for i in range(len(seq) - n + 1):
        if seq[i:i + n] == sub:
            return i
    return None


@dataclass
class PadCollator:
    """right-pad input_ids/labels to the longest in batch."""
    pad_token_id: int
    label_pad: int = -100

    def __call__(self, features):
        max_len = max(len(f["input_ids"]) for f in features)
        out = {"input_ids": [], "attention_mask": [], "labels": []}
        for f in features:
            n = len(f["input_ids"])
            pad = max_len - n
            out["input_ids"].append(f["input_ids"] + [self.pad_token_id] * pad)
            out["attention_mask"].append(f["attention_mask"] + [0] * pad)
            out["labels"].append(f["labels"] + [self.label_pad] * pad)
        return {k: torch.tensor(v, dtype=torch.long) for k, v in out.items()}


class ListDataset(torch.utils.data.Dataset):
    def __init__(self, rows: list[dict]):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


# ──────────────── main ────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--base", default=DEFAULT_BASE)
    p.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--micro-bs", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--max-seq-length", type=int, default=2048)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--val-split", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--resume-adapter", type=Path, default=None,
                   help="기존 LoRA adapter 경로를 주면 fresh LoRA 대신 "
                        "PeftModel.from_pretrained 로 resume 학습 (v3→v4 용).")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    args.output.mkdir(parents=True, exist_ok=True)

    print(f"[train_sft] base={args.base}")
    print(f"[train_sft] tokenizer={args.tokenizer}")
    print(f"[train_sft] data={args.data}")
    print(f"[train_sft] output={args.output}")

    tok = AutoTokenizer.from_pretrained(str(args.tokenizer), trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    print(f"[train_sft] tokenizer vocab={len(tok)} pad_id={tok.pad_token_id}")

    # ── data ──
    all_rows = load_sft_dataset(args.data, tok, args.max_seq_length)
    print(f"[train_sft] loaded {len(all_rows)} samples")
    if not all_rows:
        raise SystemExit("no samples after tokenization")

    # shuffle + split
    import random
    rng = random.Random(args.seed)
    rng.shuffle(all_rows)
    n_val = max(1, int(len(all_rows) * args.val_split))
    val_rows = all_rows[:n_val]
    train_rows = all_rows[n_val:]
    print(f"[train_sft] train={len(train_rows)} val={len(val_rows)}")

    if args.dry_run:
        print("[train_sft] dry-run: shapes")
        for r in train_rows[:2]:
            print(f"  id={r['_id']} len={len(r['input_ids'])} labels_nonpad={sum(1 for x in r['labels'] if x != -100)}")
        return

    # ── model ──
    print("[train_sft] loading base model (bf16)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        torch_dtype=torch.bfloat16,
    )
    model.resize_token_embeddings(len(tok), mean_resizing=False)

    if args.resume_adapter:
        print(f"[train_sft] RESUME from adapter: {args.resume_adapter}")
        if not args.resume_adapter.exists():
            raise SystemExit(f"resume adapter not found: {args.resume_adapter}")
        model = PeftModel.from_pretrained(
            model,
            str(args.resume_adapter),
            is_trainable=True,
        )
        print("[train_sft] adapter loaded as trainable — LoRA delta 를 이어 학습")
    else:
        print("[train_sft] fresh LoRA adapter 생성")
        lora_cfg = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=list(LORA_TARGETS) + ["embed_tokens", "lm_head"],
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    # ── trainer ──
    collator = PadCollator(pad_token_id=tok.pad_token_id)
    train_ds = ListDataset(train_rows)
    val_ds = ListDataset(val_rows)

    # step 계산: len(train) / (bs × grad_accum) × epochs
    total_steps = max(
        1,
        (len(train_rows) // max(1, args.micro_bs * args.grad_accum)) * args.epochs,
    )
    print(f"[train_sft] approx total_steps={total_steps}")

    targs = TrainingArguments(
        output_dir=str(args.output),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.micro_bs,
        per_device_eval_batch_size=args.micro_bs,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=1,
        save_strategy="epoch",
        eval_strategy="epoch" if val_rows else "no",
        save_total_limit=2,
        report_to=[],
        seed=args.seed,
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
    )

    print("[train_sft] starting training ...")
    trainer.train()

    # adapter 저장
    adapter_dir = args.output / "adapter"
    print(f"[train_sft] saving adapter → {adapter_dir}")
    model.save_pretrained(str(adapter_dir))
    tok.save_pretrained(str(adapter_dir))
    # 메타
    (args.output / "train_meta.json").write_text(
        json.dumps({
            "base": args.base,
            "data": str(args.data),
            "epochs": args.epochs,
            "lr": args.lr,
            "train_samples": len(train_rows),
            "val_samples": len(val_rows),
            "lora_rank": args.lora_rank,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("[train_sft] done.")


if __name__ == "__main__":
    main()
