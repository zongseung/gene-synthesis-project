"""dongui_bogam SFT mainline trainer.

`--mode sft` (default) — LoRA SFT for HanMed-LLM (Bllossom 또는 Gemma preset).
`--mode cpt` — root `src/training/cpt_trainer.py` 로 위임 (레거시 CPT 경로).

ver6 부터 이 파일은 SFT 가 mainline 이므로 이름을 sft_trainer.py 로 사용.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[4]
EXP_ROOT = Path(__file__).resolve().parents[2]
TOKENIZER_PATH = ROOT / "data" / "tokenizer" / "hanmed_bllossom_ext"
DEFAULT_BASE = "MLP-KTLim/llama-3-Korean-Bllossom-8B"
DEFAULT_OUTPUT = EXP_ROOT / "outputs_ver5_book008_full"
DEFAULT_SFT_DATA = EXP_ROOT / "data" / "sft" / "book008_full_sft.jsonl"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.seed import set_global_seed  # noqa: E402

LORA_TARGET_MODULES_BLLOSSOM = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "embed_tokens",
    "lm_head",
]

# ver6 Gemma 전환: embed_tokens · lm_head 제외 (docs/ver6 §3.2.1).
LORA_TARGET_MODULES_GEMMA = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

# Gemma-3 chat template assistant turn opener. Bllossom (Llama-3 ChatML) 과 다름.
RESPONSE_TEMPLATE_BLLOSSOM = "<|start_header_id|>assistant<|end_header_id|>\n\n"
RESPONSE_TEMPLATE_GEMMA = "<start_of_turn>model\n"


@dataclass
class DispatchArgs:
    mode: str
    remainder: list[str]


def parse_dispatch(argv: Sequence[str] | None = None) -> DispatchArgs:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--mode", choices=["cpt", "sft"], default="cpt")
    known, remainder = parser.parse_known_args(argv)
    return DispatchArgs(mode=known.mode, remainder=remainder)


def parse_sft_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="dongui_bogam SFT trainer — LoRA SFT on Bllossom or Gemma.")
    p.add_argument("--mode", choices=["sft"], default="sft")
    p.add_argument("--base", default=DEFAULT_BASE)
    p.add_argument(
        "--preset",
        choices=["bllossom", "gemma"],
        default="bllossom",
        help="family preset — gemma 선택 시 target_modules · response_template · tokenizer 를 Gemma-3 기본값으로 설정.",
    )
    p.add_argument("--tokenizer", type=Path, default=None, help="omit → preset 기본. bllossom=hanmed_bllossom_ext, gemma=base.")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--sft-data", type=Path, default=DEFAULT_SFT_DATA)
    p.add_argument("--resume-adapter", type=Path, default=None)
    p.add_argument("--sft-val-split", type=float, default=0.15)
    p.add_argument("--sft-epochs", type=int, default=3)
    p.add_argument("--sft-lr", type=float, default=2e-5)
    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--micro-bs", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--max-seq-length", type=int, default=2048)
    p.add_argument("--grad-checkpointing", action=argparse.BooleanOptionalAction, default=True,
                   help="gradient checkpointing (DDP+peft hang 시 --no-grad-checkpointing)")
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--eval-strategy", choices=["no", "epoch"], default="epoch")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--resume", type=str, default=None)
    args = p.parse_args(argv)
    # preset → tokenizer / target_modules / response_template 기본값 주입
    if args.tokenizer is None:
        args.tokenizer = TOKENIZER_PATH if args.preset == "bllossom" else Path(args.base)
    if args.preset == "gemma":
        args.target_modules = LORA_TARGET_MODULES_GEMMA
        args.response_template = RESPONSE_TEMPLATE_GEMMA
    else:
        args.target_modules = LORA_TARGET_MODULES_BLLOSSOM
        args.response_template = RESPONSE_TEMPLATE_BLLOSSOM
    return args


def load_root_cpt_module():
    path = ROOT / "src" / "training" / "cpt_trainer.py"
    spec = importlib.util.spec_from_file_location("root_cpt_trainer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load root CPT trainer from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ensure_hash_seed() -> None:
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise RuntimeError("PYTHONHASHSEED must be '0' for deterministic experiment runs.")


def ensure_runtime_ready(dry_run: bool) -> None:
    if dry_run:
        return
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in the current environment; full 8B SFT training cannot start.")


def load_sft_dataset(path: Path, val_split: float, seed: int):
    from datasets import load_dataset

    if not path.exists():
        raise FileNotFoundError(f"SFT data not found: {path}")
    cache_dir = EXP_ROOT / ".cache" / "hf_datasets"
    cache_dir.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("json", data_files=str(path), split="train", cache_dir=str(cache_dir))
    if len(ds) < 2:
        return ds, ds
    test_size = max(min(val_split, 0.5), 1 / len(ds))
    split = ds.train_test_split(test_size=test_size, seed=seed)
    return split["train"], split["test"]


def build_model_and_tokenizer(args: argparse.Namespace):
    import torch
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(args.tokenizer))
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16)
    # Gemma 는 vocab 확장 없음. Bllossom 은 hanmed_bllossom_ext (4 token 확장) 때문에 resize 필요.
    if args.preset == "bllossom" and model.get_input_embeddings().num_embeddings != len(tok):
        model.resize_token_embeddings(len(tok), mean_resizing=False)

    if args.resume_adapter:
        model = PeftModel.from_pretrained(model, str(args.resume_adapter), is_trainable=True)
    else:
        lora_cfg = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=args.target_modules,
        )
        model = get_peft_model(model, lora_cfg)

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    return model, tok


def find_subsequence(seq: list[int], sub: list[int]) -> int | None:
    for idx in range(len(seq) - len(sub) + 1):
        if seq[idx : idx + len(sub)] == sub:
            return idx
    return None


def tokenize_messages(example: dict, tok, response_template_ids: list[int], max_seq_length: int) -> dict:
    text = tok.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)
    enc = tok(text, truncation=True, max_length=max_seq_length, padding=False, add_special_tokens=False)
    labels = list(enc["input_ids"])
    start = find_subsequence(labels, response_template_ids)
    if start is None:
        labels = [-100] * len(labels)
    else:
        cutoff = start + len(response_template_ids)
        labels[:cutoff] = [-100] * cutoff
    # Gemma-3 (multimodal) 는 학습 시 token_type_ids 필수 (0=text, 1=image).
    # text-only SFT 이므로 전부 0.
    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "labels": labels,
        "token_type_ids": [0] * len(enc["input_ids"]),
    }


class CompletionOnlyCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict]) -> dict:
        import torch

        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, attention_mask, labels, token_type_ids = [], [], [], []
        has_tti = "token_type_ids" in features[0]
        for feature in features:
            pad = max_len - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [self.pad_token_id] * pad)
            attention_mask.append(feature["attention_mask"] + [0] * pad)
            labels.append(feature["labels"] + [-100] * pad)
            if has_tti:
                token_type_ids.append(feature["token_type_ids"] + [0] * pad)
        batch = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
        if has_tti:
            batch["token_type_ids"] = torch.tensor(token_type_ids, dtype=torch.long)
        return batch


def make_training_args(args: argparse.Namespace):
    from transformers import TrainingArguments

    do_eval = args.eval_strategy != "no"
    return TrainingArguments(
        output_dir=str(args.output),
        per_device_train_batch_size=args.micro_bs,
        per_device_eval_batch_size=args.micro_bs,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.sft_epochs,
        learning_rate=args.sft_lr,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        bf16=True,
        gradient_checkpointing=args.grad_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=5,
        save_strategy="epoch",
        eval_strategy=args.eval_strategy,
        load_best_model_at_end=do_eval,
        metric_for_best_model="eval_loss" if do_eval else None,
        greater_is_better=False if do_eval else None,
        remove_unused_columns=False,
        report_to="none",
        save_total_limit=2,
        seed=args.seed,
        max_steps=args.max_steps,
        # DDP + peft LoRA: 일부 param 이 forward 에서만 쓰이고 backward gradient 없음 →
        # find_unused_parameters=True 로 all-reduce 시 skip.
        # (gradient_checkpointing=False 와 함께 쓰면 hang 없이 작동하는 조합)
        ddp_find_unused_parameters=True,
    )


def run_sft(args: argparse.Namespace) -> int:
    from transformers import Trainer

    ensure_hash_seed()
    ensure_runtime_ready(args.dry_run)
    set_global_seed(args.seed)

    train_ds, val_ds = load_sft_dataset(args.sft_data, args.sft_val_split, args.seed)
    model, tok = build_model_and_tokenizer(args)
    response_template_ids = tok(args.response_template, add_special_tokens=False).input_ids

    train_ds = train_ds.map(
        lambda ex: tokenize_messages(ex, tok, response_template_ids, args.max_seq_length),
        remove_columns=train_ds.column_names,
    )
    val_ds = val_ds.map(
        lambda ex: tokenize_messages(ex, tok, response_template_ids, args.max_seq_length),
        remove_columns=val_ds.column_names,
    )

    if args.dry_run:
        print(
            json.dumps(
                {
                    "mode": "sft",
                    "train_rows": len(train_ds),
                    "val_rows": len(val_ds),
                    "response_template_ids": response_template_ids,
                    "output": str(args.output),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    trainer = Trainer(
        model=model,
        args=make_training_args(args),
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tok,
        data_collator=CompletionOnlyCollator(tok.pad_token_id),
    )
    trainer.train(resume_from_checkpoint=args.resume)

    args.output.mkdir(parents=True, exist_ok=True)
    adapter_dir = args.output / "adapter"
    trainer.save_model(str(adapter_dir))
    tok.save_pretrained(str(adapter_dir))
    manifest = {
        "mode": "sft",
        "preset": args.preset,
        "base": args.base,
        "tokenizer": str(args.tokenizer),
        "sft_data": str(args.sft_data),
        "resume_adapter": str(args.resume_adapter) if args.resume_adapter else None,
        "train_rows": len(train_ds),
        "val_rows": len(val_ds),
        "epochs": args.sft_epochs,
        "lr": args.sft_lr,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "target_modules": args.target_modules,
        "response_template": args.response_template,
        "max_steps": args.max_steps,
    }
    with (args.output / "train_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(json.dumps({"saved_adapter": str(adapter_dir), **manifest}, ensure_ascii=False, indent=2))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    dispatch = parse_dispatch(argv)
    if dispatch.mode == "cpt":
        return load_root_cpt_module().main(dispatch.remainder)
    args = parse_sft_args(["--mode", "sft", *dispatch.remainder])
    return run_sft(args)


if __name__ == "__main__":
    raise SystemExit(main())
