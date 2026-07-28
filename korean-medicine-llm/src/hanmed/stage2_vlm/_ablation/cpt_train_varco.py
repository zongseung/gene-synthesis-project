"""VARCO-VISION 한의학 CPT (Stage 1, 선택) — 텍스트 전용 packed corpus causal LM.

전략: claudedocs/training_strategy_hanmed.md §1단계.
- 베이스: models/VARCO-VISION-2.0-14B (LlavaOnevision: SigLIP 비전 + Qwen3 언어).
- VARCO 의 **언어모델 부분만** LoRA 로 가볍게 한의학 한국어에 적응시킨다.
  비전타워(vision_tower) · projector(multi_modal_projector) 는 동결 (멀티모달 능력 보존).
- 입력은 사전 packed 된 {"input_ids":[2048]} jsonl → next-token cross-entropy.
- rank-stabilized LoRA(r=64), bf16, 1 epoch, 작은 LR, early stopping(val perplexity).

LlavaOnevisionForConditionalGeneration 는 pixel_values 없이 input_ids 만 주면
내부 Qwen3 언어모델이 순수 causal LM 으로 동작한다 (이미지 토큰 미포함 packed text).
→ CPT 로 학습한 LoRA 어댑터가 이후 SFT 와 동일 아키텍처에서 호환된다.

사용:
  # dry-run (모델 로드 없이 데이터로더/콜레이트/스텝계산만)
  .venv/bin/python -m hanmed.stage2_vlm._ablation.cpt_train_varco \
      --config configs/cpt_varco.yaml --dry_run

  # smoke (모델 로드+forward+loss+LoRA 부착 확인, 2 step)
  .venv/bin/python -m hanmed.stage2_vlm._ablation.cpt_train_varco \
      --config configs/cpt_varco.yaml --max_steps 2 --max_records 64

  # 실제 학습 (2-GPU DDP)
  PYTHONHASHSEED=0 .venv/bin/torchrun --nproc_per_node=2 \
      -m hanmed.stage2_vlm._ablation.cpt_train_varco --config configs/cpt_varco.yaml
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import yaml

# 언어모델 LoRA 만 매칭 (vision_tower / multi_modal_projector 제외).
# peft 는 target_modules 가 str 이면 re.fullmatch 로 모듈명 전체를 매칭한다.
LM_LORA_REGEX = (
    r"model\.language_model\.layers\.\d+\."
    r"(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"
)


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VARCO-VISION 한의학 CPT (text-only packed).")
    p.add_argument("--config", required=True, help="configs/cpt_varco.yaml")
    p.add_argument("--dry_run", action="store_true",
                   help="모델 로드 없이 데이터로더/콜레이트/스텝계산까지만 검증.")
    p.add_argument("--max_steps", type=int, default=None, help="smoke 용 step 제한.")
    p.add_argument("--max_records", type=int, default=None, help="smoke 용 record 제한.")
    p.add_argument("--load_4bit", action="store_true", help="OOM 회피용 4bit 강제.")
    # config override
    p.add_argument("--output_dir", default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--micro_bs", type=int, default=None)
    return p.parse_args()


def init_distributed() -> tuple[int, int, int]:
    import torch

    if "LOCAL_RANK" in os.environ and not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        world = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local_rank)
        return world, rank, local_rank
    return 1, 0, 0


def build_dataset(data_path: str, seq_len: int, pad_token_id: int, val_split: float,
                  seed: int, max_records: int | None):
    """packed {"input_ids":[seq_len]} jsonl → labels(pad=-100)+attention_mask 부착."""
    from datasets import load_dataset

    ds = load_dataset("json", data_files=str(data_path), split="train")
    if max_records:
        ds = ds.select(range(min(max_records, len(ds))))

    def _attach(batch):
        out_ids, out_lab, out_att = [], [], []
        for ids in batch["input_ids"]:
            ids = list(ids)[:seq_len]
            labels = [t if t != pad_token_id else -100 for t in ids]
            attn = [0 if t == pad_token_id else 1 for t in ids]
            out_ids.append(ids)
            out_lab.append(labels)
            out_att.append(attn)
        return {"input_ids": out_ids, "labels": out_lab, "attention_mask": out_att}

    ds = ds.map(_attach, batched=True,
                remove_columns=[c for c in ds.column_names if c != "input_ids"],
                desc="attach labels")

    if len(ds) < 50:
        return ds, ds.select(range(min(2, len(ds))))
    split = ds.train_test_split(test_size=val_split, seed=seed)
    return split["train"], split["test"]


def build_model(cfg: dict, load_4bit: bool):
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import LlavaOnevisionForConditionalGeneration

    kwargs = dict(torch_dtype=torch.bfloat16)
    if load_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
            # 언어모델만 학습하므로 비전/projector 양자화 무관하나, projector 는
            # 학습 대상이 아니어도 일관성 위해 제외 불필요.
        )
        kwargs["device_map"] = {"": 0}
    else:
        kwargs["device_map"] = {"": int(os.environ.get("LOCAL_RANK", 0))}

    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        cfg["base_model"], **kwargs)

    if load_4bit:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lora_cfg = LoraConfig(
        r=cfg["lora_rank"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        use_rslora=cfg.get("use_rslora", True),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=LM_LORA_REGEX,   # 언어모델만 (vision/projector 동결)
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    return model


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    if args.lr is not None:
        cfg["lr"] = args.lr
    if args.micro_bs is not None:
        cfg["micro_bs"] = args.micro_bs
    load_4bit = args.load_4bit or cfg.get("load_4bit", False)

    if args.dry_run:
        world, rank, local_rank = 1, 0, 0
    else:
        world, rank, local_rank = init_distributed()

    def log(m):
        if rank == 0:
            print(m, flush=True)

    # pad_token_id: VARCO Qwen3 tokenizer
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    seq_len = cfg["seq_len"]
    micro_bs = cfg["micro_bs"]
    grad_accum = cfg["grad_accum"]
    eff_tok = micro_bs * grad_accum * seq_len * world

    log("=" * 72)
    log("VARCO-VISION 한의학 CPT (text-only packed causal LM)")
    log("=" * 72)
    log(f"  base         : {cfg['base_model']}")
    log(f"  data         : {cfg['data']}")
    log(f"  pad_token_id : {pad_id}")
    log(f"  seq_len      : {seq_len}  world={world} rank={rank}")
    log(f"  micro_bs×ga×world = {micro_bs}×{grad_accum}×{world} → {eff_tok:,} tok/step")
    log(f"  LoRA r={cfg['lora_rank']} α={cfg['lora_alpha']} rslora={cfg.get('use_rslora', True)}")
    log(f"  target regex : {LM_LORA_REGEX}")
    log(f"  lr={cfg['lr']} epochs={cfg['epochs']} load_4bit={load_4bit}")

    log("[1/3] Building dataset…")
    train_ds, val_ds = build_dataset(
        cfg["data"], seq_len, pad_id, cfg["val_split"], cfg["seed"], args.max_records)
    log(f"  train={len(train_ds)}  val={len(val_ds)}")
    # collate sanity
    from transformers import default_data_collator
    sample = default_data_collator([train_ds[0], train_ds[min(1, len(train_ds) - 1)]])
    log(f"  collate sanity: input_ids {tuple(sample['input_ids'].shape)}  "
        f"labels {tuple(sample['labels'].shape)}  "
        f"non-masked-labels={(sample['labels'] != -100).sum().item()}")

    if args.dry_run:
        log("[DRY-RUN] 데이터로더/콜레이트/스텝계산 완료 — 모델 로드 skip. exit 0.")
        return 0

    log("[2/3] Loading VARCO + LoRA…")
    model = build_model(cfg, load_4bit)

    from transformers import (EarlyStoppingCallback, Trainer, TrainingArguments,
                              default_data_collator)
    targs = TrainingArguments(
        output_dir=cfg["output_dir"],
        per_device_train_batch_size=micro_bs,
        per_device_eval_batch_size=micro_bs,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=cfg["epochs"],
        max_steps=args.max_steps if args.max_steps is not None else -1,
        learning_rate=cfg["lr"],
        weight_decay=cfg["weight_decay"],
        warmup_ratio=cfg["warmup_ratio"],
        lr_scheduler_type=cfg.get("lr_scheduler", "cosine"),
        bf16=True,
        gradient_checkpointing=cfg["grad_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=cfg["logging_steps"],
        save_strategy="steps", eval_strategy="steps",
        save_steps=cfg["save_steps"], eval_steps=cfg["eval_steps"],
        save_total_limit=cfg["save_total_limit"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss", greater_is_better=False,
        optim="paged_adamw_8bit" if load_4bit else "adamw_torch",
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=2,
        seed=cfg["seed"],
        ddp_find_unused_parameters=False,
    )
    callbacks = []
    if not args.max_steps:
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=cfg.get("early_stopping_patience", 2)))

    trainer = Trainer(
        model=model, args=targs,
        train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=default_data_collator, callbacks=callbacks)

    log("[3/3] Trainer.train()…")
    trainer.train()

    if rank == 0:
        out = Path(cfg["output_dir"]) / "adapter"
        out.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(out))
        tok.save_pretrained(str(out))
        with open(Path(cfg["output_dir"]) / "train_manifest.json", "w") as fh:
            json.dump({"stage": "cpt", **{k: cfg[k] for k in
                      ("base_model", "data", "lora_rank", "lora_alpha", "lr", "epochs")}},
                      fh, ensure_ascii=False, indent=2)
        log(f"✔ adapter saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
