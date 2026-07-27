"""VARCO-VISION-2.0 (LLaVA-OneVision + Qwen3) QLoRA SFT 트레이너.

mm_{train,val}.jsonl (이미지 1장 = conversations 1쌍) 로 비전-언어 모델을 SFT 한다.
4-bit QLoRA (bitsandbytes) + LoRA(peft) 로 단일 A6000(48GB)에 14B 적재.

이미지 경로: jsonl의 "image"는 논리 경로(예 151/001_칡/xxx.jpg). --image_root 와 결합.
누락 이미지는 회색 placeholder로 대체하고 수를 로깅(스모크 테스트용).

사용:
  PYTHONPATH=src .venv/bin/python -m hanmed_mm.training.sft_mm_trainer \
      --model models/VARCO-VISION-2.0-14B \
      --train data/sft/mm_train.jsonl --val data/sft/mm_val.jsonl \
      --image_root /mnt/nas-rawtext/한의학/aihub_images \
      --out outputs/mm_sft_run1 --epochs 1 --bsz 2 --grad_accum 8
"""
from __future__ import annotations
import argparse, json, os

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (AutoProcessor, BitsAndBytesConfig,
                          LlavaOnevisionForConditionalGeneration,
                          Trainer, TrainingArguments)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

_PLACEHOLDER = Image.new("RGB", (384, 384), (128, 128, 128))


class MMSFTDataset(Dataset):
    def __init__(self, path, image_root, max_samples=None):
        self.rows = [json.loads(l) for l in open(path, encoding="utf-8")]
        if max_samples:
            self.rows = self.rows[:max_samples]
        self.image_root = image_root
        self.n_placeholder = 0

    def __len__(self):
        return len(self.rows)

    def _load_image(self, rel):
        fp = os.path.join(self.image_root, rel) if self.image_root else rel
        try:
            return Image.open(fp).convert("RGB")
        except Exception:
            self.n_placeholder += 1
            return _PLACEHOLDER

    def __getitem__(self, i):
        r = self.rows[i]
        human = r["conversations"][0]["value"].replace("<image>", "").strip()
        gpt = r["conversations"][1]["value"]
        return {
            "image": self._load_image(r["image"]),
            "messages": [
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": human}]},
                {"role": "assistant", "content": [{"type": "text", "text": gpt}]},
            ],
        }


class Collator:
    """full 대화와 prompt(생성프롬프트까지)를 각각 토큰화해 응답 토큰만 학습(label 마스킹)."""
    def __init__(self, processor):
        self.p = processor

    def __call__(self, batch):
        images = [b["image"] for b in batch]
        full_txt = [self.p.apply_chat_template(b["messages"], tokenize=False) for b in batch]
        prompt_txt = [self.p.apply_chat_template(b["messages"][:1], tokenize=False,
                                                 add_generation_prompt=True) for b in batch]
        enc = self.p(text=full_txt, images=images, return_tensors="pt",
                     padding=True)
        labels = enc["input_ids"].clone()
        # prompt 길이만큼 마스킹 (이미지 토큰 확장은 full/prompt 동일하므로 prefix 일치)
        for i, pt in enumerate(prompt_txt):
            plen = self.p(text=[pt], images=[images[i]], return_tensors="pt")["input_ids"].shape[1]
            labels[i, :plen] = -100
        labels[enc["input_ids"] == self.p.tokenizer.pad_token_id] = -100
        enc["labels"] = labels
        return enc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/VARCO-VISION-2.0-14B")
    ap.add_argument("--train", default="data/sft/mm_train.jsonl")
    ap.add_argument("--val", default="data/sft/mm_val.jsonl")
    ap.add_argument("--image_root", default="")
    ap.add_argument("--out", default="outputs/mm_sft_run1")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--bsz", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--max_samples", type=int, default=0, help="스모크 테스트용 샘플 제한")
    ap.add_argument("--max_steps", type=int, default=-1)
    args = ap.parse_args()

    processor = AutoProcessor.from_pretrained(args.model)
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # connector(projector)는 4-bit 양자화에서 제외 → bf16 유지해야 풀 파인튜닝 가능.
    # (양자화된 Linear는 modules_to_save로 학습 불가)
    # skip 키는 반드시 전체 모듈 경로("model.multi_modal_projector")여야 한다.
    # transformers의 should_convert_module은 re.match(문자열 시작 고정)로 매칭하므로
    # "multi_modal_projector"만 주면 "model.multi_modal_projector.*"와 안 맞아 skip이 무시됨.
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True,
                             llm_int8_skip_modules=["model.multi_modal_projector"])
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        args.model, quantization_config=bnb, torch_dtype=torch.bfloat16, device_map={"": 0})
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    # 비전인코더 동결(설계 §8.3), LLM=LoRA, connector=풀 파인튜닝(시각특징→언어 매핑 도메인 적응).
    lora = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_r * 2, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        modules_to_save=["multi_modal_projector"],   # connector 풀 학습+저장
        task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()   # connector params가 trainable로 잡히는지 확인
    model.config.use_cache = False

    ms = args.max_samples or None
    train_ds = MMSFTDataset(args.train, args.image_root, ms)
    val_ds = MMSFTDataset(args.val, args.image_root, ms)

    targs = TrainingArguments(
        output_dir=args.out, num_train_epochs=args.epochs, max_steps=args.max_steps,
        per_device_train_batch_size=args.bsz, per_device_eval_batch_size=args.bsz,
        gradient_accumulation_steps=args.grad_accum, learning_rate=args.lr,
        bf16=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=1, save_strategy="epoch", eval_strategy="no",
        warmup_ratio=0.03, lr_scheduler_type="cosine", report_to=[],
        remove_unused_columns=False, dataloader_num_workers=2, optim="paged_adamw_8bit")
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds,
                      data_collator=Collator(processor))
    trainer.train()
    trainer.save_model(args.out)
    processor.save_pretrained(args.out)
    print(f"\n완료 → {args.out}  (placeholder 이미지 {train_ds.n_placeholder}장)")


if __name__ == "__main__":
    main()
