"""VARCO-VISION 한의학 멀티모달 SFT (Stage 2, 메인).

전략: claudedocs/training_strategy_hanmed.md §2단계 + §3 환각방지.
- 설진(tongue, 문헌인용+abstain) + 약초(herb, 식별/독초) 통합 SFT.
- **균형 샘플링**: 설진:약초 ≈ 1:9.2 불균형 → WeightedRandomSampler 로 설진 노출을
  목표 비율(기본 0.34)까지 끌어올림. 단순 5~10배 복제 금지(§3 환각 경고).
- **비전인코더 동결**, LLM=LoRA(rank64, rank-stabilized), projector 풀학습
  (modules_to_save) — 단계 단순화(Prismatic single-stage, projector 워밍업 생략).
- epoch 1~3, 작은 LR, early stopping(val loss == perplexity 모니터).

데이터 포맷(공통): {"image": 논리경로, "conversations":[{from:human,value:"<image>\n.."},
                  {from:gpt,value:".."}], ...}
이미지 해석:
  - 설진: image="shezhenv3/{split}/X.jpg" → {tongue_image_root}/{split}/images/X.jpg
  - 약초: image="151/001_칡/x.jpg"        → {herb_image_root}/151/001_칡/x.jpg
누락 이미지는 회색 placeholder 로 대체하고 개수를 로깅(smoke/샤딩 미완 대비).

사용:
  # dry-run (모델 없이 데이터셋/샘플러/콜레이트만)
  PYTHONPATH=src .venv/bin/python -m hanmed_mm.training.sft_train_varco \
      --config configs/sft_varco.yaml --dry_run

  # smoke (모델 로드+forward+loss+LoRA, 2 step, 극소 샘플)
  PYTHONPATH=src .venv/bin/python -m hanmed_mm.training.sft_train_varco \
      --config configs/sft_varco.yaml --max_steps 2 --max_samples 16

  # 실제 학습 (단일 A6000; 14B bf16 + LoRA + grad ckpt)
  PYTHONPATH=src .venv/bin/python -m hanmed_mm.training.sft_train_varco \
      --config configs/sft_varco.yaml
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
from pathlib import Path

import yaml

# 언어모델 LoRA 만 (vision_tower 동결). projector 는 modules_to_save 로 풀학습.
LM_LORA_REGEX = (
    r"model\.language_model\.layers\.\d+\."
    r"(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"
)


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VARCO-VISION 한의학 멀티모달 SFT.")
    p.add_argument("--config", required=True)
    p.add_argument("--dry_run", action="store_true",
                   help="모델 없이 데이터셋/균형샘플러/콜레이트까지만 검증.")
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--max_samples", type=int, default=None,
                   help="smoke 용 각 소스 record 제한.")
    p.add_argument("--load_4bit", action="store_true")
    p.add_argument("--output_dir", default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
# 데이터셋
# ---------------------------------------------------------------------------
def _resolve_tongue(rel: str, root: str) -> str:
    # "shezhenv3/train/A (1000).jpg" → {root}/train/images/A (1000).jpg
    parts = rel.split("/")
    if len(parts) >= 3:
        split, fname = parts[1], parts[-1]
        return os.path.join(root, split, "images", fname)
    return os.path.join(root, rel)


def _resolve_herb(rel: str, root: str) -> str:
    return os.path.join(root, rel) if root else rel


def subsample_val(rows: list, cap: int) -> list:
    """검증셋을 소스 비율 유지한 채 cap 행으로 줄인다. 결정론적.

    eval 은 체크포인트를 고르려고 도는 것이라 전량이 필요 없다. 2차 val 6,085행을
    다 돌리면 1회 63분 × 9회 = 9.5시간으로, 학습 47~62시간의 15~20%가 여기 쓰인다.
    """
    if len(rows) <= cap:
        return rows
    frac = cap / len(rows)
    by_src = collections.OrderedDict()
    for r in rows:
        by_src.setdefault(r["_src"], []).append(r)
    keep = []
    for src, grp in by_src.items():
        grp = sorted(grp, key=lambda r: hashlib.md5(
            json.dumps(r.get("conversations"), ensure_ascii=False,
                       sort_keys=True).encode()).hexdigest())
        keep.extend(grp[:max(1, round(len(grp) * frac))])
    return keep


class UnifiedMMDataset:
    """설진+약초 통합. 각 row 에 source('tongue'|'herb') 태그를 달아 균형 샘플링에 사용."""

    def __init__(self, cfg: dict, split: str, max_samples: int | None = None):
        from PIL import Image
        self._Image = Image
        self._placeholder = Image.new("RGB", (384, 384), (128, 128, 128))
        self.n_placeholder = 0
        # 약초 이미지: tar 샤드 인덱스가 설정되면 풀린 폴더 대신 샤드에서 직접 읽음
        self._herb_shard_index = cfg.get("herb_shard_index")
        self._herb_shard_dir = cfg.get("herb_shard_dir", "data/shards")
        self._shard_reader = None  # 워커별 lazy 생성(fork 안전)

        # 미설정 소스는 건너뛴다 — 1차(text 만)와 2차(tongue+herb+text replay)가 같은 코드를 쓴다.
        suffix = "train" if split == "train" else "val"
        sources = [(cfg[f"{src}_{suffix}"], src, cfg.get(f"{src}_image_root", ""))
                   for src in ("tongue", "herb", "text") if cfg.get(f"{src}_{suffix}")]
        self.rows = []
        for path, src, root in sources:
            cnt = 0
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    r = json.loads(line)
                    r["_src"] = src
                    r["_root"] = root
                    self.rows.append(r)
                    cnt += 1
                    if max_samples and cnt >= max_samples:
                        break
        if split != "train" and cfg.get("val_max_rows"):
            self.rows = subsample_val(self.rows, cfg["val_max_rows"])
        self.sources = [r["_src"] for r in self.rows]

    def __len__(self):
        return len(self.rows)

    def _load(self, rel: str, src: str, root: str):
        # 약초 + 샤드 인덱스 설정 시: tar 샤드에서 직접 읽기(/ ↔ __ 경로 불일치 해소)
        if src == "herb" and self._herb_shard_index:
            if self._shard_reader is None:
                from hanmed.shared.shard_image_reader import ShardImageReader
                self._shard_reader = ShardImageReader(self._herb_shard_index, self._herb_shard_dir)
            img = self._shard_reader.get(rel)
            if img is not None:
                return img
            self.n_placeholder += 1   # 아직 미샤딩(612 등)은 placeholder
            return self._placeholder
        fp = _resolve_tongue(rel, root) if src == "tongue" else _resolve_herb(rel, root)
        try:
            return self._Image.open(fp).convert("RGB")
        except Exception:
            self.n_placeholder += 1
            return self._placeholder

    def __getitem__(self, i):
        r = self.rows[i]
        conv = r["conversations"]
        human = conv[0]["value"].replace("<image>", "").strip()
        gpt = conv[1]["value"]
        img = self._load(r["image"], r["_src"], r["_root"]) if r.get("image") else None
        user = [{"type": "text", "text": human}]
        if img is not None:
            user.insert(0, {"type": "image"})
        return {
            "image": img,
            "messages": [
                {"role": "user", "content": user},
                {"role": "assistant", "content": [{"type": "text", "text": gpt}]},
            ],
        }


class Collator:
    """full 대화/prompt 를 각각 토큰화해 응답 토큰만 학습(label 마스킹)."""

    def __init__(self, processor, max_seq_len: int | None = None):
        self.p = processor
        self.max_seq_len = max_seq_len          # D3 — 설정된 길이를 실제로 강제

    def _encode(self, text, images, **kw):
        # 이미지 행은 자르지 않는다 — 시퀀스 길이가 anyres 그리드로 정해져 있어
        # 자르면 image 토큰 수가 어긋나며 processor 가 ValueError 를 낸다.
        if self.max_seq_len and not images:
            kw.update(truncation=True, max_length=self.max_seq_len)
        # 텍스트 전용 행(1차 replay)은 images=None 으로 넘겨야 pixel_values 가 생기지 않는다.
        return self.p(text=text, images=images or None, return_tensors="pt", **kw)

    def __call__(self, batch):
        images = [b["image"] for b in batch if b["image"] is not None]
        if images and len(images) != len(batch):
            raise ValueError("한 배치에 이미지 유·무 행이 섞였다 (micro_bs=1 전제).")
        full = [self.p.apply_chat_template(b["messages"], tokenize=False) for b in batch]
        prompt = [self.p.apply_chat_template(b["messages"][:1], tokenize=False,
                                             add_generation_prompt=True) for b in batch]
        enc = self._encode(full, images, padding=True)
        labels = enc["input_ids"].clone()
        for i, pt in enumerate(prompt):
            plen = self._encode([pt], images[i:i + 1])["input_ids"].shape[1]
            # truncation 으로 프롬프트가 전부를 덮으면 loss 가 nan 이 된다 → 최소 1토큰 남김
            labels[i, :min(plen, labels.shape[1] - 1)] = -100
        labels[enc["input_ids"] == self.p.tokenizer.pad_token_id] = -100
        enc["labels"] = labels
        return enc


# ---------------------------------------------------------------------------
# 균형 샘플러
# ---------------------------------------------------------------------------
def build_weighted_sampler(sources, target_fracs: dict, seed: int):
    """source 별 노출 비율이 target_fracs 에 맞도록 sample 가중치 부여.

    group g 의 목표 노출비율 f_g, 원소 수 n_g → 가중치 = f_g / n_g.
    target_fracs 에 없는 source 는 남은 비율을 균등 분배받는다 (보통 herb).
    (단순 복제 대신 with-replacement 샘플링으로 다양성 유지 — §3 환각방지.)
    """
    from collections import Counter

    import torch
    from torch.utils.data import WeightedRandomSampler

    n = Counter(sources)
    rest = [s for s in n if s not in target_fracs]
    left = 1.0 - sum(target_fracs.get(s, 0.0) for s in n)
    frac = {s: target_fracs.get(s, left / max(len(rest), 1)) for s in n}
    weights = [frac[s] / n[s] for s in sources]
    g = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(weights, num_samples=len(sources),
                                 replacement=True, generator=g), dict(n)


# ---------------------------------------------------------------------------
# 모델
# ---------------------------------------------------------------------------
def load_kwargs(load_4bit: bool) -> dict:
    """from_pretrained 인자. 4bit·bf16 모두 LOCAL_RANK 의 GPU 에 적재한다 (D2).

    device_map 을 rank 로 잡지 않으면 torchrun 의 모든 rank 가 GPU 0 에 올라간다.
    """
    import torch
    from transformers import BitsAndBytesConfig

    kw = {"torch_dtype": torch.bfloat16,
          "device_map": {"": int(os.environ.get("LOCAL_RANK", 0))}}
    if load_4bit:
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
            # projector 는 풀학습이므로 양자화 제외(전체 모듈경로 명시 — re.match 매칭).
            llm_int8_skip_modules=["model.multi_modal_projector"])
    return kw


def has_image_source(cfg: dict) -> bool:
    return any(cfg.get(f"{src}_train") for src in ("tongue", "herb"))


def build_model(cfg: dict, load_4bit: bool):
    from peft import LoraConfig, get_peft_model
    from transformers import LlavaOnevisionForConditionalGeneration

    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        cfg["base_model"], **load_kwargs(load_4bit))
    if load_4bit:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    # 이미지 소스가 없는 라운드(1차 텍스트 SFT)에서는 projector 를 학습 대상에서 뺀다.
    # 텍스트는 비전 타워를 지나지 않아 gradient 가 0이고, 그대로 두면 DDP 가
    # 「unused parameter」로 죽는다. 학습할 신호가 없는 파라미터라 손해도 없다.
    lora = LoraConfig(
        r=cfg["lora_rank"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
        use_rslora=cfg.get("use_rslora", True),
        target_modules=LM_LORA_REGEX,            # 언어모델만 (vision 동결)
        modules_to_save=["multi_modal_projector"] if has_image_source(cfg) else None,
        bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    return model


# ---------------------------------------------------------------------------
# Trainer (균형 샘플러 주입)
# ---------------------------------------------------------------------------
def make_trainer(model, processor, train_ds, val_ds, sampler, cfg, args, load_4bit):
    from transformers import EarlyStoppingCallback, Trainer, TrainingArguments

    class BalancedTrainer(Trainer):
        def _get_train_sampler(self, *a, **k):
            return sampler if sampler is not None else super()._get_train_sampler(*a, **k)

    targs = TrainingArguments(
        output_dir=cfg["output_dir"],
        per_device_train_batch_size=cfg["micro_bs"],
        per_device_eval_batch_size=cfg["micro_bs"],
        gradient_accumulation_steps=cfg["grad_accum"],
        num_train_epochs=cfg["epochs"],
        max_steps=args.max_steps if args.max_steps is not None else -1,
        learning_rate=cfg["lr"], weight_decay=cfg["weight_decay"],
        warmup_ratio=cfg["warmup_ratio"], lr_scheduler_type=cfg.get("lr_scheduler", "cosine"),
        bf16=True, gradient_checkpointing=cfg["grad_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=cfg["logging_steps"],
        save_strategy="steps", eval_strategy="steps",
        save_steps=cfg["save_steps"], eval_steps=cfg["eval_steps"],
        save_total_limit=cfg["save_total_limit"],
        load_best_model_at_end=True, metric_for_best_model="eval_loss",
        greater_is_better=False,
        optim="paged_adamw_8bit" if load_4bit else "adamw_torch",
        report_to="none", remove_unused_columns=False,
        dataloader_num_workers=2, seed=cfg["seed"],
        # 2차(이미지+텍스트 replay)는 텍스트 전용 micro-batch 에서 projector·vision 이
        # 통째로 안 쓰인다 → DDP 가 unused parameter 로 죽으므로 탐지를 켜야 한다.
        # 매 스텝 오버헤드가 붙지만 replay 를 섞는 대가다. 단일 소스 라운드는 끈다.
        ddp_find_unused_parameters=bool(cfg.get("text_train")) and has_image_source(cfg))
    callbacks = []
    if not args.max_steps:
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=cfg.get("early_stopping_patience", 2)))
    return BalancedTrainer(model=model, args=targs,
                           train_dataset=train_ds, eval_dataset=val_ds,
                           data_collator=Collator(processor, cfg.get("max_seq_len")),
                           callbacks=callbacks)


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    load_4bit = args.load_4bit or cfg.get("load_4bit", False)

    print("=" * 72)
    print("VARCO-VISION 한의학 멀티모달 SFT (tongue+herb, 균형 샘플링)")
    print("=" * 72)

    print("[1/4] Building datasets…")
    train_ds = UnifiedMMDataset(cfg, "train", args.max_samples)
    val_ds = UnifiedMMDataset(cfg, "val", args.max_samples)
    from collections import Counter
    print(f"  train={len(train_ds)} {dict(Counter(train_ds.sources))}  val={len(val_ds)}")

    print("[2/4] Building balanced sampler…")
    # 소스가 하나면(1차 텍스트 SFT) 균형 샘플링이 필요 없다 — 기본 셔플이 전량을 한 번씩 본다.
    sampler = None
    if len(set(train_ds.sources)) > 1:
        # herb 는 target 미지정 → 잔여 비율. text 는 1차 replay 노출 비율(행 기준).
        targets = {"tongue": cfg["tongue_target_frac"]}
        if cfg.get("text_target_frac"):
            targets["text"] = cfg["text_target_frac"]
        sampler, counts = build_weighted_sampler(train_ds.sources, targets, cfg["seed"])
        print(f"  목표 노출비율={targets}  n={counts}")
        # 샘플러 노출 비율 실측 (smoke 검증)
        import torch
        draws = list(torch.utils.data.DataLoader(
            range(len(train_ds.sources)), sampler=sampler, batch_size=512))
        flat = torch.cat([d for d in draws]).tolist()
        obs = Counter(train_ds.sources[i] for i in flat)
        print("  실측 노출비율 ≈ "
              + ", ".join(f"{k}={v / len(flat):.3f}" for k, v in sorted(obs.items())))
    else:
        print("  단일 소스 → 기본 셔플 샘플러")

    if args.dry_run:
        # 콜레이트는 processor 필요 → processor 만 로드(모델 X)해서 1 배치 검증
        from transformers import AutoProcessor
        proc = AutoProcessor.from_pretrained(cfg["base_model"])
        if proc.tokenizer.pad_token_id is None:
            proc.tokenizer.pad_token = proc.tokenizer.eos_token
        batch = Collator(proc, cfg.get("max_seq_len"))([train_ds[0]])
        pv = batch.get("pixel_values")
        print(f"[3/4] collate sanity: input_ids {tuple(batch['input_ids'].shape)}  "
              f"labels {tuple(batch['labels'].shape)}  "
              f"pixel_values {tuple(pv.shape) if pv is not None else 'none(text-only)'}  "
              f"non-masked-labels={(batch['labels'] != -100).sum().item()}")
        print(f"  placeholder 이미지(train) {train_ds.n_placeholder}장")
        print("[DRY-RUN] 데이터셋/샘플러/콜레이트 완료 — 모델 로드 skip. exit 0.")
        return 0

    print("[3/4] Loading VARCO + LoRA…")
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(cfg["base_model"])
    if proc.tokenizer.pad_token_id is None:
        proc.tokenizer.pad_token = proc.tokenizer.eos_token
    model = build_model(cfg, load_4bit)

    trainer = make_trainer(model, proc, train_ds, val_ds, sampler, cfg, args, load_4bit)
    print("[4/4] Trainer.train()…")
    trainer.train()

    out = Path(cfg["output_dir"]) / "adapter"
    out.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out))
    proc.save_pretrained(str(out))
    with open(Path(cfg["output_dir"]) / "train_manifest.json", "w") as fh:
        json.dump({"stage": "sft", "tongue_target_frac": cfg["tongue_target_frac"],
                   **{k: cfg[k] for k in ("base_model", "lora_rank", "lr", "epochs")},
                   "placeholder_images": train_ds.n_placeholder},
                  fh, ensure_ascii=False, indent=2)
    print(f"✔ adapter saved: {out}  (placeholder {train_ds.n_placeholder}장)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
