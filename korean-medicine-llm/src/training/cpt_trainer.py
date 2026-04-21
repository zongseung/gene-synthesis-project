"""Stage 1 CPT trainer — HanMed-LLM ver2 (§04a §C.1 / §C.3 / §C.5 / §11.2 W5).

본 스크립트는 **Stage 1 CPT (causal LM self-supervised)** 학습을 단일 책임으로 수행한다.
데이터 재처리·tokenizer 재생성·평가는 범위 밖이다 (§04a §A.2/§A.3/§D 참조).

참조:
- §C.1 Objective : next-token prediction on all non-pad tokens (cross-entropy)
- §C.2 Mix      : HanMed 40% = bilingual 5% + zh_only 25% + ko_only 10%
                  (Wiki-ko 30% / CBETA 20% / 예비 10% 는 M2 수집 대기 — CLI optional)
- §C.3 Packing  : seq_len 2048 (pre-packed jsonl 그대로 입력), pad 우측
- §C.5 Hyper    : Bllossom-8B + LoRA r=32/α=64 (target 7개) + bf16
                  micro_bs 2 × grad_accum 16 × seq_len 2048 = 65,536 tok/step (1 GPU)
                  LR 1e-4 / cosine / warmup_steps = int(total_steps * 0.05)
                  total_steps = ceil(cap_tokens / (micro_bs * grad_accum * seq_len * N_GPU))
- §10.3 P-CPT   : Stage 1 은 plain text (chat template token 미포함) — 현 packed 파일은
                  `<|begin_of_text|>` BOS + `<|eot_id|>` sep + `<|end_of_text|>` pad 포맷

CLI 요약:
    PYTHONHASHSEED=0 .venv/bin/python src/training/cpt_trainer.py \
        --cap-tokens 20_400_000 --epoch-variant 3 --dry-run

DDP:
    PYTHONHASHSEED=0 torchrun --nproc_per_node=2 \
        .venv/bin/python src/training/cpt_trainer.py [...]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

# `src/utils/seed.py` 를 import 가능하게 루트 경로 삽입 (repo root == .venv 상위)
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.utils.seed import set_global_seed  # noqa: E402


# =============================================================================
# 기본값 — §C.5 표 그대로
# =============================================================================

DEFAULT_BASE = "MLP-KTLim/llama-3-Korean-Bllossom-8B"
DEFAULT_TOKENIZER = "data/tokenizer/hanmed_bllossom_ext"
DEFAULT_OUTPUT = "outputs/cpt_bllossom"
DEFAULT_CAP_TOKENS = 20_400_000  # §C.4.3 Core 14 R3a (unique 2.72M × 3 / 0.40)
DEFAULT_EPOCH_VARIANT = 3
DEFAULT_LORA_RANK = 32
DEFAULT_LORA_ALPHA = 64
DEFAULT_LORA_DROPOUT = 0.05
DEFAULT_LR = 1e-4
DEFAULT_WARMUP_RATIO = 0.05
DEFAULT_MICRO_BS = 2
DEFAULT_GRAD_ACCUM = 16
DEFAULT_SEQ_LEN = 2048
DEFAULT_SEED = 42

# §C.5 target_modules — 7개 고정
LORA_TARGET_MODULES: tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

# ver4 §A ablation — synth 제거 (template overfit). raw-only baseline.
#
# ver4 §R1 (2026-04-21) — mix 재설계. 근거:
#   claudedocs/research_hanmed_cpt_methodology_20260421.md §0
#   - 이전 mix(bilingual 0.45 / zh 0.20 / ko 0.35, general 0%) 에서 관측된
#     3가지 실패: (F1) style-over-fact 환각, (F2) 도메인 트리거 시 KO→ZH
#     언어 붕괴, (F3) rare-token degeneracy.
#   - BianCang(460M, TCM) 은 general domain replay ~16% (72/460M) 로
#     도메인 성능 +60pp 유지하며 기본 능력 보존 [arXiv 2411.11027 Table 1].
#   - GeRe 는 LoRA 조건 1K 고정 general replay 로 F1 +6.5pp 입증
#     [arXiv 2508.04676 Table III].
#
# 목표 불변식: general replay ≥ 0.15, KO 앵커(ko_only + wiki_ko) ≥ 0.50,
# zh_only ≤ 0.10.
#
# 주의: wiki_ko packed jsonl 이 아직 미생성 (data/replay/wiki_ko_packed_2048.jsonl).
# 학습 실행 전 replay corpus 수집·packing 필요. 없으면 build_datasets 가
# FileNotFoundError 로 조기 실패.
DEFAULT_MIX = (
    "hanmed_ko_only:0.45,"
    "hanmed_bilingual:0.30,"
    "hanmed_zh_only:0.10,"
    "wiki_ko:0.15"
)

# 각 corpus 식별자 → packed jsonl 경로 (B.1 / B.2)
CORPUS_PATHS: dict[str, str] = {
    "hanmed_bilingual": "data/cpt_processed/hanmed_bilingual_packed_2048.jsonl",
    "hanmed_zh_only": "data/cpt_processed/hanmed_zh_only_packed_2048.jsonl",
    "hanmed_ko_only": "data/cpt_processed/hanmed_ko_only_packed_2048.jsonl",
    "hanmed_synth_facts": "data/cpt_processed/hanmed_synth_facts_packed_2048.jsonl",
    # ver4 §R1 (2026-04-21): wiki_ko 는 src/data/builder/build_wiki_ko.py 로
    # 수집 후 preprocess.py 를 거쳐 cpt_processed/ 에 packed 로 생성됨 (hanmed
    # 코퍼스와 동일한 디렉토리 · 동일 seq_len 2048).
    "wiki_ko": "data/cpt_processed/wiki_ko_packed_2048.jsonl",
    # 이하 M2 수집 대기 — CLI 로 경로 override 가능
    "cbeta": "data/replay/cbeta_packed_2048.jsonl",
    "aihub": "data/replay/aihub_ko_packed_2048.jsonl",
}


# =============================================================================
# 데이터 구조
# =============================================================================

@dataclass
class MixSpec:
    """정규화된 혼합 사양: corpus_id → probability (합계 1.0)."""

    entries: list[tuple[str, float]]

    @property
    def ids(self) -> list[str]:
        return [c for c, _ in self.entries]

    @property
    def probs(self) -> list[float]:
        return [p for _, p in self.entries]


# =============================================================================
# CLI / 설정
# =============================================================================

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="HanMed-LLM Stage 1 CPT trainer (§04a §C.5 / §11.2 W5)",
    )

    # 경로/자원
    p.add_argument(
        "--corpus",
        type=str,
        default="data/cpt_processed/corpus_v1.json",
        help="corpus_v1.json manifest 경로 (§B.3). dry-run 에서는 미존재해도 허용.",
    )
    p.add_argument("--base", type=str, default=DEFAULT_BASE, help="HF base model id (§C.5).")
    p.add_argument(
        "--tokenizer",
        type=str,
        default=DEFAULT_TOKENIZER,
        help="학습 tokenizer 경로 (§A.4, Bllossom + 4 special tokens).",
    )
    p.add_argument("--output", type=str, default=DEFAULT_OUTPUT, help="학습 결과 output_dir.")

    # §C.4.3 토큰 예산
    p.add_argument(
        "--cap-tokens",
        type=int,
        default=DEFAULT_CAP_TOKENS,
        help="총 학습 토큰 cap (§C.4.3). Core 14 R3a default 20.4M.",
    )
    p.add_argument(
        "--epoch-variant",
        type=int,
        choices=(3, 5),
        default=DEFAULT_EPOCH_VARIANT,
        help="§C.4.3 repeat_factor variant (R3a=3 / R3b=5). run_name 에 기록.",
    )

    # §C.5 LoRA
    p.add_argument("--lora-rank", type=int, default=DEFAULT_LORA_RANK)
    p.add_argument("--lora-alpha", type=int, default=DEFAULT_LORA_ALPHA)
    p.add_argument("--lora-dropout", type=float, default=DEFAULT_LORA_DROPOUT)

    # §C.5 학습
    p.add_argument("--lr", type=float, default=DEFAULT_LR)
    p.add_argument("--warmup-ratio", type=float, default=DEFAULT_WARMUP_RATIO)
    p.add_argument("--micro-bs", type=int, default=DEFAULT_MICRO_BS)
    p.add_argument("--grad-accum", type=int, default=DEFAULT_GRAD_ACCUM)
    p.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)

    # 믹스 (§C.2) — `id:p,id:p,...` 문자열. 필요 시 replay corpus 추가.
    p.add_argument(
        "--mix",
        type=str,
        default=DEFAULT_MIX,
        help=(
            "Comma-separated `corpus_id:prob` list. "
            "R1 default (2026-04-21): ko_only 0.45 / bilingual 0.30 / zh_only 0.10 / wiki_ko 0.15. "
            "불변식: general≥0.15, KO 앵커≥0.50, zh_only≤0.10. "
            "합계는 자동 정규화되며 default 는 이미 sum=1.00."
        ),
    )
    p.add_argument(
        "--extra-corpus",
        type=str,
        action="append",
        default=[],
        help=(
            "replay corpus 경로 override. 형식 `corpus_id=/abs/path.jsonl`. "
            "여러 번 반복 가능 (Wiki-ko, CBETA, AI Hub)."
        ),
    )

    # 재현성/동작
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="모델 로드 전까지만 실행 (argparse/dataset/step 계산 검증용).",
    )
    p.add_argument("--resume", type=str, default=None, help="Trainer checkpoint 경로.")

    return p.parse_args(argv)


def parse_mix(mix_str: str) -> MixSpec:
    """`id:p,id:p,...` 문자열 → MixSpec (합계 1.0 으로 정규화).

    §C.2 정합: v0 default 는 HanMed 40% 만 → 정규화 후 내부 비율 유지 (0.125/0.625/0.25).
    Wiki-ko/CBETA/AI Hub 가 포함된 6-way 전체 믹스가 들어오면 그대로 1.0 유지.
    """
    entries: list[tuple[str, float]] = []
    for chunk in mix_str.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise ValueError(f"mix entry must be 'id:prob', got {chunk!r}")
        cid, prob = chunk.rsplit(":", 1)
        entries.append((cid.strip(), float(prob)))

    if not entries:
        raise ValueError("mix is empty")

    total = sum(p for _, p in entries)
    if total <= 0:
        raise ValueError(f"mix probabilities must sum to > 0, got {total}")

    normalized = [(c, p / total) for c, p in entries]
    return MixSpec(entries=normalized)


def resolve_corpus_paths(
    mix: MixSpec,
    extras: list[str],
) -> dict[str, Path]:
    """corpus_id → packed jsonl Path. `--extra-corpus id=path` 가 default 를 override."""
    overrides: dict[str, str] = {}
    for item in extras:
        if "=" not in item:
            raise ValueError(f"--extra-corpus expects 'id=path', got {item!r}")
        cid, path = item.split("=", 1)
        overrides[cid.strip()] = path.strip()

    resolved: dict[str, Path] = {}
    for cid in mix.ids:
        raw = overrides.get(cid) or CORPUS_PATHS.get(cid)
        if raw is None:
            raise KeyError(
                f"unknown corpus id {cid!r}; register in CORPUS_PATHS or pass "
                f"--extra-corpus {cid}=/path/to.jsonl"
            )
        resolved[cid] = Path(raw)
    return resolved


# =============================================================================
# 분산 (§W5 DDP)
# =============================================================================

def init_distributed() -> tuple[int, int, int]:
    """(world_size, rank, local_rank). torchrun 외에도 single-GPU 를 허용."""
    import torch  # 지연 import (dry-run 초기 출력 속도를 위해)

    if "LOCAL_RANK" in os.environ and not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        world = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local_rank)
        return world, rank, local_rank

    return 1, 0, 0


# =============================================================================
# 학습 step 계산 (§C.5 공식)
# =============================================================================

def compute_schedule(
    cap_tokens: int,
    micro_bs: int,
    grad_accum: int,
    seq_len: int,
    world_size: int,
    warmup_ratio: float,
) -> tuple[int, int, int]:
    """(effective_batch_tokens, total_steps, warmup_steps).

    §C.5:
        total_steps  = ceil(cap_tokens / (micro_bs * grad_accum * seq_len * N_GPU))
        warmup_steps = int(total_steps * warmup_ratio)
    """
    eff_tokens_per_step = micro_bs * grad_accum * seq_len * world_size
    if eff_tokens_per_step <= 0:
        raise ValueError(f"effective tokens/step must be > 0, got {eff_tokens_per_step}")
    total_steps = math.ceil(cap_tokens / eff_tokens_per_step)
    warmup_steps = int(total_steps * warmup_ratio)
    return eff_tokens_per_step, total_steps, warmup_steps


# =============================================================================
# Dataset
# =============================================================================

def build_dataset(
    mix: MixSpec,
    corpus_paths: dict[str, Path],
    seq_len: int,
    pad_token_id: int,
    seed: int,
):
    """interleaved packed CPT dataset.

    각 packed jsonl 의 record 는 `{"input_ids": [len=seq_len]}` (§B.2).
    - labels = input_ids (CLM, pad 포함 — collator 에서 pad 마스킹은 하지 않음.
      단 attention_mask 를 통해 loss 영향 조절은 hf default 동작에 맡긴다.)
    - attention_mask = 1 where input_ids != pad_id, else 0

    §C.1 문구상 "cross-entropy over all non-pad tokens" 이므로 pad 위치의 label 은
    -100 (ignore_index) 로 치환한다.
    """
    from datasets import concatenate_datasets, interleave_datasets, load_dataset  # 지연 import

    # §C.7: held-out 2% random split per corpus, seed=42
    val_fraction = 0.02

    per_corpus_train = []
    per_corpus_val = []
    record_counts: dict[str, dict[str, int]] = {}
    for cid in mix.ids:
        path = corpus_paths[cid]
        if not path.exists():
            raise FileNotFoundError(
                f"packed corpus missing for {cid!r}: {path} "
                f"(W3 Stage 2 or replay 수집 필요)"
            )
        ds = load_dataset("json", data_files=str(path), split="train")

        def _attach_labels(batch):
            new_input_ids: list[list[int]] = []
            new_labels: list[list[int]] = []
            new_attn: list[list[int]] = []
            for ids in batch["input_ids"]:
                if len(ids) != seq_len:
                    raise ValueError(
                        f"packed record length {len(ids)} != seq_len {seq_len} in {path}"
                    )
                labels = [tid if tid != pad_token_id else -100 for tid in ids]
                attn = [0 if tid == pad_token_id else 1 for tid in ids]
                new_input_ids.append(list(ids))
                new_labels.append(labels)
                new_attn.append(attn)
            return {
                "input_ids": new_input_ids,
                "labels": new_labels,
                "attention_mask": new_attn,
            }

        ds = ds.map(
            _attach_labels,
            batched=True,
            remove_columns=[c for c in ds.column_names if c != "input_ids"],
            desc=f"attach labels [{cid}]",
        )

        n_total = len(ds)
        # 최소 1 val record 보장 (작은 corpus 에서도 평가 가능)
        if n_total < 50:
            ds_train, ds_val = ds, ds.select(range(min(1, n_total)))
        else:
            split = ds.train_test_split(test_size=val_fraction, seed=seed)
            ds_train, ds_val = split["train"], split["test"]

        record_counts[cid] = {
            "total": n_total,
            "train": len(ds_train),
            "val": len(ds_val),
        }
        per_corpus_train.append(ds_train)
        per_corpus_val.append(ds_val)

    interleaved_train = interleave_datasets(
        per_corpus_train,
        probabilities=mix.probs,
        seed=seed,
        stopping_strategy="all_exhausted",
    )
    # val 은 mix 비중 없이 concat — per-corpus breakdown 은 별도 eval 로.
    val_combined = concatenate_datasets(per_corpus_val)
    return interleaved_train, val_combined, record_counts


# =============================================================================
# 모델 (§C.5)
# =============================================================================

def build_model(
    base: str,
    tokenizer_dir: str,
    lora_rank: int,
    lora_alpha: int,
    lora_dropout: float,
):
    """Bllossom-8B bf16 + LoRA (§C.5 target_modules 7개) + grad ckpt."""
    import torch  # noqa: F401
    from peft import LoraConfig, TaskType, get_peft_model  # noqa: F401
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: F401

    tok = AutoTokenizer.from_pretrained(tokenizer_dir)
    if tok.pad_token is None:
        # Bllossom-8B ext tokenizer 는 pad_id 128001 을 이미 보유 — 방어용.
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base,
        torch_dtype=torch.bfloat16,
    )
    # vocab 128,256 → 128,260 반영 (§A.4)
    # mean_resizing=False: DDP 에서 rank 별 RNG 로 init 하면 embedding 값이
    # 불일치 → `_verify_params_across_processes` broadcast 단계에서 hang.
    # 새 4 토큰은 0-init 후 학습으로 맞춘다.
    model.resize_token_embeddings(len(tok), mean_resizing=False)

    lora_cfg = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        # ver4 §5.3 재설계 (docs/ver4/05_new_token_training_methods.md B안):
        # 기존 modules_to_save=["embed_tokens","lm_head"] 는 PEFT 의
        # ModulesToSaveWrapper 가 rank 간 param iteration 불일치를 만들어
        # DDP `_verify_params_across_processes` AllGather 에서 hang.
        # PEFT 0.13.2 는 trainable_token_indices 미지원(0.14+), 업그레이드는
        # transformers 5.5.4 호환 리스크가 커 회피.
        # 대안: embed_tokens / lm_head 를 LoRA target 에 포함 (PEFT 0.8+ 지원).
        # Chinese-LLaMA-Alpaca (arXiv:2304.08177) Stage 2 와 동형.
        # 새 4 token 은 mean_resizing=False 로 0-init 된 뒤 LoRA delta (rank 32) 로 학습.
        target_modules=list(LORA_TARGET_MODULES) + ["embed_tokens", "lm_head"],
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # §C.5 grad ckpt 는 TrainingArguments(gradient_checkpointing=True) 에서 단일 지점
    # 관리 (use_reentrant=False). 여기서 중복 enable 하면 HF Trainer 가 prepare 직전
    # 재토글하면서 PEFT 의 ModulesToSaveWrapper 와 타이밍 충돌 → DDP init hang.
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    return model, tok


# =============================================================================
# TrainingArguments (§C.5)
# =============================================================================

def make_training_args(
    output: str,
    total_steps: int,
    warmup_steps: int,
    args: argparse.Namespace,
    rank: int,
    run_name: str,
):
    from transformers import TrainingArguments  # 지연 import

    report_to = ["wandb"] if rank == 0 else "none"

    # ver4 §5.3 (4): save_steps / eval_steps 둘 다 50 으로 정렬.
    # (기존: save=total_steps//4, eval=min(50, total_steps//3) → step 150 근처에서
    #  교차해 best eval step 에 save 가 없던 버그)
    save_steps = 50
    eval_steps = 50

    return TrainingArguments(
        output_dir=output,
        per_device_train_batch_size=args.micro_bs,
        per_device_eval_batch_size=args.micro_bs,
        gradient_accumulation_steps=args.grad_accum,
        # ver4 §5.3 (2): num_train_epochs=3 명시. max_steps 가 설정되어 있으면
        # HF Trainer 는 max_steps 를 우선하나, epoch-aware 스케줄러(e.g. cosine)
        # 와 재현성 manifest 를 위해 명시값을 주입한다.
        num_train_epochs=args.epoch_variant,
        max_steps=total_steps,
        warmup_steps=warmup_steps,
        learning_rate=args.lr,
        weight_decay=0.0,
        # ver4 §5.3 (5): LR floor — cosine_with_min_lr + min_lr_rate=0.1
        # (transformers ≥ 4.43 지원. min_lr_rate 는 lr_scheduler_kwargs 경유 전달.)
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr_rate": 0.1},
        logging_steps=10,
        save_strategy="steps",
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_steps=save_steps,
        save_total_limit=2,
        # ver4 §5.3 (3): best model 선택 — eval_loss 최소값 채택.
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        optim="adamw_torch",
        seed=args.seed,
        dataloader_num_workers=2,
        remove_unused_columns=False,
        # grad ckpt 재복원. off 로 두면 LoRA-on-embed/lm_head + micro_bs=2 + seq 2048
        # 조합에서 activation 이 A6000 48GB 한계 초과 (dropout forward OOM).
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to=report_to,
        run_name=run_name,
        # modules_to_save 제거 + LoRA-on-embed/lm_head 로 전환됨 → unused param 없음.
        ddp_find_unused_parameters=False,
    )


# =============================================================================
# wandb (§W5, rank-0 only)
# =============================================================================

def init_wandb_if_rank0(rank: int, run_name: str, config: dict) -> None:
    if rank != 0:
        return
    os.environ.setdefault("WANDB_MODE", "offline")
    try:
        import wandb

        wandb.init(
            project="HanMed-CPT",
            name=run_name,
            config=config,
            reinit=False,
        )
    except ImportError:
        # wandb 미설치 시 silent — report_to 에서 알아서 제거되도록 handle
        os.environ["WANDB_DISABLED"] = "true"


# =============================================================================
# main
# =============================================================================

def _log(rank: int, msg: str) -> None:
    if rank == 0:
        print(msg, flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    # PYTHONHASHSEED=0 강제 (§A.5)
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise RuntimeError(
            "PYTHONHASHSEED must be '0'. Rerun with: "
            "`PYTHONHASHSEED=0 .venv/bin/python src/training/cpt_trainer.py ...`"
        )
    set_global_seed(args.seed)

    mix = parse_mix(args.mix)
    corpus_paths = resolve_corpus_paths(mix, args.extra_corpus)

    # DDP init (dry-run 에서도 single-GPU fallback 으로 world=1 을 얻는다)
    if args.dry_run:
        world_size, rank, local_rank = 1, 0, 0
    else:
        world_size, rank, local_rank = init_distributed()

    eff_tokens_per_step, total_steps, warmup_steps = compute_schedule(
        cap_tokens=args.cap_tokens,
        micro_bs=args.micro_bs,
        grad_accum=args.grad_accum,
        seq_len=args.seq_len,
        world_size=world_size,
        warmup_ratio=args.warmup_ratio,
    )

    cap_mtokens = args.cap_tokens // 1_000_000
    run_name = f"cpt_bllossom_{args.epoch_variant}e_{cap_mtokens}M"

    _log(rank, "=" * 72)
    _log(rank, f"HanMed-LLM Stage 1 CPT trainer — §04a §C.5 / §11.2 W5")
    _log(rank, "=" * 72)
    _log(rank, f"  base            : {args.base}")
    _log(rank, f"  tokenizer       : {args.tokenizer}")
    _log(rank, f"  output_dir      : {args.output}")
    _log(rank, f"  world_size      : {world_size} (rank={rank}, local_rank={local_rank})")
    _log(rank, f"  cap_tokens      : {args.cap_tokens:,} (epoch_variant={args.epoch_variant})")
    _log(rank, f"  seq_len         : {args.seq_len}")
    _log(rank, f"  micro_bs        : {args.micro_bs} × grad_accum {args.grad_accum} "
               f"× world {world_size} = {args.micro_bs * args.grad_accum * world_size} "
               f"(effective batch size in sequences)")
    _log(rank, f"  tokens / step   : {eff_tokens_per_step:,}")
    _log(rank, f"  total_steps     : {total_steps} "
               f"(= ceil({args.cap_tokens:,} / {eff_tokens_per_step:,}))")
    _log(rank, f"  warmup_steps    : {warmup_steps} "
               f"(= int({total_steps} × {args.warmup_ratio}))")
    _log(rank, f"  LoRA            : r={args.lora_rank}, α={args.lora_alpha}, "
               f"dropout={args.lora_dropout}, targets={LORA_TARGET_MODULES}")
    _log(rank, f"  LR              : {args.lr} (cosine w/ warmup, wd=0.0)")
    _log(rank, f"  run_name        : {run_name}")
    _log(rank, "")
    _log(rank, "  Mix (§C.2, post-normalization) :")
    for cid, p in mix.entries:
        _log(rank, f"    {cid:<22s} p={p:.4f}  path={corpus_paths[cid]}")
    _log(rank, f"    (sum of probs = {sum(mix.probs):.4f})")
    _log(rank, "")

    # ---- Dataset ---------------------------------------------------------
    # Pad id 는 tokenizer_config.json / manifest 에서 128001 고정. dry-run 에서도
    # 같은 id 를 쓴다 (이미 packed 파일이 동일 pad 로 만들어짐).
    pad_token_id = 128001

    _log(rank, "[1/4] Building dataset (interleave over packed jsonl) + val split 2%…")
    train_dataset, val_dataset, record_counts = build_dataset(
        mix=mix,
        corpus_paths=corpus_paths,
        seq_len=args.seq_len,
        pad_token_id=pad_token_id,
        seed=args.seed,
    )
    _log(rank, "  Dataset records per corpus (train / val):")
    for cid, counts in record_counts.items():
        _log(
            rank,
            f"    {cid:<22s} total={counts['total']:>6d}  train={counts['train']:>6d}  val={counts['val']:>4d}",
        )
    total_train = sum(c["train"] for c in record_counts.values())
    total_val = sum(c["val"] for c in record_counts.values())
    _log(rank, f"  Union train = {total_train}, val = {total_val} (§C.7 held-out 2%)")

    if args.dry_run:
        _log(rank, "")
        _log(rank, "[DRY-RUN] argparse/dataset/step 계산 완료 — 모델 로드 skip.")
        _log(rank, "[DRY-RUN] exit 0.")
        return 0

    # ---- Model ----------------------------------------------------------
    _log(rank, "[2/4] Loading base model + LoRA wrap…")
    model, tok = build_model(
        base=args.base,
        tokenizer_dir=args.tokenizer,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )

    # ---- TrainingArguments + wandb --------------------------------------
    _log(rank, "[3/4] Building TrainingArguments + wandb…")
    targs = make_training_args(
        output=args.output,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        args=args,
        rank=rank,
        run_name=run_name,
    )

    wandb_config = {
        "cap_tokens": args.cap_tokens,
        "epoch_variant": args.epoch_variant,
        "seq_len": args.seq_len,
        "micro_bs": args.micro_bs,
        "grad_accum": args.grad_accum,
        "world_size": world_size,
        "total_steps": total_steps,
        "warmup_steps": warmup_steps,
        "lr": args.lr,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "mix": dict(mix.entries),
        "base": args.base,
        "tokenizer": args.tokenizer,
        "seed": args.seed,
    }
    init_wandb_if_rank0(rank, run_name, wandb_config)

    # ---- Trainer --------------------------------------------------------
    from transformers import Trainer, default_data_collator  # 지연 import

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=default_data_collator,
    )

    _log(rank, "[4/4] Starting Trainer.train()…")
    trainer.train(resume_from_checkpoint=args.resume)

    # ---- Save (rank 0 only) --------------------------------------------
    if rank == 0:
        output_root = Path(args.output)
        adapter_dir = output_root / "adapter"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(adapter_dir))
        tok.save_pretrained(str(adapter_dir))

        # corpus_v1.json 사본 (재현성 — §W5)
        corpus_manifest = Path(args.corpus)
        if corpus_manifest.exists():
            shutil.copy2(corpus_manifest, output_root / "corpus_v1.json")

        # 학습 메타데이터
        meta = {
            "run_name": run_name,
            "total_steps": total_steps,
            "warmup_steps": warmup_steps,
            "effective_tokens_per_step": eff_tokens_per_step,
            "world_size": world_size,
            "cap_tokens": args.cap_tokens,
            "epoch_variant": args.epoch_variant,
            "mix": dict(mix.entries),
            "base": args.base,
            "tokenizer": args.tokenizer,
        }
        with open(output_root / "train_manifest.json", "w", encoding="utf-8") as fh:
            json.dump(meta, fh, ensure_ascii=False, indent=2)

        _log(rank, f"✔ adapter saved : {adapter_dir}")
        _log(rank, f"✔ manifest copy : {output_root / 'corpus_v1.json'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
