"""Stage 1 분류기 학습 — DDP 2-GPU, bf16(GradScaler 없음), AdamW+cosine, best by val.

멀티태스크 손실 = w_s·CE(species, class-balanced) + w_p·CE(part) + w_t·BCE(toxic).
독초 recall이 안전의 핵심이라 toxic 가중을 높게 둘 수 있음(--w_toxic).
early stopping 없음, val_species_acc 기준 best_model.pth 저장(rank0).

사용:
  # vocab 먼저 생성(샤딩 완료 후)
  PYTHONPATH=src .venv/bin/python -m hanmed_mm.classifier.labels  # (아래 main에서 자동 생성도 함)
  # 2-GPU 학습
  PYTHONHASHSEED=0 PYTHONPATH=src .venv/bin/torchrun --nproc_per_node=2 \
    -m hanmed_mm.classifier.train \
    --shards "data/shards/151_train_w*.tar" --val_shards "data/shards/151_val_w*.tar" \
    --backbone convnextv2 --img 512 --bsz 32 --epochs 30 --out outputs/clf_run1
  # 단일 GPU 디버그
  PYTHONPATH=src .venv/bin/python -m hanmed_mm.classifier.train ... --single_gpu

전제: webdataset, timm 설치 필요 (.venv/bin/pip install webdataset timm).
"""
from __future__ import annotations
import argparse
import glob
import json
import os

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from hanmed_mm.classifier.dataset import make_loader
from hanmed_mm.classifier.labels import build_vocab, load_vocab, species_weights
from hanmed_mm.classifier.model import build_model


# ----------------------------------------------------------------- DDP 유틸
def ddp_setup(single_gpu: bool):
    if single_gpu or "RANK" not in os.environ:
        return 0, 1, 0, False
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local)
    return rank, world, local, True


def is_main(rank):
    return rank == 0


def log(rank, *a):
    if is_main(rank):
        print(*a, flush=True)


# ----------------------------------------------------------------- 평가
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    n = sp_hit = pt_hit = 0
    tox_tp = tox_fn = 0      # 독초 recall 추적(안전 최우선)
    for img, sp, pt, tox in loader:
        img = img.to(device, non_blocking=True)
        sp = sp.to(device); pt = pt.to(device); tox = tox.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(img)
        sp_hit += (out["species"].argmax(1) == sp).sum().item()
        pt_hit += (out["part"].argmax(1) == pt).sum().item()
        pred_tox = (out["toxic"].float().sigmoid() >= 0.5)
        is_tox = tox.bool()
        tox_tp += (pred_tox & is_tox).sum().item()
        tox_fn += (~pred_tox & is_tox).sum().item()
        n += sp.size(0)
    model.train()
    recall = tox_tp / (tox_tp + tox_fn) if (tox_tp + tox_fn) else float("nan")
    return {"n": n, "species_acc": sp_hit / max(n, 1),
            "part_acc": pt_hit / max(n, 1), "toxic_recall": recall}


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", required=True, help="train 샤드 brace glob")
    ap.add_argument("--val_shards", default="", help="val 샤드 glob(없으면 train 일부)")
    ap.add_argument("--manifest_glob", default="data/shards/*_train_manifest.json")
    ap.add_argument("--vocab", default="data/shards/label_vocab.json")
    ap.add_argument("--backbone", default="convnextv2", choices=["convnextv2", "dinov2"])
    ap.add_argument("--img", type=int, default=512)
    ap.add_argument("--bsz", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup_ratio", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--w_species", type=float, default=1.0)
    ap.add_argument("--w_part", type=float, default=0.3)
    ap.add_argument("--w_toxic", type=float, default=1.0, help="독초 recall 중요 → 높게")
    ap.add_argument("--out", default="outputs/clf_run1")
    ap.add_argument("--single_gpu", action="store_true")
    args = ap.parse_args()

    rank, world, local, ddp = ddp_setup(args.single_gpu)
    device = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")

    # vocab (rank0 생성 → 동기화 후 전원 로드)
    if is_main(rank) and not os.path.exists(args.vocab):
        os.makedirs(os.path.dirname(args.vocab), exist_ok=True)
        build_vocab(args.manifest_glob, args.vocab)
    if ddp:
        dist.barrier()
    vocab = load_vocab(args.vocab)
    n_species = len(vocab["species"])
    n_parts = len(vocab["parts"])
    log(rank, f"vocab: species {n_species} / parts {n_parts}")

    # epoch 길이(총 학습 샘플 수) — 매니페스트 합
    total = sum(json.load(open(m, encoding="utf-8"))["num_samples"]
                for m in glob.glob(args.manifest_glob)) or (args.bsz * 1000)
    epoch_samples = max(1, total // world)
    steps_per_epoch = max(1, epoch_samples // args.bsz)
    log(rank, f"train samples ≈ {total} (rank당 {epoch_samples}, {steps_per_epoch} step/epoch)")

    # 데이터
    train_loader = make_loader(args.shards, vocab, args.img, args.bsz,
                               args.workers, train=True, epoch_samples=epoch_samples)
    val_urls = args.val_shards or args.shards
    val_loader = make_loader(val_urls, vocab, args.img, args.bsz,
                             max(2, args.workers // 2), train=False,
                             epoch_samples=min(epoch_samples, 20000))

    # 모델
    model = build_model(args.backbone, n_species, n_parts, pretrained=True).to(device)
    if ddp:
        model = DDP(model, device_ids=[local], find_unused_parameters=False)

    # 손실 (species class-balanced)
    sp_w = torch.tensor(species_weights(args.manifest_glob, vocab, "sqrt"),
                        dtype=torch.float32, device=device)
    ce_species = nn.CrossEntropyLoss(weight=sp_w)
    ce_part = nn.CrossEntropyLoss(ignore_index=vocab["parts"]["unknown"])
    bce_toxic = nn.BCEWithLogitsLoss()

    # 옵티마이저 + cosine warmup
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    from transformers import get_cosine_schedule_with_warmup
    total_steps = steps_per_epoch * args.epochs
    sched = get_cosine_schedule_with_warmup(
        opt, int(total_steps * args.warmup_ratio), total_steps)

    if is_main(rank):
        os.makedirs(args.out, exist_ok=True)
    best = -1.0
    for epoch in range(args.epochs):
        model.train()
        for step, (img, sp, pt, tox) in enumerate(train_loader):
            img = img.to(device, non_blocking=True)
            sp = sp.to(device); pt = pt.to(device); tox = tox.to(device).float()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(img)
                loss = (args.w_species * ce_species(out["species"], sp)
                        + args.w_part * ce_part(out["part"], pt)
                        + args.w_toxic * bce_toxic(out["toxic"].float(), tox))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            if is_main(rank) and step % 50 == 0:
                log(rank, f"[e{epoch} s{step}/{steps_per_epoch}] loss {loss.item():.4f} "
                          f"lr {sched.get_last_lr()[0]:.2e}")
            if step + 1 >= steps_per_epoch:
                break

        # 검증(rank0)
        if is_main(rank):
            m = evaluate(model.module if ddp else model, val_loader, device)
            log(rank, f"== epoch {epoch} val: species {m['species_acc']:.4f} "
                      f"part {m['part_acc']:.4f} toxic_recall {m['toxic_recall']:.4f} (n={m['n']})")
            score = m["species_acc"]
            if score > best:
                best = score
                ckpt = {"model_state_dict": (model.module if ddp else model).state_dict(),
                        "config": vars(args), "vocab": vocab, "val": m, "epoch": epoch}
                torch.save(ckpt, os.path.join(args.out, "best_model.pth"))
                log(rank, f"  ↑ best 갱신(species_acc {best:.4f}) → best_model.pth")
        if ddp:
            dist.barrier()

    if ddp:
        dist.destroy_process_group()
    log(rank, "학습 종료.")


if __name__ == "__main__":
    main()
