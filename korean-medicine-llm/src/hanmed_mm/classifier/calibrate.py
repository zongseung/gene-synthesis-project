"""학습 후 보정 — temperature scaling + abstain 임계(설계 §6 후처리).

1) species 로짓에 단일 온도 T를 LBFGS로 적합(NLL 최소화) → calibrated 확률.
2) MSP(max softmax prob) 기반 abstain 임계 τ를 신뢰도-커버리지 곡선에서 선택.
   (energy/OOD는 후속 확장 지점.)
결과를 best_model.pth 옆 calibration.json 에 저장 → infer가 로드.

사용:
  PYTHONPATH=src .venv/bin/python -m hanmed_mm.classifier.calibrate \
    --ckpt outputs/clf_run1/best_model.pth --val_shards "data/shards/151_val_w*.tar" \
    --target_acc 0.9
"""
from __future__ import annotations
import argparse
import json
import os

import torch
import torch.nn.functional as F

from hanmed_mm.classifier.dataset import make_loader
from hanmed_mm.classifier.model import build_model


@torch.no_grad()
def collect_logits(model, loader, device):
    logits, labels = [], []
    for img, sp, pt, tox in loader:
        img = img.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(img)
        logits.append(out["species"].float().cpu())
        labels.append(sp)
    return torch.cat(logits), torch.cat(labels)


def fit_temperature(logits, labels):
    T = torch.nn.Parameter(torch.ones(1))
    opt = torch.optim.LBFGS([T], lr=0.01, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / T.clamp_min(1e-2), labels)
        loss.backward()
        return loss

    opt.step(closure)
    return float(T.detach().clamp_min(1e-2).item())


def pick_abstain_tau(probs, correct, target_acc: float):
    """MSP 임계를 올려가며 '커버된 예측'의 정확도가 target_acc 이상이 되는 최소 τ 선택."""
    conf, _ = probs.max(1)
    best_tau, best_cov = 1.0, 0.0
    for tau in [i / 100 for i in range(50, 100)]:
        keep = conf >= tau
        if keep.sum() < 10:
            continue
        acc = correct[keep].float().mean().item()
        cov = keep.float().mean().item()
        if acc >= target_acc and cov > best_cov:
            best_cov, best_tau = cov, tau
    return best_tau, best_cov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--val_shards", required=True)
    ap.add_argument("--img", type=int, default=512)
    ap.add_argument("--bsz", type=int, default=32)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--target_acc", type=float, default=0.9)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    vocab = ckpt["vocab"]
    cfg = ckpt["config"]
    model = build_model(cfg["backbone"], len(vocab["species"]), len(vocab["parts"]), pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    loader = make_loader(args.val_shards, vocab, args.img, args.bsz, args.workers,
                         train=False, epoch_samples=50000)
    logits, labels = collect_logits(model, loader, device)
    print(f"수집 {logits.size(0)}개 val 샘플")

    T = fit_temperature(logits, labels)
    probs = F.softmax(logits / T, dim=1)
    correct = probs.argmax(1) == labels
    tau, cov = pick_abstain_tau(probs, correct, args.target_acc)
    print(f"temperature T = {T:.4f}")
    print(f"abstain τ = {tau:.3f}  (커버리지 {cov:.3f} @ 정확도≥{args.target_acc})")

    out = {"temperature": T, "tau_conf": tau, "coverage": cov,
           "target_acc": args.target_acc}
    out_path = os.path.join(os.path.dirname(args.ckpt), "calibration.json")
    json.dump(out, open(out_path, "w", encoding="utf-8"), indent=2)
    print(f"저장 → {out_path}")


if __name__ == "__main__":
    main()
