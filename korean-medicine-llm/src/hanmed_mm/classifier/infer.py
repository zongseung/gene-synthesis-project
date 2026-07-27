"""추론 — 이미지 → ClassifierResult (설계 pipeline/types 계약과 정렬).

best_model.pth + calibration.json 로드. species 확률은 온도보정 적용.
topk는 (species_ko, prob). 학명(scientific_name) 피벗은 파이프라인 crosswalk 단계에서.
is_ood/toxic_prob/confidence 제공 → safety_gate가 abstain/toxic 판정.

사용(파이썬):
  clf = Classifier("outputs/clf_run1/best_model.pth")
  res = clf.predict("some.jpg")   # → dict(topk, part, toxic_prob, confidence, is_ood)
"""
from __future__ import annotations
import json
import os

import torch
import torch.nn.functional as F
from PIL import Image

from hanmed_mm.classifier.dataset import build_transform
from hanmed_mm.classifier.labels import PARTS
from hanmed_mm.classifier.model import build_model


class Classifier:
    def __init__(self, ckpt_path: str, img_size: int | None = None, device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.vocab = ckpt["vocab"]
        cfg = ckpt["config"]
        self.img_size = img_size or cfg.get("img", 512)
        self.model = build_model(cfg["backbone"], len(self.vocab["species"]),
                                 len(self.vocab["parts"]), pretrained=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device).eval()
        self.tf = build_transform(self.img_size, train=False)
        self.species_inv = {i: s for s, i in self.vocab["species"].items()}
        self.part_inv = {i: p for p, i in self.vocab["parts"].items()}

        # 보정값
        calib_path = os.path.join(os.path.dirname(ckpt_path), "calibration.json")
        if os.path.exists(calib_path):
            c = json.load(open(calib_path, encoding="utf-8"))
            self.T = c.get("temperature", 1.0)
            self.tau_conf = c.get("tau_conf", 0.0)
        else:
            self.T, self.tau_conf = 1.0, 0.0

    @torch.no_grad()
    def predict(self, image, topk: int = 5) -> dict:
        if isinstance(image, (str, os.PathLike)):
            image = Image.open(image)
        x = self.tf(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = self.model(x)
        sp_prob = F.softmax(out["species"].float() / self.T, dim=1)[0]
        conf, _ = sp_prob.max(0)
        k = min(topk, sp_prob.numel())
        pv, pi = sp_prob.topk(k)
        part_idx = int(out["part"].float().argmax(1)[0])
        toxic_prob = float(out["toxic"].float().sigmoid()[0])
        confidence = float(conf)
        return {
            "topk": [(self.species_inv[int(i)], float(p)) for p, i in zip(pv, pi)],
            "part": self.part_inv.get(part_idx, "unknown"),
            "toxic_prob": toxic_prob,
            "confidence": confidence,
            "is_ood": confidence < self.tau_conf,   # 임계 미만 → abstain 후보
        }
