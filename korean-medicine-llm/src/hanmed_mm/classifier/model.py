"""멀티태스크 분류기 — 교체 가능한 백본(ConvNeXt-V2 / DINOv2) + 3개 헤드.

설계 §8-1: 백본은 교체 가능 인터페이스. 독초 recall 기준으로 baseline 후 선택(ablation).
헤드: species(N-way), part(7-way), toxic(binary). 고해상(448~512px)으로 VLM 384 한계 보완.
"""
from __future__ import annotations

import torch
import torch.nn as nn

# timm 모델 식별자 (교체 지점)
_BACKBONES = {
    "convnextv2": "convnextv2_base.fcmae_ft_in22k_in1k",   # 고해상 CNN, 입력크기 유연
    "dinov2": "vit_base_patch14_dinov2.lvd142m",           # 자가지도 ViT (img_size는 14의 배수, 예 518)
}


class MultiTaskClassifier(nn.Module):
    def __init__(self, backbone: str = "convnextv2", n_species: int = 1000,
                 n_parts: int = 7, pretrained: bool = True, dropout: float = 0.1):
        super().__init__()
        import timm
        model_id = _BACKBONES.get(backbone, backbone)
        # num_classes=0 → pooled feature 추출기.
        kwargs = dict(pretrained=pretrained, num_classes=0)
        # ViT 계열만 dynamic_img_size 필요(pos-embed 보간). ConvNeXt는 완전 conv라 가변입력 기본 지원.
        if "vit" in model_id or "dinov2" in model_id:
            kwargs["dynamic_img_size"] = True
        self.backbone = timm.create_model(model_id, **kwargs)
        d = self.backbone.num_features
        self.dropout = nn.Dropout(dropout)
        self.head_species = nn.Linear(d, n_species)
        self.head_part = nn.Linear(d, n_parts)
        self.head_toxic = nn.Linear(d, 1)
        self.backbone_name = backbone

    def forward(self, x):
        f = self.dropout(self.backbone(x))
        return {
            "species": self.head_species(f),
            "part": self.head_part(f),
            "toxic": self.head_toxic(f).squeeze(-1),   # (B,) logit
        }


def build_model(backbone: str, n_species: int, n_parts: int, pretrained: bool = True) -> MultiTaskClassifier:
    return MultiTaskClassifier(backbone, n_species, n_parts, pretrained)
