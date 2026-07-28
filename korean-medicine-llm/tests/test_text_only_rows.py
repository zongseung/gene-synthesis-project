"""S5 — 1차 replay 용 텍스트 전용 행이 Dataset → Collator 를 통과한다.

이미지 블록이 붙지 않아야 하고(pixel_values 없음), 답변 토큰은 학습 대상으로 남아야 한다.
"""
import json
import os

import pytest

from hanmed.training.train import Collator, UnifiedMMDataset

BASE = "models/VARCO-VISION-2.0-14B"


def _cfg(tmp_path, rows):
    p = tmp_path / "text.jsonl"
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    return {
        "tongue_train": str(empty), "tongue_val": str(empty), "tongue_image_root": "",
        "herb_train": str(empty), "herb_val": str(empty), "herb_image_root": "",
        "text_train": str(p), "text_val": str(p),
    }


TEXT_ROW = {"conversations": [
    {"from": "human", "value": "동의보감 탕액편 > 인삼의 기록은?"},
    {"from": "gpt", "value": "성질은 약간 따뜻하고 맛은 달며 독이 없다."},
]}


def test_text_only_row_has_no_image_block(tmp_path):
    ds = UnifiedMMDataset(_cfg(tmp_path, [TEXT_ROW]), "train")
    item = ds[0]
    assert item["image"] is None
    types = [c["type"] for c in item["messages"][0]["content"]]
    assert "image" not in types


@pytest.mark.skipif(not os.path.exists(BASE), reason="베이스 모델 없음")
def test_text_only_batch_has_no_pixel_values(tmp_path):
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(BASE)
    if proc.tokenizer.pad_token_id is None:
        proc.tokenizer.pad_token = proc.tokenizer.eos_token
    ds = UnifiedMMDataset(_cfg(tmp_path, [TEXT_ROW]), "train")
    batch = Collator(proc)([ds[0]])
    assert "pixel_values" not in batch
    assert (batch["labels"] != -100).sum() > 0


@pytest.mark.skipif(not os.path.exists(BASE), reason="베이스 모델 없음")
def test_image_row_is_not_truncated(tmp_path):
    """이미지 행을 자르면 image 토큰 수가 anyres 그리드와 어긋나 processor 가 죽는다 (D3)."""
    from PIL import Image
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(BASE)
    if proc.tokenizer.pad_token_id is None:
        proc.tokenizer.pad_token = proc.tokenizer.eos_token
    item = {
        "image": Image.new("RGB", (384, 384), (128, 128, 128)),
        "messages": [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "이 약초는?"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "칡입니다."}]},
        ],
    }
    batch = Collator(proc, max_seq_len=64)([item])
    assert batch["input_ids"].shape[1] > 64
