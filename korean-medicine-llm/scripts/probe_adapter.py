"""Stage 1 CPT 후 base vs adapter 샘플 비교 (single-device · disable_adapter 경로).

동일 GPU 에 base 를 한 번만 로드하고 PEFT adapter 를 attach 한 뒤,
`with model.disable_adapter():` 로 BASE continuation 을, 그대로 두고
CPT continuation 을 측정한다. 두 경로가 동일 파라미터 메모리 · 동일 RNG · 동일
토크나이저를 공유하므로 이중 로드(GPU0/GPU1) 대비 수치적으로 공정한 A/B 가 된다.

SFT 이전이므로 "질문-답변" 포맷 기대 금지. CPT 는 continuation.
"""
from __future__ import annotations

import os
import sys
import torch

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

from transformers import AutoTokenizer, AutoModelForCausalLM  # noqa: E402
from peft import PeftModel  # noqa: E402

BASE = "MLP-KTLim/llama-3-Korean-Bllossom-8B"
TOK_DIR = "data/tokenizer/hanmed_bllossom_ext"
ADAPTER = "outputs/cpt_bllossom/adapter"
DEVICE = "cuda:0"

PROMPTS = [
    "한의학에서 사상체질이란 ",
    "陰陽五行이란 ",
    "補中益氣湯은 ",
    "傷寒論에서 말하기를, ",
    "사상체질의학의 창시자 이제마는 ",
    "<KO>오장육부</KO>는 ",
    "한의학의 기본 이론인 기혈(氣血)은 ",
]

GEN = dict(
    max_new_tokens=120,
    do_sample=False,  # greedy: deterministic A/B 비교
    repetition_penalty=1.1,
)


def load_model_with_adapter(device: str):
    print(f"[load base + adapter → {device}]", flush=True)
    tok = AutoTokenizer.from_pretrained(TOK_DIR)
    m = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16).to(device)
    m.resize_token_embeddings(len(tok), mean_resizing=False)
    m = PeftModel.from_pretrained(m, ADAPTER)
    m.eval()
    return m, tok


def gen(model, tok, prompt: str, device: str) -> str:
    ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        out = model.generate(ids, pad_token_id=tok.pad_token_id, **GEN)
    return tok.decode(out[0], skip_special_tokens=False)


def main():
    torch.manual_seed(42)
    model, tok = load_model_with_adapter(DEVICE)
    print("\n" + "=" * 78)
    for i, p in enumerate(PROMPTS, 1):
        print(f"\n--- [{i}/{len(PROMPTS)}] prompt: {p!r}")
        # BASE: adapter 일시 비활성화 (동일 weight · 동일 tokenizer)
        torch.manual_seed(42 + i)
        with model.disable_adapter():
            out_b = gen(model, tok, p, DEVICE)
        # CPT: adapter 활성 상태
        torch.manual_seed(42 + i)
        out_a = gen(model, tok, p, DEVICE)
        print("[BASE]")
        print(out_b)
        print("[CPT]")
        print(out_a)
    print("\n" + "=" * 78)


if __name__ == "__main__":
    sys.exit(main())
