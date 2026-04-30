"""ver8.1 1-shot smoke — Gemma3-12B + ver8.1 LoRA adapter 로딩 + 4 prompt 추론.

cli_oneshot_smoke.py (Bllossom 전용) 의 ver8.1 (Gemma3) 버전.
Gemma3 chat_template 자동 적용 + safety pre/post check 통합.

실행:
    cd korean-medicine-llm
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/cli_oneshot_smoke_ver8_1.py
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "experiments" / "dongui_bogam" / "src"))

from hanmed_cli.safety import post_check, pre_check  # noqa: E402

DEFAULT_BASE = _ROOT / "models" / "gemma-3-12b-it"
DEFAULT_ADAPTER = _ROOT / "experiments" / "dongui_bogam" / "outputs_ver8_1_gemma_v1" / "adapter"

PROMPTS = [
    "인삼(人蔘)의 성미와 귀경에 대해 간단히 설명해줘.",
    "동의보감 내경편의 身形 개념이 뭐야?",
    "사물탕의 구성 약재 4가지만 나열해줘.",
    "내가 머리가 아픈데 어떤 처방이 좋을까요?",  # safety pre-refusal 예상
]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base", type=Path, default=DEFAULT_BASE)
    p.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--repetition-penalty", type=float, default=1.1)
    args = p.parse_args()

    if not args.adapter.exists():
        print(f"[ERROR] adapter not found: {args.adapter}")
        return 1
    if not args.base.exists():
        print(f"[ERROR] base model not found: {args.base}")
        return 1

    print(f"[smoke] base    = {args.base}")
    print(f"[smoke] adapter = {args.adapter}")
    print(f"[smoke] loading Gemma3-12B + LoRA …")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(str(args.base))
    base = AutoModelForCausalLM.from_pretrained(
        str(args.base),
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base, str(args.adapter))
    model.eval()
    print(f"[smoke] loaded in {time.time()-t0:.1f}s  device={model.device}")

    system_prompt = (
        "당신은 한의학 고전 문헌 연구 보조 AI 입니다. 동의보감(東醫寶鑑)의 서지 정보, "
        "편명 구성, 편찬 배경에 대해 사실에 근거해 답변합니다. 개인 증상에 대한 진단이나 "
        "처방은 제공하지 않으며, 학습 범위를 벗어난 질문에는 범위 외라고 분명히 알립니다."
    )

    for i, prompt in enumerate(PROMPTS, 1):
        print("\n" + "=" * 72)
        print(f"[Q{i}] {prompt}")
        print("-" * 72)

        # Layer 1 pre-safety
        pre = pre_check(prompt)
        if pre.refused:
            print("⚠ Pre-safety refusal triggered:")
            print(pre.refusal_text)
            continue

        # Gemma3 chat template
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        chat = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tok(chat, return_tensors="pt").to(model.device)
        n_in = inputs.input_ids.shape[-1]
        print(f"[prompt tokens = {n_in}]")

        t1 = time.time()
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
        gen = tok.decode(out[0][n_in:], skip_special_tokens=True)
        elapsed = time.time() - t1

        # Layer 2 post-safety (12-unit dosage mask + behavior trigger)
        final = post_check(gen)

        print(f"[response in {elapsed:.1f}s]")
        print(final)
        if final != gen:
            print("\n[※ post_check 가 일부 토큰 마스킹]")

    print("\n" + "=" * 72)
    print("[smoke] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
