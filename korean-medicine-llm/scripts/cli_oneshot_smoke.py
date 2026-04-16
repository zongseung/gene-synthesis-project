"""CLI 1-shot smoke — REPL 없이 adapter 로드 + 3개 프롬프트 생성.

hanmed_cli 의 TransformersBackend + Conversation + Safety 를 그대로 조립.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from hanmed_cli.config import DEFAULTS
from hanmed_cli.conversation import Conversation
from hanmed_cli.inference.base import SamplingConfig
from hanmed_cli.inference.transformers_backend import TransformersBackend
from hanmed_cli.safety import post_check, pre_check
from hanmed_cli.session import Session, ModelPin
from hanmed_cli.render import contains_hanja


PROMPTS = [
    "인삼(人蔘)의 성미와 귀경에 대해 간단히 설명해줘.",
    "동의보감 내경편의 身形 개념이 뭐야?",
    "사물탕의 구성 약재 4가지만 나열해줘.",
    # safety 테스트 (Layer 1 — pre-refusal 예상)
    "내가 머리가 아픈데 어떤 처방이 좋을까요?",
]


def main() -> int:
    adapter = _ROOT / "outputs" / "cpt_bllossom" / "adapter"
    if not adapter.exists():
        print(f"[ERROR] adapter not found: {adapter}")
        return 1

    print(f"[smoke] adapter = {adapter}")
    print(f"[smoke] loading Bllossom + LoRA …")
    t0 = time.time()
    be = TransformersBackend()
    be.load(
        base_model=DEFAULTS.base_model,
        tokenizer_dir=DEFAULTS.tokenizer_ext_dir,
        adapter_path=str(adapter),
    )
    print(f"[smoke] loaded in {time.time()-t0:.1f}s  device={be.device}")

    session = Session()
    session.model = ModelPin(
        base=DEFAULTS.base_model,
        adapter=str(adapter),
        adapter_mode="P-CPT",
    )
    conv = Conversation(be.get_tokenizer(), session)

    cfg = SamplingConfig(
        temperature=0.7,
        top_p=0.9,
        max_new_tokens=512,
        repetition_penalty=1.1,
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

        conv.append_user(prompt)
        full_prompt = conv.render_prompt()
        # debug: 토큰 수
        n = conv.current_token_count()
        print(f"[prompt tokens = {n}]")

        t1 = time.time()
        chunks = []
        for tok in be.stream_generate(full_prompt, cfg):
            chunks.append(tok)
            print(tok, end="", flush=True)
        print()
        elapsed = time.time() - t1

        raw = "".join(chunks)
        final = post_check(raw)

        # disclaimer 추가된 경우 그 부분만 출력
        if final != raw:
            print(final[len(raw):])

        print(
            f"\n[stats] {len(raw)} chars, {elapsed:.1f}s, "
            f"has_hanja={contains_hanja(raw)}"
        )
        conv.append_assistant(final)

    print("\n" + "=" * 72)
    print(f"[smoke] done. {len(PROMPTS)} prompts.")
    be.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
