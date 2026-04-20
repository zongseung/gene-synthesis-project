"""ver4 §03.01 · adapter → merged model 빌드.

PEFT LoRA 를 base 에 merge 해 단일 HF safetensors 모델로 저장.
vLLM 서빙에서 --enable-lora 없이 바로 로드 가능하게 함.

사용:
    PYTHONHASHSEED=0 .venv/bin/python scripts/build_merged_model.py \
        --adapter outputs/cpt_bllossom/best_model \
        --output outputs/hanmed_merged_v0.1

§5.3 (3) best_model_checkpoint 활성 상태에서 생성된 체크포인트를 어댑터 입력으로 사용.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="MLP-KTLim/llama-3-Korean-Bllossom-8B")
    ap.add_argument("--adapter", type=Path, required=True,
                    help="PEFT adapter 디렉토리 (e.g., outputs/cpt_bllossom/best_model)")
    ap.add_argument("--tokenizer", type=Path, default=Path("data/tokenizer/hanmed_bllossom_ext"))
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    args = ap.parse_args()

    if not args.adapter.exists():
        raise SystemExit(f"adapter 디렉토리 없음: {args.adapter}")

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

    print(f"[1/4] base 로드: {args.base}")
    base = AutoModelForCausalLM.from_pretrained(args.base, dtype=dtype, device_map="cpu")

    print(f"[2/4] tokenizer 로드 (ext vocab): {args.tokenizer}")
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    base.resize_token_embeddings(len(tok))

    print(f"[3/4] adapter 로드 + merge: {args.adapter}")
    merged = PeftModel.from_pretrained(base, args.adapter)
    merged = merged.merge_and_unload()

    print(f"[4/4] 저장: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(args.output, safe_serialization=True)
    tok.save_pretrained(args.output)

    print(f"✓ merged model: {args.output}")
    print("  next: cd docker && docker compose up -d --build")


if __name__ == "__main__":
    main()
