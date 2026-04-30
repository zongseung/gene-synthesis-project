"""ver8.1 § serving · Gemma3-12B-IT + ver8.1 LoRA → merged HF safetensors.

PEFT LoRA 를 base 에 merge 해 단일 HF 모델로 저장. vLLM 서빙에서
`--enable-lora` 없이 바로 로드 가능하게 함.

Phase B (Gemma3) 특이사항:
  - Gemma3-12B-IT 는 multimodal (`Gemma3ForConditionalGeneration`).
    vision_config + text_config 분리. LoRA 는 text tower 의 q/k/v/o + gate/up/down
    만 학습 (`adapter_config.json` target_modules 참조).
  - merge 시 multimodal 전체 weight 로 저장 (vision tower 그대로 + language tower
    LoRA merged). vLLM Gemma3 는 multimodal 모델로 로드하지만 텍스트 추론만 사용.
  - tokenizer 는 base Gemma3 tokenizer 그대로 (Bllossom 같은 ext vocab 없음).
    adapter 디렉토리에 tokenizer.json + chat_template.jinja 가 함께 저장됨 → 그쪽
    우선 사용 (LoRA training 시 사용된 동일 chat_template 보장).

사용:
    PYTHONHASHSEED=0 .venv/bin/python \\
        experiments/dongui_bogam/scripts/build_merged_model_ver8_1.py

기본 입출력:
    base    : models/gemma-3-12b-it
    adapter : experiments/dongui_bogam/outputs_ver8_1_gemma_v1/adapter
    output  : experiments/dongui_bogam/outputs_ver8_1_gemma_v1/merged

권장 자원:
    GPU 1장 (24 GB+) 또는 CPU + 64 GB RAM (CPU merge 는 ~20분 소요).
    저장 후 ~24 GB safetensors.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]  # korean-medicine-llm/
DEFAULT_BASE = ROOT / "models" / "gemma-3-12b-it"
DEFAULT_ADAPTER = ROOT / "experiments" / "dongui_bogam" / "outputs_ver8_1_gemma_v1" / "adapter"
DEFAULT_OUTPUT = ROOT / "experiments" / "dongui_bogam" / "outputs_ver8_1_gemma_v1" / "merged"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, default=DEFAULT_BASE)
    ap.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                    help="merge 위치. cpu 는 안전·느림, cuda 는 24GB+ 필요.")
    args = ap.parse_args()

    if not args.base.exists():
        raise SystemExit(f"base 모델 디렉토리 없음: {args.base}")
    if not args.adapter.exists():
        raise SystemExit(f"adapter 디렉토리 없음: {args.adapter}")
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(
            f"output 디렉토리가 비어있지 않음: {args.output}\n"
            f"덮어쓰려면 먼저 비우세요."
        )

    dtype = {"bfloat16": torch.bfloat16,
             "float16": torch.float16,
             "float32": torch.float32}[args.dtype]
    device_map = "cuda" if args.device == "cuda" else "cpu"

    print(f"[1/5] base 로드 (Gemma3 multimodal): {args.base}")
    base = AutoModelForImageTextToText.from_pretrained(
        args.base, dtype=dtype, device_map=device_map
    )

    print(f"[2/5] tokenizer 로드 (adapter 디렉토리 우선): {args.adapter}")
    # adapter 디렉토리에 tokenizer.json + chat_template.jinja 가 LoRA 학습 시 저장됨.
    # 동일 chat_template 보장을 위해 adapter 의 tokenizer 우선 사용.
    tok_src = args.adapter if (args.adapter / "tokenizer.json").exists() else args.base
    tok = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=True)

    print(f"[3/5] adapter 로드 + merge: {args.adapter}")
    merged = PeftModel.from_pretrained(base, args.adapter)
    merged = merged.merge_and_unload()

    print(f"[4/5] 저장 (safetensors): {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(args.output, safe_serialization=True)
    tok.save_pretrained(args.output)

    print("[5/5] chat_template + multimodal processor 보조 파일 복사")
    # adapter 우선 (학습 시 사용된 chat_template 보장)
    for fname in ("chat_template.jinja", "chat_template.json"):
        src = args.adapter / fname
        if src.exists():
            shutil.copy2(src, args.output / fname)
            print(f"  ✓ {fname} (from adapter)")
    # base 에만 있는 multimodal processor / tokenizer 보조 파일
    # (vLLM 의 Gemma3 multimodal handler 가 이걸 요구 — 누락 시 startup fail)
    for fname in ("preprocessor_config.json", "processor_config.json",
                  "special_tokens_map.json", "added_tokens.json",
                  "tokenizer.model", "generation_config.json"):
        dst = args.output / fname
        if dst.exists():
            continue
        src = args.base / fname
        if src.exists():
            shutil.copy2(src, dst)
            print(f"  ✓ {fname} (from base)")

    print(f"\n✓ merged model: {args.output}")
    print("  next: docker compose -f experiments/dongui_bogam/docker/compose.ver8_1.yml up -d --build")


if __name__ == "__main__":
    main()
