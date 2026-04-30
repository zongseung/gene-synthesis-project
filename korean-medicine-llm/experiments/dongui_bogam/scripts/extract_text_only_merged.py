"""multimodal merged model (Gemma3ForConditionalGeneration) → text-only Gemma3ForCausalLM.

merge 된 multimodal model 의 weight 에서 language tower 만 추출하고
prefix `model.language_model.*` → `model.*` 로 rename, config 도 text-only
형태로 재작성. vLLM 의 Gemma3 multimodal handler 우회 (image_token_id 누락 버그).

사용:
    .venv/bin/python experiments/dongui_bogam/scripts/extract_text_only_merged.py \\
        --src experiments/dongui_bogam/outputs_ver8_2_gemma_v1/merged \\
        --dst experiments/dongui_bogam/outputs_ver8_2_gemma_v1/merged_text
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SRC = ROOT / "experiments" / "dongui_bogam" / "outputs_ver8_1_gemma_v1" / "merged"
DEFAULT_DST = ROOT / "experiments" / "dongui_bogam" / "outputs_ver8_1_gemma_v1" / "merged_text"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC,
                    help="merged multimodal model 디렉토리 (build_merged_model_ver8_1.py 산출).")
    ap.add_argument("--dst", type=Path, default=DEFAULT_DST,
                    help="text-only 추출본 저장 디렉토리.")
    args = ap.parse_args()
    SRC = args.src
    DST = args.dst

    if not SRC.exists():
        raise SystemExit(f"source 디렉토리 없음: {SRC}")
    DST.mkdir(parents=True, exist_ok=True)

    # ── 1. config.json — text-only ────────────────────────────────────────
    mm_cfg = json.load(open(SRC / "config.json"))
    text_cfg = dict(mm_cfg.get("text_config") or {})
    text_cfg["architectures"] = ["Gemma3ForCausalLM"]
    text_cfg["model_type"] = "gemma3_text"
    if "torch_dtype" not in text_cfg:
        text_cfg["torch_dtype"] = mm_cfg.get("dtype") or mm_cfg.get("torch_dtype") or "bfloat16"
    if "transformers_version" in mm_cfg and "transformers_version" not in text_cfg:
        text_cfg["transformers_version"] = mm_cfg["transformers_version"]
    json.dump(text_cfg, open(DST / "config.json", "w"),
              indent=2, ensure_ascii=False)
    print(f"[1/3] config.json → text-only ({text_cfg['architectures']})")

    # ── 2. weight key rename ──────────────────────────────────────────────
    print("[2/3] weight rename: model.language_model.* → model.*")
    new_state = {}
    skip_prefixes = ("model.vision_tower.", "model.multi_modal_projector.")
    with safe_open(SRC / "model.safetensors", framework="pt") as f:
        keys = list(f.keys())
        n_total = len(keys)
        n_skip = 0
        for k in keys:
            if any(k.startswith(p) for p in skip_prefixes):
                n_skip += 1
                continue
            if k.startswith("model.language_model."):
                # multimodal weight 의 prefix 가 `model.language_model.model.X` 형태이므로
                # `model.language_model.` 를 통째로 제거하면 text-only 형태인 `model.X` 로 정확히 매핑됨.
                # (이전 bug: "model." + "model.X" = "model.model.X" 가 됨)
                new_k = k[len("model.language_model."):]
            else:
                new_k = k  # lm_head 등 root 위치 weight
            new_state[new_k] = f.get_tensor(k)
        print(f"  total={n_total}, kept={len(new_state)}, skipped(vision/proj)={n_skip}")

    save_file(new_state, DST / "model.safetensors", metadata={"format": "pt"})
    print(f"  saved {DST / 'model.safetensors'}")

    # ── 3. tokenizer / chat_template / generation_config 그대로 복사 ──────
    print("[3/3] tokenizer + chat_template 복사")
    for fname in ("tokenizer.json", "tokenizer.model", "tokenizer_config.json",
                  "special_tokens_map.json", "added_tokens.json",
                  "chat_template.jinja", "chat_template.json",
                  "generation_config.json"):
        src_f = SRC / fname
        if src_f.exists():
            shutil.copy2(src_f, DST / fname)
            print(f"  ✓ {fname}")

    print(f"\n✓ text-only merged: {DST}")
    print(f"  next: HANMED_MERGED_DIR={DST} \\")
    print("        docker compose -f docker/compose.ver8_1.yml up -d --force-recreate hanmed_vllm_ver8_1")


if __name__ == "__main__":
    main()
