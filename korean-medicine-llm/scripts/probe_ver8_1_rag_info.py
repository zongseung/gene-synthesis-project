"""ver8.1 RAG — INFO mode prototype (yaml-driven).

진료성 query (예: "내가 두통이 있는데 사물탕 먹어도 돼?") 에 대해 STRICT 거부
대신 본문 매핑 사실만 안내하는 모드. v4 hybrid retrieval 재사용.

System prompt · fewshots · test queries 는 외부 yaml 에서 로드:
  experiments/dongui_bogam/eval/info_mode_prompts.yaml

핵심 룰 (yaml system 에 명시):
  1. 임상 판정 금지.
  2. 본문 매핑 사실만 — [발췌] 안의 글자만 사용.
  3. 3-tier 매칭 — ✓ direct / △ via variant / ✗.
  4. 4-section 답변 형식 + 면책 footer 강제.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import faiss
import torch
import yaml
from peft import PeftModel
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments" / "dongui_bogam" / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

# v4 의 retrieval 로직 재사용 (한자 정규화 + 양방향 alias + NAME_RE 타이트)
from probe_ver8_1_rag_v4 import (  # noqa: E402
    EMB_MODEL, INDEX, META, BASE_DIR, ADAPTER, K,
    hybrid_search,
)
from hanmed_cli.safety import post_check  # noqa: E402

PROMPTS_YAML = ROOT / "experiments" / "dongui_bogam" / "eval" / "info_mode_prompts.yaml"


def load_prompts(path: Path) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    required = {"system", "fewshots", "test_queries"}
    missing = required - cfg.keys()
    if missing:
        raise ValueError(f"{path} missing keys: {missing}")
    if not cfg["fewshots"]:
        raise ValueError(f"{path} fewshots is empty")
    for i, fs in enumerate(cfg["fewshots"]):
        if "user" not in fs or "assistant" not in fs:
            raise ValueError(f"{path} fewshots[{i}] missing user/assistant")
    return cfg


def build_messages(system: str, fewshots: list[dict], user_msg: str) -> list[dict]:
    msgs = [{"role": "system", "content": system}]
    for fs in fewshots:
        msgs.append({"role": "user", "content": fs["user"].strip()})
        msgs.append({"role": "assistant", "content": fs["assistant"].strip()})
    msgs.append({"role": "user", "content": user_msg})
    return msgs


def main():
    cfg = load_prompts(PROMPTS_YAML)
    print(f"[prompts] loaded {PROMPTS_YAML.name}: "
          f"{len(cfg['fewshots'])} fewshots, {len(cfg['test_queries'])} queries")

    print("\n=== load FAISS + meta + encoder ===")
    t0 = time.time()
    index = faiss.read_index(str(INDEX))
    meta = [json.loads(l) for l in open(META)]
    encoder = SentenceTransformer(EMB_MODEL, device="cuda:0")
    print(f"  loaded ({time.time()-t0:.1f}s)")

    print("\n=== load Gemma3-12B + ver8.1 LoRA ===")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(str(BASE_DIR))
    base = AutoModelForCausalLM.from_pretrained(
        str(BASE_DIR), dtype=torch.bfloat16, device_map={"": "cuda:0"}
    )
    model = PeftModel.from_pretrained(base, str(ADAPTER)).eval()
    print(f"  loaded in {time.time()-t0:.1f}s")

    gen_kwargs = dict(
        max_new_tokens=500,
        do_sample=False,
        repetition_penalty=1.1,
        no_repeat_ngram_size=8,
    )

    for qi, query in enumerate(cfg["test_queries"], 1):
        print("\n" + "=" * 80)
        print(f"[INFO-{qi}] {query}")
        print("-" * 80)

        # NOTE: 의도적으로 pre_check 우회 (INFO mode prototype 검증).
        # 실제 통합 시엔 pre_check 가 mode 분기를 반환하도록 확장 필요.

        results, names = hybrid_search(query, encoder, index, meta, k=K)
        print(f"[hybrid retrieved top-{K}]  extracted_names={names[:6]}")
        for rank, item in enumerate(results, 1):
            r = item["rec"]
            body = (r["trans_ko"] or r["original"]).replace("\n", " ").strip()
            tag = f"{item['src']}/sim={item['sim']:.3f}"
            print(f"  [{rank}] {tag} {r['up_path_nm']}  →  {body[:60]}")

        excerpt_block = "\n\n".join([
            f"[{i+1}] {item['rec']['up_path_nm']} ({item['rec']['content_level']})\n"
            f"{(item['rec']['trans_ko'] or item['rec']['original']).strip()}"
            for i, item in enumerate(results)
        ])
        user_msg = (
            "[동의보감 발췌]\n\n"
            f"{excerpt_block}\n\n"
            f"[질문] {query}"
        )

        msgs = build_messages(cfg["system"], cfg["fewshots"], user_msg)
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tok(prompt, return_tensors="pt").to(model.device)
        n_in = ids.input_ids.shape[-1]

        t1 = time.time()
        with torch.inference_mode():
            out = model.generate(**ids, pad_token_id=tok.pad_token_id, **gen_kwargs)
        text = tok.decode(out[0, n_in:], skip_special_tokens=True).strip()
        elapsed = time.time() - t1

        final = post_check(text)
        print(f"\n[answer in {elapsed:.1f}s, prompt={n_in} tokens, "
              f"INFO mode, {len(cfg['fewshots'])} fewshots]")
        print(final)

    print("\n" + "=" * 80)
    print("[probe-rag-info] done")


if __name__ == "__main__":
    main()
