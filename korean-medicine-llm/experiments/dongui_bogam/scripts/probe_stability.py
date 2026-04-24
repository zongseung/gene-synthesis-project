from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

DEFAULT_BASE = "MLP-KTLim/llama-3-Korean-Bllossom-8B"
ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TOKENIZER = ROOT / "data" / "tokenizer" / "hanmed_bllossom_ext"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Probe entity stability across paraphrased fact questions.")
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--fact-variants", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--base", default=DEFAULT_BASE)
    p.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    p.add_argument("--system", default="당신은 한의학 고전 문헌 연구 보조 AI입니다. 질문에 맞는 사실만 답하세요.")
    p.add_argument("--max-new-tokens", type=int, default=220)
    return p.parse_args()


def load_questions(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data["facts"]


def build_model(args):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(args.tokenizer))
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16)
    if model.get_input_embeddings().num_embeddings != len(tok):
        model.resize_token_embeddings(len(tok))
    model = PeftModel.from_pretrained(model, str(args.adapter))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    eot = tok.convert_tokens_to_ids("<|eot_id|>")
    stops = [tok.eos_token_id]
    if eot is not None and eot != tok.eos_token_id:
        stops.append(eot)
    return tok, model, device, stops


def generate(tok, model, device, stops, system: str, question: str, max_new_tokens: int) -> str:
    import torch

    prompt = tok.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True,
    )
    ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        out = model.generate(
            ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=stops,
            pad_token_id=tok.pad_token_id,
            repetition_penalty=1.1,
            no_repeat_ngram_size=6,
        )
    return tok.decode(out[0, ids.shape[1] :], skip_special_tokens=True).strip()


def main() -> int:
    args = parse_args()
    facts = load_questions(args.fact_variants)
    tok, model, device, stops = build_model(args)
    rows = []
    for fact in facts:
        responses = [generate(tok, model, device, stops, args.system, q, args.max_new_tokens) for q in fact["questions"]]
        target = fact["expected_entity"]
        hit_count = sum(1 for response in responses if target in response)
        rows.append(
            {
                "id": fact["id"],
                "expected_entity": target,
                "hit_count": hit_count,
                "stability": round(hit_count / max(len(responses), 1), 4),
                "responses": responses,
            }
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump({"facts": rows, "mean_stability": round(sum(r["stability"] for r in rows) / max(len(rows), 1), 4)}, f, ensure_ascii=False, indent=2)
    print(f"[probe_stability] facts={len(rows)} out={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
