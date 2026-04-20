"""Factual hallucination probe for HanMed-LLM (EXP-V4-01 / post-training).

사용:
    # base Bllossom 단독 (EXP-V4-01 baseline)
    .venv/bin/python scripts/probe_factual.py --mode base

    # adapter 로드
    .venv/bin/python scripts/probe_factual.py --mode adapter \
        --adapter outputs/cpt_bllossom/adapter

    # 외부 질문셋 + 결과 jsonl 저장
    .venv/bin/python scripts/probe_factual.py --mode adapter \
        --questions eval/hanmed_eval_v0/T1_factual.jsonl \
        --output outputs/probes/run_YYYYMMDD.jsonl

질문 jsonl 스키마: {"id": str, "question": str, "expected": str (opt)}
결과 jsonl 스키마: {"id", "question", "expected", "response", "n_input_tokens",
                  "n_output_tokens", "latency_sec", "mode", "adapter", "rev"}
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASE = "MLP-KTLim/llama-3-Korean-Bllossom-8B"
DEFAULT_TOKENIZER = REPO_ROOT / "data" / "tokenizer" / "hanmed_bllossom_ext"
DEFAULT_ADAPTER = REPO_ROOT / "outputs" / "cpt_bllossom" / "adapter"
DEFAULT_SYSTEM_PROMPT = (
    "당신은 한의학 고전에 밝은 연구 조교입니다. 사실만, 간결하게 답하세요."
)

FALLBACK_QUESTIONS = [
    {
        "id": "hanmed_v0_Q1",
        "question": "동의보감(東醫寶鑑)을 편찬한 저자는 누구이며, 어느 왕의 명으로 언제 완성되었나요? 한 단락으로 답하세요.",
        "expected": "허준(許浚) / 선조(宣祖)의 명 / 1610년 완성, 1613년 간행",
    },
    {
        "id": "hanmed_v0_Q2",
        "question": "사상의학(四象醫學)을 창시한 인물은 누구이며, 대표 저서는 무엇인가요? 한 단락으로 답하세요.",
        "expected": "이제마(李濟馬) / 『동의수세보원(東醫壽世保元)』 (1894)",
    },
    {
        "id": "hanmed_v0_Q3",
        "question": "향약집성방(鄕藥集成方)은 조선의 어느 왕 시기에 누가 편찬하였나요? 한 단락으로 답하세요.",
        "expected": "세종(世宗) / 유효통·노중례·박윤덕 등 (1433)",
    },
    {
        "id": "hanmed_v0_Q4",
        "question": "한의학에서 말하는 오장(五臟)은 구체적으로 어떤 장부를 가리키나요? 다섯 가지를 모두 나열하세요.",
        "expected": "간(肝) · 심(心) · 비(脾) · 폐(肺) · 신(腎)",
    },
]


def load_questions(path: Path | None) -> list[dict]:
    if path is None:
        return FALLBACK_QUESTIONS
    qs: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            qs.append(json.loads(line))
    if not qs:
        raise SystemExit(f"no questions found in {path}")
    return qs


def build_model(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(args.tokenizer))
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]

    print(f"=== loading base: {args.base} (dtype={args.dtype}) ===", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=dtype)

    if model.get_input_embeddings().num_embeddings != len(tok):
        print(
            f"=== resizing embeddings {model.get_input_embeddings().num_embeddings} -> {len(tok)} ===",
            flush=True,
        )
        model.resize_token_embeddings(len(tok))

    if args.mode == "adapter":
        from peft import PeftModel

        adapter_path = str(args.adapter)
        print(f"=== loading adapter: {adapter_path} ===", flush=True)
        model = PeftModel.from_pretrained(model, adapter_path)

    device = "cuda" if (args.device == "auto" and __import__("torch").cuda.is_available()) else args.device
    model.to(device).eval()

    eot = tok.convert_tokens_to_ids("<|eot_id|>")
    stops = [tok.eos_token_id]
    if eot is not None and eot != tok.eos_token_id:
        stops.append(eot)

    return tok, model, stops, device


def generate_once(tok, model, stops, device, question: str, system: str, max_new: int, rep_penalty: float):
    import torch

    msgs = [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt").input_ids.to(device)

    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            ids,
            max_new_tokens=max_new,
            do_sample=False,
            eos_token_id=stops,
            pad_token_id=tok.pad_token_id,
            repetition_penalty=rep_penalty,
        )
    latency = time.time() - t0

    gen_ids = out[0, ids.shape[1]:]
    response = tok.decode(gen_ids, skip_special_tokens=True).strip()
    return {
        "response": response,
        "n_input_tokens": int(ids.shape[1]),
        "n_output_tokens": int(gen_ids.shape[0]),
        "latency_sec": round(latency, 3),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["base", "adapter"], required=True)
    p.add_argument("--base", default=DEFAULT_BASE)
    p.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    p.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    p.add_argument("--questions", type=Path, default=None, help="jsonl of {id, question, expected?}. omit for fallback 4 Qs.")
    p.add_argument("--output", type=Path, default=None, help="jsonl output path. omit for stdout-only.")
    p.add_argument("--system", default=DEFAULT_SYSTEM_PROMPT)
    p.add_argument("--max-new-tokens", type=int, default=300)
    p.add_argument("--repetition-penalty", type=float, default=1.1)
    p.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    p.add_argument("--device", default="auto")
    p.add_argument("--rev", default="", help="tag to record in output (e.g. adapter checkpoint step, run id)")
    args = p.parse_args()

    if args.mode == "adapter" and not args.adapter.exists():
        raise SystemExit(f"adapter path not found: {args.adapter}")

    questions = load_questions(args.questions)
    print(f"=== probe: mode={args.mode}, N={len(questions)} ===", flush=True)

    tok, model, stops, device = build_model(args)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    results = []
    for i, q in enumerate(questions, 1):
        qid = q.get("id", f"Q{i:03d}")
        qtext = q["question"]
        expected = q.get("expected", "")
        print(f"\n===== {qid} =====\n{qtext}", flush=True)
        try:
            out = generate_once(
                tok, model, stops, device,
                question=qtext, system=args.system,
                max_new=args.max_new_tokens, rep_penalty=args.repetition_penalty,
            )
            print(f"--- response ({out['n_output_tokens']} tok, {out['latency_sec']}s) ---\n{out['response']}", flush=True)
            if expected:
                print(f"--- expected ---\n{expected}", flush=True)
        except Exception as e:
            out = {"response": f"[ERROR] {e.__class__.__name__}: {e}", "n_input_tokens": 0, "n_output_tokens": 0, "latency_sec": 0.0}
            print(f"--- ERROR ---\n{out['response']}", flush=True)

        record = {
            "run_id": run_id,
            "id": qid,
            "question": qtext,
            "expected": expected,
            **out,
            "mode": args.mode,
            "adapter": str(args.adapter) if args.mode == "adapter" else None,
            "rev": args.rev,
            "max_new_tokens": args.max_new_tokens,
            "repetition_penalty": args.repetition_penalty,
        }
        results.append(record)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\n=== wrote {len(results)} results → {args.output} ===", flush=True)

    total_in = sum(r["n_input_tokens"] for r in results)
    total_out = sum(r["n_output_tokens"] for r in results)
    total_t = sum(r["latency_sec"] for r in results)
    mean_out = total_out / max(1, len(results))
    print(
        f"\n=== summary | N={len(results)} | in_tok={total_in} out_tok={total_out} "
        f"mean_out={mean_out:.1f} total_latency={total_t:.1f}s ===",
        flush=True,
    )


if __name__ == "__main__":
    sys.exit(main())
