"""Gemma-3 12B-IT zero-training probe via transformers direct (vLLM v0.9.2 bug workaround).

Clinical case + Q1~Q4. GA/GB/GC 분기용. 결과 outputs/probes/gemma_zero_probe_tf_*.jsonl.
Usage:
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/gemma_zero_probe_transformers.py
"""
from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "models/gemma-3-12b-it"

SYSTEM = "당신은 한의학 고전에 밝은 연구 조교입니다. 사실만, 간결하게 답하세요."

CLINICAL_Q = (
    "임신 4개월의 여성이 피곤을 많이 느끼고 다니는 데 힘이 많이 듭니다. "
    "움직임이 많아지면 더 힘들어 합니다. 밥 맛도 별로 없고 때로는 구역감도 느낍니다. "
    "얼굴색이 하얀색으로 보여지고 손톱도 창백합니다. 잠들기가 어려운 경우 등 "
    "수면도 충분하지 않습니다. 때로는 알 수 없는 땀이 흘러 나오는 경우도 있습니다. "
    "이 증상에 대한 변증 등 한의학적 해석과 처방(또는 한약)을 추천해 주세요"
)

FACTUAL = [
    ("Q1_author", "동의보감(東醫寶鑑)을 편찬한 저자는 누구이며, 어느 왕의 명으로 언제 완성되었나요? 한 단락으로 답하세요.",
     "허준(許浚) / 선조(宣祖) / 1610년 완성, 1613년 간행"),
    ("Q2_sasang", "사상의학(四象醫學)을 창시한 인물은 누구이며, 대표 저서는 무엇인가요? 한 단락으로 답하세요.",
     "이제마(李濟馬) / 『동의수세보원』 (1894)"),
    ("Q3_hyangyak", "향약집성방(鄕藥集成方)은 조선의 어느 왕 시기에 누가 편찬하였나요? 한 단락으로 답하세요.",
     "세종(世宗) / 유효통·노중례·박윤덕 (1433)"),
    ("Q4_ojang", "한의학에서 말하는 오장(五臟)은 구체적으로 어떤 장부를 가리키나요? 다섯 가지를 모두 나열하세요.",
     "간(肝) · 심(心) · 비(脾) · 폐(肺) · 신(腎)"),
]


BIAN_TERMS = [
    "氣血兩虛", "기혈양허", "胎動不安", "태동불안", "血虛", "혈허",
    "心脾兩虛", "심비양허", "脾胃虛弱", "비위허약", "氣虛", "기허",
    "自汗", "자한", "不眠", "불면", "貧血", "빈혈", "失眠", "실면",
]
FORMULA_TERMS = [
    "八珍湯", "팔진탕", "膠艾湯", "교애탕", "當歸芍藥散", "당귀작약산",
    "安胎飮", "안태음", "補中益氣湯", "보중익기탕", "四物湯", "사물탕",
    "十全大補湯", "십전대보탕", "歸脾湯", "귀비탕", "人蔘養榮湯", "인삼양영탕",
]


def analyze(text: str) -> dict:
    hits_bian = [t for t in BIAN_TERMS if t in text]
    hits_formula = [t for t in FORMULA_TERMS if t in text]
    # 최장 반복 run: 15자 구절이 얼마나 반복되는지
    rep = 0
    if len(text) > 200:
        for i in range(0, len(text) - 15, 5):
            chunk = text[i:i + 15]
            cnt = text.count(chunk)
            if cnt > rep:
                rep = cnt
            if rep >= 5:
                break
    return {
        "bian_hits": hits_bian,
        "formula_hits": hits_formula,
        "has_bian": bool(hits_bian),
        "has_formula": bool(hits_formula),
        "max_15gram_run": rep,
    }


def gen(model, tok, system: str, question: str, max_new: int = 512) -> dict:
    msgs = [{"role": "user", "content": f"{system}\n\n{question}"}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            ids,
            max_new_tokens=max_new,
            do_sample=False,
            repetition_penalty=1.1,
            pad_token_id=tok.pad_token_id,
        )
    dt = time.time() - t0
    gen_ids = out[0, ids.shape[1]:]
    text = tok.decode(gen_ids, skip_special_tokens=True).strip()
    return {
        "response": text,
        "prompt_tokens": int(ids.shape[1]),
        "completion_tokens": int(gen_ids.shape[0]),
        "latency_sec": round(dt, 2),
    }


def main():
    print(f"=== loading {MODEL_PATH} (bf16, cuda) ===", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cuda:0"
    )
    model.eval()
    print(f"loaded; params ≈ {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B", flush=True)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path("outputs/probes")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"gemma_zero_probe_tf_{run_id}.jsonl"

    results = []

    # Clinical
    print(f"\n[CLINICAL] ({len(CLINICAL_Q)} chars)", flush=True)
    g = gen(model, tok, SYSTEM, CLINICAL_Q, max_new=800)
    g["analysis"] = analyze(g["response"])
    rec = {"id": "CLINICAL_preg4mo", "question": CLINICAL_Q, **g, "expected": "변증명+처방명 기대"}
    results.append(rec)
    print(f"  {g['completion_tokens']} tok / {g['latency_sec']}s", flush=True)
    print(f"  bian_hits={g['analysis']['bian_hits']}", flush=True)
    print(f"  formula_hits={g['analysis']['formula_hits']}", flush=True)
    print(f"  max_15gram_run={g['analysis']['max_15gram_run']}", flush=True)
    print(f"  --- response (first 1200 chars) ---\n{g['response'][:1200]}", flush=True)
    if len(g["response"]) > 1200:
        print(f"  ... ({len(g['response']) - 1200} more chars)", flush=True)

    # Factual
    for qid, q, exp in FACTUAL:
        print(f"\n[{qid}]", flush=True)
        print(f"Q: {q}", flush=True)
        g = gen(model, tok, SYSTEM, q, max_new=300)
        rec = {"id": qid, "question": q, "expected": exp, **g}
        results.append(rec)
        print(f"  {g['completion_tokens']} tok / {g['latency_sec']}s", flush=True)
        print(f"  expected: {exp}", flush=True)
        print(f"  --- response ---\n{g['response']}", flush=True)

    with out_path.open("w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n=== wrote {len(results)} results → {out_path} ===", flush=True)


if __name__ == "__main__":
    main()
