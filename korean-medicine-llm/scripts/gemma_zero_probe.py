"""Gemma-3 12B-IT zero-training probe.

Clinical (임신 4개월 여성) + Q1~Q4 fallback factual. GA/GB/GC 분기용.
결과: outputs/probes/gemma_zero_train_probe.jsonl
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "gemma-3-12b-it"
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


def ask(qid: str, question: str, expected: str, max_tokens: int = 600) -> dict:
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": question},
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "frequency_penalty": 0.3,
        "presence_penalty": 0.2,
    }
    t0 = time.time()
    r = requests.post(URL, json=payload, timeout=300)
    dt = time.time() - t0
    r.raise_for_status()
    d = r.json()
    msg = d["choices"][0]["message"]["content"]
    finish = d["choices"][0].get("finish_reason")
    usage = d.get("usage", {})
    return {
        "id": qid,
        "question": question,
        "expected": expected,
        "response": msg,
        "finish_reason": finish,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "latency_sec": round(dt, 2),
        "params": {k: v for k, v in payload.items() if k != "messages"},
    }


def analyze_clinical(text: str) -> dict:
    """간단 휴리스틱: 변증명·처방명·반복 loop 체크."""
    bian_terms = ["氣血兩虛", "기혈양허", "胎動不安", "태동불안", "血虛", "혈허",
                  "心脾兩虛", "심비양허", "脾胃虛弱", "비위허약", "自汗", "자한",
                  "不眠", "불면", "貧血"]
    formula_terms = ["八珍湯", "팔진탕", "膠艾湯", "교애탕", "當歸芍藥散", "당귀작약산",
                     "安胎飮", "안태음", "補中益氣湯", "보중익기탕", "四物湯", "사물탕",
                     "人蔘養榮湯", "십전대보탕", "歸脾湯", "귀비탕"]
    hits_bian = [t for t in bian_terms if t in text]
    hits_formula = [t for t in formula_terms if t in text]
    # repetition: 15자 이상 구절이 3회 이상 반복
    import re
    words = text.split()
    rep = 0
    if len(text) > 200:
        for i in range(len(text) - 15):
            chunk = text[i:i + 15]
            if text.count(chunk) >= 3:
                rep = text.count(chunk)
                break
    return {
        "bian_hits": hits_bian,
        "formula_hits": hits_formula,
        "has_bian": bool(hits_bian),
        "has_formula": bool(hits_formula),
        "repetition_15gram_count": rep,
    }


def main():
    out_dir = Path("outputs/probes")
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"gemma_zero_train_probe_{run_id}.jsonl"

    print(f"=== Gemma zero-training probe run_id={run_id} ===")
    results = []

    # Clinical
    print("\n[CLINICAL]")
    print(f"Q: {CLINICAL_Q[:80]}...")
    r = ask("CLINICAL_preg4mo", CLINICAL_Q, "변증+처방명 기대", max_tokens=800)
    r["analysis"] = analyze_clinical(r["response"])
    print(f"  finish={r['finish_reason']}  completion_tok={r['completion_tokens']}  t={r['latency_sec']}s")
    print(f"  bian_hits={r['analysis']['bian_hits']}")
    print(f"  formula_hits={r['analysis']['formula_hits']}")
    print(f"  repetition_15gram={r['analysis']['repetition_15gram_count']}")
    print(f"  ---response (first 800 chars)---\n{r['response'][:800]}")
    if len(r["response"]) > 800:
        print(f"  ...({len(r['response']) - 800} more chars)")
    results.append(r)

    # Factual
    for qid, q, exp in FACTUAL:
        print(f"\n[{qid}]")
        print(f"Q: {q}")
        r = ask(qid, q, exp, max_tokens=400)
        print(f"  finish={r['finish_reason']}  completion_tok={r['completion_tokens']}  t={r['latency_sec']}s")
        print(f"  expected: {exp}")
        print(f"  ---response---\n{r['response']}")
        results.append(r)

    with out_path.open("w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n=== wrote {len(results)} results → {out_path} ===")


if __name__ == "__main__":
    main()
