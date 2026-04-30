"""ver8.1 RAG smoke — book_008 FAISS retrieval + Gemma3-12B + ver8.1 LoRA.

이전 9-question (smoke 4 + grounded 5 + strengths 5 = 14) 를 RAG context 와 함께
재실행해 환각률 비교.

실행:
    cd korean-medicine-llm
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/probe_ver8_1_rag.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import faiss
from peft import PeftModel
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments" / "dongui_bogam" / "src"))

from hanmed_cli.safety import post_check, pre_check  # noqa: E402

BASE_DIR = ROOT / "models" / "gemma-3-12b-it"
ADAPTER = ROOT / "experiments" / "dongui_bogam" / "outputs_ver8_1_gemma_v1" / "adapter"
INDEX = ROOT / "data" / "rag" / "book_008.index"
META = ROOT / "data" / "rag" / "book_008.meta.jsonl"
EMB_MODEL = "BAAI/bge-m3"

# 14 question = smoke 4 + grounded 5 + strengths 5 (이전 probe 들과 동일 set)
QUESTIONS = [
    # === smoke 4 ===
    ("Q1", "인삼(人蔘)의 성미와 귀경에 대해 간단히 설명해줘."),
    ("Q2", "동의보감 내경편의 身形 개념이 뭐야?"),
    ("Q3", "사물탕의 구성 약재 4가지만 나열해줘."),
    ("Q4", "내가 머리가 아픈데 어떤 처방이 좋을까요?"),
    # === grounded 5 ===
    ("D1", "동의보감 내경편의 '통설산(通泄散)' 은 어떤 증상에 쓰는 처방이며 구성 약재는 무엇인가요?"),
    ("D2", "유옹(乳癰) 에 쓰는 '단삼고(丹蔘膏)' 의 조성과 적응증을 알려주세요."),
    ("D3", "동의보감에서 중풍으로 가래가 심하고 약이 듣지 않을 때 어떤 침구 처치를 권하나요?"),
    ("D4", "동의보감 소아문의 '진경환(鎭驚丸)' 은 어떤 증에 쓰며 구성 약재는 무엇인가요?"),
    ("D5", "허로로 양기가 부족할 때 쓰는 '증손낙령탕(增損樂令湯)' 의 구성 약재를 알려주세요."),
    # === strengths 5 ===
    ("S1", "동의보감의 다섯 편(內景·外形·雜病·湯液·鍼灸) 의 큰 구성과 각 편이 다루는 주제를 짧게 정리해 주세요."),
    ("S2", "보중익기탕(補中益氣湯) 은 동의보감의 어느 편·문에 수록된 처방이며 주된 효능은 무엇인가요?"),
    ("S3", "동의보감은 누가 언제 편찬했고, 어떤 시대적 배경에서 만들어졌나요?"),
    ("S4", "'內景篇(내경편)' 이라는 편명의 뜻과 동의보감에서 이 편이 다루는 범주를 설명해 주세요."),
    ("S5", "다음 동의보감 본문 대목의 핵심 의미를 짚어 주세요.\n발췌: 醫者雅言軒岐, 軒岐上窮天紀, 下極人理, 宜不屑乎記述."),
]

K = 5  # top-k retrieval

SYSTEM_RAG = (
    "당신은 한의학 고전 문헌 연구 보조 AI 입니다. 사용자 질문에 답할 때, 아래 "
    "[동의보감 발췌] 의 내용에만 근거하여 답하세요. 발췌에 없는 처방·약재·편명·"
    "인용은 만들지 마세요. 답변 시 [1], [2] 같은 인용 표기로 어느 발췌를 근거로 "
    "했는지 표시하세요. 발췌 자료에서 확인할 수 없으면 '발췌 자료에서는 확인할 "
    "수 없습니다' 라고 답하세요."
)


def load_meta():
    with open(META, "r", encoding="utf-8") as fp:
        return [json.loads(ln) for ln in fp]


def main():
    print("=== load FAISS index + meta ===")
    t0 = time.time()
    index = faiss.read_index(str(INDEX))
    meta = load_meta()
    print(f"  index ntotal = {index.ntotal:,}, meta {len(meta):,} records ({time.time()-t0:.1f}s)")

    print("\n=== load embedding model (BGE-M3) ===")
    t0 = time.time()
    encoder = SentenceTransformer(EMB_MODEL, device="cuda:0")
    print(f"  loaded in {time.time()-t0:.1f}s")

    print("\n=== load Gemma3-12B + ver8.1 LoRA adapter ===")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(str(BASE_DIR))
    base = AutoModelForCausalLM.from_pretrained(
        str(BASE_DIR), dtype=torch.bfloat16, device_map={"": "cuda:0"}
    )
    model = PeftModel.from_pretrained(base, str(ADAPTER)).eval()
    print(f"  loaded in {time.time()-t0:.1f}s")

    gen_kwargs = dict(
        max_new_tokens=500,
        do_sample=True,
        temperature=0.5,
        top_p=0.9,
        repetition_penalty=1.2,
        no_repeat_ngram_size=8,
    )

    for qid, query in QUESTIONS:
        print("\n" + "=" * 80)
        print(f"[{qid}] {query}")
        print("-" * 80)

        # safety pre-check
        pre = pre_check(query)
        if pre.refused:
            print("⚠ Pre-safety refusal triggered:")
            print(pre.refusal_text)
            continue

        # retrieve top-k
        qe = encoder.encode([query], normalize_embeddings=True,
                            convert_to_numpy=True).astype("float32")
        D, I = index.search(qe, K)
        sources = []
        print(f"[retrieved top-{K}]")
        for rank, idx in enumerate(I[0]):
            r = meta[idx]
            sim = D[0][rank]
            body = (r["trans_ko"] or r["original"]).replace("\n", " ").strip()
            sources.append({
                "rank": rank + 1,
                "id": r["id"],
                "path": r["up_path_nm"],
                "level": r["content_level"],
                "body": body,
                "sim": float(sim),
            })
            preview = body[:60]
            print(f"  [{rank+1}] sim={sim:.3f} {r['up_path_nm']}  →  {preview}")

        # compose RAG prompt
        excerpt_block = "\n\n".join([
            f"[{s['rank']}] {s['path']} ({s['level']})\n{s['body']}"
            for s in sources
        ])
        user_msg = (
            "[동의보감 발췌]\n"
            f"{excerpt_block}\n\n"
            f"[질문] {query}"
        )

        # generate
        msgs = [
            {"role": "system", "content": SYSTEM_RAG},
            {"role": "user", "content": user_msg},
        ]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tok(prompt, return_tensors="pt").to(model.device)
        n_in = ids.input_ids.shape[-1]

        t1 = time.time()
        with torch.inference_mode():
            out = model.generate(**ids, pad_token_id=tok.pad_token_id, **gen_kwargs)
        text = tok.decode(out[0, n_in:], skip_special_tokens=True).strip()
        elapsed = time.time() - t1

        # safety post-check
        final = post_check(text)

        print(f"\n[answer in {elapsed:.1f}s, prompt={n_in} tokens]")
        print(final)
        if final != text:
            print("\n[※ post_check 마스킹 적용됨]")

    print("\n" + "=" * 80)
    print("[probe-rag] done")


if __name__ == "__main__":
    main()
