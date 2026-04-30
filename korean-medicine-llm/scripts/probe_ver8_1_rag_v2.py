"""ver8.1 RAG smoke v2 — temperature 0.1 + 강화 prompt + 1-shot example.

v1 (probe_ver8_1_rag.py) 결과:
  - D2~D5: 정답 (RAG 효과)
  - Q3 사물탕: retrieved 정답인데 모델 무시 (학습 prior 우위)
  - S4 內景篇: retrieved 빈약 → 환각 cascade

v2 변경점:
  1. do_sample=False (greedy)  ← retrieved 토큰 우선
  2. SYSTEM_RAG 강화: "글자 그대로 옮길 것, 한 글자라도 변형 금지"
  3. 약재 목록 형식 강제: "발췌의 ㆍ , 로 분리된 토큰만 나열"
  4. 1-shot example (D2 단삼고 형식)
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

QUESTIONS = [
    ("Q1", "인삼(人蔘)의 성미와 귀경에 대해 간단히 설명해줘."),
    ("Q2", "동의보감 내경편의 身形 개념이 뭐야?"),
    ("Q3", "사물탕의 구성 약재 4가지만 나열해줘."),
    ("Q4", "내가 머리가 아픈데 어떤 처방이 좋을까요?"),
    ("D1", "동의보감 내경편의 '통설산(通泄散)' 은 어떤 증상에 쓰는 처방이며 구성 약재는 무엇인가요?"),
    ("D2", "유옹(乳癰) 에 쓰는 '단삼고(丹蔘膏)' 의 조성과 적응증을 알려주세요."),
    ("D3", "동의보감에서 중풍으로 가래가 심하고 약이 듣지 않을 때 어떤 침구 처치를 권하나요?"),
    ("D4", "동의보감 소아문의 '진경환(鎭驚丸)' 은 어떤 증에 쓰며 구성 약재는 무엇인가요?"),
    ("D5", "허로로 양기가 부족할 때 쓰는 '증손낙령탕(增損樂令湯)' 의 구성 약재를 알려주세요."),
    ("S1", "동의보감의 다섯 편(內景·外形·雜病·湯液·鍼灸) 의 큰 구성과 각 편이 다루는 주제를 짧게 정리해 주세요."),
    ("S2", "보중익기탕(補中益氣湯) 은 동의보감의 어느 편·문에 수록된 처방이며 주된 효능은 무엇인가요?"),
    ("S3", "동의보감은 누가 언제 편찬했고, 어떤 시대적 배경에서 만들어졌나요?"),
    ("S4", "'內景篇(내경편)' 이라는 편명의 뜻과 동의보감에서 이 편이 다루는 범주를 설명해 주세요."),
    ("S5", "다음 동의보감 본문 대목의 핵심 의미를 짚어 주세요.\n발췌: 醫者雅言軒岐, 軒岐上窮天紀, 下極人理, 宜不屑乎記述."),
]

K = 5

SYSTEM_RAG = (
    "당신은 한의학 고전 문헌 연구 보조 AI 입니다. 사용자 질문에 답할 때 다음 규칙을 "
    "엄격히 지키세요.\n\n"
    "1. [동의보감 발췌] 안의 글자만 사용해서 답하세요. 발췌에 없는 약재명·처방명·"
    "편명·인용·약재 개수는 절대 만들지 마세요.\n"
    "2. 약재 이름·처방 이름은 발췌의 표기를 한 글자도 바꾸지 말고 그대로 옮기세요. "
    "한자를 다른 한자로 변환하거나 새로 만드는 것을 금지합니다.\n"
    "3. 약재 목록을 답할 때는 발췌의 'ㆍ' 또는 ',' 로 구분된 토큰만 그대로 나열하세요. "
    "발췌에 없는 약재를 추가하지 마세요.\n"
    "4. 발췌만으로 답할 수 없으면 '발췌 자료에서는 확인할 수 없습니다' 라고 답하세요. "
    "추측하지 마세요.\n"
    "5. 답변 끝에 [1], [2] 형태로 어느 발췌를 근거로 했는지 표시하세요."
)

# Few-shot example (D2 단삼고 정답 형태)
FEWSHOT_USER = """[동의보감 발췌]

[1] 外形篇卷之三 > 乳 > 乳癰 > 丹參膏 (SS)
유옹으로 멍울이 생겨 찌르듯이 아프며 터진 후에 아물지 않는 것을 치료한다. 단삼ㆍ적작약ㆍ백지를 모두 같은 양으로 썰고 술에 2일간 담갔다가 돼지기름 반 근에 넣고 졸여 고약을 만든다.

[2] 外形篇卷之三 > 乳 > 乳癰 (ZZ)
유옹에는 이미 터졌거나 아직 터지지 않았거나 단삼고를 통용한다.

[질문] 단삼고는 어떤 증상에 쓰는 처방이며 구성은 무엇인가요?"""

FEWSHOT_ASSISTANT = """단삼고는 유옹(乳癰)에 사용하며, 적응증은 멍울이 생겨 찌르듯이 아프며 터진 후에 아물지 않는 것입니다. [1]

구성 약재: 단삼ㆍ적작약ㆍ백지 (모두 같은 양).

제법: 약재를 썰고 술에 2일간 담근 뒤 돼지기름 반 근에 넣고 졸여 고약을 만듭니다. [1]"""


def load_meta():
    with open(META, "r", encoding="utf-8") as fp:
        return [json.loads(ln) for ln in fp]


def main():
    print("=== load FAISS + meta + encoder ===")
    t0 = time.time()
    index = faiss.read_index(str(INDEX))
    meta = load_meta()
    encoder = SentenceTransformer(EMB_MODEL, device="cuda:0")
    print(f"  loaded ({time.time()-t0:.1f}s)  index ntotal={index.ntotal:,}")

    print("\n=== load Gemma3-12B + ver8.1 LoRA ===")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(str(BASE_DIR))
    base = AutoModelForCausalLM.from_pretrained(
        str(BASE_DIR), dtype=torch.bfloat16, device_map={"": "cuda:0"}
    )
    model = PeftModel.from_pretrained(base, str(ADAPTER)).eval()
    print(f"  loaded in {time.time()-t0:.1f}s")

    # v2: greedy + 짧은 max_new_tokens + 낮은 repetition_penalty
    gen_kwargs = dict(
        max_new_tokens=400,
        do_sample=False,             # greedy — retrieved 토큰 우선
        repetition_penalty=1.1,
        no_repeat_ngram_size=8,
    )

    for qid, query in QUESTIONS:
        print("\n" + "=" * 80)
        print(f"[{qid}] {query}")
        print("-" * 80)

        pre = pre_check(query)
        if pre.refused:
            print("⚠ Pre-safety refusal:")
            print(pre.refusal_text)
            continue

        # retrieve top-K
        qe = encoder.encode([query], normalize_embeddings=True,
                            convert_to_numpy=True).astype("float32")
        D, I = index.search(qe, K)
        sources = []
        print(f"[retrieved top-{K}]")
        for rank, idx in enumerate(I[0]):
            r = meta[idx]
            body = (r["trans_ko"] or r["original"]).replace("\n", " ").strip()
            sources.append({
                "rank": rank + 1, "id": r["id"],
                "path": r["up_path_nm"], "level": r["content_level"],
                "body": body, "sim": float(D[0][rank]),
            })
            preview = body[:60]
            print(f"  [{rank+1}] sim={D[0][rank]:.3f} {r['up_path_nm']}  →  {preview}")

        excerpt_block = "\n\n".join([
            f"[{s['rank']}] {s['path']} ({s['level']})\n{s['body']}"
            for s in sources
        ])
        user_msg = (
            "[동의보감 발췌]\n\n"
            f"{excerpt_block}\n\n"
            f"[질문] {query}"
        )

        # 3-turn: system + 1-shot user/assistant + 실제 user
        msgs = [
            {"role": "system", "content": SYSTEM_RAG},
            {"role": "user", "content": FEWSHOT_USER},
            {"role": "assistant", "content": FEWSHOT_ASSISTANT},
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

        final = post_check(text)

        print(f"\n[answer in {elapsed:.1f}s, prompt={n_in} tokens, greedy]")
        print(final)
        if final != text:
            print("\n[※ post_check 마스킹 적용됨]")

    print("\n" + "=" * 80)
    print("[probe-rag-v2] done")


if __name__ == "__main__":
    main()
