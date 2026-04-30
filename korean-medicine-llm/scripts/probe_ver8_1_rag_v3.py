"""ver8.1 RAG v3 — Hybrid retrieval (dense + path leaf boost) + greedy + 강화 prompt.

v2 (greedy + 강화 prompt) 가 Q3 사물탕 환각 잔존:
  retrieved top-5 에 사물탕 본문 (內景篇卷之二 > 血 > 通治血病藥餌 > 四物湯,
  body: "숙지황·백작약·천궁·당귀 각 1.25돈") 가 빠져 있음.
  짧은 SS body (60자) 라 BGE-M3 dense retrieval 우선순위 낮음.

v3 변경점:
  1. 사용자 query 에서 처방·약재·경혈 이름 추출 (regex 기반)
  2. path leaf 정확 매칭 record (한자 OR 한글) 우선 boost
  3. 너무 짧은 path-only record (level=AA/Z2/EP+빈body) 는 제외
  4. boost 결과 + dense 결과 합쳐 top-K 구성 (중복 제거)
"""
from __future__ import annotations

import json
import os
import re
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

# 14 question (이전과 동일)
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
MIN_BODY_LEN = 30  # path-only record 제외 기준

# 한국어 → 한자 alias (14 question 에 등장하는 처방·편 한정).
# query 가 한국어만 명시한 경우 (예: "사물탕") 한자 leaf 매칭을 가능케 함.
HANJA_ALIAS = {
    "사물탕": "四物湯",
    "보중익기탕": "補中益氣湯",
    "단삼고": "丹參膏",
    "통설산": "通泄散",
    "진경환": "鎭驚丸",
    "증손낙령탕": "增損樂令湯",
    "내경편": "內景篇",
    "외형편": "外形篇",
    "잡병편": "雜病篇",
    "탕액편": "湯液篇",
    "침구편": "鍼灸篇",
    "신형": "身形",
    "인삼": "人參",
}


SYSTEM_RAG = (
    "당신은 한의학 고전 문헌 연구 보조 AI 입니다. 사용자 질문에 답할 때 다음 규칙을 "
    "엄격히 지키세요.\n\n"
    "1. [동의보감 발췌] 안의 글자만 사용해서 답하세요. 발췌에 없는 약재명·처방명·"
    "편명·인용·약재 개수는 절대 만들지 마세요.\n"
    "2. 약재 이름·처방 이름은 발췌의 표기를 한 글자도 바꾸지 말고 그대로 옮기세요.\n"
    "3. 약재 목록을 답할 때는 발췌의 'ㆍ' 또는 ',' 로 구분된 토큰만 그대로 나열하세요.\n"
    "4. 발췌만으로 답할 수 없으면 '발췌 자료에서는 확인할 수 없습니다' 라고 답하세요.\n"
    "5. 답변 끝에 [N] 으로 어느 발췌를 근거로 했는지 표시하세요."
)

FEWSHOT_USER = """[동의보감 발췌]

[1] 外形篇卷之三 > 乳 > 乳癰 > 丹參膏 (SS)
유옹으로 멍울이 생겨 찌르듯이 아프며 터진 후에 아물지 않는 것을 치료한다. 단삼ㆍ적작약ㆍ백지를 모두 같은 양으로 썰고 술에 2일간 담갔다가 돼지기름 반 근에 넣고 졸여 고약을 만든다.

[질문] 단삼고는 어떤 증상에 쓰는 처방이며 구성은 무엇인가요?"""

FEWSHOT_ASSISTANT = """단삼고는 유옹(乳癰)에 사용하며, 적응증은 멍울이 생겨 찌르듯이 아프며 터진 후에 아물지 않는 것입니다.

구성 약재: 단삼ㆍ적작약ㆍ백지 (모두 같은 양). [1]"""


# 처방·증·약재 이름 추출 — 한국어/한자 토큰 + 의서 접미어
NAME_RE = re.compile(
    r"[가-힣]+(?:탕|산|환|단|음|고|환자|법|음자|환원)|"
    r"[一-龥]+(?:湯|散|丸|丹|飮|膏|法|圓)|"
    r"[가-힣一-龥]{2,}"  # 일반 명사 (보조)
)


def extract_names_from_query(query: str) -> list[str]:
    """query 에서 처방·증·고유명사 후보 추출.
    한국어/한자 둘 다 모으고, 한국어→한자 alias 도 함께 expand."""
    candidates = NAME_RE.findall(query)
    suffixes = ("탕", "산", "환", "단", "음", "고", "음자", "湯", "散", "丸", "丹", "飮", "膏")
    primary = [c for c in candidates if c.endswith(suffixes)]
    hanja = [c for c in candidates if re.fullmatch(r"[一-龥]{2,5}", c)]
    names = list(dict.fromkeys(primary + hanja))
    # 한국어 → 한자 alias expand
    expanded = []
    for n in names:
        expanded.append(n)
        if n in HANJA_ALIAS and HANJA_ALIAS[n] not in names:
            expanded.append(HANJA_ALIAS[n])
    return list(dict.fromkeys(expanded))


def boost_search(meta: list, names: list[str], boost_k: int = 3) -> list[dict]:
    """path leaf 정확 매칭 우선, 그 다음 substring 매칭.
    같은 leaf 의 여러 record 는 body 길이 큰 순으로 정렬."""
    if not names:
        return []
    exact_hits = []   # leaf == name (정확)
    substr_hits = []  # name in leaf (변형처방 등)
    seen = set()
    for name in names:
        for r in meta:
            if r["id"] in seen:
                continue
            path = r.get("up_path_nm") or ""
            if not path:
                continue
            leaf = path.split(" > ")[-1].strip()
            body = (r.get("trans_ko") or r.get("original") or "").strip()
            if len(body) < MIN_BODY_LEN:
                continue
            if leaf == name:
                exact_hits.append(r)
                seen.add(r["id"])
            elif name in leaf:
                substr_hits.append(r)
                seen.add(r["id"])
    # body 길이 (정보량) 우선
    exact_hits.sort(key=lambda r: -len(r.get("trans_ko") or ""))
    return (exact_hits + substr_hits)[:boost_k]


def hybrid_search(query, encoder, index, meta, k=5, boost_k=2):
    # boost (path leaf 정확 매칭, body 30자 이상)
    names = extract_names_from_query(query)
    boost_recs = boost_search(meta, names, boost_k=boost_k)
    boost_ids = {r["id"] for r in boost_recs}

    # dense retrieval (top-k 충분히 많이)
    qe = encoder.encode([query], normalize_embeddings=True,
                        convert_to_numpy=True).astype("float32")
    D, I = index.search(qe, k * 2)
    dense_results = []
    for rank, idx in enumerate(I[0]):
        r = meta[idx]
        if r["id"] not in boost_ids:
            dense_results.append({"rec": r, "sim": float(D[0][rank]), "src": "dense"})

    # 결합: boost 우선, 그 후 dense
    final = [{"rec": r, "sim": 1.0, "src": "boost"} for r in boost_recs]
    for d in dense_results:
        if len(final) >= k:
            break
        final.append(d)
    return final[:k], names


def main():
    print("=== load FAISS + meta + encoder ===")
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
        max_new_tokens=400,
        do_sample=False,             # greedy
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

        results, names = hybrid_search(query, encoder, index, meta, k=K)
        print(f"[hybrid retrieved top-{K}]  extracted_names={names[:5]}")
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
        print(f"\n[answer in {elapsed:.1f}s, prompt={n_in} tokens, greedy + hybrid]")
        print(final)
        if final != text:
            print("\n[※ post_check 마스킹]")

    print("\n" + "=" * 80)
    print("[probe-rag-v3] done")


if __name__ == "__main__":
    main()
