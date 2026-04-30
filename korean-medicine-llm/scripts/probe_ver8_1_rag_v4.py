"""ver8.1 RAG v4 — Hanja 이형자 정규화 + 양방향 alias + NAME_RE 타이트닝.

v3 (probe_ver8_1_rag_v3.log) 에서 발견된 한자 병기 관련 오류:
  (1) Q1 「인삼(人蔘)」: query 의 `蔘` 가 코퍼스 표준자 `參` 와 매칭 실패
      → boost search 0건, dense 만으로 retrieve.
  (2) Q1·D3: NAME_RE 보조 regex `[가-힣一-龥]{2,}` 가 너무 헐거워
      `'간단'`, `'심하고'` 같은 일반 한국어 명사를 처방명 후보로 잘못 추출.
  (3) HANJA_ALIAS 가 `ko→hanja` 단방향 — query 가 한자만 있을 경우
      한국어 leaf 매칭 불가.

v4 변경점:
  1. HANJA_VARIANTS 이형자 → 표준자 (book_008 코퍼스 기준) 정규화 테이블 추가.
     코퍼스 grep: 蔘 0건 / 參 1472건, 鎮 0건 / 鎭 92건 → 표준자는 參·鎭.
  2. HANJA_ALIAS 양방향 expand (한국어↔한자 대칭).
  3. NAME_RE 에서 보조 regex 제거. 의서 접미어 보유 토큰 또는 fully-hanja
     2~5자 토큰만 후보로 인정.
  4. extract_names_from_query 가 normalize 적용 후 alias 양방향 expand.
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

# 14 question (v3 와 동일 — 비교 가능하게)
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
MIN_BODY_LEN = 30

# ── (1) 한자 이형자 → 표준자 정규화 ────────────────────────────────────────
# book_008.meta.jsonl 코퍼스의 표준자에 맞춤 (grep 으로 확인).
HANJA_VARIANTS = {
    "蔘": "參",   # 인삼·사삼·단삼·현삼 (코퍼스 1472건 모두 參)
    "鎮": "鎭",   # 鎭驚丸 등 (코퍼스 92건 모두 鎭)
    "飲": "飮",   # 보혈음·생맥음 (간혹 飲 표기 query 대응)
    "証": "證",   # 諸證·증후
    "醫": "醫",   # identity (동일 자 — placeholder)
}


def normalize_hanja(s: str) -> str:
    """이형자를 코퍼스 표준자로 치환."""
    return "".join(HANJA_VARIANTS.get(c, c) for c in s)


# ── (3) 한국어 ↔ 한자 양방향 alias ─────────────────────────────────────────
KO_TO_HANJA = {
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
    "유옹": "乳癰",
}
HANJA_TO_KO = {v: k for k, v in KO_TO_HANJA.items()}


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


# ── (2) NAME_RE 타이트닝 ──────────────────────────────────────────────────
# 한국어 처방은 prefix ≥ 2자 요구 ('간'+'단' 같은 1자 prefix false positive 차단).
# fully-hanja 2~5자 토큰은 편명·약재·증 표기로 인정.
# 보조 regex `[가-힣一-龥]{2,}` 제거.
NAME_RE = re.compile(
    r"[가-힣]{2,}(?:탕|산|환|단|음자|음|고)|"  # 한국어 처방 (prefix ≥ 2자)
    r"[一-龥]+(?:湯|散|丸|丹|飮|膏|圓)|"        # 한자 처방
    r"[一-龥]{2,5}"                             # fully-hanja 2~5자
)


def extract_names_from_query(query: str) -> list[str]:
    """query → 처방·증·고유명사 후보 (정규화 + 양방향 alias expand).

    1. normalize_hanja 로 이형자 → 표준자.
    2. NAME_RE 로 후보 추출.
    3. alias dict keys 가 query 에 substring 으로 들어있으면 명시 추가
       (NAME_RE 가 못 잡은 한국어 약재·편명 — 예: '인삼' — 보장).
    4. KO↔HJ 양방향 expand.
    """
    norm_query = normalize_hanja(query)
    raw = NAME_RE.findall(norm_query)

    # alias keys substring 보장 (NAME_RE 가 놓친 한국어 약재·편명 커버)
    for ko in KO_TO_HANJA:
        if ko in norm_query:
            raw.append(ko)
    for hj in HANJA_TO_KO:
        if hj in norm_query:
            raw.append(hj)

    candidates = list(dict.fromkeys(raw))
    expanded: list[str] = []
    for n in candidates:
        expanded.append(n)
        if n in KO_TO_HANJA:
            expanded.append(KO_TO_HANJA[n])
        if n in HANJA_TO_KO:
            expanded.append(HANJA_TO_KO[n])
    return list(dict.fromkeys(expanded))


def boost_search(meta: list, names: list[str], boost_k: int = 3) -> list[dict]:
    """path leaf 정확 매칭 우선, 그 다음 substring 매칭.
    leaf 와 name 모두 normalize_hanja 적용 후 비교 — 코퍼스가 표준자만 써도
    query 의 이형자가 매칭됨."""
    if not names:
        return []
    norm_names = [normalize_hanja(n) for n in names]
    exact_hits, substr_hits = [], []
    seen = set()
    for name, name_n in zip(names, norm_names):
        for r in meta:
            if r["id"] in seen:
                continue
            path = r.get("up_path_nm") or ""
            if not path:
                continue
            leaf_n = normalize_hanja(path.split(" > ")[-1].strip())
            body = (r.get("trans_ko") or r.get("original") or "").strip()
            if len(body) < MIN_BODY_LEN:
                continue
            if leaf_n == name_n:
                exact_hits.append(r)
                seen.add(r["id"])
            elif name_n in leaf_n:
                substr_hits.append(r)
                seen.add(r["id"])
    exact_hits.sort(key=lambda r: -len(r.get("trans_ko") or ""))
    return (exact_hits + substr_hits)[:boost_k]


def hybrid_search(query, encoder, index, meta, k=5, boost_k=2):
    names = extract_names_from_query(query)
    boost_recs = boost_search(meta, names, boost_k=boost_k)
    boost_ids = {r["id"] for r in boost_recs}

    qe = encoder.encode([query], normalize_embeddings=True,
                        convert_to_numpy=True).astype("float32")
    D, I = index.search(qe, k * 2)
    dense_results = []
    for rank, idx in enumerate(I[0]):
        r = meta[idx]
        if r["id"] not in boost_ids:
            dense_results.append({"rec": r, "sim": float(D[0][rank]), "src": "dense"})

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
        do_sample=False,
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
    print("[probe-rag-v4] done")


if __name__ == "__main__":
    main()
