"""book_008 raw 34,040 records → BGE-M3 embedding → FAISS index.

산출:
  data/rag/book_008.index    — FAISS IndexFlatIP (cosine, normalized)
  data/rag/book_008.meta.jsonl — 각 vector idx 의 메타 (record id / path / level / trans_ko)

검색은 cosine similarity (normalized inner product). 34,040 vector 는 작은 규모라 IndexFlatIP 가 충분 (GPU/CPU 모두 빠름, 정확도 손실 0).

실행:
  cd korean-medicine-llm
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/build_rag_index.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import faiss

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw" / "mediclassics_unified" / "book_008"
OUT_DIR = ROOT / "data" / "rag"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "BAAI/bge-m3"  # multilingual + 한자 + 한국어 강. 1024-dim.


def load_records():
    """vol_01..23.jsonl 전체 로드. record_id = volume_id_seq."""
    records = []
    for f in sorted(RAW_DIR.glob("vol_*.jsonl")):
        with open(f, "r", encoding="utf-8") as fp:
            for ln in fp:
                r = json.loads(ln)
                records.append({
                    "id": f"book_008/vol_{r['volume_id']:02d}/seq_{r['content_seq']}",
                    "volume_id": r["volume_id"],
                    "content_seq": r["content_seq"],
                    "content_level": r["content_level"],
                    "up_path_nm": r.get("up_path_nm"),
                    "trans_ko": r.get("trans_ko", "").strip(),
                    "original": r.get("original", "").strip(),
                })
    return records


def make_embedding_text(rec: dict) -> str:
    """검색 시 사용할 텍스트. up_path + trans_ko + original 한자.
    빈 trans_ko 면 original 만, 둘 다 비어있으면 path 만."""
    parts = []
    if rec["up_path_nm"]:
        parts.append(rec["up_path_nm"])
    if rec["trans_ko"]:
        parts.append(rec["trans_ko"])
    if rec["original"] and rec["original"] != rec["trans_ko"]:
        # 한자 원문도 포함 (한자 검색 쿼리 대응)
        parts.append(rec["original"])
    return "  ".join(parts) if parts else f"[{rec['content_level']} {rec['id']}]"


def main():
    print(f"=== load records ===")
    t0 = time.time()
    recs = load_records()
    print(f"  loaded {len(recs):,} records in {time.time()-t0:.1f}s")

    print(f"\n=== embedding ({MODEL_NAME}) ===")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"  device = {device}")
    t0 = time.time()
    model = SentenceTransformer(MODEL_NAME, device=device)
    print(f"  model loaded in {time.time()-t0:.1f}s, dim={model.get_sentence_embedding_dimension()}")

    texts = [make_embedding_text(r) for r in recs]
    t0 = time.time()
    embs = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,  # cosine 용 L2 normalize
        convert_to_numpy=True,
    )
    print(f"  embedded {len(embs):,} × {embs.shape[1]} in {time.time()-t0:.1f}s")
    embs = embs.astype("float32")

    print(f"\n=== build FAISS index (IndexFlatIP) ===")
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    print(f"  index ntotal = {index.ntotal}")

    out_index = OUT_DIR / "book_008.index"
    out_meta = OUT_DIR / "book_008.meta.jsonl"
    faiss.write_index(index, str(out_index))
    print(f"  saved index -> {out_index}  ({out_index.stat().st_size / 1024 / 1024:.1f} MB)")

    with open(out_meta, "w", encoding="utf-8") as fp:
        for r in recs:
            fp.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  saved meta  -> {out_meta}  ({out_meta.stat().st_size / 1024 / 1024:.1f} MB)")

    # === 짧은 retrieval test ===
    print(f"\n=== retrieval smoke test ===")
    queries = [
        "사물탕의 구성 약재",
        "보중익기탕의 효능",
        "동의보감 內景篇 身形",
        "통설산은 어떤 증상에 쓰나",
        "증손낙령탕의 구성",
    ]
    qe = model.encode(queries, normalize_embeddings=True, convert_to_numpy=True).astype("float32")
    D, I = index.search(qe, 5)
    for i, q in enumerate(queries):
        print(f"\n  query: {q}")
        for rank, (score, idx) in enumerate(zip(D[i], I[i]), 1):
            r = recs[idx]
            preview = (r["trans_ko"] or r["original"])[:80].replace("\n", " ")
            print(f"    {rank}. [{score:.3f}] {r['up_path_nm']}  →  {preview}")


if __name__ == "__main__":
    main()
