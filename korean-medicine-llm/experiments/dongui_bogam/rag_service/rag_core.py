"""RAG core — hybrid retrieval (boost + dense) for HanMed ver8.1.

probe_ver8_1_rag_v4.py 의 핵심 로직을 sidecar 환경으로 이식:
  - HANJA_VARIANTS: 蔘→參·鎮→鎭·飲→飮·証→證 정규화
  - KO ↔ HJ 양방향 alias
  - NAME_RE: prefix ≥ 2 한국어 처방, fully-hanja 2~5자, alias keys substring
  - boost_search: leaf normalize 후 정확/부분 매칭
  - hybrid_search: boost (top boost_k) + dense (FAISS) 결합
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# ── 한자 이형자 → 표준자 (book_008 corpus 기준) ─────────────────────────────
HANJA_VARIANTS: dict[str, str] = {
    "蔘": "參",
    "鎮": "鎭",
    "飲": "飮",
    "証": "證",
}


def normalize_hanja(s: str) -> str:
    return "".join(HANJA_VARIANTS.get(c, c) for c in s)


# ── 한국어 ↔ 한자 양방향 alias ────────────────────────────────────────────
KO_TO_HANJA: dict[str, str] = {
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
HANJA_TO_KO: dict[str, str] = {v: k for k, v in KO_TO_HANJA.items()}


NAME_RE = re.compile(
    r"[가-힣]{2,}(?:탕|산|환|단|음자|음|고)|"
    r"[一-龥]+(?:湯|散|丸|丹|飮|膏|圓)|"
    r"[一-龥]{2,5}"
)


def extract_names(query: str) -> list[str]:
    norm_query = normalize_hanja(query)
    raw = NAME_RE.findall(norm_query)
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


@dataclass
class RetrievedItem:
    rec: dict[str, Any]
    sim: float
    src: str  # "boost" | "dense"


class RagCore:
    """In-memory FAISS + meta + encoder. Singleton-style (process-local)."""

    def __init__(self, index_path: Path, meta_path: Path,
                 encoder_name: str, encoder_device: str = "cpu",
                 min_body_len: int = 30) -> None:
        self.index = faiss.read_index(str(index_path))
        self.meta: list[dict[str, Any]] = [json.loads(l) for l in open(meta_path)]
        # local_files_only=True : HF API 호출 없이 cache 만 사용 (offline-first).
        # cache miss 시 명확한 error 로 빠르게 fail (이전엔 hang 됐음).
        self.encoder = SentenceTransformer(
            encoder_name, device=encoder_device, local_files_only=True
        )
        self.min_body_len = min_body_len

    def boost_search(self, names: list[str], boost_k: int) -> list[dict[str, Any]]:
        if not names:
            return []
        norm_names = [normalize_hanja(n) for n in names]
        exact_hits, substr_hits = [], []
        seen: set[str] = set()
        for name_n in norm_names:
            for r in self.meta:
                rid = r.get("id")
                if rid is None or rid in seen:
                    continue
                path = r.get("up_path_nm") or ""
                if not path:
                    continue
                leaf_n = normalize_hanja(path.split(" > ")[-1].strip())
                body = (r.get("trans_ko") or r.get("original") or "").strip()
                if len(body) < self.min_body_len:
                    continue
                if leaf_n == name_n:
                    exact_hits.append(r)
                    seen.add(rid)
                elif name_n in leaf_n:
                    substr_hits.append(r)
                    seen.add(rid)
        exact_hits.sort(key=lambda r: -len(r.get("trans_ko") or ""))

        # 약재 query 보강: 湯液篇 (약재 主篇) 의 substr-only 매칭이라도 최소 1슬롯
        # 보장. 同名 leaf (單方 > 人參 등) 가 exact 슬롯 다 차지해도 약재의 성미·
        # 주치 메인 record (湯液篇 > 草部·木部 등) 가 retrieval 에 누락되지 않게.
        primary = [r for r in substr_hits
                   if "湯液篇" in (r.get("up_path_nm") or "")]
        if primary:
            primary.sort(key=lambda r: -len(r.get("trans_ko") or ""))
            merged = primary[:1] + [r for r in exact_hits + substr_hits
                                    if r["id"] != primary[0]["id"]]
            return merged[:boost_k + 1]
        return (exact_hits + substr_hits)[:boost_k]

    def hybrid_search(self, query: str, k: int = 5,
                      boost_k: int = 2) -> tuple[list[RetrievedItem], list[str]]:
        names = extract_names(query)
        boost_recs = self.boost_search(names, boost_k=boost_k)
        boost_ids = {r["id"] for r in boost_recs}

        qe = self.encoder.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True
        ).astype("float32")
        D, I = self.index.search(qe, k * 2)

        dense_results: list[RetrievedItem] = []
        for rank, idx in enumerate(I[0]):
            r = self.meta[int(idx)]
            if r.get("id") in boost_ids:
                continue
            dense_results.append(
                RetrievedItem(rec=r, sim=float(D[0][rank]), src="dense")
            )

        final: list[RetrievedItem] = [
            RetrievedItem(rec=r, sim=1.0, src="boost") for r in boost_recs
        ]
        for d in dense_results:
            if len(final) >= k:
                break
            final.append(d)
        return final[:k], names

    @staticmethod
    def build_excerpt_block(items: list[RetrievedItem]) -> str:
        return "\n\n".join([
            f"[{i+1}] {it.rec.get('up_path_nm') or '(no path)'} "
            f"({it.rec.get('content_level', '?')})\n"
            f"{(it.rec.get('trans_ko') or it.rec.get('original') or '').strip()}"
            for i, it in enumerate(items)
        ])
