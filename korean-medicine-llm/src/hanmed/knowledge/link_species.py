"""종 ↔ 한자 약재명 링크 확장 — 한자 독음 대조.

claudedocs/vlm_plan/03_dataset.md §3.3.

기존 링크는 한자 표제어 정확 일치라 「왕불유행 ↔ 王不留行」 같은 한글 약재명을 놓친다.
`hanja` 로 표제어 독음을 만들고 두음법칙을 양쪽에 같은 규칙으로 적용해 대조한다.

두음법칙을 정규화로 처리하는 이유: hanja 패키지는 王不留行 을 「왕불류행」으로 내지만
우리 라벨은 「왕불유행」이다. 한쪽만 고치면 반대 방향(綠豆 녹두 ↔ 록두)이 깨진다.

**동음이의는 링크하지 않는다** — 실측 오탐 가자 = 訶子(가자, 약재) / 茄子(가지, 채소).
"""
from __future__ import annotations

import glob
import json
import os

# 초성 index: ㄴ=2, ㄹ=5, ㅇ=11 / 중성 index: ㅑ2 ㅒ3 ㅕ6 ㅖ7 ㅛ12 ㅠ17 ㅣ20
_CHO_N, _CHO_R, _CHO_O = 2, 5, 11
_I_Y_VOWELS = {2, 3, 6, 7, 12, 17, 20}
_KB_DIR = "data/safety_kb/classical"


def normalize_reading(s: str) -> str:
    """두음법칙 정규화. 표제어 독음과 라벨을 같은 표기로 모은다.

    ㄹ·ㄴ + ㅣ/y계 모음 → ㅇ  (류→유, 니→이)
    ㄹ + 그 밖의 모음    → ㄴ  (록→녹)
    """
    out = []
    for ch in s:
        o = ord(ch) - 0xAC00
        if 0 <= o < 11172:
            cho, jung, jong = o // 588, (o % 588) // 28, o % 28
            if cho in (_CHO_N, _CHO_R):
                if jung in _I_Y_VOWELS:
                    cho = _CHO_O
                elif cho == _CHO_R:
                    cho = _CHO_N
                ch = chr(0xAC00 + (cho * 21 + jung) * 28 + jong)
        out.append(ch)
    return "".join(out)


def build_index(*term_groups) -> dict[str, list[tuple[str, int]]]:
    """한자 표제어들 → {정규화 독음: [(표제어, 출전순위)…]}.

    인자 순서가 출전 우선순위다 (동의보감 > 향약집성방 > 본초강목, 03_dataset §3.3).
    같은 독음에 여러 표제어가 걸려도 상위 출전에 하나만 있으면 그것으로 확정한다 —
    水芹(동의보감)/水斳(향약집성방)처럼 이체자로 갈리는 경우가 대부분이라
    전부 보류하면 멀쩡한 링크를 잃는다. 같은 출전 안에서 갈리면 보류한다.
    """
    import hanja
    idx: dict[str, list[tuple[str, int]]] = {}
    for rank, terms in enumerate(term_groups):
        for t in terms:
            key = normalize_reading(hanja.translate(t, "substitution"))
            hits = idx.setdefault(key, [])
            if t not in [h[0] for h in hits]:
                hits.append((t, rank))
    return idx


def kb_terms(kb_dir: str = _KB_DIR) -> list[str]:
    """동의보감 본초 KB 표제어 700종."""
    out = []
    for f in sorted(glob.glob(os.path.join(kb_dir, "*.json"))):
        with open(f, encoding="utf-8") as fh:
            out.append(json.load(fh)["herb_hanja"])
    return out


_default_index: dict[str, list[str]] | None = None


def link(label: str, index=None) -> list[str]:
    """한글 약재명/종명 → 한자 표제어.

    최상위 출전에 후보가 하나면 링크, 같은 출전에서 갈리면 보류(빈 리스트).
    """
    global _default_index
    if index is None:
        if _default_index is None:
            _default_index = build_index(kb_terms())
        index = _default_index
    hits = index.get(normalize_reading(label or ""), [])
    if not hits:
        return []
    top = min(r for _, r in hits)
    best = [t for t, r in hits if r == top]
    return best if len(best) == 1 else []


def main() -> int:
    import argparse
    from collections import Counter

    p = argparse.ArgumentParser(description="독음 기반 링크 확장 현황 보고.")
    p.add_argument("--species", default="data/annotations/species_annotation.jsonl")
    p.add_argument("--kb_dir", default=_KB_DIR)
    a = p.parse_args()

    idx = build_index(kb_terms(a.kb_dir))
    stat, newly = Counter(), []
    with open(a.species, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            hit = link(r.get("herb_name_label") or "", idx) or link(r["species_ko"], idx)
            stat[r["knowledge_status"]] += 1
            if hit:
                stat["reading_link"] += 1
                if r["knowledge_status"] != "linked":
                    newly.append((r["species_ko"], hit[0]))
    print(f"표제어 {sum(len(v) for v in idx.values())} · 고유독음 {len(idx)} · "
          f"동음이의 {sum(1 for v in idx.values() if len(v) > 1)}")
    print(f"현황 {dict(stat)}")
    print(f"신규 링크 {len(newly)}종: " + ", ".join(f"{k}→{v}" for k, v in newly[:10]) + " …")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
