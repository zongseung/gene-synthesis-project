"""분류기 라벨 어휘 — species는 샤드 매니페스트에서 수집, part는 고정 어휘.

species 클래스명은 샤드 json의 `species_ko`(151='005_유럽감초', 612='난쟁이아욱')를
그대로 클래스 키로 쓴다(데이터셋 간 종이 다르므로 union). 학습/추론이 동일 vocab을
공유하도록 `label_vocab.json` 으로 고정 저장한다.
"""
from __future__ import annotations
import glob
import json

# label_index._norm_part 산출과 동일한 표준 부위 집합
PARTS = ["flower", "leaf", "fruit", "root", "stem", "whole", "unknown"]


def build_vocab(manifest_glob: str, out_path: str) -> dict:
    """샤드 매니페스트들의 species_counts 합집합으로 species 어휘 구성 후 저장."""
    species: set[str] = set()
    n_manifest = 0
    for mp in sorted(glob.glob(manifest_glob)):
        m = json.load(open(mp, encoding="utf-8"))
        species.update(m.get("species_counts", {}).keys())
        n_manifest += 1
    species_sorted = sorted(species)
    vocab = {
        "species": {s: i for i, s in enumerate(species_sorted)},
        "parts": {p: i for i, p in enumerate(PARTS)},
        "n_manifest": n_manifest,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    print(f"vocab 저장: {out_path}  (species {len(species_sorted)}종 / part {len(PARTS)} / 매니페스트 {n_manifest}개)")
    return vocab


def load_vocab(path: str) -> dict:
    return json.load(open(path, encoding="utf-8"))


def species_weights(manifest_glob: str, vocab: dict, mode: str = "sqrt") -> list[float]:
    """클래스 불균형 보정용 species 가중치 (sqrt-inverse-freq, ver1 오버샘플 철학 승계)."""
    import math
    counts = [0] * len(vocab["species"])
    for mp in sorted(glob.glob(manifest_glob)):
        m = json.load(open(mp, encoding="utf-8"))
        for sp, c in m.get("species_counts", {}).items():
            if sp in vocab["species"]:
                counts[vocab["species"][sp]] += c
    total = sum(counts) or 1
    w = []
    for c in counts:
        c = max(c, 1)
        if mode == "sqrt":
            w.append((total / c) ** 0.5)
        else:  # inverse
            w.append(total / c)
    # 평균 1.0으로 정규화
    mean = sum(w) / len(w)
    return [x / mean for x in w]
