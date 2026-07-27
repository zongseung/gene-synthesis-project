#!/usr/bin/env python3
"""track6_herb_image — 약초 이미지 벤치 트랙 빌더 (결함 B: 이미지 문항 0개 해소).

대상 이미지 풀 = herb_shard_index 전체 - (mm_train_resolved + mm_val_resolved 의
image). 학습에 쓰인 이미지는 벤치에서 배제해야 벤치 점수가 학습 오염 없이 일반화를
잰다. 종당 5장(has_similar_class 종은 10장 — 오식별이 곧 위험이라 두껍게 본다)을
`build_sft_mm.allocate()` 로 부위 층화 추출한다(새 로직 없음, 재사용).

문항 3종:
  species_id        — gold = species_ko (샤드 논리경로가 이미 종 라벨)
  toxicity           — gold = ontology.species.tox_status (3값)
  efficacy_abstain    — gold = "abstain", knowledge_status != 'linked' 종만
정답은 전부 기존 라벨(ontology.sqlite species 테이블 + 샤드 인덱스)에서만 온다.

사용:
  PYTHONPATH=src .venv/bin/python scripts/build_herb_image_bench.py
"""
from __future__ import annotations
import argparse
import collections
import glob
import hashlib
import json
import os
import sqlite3
import sys

sys.path.insert(0, "src")

import pyarrow.parquet as pq  # noqa: E402

from hanmed_mm.classifier.siglip_probe import species_of  # noqa: E402
from hanmed_mm.data.build_ontology import DB_PATH  # noqa: E402
from hanmed_mm.data.build_sft_mm import allocate  # noqa: E402

DEF_SHARD_INDEX = "data/shards/herb_shard_index.json"
DEF_TRAIN = "data/sft/mm_train_resolved.jsonl"
DEF_VAL = "data/sft/mm_val_resolved.jsonl"
DEF_LABEL_SHARDS = "data/label_index/shards/*.parquet"
DEF_OUT = "data/eval/hanmed_bench/track6_herb_image.jsonl"
DEF_MANIFEST = "data/eval/hanmed_bench/manifest.json"

CAP_SIMILAR, CAP_DEFAULT = 10, 5

QUESTION = {
    "species_id": "이 사진 속 약용식물의 이름은?",
    "toxicity": "이 식물을 섭취해도 안전한가요?",
    "efficacy_abstain": "이 약초의 효능은?",
}
SCORING = {
    "species_id": "species_top1_acc",
    "toxicity": "tox_status_acc",
    "efficacy_abstain": "abstain_judgment",
}


def _eid(prefix: str, *parts) -> str:
    h = hashlib.sha1("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()[:10]
    return f"{prefix}_{h}"


def load_used_images(*paths) -> set:
    used = set()
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                used.add(json.loads(line)["image"])
    return used


def load_part_index(shards_glob: str) -> dict:
    """image_filename → part. label_index 파케가 권위 있는 소스(파일명 파싱보다 우선)."""
    idx = {}
    for fp in sorted(glob.glob(shards_glob)):
        df = pq.read_table(fp, columns=["image_filename", "part"]).to_pandas()
        for fn, part in zip(df["image_filename"], df["part"]):
            idx[fn] = part
    return idx


def load_species_table(db_path: str) -> dict:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    try:
        rows = con.execute(
            "SELECT species_ko, knowledge_status, tox_status, has_similar_class "
            "FROM species"
        ).fetchall()
    finally:
        con.close()
    return {r["species_ko"]: dict(r) for r in rows}


def build_pool(shard_index_path: str, used_images: set, part_index: dict) -> dict:
    """종 → 부위 → row 리스트. row = (species_ko, dataset, part, filename, logical, "pool")
    — build_sft_mm.allocate() 가 기대하는 6-tuple 모양(정렬 키는 r[3])에 맞춘다."""
    with open(shard_index_path, encoding="utf-8") as f:
        index = json.load(f)
    pool = collections.defaultdict(lambda: collections.defaultdict(list))
    for logical in index:
        if logical in used_images:
            continue
        segs = logical.split("/")
        if len(segs) < 3:
            continue
        filename = segs[-1]
        part = part_index.get(filename)
        if part is None:
            continue  # 부위 라벨이 없는 이미지는 층화 대상에서 제외(근거 없는 배분 금지)
        sp = species_of(logical)
        pool[sp][part].append((sp, segs[0], part, filename, logical, "pool"))
    return pool


def build_track6(shard_index_path: str, used_images: set, part_index: dict,
                  species_table: dict) -> list:
    pool = build_pool(shard_index_path, used_images, part_index)
    rows = []
    for sp in sorted(pool):
        meta = species_table.get(sp)
        if meta is None:
            continue  # ontology 밖 종은 gold 를 못 만든다(결함 8: 라벨 신설 금지)
        has_similar = bool(meta["has_similar_class"])
        cap = CAP_SIMILAR if has_similar else CAP_DEFAULT
        chosen = allocate(pool[sp], cap)
        for _, _dataset, _part, _filename, logical, _ in chosen:
            probe_types = ["species_id", "toxicity"]
            if meta["knowledge_status"] != "linked":
                probe_types.append("efficacy_abstain")
            gold = {
                "species_id": sp,
                "toxicity": meta["tox_status"],
                "efficacy_abstain": "abstain",
            }
            for pt in probe_types:
                rows.append({
                    "id": _eid("t6", pt, logical),
                    "track": "herb_image",
                    "question": QUESTION[pt],
                    "image": logical,
                    "gold": gold[pt],
                    "scoring": SCORING[pt],
                    "species_ko": sp,
                    "has_similar_class": has_similar,
                    "probe_type": pt,
                })
    return rows


def write_jsonl(path: str, rows: list):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def update_manifest(manifest_path: str, rows: list):
    """기존 5트랙 manifest 에 track6 추가 + track1 source 오기(stale) 정정.

    track1 은 3분할(Task 1) 이후 실제로는 tongue_sft_test.jsonl 에서 나오는데
    manifest 문자열이 예전 tongue_sft_val.jsonl 을 그대로 가리키고 있었다."""
    with open(manifest_path, encoding="utf-8") as f:
        man = json.load(f)
    man["tracks"]["track1_tongue_byeonjeung"]["source"] = \
        "data/sft/tongue_sft/tongue_sft_test.jsonl"

    by_probe = collections.Counter(r["probe_type"] for r in rows)
    n_similar = sum(1 for r in rows if r["has_similar_class"])
    man["tracks"]["track6_herb_image"] = {
        "n": len(rows),
        "n_species": len({r["species_ko"] for r in rows}),
        "by_probe_type": dict(by_probe),
        "n_similar_class": n_similar,
        "n_non_similar_class": len(rows) - n_similar,
        "source": "data/shards/herb_shard_index.json (mm_train/val_resolved 제외)",
        "metric": "species_top1_acc + tox_status_acc + abstain_judgment",
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(man, f, ensure_ascii=False, indent=2)
    return man["tracks"]["track6_herb_image"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard_index", default=DEF_SHARD_INDEX)
    ap.add_argument("--train", default=DEF_TRAIN)
    ap.add_argument("--val", default=DEF_VAL)
    ap.add_argument("--label_shards", default=DEF_LABEL_SHARDS)
    ap.add_argument("--db", default=DB_PATH)
    ap.add_argument("--out", default=DEF_OUT)
    ap.add_argument("--manifest", default=DEF_MANIFEST)
    args = ap.parse_args()

    used = load_used_images(args.train, args.val)
    part_index = load_part_index(args.label_shards)
    species_table = load_species_table(args.db)

    rows = build_track6(args.shard_index, used, part_index, species_table)
    write_jsonl(args.out, rows)
    stats = update_manifest(args.manifest, rows)

    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"\n[OK] {args.out} 에 {len(rows)}행 생성, manifest.json 갱신")


if __name__ == "__main__":
    main()
