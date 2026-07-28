"""원천데이터 zip → label_index parquet (612 Validation 전용 보완).

612 Training 원천 zip은 121종 중 30종만 받아져 있고 나머지 1,979GB 는 미다운로드다.
반면 Validation 원천 zip은 121종 전부 로컬에 있다. 라벨링데이터(JSON)는 Training 쪽만
있으므로 val 이미지에는 라벨 zip이 없는데, build_sft_mm 이 parquet 에서 실제로 읽는 건
dataset/species_ko/part/image_filename/label_zip 다섯 컬럼뿐이고(독성·효능은
species_annotation.jsonl 에서 옴), 이 다섯은 전부 zip 이름과 내부 경로에서 도출된다.
→ 원천 zip 목록만으로 동일 스키마 parquet 을 만들어 build_sft_mm 을 무수정 재사용한다.

zip 내부 구조: `{부위한글}/{종}_{부위}_{id}.jpg`  (꽃/열매/잎/전초)

label_zip 은 파일명이 아니라 **split 태그**로 쓰인다(build_sft_mm._split_of 가 VS_ →val,
그 외→train 로 판정). 종별로 파일명 정렬 후 앞 train_cap 장은 TS_, 다음 val_cap 장은 VS_
로 태깅해 train/val 이미지가 겹치지 않게 한다. 물리 파일은 둘 다 VS_ zip 안에 있다.

사용:
  PYTHONPATH=src .venv/bin/python -m hanmed_mm.data.label_index_from_source \
      --split 2.Validation --out data/label_index/shards
"""
from __future__ import annotations
import argparse, glob, os, zipfile
from concurrent.futures import ThreadPoolExecutor

import pyarrow as pa
import pyarrow.parquet as pq

from hanmed.shared.label_index import COLUMNS, SCHEMA, _cp949, _norm_part

IMG_EXT = (".jpg", ".jpeg", ".png")


def zip_rows(zip_path: str, train_cap: int, val_cap: int) -> list[dict]:
    """원천 zip 하나 → 종당 (train_cap + val_cap) 행. 부위는 최상위 폴더에서."""
    species = os.path.basename(zip_path)[3:-4]          # 'VS_가지.zip' → '가지'
    with zipfile.ZipFile(zip_path) as z:
        members = sorted(
            _cp949(n) for n in z.namelist()
            if not n.endswith("/") and n.lower().endswith(IMG_EXT)
        )
    rows = []
    for i, disp in enumerate(members[: train_cap + val_cap]):
        parts = disp.split("/")
        rows.append({
            "dataset": "612",
            "species_ko": species,
            "part": _norm_part(parts[0] if len(parts) > 1 else None),
            "image_filename": os.path.basename(disp),
            # split 태그(물리 zip 아님) — 앞쪽 train_cap 장만 train
            "label_zip": f"{'TS' if i < train_cap else 'VS'}_{species}.zip",
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/mnt/nas-rawtext/한의학/aihub")
    ap.add_argument("--dataset", default="612")
    ap.add_argument("--split", default="2.Validation", help="01.데이터 하위 폴더명")
    ap.add_argument("--out", default="data/label_index/shards")
    ap.add_argument("--train_cap", type=int, default=100, help="build_sft_mm --train_cap 과 맞출 것")
    ap.add_argument("--val_cap", type=int, default=15)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    pat = os.path.join(args.root, args.dataset, "*", "01.데이터", args.split, "원천데이터", "*.zip")
    zips = sorted(p for p in glob.glob(pat) if not p.endswith(".part"))
    print(f"원천 zip {len(zips)}개 ({args.split})", flush=True)
    if not zips:
        return

    os.makedirs(args.out, exist_ok=True)
    total = 0

    def work(zp):
        base = os.path.basename(zp)
        shard = os.path.join(args.out, f"{args.dataset}__SRC_{base}.parquet")
        if os.path.exists(shard):
            return base, -1
        rows = zip_rows(zp, args.train_cap, args.val_cap)
        if rows and not args.dry_run:
            pq.write_table(
                pa.Table.from_pylist([{c: r.get(c) for c in COLUMNS} for r in rows], schema=SCHEMA),
                shard,
            )
        return base, len(rows)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:   # zip 중앙디렉터리만 읽음(IO 바운드)
        for base, n in ex.map(work, zips):
            if n >= 0:
                total += n
            print(f"  {base}: {'skip' if n < 0 else n}행", flush=True)
    print(f"\n완료: {total}행 → {args.out}{' (dry_run)' if args.dry_run else ''}", flush=True)


if __name__ == "__main__":
    main()
