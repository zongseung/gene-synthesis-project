"""라벨 zip(151/612) → 통합 라벨 인덱스 (이미지 1장 = 1행).

이미지 본체 없이 라벨 JSON만 읽어, 학명 크로스워크·균형 샘플링의 기반이 되는
parquet 인덱스를 만든다. 종-zip당 워커 하나(배치 병렬), resume 지원.

스키마 차이:
  151: metadata.kind=코드, part=영문, 종이름은 멤버 폴더명('001_칡')에만 존재.
  612: plantinfo.class_name=한글명, herb_name=생약명, classification_info=판별대상/유사,
       toxic_info=Y/N, instance_info.instance_name=부위(한글).

사용:
  PYTHONPATH=src .venv/bin/python -m hanmed_mm.data.label_index \
      --root /mnt/nas-rawtext/한의학/aihub --out data/label_index --workers 8
"""
from __future__ import annotations
import argparse, glob, json, os, zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed

import pyarrow as pa
import pyarrow.parquet as pq

# 부위 정규화 (151 영문 / 612 한글 → 표준)
PART_MAP = {
    "flower": "flower", "꽃": "flower",
    "leaf": "leaf", "잎": "leaf",
    "fruit": "fruit", "열매": "fruit",
    "root": "root", "뿌리": "root",
    "stem": "stem", "줄기": "stem",
    "whole": "whole", "entire": "whole", "전초": "whole",
}
COLUMNS = ["dataset", "species_ko", "species_code", "herb_name", "part",
           "is_poisonous", "classification", "image_filename", "width", "height",
           "region", "label_zip"]
# 명시적 스키마: all-null 컬럼이 null 타입으로 추론돼 샤드 concat이 깨지는 것 방지
SCHEMA = pa.schema([
    ("dataset", pa.string()), ("species_ko", pa.string()), ("species_code", pa.string()),
    ("herb_name", pa.string()), ("part", pa.string()), ("is_poisonous", pa.bool_()),
    ("classification", pa.string()), ("image_filename", pa.string()),
    ("width", pa.int64()), ("height", pa.int64()), ("region", pa.string()),
    ("label_zip", pa.string()),
])


def _cp949(name: str) -> str:
    """zipfile이 cp437로 잘못 디코드한 한글 멤버명을 복원."""
    try:
        return name.encode("cp437").decode("cp949")
    except Exception:
        return name


def _norm_part(p):
    if not p:
        return "unknown"
    return PART_MAP.get(str(p).strip().lower(), PART_MAP.get(str(p).strip(), str(p).strip().lower()))


def _row_151(d, member_disp, zip_base):
    md = d.get("metadata", {})
    img = d.get("imagedata", {})
    top = member_disp.split("/")[0]              # '001_칡'
    code, _, name = top.partition("_")
    return {
        "dataset": "151",
        "species_ko": name or top,
        "species_code": code or None,
        "herb_name": None,
        "part": _norm_part(md.get("part")),
        "is_poisonous": bool(md.get("is_poisonous", False)),
        "classification": "similar" if md.get("is_compare") else "target",
        "image_filename": img.get("filename"),
        "width": img.get("width"),
        "height": img.get("height"),
        "region": md.get("place"),
        "label_zip": zip_base,
    }


def _row_612(d, zip_base):
    pi = d.get("plantinfo", {})
    img = d.get("image", {})
    part = (pi.get("instance_info") or {}).get("instance_name")
    return {
        "dataset": "612",
        "species_ko": pi.get("class_name"),
        "species_code": None,
        "herb_name": pi.get("herb_name"),
        "part": _norm_part(part),
        "is_poisonous": str(pi.get("toxic_info", "")).strip().upper() == "Y",
        "classification": "similar" if pi.get("classification_info") == "유사" else "target",
        "image_filename": img.get("file_name"),
        "width": img.get("width"),
        "height": img.get("height"),
        "region": img.get("region_name"),
        "label_zip": zip_base,
    }


def parse_zip(args):
    """워커: 라벨 zip 하나 → parquet 샤드. (zip_path, ds, out_dir) -> (zip, n, n_err)"""
    zip_path, ds, out_dir = args
    zip_base = os.path.basename(zip_path)
    shard = os.path.join(out_dir, f"{ds}__{zip_base}.parquet")
    if os.path.exists(shard):
        return (zip_base, -1, 0)  # 이미 처리됨(resume)
    rows, n_err = [], 0
    try:
        with zipfile.ZipFile(zip_path) as z:
            for zi in z.infolist():
                if zi.is_dir() or not zi.filename.lower().endswith(".json"):
                    continue
                try:
                    d = json.loads(z.read(zi.filename).decode("utf-8", "replace"))
                    if ds == "151":
                        rows.append(_row_151(d, _cp949(zi.filename), zip_base))
                    else:
                        rows.append(_row_612(d, zip_base))
                except Exception:
                    n_err += 1
    except Exception as e:
        return (f"{zip_base} [ZIP ERR: {e.__class__.__name__}]", 0, 0)
    if rows:
        table = pa.Table.from_pylist([{c: r.get(c) for c in COLUMNS} for r in rows], schema=SCHEMA)
        os.makedirs(out_dir, exist_ok=True)
        pq.write_table(table, shard)
    return (zip_base, len(rows), n_err)


def find_label_zips(root):
    """고정깊이 glob (재귀 ** 금지 — NAS에서 느림)."""
    out = []
    for ds in ("151", "612"):
        pat = os.path.join(root, ds, "*", "01.데이터", "*", "라벨링데이터", "*.zip")
        out += [(p, ds) for p in sorted(glob.glob(pat))]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/mnt/nas-rawtext/한의학/aihub")
    ap.add_argument("--out", default="data/label_index")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    shard_dir = os.path.join(args.out, "shards")
    os.makedirs(shard_dir, exist_ok=True)
    zips = find_label_zips(args.root)
    print(f"라벨 zip {len(zips)}개 발견 (151={sum(1 for _,d in zips if d=='151')}, "
          f"612={sum(1 for _,d in zips if d=='612')})  workers={args.workers}", flush=True)
    if not zips:
        print("라벨 zip 없음 — 아직 다운로드 안 됐을 수 있음.", flush=True)
        return

    tasks = [(p, d, shard_dir) for p, d in zips]
    total = done = skipped = errs = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(parse_zip, t): t[0] for t in tasks}
        for i, f in enumerate(as_completed(futs), 1):
            name, n, ne = f.result()
            if n == -1:
                skipped += 1
            else:
                total += max(n, 0); errs += ne; done += 1
            if i % 10 == 0 or n == -1:
                print(f"[{i}/{len(tasks)}] {name}: rows={n} err={ne} "
                      f"(누적 rows={total}, skip={skipped})", flush=True)
    print(f"\n완료: zip {done}개 처리 / {skipped}개 skip / 총 {total}행 / 파싱오류 {errs}", flush=True)

    # 통계 (샤드 전체 읽어 요약)
    shards = glob.glob(os.path.join(shard_dir, "*.parquet"))
    if shards:
        t = pa.concat_tables([pq.read_table(s) for s in shards])
        import pyarrow.compute as pc
        print(f"\n=== 인덱스 통계 (총 {t.num_rows}행) ===")
        for col in ("dataset", "part", "classification", "is_poisonous"):
            vc = pc.value_counts(t[col])
            pairs = [(str(x['values']), x['counts']) for x in vc.to_pylist()]
            print(f"  {col}: {dict(pairs)}")
        sp = pc.value_counts(t["species_ko"])
        print(f"  종 수: {len(sp)} 개")
        print(f"  생약명 보유(612): {pc.sum(pc.is_valid(t['herb_name'])).as_py()}행")


if __name__ == "__main__":
    main()
