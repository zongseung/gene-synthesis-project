"""원천 이미지 zip(NAS) → 768px 리사이즈 → WebDataset tar 샤드(NVMe).

각 샘플 = `{key}.jpg`(긴 변 768px 리사이즈) + `{key}.json`(라벨). 종(species)을
라운드로빈으로 섞어 패킹하므로 한 샤드가 여러 종을 포함한다 → 학습 시 디스크는
순차로 읽되(WebDataset) 셔플 버퍼로 배치 다양성을 얻는다.

설계 근거: claudedocs/design_data_serving_sharding_20260629.md (§5 샤딩, §5-4 리사이즈)
  - 원본 고해상(전체 ~2.3TB) > NVMe 2.1T → 768px 리사이즈로 ~300GB로 축소.
  - NAS 원본은 **읽기만** 하며 변경/삭제하지 않는다(재현·재처리 안전판).

라벨: 종(species)·부위(part)는 zip 내부 경로/파일명에서 직접 도출(인덱스 불필요).
  --label_index (label_index.py 산출 parquet)를 주면 (dataset, image_filename)으로
  조인해 is_poisonous/classification/herb_name 을 추가 임베드한다.

쓰기는 stdlib tarfile(설치 불필요). 읽기는 학습 시 webdataset 사용(WebDataset 호환 포맷).

사용:
  # 151 학습셋 시범 샤딩 (다운로드 완료분)
  PYTHONPATH=src .venv/bin/python -m hanmed.stage2_vlm.shard \
      --root /mnt/nas-rawtext/한의학/aihub --dataset 151 --split train \
      --out data/shards --long_side 768 --shard_size 1e9 --interleave 8 --workers 4

  # 매핑·종 분포만 점검(이미지 안 씀)
  ... --dry_run
"""
from __future__ import annotations
import argparse, collections, glob, io, json, os, re, tarfile, zipfile
from concurrent.futures import ProcessPoolExecutor

from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # AI Hub 고해상 원본 신뢰 → DecompressionBomb 제한 해제(대형 이미지 스킵 방지)

from hanmed.shared.label_index import _cp949, _norm_part  # 재사용(중복 방지)

IMG_EXT = (".jpg", ".jpeg", ".png")
SPLIT_DIR = {"train": "1.Training", "val": "2.Validation"}
PREFIX_612 = ("TS_", "VS_", "TL_", "VL_")


# ---------------------------------------------------------------- 종/부위 도출
def _species_612(zip_base: str) -> str:
    """TS_난쟁이아욱.zip → 난쟁이아욱 (접두 제거)."""
    stem = zip_base[:-4] if zip_base.lower().endswith(".zip") else zip_base
    for pre in PREFIX_612:
        if stem.startswith(pre):
            return stem[len(pre):]
    return stem


def _part_151(fname: str):
    """005_00000002_leaf.jpg → leaf (파일명 마지막 토큰)."""
    stem = os.path.splitext(fname)[0]
    return stem.rsplit("_", 1)[-1] if "_" in stem else None


def _safe_key(s: str) -> str:
    """tar 멤버 키: '.'(webdataset 확장자 구분자)·구분자 제거."""
    return re.sub(r"[^0-9A-Za-z가-힣_-]", "_", s)


# ---------------------------------------------------------------- zip 탐색
def find_source_zips(root: str, dataset: str, split: str) -> list[str]:
    """원천데이터 zip (.part 제외). 고정깊이 glob (재귀 ** 금지 — NAS 느림)."""
    pat = os.path.join(root, dataset, "*", "01.데이터", SPLIT_DIR[split], "원천데이터", "*.zip")
    return sorted(p for p in glob.glob(pat) if not p.endswith(".part"))


# ---------------------------------------------------------------- 라벨 인덱스(선택)
def load_label_lookup(label_index_dir: str) -> dict:
    """parquet 인덱스 → {(dataset, image_filename): {추가 라벨필드}}. 없으면 {}."""
    shards = glob.glob(os.path.join(label_index_dir, "shards", "*.parquet"))
    if not shards:
        return {}
    import pyarrow.parquet as pq
    # 샤드별 독립 로드(concat 금지): 일부 샤드가 all-null 컬럼을 null 타입으로 굳혀
    # concat_tables 가 schema mismatch 로 깨지는 stale-shard 문제를 우회.
    fields = ("dataset", "image_filename", "is_poisonous", "classification", "herb_name", "species_code")
    extra_fields = ("is_poisonous", "classification", "herb_name", "species_code")
    out = {}
    for s in shards:
        t = pq.read_table(s)
        names = set(t.column_names)
        cols = {f: (t[f].to_pylist() if f in names else [None] * t.num_rows) for f in fields}
        for i in range(t.num_rows):
            fn = cols["image_filename"][i]
            if not fn:
                continue
            out[(cols["dataset"][i], fn)] = {k: cols[k][i] for k in extra_fields}
    return out


# ---------------------------------------------------------------- 샘플 스트림
def _zip_image_gen(z: zipfile.ZipFile, dataset: str, sp612, label_lookup, stats):
    """zip 하나의 이미지 멤버를 순차로 yield: (key, raw_bytes, label_dict).

    손상 멤버(Bad CRC-32 / zlib invalid block 등)는 `z.read` 에서 예외를 던지는데,
    그대로 두면 워커 프로세스가 죽어 전체 샤딩이 중단된다 → 멤버 단위로 캐치·스킵하고
    `stats["corrupt"]` 로 집계(전체 abort 방지)."""
    for zi in z.infolist():
        if zi.is_dir():
            continue
        disp = _cp949(zi.filename)
        if not disp.lower().endswith(IMG_EXT):
            continue
        fname = os.path.basename(disp)
        if dataset == "151":
            species = disp.split("/")[0]            # '005_유럽감초'
            part = _part_151(fname)                  # 파일명 접미
        else:
            species = sp612                          # zip명에서
            part = disp.split("/")[0]                # 최상위 폴더(부위 한글)
        label = {"dataset": dataset, "species_ko": species,
                 "image_filename": fname, "part": _norm_part(part)}
        if label_lookup:
            extra = label_lookup.get((dataset, fname))
            if extra:
                label.update({k: v for k, v in extra.items() if v is not None})
            else:
                label["_label_index_miss"] = True
        try:
            raw = z.read(zi)                          # 손상 멤버면 여기서 예외
        except Exception as e:                        # BadZipFile / zlib.error 등
            stats["corrupt"] += 1
            print(f"    ⚠️ 손상 멤버 스킵: {disp} — {type(e).__name__}: {e}", flush=True)
            continue
        yield (_safe_key(f"{dataset}__{species}__{os.path.splitext(fname)[0]}"),
               raw, label)


def interleaved_samples(zips, dataset, interleave, label_lookup, stats):
    """최대 `interleave`개 zip을 동시에 열어 라운드로빈으로 샘플 yield (종 섞기).
    각 zip은 내부적으로 순차 읽기(HDD 친화). zip 소진 시 다음 zip 투입.
    `stats` 는 손상 멤버 집계용 mutable dict(워커 프로세스별)."""
    zq = collections.deque(zips)

    def open_next():
        while zq:
            zp = zq.popleft()
            try:
                z = zipfile.ZipFile(zp)
            except Exception as e:                    # zip 자체가 못 열리면 통째 스킵
                stats["bad_zip"] += 1
                print(f"    ⚠️ 손상 zip 스킵: {os.path.basename(zp)} — "
                      f"{type(e).__name__}: {e}", flush=True)
                continue
            base = _cp949(os.path.basename(zp))
            sp612 = _species_612(base) if dataset == "612" else None
            return [base, _zip_image_gen(z, dataset, sp612, label_lookup, stats), z]
        return None

    live = []
    while len(live) < interleave:
        nx = open_next()
        if nx is None:
            break
        live.append(nx)

    i = 0
    while live:
        i %= len(live)
        _, gen, z = live[i]
        try:
            yield next(gen)
            i += 1
        except StopIteration:
            z.close()
            nx = open_next()
            if nx is None:
                live.pop(i)          # i 그대로(다음 원소가 당겨짐)
            else:
                live[i] = nx
                i += 1


# ---------------------------------------------------------------- 리사이즈(워커)
def _resize(task, long_side, quality):
    """(key, raw, label) → (key, jpg_bytes, json_bytes) | (key, None, None) on error."""
    key, raw, label = task
    try:
        im = Image.open(io.BytesIO(raw)).convert("RGB")
        w, h = im.size
        if max(w, h) > long_side:                    # 작은 건 확대 안 함
            s = long_side / max(w, h)
            im = im.resize((round(w * s), round(h * s)), Image.LANCZOS)
        label["width"], label["height"] = im.size
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=quality)
        return (key, buf.getvalue(), json.dumps(label, ensure_ascii=False).encode("utf-8"))
    except Exception:
        return (key, None, None)


# ---------------------------------------------------------------- 샤드 라이터
class ShardWriter:
    """stdlib tarfile 기반. 샘플(jpg+json) 단위로 쓰고 *그 후* 크기 검사 → 짝 무결성."""
    def __init__(self, pattern: str, maxsize: int):
        self.pattern, self.maxsize = pattern, maxsize
        self.idx = -1
        self.tar = None
        self.size = self.count = self.total = 0
        self.shards = []
        self._roll()

    def _roll(self):
        if self.tar:
            self.tar.close()
        self.idx += 1
        self.size = self.count = 0
        path = self.pattern % self.idx
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.tar = tarfile.open(path, "w")
        self.shards.append(path)

    def write(self, key: str, files: dict):
        for ext, data in files.items():                # jpg, json 한꺼번에
            info = tarfile.TarInfo(f"{key}.{ext}")
            info.size = len(data)
            self.tar.addfile(info, io.BytesIO(data))
            self.size += len(data) + 512
        self.count += 1
        self.total += 1
        if self.size >= self.maxsize:                  # 샘플 다 쓴 "후" 검사 → 롤오버
            self._roll()

    def close(self):
        last = self.shards[-1]
        empty = self.count == 0
        if self.tar:
            self.tar.close()
        if empty and len(self.shards) > 1:             # 끝에 생긴 빈 샤드 제거
            os.remove(last)
            self.shards.pop()
        return self.shards


# ---------------------------------------------------------------- 병렬 배치 워커
def shard_batch(job):
    """워커 프로세스 1개 = zip 배치 1개. 배치를 인터리브(종 섞기)·768px 리사이즈해
    자기 전용 샤드(`..._wNN_*.tar`)에 기록. 부분 매니페스트 반환.

    참고: --label_index 사용 시 워커마다 parquet을 독립 로드(메모리 W배). 612 대용량
    인덱스에서 부담되면 인덱스 생략(종/부위는 inline 도출됨) 후 학습 시 join 권장."""
    wid, zips, p = job
    label_lookup = load_label_lookup(p["label_index"]) if p["label_index"] else {}
    pattern = os.path.join(p["out"], f'{p["dataset"]}_{p["split"]}_w{wid:02d}_%06d.tar')
    writer = ShardWriter(pattern, p["shard_size"])
    species_counts = collections.Counter()
    stats = {"corrupt": 0, "bad_zip": 0}          # 손상 멤버·손상 zip 집계(워커별)
    miss = errs = 0
    for key, raw, label in interleaved_samples(zips, p["dataset"], p["interleave"], label_lookup, stats):
        if label_lookup and label.get("_label_index_miss"):
            miss += 1
        k, jpg, js = _resize((key, raw, label), p["long_side"], p["quality"])
        if jpg is None:
            errs += 1
            continue
        writer.write(k, {"jpg": jpg, "json": js})
        species_counts[k.split("__")[1] if "__" in k else "?"] += 1
        if writer.total % 2000 == 0:
            print(f"  [w{wid:02d}] {writer.total}장 → 샤드 {len(writer.shards)}개 (err {errs})", flush=True)
    shards = writer.close()
    return {"wid": wid, "num_shards": len(shards), "num_samples": writer.total,
            "decode_errors": errs, "label_index_miss": miss,
            "corrupt_members": stats["corrupt"], "bad_zips": stats["bad_zip"],
            "species_counts": dict(species_counts),
            "shards": [os.path.basename(s) for s in shards]}


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/mnt/nas-rawtext/한의학/aihub")
    ap.add_argument("--dataset", choices=["151", "612"], required=True)
    ap.add_argument("--split", choices=["train", "val"], required=True)
    ap.add_argument("--out", default="data/shards")
    ap.add_argument("--label_index", default="", help="parquet 인덱스 dir(선택) → 라벨 보강")
    ap.add_argument("--long_side", type=int, default=768)
    ap.add_argument("--quality", type=int, default=90)
    ap.add_argument("--shard_size", type=float, default=1e9, help="샤드 목표 바이트(소프트)")
    ap.add_argument("--interleave", type=int, default=8, help="워커당 동시에 여는 zip 수(종 섞기)")
    ap.add_argument("--workers", type=int, default=8, help="병렬 배치 워커 프로세스 수")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    manifest_path = os.path.join(args.out, f"{args.dataset}_{args.split}_manifest.json")
    if os.path.exists(manifest_path) and not args.overwrite:
        print(f"이미 샤딩됨: {manifest_path} (덮어쓰려면 --overwrite)", flush=True)
        return

    zips = find_source_zips(args.root, args.dataset, args.split)
    print(f"{args.dataset}/{args.split} 원천 zip {len(zips)}개", flush=True)
    if not zips:
        print("원천 zip 없음 — 다운로드 미완 가능.", flush=True)
        return
    if args.dry_run:
        for z in zips[:8]:
            b = _cp949(os.path.basename(z))
            # 151 종은 zip 내부 폴더에서 도출(여기선 zip명에서 근사 표시), 612는 접두 제거
            sp = _species_612(b) if args.dataset == "612" else os.path.splitext(b)[0]
            print(f"  {b}  → species≈{sp}")
        return

    # 종(zip)을 라운드로빈으로 W개 배치에 분배 → 각 배치가 다양한 종을 포함(샤드 내 혼합 보존)
    W = max(1, min(args.workers, len(zips)))
    batches = [zips[i::W] for i in range(W)]
    params = {"dataset": args.dataset, "split": args.split, "out": args.out,
              "label_index": args.label_index, "long_side": args.long_side,
              "quality": args.quality, "interleave": args.interleave,
              "shard_size": int(args.shard_size)}
    os.makedirs(args.out, exist_ok=True)
    print(f"병렬 배치: {W} 워커 × 평균 {len(zips)/W:.1f} zip "
          f"(라벨 인덱스 {'사용' if args.label_index else '미사용'})", flush=True)

    parts = []
    with ProcessPoolExecutor(max_workers=W) as ex:
        for r in ex.map(shard_batch, [(i, batches[i], params) for i in range(W)]):
            parts.append(r)
            print(f"  [w{r['wid']:02d}] 완료: {r['num_samples']}장 / 샤드 {r['num_shards']}개 "
                  f"/ 종 {len(r['species_counts'])} / 디코드오류 {r['decode_errors']} "
                  f"/ 손상멤버 {r['corrupt_members']} / 손상zip {r['bad_zips']}", flush=True)

    # 부분 매니페스트 병합
    sc = collections.Counter()
    shards = []
    for p in parts:
        sc.update(p["species_counts"])
        shards += p["shards"]
    total = sum(p["num_samples"] for p in parts)
    errs = sum(p["decode_errors"] for p in parts)
    miss = sum(p["label_index_miss"] for p in parts)
    corrupt = sum(p.get("corrupt_members", 0) for p in parts)
    bad_zips = sum(p.get("bad_zips", 0) for p in parts)
    manifest = {
        "dataset": args.dataset, "split": args.split,
        "long_side": args.long_side, "quality": args.quality,
        "shard_size": int(args.shard_size), "workers": W,
        "num_shards": len(shards), "num_samples": total,
        "decode_errors": errs, "label_index_miss": miss,
        "corrupt_members": corrupt, "bad_zips": bad_zips,
        "num_species": len(sc),
        "species_counts": dict(sc),
        "shards": sorted(shards),
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"\n완료: {total}장 / 샤드 {len(shards)}개 / 종 {len(sc)} "
          f"/ 디코드오류 {errs} / 라벨미스 {miss} / 손상멤버 {corrupt} / 손상zip {bad_zips}", flush=True)
    print(f"  매니페스트 → {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
