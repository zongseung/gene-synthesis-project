#!/usr/bin/env python3
"""herb_shard_index.json 빌더 — WebDataset tar 샤드 → 논리경로 인덱스.

트레이너(ShardImageReader)는 약초 이미지를 `herb_shard_index.json`
(논리경로 → [tar_basename, member])로만 해소한다. shard.py 는 tar+manifest 만
만들고 이 인덱스는 안 만들므로, 샤딩 후 이 스크립트로 인덱스를 (재)생성해야 한다.

매핑 규칙(검증됨): tar member `{ds}__{종}__{파일}.jpg` → 논리 `{ds}/{종}/{파일}.jpg`
                = member.replace('__', '/'). SFT jsonl 의 image 필드와 정확히 일치.

사용:
  # 612 만 추가(기존 151 인덱스 보존, 병합):
  PYTHONPATH=src .venv/bin/python scripts/build_herb_shard_index.py --datasets 612 --merge

  # 전체 재빌드(모든 데이터셋 tar 스캔):
  PYTHONPATH=src .venv/bin/python scripts/build_herb_shard_index.py --datasets 151 612

  # 실제 쓰지 않고 통계만:
  ... --datasets 612 --merge --dry_run
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import sys
import tarfile
from collections import Counter


def scan_dataset_tars(shard_dir: str, dataset: str):
    """{shard_dir}/{dataset}_*.tar 을 스캔해 (logical, [tar_base, member]) 를 yield.
    스트리밍 순회로 truncated tar 도 직전까지 읽고, 손상 tar 는 (이름, 읽은수) 로 보고."""
    tars = sorted(glob.glob(os.path.join(shard_dir, f"{dataset}_*.tar")))
    broken = []
    for t in tars:
        base = os.path.basename(t)
        cnt = 0
        try:
            with tarfile.open(t) as tf:
                for m in tf:                      # 스트리밍 — 끊겨도 직전까지
                    name = m.name
                    if not name.endswith(".jpg"):
                        continue
                    member = os.path.basename(name)   # 평평한 멤버명 방어
                    logical = member.replace("__", "/")
                    cnt += 1
                    yield logical, [base, member]
        except tarfile.ReadError as e:
            broken.append((base, cnt, str(e)))
    scan_dataset_tars.broken = broken            # 호출측에서 회수
    scan_dataset_tars.tar_count = len(tars)


def main():
    ap = argparse.ArgumentParser(description="herb_shard_index.json 빌더")
    ap.add_argument("--shard_dir", default="data/shards")
    ap.add_argument("--out", default="data/shards/herb_shard_index.json")
    ap.add_argument("--datasets", nargs="+", required=True,
                    help="인덱싱할 데이터셋 (예: 612 또는 151 612)")
    ap.add_argument("--merge", action="store_true",
                    help="기존 --out 인덱스에 병합(지정 데이터셋만 갱신). 없으면 신규 생성.")
    ap.add_argument("--dry_run", action="store_true", help="쓰지 않고 통계만")
    args = ap.parse_args()

    index = {}
    if args.merge and os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            index = json.load(f)
        print(f"기존 인덱스 로드: {len(index)} 엔트리 ({args.out})", flush=True)
        # 병합 시, 갱신 대상 데이터셋의 낡은 엔트리는 제거 후 재삽입(부분/이전 샤딩 잔재 제거)
        for ds in args.datasets:
            before = len(index)
            index = {k: v for k, v in index.items() if not k.startswith(f"{ds}/")}
            if before - len(index):
                print(f"  {ds}/ 기존 엔트리 {before - len(index)}개 제거(재삽입 예정)", flush=True)

    added = Counter()
    species = {ds: set() for ds in args.datasets}
    all_broken = []
    for ds in args.datasets:
        n0 = len(index)
        for logical, loc in scan_dataset_tars(args.shard_dir, ds):
            index[logical] = loc
            added[ds] += 1
            parts = logical.split("/")
            if len(parts) >= 2:
                species[ds].add(parts[1])
        brk = getattr(scan_dataset_tars, "broken", [])
        tc = getattr(scan_dataset_tars, "tar_count", 0)
        all_broken += brk
        print(f"[{ds}] tar {tc}개 → 엔트리 +{added[ds]} / 종 {len(species[ds])}"
              f" / 손상tar {len(brk)}", flush=True)
        for b, c, e in brk:
            print(f"    ⚠️ 손상: {b} (읽은 {c}장) — {e}", flush=True)

    print(f"\n최종 인덱스: {len(index)} 엔트리", flush=True)
    for ds in args.datasets:
        print(f"  {ds}: {added[ds]} 엔트리 / {len(species[ds])} 종", flush=True)

    if all_broken:
        print(f"\n❌ 손상 tar {len(all_broken)}개 — 샤딩 미완/중단 의심. "
              "샤딩 완료 후 다시 실행 권장.", flush=True)

    if args.dry_run:
        print("\n(dry_run — 쓰지 않음)", flush=True)
        return 0

    tmp = args.out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False)
    os.replace(tmp, args.out)
    print(f"\n✅ 저장 → {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
