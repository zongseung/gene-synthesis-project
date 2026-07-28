"""원천데이터 zip(NAS) → 로컬 이미지 스테이징.

SFT jsonl(mm_*.jsonl)이 참조하는 이미지만 골라, NAS 원천데이터 zip에서 추출해
trainer의 --image_root 가 기대하는 논리 경로(`{dataset}/{code_species}/{filename}`)로 저장한다.
NAS가 느리므로: 종별 병렬 · 이미 받은 파일 skip(resume) · per-zip try/except · .part(미완) 제외.

논리 경로는 build_sft_mm._image_path 와 정확히 일치해야 한다(= trainer가 그대로 조인).

사용:
  PYTHONPATH=src .venv/bin/python -m hanmed.stage2_vlm.stage_images \
      --sft data/sft/mm_train.jsonl data/sft/mm_val.jsonl \
      --out data/images --linked_only --annotation data/annotations/species_annotation.jsonl \
      --workers 4 --limit_species 10            # 우선 10종만 증분 스테이징
  # 매핑·커버리지만 점검: --dry_run
"""
from __future__ import annotations
import argparse, glob, json, os, zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed


def _cp949(name: str) -> str:
    try:
        return name.encode("cp437").decode("cp949")
    except Exception:
        return name


def find_source_zips(root: str) -> list[str]:
    """원천데이터 zip 경로 목록 (.part 제외). 고정깊이 glob."""
    out = []
    for ds in ("151", "612"):
        pat = os.path.join(root, ds, "*", "01.데이터", "*", "원천데이터", "*.zip")
        out += [p for p in glob.glob(pat) if not p.endswith(".part")
                and not os.path.basename(p).count(".part")]
    return sorted(out)


def needed_images(sft_paths, out_root):
    """SFT jsonl → {(dataset, sub): {filename: local_path}}  (sub=code_species or species)."""
    need = {}
    for p in sft_paths:
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            rel = r["image"]                       # dataset/sub/filename
            ds = r["dataset"]
            parts = rel.split("/")
            if len(parts) < 3:
                continue
            sub, fname = parts[1], parts[-1]
            need.setdefault((ds, sub), {})[fname] = os.path.join(out_root, rel)
    return need


def stage_one(args):
    """워커: (ds, sub, want{fname:dest}, zip_candidates, cap) → (sub, staged, missing)."""
    ds, sub, want, zips, cap = args
    todo = {f: d for f, d in want.items() if not os.path.exists(d)}   # resume
    if cap:
        todo = dict(list(todo.items())[:cap])                        # 종당 추출 상한
    if not todo:
        return (sub, 0, 0)
    staged = 0
    for zp in zips:
        if not todo:
            break
        try:
            with zipfile.ZipFile(zp) as z:
                for zi in z.infolist():
                    if zi.is_dir():
                        continue
                    base = os.path.basename(_cp949(zi.filename))
                    dest = todo.get(base)
                    if not dest:
                        continue
                    os.makedirs(os.path.dirname(dest), exist_ok=True)
                    with z.open(zi) as src, open(dest, "wb") as f:
                        f.write(src.read())
                    staged += 1
                    del todo[base]
        except Exception:
            continue
    return (sub, staged, len(todo))


def map_zips(need, all_zips):
    """(ds, sub) → 후보 source zip. 151은 code 접두 정확매칭, 612는 prefix 제거 후 종명 정확매칭.
    (substring 매칭 금지 — '나팔꽃'이 '둥근잎미국나팔꽃'에 걸리는 오매칭 방지)"""
    by_ds = {"151": [], "612": []}
    for z in all_zips:
        ds = "151" if "/151/" in z else ("612" if "/612/" in z else None)
        if ds:
            by_ds[ds].append((z, _cp949(os.path.basename(z))))
    out = {}
    for (ds, sub) in need:
        cands = []
        if ds == "151":
            code = sub.split("_", 1)[0]                       # '009_나팔꽃' → '009'
            cands = [z for z, bn in by_ds["151"] if bn.startswith(code + "_")]
        else:  # 612: sub=종명, zip은 TS_/VS_/TL_/VL_ 접두
            for z, bn in by_ds["612"]:
                stem = bn[:-4] if bn.lower().endswith(".zip") else bn
                for pre in ("TS_", "VS_", "TL_", "VL_"):
                    if stem.startswith(pre):
                        stem = stem[len(pre):]
                        break
                if stem == sub:
                    cands.append(z)
        out[(ds, sub)] = cands
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/mnt/nas-rawtext/한의학/aihub")
    ap.add_argument("--sft", nargs="+", default=["data/sft/mm_train.jsonl", "data/sft/mm_val.jsonl"])
    ap.add_argument("--out", default="data/images")
    ap.add_argument("--annotation", default="data/annotations/species_annotation.jsonl")
    ap.add_argument("--linked_only", action="store_true", help="고전지식 linked 종만 스테이징")
    ap.add_argument("--limit_species", type=int, default=0, help="우선 N종만(증분)")
    ap.add_argument("--max_per_species", type=int, default=0, help="종당 추출 이미지 상한(검증용, 0=전체)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dry_run", action="store_true", help="zip 매핑·커버리지만 출력")
    args = ap.parse_args()

    need = needed_images(args.sft, args.out)
    if args.linked_only:
        linked_sub = set()
        for line in open(args.annotation, encoding="utf-8"):
            a = json.loads(line)
            if a["knowledge_status"] == "linked":
                code = a.get("species_code")
                linked_sub.add(f"{code}_{a['species_ko']}" if code else a["species_ko"])
        need = {(ds, sub): v for (ds, sub), v in need.items() if sub in linked_sub}

    subs = sorted(need.keys())
    if args.limit_species:
        subs = subs[:args.limit_species]
        need = {k: need[k] for k in subs}

    print(f"대상 종 {len(need)}개 · 필요 이미지 {sum(len(v) for v in need.values())}장", flush=True)
    print("원천데이터 zip 탐색 중(NAS — 느릴 수 있음)…", flush=True)
    all_zips = find_source_zips(args.root)
    print(f"  source zip {len(all_zips)}개 발견", flush=True)
    zmap = map_zips(need, all_zips)

    no_zip = [k for k, z in zmap.items() if not z]
    print(f"  zip 매칭됨 {len(need)-len(no_zip)}종 / 미발견 {len(no_zip)}종"
          f"{' (다운로드 미완 가능)' if no_zip else ''}", flush=True)
    if args.dry_run:
        for k in subs[:8]:
            print(f"    {k} → {[os.path.basename(z) for z in zmap[k]][:2]}")
        return

    tasks = [(ds, sub, need[(ds, sub)], zmap[(ds, sub)], args.max_per_species)
             for (ds, sub) in subs if zmap[(ds, sub)]]
    total_staged = total_missing = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(stage_one, t): t[1] for t in tasks}
        for i, f in enumerate(as_completed(futs), 1):
            sub, staged, missing = f.result()
            total_staged += staged; total_missing += missing
            print(f"[{i}/{len(tasks)}] {sub}: +{staged}장 (missing {missing})  "
                  f"누적 {total_staged}장", flush=True)
    print(f"\n완료: {total_staged}장 스테이징 → {args.out} / 못 찾은 이미지 {total_missing}장", flush=True)


if __name__ == "__main__":
    main()
