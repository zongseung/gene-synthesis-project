"""종 단위 지식 카드 빌더 — 이미지 라벨 ↔ 동의보감 고전지식(학명 피벗).

label_index(이미지 라벨) · 04_linked(본초↔crosswalk) · safety_kb(고전지식) · crosswalk(학명)를
species_ko 키로 병합해 `species_annotation.jsonl`(종 1행=1 지식 카드)을 만든다.
이미지마다 복제하지 않고, 학습/RAG는 (dataset, species_ko)로 런타임 조인한다.

조인 체인:
  image label(species_ko) → crosswalk(species_ko↔scientific_name)
                          → 04_linked 역인덱스(species_ko→herb_hanja)
                          → safety_kb/classical(효능·주치·금기·독성·출처)

설계: claudedocs/design_safety_kb_image_annotation_20260625.md

사용:
  PYTHONPATH=src .venv/bin/python -m hanmed.knowledge.annotate \
      --shards 'data/label_index/shards/*.parquet' \
      --linked _workspace/boncho/04_linked.jsonl \
      --safety_kb data/safety_kb/classical \
      --crosswalk data/crosswalk.parquet \
      --out data/annotations
"""
from __future__ import annotations
import argparse, glob, json, os, re, collections

import pandas as pd
import pyarrow.parquet as pq


def _norm_herb(h: str) -> str:
    """'人參/main' 같은 verify 접미사 제거."""
    return re.sub(r"/(main|전역|.*)$", "", str(h)).strip()


# ── 1. 본초 역인덱스: species_ko → [(herb_hanja, confidence)] ───────────────
def build_species2herb(linked_jsonl: str) -> dict[str, list[tuple[str, str]]]:
    sp2herb: dict[str, list[tuple[str, str]]] = collections.defaultdict(list)
    with open(linked_jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            conf = r.get("link_confidence")
            if conf not in ("resolved", "ambiguous"):
                continue
            m = r.get("crosswalk_match")
            if isinstance(m, list):          # ambiguous 1:N 후보 리스트
                m = m[0] if m and isinstance(m[0], dict) else {}
            if not isinstance(m, dict):
                continue
            sp = m.get("species_ko")
            if sp:
                sp2herb[sp].append((_norm_herb(r["herb_hanja"]), conf))
    return dict(sp2herb)


# ── 2. 고전지식 적재: herb_hanja → knowledge ──────────────────────────────
def load_safety_kb(classical_dir: str) -> dict[str, dict]:
    kb: dict[str, dict] = {}
    for fp in glob.glob(os.path.join(classical_dir, "*.json")):
        d = json.load(open(fp, encoding="utf-8"))
        kb[_norm_herb(d["herb_hanja"])] = d      # 동명이부 충돌은 드묾(첫/마지막 우선)
    return kb


# ── 3. 이미지 종 집계: species_ko → 메타 ───────────────────────────────────
def collect_image_species(shards_glob: str) -> dict[str, dict]:
    agg: dict[str, dict] = {}
    for s in sorted(glob.glob(shards_glob)):
        df = pq.read_table(
            s, columns=["dataset", "species_ko", "species_code", "herb_name",
                        "is_poisonous", "classification"]
        ).to_pandas()
        for sp, g in df.groupby("species_ko"):
            if not sp:
                continue
            a = agg.setdefault(sp, {
                "datasets": set(), "species_code": None, "herb_name_label": None,
                "n_images": 0, "n_pois_true": 0, "has_similar": False,
            })
            a["datasets"].update(x for x in g["dataset"].dropna().unique())
            a["n_images"] += len(g)
            a["n_pois_true"] += int(g["is_poisonous"].fillna(False).sum())
            if (g["classification"] == "similar").any():
                a["has_similar"] = True
            code = g["species_code"].dropna()
            if a["species_code"] is None and len(code):
                a["species_code"] = str(code.iloc[0])
            hn = g["herb_name"].dropna()              # 612 라벨 생약명(있으면)
            if a["herb_name_label"] is None and len(hn):
                a["herb_name_label"] = str(hn.iloc[0])
    return agg


# ── 4. 병합 ───────────────────────────────────────────────────────────────
def _classical_block(herbs: list[tuple[str, str]], kb: dict[str, dict]) -> tuple[dict | None, str]:
    """linked herb_hanja들의 지식을 병합. (block, knowledge_status) 반환."""
    confs = {c for _, c in herbs}
    status = "linked" if "resolved" in confs else "ambiguous"
    효능, 주치, 금기, 출처, hanja = [], [], [], [], []
    seongmi, 독성 = None, None
    for h, _ in herbs:
        k = kb.get(h)
        if not k:
            continue
        hanja.append(h)
        효능 += [x for x in k.get("효능", []) if x not in 효능]
        주치 += [x for x in k.get("주치", []) if x not in 주치]
        금기 += [x for x in k.get("금기", []) if x not in 금기]
        for refs in (k.get("field_provenance") or {}).values():
            출처 += [r for r in refs if r not in 출처]
        if seongmi is None and k.get("seongmi"):
            seongmi = k["seongmi"]
            독성 = (k["seongmi"] or {}).get("독성")
    if not hanja:
        return None, "unlinked"
    return {
        "herb_hanja": hanja, "효능": 효능, "주치": 주치, "금기": 금기,
        "seongmi": seongmi, "독성_고전": 독성, "출처": 출처,
        "link_confidence": "resolved" if "resolved" in confs else "ambiguous",
    }, status


def merge_annotation(img_agg, sp2herb, kb, sci_map) -> list[dict]:
    rows = []
    for sp, a in sorted(img_agg.items()):
        herbs = sp2herb.get(sp, [])
        classical, status = (_classical_block(herbs, kb) if herbs else (None, "unlinked"))
        n = a["n_images"]
        sci, sci_conf = sci_map.get(sp) or (None, None)
        rows.append({
            "species_ko": sp,
            "dataset": sorted(a["datasets"]),
            "species_code": a["species_code"],
            "scientific_name": sci,
            "scientific_name_confidence": sci_conf,   # resolved 일 때만 학명 단정 가능
            "herb_name_label": a["herb_name_label"],             # 612 라벨 생약명(있으면)
            "is_poisonous": a["n_pois_true"] > 0,                 # 안전 방향: 하나라도 양성이면 양성
            "pois_conflict": 0 < a["n_pois_true"] < n,            # 종 내 라벨 불일치 플래그
            "has_similar_class": a["has_similar"],
            "n_images": n,
            "classical": classical,
            "knowledge_status": status,
        })
    return rows


def load_sci_map(crosswalk_path: str) -> dict[str, tuple[str, str]]:
    """species_ko → (학명, crosswalk confidence).

    confidence 를 버리면 안 된다. 은조롱의 Cynanchum wilfordii 는 ambiguous(gbif)
    인데 학명만 넘어가 SFT 가 「식별 결과는 은조롱(Cynanchum wilfordii)입니다」로
    단정했다. 소비자(build_sft_mm.render_T1)가 resolved 만 단정하도록 같이 넘긴다.
    """
    df = pd.read_parquet(crosswalk_path)
    out = {}
    for _, r in df.iterrows():
        sp, sci = r.get("species_ko"), r.get("scientific_name")
        conf = r.get("confidence")
        if sp and isinstance(sci, str) and sci.strip():
            out.setdefault(sp, (sci, conf if isinstance(conf, str) else None))
    return out


# ── 5. 출력 ───────────────────────────────────────────────────────────────
def write_annotation(rows, out_jsonl):
    os.makedirs(os.path.dirname(out_jsonl), exist_ok=True)
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_build_report(rows, out_md):
    n = len(rows)
    by_status = collections.Counter(r["knowledge_status"] for r in rows)
    pois = sum(1 for r in rows if r["is_poisonous"])
    conflict = [r["species_ko"] for r in rows if r["pois_conflict"]]
    dangling = [r["species_ko"] for r in rows
                if r["knowledge_status"] != "unlinked" and not (r["classical"] and r["classical"]["herb_hanja"])]
    linked_sci = sum(1 for r in rows if r["scientific_name"])
    lines = [
        "# species_annotation 빌드 리포트", "",
        f"- 종(species_ko): **{n}**",
        f"- knowledge_status: {dict(by_status)}",
        f"- 학명 보유: {linked_sci}/{n}",
        f"- is_poisonous=True: {pois}종 (독초판별 양성)",
        f"- 독성 라벨 불일치(pois_conflict) 종 {len(conflict)}개: {conflict[:20]}",
        f"- dangling(linked인데 safety_kb 없음): {len(dangling)}개 {dangling[:20]}",
        "",
        "## knowledge_status 정의",
        "- linked: resolved 본초 연결 + 고전지식 보유",
        "- ambiguous: 1:N 기원종(복수 herb_hanja) — T3 학습은 기본 제외 권장",
        "- unlinked: 이미지 종이 크로스워크/본초에 미연결 (정상: 비식물·미수록·다운로드 미완)",
    ]
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default="data/label_index/shards/*.parquet")
    ap.add_argument("--linked", default="_workspace/boncho/04_linked.jsonl")
    ap.add_argument("--safety_kb", default="data/safety_kb/classical")
    ap.add_argument("--crosswalk", default="data/crosswalk.parquet")
    ap.add_argument("--out", default="data/annotations")
    args = ap.parse_args()

    print("역인덱스·고전지식·이미지 종 적재 중…", flush=True)
    sp2herb = build_species2herb(args.linked)
    kb = load_safety_kb(args.safety_kb)
    sci_map = load_sci_map(args.crosswalk)
    img_agg = collect_image_species(args.shards)
    print(f"  이미지 종 {len(img_agg)} · 본초연결 종 {len(sp2herb)} · safety_kb {len(kb)} · 학명 {len(sci_map)}",
          flush=True)

    rows = merge_annotation(img_agg, sp2herb, kb, sci_map)
    out_jsonl = os.path.join(args.out, "species_annotation.jsonl")
    write_annotation(rows, out_jsonl)
    write_build_report(rows, os.path.join(args.out, "build_report.md"))

    by_status = collections.Counter(r["knowledge_status"] for r in rows)
    print(f"\n완료: {len(rows)}종 → {out_jsonl}")
    print(f"  knowledge_status: {dict(by_status)}")
    print(f"  is_poisonous=True: {sum(1 for r in rows if r['is_poisonous'])}종")


if __name__ == "__main__":
    main()
