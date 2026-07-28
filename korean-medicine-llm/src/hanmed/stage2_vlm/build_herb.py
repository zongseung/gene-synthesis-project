"""VARCO-VISION SFT JSONL 빌더 — species_annotation + 이미지 라벨 → 대화쌍.

종 단위 지식 카드(`species_annotation.jsonl`)와 이미지 라벨 인덱스를 결합해
이미지 1장 = 대화 1쌍(LLaVA/VARCO conversations 포맷)을 생성한다.

태스크 (annotation 필드로 결정론 생성 → 환각 0):
  T1 약초ID    : 전체 종, 응답=species_ko(+학명, crosswalk confidence=resolved 일 때만)
  T2 독초판별  : 전체 종, 응답=tox_status 3값 (필드 없으면 실패 — 폴백 금지)
  T3 효능설명  : linked 종만, 응답=동의보감 효능·주치 + 출전 (ambiguous 기본 제외)

규칙: T3는 linked(resolved)만 — unlinked/ambiguous는 고전 응답 미생성(환각 차단).
      종당 이미지 샘플링 상한으로 클래스 균형, 드롭 수 로깅.

설계: claudedocs/design_safety_kb_image_annotation_20260625.md

사용:
  PYTHONPATH=src .venv/bin/python -m hanmed.stage2_vlm.build_herb \
      --annotation data/annotations/species_annotation.jsonl \
      --shards 'data/label_index/shards/*.parquet' \
      --out data/sft --per_species_cap 40
"""
from __future__ import annotations
import argparse, glob, json, os, collections

import pyarrow.parquet as pq

from hanmed.shared import _pick
from hanmed.shared.parts import PART_KO, NON_PART, PART_COMPAT

IMG_TOKEN = "<image>"

# 질문 패러프레이즈 (포맷 다양성 — 파일명 해시로 결정론 선택)
Q_ID = ["이 사진 속 약용식물의 이름은 무엇인가요?",
        "이 약초는 무엇입니까?",
        "사진에 보이는 식물의 이름을 알려주세요."]
Q_TOX = ["이 식물은 독성이 있나요?",
         "이 약초를 섭취해도 안전한가요?",
         "사진 속 식물의 독성 여부를 알려주세요."]
Q_EFF = ["이 약초는 어떤 효능이 있나요?",
         "이 약재는 어디에 좋은가요?",
         "이 약초의 주된 쓰임을 알려주세요."]

# 답변 프레이밍 변형 (사실 불변, 말투·도입부만 다양화 — 파일명 해시로 결정론 선택).
# 같은 종이라도 이미지마다 표현이 달라져 byte 단위 반복 암기·환각을 줄인다.
A_ID = [
    "{name}입니다.",
    "사진 속 식물은 {name}입니다.",
    "이 약용식물은 {name}입니다.",
    "식별 결과는 {name}입니다.",
]
A_TOX_TOXIC = [
    "독성이 있는 식물로 분류됩니다. 함부로 섭취하면 안 됩니다.",
    "이 식물은 독성이 있는 것으로 분류되므로 함부로 섭취해서는 안 됩니다.",
    "독성 식물로 분류됩니다. 섭취에 각별한 주의가 필요합니다.",
]
A_TOX_SAFE = [
    "독초로 분류되지 않은 식물입니다.",
    "문헌상 독초로 분류되지 않은 식물입니다.",
    "독성 식물로 분류되지는 않습니다.",
]
A_TOX_UNVERIFIED = [
    "이 식물의 독성 여부는 문헌으로 확인되지 않았습니다. 독성이 없다고 단정할 수 없으니 함부로 섭취하지 마세요.",
    "이 식물의 독성 여부가 문헌으로 확인되지 않아 안전하다고 단정할 수 없습니다. 함부로 섭취하지 마세요.",
    "독성 여부가 문헌으로 확인되지 않은 식물입니다. 안전을 단정할 수 없으므로 임의로 섭취하지 마세요.",
]
_TOX_SIMILAR_TOXIC = " 식용·약용 식물과 혼동하기 쉬우니 특히 주의가 필요합니다."
_TOX_SIMILAR_SAFE = " 다만 유사한 독초와 혼동될 수 있으니 정확한 식별이 중요합니다."

# 효능 도입부 변형. 모두 소유격("의")으로 끝나 뒤따르는 효능/주치 문장에 자연스럽게 이어진다.
A_EFF_LEAD = [
    "동의보감 탕액편에 따르면 {sp}(한약명 {hanja})의",
    "전통 본초서 동의보감(탕액편) 기준으로 보면 {sp}(한약명 {hanja})의",
    "동의보감 탕액편에 기록된 {sp}(한약명 {hanja})의",
]
# 효능/주치 미확인·미커버(unlinked) 종용 abstain — 모델이 "모른다"를 학습하게 한다.
A_EFF_ABSTAIN = [
    "동의보감 본초 문헌에서 이 약재의 효능 기술을 확인하지 못해, 효능을 단정하지 않습니다.",
    "이 약재의 효능은 동의보감 본초 문헌에서 확인되지 않아, 효능을 단정하지 않습니다.",
    "동의보감 등 본초 문헌에서 효능 기술을 찾지 못해, 이 약재의 효능을 단정하기 어렵습니다.",
]

def _euro(word):
    """받침에 맞는 조사 '으로'/'로' 선택. 받침 없음·ㄹ받침 → '로', 그 외 → '으로'.
    (예: 잎→잎으로, 꽃→꽃으로, 껍질→껍질로, 뿌리→뿌리로)"""
    if not word:
        return "로"
    code = ord(word[-1])
    if 0xAC00 <= code <= 0xD7A3:
        jong = (code - 0xAC00) % 28
        if jong not in (0, 8):   # 받침 있고 ㄹ(8)이 아니면 '으로'
            return "으로"
    return "로"


def _split_of(label_zip: str) -> str:
    """label_zip → train/val. 151: *train*→train,*val*→val / 612: TL_*→train, VS_*→val."""
    z = (label_zip or "").lower()
    if "val" in z or z.startswith("vs_") or "vs_" in z:
        return "val"
    return "train"


def render_T1(ann, q, key):
    """약초 ID. 사실은 종명 하나뿐이라 **말투만** 돌린다 — 시각 주장을 새로 만들지 않는다.

    문구가 하나였을 때 종당 고유 답변이 1개라 같은 문장이 100회 반복됐다(실측).
    조사 문제를 피하려고 모든 템플릿이 「…입니다」로 끝난다 — 학명 괄호 뒤에는
    받침 판정을 할 수 없다.

    학명은 crosswalk confidence 가 resolved 일 때만 단정한다. 은조롱의
    Cynanchum wilfordii 는 ambiguous(gbif) 인데도 정답으로 단정돼 있었다.
    필드가 아예 없으면 확인된 것으로 치지 않는다 — 한국명만 내면 틀릴 일이 없고,
    빠뜨렸을 때의 손해가 단정했을 때의 손해보다 작다.
    """
    sci = (ann.get("scientific_name")
           if ann.get("scientific_name_confidence") == "resolved" else None)
    name = f"{ann['species_ko']}({sci})" if sci else ann["species_ko"]
    return {"from": "gpt", "value": _pick(A_ID, key).format(name=name)}, q


def render_T2(ann, q, key):
    """안전 게이트: tox_status(toxic/safe_documented/unverified)로 3분기.
    unverified(미검증)는 '안전' 단정 금지 → abstain(안전 우선).

    **폴백 없음.** tox_status 가 없거나 모르는 값이면 답변을 만들지 않고 죽는다.
    예전 폴백은 is_poisonous 를 봤는데, 그 라벨은 「미검증」과 「무독 확인」을 구분하지
    못해 False 인 종 전부가 「독초로 분류되지 않습니다」라는 안전 단정으로 렌더됐다
    (오늘 unverified 8,085행이 전부 해당). 이 프로젝트 최악의 실패 모드라
    조용한 기본값을 두지 않는다.
    """
    tox = ann.get("tox_status")
    if tox == "toxic":
        a = _pick(A_TOX_TOXIC, key)
        if ann.get("has_similar_class"):
            a += _TOX_SIMILAR_TOXIC
    elif tox == "safe_documented":
        a = _pick(A_TOX_SAFE, key)
        if ann.get("has_similar_class"):
            a += _TOX_SIMILAR_SAFE
    elif tox == "unverified":
        # 독성 미검증 → 안전하다고 단정하지 않고 섭취 자제 권고.
        a = _pick(A_TOX_UNVERIFIED, key)
    else:
        raise ValueError(
            f"{ann.get('species_ko')}: tox_status={tox!r} — 독성 답변을 만들 수 없다. "
            "species_annotation.jsonl 을 만드는 hanmed.knowledge.annotate 가 종마다 "
            "toxic/safe_documented/unverified 중 하나를 넣어야 한다."
        )
    return {"from": "gpt", "value": a}, q


def render_T3(ann, q, key, part=None):
    """동의보감 효능·주치 인용. 효능/주치가 비거나 unlinked면 None 대신 abstain 답변.
    part(이미지 부위, 영문)가 약용부위(medicinal_part_ko)와 불일치하면 효능 끝에 캐비엇 부착."""
    c = ann.get("classical") or {}
    효능 = c.get("효능") or []
    주치 = c.get("주치") or []
    if not 효능 and not 주치:
        # abstain: 모델이 미커버 종에 효능을 환각하지 않고 "모른다"를 학습하게 한다.
        return {"from": "gpt", "value": _pick(A_EFF_ABSTAIN, key)}, q
    hanja = "·".join(c.get("herb_hanja", []))
    # 도입부는 말투만 변형(사실 불변). 가변 내용(효능/주치) 뒤엔 항상 고정어(등·으로는)로 조사 어색함 방지.
    parts = [_pick(A_EFF_LEAD, key).format(sp=ann["species_ko"], hanja=hanja)]
    if 효능:
        parts.append(f" 주요 효능으로는 {', '.join(효능[:6])} 등이 있습니다.")
    if 주치:
        parts.append(f" 주치는 {', '.join(주치[:5])} 등입니다.")
    금기 = c.get("금기") or []
    if 금기:
        parts.append(f" 금기로는 {', '.join(금기[:3])} 등이 있어 주의가 필요합니다.")
    # 부위 불일치 캐비엇: 약용부위가 명시돼 있고, 이미지 부위가 그와 호환되지 않을 때만.
    # medicinal_part_ko 가 비어 있으면(미상) 거짓 단정 방지 위해 캐비엇 생략.
    med = ann.get("medicinal_part_ko") or []
    if med and part:
        compat = PART_COMPAT.get(part, set())
        if not (compat & set(med)):
            img_ko = PART_KO.get(part, part)
            parts.append(
                f" (다만 약재로 쓰는 부위는 {', '.join(med)}이며, 사진은 {img_ko}{_euro(img_ko)} 보입니다."
                " 부위에 따라 약효가 다를 수 있습니다.)"
            )
    return {"from": "gpt", "value": "".join(parts)}, q


# 군락·전초 사진이 표본에서 차지할 상한. 라운드로빈이 부위 수로 균등 배분하면
# 장수가 적은 군락이 5.7% → 17.3% 로 뛴다(실측). 종 식별에 가장 덜 유용한 사진이다.
NON_PART_MAX_FRAC = 0.10


def allocate(parts: dict, cap: int) -> list:
    """부위별 라운드로빈으로 cap 장 배분. 결정론적(파일명 정렬).

    앞에서 cap 장을 자르면 부위가 한쪽으로 쏠린다 — 실측으로 206종 중 76종이
    부위 1종류만 뽑혔다. 데이터에는 종당 부위가 중앙 4종류 있는데 버리고 있었다.
    """
    buckets = {p: sorted(v, key=lambda r: r[3]) for p, v in parts.items() if v}
    limit = {p: (max(1, int(cap * NON_PART_MAX_FRAC)) if p in NON_PART else cap)
             for p in buckets}
    order = sorted(buckets)
    out, taken, i = [], collections.Counter(), 0
    def room(p):
        return taken[p] < min(len(buckets[p]), limit[p])
    while len(out) < cap and any(room(p) for p in order):
        p = order[i % len(order)]
        if room(p):
            out.append(buckets[p][taken[p]])
            taken[p] += 1
        i += 1
    # 비-군락 사진이 부족해 cap 을 못 채우면 남은 것으로 채운다(상한보다 장수 우선).
    for p in order:
        while len(out) < cap and taken[p] < len(buckets[p]):
            out.append(buckets[p][taken[p]])
            taken[p] += 1
    return out


def iter_image_rows(shards_glob, train_cap, val_cap, shard_index=None):
    """종당 split별 상한까지, 부위 층화. yield (species_ko, dataset, part, fname, zip, split)."""
    caps = {"train": train_cap, "val": val_cap}
    pool = collections.defaultdict(lambda: collections.defaultdict(list))
    code_of = {}
    for s in sorted(glob.glob(shards_glob)):
        df = pq.read_table(
            s, columns=["dataset", "species_ko", "species_code", "part",
                        "image_filename", "label_zip"]
        ).to_pandas()
        for r in df.itertuples(index=False):
            sp = r.species_ko
            if not sp or not r.image_filename:
                continue
            code_of.setdefault(sp, r.species_code)
            split = _split_of(r.label_zip)
            pool[(sp, split)][r.part].append(
                (sp, r.dataset, r.part, r.image_filename, r.label_zip, split))

    index = None
    if shard_index and os.path.exists(shard_index):
        with open(shard_index, encoding="utf-8") as fh:
            index = set(json.load(fh))

    for (sp, split), parts in sorted(pool.items()):
        if index is not None:
            # 샤드에 실제로 있는 이미지만 남긴다. 부위 층화가 없는 이미지를 고르면
            # 그 자리는 학습에서 통째로 버려진다 — 612 는 종별로 보유 부위가 다르다
            # (예: 가는장구채는 꽃·전초·잎만 있고 열매는 샤드에 없음).
            kept = {p: [r for r in v
                        if _image_path(r[1], r[0], code_of.get(r[0]), r[3]) in index]
                    for p, v in parts.items()}
            if any(kept.values()):        # 하나도 없으면 원본을 남겨 재샤딩 후 쓴다
                parts = {p: v for p, v in kept.items() if v}
        yield from allocate(parts, caps[split])


def _image_path(dataset, species_ko, code, filename):
    """학습 dataloader가 NAS root와 결합할 논리 경로. (원천데이터 zip 해소는 학습측 책임)

    code 는 612 에서 없다. parquet 경유로 오면 None 이 아니라 NaN 인데 NaN 은 truthy 라
    「612/nan_가는장구채/…」 같은 경로가 만들어진다 → NaN 도 없는 것으로 본다.
    """
    has_code = code is not None and code == code and str(code) != ""
    sub = f"{code}_{species_ko}" if has_code else species_ko
    return f"{dataset}/{sub}/{filename}"


def build_sft(annotation_jsonl, shards_glob, out_dir, train_cap, val_cap, include_ambiguous,
              abstain_eff_train_cap=20, abstain_eff_val_cap=3, shard_index=None):
    ann_map = {}
    for line in open(annotation_jsonl, encoding="utf-8"):
        a = json.loads(line)
        ann_map[a["species_ko"]] = a

    writers = {}
    counts = collections.Counter()
    # abstain eff 는 종 단위 신호라 종당 소수면 충분. 상한을 둬 over-refusal·반복 환각 방지.
    abstain_caps = {"train": abstain_eff_train_cap, "val": abstain_eff_val_cap}
    abstain_seen = collections.Counter()   # (species_ko, split) → abstain eff 수
    os.makedirs(out_dir, exist_ok=True)

    def emit(split, rec):
        if split not in writers:
            writers[split] = open(os.path.join(out_dir, f"mm_{split}.jsonl"), "w", encoding="utf-8")
        writers[split].write(json.dumps(rec, ensure_ascii=False) + "\n")
        counts[(split, rec["task"])] += 1

    for sp, dataset, part, filename, label_zip, split in iter_image_rows(
            shards_glob, train_cap, val_cap, shard_index):
        ann = ann_map.get(sp)
        if not ann:
            continue
        img = _image_path(dataset, sp, ann.get("species_code"), filename)
        base = {"image": img, "dataset": dataset, "species_ko": sp, "part": part}
        key = filename

        # T1 약초ID (전체)
        gpt, q = render_T1(ann, _pick(Q_ID, key), key)
        emit(split, {**base, "task": "id",
                     "conversations": [{"from": "human", "value": f"{IMG_TOKEN}\n{q}"}, gpt]})
        # T2 독초판별 (전체)
        gpt, q = render_T2(ann, _pick(Q_TOX, key), key)
        emit(split, {**base, "task": "tox",
                     "conversations": [{"from": "human", "value": f"{IMG_TOKEN}\n{q}"}, gpt]})
        # T3 효능: linked(+옵션 ambiguous)는 고전 인용, 그 외(unlinked 등)는 abstain.
        # abstain 학습 신호는 종 단위라 종당 abstain_eff_cap 까지만 emit (over-refusal·반복 환각 방지).
        # real 효능(카드 보유 종)은 종당 이미지 상한(train_cap)까지 전부 emit.
        status = ann.get("knowledge_status")
        use_classical = status == "linked" or (status == "ambiguous" and include_ambiguous)
        ann_eff = ann if use_classical else {"species_ko": sp}
        c_eff = ann_eff.get("classical") or {}
        is_abstain_eff = not ((c_eff.get("효능") or []) or (c_eff.get("주치") or []))
        if not (is_abstain_eff and abstain_seen[(sp, split)] >= abstain_caps[split]):
            if is_abstain_eff:
                abstain_seen[(sp, split)] += 1
            gpt, q = render_T3(ann_eff, _pick(Q_EFF, key), key, part)
            emit(split, {**base, "task": "eff",
                         "conversations": [{"from": "human", "value": f"{IMG_TOKEN}\n{q}"}, gpt]})

    for w in writers.values():
        w.close()
    return counts


def write_resolved(out_dir: str, shard_index: str) -> dict:
    """이미지가 실제로 샤드에 있는 행만 추린 mm_*_resolved.jsonl 생성.

    학습 config 가 보는 파일이다. 없으면 트레이너가 회색 placeholder 를 먹는다.
    612 는 30/121종만 보유라 상당수가 여기서 빠진다.
    """
    with open(shard_index, encoding="utf-8") as fh:
        index = json.load(fh)
    stat = {}
    for split in ("train", "val"):
        src = os.path.join(out_dir, f"mm_{split}.jsonl")
        if not os.path.exists(src):
            continue
        kept = dropped = 0
        with open(src, encoding="utf-8") as fin, \
                open(os.path.join(out_dir, f"mm_{split}_resolved.jsonl"), "w",
                     encoding="utf-8") as fout:
            for line in fin:
                if json.loads(line)["image"] in index:
                    fout.write(line)
                    kept += 1
                else:
                    dropped += 1
        stat[split] = (kept, dropped)
    return stat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotation", default="data/annotations/species_annotation.jsonl")
    ap.add_argument("--shards", default="data/label_index/shards/*.parquet")
    ap.add_argument("--out", default="data/sft")
    ap.add_argument("--train_cap", type=int, default=100, help="종당 train 이미지 상한")
    ap.add_argument("--val_cap", type=int, default=15, help="종당 val 이미지 상한")
    ap.add_argument("--include_ambiguous", action="store_true",
                    help="ambiguous 종도 T3 효능 생성 (기본 제외)")
    ap.add_argument("--abstain_eff_train_cap", type=int, default=20,
                    help="종당 abstain 효능 train 상한 (over-refusal 방지)")
    ap.add_argument("--abstain_eff_val_cap", type=int, default=3,
                    help="종당 abstain 효능 val 상한")
    ap.add_argument("--shard_index", default="data/shards/herb_shard_index.json",
                    help="이미지 실재 행만 추린 mm_*_resolved.jsonl 도 생성")
    args = ap.parse_args()

    counts = build_sft(args.annotation, args.shards, args.out,
                       args.train_cap, args.val_cap, args.include_ambiguous,
                       args.abstain_eff_train_cap, args.abstain_eff_val_cap,
                       args.shard_index)
    print(f"완료 → {args.out}/mm_*.jsonl  (train≤{args.train_cap}, val≤{args.val_cap} 장/종)")
    total = sum(counts.values())
    by_split = collections.Counter()
    for (split, task), n in sorted(counts.items()):
        print(f"  {split:5s} {task:4s}: {n}")
        by_split[split] += n
    print(f"  split 합계: {dict(by_split)} / 총 {total} 대화쌍")
    if args.shard_index and os.path.exists(args.shard_index):
        for split, (kept, dropped) in write_resolved(args.out, args.shard_index).items():
            print(f"  resolved {split}: {kept}행 (이미지 없어 제외 {dropped})")


if __name__ == "__main__":
    main()
