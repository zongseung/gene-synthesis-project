"""설진(혀) SFT 생성기 — 동의보감 한글 grounding + abstain 게이트.

소스: TMC-Tongue COCO 정답 라벨(인간 어노테이션). VARCO 추론이 아니라 GT 라벨을
변증 학습타깃으로 변환한다(생성단계 멀티모달 환각 회피).

규율(품질평가 9건 반영):
- 변증 인용은 rule KB의 quote_ko(동의보감 trans_ko verbatim)만. 한자 인용 금지.
- abstain 카테고리(자설·치흔·수설)는 피처 서술만, 변증 결론 금지.
- build manifest의 drop/exclude/license 자동 적용.
- 출력 후 glyph-literal 게이트로 quote_ko 재검증.

3분할(결함 A 수정): 소스 COCO train 을 다시 train/val 로 쪼개 학습 검증셋을 벤치와
분리한다. 소스 COCO val 은 전량 test 로만 내보내고 학습에는 절대 쓰지 않는다.
train→{train,val} 분할은 파일명의 md5 정렬(결정론, PYTHONHASHSEED 비의존)로 고른다.

사용: PYTHONPATH=src python -m hanmed_mm.data.gen_tongue_sft \
        --coco_root <shezhenv3-coco> --rule_kb data/safety_kb/tongue_rule_kb.json \
        --manifest data/sft/tongue_sft_build_manifest.json --out data/sft/tongue_sft \
        --holdout_val 556
"""
from __future__ import annotations
import argparse, collections, hashlib, json, os


def load_json(p):
    with open(p) as f:
        return json.load(f)


# 문헌 grounding 프레이밍 — 진단을 단정하지 않고 "문헌이 이렇게 서술한다"로만.
# 동의보감 변증은 상한 경과·맥·증상을 종합한 맥락 기술이므로, 사진 단독 진단으로 환원하지 않는다.
_OBS = ["{h}가 관찰됩니다", "혀에서 {h}가 보입니다", "{h} 소견이 확인됩니다", "설진상 {h}가 나타납니다", "{h}가 비교적 뚜렷합니다"]
_CITE = [
    "동의보감 〈{sec}〉은 이를 「{q}」라고 기술하며, 문헌상 {b} 맥락에서 다룹니다",
    "동의보감 〈{sec}〉에서는 「{q}」라 하여 {b}와 관련해 서술합니다",
    "이에 대해 동의보감 〈{sec}〉은 「{q}」라고 기록합니다(문헌상 {b} 맥락)",
    "동의보감 〈{sec}〉의 「{q}」라는 구절이 이를 {b} 맥락에서 설명합니다",
    "문헌상으로는 동의보감 〈{sec}〉이 「{q}」라 적어 {b}와 연결합니다",
]
_CAVEATS = [
    " (위 내용은 동의보감 문헌의 서술을 인용·해설한 것입니다. 원문은 상한 경과·맥·증상을 종합한 맥락의 기술이므로, 사진 한 장만으로 변증을 단정하지 않으며 진료를 대신하지 않습니다.)",
    " (이 설명은 동의보감 원문 서술을 옮긴 문헌 해설입니다. 고전 변증은 맥·증상을 종합한 맥락 기술이라 사진 한 장으로 단정할 수 없으며, 진료를 대신하지 않습니다.)",
    " (위는 동의보감 기록을 인용한 문헌 정리입니다. 원문 변증은 경과·맥·증을 함께 살핀 것이므로 이미지 단독 판단은 삼가야 하며, 진료를 대신하지 않습니다.)",
    " (본 해설은 17세기 동의보감 문헌의 서술 인용입니다. 변증은 맥과 증상을 종합한 맥락에서 성립하므로 사진만으로 확정하지 않으며, 진료를 대신하지 않습니다.)",
]


def _leaf(up_path_nm):
    return (up_path_nm or "").split(">")[-1].strip() or "口舌"


def build_answer(signs_present, kb, idx=0):
    """signs_present: [category_code]. kb: rule KB. → (answer_text, provenance_list).
    idx: 결정론적 템플릿 로테이션(다양성). 인용문 quote_ko 는 글자단위 보존."""
    rules = kb["rules"]
    grounded, abstained = [], []
    for code in signs_present:
        r = rules.get(code)
        if r is None:
            continue
        (abstained if r.get("abstain") else grounded).append((code, r)) if (r.get("abstain") or r.get("quote_ko")) else None

    parts, prov = [], []
    for j, (code, r) in enumerate(grounded):
        h = r["hanja_label"]; sec = _leaf(r["source_ref"].get("up_path_nm"))
        obs = _OBS[(idx + j) % len(_OBS)].format(h=h)
        cite = _CITE[(idx // len(_OBS) + j) % len(_CITE)].format(sec=sec, q=r["quote_ko"], b=r["byeonjeung"])
        parts.append(f"{obs}. {cite}.")
        prov.append({
            "category": code, "byeonjeung_literature": r["byeonjeung"],
            "source_ref": r["source_ref"], "quote_ko": r["quote_ko"],
            "verify": "glyph_literal_trans_ko", "framing": "literature_grounding",
        })
    for code, r in abstained:
        parts.append(
            f"{r['hanja_label']}도 함께 보입니다. 다만 동의보감에는 이 소견에 대한 설진 서술이 없어 문헌 근거를 제시하지 않습니다."
        )
    if not parts:
        return None, None  # 논의할 소견 없음 → 레코드 생략
    return " ".join(parts) + _CAVEATS[idx % len(_CAVEATS)], prov


def _write_jsonl(outdir, fname, recs):
    with open(os.path.join(outdir, fname), "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco_root", required=True)
    ap.add_argument("--rule_kb", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", default="train,val")
    ap.add_argument("--holdout_val", type=int, default=556,
                     help="소스 train 에서 학습검증용으로 떼어낼 이미지 수(md5(file_name) 오름차순 상위 N)")
    args = ap.parse_args()

    kb = load_json(args.rule_kb)
    man = load_json(args.manifest)
    os.makedirs(args.out, exist_ok=True)

    drop = set(man["fix3_drop_classes"]["organ_zone_codes"])
    # 극소 클래스 코드(이름만; 라벨명 기준)
    tiny = {"jiankangshe"}  # 정상 9건 — 변증 소스로 부적합(피처 서술만 가능하나 표본 극소 → drop)
    exclude_files = {(m["split"], m["file"]) for m in man["fix4_exclude_dim_mismatch"]["files"]}

    stats = collections.Counter()
    for split in args.splits.split(","):
        jp = os.path.join(args.coco_root, split, "annotations", f"{split}.json")
        if not os.path.exists(jp):
            print(f"[skip] {jp} 없음"); continue
        d = load_json(jp)
        catname = {c["id"]: c["name"] for c in d["categories"]}
        by_img = collections.defaultdict(set)
        for a in d["annotations"]:
            by_img[a["image_id"]].add(catname.get(a["category_id"]))
        imgs = {im["id"]: im for im in d["images"]}

        # 질문 다양화(문헌 grounding 프레이밍) — 진단 요청이 아니라 문헌 서술 요청
        QUESTIONS = [
            "<image>\n이 혀의 설진 소견을 짚고, 동의보감 문헌이 그 소견을 어떻게 서술하는지 인용해 주세요.",
            "<image>\n사진 속 혀에서 보이는 설진 소견과, 그에 대한 동의보감의 기술을 알려주세요.",
            "<image>\n이 혀 사진의 설질·설태 소견을 관찰하고, 동의보감 원문이 해당 소견을 다루는 대목을 인용해 설명해 주세요.",
            "<image>\n이 설진 이미지의 소견을 살핀 뒤, 동의보감이 그 소견을 서술한 대목을 찾아 인용해 주세요.",
            "<image>\n혀 사진에서 관찰되는 설진 소견과, 동의보감 문헌이 이를 어떻게 기록하는지 함께 설명해 주세요.",
        ]
        recs = []
        for idx, (iid, signs) in enumerate(by_img.items()):
            im = imgs.get(iid)
            if im is None:
                continue
            if (split, im["file_name"]) in exclude_files:
                stats["excl_dim"] += 1; continue
            # drop 장부분구 + 극소 + None
            signs = {s for s in signs if s and s not in drop and s not in tiny}
            if not signs:
                stats["empty_after_filter"] += 1; continue
            answer, prov = build_answer(sorted(signs), kb, idx=idx)
            if answer is None:
                stats["no_discussable"] += 1; continue
            # 정렬·홀드아웃 선택에 쓸 원본 파일명을 레코드와 함께 보관(출력에는 포함 안 함).
            recs.append((im["file_name"], {
                "image": f"shezhenv3/{split}/{im['file_name']}",
                "dataset": "shezhenv3-tongue",
                "license": "CC0",  # fix5: 본셋은 CC0. KISTI 정상비교는 별도 NC 셋(미포함)
                "task": "설진_문헌인용",
                "signs": sorted(signs),
                "conversations": [
                    {"from": "human", "value": QUESTIONS[idx % len(QUESTIONS)]},
                    {"from": "gpt", "value": answer},
                ],
                "provenance": prov,
            }))
            stats[f"{split}_records"] += 1
            stats[f"{split}_prov_grounded"] += len(prov)

        if split == "train":
            # 결함 A 수정: 벤치(track1~3)는 소스 val 을 그대로 쓰므로, 학습 검증셋은
            # 반드시 train 에서만 떼어야 벤치와 이미지가 겹치지 않는다.
            # 내장 hash() 는 PYTHONHASHSEED 마다 값이 달라 재현 불가 → md5 hexdigest 정렬 사용.
            ordered = sorted(recs, key=lambda kv: hashlib.md5(kv[0].encode()).hexdigest())
            n = args.holdout_val
            if n >= len(recs):
                # 조용히 빈 train 파일을 쓰면 학습이 0 스텝으로 "성공"해 버린다.
                raise SystemExit(
                    f"--holdout_val {n} >= train 레코드 {len(recs)} — 학습셋이 비어버린다. "
                    "값을 줄이거나 원천 데이터를 확인할 것.")
            holdout, remainder = ordered[:n], ordered[n:]
            _write_jsonl(args.out, "tongue_sft_train.jsonl", [r for _, r in remainder])
            _write_jsonl(args.out, "tongue_sft_val.jsonl", [r for _, r in holdout])
            print(f"[train] {len(remainder)} 레코드 → tongue_sft_train.jsonl")
            print(f"[train→val holdout] {len(holdout)} 레코드 → tongue_sft_val.jsonl")
        elif split == "val":
            # 소스 val 은 벤치 전용(track1~3 원천) — 학습에 절대 섞지 않고 전량 test 로만 낸다.
            _write_jsonl(args.out, "tongue_sft_test.jsonl", [r for _, r in recs])
            print(f"[val→test] {len(recs)} 레코드 → tongue_sft_test.jsonl")
        else:
            _write_jsonl(args.out, f"tongue_sft_{split}.jsonl", [r for _, r in recs])
            print(f"[{split}] {len(recs)} 레코드 → tongue_sft_{split}.jsonl")

    print("통계:", dict(stats))


if __name__ == "__main__":
    main()
