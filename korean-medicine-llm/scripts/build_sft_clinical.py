"""ver6 · Clinical QA builder for HanMed-LLM (Gemma-3 LoRA SFT).

raw/mediclassics_unified/book_008 에서 clinical 편(부인·허로·자한·불면·식욕 등)
레코드를 추출해 symptom→변증→처방 QA 쌍을 생성.

ver6 4원칙 강제:
  (1) 답변은 실재 변증어/처방명 1+ 포함 (entity_whitelist 검증)
  (2) 모든 인용은 `up_path_nm` 실재 경로
  (3) 템플릿 pool 5종 rotate + 자유형 20%
  (4) 답변 종결은 내용 기반 (고정 closing 2종만 safety 에 허용)

Usage:
    .venv/bin/python scripts/build_sft_clinical.py --out experiments/dongui_bogam/data/sft/phaseB_qa_clinical_v1.jsonl
    .venv/bin/python scripts/build_sft_clinical.py --sample 50 --out /tmp/clinical_sample_50.jsonl  # 전문가 리뷰용
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw" / "mediclassics_unified" / "book_008"

SYSTEM_PROMPT = (
    "당신은 한의학 고전 문헌 연구 보조 AI 입니다. 동의보감(東醫寶鑑)의 증상 · 변증 · 처방 설명을 "
    "실제 본문에 근거해 정확히 답변합니다. 개인 진단이나 구체 용량 처방은 제공하지 않습니다."
)

# 임상 path 키워드 (up_path_nm 에 포함되면 clinical record 로 분류).
CLINICAL_KEYWORDS = [
    "婦人", "虛勞", "自汗", "盜汗", "不眠", "不食", "失眠",
    "食慾", "胎", "氣血", "血虛", "氣虛", "心脾",
    "風 ", "寒 ", "暑 ", "濕 ", "燥 ", "火 ",
]

# 변증명 whitelist (답변 검증용). 동의보감 본문에서 자주 쓰이는 변증어.
BIAN_WHITELIST = [
    "氣虛", "기허", "血虛", "혈허", "氣血兩虛", "기혈양허",
    "陽虛", "양허", "陰虛", "음허",
    "胎動不安", "태동불안", "心脾兩虛", "심비양허",
    "脾胃虛弱", "비위허약", "自汗", "자한",
    "盜汗", "도한", "不眠", "불면", "失眠", "실면",
    "濕熱", "습열", "痰飮", "담음",
]

# 처방명 whitelist (실재 동의보감 인용 처방).
FORMULA_WHITELIST = [
    "補中益氣湯", "보중익기탕", "四物湯", "사물탕",
    "八珍湯", "팔진탕", "十全大補湯", "십전대보탕",
    "歸脾湯", "귀비탕", "人蔘養榮湯", "인삼양영탕",
    "當歸芍藥散", "당귀작약산", "膠艾湯", "교애탕",
    "安胎飮", "안태음", "桂枝湯", "계지탕",
    "黃芪建中湯", "황기건중탕",
]

# 질문 템플릿 5종 (원칙 3 — 답변 템플릿은 별도 rotate).
QUESTION_TEMPLATES = [
    "동의보감에서 '{symptom}' 증상은 어떤 변증으로 해석하며 어떤 처방을 쓰는지 본문 근거로 설명해 주세요.",
    "'{symptom}' 이라는 증상이 동의보감에서 어떻게 서술되어 있는지, 원문 맥락과 함께 알려 주세요.",
    "한 환자가 '{symptom}' 을 호소합니다. 동의보감 기준 변증 범주와 해당 처방 계열을 간단히 정리해 주세요.",
    "{path_name} 에서 설명하는 '{symptom}' 의 병리와 대응 처방을 원문 인용을 포함해 답해 주세요.",
    "동의보감 {path_name} 편에 나오는 '{symptom}' 서술의 핵심 변증과 주 처방을 짚어 주세요.",
]

# 답변 템플릿 5종 + 1 자유형 flag.
ANSWER_TEMPLATES = [
    # T1 — 변증 / 설명 / 처방 / 인용 (구조형)
    "**변증**: {bian}\n\n**원문 설명**: {excerpt}\n\n**대응 처방**: {formulas}\n\n[출처: {cite}]",
    # T2 — 인용 우선형
    "동의보감 {cite} 에 따르면, {excerpt}\n\n이 서술은 {bian} 범주의 변증으로 정리할 수 있으며, 본문에서 언급되는 대표 처방으로는 {formulas} 등이 있습니다.",
    # T3 — 간략형
    "{excerpt}\n\n변증: {bian} / 처방 계열: {formulas}\n출처: {cite}",
    # T4 — 설명형
    "이 증상은 동의보감에서 {bian} 으로 해석됩니다. {excerpt} 관련 처방으로는 {formulas} 이 대표적입니다 ({cite}).",
    # T5 — 문답형
    "원문({cite})을 보면 {excerpt} 이에 해당하는 변증은 {bian} 이며, 활용 처방은 {formulas} 입니다.",
]


def load_records() -> list[dict]:
    recs = []
    for p in sorted(RAW_DIR.glob("vol_*.jsonl")):
        with p.open() as f:
            for ln in f:
                d = json.loads(ln)
                up = d.get("up_path_nm") or ""
                if any(k in up for k in CLINICAL_KEYWORDS):
                    recs.append(d)
    return recs


def extract_symptom(text: str) -> str | None:
    """본문에서 증상 표현 후보 추출 (간단 휴리스틱). 실제 정교화는 D3 리뷰 후."""
    if not text:
        return None
    # 괄호 안 한자 제거, 첫 문장만 사용
    t = re.sub(r"\([^\)]*\)", "", text)
    sentence = t.split(".")[0].strip() if "." in t else t[:60].strip()
    # 너무 길면 잘라내기
    return sentence[:80] if 10 <= len(sentence) <= 120 else None


def detect_bian(text: str) -> list[str]:
    return [b for b in BIAN_WHITELIST if b in text]


def detect_formulas(text: str) -> list[str]:
    return [f for f in FORMULA_WHITELIST if f in text]


def compose_qa(rec: dict, seed_idx: int, rng: random.Random) -> dict | None:
    trans = rec.get("trans_ko") or ""
    up = rec.get("up_path_nm") or ""
    if not trans or not up:
        return None
    symptom = extract_symptom(trans)
    if symptom is None:
        return None
    bian_hits = detect_bian(trans)
    formula_hits = detect_formulas(trans)
    # 원칙 1 — 실체 없으면 skip
    if not bian_hits and not formula_hits:
        return None

    # up_path_nm → 한국식 "雜病篇 卷之十 婦人" 형태로 정규화
    cite = up.strip()
    path_parts = [p.strip() for p in up.split(">")]
    path_name = path_parts[-1] if path_parts else up

    q_tpl = QUESTION_TEMPLATES[seed_idx % len(QUESTION_TEMPLATES)]
    question = q_tpl.format(symptom=symptom, path_name=path_name)

    # 답변 템플릿 rotate + 20% 자유형
    free_form = rng.random() < 0.2
    bian_str = " · ".join(bian_hits[:3]) if bian_hits else "해당 없음"
    formula_str = " · ".join(formula_hits[:3]) if formula_hits else "본문 참고"
    excerpt = trans.strip().replace("\r", " ").replace("\n", " ")[:280]

    if free_form:
        answer = (
            f"{excerpt} 이 대목은 {bian_str} 계열 증후로 정리할 수 있고, "
            f"관련 처방으로 {formula_str} 을(를) 떠올릴 수 있습니다. "
            f"세부는 전문 한의사와 상의하세요. (출처: {cite})"
        )
    else:
        a_tpl = ANSWER_TEMPLATES[seed_idx % len(ANSWER_TEMPLATES)]
        answer = a_tpl.format(
            bian=bian_str,
            formulas=formula_str,
            excerpt=excerpt,
            cite=cite,
        )

    return {
        "id": f"book_008_{rec.get('volume_id','?')}_{rec.get('content_seq','?')}",
        "category": "clinical",
        "subcat": path_name,
        "record_ref": f"book_008/vol_{rec.get('volume_id','?')}/seq_{rec.get('content_seq','?')}",
        "up_path_nm": cite,
        "question": question,
        "assistant": answer,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
        "_meta": {
            "bian_hits": bian_hits,
            "formula_hits": formula_hits,
            "free_form": free_form,
            "template_idx": seed_idx % len(ANSWER_TEMPLATES),
        },
    }


def validate(pair: dict) -> list[str]:
    """원칙 1·2 기계 검증. 위반 사유 목록 반환, 빈 list 면 통과."""
    errs = []
    ans = pair["assistant"]
    # 원칙 1: bian 또는 formula 1+
    if not (pair["_meta"]["bian_hits"] or pair["_meta"]["formula_hits"]):
        errs.append("no_entity")
    # 원칙 2: 출처가 up_path_nm 과 일치
    m = re.search(r"\[?출처:\s*([^\]\)]+)", ans)
    if m:
        cite_in_ans = m.group(1).strip()
        if cite_in_ans != pair["up_path_nm"]:
            errs.append(f"cite_mismatch:{cite_in_ans!r}")
    return errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sample", type=int, default=0, help="N 쌍만 생성 (0 = 전체)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-per-subcat", type=int, default=120, help="subcat 당 최대 쌍 수 (편향 방지)")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    recs = load_records()
    print(f"[load] clinical records: {len(recs)}")

    subcat_counts = collections.Counter()
    pairs = []
    rejections = []
    for i, rec in enumerate(rng.sample(recs, len(recs))):
        pair = compose_qa(rec, seed_idx=i, rng=rng)
        if pair is None:
            continue
        if subcat_counts[pair["subcat"]] >= args.max_per_subcat:
            continue
        errs = validate(pair)
        if errs:
            rejections.append({"id": pair["id"], "errs": errs})
            continue
        pairs.append(pair)
        subcat_counts[pair["subcat"]] += 1
        if args.sample and len(pairs) >= args.sample:
            break

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    rej_path = args.out.with_suffix(".rejections.jsonl")
    with rej_path.open("w") as f:
        for r in rejections:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # stats
    stats = {
        "total_pairs": len(pairs),
        "rejections": len(rejections),
        "subcat_dist": dict(subcat_counts.most_common(20)),
        "free_form_ratio": round(
            sum(1 for p in pairs if p["_meta"]["free_form"]) / max(1, len(pairs)), 3
        ),
    }
    stats_path = args.out.with_suffix(".stats.json")
    with stats_path.open("w") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"[write] pairs={len(pairs)} rejections={len(rejections)} → {args.out}")
    print(f"[stats] {json.dumps(stats, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
