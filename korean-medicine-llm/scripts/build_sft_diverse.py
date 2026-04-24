"""build_sft_diverse.py — ver5 Phase B SFT 데이터 다양화 빌더.

옵션 2 구현 (docs/ver5/08_sft_build_plan.md 연장):
- 기존 build_sft_qa.py --mode full_corpus 는 단일 고정 포맷 ("원문 발췌: X → 번역")
  으로 34,039 쌍을 만들었지만, 이는 실제 사용자 자연어 질문과 분포 괴리가 큼.
- 본 스크립트는 같은 record 에서 **여러 질문/답변 포맷** 을 생성해 입력 다양성을
  확보하고, 편명 (path 라벨 복붙) 제거 + 용량 표기 마스킹 을 적용한다.

질문 포맷:
  F1 structured   : 기존 build_sft_qa.py 포맷 (유지)
  F2 direct       : "동의보감에서 {topic}은 어떻게 설명되나요?"
  F3 path-based   : "동의보감 {short_path}에서 {topic}을 알려주세요"
  F4 open-ended   : "동의보감에 기록된 {leaf_category}의 하위 항목은?" (총목 전용)

답변 포맷:
  A1 quote-based  : 기존 ("현대 한국어: X / 해설: Y")
  A2 natural      : "{topic}은 동의보감 {path}에 기록된 … 입니다. [출처: …]"

용량 마스킹: "2돈", "1냥", "3푼", "5알" 같은 수치+단위 를 "[용량]" 으로 치환
  (literal quote 에도 적용 — 모델이 용량 패턴 자체를 학습하지 않도록)
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# build_sft_qa.py 에서 분류/경로/컨텍스트 헬퍼 재사용
_SCRIPT = Path(__file__).resolve()
sys.path.insert(0, str(_SCRIPT.parent))
import build_sft_qa as bq  # type: ignore


# ────────────────────────────── 용량 마스킹 ──────────────────────────────

DOSAGE_PATTERNS = [
    re.compile(r"\d+(?:\.\d+)?\s*(?:돈|푼|냥|전|알|첩|근|홉|되|말|구|편)"),
    re.compile(r"\d+(?:\.\d+)?\s*(?:g|mg|kg|ml|L)\b"),
    re.compile(r"각\s*\d+(?:\.\d+)?(?:돈|푼|냥|전|알|g|mg)"),
]

PRESCRIPTION_NAME_PATTERN = re.compile(
    r"[一-龥]{1,4}(?:탕|산|환|고|음|煎|丸|散|湯|膏|飮|丹|단)(?!요|에|과|이|을|의|,)"
)

def has_dosage_or_prescription(text: str) -> bool:
    """원문에 용량·처방 수치/처방명이 있으면 True — seed 에서 제외."""
    if not text:
        return False
    for pat in DOSAGE_PATTERNS:
        if pat.search(text):
            return True
    if PRESCRIPTION_NAME_PATTERN.search(text):
        return True
    return False


def mask_dosage(text: str) -> str:
    """v3.1: no-op. 처방 포함 record 는 is_safe_record() 로 사전 제외하므로
    마스킹은 하지 않는다 ([용량] 토큰 학습 문제 차단)."""
    return text or ""


# ───────────────────────── record 당 변형 generator ─────────────────────────

def leaf_of_path(parts: list[str]) -> str:
    return parts[-1] if parts else ""


def short_path_ko(parts: list[str]) -> str:
    """路径 한자 그대로 3-leaf 요약."""
    if not parts:
        return "동의보감"
    if len(parts) <= 2:
        return " > ".join(parts)
    return " > ".join([parts[0], parts[-1]])


def natural_question_candidates(record_type: str, row: dict, topic: str, parts: list[str]) -> list[str]:
    """record 하나에서 자연어 질문 후보들을 생성 (2~3개)."""
    cand: list[str] = []
    path_str = short_path_ko(parts)

    if record_type == "병증 설명":
        cand.append(f"동의보감에서 {topic}은(는) 어떤 증상이나 병으로 설명되나요?")
        cand.append(f"동의보감 {path_str} 대목의 {topic}에 대한 설명을 알려주세요.")
        cand.append(f"{topic}에 관해 동의보감 원문이 어떻게 기술하는지 정리해 주세요.")
    elif record_type == "본문 설명":
        cand.append(f"동의보감에서 {topic}은(는) 어떻게 기록되어 있나요?")
        cand.append(f"동의보감 {path_str} 부분의 {topic} 내용을 정리해 주세요.")
        cand.append(f"{topic}에 대한 동의보감의 문헌적 기술을 알려주세요.")
    elif record_type == "서문":
        cand.append(f"동의보감 서문 중 {topic} 대목은 어떤 내용인가요?")
        cand.append(f"동의보감의 {path_str} 서문 내용을 정리해 주세요.")
        cand.append(f"서문에서 {topic}은(는) 어떻게 언급되나요?")
    elif record_type == "총목":
        cand.append(f"동의보감 {path_str} 에는 어떤 하위 항목이 있나요?")
        cand.append(f"동의보감 총목에서 {topic}은(는) 어떻게 분류되나요?")
    else:
        cand.append(f"동의보감에서 {topic}은(는) 어떻게 설명되나요?")
        cand.append(f"동의보감 {path_str} 대목의 내용을 정리해 주세요.")

    return cand


def structured_question(record_type: str, row: dict, max_chars: int) -> str:
    return bq.compose_full_corpus_question(record_type, row, max_chars)


# ───────────────────────────── 답변 generator ─────────────────────────────

def quote_answer(record_type: str, row: dict, max_chars: int) -> str:
    """A1: 기존 포맷 재사용."""
    base = bq.compose_full_corpus_answer(record_type, row, max_chars)
    # 용량 마스킹은 quote 블록에만 적용 (해설은 고정 문구이므로 영향 없음)
    return mask_dosage(base)


def natural_answer(record_type: str, row: dict, topic: str, parts: list[str], max_chars: int) -> str:
    """A2: 자연어 형태 + literal quote (용량 마스킹) + 짧은 맥락 + 출처 태그."""
    translated = bq.clip_text(row.get("trans_ko") or row.get("original"), max_chars)
    translated = mask_dosage(translated)
    path_str = short_path_ko(parts) or "동의보감"

    if record_type == "병증 설명":
        tail = (
            "이는 17세기 동의보감의 문헌적 설명이며, 현대 의학의 진단이나 처방을 "
            "대체하지 않습니다."
        )
        return (
            f"{topic}에 대해 동의보감은 다음과 같이 기록하고 있습니다. "
            f"\"{translated}\" {tail} [출처: 동의보감 {path_str}]"
        )
    if record_type == "본문 설명":
        tail = (
            "이 대목은 동의보감이 해당 항목을 문헌적으로 기록한 것으로, "
            "개인 처방 지침이 아닙니다."
        )
        return (
            f"동의보감 {path_str} 에서 {topic} 에 관해 다음과 같이 기록합니다. "
            f"\"{translated}\" {tail} [출처: 동의보감 {path_str}]"
        )
    if record_type == "서문":
        tail = "이 대목은 동의보감 서문 계열 기록으로 편찬 맥락을 보여 줍니다."
        return (
            f"동의보감 {path_str} 서문은 다음과 같이 기술합니다. "
            f"\"{translated}\" {tail} [출처: 동의보감 {path_str}]"
        )
    if record_type == "총목":
        tail = "이는 동의보감 총목이 해당 편의 하위 분류를 보여 주는 대목입니다."
        return (
            f"동의보감 {path_str} 의 총목에는 {topic} 항목이 다음과 같이 기록되어 있습니다. "
            f"\"{translated}\" {tail} [출처: 동의보감 {path_str}]"
        )
    # fallback
    return (
        f"동의보감 {path_str} 에 관한 기록은 다음과 같습니다. "
        f"\"{translated}\" [출처: 동의보감 {path_str}]"
    )


# ─────────────────────────────── main flow ───────────────────────────────

def build_messages(system_prompt: str, question: str, answer: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]


SYSTEM_PROMPT = (
    "당신은 한의학 고전 문헌 연구 보조 AI 입니다. 동의보감(東醫寶鑑) 원문을 "
    "인용해 답하되, 개인 증상 진단·처방·용량 지시는 제공하지 않습니다. "
    "학습 범위를 벗어나는 질문에는 범위 외임을 분명히 알립니다."
)


def expand_record(
    ref: str,
    row: dict,
    record_type: str,
    max_q_chars: int,
    max_a_chars: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    parts = bq.split_path_parts(row.get("up_path_nm"))
    topic = bq.record_topic(row)
    rows_out: list[dict[str, Any]] = []

    # F1 structured × A1 (원본 대응, 포맷 유지용 — 전체의 1/3 정도만)
    if rng.random() < 0.33:
        q = structured_question(record_type, row, max_q_chars)
        a = quote_answer(record_type, row, max_a_chars)
        rows_out.append({
            "q_format": "F1_structured", "a_format": "A1_quote",
            "question": q, "assistant": a,
        })

    # F2/F3/F4 natural × A2 natural (핵심 — 전체의 2/3)
    naturals = natural_question_candidates(record_type, row, topic, parts)
    chosen_q = rng.sample(naturals, k=min(2, len(naturals)))
    for q_natural in chosen_q:
        a_natural = natural_answer(record_type, row, topic, parts, max_a_chars)
        rows_out.append({
            "q_format": "F2_natural", "a_format": "A2_natural",
            "question": q_natural, "assistant": a_natural,
        })

    # 또한 natural Q × A1 quote 조합 (정답은 기존 맥락 스타일)
    if rng.random() < 0.4 and naturals:
        q_natural = rng.choice(naturals)
        a_quote = quote_answer(record_type, row, max_a_chars)
        rows_out.append({
            "q_format": "F2_natural", "a_format": "A1_quote",
            "question": q_natural, "assistant": a_quote,
        })

    # 메타 붙이기
    out: list[dict[str, Any]] = []
    for i, it in enumerate(rows_out):
        out.append({
            "id": f"{ref.replace('/', '_')}__v{i}",
            "category": "medical_literature",
            "record_type": record_type,
            "record_ref": ref,
            "content_level": row.get("content_level"),
            "q_format": it["q_format"],
            "a_format": it["a_format"],
            "question": it["question"],
            "assistant": it["assistant"],
            "messages": build_messages(SYSTEM_PROMPT, it["question"], it["assistant"]),
            "source_records": [ref],
            "path": parts,
        })
    return out


def upsample_template_seeds(path: Path, multiplier: int) -> list[dict[str, Any]]:
    """phaseB_qa_template_v1.jsonl 의 서지·safety·oos 16쌍을 ×multiplier 로 복제."""
    if not path.exists():
        return []
    rows = [json.loads(l) for l in path.open(encoding="utf-8")]
    out: list[dict[str, Any]] = []
    for i in range(multiplier):
        for r in rows:
            copy = dict(r)
            copy["id"] = f"{r['id']}__upx{i}"
            copy.setdefault("q_format", "template_seed")
            copy.setdefault("a_format", "template_seed")
            out.append(copy)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-dir", type=Path,
                   default=Path("data/raw/mediclassics_unified/book_008"))
    p.add_argument("--template-seeds", type=Path,
                   default=Path("data/sft/phaseB_qa_template_v1.jsonl"))
    p.add_argument("--template-upsample", type=int, default=50,
                   help="template seeds 를 ×N 복제해 mix. 16×50=800 권장")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--stats-out", type=Path, default=None)
    p.add_argument("--max-question-chars", type=int, default=220)
    p.add_argument("--max-answer-chars", type=int, default=600)
    p.add_argument("--exclude-types", nargs="*", default=["편명"],
                   help="제외할 record_type (기본 편명)")
    p.add_argument("--per-type-cap", type=int, default=None,
                   help="record_type 당 최대 수 제한 (subsample). None 이면 무제한")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = random.Random(args.seed)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    print(f"[diverse] raw_dir={args.raw_dir}")
    record_index = bq.load_record_index(args.raw_dir)
    print(f"[diverse] records loaded: {len(record_index)}")

    exclude = set(args.exclude_types or [])
    per_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    stats_filter = {"skipped_prescription": 0, "skipped_empty": 0, "skipped_excluded_type": 0}

    # 1. expand each record
    for ref, row in record_index.items():
        rtype = bq.classify_record_type(row)
        if rtype in exclude:
            stats_filter["skipped_excluded_type"] += 1
            continue
        text = bq.clean_text(row.get("trans_ko") or row.get("original"))
        if not text:
            stats_filter["skipped_empty"] += 1
            continue
        # v3.1: 처방·용량 포함 record 는 skip ([용량] 마스킹 대신 원천 제외)
        if has_dosage_or_prescription(text):
            stats_filter["skipped_prescription"] += 1
            continue
        expansions = expand_record(
            ref, row, rtype,
            args.max_question_chars, args.max_answer_chars, rng,
        )
        per_type[rtype].extend(expansions)

    total = sum(len(v) for v in per_type.values())
    print(f"[diverse] before cap: {total} from {len(per_type)} types")
    print(f"[diverse] filter stats: {stats_filter}")

    # 2. per-type cap
    if args.per_type_cap:
        for t, lst in list(per_type.items()):
            if len(lst) > args.per_type_cap:
                rng.shuffle(lst)
                per_type[t] = lst[:args.per_type_cap]

    all_rows: list[dict[str, Any]] = []
    for lst in per_type.values():
        all_rows.extend(lst)

    # 3. template seeds upsample
    seeds_upsampled = upsample_template_seeds(args.template_seeds, args.template_upsample)
    print(f"[diverse] template seeds (upsampled ×{args.template_upsample}): {len(seeds_upsampled)}")
    all_rows.extend(seeds_upsampled)

    # 4. shuffle
    rng.shuffle(all_rows)
    print(f"[diverse] total: {len(all_rows)}")

    # 5. write
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[diverse] wrote {args.out}")

    stats = {
        "total": len(all_rows),
        "by_record_type": dict(Counter(r.get("record_type", "template_seed") for r in all_rows)),
        "by_q_format": dict(Counter(r.get("q_format", "?") for r in all_rows)),
        "by_a_format": dict(Counter(r.get("a_format", "?") for r in all_rows)),
        "excluded_types": list(exclude),
        "template_upsample": args.template_upsample,
    }
    print(f"[diverse] stats: {json.dumps(stats, ensure_ascii=False)}")
    if args.stats_out:
        args.stats_out.parent.mkdir(parents=True, exist_ok=True)
        args.stats_out.write_text(json.dumps(stats, ensure_ascii=False, indent=2),
                                  encoding="utf-8")


if __name__ == "__main__":
    main()
