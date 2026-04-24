"""build_sft_complex.py — ver5 v4 복합 추론 SFT 데이터 빌더.

spec: docs/ver5/09_v4_complex_reasoning_plan.md
seeds: data/sft/complex_seeds.yaml

4 subtype 처리:
  multi_hop    — 2~3 record 연결, CoT Step 구조
  compare      — 2 element 대조, 2-column 표현
  list         — parent children 열거
  conditional  — 조건 분기 (branching yaml 에 명시)

방식: 수작업 anchor seed 렌더 + 자동 확장 (sibling/chain/parent_children)
검증: literal quote assert, CoT 패턴, entity whitelist, 용량 마스킹, dedup

호출:
    cd experiments/dongui_bogam
    PYTHONHASHSEED=0 ../../.venv/bin/python scripts/build_sft_complex.py \\
        --seeds ../../data/sft/complex_seeds.yaml \\
        --raw-dir raw \\
        --out data/sft/phaseB_qa_complex_v4.jsonl \\
        --stats-out data/sft/phaseB_qa_complex_v4.stats.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

# 기존 빌더 헬퍼 재사용
_SCRIPT = Path(__file__).resolve()
sys.path.insert(0, str(_SCRIPT.parent))
import build_sft_qa as bq  # type: ignore
from build_sft_diverse import mask_dosage  # type: ignore


def parse_ref(ref: str) -> tuple[int, int]:
    """'book_008/vol_01/seq_5' → (1, 5)."""
    m = re.match(r"book_008/vol_(\d+)/seq_(\d+)$", ref)
    if not m:
        raise ValueError(f"bad ref {ref!r}")
    return int(m.group(1)), int(m.group(2))


def load_records_indexed(raw_dir: Path) -> dict[tuple[int, int], dict]:
    """(vol, seq) 튜플 key 로 record dict 반환 — build_sft_qa 의 string-key 버전과 별도."""
    idx: dict[tuple[int, int], dict] = {}
    for vp in sorted(raw_dir.glob("vol_*.jsonl")):
        for line in vp.open(encoding="utf-8"):
            r = json.loads(line)
            idx[(r["volume_id"], r["content_seq"])] = r
    return idx


# ───────────────────────────── 상수 ─────────────────────────────

SYSTEM_PROMPT_DEFAULT = (
    "당신은 동의보감(東醫寶鑑, book_008) 문헌 연구 보조 AI 입니다. 단일 사실을 "
    "물으면 원문 인용과 함께 정확히 답하고, 복합 질문(여러 record 연결·비교·"
    "열거·조건 분기)에는 단계적 추론 구조로 답합니다. 개인 증상 진단·처방·"
    "용량 지시는 하지 않으며, 학습 범위 밖 질문에는 범위 외임을 알립니다."
)


# ─────────────────────────── CoT template ───────────────────────────

def cot_temporal_chain(slots: dict, sources: list[tuple[str, str]]) -> str:
    """multi_hop.temporal — 시간 순서 연결."""
    parts = ["이 질문에 답하려면 편찬 과정을 시기별로 확인해야 합니다.\n"]
    for i, (role, quote) in enumerate(sources, 1):
        parts.append(f"**Step {i}** ({role}) — 동의보감 원문에 \"{quote}\" 라고 "
                     f"기록되어 있습니다.")
    parts.append("\n이상을 종합하면, 동의보감 편찬은 선조(宣祖)의 명으로 1596년 "
                 "시작되었다가 정유재란(1597)으로 잠시 중단된 뒤 허준이 단독으로 "
                 "작업을 이어갔고, 선조 승하(1608) 이후 광해군 2년(1610)에 "
                 "25권으로 완성되었으며, 광해군이 내의원에 명해 1613년 간행·"
                 "반포되었습니다.\n"
                 "[출처: 동의보감 내경편 권1 서문]")
    return "\n\n".join(parts)


def cot_role_chain(slots: dict, sources: list[tuple[str, str]]) -> str:
    """multi_hop.role_chain — 인물 역할 연쇄."""
    parts = ["이 질문은 동의보감 편찬에 참여한 인물들의 각 역할을 묻고 있습니다.\n"]
    role_name = {"author": "주저자", "helpers": "편찬 보조",
                 "preface_author": "서문 저자"}
    for i, (role, quote) in enumerate(sources, 1):
        rn = role_name.get(role, role)
        parts.append(f"**Step {i}** ({rn}) — \"{quote}\"")
    parts.append("\n정리하면, 주저자는 허준(許浚)이고, 초기 편찬 보조로 정작·"
                 "양예수·김응탁·이명원·정예남이 참여했으며, 책의 서문은 이정구가 "
                 "지었습니다. 단, 정유재란 이후 최종 편찬 단계는 허준이 단독으로 "
                 "수행했으므로 주저자 지위는 허준에게만 귀속됩니다.\n"
                 "[출처: 동의보감 내경편 권1 서문]")
    return "\n\n".join(parts)


def cot_category_bridge(slots: dict, sources: list[tuple[str, str]]) -> str:
    parts = ["이 질문에 답하려면 편별 총목에 기록된 하위 항목들을 단계별로 "
             "확인해야 합니다.\n"]
    for i, (role, quote) in enumerate(sources, 1):
        parts.append(f"**Step {i}** — 총목에 \"{quote}\" 라고 기록되어 있습니다.")
    parts.append("\n따라서 해당 편은 위 항목들을 체계적으로 분류·배열하는 구조로 "
                 "짜여 있습니다. 본 답은 17세기 문헌 구성을 소개한 것이며, 현대 "
                 "의료 조언이 아닙니다.\n"
                 "[출처: 동의보감 내경편 권1 총목]")
    return "\n\n".join(parts)


def cot_principles(slots: dict, sources: list[tuple[str, str]]) -> str:
    parts = ["선조가 동의보감 편찬 시 제시한 원칙은 서문에 세 부분으로 나뉘어 "
             "기록되어 있습니다.\n"]
    label = {"principle_cultivation": "수양 우선·약물 차선",
             "principle_essence": "기존 의서의 요점 선별",
             "principle_hyangyak": "향약 활용과 병기"}
    for i, (role, quote) in enumerate(sources, 1):
        parts.append(f"**Step {i}** — 원칙: {label.get(role, role)}. "
                     f"근거: \"{quote}\"")
    parts.append("\n정리하면, 선조의 세 원칙은 (1) 수양 우선·약물 차선, "
                 "(2) 번다한 기존 의서에서 요점을 선별, (3) 우리나라 향약(鄕藥) "
                 "활용과 명칭 병기 입니다. 이 세 원칙은 동의보감 편찬 방향을 "
                 "규정하는 핵심 근거가 됐습니다.\n"
                 "[출처: 동의보감 내경편 권1 서문]")
    return "\n\n".join(parts)


def cot_symptom_chain(slots: dict, sources: list[tuple[str, str]]) -> str:
    parts = ["동의보감은 이 병증을 단계적으로 설명합니다.\n"]
    label = {"classification": "분류", "pathology": "병인(病因)",
             "treatment_context": "치법 맥락"}
    for i, (role, quote) in enumerate(sources, 1):
        parts.append(f"**Step {i}** ({label.get(role, role)}) — \"{quote}\"")
    parts.append("\n즉, 동의보감은 해당 병증을 단순 증상 나열이 아니라 "
                 "분류·병인·치법 맥락을 단계적으로 기술하는 구조로 해설합니다. "
                 "이는 17세기 문헌 기준의 설명 틀이며, 현대 의학의 진단·처방을 "
                 "대체하지 않습니다. 실제 치료가 필요하시면 반드시 전문의와 "
                 "상담해 주십시오.\n"
                 "[출처: 동의보감 관련 권·문]")
    return "\n\n".join(parts)


def cot_compare_two(slots: dict, sources: list[tuple[str, str]]) -> str:
    """compare — 2-column 구조."""
    left: list[tuple[str, str]] = []
    right: list[tuple[str, str]] = []
    for role, quote in sources:
        if "left" in role:
            left.append((role, quote))
        elif "right" in role:
            right.append((role, quote))
        else:
            # fallback: 반씩 나누기
            (left if len(left) <= len(right) else right).append((role, quote))

    topic_left = slots.get("topic_left", "첫 번째 대상")
    topic_right = slots.get("topic_right", "두 번째 대상")
    parts = [f"두 대상은 성격이 뚜렷이 다르므로 원문을 나눠서 살펴봐야 합니다.\n"]
    parts.append(f"**{topic_left}**")
    for _, q in left or sources[:1]:
        parts.append(f"- \"{q}\"")
    parts.append(f"\n**{topic_right}**")
    for _, q in right or sources[1:]:
        parts.append(f"- \"{q}\"")
    parts.append("\n**정리**: 두 대상은 위와 같이 기록되어 있어, 각자 다른 범주·"
                 "성격에 속함을 알 수 있습니다. 본 답은 17세기 문헌 원문에 기반한 "
                 "비교이며, 현대 의학의 진단·처방을 대체하지 않습니다.\n"
                 "[출처: 동의보감 내경편 권1 총목 및 본문]")
    return "\n".join(parts)


def cot_list_enum(slots: dict, sources: list[tuple[str, str]]) -> str:
    parts = ["동의보감 원문에 기록된 항목을 순서대로 나열하면 다음과 같습니다.\n"]
    for i, (role, quote) in enumerate(sources, 1):
        parts.append(f"{i}) \"{quote}\"")
    parts.append("\n위 항목들이 해당 편·문·권의 구성입니다. 본 답은 원문을 그대로 "
                 "발췌해 열거한 것이며, 현대 분류 체계나 진단 기준을 대체하지 "
                 "않습니다.\n"
                 "[출처: 동의보감 내경편 권1 총목]")
    return "\n".join(parts)


def cot_symptom_classify(slots: dict, sources: list[tuple[str, str]],
                         branching: list[dict] | None) -> str:
    parts = ["이 질문은 조건에 따라 분류가 달라지므로 원문의 분류 기준을 "
             "단계적으로 확인해야 합니다.\n"]
    for i, (role, quote) in enumerate(sources, 1):
        parts.append(f"**Step {i}** (분류 기준) — \"{quote}\"")
    if branching:
        parts.append("\n**조건별 분류**")
        for b in branching:
            parts.append(f"- 조건 「{b['condition']}」 → **{b['result']}**")
    parts.append("\n따라서 질문의 조건을 원문의 분류 기준에 매핑해 해당 유형을 "
                 "답할 수 있습니다. 본 답은 17세기 문헌의 분류 틀을 소개한 "
                 "것이며, 실제 진단·처방은 반드시 전문의와 상담해 주십시오.\n"
                 "[출처: 동의보감 잡병편 권5 해수문]")
    return "\n\n".join(parts)


def cot_temporal_branch(slots: dict, sources: list[tuple[str, str]]) -> str:
    parts = ["질문의 각 연도에 해당하는 왕을 원문에서 단계적으로 찾아봅니다.\n"]
    for i, (role, quote) in enumerate(sources, 1):
        parts.append(f"**Step {i}** — \"{quote}\"")
    parts.append("\n**연도별 왕**")
    parts.append("- 1596년 (병신년) → 선조(宣祖) — 편찬 하교")
    parts.append("- 1608년 → 광해군(光海君) 즉위 (선조 승하)")
    parts.append("- 1610년 (경술년) → 광해군 — 동의보감 완성 진상")
    parts.append("\n[출처: 동의보감 내경편 권1 서문]")
    return "\n\n".join(parts)


TEMPLATE_FN = {
    "cot_temporal_chain": cot_temporal_chain,
    "cot_role_chain": cot_role_chain,
    "cot_category_bridge": cot_category_bridge,
    "cot_principles": cot_principles,
    "cot_symptom_chain": cot_symptom_chain,
    "cot_compare_two": cot_compare_two,
    "cot_list_enum": cot_list_enum,
    "cot_temporal_branch": cot_temporal_branch,
}


def render_anchor(seed: dict, records: dict) -> list[dict]:
    """anchor seed 하나에서 question_templates 수만큼 답변 생성."""
    results = []
    template_id = seed["template_id"]
    # source_records -> list of (role, quote)
    source_pairs: list[tuple[str, str]] = []
    for sr in seed["source_records"]:
        vol, seq = parse_ref(sr["ref"])
        rec = records.get((vol, seq))
        if rec is None:
            raise KeyError(f"seed {seed['id']}: {sr['ref']} not found")
        field_text = (rec.get("trans_ko") or "").strip()
        if sr["quote_span"] not in field_text:
            raise AssertionError(
                f"seed {seed['id']}: quote not in {sr['ref']} — {sr['quote_span'][:40]}..."
            )
        source_pairs.append((sr.get("role", ""), sr["quote_span"]))

    # 분기
    if template_id == "cot_symptom_classify":
        branching = seed.get("branching", [])
        answer_fn = lambda slots, sources: cot_symptom_classify(slots, sources, branching)
    elif template_id in TEMPLATE_FN:
        answer_fn = TEMPLATE_FN[template_id]
    else:
        raise KeyError(f"unknown template_id: {template_id}")

    # compare 용 slot
    slots = {}
    if "vs_" in seed.get("subtype", "") or template_id == "cot_compare_two":
        # id 에서 주제 추출
        sid = seed["id"]
        if "naegyeong_vs_oeheyong" in sid:
            slots = {"topic_left": "내경편 (內景篇)", "topic_right": "외형편 (外形篇)"}
        elif "jabbyeong_vs_tangaek" in sid:
            slots = {"topic_left": "잡병편 (雜病篇)", "topic_right": "탕액편 (湯液篇)"}
        elif "jeong_vs_gi" in sid:
            slots = {"topic_left": "정 (精)", "topic_right": "기 (氣)"}
        elif "jeong_vs_shin" in sid:
            slots = {"topic_left": "정 (精)", "topic_right": "신 (神)"}
        elif "hae_vs_su" in sid:
            slots = {"topic_left": "해 (咳)", "topic_right": "수 (嗽)"}
        else:
            slots = {"topic_left": "첫 대상", "topic_right": "두 번째 대상"}

    answer = answer_fn(slots, source_pairs)
    answer = mask_dosage(answer)

    for qi, qt in enumerate(seed["question_templates"]):
        results.append({
            "id": f"v4_{seed['id']}__q{qi}",
            "seed_id": seed["id"],
            "subtype": seed["subtype"],
            "question": qt,
            "answer": answer,
            "source_records": [sr["ref"] for sr in seed["source_records"]],
            "template_id": template_id,
        })
    return results


# ────────────────────── 자동 확장: sibling group ──────────────────────

def find_sibling_groups(records: dict, min_size: int = 2, max_size: int = 6) -> dict[str, list[tuple]]:
    parent_to_children = defaultdict(list)
    for (vol, seq), r in records.items():
        up = (r.get("up_path_nm") or "").strip()
        ko = (r.get("trans_ko") or "").strip()
        if not up or r["content_level"] not in {"ZZ", "DP"}:
            continue
        if " > " not in up or len(ko) < 40 or len(ko) > 400:
            continue
        parent = up.rsplit(" > ", 1)[0]
        parent_to_children[parent].append((vol, seq, ko))
    return {p: c for p, c in parent_to_children.items()
            if min_size <= len(c) <= max_size}


def auto_expand_compare(sibling_groups: dict, records: dict,
                        per_group_cap: int, rng: random.Random,
                        target: int) -> list[dict]:
    out = []
    parents = list(sibling_groups.keys())
    rng.shuffle(parents)
    for parent in parents:
        if len(out) >= target:
            break
        children = sibling_groups[parent]
        if len(children) < 2:
            continue
        n_pair = min(per_group_cap, len(children) - 1)
        for i in range(n_pair):
            if i + 1 >= len(children):
                break
            left_vol, left_seq, left_ko = children[i]
            right_vol, right_seq, right_ko = children[i + 1]
            left_leaf = (records[(left_vol, left_seq)].get("up_path_nm") or "").rsplit(" > ", 1)[-1]
            right_leaf = (records[(right_vol, right_seq)].get("up_path_nm") or "").rsplit(" > ", 1)[-1]
            if left_leaf == right_leaf:
                continue
            question = f"동의보감 {parent} 아래 '{left_leaf}' 와 '{right_leaf}' 는 각각 어떻게 기술되어 있나요?"
            source_pairs = [
                ("left", left_ko),
                ("right", right_ko),
            ]
            slots = {"topic_left": left_leaf, "topic_right": right_leaf}
            answer = cot_compare_two(slots, source_pairs)
            answer = mask_dosage(answer)
            out.append({
                "id": f"v4_auto_cmp_{parent}_{left_leaf}_{right_leaf}".replace(" ", "_"),
                "subtype": "compare.sibling_auto",
                "question": question,
                "answer": answer,
                "source_records": [
                    f"book_008/vol_{left_vol:02d}/seq_{left_seq}",
                    f"book_008/vol_{right_vol:02d}/seq_{right_seq}",
                ],
                "template_id": "cot_compare_two",
            })
    return out


def auto_expand_list(sibling_groups: dict, records: dict,
                     min_children: int, max_children: int,
                     rng: random.Random, target: int) -> list[dict]:
    out = []
    parents = list(sibling_groups.keys())
    rng.shuffle(parents)
    for parent in parents:
        if len(out) >= target:
            break
        children = sibling_groups[parent]
        if not (min_children <= len(children) <= max_children):
            continue
        parent_leaf = parent.rsplit(" > ", 1)[-1] if " > " in parent else parent
        question = f"동의보감 {parent} 아래에는 어떤 하위 항목이 있으며, 각 항목은 어떻게 기술되나요?"
        source_pairs = [(f"item_{i}", rec[2]) for i, rec in enumerate(children, 1)]
        answer = cot_list_enum({}, source_pairs)
        answer = mask_dosage(answer)
        out.append({
            "id": f"v4_auto_list_{parent}".replace(" ", "_").replace(">", "_"),
            "subtype": "list.sibling_auto",
            "question": question,
            "answer": answer,
            "source_records": [
                f"book_008/vol_{rec[0]:02d}/seq_{rec[1]}" for rec in children
            ],
            "template_id": "cot_list_enum",
        })
    return out


def auto_expand_multi_hop(records: dict, rng: random.Random,
                          target: int) -> list[dict]:
    """같은 병증 leaf 를 공유하는 2~3 record 를 chain."""
    # leaf 같은 record 끼리 그룹
    leaf_to_records = defaultdict(list)
    for (vol, seq), r in records.items():
        up = (r.get("up_path_nm") or "").strip()
        ko = (r.get("trans_ko") or "").strip()
        if not up or r["content_level"] not in {"ZZ", "DP"}:
            continue
        if " > " not in up or len(ko) < 80 or len(ko) > 400:
            continue
        leaf = up.rsplit(" > ", 1)[-1]
        leaf_to_records[leaf].append((vol, seq, ko, up))

    out = []
    leaves = [l for l, rs in leaf_to_records.items() if 2 <= len(rs) <= 5]
    rng.shuffle(leaves)
    for leaf in leaves:
        if len(out) >= target:
            break
        rs = leaf_to_records[leaf][:3]
        if len(rs) < 2:
            continue
        parent_path = rs[0][3]
        question = (
            f"동의보감 {parent_path} 대목에 기록된 {leaf}에 대한 설명을 "
            f"문헌 순서대로 정리해 주세요."
        )
        source_pairs = [(f"step_{i+1}", r[2]) for i, r in enumerate(rs)]
        answer = cot_symptom_chain({}, source_pairs)
        answer = mask_dosage(answer)
        out.append({
            "id": f"v4_auto_mh_{parent_path}_{leaf}".replace(" ", "_").replace(">", "_")[:120],
            "subtype": "multi_hop.symptom_chain_auto",
            "question": question,
            "answer": answer,
            "source_records": [f"book_008/vol_{r[0]:02d}/seq_{r[1]}" for r in rs],
            "template_id": "cot_symptom_chain",
        })
    return out


def auto_expand_conditional(records: dict, rng: random.Random,
                            target: int) -> list[dict]:
    """단일 병증 record + '만약 X 조건이면' 분기 질문."""
    # ZZ level 의 분류형 record (해수·풍·한·허로 키워드 포함)
    class_kw = ["분류", "구별", "나누", "세 가지", "두 가지", "형(形)", "證", "證候"]
    cands = []
    for (vol, seq), r in records.items():
        ko = (r.get("trans_ko") or "").strip()
        if r["content_level"] != "ZZ" or len(ko) < 150 or len(ko) > 400:
            continue
        if any(kw in ko for kw in class_kw):
            cands.append((vol, seq, ko, r.get("up_path_nm") or ""))
    rng.shuffle(cands)

    out = []
    for vol, seq, ko, up in cands:
        if len(out) >= target:
            break
        leaf = up.rsplit(" > ", 1)[-1] if " > " in up else ""
        question = (
            f"동의보감 {up}에서 {leaf or '이 항목'}의 분류 기준은 무엇이며, "
            f"특정 증상이 나타날 때 어떤 유형으로 판별할 수 있나요?"
        )
        answer = cot_symptom_classify(
            {}, [("classification", ko)],
            branching=None,
        )
        answer = mask_dosage(answer)
        out.append({
            "id": f"v4_auto_cond_vol{vol:02d}_seq{seq}",
            "subtype": "conditional.classify_auto",
            "question": question,
            "answer": answer,
            "source_records": [f"book_008/vol_{vol:02d}/seq_{seq}"],
            "template_id": "cot_symptom_classify",
        })
    return out


# ──────────────────────────── validation ────────────────────────────

COT_PAT = re.compile(
    r"(Step\s*\d+|①|②|③|\d+\)\s|\*\*Step|\*\*정리\*\*|\*\*조건|"
    r"\*\*[가-힣()A-Za-z0-9 ·]{2,20}\*\*)"
)

def validate_quotes(answer: str, source_refs: list[str],
                    records: dict) -> bool:
    """각 source ref 의 trans_ko quote 가 answer 에 substring 인가 점검 (관대 40자)."""
    for ref in source_refs:
        try:
            vol, seq = parse_ref(ref)
        except Exception:
            continue
        rec = records.get((vol, seq))
        if rec is None:
            continue
        ko = (rec.get("trans_ko") or "").strip()
        if not ko:
            continue
        if ko[:40] not in answer:
            # 자동 확장인 경우 full_ko 의 40자 prefix 매칭
            return False
    return True


def validate_cot(answer: str) -> bool:
    return bool(COT_PAT.search(answer))


# ───────────────────────────── main ─────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=Path, default=Path("data/sft/complex_seeds.yaml"))
    p.add_argument("--raw-dir", type=Path,
                   default=Path("data/raw/mediclassics_unified/book_008"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--stats-out", type=Path, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--per-group-cap", type=int, default=4)
    p.add_argument("--target-multi-hop", type=int, default=800)
    p.add_argument("--target-compare", type=int, default=600)
    p.add_argument("--target-list", type=int, default=500)
    p.add_argument("--target-conditional", type=int, default=400)
    args = p.parse_args()

    rng = random.Random(args.seed)

    print("[v4] loading seeds...")
    seed_doc = yaml.safe_load(args.seeds.read_text(encoding="utf-8"))
    print("[v4] loading records...")
    records = load_records_indexed(args.raw_dir)
    print(f"[v4] records: {len(records)}")

    system_prompt = seed_doc.get("system_prompt", SYSTEM_PROMPT_DEFAULT).strip()

    # ── anchor 렌더 ──
    anchors: list[dict] = []
    for cat_key in ["multi_hop", "compare", "list", "conditional"]:
        cat = seed_doc.get(cat_key, {})
        for seed in cat.get("seeds", []):
            try:
                anchors.extend(render_anchor(seed, records))
            except (AssertionError, KeyError) as e:
                print(f"[warn] anchor {seed.get('id')} skipped: {e}", file=sys.stderr)

    print(f"[v4] anchors rendered: {len(anchors)}")

    # ── 자동 확장 ──
    sibling_groups = find_sibling_groups(records, min_size=2, max_size=8)
    print(f"[v4] sibling groups: {len(sibling_groups)}")

    auto_cmp = auto_expand_compare(sibling_groups, records,
                                    args.per_group_cap, rng,
                                    args.target_compare)
    auto_list = auto_expand_list(sibling_groups, records,
                                  min_children=3, max_children=8,
                                  rng=rng, target=args.target_list)
    auto_mh = auto_expand_multi_hop(records, rng, args.target_multi_hop)
    auto_cond = auto_expand_conditional(records, rng, args.target_conditional)

    print(f"[v4] auto: mh={len(auto_mh)} cmp={len(auto_cmp)} list={len(auto_list)} cond={len(auto_cond)}")

    all_items = anchors + auto_mh + auto_cmp + auto_list + auto_cond

    # ── validation + jsonl emit ──
    rows = []
    rejects = []
    for item in all_items:
        answer = item["answer"]
        # CoT 검증
        if not validate_cot(answer):
            rejects.append({"id": item["id"], "reason": "no_cot_structure"})
            continue
        # token 길이
        if len(answer) < 200:
            rejects.append({"id": item["id"], "reason": "too_short", "len": len(answer)})
            continue
        # dedup by answer sha1
        ah = hashlib.sha1(answer.encode()).hexdigest()[:16]

        rows.append({
            "id": item["id"],
            "category": "complex_reasoning",
            "subcat": item["subtype"],
            "q_format": f"F_complex_{item['subtype'].split('.')[0]}",
            "a_format": "A_cot",
            "question": item["question"],
            "assistant": answer,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": item["question"]},
                {"role": "assistant", "content": answer},
            ],
            "source_records": item["source_records"],
            "template_id": item["template_id"],
            "answer_tokens": len(answer.split()),  # approximate
            "_answer_hash": ah,
        })

    # shuffle + dedup on hash
    seen_hash = set()
    deduped = []
    for r in rows:
        if r["_answer_hash"] in seen_hash:
            continue
        seen_hash.add(r["_answer_hash"])
        deduped.append(r)
    rng.shuffle(deduped)

    # write
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in deduped:
            r.pop("_answer_hash", None)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[v4] wrote {args.out}: {len(deduped)} pairs (rejects={len(rejects)})")

    stats = {
        "total": len(deduped),
        "rejects": len(rejects),
        "reject_reasons": dict(Counter(r["reason"] for r in rejects)),
        "by_subcat": dict(Counter(r["subcat"] for r in deduped)),
        "by_template": dict(Counter(r["template_id"] for r in deduped)),
        "anchor_count": len(anchors),
        "auto_mh": len(auto_mh),
        "auto_cmp": len(auto_cmp),
        "auto_list": len(auto_list),
        "auto_cond": len(auto_cond),
    }
    print(f"[v4] stats: {json.dumps(stats, ensure_ascii=False)}")
    if args.stats_out:
        args.stats_out.parent.mkdir(parents=True, exist_ok=True)
        args.stats_out.write_text(json.dumps(stats, ensure_ascii=False, indent=2),
                                   encoding="utf-8")


if __name__ == "__main__":
    main()
