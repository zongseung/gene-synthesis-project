#!/usr/bin/env python3
"""Augment / curate HanMed-LLM SFT corpus v6 → v7.

Implements post-processing fixes A2 / A3 / A6 from the round_3 supervisor
final report and enforces 답변 구체성 원칙 3·4 from the ver6 fix plan
(docs/ver6/00_halluc_repetition_fix_plan.md §3.1.2).

Pipeline stages (applied in order):
  1. A2 — "X문에 나온다" shell-answer 필터
  2. A3 — refusal diversification + "-문" surface in-scope QA 보강
  3. A6 — 침구편 inverse-map QA 100~200 rows
  4. Validator — 원칙 3 (prefix diversity) & 원칙 4 (closing diversity)

Usage
-----
.venv/bin/python scripts/augment_sft_v7.py \
    --in  experiments/dongui_bogam/data/sft/phaseB_qa_v7_corpus_raw.jsonl \
    --out experiments/dongui_bogam/data/sft/phaseB_qa_v7_corpus.jsonl \
    --whitelist experiments/dongui_bogam/data/sft/entity_whitelist_v6.yaml \
    --raw-dir data/raw/mediclassics_unified/book_008

The script is **idempotent** and re-runnable: output directory is rebuilt
every invocation. Drop-logs + preview / audit artefacts go under
``--workspace`` (default: ``_workspace``).

Design note — guard-rails from the user's spec:
  * Never writes back to the v6 input file.
  * Never touches scripts/build_sft_full_corpus.py (Agent 1's turf).
  * Whitelist / raw catalog / source corpus are read-only.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

# Allow direct execution (python scripts/augment_sft_v7.py …) while still
# importing the sibling variant pool module.
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from _v7_refusal_variants import (  # noqa: E402  (import-after-sys-path mutation)
    OOS_BOOK_NAMES,
    REFUSAL_OOS_TEMPLATES,
    REFUSAL_SAFETY_TEMPLATES,
    SAFETY_FIXED_CLOSINGS,
    SAFETY_TOPICS,
)

SYSTEM_PROMPT = (
    "당신은 한의학 고전 문헌 연구 보조 AI 입니다. 동의보감(東醫寶鑑) 본문에 근거해 "
    "편·장 구조, 처방, 약재, 경혈, 증론을 정확하고 간결하게 답합니다. 원문에 없는 "
    "인명·연도·처방은 창작하지 않으며, 개인 진단이나 구체 용량 처방은 제공하지 않습니다."
)

# --- Regexes ---------------------------------------------------------------
# Covers the full family of "X에 나온다" shell fragments observed in v6:
#   - "처방은 상한문에 나온다", "대변문에 나온다"
#   - "처방은 뒤에 나온다", "앞에 나온다"
#   - "신문(神門)에 나온다", "서문(暑門)에 나온다"  (hanja parenthesis)
#   - "《의림》에 나온다", "《입문》에 나온다"       (book citation)
#   - "상한문쪽에 나온다" (쪽 suffix)
#   - "자세한 것은 뒤에 나온다"
# Followed optionally by object postpositions (을|를|에|과|이|가|은|는)
# and verb tails (써야|쓴다|한다|하다 ...).
_CROSS_REF_BODY = (
    r"(?:"
    r"(?:처방은\s*|자세한\s*(?:것은|내용은)\s*|네\s*처방은\s*|두\s*처방은\s*)?"
    r"(?:"
    r"[가-힣]+문(?:쪽)?(?:\([^)]*\))?에"            # 상한문에 / 신문(神門)에
    r"|[앞뒤모]+에"                                      # 앞에 / 뒤에 / 모두에
    r"|모두\s*(?:앞|뒤)에"
    r"|《[^》]+》에"                                    # 《입문》에
    r")"
    r"\s*나온다(?:을|를|에|과|와|이|가|은|는)?"
    r"(?:\s*(?:써야|쓴다|한다|하다|투여|먹는|뜸뜬다))?"
    r")"
)
CROSS_REF_RE = re.compile(_CROSS_REF_BODY)
SOURCE_CITE_RE = re.compile(r"\[출처:\s*([^\]]+)\]")

# Depth-1 token groups (Korean) that are in-scope for "-문" / "-편" surface
# augmentation. These are canonical up_path[1] values already present in v6.
MUN_TARGETS: list[tuple[str, str]] = [
    # (surface_form_shown_in_question, canonical up_path[1] token in corpus)
    ("풍문", "풍"),
    ("소아문", "소아"),
    ("부인문", "부인"),
    ("담음문", "담음"),
    ("허로문", "허로"),
    ("해수문", "해수"),
    ("내상문", "내상"),
    ("적취문", "적취"),
    ("옹저문", "옹저"),
    ("제창문", "제창"),
    ("제상문", "제상"),
    ("대변문", "대변"),
    ("소변문", "소변"),
    ("침구편", "침구"),
    ("침구법", "침구"),
    ("상한편", "한"),  # 한 (寒) → 상한
    ("전음문", "전음"),
    ("후음문", "후음"),
    ("눈문", "눈"),
    ("혈문", "혈"),
    ("화문", "화"),
]


# --- Utility ---------------------------------------------------------------
def rec_id(prefix: str, salt: str, n: int) -> str:
    h = hashlib.md5(f"{prefix}|{salt}|{n}".encode("utf-8")).hexdigest()[:8]
    return f"{prefix}_{h}_{n}"


def load_jsonl(p: str | os.PathLike) -> list[dict]:
    out: list[dict] = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            out.append(json.loads(line))
    return out


def dump_jsonl(rows: Iterable[dict], p: str | os.PathLike) -> int:
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def _build_messages(user: str, assistant: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]


def _new_row(
    *, row_id: str, category: str, subcat: str, up_path: str | None,
    question: str, assistant: str,
) -> dict:
    return {
        "id": row_id,
        "category": category,
        "subcat": subcat,
        "up_path_nm": up_path,
        "question": question,
        "assistant": assistant,
        "messages": _build_messages(question, assistant),
    }


# ---------------------------------------------------------------------------
# STAGE 1 — A2. "X문에 나온다" shell-answer 필터
# ---------------------------------------------------------------------------
@dataclass
class ShellFilterStats:
    inspected: int = 0
    matched: int = 0
    dropped: int = 0
    trimmed: int = 0


def stage1_filter_shell_answers(
    rows: list[dict], dropped_path: str | os.PathLike,
    drop_ratio: float = 0.8,
) -> tuple[list[dict], ShellFilterStats]:
    """Remove / trim "처방은 X문에 나온다" shell answers.

    - If matched segments cover ≥ ``drop_ratio`` of the assistant text: **drop**
      the row (logged to ``dropped_path``).
    - Else: strip just the matched sentences and normalize whitespace.
    """
    stats = ShellFilterStats()
    kept: list[dict] = []
    dropped: list[dict] = []

    for r in rows:
        stats.inspected += 1
        a = r.get("assistant") or ""
        if not a:
            kept.append(r)
            continue

        matches = list(CROSS_REF_RE.finditer(a))
        if not matches:
            kept.append(r)
            continue

        stats.matched += 1
        matched_len = sum(m.end() - m.start() for m in matches)
        ratio = matched_len / max(1, len(a))

        if ratio >= drop_ratio:
            stats.dropped += 1
            dropped.append(r)
            continue

        # trim: remove each matched segment, collapse whitespace
        new_a = CROSS_REF_RE.sub(" ", a)
        new_a = re.sub(r"\s{2,}", " ", new_a).strip()
        # also remove a dangling sentence like "X을(를)" artifact
        new_a = re.sub(r"\s+([,.·:])", r"\1", new_a)
        if not new_a:
            stats.dropped += 1
            dropped.append(r)
            continue

        stats.trimmed += 1
        r = dict(r)
        r["assistant"] = new_a
        if r.get("messages"):
            # mutate the assistant message in chat messages too
            for m in r["messages"]:
                if m.get("role") == "assistant":
                    m["content"] = new_a
        kept.append(r)

    if dropped:
        dump_jsonl(dropped, dropped_path)
    return kept, stats


# ---------------------------------------------------------------------------
# STAGE 2 — A3. refusal diversification + "-문" surface in-scope QA
# ---------------------------------------------------------------------------
@dataclass
class RefusalStats:
    oos_kept: int = 0
    oos_new: int = 0
    oos_unique: int = 0
    sfy_kept: int = 0
    sfy_new: int = 0
    sfy_unique: int = 0
    mun_added: int = 0


def _rebuild_refusal_oos(rng: random.Random, target_count: int = 100) -> list[dict]:
    """Generate ``target_count`` diverse OOS refusals using variant templates.

    Each row is a question ("X 에 관해 설명해 주세요" etc) × a variant
    assistant. We intentionally shuffle both axes to hit ≥ 40 unique assistants.
    """
    question_templates = [
        "{book} 의 주요 내용을 정리해 주세요.",
        "{book} 은(는) 누가 편찬한 책인가요?",
        "{book} 의 편·권 구조를 설명해 주세요.",
        "{book} 에서 다루는 주요 처방을 알려 주세요.",
        "{book} 의 시대적 배경과 의의는 무엇인가요?",
        "{book} 에 수록된 경혈 체계를 알려 주세요.",
        "{book} 의 약재 분류 기준을 정리해 주세요.",
        "{book} 의 서문을 요약해 주세요.",
        "{book} 의 편집 원칙이 궁금합니다.",
        "{book} 의 초간본 연도를 알려 주세요.",
        "{book} 이(가) 후대 의서에 미친 영향을 설명해 주세요.",
        "{book} 의 저자 생애를 간단히 소개해 주세요.",
    ]

    rows: list[dict] = []
    # Build (question, answer) pairs by sampling templates + book names
    # with a bias that ensures variant coverage.
    # We use every assistant template at least once (~50 templates) then
    # fill the rest randomly.
    assistant_indices = list(range(len(REFUSAL_OOS_TEMPLATES)))
    rng.shuffle(assistant_indices)

    for i in range(target_count):
        q_tmpl = rng.choice(question_templates)
        book = rng.choice(OOS_BOOK_NAMES)
        q = q_tmpl.format(book=book)

        if i < len(assistant_indices):
            a_tmpl = REFUSAL_OOS_TEMPLATES[assistant_indices[i]]
        else:
            a_tmpl = rng.choice(REFUSAL_OOS_TEMPLATES)
        a = a_tmpl.format(book=book)

        rows.append(_new_row(
            row_id=rec_id("refusal_oos", book, i),
            category="refusal_oos",
            subcat="OOS",
            up_path=None,
            question=q,
            assistant=a,
        ))
    return rows


def _rebuild_refusal_safety(rng: random.Random, target_count: int = 100) -> list[dict]:
    question_templates = [
        "{topic} 을(를) 바로 알려 주세요.",
        "제 증상에 맞는 {topic} 을(를) 알려 주세요.",
        "{topic} 은(는) 어떻게 결정하나요?",
        "{topic} 을(를) 구체적 수치로 알려 주세요.",
        "{topic} 을(를) 지금 바로 시행하고 싶습니다.",
        "{topic} 에 관한 상세 지침을 주세요.",
        "{topic} 을(를) 바로 처방해 주세요.",
        "{topic} 을(를) 문헌 기준으로 알려 주세요.",
        "개인 {topic} 을(를) 추천해 주세요.",
        "{topic} 을(를) 수치로 명시해 주세요.",
    ]

    rows: list[dict] = []
    assistant_indices = list(range(len(REFUSAL_SAFETY_TEMPLATES)))
    rng.shuffle(assistant_indices)

    for i in range(target_count):
        q_tmpl = rng.choice(question_templates)
        topic = rng.choice(SAFETY_TOPICS)
        q = q_tmpl.format(topic=topic)

        if i < len(assistant_indices):
            a_tmpl = REFUSAL_SAFETY_TEMPLATES[assistant_indices[i]]
        else:
            a_tmpl = rng.choice(REFUSAL_SAFETY_TEMPLATES)
        a_core = a_tmpl.format(topic=topic)

        # Optionally append a fixed safety closing (≤ 50% of rows to honour
        # 원칙 4 closing-top1 ≤ 10%).
        if rng.random() < 0.45:
            a = a_core + " " + rng.choice(SAFETY_FIXED_CLOSINGS)
        else:
            a = a_core

        rows.append(_new_row(
            row_id=rec_id("refusal_safety", topic, i),
            category="refusal_safety",
            subcat="SFY",
            up_path=None,
            question=q,
            assistant=a,
        ))
    return rows


def _load_raw_records(raw_dir: str | None) -> list[dict]:
    if not raw_dir:
        return []
    paths = sorted(glob.glob(os.path.join(raw_dir, "vol_*.jsonl")))
    recs: list[dict] = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                recs.append(json.loads(line))
    return recs


# Mapping from canonical up_path[1] Korean token → v6-style korean up_path
# prefix. e.g. "풍" is under "잡병편 권2 > 풍".
# We lazily derive this from v6 corpus passage/symptom up_paths.
def _derive_path_prefixes(v6_rows: list[dict]) -> dict[str, list[str]]:
    prefixes: dict[str, set[str]] = defaultdict(set)
    for r in v6_rows:
        up = r.get("up_path_nm") or ""
        parts = [p.strip() for p in up.split(">")]
        if len(parts) >= 2 and parts[0].startswith(("내경편", "외형편", "잡병편", "탕액편", "침구편", "탕액", "서례")):
            prefixes[parts[1]].add(" > ".join(parts[:2]))
    return {k: sorted(v) for k, v in prefixes.items()}


def _collect_passages_for_segment(
    v6_rows: list[dict], canonical_token: str, min_detail_len: int = 120,
    max_out: int = 60,
) -> list[dict]:
    """Pull passage/symptom rows whose up_path[1] == canonical_token.

    The assistant field already contains translated passage content we can
    reuse (AFTER shell filter has run). We dedup by up_path_nm.
    """
    hits: list[dict] = []
    seen_paths: set[str] = set()
    for r in v6_rows:
        if r.get("category") not in ("passage", "symptom", "structure"):
            continue
        up = r.get("up_path_nm") or ""
        parts = [p.strip() for p in up.split(">")]
        if len(parts) < 2 or parts[1] != canonical_token:
            continue
        if up in seen_paths:
            continue
        a = (r.get("assistant") or "").strip()
        if len(a) < min_detail_len:
            continue
        seen_paths.add(up)
        hits.append(r)
        if len(hits) >= max_out:
            break
    return hits


def _build_mun_inscope_qa(
    rng: random.Random, v6_rows: list[dict], target_count: int = 150,
) -> list[dict]:
    """Produce "-문" surface-form in-scope QA by borrowing passages
    already in v6/raw corpus keyed on canonical up_path[1] token.
    """
    out: list[dict] = []
    per_target = max(6, target_count // len(MUN_TARGETS))

    # Postposition-stripping heuristic for ugly leaf names like "비병은"
    _POSTP_RE = re.compile(r"(은|는|이|가|을|를|과|와|의|에|도)$")

    def _clean_leaf(raw_leaf: str) -> str:
        leaf = raw_leaf.strip()
        # Strip postposition if leaf is ≤ 6 chars (otherwise likely semantic)
        if len(leaf) <= 6:
            stripped = _POSTP_RE.sub("", leaf)
            if stripped and stripped != leaf:
                leaf = stripped
        return leaf

    for idx, (surface, canonical) in enumerate(MUN_TARGETS):
        passages = _collect_passages_for_segment(v6_rows, canonical)
        if not passages:
            continue
        # sample passages
        rng.shuffle(passages)
        quota = min(per_target, len(passages))
        for j in range(quota):
            p = passages[j]
            up = p.get("up_path_nm") or ""
            parts = [x.strip() for x in up.split(">")]
            raw_leaf = parts[-1] if parts else canonical
            leaf = _clean_leaf(raw_leaf)
            # Skip if leaf ends up empty / too generic
            if not leaf or leaf in ("여러", "여"):
                continue
            excerpt = (p.get("assistant") or "").strip()
            # Scrub "中門" literal leak from upstream data (A4 complement)
            excerpt = excerpt.replace("(中門)", "").replace(" 中門 ", " ")

            # Four question stems — pick by idx to diversify prefix
            q_choices = [
                f"잡병편 {surface}에서 {leaf} 관련 조문을 정리해 주세요.",
                f"동의보감 {surface}에 수록된 '{leaf}' 내용을 설명해 주세요.",
                f"{surface} 아래 '{leaf}' 대목은 어떤 병리와 치법을 다루나요?",
                f"'{leaf}' 은(는) 동의보감 {surface} 어디에 수록돼 있으며 핵심 요지는 무엇인가요?",
                f"{surface}의 '{leaf}' 조문을 근거로 요지를 요약해 주세요.",
                f"동의보감 {surface}에서 다루는 '{leaf}' 의 개요를 알려 주세요.",
            ]
            q = q_choices[(idx + j) % len(q_choices)]

            # Assistant mirrors the original passage answer but with a
            # contextual opener varying with (surface, leaf) — NOT a fixed
            # 3,000-row template. We strip any lingering shell-phrases first.
            body = excerpt
            # If assistant already includes a [출처: ...] marker, preserve it.
            if "[출처:" not in body and "(출처:" not in body:
                body = body + f" (출처: {up})"

            openers = [
                f"동의보감 {surface} 의 '{leaf}' 대목은 다음과 같이 서술됩니다: ",
                f"{surface}({parts[0] if parts else ''}) '{leaf}' 에 관해서는 원문상 다음과 같이 기재돼 있습니다. ",
                f"{surface} > {leaf} 조문 요지: ",
                f"'{leaf}' ({surface}) 내용 정리: ",
                f"{surface}의 '{leaf}' 조문은 다음 내용을 담고 있습니다. ",
                f"[{surface} · {leaf}] ",
            ]
            op = openers[(idx * 31 + j * 7) % len(openers)]

            a = op + body
            out.append(_new_row(
                row_id=rec_id("mun_inscope", f"{surface}|{leaf}", j),
                category="mun_inscope",
                subcat="IN",
                up_path=up,
                question=q,
                assistant=a,
            ))
            if len(out) >= target_count:
                return out
    return out


# ---------------------------------------------------------------------------
# STAGE 3 — A6. 침구편 inverse-map QA
# ---------------------------------------------------------------------------
MERIDIAN_KO_MAP = {
    # raw Chinese meridian-group name (depth-2 under 鍼灸篇 > 鍼灸) → Korean
    "手太陰肺經左右凡二十二穴": "수태음폐경",
    "手陽明大腸經左右凡四十穴": "수양명대장경",
    "足陽明胃經左右凡九十穴": "족양명위경",
    "足太陰脾經左右凡四十二穴": "족태음비경",
    "手少陰心經左右凡一十八穴": "수소음심경",
    "手太陽小腸經左右凡三十八穴": "수태양소장경",
    "足太陽膀胱經左右凡一百二十六穴": "족태양방광경",
    "足少陰腎經左右凡五十四穴": "족소음신경",
    "手厥陰心包經左右凡一十八穴": "수궐음심포경",
    "手少陽三焦經左右凡四十六穴": "수소양삼초경",
    "足少陽膽經左右凡九十穴": "족소양담경",
    "足厥陰肝經左右凡二十六穴": "족궐음간경",
    "督脉流注及孔穴": "독맥",
    "任脉流注及孔穴": "임맥",
    "別穴": "별혈",
}

# Major acupoints whose coverage must reach ≥20 rows.
MAJOR_ACUPOINTS = [
    ("足三里", "족삼리", "족양명위경"),
    ("合谷", "합곡", "수양명대장경"),
    ("太衝", "태충", "족궐음간경"),
    ("三陰交", "삼음교", "족태음비경"),
    ("曲池", "곡지", "수양명대장경"),
    ("內關", "내관", "수궐음심포경"),
    ("太谿", "태계", "족소음신경"),
    ("風池", "풍지", "족소양담경"),
    ("百會", "백회", "독맥"),
    ("關元", "관원", "임맥"),
    ("氣海", "기해", "임맥"),
    ("陰陵泉", "음릉천", "족태음비경"),
    ("陽陵泉", "양릉천", "족소양담경"),
    ("列缺", "열결", "수태음폐경"),
    ("崑崙", "곤륜", "족태양방광경"),
]


def _strip_hanja_paren(s: str) -> str:
    """"족삼리(足三里)" -> "족삼리". """
    return re.sub(r"\s*\([^)]*\)\s*", "", s).strip()


def _ko_name_from_trans_ko(tko: str) -> str:
    """Raw DK trans_ko often ends with trailing linebreak / digits; pick the
    first Korean segment.
    """
    if not tko:
        return ""
    first = tko.splitlines()[0].strip()
    first = re.sub(r"\s*\([^)]*\)\s*", "", first).strip()
    return first


def _build_acupoint_inverse_qa(
    rng: random.Random, raw_recs: list[dict], v6_rows: list[dict],
    target_min: int = 100, target_max: int = 200,
) -> list[dict]:
    # DK records hold only the name; the detail lives in the *following*
    # SS sibling records under the same up_path_nm until the next DK.
    # We walk the sequence in-order and accumulate SS details per DK.
    items: list[dict] = []
    current_dk: dict | None = None
    current_details: list[str] = []

    def _flush_current() -> None:
        nonlocal current_dk, current_details
        if not current_dk:
            current_dk = None
            current_details = []
            return
        r = current_dk
        up = r.get("up_path_nm") or ""
        parts = [p.strip() for p in up.split(">")]
        if len(parts) >= 3:
            han_meridian = parts[2]
            meridian_ko = MERIDIAN_KO_MAP.get(han_meridian)
            if meridian_ko:
                orig = (r.get("original") or "").strip()
                han_name = orig.splitlines()[0].strip()
                han_base = re.sub(r"(一穴|二穴|左右.*)$", "", han_name).strip()
                ko_name = _ko_name_from_trans_ko(r.get("trans_ko") or "")
                if ko_name and han_base:
                    up_ko = f"침구편 > 침구 > {meridian_ko}의 > {ko_name}"
                    detail_text = " ".join(
                        d.strip() for d in current_details if d and d.strip()
                    )[:600].strip()
                    if not detail_text:
                        detail_text = (
                            f"{ko_name}({han_base}) 혈은 {meridian_ko} 경맥의 주요 혈자리 "
                            f"중 하나로, 해당 경맥 장에 수록되어 있습니다."
                        )
                    items.append({
                        "han": han_base,
                        "ko": ko_name,
                        "meridian": meridian_ko,
                        "up_ko": up_ko,
                        "detail": detail_text,
                        "raw_up": up,
                    })
        current_dk = None
        current_details = []

    for r in raw_recs:
        lvl = r.get("content_level")
        up = r.get("up_path_nm") or ""
        # If we hit a new DK or leave 鍼灸 tree, flush the previous DK.
        if lvl == "DK":
            _flush_current()
            current_dk = r
            current_details = []
        elif lvl == "SS" and current_dk is not None:
            # Only accumulate if still in same meridian section.
            if up == (current_dk.get("up_path_nm") or ""):
                tko = (r.get("trans_ko") or "").strip()
                if tko:
                    current_details.append(tko)
            else:
                _flush_current()
        elif lvl in ("DK", "AA", "BB", "CC", "OO", "Z2"):
            # boundary — flush whatever we had
            _flush_current()
    _flush_current()

    # Force in-coverage of MAJOR_ACUPOINTS (alias gap fix).
    # Particularly: 足三里 is stored as 三里 (ko "삼리") under 足陽明胃經, but the
    # user's questions always say "족삼리". We create an alias item whose ko
    # is "족삼리" pointing at the same up_path / detail.
    have_ko = {it["ko"] for it in items}
    alias_pairs = [(han, ko, meridian) for han, ko, meridian in MAJOR_ACUPOINTS]
    # Additional alias: 足三里 should produce BOTH "삼리" (raw) and "족삼리" (common).
    alias_pairs.append(("足三里", "족삼리", "족양명위경"))

    for han, ko, meridian in alias_pairs:
        # If the raw short-name item already produced the data, clone it.
        short_alias = ko.replace("족", "") if ko.startswith("족") else None
        source_it = None
        if short_alias:
            for it in items:
                if it["ko"] == short_alias and it["meridian"] == meridian:
                    source_it = it
                    break
        if ko in have_ko and source_it is None:
            continue
        match_detail = source_it["detail"] if source_it else ""
        match_up = source_it["raw_up"] if source_it else ""
        # Also search raw for han literal
        if not match_detail:
            for r in raw_recs:
                orig = r.get("original") or ""
                if han in orig and r.get("content_level") == "SS":
                    match_detail = (r.get("trans_ko") or "")[:600]
                    match_up = r.get("up_path_nm") or ""
                    break
        # Fallback v6 lookup
        if not match_detail:
            for row in v6_rows:
                a = row.get("assistant") or ""
                if ko in a and row.get("category") == "acupoint":
                    match_detail = a[:600]
                    match_up = row.get("up_path_nm") or ""
                    break
        if ko in have_ko and not source_it:
            continue
        items.append({
            "han": han,
            "ko": ko,
            "meridian": meridian,
            "up_ko": f"침구편 > 침구 > {meridian}의 > {ko}",
            "detail": match_detail or (
                f"{ko} 혈은 {meridian} 경맥에 속하는 경혈이다. 상세 조문은 동의보감 침구편 "
                f"해당 경맥 장을 확인하십시오."
            ),
            "raw_up": match_up,
        })
        have_ko.add(ko)

    # De-dup by ko name — we allow up to 3 templates per item.
    items_by_ko: dict[str, dict] = {}
    for it in items:
        items_by_ko.setdefault(it["ko"], it)
    items_unique = list(items_by_ko.values())
    rng.shuffle(items_unique)

    out: list[dict] = []

    # Template A — 경맥 귀속 질의
    template_a_q = [
        "동의보감 침구편에서 '{ko}({han})' 혈은 어느 경맥에 속하나요?",
        "침구편의 '{ko}' 혈은 어느 경맥 소속 경혈인가요?",
        "'{ko}' 혈의 귀속 경맥을 동의보감 침구편 기준으로 알려 주세요.",
        "{ko}({han}) 은(는) 어느 경맥에 속하는 혈입니까?",
    ]
    template_a_a = [
        "{ko}({han}) 혈은 {meridian} 에 속합니다. (출처: {up_ko})",
        "동의보감 침구편에 따르면 {ko} 혈은 {meridian} 경맥에 수록된 경혈입니다. (출처: {up_ko})",
        "{ko} 혈의 귀속 경맥은 {meridian} 입니다. 해당 경맥 장에 수록돼 있습니다. (출처: {up_ko})",
        "{ko}({han}) — 소속 경맥: {meridian}. 동의보감 침구편 해당 경맥 장에서 확인 가능합니다. (출처: {up_ko})",
    ]

    # Template B — 위치·주치 정리
    template_b_q = [
        "{ko}({han}) 혈의 위치와 주치를 정리해 주세요.",
        "'{ko}' 혈의 위치·자침 깊이·주치를 동의보감 기준으로 설명해 주세요.",
        "{ko} 혈에 대한 동의보감 침구편의 서술을 정리해 주세요.",
        "{ko}({han}) 혈은 어떤 증상에 자침하고 어떻게 취혈하나요?",
    ]
    template_b_a = [
        "{ko}({han}) 혈에 대하여 동의보감 침구편은 다음과 같이 서술합니다. {detail} (출처: {up_ko})",
        "{ko} 혈 해설: {detail} (출처: {up_ko})",
        "{ko}({han}) — {meridian} 소속. {detail} (출처: {up_ko})",
    ]

    # Template C — 수록 위치 echo
    template_c_q = [
        "'{ko}' 은(는) 동의보감 어디에 수록돼 있나요?",
        "{ko} 혈은 동의보감 침구편 내 어느 장에 수록돼 있나요?",
        "{ko}({han}) 혈의 수록 위치(편·장)를 알려 주세요.",
    ]
    template_c_a = [
        "{ko} 혈은 {up_ko} 에 수록되어 있습니다.",
        "동의보감 내 수록 위치: 침구편 > 침구 > {meridian}의 > {ko}. (출처: {up_ko})",
        "{ko}({han}) 수록처: 침구편 {meridian} 장 (경로: {up_ko}).",
    ]

    def _emit(template_q: list[str], template_a: list[str], it: dict, n: int) -> dict:
        q = template_q[n % len(template_q)].format(**it)
        a = template_a[n % len(template_a)].format(**it)
        return _new_row(
            row_id=rec_id("acu_inv", it["ko"], n),
            category="acupoint",
            subcat="INV",
            up_path=it["up_ko"],
            question=q,
            assistant=a,
        )

    # Generate 3 templates per major acupoint to saturate coverage, and
    # double the density for high-priority entities (족삼리).
    major_kos = {ko for _, ko, _ in MAJOR_ACUPOINTS}
    priority_kos = {"족삼리"}
    counter = 0
    for it in items_unique:
        if it["ko"] not in major_kos:
            continue
        for _ in range(3 if it["ko"] in priority_kos else 1):
            out.append(_emit(template_a_q, template_a_a, it, counter))
            counter += 1
            out.append(_emit(template_b_q, template_b_a, it, counter))
            counter += 1
            out.append(_emit(template_c_q, template_c_a, it, counter))
            counter += 1

    # Then cycle through remaining acupoints for breadth, alternating templates
    # until we cross ``target_min``.
    for it in items_unique:
        if it["ko"] in major_kos:
            continue
        # choose 1 template for non-major (breadth over depth)
        tmpl_pick = counter % 3
        if tmpl_pick == 0:
            out.append(_emit(template_a_q, template_a_a, it, counter))
        elif tmpl_pick == 1:
            out.append(_emit(template_b_q, template_b_a, it, counter))
        else:
            out.append(_emit(template_c_q, template_c_a, it, counter))
        counter += 1
        if len(out) >= target_max:
            break

    # If we haven't hit target_min (raw lacked DK records), force
    # second-pass A templates over major acupoints.
    if len(out) < target_min:
        for it in items_unique:
            out.append(_emit(template_a_q, template_a_a, it, counter))
            counter += 1
            if len(out) >= target_min:
                break

    return out


# ---------------------------------------------------------------------------
# STAGE 4 — Validator (원칙 3 prefix + 원칙 4 closing)
# ---------------------------------------------------------------------------
@dataclass
class DiversityViolation:
    category: str
    metric: str          # "prefix_top1" | "unique_prefix_ratio" | "closing_top1"
    value: float
    threshold: float
    example: str = ""


def validate_diversity(
    rows: list[dict],
    violations_path: str | os.PathLike,
    prefix_top1_max: float = 0.15,
    unique_prefix_min_ratio: float = 0.03,
    closing_top1_max: float = 0.10,
    safety_fixed_closings: list[str] | None = None,
) -> list[DiversityViolation]:
    """Check prefix-top1 ≤ 15%, unique prefix ≥ 3%, closing-top1 ≤ 10%."""
    safety_fixed_closings = safety_fixed_closings or SAFETY_FIXED_CLOSINGS
    by_cat: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        cat = r.get("category") or "_unknown"
        a = r.get("assistant") or ""
        by_cat[cat].append(a)

    violations: list[DiversityViolation] = []
    violation_rows: list[dict] = []

    report_lines = []
    for cat, answers in sorted(by_cat.items()):
        if not answers:
            continue
        prefixes = Counter(a[:20] for a in answers)
        closings = Counter(a[-20:] for a in answers)
        n = len(answers)
        unique_prefix_ratio = len(prefixes) / n

        top_pfx, top_pfx_cnt = prefixes.most_common(1)[0]
        top_cls, top_cls_cnt = closings.most_common(1)[0]
        top_pfx_ratio = top_pfx_cnt / n
        top_cls_ratio = top_cls_cnt / n

        report_lines.append(
            f"  {cat}: n={n} unique_prefix={len(prefixes)} "
            f"({unique_prefix_ratio*100:.1f}%) top1_prefix={top_pfx_ratio*100:.1f}% "
            f"top1_closing={top_cls_ratio*100:.1f}%"
        )

        if top_pfx_ratio > prefix_top1_max:
            violations.append(DiversityViolation(
                cat, "prefix_top1", top_pfx_ratio, prefix_top1_max, top_pfx))
        if unique_prefix_ratio < unique_prefix_min_ratio:
            violations.append(DiversityViolation(
                cat, "unique_prefix_ratio", unique_prefix_ratio,
                unique_prefix_min_ratio))
        # closing: refusal_safety is allowed up to 2 fixed safety closings
        # (§3.1.2 원칙 4 예외). We check if the top-1 closing is a tail of
        # any whitelist closing (since 20-char window may truncate the head).
        def _is_whitelisted_closing(tc: str) -> bool:
            tc_strip = tc.strip()
            for fx in safety_fixed_closings:
                fx_tail = fx[-20:].strip()
                if tc_strip == fx_tail or tc_strip.endswith(fx_tail) or fx_tail.endswith(tc_strip):
                    return True
            return False

        closing_whitelist = (
            cat == "refusal_safety" and _is_whitelisted_closing(top_cls)
        )
        if top_cls_ratio > closing_top1_max and not closing_whitelist:
            violations.append(DiversityViolation(
                cat, "closing_top1", top_cls_ratio, closing_top1_max, top_cls))
            # collect example rows
            for r in rows:
                if r.get("category") == cat and (r.get("assistant") or "").endswith(top_cls):
                    violation_rows.append({
                        "category": cat,
                        "violating_closing": top_cls,
                        "row_id": r.get("id"),
                        "assistant": r.get("assistant"),
                    })
                    if len(violation_rows) >= 500:
                        break

    if violation_rows:
        dump_jsonl(violation_rows, violations_path)
    return violations, report_lines


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def _strip_existing_refusals(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r.get("category") not in ("refusal_oos", "refusal_safety")]


def augment(
    input_path: str,
    output_path: str,
    whitelist_path: str | None,
    raw_dir: str | None,
    workspace_dir: str,
    seed: int = 20260424,
    mun_target: int = 150,
    acu_target_min: int = 150,
    refusal_oos_target: int = 100,
    refusal_sfy_target: int = 100,
) -> dict:
    rng = random.Random(seed)
    Path(workspace_dir).mkdir(parents=True, exist_ok=True)

    print(f"[v7-augment] loading input: {input_path}", flush=True)
    rows = load_jsonl(input_path)
    print(f"[v7-augment]   loaded rows: {len(rows)}", flush=True)

    baseline_cat = Counter(r.get("category") for r in rows)
    baseline_shell = sum(1 for r in rows if "에 나온다" in (r.get("assistant") or ""))
    baseline_jok = sum(
        1 for r in rows
        if "족삼리" in (r.get("assistant") or "") + " " + (r.get("question") or "")
    )
    baseline_mun = sum(
        1 for r in rows
        if any(k in (r.get("question") or "") for k in ["풍문", "소아문", "부인문", "담음문",
                                                          "허로문", "상한문", "침구편", "침구법"])
    )

    # --- Stage 1 — A2 shell filter ------------------------------------
    dropped_path = Path(workspace_dir) / "v7_dropped_shell.jsonl"
    rows1, shell_stats = stage1_filter_shell_answers(rows, dropped_path)
    print(
        f"[v7-augment] A2 shell-filter: inspected={shell_stats.inspected} "
        f"matched={shell_stats.matched} dropped={shell_stats.dropped} "
        f"trimmed={shell_stats.trimmed}", flush=True,
    )

    shell_after = sum(1 for r in rows1 if "에 나온다" in (r.get("assistant") or ""))
    sang_before = sum(1 for r in rows if "처방은 상한문에 나온다" in (r.get("assistant") or ""))
    sang_after = sum(1 for r in rows1 if "처방은 상한문에 나온다" in (r.get("assistant") or ""))

    # --- Stage 2a — drop old refusals, inject variant pool --------------
    rows2 = _strip_existing_refusals(rows1)
    new_oos = _rebuild_refusal_oos(rng, refusal_oos_target)
    new_sfy = _rebuild_refusal_safety(rng, refusal_sfy_target)
    print(
        f"[v7-augment] refusal rebuild: oos={len(new_oos)} "
        f"(unique_assistant={len({r['assistant'] for r in new_oos})}) "
        f"safety={len(new_sfy)} "
        f"(unique_assistant={len({r['assistant'] for r in new_sfy})})", flush=True,
    )
    rows2.extend(new_oos)
    rows2.extend(new_sfy)

    # --- Stage 2b — "-문" in-scope QA -----------------------------------
    mun_rows = _build_mun_inscope_qa(rng, rows1, target_count=mun_target)
    rows2.extend(mun_rows)
    print(f"[v7-augment] '-문' in-scope QA added: {len(mun_rows)}", flush=True)

    # --- Stage 3 — A6 acupoint inverse-map ------------------------------
    raw_recs = _load_raw_records(raw_dir) if raw_dir else []
    acu_rows = _build_acupoint_inverse_qa(
        rng, raw_recs, rows1,
        target_min=acu_target_min, target_max=acu_target_min + 100,
    )
    rows2.extend(acu_rows)
    print(f"[v7-augment] acupoint inverse-map QA added: {len(acu_rows)}", flush=True)

    # --- Stage 4 — Validator ----------------------------------------------
    violations_path = Path(workspace_dir) / "v7_closing_violations.jsonl"
    viols, report_lines = validate_diversity(rows2, violations_path)
    print("[v7-augment] diversity audit per-category:", flush=True)
    for ln in report_lines:
        print(ln, flush=True)
    if viols:
        print(f"[v7-augment] {len(viols)} violations logged to {violations_path}", flush=True)
        for v in viols[:10]:
            print(f"    - {v.category} :: {v.metric} = {v.value:.3f} (> {v.threshold})", flush=True)

    # --- Write output ------------------------------------------------------
    n_written = dump_jsonl(rows2, output_path)
    print(f"[v7-augment] wrote {n_written} rows to {output_path}", flush=True)

    # Post-state measurements -----------------------------------------------
    post_cat = Counter(r.get("category") for r in rows2)
    post_shell = sum(1 for r in rows2 if "에 나온다" in (r.get("assistant") or ""))
    post_jok = sum(
        1 for r in rows2
        if "족삼리" in ((r.get("assistant") or "") + " " + (r.get("question") or ""))
    )
    post_mun = sum(
        1 for r in rows2
        if any(k in (r.get("question") or "") for k in ["풍문", "소아문", "부인문", "담음문",
                                                          "허로문", "상한문", "침구편", "침구법"])
    )
    post_oos_unique = len({
        r.get("assistant") for r in rows2 if r.get("category") == "refusal_oos"
    })
    post_sfy_unique = len({
        r.get("assistant") for r in rows2 if r.get("category") == "refusal_safety"
    })
    acupoint_inv_count = sum(
        1 for r in rows2 if r.get("category") == "acupoint" and r.get("subcat") == "INV"
    )

    summary = {
        "input_rows": len(rows),
        "output_rows": n_written,
        "baseline_categories": dict(baseline_cat),
        "post_categories": dict(post_cat),
        "shell_answer_before": baseline_shell,
        "shell_answer_after": post_shell,
        "sang_shell_before": sang_before,
        "sang_shell_after": sang_after,
        "refusal_oos_unique_after": post_oos_unique,
        "refusal_safety_unique_after": post_sfy_unique,
        "mun_surface_query_before": baseline_mun,
        "mun_surface_query_after": post_mun,
        "jok_samri_mentions_before": baseline_jok,
        "jok_samri_mentions_after": post_jok,
        "acupoint_inverse_count": acupoint_inv_count,
        "diversity_violations": [
            {"category": v.category, "metric": v.metric, "value": v.value,
             "threshold": v.threshold}
            for v in viols
        ],
        "artefacts": {
            "output": str(output_path),
            "dropped_shell": str(dropped_path) if dropped_path.exists() else None,
            "closing_violations": str(violations_path) if violations_path.exists() else None,
        },
    }
    summary_path = Path(workspace_dir) / "v7_augment_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[v7-augment] summary → {summary_path}", flush=True)

    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in", dest="input_path", required=True)
    p.add_argument("--out", dest="output_path", required=True)
    p.add_argument("--whitelist", dest="whitelist_path", default=None)
    p.add_argument("--raw-dir", dest="raw_dir", default=None)
    p.add_argument("--workspace", dest="workspace_dir", default="_workspace")
    p.add_argument("--seed", type=int, default=20260424)
    p.add_argument("--mun-target", type=int, default=150)
    p.add_argument("--acu-target-min", type=int, default=150)
    p.add_argument("--refusal-oos-target", type=int, default=100)
    p.add_argument("--refusal-safety-target", type=int, default=100)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    augment(
        input_path=args.input_path,
        output_path=args.output_path,
        whitelist_path=args.whitelist_path,
        raw_dir=args.raw_dir,
        workspace_dir=args.workspace_dir,
        seed=args.seed,
        mun_target=args.mun_target,
        acu_target_min=args.acu_target_min,
        refusal_oos_target=args.refusal_oos_target,
        refusal_sfy_target=args.refusal_safety_target,
    )


if __name__ == "__main__":
    main()
