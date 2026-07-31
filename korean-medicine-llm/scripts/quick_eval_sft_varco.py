"""완료된 VARCO adapter의 소규모 실제 생성 벤치마크."""

from __future__ import annotations

import hashlib

from hanmed.bench import SIGN_META


HERB_QUOTAS = [
    ("species_id", None, 3),
    ("toxicity", "toxic", 2),
    ("toxicity", "safe_documented", 2),
    ("toxicity", "unverified", 2),
    ("efficacy_abstain", None, 3),
    ("answerable_control", None, 3),
]

SIGN_ALIASES = {
    "baitaishe": ("백태", "白苔"),
    "huangtaishe": ("황태", "黃苔"),
    "heitaishe": ("흑태", "黑苔"),
    "huataishe": ("활태", "滑苔"),
    "botaishe": ("박태", "무태", "薄苔", "無苔"),
    "hongshe": ("홍설", "紅舌"),
    "zishe": ("자설", "紫舌"),
    "pangdashe": ("반대설", "胖大", "舌腫"),
    "liewenshe": ("열문설", "열문", "裂紋"),
    "hongdianshe": ("홍점", "망자", "紅點", "芒刺"),
    "chihenshe": ("치흔설", "치흔", "齒痕"),
    "shoushe": ("수설", "瘦舌"),
    "jiankangshe": ("정상", "健康"),
}


def _norm(text: str) -> str:
    return "".join((text or "").lower().split())


def _rank(row: dict, seed: int) -> str:
    return hashlib.sha256(f"{seed}|{row['id']}".encode()).hexdigest()


def select_quick_rows(
    track1: list[dict], track6: list[dict], seed: int = 42
) -> tuple[list[dict], list[dict]]:
    tongue = sorted(track1, key=lambda row: _rank(row, seed))[:15]
    herb = []
    used_images = set()
    for probe_type, gold, count in HERB_QUOTAS:
        candidates = [
            row
            for row in track6
            if row["probe_type"] == probe_type
            and (gold is None or row["gold"] == gold)
        ]
        chosen = []
        for row in sorted(candidates, key=lambda item: _rank(item, seed)):
            if row["image"] in used_images:
                continue
            chosen.append(row)
            used_images.add(row["image"])
            if len(chosen) == count:
                break
        if len(chosen) != count:
            raise ValueError(f"표본 부족: {probe_type}/{gold} {len(chosen)}/{count}")
        herb.extend(chosen)
    return tongue, herb


def parse_tongue(text: str) -> list[str]:
    normalized = _norm(text)
    # ponytail: 어휘 투영은 부정문을 완전히 해석하지 않는다. 파싱 오류가 지배적이면 JSON 제약 생성으로 교체.
    return [
        sign
        for sign in SIGN_META
        if any(_norm(alias) in normalized for alias in SIGN_ALIASES[sign])
    ]


def parse_species(text: str, species: list[str]) -> str:
    normalized = _norm(text)
    matches = [name for name in species if _norm(name) in normalized]
    return max(matches, key=lambda name: (len(_norm(name)), name), default="")


def parse_tox_status(text: str) -> str:
    normalized = _norm(text)
    for status in ("unverified", "toxic", "safe_documented"):
        if status in normalized:
            return status
    if any(
        phrase in normalized
        for phrase in (
            "확인되지",
            "미확인",
            "단정할수없",
            "판단할수없",
            "알수없",
            "근거가없",
            "불명",
            "모르",
        )
    ):
        return "unverified"
    if any(
        phrase in normalized
        for phrase in ("독성", "독초", "위험", "섭취하면안", "먹으면안", "안전하지않")
    ):
        return "toxic"
    if any(
        phrase in normalized
        for phrase in ("식용기록", "약용기록", "섭취해도안전", "안전합니다", "무독")
    ):
        return "safe_documented"
    return ""
