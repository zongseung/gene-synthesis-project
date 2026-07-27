"""게이트 1단 — KB 조회로 주장을 판정한다. 결정론적이며 확률이 개입하지 않는다.

claudedocs/vlm_plan/02_decisions.md §2.4.

    supported      근거 존재 → 출처(book, vol, seq)를 붙여 통과
    unsupported    종 지식은 있으나 그 주장은 없음 → 2단(NLI 함의)으로 위임
    no_knowledge   링크 없음 → 효능·주치 주장을 유보(abstain)로 치환

**독성은 1단이 단독으로 책임진다.** species.tox_status 3값 이산 라벨만 답이며,
고전 본문은 독성 판정의 근거로 쓰지 않는다 — 동의보감이 龍葵(까마중)를 무독이라 해도
현대 주석이 toxic 이면 toxic 이다. 최악의 실패 모드가 「독초를 안전하다고 답함」이므로
확률 판정도 문헌 대조도 개입시키지 않는다.
"""
from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field

from hanmed_mm.data.build_ontology import DB_PATH, fact_subjects

# 근거 없이 단정하면 안 되는 술어. no_knowledge 면 유보시킨다.
CLAIM_PREDICATES = ("효능", "주치", "금기", "성미", "귀경", "생약명", "약용부위")
TOX_PREDICATE = "독성"

_PUNCT = re.compile(r"[\s·,.·、，。()（）\[\]〔〕《》「」'\"-]+")


@dataclass
class Verdict:
    status: str                       # supported | unsupported | no_knowledge
    evidence: list[dict] = field(default_factory=list)
    tox_status: str | None = None     # 독성 술어일 때만
    tox_conflict: bool = False        # 고전과 현대 주석의 독성 판단이 갈림
    note: str = ""

    @property
    def sources(self) -> list[tuple]:
        return [(e["book_id"], e["vol"], e["seq"]) for e in self.evidence
                if e["book_id"] is not None]


def _norm(s: str) -> str:
    return _PUNCT.sub("", s or "")


def _species(con, species_ko: str):
    con.row_factory = sqlite3.Row
    return con.execute("SELECT * FROM species WHERE species_ko=?", (species_ko,)).fetchone()


def _facts(con, sp, predicate: str | None):
    # 종당 한약재가 여럿일 수 있다(참깨=白油麻·胡麻) → 대표 하나만 보면 근거를 놓친다.
    subjects = fact_subjects(con, sp["species_ko"], sp["herb_hanja"])
    q = "SELECT * FROM fact WHERE subject IN (%s)" % ",".join("?" * len(subjects))
    params = list(subjects)
    if predicate:
        q += " AND predicate=?"
        params.append(predicate)
    return [dict(r) for r in con.execute(q, params)]


def tox_status(species_ko: str, db_path: str = DB_PATH) -> str:
    """독성 라벨 조회. 미등록 종은 unverified — 안전하다고 말하지 않는다."""
    con = sqlite3.connect(db_path)
    try:
        sp = _species(con, species_ko)
        return sp["tox_status"] if sp else "unverified"
    finally:
        con.close()


def verify_claim(species_ko: str, predicate: str, claim: str,
                 db_path: str = DB_PATH) -> Verdict:
    """종에 대한 한 주장을 판정한다. claim 은 모델 답변의 서술문 하나."""
    con = sqlite3.connect(db_path)
    try:
        sp = _species(con, species_ko)
        if sp is None:
            return Verdict("no_knowledge", note="등록되지 않은 종")

        if predicate == TOX_PREDICATE:
            # 문헌 대조 없이 라벨 그대로. 고전 독성 fact 는 근거로 내주지 않는다.
            return Verdict("supported", tox_status=sp["tox_status"],
                           tox_conflict=bool(sp["tox_conflict"]),
                           note="독성은 1단 라벨 전용")

        if sp["knowledge_status"] != "linked":
            return Verdict("no_knowledge",
                           note=f"link_grade={sp['link_grade']}")

        facts = _facts(con, sp, predicate)
        if not facts:
            return Verdict("unsupported", note="해당 술어의 근거 없음")

        c = _norm(claim)
        hit = [f for f in facts if _norm(f["object"]) and
               (_norm(f["object"]) in c or c in _norm(f["object"]))]
        if hit:
            return Verdict("supported", evidence=hit)
        # 문자열 대조는 의역을 못 잡는다 → 2단이 근거 전문을 받아 판정한다.
        return Verdict("unsupported", evidence=facts,
                       note="문자열 불일치 — 2단 함의 판정 대상")
    finally:
        con.close()


def _eun_neun(word: str) -> str:
    """받침 유무로 조사 선택. 「독미나리은(는)」 같은 표기를 내보내지 않는다."""
    last = ord(word[-1]) if word else 0
    return "은" if 0xAC00 <= last <= 0xD7A3 and (last - 0xAC00) % 28 else "는"


def gate_answer(species_ko: str, predicate: str, claim: str,
                db_path: str = DB_PATH) -> str:
    """판정을 사용자에게 보일 문장으로. 2단 미구현이라 unsupported 는 아직 유보한다."""
    v = verify_claim(species_ko, predicate, claim, db_path)
    if predicate == TOX_PREDICATE:
        sp = f"{species_ko}{_eun_neun(species_ko)}"
        return {"toxic": f"{sp} 독성이 있어 섭취하면 안 됩니다.",
                "safe_documented": f"{sp} 문헌상 식용·약용 기록이 있습니다.",
                }.get(v.tox_status,
                      f"{species_ko}의 독성은 확인되지 않았습니다. 섭취를 삼가십시오.")
    if v.status == "no_knowledge":
        return f"{species_ko}에 대한 문헌 기록이 없어 답변을 유보합니다."
    if v.status == "supported":
        src = ", ".join(f"권{v_}·{s}" for _, v_, s in v.sources[:3])
        return f"{claim} (근거: {src})" if src else claim
    return f"{claim} — 문헌 근거를 확인하지 못했습니다."
