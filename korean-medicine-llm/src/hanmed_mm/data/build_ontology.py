"""근거 사전 — data/ontology.sqlite 를 만들고 조회한다. (학습 대상 아님)

claudedocs/vlm_plan/03_dataset.md §3.3.

게이트 1단이 「이 주장에 근거가 있는가」를 결정론적으로 답하려면 출처가 붙은 사실이
필요하다. 여기 들어가는 모든 fact 는 (book_id, vol, seq) 를 갖는다.

소스 4종:
  동의보감    data/safety_kb/classical/*.json   700종. 효능·주치·금기·성미·귀경·독성
  본초강목    book_190                          〔釋名〕→생약명 〔氣味〕→성미
                                                〔主治〕→주치 〔發明〕→효능
  향약집성방  book_093                          〔鄕名〕→생약명
  종 주석     species_annotation.jsonl          학명·약용부위 + species 테이블
  설진 KB     tongue_rule_kb.json               변증 (subject_type=tongue_sign)

**의도적으로 넣지 않은 것**: 본초강목 〔集解〕(산지·형태 서술)·〔附方〕(처방)은 효능 근거가
아니라서 제외했다. 향약집성방 本草부 본문(DH 표제어 아래 ZZ 문단)은 어느 항목인지
표시가 없어 predicate 를 추정해야 하므로 넣지 않았다 — 근거 사전에 추측을 넣지 않는다.

사용:
  PYTHONPATH=src .venv/bin/python -m hanmed_mm.data.build_ontology
"""
from __future__ import annotations

import glob
import json
import os
import re
import sqlite3

from hanmed_mm.data.link_species import build_index, kb_terms, link

DB_PATH = "data/ontology.sqlite"

# 폐쇄집합 10종. 게이트 1단은 이 술어만 판정한다.
PREDICATES = ("효능", "주치", "금기", "성미", "귀경", "독성",
              "학명", "생약명", "약용부위", "변증")

BOOK_DONGUI, BOOK_BONCHO, BOOK_HYANGYAK = 8, 190, 93
# 본초강목 절 표시 → 술어. 나머지 절은 위 docstring 의 사유로 제외.
BONCHO_SECTIONS = {"〔釋名〕": "생약명", "〔氣味〕": "성미",
                   "〔主治〕": "주치", "〔發明〕": "효능"}
_HANJA_RUN = re.compile(r"^[一-鿿]+")

SCHEMA = """
CREATE TABLE fact(
  subject TEXT NOT NULL,        -- 한자 표제어 또는 설진 소견 코드
  subject_type TEXT NOT NULL,   -- herb | tongue_sign
  predicate TEXT NOT NULL,      -- PREDICATES
  object TEXT NOT NULL,
  book_id INTEGER, vol INTEGER, seq INTEGER,
  link_grade TEXT               -- exact_hanja | reading | candidate
);
CREATE TABLE species(
  species_ko TEXT PRIMARY KEY,
  herb_hanja TEXT,                  -- 대표 1개(표시·하위호환). 조회는 species_herb 를 쓴다
  knowledge_status TEXT NOT NULL,   -- linked | ambiguous | unlinked
  tox_status TEXT NOT NULL,         -- toxic | safe_documented | unverified
  has_similar_class INTEGER NOT NULL,
  link_grade TEXT,
  tox_conflict INTEGER NOT NULL     -- 종 주석은 toxic 인데 고전은 무독이라 한 경우
);
-- 종↔한약재는 1:N 이다. 참깨는 白油麻·胡麻 둘 다인데 대표 하나만 보면 胡麻 쪽 9개
-- 사실(익기력 등)이 통째로 안 보여, SFT 가 옳게 인용한 주장을 게이트가 근거 없음으로
-- 되돌린다(거짓 과잉거부).
CREATE TABLE species_herb(
  species_ko TEXT NOT NULL,
  herb_hanja TEXT NOT NULL,
  PRIMARY KEY (species_ko, herb_hanja)
);
CREATE INDEX fact_subject ON fact(subject, predicate);
"""


def headword(path_segment: str) -> str:
    """「附子  《本經》下品」 → 「附子」. 출전 표기가 붙어 정확 일치로는 안 잡힌다."""
    m = _HANJA_RUN.match((path_segment or "").strip())
    return m.group(0) if m else ""


def section_of(up_path: str) -> str:
    """경로에서 〔…〕 절 표시를 찾는다. 부위(根·葉) 세그먼트가 끼어 있어도 잡힌다."""
    for seg in reversed((up_path or "").split(" > ")):
        if seg.startswith("〔"):
            return seg
    return ""


def _fact(subject, predicate, obj, book_id, vol, seq, grade="exact_hanja",
          subject_type="herb"):
    return (subject, subject_type, predicate, obj, book_id, vol, seq, grade)


def donguibogam_facts(kb_dir: str = "data/safety_kb/classical"):
    """동의보감 본초 KB. field_provenance 의 seq 를 source_refs 로 vol 까지 해석."""
    for f in sorted(glob.glob(os.path.join(kb_dir, "*.json"))):
        with open(f, encoding="utf-8") as fh:
            kb = json.load(fh)
        subj = kb["herb_hanja"]
        vol_of = {r["content_seq"]: r["volume_id"] for r in kb.get("source_refs", [])}
        prov = kb.get("field_provenance", {})

        def refs(field):
            out = [int(s.split(":")[1]) for s in prov.get(field, []) if ":" in s]
            return out or [r["content_seq"] for r in kb.get("source_refs", [])[:1]]

        for field in ("효능", "주치", "금기", "귀경"):
            for obj in kb.get(field) or []:
                for seq in refs(field):
                    yield _fact(subj, field, obj, BOOK_DONGUI, vol_of.get(seq), seq)
        sm = kb.get("seongmi") or {}
        for key, pred in (("성질", "성미"), ("맛", "성미"), ("독성", "독성")):
            if sm.get(key):
                for seq in refs("성미"):
                    yield _fact(subj, pred, f"{key}:{sm[key]}", BOOK_DONGUI,
                                vol_of.get(seq), seq)


def _classic_records(classics_dir: str, book_dir: str):
    for f in sorted(glob.glob(os.path.join(classics_dir, book_dir, "vol_*.jsonl"))):
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                yield json.loads(line)


def bonchogangmok_facts(classics_dir: str, subjects: set[str]):
    """본초강목. 표제어는 경로 3번째 세그먼트의 선두 한자 런."""
    for rec in _classic_records(classics_dir, "book_190"):
        path = rec.get("up_path_nm") or ""
        pred = BONCHO_SECTIONS.get(section_of(path))
        trans = (rec.get("trans_ko") or "").strip()
        if not pred or not trans:
            continue
        segs = path.split(" > ")
        subj = headword(segs[2]) if len(segs) > 2 else ""
        if subj in subjects:
            yield _fact(subj, pred, trans, BOOK_BONCHO,
                        rec["volume_id"], rec["content_seq"])


def hyangyak_facts(classics_dir: str, subjects: set[str]):
    """향약집성방 本草부 〔鄕名〕. 「黃精 〔鄕名〕 竹大根」 형태."""
    for rec in _classic_records(classics_dir, "book_093"):
        orig = (rec.get("original") or "").strip()
        if "〔鄕名〕" not in orig:
            continue
        subj = headword(orig)
        if subj in subjects:
            hyangmyeong = orig.split("〔鄕名〕", 1)[1].strip()
            yield _fact(subj, "생약명", hyangmyeong, BOOK_HYANGYAK,
                        rec["volume_id"], rec["content_seq"])


def tongue_facts(path: str = "data/safety_kb/tongue_rule_kb.json"):
    """설진 소견 → 변증. quote_ko 는 번역문의 정확한 부분문자열이다.

    abstain 소견 3종(紫舌·齒痕·瘦舌)은 출전이 없다 → 사실로 넣지 않는다.
    게이트가 근거 부재로 유보하는 것이 정확한 동작이다.
    """
    with open(path, encoding="utf-8") as fh:
        rules = json.load(fh)["rules"]
    for code, r in rules.items():
        ref = r.get("source_ref")
        if r.get("abstain") or not ref:
            continue
        yield _fact(code, "변증", r["byeonjeung"], ref["book_id"], ref["volume_id"],
                    ref["content_seq"], subject_type="tongue_sign")
        yield _fact(code, "변증", r["quote_ko"], ref["book_id"], ref["volume_id"],
                    ref["content_seq"], subject_type="tongue_sign")


def classic_headwords(classics_dir: str) -> tuple[list[str], list[str]]:
    """(향약집성방, 본초강목) 표제어. 동의보감에 없는 종도 독음으로 링크되게 한다."""
    boncho, hyangyak = set(), set()
    for rec in _classic_records(classics_dir, "book_190"):
        path = rec.get("up_path_nm") or ""
        if BONCHO_SECTIONS.get(section_of(path)):
            segs = path.split(" > ")
            if len(segs) > 2 and headword(segs[2]):
                boncho.add(headword(segs[2]))
    for rec in _classic_records(classics_dir, "book_093"):
        orig = (rec.get("original") or "").strip()
        if "〔鄕名〕" in orig and headword(orig):
            hyangyak.add(headword(orig))
    return sorted(hyangyak), sorted(boncho)


def kb_toxicity(kb_dir: str = "data/safety_kb/classical") -> dict[str, str]:
    """한자 표제어 → 고전 독성 표기(무독·유독·유소독)."""
    out = {}
    for f in sorted(glob.glob(os.path.join(kb_dir, "*.json"))):
        with open(f, encoding="utf-8") as fh:
            kb = json.load(fh)
        out[kb["herb_hanja"]] = (kb.get("seongmi") or {}).get("독성") or ""
    return out


def resolve_species(species_path: str, idx: dict[str, list[str]]):
    """종 → (주석, 한자 표제어 전체, link_grade, 유효 상태, 독성충돌). 기존 링크 + 독음 확장.

    표제어는 **리스트 전체**를 낸다. 예전엔 [0] 만 저장해 참깨(白油麻·胡麻)의 胡麻
    사실이 조회에서 사라졌다. 대표 1개는 호출측이 [0] 으로 고른다.

    ambiguous 는 candidate 로 기록만 하고 게이트에서 쓰지 않는다 (§3.3).

    **독음 링크(reading)는 게이트에서 쓰지 않는다.** 27종 전수를 학명과 대조한 결과
    7종이 명백한 오류였다(오류율 26%):

        서양등골나물 Ageratina altissima → 蕎麥(메밀)
        물억새·수크령 → 茅根        묏미나리 Ostericum → 水芹(미나리, Oenanthe)
        매듭풀 Kummerowia → 萹蓄     석잠풀 Stachys → 澤蘭(쉽싸리 Lycopus)
        배초향 → 排草香 (표준 한약명은 藿香)

    원인은 링크가 아니라 그 위에 있다 — `herb_name_label` 자체가 틀린 종이 있다
    (서양등골나물의 라벨이 「교맥」). 근거 사전은 틀린 근거를 출처까지 붙여 내주는 것이
    가장 나쁘므로, 맞는 18종을 잃더라도 candidate 와 같이 기록만 한다.
    학명 대조가 가능해지면(crosswalk 의 공정서 매핑) 되살릴 수 있게 grade 는 남긴다.

    **독성 충돌은 표시만 한다.** 종 주석이 toxic 인데 고전 표제어가 무독인 경우
    (龍葵 처럼 고전과 현대 판단이 실제로 갈리기도 한다). 게이트는 어차피 독성을
    tox_status 라벨로만 답하므로 링크를 끊지 않고 사람이 볼 목록으로 남긴다.
    """
    tox_kb = kb_toxicity()
    with open(species_path, encoding="utf-8") as fh:
        for line in fh:
            a = json.loads(line)
            cls = a.get("classical") or {}
            hanja_list = cls.get("herb_hanja") or []
            status = a["knowledge_status"]
            if status == "linked" and hanja_list:
                grade = "exact_hanja"
            elif status == "ambiguous" and hanja_list:
                grade = "candidate"
            else:
                hit = link(a.get("herb_name_label") or "", idx) or link(a["species_ko"], idx)
                hanja_list, grade = hit, ("reading" if hit else None)
            hanja_list = list(dict.fromkeys(hanja_list or []))
            hw = hanja_list[0] if hanja_list else None
            conflict = bool(hw and a["tox_status"] == "toxic"
                            and "무독" in tox_kb.get(hw, ""))
            # 사람이 확인한 exact_hanja 만 게이트가 쓴다. reading·candidate 는 기록만.
            eff = "linked" if grade == "exact_hanja" else ("ambiguous" if grade else "unlinked")
            yield a, hanja_list, grade, eff, conflict


def build(db_path: str = DB_PATH,
          classics_dir: str = "ver1/data/raw/mediclassics_unified",
          species_path: str = "data/annotations/species_annotation.jsonl") -> dict:
    if os.path.exists(db_path):
        os.remove(db_path)
    con = sqlite3.connect(db_path)
    con.executescript(SCHEMA)

    subjects, stat = set(), {}
    rows = []
    idx = build_index(kb_terms(), *classic_headwords(classics_dir))
    stat["reading_index"] = len(idx)
    links = []
    for a, hanja_list, grade, eff, conflict in resolve_species(species_path, idx):
        rows.append((a["species_ko"], hanja_list[0] if hanja_list else None,
                     eff, a["tox_status"],
                     int(bool(a.get("has_similar_class"))), grade, int(conflict)))
        if eff == "linked":
            subjects.update(hanja_list)
            links += [(a["species_ko"], h) for h in hanja_list]
        # 종 주석에서 오는 사실 — 고전이 아니라 출처는 book_id NULL.
        if a.get("scientific_name"):
            con.execute("INSERT INTO fact VALUES (?,?,?,?,?,?,?,?)",
                        _fact(a["species_ko"], "학명", a["scientific_name"],
                              None, None, None, grade))
        for part in a.get("medicinal_part_ko") or []:
            con.execute("INSERT INTO fact VALUES (?,?,?,?,?,?,?,?)",
                        _fact(a["species_ko"], "약용부위", part, None, None, None, grade))
    con.executemany("INSERT INTO species VALUES (?,?,?,?,?,?,?)", rows)
    con.executemany("INSERT INTO species_herb VALUES (?,?)", links)
    stat["species"] = len(rows)
    stat["linked"] = sum(1 for r in rows if r[2] == "linked")
    stat["species_herb_links"] = len(links)
    # 대표 1개만 저장하던 시절 조용히 버려지던 링크 수. 0 이 아니면 1:N 이 실재한다.
    stat["extra_herb_links"] = len(links) - len({s for s, _ in links})
    stat["tox_conflict"] = sum(1 for r in rows if r[6])

    for name, gen in (("donguibogam", donguibogam_facts()),
                      ("bonchogangmok", bonchogangmok_facts(classics_dir, subjects)),
                      ("hyangyak", hyangyak_facts(classics_dir, subjects)),
                      ("tongue", tongue_facts())):
        facts = list(gen)
        con.executemany("INSERT INTO fact VALUES (?,?,?,?,?,?,?,?)", facts)
        stat[name] = len(facts)
        stat[name + "_subjects"] = len({f[0] for f in facts})
    con.commit()
    con.close()
    return stat


# ---------------------------------------------------------------------------
# 조회 (게이트 1단이 쓰는 진입점)
# ---------------------------------------------------------------------------
def fact_subjects(con, species_ko: str, primary: str | None = None) -> list[str]:
    """종이 근거로 삼는 fact.subject 목록 — 연결된 한약재 **전부** + 종명 자체.

    게이트(gate.verify)와 lookup 이 같은 답을 봐야 해서 여기 한 곳에만 둔다.
    primary 는 species.herb_hanja (구 DB 나 아직 species_herb 가 없는 경우의 안전망).
    """
    linked = [r[0] for r in con.execute(
        "SELECT herb_hanja FROM species_herb WHERE species_ko=? ORDER BY herb_hanja",
        (species_ko,))]
    return [s for s in dict.fromkeys(linked + [primary, species_ko]) if s]


def lookup(species_ko: str, predicate: str | None = None,
           db_path: str = DB_PATH) -> list[dict]:
    """종의 사실을 출처와 함께 반환. 링크가 없거나 candidate 면 빈 리스트.

    반환 fact 마다 book_id·vol·seq 가 붙는다 (학명·약용부위는 고전 출처가 없어 None).
    """
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    try:
        sp = con.execute("SELECT * FROM species WHERE species_ko=?", (species_ko,)).fetchone()
        if sp is None or sp["knowledge_status"] != "linked":
            return []
        subjects = fact_subjects(con, species_ko, sp["herb_hanja"])
        q = ("SELECT * FROM fact WHERE subject IN (%s)"
             % ",".join("?" * len(subjects)))
        params = list(subjects)
        if predicate:
            q += " AND predicate=?"
            params.append(predicate)
        return [dict(r) for r in con.execute(q, params)]
    finally:
        con.close()


def main() -> int:
    import argparse
    p = argparse.ArgumentParser(description="근거 사전 빌드.")
    p.add_argument("--db", default=DB_PATH)
    p.add_argument("--classics_dir", default="ver1/data/raw/mediclassics_unified")
    p.add_argument("--species", default="data/annotations/species_annotation.jsonl")
    a = p.parse_args()
    stat = build(a.db, a.classics_dir, a.species)
    print(f"✔ {a.db}")
    for k, v in stat.items():
        print(f"  {k:24s} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
