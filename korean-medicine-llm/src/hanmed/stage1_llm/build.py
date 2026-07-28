"""1차 텍스트 SFT 렌더러 — 고전 번역문을 instruct 형식으로 감싼다.

claudedocs/vlm_plan/03_dataset.md §3.1.

원칙: **답변은 trans_ko verbatim**(앞뒤 공백만 제거). 질문만 메타데이터에서 조립하므로
답변에 창작이 0이고 근거율은 정의상 100%다. ver1 합성 SFT 폐기 사유는 02_decisions §2.3.

산출(기본):
  data/sft/text_train.jsonl   1차 SFT 학습분
  data/sft/text_val.jsonl     검증분
  data/sft/text_replay.jsonl  2차 멀티모달 SFT 의 replay 분(train 의 부분집합)

사용:
  PYTHONPATH=src .venv/bin/python -m hanmed.stage1_llm.build
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
from pathlib import Path

# mediclassics book_id → (한자 서명, 한글 서명). 본초 계열 7책만 사용한다.
BOOKS = {
    190: ("本草綱目", "본초강목"),
    8: ("東醫寶鑑", "동의보감"),
    93: ("鄕藥集成方", "향약집성방"),
    24: ("本草精華", "본초정화"),
    154: ("本經疏證", "본경소증"),
    38: ("食療纂要", "식료찬요"),
    94: ("鄕藥採取月令", "향약채취월령"),
}

Q_TEMPLATES = (
    "{book} 「{path}」의 기록은?",
    "{book} 「{path}」에는 무엇이라 적혀 있는가?",
    "{book}의 {path} 항목을 인용해 달라.",
    "{book} 「{path}」 원문 번역을 알려 줘.",
    "{book} 「{path}」에 대해 문헌은 어떻게 기술하는가?",
)

Q_ABSTAIN = (
    "{sp}의 효능은?",
    "{sp}은(는) 어떤 약효가 있는가?",
    "{sp}의 주치와 효능을 알려 줘.",
)
A_ABSTAIN = "해당 종은 동의보감·본초강목에 기록이 없습니다."

MIN_CHARS = 20


def _pick(opts: tuple[str, ...], key: str) -> str:
    """어투 회전. 내장 hash() 는 PYTHONHASHSEED 에 좌우되므로 쓰지 않는다."""
    h = hashlib.md5(key.encode("utf-8")).digest()
    return opts[h[0] % len(opts)]


def render(rec: dict) -> dict:
    """고전 레코드 1건 → SFT 행. 답변은 번역문 그대로."""
    _, book_ko = BOOKS[rec["book_id"]]
    path = rec["up_path_nm"]
    key = f"{rec['book_id']}/{rec['volume_id']}/{rec['content_seq']}"
    return {
        "conversations": [
            {"from": "human", "value": _pick(Q_TEMPLATES, key).format(book=book_ko, path=path)},
            {"from": "gpt", "value": rec["trans_ko"].strip()},
        ],
        "task": "classic_quote",
        "book_id": rec["book_id"],
        "volume_id": rec["volume_id"],
        "content_seq": rec["content_seq"],
        "up_path_nm": path,
    }


def abstain_row(species_ko: str) -> dict:
    """knowledge_status=unlinked 종 → 유보. KB 상태를 읽은 것이라 창작이 아니다."""
    return {
        "conversations": [
            {"from": "human", "value": _pick(Q_ABSTAIN, species_ko).format(sp=species_ko)},
            {"from": "gpt", "value": A_ABSTAIN},
        ],
        "task": "abstain",
        "species_ko": species_ko,
    }


def usable(rec: dict) -> bool:
    return (rec.get("book_id") in BOOKS
            and bool(rec.get("up_path_nm"))
            and len((rec.get("trans_ko") or "").strip()) >= MIN_CHARS)


def unlinked_species(db_path: str) -> list[str]:
    """근거 사전이 정한 unlinked 종. 게이트의 no_knowledge 판정과 같은 원천을 쓴다.

    종 주석 파일을 직접 읽으면 안 된다 — 독음 링크 확장으로 linked 가 된 종에까지
    「기록이 없습니다」를 학습시키게 되어 게이트와 답이 어긋난다.
    """
    import sqlite3
    con = sqlite3.connect(db_path)
    try:
        return [r[0] for r in con.execute(
            "SELECT species_ko FROM species WHERE knowledge_status!='linked' ORDER BY 1")]
    finally:
        con.close()


def iter_rows(classics_dir: str, db_path: str):
    for book_id in BOOKS:
        for f in sorted(glob.glob(os.path.join(classics_dir, f"book_{book_id:03d}", "vol_*.jsonl"))):
            with open(f, encoding="utf-8") as fh:
                for line in fh:
                    rec = json.loads(line)
                    if usable(rec):
                        yield render(rec)
    if os.path.exists(db_path):
        for sp in unlinked_species(db_path):
            yield abstain_row(sp)
    else:
        print(f"⚠ {db_path} 없음 — abstain 행을 생성하지 않았다. "
              f"먼저 build_ontology 를 돌릴 것.")


def _bucket(row: dict) -> int:
    """행 고유키 → 0..999. 결정론적 split·sampling 용."""
    key = json.dumps(row["conversations"][0], ensure_ascii=False, sort_keys=True)
    return int.from_bytes(hashlib.md5(key.encode("utf-8")).digest()[:2], "big") % 1000


def main() -> int:
    p = argparse.ArgumentParser(description="1차 텍스트 SFT 데이터 생성.")
    p.add_argument("--classics_dir", default="ver1/data/raw/mediclassics_unified")
    p.add_argument("--ontology", default="data/ontology.sqlite",
                   help="abstain 대상 종의 원천. build_ontology 산출물")
    p.add_argument("--out_dir", default="data/sft")
    p.add_argument("--val_permille", type=int, default=10, help="검증 비율 (1/1000 단위)")
    p.add_argument("--replay", type=int, default=4000, help="2차 replay 행 수")
    a = p.parse_args()

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train, val = [], []
    for row in iter_rows(a.classics_dir, a.ontology):
        (val if _bucket(row) < a.val_permille else train).append(row)
    # replay 는 train 의 결정론적 부분집합 (버킷 상위 구간).
    replay = [r for r in train if _bucket(r) < a.val_permille + max(
        1, round(a.replay / max(len(train), 1) * 1000))][:a.replay]

    for name, rows in (("text_train", train), ("text_val", val), ("text_replay", replay)):
        path = out / f"{name}.jsonl"
        with open(path, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        n_abs = sum(1 for r in rows if r["task"] == "abstain")
        print(f"✔ {path}  {len(rows)}행 (abstain {n_abs})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
