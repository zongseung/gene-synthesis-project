"""결함 B 검증 — 약초 이미지 벤치(track6)가 학습 이미지와 겹치지 않고,
gold 가 기존 라벨(ontology.sqlite)에서만 온 값인지 확인한다."""
import collections
import json
import os
import sqlite3
import subprocess
import sys

import pytest

BENCH_DIR = "data/eval/hanmed_bench"
TRACK6 = os.path.join(BENCH_DIR, "track6_herb_image.jsonl")
MANIFEST = os.path.join(BENCH_DIR, "manifest.json")
TRAIN = "data/sft/mm_train_resolved.jsonl"
VAL = "data/sft/mm_val_resolved.jsonl"
SHARD_INDEX = "data/shards/herb_shard_index.json"
DB_PATH = "data/ontology.sqlite"

needs_bench = pytest.mark.skipif(
    not all(os.path.exists(p) for p in [TRACK6, TRAIN, VAL, SHARD_INDEX]),
    reason="track6 벤치/학습 산출물 없음 — scripts/build_herb_image_bench.py 먼저 실행",
)
needs_db = pytest.mark.skipif(not os.path.exists(DB_PATH), reason="ontology.sqlite 없음")


def _rows(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def _images(path, key="image"):
    return {d[key] for d in _rows(path) if key in d}


@needs_bench
def test_track6_excludes_train_images():
    assert _images(TRACK6) & _images(TRAIN) == set()


@needs_bench
def test_track6_excludes_val_images():
    assert _images(TRACK6) & _images(VAL) == set()


@needs_bench
def test_all_images_resolve_in_shard_index():
    with open(SHARD_INDEX, encoding="utf-8") as f:
        index = set(json.load(f))
    assert _images(TRACK6) <= index


@needs_bench
def test_toxicity_gold_is_one_of_three_values():
    rows = [r for r in _rows(TRACK6) if r["probe_type"] == "toxicity"]
    assert rows
    assert {r["gold"] for r in rows} <= {"toxic", "safe_documented", "unverified"}


@needs_bench
@needs_db
def test_efficacy_abstain_species_are_all_unlinked():
    rows = [r for r in _rows(TRACK6) if r["probe_type"] == "efficacy_abstain"]
    assert rows
    con = sqlite3.connect(DB_PATH)
    try:
        species = {r["species_ko"] for r in rows}
        for sp in species:
            status = con.execute(
                "SELECT knowledge_status FROM species WHERE species_ko=?", (sp,)
            ).fetchone()[0]
            assert status != "linked", f"{sp} 는 linked 인데 efficacy_abstain 문항이 생성됨"
    finally:
        con.close()


@needs_bench
def test_similar_species_have_more_questions_than_non_similar():
    counts = collections.Counter(r["has_similar_class"] for r in _rows(TRACK6))
    assert counts[True] > counts[False]


@needs_bench
def test_generation_is_deterministic(tmp_path):
    """같은 입력으로 두 번 생성하면 바이트 단위로 동일한 jsonl 이 나와야 한다
    (hashlib + 정렬 기반 선택 — 내장 hash() 미사용)."""
    manifest_src = MANIFEST
    outs = []
    for i in range(2):
        out = tmp_path / f"track6_{i}.jsonl"
        man = tmp_path / f"manifest_{i}.json"
        man.write_text(open(manifest_src, encoding="utf-8").read(), encoding="utf-8")
        subprocess.run(
            [sys.executable, "scripts/build_herb_image_bench.py",
             "--out", str(out), "--manifest", str(man)],
            env={**os.environ, "PYTHONPATH": "src"},
            check=True, capture_output=True, text=True,
        )
        outs.append(out.read_text(encoding="utf-8"))
    assert outs[0] == outs[1]
