"""S1 — 1차 텍스트 SFT 렌더러. 답변은 번역문 verbatim (증강 0)."""
import os

import pytest

from hanmed.stage1_llm.build import abstain_row, render

REC = {
    "book_id": 8, "volume_id": 21, "content_seq": 1234, "content_level": "ZZ",
    "up_path_nm": "湯液篇卷之二 > 草部 上 > 人參",
    "original": "性微溫, 味甘, 無毒.\r\n",
    "trans_ko": "성질은 약간 따뜻하고 맛은 달며 독이 없다.\r\n\r\n",
}


def test_answer_is_translation_verbatim():
    row = render(REC)
    assert row["conversations"][1]["value"] == REC["trans_ko"].strip()


def test_source_is_preserved():
    row = render(REC)
    for k in ("book_id", "volume_id", "content_seq", "up_path_nm"):
        assert row[k] == REC[k]


def test_no_image_key():
    assert "image" not in render(REC)


def test_question_mentions_book_and_path():
    q = render(REC)["conversations"][0]["value"]
    assert "동의보감" in q and "人參" in q


def test_tone_rotation_is_deterministic_across_processes():
    """_pick 이 내장 hash() 면 PYTHONHASHSEED 에 따라 값이 바뀐다 → 재현 불가."""
    import json
    import subprocess
    import sys
    code = ("import json,sys;sys.path.insert(0,'src');"
            "from hanmed.stage1_llm.build import render;"
            f"print(render({REC!r})['conversations'][0]['value'])")
    outs = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                           env={**os.environ, "PYTHONHASHSEED": seed}).stdout
            for seed in ("0", "1", "random")}
    assert len(outs) == 1, outs
    del json


def test_abstain_row_states_no_record():
    row = abstain_row("가는잎향유")
    assert "가는잎향유" in row["conversations"][0]["value"]
    assert "기록이 없" in row["conversations"][1]["value"]
    assert "image" not in row
