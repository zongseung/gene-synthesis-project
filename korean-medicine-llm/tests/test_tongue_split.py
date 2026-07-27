"""결함 A 검증 — 설진 3분할(train/val/test)이 벤치 이미지와 겹치지 않는지 확인.
train=학습, val=학습 eval_loss(소스 train 홀드아웃), test=벤치(track1~3) 전용(소스 val 전량).
세 집합은 이미지 단위로 서로소여야 하고, test 는 벤치 이미지를 전부 포함해야 한다."""
import json
import os
import subprocess
import sys

import pytest

TONGUE_DIR = "data/sft/tongue_sft"
BENCH_DIR = "data/eval/hanmed_bench"
COCO_ROOT = ("/home/user/gene-synthesis-project/shezhen_unzipped/"
             "shezhen datasets/shezhenv3-coco/shezhenv3-coco")

TRAIN_P = os.path.join(TONGUE_DIR, "tongue_sft_train.jsonl")
VAL_P = os.path.join(TONGUE_DIR, "tongue_sft_val.jsonl")
TEST_P = os.path.join(TONGUE_DIR, "tongue_sft_test.jsonl")
TRACK_PATHS = [os.path.join(BENCH_DIR, f"track{i}_{name}.jsonl") for i, name in (
    (1, "tongue_byeonjeung"), (2, "donguibogam_citation"), (3, "abstain"))]

needs_split = pytest.mark.skipif(
    not all(os.path.exists(p) for p in [TRAIN_P, VAL_P, TEST_P, *TRACK_PATHS]),
    reason="설진 3분할/벤치 산출물 없음",
)


def _images(path):
    # track3_abstain 의 일부 행(위험 프로브)은 image 필드가 없다 — 있는 것만 모은다.
    out = set()
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            if "image" in d:
                out.add(d["image"])
    return out


def _bench_images():
    out = set()
    for p in TRACK_PATHS:
        out |= _images(p)
    return out


@needs_split
def test_val_holdout_excludes_bench_images():
    assert _images(VAL_P) & _bench_images() == set()


@needs_split
def test_train_excludes_bench_images():
    assert _images(TRAIN_P) & _bench_images() == set()


@needs_split
def test_test_file_covers_track1():
    assert _images(TRACK_PATHS[0]) <= _images(TEST_P)


@needs_split
def test_train_val_test_are_pairwise_disjoint():
    train, val, test = _images(TRAIN_P), _images(VAL_P), _images(TEST_P)
    assert train & val == set()
    assert train & test == set()
    assert val & test == set()


needs_coco = pytest.mark.skipif(not os.path.exists(COCO_ROOT), reason="원본 shezhenv3-coco 없음")


@needs_coco
def test_holdout_selection_is_deterministic(tmp_path):
    """내장 hash() 였다면 PYTHONHASHSEED 마다 val 홀드아웃이 달라졌을 것.
    md5(file_name) hexdigest 정렬을 쓰므로 해시시드가 달라도 같은 556장이 나와야 한다."""
    base_cmd = [
        sys.executable, "-m", "hanmed_mm.data.gen_tongue_sft",
        "--coco_root", COCO_ROOT,
        "--rule_kb", "data/safety_kb/tongue_rule_kb.json",
        "--manifest", "data/sft/tongue_sft_build_manifest.json",
        "--holdout_val", "556",
    ]
    out_a, out_b = tmp_path / "a", tmp_path / "b"
    out_a.mkdir(); out_b.mkdir()
    for out_dir, seed in ((out_a, "0"), (out_b, "12345")):
        subprocess.run(
            base_cmd + ["--out", str(out_dir)],
            env={**os.environ, "PYTHONHASHSEED": seed, "PYTHONPATH": "src"},
            check=True, capture_output=True, text=True,
        )
    assert _images(out_a / "tongue_sft_val.jsonl") == _images(out_b / "tongue_sft_val.jsonl")
