"""부위 층화 샘플링 — 앞에서 자르면 부위가 쏠린다(실측 206종 중 76종이 1부위)."""
import pytest

from hanmed_mm.data.build_sft_mm import (NON_PART_MAX_FRAC, _pick, allocate,
                                         render_T1, render_T2)

# allocate 가 받는 행 형식: (species, dataset, part, filename, zip, split)
def _rows(part, n):
    return [("칡", "151", part, f"{part}_{i:04d}.jpg", "z", "train") for i in range(n)]


def test_spreads_across_parts_instead_of_taking_the_first_ones():
    parts = {"leaf": _rows("leaf", 500), "root": _rows("root", 50),
             "flower": _rows("flower", 30)}
    got = allocate(parts, 100)
    assert len(got) == 100
    used = {r[2] for r in got}
    assert used == {"leaf", "root", "flower"}, "부위가 한쪽으로 쏠리면 안 된다"


def test_group_photos_are_capped():
    """군락 사진은 종 식별에 가장 덜 유용하다 — 균등 배분하면 비중이 3배로 뛴다."""
    parts = {"leaf": _rows("leaf", 500), "group": _rows("group", 500)}
    got = allocate(parts, 100)
    n_group = sum(1 for r in got if r[2] == "group")
    assert n_group <= 100 * NON_PART_MAX_FRAC + 1, n_group


def test_falls_back_to_group_when_nothing_else_left():
    """비-군락 사진이 모자라면 상한보다 장수를 우선한다."""
    parts = {"leaf": _rows("leaf", 5), "group": _rows("group", 500)}
    assert len(allocate(parts, 100)) == 100


def test_allocation_is_deterministic():
    parts = {"leaf": _rows("leaf", 200), "root": _rows("root", 200)}
    assert [r[3] for r in allocate(parts, 50)] == [r[3] for r in allocate(parts, 50)]


def test_pick_is_stable_across_processes():
    """내장 hash() 면 PYTHONHASHSEED 마다 문구가 달라져 재현이 안 된다."""
    import os
    import subprocess
    import sys
    code = ("import sys;sys.path.insert(0,'src');"
            "from hanmed_mm.data.build_sft_mm import _pick;"
            "print(_pick(['a','b','c','d'],'091_00005175_fruit.jpg'))")
    outs = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                           env={**os.environ, "PYTHONHASHSEED": s}).stdout
            for s in ("0", "1", "random")}
    assert len(outs) == 1, outs


# ── 독성 답변: 폴백 금지 ────────────────────────────────────────────────────
@pytest.mark.parametrize("ann", [
    {"species_ko": "은조롱", "is_poisonous": False},          # 필드 자체가 없음
    {"species_ko": "은조롱", "is_poisonous": False, "tox_status": None},
    {"species_ko": "은조롱", "is_poisonous": False, "tox_status": "safe"},   # 오타값
])
def test_missing_tox_status_raises_instead_of_answering_safe(ann):
    """tox_status 가 없으면 죽어야 한다. 예전 폴백은 is_poisonous=False 를
    「독초로 분류되지 않은 식물입니다」로 렌더해 미검증 종을 안전하다고 답했다."""
    with pytest.raises(ValueError, match="tox_status"):
        render_T2(ann, "q", "k")


def test_unverified_still_refuses_to_call_it_safe():
    ann = {"species_ko": "은조롱", "tox_status": "unverified"}
    assert "단정" in render_T2(ann, "q", "k")[0]["value"]


# ── 학명: 확정된 것만 단정 ─────────────────────────────────────────────────
@pytest.mark.parametrize("conf", [None, "ambiguous", "unresolved"])
def test_unconfirmed_binomial_is_omitted(conf):
    """은조롱의 Cynanchum wilfordii 는 crosswalk 상 ambiguous(gbif) 다.
    한국명은 의심 대상이 아니므로 식별 자체는 하고 학명만 뺀다."""
    ann = {"species_ko": "은조롱", "scientific_name": "Cynanchum wilfordii",
           "scientific_name_confidence": conf}
    v = render_T1(ann, "q", "k")[0]["value"]
    assert "Cynanchum" not in v and "은조롱" in v


def test_resolved_binomial_is_kept():
    ann = {"species_ko": "참깨", "scientific_name": "Sesamum indicum L.",
           "scientific_name_confidence": "resolved"}
    assert "Sesamum indicum L." in render_T1(ann, "q", "k")[0]["value"]
