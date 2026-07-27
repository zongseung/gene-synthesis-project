"""S2 — 독음 링크. 동음이의는 보류하고 단일 후보만 링크한다."""
from hanmed_mm.data.link_species import build_index, link, normalize_reading


def test_dueum_normalization_matches_both_directions():
    # hanja 는 王不留行 을 「왕불류행」으로 낸다. 라벨은 「왕불유행」.
    assert normalize_reading("왕불류행") == normalize_reading("왕불유행")
    assert normalize_reading("록두") == normalize_reading("녹두")


def test_links_single_candidate():
    idx = build_index(["王不留行", "五加皮"])
    assert link("왕불유행", idx) == ["王不留行"]
    assert link("오가피", idx) == ["五加皮"]


def test_source_priority_breaks_variant_tie():
    # 水芹(동의보감) / 水斳(향약집성방) — 이체자다. 상위 출전으로 확정한다.
    assert link("수근", build_index(["水芹"], ["水斳"])) == ["水芹"]


def test_homonym_is_held():
    # 가자 = 訶子(약재) / 茄子(가지, 채소) — 실측 오탐. 링크하지 않는다.
    idx = build_index(["訶子", "茄子"])
    assert link("가자", idx) == []


def test_unknown_label_returns_empty():
    assert link("없는이름", build_index(["王不留行"])) == []
    assert link("", build_index(["王不留行"])) == []
