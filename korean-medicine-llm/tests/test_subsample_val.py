"""검증셋 부분표집 — 소스 비율을 유지하고 결정론적이어야 한다."""
from hanmed_mm.training.sft_train_varco import subsample_val


def _rows(src, n):
    return [{"_src": src, "conversations": [{"from": "human", "value": f"{src}{i}"},
                                            {"from": "gpt", "value": "a"}]}
            for i in range(n)]


ROWS = _rows("herb", 4716) + _rows("tongue", 556) + _rows("text", 813)


def test_keeps_source_ratio():
    got = subsample_val(ROWS, 1500)
    assert abs(len(got) - 1500) <= 3          # 소스별 반올림 오차만 허용
    share = {s: sum(1 for r in got if r["_src"] == s) / len(got) for s in ("herb", "tongue", "text")}
    for s in share:
        orig = sum(1 for r in ROWS if r["_src"] == s) / len(ROWS)
        assert abs(share[s] - orig) < 0.01, (s, share[s], orig)


def test_is_deterministic():
    assert subsample_val(ROWS, 1500) == subsample_val(ROWS, 1500)


def test_no_op_when_already_small():
    small = _rows("herb", 10)
    assert subsample_val(small, 1500) is small


def test_every_source_survives_a_tiny_cap():
    """비율만 곱하면 작은 소스가 0행이 돼 그 소스의 loss 를 못 본다."""
    got = subsample_val(ROWS, 20)
    assert {r["_src"] for r in got} == {"herb", "tongue", "text"}
