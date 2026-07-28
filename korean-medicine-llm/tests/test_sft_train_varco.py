"""S0b — 4bit 적재가 LOCAL_RANK 를 반영한다 (D2).
S5  — image=None(텍스트 전용) 행이 pixel_values 없이 배치로 만들어진다."""
import pytest

from hanmed.training.train import load_kwargs


@pytest.mark.parametrize("load_4bit", [True, False])
def test_load_kwargs_honors_local_rank(monkeypatch, load_4bit):
    monkeypatch.setenv("LOCAL_RANK", "1")
    kw = load_kwargs(load_4bit)
    assert kw["device_map"] == {"": 1}
    assert ("quantization_config" in kw) is load_4bit


def test_load_kwargs_defaults_to_gpu0(monkeypatch):
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    assert load_kwargs(True)["device_map"] == {"": 0}
