import collections
import importlib.util
from pathlib import Path

from hanmed.bench.run_eval import load_jsonl

_spec = importlib.util.spec_from_file_location(
    "quick_eval_sft_varco", Path("scripts/quick_eval_sft_varco.py")
)
quick_eval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(quick_eval)

parse_species = quick_eval.parse_species
parse_tongue = quick_eval.parse_tongue
parse_tox_status = quick_eval.parse_tox_status
select_quick_rows = quick_eval.select_quick_rows


def test_response_projection():
    assert parse_tongue("혀에서 白苔와 裂紋가 관찰됩니다.") == [
        "baitaishe",
        "liewenshe",
    ]
    assert parse_species(
        "이 식물은 가는잎향유입니다.", ["향유", "가는잎향유"]
    ) == "가는잎향유"
    assert parse_tox_status(
        "안전하다고 단정할 수 없고 독성 여부가 확인되지 않았습니다."
    ) == "unverified"
    assert parse_tox_status("독성이 있어 섭취하면 안 됩니다.") == "toxic"
    assert parse_tox_status("문헌상 식용 기록이 있습니다.") == "safe_documented"


def test_quick_selection_is_deterministic_and_stratified():
    track1 = load_jsonl("data/eval/hanmed_bench/track1_tongue_byeonjeung.jsonl")
    track6 = load_jsonl("data/eval/hanmed_bench/track6_herb_image.jsonl")
    first = select_quick_rows(track1, track6)

    assert first == select_quick_rows(track1, track6)
    tongue, herb = first
    assert len(tongue) == 15
    assert len(herb) == 15
    assert len({row["image"] for row in herb}) == 15
    assert collections.Counter(row["probe_type"] for row in herb) == {
        "species_id": 3,
        "toxicity": 6,
        "efficacy_abstain": 3,
        "answerable_control": 3,
    }
    assert collections.Counter(
        row["gold"] for row in herb if row["probe_type"] == "toxicity"
    ) == {"toxic": 2, "safe_documented": 2, "unverified": 2}
