"""S4a — 게이트 1단. unlinked 종의 효능 주장은 no_knowledge, 독성은 라벨 그대로."""
import os

import pytest

from hanmed_mm.data.build_ontology import DB_PATH
from hanmed_mm.gate.verify import gate_answer, tox_status, verify_claim

pytestmark = pytest.mark.skipif(not os.path.exists(DB_PATH), reason="ontology.sqlite 없음")


def test_unlinked_species_efficacy_is_no_knowledge():
    assert verify_claim("가는잎향유", "효능", "청열해독에 씁니다").status == "no_knowledge"


def test_supported_claim_carries_provenance():
    v = verify_claim("가죽나무", "효능", "살감충 작용이 있습니다")
    assert v.status == "supported"
    assert v.sources and all(len(s) == 3 and s[2] for s in v.sources)


def test_unmatched_claim_defers_to_stage2_with_evidence():
    v = verify_claim("가죽나무", "효능", "불면증을 치료합니다")
    assert v.status == "unsupported"
    assert v.evidence, "2단이 판정하려면 근거 전문이 함께 나와야 한다"


def test_toxicity_comes_from_label_not_classics():
    """까마중: 종 주석 toxic, 동의보감 龍葵 무독. 라벨이 이긴다."""
    v = verify_claim("까마중", "독성", "안전하게 먹을 수 있습니다")
    assert v.tox_status == "toxic"
    assert v.tox_conflict is True
    assert "독성이 있어" in gate_answer("까마중", "독성", "안전합니다")


def test_wrongly_linked_toxic_species_abstains():
    """독미나리(맹독)는 독음으로 水芹(미나리)에 걸린다 → 링크 보류 → 유보."""
    assert verify_claim("독미나리", "효능", "해열에 씁니다").status == "no_knowledge"
    assert "유보" in gate_answer("독미나리", "효능", "해열에 씁니다")
    assert tox_status("독미나리") == "toxic"


def test_unknown_species_is_never_called_safe():
    assert tox_status("없는풀") == "unverified"
    assert "삼가" in gate_answer("없는풀", "독성", "먹어도 됩니다")


def test_josa_follows_final_consonant():
    assert gate_answer("독미나리", "독성", "?").startswith("독미나리는")
    assert gate_answer("까마중", "독성", "?").startswith("까마중은")
