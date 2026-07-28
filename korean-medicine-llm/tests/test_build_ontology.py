"""S3 — 근거 사전. 반환 fact 마다 출처(book, vol, seq)가 있어야 한다."""
import os

import pytest

from hanmed.knowledge.build_ontology import DB_PATH, headword, lookup, section_of

needs_db = pytest.mark.skipif(not os.path.exists(DB_PATH), reason="ontology.sqlite 없음")


def test_headword_strips_source_annotation():
    # 본초강목 표제어는 출전 표기가 붙는다 — 선두 한자 런만 표제어다.
    assert headword("附子  《本經》下品") == "附子"
    assert headword("礬石 《本經》上品 校正 並入《海藥》") == "礬石"
    assert headword("黃精 〔鄕名〕 竹大根") == "黃精"


def test_section_is_found_behind_part_segment():
    path = "本草綱目 卷十七 下 > 草之六 > 白附子  《別錄》下品 > 根 > 〔氣味〕"
    assert section_of(path) == "〔氣味〕"
    assert section_of("本草綱目卷一 上 > 序例上") == ""


@needs_db
def test_classical_facts_carry_provenance():
    facts = lookup("가죽나무")
    assert facts
    for f in facts:
        if f["book_id"] is not None:          # 학명·약용부위는 고전 출처가 없다
            assert f["vol"] is not None and f["seq"] is not None


@needs_db
def test_unlinked_species_has_no_facts():
    assert lookup("가는잎향유") == []


@needs_db
def test_toxic_species_with_conflicting_link_is_held():
    """독미나리(맹독)는 독음으로 水芹(미나리)에 걸린다 → 링크 보류, 근거 0."""
    assert lookup("독미나리") == []


@needs_db
def test_multi_hanja_species_returns_every_linked_herb():
    """참깨는 白油麻·胡麻 둘 다다. 대표 하나만 보면 胡麻 쪽 사실(익기력)이 사라져
    SFT 가 옳게 인용한 주장을 게이트가 근거 없음으로 되돌린다(거짓 과잉거부)."""
    facts = lookup("참깨", "효능")
    assert {"白油麻", "胡麻"} <= {f["subject"] for f in facts}
    assert any(f["object"] == "익기력" for f in facts)


@needs_db
def test_claim_from_secondary_herb_is_supported():
    from hanmed_mm.gate.verify import verify_claim
    assert verify_claim("참깨", "효능", "익기력").status == "supported"
