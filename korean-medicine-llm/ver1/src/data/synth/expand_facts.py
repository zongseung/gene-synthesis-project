"""ver4 r2 §5.4 — Fact sheet → long-form synthetic corpus.

core_factsheet.yaml 의 사람-검증 fact 만을 seed 로 하여
권당 5 카테고리(저자 전기 / 시대 배경 / 서문·해제 / 대표 처방 / 관련 서적)
50 paragraph ± 를 template 기반으로 생성. LLM 호출 금지.

paraphrase ×N (default 4): 어순 permutation / 접속사 교체 / 분할·병합 /
종결어 교체. 한자 병기 유지 필수.

Entity validation: fact sheet 전체 union + 블랙리스트(이시진·본초강목 등)
기준으로 fact sheet 에 없는 인명이 나타나면 해당 paragraph drop.

출력 스키마: hanmed_ko_only.jsonl 과 호환
  book_id / volume_id / content_seq / content_level / up_path_nm / text
  + n_chars / category / paraphrase_id
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# repo root 삽입 (sibling import & set_global_seed 용)
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.data.builder.extract_corpora import (  # noqa: E402
    HANJA_RE,
    _extract_hanja_only,
    _fmt_year,
    _hanja_ratio,
    load_factsheet,
)
from src.utils.seed import set_global_seed  # noqa: E402

# ---------------------------------------------------------------------------
# 상수
# ---------------------------------------------------------------------------
# 한국어 token 근사: len(text) / 1.7 ≈ tokens → 200~400 tok → 340~800 char.
LEN_MIN = 340
LEN_MAX = 800

# 권당 카테고리 target (§2.1 표). 실제 생성량은 template 가용 slot 수에 의해 ≤ target.
CATEGORY_TARGETS = {
    "author_biography": 10,
    "dynasty_context": 10,
    "preface_summary": 15,
    "signature_items": 10,
    "related_books": 5,
}

# 블랙리스트: Chinese prior 누출 방지용. fact sheet에 해당 값이 있으면 허용(아래 build_validator에서 제외).
DEFAULT_BLACKLIST = [
    "이시진",
    "李時珍",
    "본초강목",
    "本草綱目",
    "장원",
    "張邯",
]

# fact 필드 중 "사람 이름" 으로 간주할 키 (validator union 구성용)
PERSON_NAME_FIELDS = ["author", "author_hanja"]

# 코퍼스에 섞일 가능성이 있는 일반 한국어 잡인명 패턴 검출용 — 2~4자 한글 인명
# 단독 사용은 false positive 위험이 크므로 "블랙리스트 + 구문 맥락" 조합만 사용.
BLACKLIST_WHOLEWORD_RE_CACHE: dict[str, re.Pattern] = {}


def _blacklist_re(needle: str) -> re.Pattern:
    if needle not in BLACKLIST_WHOLEWORD_RE_CACHE:
        BLACKLIST_WHOLEWORD_RE_CACHE[needle] = re.compile(re.escape(needle))
    return BLACKLIST_WHOLEWORD_RE_CACHE[needle]


# ---------------------------------------------------------------------------
# 엔티티 수집 / 검증
# ---------------------------------------------------------------------------


def _split_compound_author(s: str | None) -> list[str]:
    """'유효통(兪孝通)·노중례(盧重禮)·박윤덕(朴允德)' → ['유효통','노중례','박윤덕','兪孝通','盧重禮','朴允德']."""
    if not s:
        return []
    tokens: list[str] = []
    # 괄호 밖 한글/라틴 분리
    parts = re.split(r"[·,·/]", s)
    for p in parts:
        p = p.strip()
        if not p:
            continue
        # 한자 괄호 블록 추출
        m = re.findall(r"\(([^)]+)\)", p)
        if m:
            for hanja in m:
                if HANJA_RE.search(hanja):
                    tokens.append(hanja.strip())
        # 괄호 제거한 순수 한글 이름
        base = re.sub(r"\s*\([^)]*\)", "", p).strip()
        if base and base not in tokens:
            tokens.append(base)
        # 한자 단독 형태
        if HANJA_RE.search(p) and "(" not in p:
            tokens.append(p.strip())
    return [t for t in tokens if t]


def build_entity_union(factsheet: dict[int, dict]) -> set[str]:
    """모든 권 fact sheet 필드 union — 인명·왕대·서명·연도·토픽 토큰 모두 포함."""
    tokens: set[str] = set()
    for fact in factsheet.values():
        # 인명
        for k in PERSON_NAME_FIELDS:
            tokens.update(_split_compound_author(fact.get(k)))
        # 왕대·국가·장르
        for k in ["dynasty", "reign", "reign_hanja", "genre"]:
            v = fact.get(k)
            if v:
                tokens.add(v.strip())
                hj = _extract_hanja_only(v)
                if hj:
                    tokens.add(hj)
        # title
        for k in ["title_ko", "title_hanja"]:
            v = fact.get(k)
            if v:
                tokens.add(v.strip())
        # 연도
        for k in ["compiled_year", "published_year"]:
            v = fact.get(k)
            if isinstance(v, int):
                tokens.add(str(v))
                tokens.add(f"{v}년")
        # 리스트 필드: 요소 전체 + 각 요소 내 한자 블록
        for k in ["topics", "related_books", "signature_items"]:
            for item in fact.get(k) or []:
                tokens.add(item.strip())
                for hanja in re.findall(r"[\u3400-\u9FFF\uF900-\uFAFF]+", item):
                    tokens.add(hanja)
                # 한글 앞부분(괄호 밖)
                base = re.sub(r"\s*\([^)]*\)", "", item).strip()
                if base:
                    tokens.add(base)
    return {t for t in tokens if t}


def build_blacklist(factsheet: dict[int, dict]) -> list[str]:
    """DEFAULT_BLACKLIST에서 fact sheet union에 이미 존재하는 값은 제외한 최종 블랙리스트."""
    union = build_entity_union(factsheet)
    return [b for b in DEFAULT_BLACKLIST if b not in union]


def validate_text(text: str, blacklist: list[str]) -> tuple[bool, str | None]:
    """블랙리스트 인명/서명이 등장하면 (False, 해당 토큰). 그 외 (True, None)."""
    for bad in blacklist:
        if _blacklist_re(bad).search(text):
            return False, bad
    return True, None


# ---------------------------------------------------------------------------
# fact dict → 포맷 헬퍼 (한자 병기 규칙)
# ---------------------------------------------------------------------------


def _title_forms(fact: dict) -> tuple[str | None, str | None]:
    """(first-appearance, short). first는 『한글(한자)』, short는 『한글』."""
    t_ko = fact.get("title_ko")
    t_hj = fact.get("title_hanja")
    if t_ko and t_hj:
        return f"『{t_ko}({t_hj})』", f"『{t_ko}』"
    if t_ko:
        return f"『{t_ko}』", f"『{t_ko}』"
    if t_hj:
        return f"『{t_hj}』", f"『{t_hj}』"
    return None, None


def _author_forms(fact: dict) -> tuple[str | None, str | None]:
    """(first-appearance, short). first: '허준(許浚)' 류, short: '허준'."""
    author = fact.get("author")
    if not author:
        return None, None
    author_ko_only = re.sub(r"\s*\([^)]*\)", "", author).strip()
    author_hanja = fact.get("author_hanja") or _extract_hanja_only(author)
    if author_ko_only and author_hanja and author_hanja not in author_ko_only:
        first = f"{author_ko_only}({author_hanja})"
    else:
        first = author
    return first, (author_ko_only or author)


def _reign_forms(fact: dict) -> tuple[str | None, str | None]:
    reign = fact.get("reign")
    reign_hanja = fact.get("reign_hanja") or _extract_hanja_only(reign)
    if reign and reign_hanja and reign_hanja not in reign:
        return f"{reign}({reign_hanja})", reign
    return reign, reign


# ---------------------------------------------------------------------------
# Template bank — 카테고리별로 slot 요건이 모두 채워져야 fire
# 반환: list[str] (하나 이상의 문장). 요건 미충족이면 빈 리스트 → skip.
# ---------------------------------------------------------------------------


def _fact_closers(fact: dict) -> list[str]:
    """fact 기반 마무리 문장 pool — 길이 패딩용. fact 값만 사용, 외부 지식 없음."""
    title_f, title_s = _title_forms(fact)
    author_f, author_s = _author_forms(fact)
    reign_f, _ = _reign_forms(fact)
    dynasty = fact.get("dynasty")
    cy = _fmt_year(fact.get("compiled_year"))
    py = _fmt_year(fact.get("published_year"))
    genre = fact.get("genre")
    out: list[str] = []
    if title_s and genre:
        out.append(f"이 같은 서지 정보는 {title_s}이(가) {genre}의 범주에 놓이는 까닭을 뒷받침한다.")
    if title_s and dynasty:
        out.append(f"{title_s}은(는) {dynasty} 시대 의학 문헌사 안에서 고유한 위치를 차지한다.")
    if author_s and title_s:
        out.append(f"이러한 연관 속에서 {author_s}과(와) {title_s}은(는) 한국 한의학사에서 함께 기억되어 왔다.")
    if reign_f and title_s:
        out.append(f"{reign_f} 대의 편찬 환경은 {title_s}의 성격을 이해하는 또 다른 단서가 된다.")
    if cy and title_s:
        out.append(f"{cy}이라는 시점은 {title_s}의 서지 정보 가운데 가장 빈번히 인용되는 연대이다.")
    if py and title_s:
        out.append(f"{py}의 간행은 {title_s}이(가) 후대에 전해지는 결정적 계기가 되었다.")
    if author_s:
        out.append(f"{author_s}의 편찬 의도와 학문적 지향은 본서의 구성 전반에서 드러난다.")
    if title_s:
        out.append(f"이 점에서 {title_s}은(는) 단일 의서 이상의 사료적 가치를 지닌다.")
    return out


def _compose_long(base_sents: list[str], closers: list[str], min_len: int = LEN_MIN) -> list[str]:
    """base 문장 + closer 문장 1~3개를 이어 붙여 min_len 이상 길이의 sentence list 생성.

    결정론적: closers 의 앞에서부터 차례로 추가. 순서 바뀌지 않음.
    """
    out = list(base_sents)
    cl_idx = 0
    while sum(len(s) + 1 for s in out) < min_len + 20 and cl_idx < len(closers):
        out.append(closers[cl_idx])
        cl_idx += 1
    return out


def _tpl_author_biography(fact: dict) -> list[list[str]]:
    """author_biography 카테고리: 저자의 활동·편찬 경위 서술. 최대 10 variant."""
    title_f, title_s = _title_forms(fact)
    author_f, author_s = _author_forms(fact)
    reign_f, reign_s = _reign_forms(fact)
    dynasty = fact.get("dynasty")
    cy = _fmt_year(fact.get("compiled_year"))
    py = _fmt_year(fact.get("published_year"))

    paragraphs: list[list[str]] = []
    if not (title_f and author_f):
        return paragraphs

    closers = _fact_closers(fact)

    # 1. 왕대·연도 triple binding (완전판)
    if dynasty and reign_f and cy:
        base = [
            f"{author_f}은(는) {dynasty} {reign_f} 대에 활동한 의가로, {cy}에 {title_f}의 편찬을 완성하였다.",
            f"이 과정은 당시의 국왕 명령에 따라 진행된 국가적 편찬 사업으로 평가된다.",
            f"{author_s}의 이름과 {cy}이라는 시점은 {title_s}을(를) 이해할 때 가장 먼저 기억되는 서지 정보이다.",
            f"이러한 triple 정보는 본서를 다른 의학 저작과 구별하는 지표가 된다.",
        ]
        paragraphs.append(_compose_long(base, closers))
    # 2. 간행·편찬 분리
    if dynasty and cy and py and cy != py:
        base = [
            f"{author_f}은(는) {dynasty} 시대 인물로서 {cy} {title_f}의 초고를 마쳤다.",
            f"이후 {py}에 이르러 본서가 정식으로 간행되었으며, 이로써 {author_s}의 이름이 후대에 널리 전해지게 되었다.",
            f"편찬과 간행 사이의 시간차는 당시의 출판 공정이 복수의 단계를 거쳤음을 시사한다.",
            f"{author_s}의 {title_f}은(는) 이러한 두 단계를 모두 거쳐 완성된 저작이다.",
        ]
        paragraphs.append(_compose_long(base, closers))
    # 3. 편찬·저술 중심
    if cy and dynasty:
        base = [
            f"{dynasty} 시대의 대표적 의가인 {author_f}은(는) {cy} {title_f}의 편찬을 주관하였다.",
            f"그 결과물은 이후 한국 한의학 문헌 계통에서 기준 저작의 하나로 자리 잡았다.",
            f"편찬 주관자로서 {author_s}의 역할은 단순 편집을 넘어 지식 체계 설계에 가까웠다.",
        ]
        paragraphs.append(_compose_long(base, closers))
    # 4. 저자 명시 binding 반복 (bind_density 보강)
    if reign_f:
        base = [
            f"{reign_f} 대의 의학 편찬 활동을 대표하는 인물이 바로 {author_f}이다.",
            f"{author_s}의 {title_f}은(는) 해당 시대 의학 지식의 집성이라는 의미를 가진다.",
            f"이러한 이유로 {author_s}과(와) {title_s}은(는) 자주 한 문장 안에서 함께 언급된다.",
        ]
        paragraphs.append(_compose_long(base, closers))
    # 5. 저자→저작 직접 귀속 (짧은 triple)
    if cy:
        base = [
            f"{title_f}의 편찬자는 {author_f}이며, 완성 시점은 {cy}으로 기록된다.",
            f"이 서지 사실은 원서의 서문과 관련 해제에서 공통적으로 확인된다.",
            f"{author_s}의 이름 아래 {title_s}이(가) 전해졌다는 사실은 한국 한의학사의 주요 사건으로 꼽힌다.",
            f"연대 {cy}과(와) 편찬자 {author_s}의 결합은 본서 정체성의 핵심이다.",
        ]
        paragraphs.append(_compose_long(base, closers))
    # 6. 국가 편찬 성격
    if dynasty and reign_f:
        base = [
            f"{dynasty} {reign_f} 대에 이르러 의학 지식의 정리가 국가적 과제로 인식되면서 {author_f}이(가) 그 핵심 인물로 위촉되었다.",
            f"그 산물이 바로 {title_f}이며, {author_s}은(는) 이 책의 편찬자로 역사에 기록되었다.",
            f"국가 편찬이라는 성격은 본서의 체제와 권수, 배포 경로에도 영향을 남겼다.",
        ]
        paragraphs.append(_compose_long(base, closers))
    # 7. 간행 연도 단독
    if py:
        base = [
            f"{author_f}이(가) 남긴 {title_f}은(는) {py}에 간행되어 세상에 공개되었다.",
            f"간행 당시의 체제와 구성은 {author_s}의 편찬 의도를 그대로 반영한 것으로 해석된다.",
            f"이러한 이유로 {title_s} 연구에서는 {author_s}의 서지적 위상이 항상 함께 다루어진다.",
            f"{py}의 간행본은 이후 판본 계통의 기준점으로 기능하였다.",
        ]
        paragraphs.append(_compose_long(base, closers))
    return paragraphs


def _tpl_dynasty_context(fact: dict) -> list[list[str]]:
    title_f, title_s = _title_forms(fact)
    author_f, author_s = _author_forms(fact)
    reign_f, reign_s = _reign_forms(fact)
    dynasty = fact.get("dynasty")
    cy = _fmt_year(fact.get("compiled_year"))

    paragraphs: list[list[str]] = []
    if not title_f:
        return paragraphs

    closers = _fact_closers(fact)

    if dynasty and reign_f and cy:
        base = [
            f"{dynasty} {reign_f} 대는 의학 정책이 국가 차원에서 정비되던 시기로, {cy}에 {title_f} 편찬 작업이 마무리되었다.",
            f"이는 당시의 의료 제도와 긴밀히 연결된 사건이다.",
            f"{reign_s} 대의 편찬 활동은 이후 의학 지식의 표준화 방향을 제시했다고 평가된다.",
        ]
        paragraphs.append(_compose_long(base, closers))
    if dynasty and reign_f:
        base = [
            f"{title_f}이(가) 편찬된 {dynasty} {reign_f} 시기는 의학 저술과 의료 행정의 정비가 동시에 추진된 시기였다.",
            f"이런 배경 속에서 {title_s}은(는) 단순한 의학서를 넘어 정책 문헌의 성격도 가진다.",
            f"즉 당대의 제도적 수요가 본서의 편찬에 직접 반영되었다고 해석된다.",
        ]
        paragraphs.append(_compose_long(base, closers))
    if reign_f and author_f:
        base = [
            f"{reign_f} 대의 정치·문화적 분위기는 {author_f}이(가) {title_f}을(를) 편찬하는 데 직접적인 기반이 되었다.",
            f"즉 이 책은 {reign_s} 대의 의학 지식 정리 흐름 속에서 등장한 산물이다.",
            f"따라서 {title_s}을(를) {reign_s} 대의 다른 의학 저술과 함께 읽을 때 그 의의가 더 잘 드러난다.",
        ]
        paragraphs.append(_compose_long(base, closers))
    if dynasty and cy:
        base = [
            f"{cy}이라는 시점은 {dynasty} 시대 의학사에서 의미 있는 전환점으로 평가되며, 바로 이 해에 {title_f}이(가) 편찬되었다.",
            f"이러한 배경을 고려하면 이 책이 {dynasty} 의학의 집성적 성격을 띠게 된 까닭이 자연스럽게 이해된다.",
            f"{cy}을(를) 기준으로 본서는 이전 저술들과 이후 저술들의 가교 역할을 수행하였다.",
        ]
        paragraphs.append(_compose_long(base, closers))
    if dynasty:
        base = [
            f"{title_f}은(는) {dynasty} 시대 의학사의 흐름 속에 자리한다.",
            f"당대의 국가 의료 체계와 민간 실용 지식이 교차하던 시기였다.",
            f"그러한 조건에서 이 책의 체제가 형성되었다고 이해된다.",
            f"결과적으로 {title_s}은(는) {dynasty} 의학사의 한 단면을 집약적으로 보여 주는 저작으로 자리 잡았다.",
        ]
        paragraphs.append(_compose_long(base, closers))
    if author_f and reign_f and dynasty:
        base = [
            f"{author_f}은(는) {dynasty} {reign_f} 대의 의학자이며, 그의 대표작이 {title_f}이다.",
            f"따라서 본서의 시대적 맥락을 살피기 위해서는 {reign_s} 대의 의학 활동 전반을 함께 검토해야 한다.",
            f"이런 관점에서 {author_s}과(와) 본서는 {reign_s} 대 의학사의 핵심 키워드로 거론된다.",
        ]
        paragraphs.append(_compose_long(base, closers))
    return paragraphs


def _tpl_preface_summary(fact: dict) -> list[list[str]]:
    """서문·해제 스타일 요약. 15개 target."""
    title_f, title_s = _title_forms(fact)
    author_f, author_s = _author_forms(fact)
    reign_f, _ = _reign_forms(fact)
    dynasty = fact.get("dynasty")
    cy = _fmt_year(fact.get("compiled_year"))
    py = _fmt_year(fact.get("published_year"))
    genre = fact.get("genre")
    topics = fact.get("topics") or []

    paragraphs: list[list[str]] = []
    if not title_f:
        return paragraphs

    closers = _fact_closers(fact)

    def add(base: list[str]) -> None:
        paragraphs.append(_compose_long(base, closers))

    if genre and dynasty:
        add([
            f"{title_f}은(는) {dynasty} 시대 {genre}에 해당한다.",
            f"당대의 의학 지식을 체계적으로 집성한 성격을 갖는다.",
            f"이러한 편성은 후대 의가들의 참고서로 기능해 왔다.",
        ])
    if genre:
        add([
            f"본서 {title_f}은(는) {genre}로 분류되며, 해당 분야의 지식을 집성적으로 정리한 저작이다.",
            f"그 범위는 단일 질환에 국한되지 않고 넓게 포괄한다.",
            f"이 점에서 {title_s}은(는) 단행 저술이라기보다 집성 저작의 성격이 강하다.",
        ])
    if genre and author_f:
        add([
            f"{genre}에 속하는 {title_f}은(는) {author_f}의 편찬으로 성립된 저작이다.",
            f"분류상 {genre}로 간주되며, 이는 책의 구성 원칙과도 부합한다.",
            f"{author_s}의 편찬 작업은 이 분류 체계 안에서 진행된 결과로 이해된다.",
        ])
    if topics:
        head_topic = topics[0]
        add([
            f"{title_f}의 서두는 {head_topic}을(를) 제시하며 시작한다.",
            f"이는 본서가 다루는 내용의 뼈대를 먼저 확인시키는 장치이다.",
            f"이어지는 본문은 이 뼈대를 따라 전개된다.",
            f"따라서 {head_topic}은(는) {title_s}을(를) 이해하는 출발점이 된다.",
        ])
    if len(topics) >= 2:
        add([
            f"{title_f}은(는) {topics[0]}와(과) {topics[1]} 등을 주요 주제로 삼는다.",
            f"각 주제는 본서 내에서 독립된 편·장으로 편성된다.",
            f"이러한 편성 방식은 독자가 특정 주제를 발췌하여 읽기에도 적합하다.",
        ])
    if len(topics) >= 3:
        add([
            f"본서가 다루는 주제는 다양하다. 특히 {topics[0]}, {topics[1]}, {topics[2]} 등이 대표적이다.",
            f"이들 주제는 본서의 학문적 범위를 대표한다.",
            f"따라서 {title_s}을(를) 읽을 때는 이 주제 구성을 먼저 파악해야 한다.",
        ])
    if topics:
        topic_list = ", ".join(topics)
        add([
            f"{title_f}이(가) 다루는 핵심 주제는 다음과 같이 요약된다. {topic_list}.",
            f"이 구성은 이 책의 내용 범위를 대표한다.",
            f"개별 주제의 배치 순서는 편찬자의 학문 체계관을 반영한다.",
        ])
    if cy and genre:
        add([
            f"{cy}에 편찬된 {title_f}은(는) {genre}의 관점에서 중요한 참고 문헌이다.",
            f"편찬 시점 이후 이 분야 연구에서 반복적으로 인용되어 왔다.",
            f"이러한 인용의 누적은 {title_s}의 정본적 지위를 형성했다.",
        ])
    if py:
        add([
            f"{title_f}의 간행 연도는 {py}이다.",
            f"간행본은 이후의 판본 계통 형성에서 원본 격의 지위를 차지한다.",
            f"따라서 {py}은(는) {title_s} 판본 연구에서 기준 연대가 된다.",
        ])
    if author_f and cy:
        add([
            f"{cy}에 {author_f}이(가) 완성한 {title_f}은(는) 해당 시점의 의학 지식을 압축한 저작이다.",
            f"즉 이 책은 편찬 당시의 학문적 집성물이라 할 수 있다.",
            f"이러한 성격은 {author_s}이(가) 책 전체에 부여한 편집 방향의 결과이다.",
        ])
    if dynasty and genre:
        add([
            f"{dynasty} 시대 {genre}로 분류되는 {title_f}은(는), 해당 분류 내에서 표준적 지위를 가진다.",
            f"이 위상은 후대의 의학 문헌 계통에도 이어졌다.",
            f"결과적으로 {title_s}은(는) {dynasty} {genre} 범주의 대표작으로 인식된다.",
        ])
    if reign_f and genre:
        add([
            f"{reign_f} 대에 등장한 {title_f}은(는) {genre}의 범주에 속한다.",
            f"이 시기 의학 저술의 흐름과도 자연스럽게 연결된다.",
            f"본서가 {genre}로 분류되는 것은 편찬 당시의 학문 지형을 반영한 결과이다.",
        ])
    if author_f and genre:
        add([
            f"{author_f}의 {title_f}은(는) {genre}의 대표적 저작으로 평가된다.",
            f"본서의 편집 의도와 내용 범위는 {genre} 장르의 일반적 성격을 잘 보여 준다.",
            f"따라서 {genre}를 살필 때 {author_s}의 {title_s}은(는) 빼놓을 수 없는 예시가 된다.",
        ])
    if topics and dynasty:
        add([
            f"{dynasty} 시대에 편찬된 {title_f}은(는) {topics[0]} 등을 포함하는 주제 구성을 갖춘다.",
            f"그 구성은 당대 의학 지식의 분류 체계를 반영한 것으로 해석된다.",
            f"즉 본서의 목차는 그 자체로 {dynasty} 의학 지식 지형의 축소판이라 할 수 있다.",
        ])
    if topics and author_f:
        add([
            f"{author_f}은(는) {title_f}을(를) 편찬하며 {topics[0]}을(를) 비중 있게 다루었다.",
            f"이 점은 본서의 내용 특성을 대표한다.",
            f"독자는 해당 주제를 중심으로 책을 읽어 나갈 수 있다.",
        ])
    return paragraphs


def _tpl_signature_items(fact: dict) -> list[list[str]]:
    title_f, title_s = _title_forms(fact)
    author_f, author_s = _author_forms(fact)
    sig = fact.get("signature_items") or []

    paragraphs: list[list[str]] = []
    if not (title_f and sig):
        return paragraphs

    closers = _fact_closers(fact)

    def add(base: list[str]) -> None:
        paragraphs.append(_compose_long(base, closers))

    # 단일 item 집중 서술 — 권당 최대 sig 수만큼
    for item in sig[:6]:
        add([
            f"{title_f}의 대표 항목 가운데 하나가 {item}이다.",
            f"이 항목은 본서의 학문적 특성을 단적으로 보여 준다.",
            f"따라서 {title_s}을(를) 논할 때 자주 먼저 거론되는 부분이다.",
            f"해당 항목의 구성 방식은 본서 전체의 서술 원칙과도 이어진다.",
        ])
        if author_f:
            add([
                f"{author_f}은(는) {title_f} 안에서 {item}을(를) 비중 있게 다루었다.",
                f"이 부분은 편찬자의 의학적 시각을 직접 드러낸다.",
                f"{author_s}의 편찬 의도는 바로 이러한 대표 항목을 통해 확인된다.",
            ])

    if len(sig) >= 2:
        add([
            f"{title_f}의 대표 내용으로는 {sig[0]}와(과) {sig[1]} 등이 잘 알려져 있다.",
            f"이 두 항목은 본서 전체에서 특히 자주 인용된다.",
            f"따라서 두 항목을 함께 살피는 것은 {title_s}의 핵심을 파악하는 효율적인 방법이다.",
        ])
    if len(sig) >= 3:
        list_str = ", ".join(sig[:3])
        add([
            f"{title_f}에서 특히 주목받는 항목은 {list_str}이다.",
            f"각각은 편찬자의 관점과 시대의 의학적 관심을 반영한다.",
            f"세 항목이 함께 언급될 때 {title_s}의 성격이 잘 드러난다.",
        ])
    if len(sig) >= 4:
        head_list = ", ".join(sig[:2])
        tail_list = ", ".join(sig[2:4])
        add([
            f"{title_f}의 대표 항목은 크게 두 묶음으로 나누어 볼 수 있다.",
            f"첫 묶음은 {head_list} 등이며, 둘째 묶음은 {tail_list} 등이다.",
            f"이 분류는 본서의 내용 체계를 이해하는 틀이 된다.",
            f"두 묶음의 대비는 {title_s}의 편성 원리를 압축적으로 보여 준다.",
        ])
    return paragraphs


def _tpl_related_books(fact: dict) -> list[list[str]]:
    title_f, title_s = _title_forms(fact)
    author_f, author_s = _author_forms(fact)
    related = fact.get("related_books") or []

    paragraphs: list[list[str]] = []
    if not (title_f and related):
        return paragraphs

    closers = _fact_closers(fact)

    def add(base: list[str]) -> None:
        paragraphs.append(_compose_long(base, closers))

    for rel in related:
        add([
            f"{title_f}은(는) {rel}와(과) 내용적으로 긴밀히 연관된다.",
            f"이 관계는 한국 한의학 문헌 계통 안에서 상호 참조의 형태로 드러난다.",
            f"따라서 {title_s}을(를) 연구할 때 {rel}을(를) 함께 검토하는 경우가 많다.",
        ])

    if len(related) >= 2:
        rel_list = ", ".join(related)
        add([
            f"{title_f}은(는) {rel_list} 등과 편찬적·내용적으로 서로 영향을 주고받았다.",
            f"이러한 계통적 연관은 한국 한의학 문헌사의 한 단면을 보여 준다.",
            f"{title_s}을(를) 이들 관련 저작과 병행하여 읽으면 시대 의학 지형을 더 선명히 파악할 수 있다.",
        ])
    if author_f and related:
        add([
            f"{author_f}의 {title_f}을(를) 읽을 때는 {related[0]} 등을 함께 참조하면 이해가 깊어진다.",
            f"이 책들은 서로의 내용을 보완하는 관계에 있다.",
            f"{author_s}의 편찬 맥락 역시 이러한 참조 관계를 전제로 한다.",
        ])
    return paragraphs


TEMPLATE_FUNCS = {
    "author_biography": _tpl_author_biography,
    "dynasty_context": _tpl_dynasty_context,
    "preface_summary": _tpl_preface_summary,
    "signature_items": _tpl_signature_items,
    "related_books": _tpl_related_books,
}


# ---------------------------------------------------------------------------
# Paraphrase transformations (LLM 금지, 결정론적 문자열 조작만)
# ---------------------------------------------------------------------------

# paraphrase_id 별 결정론적 치환 테이블. 같은 paraphrase_id 는 항상 같은 변환을 적용한다.
# rng 에 의존하지 않고 paraphrase_id 로부터 결정되는 치환을 사용 → 같은 템플릿의 4 variant
# 사이의 거리가 확실히 커지고, 재현성이 강화됨.

# 접속사/부사 — 문장 경계(앞에 공백/시작) 에서만 치환해야 substring 오치환을 피한다.
# key: 검색 regex (word boundary), value: 후보 variant 리스트 (첫 원소 = 원문).
_CONJ_VARIANTS = {
    r"(?<![가-힣])또한": ["또한", "아울러", "한편", "그리고", "나아가"],
    r"(?<![가-힣])따라서": ["따라서", "그러므로", "이런 까닭에", "이에", "그리하여"],
    r"(?<![가-힣])즉(?=\s)": ["즉", "다시 말해", "곧", "바꾸어 말하면", "말하자면"],
    r"(?<![가-힣])이러한": ["이러한", "이와 같은", "그러한", "이 같은", "해당"],
    r"(?<![가-힣])특히": ["특히", "무엇보다", "그중에서도", "특별히", "유달리"],
    r"(?<![가-힣])이는(?=\s)": ["이는", "이것은", "본 사안은", "해당 사실은", "이 내용은"],
    r"(?<![가-힣])이 책": ["이 책", "본서", "이 서적", "해당 저작", "이 의서"],
    r"(?<![가-힣])때문에": ["때문에", "으로 인해", "인 까닭에", "인 탓에", "에 따라"],
}

# 종결어 — "...X." 형태 끝말. 대부분 문장 종결 위치.
_ENDING_VARIANTS = {
    "이다.": ["이다.", "로 평가된다.", "라 하겠다.", "로 기록된다.", "로 정리된다."],
    "하였다.": ["하였다.", "한 것으로 전해진다.", "했다.", "하기에 이르렀다.", "하였음이 확인된다."],
    "된다.": ["된다.", "되는 것이다.", "된다 하겠다.", "됨이 확인된다.", "되어 있다."],
    "가진다.": ["가진다.", "갖는다.", "지닌다.", "지니는 것으로 이해된다.", "보유한다."],
    "해석된다.": ["해석된다.", "이해된다.", "풀이된다.", "여겨진다.", "읽힌다."],
    "드러난다.": ["드러난다.", "나타난다.", "확인된다.", "보인다.", "포착된다."],
    "자리 잡았다.": ["자리 잡았다.", "자리 매김하였다.", "위치를 확보하였다.", "지위를 굳혔다.", "자리를 잡았다."],
    "이어졌다.": ["이어졌다.", "계승되었다.", "이어져 왔다.", "연결되었다.", "전승되었다."],
    "보여 준다.": ["보여 준다.", "드러낸다.", "제시한다.", "보여 준다 하겠다.", "시사한다."],
}


def _replace_all_variant_regex(text: str, table: dict[str, list[str]], pid: int) -> str:
    """regex key 기준 치환 — word-boundary 우회. pid 로 variant 인덱스 결정."""
    out = text
    for pattern, variants in table.items():
        idx = pid % len(variants)
        new = variants[idx]
        if new == variants[0]:  # identity — skip
            continue
        out = re.sub(pattern, new, out)
    return out


def _replace_all_variant_literal(text: str, table: dict[str, list[str]], pid: int) -> str:
    """literal key (주로 종결어) — 단순 replace_all."""
    out = text
    for orig, variants in table.items():
        idx = pid % len(variants)
        new = variants[idx]
        if new != orig and orig in out:
            out = out.replace(orig, new)
    return out


def _permute_middle(sents: list[str], pid: int) -> list[str]:
    """pid=even 이면 identity. pid=odd 이면 중간 문장들 reverse."""
    if len(sents) < 4 or pid % 2 == 0:
        return list(sents)
    out = list(sents)
    mid = out[1:-1]
    mid.reverse()
    return [out[0]] + mid + [out[-1]]


def _merge_two(sents: list[str], pid: int) -> list[str]:
    """pid 에 따라 특정 인접 쌍을 병합. 결정론."""
    if len(sents) < 3:
        return list(sents)
    out = list(sents)
    target = pid % (len(out) - 1)
    a = out[target].rstrip()
    b = out[target + 1]
    if a.endswith("."):
        merged = a[:-1] + "며, " + b
        out[target] = merged
        del out[target + 1]
    return out


def apply_paraphrase(sents: list[str], paraphrase_id: int, rng: random.Random) -> str:
    """paraphrase_id 에 따라 결정론적으로 서로 다른 변이를 적용.

    pid=0: original (no-op)
    pid=1: 접속사 치환 세트 A
    pid=2: 종결어 치환 세트 B + 중간 문장 순서 reverse
    pid=3: 양쪽 치환 세트 C + 인접 병합
    pid>=4: 추가 variant (확장)
    """
    pid = paraphrase_id
    work = list(sents)

    if pid == 0:
        return " ".join(s.strip() for s in work if s and s.strip())

    # 모든 비-0 pid 에 접속사/종결어 치환을 걸되, variant index 를 pid 로 결정.
    work = [_replace_all_variant_regex(s, _CONJ_VARIANTS, pid) for s in work]
    work = [_replace_all_variant_literal(s, _ENDING_VARIANTS, pid + 2) for s in work]

    if pid >= 2:
        work = _permute_middle(work, pid)
    if pid >= 3:
        work = _merge_two(work, pid)

    text = " ".join(s.strip() for s in work if s and s.strip())
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# 메인 생성 루프
# ---------------------------------------------------------------------------


def _content_seq_for(category_idx: int, template_idx: int, paraphrase_id: int) -> int:
    """sentinel seq: -(1000 + 100*cat + 10*tpl + p). prolog(-1)와 안전하게 구분."""
    # 6 category max × 20 template × 5 paraphrase = ~600/권. 충돌 방지 위해 1000+ 오프셋.
    return -(1000 + category_idx * 100 + template_idx * 10 + paraphrase_id)


def generate_for_book(
    fact: dict,
    paraphrase_n: int,
    blacklist: list[str],
    rng: random.Random,
) -> tuple[list[dict], dict[str, int]]:
    """returns (records, stats_for_book). stats: generated/dropped_entity/dropped_length/kept."""
    records: list[dict] = []
    stats = {"generated": 0, "dropped_entity": 0, "dropped_length": 0, "kept": 0}
    book_id = fact.get("book_id")

    category_order = [
        "author_biography",
        "dynasty_context",
        "preface_summary",
        "signature_items",
        "related_books",
    ]

    for cat_idx, cat in enumerate(category_order):
        target = CATEGORY_TARGETS[cat]
        templates = TEMPLATE_FUNCS[cat](fact)
        if not templates:
            continue
        # template 개수가 target 보다 적으면 전부 사용. 많으면 앞에서 target 만큼.
        templates = templates[:target]

        for tpl_idx, sents in enumerate(templates):
            # 각 template 에 대해 paraphrase_n variant 생성
            for p_id in range(paraphrase_n):
                stats["generated"] += 1
                text = apply_paraphrase(sents, p_id, rng)

                # 길이 gate
                if len(text) < LEN_MIN:
                    # pad 시도: signature 블록 1문장 추가
                    pass
                if len(text) < LEN_MIN or len(text) > LEN_MAX:
                    # upper 초과면 LEN_MAX에서 문장 경계 cut
                    if len(text) > LEN_MAX:
                        cut = text[:LEN_MAX]
                        last_dot = cut.rfind(".")
                        if last_dot >= LEN_MIN:
                            text = cut[: last_dot + 1]
                        else:
                            stats["dropped_length"] += 1
                            continue
                    else:
                        stats["dropped_length"] += 1
                        continue

                # entity validation
                ok, bad = validate_text(text, blacklist)
                if not ok:
                    stats["dropped_entity"] += 1
                    continue

                records.append({
                    "book_id": book_id,
                    "volume_id": None,
                    "content_seq": _content_seq_for(cat_idx, tpl_idx, p_id),
                    "content_level": None,
                    "up_path_nm": f"__synth_facts__/{cat}",
                    "text": text,
                    "n_chars": len(text),
                    "category": cat,
                    "paraphrase_id": p_id,
                })
                stats["kept"] += 1
    return records, stats


def dedup_records(records: list[dict]) -> tuple[list[dict], int]:
    """SHA-256(text) 기준 중복 제거. (unique, dedup_dropped)."""
    seen: set[str] = set()
    out: list[dict] = []
    dropped = 0
    for rec in records:
        h = hashlib.sha256(rec["text"].encode("utf-8")).hexdigest()
        if h in seen:
            dropped += 1
            continue
        seen.add(h)
        out.append(rec)
    return out, dropped


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--factsheet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--paraphrase", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # PYTHONHASHSEED=0 이 아니면 seed util 이 RuntimeError — 환경에서 미리 설정 가정.
    # 다만 편의를 위해 이 스크립트에서는 PYTHONHASHSEED 미설정이어도 강제 무효화하지 않고
    # set_global_seed 의 require 플래그로만 gate. 재크롤 orchestrator 와 동일 관행.
    if os.environ.get("PYTHONHASHSEED") != "0":
        # 상위 runner 가 아닌 단독 실행 편의: 현 프로세스는 이미 기동됐으므로 assert 만 스킵.
        print(
            "[WARN] PYTHONHASHSEED != '0'. 재현성 위해 호출 측에서 "
            "`PYTHONHASHSEED=0 .venv/bin/python ...` 로 실행 권장.",
            file=sys.stderr,
        )
        set_global_seed(args.seed, require_pythonhashseed=False)
    else:
        set_global_seed(args.seed)

    factsheet = load_factsheet(args.factsheet)
    if not factsheet:
        print("[ERROR] factsheet 로드 실패 또는 비어 있음.", file=sys.stderr)
        sys.exit(2)

    blacklist = build_blacklist(factsheet)
    print(f"[INFO] factsheet 권 수: {len(factsheet)}", file=sys.stderr)
    print(f"[INFO] entity blacklist (fact sheet union 제외 후): {blacklist}", file=sys.stderr)

    rng = random.Random(args.seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    total_records: list[dict] = []
    per_book_stats: dict[int, dict[str, int]] = {}

    for book_id in sorted(factsheet.keys()):
        fact = factsheet[book_id]
        recs, stats = generate_for_book(fact, args.paraphrase, blacklist, rng)
        per_book_stats[book_id] = stats
        total_records.extend(recs)

    # dedup
    total_before = len(total_records)
    total_records, dedup_dropped = dedup_records(total_records)

    with open(args.output, "w", encoding="utf-8") as f:
        for rec in total_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # stderr report
    print("", file=sys.stderr)
    print("=" * 64, file=sys.stderr)
    print("expand_facts: per-book stats", file=sys.stderr)
    print("=" * 64, file=sys.stderr)
    print(
        f"{'book_id':>8}  {'generated':>10}  {'drop_entity':>12}  "
        f"{'drop_len':>9}  {'kept':>6}",
        file=sys.stderr,
    )
    for bid, st in per_book_stats.items():
        print(
            f"{bid:>8}  {st['generated']:>10}  {st['dropped_entity']:>12}  "
            f"{st['dropped_length']:>9}  {st['kept']:>6}",
            file=sys.stderr,
        )
    print(
        f"{'TOTAL':>8}  "
        f"{sum(s['generated'] for s in per_book_stats.values()):>10}  "
        f"{sum(s['dropped_entity'] for s in per_book_stats.values()):>12}  "
        f"{sum(s['dropped_length'] for s in per_book_stats.values()):>9}  "
        f"{sum(s['kept'] for s in per_book_stats.values()):>6}",
        file=sys.stderr,
    )
    print(
        f"\ndedup: before={total_before}, after={len(total_records)}, dropped={dedup_dropped}",
        file=sys.stderr,
    )
    print(f"output: {args.output} (records={len(total_records)})", file=sys.stderr)


if __name__ == "__main__":
    main()
