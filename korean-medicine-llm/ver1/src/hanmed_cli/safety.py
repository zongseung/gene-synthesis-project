"""§10.5.4 Safety Refusal Layer (2단계).

Layer 1 (pre-model): UNSAFE_PATTERNS 매치 → 모델 호출 생략하고 refusal 반환.
Layer 2 (post-model): CLINICAL_DIRECTIVE_PATTERNS 매치 → disclaimer 자동 append.

R3.4 교정: 기존 리터럴 N 버그 (r"(몇|N) ?회", r"N g 씩") 를 \\d+ 로 일반화.
R3.4 anti-contamination 프로토콜: regex 는 T4 20문항 곡선을 외워 작성하지 않는다
(§10.5.5). 한문 jailbreak 대응은 v0 범위 밖 (§10.5.6 열린 결정 #2 — classifier 도입은 v1).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from hanmed_cli.config import DEFAULTS


# --- Layer 1 ----------------------------------------------------------

UNSAFE_PATTERNS: tuple[re.Pattern, ...] = tuple(
    re.compile(p)
    for p in [
        # 자가 진단 / 증상 호소
        r"(내가|제가).*(아픈데|증상).*(처방|약)",
        r"(내|제) (몸|배|머리).*(처방|치료)",
        # 복용량 / 주기 (숫자 표현은 \d+ 로 일반화 — R3.4)
        r"(얼마나|하루.*몇|하루\s*\d+).*(복용|먹)",
        r"(몇|\d+)\s*회.*(복용|먹)",
        # 취약 집단
        r"(임신|임산부).*(복용|먹)",
        r"(어린이|소아|유아).*(복용|먹)",
        r"(노인|고령).*(복용|먹)",
        # 양약 대체
        r"(양약|항생제|당뇨약|혈압약).*(대신|대체)",
        # 위기
        r"(자살|자해|죽고 싶)",
        # 구체 질환 진단
        r"(암|종양|중풍).*(치료|처방|낫)",
    ]
)

REFUSAL_TEMPLATE = """저는 한의학 고전 문헌 연구 보조 AI 입니다.
환자 진단, 처방 결정, 복용량 권고와 같은 임상 의사결정은 제공하지 않습니다.
의료적 판단은 반드시 면허 한의사와 상의하시기 바랍니다.

다만 한의학 고전의 이론·본초·처방 구성 자체에 대한 설명은 도울 수 있습니다.
예) "인삼의 성미와 귀경" / "사물탕의 구성 약재" / "동의보감 내경편 身形 해석"
"""


# --- Layer 2 ----------------------------------------------------------

CLINICAL_DIRECTIVE_PATTERNS: tuple[re.Pattern, ...] = tuple(
    re.compile(p)
    for p in [
        r"하루\s*\d+\s*회",
        r"\d+\s*일\s*동안\s*복용",
        r"식후\s*\d+\s*분",
        r"(즉시|당장)\s*복용",
        r"\d+\s*g\s*씩",
    ]
)

DISCLAIMER = (
    "\n\n⚠ 본 답변은 한의학 고전 문헌 설명이며 임상 의사결정이 아닙니다.\n"
    "  복용 전 반드시 면허 한의사와 상의하세요."
)


# --- API --------------------------------------------------------------

@dataclass(frozen=True)
class PreCheckResult:
    refused: bool
    refusal_text: str | None


def pre_check(user_input: str) -> PreCheckResult:
    """Layer 1 — 모델 호출 전 regex 매칭. 매치 → refusal, 아니면 pass."""
    for pat in UNSAFE_PATTERNS:
        if pat.search(user_input):
            return PreCheckResult(refused=True, refusal_text=REFUSAL_TEMPLATE)
    return PreCheckResult(refused=False, refusal_text=None)


def post_check(response: str) -> str:
    """Layer 2 — 모델 출력에 임상 지시 패턴 포함 시 disclaimer append.
    footer 는 DEFAULTS.footer_enabled 플래그로 게이팅 (R1 2026-04-21)."""
    out = response
    for pat in CLINICAL_DIRECTIVE_PATTERNS:
        if pat.search(out):
            out = out.rstrip() + DISCLAIMER
            break
    # footer (§10.5.4) — R1 기준 DEFAULTS.footer_enabled 로 게이팅.
    if DEFAULTS.footer_enabled and DEFAULTS.footer not in out:
        out = out.rstrip() + "\n\n" + DEFAULTS.footer
    return out
