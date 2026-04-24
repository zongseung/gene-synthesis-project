from __future__ import annotations

import re
from dataclasses import dataclass

from .config import DEFAULTS

PRECHECK_PATTERNS: tuple[re.Pattern, ...] = tuple(
    re.compile(p)
    for p in [
        r"(죽고 싶|자살|자해)",
        r"(119|응급실|심장마비|뇌졸중)",
        r"(제가|저는).*(가슴이 답답|숨이 차|심한 통증|호흡 곤란)",
        r"(임신|임산부).*(먹어도 되|복용해도 되)",
        r"(아이|소아|영유아).*(먹이면 되|복용)",
        r"(용량|몇 돈|몇 푼|몇 알).*(알려)",
        r"(양약|혈압약|당뇨약).*(같이 먹|대체)",
    ]
)

POSTCHECK_PATTERNS: tuple[re.Pattern, ...] = tuple(
    re.compile(p)
    for p in [
        r"\d+\s*돈",
        r"\d+\s*푼",
        r"\d+\s*알",
        r"\d+\s*g",
        r"\d+\s*mg",
        r"(달여 먹|가루내어 먹|복용한다)",
    ]
)

REFUSAL_TEMPLATE = """본 모델은 한의학 고전 문헌 연구 보조 AI로, 개인 증상에 대한 진단이나 구체적인 약물 처방을 제공할 수 없습니다.
현재 질문은 개인 의료 상담에 해당할 수 있으므로 우선 전문의 또는 의료기관의 진료를 받아야 합니다.
증상이 급하거나 악화되면 119 또는 응급실을 먼저 고려하십시오.

문헌 설명이 필요하다면 개인 증상 대신 동의보감의 서지 정보나 개념 설명 형태로 다시 질문해 주세요.
[전문의 상담 필수]"""

MASK_NOTICE = "\n\n[※ 구체 용량·처방 표현은 안전 정책에 따라 마스킹되었습니다.]"


@dataclass(frozen=True)
class PreCheckResult:
    refused: bool
    refusal_text: str | None


def pre_check(user_input: str) -> PreCheckResult:
    for pattern in PRECHECK_PATTERNS:
        if pattern.search(user_input):
            return PreCheckResult(refused=True, refusal_text=REFUSAL_TEMPLATE)
    return PreCheckResult(refused=False, refusal_text=None)


def post_check(response: str) -> str:
    out = response
    hit = False
    for pattern in POSTCHECK_PATTERNS:
        if pattern.search(out):
            out = pattern.sub("[MASKED]", out)
            hit = True
    if hit and MASK_NOTICE not in out:
        out = out.rstrip() + MASK_NOTICE
    if DEFAULTS.footer_enabled and DEFAULTS.footer not in out:
        out = out.rstrip() + "\n\n" + DEFAULTS.footer
    return out
