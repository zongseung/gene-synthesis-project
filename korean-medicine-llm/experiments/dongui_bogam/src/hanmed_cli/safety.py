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
        # 1인칭 + 처방 요청 — 학술 검색 ("두통에 등장하는 처방을 알려") 와 구분
        r"(제가|저는|나는|내가|나에게|저에게|제게|저한테|나한테).*(처방|약재).*(부탁|해\s*주|구체적|받|해\s*달|알려|추천|제시|적어)",
        # 명시적 진료 동사 (1인칭 없어도 진료 행위 분명) — 학술의 "알려/추천" 은 제외
        r"(처방|약재)\s*[을를]?\s*(부탁|해\s*주|구체적|해\s*달)",
        # 본인 질환·증세 보유 + 처방·약재 요청 (한 문장 안)
        r"(제가|저는|나는|내가).*(질환|병|증세|아픕니다|아프|불편).*(처방|약재|약)",
        r"(질환|병|증세).*(가지고\s*있|앓고\s*있|진단|증상)",
        # 심리·정신 + 신체 동반 — 진료 영역
        r"(심리적|정신적|마음).*(불안|우울|초조|답답|장애)",
    ]
)

# pre_check 보다 약한 진료성 의도. hit 해도 거부하지는 않지만, post_check 의
# dosage/행동 마스킹을 활성화해 모델이 환각으로 만든 용량·복용지시가 새어나가는
# 것을 보조 차단한다 (context-aware masking, ver8.1 § C policy).
# 학술 검색 query (예: "사물탕의 구성", "신형 개념") 는 hit 하지 않아 본문이
# 그대로 인용된다.
CLINICAL_INTENT_PATTERNS: tuple[re.Pattern, ...] = tuple(
    re.compile(p)
    for p in [
        r"(내가|제가|나는|저는).*(아프|불편|증상|먹어도|복용해도|괜찮|어때)",
        r"어떤\s*(처방|약|한약).*(좋|쓸|먹|복용|효과)",
        r"(임신|소아|영유아|어린이|만성질환|복약).*(처방|복용|먹|약|쓸 수)",
        r"(내가|제가|나는|저는|저희|저의|내가요|제가요).*(증상|아픈데|불편한데).*(처방|약|먹)",
    ]
)

# ver8.1 round_2 동기화 — audit.py 의 12-unit DOSAGE_PATTERNS 와 정확히 일치.
# 학습 corpus phaseB_qa_v8_1_corpus.jsonl 에 잔존하는 quoted citation dosage 22 rows 가
# inference 시 모델 출력으로 새어 나오는 것을 defense-in-depth 로 차단.
# 근거: docs/ver8.1/04_round_2_log_and_convergence.md §5.3, ver5/06_safety §3.2.
POSTCHECK_PATTERNS: tuple[re.Pattern, ...] = tuple(
    re.compile(p)
    for p in [
        # 한글 단위 11종 (audit.py DOSAGE_PATTERNS 와 동일)
        r"(?<![0-9])\d+(?:\.\d+)?\s*(?:돈|푼|냥|전|알|첩|근|홉|되|말|구)",
        # SI 단위
        r"(?<![0-9])\d+(?:\.\d+)?\s*(?:g|mg|kg|ml|L)\b",
        # 합산 표현 ("각 N단위")
        r"각\s*\d+(?:\.\d+)?(?:돈|푼|냥|전|알|첩|근|홉|되|말|구|g|mg)",
        # '편' 단위 — 카운트 ("5편의 약재"). 분류 라벨 ('내경편/외형편/잡병편/탕액편/침구편') 은
        # 앞에 한자/한글이 오므로 \d+ 요구만으로 자연 배제됨 (negative lookahead 불필요).
        r"(?<![0-9])\d+편",
        # 행동 트리거 (기존 보존)
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


def is_clinical_intent(user_input: str) -> bool:
    """pre_check 보다 약한 진료성 의도 탐지. hit 시 post_check 의 dosage 마스킹
    을 켜고, hit 하지 않으면 학술 검색으로 간주해 본문을 그대로 인용한다."""
    return any(p.search(user_input) for p in CLINICAL_INTENT_PATTERNS)


def post_check(response: str, *, mask_dosage: bool = True) -> str:
    """모델 출력 후처리. mask_dosage=False 일 때 dosage/행동 마스킹을 스킵 —
    학술 검색 컨텍스트에서 동의보감 본문을 손상 없이 인용하기 위함.

    footer 부착은 mask_dosage 와 무관하게 항상 수행 (DEFAULTS.footer_enabled).
    """
    out = response
    hit = False
    if mask_dosage:
        for pattern in POSTCHECK_PATTERNS:
            if pattern.search(out):
                out = pattern.sub("[MASKED]", out)
                hit = True
        if hit and MASK_NOTICE not in out:
            out = out.rstrip() + MASK_NOTICE
    if DEFAULTS.footer_enabled and DEFAULTS.footer not in out:
        out = out.rstrip() + "\n\n" + DEFAULTS.footer
    return out
