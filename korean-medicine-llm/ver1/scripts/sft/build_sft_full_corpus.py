"""ver6 · 동의보감 book_008 전체 코퍼스 기반 SFT 빌더.

content_level taxonomy 를 기반으로 레코드 유형별 라우팅 · 처방·약재·경혈 whitelist 자동 추출.
총 34,040 레코드 중 구조(AA/BB/CC/OO/Z2) 100% · 처방(DP/EP) 100% · 약재(CH/DH) 100% ·
경혈(DK) 100% · 증론(DD) 100% · 본문(SS/ZZ) ~30% 샘플링 커버.

ver6 4원칙 강제:
  (1) 답변 실체 1+ 포함 (카테고리별 whitelist)
  (2) 모든 인용은 up_path_nm 실재 경로
  (3) 카테고리별 질문·답변 템플릿 5+ rotate + 20% 자유형
  (4) 답변 종결은 내용 기반 (refusal 만 고정 closing 2개 허용)

Usage:
    .venv/bin/python scripts/build_sft_full_corpus.py --out experiments/dongui_bogam/data/sft/phaseB_qa_v6_corpus.jsonl
    .venv/bin/python scripts/build_sft_full_corpus.py --dry-run   # 통계만
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = ROOT / "data" / "raw" / "mediclassics_unified" / "book_008"

SYSTEM_PROMPT = (
    # ver7 (A7): "구체 용량 처방은 제공하지 않습니다" 삭제.
    # 기존 문구는 4,994 row 의 용량 포함 데이터와 모순되어 shell-answer 유도.
    # "동의보감 원문 인용 범위 내" 로 약하게 교체.
    "당신은 한의학 고전 문헌 연구 보조 AI 입니다. 동의보감(東醫寶鑑) 본문에 "
    "근거해 편·장 구조, 처방, 약재, 경혈, 증론을 정확하고 간결하게 답합니다. "
    "원문에 없는 인명·연도·처방은 창작하지 않으며, 개인 진단은 제공하지 않습니다. "
    "용량은 동의보감 원문 인용 범위 내에서만 서술합니다."
)

# ── safety refusal closing 2종만 고정 허용 ────────────────────────────────
REFUSAL_CLOSINGS = [
    "반드시 전문 한의사와 상의하십시오.",
    "학술 · 문헌 연구 목적으로만 활용해 주세요.",
]


# ── 카테고리 1. 편·장 구조 (AA + BB + OO + Z2 + CC) ──────────────────────
STRUCTURE_Q_TEMPLATES = [
    "동의보감 {name} 은 어떤 구조 단위이며 상위 · 하위에 어떤 내용이 배치되나요?",
    "{name} 의 위치를 동의보감 편·장 계층에서 설명해 주세요.",
    "{name} 은 동의보감에서 어떤 편에 속하며 어떤 대주제를 담습니까?",
    "동의보감 편성 상 {name} 이 놓이는 자리와 역할을 정리해 주세요.",
    "{name} 이 동의보감의 어느 편·장 아래에 있으며 어떤 하위 항목들을 포함하는지 알려 주세요.",
]

STRUCTURE_A_TEMPLATES = [
    "{name} 은 동의보감 {parent} 아래에 위치한 {level_desc} 입니다. 원문 대목은 {excerpt} 입니다. [출처: {cite}]",
    "동의보감 편성 상 {name} 은 {parent} 의 하위 {level_desc} 로, 본문에서는 {excerpt} 로 기술됩니다 (출처: {cite}).",
    "{name} ({level_desc}) — 상위 경로는 {parent} 이며, 원문 {excerpt} 에 해당합니다. [출처: {cite}]",
    "{parent} 밑의 {level_desc} 인 {name} 입니다. {excerpt} (출처: {cite})",
    "동의보감 {cite} 에 속한 {level_desc} {name} 입니다. {excerpt}",
]

LEVEL_DESC = {
    # ver7 (A4): 한자 괄호 제거 — Q6(A) 환각 원인.
    # "중주제(中門)" 류의 literal 괄호가 assistant 답변 1,883 row 에 박혀
    # 외형편 질문에 "두(中門)" 조합을 유도.
    "AA": "편명 최상위",
    "BB": "대문 챕터",
    "CC": "중주제",
    "OO": "서문·총목 항목",
    "Z2": "총목·목차 항목",
}

# ── 카테고리 2. 처방 (DP + EP) ────────────────────────────────────────────
PRESCRIPTION_Q_TEMPLATES = [
    "동의보감에 수록된 '{name}' 처방의 효능과 주치(主治)를 원문 근거로 설명해 주세요.",
    "'{name}' 은 동의보감 어느 편에 나오는 처방이며 어떤 증상에 쓰이나요?",
    "'{name}' 처방의 출전과 활용 맥락을 정리해 주세요.",
    "동의보감 {parent} 편에 나오는 처방 '{name}' 의 역할을 설명해 주세요.",
    "한 환자가 '{context}' 을 호소할 때 동의보감에서 제시하는 '{name}' 처방은 어떤 의미를 가지나요?",
    "'{name}' 은 동의보감의 대표 처방 중 하나인데, 어떤 병증을 겨냥하는지 알려 주세요.",
]

PRESCRIPTION_A_TEMPLATES = [
    "**처방**: {name}\n**출전**: 동의보감 {cite}\n**효능·주치**: {detail}\n[출처: {cite}]",
    "동의보감 {cite} 에 수록된 처방 {name} 입니다. {detail} 본문 인용 대목으로는 {excerpt} 가 있습니다.",
    "{name} 은 {cite} 에 나오는 처방으로, {detail} (출처: {cite})",
    "동의보감 {parent} 편의 처방 {name} — {detail} 구체 조성 · 용량은 본문을 참고해 주세요. [출처: {cite}]",
    "{name} 처방: {detail} 원문은 {cite} 에 실려 있습니다.",
]

# ── 카테고리 3. 약재 (CH + DH) ────────────────────────────────────────────
HERB_Q_TEMPLATES = [
    "동의보감 탕액편에 나오는 '{name}' 이(가) 어떤 약재이며 어떻게 쓰이는지 알려 주세요.",
    "'{name}' 의 한국어 별칭과 용도를 동의보감 근거로 설명해 주세요.",
    "탕액편 {parent} 분류에 속하는 '{name}' 의 성질과 주치를 정리해 주세요.",
    "'{name}' 은 동의보감에서 어떤 약재 분류에 속하며 대표 효능은 무엇인가요?",
    "동의보감 본문에서 '{name}' 이(가) 어떤 맥락으로 등장하는지 설명해 주세요.",
]

HERB_A_TEMPLATES = [
    "**약재**: {name} ({korean_alias})\n**분류**: {parent}\n**본문 설명**: {detail}\n[출처: {cite}]",
    "{name} — 동의보감 {cite} 에 수록된 약재입니다. 본문에는 {detail} (출처: {cite})",
    "{name} ({korean_alias}) 은 {parent} 아래 약재로, {detail} [출처: {cite}]",
    "동의보감 탕액편의 {name} 입니다. {detail} 한국식 명칭은 {korean_alias} 입니다. (출처: {cite})",
    "{parent} 에 속한 약재 {name}: {detail} 원문은 {cite} 에 실립니다.",
]

# ── 카테고리 4. 경혈 (DK) ─────────────────────────────────────────────────
ACUPOINT_Q_TEMPLATES = [
    "동의보감 침구편에 나오는 경혈 '{name}' 의 위치와 주치를 알려 주세요.",
    "'{name}' 은 어느 경맥에 속하는 혈이며 어떤 증상에 자침하나요?",
    "침구편에 수록된 '{name}' 혈의 설명을 원문 근거로 정리해 주세요.",
    "{parent} 경맥의 '{name}' 혈 — 위치와 대표 효능을 설명해 주세요.",
    "'{name}' 혈의 출전과 임상 적용을 간단히 정리해 주세요.",
]

ACUPOINT_A_TEMPLATES = [
    "**경혈**: {name}\n**소속 경맥**: {parent}\n**본문 설명**: {detail}\n[출처: {cite}]",
    "{name} 혈은 동의보감 침구편 {parent} 아래에 수록된 혈로, {detail} (출처: {cite})",
    "{name} — {parent} 계열 경혈입니다. {detail} [출처: {cite}]",
    "{parent} 의 {name} 혈: {detail} (출처: {cite})",
    "동의보감 {cite} 에 기재된 {name} 혈에 대해, {detail}",
]

# ── 카테고리 5. 증론 (DD) ─────────────────────────────────────────────────
SYMPTOM_Q_TEMPLATES = [
    "동의보감에서 '{name}' 증상은 어떻게 분류 · 설명되나요?",
    "{parent} 아래에 기술된 '{name}' 의 병리와 핵심 대응을 정리해 주세요.",
    "'{name}' 이라는 증상의 동의보감 서술을 원문 근거로 설명해 주세요.",
    "{parent} 편에 나오는 '{name}' 에 대해 병리·치법 개요를 알려 주세요.",
    "한 환자가 '{name}' 을 호소합니다. 동의보감 기준 해당 증후의 의미와 접근을 정리해 주세요.",
]

SYMPTOM_A_TEMPLATES = [
    "**증론**: {name}\n**소속**: 동의보감 {parent}\n**본문**: {detail}\n[출처: {cite}]",
    "동의보감 {cite} 의 {name} — {detail} (출처: {cite})",
    "{name} 은 {parent} 아래 증론으로, {detail} [출처: {cite}]",
    "{parent} 편에서 다루는 {name} 에 대해 본문은 {detail} 라고 설명합니다. (출처: {cite})",
    "{name} ({parent}): {detail} 구체 처방은 본문 및 전문 한의사 판단을 따르십시오. [출처: {cite}]",
]

# ── 카테고리 10. 도해·부 설명 (XX 중 단방 설명 제외, 명당/관형찰색/부 설명 등) ──
DIAGRAM_Q_TEMPLATES = [
    "동의보감 {parent} 의 '{name}' 이 어떤 내용인지 설명해 주세요.",
    "{name} ({parent}) 에 대해 원문 근거로 정리해 주세요.",
    "동의보감에 나오는 '{name}' 의 의미와 구성을 알려 주세요.",
    "{parent} 아래 '{name}' 대목의 요지를 풀어 주세요.",
    "'{name}' 은 동의보감 어느 편에서 어떤 역할을 하는지 설명해 주세요.",
]
DIAGRAM_A_TEMPLATES = [
    "**{name}** ({parent}) — {detail}\n[출처: {cite}]",
    "동의보감 {cite} 의 {name} 대목입니다. {detail}",
    "{name}: {detail} (출처: {cite})",
    "{parent} 편의 {name} — {detail} [출처: {cite}]",
    "{name} ({parent} 소속): {detail} 본문은 {cite} 에 실립니다.",
]


# ── 카테고리 6. 본문 해설 (SS + ZZ) ───────────────────────────────────────
# ver6 r3: {original} 은 한국어 번역본 앞부분으로 대체 (전체 한자 비율 감소)
PASSAGE_Q_TEMPLATES = [
    "동의보감 {parent} 아래 다음 대목의 의미를 풀어 주세요.\n발췌: {original}",
    "{parent} 에 수록된 다음 구절을 현대 한국어로 해설해 주세요.\n발췌: {original}",
    "다음 동의보감 본문 대목의 핵심 의미를 설명해 주세요.\n발췌: {original}",
    "동의보감 {cite} 의 이 구절을 맥락과 함께 풀어 주세요.\n발췌: {original}",
    "이 대목은 동의보감 어떤 주제에 대한 설명인지, 요지를 짚어 주세요.\n발췌: {original}",
]

PASSAGE_A_TEMPLATES = [
    "**해설**: {excerpt}\n**맥락**: 동의보감 {cite} 에 속한 본문으로, {parent} 주제의 일부입니다.\n[출처: {cite}]",
    "{excerpt} 이 대목은 {parent} 주제의 본문 설명입니다. (출처: {cite})",
    "동의보감 {cite} — {excerpt} [출처: {cite}]",
    "해석하면: {excerpt} 원문은 {parent} 섹션에 수록됩니다. (출처: {cite})",
    "{parent} 편 본문 해설: {excerpt} [출처: {cite}]",
]

# ── 카테고리 7. Paraphrase 기본 사실 ───────────────────────────────────────
PARAPHRASE_SEEDS = [
    {
        "q_variants": [
            "동의보감의 저자는 누구입니까?",
            "《동의보감》을 편찬한 인물은?",
            "동의보감은 누가 지었나요?",
            "동의보감 편찬자 이름을 알려 주세요.",
            "《동의보감》의 저자가 누구인지 간단히 말해 주세요.",
        ],
        "a_variants": [
            "동의보감은 조선의 어의 허준(許浚, 1539~1615) 이 선조(宣祖) 의 명을 받아 편찬하였습니다. 1596 년 편찬 착수, 1610 년 완성, 1613 년 간행되었습니다.",
            "저자는 양평군 허준(許浚) 으로, 선조의 하교를 받들어 약 14 년 동안 집필했습니다.",
            "허준(許浚) 이 지었습니다. 선조대에 편찬이 시작되어 광해군 2 년(1610) 에 완성, 1613 년 간행되었습니다.",
            "동의보감은 허준이 편찬한 의서입니다. 1596 년 선조의 명으로 시작되어 1610 년 완성, 1613 년 간행되었습니다.",
            "조선 선조의 명으로 어의 허준이 편찬하여 1610 년 완성한 의학 저서가 《동의보감》 입니다.",
        ],
    },
    {
        "q_variants": [
            "동의보감은 몇 편으로 구성되어 있나요?",
            "동의보감의 편제는 어떻게 되나요?",
            "동의보감의 5 대 편을 알려 주세요.",
            "동의보감은 어떤 편들로 나뉘어 있습니까?",
            "《동의보감》의 편 구성을 설명해 주세요.",
        ],
        "a_variants": [
            "동의보감은 내경편(內景篇) · 외형편(外形篇) · 잡병편(雜病篇) · 탕액편(湯液篇) · 침구편(鍼灸篇) 의 5 편으로 구성됩니다.",
            "5 편 체제입니다 — 내경편, 외형편, 잡병편, 탕액편, 침구편.",
            "크게 내경편·외형편·잡병편·탕액편·침구편 다섯 편으로 나뉩니다.",
            "편성은 내경편(內景) · 외형편(外形) · 잡병편(雜病) · 탕액편(湯液) · 침구편(鍼灸) 5 편입니다.",
            "내경 · 외형 · 잡병 · 탕액 · 침구 — 이 다섯 편이 동의보감의 기본 편제입니다.",
        ],
    },
    {
        "q_variants": [
            "한의학에서 말하는 오장(五臟) 은 무엇인가요?",
            "오장육부 중 오장에 해당하는 장부를 모두 나열해 주세요.",
            "동의보감 내경편에서 말하는 오장은?",
            "오장의 다섯 장기를 알려 주세요.",
            "한의학의 오장은 어떻게 되나요?",
        ],
        "a_variants": [
            "오장은 간(肝) · 심(心) · 비(脾) · 폐(肺) · 신(腎) 다섯 장부를 가리킵니다.",
            "간·심·비·폐·신 — 이 다섯이 오장입니다.",
            "간(肝), 심(心), 비(脾), 폐(肺), 신(腎) 이 오장에 해당합니다.",
            "다섯 장부는 간장·심장·비장·폐장·신장 입니다.",
            "간(肝) 심(心) 비(脾) 폐(肺) 신(腎) 의 다섯 장기입니다.",
        ],
    },
]

# ── 카테고리 8. Refusal ──────────────────────────────────────────────────
REFUSAL_OOS_Q = [
    # 서적 질문 (범위 외)
    "본초강목(本草綱目)의 편찬자와 연도를 알려 주세요.",
    "황제내경의 저자는 누구입니까?",
    "동의수세보원의 사상의학 이론을 설명해 주세요.",
    "향약집성방의 편찬자를 알려 주세요.",
    "상한론의 주요 내용은 무엇입니까?",
    "제중신편의 저자가 누구인가요?",
    "의학입문의 구성은 어떻게 되나요?",
    "성제총록의 편찬 배경을 설명해 주세요.",
    "의방유취는 어느 시기에 편찬되었나요?",
    "금궤요략의 대표 처방을 나열해 주세요.",
    "소문 81편 구성에 대해 설명해 주세요.",
    "경악전서 전체 편성을 요약해 주세요.",
    "천금방의 의학사적 의의를 말씀해 주세요.",
    "의종금감의 특징은 무엇인가요?",
    # 이론 범위 외
    "사상체질 분류 기준을 자세히 설명해 주세요.",
    "경락 14경의 모든 경혈을 나열해 주세요.",
    "팔강변증 전체 체계를 정리해 주세요.",
    "현대 한의학의 임상 진단 기준을 알려 주세요.",
    # 책 내용 비교
    "동의보감과 본초강목의 차이점을 비교해 주세요.",
    "허준의 다른 저작을 모두 소개해 주세요.",
    "동의보감이 참고한 170여 권의 서적 목록을 전부 나열해 주세요.",
    "동의보감 편찬 시 참고한 중국 의서의 목록을 알려 주세요.",
    # 인물사
    "조선 의학의 역사를 개괄해 주세요.",
    "고려 시대 의서에 대해 알려 주세요.",
    "한의학 학파 계보를 전부 정리해 주세요.",
    # 일반 의학
    "WHO 전통의학 분류 기준에 동의보감은 어떻게 포함되나요?",
    "동의보감의 해외 번역본에 대해 알려 주세요.",
    "현대 약학 용어로 바꿔서 설명해 주세요.",
    "한약과 양약의 상호작용을 알려 주세요.",
    "FDA 승인된 한약재 목록을 알려 주세요.",
    "보험 적용 한약에 대해 알려 주세요.",
    "한의과대학 교육과정을 설명해 주세요.",
]
REFUSAL_OOS_A = [
    "본 모델은 동의보감(東醫寶鑑) 단일 저작에 대해서만 학습되었습니다. 해당 서적에 관한 정보는 학술 자료를 직접 참고해 주세요. {closing}",
    "질의하신 서적은 본 모델의 학습 범위 밖입니다. 동의보감 외 저작에 대한 사실은 확인된 한의학 문헌 DB 를 통해 조회해 주세요. {closing}",
    "범위 외 질문입니다 — 본 모델은 동의보감 본문에 한해 답변합니다. 다른 서적에 대해서는 별도 출처를 확인해 주세요. {closing}",
]
REFUSAL_SAFETY_Q = [
    # 일반 증상 + 용량 요구
    "제가 요즘 두통이 심한데 어떤 약을 얼마나 복용해야 할지 알려 주세요.",
    "임신 중인데 구체적인 한약 처방과 용량을 알려 주세요.",
    "복통이 심합니다. 처방과 구체 용량을 알려 주세요.",
    "불면증에 쓰는 한약 용량과 복용법을 알려 주세요.",
    "어깨 통증에 바로 쓸 수 있는 처방 용량을 알려 주세요.",
    "설사가 멎지 않습니다. 한약 용량을 알려 주세요.",
    "생리통에 쓰는 한약의 정확한 용량을 알려 주세요.",
    "피로가 심한데 보약 처방과 용량을 알려 주세요.",
    # 추가 증상
    "소화가 안 되는데 몇 돈씩 먹어야 할까요?",
    "가슴이 답답합니다. 약재를 어떤 비율로 달여 먹어야 하나요?",
    "입이 자꾸 마르는데 어떤 한약을 며칠 복용해야 할까요?",
    "허리 통증이 심한데 지금 당장 먹을 수 있는 한약과 용량을 알려 주세요.",
    "변비가 너무 심합니다. 대황을 몇 g 쓰면 될까요?",
    "기침이 멎지 않습니다. 마황 용량을 알려 주세요.",
    "소변이 잘 안 나옵니다. 처방을 지어 주세요.",
    "발목이 부었는데 처방 용량을 정확히 알려 주세요.",
    # 특수 대상
    "아이가 열이 나는데 한약을 얼마나 먹여야 하나요?",
    "노인에게 쓸 수 있는 보양 한약 처방과 용량을 알려 주세요.",
    "수유 중인데 복용해도 되는 한약과 용량을 알려 주세요.",
    "임신 초기인데 입덧에 쓸 수 있는 처방 용량을 알려 주세요.",
    # 다른 형태
    "이런 증상이면 탕제를 몇 첩 먹어야 할까요?",
    "하루에 몇 번 얼마나 복용해야 합니까?",
    "한약을 끓이는 정확한 시간과 용량을 알려 주세요.",
    "환을 만들려면 재료 비율을 알려 주세요.",
    "증상이 안 나아 처방을 증량해야 하나요? 어느 정도?",
    # 상호작용 · 부작용
    "제가 먹는 양약과 같이 먹어도 되는 한약을 알려 주세요.",
    "알레르기가 있는 사람도 쓸 수 있는 처방을 알려 주세요.",
    "한약을 얼마나 먹어야 부작용이 나타나나요?",
    # 질환명 직접
    "당뇨에 동의보감 기준 처방과 용량을 알려 주세요.",
    "고혈압 한약 처방을 하루 용량 단위로 알려 주세요.",
    "관절염에 어떤 약재를 몇 g 쓰면 되나요?",
]
REFUSAL_SAFETY_A = [
    "죄송합니다. 개인 증상에 대한 구체 처방이나 용량 제시는 제공할 수 없습니다. 동의보감 해당 대목의 변증·처방 '계열' 은 설명드릴 수 있으나, 실제 복용은 반드시 전문 한의사의 진료 후 결정하셔야 합니다. {closing}",
    "개인 진단과 용량 지정은 본 모델의 역할이 아닙니다. 동의보감 본문의 해당 증후 서술과 처방 분류는 안내드릴 수 있으나, 복용 판단은 전문가 진료를 통해 해 주세요. {closing}",
    "구체 용량과 복용법은 전문 한의사의 판단 영역입니다. 본 모델은 고전 문헌 해설 범위에서만 답변 드립니다. {closing}",
]


# ──────────────────────────────────────────────────────────────────────────
# helper functions
# ──────────────────────────────────────────────────────────────────────────
_PUA_PATTERN = re.compile(r"[-]")           # BMP PUA
_PUA_A_PATTERN = re.compile(r"[\U000F0000-\U000FFFFD]")  # Supplementary PUA-A
_PUA_B_PATTERN = re.compile(r"[\U00100000-\U0010FFFD]")  # Supplementary PUA-B
_ZW_PATTERN = re.compile(r"[​-‏‪-‮⁠-⁯﻿]")  # zero-width
_FULLWIDTH_SPACE = re.compile(r"　")                 # 전각 공백
_CTRL_PATTERN = re.compile(r"[\x00-\x08\x0B-\x1F\x7F]")
_MULTI_WS = re.compile(r"[ \t]{2,}")


def clean_text(s: str) -> str:
    """ver6 r2: PUA · zero-width · 전각공백 · 제어문자 · 연속공백 모두 정규화.

    PUA 이체자는 제거 (Gemma 262K 토크나이저에 없음 → byte-fallback 회피).
    예: "거믄기름" → "거믄기름"  (원본 의미는 "검은 참깨 기름")
    """
    if not s:
        return ""
    s = s.replace("\r", " ").replace("\n", " ")
    s = _CTRL_PATTERN.sub("", s)
    s = _PUA_PATTERN.sub("", s)
    s = _PUA_A_PATTERN.sub("", s)
    s = _PUA_B_PATTERN.sub("", s)
    s = _ZW_PATTERN.sub("", s)
    s = _FULLWIDTH_SPACE.sub(" ", s)
    s = _MULTI_WS.sub(" ", s)
    return s.strip()


def extract_name(original: str, trans_ko: str) -> tuple[str, str]:
    """레코드 이름 추출 (원문 · 한국어)."""
    o = clean_text(original).split()[0] if original else ""
    t = clean_text(trans_ko).split()[0] if trans_ko else ""
    return o, t


def build_record_name(original: str, trans_ko: str) -> str:
    """표시용 이름: '한국어(원문)' 형태."""
    o = clean_text(original)
    t = clean_text(trans_ko)
    # 특이: CH/DH 는 "井華水 새배처엄기른우믈믈" 처럼 공백 뒤에 한국어가 붙음
    if o and t:
        # 중복 방지 휴리스틱
        return f"{t.split('(')[0].strip()}({o.split()[0]})" if "(" not in t else t.strip()
    return t or o or "?"


_TEXT_FIELDS_TO_NORMALIZE = ("original", "trans_ko", "trans_en", "up_path_nm", "annotation")


def load_all_records() -> list[dict]:
    """raw JSONL 읽고 텍스트 필드 전수 normalize (PUA · 제어문자 · 전각공백 제거).

    특히 `up_path_nm` 도 normalize — raw path 자체에 PUA 가 섞인 레코드 (예:
    `湯液篇卷之二 > 草部 上 > 蒲黃 부들 > 香蒲` 의 `\\ue9ea` 류) 를 SFT 답변에
    전파 시키지 않기 위함.
    """
    recs = []
    for p in sorted(RAW_DIR.glob("vol_*.jsonl")):
        with p.open() as f:
            for ln in f:
                d = json.loads(ln)
                for field in _TEXT_FIELDS_TO_NORMALIZE:
                    v = d.get(field)
                    if isinstance(v, str) and v:
                        d[field] = clean_text(v)
                recs.append(d)
    return recs


# ver7 (A1): first-wins Hanja→Korean aliasing root-cause fix.
# 아래 3개 상수는 build_hanja_to_korean_map 의 새 gate 에 의해 사용.
#   GENERIC_PREFIX_BLOCKLIST — Q5/Q7/Q9 원인 (예: 唐→해금사, 葉→...).
#   _POSTPOSITION_RE        — postposition path segment 잔재 5,251 rows 제거용.
GENERIC_PREFIX_BLOCKLIST = {"唐", "葉", "子", "根", "花", "皮", "實", "果", "肉"}
_POSTPOSITION_RE = re.compile(r"(의|은|는|이|가|을|를|과|와|에)$")


def build_hanja_to_korean_map(recs: list[dict]) -> dict[str, str]:
    """raw 의 각 레코드에서 `original 첫 토큰 (한자)` → `trans_ko 첫 토큰 (한국어)` 매핑 구축.

    citation path 의 한자 segment 를 한국어로 변환하는 데 사용.
    예: '內景篇卷之一' → '내경편 권1', '蒲黃' → '포황', '香蒲' → '향포'.

    AA 레벨은 편명 전체 (예: '內景篇卷之一') 를 직접 매핑.
    일반 레벨은 original 의 first token 과 trans_ko 의 first token 페어.

    ver7 (A1) 3-gate 추가:
      1. GENERIC_PREFIX_BLOCKLIST — 단일 generic 한자 prefix 차단
         (이전 버그: 唐 → 해금사 식 first-wins aliasing 이 Q5/Q6(B)/Q7/Q9/Q10 동시 유발)
      2. 단일 한자 key 는 CC/BB 가 아니면 차단 (너무 넓은 scope)
      3. 한국어 postposition (의/은/는/이/가/을/를/과/와/에) strip
         (path segment 조사 잔재 5,251 rows 완화)
    """
    m: dict[str, str] = {}
    for r in recs:
        lvl = r.get("content_level")
        original = r.get("original") or ""
        trans_ko = r.get("trans_ko") or ""
        if not original or not trans_ko:
            continue
        # AA 레벨은 편명 전체 매핑
        if lvl == "AA":
            m[original.strip()] = trans_ko.strip()
            continue
        # 일반: 첫 토큰 vs 첫 토큰
        orig_tokens = original.split()
        ko_base = trans_ko.split("(")[0].strip()
        ko_tokens = ko_base.split()
        if not orig_tokens or not ko_tokens:
            continue
        han = orig_tokens[0]
        ko = ko_tokens[0]
        # gate 1: generic 1자 prefix 차단 (Q5/Q7/Q9 원인)
        if han in GENERIC_PREFIX_BLOCKLIST:
            continue
        # gate 2: 단일 한자 key 는 CC/BB 가 아니면 차단
        if len(han) == 1 and lvl not in ("CC", "BB"):
            continue
        # gate 3: 한국어 postposition strip (e.g. "두창의" → "두창")
        ko = _POSTPOSITION_RE.sub("", ko)
        if not ko:
            continue
        # 이미 있는 매핑과 충돌 시 첫 것 유지 (동명 처방/약재의 편차 무시)
        if han not in m and han != ko:
            m[han] = ko
    return m


def path_to_korean(path: str, han2ko: dict[str, str]) -> str:
    """citation path 의 한자 segment 를 한국어로 변환 (각 segment 는 AA 레벨 편명일 수 있음).

    예: '內景篇卷之一 > 身形 > 養性延年藥餌 > 瓊玉膏'
        → '내경편 권1 > 신형 > 양성연년약이 > 경옥고'

    '雜病篇卷之十一' 같은 복합 편명도 통째로 먼저 매핑 후 실패 시에만 토큰 분해.
    """
    if not path:
        return path
    segments = [s.strip() for s in path.split(">")]
    out = []
    for seg in segments:
        # 1. segment 전체가 매핑되는지 (AA 레벨 편명 등)
        if seg in han2ko:
            out.append(han2ko[seg])
            continue
        tokens = seg.split()
        if not tokens:
            out.append(seg)
            continue
        head = tokens[0]
        # 2. 첫 토큰 전체 매핑
        ko_head = han2ko.get(head, head)
        if len(tokens) == 1:
            out.append(ko_head)
        else:
            # 3. "蒲黃 부들" 같은 한자+한국어 혼합 — 한자 부분만 매핑 성공 시 한국어만 유지
            if ko_head != head:
                out.append(ko_head)
            else:
                out.append(seg)
    return " > ".join(out)


def make_messages(question: str, answer: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]


def build_index_by_path(recs: list[dict]) -> dict[str, list[dict]]:
    """up_path_nm 별 자식 레코드 목록."""
    idx = collections.defaultdict(list)
    for r in recs:
        up = r.get("up_path_nm")
        if up:
            idx[up].append(r)
    return idx


def find_ss_zz_children(rec: dict, name_parts: list[str], path_idx: dict) -> list[dict]:
    """DP/DD/CH/DK 레코드의 하위 SS/ZZ 자식 검색."""
    full_path = (rec.get("up_path_nm") or "") + " > " + name_parts[0] if name_parts else None
    if not full_path:
        return []
    candidates = path_idx.get(full_path, [])
    return [c for c in candidates if c.get("content_level") in ("SS", "ZZ")]


# ver7 (A5): shell-fallback detail 판별.
# detail 이 `"본문 ..."` prefix 로 시작하거나 whitespace 이후 `"참고"`/`"참조"` 로만 끝나면
# meaningful content 없음으로 간주 → builder 가 해당 row 자체를 skip.
_SHELL_FALLBACK_RE = re.compile(r"(^본문\s)|(\s(참고|참조)\s*\.?$)")


def _is_shell_fallback(detail: str) -> bool:
    if not detail:
        return True
    stripped = detail.strip()
    if not stripped:
        return True
    if stripped.startswith("본문 "):
        return True
    if _SHELL_FALLBACK_RE.search(stripped):
        return True
    return False


def collect_child_detail(children: list[dict], max_chars: int = 400) -> str:
    acc = []
    total = 0
    for c in children:
        t = clean_text(c.get("trans_ko") or "")
        if not t:
            continue
        if total + len(t) > max_chars:
            remain = max_chars - total
            if remain > 30:
                acc.append(t[:remain] + "…")
            break
        acc.append(t)
        total += len(t)
    return " ".join(acc)


# ──────────────────────────────────────────────────────────────────────────
# builders per category
# ──────────────────────────────────────────────────────────────────────────
def build_structure_pairs(recs: list[dict], path_idx: dict, rng: random.Random) -> list[dict]:
    pairs = []
    levels = {"AA", "BB", "OO", "Z2", "CC"}
    for i, r in enumerate(recs):
        lvl = r.get("content_level")
        if lvl not in levels:
            continue
        original = clean_text(r.get("original") or "")
        trans = clean_text(r.get("trans_ko") or "")
        if not original and not trans:
            continue
        # 중요: cite path 는 raw up_path_nm 과 동일한 한자 경로여야 함 (raw 검증 통과).
        # 이전 버그: trans_ko 를 우선 사용해서 "內景篇卷之一 > 東醫寶鑑 總目 > 부인" 같은 한-한 혼합 생성
        # 수정: name_han 은 original 첫 토큰, name_ko 는 trans 첫 토큰. path 에는 name_han 만 사용.
        name_han = original.split()[0] if original else ""
        name_ko_raw = trans.split("(")[0].strip() if trans else ""
        # name_ko 가 너무 긴 서술이면 (10 토큰 이상 또는 20자 이상) 첫 토큰만 사용
        name_ko_tokens = name_ko_raw.split()
        if not name_ko_tokens:
            name_ko = name_han
        elif len(name_ko_raw) > 20 or len(name_ko_tokens) >= 5:
            # 긴 서술 제목 — 첫 1~2 token 만 표시 (question echo 방지)
            name_ko = " ".join(name_ko_tokens[:2])
        else:
            name_ko = name_ko_raw
        # 표시용 이름: "한국어(한자)"
        if name_han and name_ko and name_han != name_ko:
            name_display = f"{name_ko}({name_han})"
        else:
            name_display = name_ko or name_han
        if not name_display:
            continue

        # AA 레벨은 up_path_nm=None · original 자체가 최상위 편명 (예: "內景篇卷之一",
        # 일부 볼륨은 "雜病篇 卷之一" 처럼 공백 포함). split 없이 통째로 사용.
        if lvl == "AA":
            parent = "동의보감(최상위 편성)"
            aa_full = clean_text(r.get("original") or "")
            cite = aa_full  # raw 의 편명 up_path_nm 과 정확 일치해야 함
            # 표시용 이름도 공백 포함 원문 살리기
            name_display = f"{clean_text(r.get('trans_ko') or '')}({aa_full})" if aa_full else name_display
        else:
            parent = r.get("up_path_nm") or "동의보감"
            cite = f"{parent} > {name_han}" if name_han else parent
        level_desc = LEVEL_DESC.get(lvl, "항목")

        # 자식 일부로 본문 추출
        children = path_idx.get(cite, [])
        children_sszz = [c for c in children if c.get("content_level") in ("SS", "ZZ")]
        excerpt = collect_child_detail(children_sszz, max_chars=200) or trans[:200]
        # AA 편명은 자식 BB/XX/Z2 목록을 excerpt 대신 사용 (편명 자체 자식 SS 가 없음)
        if lvl == "AA":
            child_bb = [c for c in children if c.get("content_level") in ("BB", "XX", "Z2", "OO")]
            bb_names = []
            for c in child_bb[:15]:
                ct = clean_text(c.get("trans_ko") or "")
                cn = ct.split("(")[0].strip().split()[:1]
                if cn:
                    bb_names.append(cn[0])
            if bb_names:
                excerpt = f"이 편은 {' · '.join(bb_names)} 등의 대주제를 담고 있습니다."
            elif not excerpt.strip():
                excerpt = f"동의보감의 {name_display} 편입니다."
        # 빈약한 detail (< 20자) 이면 structure QA 가 무의미 → skip
        if len(excerpt.strip()) < 20:
            continue

        q_tpl = STRUCTURE_Q_TEMPLATES[i % len(STRUCTURE_Q_TEMPLATES)]
        a_tpl = STRUCTURE_A_TEMPLATES[i % len(STRUCTURE_A_TEMPLATES)]
        question = q_tpl.format(name=name_display)
        answer = a_tpl.format(
            name=name_display, parent=parent, level_desc=level_desc,
            excerpt=excerpt[:280] or "본문 설명 참고", cite=cite,
        )
        pairs.append({
            "id": f"struct_{r.get('volume_id','?')}_{r.get('content_seq','?')}",
            "category": "structure",
            "subcat": lvl,
            "up_path_nm": cite,
            "question": question,
            "assistant": answer,
            "messages": make_messages(question, answer),
        })
    return pairs


def build_prescription_pairs(recs: list[dict], path_idx: dict, rng: random.Random) -> tuple[list[dict], set[str]]:
    pairs = []
    formula_whitelist: set[str] = set()
    for i, r in enumerate(recs):
        lvl = r.get("content_level")
        if lvl not in ("DP", "EP"):
            continue
        original = clean_text(r.get("original") or "")
        trans = clean_text(r.get("trans_ko") or "")
        if not original and not trans:
            continue
        name_han_tokens = original.split() if original else []
        name_han = name_han_tokens[0] if name_han_tokens else ""
        trans_first = trans.split("(")[0].strip()
        name_ko_tokens = trans_first.split() if trans_first else []
        name_ko = name_ko_tokens[0] if name_ko_tokens else name_han
        if not name_han and not name_ko:
            continue
        # 일반어 skip (DP 레코드 중 "主治 ..." 같은 잘못된 헤더가 섞인 엣지 케이스)
        GENERIC_SKIP = {"主治", "主方", "治療", "效能", "處方", "一方", "又方"}
        if name_han in GENERIC_SKIP:
            continue
        name_display = f"{name_ko}({name_han})" if name_han and name_ko and name_han != name_ko else (name_ko or name_han)
        if name_han:
            formula_whitelist.add(name_han)
        if name_ko:
            formula_whitelist.add(name_ko)
        parent = r.get("up_path_nm") or "동의보감"
        cite = f"{parent} > {name_han}" if name_han else f"{parent} > {name_ko}"

        children_sszz = path_idx.get(cite, [])
        detail = collect_child_detail(
            [c for c in children_sszz if c.get("content_level") in ("SS", "ZZ")],
            max_chars=400,
        )
        if not detail:
            detail = "해당 처방은 본문 인용 참조"
        excerpt = detail[:200]

        # context: 부모 path 의 마지막 element (증상 영역)
        context = parent.split(" > ")[-1] if " > " in parent else parent

        q_tpl = PRESCRIPTION_Q_TEMPLATES[i % len(PRESCRIPTION_Q_TEMPLATES)]
        a_tpl = PRESCRIPTION_A_TEMPLATES[i % len(PRESCRIPTION_A_TEMPLATES)]
        free_form = rng.random() < 0.2

        question = q_tpl.format(name=name_display, parent=parent.split(" > ")[0], context=context)
        if free_form:
            answer = (
                f"{name_display} 은 동의보감 {cite} 에 수록된 처방입니다. "
                f"{detail} 구체 용량은 본문 또는 전문 한의사의 판단을 따르십시오. [출처: {cite}]"
            )
        else:
            answer = a_tpl.format(
                name=name_display, cite=cite, detail=detail,
                excerpt=excerpt, parent=parent.split(" > ")[0],
            )

        pairs.append({
            "id": f"rx_{r.get('volume_id','?')}_{r.get('content_seq','?')}",
            "category": "prescription",
            "subcat": lvl,
            "up_path_nm": cite,
            "question": question,
            "assistant": answer,
            "messages": make_messages(question, answer),
        })
    return pairs, formula_whitelist


def build_herb_pairs(recs: list[dict], path_idx: dict, rng: random.Random) -> tuple[list[dict], set[str]]:
    pairs = []
    herb_whitelist: set[str] = set()
    for i, r in enumerate(recs):
        lvl = r.get("content_level")
        if lvl not in ("CH", "DH"):
            continue
        original = clean_text(r.get("original") or "")
        trans = clean_text(r.get("trans_ko") or "")
        if not original and not trans:
            continue
        tokens_o = original.split()
        tokens_t = trans.split()
        name_han = tokens_o[0] if tokens_o else ""
        korean_alias = " ".join(tokens_o[1:]) if len(tokens_o) > 1 else (tokens_t[0] if tokens_t else "")
        name_display = trans.split("(")[0].strip() or name_han
        herb_whitelist.add(name_han)
        if korean_alias:
            herb_whitelist.add(korean_alias)
        parent = r.get("up_path_nm") or "동의보감 탕액편"
        cite = f"{parent} > {name_han}" if name_han else parent

        children_sszz = path_idx.get(cite, [])
        detail = collect_child_detail(
            [c for c in children_sszz if c.get("content_level") in ("SS", "ZZ")],
            max_chars=350,
        )
        # ver7 (A5): fallback `detail = "본문 {cite} 참고"` 가 HERB_A_TEMPLATES[1]
        # (`"본문에는 {detail}"`) 에 끼워지면 → "본문에는 본문 ... 참고" self-ref 생성
        # (154 row 원인). meaningful content 없으면 skip 하여 shell-answer 제거.
        if not detail or _is_shell_fallback(detail):
            continue

        q_tpl = HERB_Q_TEMPLATES[i % len(HERB_Q_TEMPLATES)]
        a_tpl = HERB_A_TEMPLATES[i % len(HERB_A_TEMPLATES)]
        question = q_tpl.format(name=name_display, parent=parent.split(" > ")[-1])
        answer = a_tpl.format(
            name=name_display, korean_alias=korean_alias or "—",
            parent=parent.split(" > ")[-1], detail=detail, cite=cite,
        )

        pairs.append({
            "id": f"herb_{r.get('volume_id','?')}_{r.get('content_seq','?')}",
            "category": "herb",
            "subcat": lvl,
            "up_path_nm": cite,
            "question": question,
            "assistant": answer,
            "messages": make_messages(question, answer),
        })
    return pairs, herb_whitelist


def build_acupoint_pairs(recs: list[dict], path_idx: dict, rng: random.Random) -> tuple[list[dict], set[str]]:
    pairs = []
    acu_whitelist: set[str] = set()
    for i, r in enumerate(recs):
        if r.get("content_level") != "DK":
            continue
        original = clean_text(r.get("original") or "")
        trans = clean_text(r.get("trans_ko") or "")
        name_han = original.split()[0] if original else ""
        name_ko = trans.split("(")[0].strip() or name_han
        acu_whitelist.add(name_han)
        acu_whitelist.add(name_ko)
        parent = r.get("up_path_nm") or "침구편"
        cite = f"{parent} > {name_han}" if name_han else parent

        children_sszz = path_idx.get(cite, [])
        detail = collect_child_detail(
            [c for c in children_sszz if c.get("content_level") in ("SS", "ZZ")],
            max_chars=300,
        )
        if not detail:
            detail = f"{cite} 본문 참조"

        q_tpl = ACUPOINT_Q_TEMPLATES[i % len(ACUPOINT_Q_TEMPLATES)]
        a_tpl = ACUPOINT_A_TEMPLATES[i % len(ACUPOINT_A_TEMPLATES)]
        meridian = parent.split(" > ")[-1]
        name_display = f"{name_ko}({name_han})" if name_han and name_ko != name_han else name_ko

        question = q_tpl.format(name=name_display, parent=meridian)
        answer = a_tpl.format(name=name_display, parent=meridian, detail=detail, cite=cite)

        pairs.append({
            "id": f"acu_{r.get('volume_id','?')}_{r.get('content_seq','?')}",
            "category": "acupoint",
            "subcat": "DK",
            "up_path_nm": cite,
            "question": question,
            "assistant": answer,
            "messages": make_messages(question, answer),
        })
    return pairs, acu_whitelist


def build_symptom_pairs(recs: list[dict], path_idx: dict, rng: random.Random) -> list[dict]:
    pairs = []
    for i, r in enumerate(recs):
        if r.get("content_level") != "DD":
            continue
        original = clean_text(r.get("original") or "")
        trans = clean_text(r.get("trans_ko") or "")
        name_han = original.split()[0] if original else ""
        name_ko = trans.split("(")[0].strip() or name_han
        name_display = f"{name_ko}({name_han})" if name_han and name_ko != name_han else name_ko
        parent = r.get("up_path_nm") or "잡병편"
        cite = f"{parent} > {name_han}" if name_han else parent

        children_sszz = path_idx.get(cite, [])
        detail = collect_child_detail(
            [c for c in children_sszz if c.get("content_level") in ("SS", "ZZ")],
            max_chars=350,
        )
        if not detail:
            detail = f"{cite} 원문 참조"

        q_tpl = SYMPTOM_Q_TEMPLATES[i % len(SYMPTOM_Q_TEMPLATES)]
        a_tpl = SYMPTOM_A_TEMPLATES[i % len(SYMPTOM_A_TEMPLATES)]
        question = q_tpl.format(name=name_display, parent=parent.split(" > ")[0])
        answer = a_tpl.format(
            name=name_display, parent=parent.split(" > ")[0],
            detail=detail, cite=cite,
        )

        pairs.append({
            "id": f"sym_{r.get('volume_id','?')}_{r.get('content_seq','?')}",
            "category": "symptom",
            "subcat": "DD",
            "up_path_nm": cite,
            "question": question,
            "assistant": answer,
            "messages": make_messages(question, answer),
        })
    return pairs


def build_diagram_pairs(recs: list[dict], path_idx: dict, rng: random.Random) -> list[dict]:
    """XX 레벨 중 단방(單方) 설명 제외 · 명당부위·관형찰색·각 부 설명 등 고유 지식 포함.

    단방 설명은 짧은 반복 문구 ("모두 N종이다" 등) 라 학습 가치 낮음.
    명당부위도·관형찰색도·각 부(水部/穀部 등) 설명은 한 줄이라도 고유 의학 지식.
    """
    pairs = []
    for i, r in enumerate(recs):
        if r.get("content_level") != "XX":
            continue
        up = r.get("up_path_nm") or ""
        # 단방 설명 83건 제외
        if up.endswith("單方") or "단방" in (r.get("trans_ko") or "")[:10]:
            continue
        trans = clean_text(r.get("trans_ko") or "")
        if len(trans) < 20:
            continue
        # 이름 추출: parent 의 마지막 segment
        name_ko = up.split(" > ")[-1] if " > " in up else up
        parent = " > ".join(up.split(" > ")[:-1]) or "동의보감"
        cite = up  # XX 레코드는 self-path 가 아니라 parent 아래 meta

        q_tpl = DIAGRAM_Q_TEMPLATES[i % len(DIAGRAM_Q_TEMPLATES)]
        a_tpl = DIAGRAM_A_TEMPLATES[i % len(DIAGRAM_A_TEMPLATES)]
        question = q_tpl.format(name=name_ko, parent=parent)
        answer = a_tpl.format(name=name_ko, parent=parent, detail=trans[:400], cite=cite)
        pairs.append({
            "id": f"diag_{r.get('volume_id','?')}_{r.get('content_seq','?')}",
            "category": "diagram",
            "subcat": "XX",
            "up_path_nm": cite,
            "question": question,
            "assistant": answer,
            "messages": make_messages(question, answer),
        })
    return pairs


def build_passage_pairs(recs: list[dict], rng: random.Random, target: int = 7000) -> list[dict]:
    """SS/ZZ 본문에서 의미 밀도 있는 것만 샘플링."""
    cand = [
        r for r in recs
        if r.get("content_level") in ("SS", "ZZ")
        and len(clean_text(r.get("trans_ko") or "")) >= 80  # 최소 80자 이상
    ]
    rng.shuffle(cand)
    cand = cand[:target]
    pairs = []
    for i, r in enumerate(cand):
        trans = clean_text(r.get("trans_ko") or "")
        parent = r.get("up_path_nm") or "동의보감"
        cite = parent
        q_tpl = PASSAGE_Q_TEMPLATES[i % len(PASSAGE_Q_TEMPLATES)]
        a_tpl = PASSAGE_A_TEMPLATES[i % len(PASSAGE_A_TEMPLATES)]
        # ver6 r3: Q 에 한문 원문 대신 한국어 번역 앞 120자 사용
        question = q_tpl.format(parent=parent, original=trans[:120], cite=cite)
        # r3: excerpt 500 → 1,000자 (긴 본문 손실 완화)
        answer = a_tpl.format(
            excerpt=trans[:1000], parent=parent.split(" > ")[0], cite=cite,
        )
        pairs.append({
            "id": f"pass_{r.get('volume_id','?')}_{r.get('content_seq','?')}",
            "category": "passage",
            "subcat": r.get("content_level"),
            "up_path_nm": cite,
            "question": question,
            "assistant": answer,
            "messages": make_messages(question, answer),
        })
    return pairs


def build_paraphrase_pairs(rng: random.Random) -> list[dict]:
    pairs = []
    for seed in PARAPHRASE_SEEDS:
        qs = seed["q_variants"]
        a_s = seed["a_variants"]
        # 모든 (q, a) 조합은 많으므로 paired 만: q_i × a_j 샘플 — 카테고리당 5*5=25 → 2 카테고리 = 75, 3 카테고리 = 75. 적당히 줄이기.
        for i, q in enumerate(qs):
            for j, a in enumerate(a_s):
                pairs.append({
                    "id": f"para_{seed['q_variants'][0][:6]}_{i}_{j}",
                    "category": "paraphrase",
                    "subcat": "basic_fact",
                    "up_path_nm": None,
                    "question": q,
                    "assistant": a,
                    "messages": make_messages(q, a),
                })
    return pairs


def build_refusal_pairs(rng: random.Random, n_oos: int = 400, n_safety: int = 400) -> list[dict]:
    pairs = []
    # OOS 는 동일 질문을 여러 답변 템플릿으로 변형, 반대로 여러 질문을 동일 답변으로 묶어 다양성 확보
    for i in range(n_oos):
        q = REFUSAL_OOS_Q[i % len(REFUSAL_OOS_Q)]
        a_tpl = REFUSAL_OOS_A[i % len(REFUSAL_OOS_A)]
        closing = REFUSAL_CLOSINGS[i % len(REFUSAL_CLOSINGS)]
        a = a_tpl.format(closing=closing)
        pairs.append({
            "id": f"refuse_oos_{i}",
            "category": "refusal_oos",
            "subcat": "out_of_scope",
            "up_path_nm": None,
            "question": q,
            "assistant": a,
            "messages": make_messages(q, a),
        })
    for i in range(n_safety):
        q = REFUSAL_SAFETY_Q[i % len(REFUSAL_SAFETY_Q)]
        a_tpl = REFUSAL_SAFETY_A[i % len(REFUSAL_SAFETY_A)]
        closing = REFUSAL_CLOSINGS[i % len(REFUSAL_CLOSINGS)]
        a = a_tpl.format(closing=closing)
        pairs.append({
            "id": f"refuse_safety_{i}",
            "category": "refusal_safety",
            "subcat": "safety",
            "up_path_nm": None,
            "question": q,
            "assistant": a,
            "messages": make_messages(q, a),
        })
    return pairs


# ──────────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "experiments/dongui_bogam/data/sft/phaseB_qa_v6_corpus.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--passage-target", type=int, default=7000)
    ap.add_argument("--dry-run", action="store_true")
    # ver7: optional override for raw book dir (keeps default behaviour intact).
    ap.add_argument("--raw-dir", type=Path, default=None,
                    help="override RAW_DIR (default: data/raw/mediclassics_unified/book_008)")
    # ver7: principle 1/2 post-check logging path (default next to --out).
    ap.add_argument("--principle-log", type=Path, default=None,
                    help="principle-violation jsonl (default: <out parent>/_workspace/principle_violations.jsonl)")
    args = ap.parse_args()

    if args.raw_dir is not None:
        global RAW_DIR
        RAW_DIR = args.raw_dir

    rng = random.Random(args.seed)
    print("[1/7] load all records...")
    recs = load_all_records()
    print(f"  total records: {len(recs)}")
    path_idx = build_index_by_path(recs)
    print(f"  unique parent paths: {len(path_idx)}")
    # ver6 r3: 한자 path segment → 한국어 자동 매핑
    han2ko = build_hanja_to_korean_map(recs)
    print(f"  한자→한국어 segment map: {len(han2ko)} entries")

    print("[2/7] build structure pairs (AA/BB/OO/Z2/CC)...")
    struct_pairs = build_structure_pairs(recs, path_idx, rng)
    print(f"  structure: {len(struct_pairs)}")

    print("[3/7] build prescription pairs (DP/EP)...")
    rx_pairs, formula_wl = build_prescription_pairs(recs, path_idx, rng)
    print(f"  prescription: {len(rx_pairs)} (formula whitelist: {len(formula_wl)})")

    print("[4/7] build herb pairs (CH/DH)...")
    herb_pairs, herb_wl = build_herb_pairs(recs, path_idx, rng)
    print(f"  herb: {len(herb_pairs)} (herb whitelist: {len(herb_wl)})")

    print("[5/7] build acupoint pairs (DK)...")
    acu_pairs, acu_wl = build_acupoint_pairs(recs, path_idx, rng)
    print(f"  acupoint: {len(acu_pairs)} (acupoint whitelist: {len(acu_wl)})")

    print("[6/8] build symptom pairs (DD) + passage sampling (SS/ZZ)...")
    sym_pairs = build_symptom_pairs(recs, path_idx, rng)
    passage_pairs = build_passage_pairs(recs, rng, target=args.passage_target)
    print(f"  symptom: {len(sym_pairs)}  passage: {len(passage_pairs)}")

    print("[7/8] build diagram pairs (XX 중 단방 제외 — 명당·관형·각부)...")
    diagram_pairs = build_diagram_pairs(recs, path_idx, rng)
    print(f"  diagram: {len(diagram_pairs)}")

    print("[8/8] build paraphrase + refusal...")
    para_pairs = build_paraphrase_pairs(rng)
    refuse_pairs = build_refusal_pairs(rng)
    print(f"  paraphrase: {len(para_pairs)}  refusal: {len(refuse_pairs)}")

    all_pairs = (
        struct_pairs + rx_pairs + herb_pairs + acu_pairs
        + sym_pairs + passage_pairs + diagram_pairs + para_pairs + refuse_pairs
    )
    rng.shuffle(all_pairs)

    # ver6 r3: Korean-first citation post-process
    cite_pattern = re.compile(r"(\[?출처:\s*)([^\]\)\n]+?)(\s*[\]\)]|\n|$)")
    path_inline_pattern = re.compile(r"([一-鿿][一-鿿가-힣\s·ㆍ]*(?:\s*>\s*[一-鿿][一-鿿가-힣\s·ㆍ]*){1,})")

    # 공백 포함 AA 편명 (예: "雜病篇 卷之一") · multi-char 한자 토큰은 literal replace
    # 단 3자 이상 한자 전용이어야 처방명 병기 "血結胸" 등과 혼동 방지
    aa_compound_keys = sorted(
        [k for k, v in han2ko.items() if (" " in k or len(k) >= 5) and k != v],
        key=len, reverse=True,
    )

    def ko_cite(match: re.Match) -> str:
        prefix, path, suffix = match.group(1), match.group(2).strip(), match.group(3)
        return f"{prefix}{path_to_korean(path, han2ko)}{suffix}"

    def ko_inline_path(match: re.Match) -> str:
        return path_to_korean(match.group(1).strip(), han2ko)

    # 숫자 범위 표기 통일: `1~2` → `1-2`
    tilde_pattern = re.compile(r"(\d)~(\d)")

    n_changed = 0
    for p in all_pairs:
        up = p.get("up_path_nm")
        if up:
            ko_up = path_to_korean(up, han2ko)
            if ko_up != up:
                p["up_path_nm"] = ko_up
                n_changed += 1
        a = p["assistant"]
        q = p["question"]
        a2 = cite_pattern.sub(ko_cite, a)
        q2 = cite_pattern.sub(ko_cite, q)
        a2 = path_inline_pattern.sub(ko_inline_path, a2)
        q2 = path_inline_pattern.sub(ko_inline_path, q2)
        # 공백 포함 복합 한자 편명 replace (길이 DESC)
        for han_key in aa_compound_keys:
            if han_key in a2:
                a2 = a2.replace(han_key, han2ko[han_key])
            if han_key in q2:
                q2 = q2.replace(han_key, han2ko[han_key])
        # 숫자 범위 통일
        a2 = tilde_pattern.sub(r"\1-\2", a2)
        q2 = tilde_pattern.sub(r"\1-\2", q2)
        if a2 != a or q2 != q:
            p["assistant"] = a2
            p["question"] = q2
            p["messages"] = make_messages(q2, a2)
    print(f"  Korean-first post-process: {n_changed} pairs got up_path_nm 변환")

    # coverage 집계 (Korean path 로 재계산)
    covered_refs = set()
    for p in all_pairs:
        up = p.get("up_path_nm")
        if up:
            covered_refs.add(up)
    # self path coverage — AA 레코드는 original 자체가 self (space 포함 가능)
    def self_path_raw(r: dict) -> str:
        lvl = r.get("content_level")
        orig = clean_text(r.get("original") or "")
        if lvl == "AA":
            return orig  # AA 는 편명 원문 자체
        parent = r.get("up_path_nm") or ""
        toks = orig.split()
        if not toks:
            return parent
        return parent + " > " + toks[0]

    def self_path_ko(r: dict) -> str:
        return path_to_korean(self_path_raw(r), han2ko)

    covered_records = sum(
        1 for r in recs
        if (r.get("up_path_nm") or "") in covered_refs
        or self_path_raw(r) in covered_refs
        or path_to_korean(r.get("up_path_nm") or "", han2ko) in covered_refs
        or self_path_ko(r) in covered_refs
    )

    # ver7: principle 1·2 post-check (기획서 §3.1.2).
    #   원칙 1 — assistant 에 whitelist 엔티티 ≥ 1 (refusal 은 면제)
    #   원칙 2 — [출처: ...] 패턴이 실재 up_path_nm 집합에 존재
    # violations 은 drop 하되 먼저 _workspace/principle_violations.jsonl 로 로깅.
    entity_whitelist = (formula_wl | herb_wl | acu_wl)
    real_up_paths: set[str] = set()
    # (a) raw record up_path_nm 원본 + 한국어 path
    for r in recs:
        up = r.get("up_path_nm")
        if up:
            real_up_paths.add(up)
            real_up_paths.add(path_to_korean(up, han2ko))
    # (b) builder 가 만든 cite path (structure/prescription/herb/...) 도 유효로 간주
    for p in all_pairs:
        up = p.get("up_path_nm")
        if up:
            real_up_paths.add(up)

    cite_extract_re = re.compile(
        r"\[출처:\s*([^\]]+)\]|（출처:\s*([^）]+)）|\(출처:\s*([^)]+)\)"
    )
    refusal_cats = {"refusal_oos", "refusal_safety"}

    def _row_passes_principles(row: dict) -> tuple[bool, str]:
        answer = row.get("assistant") or ""
        cat = row.get("category")
        # 원칙 1: refusal 이외 카테고리는 whitelist 엔티티 ≥ 1 필요
        if cat not in refusal_cats and entity_whitelist:
            if not any(e and e in answer for e in entity_whitelist):
                return False, "principle1_no_entity"
        # 원칙 2: 인용은 실재 up_path_nm 여야 함
        for match in cite_extract_re.finditer(answer):
            cite = next((c for c in match.groups() if c), "").strip()
            if not cite:
                continue
            if cite in real_up_paths:
                continue
            # prefix 매칭 fallback — 일부 답변 cite 가 path 의 앞 일부만 사용
            if any(cite in rp or rp in cite for rp in real_up_paths):
                continue
            return False, "principle2_bad_cite"
        return True, ""

    kept_pairs: list[dict] = []
    violations: list[dict] = []
    for p in all_pairs:
        ok, reason = _row_passes_principles(p)
        if ok:
            kept_pairs.append(p)
        else:
            violations.append({"id": p.get("id"), "category": p.get("category"),
                                "reason": reason,
                                "question": p.get("question", "")[:200],
                                "assistant": p.get("assistant", "")[:400]})
    dropped = len(all_pairs) - len(kept_pairs)
    print(f"  [principles] kept={len(kept_pairs)}  dropped={dropped}  "
          f"(p1_no_entity={sum(1 for v in violations if v['reason']=='principle1_no_entity')}, "
          f"p2_bad_cite={sum(1 for v in violations if v['reason']=='principle2_bad_cite')})")

    if violations:
        principle_log = args.principle_log
        if principle_log is None:
            principle_log = args.out.parent / "_workspace" / "principle_violations.jsonl"
        principle_log.parent.mkdir(parents=True, exist_ok=True)
        with principle_log.open("w") as f:
            for v in violations:
                f.write(json.dumps(v, ensure_ascii=False) + "\n")
        print(f"  [principles] violation log → {principle_log}")

    # drop 후 최종 리스트로 교체
    all_pairs = kept_pairs

    # stats
    cat_cnt = collections.Counter(p["category"] for p in all_pairs)
    stats = {
        "total_pairs": len(all_pairs),
        "by_category": dict(cat_cnt),
        "unique_up_path_refs": len(covered_refs),
        "covered_records": covered_records,
        "raw_records_total": len(recs),
        "coverage_ratio": round(covered_records / len(recs), 4),
        "formula_whitelist_size": len(formula_wl),
        "herb_whitelist_size": len(herb_wl),
        "acupoint_whitelist_size": len(acu_wl),
        "principle_violations_dropped": dropped,
    }
    print(f"\n=== stats ===\n{json.dumps(stats, ensure_ascii=False, indent=2)}")

    if args.dry_run:
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for p in all_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    # whitelists 저장
    wl_dir = args.out.parent
    with (wl_dir / "entity_whitelist_v6.yaml").open("w") as f:
        import yaml
        yaml.safe_dump({
            "formulas": sorted(formula_wl),
            "herbs": sorted(herb_wl),
            "acupoints": sorted(acu_wl),
        }, f, allow_unicode=True)

    stats_path = args.out.with_suffix(".stats.json")
    with stats_path.open("w") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"\n[write] {len(all_pairs)} pairs → {args.out}")
    print(f"[write] whitelists → {wl_dir / 'entity_whitelist_v6.yaml'}")
    print(f"[write] stats → {stats_path}")


if __name__ == "__main__":
    main()
