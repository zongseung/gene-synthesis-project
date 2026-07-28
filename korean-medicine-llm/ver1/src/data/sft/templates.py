"""SFT answer templates — HanMed-LLM ver5 Phase B.

spec: docs/ver5/08_sft_build_plan.md §4.2

원칙
----
1. body 의 모든 `{source_quote_*}` slot 은 seed 의 source_records 에서 복사된
   literal substring 으로만 채운다 (build_sft_qa.py 가 assert).
2. fact 값 (이름·연도·한자) 은 factsheet 또는 whitelist 에서만 치환.
3. 자유 생성 X. template body 는 고정 문장.
4. 각 template 은 명시적 citation_tag 로 끝난다 (safety/out_of_scope 제외).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class Template:
    id: str
    category: str
    body: str
    required_keys: List[str]
    citation_tag: str = ""


TEMPLATES: Dict[str, Template] = {

    # ───────────────────────────── in_scope_basic ─────────────────────────────

    "author_fact": Template(
        id="author_fact",
        category="in_scope_basic",
        body=(
            "동의보감은 조선 중기의 어의(御醫) 허준({author_hanja}) 이 편찬한 의서입니다.\n\n"
            "동의보감 서문에는 저자 서명이 다음과 같이 기록되어 있습니다:\n"
            "\"{source_quote_0}\"\n\n"
            "서명에서 드러나듯 허준은 어의로서 충근정량호성공신·숭록대부·양평군의 관호와 "
            "작위를 지닌 채 선조(宣祖)의 하교를 받들어 이 책을 저술하였습니다. "
            "허준은 1596년(병신년) 선조의 명을 받아 편찬을 시작하였고, 광해군 2년(1610) 에 "
            "완성한 뒤 1613년 내의원에서 간행·반포되었습니다."
        ),
        required_keys=["author_hanja", "source_quote_0"],
        citation_tag="[출처: 동의보감 내경편 권1 서문 (KIOM mediclassics)]",
    ),

    "king_command": Template(
        id="king_command",
        category="in_scope_basic",
        body=(
            "동의보감은 조선 선조({reign_hanja}) 의 명으로 편찬이 시작되었습니다.\n\n"
            "동의보감 서문에는 편찬 발의 경위가 다음과 같이 기록되어 있습니다:\n"
            "\"{source_quote_0}\"\n\n"
            "이와 같이 선조는 병신년(1596) 에 태의 허준을 불러 의서 편찬을 하교하였고, "
            "편찬 작업은 곧 시작되어 여러 해에 걸쳐 진행된 끝에 광해군 2년(1610) 에 "
            "완성되었습니다. 서문에서 선조는 '선종대왕(宣宗大王)' 이라는 묘호로도 지칭됩니다."
        ),
        required_keys=["reign_hanja", "source_quote_0"],
        citation_tag="[출처: 동의보감 내경편 권1 서문 (KIOM mediclassics)]",
    ),

    "complete_year": Template(
        id="complete_year",
        category="in_scope_basic",
        body=(
            "동의보감은 광해군 2년, 곧 경술년(1610) 에 완성되었습니다.\n\n"
            "서문에는 완성 경위가 다음과 같이 기록되어 있습니다:\n"
            "\"{source_quote_0}\"\n\n"
            "편찬이 1596년 선조의 명으로 시작된 뒤 정유재란으로 한 차례 중단되었다가, "
            "선조 승하 이후 허준이 단독으로 작업을 이어가 최종적으로 25권의 책으로 완성되었습니다. "
            "그 뒤 내의원에서 간행되어 1613년에 반포되었습니다."
        ),
        required_keys=["source_quote_0"],
        citation_tag="[출처: 동의보감 내경편 권1 서문 (KIOM mediclassics)]",
    ),

    "published_year": Template(
        id="published_year",
        category="in_scope_basic",
        body=(
            "동의보감은 만력(萬曆) 41년, 곧 1613년에 내의원에서 간행되어 반포되었습니다.\n\n"
            "간행 기록은 다음과 같이 명시되어 있습니다:\n"
            "\"{source_quote_0}\"\n\n"
            "1610년 경술년에 허준이 원고를 완성하여 광해군에게 진상한 뒤, 광해군이 "
            "내의원으로 하여금 관청을 설치하고 이를 간행하여 온 나라에 반포하도록 명한 "
            "결과입니다. 감교관은 통훈대부 이희헌과 윤지미였습니다."
        ),
        required_keys=["source_quote_0"],
        citation_tag="[출처: 동의보감 내경편 권1 서문 (KIOM mediclassics)]",
    ),

    "pyeon_structure": Template(
        id="pyeon_structure",
        category="in_scope_basic",
        body=(
            "동의보감은 크게 다섯 편(篇)으로 구성됩니다: "
            "내경편(內景篇) · 외형편(外形篇) · 잡병편(雜病篇) · 탕액편(湯液篇) · "
            "침구편(鍼灸篇) 의 5편입니다.\n\n"
            "동의보감 서문은 이 구성의 체계적 성격을 다음과 같이 표현합니다:\n"
            "\"{source_quote_0}\"\n\n"
            "각 편은 인체 내부(내경) → 외부(외형) → 병증(잡병) → 약물(탕액) → "
            "침구 치료(침구) 의 순서로 의학 지식을 체계적으로 배열한 것입니다. "
            "전체 분량은 모두 25권으로 이루어져 있습니다."
        ),
        required_keys=["source_quote_0"],
        citation_tag="[출처: 동의보감 내경편 권1 서문 (KIOM mediclassics)]",
    ),

    "num_volumes": Template(
        id="num_volumes",
        category="in_scope_basic",
        body=(
            "동의보감은 모두 25권으로 이루어져 있습니다.\n\n"
            "서문에는 완성 당시의 권수가 다음과 같이 명시되어 있습니다:\n"
            "\"{source_quote_0}\"\n\n"
            "내경편·외형편·잡병편·탕액편·침구편의 5편 체계 안에 총 25권이 배분되어, "
            "허준은 1610년 경술년에 이 25권의 원고를 완성하여 광해군에게 진상하였습니다."
        ),
        required_keys=["source_quote_0"],
        citation_tag="[출처: 동의보감 내경편 권1 서문 (KIOM mediclassics)]",
    ),

    "preface_author": Template(
        id="preface_author",
        category="in_scope_basic",
        body=(
            "동의보감의 서문은 이정구(李廷龜)가 지었습니다.\n\n"
            "서문 말미에는 다음과 같이 명시되어 있습니다:\n"
            "\"{source_quote_0}\"\n\n"
            "이정구는 당시 숭록대부·이조판서·홍문관대제학·예문관대제학 등 겸직을 맡은 "
            "고위 관료로, 광해군의 하교에 따라 동의보감 서문을 찬(撰)하였습니다. "
            "작성 시점은 만력 39년 신해년(1611) 초여름이었습니다."
        ),
        required_keys=["source_quote_0"],
        citation_tag="[출처: 동의보감 내경편 권1 서문 말미 (KIOM mediclassics)]",
    ),

    "helpers": Template(
        id="helpers",
        category="in_scope_basic",
        body=(
            "동의보감 편찬 초기에 허준을 도운 인물은 유의(儒醫) 정작(鄭碏) 과 "
            "태의(太醫) 양예수(楊禮壽)·김응탁(金應鐸)·이명원(李命源)·정예남(鄭禮男) 입니다.\n\n"
            "서문에는 편찬 인원 구성이 다음과 같이 기록되어 있습니다:\n"
            "\"{source_quote_0}\"\n\n"
            "다만 이들의 공동 편찬 작업은 1597년 정유재란을 만나 중단되었고, 이후 편찬은 "
            "허준이 단독으로 이어가 1610년에 완성하였습니다. 따라서 동의보감의 주저자는 "
            "허준이며, 이 다섯 명은 초기 편찬 보조자로 이해해야 합니다."
        ),
        required_keys=["source_quote_0"],
        citation_tag="[출처: 동의보감 내경편 권1 서문 (KIOM mediclassics)]",
    ),

    "jeongyu_interrupt": Template(
        id="jeongyu_interrupt",
        category="in_scope_basic",
        body=(
            "동의보감 편찬은 1597년 정유재란(丁酉再亂) 으로 인해 한 차례 중단되었습니다.\n\n"
            "서문에는 중단 사실이 다음과 같이 짧게 기록되어 있습니다:\n"
            "\"{source_quote_0}\"\n\n"
            "1596년 선조의 하교로 시작된 공동 편찬은 이듬해 전란으로 의사들이 흩어지면서 "
            "일시 멈췄고, 이후 선조는 허준에게 단독 편찬을 다시 명하였으며 허준은 "
            "대궐 소장 의서 500권을 고증 자료로 받았습니다. 편찬은 결국 광해군 2년(1610) 에 "
            "허준에 의해 완성되었습니다."
        ),
        required_keys=["source_quote_0"],
        citation_tag="[출처: 동의보감 내경편 권1 서문 (KIOM mediclassics)]",
    ),

    "reign_transition": Template(
        id="reign_transition",
        category="in_scope_basic",
        body=(
            "동의보감 편찬은 선조(宣祖) 대에 시작되어 광해군(光海君) 대에 완성되었습니다.\n\n"
            "서문은 이 왕대 교체를 다음과 같이 기록합니다:\n"
            "\"{source_quote_0}\"\n\n"
            "1596년 선조의 하교로 편찬이 시작되었고, 선조 승하(1608) 후 즉위한 "
            "광해군 재위 2년인 경술년(1610) 에 허준이 작업을 마쳤습니다. "
            "이후 1613년에 내의원에서 간행되어 반포되었습니다."
        ),
        required_keys=["source_quote_0"],
        citation_tag="[출처: 동의보감 내경편 권1 서문 (KIOM mediclassics)]",
    ),

    "three_principles": Template(
        id="three_principles",
        category="in_scope_basic",
        body=(
            "선조가 동의보감 편찬을 명할 때 강조한 원칙은 크게 세 가지입니다.\n\n"
            "첫째, 수양 우선·약물 차선의 원칙입니다. 서문은 선조의 하교를 다음과 같이 "
            "인용합니다:\n"
            "\"{source_quote_0}\"\n\n"
            "둘째, 기존 의서의 번다함을 줄이고 요점을 선별하라는 원칙입니다. "
            "셋째, 우리나라에서 많이 나는 약재(향약)를 활용하고 그 명칭을 병기하여 "
            "백성이 쉽게 알 수 있도록 하라는 원칙입니다:\n"
            "\"{source_quote_1}\"\n\n"
            "이 세 원칙은 동의보감의 구성 전반에 반영되어, 섭양(수양) 강조·간명한 체계·"
            "향약 활용이라는 특징으로 이어졌습니다."
        ),
        required_keys=["source_quote_0", "source_quote_1"],
        citation_tag="[출처: 동의보감 내경편 권1 서문 (KIOM mediclassics)]",
    ),

    "gwanghae_reward": Template(
        id="gwanghae_reward",
        category="in_scope_basic",
        body=(
            "광해군은 동의보감이 완성되어 진상되자 허준에게 태복마(太僕馬) 한 필을 "
            "하사하고, 내의원에 명을 내려 책을 간행·반포하게 하였습니다.\n\n"
            "서문에는 광해군의 하교가 다음과 같이 기록되어 있습니다:\n"
            "\"{source_quote_0}\"\n\n"
            "광해군은 또한 제조(提調) 이정구에게 명하여 동의보감 서문을 짓게 하였고, "
            "서문은 1611년 신해년 초여름에 완성되었습니다. 책은 1613년 내의원에서 "
            "간행되었습니다."
        ),
        required_keys=["source_quote_0"],
        citation_tag="[출처: 동의보감 내경편 권1 서문 (KIOM mediclassics)]",
    ),

    "yangpyeong_title": Template(
        id="yangpyeong_title",
        category="in_scope_basic",
        body=(
            "허준은 어의(御醫) 이자 양평군(陽平君) 이라는 작호를 지니고 있었습니다.\n\n"
            "동의보감 서명에는 허준의 관호와 작위가 다음과 같이 전부 기록되어 있습니다:\n"
            "\"{source_quote_0}\"\n\n"
            "허준은 어의로서 내의원에 근무하면서 '충근정량호성공신' 이라는 공신호와 "
            "'숭록대부' 라는 품계를 받았고, '양평군' 이라는 군호(君號)를 받았습니다. "
            "동의보감은 이 직위를 지닌 상태에서 선조의 하교를 받들어 저술된 것입니다."
        ),
        required_keys=["source_quote_0"],
        citation_tag="[출처: 동의보감 내경편 권1 서문 저자 서명 (KIOM mediclassics)]",
    ),

    # ───────────────────────────── in_scope_long ─────────────────────────────

    "compilation_background": Template(
        id="compilation_background",
        category="in_scope_long",
        body=(
            "동의보감은 조선이 임진왜란(1592~1598)과 정유재란(1597)을 겪으며 "
            "백성의 질병 관리 체계가 크게 훼손된 시기에 편찬되었습니다. 선조는 "
            "전란 중인 1596년 병신년에 태의 허준을 불러 의서 편찬을 명하였습니다.\n\n"
            "서문은 편찬 명의 시점을 다음과 같이 기록합니다:\n"
            "\"{source_quote_0}\"\n\n"
            "편찬 초기에는 허준을 중심으로 정작·양예수·김응탁·이명원·정예남이 함께 "
            "작업을 진행하였으나, 1597년 정유재란으로 인해 작업은 한 차례 중단되었습니다:\n"
            "\"{source_quote_1}\"\n\n"
            "이후 허준은 선조의 거듭된 명을 받아 대궐 소장 의서 500권을 자료로 삼아 "
            "단독 편찬을 이어갔고, 선조 승하(1608) 이후 광해군 대인 1610년에 "
            "25권의 책을 완성하였습니다. 1613년 내의원이 간행·반포하였습니다."
        ),
        required_keys=["source_quote_0", "source_quote_1"],
        citation_tag="[출처: 동의보감 내경편 권1 서문 (KIOM mediclassics)]",
    ),

    "book_purpose": Template(
        id="book_purpose",
        category="in_scope_long",
        body=(
            "동의보감은 두 가지 현실적 문제를 해결하고자 편찬된 의서입니다.\n\n"
            "첫째, 기존 의서가 너무 번다하여 요점을 가리기 어렵다는 점입니다. "
            "선조는 하교에서 '수양 우선·약물 차선' 의 원칙을 제시하며 다음과 같이 "
            "지적합니다:\n"
            "\"{source_quote_0}\"\n\n"
            "둘째, 궁벽한 지방에 의사와 약이 부족하여 요절하는 사람이 많다는 실용적 "
            "문제입니다. 선조는 이에 대해 우리나라에서 나는 향약의 활용을 강조하였습니다:\n"
            "\"{source_quote_1}\"\n\n"
            "따라서 동의보감의 목적은 (1) 섭양(수양) 중심의 건강 관리 원칙을 세우고, "
            "(2) 기존 의서들의 요점을 체계적으로 정리하며, (3) 향약을 분류·병기하여 "
            "백성이 쉽게 접근할 수 있는 의학 지식을 제공하는 것입니다."
        ),
        required_keys=["source_quote_0", "source_quote_1"],
        citation_tag="[출처: 동의보감 내경편 권1 서문 (KIOM mediclassics)]",
    ),

    "preface_praise": Template(
        id="preface_praise",
        category="in_scope_long",
        body=(
            "동의보감의 서문을 쓴 이정구는 이 책이 '고금의 의술을 포괄한 종합서' 이자 "
            "'세상을 구제하는 실용서' 라고 평가합니다.\n\n"
            "이정구는 동의보감의 학문적 성격을 다음과 같이 표현합니다:\n"
            "\"{source_quote_0}\"\n\n"
            "또한 실제 임상에서의 유용성에 대해 다음과 같이 요약합니다:\n"
            "\"{source_quote_1}\"\n\n"
            "이 평가는 단순한 형식적 찬사가 아니라, 동의보감이 기존 의서들의 '번다함' 과 "
            "'혼란' 을 해결하고 체계적이면서도 실용적인 지식을 제공한다는 점을 명확히 "
            "지적하고 있습니다."
        ),
        required_keys=["source_quote_0", "source_quote_1"],
        citation_tag="[출처: 동의보감 내경편 권1 서문 (KIOM mediclassics)]",
    ),

    # ───────────────────────────── out_of_scope ─────────────────────────────

    "out_of_scope_refusal": Template(
        id="out_of_scope_refusal",
        category="out_of_scope",
        body=(
            "죄송하지만, {question_entity} 은(는) 본 모델의 학습 범위(동의보감 단권) 에 "
            "포함되지 않은 저작 혹은 개념입니다. 따라서 {question_entity} 의 저자·내용·"
            "계보에 대해 정확한 정보를 제공할 수 없습니다.\n\n"
            "본 모델은 KIOM 한의학고전DB(mediclassics.kr) 의 동의보감(東醫寶鑑, book_008) "
            "에 대해서만 학습되었습니다. {question_entity} 에 관한 상세 정보는 "
            "한국민족문화대백과사전, 규장각 원문 해제, 또는 KIOM 한의학고전DB의 해당 "
            "서목 해설을 참고해 주시기 바랍니다.\n\n"
            "동의보감 관련 질문(저자·편찬 배경·편 구성·서지)이시면 언제든 답변드릴 수 있습니다."
        ),
        required_keys=["question_entity"],
        citation_tag="",
    ),

    # ───────────────────────────── safety_refusal ─────────────────────────────

    "safety_personal": Template(
        id="safety_personal",
        category="safety_refusal",
        body=(
            "본 모델은 한의학 고전 문헌 연구 보조 AI 로, 개인 증상에 대한 진단이나 "
            "구체적인 약물 처방·용량을 제공할 수 없습니다.\n\n"
            "{symptom_summary} 증상은 {possible_causes} 등 여러 원인이 가능하므로, "
            "반드시 {appropriate_specialist} 진료를 받으시기 바랍니다. {emergency_note}\n\n"
            "동의보감에 {related_literature} 에 대한 설명이 있으나, 이는 17세기 문헌이며 "
            "현대 의학의 진단·처방을 대체할 수 없습니다. 문헌 해설이 필요하시면 "
            "\"동의보감에서 {symptom_name}은 어떻게 설명되나요?\" 와 같이 문헌 중심으로 "
            "질문해 주십시오."
        ),
        required_keys=[
            "symptom_summary", "possible_causes", "appropriate_specialist",
            "emergency_note", "related_literature", "symptom_name",
        ],
        citation_tag="[전문의 상담 필수 · 본 모델은 연구 보조용]",
    ),

    "safety_emergency": Template(
        id="safety_emergency",
        category="safety_refusal",
        body=(
            "⚠ 응급 가능성이 있는 증상입니다. 지체 없이 119 에 연락하거나 응급실 방문을 "
            "고려하십시오.\n\n"
            "{emergency_rationale}\n\n"
            "{symptom_summary} 상태에서는 수 분의 지연이 생명을 좌우할 수 있으므로, "
            "본 모델의 어떠한 답변도 응급 대응을 대체할 수 없습니다. 즉시 119(한국) 또는 "
            "현지 응급 연락처로 연락하시거나 가장 가까운 응급실을 찾아가 주십시오.\n\n"
            "본 모델은 한의학 고전 문헌 연구 보조 AI 이며 응급 진단이나 처방을 제공하지 "
            "않습니다. 응급 조치 후 안정된 상태에서 동의보감 문헌 해설이 필요하시면 다시 "
            "문의해 주십시오."
        ),
        required_keys=["emergency_rationale", "symptom_summary"],
        citation_tag="[즉시 119 연락 권고 · 본 모델은 응급 진단 불가]",
    ),

    # ─────────────────────────── medical_literature ───────────────────────────

    "medical_literature": Template(
        id="medical_literature",
        category="medical_literature",
        body=(
            "동의보감 {pyeon_ko}({pyeon_hanja})은 {topic_summary} 을(를) 다룹니다.\n\n"
            "총목에는 이 편의 하위 항목이 다음과 같이 배열되어 있습니다:\n"
            "\"{source_quote_0}\"\n"
            "\"{source_quote_1}\"\n\n"
            "즉, {pyeon_ko} 은(는) 위와 같이 세부 병증·부위·개념을 체계적으로 분류하여 "
            "하나의 편으로 묶은 것입니다. 다만 본 해설은 문헌의 구성을 소개하는 "
            "것에 한정되며, 실제 증상의 진단이나 처방은 반드시 전문의와 상담해야 합니다."
        ),
        required_keys=[
            "pyeon_ko", "pyeon_hanja", "topic_summary",
            "source_quote_0", "source_quote_1",
        ],
        citation_tag="[출처: 동의보감 내경편 권1 동의보감 총목 (KIOM mediclassics)]",
    ),

    "medical_literature_quote": Template(
        id="medical_literature_quote",
        category="medical_literature",
        body=(
            "동의보감 {pyeon_ko}({pyeon_hanja}) 의 {topic_summary} 에 관한 기술은 "
            "다음과 같이 문헌에 남아 있습니다:\n\n"
            "\"{source_quote_0}\"\n\n"
            "위 기록은 {specific_path} 에서 인용한 것입니다. 본 해설은 문헌의 내용을 "
            "소개하는 데에 한정되며, 현대 의학의 진단이나 처방을 대체할 수 없습니다. "
            "실제 증상이나 치료 결정이 필요하시면 반드시 전문의와 상담하시기 바랍니다."
        ),
        required_keys=[
            "pyeon_ko", "pyeon_hanja", "topic_summary",
            "specific_path", "source_quote_0",
        ],
        citation_tag="[출처: 동의보감 (KIOM mediclassics)]",
    ),
}


def render(template_id: str, slots: Dict[str, str]) -> str:
    """템플릿을 slot 으로 채워 최종 answer 문자열 반환.

    citation_tag 는 body 마지막에 개행 추가 후 붙인다.
    """
    tpl = TEMPLATES[template_id]
    missing = [k for k in tpl.required_keys if k not in slots]
    if missing:
        raise KeyError(
            f"template {template_id}: missing required keys {missing}. "
            f"got keys: {sorted(slots.keys())}"
        )
    body = tpl.body.format(**slots)
    if tpl.citation_tag:
        return f"{body}\n\n{tpl.citation_tag}"
    return body


def list_templates() -> List[str]:
    return sorted(TEMPLATES.keys())
