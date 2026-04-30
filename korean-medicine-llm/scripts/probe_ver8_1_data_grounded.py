"""ver8.1 adapter — 데이터 내 실존 토픽 5문 probe.

phaseB_qa_v7_corpus.jsonl 에 실제로 들어 있는 5개 passage/prescription 대목
을 기반으로 질문을 던진다. 각 질문마다 원문 정답을 주석으로 명시해
데이터가 있는데도 답이 틀리면 순수 학습 부족으로 판정할 수 있다.
"""
from __future__ import annotations
import os, torch
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE = "models/gemma-3-12b-it"
ADAPTER = "experiments/dongui_bogam/outputs_ver8_1_gemma_v1/adapter"
DEVICE = "cuda:0"

SYSTEM = (
    "당신은 한의학 고전 문헌 연구 보조 AI 입니다. 동의보감(東醫寶鑑) 본문에 근거해 "
    "편·장 구조, 처방, 약재, 경혈, 증론을 정확하고 간결하게 답합니다. 원문에 없는 "
    "인명·연도·처방은 창작하지 않으며, 용량은 동의보감 원문 인용 범위 내에서만 서술합니다."
)

# 각 Q 아래 GOLD 는 phaseB_qa_v7_corpus.jsonl 의 정답 요지 (원문에서 직접 추출).
QUESTIONS = [
    # D1 — 내경편 권1 > 신 > 전광 > 통설산
    #  GOLD: 갑자기 생긴 전광(癲狂)이 멎지 않거나 풍연(風涎)으로 기가 막혀 졸도하는 것 치료.
    #        과체 가루 3돈 + 경분 2.5푼. 물 반 홉에 섞어 먹인다. 《단심》
    "동의보감 내경편의 '통설산(通泄散)' 은 어떤 증상에 쓰는 처방이며 구성 약재는 무엇인가요?",
    # D2 — 외형편 권3 > 젖가슴 > 유옹 > 단삼고
    #  GOLD: 유옹(乳癰) 으로 멍울·통증, 터진 후 안 아무는 것. 단삼·적작약·백지 동량
    #        + 돼지기름 반 근 + 황랍 1냥. 고약을 만들어 바른다. 《입문》
    "유옹(乳癰) 에 쓰는 '단삼고(丹蔘膏)' 의 조성과 적응증을 알려주세요.",
    # D3 — 잡병편 권2 > 풍 > 침구법
    #  GOLD: 중풍에 가래가 심해 톱질 소리가 나고 약 효과 없을 때 배꼽 아래 기해·관원에
    #        200~300장 뜸을 뜬다. 《강목》
    "동의보감에서 중풍으로 가래가 심하고 약이 듣지 않을 때 어떤 침구 처치를 권하나요?",
    # D4 — 잡병편 권11 > 소아 > 급경풍 > 진경환
    #  GOLD: 급경풍 치료. 놀람 진정·신 안정·열 내리고 담 삭힘.
    #        우담남성 5돈, 주사 3.5돈, 호박·천축황·웅황 각 3돈, 우황 2돈, 진주 1돈,
    #        사향 0.5돈, 금박 10장. 환. 《정전》
    "동의보감 소아문의 '진경환(鎭驚丸)' 은 어떤 증에 쓰며 구성 약재는 무엇인가요?",
    # D5 — 잡병편 권4 > 허로 > 양허 > 증손낙령탕
    #  GOLD: 허로로 양기 부족. 반하 1.5돈, 황기·인삼·귤피·백복령·당귀·계심·세신·전호·
    #        맥문동·백작약·감초 각 7푼, 부자·숙지황 각 3.5푼, 원지 2푼. 《득효》
    "허로로 양기가 부족할 때 쓰는 '증손낙령탕(增損樂令湯)' 의 구성 약재를 알려주세요.",
]

GEN = dict(
    max_new_tokens=500,
    do_sample=True,
    temperature=0.5,
    repetition_penalty=1.2,
    no_repeat_ngram_size=8,
)

def main():
    tok = AutoTokenizer.from_pretrained(BASE)
    base = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map={"": DEVICE})
    model = PeftModel.from_pretrained(base, ADAPTER).eval()

    for i, q in enumerate(QUESTIONS, 1):
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": q}]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tok(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = model.generate(**ids, pad_token_id=tok.pad_token_id, **GEN)
        text = tok.decode(out[0, ids.input_ids.shape[1]:], skip_special_tokens=True)
        print(f"\n==== D{i} ====\n{q}\n---- A{i} ----\n{text.strip()}")

if __name__ == "__main__":
    main()
