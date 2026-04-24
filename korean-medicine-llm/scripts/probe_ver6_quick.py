"""ver6 Gemma-3-12b SFT adapter 빠른 답변 확인 (3개 질문)."""
from __future__ import annotations
import os, torch
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE = "models/gemma-3-12b-it"
ADAPTER = "experiments/dongui_bogam/outputs_ver6_gemma_v1/adapter"
DEVICE = "cuda:0"

SYSTEM = (
    "당신은 한의학 고전 문헌 연구 보조 AI 입니다. 동의보감(東醫寶鑑) 본문에 근거해 "
    "편·장 구조, 처방, 약재, 경혈, 증론을 정확하고 간결하게 답합니다. 원문에 없는 "
    "인명·연도·처방은 창작하지 않으며, 개인 진단이나 구체 용량 처방은 제공하지 않습니다."
)

QUESTIONS = [
    "동의보감 내경편에서 담(痰)으로 인해 생기는 병의 대표적 증상을 알려주세요.",
    "補中益氣湯의 구성 약재와 주요 효능을 동의보감 기준으로 설명해주세요.",
    "동의보감에서 말하는 오장(五臟)과 육부(六腑)의 기본 관계는 무엇인가요?",
    "동의보감 잡병편에서 풍(風)의 원인과 주요 증후를 요약해주세요.",
    "동의보감에서 인삼(人蔘)의 효능과 주로 쓰이는 증상은 무엇인가요?",
    "동의보감 외형편에서 두통(頭痛)은 어떻게 분류되나요?",
    "동의보감 부인문에서 산후풍(産後風)에 대한 설명을 해주세요.",
    "동의보감 소아문에서 경풍(驚風)의 원인과 치법을 알려주세요.",
    "동의보감 탕액편에서 당귀(當歸)의 성미와 주된 효능은?",
    "동의보감 침구편에서 족삼리(足三里) 혈의 위치와 주요 주치를 알려주세요.",
]

GEN = dict(
    max_new_tokens=400,
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
        print(f"\n==== Q{i} ====\n{q}\n---- A{i} ----\n{text.strip()}")

if __name__ == "__main__":
    main()
