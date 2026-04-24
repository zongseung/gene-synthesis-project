"""ver7 adapter 임산부·부인문 특화 7문 probe.

동의보감 부인문 (잡병편 권10) 범위 내 임신·태교·산후 관련 실사용 질문으로
v7 의 부인문 커버리지와 refusal 회피·조성 환각을 동시 측정.
"""
from __future__ import annotations
import os, torch
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE = "models/gemma-3-12b-it"
ADAPTER = "experiments/dongui_bogam/outputs_ver7_gemma_patched/adapter"
DEVICE = "cuda:0"

SYSTEM = (
    "당신은 한의학 고전 문헌 연구 보조 AI 입니다. 동의보감(東醫寶鑑) 본문에 근거해 "
    "편·장 구조, 처방, 약재, 경혈, 증론을 정확하고 간결하게 답합니다. 원문에 없는 "
    "인명·연도·처방은 창작하지 않으며, 용량은 동의보감 원문 인용 범위 내에서만 서술합니다."
)

QUESTIONS = [
    "동의보감 부인문에서 임신오조(姙娠惡阻)의 원인과 증상을 설명해 주세요.",
    "동의보감에서 임신 중에 금기시하는 약재는 어떤 것들이 있나요?",
    "태동불안(胎動不安) 에 대해 동의보감은 어떻게 설명하고 있나요?",
    "임산부의 감기에 대해 동의보감은 어떤 처방을 제시하나요?",
    "동의보감 부인문에서 태루하혈(胎漏下血) 은 어떤 병증이며 어떻게 다루나요?",
    "동의보감에서 산후 악로부진(惡露不盡) 의 원인과 치법은?",
    "임신 중 맥진에서 임신맥(妊娠脈) 의 특징은 무엇인가요?",
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
        print(f"\n==== P{i} ====\n{q}\n---- A{i} ----\n{text.strip()}")

if __name__ == "__main__":
    main()
