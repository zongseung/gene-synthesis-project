"""ver7 adapter 처방·약재·혈 중심 10문 probe (Q11~Q20).

기존 Q1~Q10 (probe_ver7_quick) 외에 처방 조성·약재 편 분류·혈 주치 정확도를
넓게 측정하기 위한 보조 probe. 목적: 남은 환각 유형 (조성·편 오분류·한자 오타)
을 다양한 카테고리에서 재현해 다음 iteration 타깃 식별.
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
    # 처방 조성 (실재 처방)
    "사물탕(四物湯)의 구성 약재 네 가지를 알려주세요.",
    "육미지황원(六味地黃元)의 구성 여섯 약재를 알려주세요.",
    "소시호탕(小柴胡湯)의 구성과 주치를 설명해 주세요.",
    "십전대보탕(十全大補湯)은 어떤 증에 쓰는 처방인가요?",
    # 약재 편 분류 (탕액편 내부)
    "숙지황(熟地黃)은 탕액편 어느 부(部)에 속하며 주된 효능은?",
    "황기(黃芪)의 성미와 주된 효능을 알려주세요.",
    # 혈자리 inverse
    "합곡(合谷) 혈은 어느 경맥에 속하며 주요 주치는 무엇인가요?",
    "태충(太衝) 혈의 위치와 소속 경맥을 알려주세요.",
    # 편 연결 (정확한 up_path 확인)
    "동의보감 내경편에서 정(精)·기(氣)·신(神) 삼보는 어떻게 구분되나요?",
    "동의보감 외형편 제1권에서 다루는 대표 주제를 나열해 주세요.",
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

    for i, q in enumerate(QUESTIONS, 11):
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": q}]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tok(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = model.generate(**ids, pad_token_id=tok.pad_token_id, **GEN)
        text = tok.decode(out[0, ids.input_ids.shape[1]:], skip_special_tokens=True)
        print(f"\n==== Q{i} ====\n{q}\n---- A{i} ----\n{text.strip()}")

if __name__ == "__main__":
    main()
