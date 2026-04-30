"""ver8.1 adapter — 학습 데이터 distribution 강한 영역 5문 probe.

이전 grounded probe (5문) 가 처방→조성 환각을 입증했다면, 본 probe 는
학습 데이터 (phaseB_qa_v8_1_corpus) 의 강한 카테고리만 골라 모델이
실제로 무엇을 잘하는지 본다.

선정 카테고리 + 학습 rows:
  1. structure / 편 구조 / 1,932 rows
  2. prescription direct-efficacy / 1,011 rows  (효능, 구성 X)
  3. paraphrase / 동의보감 기본 사실 / 75 rows
  4. structure / 편명 한자 뜻 / 2,207 (구조 카테고리 내 OO/AA/BB)
  5. passage / 본문 요약 / 7,000 rows  ← top cluster
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

QUESTIONS = [
    # S1 — structure: 편 구성 (학습 데이터에 5편 23권 빈번 등장)
    "동의보감의 다섯 편(內景·外形·雜病·湯液·鍼灸) 의 큰 구성과 각 편이 다루는 주제를 짧게 정리해 주세요.",
    # S2 — prescription direct-efficacy (PRESCRIPTION_Q_TEMPLATES[1], 1,011 rows)
    "보중익기탕(補中益氣湯) 은 동의보감의 어느 편·문에 수록된 처방이며 주된 효능은 무엇인가요?",
    # S3 — paraphrase / basic_fact (75 rows, 저자·연도·구성)
    "동의보감은 누가 언제 편찬했고, 어떤 시대적 배경에서 만들어졌나요?",
    # S4 — structure / 편명 한자 뜻 (1,883 rows of CC + 23 rows of AA)
    "'內景篇(내경편)' 이라는 편명의 뜻과 동의보감에서 이 편이 다루는 범주를 설명해 주세요.",
    # S5 — passage 본문 요약 (top 1,400 prefix cluster)
    "다음 동의보감 본문 대목의 핵심 의미를 짚어 주세요.\n발췌: 醫者雅言軒岐, 軒岐上窮天紀, 下極人理, 宜不屑乎記述, 而猶且說問著難, 垂法後世, 則醫之有書, 厥惟遠哉.",
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
    base = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16,
                                                device_map={"": DEVICE})
    model = PeftModel.from_pretrained(base, ADAPTER).eval()
    print(f"[probe] adapter = {ADAPTER}  loaded  device={DEVICE}")

    for i, q in enumerate(QUESTIONS, 1):
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": q}]
        prompt = tok.apply_chat_template(msgs, tokenize=False,
                                          add_generation_prompt=True)
        ids = tok(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = model.generate(**ids, pad_token_id=tok.pad_token_id, **GEN)
        text = tok.decode(out[0, ids.input_ids.shape[1]:],
                          skip_special_tokens=True)
        print(f"\n==== S{i} ====\n{q}\n---- A{i} ----\n{text.strip()}")


if __name__ == "__main__":
    main()
