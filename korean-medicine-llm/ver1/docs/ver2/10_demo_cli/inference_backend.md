# 10.4 Inference Backend

## 후보 비교 (A6000 48GB, Bllossom-8B bf16 기준)

| Backend | tok/s | 장점 | 단점 | v0 |
|---|---|---|---|---|
| **vLLM ≥ 0.6** | 40~80 | PagedAttention, continuous batching, LoRA serving (dynamic), OpenAI API 호환 | 설치 복잡, CUDA 필수 | ✅ primary |
| **transformers + peft** | 15~30 | 의존성 적음, LoRA 네이티브 | 느림, KV cache 단순 | ✅ fallback (debug) |
| **llama.cpp (GGUF)** | 25~50 (Q4_K_M) | CPU 가능, quantization | LoRA 병합 필요, 품질 저하 | v1 옵션 |
| SGLang | 50~90 | structured output 강력 | 신생, LoRA 지원 제한 | v1 검토 |
| TGI | 40~70 | HuggingFace 공식 | Docker 의존, overhead | 제외 |

## 10.4.1 vLLM (primary)

> **주의**: 아래 snippet 은 **가독성 pseudo-code**. 실제 구현 시 tokenizer import, error handling, async wrapping 필요.

```python
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

llm = LLM(
    model="MLP-KTLim/llama-3-Korean-Bllossom-8B",
    dtype="bfloat16",
    enable_lora=True,
    max_lora_rank=32,              # §04a §C.5 와 정합
    max_loras=1,
    gpu_memory_utilization=0.85,
    max_model_len=8192,            # 10.5 conversation window
)

lora = LoRARequest("hanmed_cpt", 1, "outputs/cpt_bllossom/adapter")

outputs = llm.generate(
    prompts=[chatml_formatted],
    sampling_params=SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=1024,
        repetition_penalty=1.1,
        stop_token_ids=[tokenizer.eos_token_id, 128009],  # Llama-3 <|eot_id|>
    ),
    lora_request=lora,
)
```

**Streaming**: vLLM `AsyncLLMEngine` + async iterator.

**LoRA swapping**: runtime 에 `LoRARequest` 교체 가능 → P-CPT / P-SFT 전환 디버그 편의.

## 10.4.2 transformers fallback

> **주의**: 아래 snippet 은 **가독성 pseudo-code**. `input_ids`, 에러 핸들링 누락.

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from peft import PeftModel
from threading import Thread

tok = AutoTokenizer.from_pretrained("MLP-KTLim/llama-3-Korean-Bllossom-8B")
model = AutoModelForCausalLM.from_pretrained(
    "MLP-KTLim/llama-3-Korean-Bllossom-8B",
    torch_dtype=torch.bfloat16,
    device_map="cuda",
)
model = PeftModel.from_pretrained(model, "outputs/cpt_bllossom/adapter")
model.eval()

streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
thread = Thread(target=model.generate, kwargs=dict(
    input_ids=input_ids,
    streamer=streamer,
    max_new_tokens=1024,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.1,
))
thread.start()
for token in streamer:
    print(token, end="", flush=True)
```

**언제 이걸 쓰나**:
- vLLM 설치 실패 환경 (CUDA 버전 미스매치 등)
- 단일 prompt 디버그 (throughput 중요하지 않음)
- CPU 전용 환경 (성능 느리지만 동작)

## 10.4.3 Backend 추상 인터페이스

```python
# inference/base.py
from abc import ABC, abstractmethod
from typing import AsyncIterator
from dataclasses import dataclass

@dataclass
class SamplingConfig:
    temperature: float = 0.7
    top_p: float = 0.9
    max_new_tokens: int = 1024
    repetition_penalty: float = 1.1

class Backend(ABC):
    @abstractmethod
    def load(self, base_model: str, adapter_path: str, **kwargs) -> None: ...

    @abstractmethod
    def stream_generate(self, prompt: str, cfg: SamplingConfig) -> AsyncIterator[str]: ...

    @abstractmethod
    def close(self) -> None: ...
```

## 10.4.4 성능 목표 (A6000)

| 지표 | vLLM | transformers |
|---|---|---|
| First-token latency (warm) | < 0.5 s | < 1.5 s |
| Throughput | ≥ 40 tok/s | ≥ 15 tok/s |
| Cold start | 15~25 s | 20~35 s |
| Peak GPU mem (8K context) | 25~28 GB | 28~32 GB |

R3.5 정리:
- 위 `Cold start` 는 **첫 모델 로드 포함** 시간이다.
- `10.1 E1` 의 `REPL prompt < 5 s` 와 충돌하지 않는다. CLI 는 backend lazy-load 를 전제로 하므로, REPL 진입 시간과 첫 질의 응답 시간을 분리 측정한다.

## 10.4.5 GGUF 경로 (v1 옵션)

CPT adapter 를 base 에 merge 후 GGUF 변환:

```bash
# 1. merge
python scripts/merge_lora.py --base Bllossom-8B --adapter outputs/cpt_bllossom/adapter --output outputs/cpt_merged

# 2. GGUF 변환 (llama.cpp)
python llama.cpp/convert_hf_to_gguf.py outputs/cpt_merged --outfile hanmed-cpt.gguf

# 3. quantization (옵션)
./llama-quantize hanmed-cpt.gguf hanmed-cpt-q4_k_m.gguf Q4_K_M
```

CPU / Apple Silicon 배포 시 유리. 정밀도 손실 (bf16 → Q4_K_M) 로 T2 QA 정확도 1~3%p 감소 예상 — v1 에서 실측.

## 10.4.6 열린 결정

1. **vLLM 버전 핀**: 0.6.x vs 0.7.x — LoRA dynamic load API 변화 모니터링
2. **Tensor parallel**: v0 에서 불필요. v1 에서 S2 동시 사용자 4+ 이면 `tp=2` 활성화
3. **Quantization**: AWQ/GPTQ 4-bit — RTX 3090 24GB 에서 돌리려면 필요. v1 이후
