"""§10.4.2 transformers + PEFT backend.

- Bllossom-8B bf16 로드 + (옵션) LoRA adapter wrap
- `TextIteratorStreamer` 로 토큰 스트림 생성
- CUDA 사용 가능하면 `cuda`, 아니면 `cpu` (성능 낮음 — debug only)

v0 primary backend. vLLM 은 v1 (§10.4 표).
"""

from __future__ import annotations

from threading import Thread
from typing import Iterator

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from hanmed_cli.config import DEFAULTS
from hanmed_cli.inference.base import Backend, SamplingConfig


_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


class TransformersBackend(Backend):
    @property
    def name(self) -> str:
        return "transformers"

    def __init__(self) -> None:
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load(
        self,
        *,
        base_model: str,
        tokenizer_dir: str,
        adapter_path: str | None,
        dtype: str = DEFAULTS.dtype,
        max_model_len: int = DEFAULTS.max_model_len,
    ) -> None:
        torch_dtype = _DTYPE_MAP.get(dtype, torch.bfloat16)

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch_dtype,
        )
        # vocab 128,256 → 128,260 (§A.4 special tokens)
        if model.get_input_embeddings().num_embeddings != len(self.tokenizer):
            model.resize_token_embeddings(len(self.tokenizer))

        if adapter_path:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter_path)

        model.to(self.device)
        model.eval()
        self.model = model

    def stream_generate(
        self,
        prompt: str,
        cfg: SamplingConfig,
    ) -> Iterator[str]:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("backend.load() must be called first")

        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=False,
        )

        stop_ids = [self.tokenizer.eos_token_id]
        if DEFAULTS.stop_token_id_eot not in stop_ids:
            stop_ids.append(DEFAULTS.stop_token_id_eot)

        gen_kwargs = dict(
            input_ids=input_ids,
            streamer=streamer,
            max_new_tokens=cfg.max_new_tokens,
            do_sample=cfg.temperature > 0,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            repetition_penalty=cfg.repetition_penalty,
            eos_token_id=stop_ids,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        thread = Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()
        for tok in streamer:
            yield tok
        thread.join()

    def get_tokenizer(self):
        if self.tokenizer is None:
            raise RuntimeError("backend.load() must be called first")
        return self.tokenizer

    def close(self) -> None:
        self.model = None
        self.tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
