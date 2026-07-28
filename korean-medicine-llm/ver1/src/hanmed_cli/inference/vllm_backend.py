"""§10.4.1 vLLM backend — v1 skeleton.

v0 에서는 복잡도 최소화 원칙(memory)에 따라 transformers backend 사용.
본 파일은 v1 에서 완성할 스켈레톤.

현재 상태: import 시 vllm 미설치면 ImportError 우아하게 전달.
"""

from __future__ import annotations

from typing import Iterator

from hanmed_cli.config import DEFAULTS
from hanmed_cli.inference.base import Backend, SamplingConfig


class VLLMBackend(Backend):
    @property
    def name(self) -> str:
        return "vllm"

    def __init__(self) -> None:
        try:
            import vllm  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "vllm backend 는 v1 에서 활성화됩니다. v0 에서는 --backend transformers 를 쓰세요. "
                "(§10.4 표: vLLM primary 는 v1 이후)"
            ) from exc
        self._llm = None
        self._tokenizer = None

    def load(
        self,
        *,
        base_model: str,
        tokenizer_dir: str,
        adapter_path: str | None,
        dtype: str = DEFAULTS.dtype,
        max_model_len: int = DEFAULTS.max_model_len,
    ) -> None:
        # v1 에서 구현:
        #     from vllm import LLM
        #     from vllm.lora.request import LoRARequest
        #     self._llm = LLM(
        #         model=base_model, dtype=dtype, enable_lora=bool(adapter_path),
        #         max_lora_rank=DEFAULTS.max_lora_rank,
        #         gpu_memory_utilization=DEFAULTS.gpu_memory_utilization,
        #         max_model_len=max_model_len,
        #     )
        raise NotImplementedError("VLLMBackend.load — v1 에서 구현 예정")

    def stream_generate(
        self,
        prompt: str,
        cfg: SamplingConfig,
    ) -> Iterator[str]:
        raise NotImplementedError("VLLMBackend.stream_generate — v1 에서 구현 예정")

    def get_tokenizer(self):
        raise NotImplementedError("VLLMBackend.get_tokenizer — v1 에서 구현 예정")

    def close(self) -> None:
        self._llm = None
        self._tokenizer = None
