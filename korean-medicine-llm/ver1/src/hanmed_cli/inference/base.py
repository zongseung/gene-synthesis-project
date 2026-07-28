"""§10.4.3 Backend 추상 인터페이스.

구현체 (TransformersBackend, VLLMBackend) 가 이 프로토콜을 따른다.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator

from hanmed_cli.config import DEFAULTS


@dataclass
class SamplingConfig:
    temperature: float = DEFAULTS.temperature
    top_p: float = DEFAULTS.top_p
    max_new_tokens: int = DEFAULTS.max_new_tokens
    repetition_penalty: float = DEFAULTS.repetition_penalty


class Backend(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """'transformers' | 'vllm' | ..."""

    @abstractmethod
    def load(
        self,
        *,
        base_model: str,
        tokenizer_dir: str,
        adapter_path: str | None,
        dtype: str = DEFAULTS.dtype,
        max_model_len: int = DEFAULTS.max_model_len,
    ) -> None:
        """base + (optional) LoRA adapter 로드. idempotent."""

    @abstractmethod
    def stream_generate(
        self,
        prompt: str,
        cfg: SamplingConfig,
    ) -> Iterator[str]:
        """prompt 받아 토큰 chunk 를 순차 yield (str, 이미 decode된 부분 문자열)."""

    @abstractmethod
    def close(self) -> None:
        """자원 해제."""

    # optional: tokenizer 반환 (conversation 이 ChatML 템플릿 만들 때 사용)
    @abstractmethod
    def get_tokenizer(self):
        """PreTrainedTokenizerBase."""
