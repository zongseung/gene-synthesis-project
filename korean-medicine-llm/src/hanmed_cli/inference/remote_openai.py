"""§10.4 + ver4 §03.02 · Remote OpenAI-compatible backend.

vLLM / Ollama / 기타 OpenAI API 호환 서버에 HTTP 로 토큰 스트리밍.

`Backend.stream_generate(prompt: str, cfg)` 계약 유지:
    - 이미 `Conversation.render_prompt()` 가 tokenizer chat template 을 적용해
      ChatML wrapped string 을 만들어 넘김.
    - 서버쪽에서 다시 chat template 을 씌우면 이중 wrap → `/v1/completions`
      (prompt-based) 엔드포인트 사용.

`get_tokenizer()`:
    - Conversation 이 `apply_chat_template(tokenize=False)` 로 문자열 렌더 +
      `apply_chat_template(tokenize=True)` 로 토큰 수 측정.
    - 로컬에 base tokenizer 가 있으면 이를 사용 (sliding window 정확).
    - 없으면 LlamaStub (Llama-3 ChatML 템플릿 하드코딩 + char 기반 근사 토큰수).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterator

import httpx

from hanmed_cli.config import DEFAULTS
from hanmed_cli.inference.base import Backend, SamplingConfig


_LLAMA3_CHAT_TEMPLATE = (
    "{% for m in messages %}"
    "<|start_header_id|>{{ m.role }}<|end_header_id|>\n\n"
    "{{ m.content }}<|eot_id|>"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
    "{% endif %}"
)


class _LlamaStub:
    """tokenizer-less fallback — Llama-3 chat template 하드코딩, 토큰수는 char/3 근사."""

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = False,
    ):
        # Minimal jinja-like substitution — 의존성 피해 직접 문자열 조합.
        parts: list[str] = []
        for m in messages:
            parts.append(f"<|start_header_id|>{m['role']}<|end_header_id|>\n\n")
            parts.append(f"{m['content']}<|eot_id|>")
        if add_generation_prompt:
            parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
        rendered = "".join(parts)
        if tokenize:
            # CJK 평균 ~3 char/token (Bllossom ext tokenizer 경험치)
            return [0] * max(1, len(rendered) // 3)
        return rendered


class RemoteOpenAIBackend(Backend):
    """OpenAI-compatible HTTP backend (vLLM /v1/completions streaming)."""

    def __init__(self) -> None:
        self._endpoint: str | None = None
        self._model: str | None = None
        self._client: httpx.Client | None = None
        self._tokenizer = _LlamaStub()

    @property
    def name(self) -> str:
        return "remote_openai"

    def load(
        self,
        *,
        base_model: str,
        tokenizer_dir: str,
        adapter_path: str | None,
        dtype: str = DEFAULTS.dtype,
        max_model_len: int = DEFAULTS.max_model_len,
        endpoint: str | None = None,
        api_key: str | None = None,
    ) -> None:
        """
        base_model 은 served-model-name (e.g., "hanmed-p-a-plus") 으로 재활용.
        tokenizer_dir / adapter_path / dtype / max_model_len 은 원격서 무관이라 무시.
        endpoint / api_key 는 env 또는 인자로.
        """
        self._endpoint = (endpoint or os.environ.get("HANMED_ENDPOINT") or "http://localhost:8000/v1").rstrip("/")
        self._model = base_model or os.environ.get("HANMED_MODEL") or "hanmed-p-a-plus"
        key = api_key or os.environ.get("HANMED_API_KEY") or "EMPTY"
        self._client = httpx.Client(
            base_url=self._endpoint,
            headers={"Authorization": f"Bearer {key}"},
            timeout=httpx.Timeout(60.0, read=None),
        )
        # health check: /models 조회
        resp = self._client.get("/models")
        resp.raise_for_status()
        # 로컬 extended tokenizer 가 있으면 _LlamaStub 대신 사용 (sliding window 정확)
        tok_path = Path(tokenizer_dir) if tokenizer_dir else None
        if tok_path and tok_path.exists():
            try:
                from transformers import AutoTokenizer
                self._tokenizer = AutoTokenizer.from_pretrained(str(tok_path), trust_remote_code=True)
            except Exception:
                # transformers 없거나 로드 실패 시 stub 유지
                pass

    def get_tokenizer(self):
        return self._tokenizer

    def stream_generate(self, prompt: str, cfg: SamplingConfig) -> Iterator[str]:
        """prompt (chat-template 적용된 문자열) → vLLM /v1/completions SSE 스트림."""
        assert self._client and self._model, "load() 먼저 호출"
        payload = {
            "model": self._model,
            "prompt": prompt,
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
            "max_tokens": cfg.max_new_tokens,
            "stream": True,
        }
        if cfg.repetition_penalty and cfg.repetition_penalty != 1.0:
            # vLLM extension
            payload["repetition_penalty"] = cfg.repetition_penalty

        with self._client.stream("POST", "/completions", json=payload) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                    delta = obj.get("choices", [{}])[0].get("text", "")
                    if delta:
                        yield delta
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue

    def close(self) -> None:
        if self._client:
            self._client.close()
            self._client = None
