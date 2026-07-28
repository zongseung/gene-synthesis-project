"""§10.4 Inference backends.

Backend 추상 인터페이스(`base.Backend`) + 두 구현 (transformers / vllm).
선택은 `get_backend(name)` 로.
"""

from __future__ import annotations

from hanmed_cli.inference.base import Backend, SamplingConfig


def get_backend(name: str) -> Backend:
    """`name` ∈ {"transformers", "vllm", "auto"}.

    "auto": vllm import 가능 + CUDA 사용 가능 → vllm, 아니면 transformers.
    v0: transformers 를 primary 로 사용 (복잡도 최소화, §10 기획 memory).
    """
    if name == "transformers":
        from hanmed_cli.inference.transformers_backend import TransformersBackend

        return TransformersBackend()
    if name == "vllm":
        from hanmed_cli.inference.vllm_backend import VLLMBackend

        return VLLMBackend()
    if name == "remote_openai":
        from hanmed_cli.inference.remote_openai import RemoteOpenAIBackend

        return RemoteOpenAIBackend()
    if name == "auto":
        try:
            import vllm  # noqa: F401

            from hanmed_cli.inference.vllm_backend import VLLMBackend

            return VLLMBackend()
        except ImportError:
            from hanmed_cli.inference.transformers_backend import TransformersBackend

            return TransformersBackend()
    raise ValueError(f"unknown backend: {name!r}")


__all__ = ["Backend", "SamplingConfig", "get_backend"]
