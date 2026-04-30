"""RAG sidecar 설정 — 모든 경로/엔드포인트/모델 이름은 환경변수 우선.

Defaults 는 compose.ver8_1.yml 의 volume mount 와 일치하도록 컨테이너 내부 절대경로.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env_path(key: str, default: str) -> Path:
    return Path(os.environ.get(key, default))


@dataclass(frozen=True)
class Settings:
    # vLLM upstream
    vllm_url: str = os.environ.get("VLLM_URL", "http://hanmed_vllm_ver8_1:8000")
    vllm_model: str = os.environ.get("VLLM_MODEL_NAME", "hanmed-ver8_1")
    vllm_timeout_s: float = float(os.environ.get("VLLM_TIMEOUT_S", "60"))

    # RAG corpus
    faiss_index_path: Path = _env_path("RAG_INDEX_PATH", "/rag_data/book_008.index")
    faiss_meta_path: Path = _env_path("RAG_META_PATH", "/rag_data/book_008.meta.jsonl")
    encoder_name: str = os.environ.get("RAG_ENCODER", "BAAI/bge-m3")
    encoder_device: str = os.environ.get("RAG_ENCODER_DEVICE", "cpu")

    # INFO mode prompt yaml (Phase 3 에서는 미사용 — Phase 6 후속)
    info_prompts_yaml: Path = _env_path(
        "INFO_PROMPTS_YAML", "/prompts/info_mode_prompts.yaml"
    )

    # Retrieval
    top_k: int = int(os.environ.get("RAG_TOP_K", "5"))
    boost_k: int = int(os.environ.get("RAG_BOOST_K", "2"))
    min_body_len: int = int(os.environ.get("RAG_MIN_BODY_LEN", "30"))

    # Generation
    max_tokens: int = int(os.environ.get("GEN_MAX_TOKENS", "400"))
    temperature: float = float(os.environ.get("GEN_TEMPERATURE", "0.0"))
    repetition_penalty: float = float(os.environ.get("GEN_REPETITION_PENALTY", "1.1"))


settings = Settings()
