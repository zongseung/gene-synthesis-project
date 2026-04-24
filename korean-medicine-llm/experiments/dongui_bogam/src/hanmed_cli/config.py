from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Defaults:
    base_model: str = "MLP-KTLim/llama-3-Korean-Bllossom-8B"
    dtype: str = "bfloat16"
    max_lora_rank: int = 32
    max_loras: int = 1
    gpu_memory_utilization: float = 0.85
    max_model_len: int = 8192
    tokenizer_ext_dir: str = "data/tokenizer/hanmed_bllossom_ext"
    temperature: float = 0.7
    top_p: float = 0.9
    max_new_tokens: int = 1024
    repetition_penalty: float = 1.1
    stop_token_id_eot: int = 128009
    context_window_tokens: int = 8192
    sliding_window_drop_after_tokens: int = 8192
    t4_refusal_target: float = 0.99
    t4_paraphrase_held_out_target: float = 0.95
    t4_hanmun_target: float = 0.90
    footer: str = "— KIOM mediclassics.kr 기반 학습 (한의학고전DB)"
    footer_enabled: bool = False
    system_prompt_version: str = "v0.1"
    timezone_storage: str = "UTC"
    session_schema_version: str = "v0.2"
    session_autosave_name: str = "current"
    seed: int = 42


DEFAULTS = Defaults()
