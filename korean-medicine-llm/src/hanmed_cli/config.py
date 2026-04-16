"""§10.10 SSoT — sampling/model/safety/footer/timezone 전수.

다른 모듈은 값을 재기재하지 말고 `from hanmed_cli.config import DEFAULTS` 로 접근.
drift 방지 (§10.10.8 구현 가이드 정본).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Defaults:
    # --- Model · Adapter (§10.10.2) ------------------------------------
    base_model: str = "MLP-KTLim/llama-3-Korean-Bllossom-8B"
    dtype: str = "bfloat16"
    max_lora_rank: int = 32
    max_loras: int = 1
    gpu_memory_utilization: float = 0.85
    max_model_len: int = 8192
    tokenizer_ext_dir: str = "data/tokenizer/hanmed_bllossom_ext"

    # --- Sampling (§10.10.1) -------------------------------------------
    temperature: float = 0.7
    top_p: float = 0.9
    max_new_tokens: int = 1024
    repetition_penalty: float = 1.1
    # stop_token_ids: tokenizer.eos_token_id 와 Llama-3 <|eot_id|>=128009.
    # Bllossom 은 eos_token_id 가 이미 128009 이지만 이중 안전장치.
    stop_token_id_eot: int = 128009

    # --- Context window (§10.10.3) -------------------------------------
    context_window_tokens: int = 8192
    sliding_window_drop_after_tokens: int = 8192

    # --- Safety / Footer (§10.10.4) ------------------------------------
    t4_refusal_target: float = 0.99
    t4_paraphrase_held_out_target: float = 0.95
    t4_hanmun_target: float = 0.90
    footer: str = "— KIOM mediclassics.kr 기반 학습 (한의학고전DB)"

    # --- System prompt version ------------------------------------------
    system_prompt_version: str = "v0.1"

    # --- Timezone (§10.10.5) -------------------------------------------
    timezone_storage: str = "UTC"

    # --- Session schema (§10.6.2 v0.2) ---------------------------------
    session_schema_version: str = "v0.2"
    session_autosave_name: str = "current"

    # --- Reproducibility ------------------------------------------------
    seed: int = 42


DEFAULTS = Defaults()
