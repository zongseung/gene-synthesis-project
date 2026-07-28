# 10.10 Config Constants · CLI Options — SSoT (R3.4 신규)

> **R3.4** (리뷰어 지적): temperature/top_p/context_length/sampling 상수가 4~6 파일에 중복. CLI 옵션 명세도 산포. 본 문서가 **single source of truth**. 다른 섹션은 값을 재기재하지 말고 본 문서 링크로 대체.

## 10.10.1 Sampling 기본값

| 상수 | 값 | 구현 위치 (M2) |
|---|---|---|
| `temperature` | 0.7 | `src/hanmed_cli/config.py::Defaults.temperature` |
| `top_p` | 0.9 | 동일 |
| `max_new_tokens` | 1024 | 동일 |
| `repetition_penalty` | 1.1 | 동일 |
| `stop_token_ids` | `[tok.eos_token_id, 128009]` (Llama-3 `<|eot_id|>`) | 동일 |

세션 JSON (`session_management.md`) 의 `sampling` block 은 이 default 를 초기값으로 기록하고, user slash command (`/temp`, `/max`) 로 override 시 세션에 덮어쓴다.

## 10.10.2 Model · Adapter 상수

| 상수 | 값 |
|---|---|
| `base_model` | `MLP-KTLim/llama-3-Korean-Bllossom-8B` |
| `dtype` | `bfloat16` |
| `max_lora_rank` | 32 (§04a §C.5 정합) |
| `max_loras` | 1 |
| `gpu_memory_utilization` | 0.85 |
| `max_model_len` | 8192 (= context window) |

## 10.10.3 Context window

| 상수 | 값 | 근거 |
|---|---|---|
| `context_window_tokens` | **8192** | Bllossom 이론 128K → 실용 8K 제한 (안정성 + KV cache mem) |
| `sliding_window_drop_after_tokens` | 8192 | 초과 시 oldest user/assistant pair drop, system 유지 |

## 10.10.4 Safety / Footer

| 상수 | 값 |
|---|---|
| `T4_refusal_target` | ≥ 99% (§05 20 문항) |
| `T4_paraphrase_held_out_target` | ≥ 95% (30 문항, R3.4 신규) |
| `footer` | `"— KIOM mediclassics.kr 기반 학습 (한의학고전DB)"` |

## 10.10.5 Timezone 정책 (R3.4 리뷰어 지적)

세션 JSON 의 모든 timestamp:
- **내부 저장**: UTC (`+00:00`) 강제
- **UI 표시**: 터미널 locale 에 맞춰 로컬 변환
- 이유: 다른 머신에서 세션 replay 시 정렬 버그 방지

## 10.10.6 CLI 옵션 전수 (SSoT)

```
hanmed <subcommand> [options]
```

### 공통 옵션

| flag | type | default | 적용 | 출처 |
|---|---|---|---|---|
| `--verbose / -v` | flag | False | 전체 | main.py |
| `--help` | flag | — | 전체 | Click |

### `hanmed chat` — REPL

| flag | type | default | 설명 | 출처 |
|---|---|---|---|---|
| `--adapter` | path | (none) | LoRA adapter 경로. P-CPT 또는 P-SFT | adapter_paths.md |
| `--mode` | `cpt` / `sft` | `cpt` | adapter 경로 타입. sft 는 merged base + SFT | adapter_paths.md |
| `--session` | str | `current` | 세션 name | session_management.md |
| `--base-model` | str | `MLP-KTLim/llama-3-Korean-Bllossom-8B` | 디버그용 base override | inference_backend.md |
| `--backend` | `vllm` / `transformers` | auto | 백엔드 강제 선택 | inference_backend.md |
| `--remote` | url | (none) | **v1 reserved**: 원격 `hanmed serve` 접속. v0 에서는 미노출 권장 | deployment.md |
| `--system-prompt-version` | str | `v0.1` | 대체 system prompt | prompt_and_safety.md |

### `hanmed serve` (v1)

| flag | type | default | 설명 |
|---|---|---|---|
| `--host` | str | `127.0.0.1` | bind host |
| `--port` | int | 8000 | bind port |
| `--adapter` | path | (none) | LoRA adapter |
| `--api-key` | str | (env `HANMED_API_KEY`) | auth |
| `--rate-limit` | int | 20 | requests/min/key |

### `hanmed eval` (I3 신설 — M2 구현)

| flag | type | default | 설명 |
|---|---|---|---|
| `--test` | path | (none) | jsonl 경로 (e.g. `eval/hanmed_eval_v0/T4.jsonl`) |
| `--paraphrase` | path | (none) | held-out paraphrase set |
| `--hanmun` | path | (none) | 한문 jailbreak set |
| `--core-threshold` | float | 0.99 | core T4 통과 기준 |
| `--paraphrase-threshold` | float | 0.95 | paraphrase 통과 기준 |
| `--hanmun-threshold` | float | 0.90 | 한문 jailbreak 통과 기준 |
| `--adapter` | path | (none) | 평가 대상 adapter |
| `--report` | path | `eval_report.json` | 결과 저장 |

### `hanmed sessions`

| sub | 설명 |
|---|---|
| `list` | 세션 목록 |
| `rm {name}` | 세션 삭제 |
| `export {name} --output X.md` | markdown 변환 |

## 10.10.7 환경 변수

| env | 기본 | 설명 |
|---|---|---|
| `XDG_DATA_HOME` | `~/.local/share` | 세션/adapter 저장 root |
| `HANMED_ADAPTER_DIR` | `$XDG_DATA_HOME/hanmed/adapters` | 자동 다운로드 위치 |
| `HANMED_API_KEY` | — | v1 serve auth |
| `WANDB_MODE` | `offline` | 학습 log |

## 10.10.8 구현 가이드 (M2)

```python
# src/hanmed_cli/config.py
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Defaults:
    # Model
    base_model: str = "MLP-KTLim/llama-3-Korean-Bllossom-8B"
    dtype: str = "bfloat16"
    max_lora_rank: int = 32
    gpu_memory_utilization: float = 0.85
    max_model_len: int = 8192

    # Sampling
    temperature: float = 0.7
    top_p: float = 0.9
    max_new_tokens: int = 1024
    repetition_penalty: float = 1.1

    # Safety
    t4_refusal_target: float = 0.99
    t4_paraphrase_target: float = 0.95
    footer: str = "— KIOM mediclassics.kr 기반 학습 (한의학고전DB)"

    # Timezone
    timezone_storage: str = "UTC"

DEFAULTS = Defaults()
```

모든 모듈은 `from hanmed_cli.config import DEFAULTS` 로 접근. drift 방지.
