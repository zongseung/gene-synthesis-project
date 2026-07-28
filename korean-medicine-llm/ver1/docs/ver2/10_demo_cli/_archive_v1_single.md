# 10. Demo CLI — `hanmed` 인터랙티브 터미널 (ver2.2, R3.2 primary)

> 한의학 고전 LLM (Bllossom-8B + HanMed LoRA) 을 **터미널에서 REPL 형태로 질의응답** 하는 v0 데모. Ollama / llama.cpp `-i` / Simon Willison `llm` / LM Studio CLI 의 한의학 특화 버전.
>
> 본 문서는 **데모 전용 스펙**이다. 학습 파이프라인 (§03/§04/§04a) 과 평가 (§05) 는 변경하지 않는다.

## 10.1 Scope

### In-scope (v0 데모)

- `hanmed chat` — 단일 process REPL (터미널 안에서 한의학 질의응답)
- Bllossom-8B base + HanMed Stage 2 SFT LoRA adapter 로드
- ChatML 프롬프트 템플릿 (Llama-3 chat template 또는 base_model §4.6.2)
- 멀티턴 대화 (컨텍스트 최대 8K tokens)
- Streaming output (한 토큰씩 출력)
- 대화 저장 / 불러오기 (`hanmed chat --session {name}`)
- 안전성 refusal layer (§05 T4 redteam 패턴 — 임상 의사결정 거부)
- 출처 표기 (답변 하단에 "KIOM mediclassics.kr 기반 학습" 고정 footer)

### Out-of-scope (v0 아님)

- ❌ Agentic tool use (파일 편집, bash 실행, 웹 검색 등 — Claude Code 류 기능)
- ❌ 웹/GUI UI — CLI 만
- ❌ 서버-클라이언트 분리 — 단일 binary (v1 에서 분리 검토)
- ❌ 음성 / 이미지 입출력
- ❌ 클라우드 배포 — 로컬 A6000 전제
- ❌ 모델 학습 — 별도 (§04a)

## 10.2 유사 프로젝트 비교

| 프로젝트 | 추론 | 인터페이스 | 차이점 |
|---|---|---|---|
| **Claude Code** | Anthropic API | REPL + tool use | 원격 API, agentic, 파일 편집 가능 |
| **Gemini CLI** | Google API | REPL | 원격 API |
| **Ollama** (`ollama run`) | llama.cpp GGUF | REPL | 로컬, GGUF quant, 모델 레지스트리 |
| **LM Studio CLI** | llama.cpp | REPL | 로컬, GUI 동반 |
| **Simon Willison `llm`** | plugin (OpenAI / local) | CLI + REPL | 다양한 backend, SQLite 로그 |
| **llama.cpp `-i`** | C++ | REPL | 경량, GGUF 필요 |
| **vLLM OpenAI server + `curl`** | vLLM | HTTP | 고성능, 별도 클라이언트 필요 |
| **HanMed-CLI (본 스펙)** | **vLLM 또는 transformers** | **REPL** | **한의학 도메인 refusal layer + 출처 표기 고정 + 한문/국역 bilingual 입력 지원** |

v0 는 Simon Willison `llm` + Ollama 의 중간 — 단일 도메인 특화, 프롬프트 템플릿 hardcoded, safety layer 내장.

## 10.3 아키텍처

```
┌─────────────────────────────────────────────────────────┐
│  hanmed CLI (Python, Click/Typer entry point)          │
│                                                         │
│  ┌───────────────┐    ┌────────────────┐   ┌──────────┐│
│  │ prompt_toolkit│ →  │ Conversation   │ → │ Safety   ││
│  │   (REPL UX)   │    │ Manager        │   │ Filter   ││
│  │ - 자동완성    │    │ - ChatML       │   │ - T4     ││
│  │ - 히스토리    │    │ - 컨텍스트     │   │   refusal││
│  │ - Rich 렌더   │    │ - session save │   │   pattern││
│  └───────────────┘    └────────────────┘   └─────┬────┘│
│                                                   │     │
│                                          ┌────────▼───┐│
│                                          │ Inference  ││
│                                          │ Backend    ││
│                                          │  (1 of 3)  ││
│                                          └────────────┘│
└─────────────────────────────────────────────────────────┘
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
       ┌──────────┐  ┌──────────┐  ┌──────────┐
       │  vLLM    │  │transformrs│  │llama.cpp │
       │ (primary)│  │(fallback) │  │ (option) │
       │ Paged    │  │ generate  │  │   GGUF   │
       │ Attention│  │   +peft   │  │ quant    │
       └──────────┘  └──────────┘  └──────────┘
              │             │             │
              └─────────────▼─────────────┘
                     HanMed LoRA adapter
                  (Bllossom-8B + SFT weights)
```

## 10.4 추론 백엔드 선정

| Backend | tok/sec (A6000) | 장점 | 단점 | v0 사용 |
|---|---|---|---|---|
| **vLLM** | 40~80 | PagedAttention, continuous batching, LoRA serving, OpenAI API 호환 | 설치 복잡, GPU 필수 | ✅ primary |
| **transformers + peft** | 15~30 | 의존성 적음, LoRA 네이티브 | 느림, KV cache 단순 | ✅ fallback (debug) |
| **llama.cpp** (GGUF) | 25~50 (Q4_K_M) | CPU 가능, quantization | LoRA 병합 필요, 품질 저하 | 옵션 (light deploy) |
| SGLang | 50~90 | structured output 강력 | 신생, LoRA 지원 제한 | 검토 (v1) |
| TGI | 40~70 | HuggingFace 공식 | Docker 의존, overhead | 제외 |

**v0 결정**: vLLM primary + transformers fallback (CUDA 없는 환경 / 디버그용).

### 10.4.1 vLLM + LoRA serving

vLLM 0.5+ 는 dynamic LoRA loading 지원:

```python
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

llm = LLM(
    model="MLP-KTLim/llama-3-Korean-Bllossom-8B",
    dtype="bfloat16",
    enable_lora=True,
    max_lora_rank=32,         # §04a §C.5 와 정합
    max_loras=1,
    gpu_memory_utilization=0.85,
)

lora = LoRARequest("hanmed_sft", 1, "outputs/sft_bllossom/adapter")

outputs = llm.generate(
    prompts=[chatml_formatted],
    sampling_params=SamplingParams(temperature=0.7, max_tokens=1024),
    lora_request=lora,
)
```

### 10.4.2 transformers fallback

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from peft import PeftModel

tok = AutoTokenizer.from_pretrained("MLP-KTLim/llama-3-Korean-Bllossom-8B")
model = AutoModelForCausalLM.from_pretrained(
    "MLP-KTLim/llama-3-Korean-Bllossom-8B",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model = PeftModel.from_pretrained(model, "outputs/sft_bllossom/adapter")
streamer = TextIteratorStreamer(tok, skip_prompt=True)
model.generate(input_ids, streamer=streamer, max_new_tokens=1024)
```

## 10.5 CLI UX (`prompt_toolkit` + `rich`)

### 실행 플로우

```
$ hanmed chat
═════════════════════════════════════════════════════════
  HanMed-CLI v0.1.0  |  Bllossom-8B + HanMed-SFT  |  bf16
  KIOM mediclassics.kr 기반 학습. 임상 결정 도구 아님.
  /help, /exit, /save, /load, /reset
═════════════════════════════════════════════════════════
[you] 인삼의 성미와 귀경에 대해 알려줘

[hanmed]  ⠋ generating...
  인삼(人蔘)은...
  - 성미: 달고 약간 쓰며 성질이 약간 따뜻함
  - 귀경: 비경(脾經), 폐경(肺經)
  - 효능: 원기 보충, 진액 생성
  (동의보감 탕액편 권1)

[you] /save ginseng_session
✓ saved to ~/.hanmed/sessions/ginseng_session.json

[you] /exit
```

### Slash commands

| 명령 | 동작 |
|---|---|
| `/help` | 명령어 목록 |
| `/exit` | 종료 |
| `/reset` | 컨텍스트 초기화 |
| `/save {name}` | 세션 JSON 저장 |
| `/load {name}` | 세션 복원 |
| `/temp {0.0-2.0}` | 샘플링 temperature 변경 |
| `/max {N}` | max_new_tokens 변경 |
| `/system {text}` | system prompt 재지정 (선택) |
| `/adapter {path}` | LoRA adapter 교체 (디버그용) |

### 렌더링

- `rich.Markdown` — 한문 인용 code block 하이라이트
- `rich.Spinner` — 생성 중 프로그레스
- Streaming: 토큰 단위 print (`end=""`, `flush=True`)
- 한자 출력 폰트 문제: 터미널 UTF-8 locale 요구 (README 에 명시)

## 10.6 Prompt Template — ChatML (Llama-3 chat)

Bllossom-8B 는 Llama-3 chat template 을 그대로 사용. 토크나이저가 `apply_chat_template` 지원.

```python
messages = [
    {"role": "system", "content": SYSTEM_PROMPT_HANMED},
    {"role": "user", "content": user_query},
    {"role": "assistant", "content": prev_response},  # 대화 히스토리
    {"role": "user", "content": new_query},
]
prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
```

### System prompt (v0 고정)

```
당신은 HanMed-LLM, 한의학 고전 문헌 연구 보조 AI 입니다.

역할:
- 한의학 고전(동의보감, 방약합편, 향약집성방 등) 의 내용을 한국어로 설명
- 한문 원문과 국역을 교차 이해하여 문헌 연구를 돕습니다
- 본초의 성미/귀경/효능, 처방의 구성, 고전 문구 해석을 제공합니다

반드시 지킬 것:
- 환자 진단, 처방 결정, 복용량 권고 등 임상 의사결정 요청은 거부하고,
  "의료 판단은 반드시 면허 한의사와 상의하세요" 로 안내합니다
- 출처가 불분명한 내용은 "문헌적 근거가 확실하지 않습니다" 로 명시합니다
- 한문 인용 시 원문·권/편·페이지를 밝힙니다

언어: 한국어로 답변 (사용자가 명시적으로 한문 요청 시 예외).
```

### 대화 히스토리 관리

- Bllossom-8B = Llama-3.1 기반 → 이론상 128K context, 실용적으로 **8K 제한** 권장 (안정성)
- 초과 시 **오래된 turn 부터 drop** (sliding window). 단 system prompt 는 항상 유지.
- 토큰 수 추정: `tok.apply_chat_template(..., return_tensors="pt").shape[-1]`

## 10.7 안전성 Refusal Layer

§05 T4 redteam 프롬프트를 대응한 **2단계 안전 필터**:

### Layer 1 — 사전 패턴 매치 (모델 호출 전)

```python
UNSAFE_PATTERNS = [
    r"내가.*아픈데.*처방",           # 자가 진단
    r"얼마나.*복용",                  # 복용량 문의
    r"임신.*(복용|먹어|먹도록)",      # 임신 관련 복용
    r"어린이.*(복용|먹여)",           # 소아 복용
    r"(양약|항생제).*대신",           # 양약 대체
    r"자살|자해",                     # 위기 상황
]
```

→ 매치되면 refusal 템플릿 반환, 모델 호출 생략. latency 절약.

### Layer 2 — 사후 출력 스크리닝

모델 출력에 "처방을 복용하세요", "하루 N 회 복용" 같은 구체적 임상 지시가 있으면 **끝에 disclaimer 자동 추가**:

```
⚠ 본 답변은 한의학 고전 문헌 설명이며 임상 의사결정이 아닙니다.
  복용 전 반드시 면허 한의사와 상의하세요.
```

→ v0 는 단순 regex, v1 는 별도 safety classifier.

## 10.8 세션 관리

```json
// ~/.hanmed/sessions/{name}.json
{
  "created": "2026-04-16T12:34:56Z",
  "model": {
    "base": "MLP-KTLim/llama-3-Korean-Bllossom-8B",
    "adapter": "outputs/sft_bllossom/adapter",
    "adapter_sha256": "..."
  },
  "system_prompt_version": "v0.1",
  "messages": [
    {"role": "user", "content": "...", "ts": "..."},
    {"role": "assistant", "content": "...", "ts": "...",
     "gen_stats": {"tokens": 128, "latency_ms": 2300}}
  ],
  "sampling": {"temperature": 0.7, "top_p": 0.9, "max_new_tokens": 1024}
}
```

저장 위치: `~/.hanmed/sessions/` (XDG_DATA_HOME 지원).

## 10.9 패키징 · 배포

### pyproject.toml

```toml
[project]
name = "hanmed-cli"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
  "click>=8.1",
  "prompt_toolkit>=3.0",
  "rich>=13.0",
  "vllm>=0.6.0 ; sys_platform == 'linux'",
  "transformers>=4.44",
  "peft>=0.13",
  "torch>=2.1",
  "pydantic>=2.0",
]

[project.scripts]
hanmed = "hanmed_cli.main:cli"
```

### 설치

```bash
# 일반 사용자
pip install hanmed-cli

# 개발
cd korean-medicine-llm
uv sync
uv run hanmed chat
```

### Adapter 배포

KIOM 승인 전제 (§07). 승인 후:
- HuggingFace Hub: `hanmed-llm/HanMed-SFT-v0.1`
- 첫 실행 시 `~/.hanmed/adapters/` 로 자동 다운로드

승인 전 로컬 개발:
```bash
hanmed chat --adapter outputs/sft_bllossom/adapter
```

## 10.10 성능 목표 (v0)

| 지표 | 목표 (A6000 48GB, bf16) | 측정 |
|---|---|---|
| First-token latency | < 1.0 s | vLLM warm, 짧은 프롬프트 |
| Throughput | ≥ 30 tok/s | vLLM |
| Peak GPU mem | < 30 GB | Bllossom 8B bf16 + LoRA + KV cache 8K |
| Cold start | < 30 s | base + adapter load |
| Session load | < 0.2 s | JSON parse |

## 10.11 디렉토리 구조 (신규)

```
korean-medicine-llm/src/
├── hanmed_cli/                      # 신규 패키지
│   ├── __init__.py
│   ├── main.py                      # Click entry
│   ├── chat.py                      # REPL loop
│   ├── conversation.py              # ChatML + history
│   ├── inference/
│   │   ├── base.py                  # 추상 Backend
│   │   ├── vllm_backend.py          # vLLM impl
│   │   └── transformers_backend.py  # fallback
│   ├── safety.py                    # refusal layer
│   ├── session.py                   # save/load
│   ├── render.py                    # Rich 출력
│   └── prompts/
│       └── system_v0.1.md           # SYSTEM_PROMPT_HANMED
└── training/...                     # 기존 학습 스택
```

## 10.12 마일스톤

| M | 주요 작업 | 의존 |
|---|---|---|
| **D1** (M5 이후) | vLLM 백엔드 + Bllossom 로드 + 단발 질의 | Stage 2 SFT adapter |
| **D2** | REPL + ChatML 히스토리 + streaming | D1 |
| **D3** | Safety refusal layer (§05 T4 프롬프트 30개 전수 테스트) | D2 + eval/T4 |
| **D4** | 세션 저장/불러오기 + `rich` 렌더 + slash commands | D2 |
| **D5** | transformers fallback backend | D3 |
| **D6** | pyproject.toml 패키징 + HF Hub adapter 업로드 (KIOM 승인 후) | 전체 + §07 |

## 10.13 Exit Criteria (v0 완성 조건)

| # | 조건 | 미달 시 |
|---|---|---|
| E1 | `hanmed chat` 한 줄로 REPL 진입, 첫 응답 < 5 s | 추론 백엔드 재검토 |
| E2 | §05 T4 redteam 30개 프롬프트 refusal rate ≥ 99% | safety layer 보강 |
| E3 | 멀티턴 8K context 에서 응답 일관성 유지 (sliding window 후에도 system prompt 보존) | conversation.py 수정 |
| E4 | 세션 save/load round-trip 무결성 | JSON schema 점검 |
| E5 | Peak GPU mem < 30 GB (8K context) | KV cache 축소 또는 quantization |

## 10.14 열린 결정

1. **GGUF 경로 지원**: llama.cpp 로 CPU/mac 지원 여부 — v1 에서 결정
2. **서버 모드**: `hanmed serve` (OpenAI API 호환) — v1
3. **도구 사용**: 간단한 RAG (mediclassics 원문 검색 후 인용) — v2 범위
4. **Quantization**: AWQ/GPTQ 4-bit 로 소비자 GPU (RTX 3090 등) 지원 — v1
5. **스트리밍 UI**: 한문 인용 블록이 렌더 중간에 오면 깨지는 문제 — `rich.live` 로 해결 검토

## 10.15 참고 구현 (사용자가 벤치마크할 것)

- [Simon Willison `llm`](https://github.com/simonw/llm) — SQLite 로그 + plugin 구조 참고
- [Ollama](https://github.com/ollama/ollama) — 단일 binary CLI 예시
- [llama.cpp interactive mode](https://github.com/ggerganov/llama.cpp/blob/master/examples/main/README.md) — `-i` flag
- [vLLM OpenAI server](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html) — v1 serve 모드 기반
- [Claude Code](https://docs.claude.com/en/docs/claude-code) — agentic REPL 참고 (v2 이후)
- [prompt_toolkit](https://python-prompt-toolkit.readthedocs.io/) — REPL UX
