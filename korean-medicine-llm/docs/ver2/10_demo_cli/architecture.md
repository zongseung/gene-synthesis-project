# 10.2 Architecture

```
┌─────────────────────────────────────────────────────────┐
│  hanmed CLI (Python package, Click entry)              │
│                                                         │
│  ┌───────────────┐   ┌────────────────┐  ┌──────────┐ │
│  │ prompt_toolkit│ → │ Conversation   │→ │ Safety   │ │
│  │   (REPL UX)   │   │ Manager        │  │ Filter   │ │
│  │ - 자동완성     │   │ - ChatML       │  │ - T4     │ │
│  │ - 히스토리     │   │ - 8K sliding   │  │   refusal│ │
│  │ - Rich 렌더    │   │ - session save │  │ - footer │ │
│  └───────────────┘   └────────────────┘  └────┬─────┘ │
│                                                │        │
│                                       ┌────────▼────┐   │
│                                       │ Inference   │   │
│                                       │ Backend     │   │
│                                       │  (abstract) │   │
│                                       └──┬──────┬───┘   │
└─────────────────────────────────────────┼──────┼────────┘
                                          ▼      ▼
                          ┌──────────────────┐ ┌────────────────┐
                          │ vLLM (primary)   │ │ transformers   │
                          │ - PagedAttn      │ │   (fallback)   │
                          │ - LoRA serving   │ │ - peft loader  │
                          │ - OpenAI API 호환│ │ - generate()   │
                          └──────────────────┘ └────────────────┘
                                     │                │
                                     └───────┬────────┘
                                             ▼
                             ┌───────────────────────────────┐
                             │ Bllossom-8B + HanMed adapter  │
                             │ (P-CPT or P-SFT, §10.3)       │
                             └───────────────────────────────┘
```

## 모듈 책임

| 모듈 | 책임 | 외부 의존 |
|---|---|---|
| `main.py` | Click CLI entry, 옵션 파싱, 백엔드 선택 | click, hanmed_cli.* |
| `chat.py` | REPL loop, slash commands, streaming | prompt_toolkit, rich |
| `conversation.py` | ChatML 포맷, 히스토리 sliding window | transformers (tokenizer only) |
| `safety.py` | 2-layer refusal (pre-regex + post-disclaimer) | re |
| `session.py` | JSON 세션 save/load | pathlib, json |
| `render.py` | Rich markdown, 한문 block 강조 | rich |
| `inference/base.py` | `Backend` 추상 클래스 | — |
| `inference/vllm_backend.py` | vLLM LoRA serving | vllm |
| `inference/transformers_backend.py` | HF generate + peft | transformers, peft, torch |

모듈 간 호출 그래프는 DAG (cycle 없음) — 테스트 격리 쉬움.

## 디렉토리 구조

```
korean-medicine-llm/src/hanmed_cli/
├── __init__.py
├── main.py                          # Click entry
├── chat.py                          # REPL loop
├── conversation.py                  # ChatML + history
├── safety.py                        # refusal layer
├── session.py                       # save/load
├── render.py                        # Rich 출력
├── inference/
│   ├── __init__.py
│   ├── base.py                      # Backend 추상 클래스
│   ├── vllm_backend.py
│   └── transformers_backend.py
└── prompts/
    └── system_v0.1.md               # SYSTEM_PROMPT_HANMED
```

관련 tests:
```
tests/hanmed_cli/
├── test_safety_pre_patterns.py      # §05 T4 20개 refusal + 30 paraphrase held-out
├── test_conversation_sliding.py     # 8K context window 보존
├── test_session_roundtrip.py        # save/load 무결성
└── test_chatml_template.py          # Llama-3 chat template 적용
```

## 데이터 플로우 (한 turn)

```
1. 사용자 입력 "인삼의 성미 알려줘"
   │
2. Safety Filter.pre_check(user_input)
   │  - 임상 의사결정 regex 매치? → refusal 즉시 반환
   │  - 아니면 통과
   │
3. Conversation.append_user(user_input)
   │  - Llama-3 chat template 적용
   │  - 8K 초과면 가장 오래된 turn drop (system 유지)
   │
4. Inference Backend.generate(messages, sampling)
   │  - vLLM: LLM.generate() with LoRARequest
   │  - transformers: model.generate() with peft adapter
   │  - streaming iterator
   │
5. Render.stream(iterator)
   │  - 토큰 단위 stdout flush
   │  - 한문 block 감지 시 rich.Syntax 강조
   │
6. Safety Filter.post_check(response)
   │  - "하루 3회", "5일 동안 복용" 류 패턴 → disclaimer 자동 추가
   │  - footer "— KIOM mediclassics.kr 기반 학습" 고정
   │
7. Conversation.append_assistant(response)
   │
8. (optional) Session.autosave()
```

## 에러 처리

| 상황 | 동작 |
|---|---|
| GPU OOM | streaming 중단, `max_new_tokens` 절반 낮춰 재시도 1회. 실패 시 사용자에게 안내 |
| adapter load 실패 | CLI 시작 시 명시적 에러, exit 1 |
| `prompt_toolkit` 이상 종료 (Ctrl+D) | 세션 autosave 후 정상 종료 |
| 모델 무한 생성 (repeat loop) | stopping criteria: repetition_penalty 1.1, max_new_tokens 1024 |
| safety filter positive (pre) | 모델 호출 생략, 표준 refusal 템플릿 반환, latency ~ 0 |

## 상태 관리

- Conversation: in-memory, 세션 save 시 JSON serialize
- Backend: lazy init (첫 generate 때 모델 로드)
- Session: 파일 시스템 (`$XDG_DATA_HOME/hanmed/sessions/*.json`, fallback `~/.local/share/hanmed/sessions/*.json`) + atomic write
- Config: v0 runtime env 는 최소 (`XDG_DATA_HOME`, `HANMED_ADAPTER_DIR`). `HANMED_API_KEY` 는 v1 `serve` 전용

R3.5 메모:
- backend lazy init 을 유지하면 REPL 진입은 빠르지만 첫 질의 latency 는 cold model load 를 포함한다. 따라서 성능 보고는 `REPL ready`, `cold first-token`, `warm first-token` 3지표로 분리한다.
- 세션 재현성 확보를 위해 message timestamp 는 UTC 저장, UI 에서만 locale 변환한다.

## 확장 포인트 (v1 계획)

- `hanmed serve` — vLLM OpenAI API wrap (10.7 S2)
- plugin 시스템 — Simon Willison `llm` 스타일 (safety patterns, render 변형 등을 plugin 으로)
- GGUF 경로 — llama.cpp backend 추가 (CPU 지원)
