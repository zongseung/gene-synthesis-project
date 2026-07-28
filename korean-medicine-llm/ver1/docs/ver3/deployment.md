# ver3 · 문서 2 — Deployment (v0 로컬 REPL → v1 vLLM serve + Docker + 클라우드)

> **역할**: v0 로컬 REPL (`hanmed chat` transformers backend) 이 M2 smoke 까지 완료된 상태에서 **v1 vLLM backend + `hanmed serve` + Docker 패키징 + 클라우드 배포** 를 M3 에 공식 착수한다. ver2.2 §10.7 배포 매트릭스 L / S1 / S2 / C 의 **M2 · M3 · M4 실행 계획** 을 확정.

> **작성 근거**: pilot adapter (`outputs/cpt_bllossom/adapter/`) 는 이미 산출. `chat_template.jinja` 보존 확인 → P-CPT 경로 vLLM 적재 가능. CLI v0 smoke (Q1~Q4) 는 transformers backend 로 이미 통과. ver2.2 §10.4 의 "vLLM backend skeleton" 을 **실가동** 단계로 옮긴다.

## 1. 배포 스테이지 매트릭스 (ver2 §10.7 확장)

ver2.2 §10.7 의 L / S1 / S2 / C 4단계를 ver3 시점 상태와 M2/M3/M4 일정으로 확장:

| 경로 | 버전 | 환경 | 현재 상태 (ver3) | M2 액션 | M3 액션 | M4 액션 |
|---|---|---|---|---|---|---|
| **L-로컬** | **v0** | 개발자 A6000 단일 | ✅ 완료 (transformers backend, smoke Q1~Q4 통과) | 유지 (smoke suite 보강) | vLLM backend 활성화 | 유지 |
| **S1-기관 SSH** | v0 | 가천대 GPU 서버 | ✅ 가능 (uv sync + REPL) | IT 팀 SSH 계정 요청 | Unix 계정 분리 확정 | 유지 |
| **S2-기관 HTTP** | **v1** | 가천대 GPU 서버 + `hanmed serve` | ❌ 미개발 | skeleton 설계 | **구현 + vLLM 연동** | 기관 내부 4~8 동시 사용 검증 |
| **C-클라우드** | **v1** | RunPod A6000 / AWS g5 | ❌ 미배포 | Docker 설계 | **Docker 빌드 + registry** | RunPod 파일럿 + KIOM 승인 병행 |
| **C-공개 웹** | **v2** | 공개 웹 + 상업 API | ❌ | — | — | 상업 라이선스 + 의료기기 규제 (ver3 scope 초과) |

**핵심 변화 (ver2.2 → ver3)**:
- v1 `hanmed serve` 를 2026 Q3 이 아닌 **M3 (2026 Q2 말)** 에 착수 — pilot adapter 가 이미 존재하므로 서버 측 기술 준비 지연 사유 없음
- v1 C (RunPod) 파일럿을 KIOM 승인 전에 **내부 VPC / private registry 로만** 실행 가능 → "승인 대기 = 배포 대기" 의 사슬 해제

## 2. vLLM backend 활성화 (ver2.2 §10.4.1 skeleton → 실가동)

### 2.1 환경 요구사항

| 항목 | 값 | 근거 |
|---|---|---|
| vllm | **≥ 0.6.0** | ver2.2 §10.8.1 `[project.optional-dependencies]` `vllm = ["vllm>=0.6.0"]` |
| PyTorch | 2.1+ | ver2.2 §10.8.1 |
| CUDA | 12.1+ | pilot 환경 동일 (manifest 에서 2.1.x 확인) |
| GPU | A6000 48GB 1대 | v1 S2 single-GPU. tensor_parallel 은 v2 |
| Python | 3.10~3.12 | pilot 환경 (`.venv/lib/python3.12`) |

`pyproject.toml` 수정 (`korean-medicine-llm/pyproject.toml`):
- `[project.optional-dependencies]` → `vllm = ["vllm>=0.6.0"]` 그대로 유지
- `uv sync --extra vllm` 으로 설치

**M3 gate**: `uv run python -c "from vllm import LLM; print(LLM.__module__)"` 이 성공.

### 2.2 `VLLMBackend` 구현 명세 (`src/hanmed_cli/inference/vllm_backend.py`)

ver2.2 §10.4.1 pseudo-code 를 실가동 코드로 승격. ver2.2 §10.4.3 Backend abstract 에 합치:

```python
# src/hanmed_cli/inference/vllm_backend.py (planned, ~250 LoC)

from typing import AsyncIterator
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from vllm.lora.request import LoRARequest
from .base import Backend, SamplingConfig

class VLLMBackend(Backend):
    """ver2.2 §10.4.1 실가동. LoRA dynamic 로드 지원."""

    def __init__(self) -> None:
        self.engine: AsyncLLMEngine | None = None
        self.lora_request: LoRARequest | None = None

    def load(
        self,
        base_model: str = "MLP-KTLim/llama-3-Korean-Bllossom-8B",
        adapter_path: str = "outputs/cpt_bllossom/adapter",
        *,
        max_model_len: int = 8192,          # §10.10.2
        max_lora_rank: int = 32,            # §04a §C.5 LoRA r=32 (pilot 실측)
        gpu_memory_utilization: float = 0.85,
        tokenizer_path: str | None = "data/tokenizer/hanmed_bllossom_ext",
        enforce_eager: bool = False,
    ) -> None:
        args = AsyncEngineArgs(
            model=base_model,
            tokenizer=tokenizer_path or base_model,
            dtype="bfloat16",               # ver2.2 §4.3
            enable_lora=True,
            max_lora_rank=max_lora_rank,
            max_loras=1,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            enforce_eager=enforce_eager,    # CUDA graph 문제 시 True
        )
        self.engine = AsyncLLMEngine.from_engine_args(args)
        self.lora_request = LoRARequest(
            lora_name="hanmed_cpt",
            lora_int_id=1,
            lora_path=adapter_path,
        )

    async def stream_generate(
        self, prompt: str, cfg: SamplingConfig
    ) -> AsyncIterator[str]:
        assert self.engine is not None and self.lora_request is not None
        params = SamplingParams(
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_tokens=cfg.max_new_tokens,
            repetition_penalty=cfg.repetition_penalty,
            stop_token_ids=[128009],        # Llama-3 <|eot_id|>
        )
        request_id = f"req-{id(prompt)}"
        async for output in self.engine.generate(
            prompt, params, request_id,
            lora_request=self.lora_request,
        ):
            # delta token 만 yield
            yield output.outputs[0].text

    def close(self) -> None:
        if self.engine is not None:
            # AsyncLLMEngine shutdown (vLLM 0.6 API)
            self.engine = None
```

**핵심 설계 결정**:
- `enable_lora=True` + `max_lora_rank=32` 로 PagedAttention 에 LoRA 동적 로드 허용. pilot adapter 의 r=32 와 정합
- `max_model_len=8192` — ver2.2 §10.5.3 "실용 8K" 와 일치. `max_position_embeddings` 는 Llama-3 에서 8192~131072 중 선택 가능하나 KV cache 절약 위해 8192
- `gpu_memory_utilization=0.85` — A6000 48GB 에서 여유 7GB (OS + driver + 기타 프로세스)
- `stop_token_ids=[128009]` — pilot adapter 의 chat template 이 Llama-3 `<|eot_id|>` 사용 (`adapter/chat_template.jinja` 확인)
- `enforce_eager` fallback — vLLM CUDA graph 빌드 실패 시 eager 로 degrade (throughput 20~30% 하락하지만 compatibility 확보)

### 2.3 Throughput 목표 (ver2.2 §10.4.4 확장)

| 지표 | 목표 | 근거 |
|---|---|---|
| First-token latency (warm, vLLM) | **< 0.5s** | ver2.2 §10.4.4 |
| Throughput | **≥ 40 tok/s** | ver2.2 §10.4.4 |
| vs transformers (pilot smoke 관측치 ~15~25 tok/s) | **2× 이상** | 동일 §10.4.4 |
| Cold start (vLLM 엔진 초기화 + adapter 로드) | **< 25s** | 모델 가중치 로드 + KV cache pre-allocate |
| Peak GPU mem @ 8K ctx | **< 30 GB** | `gpu_memory_utilization=0.85` × 48GB = 40.8GB + padding |
| 동시 사용자 (continuous batching) | **2~4** | single A6000 기준 보수 추정. vLLM 0.6 scheduler |

**M3 실측 과제**: `scripts/bench_vllm.py` (planned, ~80 LoC) 로 A6000 에서 위 지표 전수 측정.

### 2.4 CLI backend 스위치

`src/hanmed_cli/main.py` 에 `--backend` 플래그 추가:

```bash
# vLLM (primary from M3)
hanmed chat --backend vllm --adapter outputs/cpt_bllossom/adapter

# transformers (fallback, 현재 v0 default)
hanmed chat --backend transformers --adapter outputs/cpt_bllossom/adapter

# auto-detect (default): vllm import 성공하면 vllm, 실패하면 transformers
hanmed chat --adapter outputs/cpt_bllossom/adapter
```

`src/hanmed_cli/config.py` 의 backend resolver 에 try-import 추가.

## 3. `hanmed serve` 서브커맨드 스펙 (§10.7 S2)

v1 S2 기관 HTTP 모드. **vLLM OpenAI-compatible API** 래핑.

### 3.1 Click group 확장

`src/hanmed_cli/main.py` 의 `cli` group 에 `serve` 서브커맨드 추가:

```python
# src/hanmed_cli/serve.py (planned, ~200 LoC)

@click.command()
@click.option("--host", default="0.0.0.0")
@click.option("--port", default=8000, type=int)
@click.option("--base-model", default="MLP-KTLim/llama-3-Korean-Bllossom-8B")
@click.option("--adapter", default="outputs/cpt_bllossom/adapter")
@click.option("--tokenizer", default="data/tokenizer/hanmed_bllossom_ext")
@click.option("--api-key", envvar="HANMED_API_KEY", required=True,
              help="Bearer token. 환경변수 HANMED_API_KEY 로도 설정 가능.")
@click.option("--rate-limit", default=20, type=int,
              help="requests per minute per API key (default 20)")
@click.option("--max-model-len", default=8192, type=int)
@click.option("--safety-mode", default="on",
              type=click.Choice(["on", "off"]),
              help="pre-pattern refusal (§10.5.4 Layer 1)")
@click.option("--footer",
              default="KIOM mediclassics.kr 기반 학습 (한의학고전DB)",
              help="응답 footer (§10.5.4 고정 footer)")
def serve(host, port, base_model, adapter, tokenizer, api_key, rate_limit,
          max_model_len, safety_mode, footer):
    """v1 S2 — hanmed serve (vLLM OpenAI API wrapper)."""
    # 1. vLLM OpenAI server 는 공식 명령 `vllm serve` 를 내부 호출 대신
    #    FastAPI 로 직접 라우팅하여 middleware (auth, rate limit, safety) 삽입
    from .server import build_app
    import uvicorn
    app = build_app(
        base_model=base_model,
        adapter=adapter,
        tokenizer=tokenizer,
        api_key=api_key,
        rate_limit_per_min=rate_limit,
        max_model_len=max_model_len,
        safety_mode=safety_mode,
        footer=footer,
    )
    uvicorn.run(app, host=host, port=port)
```

### 3.2 HTTP 엔드포인트 (OpenAI-compatible subset)

| 엔드포인트 | 메서드 | 용도 | auth |
|---|---|---|---|
| `/v1/models` | GET | 현재 로드된 model + adapter 목록 | Bearer |
| `/v1/chat/completions` | POST | OpenAI chat completion 호환 | Bearer |
| `/v1/completions` | POST | OpenAI text completion 호환 | Bearer |
| `/health` | GET | liveness probe (200 OK) | none |
| `/metrics` | GET | Prometheus metrics (v1 옵션) | IP allowlist |

**non-OpenAI 필드 추가**:
- `/v1/chat/completions` response 에 `hanmed_footer` 필드 (§10.5.4 footer 텍스트)
- `hanmed_safety_triggered: bool` — Layer 1 pattern match 로 모델 호출 건너뛴 경우 true

### 3.3 Middleware 구조

```
HTTP request
    │
    ▼
[1] Bearer auth check (X-API-Key 또는 Authorization: Bearer ...)
    │  fail → 401
    ▼
[2] Rate limiter (slowapi 기반, key 별 분단위 bucket)
    │  over → 429
    ▼
[3] Safety Layer 1 (§10.5.4 UNSAFE_PATTERNS)
    │  match → 모델 호출 건너뛰고 REFUSAL_TEMPLATE + footer 반환
    ▼
[4] vLLM AsyncEngine generate
    │
    ▼
[5] Safety Layer 2 (§10.5.4 CLINICAL_DIRECTIVE_PATTERNS) + disclaimer 주입
    │
    ▼
[6] Footer append
    │
    ▼
HTTP response
```

### 3.4 Auth 모델

- **Bearer token** (`HANMED_API_KEY` env 또는 `--api-key`) — v1 단일 key 운영 (기관 내부)
- v2 에서 **multi-key + DB** (Postgres / SQLite) 확장 검토 — 사용자별 rate limit, 폐기, 감사 로그
- KIOM 승인 이후 **IP allowlist** 병행 가능 (VPN 대역만 허용)

### 3.5 Rate limit

- 기본 **20 req/min/key** (ver2.2 §10.10.2 와 합치)
- slowapi 라이브러리 사용 (FastAPI 표준)
- 초과 시 HTTP 429 + `Retry-After` 헤더

### 3.6 로깅 (privacy 준수)

- **프롬프트 원문 저장 금지** (§07.1 KIOM 재배포 해석)
- 집계 metric 만:
  - request timestamp · API key hash · prompt token count · response token count · latency · safety_triggered bool · refusal_triggered bool
- 저장 위치: `logs/serve/YYYY-MM-DD.jsonl` (기관 서버 로컬)
- rotation: 30일 후 삭제 (KIOM 정기 보고용 월별 집계만 보존)

### 3.7 `hanmed serve` exit criteria (M3 gate)

| # | 조건 | 측정 |
|---|---|---|
| SE1 | localhost:8000 에서 `/health` 200 응답 | curl |
| SE2 | Bearer token 누락 → 401 | curl without auth |
| SE3 | 21번째 req/min → 429 | 부하 스크립트 |
| SE4 | `/v1/chat/completions` 정상 응답 + footer + Q1 한의학 register | integration test |
| SE5 | Q4 (증상 호소) → safety_triggered=true + REFUSAL_TEMPLATE | integration test |
| SE6 | 동시 2 사용자 throughput ≥ 60 tok/s 합산 | bench |
| SE7 | 세션 10시간 유지 후 메모리 누수 < 5% | `nvidia-smi` watch |

## 4. Docker 패키징

ver2.2 §10.8.2 에는 3-line Dockerfile 스케치만 존재. ver3 M3 에서 **multi-stage build** + **immutable adapter pin** 으로 확장.

### 4.1 Dockerfile (`docker/Dockerfile.hanmed-serve`, planned)

```dockerfile
# syntax=docker/dockerfile:1.6

# -------- Stage 1: base CUDA runtime + Python --------
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3-pip git curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:${PATH}"

# -------- Stage 2: build dependencies --------
FROM base AS builder

WORKDIR /build
COPY pyproject.toml uv.lock ./
COPY src/ ./src/

# uv 로 .venv 생성 + vllm extra 설치
RUN uv sync --frozen --extra vllm

# -------- Stage 3: runtime image --------
FROM base AS runtime

WORKDIR /app

# builder 에서 .venv 복사
COPY --from=builder /build/.venv /app/.venv
COPY --from=builder /build/src /app/src
COPY pyproject.toml /app/pyproject.toml

# adapter · tokenizer volume mount point
VOLUME ["/app/outputs", "/app/data/tokenizer", "/root/.cache/huggingface", "/root/.hanmed"]

ENV PATH="/app/.venv/bin:${PATH}" \
    HANMED_API_KEY="" \
    HANMED_ADAPTER="/app/outputs/cpt_bllossom/adapter" \
    HANMED_TOKENIZER="/app/data/tokenizer/hanmed_bllossom_ext" \
    HANMED_BASE_MODEL="MLP-KTLim/llama-3-Korean-Bllossom-8B"

EXPOSE 8000

# 기본 진입점 = hanmed serve
ENTRYPOINT ["hanmed"]
CMD ["serve", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--adapter", "/app/outputs/cpt_bllossom/adapter", \
     "--tokenizer", "/app/data/tokenizer/hanmed_bllossom_ext"]
```

### 4.2 이미지 크기 목표

| 구성 | 예상 크기 |
|---|---|
| CUDA 12.1 runtime base | ~2.5 GB |
| Python 3.10 + uv | +0.1 GB |
| `.venv` (torch, vllm, transformers, peft, click, fastapi, uvicorn) | ~5~6 GB |
| src + pyproject | ~0.05 GB |
| **base image (w/o adapter)** | **~8~9 GB** |
| adapter (`outputs/cpt_bllossom/adapter/`) | ~500 MB (r=32 기준) — **volume mount 권장, image bake 비권장** |

**ver3 결정**: adapter 는 image 에 bake 하지 않고 **volume mount**. 이유:
- KIOM 라이선스 (§07.1): adapter 공개는 서면 승인 필요 → private image 에 bake 하면 registry 접근자가 자동 취득
- 버전 관리: adapter 별도 배포 (HF private repo 또는 S3 pre-signed URL) 로 CLI 와 분리 (§10.8.4 "adapter ≠ CLI version")

### 4.3 실행

```bash
# adapter · HF cache · session volume 을 host 에서 mount
docker run --gpus all -d \
  --name hanmed-serve \
  -p 8000:8000 \
  -v $PWD/outputs:/app/outputs:ro \
  -v $PWD/data/tokenizer:/app/data/tokenizer:ro \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ~/.hanmed:/root/.hanmed \
  -e HANMED_API_KEY="${HANMED_API_KEY}" \
  ghcr.io/zongseung/hanmed-cli:0.2.0
```

### 4.4 docker-compose.yml (기관 서버 운영 편의)

```yaml
# docker/docker-compose.yml (planned)
version: "3.9"
services:
  hanmed-serve:
    image: ghcr.io/zongseung/hanmed-cli:0.2.0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    ports:
      - "8000:8000"
    volumes:
      - ../outputs:/app/outputs:ro
      - ../data/tokenizer:/app/data/tokenizer:ro
      - hf-cache:/root/.cache/huggingface
      - hanmed-sessions:/root/.hanmed
    environment:
      HANMED_API_KEY: ${HANMED_API_KEY}
      HANMED_ADAPTER: /app/outputs/cpt_bllossom/adapter
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 5s
      retries: 3

volumes:
  hf-cache:
  hanmed-sessions:
```

### 4.5 Docker M3 exit

| # | 조건 |
|---|---|
| DE1 | `docker build -f docker/Dockerfile.hanmed-serve -t hanmed-cli:0.2.0 .` 성공 |
| DE2 | 이미지 크기 ≤ 10 GB (adapter 제외) |
| DE3 | `docker run --gpus all hanmed-cli:0.2.0` 후 `/health` 200 응답 |
| DE4 | volume mount 로 adapter 적재 → CLI smoke Q1~Q4 통과 |
| DE5 | GHCR push 성공 (`ghcr.io/zongseung/hanmed-cli:0.2.0`) |

## 5. 클라우드 배포 사전조건 (ver2.2 §10.7 C + §07 확장)

### 5.1 KIOM 서면 승인 프로토콜 (`ver2/07_license_ethics/license_ethics.md` §7.1.3)

| 단계 | 기간 | 산출물 |
|---|---|---|
| 1차 문의 (이메일) | 1~4주 | `kiombook@kiom.re.kr` 문의서 + 연구계획 |
| 자료 제출 | 1~2주 | 이용 범위 · 공개 계획 · adapter sha256 pin |
| 법무 검토 | 2~6주 | KIOM 내부 절차 |
| MOU / 이용허락서 | 2~6주 | 서면 |
| **총** | **2~6개월** | — |

**ver3 M3 착수**: **cloud C 경로 파일럿은 KIOM 승인 이전에도 private VPC 에서 가능** — 단 (a) image + adapter 가 registry 외부로 유출되지 않음, (b) 외부 접근자 없음, (c) 로그에 프롬프트 저장 없음 — 3 조건 충족 시. M4 공개 승격은 서면 승인 필수.

### 5.2 모델 카드 (HuggingFace Hub `hanmed-llm/HanMed-CPT-v0.1`)

ver2.2 §10.8.3 model card 필수 항목 확장:

- 학습 데이터 출처 — `data/cpt_processed/corpus_v2.json` SHA-256 pin 인용
- 학습 레시피 — Bllossom-8B + LoRA r=32 α=64 dropout 0.05, cap=60M (M3 본 run), epoch 3, Adam β=(0.9, 0.95), LR 1e-4, cosine warmup 5%
- pilot 실측 인용 — eval_loss 1.887, ppl 6.60, 156 steps, 2h 53m (Core 14, cap 20.4M)
- 한계 — 임상 의사결정 금지 (§07.6.2 disclaimer)
- 평가 결과 — §05 T1~T5 primary 수치
- 라이선스 — "Llama 3 Community License (base) ∩ KIOM 비상업 이용 (corpus)" — 최종 조항은 KIOM 승인 이후 확정
- adapter SHA-256 pin — `outputs/cpt_bllossom_main_60M/adapter/adapter_model.safetensors`

### 5.3 출처 표기 footer (고정)

모든 API 응답 (cloud `/v1/chat/completions` · `/v1/completions`) 에 고정 footer:

```
— KIOM mediclassics.kr 기반 학습 (한의학고전DB)
```

(ver2.2 §10.5.4 footer 문자열 그대로)

### 5.4 사용량 로깅 원칙 (§3.6 재인용)

- 프롬프트 원문 저장 금지
- 집계 metric 만: request count · token count · latency · safety event
- KIOM 정기 보고용 월별 aggregate JSON → 수동 제출

### 5.5 IRB / 의료기기 규제 자문 (§07.6)

- v1 (기관 내부 · RunPod private) 범위 = **연구 보조 도구** → IRB 면제 유지
- v2 (공개 웹 · 환자 대상) 범위 = **보건복지부 / 식약처 자문 필수** (ver3 scope 초과)
- M4 KIOM 승인 병행 트랙에서 IRB 면제 재확인 (v1 범위 이내)

## 6. 클라우드 후보 비교 (ver2.2 §10.7 확장)

ver2.2 §10.7 표를 2026-04-17 시점 재측정:

| 제공자 | GPU | 시간당 비용 | LoRA serving 지원 | 한국 리전 | 추천 시나리오 |
|---|---|---|---|---|---|
| **RunPod** | A6000 48GB | ~$0.79 | ✅ vLLM 직접 | 간접 (SG / US) | **v1 primary** — 비용 최저, A6000 이 pilot/main 과 동일 |
| AWS g5.2xlarge | A10G 24GB | ~$1.00 | ✅ vLLM | ICN | A10G 메모리 부족 위험 (Bllossom-8B bf16 = 16GB + KV cache → 24GB 여유 < 5GB). CLI Q4 길이 응답 시 OOM |
| AWS g5.4xlarge | A10G 24GB | ~$1.62 | ✅ | ICN | 동일 GPU, 더 많은 vCPU |
| AWS g6.2xlarge | L4 24GB | ~$1.09 | ✅ | ICN | A10G 와 유사 제약 |
| AWS p4d.24xlarge | A100 40GB × 8 | ~$32 | ✅ | ICN | 과도, v2 tensor_parallel 전용 |
| Lambda Labs | A6000 48GB | ~$0.80 | ✅ | US | 재고 부족 빈번 |
| Paperspace | A6000 48GB | ~$1.10 | ✅ | US | IDE 통합 유리 |
| **HF Inference Endpoints** | L4 / A10 / A100 | ~$0.60~1.50 | ⚠️ **LoRA serving 제한** (2026-04-17 시점 LoRA hot-swap 미정식 지원) | US/EU | adapter merge 후에만 사용. **v2 옵션** |
| **자체 on-prem (가천대)** | A6000 | 0 | ✅ | KR | **v0 S2 + v1 파일럿 최우선** — 라이선스 안전 |

### 6.1 v1 cloud primary 결정 (ver2.2 §10.7 재확인)

**RunPod A6000 48GB, ~$0.79/h**:
- pilot/main 학습과 동일 GPU → 추가 성능 튜닝 불필요
- vLLM LoRA serving 공식 지원
- GPU secure cloud (VPC 분리) 선택 가능 → KIOM 승인 전 private 운영

### 6.2 비용 예측 (월 100 active hours)

- GPU: $0.79/h × 100h = **$79/월**
- persistent storage (adapter + HF cache) 50GB: ~$3.5/월
- egress (API response 트래픽, 100GB): ~$5/월
- **월 합계 ≈ $87~90/월** (ver2.2 §10.7 비용 추정 "$90/월" 과 정합)

**월 500 active hours (본격 파일럿)**: ~$430/월. 예산 상한 결정은 M4 시점.

## 7. CI/CD 파이프라인 (ver2.2 §10.8.5 v1 확장)

### 7.1 GitHub Actions workflow (`.github/workflows/ci.yml`, planned)

```yaml
name: CI

on:
  pull_request:
  push:
    branches: [main]
    tags: ["v*"]

jobs:
  test:
    runs-on: ubuntu-22.04
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
      - run: uv sync --extra dev
      - run: uv run pytest tests/hanmed_cli --cov=src/hanmed_cli --cov-report=xml
      - run: uv run coverage report --fail-under=80

  build-docker:
    needs: test
    if: startsWith(github.ref, 'refs/tags/v')
    runs-on: ubuntu-22.04
    permissions:
      contents: read
      packages: write
    steps:
      - uses: actions/checkout@v4
      - uses: docker/setup-buildx-action@v3
      - uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      - run: |
          docker buildx build --platform linux/amd64 \
            -f docker/Dockerfile.hanmed-serve \
            -t ghcr.io/${{ github.repository_owner }}/hanmed-cli:${{ github.ref_name }} \
            -t ghcr.io/${{ github.repository_owner }}/hanmed-cli:latest \
            --push .

  release-pypi:
    needs: test
    if: startsWith(github.ref, 'refs/tags/v') && !contains(github.ref, 'internal')
    runs-on: ubuntu-22.04
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
      - run: uv build
      - uses: pypa/gh-action-pypi-publish@release/v1
        with:
          password: ${{ secrets.PYPI_TOKEN }}
        # 주의: KIOM 승인 전 publish 차단 — 태그 convention `v0.x-internal` 은 skip
```

### 7.2 Adapter SHA-256 pin (immutable release)

`pyproject.toml` 의 `[tool.hanmed.release]` 섹션에 매 release 시 기록 (ver2.2 §10.8.4 확장):

```toml
[tool.hanmed.release]
version = "0.2.0"
adapter_name = "HanMed-CPT-v0.1"
adapter_sha256 = "<sha256 of outputs/cpt_bllossom_main_60M/adapter/adapter_model.safetensors>"
adapter_source = "hf:hanmed-llm/HanMed-CPT-v0.1@<commit>"  # KIOM 승인 후에만 public
base_model = "MLP-KTLim/llama-3-Korean-Bllossom-8B"
base_revision = "3c9b6f7..."  # immutable HF snapshot revision (never "main")
corpus_manifest_sha256 = "<sha256 of corpus_v2.json>"
```

CI job `verify-adapter`: tag 빌드 시 adapter sha256 이 `pyproject.toml` 의 값과 일치하는지 검증. 불일치 → build fail.

### 7.3 Release 태그 규약

| 태그 예시 | 의미 | publish |
|---|---|---|
| `v0.1.x-internal` | 기관 내부 테스트 | Docker → private GHCR only |
| `v0.2.x-preview` | M3 완료 후 preview | Docker → GHCR, PyPI skip |
| `v1.0.0` | KIOM 승인 이후 공개 | Docker GHCR + public, PyPI publish |

## 8. v0 → v1 → v2 승격 gate (ver2.2 §10.9.3/.4 확장)

### 8.1 v0 → v1 exit

ver2.2 §10.9.3 표 + ver3 확장:

| 조건 | 내용 | 측정 시점 |
|---|---|---|
| `hanmed serve` 구현 완료 | §3 SE1~SE7 통과 | M3 end |
| vLLM backend exit | §2.3 throughput 목표 | M3 bench |
| Docker 이미지 빌드 | §4.5 DE1~DE5 | M3 end |
| CI/CD 파이프라인 | §7.1 test + docker jobs green | M3 end |
| Stage 2 SFT adapter | (문서 1 §6) `outputs/sft_bllossom_v0.1` | M3 end |
| KIOM 1차 문의 송부 | §7.1.3 이메일 | M3 week 1 (병렬) |
| E6 chat template preservation | §10.9.2 E6 (ΔEOT-rate < 2%p, 3 seed, 200 generic prompt) | M3 end |

### 8.2 v1 → v2 exit

| 조건 | 내용 | 예상 시점 |
|---|---|---|
| KIOM 서면 승인 (MOU/이용허락서) | §7.1.3 | M4+ |
| 상업 라이선스 협상 (opt) | §07.1.2 "상업 이용 → kiombook@kiom.re.kr" | 2027+ |
| 의료기기 규제 자문 | 보건복지부 / 식약처 | 2027+ |
| 클라우드 public deployment | RunPod → public endpoint | M4+ |
| 웹 UI | React/Next.js 클라이언트 | ver3 scope 초과 |
| 다국어 safety classifier | §10.5.6 열린 결정 1 | ver3 scope 초과 |

## 9. 로드맵 — M2 / M3 / M4

### 9.1 M2 (현재 — vLLM 붙이기)

| 주차 | 작업 | 산출물 | gate |
|---|---|---|---|
| W1 | vLLM 0.6+ 환경 구성 + smoke (`uv sync --extra vllm`) | `.venv/` with vllm | M3 prep |
| W1 | `src/hanmed_cli/inference/vllm_backend.py` 구현 (§2.2) | 250 LoC | backend tests |
| W2 | `--backend {vllm|transformers|auto}` 스위치 (§2.4) | CLI update | smoke Q1~Q4 양 backend |
| W2 | `scripts/bench_vllm.py` 구현 + A6000 측정 | bench log | §2.3 목표 수치 |
| W3 | v0 smoke 보강 (Q1~Q4 외 10 추가) | `tests/hanmed_cli/test_smoke.py` | E2 기초 |

**M2 exit**: vLLM backend 로 CLI 단발 generate 성공 + throughput ≥ 40 tok/s 실측.

### 9.2 M3 (serve + Docker)

| 주차 | 작업 | 산출물 | gate |
|---|---|---|---|
| W1 | `hanmed serve` 서브커맨드 구현 (§3.1) | `src/hanmed_cli/serve.py` + `server.py` | SE1~SE5 |
| W1 | FastAPI middleware (auth + rate limit + safety) | ~300 LoC | SE2, SE3 |
| W2 | serve integration test (pytest + httpx) | `tests/hanmed_cli/test_serve.py` | SE1~SE7 |
| W2 | Dockerfile + docker-compose.yml (§4) | `docker/` | DE1~DE4 |
| W3 | CI/CD workflow (§7) | `.github/workflows/ci.yml` | GHCR push green |
| W3 | Stage 2 SFT adapter 산출 (문서 1 §6.5) | `outputs/sft_bllossom_v0.1/adapter` | P-SFT 경로 ready |
| W4 | `--mode sft` 런타임 스위치 (§10.3 P-SFT) | CLI update | P-SFT smoke |

**M3 exit**: v0 → v1 exit 전 조건 (§8.1) 통과.

### 9.3 M4 (클라우드 + KIOM 승인 병행)

| 주차 | 작업 | 산출물 |
|---|---|---|
| W1 | RunPod A6000 private GPU secure cloud 계약 + 초기 설정 | runbook |
| W1 | Docker image push (GHCR private tag `v0.2.x-internal`) | registry |
| W2 | RunPod 파일럿 배포 + integration test (VPC 내부) | test log |
| W2 | 월별 집계 로그 포맷 확정 (§3.6) + 첫 리포트 | `reports/usage_2026-05.json` |
| W3 | 기관 HTTP S2 배포 (가천대 GPU + docker-compose) | deploy log |
| W3 | KIOM 1차 문의 송부 + 자료 제출 | email + 연구계획 PDF |
| W4 | 전문가 3-5인 기관 내부 테스트 세션 | feedback log |

**M4 exit**: 기관 내부 S2 stable 운영 + KIOM 승인 프로세스 진행 중. v1 공개 배포 (v1 → v2) 는 KIOM 승인 완료 시점에 별도 trigger.

## 10. 열린 결정

1. **vLLM 버전 핀 0.6 vs 0.7** — 2026-04-17 시점 vLLM 0.7.x 가 LoRA dynamic API 일부 변경. ver3 M3 는 **0.6.x 핀** (`vllm>=0.6.0,<0.7`) 으로 고정, 0.7 migration 은 M4 이후 검토.
2. **Docker image CUDA 12.1 vs 12.2** — pilot 환경이 CUDA 12.x (정확 12.1 확인 필요). vLLM 0.6 은 CUDA 12.1~12.4 지원. ver3 기본 12.1, migration 시 build matrix 확장.
3. **HF Inference Endpoints LoRA 지원 재검토** — 2026-04-17 기준 LoRA hot-swap 미지원. 공식 지원 발표 시 C 경로 단순화 가능 (adapter merge 불필요). 분기별 재검토.
4. **Tensor parallel 2-GPU 도입 시점** — v1 S2 single A6000 은 동시 사용자 2~4 수용. 4+ 초과 시점에 `tp=2` 활성화 trigger. 가천대 GPU 가용성 확인 후 결정.
5. **API key 관리 시스템** — v1 single key (env) → v2 multi-key (DB) 전환 시점. KIOM 승인 이후 결정 권장.
6. **adapter 배포 경로** — HF private repo vs S3 pre-signed URL. KIOM 승인 조건에 따라 결정 (재배포 금지 해석이면 S3 pre-signed + IP allowlist 조합).
7. **가천대 IT 팀 서버 준비 기간** — S2 deploy 에 GPU 서버 설정 필요. ver3 M3 W3 일정은 IT 팀 대응 기간에 의존.
8. **Rate limit 기본값 20/min 적정성** — 기관 내부 5명 가정. 실측 후 조정. prometheus `/metrics` 노출 선행 필요.
9. **safety Layer 1 regex vs classifier** — ver2.2 §10.5.6 열린 결정 1. ver3 M3 는 regex 유지, classifier 는 v2. paraphrase held-out refusal < 95% 트리거 시 M4 로 당김.
10. **pyproject.toml license 문구** — 현재 `"TBD — KIOM 승인 후 확정"` (§10.8.1). M4 KIOM 승인 완료 시점에 정확 문구 삽입. 이전까지 PyPI publish 차단.

## 11. 부록 A — pilot adapter 디렉토리 구조 (M2 시점 실측)

```
outputs/cpt_bllossom/
├── adapter/
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   ├── chat_template.jinja           # Llama-3 chat template 보존 확인
│   ├── README.md
│   ├── tokenizer_config.json
│   ├── tokenizer.json
│   └── training_args.bin
├── checkpoint-117/                    # 중간 save
├── checkpoint-156/                    # final (eval_loss 1.887)
├── train.log                          # 389 lines, tqdm + metric
└── train_manifest.json                # run config pin
```

`adapter_config.json` 예상 필드 (`peft` 기본):
- `base_model_name_or_path = "MLP-KTLim/llama-3-Korean-Bllossom-8B"`
- `r = 32`, `lora_alpha = 64`, `lora_dropout = 0.05`
- `target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`
- `peft_type = "LORA"`
- `task_type = "CAUSAL_LM"`

## 12. ver2 cross-reference (필수 읽기)

- `ver2/07_license_ethics/license_ethics.md` **§7.1** — KIOM 라이선스 · §7.1.3 문의 일정
- `ver2/07_license_ethics/license_ethics.md` **§7.6** — 의료 규제 · disclaimer
- `ver2/07_license_ethics/license_ethics.md` **§7.9** — 전문가 계약 NDA
- `ver2/10_demo_cli/deployment.md` **§10.7** — L/S/C 배포 매트릭스 (ver3 §1 에서 확장)
- `ver2/10_demo_cli/inference_backend.md` **§10.4.1** — vLLM skeleton (ver3 §2 에서 실가동)
- `ver2/10_demo_cli/inference_backend.md` **§10.4.4** — 성능 목표
- `ver2/10_demo_cli/prompt_and_safety.md` **§10.5.4** — Safety 2-layer
- `ver2/10_demo_cli/prompt_and_safety.md` **§10.5.5** — T4 평가 프로토콜
- `ver2/10_demo_cli/adapter_paths.md` **§10.3 P-SFT** — `--mode sft` 런타임 스위치
- `ver2/10_demo_cli/packaging.md` **§10.8** — pyproject · Docker 초안 (ver3 §4 에서 multi-stage)
- `ver2/10_demo_cli/packaging.md` **§10.8.5** — CI/CD v1 (ver3 §7 에서 구체화)
- `ver2/10_demo_cli/milestones_and_exit.md` **§10.9** — E1~E7 + 승격 gate
