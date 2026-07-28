# ver4 · 03.01 vLLM + Docker 서빙 기획서

**목표**: EXP-V4-03 산출 adapter 를 OpenAI API 호환 엔드포인트로 노출. `docker compose up -d` 한 줄로 재현.

---

## 1. 아키텍처

```
┌───────────────────── 사용자 머신 ──────────────────────┐
│                                                        │
│   $ hanmed                                             │
│   │                                                    │
│   ▼                                                    │
│  ┌────────────────────┐                                │
│  │ hanmed_cli (Click) │──── splash (§10.11 v3)         │
│  │  • RemoteOpenAI    │                                │
│  │    Backend         │                                │
│  └─────────┬──────────┘                                │
└────────────┼───────────────────────────────────────────┘
             │  HTTP /v1/chat/completions (SSE stream)
             ▼
┌──────────── Docker container ──────────────┐
│  vllm/vllm-openai:v0.7.0 (CUDA 12.4)       │
│  ┌──────────────────────────────────────┐  │
│  │ vLLM OpenAI Server                   │  │
│  │  --model /model (merged)             │  │
│  │  --port 8000                         │  │
│  │  --dtype bfloat16                    │  │
│  │  --max-model-len 4096                │  │
│  └──────────────┬───────────────────────┘  │
│                 │ reads                    │
│  ┌──────────────▼──────────────────────┐   │
│  │ /model  = merged Bllossom-8B +      │   │
│  │           P-A+ LoRA + ext tokenizer │   │
│  └─────────────────────────────────────┘   │
└─────────────────────────────────────────────┘
             ▲
             │ volume mount (read-only)
             │
   outputs/hanmed_merged_v0.1/   (호스트 경로)
```

## 2. Adapter 병합 (Merge) 결정

### 2.1 비교

| 항목 | **Merged** (권장) | Dynamic LoRA (`--enable-lora`) |
|---|---|---|
| 런타임 adapter 교체 | ❌ | ✅ |
| 초기 로드 시간 | 빠름 (base 만) | 빠름 (base + adapter 1st load) |
| 추론 latency | **가장 낮음** | +2~5% overhead |
| GPU 메모리 | base 와 동일 | base + adapter shards |
| 디스크 아티팩트 | 완전 통합 1건 | base + adapter 분리 유지 |
| 재학습 시 재빌드 | 필요 | 어댑터만 교체 |

### 2.2 결정 — **Merged 선택**

- HanMed 는 현재 adapter 1종 (P-A+ CPT) 만 운영. SFT/DPO adapter 추가 예정 없음 (ver4 r1 에서 SFT 기각)
- Dynamic LoRA 는 multi-tenant 서빙에 유리하나 본 프로젝트 scope 바깥
- merge 한 번의 공수 vs 매 요청 overhead 감수는 불균형

### 2.3 Merge script 명세 — `scripts/build_merged_model.py`

**입력**:
- `--base` default `MLP-KTLim/llama-3-Korean-Bllossom-8B`
- `--adapter` default `outputs/cpt_bllossom/best_model` (load_best_model_at_end 산출)
- `--tokenizer` default `data/tokenizer/hanmed_bllossom_ext`
- `--output` default `outputs/hanmed_merged_v0.1`

**처리**:
```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype="bfloat16")
# 신규 special token 4 개 반영된 tokenizer 사용
tok = AutoTokenizer.from_pretrained(args.tokenizer)
base.resize_token_embeddings(len(tok))

# modules_to_save=["embed_tokens","lm_head"] 적용 상태에서 로드
merged = PeftModel.from_pretrained(base, args.adapter)
merged = merged.merge_and_unload()

merged.save_pretrained(args.output, safe_serialization=True)
tok.save_pretrained(args.output)
```

**출력 크기**: ~16 GB (8B × bf16). safetensors 분할.

**검증**: `AutoModelForCausalLM.from_pretrained(args.output)` 로 재로드 + 간단 generate 1 회.

## 3. Dockerfile

경로: `docker/Dockerfile.vllm`

```dockerfile
# vLLM 공식 이미지 (CUDA 12.4 + PyTorch 2.5 + vLLM 0.7.0)
FROM vllm/vllm-openai:v0.7.0

# 메타
LABEL maintainer="hanmed-llm"
LABEL version="v0.1"

# 추가 python 의존은 없음 (vLLM 이 transformers 4.47+, tokenizers 등 모두 포함)

# 작업 디렉토리
WORKDIR /app

# 모델은 volume mount. 이미지엔 포함시키지 않음 (크기·재사용성)
# healthcheck: vLLM /health
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD wget -q -O /dev/null http://localhost:8000/health || exit 1

EXPOSE 8000

# entrypoint 는 compose 에서 override
ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server"]
```

- Base 이미지 버전 고정 (`v0.7.0`) — vLLM CLI 인자 API 가 minor 간 호환성 깨질 수 있어 pin.
- 모델 이미지 내장 X — 16 GB 복사는 빌드 시간 크고 이미지 재사용성 떨어짐. Mount 로 해결.

## 4. docker-compose.yml

경로: `docker/docker-compose.yml`

```yaml
services:
  hanmed_vllm:
    build:
      context: .
      dockerfile: Dockerfile.vllm
    image: hanmed-llm:v0.1
    container_name: hanmed_vllm
    runtime: nvidia
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    environment:
      - NVIDIA_VISIBLE_DEVICES=0
      - HF_HUB_OFFLINE=1
      - TRANSFORMERS_OFFLINE=1
    command:
      - --model=/model
      - --served-model-name=hanmed-p-a-plus
      - --tokenizer=/model
      - --trust-remote-code
      - --dtype=bfloat16
      - --port=8000
      - --host=0.0.0.0
      - --max-model-len=4096
      - --max-num-seqs=16
      - --gpu-memory-utilization=0.85
      - --disable-log-requests
    volumes:
      - ${HANMED_MODEL_DIR:-../outputs/hanmed_merged_v0.1}:/model:ro
    ports:
      - "127.0.0.1:8000:8000"
    restart: unless-stopped
    shm_size: 8gb
```

### 4.1 주요 옵션 근거

| 인자 | 값 | 이유 |
|---|---|---|
| `--max-model-len` | 4096 | 한의서 해제 답변 최대 길이 여유. Bllossom-8B base ctx 8k 이지만 낮춰서 KV cache 절약 |
| `--max-num-seqs` | 16 | A6000 48GB 에서 8B bf16 기준 안전 상한 |
| `--gpu-memory-utilization` | 0.85 | 나머지 15% 는 CUDA reserve / 커널 fragmentation 여유 |
| `--dtype` | bfloat16 | CLAUDE.md 규칙 · Ampere 이상 native 지원 |
| `--trust-remote-code` | on | Bllossom tokenizer 커스텀 코드 허용 |
| `--served-model-name` | `hanmed-p-a-plus` | 클라이언트 `model` 파라미터 명시적 |
| `HF_HUB_OFFLINE=1` | — | 컨테이너 기동 중 외부 다운로드 차단 (재현성 + 보안) |

### 4.2 포트 바인딩

- `127.0.0.1:8000:8000` — 외부 노출 방지. SSH 포트포워드나 별도 리버스 프록시 경유 설계.
- 공용 배포 시 nginx / caddy 앞단에 TLS + 인증 필수 (본 라운드 out of scope).

## 5. 기동 플로우

```bash
# 1. 모델 병합 (학습 완료 후 1회)
.venv/bin/python scripts/build_merged_model.py \
  --adapter outputs/cpt_bllossom/best_model \
  --output outputs/hanmed_merged_v0.1

# 2. Docker 이미지 빌드 + 기동
cd docker
docker compose up -d --build

# 3. 헬스 확인
curl -s http://localhost:8000/health
curl -s http://localhost:8000/v1/models | jq

# 4. 대화 smoke (OpenAI SDK 형식)
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "hanmed-p-a-plus",
    "messages": [{"role":"user","content":"동의보감 저자는?"}],
    "max_tokens": 256,
    "temperature": 0
  }' | jq -r '.choices[0].message.content'
```

## 6. 환경별 분기

| 환경 | 모델 경로 | vLLM 파라미터 차이 |
|---|---|---|
| dev (현 머신) | `outputs/hanmed_merged_v0.1` | 위 default |
| demo / 테스트 서버 | `/opt/hanmed/models/v0.1` | `--max-num-seqs=8` 낮춤 (동시성 소규모) |
| production | (미정) | tensor parallel, replica, auth — 별도 라운드 |

## 7. 로그 · 메트릭

### 7.1 로그

```yaml
# compose 에 추가
logging:
  driver: json-file
  options:
    max-size: "100m"
    max-file: "3"
```

vLLM 내부 로그는 stdout 으로 나가 docker logs 로 수집.

### 7.2 메트릭 (향후)

- vLLM 이 `/metrics` Prometheus 엔드포인트 제공 (옵트인 `--disable-log-requests` 와 별개)
- Grafana / Prometheus 통합은 M3 이후

## 8. 실패 모드 · 복구

| 증상 | 원인 후보 | 대응 |
|---|---|---|
| 컨테이너 OOM | `max-model-len` 크거나 `max-num-seqs` 과다 | util 0.85 → 0.80, seqs 16 → 8 |
| `HF_HUB_OFFLINE=1` 로딩 실패 | tokenizer 파일 누락 | `outputs/hanmed_merged_v0.1` 에 tokenizer.* 저장 확인 |
| Cold start 60s+ | base 모델 weight mmap + shard 로드 | 정상. `--start-period=60s` healthcheck 에 반영 |
| 응답 이상 (저자 환각 재발) | adapter merge 실패 (embedding 미반영) | `modules_to_save=["embed_tokens","lm_head"]` 확인 후 재병합 |

## 9. 체크리스트

- [ ] `scripts/build_merged_model.py` 작성 + smoke
- [ ] `docker/Dockerfile.vllm`, `docker/docker-compose.yml` 작성
- [ ] Docker daemon + nvidia-container-toolkit 동작 확인
- [ ] `docker compose build` 성공
- [ ] `docker compose up -d` 후 `/health` green
- [ ] `/v1/chat/completions` 한국어 질의 응답 확인
- [ ] 1차 환각 probe 4문항 재측정 (`T1_acc` baseline 확정용)
- [ ] `docker compose logs` 정상, 에러 0

## 10. 다음 문서

[02_hanmed_entry.md](02_hanmed_entry.md) — `hanmed` 쉘 명령 + `RemoteOpenAIBackend` 구현.
