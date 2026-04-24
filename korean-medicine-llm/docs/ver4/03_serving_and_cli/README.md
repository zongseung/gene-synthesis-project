# ver4 · 03. Serving & CLI Entry (배포 · CLI 실행)

**상태**: draft · 2026-04-21
**단계**: EXP-V4-03 학습 이후 M2 (serving phase)
**맥락**: training adapter 를 실제 사용자가 `hanmed` 한 글자로 쓰게 만드는 전 구간. 내부적으로는
   1. adapter 를 **vLLM + Docker** 로 고성능 서빙
   2. 클라이언트는 `hanmed` 단일 쉘 명령으로 **splash + REPL** 접속
두 축을 병렬 설계한다.

---

## 문서 인덱스

| # | 파일 | 내용 |
|---|---|---|
| 01 | [`01_vllm_docker.md`](01_vllm_docker.md) | vLLM 기반 OpenAI-compatible 서빙, Dockerfile, compose, adapter merge vs dynamic LoRA 판단 |
| 02 | [`02_hanmed_entry.md`](02_hanmed_entry.md) | `pyproject.toml` + Click `invoke_without_command=True` + `.venv/bin/hanmed` 엔트리 설계 |

## 한눈 요약

**What changes**

1. `outputs/cpt_bllossom/adapter` + `data/tokenizer/hanmed_bllossom_ext` → **merged model image** (`hanmed-llm:v0.1`)
2. Docker Compose 로 `vllm_openai` 서비스 기동 → `http://localhost:8000/v1` OpenAI API 호환 엔드포인트
3. `hanmed_cli` 의 `inference/base.py` 에 **RemoteOpenAIBackend** 신규 구현 (기존 `--remote` 자리 차지만 해두었던 것 실제 구현)
4. `pyproject.toml` 신설 + Click group `invoke_without_command=True` → 쉘에서 `hanmed` 만 치면 splash + REPL

**Why now**

- EXP-V4-03 재학습이 끝나면 adapter 품질이 확정됨 → 서빙 인프라·CLI 엔트리 포인트가 **그 시점 바로 필요**
- CLI Visual Identity (§10.11 v3) 는 이미 `branding.py`·`render.py` 에 통합 완료. `hanmed` 쉘 엔트리만 있으면 사용자 경험 완성
- vLLM 으로 옮기면 `transformers` backend 대비 **throughput ~3~8×**, TTFT 1/3 수준 — REPL 체감 속도 급등

**Who touches what**

| 영역 | 담당 파일 | 상태 |
|---|---|---|
| Model packaging | `scripts/build_merged_model.py` | ✅ 작성 완료 (EXP-V4-03 best_model 필요) |
| Serving | `docker/Dockerfile.vllm`, `docker/docker-compose.yml` | ✅ 작성 완료 (Docker daemon 필요 시 기동) |
| CLI backend | `src/hanmed_cli/inference/remote_openai.py` | ✅ 구현 완료 (httpx `/v1/completions` SSE + LlamaStub tokenizer) |
| CLI entry | `pyproject.toml`, `src/hanmed_cli/main.py` | ✅ 구현 완료 (`hanmed` 단일 명령 + `--splash-only`) |
| Inference registry | `src/hanmed_cli/inference/__init__.py` | ✅ `remote_openai` 등록 |

## 스코프 경계

### IN
- Single-node single-GPU 서빙 (A6000 1장)
- OpenAI-compatible API (`/v1/chat/completions`)
- Merged model 방식 (LoRA baked-in)
- Local/단일 호스트 배포 (dev + demo)
- `hanmed` 쉘 명령 → splash → REPL (v3 디자인)

### OUT (별도 라운드)
- Multi-GPU 분산 serving (vLLM tensor parallelism)
- Kubernetes / k8s operator
- Rate limiting · multi-tenant auth
- WebSocket / SSE 스트리밍 프런트엔드
- Serving model quantization (AWQ/GPTQ)
- CI/CD pipeline (GH Actions build-push)

## 선결 조건

- ✅ EXP-V4-03 학습 완료 + `best_model_checkpoint` 산출 (현재 진행 중)
- ✅ CLI Visual Identity v3 통합 (`branding.py`·`render.py`·`turtle_24col.ansi`)
- 🔲 GPU 최소 1장 inference 전용 확보 (학습 점유 해제 후)
- 🔲 Docker daemon 설치 및 nvidia-container-toolkit 동작
- 🔲 `.venv` 에 `pip` 내장 (현재 `/usr/bin/pip --user` 우회 상태 — editable 설치를 위해 pip 필요)

## 구현 진행 (2026-04-21)

| # | Step | 상태 | 산출 |
|---|---|---|---|
| 1 | `pyproject.toml` 작성 | ✅ | `[project.scripts] hanmed = "hanmed_cli.main:cli"` |
| 2 | venv pip 부트스트랩 | ✅ | `.venv/bin/python -m ensurepip` → pip 25.0.1 |
| 3 | editable 설치 | ✅ | `pip install --no-deps -e .` → `.venv/bin/hanmed` 생성 |
| 4 | `main.py` default command | ✅ | `@click.group(invoke_without_command=True)` + `--splash-only` |
| 5 | `RemoteOpenAIBackend` | ✅ | httpx SSE + Llama-3 chat template stub |
| 6 | `scripts/build_merged_model.py` | ✅ | peft `merge_and_unload` + ext tokenizer 보존 |
| 7 | Dockerfile + compose | ✅ | vLLM 0.7.0 base, GPU 0 binding, bf16 / seqs 16 |

## Smoke test 결과

```
$ .venv/bin/hanmed --version
hanmed, version 0.1.0

$ .venv/bin/hanmed --splash-only --plain
DONGUI v0.1.0 — DONGUI AI  [adapter: hanmed-p-a-plus (remote)]

$ .venv/bin/hanmed --splash-only      # TTY 모드
(거북 12-row mascot + header + divider + ┌─ hanmed ─── 렌더)

$ .venv/bin/hanmed                    # vLLM 없을 때
[dongui:error] backend 접속 실패 (http://localhost:8000/v1):
  ConnectError('[Errno 111] Connection refused')
  vLLM 서버 기동 확인: docker compose ps
→ exit 2
```

Backend registry 확인:
- ✅ `transformers`: 정상 등록
- ✅ `remote_openai`: 정상 등록
- ⚠ `vllm` (로컬): v1 reserved 로 import 시 에러 (기존 동작 유지)

## 학습 완료 후 다음 액션

```bash
# 1. best adapter → merged 모델
.venv/bin/python scripts/build_merged_model.py \
  --adapter outputs/cpt_bllossom/best_model \
  --output outputs/hanmed_merged_v0.1

# 2. vLLM 기동
cd docker && docker compose up -d --build
curl -sf http://localhost:8000/health

# 3. hanmed 실행
hanmed
```

## 일정 감각 (예상)

| 단계 | 소요 | 차단 |
|---|---|---|
| 1. Merged model build | 0.5h | EXP-V4-03 완료 |
| 2. Dockerfile + compose draft | 1h | Docker daemon |
| 3. vLLM 기동 smoke | 0.5h | 2 + GPU 확보 |
| 4. `RemoteOpenAIBackend` 구현 | 1h | 3 endpoint alive |
| 5. `pyproject.toml` + editable install | 0.5h | venv pip |
| 6. `hanmed` default command 구현 | 0.5h | 5 |
| 7. end-to-end 스모크 (`hanmed` → vLLM) | 0.5h | 4 + 6 |
| **합계** | **~4.5h** | 순차 의존 많음 |

## 성공 기준

1. 아무 설정 없이 사용자가 터미널에서 `hanmed` 치면 → 거북 약사 splash + `[you]` 프롬프트 + 인터랙션 작동
2. adapter 응답 품질 = EXP-V4-03 확정 지표 (`T1_acc ≥ 70%`, `answer_length_ratio ∈ [0.8, 1.2]`) 유지
3. REPL 입력 → 첫 토큰 시간 (TTFT) ≤ 1.5s
4. 동시 사용자 2명 기준 초당 총 토큰 ≥ 60 tok/s
5. `docker compose up -d` 단일 명령으로 복구 가능한 상태로 재현

## 관련 산출물

- 학습 spec: [`../02_plan_v4.md`](../02_plan_v4.md)
- CLI 디자인: [`../../10_cli_visual_identity/03_claude_code_style.md`](../../10_cli_visual_identity/03_claude_code_style.md)
- 현행 CLI 구조: `src/hanmed_cli/` (main·chat·render·prompts)
- 학습 artifact 예정 경로: `outputs/cpt_bllossom/best_model` (load_best_model_at_end 활성 시 자동 선택)
