# ver4 · 03.02 `hanmed` 쉘 엔트리 + Remote Backend 기획서

**목표**: 사용자가 터미널에 `hanmed` 만 치면 §10.11 v3 splash 가 뜨고 바로 REPL 이 시작되도록 패키징.

---

## 1. 현재 상태 (실측, 2026-04-21)

- `.venv/bin/hanmed` **없음**
- `pyproject.toml` · `setup.py` · `setup.cfg` 모두 **없음** (루트에서 확인)
- `@click.group()` 은 서브커맨드 없으면 `--help` 출력 (`invoke_without_command` default False)
- 현재 실행법: `.venv/bin/python -m hanmed_cli chat --adapter ...`
- `.venv` 에 `pip` 미설치 (`.venv/bin/python -m pip` 실패)

## 2. 목표 UX

```bash
$ hanmed
                                           # splash 12-row mascot + header
                                           # divider + tag
[you] 동의보감 저자는?
[dongui] 『東醫寶鑑』은 조선 선조의 명을 받아 허준(許浚)이 …
[you] /exit
```

옵션 (서브커맨드는 유지):
```
hanmed                       # default: chat REPL 시작
hanmed --splash-only         # splash 만 출력 후 exit (backend 로드 X)
hanmed chat --adapter ...    # 기존 호환
hanmed sessions list
```

## 3. 구현 4단계

### 3.1 Step 1 — `pyproject.toml` 생성

경로: `/home/user/gene-synthesis-project/korean-medicine-llm/pyproject.toml`

```toml
[project]
name = "hanmed-llm"
version = "0.1.0"
description = "HanMed CLI — Korean medicine classics assistant"
requires-python = ">=3.10"
dependencies = [
  "click>=8.1",
  "rich>=13.7",
  "prompt_toolkit>=3.0",
  "httpx>=0.27",         # RemoteOpenAIBackend
  "openai>=1.40",        # OpenAI SDK (optional, httpx 만으로 충분하면 drop)
  "transformers>=4.46",  # local backend
  "torch>=2.5",
  "peft>=0.13",          # adapter load (local backend)
]

[project.scripts]
hanmed = "hanmed_cli.main:cli"

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools]
package-dir = {"" = "src"}

[tool.setuptools.packages.find]
where = ["src"]
include = ["hanmed_cli*"]

[tool.setuptools.package-data]
"hanmed_cli.prompts" = ["*.ansi", "*.md"]
```

**핵심 포인트**:
- `[project.scripts] hanmed = "hanmed_cli.main:cli"` → `pip install -e .` 후 `.venv/bin/hanmed` 자동 생성
- `package-data` 로 `turtle_24col.ansi`, `system_v0.1.md` 등 asset 패키지 내부 포함 (distribution 시 빠지지 않게)
- `src/` layout 명시 → `import hanmed_cli` 정상 동작

### 3.2 Step 2 — `venv` 에 pip 부트스트랩

현재 `.venv/bin/python -m pip` 실패. editable 설치하려면 pip 필요. 방법 3가지:

| 방법 | 명령 | 평가 |
|---|---|---|
| A. ensurepip | `.venv/bin/python -m ensurepip --upgrade` | 표준. 단 venv 생성 시 `--without-pip` 로 만들어졌으면 동작 여부 편차 |
| B. get-pip.py | `.venv/bin/python <(curl -s https://bootstrap.pypa.io/get-pip.py)` | 확실. 네트워크 필요 |
| C. uv 로 재생성 | `uv venv --python 3.12 .venv-new && uv pip install -e .` | 기존 dep 재동기화 위험 (학습 중이면 금지) |

**권장**: **A 먼저 시도** → 실패 시 B. C 는 학습 종료 후 재빌드 옵션.

### 3.3 Step 3 — `main.py` 수정 (default command)

```python
@click.group(invoke_without_command=True)
@click.version_option(PKG_VERSION, prog_name="hanmed")
@click.option("--verbose", "-v", is_flag=True, help="verbose 로그")
@click.option("--splash-only", is_flag=True,
              help="splash 만 출력 후 exit (backend 로드 X)")
@click.option("--plain", is_flag=True,
              help="splash 배너·마스코트 생략 (pipe/redirect 용)")
@click.pass_context
def cli(ctx: click.Context, verbose: bool, splash_only: bool, plain: bool) -> None:
    """HanMed-LLM 인터랙티브 CLI (§10 ver4)."""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose

    if ctx.invoked_subcommand is not None:
        return  # sub-command 로 위임

    # bare 'hanmed' 진입 — splash + (optional REPL)
    from hanmed_cli.render import print_banner
    print_banner(PKG_VERSION, "(not loaded)", plain=plain)

    if splash_only:
        return

    # backend 없이 REPL 시작 — default 로 RemoteOpenAIBackend (localhost:8000)
    from hanmed_cli.inference import get_backend
    from hanmed_cli.session import Session
    from hanmed_cli.inference.base import SamplingConfig
    from hanmed_cli.chat import run_repl

    be = get_backend("remote_openai")
    try:
        be.load(
            endpoint=os.environ.get("HANMED_ENDPOINT", "http://localhost:8000/v1"),
            model_name="hanmed-p-a-plus",
        )
    except Exception as exc:
        print_error(f"backend 접속 실패: {exc!r}\n"
                    f"  vLLM 서버 기동 확인: docker compose ps\n"
                    f"  또는: hanmed chat --help 로 local backend 옵션")
        sys.exit(2)

    session = Session()
    sampling = SamplingConfig()
    rc = run_repl(be, session, sampling, adapter_label="hanmed-p-a-plus (vllm)", plain=plain)
    sys.exit(rc)
```

- `invoke_without_command=True` + `if ctx.invoked_subcommand is not None: return` 패턴이 Click 에서 default-command 구현의 정석.
- `--splash-only` 는 backend 없이 디자인 검증 용도. 학습 중이거나 vLLM 미기동이어도 동작.
- env `HANMED_ENDPOINT` 로 원격 서버 주소 override 가능.

### 3.4 Step 4 — `RemoteOpenAIBackend` 구현

경로: `src/hanmed_cli/inference/remote_openai.py`

```python
"""OpenAI-compatible HTTP backend (vLLM, Ollama, 로컬 서버).

기존 inference/base.py 의 Backend 인터페이스 구현:
    - load(endpoint, model_name)
    - get_tokenizer()  — server tokenizer 메타만 캐싱
    - generate(messages, sampling) -> iter[str]  (SSE streaming)
    - close()
"""
from __future__ import annotations

import json
from typing import Iterable

import httpx

from hanmed_cli.inference.base import Backend, SamplingConfig


class RemoteOpenAIBackend(Backend):
    name = "remote_openai"

    def __init__(self) -> None:
        self._endpoint: str | None = None
        self._model: str | None = None
        self._client: httpx.Client | None = None

    def load(self, *, endpoint: str, model_name: str, api_key: str = "none") -> None:
        self._endpoint = endpoint.rstrip("/")
        self._model = model_name
        self._client = httpx.Client(
            base_url=self._endpoint,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=httpx.Timeout(60.0, read=None),
        )
        # health check
        resp = self._client.get("/models")
        resp.raise_for_status()

    def get_tokenizer(self):
        # server-side 에서 토큰화. 클라이언트에선 stub 반환.
        return _StubTokenizer()

    def generate(
        self,
        messages: list[dict],
        sampling: SamplingConfig,
    ) -> Iterable[str]:
        assert self._client and self._model
        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": sampling.temperature,
            "top_p": sampling.top_p,
            "max_tokens": sampling.max_new_tokens,
            "stream": True,
        }
        with self._client.stream("POST", "/chat/completions", json=payload) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                    delta = obj["choices"][0]["delta"].get("content", "")
                    if delta:
                        yield delta
                except (json.JSONDecodeError, KeyError):
                    continue

    def close(self) -> None:
        if self._client:
            self._client.close()
            self._client = None


class _StubTokenizer:
    """vLLM 은 server 에서 토큰화. 클라이언트 prompt 포맷은 그대로 messages 전달."""
    def apply_chat_template(self, messages, **kwargs):
        return messages  # 리스트 그대로 — RemoteOpenAIBackend.generate 가 받음
```

`inference/__init__.py` 에 등록:
```python
_BACKENDS = {
    "transformers": ...,
    "remote_openai": lambda: RemoteOpenAIBackend(),
}
```

### 3.5 Conversation 호환

기존 `conversation.py` 의 `Conversation` 이 `tokenizer.apply_chat_template(...)` 로 문자열을 만들어 backend 에 넘기는 구조라면, `RemoteOpenAIBackend` 는 messages list 를 원본 그대로 받는 쪽을 원함. 세 가지 옵션:

1. **Backend 별 prompt 포맷 분기** — Conversation 이 backend.name 보고 다르게 반환
2. **RemoteOpenAIBackend.generate(prompt_str)** — chat template 적용된 string 을 `{"role":"user","content": str}` 로 한 번 싸서 보냄 (단순하지만 multi-turn 히스토리 X)
3. **messages list 로 일관** — 모든 backend 가 messages list 받도록 인터페이스 변경

권장: **3** — Conversation 이 raw messages list (system + turns) 를 유지하고, local backend 만 tokenizer 로 문자열 변환 후 generate. 이게 OpenAI API 와 자연스럽게 매핑.

이건 `inference/base.py` + `conversation.py` 공동 리팩토링 필요. 공수 ~1h. 본 문서의 "§3 4단계" 사이 Step 3.5 로 끼워 진행.

## 4. 설치 플로우

```bash
# 1. venv 에 pip 부트스트랩
.venv/bin/python -m ensurepip --upgrade

# 2. editable 설치
.venv/bin/pip install -e .

# 3. 확인
.venv/bin/hanmed --version
#  hanmed, version 0.1.0

# 4. splash 만 테스트 (vLLM 없이도 동작)
.venv/bin/hanmed --splash-only

# 5. 완전 REPL (vLLM 기동 후)
.venv/bin/hanmed
#  → splash → [you] prompt
```

### 4.1 PATH 에 추가

`.venv/bin/hanmed` 을 매번 절대경로로 칠 필요 없이:
```bash
# ~/.bashrc 에
export PATH="$HOME/gene-synthesis-project/korean-medicine-llm/.venv/bin:$PATH"
```
또는 venv activate.

## 5. 테스트 플랜

### 5.1 Unit
- `hanmed_cli.inference.remote_openai.RemoteOpenAIBackend` 의 SSE 파싱 단위 테스트 (mock httpx)

### 5.2 Smoke (`scripts/hanmed_smoke.sh`)
```bash
#!/usr/bin/env bash
set -euo pipefail

# 1. splash-only 는 항상 통과
.venv/bin/hanmed --splash-only

# 2. vLLM 헬스
curl -sf http://localhost:8000/health

# 3. CLI → vLLM round-trip
echo "동의보감 저자는?" | .venv/bin/hanmed --plain | head -20

echo "smoke OK"
```

### 5.3 Manual
- `hanmed` 쳤을 때 거북 + 헤더 + 구분선 뜨는지 (육안)
- `/exit` 정상 종료
- `Ctrl+C` 깨끗한 종료
- SSH 세션이라 컬러 깨지지 않는지 (`--plain` 테스트)

## 6. 위험 / 엣지케이스

| 이슈 | 대응 |
|---|---|
| venv pip 부트스트랩 실패 (ensurepip 미지원) | get-pip.py 또는 `--user` 설치 후 `sys.path` 주입 |
| `pip install -e .` 가 `.venv` 내 rich / click 중복 설치 | 기존 버전이 requires 만족 시 skip. conflict 시 수동 해결 |
| vLLM 서버 다운 → `hanmed` 진입 실패 | try/except 로 `--splash-only` 전환 안내 메시지 출력 |
| 터미널 폭 < 60 col | mascot 24-col 잘림. 향후 `COLUMNS` 감지해 `turtle_12col.ansi` fallback |
| `/sessions list` 같은 서브커맨드 | `invoke_without_command=True` + `if ctx.invoked_subcommand` 로 위임 — 검증 필수 |

## 7. 마일스톤

| 스텝 | 소요 | 차단 조건 |
|---|---|---|
| S1 · pyproject.toml 작성 | 10 min | — |
| S2 · venv pip 부트스트랩 | 5~15 min | 학습 방해 X |
| S3 · `pip install -e .` smoke | 5 min | S1+S2 |
| S4 · `main.py` default command 구현 | 30 min | S3 |
| S5 · `--splash-only` 동작 확인 | 5 min | S4 |
| S6 · `RemoteOpenAIBackend` 구현 | 45 min | S3 (vLLM 의존 아님, 코드만) |
| S7 · Conversation/Backend 인터페이스 messages-list 정비 | 1 h | S6 |
| S8 · vLLM 기동 후 round-trip | 15 min | 01_vllm_docker.md 완료 |
| **합계** | **~3 h** | 학습 끝난 뒤 착수 권장 (GPU 점유) |

S1~S7 은 **학습 중 GPU 안 써도 가능** — 병렬 진행.

## 8. 완료 후 README 반영

- ver4 README 의 "구현 진행 상태" 표에 신규 항목:
  - `pyproject.toml` ✅
  - `src/hanmed_cli/inference/remote_openai.py` ✅
  - `docker/Dockerfile.vllm` + `docker-compose.yml` ✅
  - `scripts/build_merged_model.py` ✅
  - `scripts/hanmed_smoke.sh` ✅

## 9. 관련 문서

- [01_vllm_docker.md](01_vllm_docker.md) — 서빙 인프라
- CLI 디자인 spec: [`../../10_cli_visual_identity/03_claude_code_style.md`](../../10_cli_visual_identity/03_claude_code_style.md)
- 현행 CLI: `src/hanmed_cli/main.py`, `src/hanmed_cli/render.py`, `src/hanmed_cli/prompts/branding.py`
