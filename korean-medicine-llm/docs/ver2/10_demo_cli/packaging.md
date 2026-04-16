# 10.8 Packaging and Distribution

## 10.8.1 pyproject.toml

```toml
[project]
name = "hanmed-cli"
version = "0.1.0"
description = "한의학 고전 LLM 인터랙티브 CLI (Bllossom-8B + HanMed-CPT)"
requires-python = ">=3.10"
license = { text = "TBD — KIOM 승인 후 확정" }
authors = [{ name = "zongseung", email = "new9279@gachon.ac.kr" }]
dependencies = [
  "click>=8.1",
  "prompt_toolkit>=3.0",
  "rich>=13.0",
  "transformers>=4.44",
  "peft>=0.13",
  "torch>=2.1",
  "pydantic>=2.0",
]

[project.optional-dependencies]
vllm = ["vllm>=0.6.0"]              # primary backend
gguf = ["llama-cpp-python>=0.3"]    # v1 옵션

[project.scripts]
hanmed = "hanmed_cli.main:cli"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

## 10.8.2 설치 경로

### 개발자 (로컬)
```bash
cd korean-medicine-llm
uv sync --all-extras
uv run hanmed chat --adapter outputs/cpt_bllossom/adapter
```

### 배포 (v1, KIOM 승인 후)
```bash
# PyPI
pip install hanmed-cli[vllm]

# adapter 는 HF Hub 에서 자동 다운로드
hanmed chat
```

### Docker (v1 옵션)
```dockerfile
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04
RUN pip install hanmed-cli[vllm]
ENTRYPOINT ["hanmed", "chat"]
```

## 10.8.3 Adapter 배포

### v0 (로컬 전용)
- adapter 파일 = `outputs/cpt_bllossom/adapter/` (git-ignored)
- CLI 는 `--adapter` 옵션으로 로컬 경로 지정

### v1 (공개, KIOM 승인 후)
- HuggingFace Hub: `hanmed-llm/HanMed-CPT-v0.1`
- model card 필수 포함:
  - 학습 데이터 출처 (KIOM mediclassics.kr)
  - 학습 레시피 (Bllossom-8B + LoRA r=32, cap tokens, epochs)
  - 한계 (임상 사용 금지)
  - 평가 결과 (§05 T1~T5)
  - 라이선스 (KIOM 승인 문구)
 - 첫 실행 시 `$XDG_DATA_HOME/hanmed/adapters/` (fallback `~/.local/share/hanmed/adapters/`) 로 자동 다운로드 (~500 MB)

## 10.8.4 버전 관리

CLI 와 adapter 는 버전을 분리:

- **CLI package**: semantic versioning `MAJOR.MINOR.PATCH`
- **Adapter artifact**: `HanMed-CPT-v0.1`, `HanMed-SFT-v0.1` 식의 별도 artifact version
- CLI breaking change = MAJOR
- CLI 기능 추가 = MINOR
- 버그 수정 = PATCH
- adapter 교체는 CLI major bump 의 필요조건이 아니다. 단, 세션 스키마/기본 prompt/응답 계약을 깨면 CLI major 검토

각 release 에 adapter sha256 + base model commit hash 고정:

```toml
# pyproject.toml 의 tool.hanmed 섹션
[tool.hanmed.release]
adapter_sha256 = "..."
base_model = "MLP-KTLim/llama-3-Korean-Bllossom-8B"
base_revision = "3c9b6f7..."  # immutable HF snapshot revision, never "main"
```

## 10.8.5 CI/CD (v1 이후)

GitHub Actions:
- `pytest tests/hanmed_cli/` on PR
- Coverage ≥ 80%
- Release tag → PyPI upload (secrets: PYPI_TOKEN)
- HF Hub adapter upload (secrets: HF_TOKEN) — KIOM 승인 검증 후에만

## 10.8.6 열린 결정

1. **uv vs pip**: 개발은 uv, 배포는 pip 표준. pyproject.toml 은 둘 다 호환
2. **Private PyPI**: KIOM 승인 전 공개 PyPI 배포 금지. GitHub Packages 또는 self-hosted?
3. **adapter 파일 포맷**: safetensors 권장 (보안, 속도). pickle 금지
4. **model card 언어**: 한국어 + 영문 bilingual
