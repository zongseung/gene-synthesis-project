#!/usr/bin/env bash
# Phase A' (동의보감) 전용 CLI 기동.
#
# 전제: scripts/deploy_phaseA.sh direct 또는 merged 로 vLLM 컨테이너 기동됨.
#      (http://localhost:8000 health check 통과)
#
# 모드:
#   direct   -- LoRA direct mode vLLM (adapter 이름 "dongui-bogam" 사용)
#   merged   -- merged model vLLM (모델 이름 "hanmed-phaseA" 사용)
#   custom   -- HANMED_MODEL 환경변수 그대로 사용
#
# 사용:
#   scripts/cli_phaseA.sh direct           # LoRA adapter CLI
#   scripts/cli_phaseA.sh merged           # merged model CLI
#   scripts/cli_phaseA.sh direct --splash-only    # splash 만
#   scripts/cli_phaseA.sh direct chat -m dongui-bogam    # subcommand 전달
#
# vLLM 없이 local transformers 로 쓰려면:
#   .venv/bin/python -m hanmed_cli.main chat --backend transformers --adapter outputs/cpt_bllossom_phaseA/adapter

set -euo pipefail

cd "$(dirname "$0")/../.."

mode="${1:-direct}"; shift || true

case "${mode}" in
  direct)
    export HANMED_ENDPOINT="${HANMED_ENDPOINT:-http://localhost:8000/v1}"
    export HANMED_MODEL="${HANMED_MODEL:-dongui-bogam}"
    ;;
  merged)
    export HANMED_ENDPOINT="${HANMED_ENDPOINT:-http://localhost:8000/v1}"
    export HANMED_MODEL="${HANMED_MODEL:-hanmed-phaseA}"
    ;;
  custom)
    : "${HANMED_ENDPOINT:?HANMED_ENDPOINT 환경변수 필요}"
    : "${HANMED_MODEL:?HANMED_MODEL 환경변수 필요}"
    ;;
  -h|--help|help)
    sed -n '2,18p' "$0"
    exit 0
    ;;
  *)
    echo "알 수 없는 모드: ${mode}"
    echo "사용법: $0 {direct|merged|custom} [CLI 옵션...]"
    exit 1
    ;;
esac

# 서버 health check (2초)
if ! curl -sf "${HANMED_ENDPOINT%/v1}/health" > /dev/null 2>&1; then
  echo "⚠ vLLM 서버 응답 없음: ${HANMED_ENDPOINT}"
  echo "  먼저 실행: scripts/deploy_phaseA.sh ${mode}"
  echo "  또는 --plain 으로 splash 만 보려면 계속 진행 ..."
fi

# 모델 존재 확인
echo "→ endpoint: ${HANMED_ENDPOINT}"
echo "→ model:    ${HANMED_MODEL}"
echo ""

# 인자 없거나 subcommand 가 없으면 기본으로 `chat --backend remote_openai` 실행.
# vLLM merged/direct 둘 다 원격 HTTP API 라 remote_openai 로 호출.
# --base-model 을 served-model-name (HANMED_MODEL) 으로 명시 — remote_openai
# backend 는 이 값을 모델 식별자로 쓰므로 vLLM 등록명과 일치해야 404 회피.
if [ $# -eq 0 ] || [ "${1:0:1}" = "-" ]; then
  exec .venv/bin/python -m hanmed_cli.main chat \
    --backend remote_openai \
    --base-model "${HANMED_MODEL}" \
    "$@"
else
  exec .venv/bin/python -m hanmed_cli.main "$@"
fi
