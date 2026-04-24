#!/usr/bin/env bash
# Phase A' (동의보감) 배포 자동화.
# 모드: direct (LoRA) | merged (adapter merge 후)
#
# 사용:
#   scripts/deploy_phaseA.sh direct                     # LoRA direct 배포
#   scripts/deploy_phaseA.sh merged                     # adapter merge + merged 배포
#   scripts/deploy_phaseA.sh merged --skip-build        # merged 파일 있으면 merge 생략
#   scripts/deploy_phaseA.sh down                       # 모든 Phase A' 컨테이너 중지

set -euo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"

ADAPTER_DIR="${REPO_ROOT}/outputs/cpt_bllossom_phaseA/adapter"
MERGED_DIR="${REPO_ROOT}/outputs/hanmed_merged_phaseA"
COMPOSE_DIRECT="docker/docker-compose.phaseA.yml"
COMPOSE_MERGED="docker/docker-compose.phaseA.merged.yml"

mode="${1:-}"; shift || true

case "${mode}" in
  direct)
    # LoRA adapter 존재 확인
    if [ ! -d "${ADAPTER_DIR}" ]; then
      echo "✗ adapter 디렉토리 없음: ${ADAPTER_DIR}"
      echo "  학습이 완료됐는지 확인: ls outputs/cpt_bllossom_phaseA/"
      exit 1
    fi
    if [ ! -f "${ADAPTER_DIR}/adapter_config.json" ]; then
      echo "✗ adapter_config.json 없음. 학습 중이거나 실패했을 수 있음."
      exit 1
    fi
    echo "✓ adapter 확인: ${ADAPTER_DIR}"
    echo ""
    echo "[1/2] Docker compose up (LoRA direct mode)…"
    cd docker
    docker compose -f docker-compose.phaseA.yml up -d --build
    cd ..
    echo ""
    echo "[2/2] Health check (최대 120초 대기)…"
    for i in {1..24}; do
      sleep 5
      if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
        echo "✓ vLLM 서버 기동 완료 (${i}번째 시도)"
        break
      fi
      echo "  waiting… (${i}/24)"
    done
    echo ""
    curl -s http://localhost:8000/v1/models | python3 -m json.tool || echo "서버 응답 없음"
    ;;

  merged)
    skip_build="false"
    if [ "${1:-}" = "--skip-build" ]; then
      skip_build="true"
    fi

    if [ ! -d "${ADAPTER_DIR}" ]; then
      echo "✗ adapter 디렉토리 없음: ${ADAPTER_DIR}"
      exit 1
    fi

    if [ "${skip_build}" = "false" ] || [ ! -f "${MERGED_DIR}/model.safetensors.index.json" ]; then
      echo "[1/3] adapter + base → merged 모델 빌드…"
      PYTHONHASHSEED=0 .venv/bin/python scripts/build_merged_model.py \
        --adapter "${ADAPTER_DIR}" \
        --output "${MERGED_DIR}"
    else
      echo "[1/3] merged 모델 이미 존재 (--skip-build): ${MERGED_DIR}"
    fi

    echo ""
    echo "[2/3] Docker compose up (merged mode)…"
    cd docker
    docker compose -f docker-compose.phaseA.merged.yml up -d --build
    cd ..

    echo ""
    echo "[3/3] Health check (최대 120초 대기)…"
    for i in {1..24}; do
      sleep 5
      if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
        echo "✓ vLLM 서버 기동 완료"
        break
      fi
      echo "  waiting… (${i}/24)"
    done
    echo ""
    curl -s http://localhost:8000/v1/models | python3 -m json.tool || echo "서버 응답 없음"
    ;;

  down)
    echo "Phase A' 컨테이너 중지…"
    cd docker
    docker compose -f docker-compose.phaseA.yml down 2>/dev/null || true
    docker compose -f docker-compose.phaseA.merged.yml down 2>/dev/null || true
    echo "✓ 중지 완료"
    ;;

  smoke)
    # 간단한 동의보감 질문 테스트
    echo "[smoke test] 동의보감 편찬자 질문…"
    curl -s http://localhost:8000/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{
        "model": "dongui-bogam",
        "messages": [
          {"role": "system", "content": "당신은 한국 전통 한의학 고전 전문가입니다."},
          {"role": "user", "content": "동의보감(東醫寶鑑)은 누가 편찬했으며 언제 완성되었나요? 한 단락으로 답하세요."}
        ],
        "max_tokens": 300,
        "temperature": 0
      }' | python3 -m json.tool
    ;;

  *)
    echo "사용법:"
    echo "  $0 direct                     # LoRA direct mode 배포"
    echo "  $0 merged                     # adapter merge + merged 배포"
    echo "  $0 merged --skip-build        # merge 생략 (기존 파일 사용)"
    echo "  $0 down                       # 모든 Phase A' 컨테이너 중지"
    echo "  $0 smoke                      # 동의보감 질문 smoke test (서버 기동 후)"
    exit 1
    ;;
esac
