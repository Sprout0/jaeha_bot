#!/usr/bin/env bash
# 재하봇 1 실행 스크립트 (독립 기동 — 가이드 5)
set -euo pipefail
cd "$(dirname "$0")"

MODE="${1:-local}"   # local | docker | dev

case "$MODE" in
  local)   # 보드/PC 네이티브 실행
    python3 -m app.main ;;
  docker)  # Jetson 컨테이너
    docker compose up --build ;;
  dev)     # PC 개발용 컨테이너
    docker build -f Dockerfile.dev -t jaeha_bot:dev .
    docker run --rm -it --device /dev/snd jaeha_bot:dev ;;
  *)
    echo "usage: ./run.sh [local|docker|dev]"; exit 1 ;;
esac
