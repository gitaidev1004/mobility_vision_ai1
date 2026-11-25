#!/usr/bin/env bash
# 개발 환경 일괄 종료 스크립트
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

for name in frontend backend ai-service; do
  pidfile="logs/$name.pid"
  if [ -f "$pidfile" ]; then
    pid="$(cat "$pidfile")"
    if kill -0 "$pid" 2>/dev/null; then
      # 자식 프로세스까지 함께 정리한다(gradlew, vite 는 자식을 띄운다).
      pkill -P "$pid" 2>/dev/null || true
      kill "$pid" 2>/dev/null || true
      echo "[dev] $name 종료 (pid $pid)"
    fi
    rm -f "$pidfile"
  fi
done

if command -v docker >/dev/null 2>&1; then
  docker compose stop postgres >/dev/null 2>&1 && echo "[dev] PostgreSQL 컨테이너 정지"
fi
echo "[dev] 종료 완료"
