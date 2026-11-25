#!/usr/bin/env bash
# 개발 환경 일괄 기동 스크립트 (Linux / macOS)
#
#   ./scripts/dev_up.sh            # 전부 기동
#   ./scripts/dev_up.sh db         # DB 만 기동
#
# 각 서비스는 백그라운드로 띄우고 로그는 logs/ 아래에 남긴다.
# 종료는 ./scripts/dev_down.sh 를 쓴다.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p logs

log()  { printf '\033[1;34m[dev]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[dev]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[dev]\033[0m %s\n' "$*" >&2; exit 1; }

wait_http() {   # wait_http <url> <이름> <최대초>
  local url="$1" name="$2" limit="${3:-90}" i=0
  until curl -fsS "$url" >/dev/null 2>&1; do
    i=$((i + 1))
    [ "$i" -ge "$limit" ] && die "$name 이(가) ${limit}초 안에 기동되지 않았다. logs/ 를 확인한다."
    sleep 1
  done
  log "$name 기동 확인 ($url)"
}

# ---------------------------------------------------------------- PostgreSQL
start_db() {
  if command -v docker >/dev/null 2>&1; then
    log "PostgreSQL 컨테이너를 기동한다"
    docker compose up -d postgres
    local i=0
    until docker compose exec -T postgres pg_isready -U "${POSTGRES_USER:-mobility}" >/dev/null 2>&1; do
      i=$((i + 1)); [ "$i" -ge 60 ] && die "PostgreSQL 이 준비되지 않았다"; sleep 1
    done
    log "PostgreSQL 준비 완료"
  else
    warn "docker 를 찾을 수 없다. 로컬에 설치된 PostgreSQL 이 떠 있다고 가정한다."
  fi
}

# ---------------------------------------------------------------- AI 서비스
start_ai() {
  cd "$ROOT/ai-service"
  [ -d .venv ] || die "ai-service/.venv 가 없다. README 3.3 절차로 가상환경을 먼저 만든다."
  log "AI 추론 서비스를 기동한다 (:8000)"
  # shellcheck disable=SC1091
  source .venv/bin/activate
  nohup uvicorn app.main:app --host 0.0.0.0 --port 8000 \
      > "$ROOT/logs/ai-service.log" 2>&1 &
  echo $! > "$ROOT/logs/ai-service.pid"
  deactivate || true
  cd "$ROOT"
  wait_http "http://localhost:8000/health" "AI 서비스" 90
}

# ---------------------------------------------------------------- 백엔드
start_backend() {
  cd "$ROOT/backend"
  log "백엔드를 기동한다 (:8080) — 최초 실행은 의존성 다운로드로 몇 분 걸릴 수 있다"
  nohup ./gradlew bootRun > "$ROOT/logs/backend.log" 2>&1 &
  echo $! > "$ROOT/logs/backend.pid"
  cd "$ROOT"
  wait_http "http://localhost:8080/actuator/health" "백엔드" 300
}

# ---------------------------------------------------------------- 프런트엔드
start_frontend() {
  cd "$ROOT/frontend"
  [ -d node_modules ] || { log "npm install 을 먼저 실행한다"; npm install; }
  log "프런트엔드 개발 서버를 기동한다 (:5173)"
  nohup npm run dev > "$ROOT/logs/frontend.log" 2>&1 &
  echo $! > "$ROOT/logs/frontend.pid"
  cd "$ROOT"
  wait_http "http://localhost:5173" "프런트엔드" 120
}

case "${1:-all}" in
  db)       start_db ;;
  ai)       start_ai ;;
  backend)  start_backend ;;
  frontend) start_frontend ;;
  all)
    start_db
    start_ai
    start_backend
    start_frontend
    echo
    log "전체 기동 완료"
    log "  프런트엔드  http://localhost:5173   (admin / admin123!)"
    log "  백엔드      http://localhost:8080/actuator/health"
    log "  AI 서비스   http://localhost:8000/docs"
    ;;
  *) die "사용법: $0 [all|db|ai|backend|frontend]" ;;
esac
