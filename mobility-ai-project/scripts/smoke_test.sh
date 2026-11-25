#!/usr/bin/env bash
# 기동 후 전체 연동을 점검하는 스모크 테스트.
# 로그인 → 대시보드 → 차량 목록 → 예지보전 진단 순으로 호출해 본다.
set -euo pipefail

BE="${BACKEND_URL:-http://localhost:8080}"
AI="${AI_URL:-http://localhost:8000}"
pass=0; fail=0

check() {  # check <설명> <명령...>
  local desc="$1"; shift
  if "$@" >/dev/null 2>&1; then
    printf '  \033[1;32mPASS\033[0m  %s\n' "$desc"; pass=$((pass + 1))
  else
    printf '  \033[1;31mFAIL\033[0m  %s\n' "$desc"; fail=$((fail + 1))
  fi
}

echo "== 헬스체크 =="
check "AI 서비스 /health"            curl -fsS "$AI/health"
check "백엔드 /actuator/health"      curl -fsS "$BE/actuator/health"

echo "== 인증 =="
TOKEN="$(curl -fsS -X POST "$BE/api/v1/auth/login" \
  -H 'Content-Type: application/json' \
  -d '{"username":"admin","password":"admin123!"}' \
  | sed -n 's/.*"accessToken":"\([^"]*\)".*/\1/p')"

if [ -n "${TOKEN:-}" ]; then
  printf '  \033[1;32mPASS\033[0m  로그인 토큰 발급\n'; pass=$((pass + 1))
else
  printf '  \033[1;31mFAIL\033[0m  로그인 토큰 발급\n'; fail=$((fail + 1))
  echo "로그인에 실패해 이후 검사를 건너뛴다."; exit 1
fi

AUTH=(-H "Authorization: Bearer $TOKEN")

echo "== 주요 API =="
check "GET /api/v1/auth/me"            curl -fsS "${AUTH[@]}" "$BE/api/v1/auth/me"
check "GET /api/v1/dashboard/summary"  curl -fsS "${AUTH[@]}" "$BE/api/v1/dashboard/summary"
check "GET /api/v1/vehicles"           curl -fsS "${AUTH[@]}" "$BE/api/v1/vehicles?page=0&size=5"
check "GET /api/v1/events"             curl -fsS "${AUTH[@]}" "$BE/api/v1/events?page=0&size=5"
check "GET /api/v1/parking/status"     curl -fsS "${AUTH[@]}" "$BE/api/v1/parking/status"
check "GET /api/v1/parking/sessions"   curl -fsS "${AUTH[@]}" "$BE/api/v1/parking/sessions?page=0&size=5"
check "GET /api/v1/maintenance"        curl -fsS "${AUTH[@]}" "$BE/api/v1/maintenance?page=0&size=5"
check "GET /api/v1/sensors"            curl -fsS "${AUTH[@]}" "$BE/api/v1/sensors?vehicleId=1&limit=10"
check "GET /api/v1/benchmarks"         curl -fsS "${AUTH[@]}" "$BE/api/v1/benchmarks"
check "POST /api/v1/pdm/.../predict"   curl -fsS -X POST "${AUTH[@]}" "$BE/api/v1/pdm/vehicles/1/predict"
check "GET /api/v1/pdm/vehicles/1/latest" curl -fsS "${AUTH[@]}" "$BE/api/v1/pdm/vehicles/1/latest"

echo "== AI 서비스 직접 호출 =="
check "GET /api/v1/models/info"        curl -fsS "$AI/api/v1/models/info"

echo
echo "통과 $pass건 / 실패 $fail건"
[ "$fail" -eq 0 ]
