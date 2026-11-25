@echo off
REM 개발 환경 일괄 기동 스크립트 (Windows)
REM 각 서비스를 별도 명령 프롬프트 창으로 띄운다.
setlocal
set ROOT=%~dp0..
cd /d "%ROOT%"

echo [dev] PostgreSQL 컨테이너를 기동한다
docker compose up -d postgres
if errorlevel 1 (
  echo [dev] docker 기동에 실패했다. 로컬 PostgreSQL 이 떠 있는지 확인한다.
)

echo [dev] AI 추론 서비스를 기동한다 ^(:8000^)
start "mobility-ai-service" cmd /k "cd /d %ROOT%\ai-service && .venv\Scripts\activate && uvicorn app.main:app --host 0.0.0.0 --port 8000"

echo [dev] 백엔드를 기동한다 ^(:8080^)
start "mobility-backend" cmd /k "cd /d %ROOT%\backend && gradlew.bat bootRun"

echo [dev] 프런트엔드를 기동한다 ^(:5173^)
start "mobility-frontend" cmd /k "cd /d %ROOT%\frontend && npm run dev"

echo.
echo [dev] 세 개의 창이 열렸다. 각 창의 로그로 기동 상태를 확인한다.
echo [dev]   프런트엔드  http://localhost:5173   ^(admin / admin123!^)
echo [dev]   백엔드      http://localhost:8080/actuator/health
echo [dev]   AI 서비스   http://localhost:8000/docs
endlocal
