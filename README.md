# 모빌리티 비전 AI 프로젝트

차량번호판 인식(LPR)과 자동차 예지보전(PdM)을 하나로 묶은 통합 시스템이다.
교재 『모빌리티 비전 AI 프로젝트』의 실습 예제 코드와 완성 프로젝트를 함께 담고 있다.

```
┌──────────────┐   /api/v1 (JWT)   ┌───────────────────┐   HTTP   ┌──────────────────┐
│  Vue 3       │ ────────────────► │  Spring Boot 4.0  │ ───────► │  FastAPI         │
│  Vite 6      │ ◄──────────────── │  Java 21 / Gradle │ ◄─────── │  PyTorch·OpenCV  │
│  :5173       │                   │  :8080            │          │  YOLO·CRNN/CTC   │
└──────────────┘                   └─────────┬─────────┘          │  :8000           │
                                             │ JDBC               └──────────────────┘
                                   ┌─────────▼─────────┐
                                   │  PostgreSQL 16    │
                                   │  :5432            │
                                   └───────────────────┘
```

---

## 1. 디렉터리 구성

> **교재 판본 안내**
> 교재는 B5 340쪽 규모의 요약판으로 제공된다. 요약판은 목차의 모든 절과 소절(529개)을
> 그대로 담되 설명을 압축했고, 실습 코드는 각 절에서 가장 중요한 것만 골라 실었다.
> 이 ZIP 에는 **요약판에 실린 코드뿐 아니라 장별 예제 전체(165개 파일)** 가 들어 있다.
> 따라서 교재에 코드가 보이지 않는 주제도 해당 장 디렉터리에서 실행해 볼 수 있다.
> 교재에 실린 코드는 ZIP 안의 같은 경로 파일과 글자 단위로 동일하다.

| 경로 | 설명 |
|---|---|
| `ch01/` ~ `ch06/` | 각 장의 실습 예제 코드. 교재에 실린 코드와 **완전히 동일**하며, 교재가 지면 관계로 싣지 못한 예제도 함께 들어 있다. |
| `mobility-ai-project/ai-service/` | FastAPI 기반 AI 추론 서비스 (번호판 검출·OCR·예지보전) |
| `mobility-ai-project/backend/` | Spring Boot 4.0.8 REST API 서버 (차량·출입·주차·정비·대시보드) |
| `mobility-ai-project/frontend/` | Vue 3 관리자 대시보드 |
| `mobility-ai-project/db/` | PostgreSQL 초기화 스크립트 |
| `mobility-ai-project/docs/screens/` | 관리자 화면 고해상도 캡처 이미지 |
| `mobility-ai-project/scripts/` | 개발용 일괄 기동 스크립트 |
| `mobility-ai-project/docker-compose.yml` | 개발 인프라(PostgreSQL) 및 전체 컨테이너 기동 |

---

## 2. 사전 준비

| 구성 요소 | 버전 | 확인 명령 |
|---|---|---|
| Zulu OpenJDK | 21 | `java -version` |
| Python | 3.12 | `python --version` |
| Node.js | 20 이상 | `node -v` |
| PostgreSQL | 16 | `psql --version` |
| Docker (선택) | 최신 | `docker --version` |

> JDK 는 Azul Zulu OpenJDK 21 을 기준으로 검증했다. `JAVA_HOME` 이 JDK 21 을 가리키는지 확인한다.
> Gradle 은 별도로 설치하지 않아도 된다. 프로젝트에 포함된 Gradle 래퍼(`gradlew`)를 사용한다.

---

## 3. 빠른 시작 (5단계)

### 3.1 환경 변수 준비

```bash
cd mobility-ai-project
cp .env.example .env
```

`.env` 의 `JWT_SECRET` 은 반드시 32바이트(256비트) 이상으로 교체한다.

### 3.2 PostgreSQL 기동

Docker 를 쓰는 경우:

```bash
docker compose up -d postgres
docker compose ps
```

직접 설치한 PostgreSQL 을 쓰는 경우:

```bash
psql -U postgres -c "CREATE USER mobility WITH PASSWORD 'mobility123';"
psql -U postgres -c "CREATE DATABASE mobility_ai OWNER mobility ENCODING 'UTF8';"
psql -U postgres -d mobility_ai -f db/00_init_extension.sql
```

테이블은 백엔드가 기동될 때 Flyway 마이그레이션으로 자동 생성된다. 직접 만들 필요가 없다.

### 3.3 AI 추론 서비스 (:8000)

```bash
cd ai-service
python -m venv .venv
# Windows:  .venv\Scripts\activate
source .venv/bin/activate
pip install --upgrade pip
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

기동 확인:

```bash
curl http://localhost:8000/health
```

```json
{"status":"UP","version":"1.0.0","device":"cpu","detector":"yolo-nano-plate","recognizer":"crnn-ctc-kr"}
```

> **학습된 가중치가 없어도 서비스는 정상 기동된다.** `models/` 에 `plate_yolo.pt`,
> `plate_crnn.pt` 가 없으면 OpenCV 기반 폴백 추론으로 자동 전환되고, 응답의
> `detector`/`recognizer` 값에 `-fallback` 접미사가 붙는다. 정확도는 낮지만 전체
> 파이프라인과 API 연동을 그대로 검증할 수 있다. 가중치를 학습해 넣는 절차는
> 교재 2장·3장을 참고한다.

### 3.4 백엔드 (:8080)

```bash
cd ../backend
# Windows:  gradlew.bat bootRun
./gradlew bootRun
```

기동 확인:

```bash
curl http://localhost:8080/actuator/health
curl -X POST http://localhost:8080/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"admin123!"}'
```

### 3.5 프론트엔드 (:5173)

```bash
cd ../frontend
npm install
npm run dev
```

브라우저에서 `http://localhost:5173` 에 접속한다.

| 계정 | 비밀번호 | 권한 |
|---|---|---|
| `admin` | `admin123!` | ADMIN (전체 관리) |
| `operator` | `operator123!` | OPERATOR (운영·정산) |
| `viewer` | `viewer123!` | VIEWER (조회 전용) |

---

## 4. 동작 확인 시나리오

1. `admin` 으로 로그인하면 대시보드에 차량 수·금일 입출차·인식률·위험 차량 통계가 표시된다.
2. **번호판 인식** 화면에서 차량 이미지를 업로드하고 게이트(`정문`)와 이벤트 유형(`입차`)을 고른다.
3. 인식 결과 패널에 검출 박스, 번호판 문자열, 신뢰도, 단계별 소요시간(검출/OCR/합계)이 표시된다.
4. 등록된 차량이면 출입이 허용되고 주차 세션이 열린다. 같은 차량을 `출차` 로 다시 인식시키면
   주차 시간과 요금이 계산된다.
5. **주차 현황** 화면에서 해당 세션을 정산 처리한다.
6. **예지보전** 화면에서 차량을 선택하고 진단을 실행하면 건강점수·이상점수·고장위험도·잔여수명(RUL)과
   부품별 점수, 정비 권고가 표시된다.
7. **모델 성능 비교** 화면에서 검출기·인식기 조합별 완전일치율과 CPU 지연시간을 비교한다.

---

## 5. 주요 REST API

### AI 추론 서비스 (`http://localhost:8000`)

| Method | Path | 설명 |
|---|---|---|
| GET | `/health` | 상태 및 로드된 모델 확인 |
| POST | `/api/v1/lpr/recognize` | 단일 이미지 번호판 인식 (multipart `file`) |
| POST | `/api/v1/lpr/recognize-frames` | 다중 프레임 결합 인식 |
| POST | `/api/v1/pdm/predict` | 센서 시계열 기반 예지보전 진단 |
| GET | `/api/v1/models/info` | 모델 레지스트리 |
| POST | `/api/v1/models/benchmark` | CPU 벤치마크 실행 |

대화형 문서: `http://localhost:8000/docs`

### 백엔드 (`http://localhost:8080`)

| Method | Path | 설명 |
|---|---|---|
| POST | `/api/v1/auth/login` | 로그인(JWT 발급) |
| GET/POST | `/api/v1/vehicles` | 차량 목록·등록 |
| POST | `/api/v1/recognitions` | 이미지 업로드 → AI 인식 → 이벤트·주차 처리 |
| GET | `/api/v1/recognitions/failures` | 인식 실패 목록 |
| GET | `/api/v1/events` | 출입 이벤트 조회 |
| GET | `/api/v1/parking/sessions` | 주차 세션 |
| POST | `/api/v1/parking/sessions/{id}/pay` | 정산 처리 |
| GET | `/api/v1/parking/status` | 주차 현황·혼잡도 |
| GET/POST | `/api/v1/maintenance` | 정비 이력 |
| GET/POST | `/api/v1/sensors` | 센서 데이터 |
| POST | `/api/v1/pdm/vehicles/{id}/predict` | 예지보전 진단 실행 |
| GET | `/api/v1/dashboard/summary` | 대시보드 집계 |
| GET | `/api/v1/benchmarks` | 모델 성능 목록 |

`/api/v1/auth/**` 와 `/actuator/health` 를 제외한 모든 요청에는
`Authorization: Bearer <accessToken>` 헤더가 필요하다.

---

## 6. 일괄 기동과 스모크 테스트

세 서비스를 따로 띄우는 대신 스크립트로 한 번에 기동할 수 있다.
사전에 3.1~3.5 의 가상환경 생성과 `npm install` 은 한 번 끝내 두어야 한다.

```bash
# Linux / macOS
./scripts/dev_up.sh            # DB → AI → 백엔드 → 프런트엔드 순서로 기동
./scripts/dev_down.sh          # 전체 종료
tail -f logs/backend.log       # 개별 로그 확인
```

```bat
REM Windows
scripts\dev_up.bat
```

기동이 끝나면 연동 상태를 한 번에 점검한다.

```bash
./scripts/smoke_test.sh
```

```
== 헬스체크 ==
  PASS  AI 서비스 /health
  PASS  백엔드 /actuator/health
== 인증 ==
  PASS  로그인 토큰 발급
== 주요 API ==
  PASS  GET /api/v1/dashboard/summary
  ...
통과 14건 / 실패 0건
```

---

## 7. 테스트

```bash
# AI 서비스
cd mobility-ai-project/ai-service && python -m pytest tests -q

# 백엔드
cd mobility-ai-project/backend && ./gradlew test
```

---

## 8. 장별 예제 코드 실행

각 장의 예제는 해당 장 디렉터리를 기준으로 실행한다.

```bash
cd ch01
python 01_environment/check_env.py
```

예제 코드는 무거운 라이브러리(`torch`, `ultralytics`, `scikit-learn` 등)가 설치되어 있지
않아도 안내 메시지를 출력하고 정상 종료하도록 작성했다. 전체 예제를 실행하려면
`mobility-ai-project/ai-service/requirements.txt` 의 패키지를 설치한 가상환경을 사용한다.

---

## 9. 문제 해결

| 증상 | 원인과 해결 |
|---|---|
| 백엔드 기동 중 `Connection refused` | PostgreSQL 이 떠 있는지(`docker compose ps`), `.env` 의 `DB_HOST`/`DB_PORT` 확인 |
| `JWT secret key must be at least 256 bits` | `.env` 의 `JWT_SECRET` 을 32자 이상으로 교체 |
| 인식 결과가 `AI_SERVICE_UNAVAILABLE` | AI 서비스(:8000)가 기동됐는지, `AI_SERVICE_BASE_URL` 이 맞는지 확인 |
| 번호판이 거의 인식되지 않음 | 가중치 미탑재 상태의 폴백 추론이다. 교재 2·3장 절차로 모델을 학습해 `ai-service/models/` 에 배치 |
| 프론트엔드에서 401 이 반복됨 | 토큰 만료(기본 1시간). 다시 로그인한다 |
| `npm install` 시 네트워크 오류 | 사내 프록시 환경이면 `npm config set proxy`/`https-proxy` 설정 |
| Gradle 이 JDK 17 을 잡음 | `JAVA_HOME` 을 JDK 21 로 지정하거나 `./gradlew -Dorg.gradle.java.home=<JDK21 경로> bootRun` |

---

## 10. 라이선스와 주의사항

- 이 코드는 교육 목적의 교재 부속 자료다.
- 차량번호는 개인정보에 해당할 수 있다. 수집·보관·이용 시 개인정보 보호법과 기관 내부 지침을
  따라야 하며, 보관 기간·접근 권한·감사 로그를 반드시 설계에 포함해야 한다(교재 6.13 참고).
- 자동 단속이나 과태료 부과처럼 법적 효력이 발생하는 용도로 쓰려면 별도의 법적 근거와
  공식 검증 절차가 필요하다. 이 시스템은 그 용도로 검증되지 않았다.
