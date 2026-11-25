# 모빌리티 비전 AI — 백엔드 (Spring Boot 4.0.8)

번호판 인식(LPR) · 출입 관리 · 주차 과금 · 예지보전(PdM)을 담당하는 REST API 서버입니다.
포트 **8080**, API prefix **`/api/v1`**.

## 요구 환경

| 항목 | 버전 |
|---|---|
| JDK | 21 (Zulu OpenJDK 권장) |
| Gradle | wrapper 8.14.3 (`./gradlew` 가 자동 설치) |
| PostgreSQL | 16 |

## 1. 데이터베이스 준비

DB와 계정만 만들어 두면 됩니다. **테이블과 시드 데이터는 기동 시 Flyway 가 자동 생성**합니다.

```bash
createdb -U postgres mobility_ai
psql -U postgres -c "CREATE USER mobility WITH PASSWORD 'mobility123';"
psql -U postgres -c "GRANT ALL PRIVILEGES ON DATABASE mobility_ai TO mobility;"
psql -U postgres -d mobility_ai -c "GRANT ALL ON SCHEMA public TO mobility;"
```

Docker 로 띄우려면:

```bash
docker run -d --name mobility-pg -p 5432:5432 \
  -e POSTGRES_DB=mobility_ai \
  -e POSTGRES_USER=mobility \
  -e POSTGRES_PASSWORD=mobility123 \
  postgres:16
```

## 2. 실행

```bash
cd backend
./gradlew bootRun
```

기본 프로파일은 `dev` 입니다. 운영 프로파일로 띄우려면:

```bash
SPRING_PROFILES_ACTIVE=prod ./gradlew bootRun
```

빌드 산출물로 실행:

```bash
./gradlew clean build          # 테스트 포함
java -jar build/libs/mobility-ai-backend-1.0.0.jar
```

## 3. 동작 확인

```bash
# 헬스 체크 (인증 불필요)
curl http://localhost:8080/actuator/health

# 로그인 → 토큰 발급
TOKEN=$(curl -s -X POST http://localhost:8080/api/v1/auth/login \
  -H 'Content-Type: application/json' \
  -d '{"username":"admin","password":"admin123!"}' \
  | sed -n 's/.*"accessToken":"\([^"]*\)".*/\1/p')

# 대시보드 요약 (인증 필요)
curl -s http://localhost:8080/api/v1/dashboard/summary -H "Authorization: Bearer $TOKEN"

# 차량 목록
curl -s "http://localhost:8080/api/v1/vehicles?page=0&size=5" -H "Authorization: Bearer $TOKEN"

# 번호판 인식 (AI 서비스가 꺼져 있어도 200 을 반환합니다)
curl -s -X POST http://localhost:8080/api/v1/recognitions \
  -H "Authorization: Bearer $TOKEN" \
  -F 'file=@sample.jpg' -F 'gate=정문' -F 'eventType=ENTRY'

# 제공 엔드포인트 목록
curl -s http://localhost:8080/api/v1/meta/endpoints -H "Authorization: Bearer $TOKEN"
```

### 시드 계정

| 아이디 | 비밀번호 | 권한 |
|---|---|---|
| `admin` | `admin123!` | ADMIN |
| `operator` | `operator123!` | OPERATOR |
| `viewer` | `viewer123!` | VIEWER |

> 시드 SQL 의 BCrypt 해시가 깨져 있거나 `{noop}` 이면, 기동 시 `DataInitializer` 가
> 위 비밀번호로 **자동 재인코딩**합니다. 해시를 잘못 붙여 넣어도 로그인이 됩니다.

## 4. 테스트

```bash
./gradlew test
```

| 테스트 | 성격 | 내용 |
|---|---|---|
| `ParkingFeeCalculatorTest` | 순수 단위 | 요금 계산 경계값(무료시간 일치, 기본시간 미만, 일일 상한 초과, 이용권) |
| `PlateFormatValidatorTest` | 순수 단위 | 한국 번호판 신형/구형 형식 판정 |
| `VehicleControllerTest` | 웹 계층 | 경로·HTTP 메서드·응답 JSON 필드명이 계약서와 1:1 인지 |

세 테스트 모두 **DB 없이** 돌아갑니다.

## 5. 환경 변수

모든 값은 `application.yml` 에서 `${ENV:기본값}` 형태로 노출되어 있습니다.

| 변수 | 기본값 | 설명 |
|---|---|---|
| `SERVER_PORT` | `8080` | 서버 포트 |
| `SPRING_PROFILES_ACTIVE` | `dev` | 프로파일 |
| `DB_URL` | `jdbc:postgresql://localhost:5432/mobility_ai` | JDBC URL |
| `DB_USERNAME` / `DB_PASSWORD` | `mobility` / `mobility123` | DB 계정 |
| `JWT_SECRET` | (개발용 문자열) | **32바이트 이상** 필수. 짧으면 기동 실패 |
| `JWT_EXPIRES_SECONDS` | `3600` | 토큰 만료(초) |
| `AI_BASE_URL` | `http://localhost:8000` | AI 추론 서비스 |
| `AI_CONNECT_TIMEOUT_MS` / `AI_READ_TIMEOUT_MS` | `2000` / `15000` | AI 호출 타임아웃 |
| `UPLOAD_DIR` | `./data/uploads` | 업로드 이미지 저장 경로 |
| `PARKING_CAPACITY` | `120` | 주차장 총 수용 대수 |
| `RECOGNITION_MIN_CONFIDENCE` | `0.60` | 통행 허용 최소 신뢰도 |
| `RECOGNITION_DEDUP_SECONDS` | `30` | 중복 이벤트 억제 창(초) |
| `PDM_READINGS_WINDOW` | `50` | 예측에 쓸 최근 센서 건수 |
| `CORS_ALLOWED_ORIGINS` | `http://localhost:5173,...` | 허용 출처 |

## 6. AI 서비스 없이 쓰기

AI 서비스(8000)가 꺼져 있어도 백엔드는 **500 을 던지지 않습니다.**

- `POST /api/v1/recognitions` → **200**, `status=NO_PLATE`, `allowed=false`,
  `rejectReason="AI_SERVICE_UNAVAILABLE"`
- `POST /api/v1/pdm/vehicles/{id}/predict` → **200**, `riskLevel=LOW`,
  `modelVersion="unavailable"` (폴백임을 구분할 수 있게 기록)
- `POST /api/v1/benchmarks/run` → 400 + 사유 메시지 (명시적 실행 요청이므로 사유를 알려 줍니다)

## 7. 구조

```
src/main/java/com/mobility/ai/
├── config/      SecurityConfig, JwtConfig, CorsConfig, AiClientConfig,
│                OpenApiConfig, DataInitializer, AppProperties
├── common/      ApiError, GlobalExceptionHandler, PageResponse, JsonCodec,
│                NotFoundException, BusinessException
├── domain/      JPA 엔티티 10종 + enum 5종
├── repository/  Spring Data JPA 리포지토리 9종 (+ 대시보드 집계 쿼리)
├── dto/         요청/응답 record (계약서 JSON 과 1:1)
├── service/     업무 로직 10종 + ParkingFeeCalculator / PlateFormatValidator (순수 함수)
├── client/      AiInferenceClient (RestClient, 타임아웃/폴백)
└── web/         REST 컨트롤러 10종
```

파일별 상세 설명은 [`MANIFEST.md`](./MANIFEST.md) 를 참고하세요.
