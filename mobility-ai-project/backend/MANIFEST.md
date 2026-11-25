# 백엔드 파일 매니페스트

Spring Boot 4.0.8 / Java 21 / Gradle 8.14.3 · 베이스 패키지 `com.mobility.ai`
총 89개 파일 (Java 83 · 설정 6)

---

## 1. 빌드 · 설정 파일

| 파일 | 설명 |
|---|---|
| `build.gradle` | Spring Boot 4.0.8 플러그인 + BOM(`platform`) 방식 의존성. `io.spring.dependency-management` 미사용 |
| `settings.gradle` | `rootProject.name = 'mobility-ai-backend'` |
| `gradle.properties` | 데몬 JVM 옵션(`-Xmx2g`, UTF-8), 병렬 빌드 |
| `gradle/wrapper/gradle-wrapper.properties` | Gradle 8.14.3 배포 URL |
| `.gitignore` | 빌드 산출물·IDE·업로드 데이터·`.env` 제외 |
| `README.md` | 백엔드 단독 실행법, 환경 변수, 동작 확인 curl |

## 2. 애플리케이션 설정 (`src/main/resources`)

| 파일 | 설명 |
|---|---|
| `application.yml` | 공통 설정. 모든 외부 값을 `${ENV:기본값}` 으로 노출 |
| `application-dev.yml` | 개발: SQL 로그 on |
| `application-prod.yml` | 운영: SQL 로그 off, 컨테이너 호스트명 기본값 |
| `db/migration/V1__init.sql` | 계약서 2장 테이블 10종 + 인덱스 |
| `db/migration/V2__seed.sql` | 계정 3종(BCrypt), 차량 10대, 요금 정책, 센서 240건, 출입/주차/인식/예측/벤치마크 시드. 모든 INSERT 는 재실행 안전 |

## 3. 진입점

| 파일 | 설명 |
|---|---|
| `MobilityAiApplication.java` | `@SpringBootApplication` + `@ConfigurationPropertiesScan` |

## 4. `config/` — 구성

| 파일 | 설명 |
|---|---|
| `AppProperties.java` | `app.*` 타입 세이프 프로퍼티(jwt / ai / storage / parking / cors / recognition / pdm) |
| `SecurityConfig.java` | Security 7 람다 DSL. stateless, csrf disable, `/api/v1/auth/**` · `/actuator/health` permitAll, 나머지 authenticated, `roles`/`scope` → `ROLE_` 매핑 |
| `JwtConfig.java` | `NimbusJwtEncoder` / `NimbusJwtDecoder` HMAC-SHA256. 키 32바이트 미만이면 기동 실패 |
| `CorsConfig.java` | `CorsConfigurationSource` (Vite 5173 허용) |
| `AiClientConfig.java` | AI 전용 `RestClient` 빈. JDK HttpClient 기반 연결/읽기 타임아웃 |
| `OpenApiConfig.java` | 외부 의존성 없는 `GET /api/v1/meta/endpoints` 문서 엔드포인트 |
| `DataInitializer.java` | **비밀번호 해시 안전장치**(BCrypt 아니면 재인코딩), 시드 계정/요금 정책 보강, 업로드 디렉터리 생성 |

## 5. `common/` — 공통

| 파일 | 설명 |
|---|---|
| `ApiError.java` | 계약서 오류 포맷 `{timestamp,status,error,message,path}` record |
| `GlobalExceptionHandler.java` | 전역 예외 → `ApiError`. 404/400/401/403/405/415/500 매핑 |
| `NotFoundException.java` | 조회 실패 → 404 |
| `BusinessException.java` | 업무 규칙 위반 → 400 |
| `PageResponse.java` | 계약서 페이지 포맷 `{content,totalElements,totalPages,page,size}` |
| `JsonCodec.java` | JSONB 전용 얇은 JSON 코덱. **Jackson import 를 이 파일 한 곳에만** 모으고 자동 구성된 `ObjectMapper` 를 주입만 함 |

## 6. `domain/` — 엔티티 10종 + enum 5종

| 파일 | 테이블 | 컬럼 수 |
|---|---|---|
| `AppUser.java` | `app_user` | 7 |
| `Vehicle.java` | `vehicle` | 12 |
| `PlateRecognition.java` | `plate_recognition` | 14 |
| `AccessEvent.java` | `access_event` | 10 |
| `ParkingPolicy.java` | `parking_policy` | 9 |
| `ParkingSession.java` | `parking_session` | 12 |
| `MaintenanceRecord.java` | `maintenance_record` | 9 |
| `SensorReading.java` | `sensor_reading` | 17 |
| `PdmPrediction.java` | `pdm_prediction` | 11 (`detail` 은 jsonb, `@JdbcTypeCode(SqlTypes.JSON)`) |
| `ModelBenchmark.java` | `model_benchmark` | 11 |

enum: `Role`(ADMIN/OPERATOR/VIEWER) · `EventType`(ENTRY/EXIT) ·
`RiskLevel`(LOW/MEDIUM/HIGH/CRITICAL) ·
`RecognitionStatus`(SUCCESS/LOW_CONFIDENCE/NO_PLATE/FORMAT_INVALID) ·
`PassType`(NONE/VISITOR/REGULAR/STAFF)

> Lombok 미사용 — 모든 게터/세터를 명시적으로 작성(교재 가독성).

## 7. `repository/` — 9종

| 파일 | 특기 사항 |
|---|---|
| `AppUserRepository.java` | `findByUsername`, `existsByUsername` |
| `VehicleRepository.java` | `findByPlateNumber`, 번호판/소유자 부분 검색 |
| `PlateRecognitionRepository.java` | 기간·상태 검색, 실패 목록, **인식률/평균 지연/상태별 집계** |
| `AccessEventRepository.java` | 기간·타입 검색, **중복 탐지**(`findRecentDuplicates`), **시간대별 통행량**(네이티브 `FILTER`) |
| `ParkingPolicyRepository.java` | 활성 정책 조회 |
| `ParkingSessionRepository.java` | 주차중/완료 분리 조회, 현재 주차 대수, 미결제 수, 기간 매출 합계 |
| `MaintenanceRecordRepository.java` | `JOIN FETCH` 로 N+1 회피 |
| `SensorReadingRepository.java` | 차량별 최근 N건 |
| `PdmPredictionRepository.java` | **`DISTINCT ON` 으로 차량별 최신 예측** 기준 위험 분포·위험 차량 목록·건수 |

## 8. `dto/` — 계약서 JSON 과 1:1

`AuthDtos` · `VehicleDtos` · `RecognitionDtos` · `EventDtos` · `ParkingDtos` ·
`MaintenanceDtos` · `SensorDtos` · `PdmDtos` · `BenchmarkDtos` · `DashboardDtos`

모두 `record` + 표준 Bean Validation 애너테이션만 사용. Jackson 애너테이션 없음.

## 9. `client/` — AI 연동

| 파일 | 설명 |
|---|---|
| `AiInferenceClient.java` | `RestClient` 로 AI 호출. **예외를 밖으로 던지지 않고** `available=false` 결과 반환. AI 의 스네이크 케이스 JSON 을 `Map` 으로 받아 수동 매핑 |
| `AiRecognitionResult.java` | 인식 응답 값 객체 + `unavailable()` 폴백 |
| `AiPdmResult.java` | 예측 응답 값 객체 + `unavailable(vehicleId)` 폴백 |
| `AiBenchmarkResult.java` | 벤치마크 응답 1건 |

## 10. `service/` — 업무 로직

| 파일 | 설명 |
|---|---|
| `ParkingFeeCalculator.java` | **순수 함수 요금 엔진.** 무료시간 / 기본시간·요금 / 단위시간·요금(올림) / 일일 상한(24h 단위) / 정기권·임직원권 무료 / 방문권 20% 할인 |
| `PlateFormatValidator.java` | **순수 정적** 한국 번호판 형식 검증 + 정규화(공백·하이픈 제거) |
| `AuthService.java` | 로그인 검증 + HS256 JWT 발급(`roles`/`scope`/`displayName` 클레임) |
| `VehicleService.java` | 차량 CRUD. 번호판 정규화 후 저장 |
| `RecognitionService.java` | **핵심 파이프라인**: 파일 저장 → AI 호출 → 인식 저장 → 차량 매칭 → ENTRY/EXIT 분기 → 응답 조립. AI 장애 시 200 + `AI_SERVICE_UNAVAILABLE` |
| `AccessEventService.java` | **중복 이벤트 제거**(동일 번호판+게이트+타입 30초 창), 이벤트 검색, 시간대별 집계(0~23 전부 채움) |
| `ParkingService.java` | 세션 개시/종료, 요금 계산기 호출, 결제, 주차 현황 |
| `MaintenanceService.java` | 정비 이력 조회/등록 |
| `SensorService.java` | 센서 조회/일괄 적재(차량 캐시) |
| `PdmService.java` | 최근 센서 N건 → AI 예측 → JSONB 저장, 최신 조회, 위험 차량, 위험 분포 |
| `DashboardService.java` | 계약서 대시보드 스키마 13개 필드 조립 |
| `BenchmarkService.java` | 벤치마크 목록/실행 후 저장 |

## 11. `web/` — 컨트롤러 10종

| 파일 | 경로 |
|---|---|
| `AuthController.java` | `POST /auth/login`, `GET /auth/me` |
| `VehicleController.java` | `GET/POST /vehicles`, `GET/PUT/DELETE /vehicles/{id}`, `GET /vehicles/by-plate/{plate}` |
| `RecognitionController.java` | `POST /recognitions`(multipart), `GET /recognitions`, `GET /recognitions/failures` |
| `EventController.java` | `GET /events` |
| `ParkingController.java` | `GET /parking/sessions`, `POST /parking/sessions/{id}/pay`, `GET /parking/status` |
| `MaintenanceController.java` | `GET/POST /maintenance` |
| `SensorController.java` | `GET/POST /sensors` |
| `PdmController.java` | `POST /pdm/vehicles/{id}/predict`, `GET /pdm/vehicles/{id}/latest`, `GET /pdm/risky` |
| `DashboardController.java` | `GET /dashboard/summary` |
| `BenchmarkController.java` | `GET /benchmarks`, `POST /benchmarks/run` |

전부 prefix `/api/v1`.

## 12. `src/test/java` — 테스트 3종

| 파일 | 성격 | 검증 내용 |
|---|---|---|
| `ParkingFeeCalculatorTest.java` | 순수 단위 (36 케이스) | 무료시간 **정확히 일치**, 기본시간 미만/일치, 단위 올림(6 파라미터 케이스), **일일 상한 초과**, 다일 상한, 이용권 4종, 방어 로직 |
| `PlateFormatValidatorTest.java` | 순수 단위 | 신형/구형 유효 5건, 무효 9건, null/빈값, 정규화 |
| `VehicleControllerTest.java` | 웹 계층 | 경로·HTTP 메서드·응답 JSON 필드명이 계약서와 1:1, 오류 포맷 400/404, 201/204 상태 코드 |

세 테스트 모두 DB·네트워크 없이 실행된다.

---

# 교재 집필자용 — 6장 각 절 대응 핵심 클래스

| 6장 절(제안) | 인용할 클래스 | 보여줄 포인트 |
|---|---|---|
| **6.1 프로젝트 구성과 Boot 4 전환** | `build.gradle`, `MobilityAiApplication` | BOM(`platform`) 방식, `io.spring.dependency-management` 제거, Jackson 3 기본, `@ConfigurationPropertiesScan` |
| **6.2 데이터 모델링과 마이그레이션** | `V1__init.sql`, `Vehicle`, `SensorReading` | 스네이크 케이스 ↔ `@Column(name=...)` 정확 대응, Flyway 로 스키마 관리 + `ddl-auto: none` |
| **6.3 JSONB 다루기** | `PdmPrediction`, `JsonCodec` | 필드를 `String` + `@JdbcTypeCode(SqlTypes.JSON)` 으로 두어 ORM 의 JSON 매퍼 의존 제거, 커스텀 `ObjectMapper` 빈 없이 Jackson import 를 한 파일로 격리 |
| **6.4 JWT 인증 (Security 7)** | `SecurityConfig`, `JwtConfig`, `AuthService` | 람다 DSL, stateless 리소스 서버, `NimbusJwtEncoder/Decoder` 대칭키, `roles`/`scope` → `ROLE_` 매핑, 키 길이 검증으로 기동 시 실패 |
| **6.5 공통 응답/예외 설계** | `ApiError`, `GlobalExceptionHandler`, `PageResponse` | 오류 포맷 1종으로 통일, `Page` 를 직접 노출하지 않는 이유 |
| **6.6 요금 계산 로직 분리** | **`ParkingFeeCalculator`** + `ParkingFeeCalculatorTest` | 순수 함수로 떼어내 DB·스프링 없이 경계값 테스트. 무료시간 "정확히 일치", 단위 올림, 일일 상한 24h 환산, 할인 적용 순서(상한 → 할인) |
| **6.7 인식 결과 후처리** | **`PlateFormatValidator`** + `PlateFormatValidatorTest` | 정규식 2종으로 신형/구형 수용, 정규화로 공백·하이픈 차이 흡수 |
| **6.8 AI 서비스 연동과 장애 격리** | **`AiInferenceClient`**, `AiClientConfig`, `AiRecognitionResult` | `RestClient` + JDK HttpClient 타임아웃, 예외를 폴백 값 객체로 변환, 스네이크 케이스 수동 매핑으로 Jackson 애너테이션 회피 |
| **6.9 인식 → 출입 → 과금 연계** | **`RecognitionService`** | 파이프라인 7단계, 상태 판정 우선순위, AI 장애 시 **200 + `allowed=false` + `AI_SERVICE_UNAVAILABLE`** (503 아님) |
| **6.10 중복 이벤트 제거** | **`AccessEventService.record`**, `AccessEventRepository.findRecentDuplicates` | 카메라 다중 프레임 문제, 30초 창 재사용으로 세션 중복·이중 과금 방지, `duplicated` 플래그로 하위 처리 차단 |
| **6.11 주차 세션 상태 전이** | `ParkingService` | 개시 시 기존 열린 세션 재사용, 종료 시 분 단위 계산(시계 역전 방어), 0원이면 자동 마감 |
| **6.12 예지보전 연계** | **`PdmService`** | 최근 N건 수집 → 시간 오름차순 정렬 → 스네이크 케이스 변환 → AI 호출 → JSONB 저장, 폴백 식별(`modelVersion="unavailable"`) |
| **6.13 대시보드 집계** | **`DashboardService`**, `PdmPredictionRepository` | 계약서 13필드 조립, "오늘" 경계 정의, `DISTINCT ON` 으로 차량별 최신 예측 집계, 빈 시간대 0 채우기 |
| **6.14 모델 성능 비교** | `BenchmarkService`, `V2__seed.sql` 벤치마크 시드 | 5개 모델 조합의 재현율/정확일치율/p50·p95 지연/메모리/크기/양자화 비교표 |
| **6.15 운영 안전장치** | **`DataInitializer`** | 해시가 BCrypt 가 아니면 재인코딩(시드 오류로 로그인 불가 방지), 시드 계정/정책 보강, 업로드 디렉터리 생성 |
| **6.16 테스트 전략** | 테스트 3종 | 순수 로직은 컨텍스트 없이, 웹 계층은 `standaloneSetup` 으로 계약(경로·필드명) 검증 — 셋 다 DB 불필요 |

## 계약서 준수 체크리스트

- [x] `io.spring.dependency-management` 미사용, BOM `platform` 방식
- [x] `com.fasterxml.jackson.*` import 0건 (`JsonCodec` 만 `tools.jackson` 사용)
- [x] 커스텀 `ObjectMapper` 빈 없음 (자동 구성 빈 주입만)
- [x] Lombok 미사용
- [x] Security 7 람다 DSL만 사용
- [x] Undertow/Jersey 미사용
- [x] DB 테이블 10종 · 컬럼명이 계약서와 완전 일치 (자동 대조 검증)
- [x] 컨트롤러 경로/메서드가 계약서 표와 1:1 (25개 엔드포인트)
- [x] 모든 소스 UTF-8 / LF / 한국어 주석
- [x] 비밀값은 `${ENV:기본값}` 형태로 노출

## 검증 기록 (의존성 다운로드 불가 환경)

Gradle 빌드는 불가능하므로 다음 방법으로 대체 검증했다.

| 검증 | 방법 | 결과 |
|---|---|---|
| 구조 점검 | `scripts/javacheck.py` (패키지↔디렉터리, 파일명↔타입명, 괄호 짝, 타입 해석, import 중복/미사용/금지, 인코딩) | 78개 파일 **문제 없음** (고의로 깨뜨린 파일로 검출력 사전 확인) |
| 컬럼 대조 | 계약서 ↔ `V1__init.sql` ↔ 엔티티 `@Column(name=...)` 자동 대조 | 10개 테이블 **전부 일치** |
| 마이그레이션 실행 | 임시 PostgreSQL 16 클러스터에서 `V1`/`V2` 실제 실행 | **성공**, 2회 반복 실행에도 행 수 불변(멱등) |
| 네이티브 쿼리 실행 | `countHourlyTraffic`, `DISTINCT ON` 위험 분포/위험 차량/건수 쿼리 직접 실행 | **전부 정상 동작** |
| 요금 로직 | 계산 규칙을 Python 으로 포팅해 테스트 기대값 24건 대조 | **불일치 0건** |
| 시드 ↔ 로직 정합성 | `parking_session` 시드 금액을 계산 엔진 결과와 대조 | 방문권 행 1건 불일치 발견 → **수정 완료**(17,000 → 할인 3,400 → 13,600) |
| BCrypt 해시 | libxcrypt 로 `$2a$10$` 해시 생성 후 역검증 | 3개 계정 **검증 통과**(길이 60, 평문 일치) |
| 계약 준수 | 금지 import·Lombok·커스텀 ObjectMapper·비람다 DSL·Undertow/Jersey 전수 grep | **위반 0건** |

### 남은 확인 사항 (네트워크 가능 환경에서 1회 필요)

컴파일과 스프링 컨텍스트 기동은 실제 의존성이 있어야 확인된다. `./gradlew build` 로
한 번 검증하기를 권한다. 특히 다음 지점은 Spring Boot 4 마이너 버전에 따라
클래스 위치가 바뀔 수 있어, 바뀌었다면 import 만 조정하면 된다.

- `JdkClientHttpRequestFactory` (`org.springframework.http.client`)
- `MacAlgorithm` (`org.springframework.security.oauth2.jose.jws`)
- `NimbusJwtEncoder` / `NimbusJwtDecoder` (`org.springframework.security.oauth2.jwt`)
- `@JdbcTypeCode` / `SqlTypes` (`org.hibernate.annotations` / `org.hibernate.type`)

`VehicleControllerTest` 는 이 위험을 피하려고 `@WebMvcTest` 슬라이스 대신
Spring Framework 의 `MockMvcBuilders.standaloneSetup` 을 쓴다(클래스 Javadoc 에
슬라이스 방식으로 바꾸는 코드도 함께 적어 두었다).
