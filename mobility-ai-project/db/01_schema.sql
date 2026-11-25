-- =====================================================================
-- V1__init.sql : 모빌리티 비전 AI 프로젝트 초기 스키마
-- 계약서 2장(데이터베이스)의 테이블 정의를 그대로 구현한다.
-- 모든 테이블/컬럼은 스네이크 케이스이며, 타임스탬프는 timezone 없는
-- TIMESTAMP(= Java LocalDateTime)로 통일한다.
-- =====================================================================

-- ---------------------------------------------------------------------
-- 1) app_user : 로그인 계정. role 은 ADMIN|OPERATOR|VIEWER.
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS app_user (
    id            BIGSERIAL    PRIMARY KEY,
    username      VARCHAR(50)  NOT NULL UNIQUE,
    password_hash VARCHAR(200) NOT NULL,
    display_name  VARCHAR(100) NOT NULL,
    role          VARCHAR(20)  NOT NULL,
    enabled       BOOLEAN      NOT NULL DEFAULT TRUE,
    created_at    TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- ---------------------------------------------------------------------
-- 2) vehicle : 등록 차량 마스터. plate_number 가 업무 키(UNIQUE).
--    pass_type 은 NONE|VISITOR|REGULAR|STAFF (정기권/방문권 요금 정책에 사용)
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS vehicle (
    id              BIGSERIAL   PRIMARY KEY,
    plate_number    VARCHAR(20) NOT NULL UNIQUE,
    owner_name      VARCHAR(50) NOT NULL,
    phone           VARCHAR(30),
    vehicle_type    VARCHAR(30),
    model_name      VARCHAR(80),
    model_year      INTEGER,
    vin             VARCHAR(40),
    pass_type       VARCHAR(20) NOT NULL DEFAULT 'NONE',
    pass_expires_at TIMESTAMP,
    active          BOOLEAN     NOT NULL DEFAULT TRUE,
    registered_at   TIMESTAMP   NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_vehicle_owner_name ON vehicle (owner_name);

-- ---------------------------------------------------------------------
-- 3) plate_recognition : AI 번호판 인식 1건의 원본 기록(추적/재현용).
--    status 는 SUCCESS|LOW_CONFIDENCE|NO_PLATE|FORMAT_INVALID
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS plate_recognition (
    id                  BIGSERIAL    PRIMARY KEY,
    plate_number        VARCHAR(20),
    confidence          DOUBLE PRECISION,
    detector_name       VARCHAR(50),
    ocr_name            VARCHAR(50),
    image_path          VARCHAR(300),
    detect_ms           DOUBLE PRECISION,
    ocr_ms              DOUBLE PRECISION,
    total_ms            DOUBLE PRECISION,
    quality_score       DOUBLE PRECISION,
    valid_format        BOOLEAN      NOT NULL DEFAULT FALSE,
    matched_vehicle_id  BIGINT       REFERENCES vehicle (id) ON DELETE SET NULL,
    status              VARCHAR(20)  NOT NULL,
    created_at          TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_recognition_created_at ON plate_recognition (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_recognition_status     ON plate_recognition (status);

-- ---------------------------------------------------------------------
-- 4) access_event : 차단기 출입 이벤트. event_type 은 ENTRY|EXIT.
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS access_event (
    id             BIGSERIAL   PRIMARY KEY,
    vehicle_id     BIGINT      REFERENCES vehicle (id) ON DELETE SET NULL,
    plate_number   VARCHAR(20),
    event_type     VARCHAR(10) NOT NULL,
    gate_name      VARCHAR(50),
    occurred_at    TIMESTAMP   NOT NULL DEFAULT CURRENT_TIMESTAMP,
    recognition_id BIGINT      REFERENCES plate_recognition (id) ON DELETE SET NULL,
    confidence     DOUBLE PRECISION,
    allowed        BOOLEAN     NOT NULL DEFAULT FALSE,
    reject_reason  VARCHAR(100)
);

CREATE INDEX IF NOT EXISTS idx_event_occurred_at ON access_event (occurred_at DESC);
CREATE INDEX IF NOT EXISTS idx_event_type        ON access_event (event_type);
CREATE INDEX IF NOT EXISTS idx_event_dedup       ON access_event (plate_number, gate_name, event_type, occurred_at DESC);

-- ---------------------------------------------------------------------
-- 5) parking_policy : 요금 정책. 활성(active=true) 정책 1건을 사용한다.
--    free_minutes  : 무료 시간(분)
--    base_minutes  : 기본 시간(분) / base_fee : 기본 요금
--    unit_minutes  : 추가 단위 시간(분) / unit_fee : 단위 요금
--    daily_max_fee : 1일 상한 요금
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS parking_policy (
    id            BIGSERIAL   PRIMARY KEY,
    name          VARCHAR(80) NOT NULL,
    free_minutes  INTEGER     NOT NULL DEFAULT 30,
    base_minutes  INTEGER     NOT NULL DEFAULT 60,
    base_fee      BIGINT      NOT NULL DEFAULT 2000,
    unit_minutes  INTEGER     NOT NULL DEFAULT 10,
    unit_fee      BIGINT      NOT NULL DEFAULT 500,
    daily_max_fee BIGINT      NOT NULL DEFAULT 20000,
    active        BOOLEAN     NOT NULL DEFAULT TRUE
);

-- ---------------------------------------------------------------------
-- 6) parking_session : 입차~출차 1회. exit_at 이 NULL 이면 주차 중.
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS parking_session (
    id              BIGSERIAL   PRIMARY KEY,
    vehicle_id      BIGINT      REFERENCES vehicle (id) ON DELETE SET NULL,
    plate_number    VARCHAR(20),
    entry_event_id  BIGINT,
    exit_event_id   BIGINT,
    entry_at        TIMESTAMP   NOT NULL,
    exit_at         TIMESTAMP,
    parked_minutes  INTEGER,
    fee_amount      BIGINT,
    discount_amount BIGINT      NOT NULL DEFAULT 0,
    paid            BOOLEAN     NOT NULL DEFAULT FALSE,
    payment_method  VARCHAR(20)
);

CREATE INDEX IF NOT EXISTS idx_session_active   ON parking_session (exit_at);
CREATE INDEX IF NOT EXISTS idx_session_plate    ON parking_session (plate_number);
CREATE INDEX IF NOT EXISTS idx_session_entry_at ON parking_session (entry_at DESC);

-- ---------------------------------------------------------------------
-- 7) maintenance_record : 정비 이력
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS maintenance_record (
    id           BIGSERIAL   PRIMARY KEY,
    vehicle_id   BIGINT      NOT NULL REFERENCES vehicle (id) ON DELETE CASCADE,
    service_date DATE        NOT NULL,
    odometer_km  INTEGER,
    item         VARCHAR(80) NOT NULL,
    part_name    VARCHAR(80),
    cost         BIGINT,
    shop_name    VARCHAR(80),
    memo         VARCHAR(500)
);

CREATE INDEX IF NOT EXISTS idx_maintenance_vehicle ON maintenance_record (vehicle_id, service_date DESC);

-- ---------------------------------------------------------------------
-- 8) sensor_reading : 차량 텔레메트리 시계열(예지보전 입력)
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS sensor_reading (
    id               BIGSERIAL PRIMARY KEY,
    vehicle_id       BIGINT    NOT NULL REFERENCES vehicle (id) ON DELETE CASCADE,
    measured_at      TIMESTAMP NOT NULL,
    odometer_km      INTEGER,
    engine_temp_c    DOUBLE PRECISION,
    engine_rpm       INTEGER,
    speed_kph        DOUBLE PRECISION,
    battery_voltage  DOUBLE PRECISION,
    battery_soh      DOUBLE PRECISION,
    tire_pressure_fl DOUBLE PRECISION,
    tire_pressure_fr DOUBLE PRECISION,
    tire_pressure_rl DOUBLE PRECISION,
    tire_pressure_rr DOUBLE PRECISION,
    brake_pad_mm     DOUBLE PRECISION,
    oil_quality      DOUBLE PRECISION,
    vibration_rms    DOUBLE PRECISION,
    fuel_level       DOUBLE PRECISION
);

CREATE INDEX IF NOT EXISTS idx_sensor_vehicle_time ON sensor_reading (vehicle_id, measured_at DESC);

-- ---------------------------------------------------------------------
-- 9) pdm_prediction : 예지보전 예측 결과. risk_level 은 LOW|MEDIUM|HIGH|CRITICAL
--    detail 은 부품별 점수/권고 목록을 담은 JSONB.
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS pdm_prediction (
    id             BIGSERIAL   PRIMARY KEY,
    vehicle_id     BIGINT      NOT NULL REFERENCES vehicle (id) ON DELETE CASCADE,
    predicted_at   TIMESTAMP   NOT NULL DEFAULT CURRENT_TIMESTAMP,
    health_score   DOUBLE PRECISION,
    anomaly_score  DOUBLE PRECISION,
    failure_risk   DOUBLE PRECISION,
    risk_level     VARCHAR(10) NOT NULL,
    rul_days       INTEGER,
    top_component  VARCHAR(50),
    detail         JSONB,
    model_version  VARCHAR(50)
);

CREATE INDEX IF NOT EXISTS idx_pdm_vehicle_time ON pdm_prediction (vehicle_id, predicted_at DESC);
CREATE INDEX IF NOT EXISTS idx_pdm_risk_level   ON pdm_prediction (risk_level);

-- ---------------------------------------------------------------------
-- 10) model_benchmark : 모델 성능 비교표(6장 벤치마크 절에서 인용)
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS model_benchmark (
    id               BIGSERIAL   PRIMARY KEY,
    model_name       VARCHAR(80) NOT NULL,
    task             VARCHAR(40) NOT NULL,
    detect_recall    DOUBLE PRECISION,
    exact_match_rate DOUBLE PRECISION,
    latency_p50_ms   DOUBLE PRECISION,
    latency_p95_ms   DOUBLE PRECISION,
    memory_mb        DOUBLE PRECISION,
    model_size_mb    DOUBLE PRECISION,
    quantized        BOOLEAN     NOT NULL DEFAULT FALSE,
    measured_at      TIMESTAMP   NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_benchmark_measured_at ON model_benchmark (measured_at DESC);
