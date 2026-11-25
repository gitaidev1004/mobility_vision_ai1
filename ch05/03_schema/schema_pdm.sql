-- ---------------------------------------------------------------------------
-- 예지보전 관련 테이블 DDL (PostgreSQL 16)
-- CONTRACT.md 의 vehicle / maintenance_record / sensor_reading / pdm_prediction
-- 정의와 컬럼명·타입을 일치시킨다.
--
-- 실행:  psql -U mobility -d mobility_ai -f schema_pdm.sql
-- ---------------------------------------------------------------------------

-- 1) 차량 기본정보 -----------------------------------------------------------
CREATE TABLE IF NOT EXISTS vehicle (
    id              BIGSERIAL     PRIMARY KEY,
    plate_number    VARCHAR(16)   NOT NULL UNIQUE,     -- 12가3456
    owner_name      VARCHAR(64),
    phone           VARCHAR(32),
    vehicle_type    VARCHAR(16),                       -- SEDAN/SUV/TRUCK/EV
    model_name      VARCHAR(64),
    model_year      INTEGER,
    vin             VARCHAR(32),                       -- 차대번호 17자
    pass_type       VARCHAR(16)   DEFAULT 'VISITOR',   -- REGULAR/VISITOR/BLOCKED
    pass_expires_at TIMESTAMP,
    active          BOOLEAN       NOT NULL DEFAULT TRUE,
    registered_at   TIMESTAMP     NOT NULL DEFAULT NOW(),
    CONSTRAINT ck_vehicle_year CHECK (model_year IS NULL
                                      OR model_year BETWEEN 1980 AND 2100)
);

COMMENT ON TABLE  vehicle               IS '차량 기본정보(예지보전의 기준 엔터티)';
COMMENT ON COLUMN vehicle.plate_number  IS '차량번호. 외부 노출 식별자';
COMMENT ON COLUMN vehicle.vin           IS '차대번호. 차량 생애주기 추적 키';

CREATE INDEX IF NOT EXISTS ix_vehicle_plate  ON vehicle (plate_number);
CREATE INDEX IF NOT EXISTS ix_vehicle_active ON vehicle (active)
    WHERE active = TRUE;

-- 2) 정비 이력 ---------------------------------------------------------------
CREATE TABLE IF NOT EXISTS maintenance_record (
    id           BIGSERIAL     PRIMARY KEY,
    vehicle_id   BIGINT        NOT NULL
                 REFERENCES vehicle (id) ON DELETE CASCADE,
    service_date DATE          NOT NULL,               -- 정비 수행일
    odometer_km  INTEGER       NOT NULL,               -- 정비 시점 주행거리
    item         VARCHAR(64)   NOT NULL,               -- ENGINE/BRAKE/TIRE/BATTERY/...
    part_name    VARCHAR(128),                         -- 교환 부품명
    cost         INTEGER       NOT NULL DEFAULT 0,     -- 정비 비용(원)
    shop_name    VARCHAR(128),
    memo         TEXT,
    CONSTRAINT ck_mr_odo  CHECK (odometer_km >= 0),
    CONSTRAINT ck_mr_cost CHECK (cost >= 0)
);

COMMENT ON TABLE  maintenance_record             IS '차량 정비/부품교환/고장 수리 이력';
COMMENT ON COLUMN maintenance_record.item        IS '정비 대상 부품군. 라벨 생성의 기준';
COMMENT ON COLUMN maintenance_record.odometer_km IS '주행거리. 시간 대신 사용하는 열화 축';

-- 차량별 최신 정비 조회가 가장 빈번하므로 (vehicle_id, service_date DESC)
CREATE INDEX IF NOT EXISTS ix_mr_vehicle_date
    ON maintenance_record (vehicle_id, service_date DESC);
CREATE INDEX IF NOT EXISTS ix_mr_item
    ON maintenance_record (item);
CREATE INDEX IF NOT EXISTS ix_mr_vehicle_item_odo
    ON maintenance_record (vehicle_id, item, odometer_km DESC);

-- 3) 센서 시계열 -------------------------------------------------------------
CREATE TABLE IF NOT EXISTS sensor_reading (
    id               BIGSERIAL    PRIMARY KEY,
    vehicle_id       BIGINT       NOT NULL
                     REFERENCES vehicle (id) ON DELETE CASCADE,
    measured_at      TIMESTAMP    NOT NULL,            -- 측정 시각(차량 기준)
    odometer_km      INTEGER,
    engine_temp_c    REAL,                             -- 냉각수 온도(℃)
    engine_rpm       INTEGER,                          -- 엔진 회전수(rpm)
    speed_kph        REAL,                             -- 차량 속도(km/h)
    battery_voltage  REAL,                             -- 제어모듈 전압(V)
    battery_soh      REAL,                             -- 배터리 건강도 0~1
    tire_pressure_fl REAL,                             -- 좌전 타이어 공기압(psi)
    tire_pressure_fr REAL,
    tire_pressure_rl REAL,
    tire_pressure_rr REAL,
    brake_pad_mm     REAL,                             -- 브레이크 패드 두께(mm)
    oil_quality      REAL,                             -- 엔진오일 품질 0~1
    vibration_rms    REAL,                             -- 진동 RMS(g)
    fuel_level       REAL,                             -- 연료 잔량 0~1
    CONSTRAINT uq_sr_vehicle_time UNIQUE (vehicle_id, measured_at),
    CONSTRAINT ck_sr_temp CHECK (engine_temp_c   IS NULL
                                 OR engine_temp_c BETWEEN -40 AND 215),
    CONSTRAINT ck_sr_rpm  CHECK (engine_rpm      IS NULL
                                 OR engine_rpm BETWEEN 0 AND 16383),
    CONSTRAINT ck_sr_soh  CHECK (battery_soh     IS NULL
                                 OR battery_soh BETWEEN 0 AND 1),
    CONSTRAINT ck_sr_oil  CHECK (oil_quality     IS NULL
                                 OR oil_quality BETWEEN 0 AND 1),
    CONSTRAINT ck_sr_fuel CHECK (fuel_level      IS NULL
                                 OR fuel_level BETWEEN 0 AND 1)
);

COMMENT ON TABLE  sensor_reading             IS 'OBD-II/CAN 에서 수집한 차량 센서 시계열';
COMMENT ON COLUMN sensor_reading.measured_at IS '차량 기준 측정 시각. 적재 시각과 다를 수 있음';
COMMENT ON COLUMN sensor_reading.battery_soh IS 'State of Health. 0~1 비율값';

-- 시계열 조회는 거의 항상 "특정 차량의 최근 N건" 형태다.
CREATE INDEX IF NOT EXISTS ix_sr_vehicle_time
    ON sensor_reading (vehicle_id, measured_at DESC);
-- 전체 구간 배치 집계용
CREATE INDEX IF NOT EXISTS ix_sr_time
    ON sensor_reading (measured_at);
-- 과열 차량만 빠르게 찾는 부분 인덱스
CREATE INDEX IF NOT EXISTS ix_sr_overheat
    ON sensor_reading (vehicle_id, measured_at DESC)
    WHERE engine_temp_c > 105;

-- 4) 예지보전 예측 결과 ------------------------------------------------------
CREATE TABLE IF NOT EXISTS pdm_prediction (
    id             BIGSERIAL    PRIMARY KEY,
    vehicle_id     BIGINT       NOT NULL
                   REFERENCES vehicle (id) ON DELETE CASCADE,
    predicted_at   TIMESTAMP    NOT NULL DEFAULT NOW(),
    health_score   REAL         NOT NULL,              -- 0~100 건강점수
    anomaly_score  REAL         NOT NULL,              -- 0~1 이상 점수
    failure_risk   REAL         NOT NULL,              -- 0~1 고장 위험도
    risk_level     VARCHAR(16)  NOT NULL,              -- LOW/MEDIUM/HIGH/CRITICAL
    rul_days       INTEGER,                            -- 잔존수명(일)
    top_component  VARCHAR(32),                        -- engine/battery/brake/tire/cooling
    detail         JSONB,                              -- 부품별 점수/권고 원본
    model_version  VARCHAR(32)  NOT NULL,
    CONSTRAINT ck_pp_health CHECK (health_score  BETWEEN 0 AND 100),
    CONSTRAINT ck_pp_anom   CHECK (anomaly_score BETWEEN 0 AND 1),
    CONSTRAINT ck_pp_risk   CHECK (failure_risk  BETWEEN 0 AND 1),
    CONSTRAINT ck_pp_level  CHECK (risk_level IN
                   ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL'))
);

COMMENT ON TABLE  pdm_prediction               IS '예지보전 AI 추론 결과 저장';
COMMENT ON COLUMN pdm_prediction.detail        IS 'components/recommendations 배열을 그대로 보관';
COMMENT ON COLUMN pdm_prediction.model_version IS '재현성을 위한 모델 버전 태그';

CREATE INDEX IF NOT EXISTS ix_pp_vehicle_time
    ON pdm_prediction (vehicle_id, predicted_at DESC);
CREATE INDEX IF NOT EXISTS ix_pp_risk
    ON pdm_prediction (risk_level, predicted_at DESC);
-- JSONB 내부 검색용(부품명으로 찾기)
CREATE INDEX IF NOT EXISTS ix_pp_detail_gin
    ON pdm_prediction USING GIN (detail);

-- 5) 차량별 최신 예측 뷰 -----------------------------------------------------
CREATE OR REPLACE VIEW v_vehicle_health AS
SELECT v.id            AS vehicle_id,
       v.plate_number,
       v.model_name,
       p.health_score,
       p.anomaly_score,
       p.failure_risk,
       p.risk_level,
       p.rul_days,
       p.top_component,
       p.predicted_at
FROM   vehicle v
LEFT JOIN LATERAL (
    SELECT *
    FROM   pdm_prediction pp
    WHERE  pp.vehicle_id = v.id
    ORDER  BY pp.predicted_at DESC
    LIMIT  1
) p ON TRUE
WHERE  v.active = TRUE;

COMMENT ON VIEW v_vehicle_health IS '차량별 최신 예지보전 결과(대시보드용)';
