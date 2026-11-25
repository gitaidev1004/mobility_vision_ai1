-- =====================================================================
-- 물류센터 차량 관리 확장 스키마 (6.8절)
-- 완성 프로젝트의 V1__init.sql 위에 추가로 적용한다.
-- 기존 테이블(vehicle, access_event, plate_recognition)은 수정하지 않고,
-- 새 테이블 3개만 더해 물류 도메인을 얹는다.
--
-- 적용:
--   psql -U mobility -d mobility_ai -f schema_extension.sql
-- =====================================================================

-- ---------------------------------------------------------------------
-- 1) carrier : 운송사 마스터. 협력사별 허용 차량/우선순위를 관리한다.
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS carrier (
    id           BIGSERIAL   PRIMARY KEY,
    name         VARCHAR(80) NOT NULL UNIQUE,
    contact      VARCHAR(50),
    phone        VARCHAR(30),
    -- 운송사 등급. 숫자가 작을수록 대기열에서 앞선다(동일 화물 종류 내에서).
    tier         INTEGER     NOT NULL DEFAULT 3,
    active       BOOLEAN     NOT NULL DEFAULT TRUE,
    created_at   TIMESTAMP   NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- ---------------------------------------------------------------------
-- 2) dock : 하역 도크. 냉장 설비 유무가 배정 제약이 된다.
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS dock (
    id           BIGSERIAL   PRIMARY KEY,
    dock_no      INTEGER     NOT NULL UNIQUE,
    zone         VARCHAR(20) NOT NULL DEFAULT 'A',
    -- 냉장/냉동 화물을 받을 수 있는 도크인지
    refrigerated BOOLEAN     NOT NULL DEFAULT FALSE,
    -- 정비/점검으로 일시 사용 중지
    enabled      BOOLEAN     NOT NULL DEFAULT TRUE
);

-- ---------------------------------------------------------------------
-- 3) dock_job : 배차 예약 1건 = 하역 작업 1건.
--    status 전이: RESERVED → ARRIVED → WAITING → LOADING → DONE → DEPARTED
--                 (RESERVED → NO_SHOW|CANCELED, ARRIVED/WAITING → CANCELED)
--    각 전이 시각을 별도 컬럼으로 남겨 대기시간/하역시간/체류시간을 계산한다.
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS dock_job (
    id             BIGSERIAL   PRIMARY KEY,
    -- 등록 차량과 연결. 예약 없이 들어온 walk-in 은 NULL 가능.
    vehicle_id     BIGINT      REFERENCES vehicle (id) ON DELETE SET NULL,
    carrier_id     BIGINT      REFERENCES carrier (id) ON DELETE SET NULL,
    -- 번호판은 vehicle 미등록 차량도 받기 위해 비정규화해 둔다(access_event 와 동일 전략).
    plate_number   VARCHAR(20) NOT NULL,
    -- REEFER|FRESH|NORMAL|RETURN
    cargo_type     VARCHAR(20) NOT NULL DEFAULT 'NORMAL',
    order_no       VARCHAR(40),
    status         VARCHAR(20) NOT NULL DEFAULT 'RESERVED',
    reserved_at    TIMESTAMP,
    arrived_at     TIMESTAMP,
    queued_at      TIMESTAMP,
    loading_at     TIMESTAMP,
    done_at        TIMESTAMP,
    departed_at    TIMESTAMP,
    dock_id        BIGINT      REFERENCES dock (id) ON DELETE SET NULL,
    -- 도착/출차를 유발한 출입 이벤트(추적용)
    arrive_event_id BIGINT     REFERENCES access_event (id) ON DELETE SET NULL,
    depart_event_id BIGINT     REFERENCES access_event (id) ON DELETE SET NULL,
    pallet_count   INTEGER,
    memo           VARCHAR(300)
);

-- 대기열 조회: status='WAITING' 인 건을 우선순위대로 뽑는 쿼리가 가장 빈번하다.
CREATE INDEX IF NOT EXISTS idx_dock_job_status   ON dock_job (status);
-- 번호판으로 "지금 진행 중인 작업" 을 찾는 조회(도착 인식 직후).
CREATE INDEX IF NOT EXISTS idx_dock_job_plate    ON dock_job (plate_number, status);
-- 일자별 집계용.
CREATE INDEX IF NOT EXISTS idx_dock_job_reserved ON dock_job (reserved_at DESC);
CREATE INDEX IF NOT EXISTS idx_dock_job_arrived  ON dock_job (arrived_at DESC);

-- ---------------------------------------------------------------------
-- 시드: 도크 4개(2번은 냉장), 운송사 3곳
-- 모든 INSERT 는 재실행 안전(WHERE NOT EXISTS).
-- ---------------------------------------------------------------------
INSERT INTO dock (dock_no, zone, refrigerated, enabled)
SELECT d.dock_no, d.zone, d.refrigerated, TRUE
FROM (VALUES
    (1, 'A', FALSE),
    (2, 'A', TRUE),
    (3, 'B', FALSE),
    (4, 'B', FALSE)
) AS d(dock_no, zone, refrigerated)
WHERE NOT EXISTS (SELECT 1 FROM dock x WHERE x.dock_no = d.dock_no);

INSERT INTO carrier (name, contact, phone, tier, active)
SELECT c.name, c.contact, c.phone, c.tier, TRUE
FROM (VALUES
    ('한결로지스', '김운송', '010-7777-1111', 1),
    ('미래운수',   '박배차', '010-7777-2222', 2),
    ('대한통운',   '이물류', '010-7777-3333', 2)
) AS c(name, contact, phone, tier)
WHERE NOT EXISTS (SELECT 1 FROM carrier x WHERE x.name = c.name);
