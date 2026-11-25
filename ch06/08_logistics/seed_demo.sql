-- =====================================================================
-- 물류센터 확장 데모 데이터 (6.8절)
-- schema_extension.sql 적용 후 실행한다. kpi_queries.sql 과
-- queue_queries.sql 의 집계 결과를 눈으로 확인하기 위한 데이터다.
--
-- 구성: 당일 완료 작업 8건 + 현재 대기 2건 + 미도착 1건 + 취소 1건
-- 모든 INSERT 는 재실행 안전(WHERE NOT EXISTS).
--
-- 적용:
--   psql -U mobility -d mobility_ai -f seed_demo.sql
-- =====================================================================

-- ---------------------------------------------------------------------
-- 화물차 차량 등록 (완성 프로젝트 vehicle 테이블 재사용)
-- 화물차 번호판도 한국 번호판 형식이라 PlateFormatValidator 가 그대로 통한다.
-- ---------------------------------------------------------------------
INSERT INTO vehicle (plate_number, owner_name, phone, vehicle_type, model_name,
                     model_year, pass_type, active, registered_at)
SELECT v.plate_number, v.owner_name, v.phone, 'TRUCK', v.model_name,
       v.model_year, 'STAFF', TRUE, CURRENT_TIMESTAMP
FROM (VALUES
    ('12가3456', '한결로지스', '010-7777-1111', '포터 II',   2021),
    ('34나5678', '미래운수',   '010-7777-2222', '마이티',     2022),
    ('87로2345', '대한통운',   '010-7777-3333', '메가트럭',   2020),
    ('78라1234', '한결로지스', '010-7777-1111', '봉고 III',   2019),
    ('56다7890', '미래운수',   '010-7777-2222', '포터 II',    2023),
    ('90마2345', '대한통운',   '010-7777-3333', '마이티',     2021),
    ('11바3456', '한결로지스', '010-7777-1111', '메가트럭',   2022),
    ('22사4567', '미등록운송', '010-7777-9999', '봉고 III',   2018)
) AS v(plate_number, owner_name, phone, model_name, model_year)
WHERE NOT EXISTS (SELECT 1 FROM vehicle x WHERE x.plate_number = v.plate_number);

-- ---------------------------------------------------------------------
-- 당일 하역 작업 8건. 모두 DEPARTED(완료) 상태로 두어 집계가 바로 나온다.
-- 시각은 base(당일 08:00) + 분 오프셋이고, dock_queue.py 시뮬레이션
-- 결과와 같은 값을 쓴다(교재 본문의 표와 SQL 출력이 일치하도록).
-- 마지막 건(22사4567)은 reserved_at 이 NULL 인 walk-in 이다.
-- ---------------------------------------------------------------------
INSERT INTO dock_job (vehicle_id, carrier_id, plate_number, cargo_type, order_no,
                      status, reserved_at, arrived_at, queued_at, loading_at,
                      done_at, departed_at, dock_id, pallet_count)
SELECT ve.id, ca.id, j.plate_number, j.cargo_type, j.order_no, 'DEPARTED',
       CASE WHEN j.reserved IS NULL THEN NULL
            ELSE base.t + (j.reserved || ' minutes')::interval END,
       base.t + (j.arrived  || ' minutes')::interval,
       base.t + (j.queued   || ' minutes')::interval,
       base.t + (j.loading  || ' minutes')::interval,
       base.t + (j.done     || ' minutes')::interval,
       base.t + (j.departed || ' minutes')::interval,
       dk.id, j.pallets
FROM (VALUES
    -- plate,      carrier,      cargo,    order,      dock, 예약, 도착, 줄선, 접안, 완료, 출차, 팔레트
    ('12가3456', '한결로지스', 'NORMAL', 'ORD-1001', 1,    0,    5,    5,    5,   40,   44,  18),
    ('34나5678', '미래운수',   'REEFER', 'ORD-1002', 2,   20,    8,    8,    8,   48,   51,  12),
    ('87로2345', '대한통운',   'NORMAL', 'ORD-1003', 1,   40,   12,   12,   75,  105,  110,  20),
    ('78라1234', '한결로지스', 'FRESH',  'ORD-1004', 2,   60,   18,   18,   48,   93,   97,  14),
    ('56다7890', '미래운수',   'RETURN', 'ORD-1005', 1,   80,   25,   25,  123,  148,  154,   6),
    ('90마2345', '대한통운',   'REEFER', 'ORD-1006', 2,  100,   30,   30,   40,   75,   78,  16),
    ('11바3456', '한결로지스', 'NORMAL', 'ORD-1007', 1,  120,   36,   36,   93,  123,  128,  22),
    ('22사4567',  NULL,        'NORMAL', NULL,       2, NULL,   44,   44,  105,  125,  129,   8)
) AS j(plate_number, carrier_name, cargo_type, order_no, dock_no,
       reserved, arrived, queued, loading, done, departed, pallets)
CROSS JOIN (SELECT date_trunc('day', CURRENT_TIMESTAMP) + interval '8 hours' AS t) AS base
JOIN vehicle ve ON ve.plate_number = j.plate_number
JOIN dock dk    ON dk.dock_no = j.dock_no
LEFT JOIN carrier ca ON ca.name = j.carrier_name
WHERE NOT EXISTS (
    SELECT 1 FROM dock_job x
    WHERE x.plate_number = j.plate_number
      AND x.status = 'DEPARTED'
      AND x.arrived_at >= date_trunc('day', CURRENT_TIMESTAMP)
);

-- ---------------------------------------------------------------------
-- 현재 대기 중인 작업 2건. queue_queries.sql Q1(대기열 조회)이
-- 빈 결과를 내지 않게 한다. queued_at 을 과거로 두어
-- "몇 분 기다렸는지" 가 NOW() 기준으로 계산되게 한다.
-- ---------------------------------------------------------------------
INSERT INTO dock_job (vehicle_id, carrier_id, plate_number, cargo_type, order_no,
                      status, reserved_at, arrived_at, queued_at, pallet_count)
SELECT ve.id, ca.id, w.plate_number, w.cargo_type, w.order_no, 'WAITING',
       NOW() - ((w.ago + 10) || ' minutes')::interval,
       NOW() - (w.ago || ' minutes')::interval,
       NOW() - (w.ago || ' minutes')::interval,
       w.pallets
FROM (VALUES
    ('56다7890', '미래운수',   'REEFER', 'ORD-1010', 25, 10),
    ('11바3456', '한결로지스', 'NORMAL', 'ORD-1011', 12, 20)
) AS w(plate_number, carrier_name, cargo_type, order_no, ago, pallets)
JOIN vehicle ve ON ve.plate_number = w.plate_number
LEFT JOIN carrier ca ON ca.name = w.carrier_name
WHERE NOT EXISTS (SELECT 1 FROM dock_job x WHERE x.order_no = w.order_no);

-- ---------------------------------------------------------------------
-- 미도착(no-show) 1건과 취소 1건. 운영 개선 지표가 0 이 아니게 한다.
-- 두 건은 차량 등록이 없어도 되므로 vehicle_id 를 비워 둔다
-- (plate_number 비정규화 덕분에 미등록 차량도 기록할 수 있다).
-- ---------------------------------------------------------------------
INSERT INTO dock_job (carrier_id, plate_number, cargo_type, order_no, status, reserved_at)
SELECT ca.id, s.plate_number, s.cargo_type, s.order_no, s.status,
       date_trunc('day', CURRENT_TIMESTAMP) + (s.reserved || ' minutes')::interval
FROM (VALUES
    ('33아7788', '미래운수', 'NORMAL', 'ORD-1008', 'NO_SHOW',  540),
    ('44자8899', '대한통운', 'FRESH',  'ORD-1009', 'CANCELED', 600)
) AS s(plate_number, carrier_name, cargo_type, order_no, status, reserved)
LEFT JOIN carrier ca ON ca.name = s.carrier_name
WHERE NOT EXISTS (SELECT 1 FROM dock_job x WHERE x.order_no = s.order_no);
