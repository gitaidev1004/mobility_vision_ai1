-- =====================================================================
-- V2__seed.sql : 교재 실습용 시드 데이터
-- 재실행 안전성을 위해 모든 INSERT 는 중복 키 회피 조건을 둔다.
-- =====================================================================

-- ---------------------------------------------------------------------
-- 계정 3종. password_hash 는 BCrypt($2a$10$...) 이며 평문은 각각
--   admin/admin123!   operator/operator123!   viewer/viewer123!
-- 해시가 비어 있거나 '{noop}' 로 시작하면 백엔드 기동 시 DataInitializer
-- 가 재인코딩한다(안전장치).
-- ---------------------------------------------------------------------
INSERT INTO app_user (username, password_hash, display_name, role, enabled)
SELECT 'admin', '$2a$10$penBqeU5sCej.KrLjvZoQu5y7ILD7nZ1zy2U.42COZ4dH4Xr9tfBe', '관리자', 'ADMIN', TRUE
WHERE NOT EXISTS (SELECT 1 FROM app_user WHERE username = 'admin');

INSERT INTO app_user (username, password_hash, display_name, role, enabled)
SELECT 'operator', '$2a$10$HzfmLePfGLoZDgiyCXcygeGDhfXCDqNPY5ek6zYeROwcnetLWJq/u', '운영자', 'OPERATOR', TRUE
WHERE NOT EXISTS (SELECT 1 FROM app_user WHERE username = 'operator');

INSERT INTO app_user (username, password_hash, display_name, role, enabled)
SELECT 'viewer', '$2a$10$OjehWyqnE9cY/5NCJFgFmuvUmt2r3ZThu6UYkZpmlwlvhJgs9SYEm', '조회자', 'VIEWER', TRUE
WHERE NOT EXISTS (SELECT 1 FROM app_user WHERE username = 'viewer');

-- ---------------------------------------------------------------------
-- 주차 요금 정책 : 무료 30분 / 기본 60분 2,000원 / 추가 10분당 500원 / 일 상한 20,000원
-- ---------------------------------------------------------------------
INSERT INTO parking_policy (name, free_minutes, base_minutes, base_fee, unit_minutes, unit_fee, daily_max_fee, active)
SELECT '기본 요금제', 30, 60, 2000, 10, 500, 20000, TRUE
WHERE NOT EXISTS (SELECT 1 FROM parking_policy WHERE name = '기본 요금제');

-- ---------------------------------------------------------------------
-- 등록 차량 10대. pass_type: NONE|VISITOR|REGULAR|STAFF
-- ---------------------------------------------------------------------
INSERT INTO vehicle (plate_number, owner_name, phone, vehicle_type, model_name, model_year, vin, pass_type, pass_expires_at, active, registered_at)
SELECT v.plate_number, v.owner_name, v.phone, v.vehicle_type, v.model_name, v.model_year, v.vin,
       v.pass_type, v.pass_expires_at, TRUE, CURRENT_TIMESTAMP - (v.days_ago || ' days')::interval
FROM (VALUES
    ('12가3456', '홍길동', '010-1111-2222', 'SEDAN',  '아반떼 CN7',     2021, 'KMHLM41AAMU000111', 'REGULAR', CURRENT_TIMESTAMP + interval '180 days', 300),
    ('34나5678', '김영희', '010-2222-3333', 'SUV',    '쏘렌토 MQ4',     2022, 'KNAPM81ABNS000222', 'REGULAR', CURRENT_TIMESTAMP + interval '90 days',  280),
    ('56다7890', '이철수', '010-3333-4444', 'SEDAN',  'K5 DL3',         2020, 'KNAGM41AALA000333', 'NONE',    NULL,                                    260),
    ('78라1234', '박민수', '010-4444-5555', 'TRUCK',  '포터 II',        2019, 'KMFZBN7HPKU000444', 'STAFF',   CURRENT_TIMESTAMP + interval '365 days', 240),
    ('90마2345', '최지은', '010-5555-6666', 'SUV',    '투싼 NX4',       2023, 'KMHJB81CBPU000555', 'VISITOR', CURRENT_TIMESTAMP + interval '7 days',   200),
    ('11바3456', '정다혜', '010-6666-7777', 'EV',     '아이오닉 5',     2023, 'KMHKR81GBPA000666', 'REGULAR', CURRENT_TIMESTAMP + interval '45 days',  180),
    ('22사4567', '강호동', '010-7777-8888', 'SEDAN',  'G80 RG3',        2022, 'KMTGA41ABNU000777', 'NONE',    NULL,                                    150),
    ('33아5678', '윤서연', '010-8888-9999', 'VAN',    '카니발 KA4',     2021, 'KNAME81ABMS000888', 'STAFF',   CURRENT_TIMESTAMP + interval '300 days', 120),
    ('44자6789', '임재현', '010-9999-0000', 'EV',     'EV6 CV',         2024, 'KNAEC81GBRA000999', 'VISITOR', CURRENT_TIMESTAMP - interval '3 days',    90),
    ('55차7890', '서지훈', '010-1212-3434', 'SUV',    '쏘렌토 하이브리드', 2024, 'KNAPM81CBRS001010', 'NONE',  NULL,                                     60)
) AS v(plate_number, owner_name, phone, vehicle_type, model_name, model_year, vin, pass_type, pass_expires_at, days_ago)
WHERE NOT EXISTS (SELECT 1 FROM vehicle x WHERE x.plate_number = v.plate_number);

-- ---------------------------------------------------------------------
-- 정비 이력
-- ---------------------------------------------------------------------
INSERT INTO maintenance_record (vehicle_id, service_date, odometer_km, item, part_name, cost, shop_name, memo)
SELECT ve.id, m.service_date, m.odometer_km, m.item, m.part_name, m.cost, m.shop_name, m.memo
FROM (VALUES
    ('12가3456', CURRENT_DATE - 240, 61000, '엔진오일 교환',   '0W-20 합성유',     88000,  '블루핸즈 강남점', '필터 동시 교환'),
    ('12가3456', CURRENT_DATE - 90,  74500, '브레이크 패드',   '전륜 패드 세트',   180000, '블루핸즈 강남점', '잔량 3.1mm 에서 교환'),
    ('34나5678', CURRENT_DATE - 150, 38000, '타이어 교환',     '245/45R19 4본',    760000, '타이어뱅크 송파', '편마모'),
    ('56다7890', CURRENT_DATE - 60,  95200, '배터리 교환',     'AGM 70Ah',         230000, '카포스 분당점',   'SOH 68% 경고'),
    ('78라1234', CURRENT_DATE - 30,  182000,'미션오일 교환',   'ATF SP-IV',        150000, '화물차정비 시흥', '주행거리 과다'),
    ('11바3456', CURRENT_DATE - 20,  24000, '에어컨 필터',     '활성탄 필터',      35000,  '오토큐 수지점',   '정기 점검'),
    ('22사4567', CURRENT_DATE - 10,  41000, '정기 점검',       NULL,               60000,  '제네시스 서비스', '이상 없음')
) AS m(plate_number, service_date, odometer_km, item, part_name, cost, shop_name, memo)
JOIN vehicle ve ON ve.plate_number = m.plate_number
WHERE NOT EXISTS (
    SELECT 1 FROM maintenance_record r
    WHERE r.vehicle_id = ve.id AND r.service_date = m.service_date AND r.item = m.item
);

-- ---------------------------------------------------------------------
-- 센서 시계열 : 차량별 24건(1시간 간격). 일부 차량은 열화 패턴을 섞는다.
-- generate_series 로 생성하되, 이미 데이터가 있으면 건너뛴다.
-- ---------------------------------------------------------------------
INSERT INTO sensor_reading (
    vehicle_id, measured_at, odometer_km, engine_temp_c, engine_rpm, speed_kph,
    battery_voltage, battery_soh, tire_pressure_fl, tire_pressure_fr,
    tire_pressure_rl, tire_pressure_rr, brake_pad_mm, oil_quality, vibration_rms, fuel_level)
SELECT
    ve.id,
    CURRENT_TIMESTAMP - (g.h || ' hours')::interval,
    80000 + (ve.id * 1500) + (24 - g.h) * 12,
    -- 차량 id 가 홀수면 과열 경향을 살짝 준다.
    88.0 + (ve.id % 3) * 3.5 + (g.h % 5) * 0.8,
    1400 + (g.h % 7) * 120,
    30.0 + (g.h % 11) * 4.5,
    12.6 - (ve.id % 4) * 0.25,
    0.95 - (ve.id % 5) * 0.06,
    33.0 - (g.h % 4) * 0.3,
    32.8 - (g.h % 4) * 0.3,
    31.5 - (ve.id % 3) * 0.7,
    31.9 - (ve.id % 3) * 0.7,
    8.5 - (ve.id % 6) * 1.1,
    0.85 - (ve.id % 7) * 0.08,
    0.22 + (ve.id % 5) * 0.07,
    0.75 - (g.h % 9) * 0.04
FROM vehicle ve
CROSS JOIN generate_series(0, 23) AS g(h)
WHERE NOT EXISTS (SELECT 1 FROM sensor_reading s WHERE s.vehicle_id = ve.id);

-- ---------------------------------------------------------------------
-- 모델 벤치마크 시드 (6장 모델 비교 절에서 인용)
-- ---------------------------------------------------------------------
INSERT INTO model_benchmark (model_name, task, detect_recall, exact_match_rate,
                             latency_p50_ms, latency_p95_ms, memory_mb, model_size_mb, quantized, measured_at)
SELECT b.model_name, b.task, b.detect_recall, b.exact_match_rate,
       b.latency_p50_ms, b.latency_p95_ms, b.memory_mb, b.model_size_mb, b.quantized, CURRENT_TIMESTAMP
FROM (VALUES
    ('YOLO-nano + CRNN/CTC',      'detect+ocr', 0.991, 0.962, 61.4,  88.1,  214.0, 9.8,  FALSE),
    ('YOLO-nano + Attention OCR', 'detect+ocr', 0.991, 0.971, 94.2,  131.5, 268.0, 14.2, FALSE),
    ('MobileNet-SSD + CRNN/CTC',  'detect+ocr', 0.964, 0.948, 42.8,  60.3,  148.0, 6.1,  FALSE),
    ('MobileNet-SSD + Paddle',    'detect+ocr', 0.964, 0.957, 70.6,  101.2, 196.0, 11.4, FALSE),
    ('YOLO-nano INT8 + CRNN INT8','detect+ocr', 0.982, 0.944, 28.5,  41.7,  96.0,  3.2,  TRUE)
) AS b(model_name, task, detect_recall, exact_match_rate, latency_p50_ms, latency_p95_ms, memory_mb, model_size_mb, quantized)
WHERE NOT EXISTS (SELECT 1 FROM model_benchmark x WHERE x.model_name = b.model_name);

-- ---------------------------------------------------------------------
-- 오늘자 출입 이벤트 + 주차 세션 (대시보드가 바로 그려지도록)
-- 입차만 한 차량 3대(주차 중), 입출차 완료 4건(요금 결제 2건/미결제 2건)
-- ---------------------------------------------------------------------

-- (1) 주차 중 세션용 ENTRY 이벤트
INSERT INTO access_event (vehicle_id, plate_number, event_type, gate_name, occurred_at, confidence, allowed, reject_reason)
SELECT ve.id, ve.plate_number, 'ENTRY', e.gate_name,
       date_trunc('day', CURRENT_TIMESTAMP) + (e.hh || ' hours')::interval, e.conf, TRUE, NULL
FROM (VALUES
    ('12가3456', '정문',   9,  0.973),
    ('34나5678', '정문',   10, 0.961),
    ('11바3456', '후문',   11, 0.948)
) AS e(plate_number, gate_name, hh, conf)
JOIN vehicle ve ON ve.plate_number = e.plate_number
WHERE NOT EXISTS (
    SELECT 1 FROM access_event a
    WHERE a.plate_number = e.plate_number AND a.event_type = 'ENTRY'
      AND a.occurred_at >= date_trunc('day', CURRENT_TIMESTAMP)
);

INSERT INTO parking_session (vehicle_id, plate_number, entry_event_id, entry_at, discount_amount, paid)
SELECT a.vehicle_id, a.plate_number, a.id, a.occurred_at, 0, FALSE
FROM access_event a
WHERE a.event_type = 'ENTRY'
  AND a.plate_number IN ('12가3456', '34나5678', '11바3456')
  AND a.occurred_at >= date_trunc('day', CURRENT_TIMESTAMP)
  AND NOT EXISTS (SELECT 1 FROM parking_session s WHERE s.entry_event_id = a.id);

-- (2) 완료 세션 : 어제 입차 → 어제 출차
INSERT INTO access_event (vehicle_id, plate_number, event_type, gate_name, occurred_at, confidence, allowed, reject_reason)
SELECT ve.id, ve.plate_number, e.event_type, e.gate_name,
       date_trunc('day', CURRENT_TIMESTAMP) - interval '1 day' + (e.hh || ' hours')::interval,
       e.conf, TRUE, NULL
FROM (VALUES
    ('56다7890', 'ENTRY', '정문', 8,  0.957),
    ('56다7890', 'EXIT',  '정문', 12, 0.962),
    ('78라1234', 'ENTRY', '후문', 9,  0.934),
    ('78라1234', 'EXIT',  '후문', 10, 0.941),
    ('90마2345', 'ENTRY', '정문', 13, 0.978),
    ('90마2345', 'EXIT',  '정문', 19, 0.969),
    ('22사4567', 'ENTRY', '지하', 7,  0.925),
    ('22사4567', 'EXIT',  '지하', 7,  0.930)
) AS e(plate_number, event_type, gate_name, hh, conf)
JOIN vehicle ve ON ve.plate_number = e.plate_number
WHERE NOT EXISTS (
    SELECT 1 FROM access_event a
    WHERE a.plate_number = e.plate_number AND a.event_type = e.event_type
      AND a.occurred_at >= date_trunc('day', CURRENT_TIMESTAMP) - interval '1 day'
      AND a.occurred_at <  date_trunc('day', CURRENT_TIMESTAMP)
);

-- 완료 세션 적재.
-- fee_amount / discount_amount 는 ParkingFeeCalculator 가 같은 입력에 대해 내놓는 값과
-- 정확히 일치해야 한다(교재에서 시드 데이터와 계산 로직을 대조해 보여 주기 때문).
-- 기본 요금제: 무료 30분 / 기본 60분 2,000원 / 추가 10분당 500원 / 일 상한 20,000원
INSERT INTO parking_session (vehicle_id, plate_number, entry_event_id, exit_event_id,
                             entry_at, exit_at, parked_minutes, fee_amount, discount_amount, paid, payment_method)
SELECT ent.vehicle_id, ent.plate_number, ent.id, ext.id, ent.occurred_at, ext.occurred_at,
       (EXTRACT(EPOCH FROM (ext.occurred_at - ent.occurred_at)) / 60)::int,
       d.fee, d.discount, d.paid, d.method
FROM (VALUES
    -- 240분, 일반: 2000 + (180/10)*500 = 11,000원
    ('56다7890', 11000::bigint, 0::bigint,    TRUE,  'CARD'),
    -- 60분, STAFF 유효 → 전액 면제
    ('78라1234', 0::bigint,     0::bigint,    TRUE,  'PASS'),
    -- 360분, VISITOR 유효: gross 2000 + (300/10)*500 = 17,000 → 20% 할인 3,400 → 13,600원
    ('90마2345', 13600::bigint, 3400::bigint, FALSE, NULL),
    -- 0분(무료 시간 내) → 0원
    ('22사4567', 0::bigint,     0::bigint,    FALSE, NULL)
) AS d(plate_number, fee, discount, paid, method)
JOIN access_event ent ON ent.plate_number = d.plate_number AND ent.event_type = 'ENTRY'
                      AND ent.occurred_at >= date_trunc('day', CURRENT_TIMESTAMP) - interval '1 day'
                      AND ent.occurred_at <  date_trunc('day', CURRENT_TIMESTAMP)
JOIN access_event ext ON ext.plate_number = d.plate_number AND ext.event_type = 'EXIT'
                      AND ext.occurred_at >= date_trunc('day', CURRENT_TIMESTAMP) - interval '1 day'
                      AND ext.occurred_at <  date_trunc('day', CURRENT_TIMESTAMP)
WHERE NOT EXISTS (SELECT 1 FROM parking_session s WHERE s.entry_event_id = ent.id);

-- ---------------------------------------------------------------------
-- 인식 이력 시드 (성공 다수 + 실패 3건) — 대시보드 인식률/실패 건수용
-- ---------------------------------------------------------------------
INSERT INTO plate_recognition (plate_number, confidence, detector_name, ocr_name, image_path,
                               detect_ms, ocr_ms, total_ms, quality_score, valid_format,
                               matched_vehicle_id, status, created_at)
SELECT r.plate_number, r.confidence, 'yolo', 'crnn', '/data/uploads/seed-' || r.seq || '.jpg',
       r.detect_ms, r.ocr_ms, r.detect_ms + r.ocr_ms, r.quality, r.valid_format,
       ve.id, r.status, date_trunc('day', CURRENT_TIMESTAMP) + (r.hh || ' hours')::interval
FROM (VALUES
    (1, '12가3456', 0.973, 38.2, 19.7, 0.86, TRUE,  'SUCCESS',        9),
    (2, '34나5678', 0.961, 40.1, 21.3, 0.83, TRUE,  'SUCCESS',        10),
    (3, '11바3456', 0.948, 36.7, 18.9, 0.79, TRUE,  'SUCCESS',        11),
    (4, '56다7890', 0.957, 39.5, 20.4, 0.81, TRUE,  'SUCCESS',        12),
    (5, '90마2345', 0.978, 35.8, 18.2, 0.88, TRUE,  'SUCCESS',        13),
    (6, NULL,       0.180, 44.3, 0.0,  0.31, FALSE, 'NO_PLATE',       14),
    (7, '00가0000', 0.410, 41.2, 22.8, 0.48, FALSE, 'LOW_CONFIDENCE', 15),
    (8, 'XY-1234',  0.712, 37.9, 20.1, 0.66, FALSE, 'FORMAT_INVALID', 16)
) AS r(seq, plate_number, confidence, detect_ms, ocr_ms, quality, valid_format, status, hh)
LEFT JOIN vehicle ve ON ve.plate_number = r.plate_number
WHERE NOT EXISTS (
    SELECT 1 FROM plate_recognition p
    WHERE p.image_path = '/data/uploads/seed-' || r.seq || '.jpg'
);

-- ---------------------------------------------------------------------
-- 예지보전 예측 시드 : 차량별 1건 (위험도 분포가 보이도록)
-- ---------------------------------------------------------------------
INSERT INTO pdm_prediction (vehicle_id, predicted_at, health_score, anomaly_score, failure_risk,
                            risk_level, rul_days, top_component, detail, model_version)
SELECT ve.id, CURRENT_TIMESTAMP - (p.hours_ago || ' hours')::interval,
       p.health, p.anomaly, p.risk, p.level, p.rul, p.top_component,
       p.detail::jsonb, 'pdm-ensemble-1.0.0'
FROM (VALUES
    ('12가3456', 72.4, 0.38, 0.27, 'MEDIUM',   41,  'brake',   '{"components":[{"name":"brake","score":58.2,"status":"WARNING","reason":"패드 잔량 4.1mm"}],"recommendations":[{"priority":1,"item":"브레이크 패드 점검","dueKm":3000,"dueDays":30}]}', 2),
    ('34나5678', 88.1, 0.14, 0.09, 'LOW',      180, 'engine',  '{"components":[{"name":"engine","score":90.4,"status":"NORMAL","reason":"냉각수 온도 정상 범위"}],"recommendations":[]}', 3),
    ('56다7890', 41.6, 0.72, 0.64, 'HIGH',     12,  'battery', '{"components":[{"name":"battery","score":33.8,"status":"CRITICAL","reason":"SOH 68%, 전압 11.9V"}],"recommendations":[{"priority":1,"item":"배터리 교환","dueKm":500,"dueDays":7}]}', 4),
    ('78라1234', 26.3, 0.88, 0.81, 'CRITICAL', 5,   'engine',  '{"components":[{"name":"engine","score":21.0,"status":"CRITICAL","reason":"진동 RMS 0.63, 냉각수 104도"}],"recommendations":[{"priority":1,"item":"엔진 정밀 진단","dueKm":200,"dueDays":3}]}', 5),
    ('90마2345', 93.5, 0.08, 0.05, 'LOW',      240, 'tire',    '{"components":[{"name":"tire","score":95.1,"status":"NORMAL","reason":"4륜 공기압 균일"}],"recommendations":[]}', 6),
    ('11바3456', 79.8, 0.25, 0.18, 'MEDIUM',   75,  'oil',     '{"components":[{"name":"oil","score":64.7,"status":"WARNING","reason":"오일 품질 지수 0.53"}],"recommendations":[{"priority":2,"item":"엔진오일 교환","dueKm":2000,"dueDays":45}]}', 7),
    ('22사4567', 85.2, 0.19, 0.12, 'LOW',      150, 'engine',  '{"components":[{"name":"engine","score":87.3,"status":"NORMAL","reason":"RPM/온도 정상"}],"recommendations":[]}', 8),
    ('33아5678', 55.9, 0.56, 0.47, 'HIGH',     21,  'tire',    '{"components":[{"name":"tire","score":48.1,"status":"WARNING","reason":"후륜 공기압 29.4psi"}],"recommendations":[{"priority":1,"item":"타이어 공기압 보충","dueKm":1000,"dueDays":14}]}', 9),
    ('44자6789', 91.0, 0.11, 0.07, 'LOW',      210, 'battery', '{"components":[{"name":"battery","score":93.2,"status":"NORMAL","reason":"SOH 95%"}],"recommendations":[]}', 10),
    ('55차7890', 68.4, 0.42, 0.31, 'MEDIUM',   36,  'brake',   '{"components":[{"name":"brake","score":61.5,"status":"WARNING","reason":"패드 잔량 4.8mm"}],"recommendations":[{"priority":2,"item":"브레이크 점검","dueKm":4000,"dueDays":40}]}', 11)
) AS p(plate_number, health, anomaly, risk, level, rul, top_component, detail, hours_ago)
JOIN vehicle ve ON ve.plate_number = p.plate_number
WHERE NOT EXISTS (SELECT 1 FROM pdm_prediction x WHERE x.vehicle_id = ve.id);
