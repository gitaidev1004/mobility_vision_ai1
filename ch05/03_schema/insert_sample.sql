-- ---------------------------------------------------------------------------
-- 예지보전 실습용 샘플 데이터
-- 실행:  psql -U mobility -d mobility_ai -f insert_sample.sql
-- schema_pdm.sql 을 먼저 실행해야 한다.
-- ---------------------------------------------------------------------------

-- 1) 차량 4대 등록 -----------------------------------------------------------
INSERT INTO vehicle (plate_number, owner_name, phone, vehicle_type,
                     model_name, model_year, vin, pass_type, active)
VALUES
  ('12가3456',    '홍길동', '010-1234-5678', 'SEDAN',
   '소나타 DN8',  2021, 'KMHL341ABMA123456', 'REGULAR', TRUE),
  ('서울 34나5678', '김영희', '010-2345-6789', 'SUV',
   '쏘렌토 MQ4',  2022, 'KNAR381BMNA234567', 'REGULAR', TRUE),
  ('87로2345',    '박철수', '010-3456-7890', 'TRUCK',
   '포터 II',     2019, 'KMFZBN7HP9U345678', 'VISITOR', TRUE),
  ('45다6789',    '이민정', '010-4567-8901', 'EV',
   '아이오닉 5',  2023, 'KMHKR81GPNU456789', 'REGULAR', TRUE)
ON CONFLICT (plate_number) DO NOTHING;

-- 2) 정비 이력 ---------------------------------------------------------------
-- 12가3456 : 주기 정비를 꾸준히 받은 정상 차량
INSERT INTO maintenance_record (vehicle_id, service_date, odometer_km,
                                item, part_name, cost, shop_name, memo)
SELECT id, DATE '2025-11-14', 71200, 'ENGINE', '엔진오일 + 오일필터',
       98000, '모빌리티정비 강남점', '정기 교환. 이상 없음'
FROM vehicle WHERE plate_number = '12가3456';

INSERT INTO maintenance_record (vehicle_id, service_date, odometer_km,
                                item, part_name, cost, shop_name, memo)
SELECT id, DATE '2026-04-02', 78640, 'BRAKE', '전륜 브레이크 패드',
       186000, '모빌리티정비 강남점', '패드 잔량 3.2mm 에서 교환'
FROM vehicle WHERE plate_number = '12가3456';

INSERT INTO maintenance_record (vehicle_id, service_date, odometer_km,
                                item, part_name, cost, shop_name, memo)
SELECT id, DATE '2026-09-18', 83910, 'ENGINE', '엔진오일 + 에어필터',
       112000, '모빌리티정비 강남점', '오일 품질 지수 0.41'
FROM vehicle WHERE plate_number = '12가3456';

-- 서울 34나5678 : 배터리 열화 이력
INSERT INTO maintenance_record (vehicle_id, service_date, odometer_km,
                                item, part_name, cost, shop_name, memo)
SELECT id, DATE '2026-02-21', 38120, 'BATTERY', 'AGM 배터리 90Ah',
       268000, '모빌리티정비 송파점', '시동 전압 10.9V 로 하강. 교환'
FROM vehicle WHERE plate_number = '서울 34나5678';

INSERT INTO maintenance_record (vehicle_id, service_date, odometer_km,
                                item, part_name, cost, shop_name, memo)
SELECT id, DATE '2026-08-07', 40460, 'TIRE', '전륜 타이어 2본',
       342000, '모빌리티정비 송파점', '편마모 확인. 휠 얼라인먼트 동시 조정'
FROM vehicle WHERE plate_number = '서울 34나5678';

-- 87로2345 : 과적 운행으로 열화가 빠른 화물차
INSERT INTO maintenance_record (vehicle_id, service_date, odometer_km,
                                item, part_name, cost, shop_name, memo)
SELECT id, DATE '2026-03-11', 124300, 'COOLING', '워터펌프 + 냉각수',
       421000, '물류정비센터', '냉각수 온도 112℃ 경고 후 입고'
FROM vehicle WHERE plate_number = '87로2345';

INSERT INTO maintenance_record (vehicle_id, service_date, odometer_km,
                                item, part_name, cost, shop_name, memo)
SELECT id, DATE '2026-07-29', 131880, 'BRAKE', '후륜 브레이크 디스크 + 패드',
       612000, '물류정비센터', '패드 잔량 1.8mm. 디스크 변형 동반'
FROM vehicle WHERE plate_number = '87로2345';

INSERT INTO maintenance_record (vehicle_id, service_date, odometer_km,
                                item, part_name, cost, shop_name, memo)
SELECT id, DATE '2026-10-05', 136740, 'ENGINE', '점화 플러그 4개',
       148000, '물류정비센터', '진동 RMS 0.74g. 실화(misfire) 의심'
FROM vehicle WHERE plate_number = '87로2345';

-- 3) 센서 시계열 (일부 구간만 수록. 전체는 load_csv_to_db.py 로 적재) --------
INSERT INTO sensor_reading (
    vehicle_id, measured_at, odometer_km,
    engine_temp_c, engine_rpm, speed_kph,
    battery_voltage, battery_soh,
    tire_pressure_fl, tire_pressure_fr, tire_pressure_rl, tire_pressure_rr,
    brake_pad_mm, oil_quality, vibration_rms, fuel_level)
SELECT v.id, t.measured_at, t.odometer_km,
       t.engine_temp_c, t.engine_rpm, t.speed_kph,
       t.battery_voltage, t.battery_soh,
       t.fl, t.fr, t.rl, t.rr,
       t.brake_pad_mm, t.oil_quality, t.vibration_rms, t.fuel_level
FROM   vehicle v
CROSS JOIN (VALUES
  (TIMESTAMP '2026-10-01 09:00:00', 84210,  92.4, 1850, 48.0,
   12.40, 0.86, 32.1, 31.8, 30.9, 31.2, 4.10, 0.62, 0.44, 0.55),
  (TIMESTAMP '2026-10-01 09:10:00', 84218,  93.1, 1920, 52.4,
   14.05, 0.86, 32.3, 31.9, 31.0, 31.3, 4.09, 0.62, 0.46, 0.54),
  (TIMESTAMP '2026-10-01 09:20:00', 84227,  94.8, 2240, 61.7,
   14.11, 0.86, 32.5, 32.0, 31.1, 31.4, 4.08, 0.61, 0.51, 0.53),
  (TIMESTAMP '2026-10-01 09:30:00', 84233,  91.6,  820,  0.0,
   13.52, 0.85, 32.4, 31.9, 31.0, 31.3, 4.08, 0.61, 0.26, 0.53),
  (TIMESTAMP '2026-10-01 09:40:00', 84241,  95.2, 2080, 55.3,
   14.08, 0.85, 32.6, 32.1, 31.2, 31.5, 4.07, 0.61, 0.49, 0.52)
) AS t(measured_at, odometer_km, engine_temp_c, engine_rpm, speed_kph,
       battery_voltage, battery_soh, fl, fr, rl, rr,
       brake_pad_mm, oil_quality, vibration_rms, fuel_level)
WHERE v.plate_number = '12가3456'
ON CONFLICT (vehicle_id, measured_at) DO NOTHING;

-- 87로2345 : 열화 후반 구간. 과열과 저전압이 함께 나타난다.
INSERT INTO sensor_reading (
    vehicle_id, measured_at, odometer_km,
    engine_temp_c, engine_rpm, speed_kph,
    battery_voltage, battery_soh,
    tire_pressure_fl, tire_pressure_fr, tire_pressure_rl, tire_pressure_rr,
    brake_pad_mm, oil_quality, vibration_rms, fuel_level)
SELECT v.id, t.measured_at, t.odometer_km,
       t.engine_temp_c, t.engine_rpm, t.speed_kph,
       t.battery_voltage, t.battery_soh,
       t.fl, t.fr, t.rl, t.rr,
       t.brake_pad_mm, t.oil_quality, t.vibration_rms, t.fuel_level
FROM   vehicle v
CROSS JOIN (VALUES
  (TIMESTAMP '2026-10-20 14:00:00', 137820, 104.6, 2380, 72.4,
   12.18, 0.59, 29.4, 31.1, 30.4, 30.2, 2.10, 0.22, 0.91, 0.38),
  (TIMESTAMP '2026-10-20 14:10:00', 137834, 107.9, 2510, 78.1,
   11.86, 0.58, 29.1, 31.0, 30.3, 30.1, 2.08, 0.21, 1.14, 0.37),
  (TIMESTAMP '2026-10-20 14:20:00', 137847, 111.2, 2640, 81.6,
   11.42, 0.57, 28.8, 30.9, 30.2, 30.0, 2.05, 0.20, 1.48, 0.36),
  (TIMESTAMP '2026-10-20 14:30:00', 137852, 113.7, 1480, 12.3,
   11.05, 0.56, 28.6, 30.8, 30.1, 29.9, 2.04, 0.19, 1.86, 0.36)
) AS t(measured_at, odometer_km, engine_temp_c, engine_rpm, speed_kph,
       battery_voltage, battery_soh, fl, fr, rl, rr,
       brake_pad_mm, oil_quality, vibration_rms, fuel_level)
WHERE v.plate_number = '87로2345'
ON CONFLICT (vehicle_id, measured_at) DO NOTHING;

-- 4) 예지보전 예측 결과 ------------------------------------------------------
INSERT INTO pdm_prediction (vehicle_id, predicted_at, health_score,
                            anomaly_score, failure_risk, risk_level,
                            rul_days, top_component, detail, model_version)
SELECT id, TIMESTAMP '2026-10-24 12:00:00', 72.4, 0.38, 0.27, 'MEDIUM',
       41, 'brake',
       '{"components":[
            {"name":"engine","score":81.2,"status":"NORMAL",
             "reason":"냉각수 온도 정상 범위"},
            {"name":"brake","score":58.4,"status":"WARNING",
             "reason":"패드 잔량 4.1mm"}],
          "recommendations":[
            {"priority":1,"item":"브레이크 패드 점검",
             "due_km":3000,"due_days":30}]}'::jsonb,
       'pdm-ensemble-1.0.0'
FROM vehicle WHERE plate_number = '12가3456';

INSERT INTO pdm_prediction (vehicle_id, predicted_at, health_score,
                            anomaly_score, failure_risk, risk_level,
                            rul_days, top_component, detail, model_version)
SELECT id, TIMESTAMP '2026-10-24 12:00:00', 31.8, 0.86, 0.74, 'CRITICAL',
       6, 'cooling',
       '{"components":[
            {"name":"cooling","score":18.2,"status":"CRITICAL",
             "reason":"냉각수 온도 113.7℃, 상한 105℃ 초과 지속"},
            {"name":"battery","score":34.6,"status":"CRITICAL",
             "reason":"전압 11.05V, SOH 0.56"},
            {"name":"brake","score":41.0,"status":"WARNING",
             "reason":"패드 잔량 2.04mm"}],
          "recommendations":[
            {"priority":1,"item":"냉각 시스템 즉시 점검",
             "due_km":200,"due_days":2},
            {"priority":2,"item":"배터리 교환","due_km":1000,"due_days":7}]}'::jsonb,
       'pdm-ensemble-1.0.0'
FROM vehicle WHERE plate_number = '87로2345';

-- 5) 적재 확인 ---------------------------------------------------------------
SELECT 'vehicle'            AS tbl, COUNT(*) FROM vehicle
UNION ALL SELECT 'maintenance_record', COUNT(*) FROM maintenance_record
UNION ALL SELECT 'sensor_reading',     COUNT(*) FROM sensor_reading
UNION ALL SELECT 'pdm_prediction',     COUNT(*) FROM pdm_prediction;
