-- 예지보전 예측 결과 저장 SQL (PostgreSQL 16)
-- 대상 테이블: pdm_prediction (CONTRACT.md 2절)
--   id, vehicle_id, predicted_at, health_score, anomaly_score, failure_risk,
--   risk_level, rul_days, top_component, detail JSONB, model_version

-- 1) 단건 저장 --------------------------------------------------------------
-- components 와 recommendations 는 열로 펼치지 않고 detail(JSONB) 에 넣는다.
-- 부품 개수와 권고 개수가 가변이므로 정규화하면 테이블이 과도하게 늘어난다.
-- JSONB 로 두면 조회 시 jsonb_array_elements 로 펼칠 수 있다.
INSERT INTO pdm_prediction (
    vehicle_id, predicted_at, health_score, anomaly_score, failure_risk,
    risk_level, rul_days, top_component, detail, model_version
) VALUES (
    :vehicle_id,
    COALESCE(CAST(:computed_at AS timestamp), now()),
    :health_score,
    :anomaly_score,
    :failure_risk,
    :risk_level,          -- LOW | MEDIUM | HIGH | CRITICAL
    :rul_days,
    :top_component,
    CAST(:detail AS jsonb),   -- {"components":[...], "recommendations":[...]}
    :model_version
)
RETURNING id, predicted_at;

-- 2) 차량별 최신 예측 1건 ---------------------------------------------------
-- GET /api/v1/pdm/vehicles/{id}/latest 가 쓰는 질의.
SELECT p.id, p.vehicle_id, v.plate_number, p.predicted_at,
       p.health_score, p.anomaly_score, p.failure_risk, p.risk_level,
       p.rul_days, p.top_component, p.detail, p.model_version
  FROM pdm_prediction p
  JOIN vehicle v ON v.id = p.vehicle_id
 WHERE p.vehicle_id = :vehicle_id
 ORDER BY p.predicted_at DESC
 LIMIT 1;

-- 3) 위험 차량 목록 ---------------------------------------------------------
-- GET /api/v1/pdm/risky?level=HIGH 가 쓰는 질의.
-- 차량마다 최신 1건만 보아야 하므로 DISTINCT ON 을 쓴다.
-- (윈도우 함수 row_number() 보다 PostgreSQL 에서 빠른 경우가 많다)
SELECT DISTINCT ON (p.vehicle_id)
       p.vehicle_id, v.plate_number, v.owner_name, v.model_name,
       p.health_score, p.failure_risk, p.risk_level, p.rul_days,
       p.top_component, p.predicted_at
  FROM pdm_prediction p
  JOIN vehicle v ON v.id = p.vehicle_id
 WHERE v.active = TRUE
   AND p.risk_level = ANY (CAST(:levels AS text[]))   -- 예: '{HIGH,CRITICAL}'
 ORDER BY p.vehicle_id, p.predicted_at DESC;

-- 4) 대시보드 위험도 분포 ---------------------------------------------------
-- GET /api/v1/dashboard/summary 의 riskDistribution 을 채운다.
WITH latest AS (
    SELECT DISTINCT ON (vehicle_id) vehicle_id, risk_level
      FROM pdm_prediction
     ORDER BY vehicle_id, predicted_at DESC
)
SELECT lv.risk_level,
       COUNT(latest.vehicle_id) AS vehicle_count
  FROM (VALUES ('LOW'), ('MEDIUM'), ('HIGH'), ('CRITICAL')) AS lv(risk_level)
  LEFT JOIN latest ON latest.risk_level = lv.risk_level
 GROUP BY lv.risk_level
 ORDER BY CASE lv.risk_level
              WHEN 'LOW' THEN 1 WHEN 'MEDIUM' THEN 2
              WHEN 'HIGH' THEN 3 ELSE 4 END;

-- 5) detail JSONB 펼치기 — 부품 교체 예정 차량 ------------------------------
-- 7일 안에 정비해야 하는 권고를 가진 차량을 뽑는다.
WITH latest AS (
    SELECT DISTINCT ON (vehicle_id) *
      FROM pdm_prediction
     ORDER BY vehicle_id, predicted_at DESC
)
SELECT l.vehicle_id, v.plate_number, l.risk_level,
       rec ->> 'item'                      AS item,
       (rec ->> 'priority')::int           AS priority,
       (rec ->> 'due_days')::int           AS due_days,
       (rec ->> 'due_km')::int             AS due_km
  FROM latest l
  JOIN vehicle v ON v.id = l.vehicle_id
  CROSS JOIN LATERAL jsonb_array_elements(l.detail -> 'recommendations') AS rec
 WHERE (rec ->> 'due_days')::int <= 7
 ORDER BY due_days ASC, priority ASC;

-- 6) 부품별 상태 집계 — 어느 부품이 전사적으로 문제인가 ---------------------
WITH latest AS (
    SELECT DISTINCT ON (vehicle_id) *
      FROM pdm_prediction
     ORDER BY vehicle_id, predicted_at DESC
)
SELECT comp ->> 'name'                        AS component,
       COUNT(*)                               AS total,
       COUNT(*) FILTER (WHERE comp ->> 'status' <> 'NORMAL') AS abnormal,
       ROUND(AVG((comp ->> 'score')::numeric), 1)            AS avg_score
  FROM latest l
  CROSS JOIN LATERAL jsonb_array_elements(l.detail -> 'components') AS comp
 GROUP BY comp ->> 'name'
 ORDER BY avg_score ASC;

-- 7) 건강점수 추이 — 차량 한 대의 최근 90일 ---------------------------------
SELECT date_trunc('day', predicted_at) AS day,
       ROUND(AVG(health_score)::numeric, 1) AS health_score,
       ROUND(AVG(failure_risk)::numeric, 3) AS failure_risk,
       MIN(rul_days)                        AS rul_days
  FROM pdm_prediction
 WHERE vehicle_id = :vehicle_id
   AND predicted_at >= now() - INTERVAL '90 days'
 GROUP BY 1
 ORDER BY 1;

-- 8) 인덱스 ------------------------------------------------------------------
-- 위 질의 패턴(차량별 최신 1건, 등급 필터)에 맞춘 인덱스를 둔다.
CREATE INDEX IF NOT EXISTS idx_pdm_vehicle_time
    ON pdm_prediction (vehicle_id, predicted_at DESC);
CREATE INDEX IF NOT EXISTS idx_pdm_risk_time
    ON pdm_prediction (risk_level, predicted_at DESC);
-- detail 안의 키로 조회할 일이 있으면 GIN 인덱스를 추가한다.
CREATE INDEX IF NOT EXISTS idx_pdm_detail_gin
    ON pdm_prediction USING gin (detail jsonb_path_ops);

-- 9) 보관 정책 ---------------------------------------------------------------
-- 예측은 센서가 들어올 때마다 쌓이므로 빠르게 커진다. 6.13절의 보관기간
-- 관리 원칙에 따라 오래된 행은 일별 요약만 남기고 지운다.
DELETE FROM pdm_prediction
 WHERE predicted_at < now() - INTERVAL '365 days';
