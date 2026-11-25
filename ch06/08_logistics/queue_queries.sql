-- =====================================================================
-- 물류센터 대기열·예약 분석 쿼리 (6.8절)
-- 현재 대기열, 운송사별 처리량, 예약 대비 도착 지연, walk-in/no-show 요약.
-- schema_extension.sql 과 seed_demo.sql 을 먼저 적용한다.
--
-- 적용:
--   psql -U mobility -d mobility_ai -f queue_queries.sql
-- =====================================================================

-- ---------------------------------------------------------------------
-- Q1. 현재 대기열 (운영 화면 좌측 패널)
--     정렬 우선순위: 화물 종류 → 운송사 등급 → 예약 시각 → 줄 선 시각.
--     dock_queue.py 의 queue_key() 와 같은 규칙을 SQL 로 표현한 것이다.
--
--     예약이 없는 walk-in 은 NULLS LAST 로 같은 등급 안에서 맨 뒤로 보낸다.
--     PostgreSQL 은 오름차순에서 NULL 을 기본적으로 마지막에 두지만,
--     명시하면 의도가 드러나고 MySQL(NULL 먼저)로 옮길 때도 안전하다.
--
--     NOW() - queued_at 으로 "지금까지 몇 분 기다렸는지" 를 실시간 계산한다.
-- ---------------------------------------------------------------------
SELECT j.plate_number,
       COALESCE(c.name, '미등록')                                  AS carrier,
       j.cargo_type,
       j.order_no,
       j.reserved_at,
       j.queued_at,
       ROUND((EXTRACT(EPOCH FROM (NOW() - j.queued_at)) / 60.0)::numeric, 0)
                                                                   AS waiting_min,
       CASE j.cargo_type
            WHEN 'REEFER' THEN 1
            WHEN 'FRESH'  THEN 2
            WHEN 'NORMAL' THEN 3
            WHEN 'RETURN' THEN 4
            ELSE 9
       END                                                         AS priority
FROM dock_job j
LEFT JOIN carrier c ON c.id = j.carrier_id
WHERE j.status = 'WAITING'
ORDER BY priority,
         COALESCE(c.tier, 9),
         j.reserved_at NULLS LAST,
         j.queued_at;

-- ---------------------------------------------------------------------
-- Q2. 운송사별 처리량과 평균 대기시간
--     LEFT JOIN 을 쓰는 이유: 운송사가 등록되어 있지만 당일 입고가 없는 곳도
--     0건으로 보여야 운영자가 "누가 안 왔는지" 를 알 수 있다.
--
--     날짜 조건을 ON 절에 둔 것이 핵심이다. WHERE 로 옮기면 당일 작업이 없는
--     운송사의 행에서 j.reserved_at 이 NULL 이 되어 걸러지고, 그 운송사가
--     결과에서 사라진다 — LEFT JOIN 을 쓴 의도가 깨진다.
--
--     COUNT(j.id) 를 쓰는 이유: COUNT(*) 는 LEFT JOIN 이 남긴 NULL 행도 세어
--     작업이 없는 운송사가 1 로 집계된다. COUNT(열) 은 NULL 을 세지 않는다.
-- ---------------------------------------------------------------------
SELECT c.name                                                      AS carrier,
       c.tier                                                      AS tier,
       COUNT(j.id)                                                 AS jobs,
       COUNT(j.id) FILTER (WHERE j.status = 'NO_SHOW')              AS no_shows,
       ROUND(AVG(EXTRACT(EPOCH FROM (j.loading_at - j.queued_at)) / 60.0)::numeric, 1)
                                                                   AS avg_wait_min,
       SUM(COALESCE(j.pallet_count, 0))                            AS pallets
FROM carrier c
LEFT JOIN dock_job j
       ON j.carrier_id = c.id
      AND j.reserved_at >= CURRENT_DATE
      AND j.reserved_at <  CURRENT_DATE + INTERVAL '1 day'
WHERE c.active
GROUP BY c.id, c.name, c.tier
ORDER BY jobs DESC, c.tier;

-- ---------------------------------------------------------------------
-- Q3. 예약 대비 도착 지연 분포
--     지연 = arrived_at - reserved_at. 음수면 조기 도착이다.
--     조기 도착도 문제다 — 예약보다 1시간 일찍 오면 대기열을 막는다.
--     그래서 지연을 구간으로 나눠 분포를 본다.
--
--     CASE 분기 순서가 결과를 좌우한다. SQL 의 CASE 는 위에서부터 처음 참인
--     조건을 택하므로 좁은 조건을 먼저 쓴다. 또 arrived_at IS NULL 을 맨 앞에
--     두어야 한다 — NULL 과의 비교는 모두 NULL 이므로 미도착 건이 ELSE 로
--     떨어져 "60분 초과 지연" 에 섞인다.
-- ---------------------------------------------------------------------
SELECT CASE
           WHEN j.arrived_at IS NULL                      THEN '미도착'
           WHEN delay_min < -30                           THEN '30분 이상 조기'
           WHEN delay_min < -5                            THEN '5~30분 조기'
           WHEN delay_min <= 15                            THEN '정시(±15분)'
           WHEN delay_min <= 60                            THEN '15~60분 지연'
           ELSE '60분 초과 지연'
       END                                                          AS bucket,
       COUNT(*)                                                     AS jobs
FROM (
    SELECT id, arrived_at,
           EXTRACT(EPOCH FROM (arrived_at - reserved_at)) / 60.0 AS delay_min
    FROM dock_job
    WHERE reserved_at >= CURRENT_DATE - INTERVAL '7 days'
) AS j
GROUP BY 1
ORDER BY 2 DESC;

-- ---------------------------------------------------------------------
-- Q4. 예약 없이 들어온 차량(walk-in) 과 미도착(no-show) 요약
--     운영 개선 지표다. walk-in 이 많으면 예약 시스템이 쓰이지 않는 것이고,
--     no-show 가 많으면 예약 신뢰도가 낮아 대기열 계획이 무의미해진다.
--
--     NULLIF(COUNT(*), 0) 으로 0 나눗셈을 막는다. SQL 에서 x / 0 은 예외를
--     던지지만 x / NULL 은 NULL 이 되므로, 쿼리가 실패하는 대신 빈 값이 나온다.
--
--     COALESCE(reserved_at, arrived_at) 로 기간을 거르는 것이 중요하다.
--     walk-in 은 reserved_at 이 NULL 이므로 reserved_at >= ... 조건만으로는
--     전부 제외되어 walk-in 비율이 항상 0% 로 나온다.
-- ---------------------------------------------------------------------
SELECT COUNT(*)                                                     AS total,
       COUNT(*) FILTER (WHERE reserved_at IS NULL)                   AS walk_ins,
       COUNT(*) FILTER (WHERE status = 'NO_SHOW')                    AS no_shows,
       COUNT(*) FILTER (WHERE status = 'CANCELED')                   AS canceled,
       ROUND((COUNT(*) FILTER (WHERE reserved_at IS NULL)::numeric
              / NULLIF(COUNT(*), 0) * 100), 1)                       AS walk_in_pct,
       ROUND((COUNT(*) FILTER (WHERE status = 'NO_SHOW')::numeric
              / NULLIF(COUNT(*), 0) * 100), 1)                       AS no_show_pct
FROM dock_job
WHERE COALESCE(reserved_at, arrived_at) >= CURRENT_DATE - INTERVAL '7 days';
