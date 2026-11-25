-- =====================================================================
-- 물류센터 핵심 운영 지표 쿼리 (6.8절)
-- 대기시간 / 하역시간 / 체류시간과 그 분위수, 도크 가동률을 구한다.
-- schema_extension.sql 과 seed_demo.sql 을 먼저 적용한다.
--
-- 적용:
--   psql -U mobility -d mobility_ai -f kpi_queries.sql
--
-- 시간 환산 규칙:
--   interval 끼리는 뺄 수 있지만 숫자로 쓰려면 EXTRACT(EPOCH FROM ...) 로
--   초 단위를 꺼내 60.0 으로 나눈다. 60 으로 나누면 정수 나눗셈이 되어
--   소수가 버려지므로 반드시 60.0 을 쓴다.
--   ROUND(x, 1) 은 numeric 만 받으므로 ::numeric 캐스트가 필요하다.
-- =====================================================================

-- ---------------------------------------------------------------------
-- Q1. 일자별 평균 대기시간 / 평균 하역시간 / 평균 체류시간
--     대기시간 = queued_at ~ loading_at  (줄 선 시간)
--     하역시간 = loading_at ~ done_at    (실제 작업 시간)
--     체류시간 = arrived_at ~ departed_at(센터 안에 있던 총 시간)
--
--     AVG 는 NULL 을 자동으로 제외하므로 아직 진행 중인 건은 빠진다.
--     COALESCE(..., 0) 으로 NULL 을 0 으로 바꾸면 "대기 0분" 으로 섞여
--     평균이 낮아진다 — 통계에서 NULL 을 0 으로 채우면 안 되는 사례다.
-- ---------------------------------------------------------------------
SELECT date_trunc('day', arrived_at)::date                       AS day,
       COUNT(*)                                                   AS arrived_jobs,
       COUNT(*) FILTER (WHERE status = 'DEPARTED')                AS completed_jobs,
       ROUND(AVG(EXTRACT(EPOCH FROM (loading_at - queued_at)) / 60.0)::numeric, 1)
                                                                  AS avg_wait_min,
       ROUND(AVG(EXTRACT(EPOCH FROM (done_at - loading_at)) / 60.0)::numeric, 1)
                                                                  AS avg_load_min,
       ROUND(AVG(EXTRACT(EPOCH FROM (departed_at - arrived_at)) / 60.0)::numeric, 1)
                                                                  AS avg_turnaround_min
FROM dock_job
WHERE arrived_at IS NOT NULL
GROUP BY 1
ORDER BY 1 DESC;

-- ---------------------------------------------------------------------
-- Q2. 대기시간 분위수 (p50 / p90 / 최대)
--     평균만 보면 긴 대기를 놓친다. 평균 15분인데 p90 이 70분이면
--     10대 중 1대는 70분을 기다린다는 뜻이고, 그 1대의 운전자가 민원을 넣는다.
--
--     PERCENTILE_CONT 는 보간 분위수(연속), PERCENTILE_DISC 는 실제 관측값이다.
--     대기시간은 연속량이므로 보간 쪽이 적절하다.
--     WITHIN GROUP (ORDER BY ...) 구문은 순서 집합 집계의 정렬 기준을 준다.
-- ---------------------------------------------------------------------
SELECT COUNT(*)                                                   AS samples,
       ROUND(AVG(wait_min)::numeric, 1)                            AS avg_min,
       ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY wait_min)::numeric, 1)
                                                                   AS p50_min,
       ROUND(PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY wait_min)::numeric, 1)
                                                                   AS p90_min,
       ROUND(MAX(wait_min)::numeric, 1)                            AS max_min
FROM (
    SELECT EXTRACT(EPOCH FROM (loading_at - queued_at)) / 60.0 AS wait_min
    FROM dock_job
    WHERE queued_at IS NOT NULL AND loading_at IS NOT NULL
      AND arrived_at >= CURRENT_DATE - INTERVAL '7 days'
) AS w;

-- ---------------------------------------------------------------------
-- Q3. 시간대별 도착/출차 통행량
--     완성 프로젝트 AccessEventRepository.countHourlyTraffic 과 같은
--     FILTER 집계 패턴을 dock_job 에 적용한다. 프론트엔드가 쓰는 자료 구조
--     {hour, entries, exits} 와 동일하므로 기존 BarChart 를 그대로 재사용한다.
--
--     generate_series(0, 23) 로 24칸을 먼저 만들고 LEFT JOIN 하는 것이 핵심이다.
--     GROUP BY EXTRACT(HOUR ...) 만 쓰면 이벤트가 없는 시간대가 아예 빠져
--     차트 x축이 끊긴다.
-- ---------------------------------------------------------------------
SELECT h.hour                                                      AS hour,
       COUNT(j.id) FILTER (WHERE EXTRACT(HOUR FROM j.arrived_at) = h.hour)  AS entries,
       COUNT(j.id) FILTER (WHERE EXTRACT(HOUR FROM j.departed_at) = h.hour) AS exits
FROM generate_series(0, 23) AS h(hour)
LEFT JOIN dock_job j
       ON (EXTRACT(HOUR FROM j.arrived_at)  = h.hour
        OR EXTRACT(HOUR FROM j.departed_at) = h.hour)
      AND COALESCE(j.arrived_at, j.departed_at) >= CURRENT_DATE
GROUP BY h.hour
ORDER BY h.hour;

-- ---------------------------------------------------------------------
-- Q4. 도크별 가동률
--     가동 시간 = 해당 도크에서 수행된 하역 시간의 합.
--     가동률 = 가동 시간 / 운영 시간. 여기서는 하루 12시간 운영을 가정한다
--     (가정이 숫자로 박혀 있을 때는 반드시 주석에 밝힌다. 실무에서는
--      dock 또는 설정 테이블에 운영 시간을 둬야 한다).
--
--     비어 있는 도크도 보여야 하므로 dock 을 기준으로 LEFT JOIN 한다.
--     가동률과 평균 대기시간을 나란히 봐야 "있는 도크를 안 쓰고 있다" 는
--     진짜 문제가 보인다.
-- ---------------------------------------------------------------------
SELECT d.dock_no,
       d.zone,
       d.refrigerated,
       COUNT(j.id)                                                 AS jobs,
       ROUND(COALESCE(SUM(EXTRACT(EPOCH FROM (j.done_at - j.loading_at)) / 60.0), 0)::numeric, 0)
                                                                   AS busy_min,
       ROUND((COALESCE(SUM(EXTRACT(EPOCH FROM (j.done_at - j.loading_at)) / 60.0), 0)
              / (12 * 60.0) * 100)::numeric, 1)                     AS utilization_pct
FROM dock d
LEFT JOIN dock_job j
       ON j.dock_id = d.id
      AND j.loading_at >= CURRENT_DATE
      AND j.done_at IS NOT NULL
WHERE d.enabled
GROUP BY d.id, d.dock_no, d.zone, d.refrigerated
ORDER BY d.dock_no;
