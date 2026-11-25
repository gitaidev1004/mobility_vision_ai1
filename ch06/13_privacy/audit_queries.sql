-- =====================================================================
-- 감사 로그 조회 SQL 모음 — 운영 중 실제로 쓰게 되는 질문들
-- =====================================================================

-- (1) 특정 사용자의 최근 7일 활동 — 내부자 점검의 출발점
SELECT occurred_at, action, target_table, plate_masked, reason, result
  FROM audit_log
 WHERE actor = 'operator'
   AND occurred_at >= CURRENT_DATE - INTERVAL '7 days'
 ORDER BY occurred_at DESC
 LIMIT 200;

-- (2) 번호판 조회가 유독 많은 사용자 — 비정상 사용 탐지
--     업무상 하루 30건을 넘기 어려운 역할이 300건을 조회했다면 확인 대상이다.
SELECT actor,
       actor_role,
       COUNT(*)                                    AS search_count,
       COUNT(DISTINCT plate_masked)                AS distinct_plates,
       MIN(occurred_at)                            AS first_at,
       MAX(occurred_at)                            AS last_at
  FROM audit_log
 WHERE action IN ('PLATE_SEARCH', 'PLATE_VIEW')
   AND occurred_at >= CURRENT_DATE - INTERVAL '30 days'
 GROUP BY actor, actor_role
HAVING COUNT(*) > 300
 ORDER BY search_count DESC;

-- (3) 사유를 적지 않은 조회 — 사후 소명이 불가능한 접근
SELECT actor, actor_role, action, COUNT(*) AS no_reason_count
  FROM audit_log
 WHERE action IN ('PLATE_SEARCH', 'IMAGE_VIEW', 'IMAGE_DOWNLOAD')
   AND (reason IS NULL OR btrim(reason) = '')
   AND occurred_at >= CURRENT_DATE - INTERVAL '30 days'
 GROUP BY actor, actor_role, action
 ORDER BY no_reason_count DESC;

-- (4) 업무시간 외 접근 — 09시 이전·19시 이후 또는 주말
SELECT occurred_at, actor, actor_role, action, plate_masked, client_ip
  FROM audit_log
 WHERE occurred_at >= CURRENT_DATE - INTERVAL '30 days'
   AND (EXTRACT(HOUR FROM occurred_at) NOT BETWEEN 9 AND 18
        OR EXTRACT(ISODOW FROM occurred_at) >= 6)
   AND action <> 'LOGIN'
 ORDER BY occurred_at DESC;

-- (5) 접근 거부 이력 — 권한 설정 오류이거나 침입 시도다
SELECT date_trunc('hour', occurred_at) AS hour,
       actor, client_ip, action, COUNT(*) AS denied_count
  FROM audit_log
 WHERE result = 'DENIED'
   AND occurred_at >= CURRENT_DATE - INTERVAL '7 days'
 GROUP BY 1, 2, 3, 4
HAVING COUNT(*) >= 3
 ORDER BY hour DESC, denied_count DESC;

-- (6) 정보주체 열람 요구 대응 — "내 차량 정보를 누가 봤나"
--     마스킹 값으로 조회하므로 원문 번호판을 같은 규칙으로 마스킹해 넣는다.
SELECT occurred_at, actor, actor_role, action, reason, result
  FROM audit_log
 WHERE plate_masked = '12가**56'
 ORDER BY occurred_at DESC;

-- (7) 이미지 대량 내려받기 — 유출 위험이 가장 높은 행위
SELECT actor,
       COUNT(*)                        AS download_count,
       SUM((detail->>'bytes')::bigint) AS total_bytes
  FROM audit_log
 WHERE action = 'IMAGE_DOWNLOAD'
   AND occurred_at >= CURRENT_DATE - INTERVAL '7 days'
 GROUP BY actor
 ORDER BY download_count DESC;

-- (8) 보관기간 배치 실행 이력 — 삭제가 실제로 돌았는지 증빙
SELECT occurred_at,
       target_table,
       action,
       detail->>'affected'      AS affected,
       detail->>'files_removed' AS files_removed,
       detail->>'rule_days'     AS rule_days,
       result
  FROM audit_log
 WHERE action LIKE 'RETENTION_%'
 ORDER BY occurred_at DESC
 LIMIT 100;

-- (9) 인식 결과 수정 이력 — 운영자가 AI 판정을 바꾼 기록
SELECT a.occurred_at, a.actor, a.target_id,
       a.detail->>'before' AS before_plate,
       a.detail->>'after'  AS after_plate,
       a.reason
  FROM audit_log a
 WHERE a.action = 'RECOGNITION_OVERRIDE'
   AND a.occurred_at >= CURRENT_DATE - INTERVAL '30 days'
 ORDER BY a.occurred_at DESC;
