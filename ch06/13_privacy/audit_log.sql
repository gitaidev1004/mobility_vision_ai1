-- =====================================================================
-- 감사 로그 테이블 — 누가 · 언제 · 무엇을 · 왜 조회/변경했는가
-- 번호판은 원문을 넣지 않고 마스킹된 형태로만 기록한다.
-- 감사 로그 자체가 개인정보 유출 경로가 되면 안 되기 때문이다.
-- =====================================================================
CREATE TABLE IF NOT EXISTS audit_log (
    id            BIGSERIAL    PRIMARY KEY,
    occurred_at   TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP,
    actor         VARCHAR(100) NOT NULL,               -- app_user.username 또는 배치명
    actor_role    VARCHAR(20)  NOT NULL,               -- ADMIN|OPERATOR|VIEWER|SYSTEM
    action        VARCHAR(40)  NOT NULL,               -- 아래 CHECK 참조
    target_table  VARCHAR(60),                         -- 대상 테이블
    target_id     BIGINT,                              -- 대상 레코드 PK
    plate_masked  VARCHAR(20),                         -- '12가**56' 형태만 저장
    reason        VARCHAR(200),                        -- 조회·변경 사유(필수 입력 권장)
    result        VARCHAR(20)  NOT NULL DEFAULT 'SUCCESS',  -- SUCCESS|DENIED|ERROR
    client_ip     INET,                                -- 접속 IP
    detail        JSONB,                               -- 추가 맥락(검색 조건 등)
    CONSTRAINT ck_audit_action CHECK (action IN (
        'LOGIN', 'LOGIN_FAILED', 'LOGOUT',
        'PLATE_SEARCH', 'PLATE_VIEW', 'IMAGE_VIEW', 'IMAGE_DOWNLOAD',
        'RECOGNITION_OVERRIDE', 'EVENT_DELETE', 'VEHICLE_EXPORT',
        'PSEUDONYM_VERIFY',
        'RETENTION_PURGE', 'RETENTION_ANONYMIZE', 'RETENTION_ARCHIVE',
        'PERMISSION_CHANGE'
    )),
    CONSTRAINT ck_audit_role CHECK (actor_role IN
        ('ADMIN', 'OPERATOR', 'VIEWER', 'SYSTEM')),
    CONSTRAINT ck_audit_result CHECK (result IN ('SUCCESS', 'DENIED', 'ERROR'))
);

-- 조회 패턴별 인덱스
-- (1) "최근 무슨 일이 있었나" — 운영 화면 기본 조회
CREATE INDEX IF NOT EXISTS ix_audit_occurred
    ON audit_log (occurred_at DESC);
-- (2) "이 사용자가 무엇을 했나" — 내부자 점검
CREATE INDEX IF NOT EXISTS ix_audit_actor
    ON audit_log (actor, occurred_at DESC);
-- (3) "이 차량을 누가 조회했나" — 정보주체 열람 요구 대응
CREATE INDEX IF NOT EXISTS ix_audit_plate
    ON audit_log (plate_masked, occurred_at DESC)
    WHERE plate_masked IS NOT NULL;
-- (4) "거부된 접근 시도" — 부분 인덱스로 작게 유지
CREATE INDEX IF NOT EXISTS ix_audit_denied
    ON audit_log (occurred_at DESC)
    WHERE result = 'DENIED';

-- 감사 로그는 수정·삭제되면 안 된다. 보관기간 배치만 예외적으로 지운다.
-- 애플리케이션 계정에는 INSERT/SELECT 만 준다.
REVOKE UPDATE, DELETE ON audit_log FROM mobility;
GRANT INSERT, SELECT ON audit_log TO mobility;
GRANT USAGE, SELECT ON SEQUENCE audit_log_id_seq TO mobility;

COMMENT ON TABLE  audit_log IS '개인정보 접근·변경 감사 로그. 번호판은 마스킹 저장';
COMMENT ON COLUMN audit_log.reason IS '조회 사유. 비어 있으면 사후 소명이 불가하므로 필수 입력을 권장';
COMMENT ON COLUMN audit_log.plate_masked IS '마스킹된 번호판. 원문을 넣지 않는다';
