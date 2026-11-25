# -*- coding: utf-8 -*-
"""보관기간 자동 삭제 배치.

보관기간 정책은 문서로만 있으면 지켜지지 않는다. "90일 보관" 이라고 써 두고
3년치 이미지를 쌓아 둔 시스템이 대부분이다. 이 배치는 기간이 지난 데이터를
정책에 따라 삭제·비식별화하고, 그 작업 자체를 감사 로그에 남긴다.

세 가지 처리 유형
    purge      — 레코드와 파일을 완전히 삭제한다(원상복구 불가).
    anonymize  — 레코드는 남기고 식별자만 지운다. 통계 연속성이 필요할 때.
    archive    — 다른 저장소로 옮기고 원본에서 지운다(법정 보존 의무가 있는 경우).

안전장치
    1) 기본은 dry-run 이다. ``--apply`` 를 명시해야 실제로 지운다.
    2) 한 번에 지우는 건수를 ``--batch`` 로 제한한다. 수십만 건을 단일
        트랜잭션으로 지우면 DB 가 잠기고 서비스가 멈춘다.
    3) 삭제 전 건수를 세어 로그에 남기고, 예상보다 10배 많으면 중단한다
       (정책 계산 실수로 전체 데이터를 지우는 사고를 막는다).
    4) 모든 실행은 ``audit_log`` 에 1건으로 기록한다.

사용법
    python 13_privacy/retention_job.py                 # dry-run (DB 없이도 동작)
    python 13_privacy/retention_job.py --apply         # 실제 삭제
    python 13_privacy/retention_job.py --sql           # 실행할 SQL 만 출력
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta

#: 예상 건수 대비 이 배수를 넘으면 중단한다.
SAFETY_FACTOR = 10.0


@dataclass(frozen=True)
class RetentionRule:
    """테이블 1종의 보관기간 정책.

    Attributes:
        table: 대상 테이블.
        time_column: 기준 시각 컬럼.
        days: 보관기간(일).
        action: purge | anonymize | archive.
        basis: 기간 산정 근거(감사 로그에 기록된다).
        files: 함께 삭제할 파일 경로 컬럼(없으면 None).
        anonymize_sql: anonymize 일 때 실행할 UPDATE 문.
    """

    table: str
    time_column: str
    days: int
    action: str
    basis: str
    files: str | None = None
    anonymize_sql: str | None = None


# ---------------------------------------------------------------------------
# 보관기간 정책 — 숫자는 예시다. 실제 기간은 기관 지침에 맞춰 조정한다.
# ---------------------------------------------------------------------------

RULES: list[RetentionRule] = [
    RetentionRule(
        table="plate_recognition",
        time_column="created_at",
        days=30,
        action="purge",
        basis="원본 이미지는 오인식 재검토 기간(30일)만 보관",
        files="image_path",
    ),
    RetentionRule(
        table="access_event",
        time_column="occurred_at",
        days=365,
        action="anonymize",
        basis="출입 통계는 1년 유지, 이후 번호판만 가명 토큰으로 대체",
        # pgcrypto 의 hmac 을 쓴다. md5(plate_number) 처럼 비밀키 없는 해시는
        # 번호판 전수 계산으로 즉시 역추적되므로 가명처리로 인정하기 어렵다.
        # :pepper 는 DB 에 저장하지 않고 배치 실행 시 환경변수로 주입한다.
        anonymize_sql=(
            "UPDATE access_event\n"
            "   SET plate_number = 'PV-' || substr(\n"
            "         encode(hmac(plate_number, :pepper, 'sha256'), 'hex'), 1, 16),\n"
            "       vehicle_id   = NULL\n"
            " WHERE occurred_at < :cutoff\n"
            "   AND plate_number NOT LIKE 'PV-%'"
        ),
    ),
    RetentionRule(
        table="parking_session",
        time_column="exit_at",
        days=1825,
        action="archive",
        basis="정산 증빙은 상사·세무 보존 기간을 고려해 별도 보관소로 이관",
    ),
    RetentionRule(
        table="sensor_reading",
        time_column="measured_at",
        days=730,
        action="purge",
        basis="예지보전 학습에 쓰는 원시 센서는 2년 후 일별 집계로 대체",
    ),
    RetentionRule(
        table="pdm_prediction",
        time_column="predicted_at",
        days=1095,
        action="purge",
        basis="모델 성능 추적용 예측 이력은 3년",
    ),
    RetentionRule(
        table="audit_log",
        time_column="occurred_at",
        days=1095,
        action="purge",
        basis="감사 로그는 다른 데이터보다 길게(3년) 유지한다",
    ),
]


def cutoff_of(rule: RetentionRule, now: datetime) -> datetime:
    """정책의 삭제 기준 시각을 계산한다.

    Args:
        rule: 보관기간 정책.
        now: 기준 시각(테스트에서 고정값을 넣을 수 있게 인수로 받는다).

    Returns:
        이 시각보다 오래된 레코드가 대상이 된다.
    """
    return now - timedelta(days=rule.days)


def count_sql(rule: RetentionRule) -> str:
    """대상 건수를 세는 SQL 을 만든다."""
    return (f"SELECT COUNT(*) FROM {rule.table}\n"
            f" WHERE {rule.time_column} < :cutoff")


def action_sql(rule: RetentionRule, batch: int) -> str:
    """처리 SQL 을 만든다.

    purge 는 배치 크기로 나눠 지운다. PostgreSQL 에서는 ``ctid`` 를 쓰면
    기본키가 없어도 서브쿼리로 상한을 걸 수 있다.

    Args:
        rule: 정책.
        batch: 1회 처리 건수 상한.

    Returns:
        실행할 SQL 문.
    """
    if rule.action == "anonymize" and rule.anonymize_sql:
        return rule.anonymize_sql
    if rule.action == "archive":
        return (f"-- 1) 보관소로 복사\n"
                f"INSERT INTO {rule.table}_archive\n"
                f"SELECT * FROM {rule.table}\n"
                f" WHERE {rule.time_column} < :cutoff\n"
                f" LIMIT {batch};\n"
                f"-- 2) 원본에서 삭제\n"
                f"DELETE FROM {rule.table}\n"
                f" WHERE id IN (SELECT id FROM {rule.table}_archive\n"
                f"               WHERE {rule.time_column} < :cutoff)")
    return (f"DELETE FROM {rule.table}\n"
            f" WHERE ctid IN (\n"
            f"   SELECT ctid FROM {rule.table}\n"
            f"    WHERE {rule.time_column} < :cutoff\n"
            f"    LIMIT {batch})")


def audit_sql() -> str:
    """배치 실행 사실을 감사 로그에 남기는 SQL."""
    return (
        "INSERT INTO audit_log\n"
        "  (occurred_at, actor, actor_role, action, target_table,\n"
        "   target_id, plate_masked, reason, result, client_ip, detail)\n"
        "VALUES\n"
        "  (now(), :actor, 'SYSTEM', :action, :table,\n"
        "   NULL, NULL, :reason, :result, NULL, :detail::jsonb)"
    )


def bind_named(sql: str, values: dict[str, object]) -> tuple[str, tuple]:
    """``:이름`` 형태의 명명 파라미터를 psycopg 의 ``%s`` 자리표로 바꾼다.

    문자열 연결로 값을 끼워 넣지 않는 것이 핵심이다. 값은 전부 드라이버가
    바인딩하므로 SQL 주입이 생기지 않는다. 등장 순서대로 튜플을 만든다.

    Args:
        sql: ``:cutoff`` 같은 명명 파라미터를 쓴 SQL.
        values: 이름 → 값.

    Returns:
        (변환된 SQL, 바인딩 값 튜플).
    """
    import re

    params: list[object] = []

    def repl(match: "re.Match[str]") -> str:
        name = match.group(1)
        if name not in values:
            return match.group(0)      # 모르는 이름은 그대로 둔다
        params.append(values[name])
        return "%s"

    # LIKE 'PV-%' 의 % 는 자리표가 아니므로 %% 로 escape 한 뒤 :이름 만 치환한다.
    out = re.sub(r":([a-zA-Z_][a-zA-Z0-9_]*)", repl, sql.replace("%", "%%"))
    return out, tuple(params)


def try_connect(dsn: str):
    """DB 연결을 시도한다. 드라이버나 서버가 없으면 None 을 돌려준다.

    교재 독자가 DB 를 띄우지 않고도 정책표와 SQL 을 확인할 수 있어야 하므로,
    연결 실패를 오류로 만들지 않는다.

    Args:
        dsn: PostgreSQL 접속 문자열.

    Returns:
        연결 객체 또는 None.
    """
    try:
        import psycopg
        return psycopg.connect(dsn, connect_timeout=3)
    except ImportError:
        print("[안내] psycopg 미설치 → DB 없이 정책/SQL 만 출력한다.")
        print("       pip install 'psycopg[binary]'")
        return None
    except Exception as exc:  # noqa: BLE001
        print(f"[안내] DB 연결 실패({type(exc).__name__}) "
              f"→ DB 없이 정책/SQL 만 출력한다.")
        return None


def print_policy_table(now: datetime) -> None:
    """보관기간 정책표를 출력한다."""
    print("\n[보관기간 정책]")
    print(f"  {'테이블':<20}{'기준컬럼':<14}{'기간':>7}{'처리':>11}  "
          f"{'삭제 기준 시각'}")
    for r in RULES:
        print(f"  {r.table:<20}{r.time_column:<14}{r.days:>5}일"
              f"{r.action:>11}  {cutoff_of(r, now):%Y-%m-%d}")
    print("\n  근거")
    for r in RULES:
        print(f"  - {r.table}: {r.basis}")


def run_dry(now: datetime, batch: int) -> None:
    """DB 없이 실행 계획만 출력한다."""
    print("\n[실행 계획 — dry-run]")
    for r in RULES:
        cut = cutoff_of(r, now)
        print(f"\n  ■ {r.table} ({r.action})")
        print(f"    cutoff = {cut:%Y-%m-%d %H:%M:%S}")
        print(f"    [건수 확인]")
        for line in count_sql(r).splitlines():
            print(f"      {line}")
        print(f"    [처리]")
        for line in action_sql(r, batch).splitlines():
            print(f"      {line}")
        if r.files:
            print(f"    [파일] {r.files} 컬럼의 파일을 레코드 삭제 전에 지운다")
            print(f"           (레코드를 먼저 지우면 고아 파일을 찾을 수 없다)")


def run_apply(conn, now: datetime, batch: int, actor: str,
              pepper: str) -> None:
    """실제로 삭제·비식별화를 수행한다.

    Args:
        conn: DB 연결.
        now: 기준 시각.
        batch: 1회 처리 상한.
        actor: 실행 주체(감사 로그에 기록).
        pepper: 가명처리 비밀키(환경변수에서 주입).
    """
    import json

    for r in RULES:
        cut = cutoff_of(r, now)
        with conn.cursor() as cur:
            bound, params = bind_named(count_sql(r), {"cutoff": cut})
            cur.execute(bound, params)
            total = int(cur.fetchone()[0])
        print(f"\n  ■ {r.table}: 대상 {total:,}건 (cutoff {cut:%Y-%m-%d})")
        if total == 0:
            continue
        if total > batch * SAFETY_FACTOR:
            print(f"    [중단] 대상이 배치 상한의 {SAFETY_FACTOR:.0f}배를 넘는다. "
                  f"정책 계산을 확인한 뒤 --batch 를 키워 다시 실행한다.")
            continue

        # 파일을 먼저 지운다. 레코드가 사라지면 경로를 알 수 없다.
        removed_files = 0
        if r.files:
            with conn.cursor() as cur:
                # 테이블·컬럼명은 코드 안 RULES 상수에서만 오므로 f-string 으로
                # 넣는다. 외부 입력은 값(cut)뿐이고 그것은 바인딩한다.
                cur.execute(
                    f"SELECT {r.files} FROM {r.table} "
                    f"WHERE {r.time_column} < %s AND {r.files} IS NOT NULL "
                    f"LIMIT {batch}", (cut,))
                for (path,) in cur.fetchall():
                    try:
                        os.remove(path)
                        removed_files += 1
                    except OSError:
                        pass   # 이미 없는 파일은 성공으로 본다

        sql = action_sql(r, batch)
        affected = 0
        with conn.cursor() as cur:
            for stmt in [s for s in sql.split(";") if s.strip()
                         and not s.strip().startswith("--")]:
                bound, params = bind_named(stmt, {"cutoff": cut,
                                                  "pepper": pepper})
                cur.execute(bound, params)
                affected += max(0, cur.rowcount)

        detail = json.dumps({"rule_days": r.days, "basis": r.basis,
                             "files_removed": removed_files,
                             "affected": affected,
                             "batch_limit": batch}, ensure_ascii=False)
        with conn.cursor() as cur:
            bound, params = bind_named(audit_sql(), {
                "actor": actor,
                "action": f"RETENTION_{r.action.upper()}",
                "table": r.table,
                "reason": r.basis,
                "result": "SUCCESS",
                "detail": detail,
            })
            cur.execute(bound, params)
        conn.commit()
        print(f"    처리 {affected:,}건 / 파일 {removed_files:,}개 삭제 "
              f"→ 감사 로그 기록 완료")


def main(argv: list[str] | None = None) -> int:
    """명령행 진입점."""
    ap = argparse.ArgumentParser(description="보관기간 자동 삭제 배치")
    ap.add_argument("--dsn",
                    default=os.environ.get(
                        "DB_URL",
                        "postgresql://mobility:mobility123@localhost:5432/mobility_ai"),
                    help="PostgreSQL 접속 문자열")
    ap.add_argument("--apply", action="store_true",
                    help="실제로 삭제한다(기본은 dry-run)")
    ap.add_argument("--batch", type=int, default=5000,
                    help="1회 처리 건수 상한")
    ap.add_argument("--actor", default="retention-batch",
                    help="감사 로그에 기록할 실행 주체")
    ap.add_argument("--sql", action="store_true", help="SQL 만 출력하고 종료")
    ap.add_argument("--now", default="", help="기준 시각(YYYY-MM-DD, 테스트용)")
    args = ap.parse_args(argv)

    now = (datetime.strptime(args.now, "%Y-%m-%d")
           if args.now else datetime.now())

    print("=" * 74)
    print(f"보관기간 자동 삭제 배치 — 기준 시각 {now:%Y-%m-%d %H:%M:%S}")
    print("=" * 74)
    print_policy_table(now)

    if args.sql:
        run_dry(now, args.batch)
        return 0

    if not args.apply:
        run_dry(now, args.batch)
        print("\n[안내] dry-run 이다. 실제로 지우려면 --apply 를 준다.")
        print("       운영에서는 --apply 실행을 승인 절차와 묶고,")
        print("       cron 또는 스프링 @Scheduled 로 매일 03:00 에 돌린다.")
        return 0

    pepper = os.environ.get("PLATE_PEPPER", "")
    if not pepper:
        print("\n[중단] --apply 에는 PLATE_PEPPER 환경변수가 필요하다.")
        print("       비밀키 없이 가명처리하면 역추적이 가능해 조치로 인정받기 어렵다.")
        return 1

    conn = try_connect(args.dsn)
    if conn is None:
        run_dry(now, args.batch)
        return 0
    try:
        print("\n[실행 — apply]")
        run_apply(conn, now, args.batch, args.actor, pepper)
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
