# -*- coding: utf-8 -*-
"""센서 CSV 를 PostgreSQL `sensor_reading` 테이블에 적재한다.

make_sensor_dataset.py 가 만든 CSV 를 그대로 받는다.
`psycopg` (psycopg3) 가 없으면 설치 안내를 출력하고 정상 종료한다.
--dry-run 을 주면 DB 접속 없이 CSV 검증만 수행한다.

실행:
  python 03_schema/load_csv_to_db.py --csv data/sensor_readings.csv --dry-run
  python 03_schema/load_csv_to_db.py --csv data/sensor_readings.csv
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import sys

# sensor_reading 에 적재할 컬럼 (CONTRACT.md 와 동일)
COLUMNS = [
    "vehicle_id", "measured_at", "odometer_km",
    "engine_temp_c", "engine_rpm", "speed_kph",
    "battery_voltage", "battery_soh",
    "tire_pressure_fl", "tire_pressure_fr",
    "tire_pressure_rl", "tire_pressure_rr",
    "brake_pad_mm", "oil_quality", "vibration_rms", "fuel_level",
]

# 정수로 넣어야 하는 컬럼
INT_COLS = {"vehicle_id", "odometer_km", "engine_rpm"}

INSERT_SQL = f"""
INSERT INTO sensor_reading ({", ".join(COLUMNS)})
VALUES ({", ".join(["%s"] * len(COLUMNS))})
ON CONFLICT (vehicle_id, measured_at) DO NOTHING
"""


def coerce(col: str, raw: str):
    """CSV 문자열을 DB 타입으로 변환한다. 빈 값은 NULL(None) 로 둔다."""
    s = (raw or "").strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return None
    if col == "measured_at":
        return s
    try:
        if col in INT_COLS:
            return int(round(float(s)))
        return float(s)
    except ValueError:
        return None


def read_rows(csv_path: pathlib.Path) -> tuple[list[tuple], list[str]]:
    """CSV 를 읽어 INSERT 파라미터 튜플 목록과 경고 목록을 만든다."""
    rows: list[tuple] = []
    warns: list[str] = []
    with csv_path.open(encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        missing = [c for c in COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise SystemExit(f"CSV 에 다음 컬럼이 없다: {missing}")
        for i, rec in enumerate(reader, start=2):
            vals = tuple(coerce(c, rec.get(c, "")) for c in COLUMNS)
            if vals[0] is None or vals[1] is None:
                warns.append(f"{i}행: vehicle_id/measured_at 누락 -> 건너뜀")
                continue
            # 물리 범위를 벗어난 값은 DB CHECK 제약에서 막히므로 미리 걸러낸다.
            temp = vals[COLUMNS.index("engine_temp_c")]
            if temp is not None and not (-40 <= temp <= 215):
                warns.append(f"{i}행: engine_temp_c={temp} 범위 밖 -> NULL 처리")
                vals = list(vals)
                vals[COLUMNS.index("engine_temp_c")] = None
                vals = tuple(vals)
            rows.append(vals)
    return rows, warns


def main() -> int:
    ap = argparse.ArgumentParser(description="센서 CSV → PostgreSQL 적재")
    ap.add_argument("--csv", default="data/sensor_readings.csv")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=5432)
    ap.add_argument("--db", default="mobility_ai")
    ap.add_argument("--user", default="mobility")
    ap.add_argument("--password", default="mobility123")
    ap.add_argument("--batch", type=int, default=500, help="커밋 배치 크기")
    ap.add_argument("--dry-run", action="store_true",
                    help="DB 접속 없이 CSV 검증만 수행한다")
    args = ap.parse_args()

    csv_path = pathlib.Path(args.csv)
    if not csv_path.exists():
        print(f"[오류] CSV 파일이 없다: {csv_path}")
        print("  먼저 다음을 실행해 데이터를 만든다.")
        print("  python 02_data/make_sensor_dataset.py --out data/sensor_readings.csv")
        return 0

    rows, warns = read_rows(csv_path)
    print(f"CSV 읽기 완료: {csv_path}  유효 {len(rows):,}행")
    for w in warns[:10]:
        print("  경고:", w)
    if len(warns) > 10:
        print(f"  ... 경고 {len(warns)}건 중 10건만 표시")

    if args.dry_run:
        print("\n[--dry-run] DB 접속을 생략한다. 적재 SQL 미리보기:")
        print(INSERT_SQL.strip())
        print("\n첫 행 파라미터:")
        for c, v in zip(COLUMNS, rows[0]):
            print(f"  {c:<18} = {v!r}")
        return 0

    try:
        import psycopg                     # psycopg3
    except ImportError:
        print("\n[안내] psycopg 패키지가 설치되어 있지 않다.")
        print("  아래 명령으로 설치한 뒤 다시 실행한다.")
        print("    pip install \"psycopg[binary]\"")
        print("  DB 없이 CSV 검증만 하려면 --dry-run 을 사용한다.")
        return 0

    dsn = (f"host={args.host} port={args.port} dbname={args.db} "
           f"user={args.user} password={args.password}")
    try:
        with psycopg.connect(dsn) as conn:
            with conn.cursor() as cur:
                inserted = 0
                for s in range(0, len(rows), args.batch):
                    chunk = rows[s:s + args.batch]
                    cur.executemany(INSERT_SQL, chunk)
                    inserted += len(chunk)
                    print(f"  적재 {inserted:,}/{len(rows):,}", end="\r")
                conn.commit()
                cur.execute("SELECT COUNT(*) FROM sensor_reading")
                total = cur.fetchone()[0]
        print(f"\n적재 완료. sensor_reading 전체 {total:,}행")
    except Exception as exc:                      # noqa: BLE001
        print(f"\n[오류] DB 작업 실패: {exc}")
        print("  PostgreSQL 이 기동 중인지, schema_pdm.sql 을 먼저 실행했는지 확인한다.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
