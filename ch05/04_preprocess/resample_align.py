# -*- coding: utf-8 -*-
"""비동기 센서들의 시간축을 정렬하고 리샘플링한다.

실제 차량에서는 센서마다 전송 주기가 다르다.
  엔진 신호   : 10~20ms  (CAN 고속 버스)
  타이어 공기압: 60초     (TPMS 무선)
  배터리 SOH  : 5분      (BMS 집계값)
이 상태로는 행렬 하나를 만들 수 없으므로 공통 격자(grid)에 맞춰야 한다.

세 가지 정렬 전략을 구현한다.
  1. resample : 공통 주기 격자로 집계(평균/최대/마지막) 후 병합
  2. asof     : 기준 센서 시각에 가장 가까운 과거 값을 끌어온다(merge_asof)
  3. union    : 모든 시각의 합집합 위에 전방 채움

numpy / pandas 만 사용한다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# 컬럼별 리샘플 집계 방식. 물리 의미에 맞춰 다르게 준다.
AGG_RULE: dict[str, str] = {
    "odometer_km": "max",        # 누적값은 구간 최댓값
    "engine_temp_c": "mean",     # 연속 상태량은 평균
    "engine_rpm": "mean",
    "speed_kph": "mean",
    "battery_voltage": "mean",
    "battery_soh": "last",       # 느리게 변하는 추정치는 마지막 값
    "tire_pressure_fl": "mean",
    "tire_pressure_fr": "mean",
    "tire_pressure_rl": "mean",
    "tire_pressure_rr": "mean",
    "brake_pad_mm": "last",      # 단조 감소 소모품은 마지막 값
    "oil_quality": "last",
    "vibration_rms": "max",      # 충격은 평균이 아니라 최댓값이 중요
    "fuel_level": "last",
}


def make_async_streams(seed: int = 20261024) -> dict[str, pd.DataFrame]:
    """서로 다른 주기를 가진 센서 스트림 3개를 만든다."""
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp("2026-10-01 09:00:00")

    # 1) 엔진 스트림: 2초 주기, 600개 (20분)
    n1 = 600
    ts1 = t0 + pd.to_timedelta(np.arange(n1) * 2, unit="s")
    speed = np.clip(45 + 18 * np.sin(np.arange(n1) / 42.0)
                    + rng.normal(0, 2.4, n1), 0, None)
    eng = pd.DataFrame({
        "measured_at": ts1,
        "engine_temp_c": np.round(91 + 3.2 * np.sin(np.arange(n1) / 85.0)
                                  + rng.normal(0, 0.35, n1), 2),
        "engine_rpm": np.round(750 + speed * 26 + rng.normal(0, 60, n1)
                               ).astype(int),
        "speed_kph": np.round(speed, 1),
        "vibration_rms": np.round(0.24 + 0.0022 * speed
                                  + np.abs(rng.normal(0, 0.04, n1)), 4),
    })

    # 2) TPMS 스트림: 60초 주기 + 지터(무선이라 시각이 어긋난다), 20개
    n2 = 20
    jitter = rng.integers(-7, 8, n2)
    ts2 = t0 + pd.to_timedelta(np.arange(n2) * 60 + jitter, unit="s")
    base = 32.3 - np.arange(n2) * 0.012
    tpms = pd.DataFrame({
        "measured_at": ts2,
        "tire_pressure_fl": np.round(base - 0.9 + rng.normal(0, 0.1, n2), 2),
        "tire_pressure_fr": np.round(base + rng.normal(0, 0.1, n2), 2),
        "tire_pressure_rl": np.round(base - 0.4 + rng.normal(0, 0.1, n2), 2),
        "tire_pressure_rr": np.round(base - 0.3 + rng.normal(0, 0.1, n2), 2),
    })

    # 3) BMS 스트림: 300초(5분) 주기, 5개
    n3 = 5
    ts3 = t0 + pd.to_timedelta(np.arange(n3) * 300, unit="s")
    bms = pd.DataFrame({
        "measured_at": ts3,
        "battery_voltage": np.round(14.08 - np.arange(n3) * 0.015
                                    + rng.normal(0, 0.02, n3), 3),
        "battery_soh": np.round(0.862 - np.arange(n3) * 0.0004, 4),
    })

    return {"engine": eng, "tpms": tpms, "bms": bms}


def resample_stream(df: pd.DataFrame, rule: str = "10s",
                    time_col: str = "measured_at") -> pd.DataFrame:
    """스트림 하나를 공통 주기 격자로 집계한다."""
    d = df.copy()
    d[time_col] = pd.to_datetime(d[time_col])
    d = d.set_index(time_col).sort_index()
    agg = {c: AGG_RULE.get(c, "mean") for c in d.columns
           if pd.api.types.is_numeric_dtype(d[c])}
    out = d.resample(rule).agg(agg)
    return out


def align_by_resample(streams: dict[str, pd.DataFrame],
                      rule: str = "10s") -> pd.DataFrame:
    """전략 1: 모든 스트림을 같은 격자로 리샘플한 뒤 옆으로 붙인다.

    격자 주기보다 느린 센서는 격자 대부분이 비므로 전방 채움으로 유지한다.
    (물리적으로 타이어 공기압은 1초 만에 바뀌지 않으므로 타당하다)
    """
    parts = [resample_stream(df, rule) for df in streams.values()]
    merged = pd.concat(parts, axis=1).sort_index()
    # 느린 센서의 빈 격자 채움
    slow = [c for c in merged.columns
            if c.startswith(("tire_", "battery_"))]
    merged[slow] = merged[slow].ffill().bfill()
    # 빠른 센서의 짧은 공백만 보간
    fast = [c for c in merged.columns if c not in slow]
    merged[fast] = merged[fast].interpolate(limit=3, limit_area="inside")
    return merged


def align_by_asof(streams: dict[str, pd.DataFrame],
                  base: str = "engine",
                  tolerance: str = "180s") -> pd.DataFrame:
    """전략 2: 기준 스트림의 시각마다 가장 가까운 과거 값을 끌어온다.

    merge_asof 는 '미래 값을 쓰지 않는다'는 점이 중요하다.
    실시간 추론과 학습의 조건이 같아져 데이터 누수(leakage)를 막는다.
    """
    out = streams[base].copy()
    out["measured_at"] = pd.to_datetime(out["measured_at"])
    out = out.sort_values("measured_at")
    for name, df in streams.items():
        if name == base:
            continue
        right = df.copy()
        right["measured_at"] = pd.to_datetime(right["measured_at"])
        right = right.sort_values("measured_at")
        out = pd.merge_asof(out, right, on="measured_at",
                            direction="backward",
                            tolerance=pd.Timedelta(tolerance))
    return out.set_index("measured_at")


def align_by_union(streams: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """전략 3: 모든 시각의 합집합을 인덱스로 쓰고 전방 채움한다.

    원본 시각을 하나도 버리지 않지만 행 수가 늘고 NaN 패턴이 복잡해진다.
    """
    frames = []
    for df in streams.values():
        d = df.copy()
        d["measured_at"] = pd.to_datetime(d["measured_at"])
        frames.append(d.set_index("measured_at"))
    idx = frames[0].index
    for f in frames[1:]:
        idx = idx.union(f.index)
    out = pd.concat([f.reindex(idx) for f in frames], axis=1)
    return out.ffill()


def report(name: str, df: pd.DataFrame) -> None:
    """정렬 결과의 형태와 결측률을 출력한다."""
    na = df.isna().sum().sum()
    cells = df.size
    span = df.index.max() - df.index.min()
    print(f"\n[{name}]  {len(df):,}행 x {len(df.columns)}열  "
          f"구간 {span}  결측 {na}/{cells:,} ({na / max(cells, 1) * 100:.2f}%)")
    cols = ["engine_temp_c", "speed_kph", "tire_pressure_fl",
            "battery_voltage"]
    cols = [c for c in cols if c in df.columns]
    print(df[cols].head(4).to_string())


def main() -> None:
    streams = make_async_streams()

    print("[원본 비동기 스트림]")
    print(f"{'스트림':<10}{'행':>7}{'열':>5}{'평균주기':>12}{'첫 시각':>22}")
    for name, df in streams.items():
        t = pd.to_datetime(df["measured_at"])
        period = (t.iloc[-1] - t.iloc[0]).total_seconds() / max(len(t) - 1, 1)
        print(f"{name:<10}{len(df):>7}{len(df.columns) - 1:>5}"
              f"{period:>10.1f}s{str(t.iloc[0]):>22}")

    a = align_by_resample(streams, rule="10s")
    report("전략1 resample(10초 격자)", a)

    b = align_by_asof(streams, base="engine", tolerance="180s")
    report("전략2 merge_asof(엔진 기준, 허용 180초)", b)

    c = align_by_union(streams)
    report("전략3 union index + ffill", c)

    print("\n[리샘플 주기별 행 수와 정보 손실]")
    print(f"  {'주기':<8}{'행 수':>8}{'온도 표준편차':>14}"
          f"{'온도 차분 표준편차':>18}{'진동 최대':>12}")
    for rule in ["2s", "10s", "30s", "60s", "300s"]:
        r = align_by_resample(streams, rule=rule)
        print(f"  {rule:<8}{len(r):>8}{r['engine_temp_c'].std():>14.4f}"
              f"{r['engine_temp_c'].diff().std():>18.4f}"
              f"{r['vibration_rms'].max():>12.4f}")
    print("\n  주기를 늘려도 저주파 성분(온도의 완만한 변동)은 남으므로")
    print("  온도 표준편차는 2.1~2.2 수준으로 거의 변하지 않는다.")
    print("  차분 표준편차는 2초 -> 10초 구간에서 평균 집계로 고주파")
    print("  잡음이 지워져 0.52 -> 0.24 로 줄지만, 60초를 넘기면 이제")
    print("  차분이 실제 온도 변화를 담게 되어 다시 커진다.")
    print("  진동은 max 집계라 어느 주기에서도 피크값이 보존된다.")
    print("  즉 집계 방식 선택이 곧 '어떤 정보를 지킬지'의 선택이다.")


if __name__ == "__main__":
    main()
