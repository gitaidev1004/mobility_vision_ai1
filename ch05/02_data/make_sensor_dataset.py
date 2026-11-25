# -*- coding: utf-8 -*-
"""차량 센서 시계열 합성 데이터 생성기.

CONTRACT.md 의 `sensor_reading` 테이블 컬럼과 동일한 이름/단위로
현실적인 차량 센서 시계열을 만든다. 세 가지 시나리오를 섞어 만든다.

  normal   : 정상 차량. 모든 신호가 운용 범위 안에서 변동한다.
  degrade  : 완만한 열화. 배터리 SOH / 브레이크 패드 / 오일 품질이 서서히 나빠진다.
  failure  : 급성 고장. 열화 후반에 과열·저전압·진동 급증이 함께 나타난다.

생성 결과를 `data/sensor_readings.csv` 로 저장한다.
numpy / pandas 만 사용한다.
"""
from __future__ import annotations

import argparse
import pathlib
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# sensor_reading 테이블과 동일한 컬럼 순서 (id 는 DB 가 부여)
COLUMNS = [
    "vehicle_id", "measured_at", "odometer_km",
    "engine_temp_c", "engine_rpm", "speed_kph",
    "battery_voltage", "battery_soh",
    "tire_pressure_fl", "tire_pressure_fr",
    "tire_pressure_rl", "tire_pressure_rr",
    "brake_pad_mm", "oil_quality", "vibration_rms", "fuel_level",
]

SCENARIOS = ("normal", "degrade", "failure")


def _drive_cycle(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """주행 사이클(속도/회전수)을 만든다.

    도심 정차 - 가속 - 정속 - 감속이 반복되는 형태를 블록 단위로 합성한다.
    """
    speed = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        mode = rng.choice(["idle", "city", "highway"], p=[0.22, 0.48, 0.30])
        if mode == "idle":
            seg = rng.integers(4, 14)
            target = 0.0
        elif mode == "city":
            seg = rng.integers(10, 40)
            target = rng.uniform(25, 60)
        else:
            seg = rng.integers(25, 80)
            target = rng.uniform(80, 108)
        seg = int(min(seg, n - i))
        # 목표 속도로 1차 지연 수렴 + 잡음
        prev = speed[i - 1] if i > 0 else 0.0
        for k in range(seg):
            prev += (target - prev) * 0.25 + rng.normal(0, 1.1)
            speed[i + k] = max(0.0, prev)
        i += seg

    # 회전수: 아이들 750rpm + 속도 비례 + 변속에 따른 산포
    rpm = 750.0 + speed * 26.0 + rng.normal(0, 70, n)
    rpm = np.where(speed < 1.0, 700 + rng.normal(0, 25, n), rpm)
    rpm = np.clip(rpm, 600, 5200)
    return speed, rpm


def _coolant_temp(speed: np.ndarray, rpm: np.ndarray,
                  rng: np.random.Generator,
                  overheat_gain: np.ndarray) -> np.ndarray:
    """냉각수 온도. 시동 후 웜업하고 부하에 따라 완만히 오른다."""
    n = len(speed)
    temp = np.zeros(n, dtype=float)
    t = 24.0                                   # 시동 시 외기 온도
    load = rpm / 5200.0
    for k in range(n):
        # 목표 온도 90도 + 부하 가산, 열화 차량은 overheat_gain 만큼 더 오른다
        target = 90.0 + 9.0 * load[k] + overheat_gain[k]
        # 저속일 때 냉각 효율이 떨어져 수렴이 느리다
        tau = 0.035 if speed[k] > 20 else 0.022
        t += (target - t) * tau + rng.normal(0, 0.18)
        temp[k] = t
    return temp


def generate_vehicle(vehicle_id: int, scenario: str, n: int,
                     start: datetime, interval_min: int,
                     odo_start: float, rng: np.random.Generator) -> pd.DataFrame:
    """차량 한 대의 센서 시계열을 만든다."""
    if scenario not in SCENARIOS:
        raise ValueError(f"알 수 없는 시나리오: {scenario}")

    # 열화 진행도 0 -> 1
    prog = np.linspace(0.0, 1.0, n)

    speed, rpm = _drive_cycle(n, rng)

    # 시나리오별 열화 프로파일 --------------------------------------
    if scenario == "normal":
        overheat = np.zeros(n)
        soh = 0.97 - 0.02 * prog + rng.normal(0, 0.002, n)
        pad = 9.2 - 1.1 * prog + rng.normal(0, 0.03, n)
        oil = 0.93 - 0.18 * prog + rng.normal(0, 0.006, n)
        volt_base = 14.1 - 0.05 * prog
        vib_base = 0.22 + 0.03 * prog
    elif scenario == "degrade":
        # 2차 곡선으로 가속되는 열화
        overheat = 6.5 * prog ** 2
        soh = 0.95 - 0.19 * prog ** 1.4 + rng.normal(0, 0.003, n)
        pad = 7.4 - 4.3 * prog + rng.normal(0, 0.04, n)
        oil = 0.88 - 0.52 * prog ** 1.2 + rng.normal(0, 0.009, n)
        volt_base = 14.0 - 1.15 * prog ** 1.5
        vib_base = 0.24 + 0.33 * prog ** 1.6
    else:   # failure
        # 85% 지점 이후 급격한 악화 (지수적 붕괴)
        knee = 0.82
        burst = np.clip((prog - knee) / (1 - knee), 0, 1)
        overheat = 5.0 * prog ** 2 + 16.0 * burst ** 2
        soh = 0.93 - 0.17 * prog - 0.22 * burst ** 1.5 + rng.normal(0, 0.004, n)
        pad = 6.1 - 3.6 * prog - 1.6 * burst + rng.normal(0, 0.05, n)
        oil = 0.84 - 0.46 * prog - 0.26 * burst + rng.normal(0, 0.01, n)
        volt_base = 13.9 - 0.9 * prog - 2.3 * burst ** 1.3
        vib_base = 0.25 + 0.30 * prog + 1.35 * burst ** 1.8

    temp = _coolant_temp(speed, rpm, rng, overheat)

    # 배터리 전압: 시동 중에는 충전 전압, 정차 시 약간 하강
    volt = volt_base + np.where(speed > 3, 0.0, -0.55) + rng.normal(0, 0.055, n)

    # 타이어 공기압(psi): 주행으로 온도가 오르면 약간 상승, 좌우 편차 존재
    base_psi = 32.4
    heat = 0.9 * (temp - 24.0) / 80.0
    psi_fl = base_psi + heat - 1.9 * prog * (scenario != "normal") + rng.normal(0, 0.16, n)
    psi_fr = base_psi + heat - 0.5 * prog * (scenario != "normal") + rng.normal(0, 0.16, n)
    psi_rl = base_psi + heat - 0.3 * prog + rng.normal(0, 0.16, n)
    psi_rr = base_psi + heat - 0.35 * prog + rng.normal(0, 0.16, n)

    # 진동 RMS: 속도와 노면 거칠기 + 열화 성분
    vib = vib_base + 0.0021 * speed + np.abs(rng.normal(0, 0.035, n))

    # 연료 잔량: 주행 거리에 비례해 소모, 중간에 한 번 주유
    used = np.cumsum(speed) / max(np.sum(speed), 1.0)
    fuel = 0.92 - 0.80 * used
    refuel_at = n // 2
    fuel[refuel_at:] += 0.58
    fuel = np.clip(fuel + rng.normal(0, 0.004, n), 0.03, 1.0)

    # 주행거리 누적(속도 * 구간 시간)
    dkm = speed * (interval_min / 60.0)
    odo = odo_start + np.cumsum(dkm)

    ts = [start + timedelta(minutes=interval_min * k) for k in range(n)]

    df = pd.DataFrame({
        "vehicle_id": vehicle_id,
        "measured_at": [t.strftime("%Y-%m-%d %H:%M:%S") for t in ts],
        "odometer_km": np.round(odo, 1),
        "engine_temp_c": np.round(temp, 2),
        "engine_rpm": np.round(rpm).astype(int),
        "speed_kph": np.round(speed, 1),
        "battery_voltage": np.round(np.clip(volt, 8.0, 15.5), 3),
        "battery_soh": np.round(np.clip(soh, 0.30, 1.0), 4),
        "tire_pressure_fl": np.round(np.clip(psi_fl, 12.0, 45.0), 2),
        "tire_pressure_fr": np.round(np.clip(psi_fr, 12.0, 45.0), 2),
        "tire_pressure_rl": np.round(np.clip(psi_rl, 12.0, 45.0), 2),
        "tire_pressure_rr": np.round(np.clip(psi_rr, 12.0, 45.0), 2),
        "brake_pad_mm": np.round(np.clip(pad, 0.5, 12.0), 2),
        "oil_quality": np.round(np.clip(oil, 0.02, 1.0), 4),
        "vibration_rms": np.round(np.clip(vib, 0.0, 6.0), 4),
        "fuel_level": np.round(fuel, 4),
    }, columns=COLUMNS)

    df.attrs["scenario"] = scenario
    return df


def inject_defects(df: pd.DataFrame, rng: np.random.Generator,
                   missing_rate: float = 0.02,
                   spike_rate: float = 0.008) -> pd.DataFrame:
    """현실의 수집 결함(결측·스파이크)을 주입한다.

    실제 OBD 로거는 통신 타임아웃으로 값이 비거나, 배선 접촉 불량으로
    물리적으로 불가능한 스파이크를 기록한다. 5.4 전처리 실습의 입력이 된다.
    """
    out = df.copy()
    num_cols = ["engine_temp_c", "engine_rpm", "speed_kph",
                "battery_voltage", "vibration_rms", "oil_quality"]

    n = len(out)
    for c in num_cols:
        # 결측: 연속 1~3개 구간으로 비운다(통신 단절 모사)
        n_miss = int(n * missing_rate)
        starts = rng.integers(0, n, size=max(1, n_miss // 2))
        for s in starts:
            ln = int(rng.integers(1, 4))
            out.loc[s:s + ln - 1, c] = np.nan
        # 스파이크: 정상 대비 3~8배 또는 음수
        n_sp = max(1, int(n * spike_rate))
        idx = rng.choice(n, size=n_sp, replace=False)
        for i in idx:
            if pd.isna(out.at[i, c]):
                continue
            if rng.random() < 0.5:
                out.at[i, c] = out.at[i, c] * rng.uniform(3.0, 8.0)
            else:
                out.at[i, c] = -abs(out.at[i, c]) * rng.uniform(0.5, 2.0)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="차량 센서 시계열 합성 생성기")
    ap.add_argument("--out", default="data/sensor_readings.csv",
                    help="저장할 CSV 경로")
    ap.add_argument("--rows", type=int, default=720,
                    help="차량 1대당 행 수(기본 720 = 10분 간격 5일)")
    ap.add_argument("--interval", type=int, default=10,
                    help="측정 간격(분)")
    ap.add_argument("--seed", type=int, default=20261024, help="난수 시드")
    ap.add_argument("--defects", action="store_true",
                    help="결측/스파이크를 주입한다(전처리 실습용)")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    start = datetime(2026, 10, 1, 7, 0, 0)

    plan = [
        (1, "normal", 84210.0),
        (2, "normal", 41780.0),
        (3, "degrade", 132450.0),
        (4, "failure", 168930.0),
    ]

    frames = []
    for vid, scen, odo in plan:
        df = generate_vehicle(vid, scen, args.rows, start,
                              args.interval, odo, rng)
        if args.defects:
            df = inject_defects(df, rng)
        frames.append(df)
        print(f"vehicle_id={vid} scenario={scen:<8} rows={len(df)} "
              f"odo {df['odometer_km'].iloc[0]:,.0f} -> "
              f"{df['odometer_km'].iloc[-1]:,.0f} km")

    all_df = pd.concat(frames, ignore_index=True)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    all_df.to_csv(out, index=False, encoding="utf-8")

    print(f"\n저장 완료: {out}  총 {len(all_df):,}행 x {len(COLUMNS)}열")
    print("\n[시나리오별 주요 지표 요약]")
    print(f"{'vid':>4}{'시나리오':>10}{'온도최대':>10}{'전압최소':>10}"
          f"{'SOH최소':>10}{'패드최소':>10}{'진동최대':>10}")
    for vid, scen, _ in plan:
        g = all_df[all_df["vehicle_id"] == vid]
        print(f"{vid:>4}{scen:>10}{g['engine_temp_c'].max():>10.1f}"
              f"{g['battery_voltage'].min():>10.2f}"
              f"{g['battery_soh'].min():>10.3f}"
              f"{g['brake_pad_mm'].min():>10.2f}"
              f"{g['vibration_rms'].max():>10.3f}")

    print("\n[앞 3행 미리보기]")
    cols = ["vehicle_id", "measured_at", "engine_temp_c", "engine_rpm",
            "speed_kph", "battery_voltage", "vibration_rms"]
    print(all_df[cols].head(3).to_string(index=False))


if __name__ == "__main__":
    main()
