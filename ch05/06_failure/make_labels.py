# -*- coding: utf-8 -*-
"""고장 라벨 생성 — 향후 N일 내 고장 여부(이진) + 고장 부품(다중 클래스).

센서 이력만으로는 지도학습을 할 수 없다. 정비 이력(maintenance_record)에
기록된 '고장으로 인한 교체' 시점을 기준으로, 각 센서 측정 행에
 - will_fail_7  : 측정 시점부터 7일 이내에 고장이 발생하는가 (0/1)
 - will_fail_30 : 30일 이내 고장 여부 (0/1)
 - fail_part    : 0=NONE 1=engine 2=battery 3=brake 4=tire 5=cooling
를 붙인다. 이 파일은 numpy/pandas 만으로 동작한다.

실행:  python 06_failure/make_labels.py
출력:  data/labeled_dataset.csv
"""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

RNG = np.random.default_rng(20261024)       # 결과 재현을 위한 고정 시드
OUT_DIR = pathlib.Path("data")

# 부품 다중 클래스 라벨 정의(0 은 '정상' 클래스)
PART_CLASSES = ["NONE", "engine", "battery", "brake", "tire", "cooling"]
HORIZONS = (7, 30)                          # 예측 지평(일 단위)


def make_vehicle_history(vehicle_id: int, days: int = 240) -> pd.DataFrame:
    """차량 한 대의 일별 센서 이력과 고장 발생일을 생성한다.

    열화는 '서서히 진행되다 임계를 넘으면 고장'이라는 가정을 따른다.
    부품마다 열화 속도를 달리 두어 같은 기간에도 서로 다른 시점에
    고장이 나도록 만든다.
    """
    t = np.arange(days)

    # 차량별로 조금씩 다른 초기 상태와 열화 속도를 뽑는다.
    start_odo = int(RNG.integers(20_000, 120_000))
    daily_km = float(RNG.uniform(25.0, 95.0))
    odometer = start_odo + np.cumsum(np.full(days, daily_km))

    # 열화 대상 부품을 한 개 고른다(정상 유지 차량도 존재).
    part_idx = int(RNG.integers(0, len(PART_CLASSES)))
    # 고장이 실제로 발생하는 날(없으면 -1)
    if part_idx == 0:
        fail_day = -1
    else:
        fail_day = int(RNG.integers(int(days * 0.55), days))

    # 열화 진행도: 고장일에 1.0 이 되도록 선형 증가, 정상 차량은 0 근처
    if fail_day > 0:
        prog = np.clip(t / float(fail_day), 0.0, 1.3)
        prog = prog ** 2.2                   # 후반에 급격히 악화되는 비선형 열화
    else:
        prog = np.zeros(days)

    # 기본 센서 신호(정상 범위) — 5.2절에서 정의한 sensor_reading 열과 동일
    engine_temp = 88.0 + 2.0 * np.sin(t / 11.0) + RNG.normal(0, 0.8, days)
    engine_rpm = 1750.0 + 60.0 * np.sin(t / 7.0) + RNG.normal(0, 25, days)
    speed = 46.0 + RNG.normal(0, 3.0, days)
    battery_v = 12.55 - 0.0004 * t + RNG.normal(0, 0.03, days)
    battery_soh = 0.98 - 0.00035 * t + RNG.normal(0, 0.002, days)
    tire_fl = 33.0 - 0.004 * t + RNG.normal(0, 0.25, days)
    tire_fr = 33.0 - 0.004 * t + RNG.normal(0, 0.25, days)
    tire_rl = 32.6 - 0.004 * t + RNG.normal(0, 0.25, days)
    tire_rr = 32.6 - 0.004 * t + RNG.normal(0, 0.25, days)
    brake_pad = 9.5 - 0.012 * t + RNG.normal(0, 0.06, days)
    oil_quality = 0.95 - 0.0016 * t + RNG.normal(0, 0.01, days)
    vibration = 0.30 + RNG.normal(0, 0.02, days)

    # 열화 부품별로 해당 센서에만 추세를 더한다(물리적 인과를 모방).
    part = PART_CLASSES[part_idx]
    if part == "engine":
        engine_temp += 14.0 * prog
        vibration += 0.55 * prog
        oil_quality -= 0.35 * prog
    elif part == "battery":
        battery_v -= 1.4 * prog
        battery_soh -= 0.28 * prog
    elif part == "brake":
        brake_pad -= 5.5 * prog
        vibration += 0.20 * prog
    elif part == "tire":
        tire_rl -= 7.0 * prog
        tire_rr -= 5.0 * prog
        vibration += 0.30 * prog
    elif part == "cooling":
        engine_temp += 20.0 * prog
        engine_rpm += 180.0 * prog

    df = pd.DataFrame({
        "vehicle_id": vehicle_id,
        "day": t,
        "odometer_km": np.round(odometer, 0),
        "engine_temp_c": np.round(engine_temp, 2),
        "engine_rpm": np.round(engine_rpm, 0),
        "speed_kph": np.round(speed, 1),
        "battery_voltage": np.round(battery_v, 3),
        "battery_soh": np.round(np.clip(battery_soh, 0.3, 1.0), 4),
        "tire_pressure_fl": np.round(tire_fl, 2),
        "tire_pressure_fr": np.round(tire_fr, 2),
        "tire_pressure_rl": np.round(np.clip(tire_rl, 8.0, None), 2),
        "tire_pressure_rr": np.round(np.clip(tire_rr, 8.0, None), 2),
        "brake_pad_mm": np.round(np.clip(brake_pad, 0.5, None), 2),
        "oil_quality": np.round(np.clip(oil_quality, 0.02, 1.0), 3),
        "vibration_rms": np.round(np.clip(vibration, 0.02, None), 3),
        "fuel_level": np.round(RNG.uniform(0.15, 0.95, days), 2),
    })
    df["fail_day"] = fail_day
    df["fail_part_name"] = part if fail_day > 0 else "NONE"
    return df


def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    """차량 단위로 이진 라벨과 다중 클래스 라벨을 부여한다."""
    out = []
    for vid, g in df.groupby("vehicle_id", sort=True):
        g = g.sort_values("day").copy()
        fail_day = int(g["fail_day"].iloc[0])
        part_name = str(g["fail_part_name"].iloc[0])

        for h in HORIZONS:
            if fail_day > 0:
                # 측정 시점 d 에서 (fail_day - d) 가 0 이상 h 이하면 양성
                lead = fail_day - g["day"].to_numpy()
                y = ((lead >= 0) & (lead <= h)).astype(np.int8)
            else:
                y = np.zeros(len(g), dtype=np.int8)
            g[f"will_fail_{h}"] = y

        # 다중 클래스: 30일 창 안에서는 해당 부품, 그 밖은 NONE(0)
        cls = np.zeros(len(g), dtype=np.int8)
        if fail_day > 0:
            idx = PART_CLASSES.index(part_name)
            cls = np.where(g["will_fail_30"].to_numpy() == 1, idx, 0).astype(np.int8)
        g["fail_part"] = cls

        # 고장 발생 이후 구간은 '정비 완료' 로 간주해 학습에서 제외한다.
        if fail_day > 0:
            g = g[g["day"] <= fail_day]
        out.append(g)
    return pd.concat(out, ignore_index=True)


def summarize(df: pd.DataFrame) -> None:
    """라벨 분포를 출력해 클래스 불균형 정도를 확인한다."""
    n = len(df)
    print(f"전체 행 수            : {n}")
    print(f"차량 수               : {df['vehicle_id'].nunique()}")
    for h in HORIZONS:
        pos = int(df[f"will_fail_{h}"].sum())
        print(f"will_fail_{h:<2d} 양성      : {pos:5d} ({pos / n * 100:5.2f}%)")
    print("\n[다중 클래스 분포]")
    vc = df["fail_part"].value_counts().sort_index()
    for k, v in vc.items():
        print(f"  {int(k)} {PART_CLASSES[int(k)]:<8s} {v:5d} ({v / n * 100:5.2f}%)")


def main() -> None:
    frames = [make_vehicle_history(vid) for vid in range(1, 41)]
    raw = pd.concat(frames, ignore_index=True)
    labeled = add_labels(raw)

    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / "labeled_dataset.csv"
    labeled.drop(columns=["fail_day"]).to_csv(out_path, index=False)

    summarize(labeled)
    print(f"\n저장 완료: {out_path}")


if __name__ == "__main__":
    main()
