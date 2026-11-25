# -*- coding: utf-8 -*-
"""수명 감소 곡선 생성과 piecewise-linear RUL 라벨링.

RUL(Remaining Useful Life) 은 '지금부터 고장까지 남은 수명'이다.
관측 초기에는 열화가 거의 없어 RUL 을 그대로 쓰면 모델이 쓸모없이 큰
값을 외우게 된다. 그래서 상한 R_early 를 두고 그보다 큰 구간은 모두
R_early 로 눌러 놓는 piecewise-linear RUL 을 표준으로 쓴다.

실행:  python 08_rul/rul_curve.py
출력:  data/rul_dataset.csv
"""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

RNG = np.random.default_rng(880224)
OUT_DIR = pathlib.Path("data")
R_EARLY = 60          # RUL 상한(일). 이보다 먼 미래는 구분하지 않는다.


def degradation_curve(life: int, mode: str) -> np.ndarray:
    """0(신품)에서 1(고장)까지 가는 열화 지표 H(t) 를 만든다.

    mode
      linear : 일정 속도로 닳는 소모품(브레이크 패드, 타이어)
      expo   : 후반에 급격히 악화(엔진 베어링, 냉각 호스)
      knee   : 일정 기간 안정 후 변곡점에서 급락(배터리 SOH)
    """
    t = np.arange(life + 1) / float(life)
    if mode == "linear":
        h = t
    elif mode == "expo":
        h = t ** 3.0
    elif mode == "knee":
        knee = 0.65
        # np.where 는 양쪽 식을 모두 계산하므로 음수 거듭제곱을 막아야 한다.
        tail = np.clip((t - knee) / (1.0 - knee), 0.0, None)
        h = np.where(t < knee, 0.25 * (t / knee), 0.25 + 0.75 * tail ** 1.6)
    else:
        raise ValueError(f"알 수 없는 mode: {mode}")
    return np.clip(h + RNG.normal(0, 0.012, h.shape), 0.0, 1.05)


def piecewise_rul(life: int, r_early: int = R_EARLY) -> np.ndarray:
    """piecewise-linear RUL 라벨을 만든다.

    t=0..life 각 시점의 실제 남은 수명은 (life - t) 이고,
    r_early 로 상한을 걸어 초기 구간을 평탄화한다.
    """
    raw = np.arange(life, -1, -1, dtype=float)
    return np.minimum(raw, float(r_early))


def make_unit(unit_id: int, part: str, mode: str) -> pd.DataFrame:
    """부품 하나의 전 수명(run-to-failure) 궤적을 만든다."""
    life = int(RNG.integers(110, 260))
    h = degradation_curve(life, mode)
    rul = piecewise_rul(life)
    t = np.arange(life + 1)

    # 열화 지표를 센서 관측치로 변환한다(5.2절 센서 열 이름 유지).
    daily_km = float(RNG.uniform(30, 90))
    df = pd.DataFrame({
        "unit_id": unit_id,
        "part": part,
        "mode": mode,
        "day": t,
        "odometer_km": np.round(np.cumsum(np.full(life + 1, daily_km)), 0),
        "health_index": np.round(h, 4),
        # 관측 센서 3종: 서로 다른 민감도로 같은 열화를 반영한다.
        "vibration_rms": np.round(0.28 + 0.95 * h + RNG.normal(0, 0.02, life + 1), 4),
        "engine_temp_c": np.round(88.0 + 22.0 * h ** 1.4
                                  + RNG.normal(0, 0.7, life + 1), 2),
        "oil_quality": np.round(np.clip(0.95 - 0.80 * h
                                        + RNG.normal(0, 0.01, life + 1),
                                        0.02, 1.0), 4),
        "rul_days": rul.astype(int),
        "rul_true": np.arange(life, -1, -1),
        "life_total": life,
    })
    return df


def main() -> None:
    specs = [("brake", "linear"), ("tire", "linear"),
             ("engine", "expo"), ("cooling", "expo"),
             ("battery", "knee"), ("ev_battery", "knee")]
    frames = []
    uid = 1
    # 부품을 번갈아 배치해 unit_id 순서에 모드가 고루 섞이게 한다.
    # 부품별로 몰아서 매기면 유닛 단위 분할 시 평가 집합이 한 모드로만
    # 채워져 다른 열화 모드의 성능을 전혀 확인할 수 없다.
    for _ in range(5):                       # 부품별 5개 유닛
        for part, mode in specs:
            frames.append(make_unit(uid, part, mode))
            uid += 1
    df = pd.concat(frames, ignore_index=True)

    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / "rul_dataset.csv"
    df.to_csv(out, index=False)

    print(f"유닛 수 {df['unit_id'].nunique()} / 전체 행 {len(df)}")
    print(f"RUL 상한(R_early) = {R_EARLY}일\n")

    print("[열화 모드별 평균 수명과 RUL 분포]")
    print(f"{'모드':<8s}{'유닛':>5s}{'평균수명':>9s}{'RUL평균':>9s}{'상한포화비율':>13s}")
    for mode, g in df.groupby("mode"):
        sat = float((g["rul_days"] == R_EARLY).mean())
        print(f"{mode:<8s}{g['unit_id'].nunique():5d}"
              f"{g['life_total'].mean():9.1f}{g['rul_days'].mean():9.1f}{sat:13.3f}")

    u = df[df["unit_id"] == 11].reset_index(drop=True)   # expo 모드 유닛 하나
    print(f"\n[unit_id=11 ({u['part'].iloc[0]}/{u['mode'].iloc[0]}, "
          f"수명 {int(u['life_total'].iloc[0])}일) 궤적 일부]")
    print(f"{'day':>5s}{'health':>8s}{'vib':>7s}{'temp':>7s}"
          f"{'rul_true':>9s}{'rul_days':>9s}")
    for i in list(range(0, len(u), max(1, len(u) // 7))) + [len(u) - 1]:
        r = u.iloc[i]
        print(f"{int(r['day']):5d}{r['health_index']:8.3f}{r['vibration_rms']:7.3f}"
              f"{r['engine_temp_c']:7.1f}{int(r['rul_true']):9d}"
              f"{int(r['rul_days']):9d}")

    print(f"\n저장 완료: {out}")


if __name__ == "__main__":
    main()
