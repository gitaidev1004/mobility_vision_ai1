# -*- coding: utf-8 -*-
"""IsolationForest 기반 차량 센서 이상 탐지.

IsolationForest 는 무작위로 특징과 분할점을 골라 트리를 만들고,
'몇 번 쪼개면 그 점이 혼자 남는가(경로 길이)'로 이상도를 측정한다.
이상점은 밀집 영역에서 떨어져 있어 적은 분할로 고립되므로 경로가 짧다.

장점: 거리/밀도 계산이 없어 고차원에서도 빠르고, 학습이 O(n log n) 이다.
단점: 축에 평행한 분할만 하므로 비스듬한(상관된) 경계를 잘 못 잡는다.

학습은 정상 차량 데이터로만 한다(semi-supervised 설정).
sklearn 미설치 시 안내 후 정상 종료한다.
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np
import pandas as pd

FEATURES = [
    "engine_temp_c", "engine_rpm", "speed_kph",
    "battery_voltage", "battery_soh",
    "tire_pressure_fl", "tire_pressure_fr",
    "tire_pressure_rl", "tire_pressure_rr",
    "brake_pad_mm", "oil_quality", "vibration_rms",
]
SCENARIO = {1: "normal", 2: "normal", 3: "degrade", 4: "failure"}
NORMAL_IDS = [1, 2]


def load(csv: pathlib.Path) -> pd.DataFrame | None:
    """정제된 센서 CSV 를 읽는다. 없으면 안내를 출력한다."""
    if not csv.exists():
        print(f"[오류] 입력 CSV 가 없다: {csv}")
        print("  아래 순서로 먼저 데이터를 만든다.")
        print("  python 02_data/make_sensor_dataset.py --defects "
              "--out data/sensor_readings_raw.csv")
        print("  python 04_preprocess/clean.py")
        return None
    return pd.read_csv(csv)


def main() -> None:
    ap = argparse.ArgumentParser(description="IsolationForest 이상 탐지")
    ap.add_argument("--csv", default="data/sensor_clean.csv")
    ap.add_argument("--out", default="data/score_iforest.csv")
    ap.add_argument("--contamination", type=float, default=0.02,
                    help="학습 데이터에 섞여 있다고 가정하는 이상 비율")
    ap.add_argument("--n-estimators", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20261024)
    args = ap.parse_args()

    try:
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        print("[안내] scikit-learn 이 설치되어 있지 않다.")
        print("  아래 명령으로 설치한 뒤 다시 실행한다.")
        print("    pip install scikit-learn")
        print("  설치 없이 진행하려면 05_anomaly/stat_threshold.py 를 사용한다.")
        return

    df = load(pathlib.Path(args.csv))
    if df is None:
        return

    cols = [c for c in FEATURES if c in df.columns]
    df["scenario"] = df["vehicle_id"].map(SCENARIO)

    # 정상 차량만 학습에 쓴다. 고장 차량이 섞이면 모델이 고장을 정상으로 배운다.
    train = df[df["vehicle_id"].isin(NORMAL_IDS)]
    print(f"학습 {len(train):,}행(정상 차량 {NORMAL_IDS}) / "
          f"전체 평가 {len(df):,}행, 특징 {len(cols)}개")

    # 스케일링 통계량도 학습 데이터에서만 구한다.
    scaler = StandardScaler().fit(train[cols])
    x_train = scaler.transform(train[cols])
    x_all = scaler.transform(df[cols])

    model = IsolationForest(
        n_estimators=args.n_estimators,
        contamination=args.contamination,
        max_samples="auto",
        random_state=args.seed,
        n_jobs=-1,
    ).fit(x_train)

    # score_samples 는 클수록 정상이다. 부호를 뒤집어 '클수록 이상' 으로 만든다.
    raw = -model.score_samples(x_all)
    df["iforest_raw"] = raw

    # 0~1 정규화(앙상블에서 다른 탐지기와 더하기 위해).
    # 기준점을 학습 분포의 '중앙값'으로 잡는다. 최솟값을 0 으로 잡으면
    # 정상 데이터의 절반이 0.5 부근에 몰려 정상/이상 구분이 흐려진다.
    base = -model.score_samples(x_train)
    lo, hi = np.quantile(base, 0.50), np.quantile(base, 0.999)
    df["iforest_score"] = np.clip((raw - lo) / max(hi - lo, 1e-12), 0.0, 1.0)

    print(f"\n[정규화 기준]  학습 중앙값={lo:.4f} -> 0,  "
          f"99.9% 분위={hi:.4f} -> 1")

    print(f"\n[차량별 이상 점수]")
    print(f"  {'vid':>4}{'시나리오':>10}{'평균':>9}{'중앙값':>9}"
          f"{'95%':>9}{'최대':>9}{'경보율':>9}")
    thr = float(np.quantile(base, 1 - args.contamination))
    for vid, g in df.groupby("vehicle_id"):
        s = g["iforest_score"]
        rate = (g["iforest_raw"] > thr).mean()
        print(f"  {vid:>4}{SCENARIO.get(vid, '?'):>10}{s.mean():>9.4f}"
              f"{s.median():>9.4f}{s.quantile(0.95):>9.4f}"
              f"{s.max():>9.4f}{rate * 100:>8.1f}%")

    print(f"\n[시간 구간별 평균 점수]  열화가 진행되며 점수가 오르는지 확인")
    print(f"  {'vid':>4}{'시나리오':>10}" + "".join(
        f"{f'{i * 25}-{(i + 1) * 25}%':>10}" for i in range(4)))
    for vid, g in df.groupby("vehicle_id"):
        g = g.reset_index(drop=True)
        q = len(g) // 4
        parts = [g["iforest_score"].iloc[i * q:(i + 1) * q].mean()
                 for i in range(4)]
        print(f"  {vid:>4}{SCENARIO.get(vid, '?'):>10}"
              + "".join(f"{p:>10.4f}" for p in parts))

    # 어떤 센서가 이상 판정에 기여했는지 본다(표준화 거리 기준의 근사).
    print(f"\n[이상 상위 1% 구간에서 가장 벗어난 센서]")
    hi_mask = df["iforest_score"] >= df["iforest_score"].quantile(0.99)
    mu = pd.Series(scaler.mean_, index=cols)
    sd = pd.Series(np.sqrt(scaler.var_), index=cols)
    dev = ((df.loc[hi_mask, cols] - mu) / sd).abs().mean()
    for c, v in dev.sort_values(ascending=False).head(6).items():
        print(f"  {c:<20}{v:>8.2f} 표준편차")

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    keep = ["vehicle_id", "measured_at", "scenario",
            "iforest_raw", "iforest_score"]
    df[keep].to_csv(out, index=False, encoding="utf-8")
    print(f"\n저장: {out}")
    print("  anomaly_score.py 가 이 파일을 읽어 앙상블에 사용한다.")


if __name__ == "__main__":
    main()
