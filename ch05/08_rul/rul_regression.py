# -*- coding: utf-8 -*-
"""회귀 기반 RUL 예측 — 선형/Ridge/RandomForest/GradientBoosting 비교.

sklearn 미설치 시 안내 후 종료한다.

실행:  python 08_rul/rul_regression.py
입력:  data/rul_dataset.csv  (08_rul/rul_curve.py 가 생성)
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

# 같은 디렉터리의 rul_metrics.py 를 재사용한다(지표를 중복 구현하지 않는다).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from rul_metrics import evaluate                            # noqa: E402

DATA = pathlib.Path("data") / "rul_dataset.csv"
SENSORS = ["vibration_rms", "engine_temp_c", "oil_quality"]


def add_features(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """유닛 단위 롤링 통계와 누적 특징을 추가한다.

    RUL 회귀에서 가장 중요한 특징은 '현재 값'이 아니라
      - 초기 대비 변화량 (얼마나 닳았는가)
      - 최근 변화 속도   (얼마나 빨리 닳고 있는가)
    이다. 둘을 모두 넣어야 '현재 상태'와 '남은 시간'을 연결할 수 있다.
    """
    df = df.sort_values(["unit_id", "day"]).copy()
    g = df.groupby("unit_id", sort=False)
    for c in SENSORS:
        df[f"{c}_ma"] = g[c].transform(
            lambda s: s.rolling(window, min_periods=2).mean())
        df[f"{c}_sd"] = g[c].transform(
            lambda s: s.rolling(window, min_periods=2).std()).fillna(0.0)
        # 초기값(첫 10일 평균) 대비 변화량
        df[f"{c}_delta"] = df[c] - g[c].transform(lambda s: s.head(10).mean())
        # 최근 변화 속도(이동평균의 1차 차분)
        df[f"{c}_rate"] = g[c].transform(
            lambda s: s.rolling(window, min_periods=2).mean().diff()).fillna(0.0)
    df["cycle"] = g.cumcount()                 # 경과 사이클(일)
    return df.bfill().ffill()


def split_units(df: pd.DataFrame, ratio: float = 0.7):
    """유닛 단위 분할. 같은 유닛이 양쪽에 들어가면 지표가 무의미해진다."""
    uids = sorted(df["unit_id"].unique())
    tr = set(uids[:int(len(uids) * ratio)])
    return (df[df["unit_id"].isin(tr)].reset_index(drop=True),
            df[~df["unit_id"].isin(tr)].reset_index(drop=True))


def main() -> None:
    try:
        from sklearn.ensemble import (GradientBoostingRegressor,
                                      RandomForestRegressor)
        from sklearn.linear_model import LinearRegression, Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        print("[안내] scikit-learn 이 설치되어 있지 않다.")
        print("       pip install scikit-learn")
        print("       설치 후 다시 실행하면 네 가지 회귀 모델을 비교한다.")
        return
    if not DATA.exists():
        print(f"[안내] {DATA} 가 없다. 먼저 python 08_rul/rul_curve.py")
        return

    df = add_features(pd.read_csv(DATA))
    feats = (SENSORS + [c for c in df.columns
                        if c.endswith(("_ma", "_sd", "_delta", "_rate"))]
             + ["cycle", "odometer_km"])
    train, test = split_units(df)
    xtr, ytr = train[feats], train["rul_days"]
    xte, yte = test[feats], test["rul_days"]
    print(f"특징 {len(feats)}개 / 학습 {len(xtr)}행({train['unit_id'].nunique()}유닛)"
          f" / 평가 {len(xte)}행({test['unit_id'].nunique()}유닛)")
    print(f"라벨 범위 {ytr.min()}~{ytr.max()}일 (R_early 상한 적용)\n")

    models = {
        "Linear": make_pipeline(StandardScaler(), LinearRegression()),
        "Ridge(a=10)": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "RandomForest": RandomForestRegressor(
            n_estimators=300, max_depth=12, min_samples_leaf=5,
            n_jobs=-1, random_state=42),
        "GBDT": GradientBoostingRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.06,
            subsample=0.85, random_state=42),
    }

    hdr = (f"{'모델':<14s}{'RMSE':>7s}{'MAE':>7s}{'Bias':>8s}"
           f"{'Score':>10s}{'Cost':>10s}{'a20':>7s}{'Late':>7s}")
    print(hdr)
    print("-" * len(hdr))
    fitted = {}
    for name, m in models.items():
        m.fit(xtr, ytr)
        # RUL 은 음수가 될 수 없고 상한도 있으므로 반드시 클리핑한다.
        pred = np.clip(m.predict(xte), 0.0, float(ytr.max()))
        mt = evaluate(yte.to_numpy(dtype=float), pred)
        fitted[name] = (m, pred)
        print(f"{name:<14s}{mt['RMSE']:7.2f}{mt['MAE']:7.2f}{mt['Bias']:8.2f}"
              f"{mt['Score']:10.1f}{mt['Cost']:10.1f}"
              f"{mt['a20']:7.3f}{mt['LateRate']:7.3f}")

    print("\nRMSE 최소 모델과 Cost 최소 모델이 다를 수 있다. 예지보전에서는")
    print("Cost 와 Late 를 먼저 보고, 그중에서 RMSE 가 낮은 것을 고른다.")

    # 과대예측을 줄이는 간단한 후처리: 예측값에서 안전 여유를 뺀다.
    best = min(fitted, key=lambda k: evaluate(
        yte.to_numpy(dtype=float), fitted[k][1])["Cost"])
    m, pred = fitted[best]
    print(f"\n[{best} 모델에 안전 여유(margin)를 적용한 효과]")
    print(f"{'margin(일)':>11s}{'RMSE':>7s}{'Cost':>10s}{'Late':>7s}{'a20':>7s}")
    for margin in (0, 3, 5, 8, 12):
        p = np.clip(pred - margin, 0.0, None)
        mt = evaluate(yte.to_numpy(dtype=float), p)
        print(f"{margin:11d}{mt['RMSE']:7.2f}{mt['Cost']:10.1f}"
              f"{mt['LateRate']:7.3f}{mt['a20']:7.3f}")
    print("margin 을 키우면 RMSE 는 나빠지지만 Cost 와 Late 는 좋아진다.")
    print("어디서 멈출지는 '불필요 교체 비용'과 '고장 비용'의 비율이 정한다.")

    # 열화 모드별 성능도 반드시 쪼개 본다. 평균만 보면 특정 모드의 실패를 놓친다.
    print(f"\n[{best} 모델 — 열화 모드별 성능]")
    print(f"{'모드':<9s}{'행수':>7s}{'RMSE':>7s}{'MAE':>7s}{'Late':>7s}")
    for mode, g in test.groupby("mode"):
        p = np.clip(m.predict(g[feats]), 0.0, float(ytr.max()))
        mt = evaluate(g["rul_days"].to_numpy(dtype=float), p)
        print(f"{mode:<9s}{len(g):7d}{mt['RMSE']:7.2f}{mt['MAE']:7.2f}"
              f"{mt['LateRate']:7.3f}")


if __name__ == "__main__":
    main()
