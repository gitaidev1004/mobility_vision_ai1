# -*- coding: utf-8 -*-
"""RandomForest 이진/다중 분류 + 특징 중요도.

scikit-learn 이 없으면 설치 안내를 출력하고 정상 종료한다.

실행:  python 06_failure/random_forest.py
입력:  data/labeled_dataset.csv  (06_failure/make_labels.py 가 생성)
출력:  data/rf_binary.joblib, data/rf_multi.joblib
"""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

DATA = pathlib.Path("data") / "labeled_dataset.csv"
OUT_DIR = pathlib.Path("data")

# 5.4절 전처리(ch05/04_preprocess/features.py)에서 만든 기본 센서 열과
# 롤링 특징을 함께 쓴다. 여기서는 원 센서 열만으로 기준선을 만든다.
BASE_FEATURES = [
    "odometer_km", "engine_temp_c", "engine_rpm", "speed_kph",
    "battery_voltage", "battery_soh", "tire_pressure_fl", "tire_pressure_fr",
    "tire_pressure_rl", "tire_pressure_rr", "brake_pad_mm",
    "oil_quality", "vibration_rms", "fuel_level",
]
ROLL_COLS = ["engine_temp_c", "battery_voltage", "brake_pad_mm",
             "vibration_rms", "oil_quality"]
PART_CLASSES = ["NONE", "engine", "battery", "brake", "tire", "cooling"]


def add_rolling_features(df: pd.DataFrame, window: int = 7) -> pd.DataFrame:
    """차량 단위 롤링 평균·표준편차·기울기를 추가한다.

    예지보전에서 중요한 것은 '현재 값'이 아니라 '변해 가는 방향'이다.
    같은 92℃ 라도 평소 88℃ 였다면 이상이고, 평소 92℃ 였다면 정상이다.
    그래서 원 센서 값과 함께 롤링 통계를 반드시 넣는다.
    """
    df = df.sort_values(["vehicle_id", "day"]).copy()
    g = df.groupby("vehicle_id", sort=False)
    for c in ROLL_COLS:
        df[f"{c}_ma{window}"] = g[c].transform(
            lambda s: s.rolling(window, min_periods=2).mean())
        df[f"{c}_sd{window}"] = g[c].transform(
            lambda s: s.rolling(window, min_periods=2).std()).fillna(0.0)
        # 기울기 근사: 이동평균의 1차 차분 (하루당 변화량)
        df[f"{c}_slope"] = g[c].transform(
            lambda s: s.rolling(window, min_periods=2).mean().diff()).fillna(0.0)
    return df.bfill().ffill()


def split_by_vehicle(df: pd.DataFrame, train_ratio: float = 0.7
                     ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """차량 단위로 학습/평가를 나눈다.

    행 단위로 무작위 분할하면 같은 차량의 인접한 날이 양쪽에 들어가
    거의 같은 데이터를 학습하고 평가하게 된다. 지표가 0.99 로 나와도
    실제 신차에서는 전혀 맞지 않는다. 반드시 차량(유닛) 단위로 나눈다.
    """
    vids = sorted(df["vehicle_id"].unique())
    n_tr = int(len(vids) * train_ratio)
    tr = set(vids[:n_tr])
    return (df[df["vehicle_id"].isin(tr)].reset_index(drop=True),
            df[~df["vehicle_id"].isin(tr)].reset_index(drop=True))


def main() -> None:
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import (classification_report, confusion_matrix,
                                     roc_auc_score)
        import joblib
    except ImportError:
        print("[안내] scikit-learn 이 설치되어 있지 않다. 아래로 설치한다.")
        print("       pip install scikit-learn joblib")
        print("       설치 후 다시 실행하면 RandomForest 학습이 진행된다.")
        return

    if not DATA.exists():
        print(f"[안내] {DATA} 가 없다. 먼저 아래를 실행한다.")
        print("       python 06_failure/make_labels.py")
        return

    df = add_rolling_features(pd.read_csv(DATA))
    feats = BASE_FEATURES + [c for c in df.columns
                             if c.endswith(("_ma7", "_sd7", "_slope"))]
    train, test = split_by_vehicle(df)
    print(f"특징 {len(feats)}개 / 학습 {len(train)}행 / 평가 {len(test)}행")

    # ---------------------------------------------------------- 이진 분류
    # class_weight='balanced' 는 양성이 희소한 고장 데이터에서 필수다.
    # 넣지 않으면 모델이 전부 '정상'으로 예측해도 정확도 88% 가 나온다.
    rf_bin = RandomForestClassifier(
        n_estimators=300, max_depth=12, min_samples_leaf=5,
        class_weight="balanced", n_jobs=-1, random_state=42)
    rf_bin.fit(train[feats], train["will_fail_30"])

    prob = rf_bin.predict_proba(test[feats])[:, 1]
    pred = (prob >= 0.5).astype(int)
    print("\n=== 이진 분류: 30일 내 고장 여부 ===")
    print(classification_report(test["will_fail_30"], pred,
                                target_names=["정상", "고장예상"], digits=3))
    print(f"ROC-AUC = {roc_auc_score(test['will_fail_30'], prob):.4f}")

    # 임계값을 0.5 로 고정하지 않는다. 재현율 목표를 먼저 정하고 역산한다.
    print("\n[임계값별 정밀도/재현율 — 운영 임계값 선택용]")
    y = test["will_fail_30"].to_numpy()
    print(f"{'임계값':>7s}{'정밀도':>9s}{'재현율':>9s}{'경보건수':>9s}")
    for th in (0.20, 0.30, 0.40, 0.50, 0.65, 0.80):
        p = (prob >= th).astype(int)
        tp = int(((y == 1) & (p == 1)).sum())
        fp = int(((y == 0) & (p == 1)).sum())
        fn = int(((y == 1) & (p == 0)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        print(f"{th:7.2f}{prec:9.3f}{rec:9.3f}{tp + fp:9d}")

    # ------------------------------------------------------ 다중 클래스 분류
    rf_multi = RandomForestClassifier(
        n_estimators=300, max_depth=14, min_samples_leaf=3,
        class_weight="balanced_subsample", n_jobs=-1, random_state=42)
    rf_multi.fit(train[feats], train["fail_part"])
    pred_m = rf_multi.predict(test[feats])

    present = sorted(set(test["fail_part"]) | set(pred_m))
    print("\n=== 다중 클래스: 고장 부품 분류 ===")
    print(classification_report(test["fail_part"], pred_m, labels=present,
                                target_names=[PART_CLASSES[i] for i in present],
                                digits=3, zero_division=0))
    print("혼동행렬(행=실제, 열=예측)")
    cm = confusion_matrix(test["fail_part"], pred_m, labels=present)
    print("        " + "".join(f"{PART_CLASSES[i][:7]:>8s}" for i in present))
    for i, row in zip(present, cm):
        print(f"{PART_CLASSES[i][:7]:>8s}" + "".join(f"{v:8d}" for v in row))

    # --------------------------------------------------------- 특징 중요도
    # 불순도 감소 기반 중요도는 값의 종류가 많은 열을 과대평가한다.
    # 최종 판단은 permutation importance 로 교차 확인하는 것이 안전하다.
    imp = pd.Series(rf_bin.feature_importances_, index=feats).sort_values(
        ascending=False)
    print("\n[이진 모델 특징 중요도 상위 12개]")
    for name, v in imp.head(12).items():
        bar = "#" * int(v * 200)
        print(f"  {name:<26s}{v:7.4f} {bar}")
    print(f"  상위 12개 누적 기여도: {imp.head(12).sum():.3f}")

    OUT_DIR.mkdir(exist_ok=True)
    joblib.dump({"model": rf_bin, "features": feats}, OUT_DIR / "rf_binary.joblib")
    joblib.dump({"model": rf_multi, "features": feats,
                 "classes": PART_CLASSES}, OUT_DIR / "rf_multi.joblib")
    print(f"\n모델 저장: {OUT_DIR / 'rf_binary.joblib'}, "
          f"{OUT_DIR / 'rf_multi.joblib'}")
    print(f"파라미터 수 대신 트리 개수 {rf_bin.n_estimators}개 / "
          f"평균 깊이 "
          f"{np.mean([t.get_depth() for t in rf_bin.estimators_]):.1f}")


if __name__ == "__main__":
    main()
