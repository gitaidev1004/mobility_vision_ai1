# -*- coding: utf-8 -*-
"""One-Class SVM 기반 차량 센서 이상 탐지.

One-Class SVM 은 정상 데이터만 보고 그 데이터를 감싸는 경계를 찾는다.
RBF 커널을 쓰면 특징 공간에서 원점으로부터 정상 영역을 떼어내는
초평면을 학습하고, 결정함수가 음수인 점을 이상으로 본다.

장점: 비선형 경계를 잡을 수 있고 정상 데이터만으로 학습된다.
단점: 학습이 O(n^2)~O(n^3) 이라 표본이 많으면 매우 느리다.
      nu(이상 비율 상한)와 gamma(경계의 날카로움)에 민감하다.
      -> 표본이 많으면 SGDOneClassSVM + Nystroem 근사를 쓴다.

sklearn 미설치 시 안내 후 정상 종료한다.
"""
from __future__ import annotations

import argparse
import pathlib
import time

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
    ap = argparse.ArgumentParser(description="One-Class SVM 이상 탐지")
    ap.add_argument("--csv", default="data/sensor_clean.csv")
    ap.add_argument("--out", default="data/score_ocsvm.csv")
    ap.add_argument("--nu", type=float, default=0.02,
                    help="이상으로 허용할 비율의 상한")
    ap.add_argument("--gamma", default="scale",
                    help="RBF 커널 폭. 'scale' 또는 실수")
    args = ap.parse_args()

    try:
        from sklearn.svm import OneClassSVM
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
    train = df[df["vehicle_id"].isin(NORMAL_IDS)]
    print(f"학습 {len(train):,}행(정상 차량 {NORMAL_IDS}) / "
          f"전체 평가 {len(df):,}행, 특징 {len(cols)}개")

    # One-Class SVM 은 거리 기반이므로 스케일링이 필수다.
    scaler = StandardScaler().fit(train[cols])
    x_train = scaler.transform(train[cols])
    x_all = scaler.transform(df[cols])

    gamma = args.gamma
    if gamma not in {"scale", "auto"}:
        gamma = float(gamma)

    t0 = time.perf_counter()
    model = OneClassSVM(kernel="rbf", nu=args.nu, gamma=gamma).fit(x_train)
    fit_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    dec = model.decision_function(x_all)       # 양수=정상, 음수=이상
    pred_ms = (time.perf_counter() - t0) * 1000

    print(f"\n학습 {fit_ms:,.0f}ms / 추론 {pred_ms:,.0f}ms "
          f"({pred_ms / len(df) * 1000:.1f}us per row)")
    print(f"서포트 벡터 {len(model.support_):,}개 "
          f"(학습 표본의 {len(model.support_) / len(x_train) * 100:.1f}%)")
    print("  서포트 벡터 수가 학습 표본에 비례해 늘면 추론도 그만큼 느려진다.")

    df["ocsvm_raw"] = -dec                     # 부호 반전: 클수록 이상
    # 0~1 정규화. 기준점을 학습 분포의 '중앙값'으로 잡는다. 최솟값(1% 분위)을
    # 0 으로 잡으면 정상 데이터의 절반이 0.5 부근에 몰려 앙상블에서
    # 정상/이상 구분이 흐려진다. 중앙값을 0 으로 두면 정상은 0 근처,
    # 경계를 벗어난 점만 1 에 가까워진다.
    base = -model.decision_function(x_train)
    lo, hi = np.quantile(base, 0.50), np.quantile(base, 0.999)
    df["ocsvm_score"] = np.clip((df["ocsvm_raw"] - lo)
                                / max(hi - lo, 1e-12), 0.0, 1.0)
    print(f"\n[정규화 기준]  학습 중앙값={lo:.4f} -> 0,  "
          f"99.9% 분위={hi:.4f} -> 1")

    print(f"\n[차량별 이상 점수]")
    print(f"  {'vid':>4}{'시나리오':>10}{'평균':>9}{'중앙값':>9}"
          f"{'95%':>9}{'최대':>9}{'경보율':>9}")
    for vid, g in df.groupby("vehicle_id"):
        s = g["ocsvm_score"]
        rate = (g["ocsvm_raw"] > 0).mean()      # 결정함수 음수 = 경계 밖
        print(f"  {vid:>4}{SCENARIO.get(vid, '?'):>10}{s.mean():>9.4f}"
              f"{s.median():>9.4f}{s.quantile(0.95):>9.4f}"
              f"{s.max():>9.4f}{rate * 100:>8.1f}%")

    print("\n  주의: degrade 와 failure 가 모두 1.0 으로 포화되어 서로")
    print("  구분되지 않는다. One-Class SVM 의 결정함수는 '경계에서")
    print("  얼마나 벗어났는가'를 선형으로 재지 않기 때문이다. 경계를")
    print("  크게 벗어나면 전부 같은 값으로 뭉친다. 따라서 이 탐지기는")
    print("  '이상인가 아닌가'의 판정에는 좋지만 '얼마나 심한가'의")
    print("  등급 구분에는 쓰기 어렵다. IsolationForest 나 AutoEncoder")
    print("  재구성 오차는 심각도에 따라 연속적으로 커지므로 보완이 된다.")
    print("  이것이 5.5(9) 에서 여러 탐지기를 앙상블하는 이유다.")

    print(f"\n[nu / gamma 민감도]  경보율(%) — 정상 차량은 낮고 고장은 높아야 한다")
    print(f"  {'nu':>6}{'gamma':>8}" + "".join(
        f"{f'vid{v}':>9}" for v in sorted(SCENARIO)))
    for nu in (0.01, 0.02, 0.05, 0.10):
        for gm in ("scale", 0.05):
            m = OneClassSVM(kernel="rbf", nu=nu, gamma=gm).fit(x_train)
            p = m.predict(x_all)                # -1=이상, 1=정상
            rates = [(p[(df["vehicle_id"] == v).to_numpy()] == -1).mean()
                     for v in sorted(SCENARIO)]
            print(f"  {nu:>6.2f}{str(gm):>8}"
                  + "".join(f"{r * 100:>8.1f}%" for r in rates))
    print("  nu 를 올리면 정상 차량 경보율(오경보)이 바로 따라 오른다.")
    print("  nu 는 '학습 데이터에서 이상으로 버릴 비율의 상한'이므로")
    print("  사실상 오경보율을 직접 지정하는 값이다.")

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    keep = ["vehicle_id", "measured_at", "scenario",
            "ocsvm_raw", "ocsvm_score"]
    df[keep].to_csv(out, index=False, encoding="utf-8")
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
