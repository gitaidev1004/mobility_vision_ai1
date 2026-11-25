# -*- coding: utf-8 -*-
"""분류 지표 직접 구현 + 모델 비교표 출력.

sklearn.metrics 를 쓰지 않고 혼동행렬·정밀도·재현율·F1·ROC-AUC 를
numpy 로 직접 구현한다. 지표의 정의를 코드로 확인하는 것이 목적이며,
라이브러리가 없는 환경에서도 비교표를 만들 수 있다.

비교 대상 모델은 '결정적 규칙 기반 점수 함수' 네 가지다. 실제 학습된
모델이 없어도 지표 계산 절차와 비교표 양식을 그대로 익힐 수 있다.

실행:  python 06_failure/compare_models.py
입력:  data/labeled_dataset.csv  (06_failure/make_labels.py 가 생성)
"""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

DATA = pathlib.Path("data") / "labeled_dataset.csv"
FEATURES = ["engine_temp_c", "battery_voltage", "battery_soh",
            "brake_pad_mm", "oil_quality", "vibration_rms",
            "tire_pressure_rl", "tire_pressure_rr"]


# ---------------------------------------------------------------- 지표 구현
def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    """혼동행렬 4개 값 (TN, FP, FN, TP) 을 반환한다."""
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tn, fp, fn, tp


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """정확도 = (TP+TN) / 전체. 불균형 데이터에서는 과대평가된다."""
    return float(np.mean(y_true == y_pred))


def precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """정밀도 = TP / (TP+FP). '고장이라 했을 때 맞을 확률'."""
    _, fp, _, tp = confusion(y_true, y_pred)
    return tp / (tp + fp) if (tp + fp) else 0.0


def recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """재현율 = TP / (TP+FN). '실제 고장을 놓치지 않는 비율'."""
    _, _, fn, tp = confusion(y_true, y_pred)
    return tp / (tp + fn) if (tp + fn) else 0.0


def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """F1 = 정밀도와 재현율의 조화평균."""
    p, r = precision(y_true, y_pred), recall(y_true, y_pred)
    return 2 * p * r / (p + r) if (p + r) else 0.0


def roc_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    """ROC-AUC 를 순위합(Mann-Whitney U) 공식으로 계산한다.

    AUC = (양성 점수가 음성 점수보다 클 확률). 동점은 0.5 로 센다.
    순위를 쓰면 O(n log n) 으로 계산된다.
    """
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=float)
    ranks[order] = np.arange(1, len(score) + 1, dtype=float)
    # 동점 처리: 같은 점수끼리 평균 순위를 부여한다.
    srt = score[order]
    i = 0
    while i < len(srt):
        j = i
        while j + 1 < len(srt) and srt[j + 1] == srt[i]:
            j += 1
        if j > i:
            avg = (i + j + 2) / 2.0
            ranks[order[i:j + 1]] = avg
        i = j + 1
    rank_pos_sum = float(np.sum(ranks[y_true == 1]))
    return (rank_pos_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def roc_curve_points(y_true: np.ndarray, score: np.ndarray,
                     steps: int = 11) -> list[tuple[float, float, float]]:
    """임계값을 바꿔 (임계값, FPR, TPR) 을 돌려준다."""
    qs = np.quantile(score, np.linspace(0.0, 1.0, steps))
    pts = []
    for th in qs:
        pred = (score >= th).astype(np.int8)
        tn, fp, fn, tp = confusion(y_true, pred)
        tpr = tp / (tp + fn) if (tp + fn) else 0.0
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        pts.append((float(th), fpr, tpr))
    return pts


# ----------------------------------------------------------- 비교용 점수 함수
def zscore(x: np.ndarray) -> np.ndarray:
    """표준화. 분산이 0 이면 0 벡터를 반환한다."""
    sd = x.std()
    return (x - x.mean()) / sd if sd > 1e-12 else np.zeros_like(x)


def model_threshold(df: pd.DataFrame) -> np.ndarray:
    """단일 임계 규칙: 브레이크 패드 잔량만 본다(5.5절 Threshold 방식)."""
    return zscore(-df["brake_pad_mm"].to_numpy(dtype=float))


def model_linear(df: pd.DataFrame) -> np.ndarray:
    """선형 결합: 전문가가 부여한 가중치로 센서를 합산한다."""
    w = {"engine_temp_c": 1.0, "battery_voltage": -1.0, "battery_soh": -1.0,
         "brake_pad_mm": -1.2, "oil_quality": -0.8, "vibration_rms": 1.1,
         "tire_pressure_rl": -0.9, "tire_pressure_rr": -0.7}
    s = np.zeros(len(df))
    for k, wk in w.items():
        s += wk * zscore(df[k].to_numpy(dtype=float))
    return s


def model_max_dev(df: pd.DataFrame) -> np.ndarray:
    """최대 이탈: 여러 센서 중 가장 많이 이탈한 값 하나로 판정한다."""
    sign = {"engine_temp_c": 1, "battery_voltage": -1, "battery_soh": -1,
            "brake_pad_mm": -1, "oil_quality": -1, "vibration_rms": 1,
            "tire_pressure_rl": -1, "tire_pressure_rr": -1}
    cols = [sign[k] * zscore(df[k].to_numpy(dtype=float)) for k in FEATURES]
    return np.max(np.vstack(cols), axis=0)


def model_trend(df: pd.DataFrame) -> np.ndarray:
    """추세 결합: 차량별 7일 이동평균의 변화량을 더해 열화 속도를 본다."""
    base = model_linear(df)
    g = pd.Series(base, index=df.index).groupby(df["vehicle_id"].to_numpy())
    slope = g.transform(lambda s: s.rolling(7, min_periods=2).mean().diff().fillna(0.0))
    return base + 6.0 * slope.to_numpy()


MODELS = {
    "Threshold(단일)": model_threshold,
    "Linear(가중합)": model_linear,
    "MaxDeviation": model_max_dev,
    "Trend+Linear": model_trend,
}


def best_f1_threshold(y: np.ndarray, s: np.ndarray) -> tuple[float, float]:
    """F1 이 최대가 되는 임계값을 격자 탐색으로 찾는다."""
    best_t, best_v = 0.0, -1.0
    for q in np.linspace(0.50, 0.995, 100):
        th = float(np.quantile(s, q))
        v = f1_score(y, (s >= th).astype(np.int8))
        if v > best_v:
            best_t, best_v = th, v
    return best_t, best_v


def main() -> None:
    if not DATA.exists():
        print(f"[안내] {DATA} 가 없다. 먼저 아래를 실행한다.")
        print("       python 06_failure/make_labels.py")
        return

    df = pd.read_csv(DATA)
    # 차량 단위 분리: 같은 차량이 학습/평가에 함께 들어가면 지표가 부풀려진다.
    vids = sorted(df["vehicle_id"].unique())
    test_vids = set(vids[int(len(vids) * 0.7):])
    test = df[df["vehicle_id"].isin(test_vids)].reset_index(drop=True)
    y = test["will_fail_30"].to_numpy(dtype=np.int8)

    print(f"평가 대상: 차량 {len(test_vids)}대 / {len(test)}행 "
          f"/ 양성 {int(y.sum())}건({y.mean() * 100:.2f}%)\n")

    rows = []
    for name, fn in MODELS.items():
        s = fn(test)
        th, _ = best_f1_threshold(y, s)
        pred = (s >= th).astype(np.int8)
        tn, fp, fn_, tp = confusion(y, pred)
        rows.append((name, accuracy(y, pred), precision(y, pred),
                     recall(y, pred), f1_score(y, pred), roc_auc(y, s),
                     f"{tn}/{fp}/{fn_}/{tp}"))

    hdr = f"{'모델':<16s}{'Acc':>7s}{'Prec':>7s}{'Recall':>8s}{'F1':>7s}{'AUC':>7s}  TN/FP/FN/TP"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r[0]:<16s}{r[1]:7.3f}{r[2]:7.3f}{r[3]:8.3f}"
              f"{r[4]:7.3f}{r[5]:7.3f}  {r[6]}")

    best = max(rows, key=lambda r: r[5])
    print(f"\nROC-AUC 최고 모델: {best[0]} (AUC={best[5]:.3f})")

    print("\n[Trend+Linear 모델의 ROC 곡선 샘플]")
    print(f"{'임계값':>9s}{'FPR':>8s}{'TPR':>8s}")
    for th, fpr, tpr in roc_curve_points(y, model_trend(test), steps=6):
        print(f"{th:9.3f}{fpr:8.3f}{tpr:8.3f}")


if __name__ == "__main__":
    main()
