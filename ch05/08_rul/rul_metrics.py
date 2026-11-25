# -*- coding: utf-8 -*-
"""RUL 평가지표 — RMSE/MAE/MAPE 와 비대칭 스코어 함수.

예지보전에서 과대예측(아직 괜찮다고 말했는데 고장)과 과소예측(아직
멀었는데 교체)은 비용이 전혀 다르다. 그래서 RMSE 하나로 모델을
선택하면 안 되고, 비대칭 페널티를 주는 스코어 함수를 함께 본다.

이 파일은 numpy 만으로 동작한다.
실행:  python 08_rul/rul_metrics.py
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------- 대칭 지표
def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """제곱평균제곱근오차. 큰 오차에 더 큰 벌점을 준다."""
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """평균절대오차. '평균 몇 일 틀렸는가' 를 그대로 읽을 수 있다."""
    return float(np.mean(np.abs(y_pred - y_true)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1.0) -> float:
    """평균절대백분율오차(%). RUL 이 0 에 가까울 때 발산하므로 eps 로 보호한다."""
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs(y_pred - y_true) / denom) * 100.0)


def bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """평균 편향. 음수면 전반적으로 과소예측(보수적), 양수면 과대예측."""
    return float(np.mean(y_pred - y_true))


# ----------------------------------------------------- 비대칭 스코어 함수
def nasa_score(y_true: np.ndarray, y_pred: np.ndarray,
               a_late: float = 10.0, a_early: float = 13.0) -> float:
    """NASA C-MAPSS 계열 비대칭 스코어. 작을수록 좋다.

    d = y_pred - y_true 라 할 때
      d < 0 (과소예측, 실제보다 빨리 고장난다고 예측) → exp(-d/a_early) - 1
      d >= 0 (과대예측, 실제보다 늦게 고장난다고 예측) → exp(d/a_late) - 1
    a_late < a_early 이므로 과대예측에 더 큰 벌점이 붙는다.

    주의: 원 논문의 관례와 달리 현장에서는 '과소예측'이 안전 측 오류다.
    여기서는 과대예측(고장을 놓치는 쪽)에 큰 벌점을 주도록 설정한다.
    """
    d = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    s = np.where(d < 0.0,
                 np.exp(-d / a_early) - 1.0,
                 np.exp(d / a_late) - 1.0)
    return float(np.sum(s))


def asymmetric_cost(y_true: np.ndarray, y_pred: np.ndarray,
                    cost_early_per_day: float = 1.0,
                    cost_late_per_day: float = 6.0) -> float:
    """선형 비용 함수. 단위는 '정비 비용 환산 일수'로 해석한다.

    과소예측 1일 = 불필요하게 일찍 교체해 낭비되는 잔여 수명 1일.
    과대예측 1일 = 고장 상태로 운행한 1일(견인·2차 손상·휴차 비용).
    현장 데이터로 두 단가 비율을 추정해 넣으면 바로 금액이 된다.
    """
    d = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    early = np.sum(np.clip(-d, 0.0, None)) * cost_early_per_day
    late = np.sum(np.clip(d, 0.0, None)) * cost_late_per_day
    return float(early + late)


def alpha_lambda_accuracy(y_true: np.ndarray, y_pred: np.ndarray,
                          alpha: float = 0.20) -> float:
    """α-λ 정확도: 예측이 실제의 ±α 범위 안에 든 비율."""
    yt = np.asarray(y_true, dtype=float)
    band = np.maximum(alpha * np.abs(yt), 1.0)
    return float(np.mean(np.abs(np.asarray(y_pred, dtype=float) - yt) <= band))


def late_rate(y_true: np.ndarray, y_pred: np.ndarray,
              tol_days: float = 3.0) -> float:
    """허용 오차를 넘어 과대예측한 비율. 안전 관련 핵심 지표."""
    d = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    return float(np.mean(d > tol_days))


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """모든 지표를 한 번에 계산해 딕셔너리로 돌려준다."""
    return {
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
        "Bias": bias(y_true, y_pred),
        "Score": nasa_score(y_true, y_pred),
        "Cost": asymmetric_cost(y_true, y_pred),
        "a20": alpha_lambda_accuracy(y_true, y_pred),
        "LateRate": late_rate(y_true, y_pred),
    }


def main() -> None:
    rng = np.random.default_rng(7)
    n = 400
    y = rng.uniform(1.0, 60.0, n)             # 실제 RUL(일)

    # 같은 오차 크기를 가지면서 편향 방향만 다른 세 가지 예측기
    noise = rng.normal(0.0, 5.0, n)
    preds = {
        "무편향(±5일)": y + noise,
        "보수적(-6일 편향)": y + noise - 6.0,
        "낙관적(+6일 편향)": y + noise + 6.0,
        "상수예측(30일)": np.full(n, 30.0),
        "정밀(±2일)": y + rng.normal(0.0, 2.0, n),
    }

    hdr = (f"{'예측기':<18s}{'RMSE':>7s}{'MAE':>7s}{'MAPE':>8s}"
           f"{'Bias':>7s}{'Score':>11s}{'Cost':>9s}{'a20':>7s}{'Late':>7s}")
    print(hdr)
    print("-" * len(hdr))
    for name, p in preds.items():
        m = evaluate(y, np.clip(p, 0.0, None))
        print(f"{name:<18s}{m['RMSE']:7.2f}{m['MAE']:7.2f}{m['MAPE']:8.1f}"
              f"{m['Bias']:7.2f}{m['Score']:11.1f}{m['Cost']:9.1f}"
              f"{m['a20']:7.3f}{m['LateRate']:7.3f}")

    print("\n[같은 RMSE, 다른 운용 위험]")
    a = evaluate(y, np.clip(y + noise - 6.0, 0.0, None))
    b = evaluate(y, np.clip(y + noise + 6.0, 0.0, None))
    print(f"보수적 RMSE={a['RMSE']:.2f}  Cost={a['Cost']:.1f}  Late={a['LateRate']:.3f}")
    print(f"낙관적 RMSE={b['RMSE']:.2f}  Cost={b['Cost']:.1f}  Late={b['LateRate']:.3f}")
    print(f"RMSE 차이는 {abs(a['RMSE'] - b['RMSE']):.2f}일에 불과하지만 "
          f"비대칭 비용은 {b['Cost'] / a['Cost']:.1f}배 차이가 난다.")

    print("\n[과대예측 일수별 스코어 벌점 — a_late=10, a_early=13]")
    print(f"{'오차(일)':>9s}{'스코어 기여':>13s}{'선형비용':>10s}")
    for d in (-20, -10, -5, 0, 5, 10, 20):
        s = nasa_score(np.array([30.0]), np.array([30.0 + d]))
        c = asymmetric_cost(np.array([30.0]), np.array([30.0 + d]))
        print(f"{d:9d}{s:13.2f}{c:10.1f}")


if __name__ == "__main__":
    main()
