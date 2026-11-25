# -*- coding: utf-8 -*-
"""통계 기반 이상 탐지: 3시그마 규칙과 EWMA/CUSUM 관리도.

공정 관리에서 쓰는 관리도(Control Chart)를 차량 센서에 적용한다.
  3시그마(Shewhart) : 한 점이 평균 +- 3표준편차를 벗어나면 이상
  EWMA              : 지수가중 이동평균이 관리한계를 벗어나면 이상
  CUSUM             : 누적합이 임계를 넘으면 이상

관리도를 센서 원신호에 그대로 걸면 쓸 수 없다. 관리도는 신호가
정상상태(stationary)라고 가정하지만 차량 센서는 그렇지 않다.
  - 냉각수 온도는 시동 후 웜업하며 오르고, 엔진 부하에 따라 달라진다.
  - 엔진오일 품질과 브레이크 패드는 정상 차량도 단조 감소한다(소모품).
  - 배터리 전압은 주행 중과 정차 중이 1V 가까이 다르다.
그래서 이 코드는 관리도를 걸기 전에 두 가지를 먼저 한다.
  1. 운전 조건 분리 : 정상상태 주행 구간만 남긴다.
  2. 잔차 변환     : 물리적으로 예상되는 값을 회귀로 뺀 잔차를 쓴다.

중요: 관리한계(UCL/LCL)는 '정상으로 확인된 기준 구간'에서만 계산한다.
전체 데이터로 계산하면 고장 구간이 기준선을 끌어올려 이상을 못 잡는다.

numpy / pandas 만 사용한다.
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np
import pandas as pd


class ControlLimits:
    """기준 구간에서 적합한 관리한계."""

    def __init__(self, center: float, sigma: float, k: float = 3.0) -> None:
        self.center = center
        self.sigma = sigma
        self.k = k
        self.ucl = center + k * sigma
        self.lcl = center - k * sigma

    def __repr__(self) -> str:
        return (f"center={self.center:.4f} sigma={self.sigma:.4f} "
                f"LCL={self.lcl:.4f} UCL={self.ucl:.4f}")


def fit_limits(baseline: pd.Series, k: float = 3.0,
               robust: bool = True) -> ControlLimits:
    """기준 구간에서 중심선과 표준편차를 추정한다.

    robust=True 면 중앙값과 MAD 로 추정한다. 기준 구간에 스파이크가
    조금 섞여 있어도 한계선이 부풀지 않는다.
    """
    s = baseline.dropna()
    if len(s) < 10:
        raise ValueError(f"기준 구간이 너무 짧다: {len(s)}개")
    if robust:
        center = float(s.median())
        mad = float((s - center).abs().median())
        sigma = mad / 0.6745 if mad > 0 else float(s.std(ddof=0))
    else:
        center = float(s.mean())
        sigma = float(s.std(ddof=0))
    return ControlLimits(center, max(sigma, 1e-9), k)


# ----------------------------------------------------------------------
# 전처리: 정상상태 구간 추출과 잔차 변환
# ----------------------------------------------------------------------
def steady_state(df: pd.DataFrame, warmup: int = 60,
                 min_speed: float = 20.0) -> pd.DataFrame:
    """정상상태 주행 구간만 남긴다.

    warmup   : 앞쪽 n 행을 버린다(냉각수 웜업 구간)
    min_speed: 이 속도 미만은 정차/아이들로 보고 버린다
    관리도는 '같은 운전 조건의 반복'을 전제하므로 조건을 먼저 맞춘다.
    """
    out = df.iloc[warmup:]
    return out[out["speed_kph"] >= min_speed].reset_index(drop=True)


def _design(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """설계행렬 [설명변수..., 1] 을 만든다."""
    return np.column_stack([df[c].to_numpy(dtype=float) for c in cols]
                           + [np.ones(len(df))])


def fit_reference(ref: pd.DataFrame, target: str,
                  drivers: list[str]) -> np.ndarray:
    """정상 차량 집단(fleet)에서 '부하 대비 신호' 관계를 적합한다.

    예: engine_temp_c ~ engine_rpm + speed_kph
    차량 자신의 앞 구간을 기준으로 삼으면, 이미 열화가 시작된 차량은
    그 열화가 기준선에 포함되어 버린다(기준 구간 오염). 그러면 표준편차가
    부풀어 정작 고장 구간을 못 잡는다. 그래서 정상으로 확인된 다른
    차량들에서 관계를 학습해 공통 기준으로 쓴다.
    """
    cols = [c for c in drivers if c in ref.columns]
    x = _design(ref, cols)
    y = ref[target].to_numpy(dtype=float)
    ok = np.isfinite(x).all(axis=1) & np.isfinite(y)
    if ok.sum() < len(cols) + 10:
        raise ValueError(f"{target}: 기준 차량 표본이 부족하다")
    coef, *_ = np.linalg.lstsq(x[ok], y[ok], rcond=None)
    return coef


def residual_against(df: pd.DataFrame, target: str,
                     drivers: list[str],
                     coef: np.ndarray) -> pd.Series:
    """적합된 기준 관계에서 벗어난 만큼(잔차)을 돌려준다.

    부하가 높아서 뜨거운 것과 고장이라 뜨거운 것을 이 단계에서 구분한다.
    """
    cols = [c for c in drivers if c in df.columns]
    pred = _design(df, cols) @ coef
    return pd.Series(df[target].to_numpy(dtype=float) - pred,
                     index=df.index, name=f"{target}_resid")


# ----------------------------------------------------------------------
# 관리도
# ----------------------------------------------------------------------
def sigma_rule(s: pd.Series, lim: ControlLimits) -> pd.Series:
    """3시그마 규칙. 한 점이라도 한계를 벗어나면 이상으로 표시한다."""
    return (s > lim.ucl) | (s < lim.lcl)


def ewma_stat(s: pd.Series, lim: ControlLimits,
              lam: float = 0.2) -> pd.Series:
    """EWMA 통계량(중심선에서 떨어진 표준화 거리).

    z_t = lam * x_t + (1 - lam) * z_{t-1}
    이론상 EWMA 의 정상상태 분산은 원 데이터의 lam/(2-lam) 배이므로
    관리한계도 그만큼 좁아지고, 작은 평균 이동을 3시그마보다 빨리 잡는다.
    다만 차량 센서 잔차는 자기상관(autocorrelation)이 강해 실제 분산이
    이론값보다 크다. 그래서 임계값은 공식이 아니라 정상 차량 집단에서
    보정한다(calibrate_threshold).
    lam 이 1 이면 3시그마 규칙과 같아지고, 작을수록 작은 변화에 민감하다.
    """
    z = s.ewm(alpha=lam, adjust=False).mean()
    return (z - lim.center).abs() / lim.sigma


def cusum_stat(s: pd.Series, lim: ControlLimits,
               k: float = 0.5) -> pd.Series:
    """표준화 CUSUM 통계량.

    C+_t = max(0, C+_{t-1} + z_t - k)
    C-_t = max(0, C-_{t-1} - z_t - k)
    k 는 허용 여유(slack)다. 지속적인 작은 편차를 누적하므로
    '완만한 열화'에 가장 민감하다. 한 번 넘으면 계속 경보가 유지되므로
    경보율이 아니라 첫 경보 시점으로 평가해야 한다.
    """
    z = ((s - lim.center) / lim.sigma).fillna(0.0).to_numpy()
    cp = np.zeros(len(z))
    cm = np.zeros(len(z))
    for i in range(1, len(z)):
        cp[i] = max(0.0, cp[i - 1] + z[i] - k)
        cm[i] = max(0.0, cm[i - 1] - z[i] - k)
    return pd.Series(np.maximum(cp, cm), index=s.index)


def calibrate_threshold(stats: list[pd.Series],
                        target_fpr: float = 0.01) -> float:
    """정상 차량들의 통계량에서 목표 오경보율에 맞는 임계값을 역산한다.

    교과서 공식은 관측값이 서로 독립이라고 가정하므로 센서 데이터에
    그대로 쓰면 정상 차량에서도 경보율이 수십 %까지 올라간다.
    대신 '정상으로 확인된 차량들에서 목표 오경보율(예: 1%)이 나오는
    분위수'를 임계값으로 쓰면, 가정이 깨져도 오경보율이 설계값에 묶인다.
    """
    pool = pd.concat(stats).dropna().to_numpy()
    if len(pool) < 50:
        raise ValueError("임계값을 정하기에 기준 표본이 부족하다")
    return float(np.quantile(pool, 1.0 - target_fpr))


def trend_life(s: pd.Series, floor: float,
               baseline: slice) -> tuple[float, float]:
    """소모품의 잔존수명을 추세로 추정한다(관리도 대신 쓴다).

    엔진오일/브레이크 패드처럼 정상 차량도 단조 감소하는 신호는
    '벗어남'이 아니라 '감소 속도'가 문제다. 기준 구간의 감소율과
    현재 감소율을 비교하고, floor 에 닿을 때까지 남은 스텝을 돌려준다.
    반환: (기준 대비 감소율 배수, floor 도달까지 남은 스텝)
    """
    y = s.to_numpy(dtype=float)
    x = np.arange(len(y), dtype=float)

    def _slope(xs: np.ndarray, ys: np.ndarray) -> float:
        ok = np.isfinite(ys)
        if ok.sum() < 3:
            return 0.0
        xc = xs[ok] - xs[ok].mean()
        d = float((xc ** 2).sum())
        return float((xc * (ys[ok] - ys[ok].mean())).sum() / d) if d else 0.0

    base = _slope(x[baseline], y[baseline])
    tail = slice(max(0, len(y) - len(y) // 4), None)
    now = _slope(x[tail], y[tail])
    ratio = (now / base) if abs(base) > 1e-12 else float("inf")
    remain = ((y[-1] - floor) / -now) if now < -1e-12 else float("inf")
    return ratio, remain


# ----------------------------------------------------------------------
# 분석
# ----------------------------------------------------------------------
def first_alarm(mask: pd.Series) -> int | None:
    """첫 경보가 발생한 인덱스 위치를 돌려준다(없으면 None)."""
    idx = np.flatnonzero(mask.to_numpy())
    return int(idx[0]) if len(idx) else None


# 관리도를 적용할 (신호, 설명변수) 조합
RESIDUAL_SPEC: dict[str, list[str]] = {
    "engine_temp_c": ["engine_rpm", "speed_kph"],
    "battery_voltage": ["engine_rpm"],
    "vibration_rms": ["speed_kph", "engine_rpm"],
}
# 관리도 대신 추세로 보는 소모품 (신호 -> 교환 하한)
CONSUMABLE: dict[str, float] = {
    "oil_quality": 0.20,
    "brake_pad_mm": 3.0,
    "battery_soh": 0.70,
}


def analyze_vehicle(df: pd.DataFrame, vid: int, scenario: str,
                    ref_coef: dict[str, np.ndarray],
                    ref_lim: dict[str, ControlLimits],
                    ref_thr: dict[str, float],
                    ref_thr_cusum: dict[str, float],
                    baseline_frac: float = 0.3,
                    interval_min: int = 10) -> list[dict]:
    """한 차량에 대해 잔차 관리도와 소모품 추세를 모두 적용한다."""
    ss = steady_state(df)
    base = slice(0, int(len(ss) * baseline_frac))

    print(f"\n{'=' * 74}")
    print(f"vehicle_id={vid}  시나리오={scenario}  "
          f"원본 {len(df)}행 -> 정상상태 {len(ss)}행")
    print("=" * 74)

    rows: list[dict] = []
    print(f"\n  [잔차 관리도]  잔차 = 실측 - (정상 차량 집단이 예측한 값)")
    print(f"  {'신호':<18}{'공통 관리한계':<34}"
          f"{'3시그마율':>9}{'EWMA최대':>9}{'CUSUM최대':>9}{'첫경보':>8}")
    for target, drivers in RESIDUAL_SPEC.items():
        resid = residual_against(ss, target, drivers, ref_coef[target])
        lim = ref_lim[target]

        m_sig = sigma_rule(resid, lim)
        s_ewma = ewma_stat(resid, lim, lam=0.2)
        s_cus = cusum_stat(resid, lim, k=0.5)
        m_ewma = s_ewma > ref_thr[target]
        m_cus = s_cus > ref_thr_cusum[target]

        f = first_alarm(m_ewma)
        lead = (f"{(len(ss) - f) * interval_min / 60:.0f}h"
                if f is not None else "-")
        n = len(ss)
        print(f"  {target:<18}{str(lim):<34}"
              f"{m_sig.sum() / n * 100:>8.1f}%{s_ewma.max():>9.1f}"
              f"{s_cus.max():>9.0f}{lead:>8}")
        rows.append({
            "vehicle_id": vid, "scenario": scenario, "signal": target,
            "n": n,
            "sigma_rate": round(m_sig.sum() / n, 4),
            "ewma_rate": round(m_ewma.sum() / n, 4),
            "cusum_rate": round(m_cus.sum() / n, 4),
            "ewma_peak": round(float(s_ewma.max()), 2),
            "cusum_peak": round(float(s_cus.max()), 1),
            "ewma_first": f,
            "ewma_lead_h": (None if f is None
                            else round((len(ss) - f) * interval_min / 60, 1)),
        })

    print(f"\n  [소모품 추세]  관리도를 쓰지 않고 감소 속도로 판정한다")
    print(f"  {'신호':<18}{'현재값':>10}{'교환하한':>10}"
          f"{'감소율 배수':>12}{'잔존 스텝':>12}{'잔존 일수':>12}")
    for sig_name, floor in CONSUMABLE.items():
        if sig_name not in df.columns:
            continue
        s = df[sig_name].reset_index(drop=True)
        nb = int(len(s) * baseline_frac)
        ratio, remain = trend_life(s, floor, slice(0, nb))
        days = remain * interval_min / 60 / 24
        print(f"  {sig_name:<18}{s.iloc[-1]:>10.3f}{floor:>10.2f}"
              f"{ratio:>12.2f}"
              f"{(f'{remain:,.0f}' if np.isfinite(remain) else '-'):>12}"
              f"{(f'{days:,.1f}' if np.isfinite(days) else '-'):>12}")
        rows.append({
            "vehicle_id": vid, "scenario": scenario, "signal": sig_name,
            "n": len(s), "sigma_rate": None, "ewma_rate": None,
            "cusum_rate": None, "ewma_first": None,
            "ewma_lead_h": (None if not np.isfinite(remain)
                            else round(remain * interval_min / 60, 1)),
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="관리도 기반 이상 탐지")
    ap.add_argument("--csv", default="data/sensor_clean.csv")
    ap.add_argument("--out", default="data/stat_scores.csv")
    args = ap.parse_args()

    path = pathlib.Path(args.csv)
    if not path.exists():
        print(f"[오류] 입력 CSV 가 없다: {path}")
        print("  python 02_data/make_sensor_dataset.py --defects "
              "--out data/sensor_readings_raw.csv")
        print("  python 04_preprocess/clean.py")
        return

    df = pd.read_csv(path)
    scenario = {1: "normal", 2: "normal", 3: "degrade", 4: "failure"}
    # 정상으로 확인된 차량만 기준(reference)으로 쓴다.
    normal_ids = [v for v, s in scenario.items() if s == "normal"]

    # --- 1단계: 정상 차량 집단에서 공통 기준을 적합한다 ---------------
    ref_ss = {vid: steady_state(
        df[df["vehicle_id"] == vid].reset_index(drop=True))
        for vid in normal_ids}
    ref_pool = pd.concat(ref_ss.values(), ignore_index=True)

    print(f"[기준 모형 적합]  정상 차량 {normal_ids} 의 "
          f"정상상태 {len(ref_pool):,}행 사용")
    ref_coef: dict[str, np.ndarray] = {}
    ref_lim: dict[str, ControlLimits] = {}
    ref_thr: dict[str, float] = {}
    ref_thr_cusum: dict[str, float] = {}
    print(f"  {'신호':<18}{'설명변수':<28}{'공통 관리한계':<34}"
          f"{'EWMA임계':>10}{'CUSUM임계':>10}")
    for target, drivers in RESIDUAL_SPEC.items():
        coef = fit_reference(ref_pool, target, drivers)
        ref_coef[target] = coef
        resid_pool = residual_against(ref_pool, target, drivers, coef)
        lim = fit_limits(resid_pool, k=3.0, robust=True)
        ref_lim[target] = lim
        # 차량별로 통계량을 만들어 모은 뒤 분위수로 임계값을 정한다.
        e_stats, c_stats = [], []
        for g in ref_ss.values():
            r = residual_against(g, target, drivers, coef)
            e_stats.append(ewma_stat(r, lim, lam=0.2))
            c_stats.append(cusum_stat(r, lim, k=0.5))
        ref_thr[target] = calibrate_threshold(e_stats, target_fpr=0.01)
        ref_thr_cusum[target] = calibrate_threshold(c_stats, target_fpr=0.01)
        print(f"  {target:<18}{'+'.join(drivers):<28}{str(lim):<34}"
              f"{ref_thr[target]:>10.2f}{ref_thr_cusum[target]:>10.1f}")

    # --- 2단계: 모든 차량을 공통 기준으로 평가한다 --------------------
    rows: list[dict] = []
    for vid, g in df.groupby("vehicle_id"):
        rows += analyze_vehicle(g.reset_index(drop=True), int(vid),
                                scenario.get(int(vid), "?"),
                                ref_coef, ref_lim,
                                ref_thr, ref_thr_cusum)

    summary = pd.DataFrame(rows)
    ctrl = summary[summary["ewma_rate"].notna()]

    order_all = ("normal", "degrade", "failure")
    print(f"\n\n{'=' * 74}")
    print("[시나리오별 통계량 최댓값]  차량 단위 판정에 쓰는 값")
    print("=" * 74)
    for metric, label in (("ewma_peak", "EWMA 최대"),
                          ("cusum_peak", "CUSUM 최대")):
        piv = ctrl.pivot_table(index="signal", columns="scenario",
                               values=metric, aggfunc="max")
        order = [c for c in order_all if c in piv.columns]
        print(f"\n  [{label}]")
        print(f"  {'신호':<18}" + "".join(f"{c:>12}" for c in order))
        for sig_name, r in piv.iterrows():
            print(f"  {sig_name:<18}"
                  + "".join(f"{r[c]:>12,.1f}" for c in order))

    print("\n  세 가지 장치를 모두 넣은 결과다.")
    print("   1) 정상상태 구간만 남겨 웜업/정차를 제외했다.")
    print("   2) 부하 효과를 잔차로 제거해 '부하가 높아 뜨거운 것'과")
    print("      '고장이라 뜨거운 것'을 분리했다.")
    print("   3) 기준선과 임계값을 차량 자신이 아니라 정상 차량 집단에서")
    print("      구했다. 자기 앞 구간을 기준으로 삼으면 이미 열화된 차량의")
    print("      열화가 기준선에 섞여 들어가 정작 고장을 못 잡는다.")
    print("\n  그 결과 통계량 최댓값이 normal < degrade < failure 순서로")
    print("  단조 증가하며 자릿수 단위로 벌어진다. 반면 행 단위 경보율은")
    print("  한 번 넘으면 계속 유지되는 CUSUM 특성 때문에 포화되므로,")
    print("  차량 단위 위험도는 '구간 최댓값'으로 정하는 편이 안정적이다.")
    print("  이렇게 만든 통계량을 5.5(9) 에서 0~1 이상 점수로 정규화한다.")

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out, index=False, encoding="utf-8")
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
