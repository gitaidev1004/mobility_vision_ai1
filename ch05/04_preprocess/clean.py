# -*- coding: utf-8 -*-
"""센서 데이터의 결측값과 이상값을 처리한다.

처리 순서가 중요하다.
  1. 물리 범위 위반 -> NaN 으로 전환 (물리적으로 불가능한 값은 정보가 아니다)
  2. 통계적 이상값 탐지(IQR / z-score) -> NaN 으로 전환
  3. 결측값 보간(선형보간 -> 전방 채움 -> 후방 채움)
  4. 잔여 결측 확인

반대 순서로 하면 스파이크가 보간에 섞여 들어가 주변 값을 오염시킨다.
numpy / pandas 만 사용한다.
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np
import pandas as pd

# 센서별 물리적으로 가능한 범위 (OBD-II 표준 + 차량 명세)
PHYSICAL_RANGE: dict[str, tuple[float, float]] = {
    "engine_temp_c": (-40.0, 215.0),
    "engine_rpm": (0.0, 16383.0),
    "speed_kph": (0.0, 255.0),
    "battery_voltage": (0.0, 20.0),
    "battery_soh": (0.0, 1.0),
    "tire_pressure_fl": (0.0, 60.0),
    "tire_pressure_fr": (0.0, 60.0),
    "tire_pressure_rl": (0.0, 60.0),
    "tire_pressure_rr": (0.0, 60.0),
    "brake_pad_mm": (0.0, 15.0),
    "oil_quality": (0.0, 1.0),
    "vibration_rms": (0.0, 10.0),
    "fuel_level": (0.0, 1.0),
}

SENSOR_COLS = list(PHYSICAL_RANGE.keys())


def drop_physical_outliers(df: pd.DataFrame,
                           cols: list[str] | None = None) -> tuple[pd.DataFrame, dict]:
    """물리 범위를 벗어난 값을 NaN 으로 바꾼다."""
    out = df.copy()
    cols = cols or [c for c in SENSOR_COLS if c in out.columns]
    report: dict[str, int] = {}
    for c in cols:
        lo, hi = PHYSICAL_RANGE[c]
        mask = out[c].notna() & ((out[c] < lo) | (out[c] > hi))
        report[c] = int(mask.sum())
        out.loc[mask, c] = np.nan
    return out, report


def detect_iqr(s: pd.Series, k: float = 1.5) -> pd.Series:
    """IQR 방식 이상값 마스크. 분포가 치우쳐도 비교적 안전하다.

    Q1 - k*IQR 보다 작거나 Q3 + k*IQR 보다 크면 이상값으로 본다.
    k=1.5 가 통상값이고, 센서처럼 꼬리가 긴 데이터는 2.5~3.0 을 쓴다.
    """
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0 or np.isnan(iqr):
        return pd.Series(False, index=s.index)
    return (s < q1 - k * iqr) | (s > q3 + k * iqr)


def detect_zscore(s: pd.Series, thr: float = 3.0) -> pd.Series:
    """z-score 방식 이상값 마스크. 정규분포 가정이 필요하다."""
    mu, sd = s.mean(), s.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return pd.Series(False, index=s.index)
    return ((s - mu).abs() / sd) > thr


def detect_robust_zscore(s: pd.Series, thr: float = 3.5) -> pd.Series:
    """MAD 기반 수정 z-score. 이상값이 많아도 기준선이 흔들리지 않는다.

    수정 z = 0.6745 * (x - median) / MAD
    """
    med = s.median()
    mad = (s - med).abs().median()
    if mad == 0 or np.isnan(mad):
        return pd.Series(False, index=s.index)
    return (0.6745 * (s - med).abs() / mad) > thr


def detect_rolling_residual(s: pd.Series, window: int = 13,
                            thr: float = 6.0) -> pd.Series:
    """이동 중앙값 대비 잔차로 판정한다. 예지보전에 가장 적합한 방식이다.

    IQR / z-score 는 '전체 분포'를 기준으로 삼기 때문에 치명적 문제가 있다.
    열화가 진행된 차량의 후반 구간은 정상 운행 결과인데도 분포 꼬리에
    놓이므로 통째로 이상값 처리되고, 보간으로 덮이면서 고장 징후가 지워진다.
    반면 이 방식은 '국소 추세에서 얼마나 튀었는가' 만 보므로
    완만한 열화 추세는 보존하고 순간 스파이크만 제거한다.
    """
    med = s.rolling(window, center=True, min_periods=1).median()
    resid = s - med
    mad = resid.abs().rolling(window, center=True, min_periods=1).median()
    # 센서가 완전히 평탄한 구간에서 mad=0 이 되면 0 나눗셈이 되므로 보호한다.
    scale = mad.replace(0, np.nan)
    scale = scale.fillna(resid.abs().median()).replace(0, 1e-9)
    return (0.6745 * resid.abs() / scale) > thr


DETECTORS = {
    "iqr": detect_iqr,
    "zscore": detect_zscore,
    "robust": detect_robust_zscore,
    "rolling": detect_rolling_residual,
}


def drop_statistical_outliers(df: pd.DataFrame, method: str = "rolling",
                              cols: list[str] | None = None,
                              group_col: str | None = "vehicle_id",
                              **kw) -> tuple[pd.DataFrame, dict]:
    """통계적 이상값을 NaN 으로 바꾼다.

    차량마다 정상 범위가 다르므로 group_col 로 그룹별 판정한다.
    차량 전체를 한 분포로 보면 열화 차량의 정상값이 전부 이상값이 된다.
    """
    detector = DETECTORS[method]
    out = df.copy()
    cols = cols or [c for c in SENSOR_COLS if c in out.columns]
    report: dict[str, int] = {}
    for c in cols:
        if group_col and group_col in out.columns:
            mask = out.groupby(group_col)[c].transform(
                lambda s: detector(s, **kw))
            mask = mask.fillna(False).astype(bool)
        else:
            mask = detector(out[c], **kw)
        report[c] = int(mask.sum())
        out.loc[mask, c] = np.nan
    return out, report


def fill_missing(df: pd.DataFrame, cols: list[str] | None = None,
                 group_col: str | None = "vehicle_id",
                 limit: int = 6,
                 final_fill: bool = True) -> tuple[pd.DataFrame, dict]:
    """결측값을 보간한다.

    선형보간(interpolate) -> 전방 채움(ffill) -> 후방 채움(bfill) 3단계.
    limit 는 연속 결측을 몇 개까지 메울지 정한다. 긴 단절 구간을 억지로
    메우면 존재하지 않은 운행을 만들어내므로 상한을 둔다.
    final_fill=True 면 limit 를 넘긴 잔여 결측까지 무제한 ffill/bfill 로
    메운다. 모델 입력은 NaN 을 허용하지 않으므로 마지막 단계에서 필요하다.
    """
    out = df.copy()
    cols = cols or [c for c in SENSOR_COLS if c in out.columns]
    before = {c: int(out[c].isna().sum()) for c in cols}

    def _fill(s: pd.Series) -> pd.Series:
        s = s.interpolate(method="linear", limit=limit, limit_area="inside")
        s = s.ffill(limit=limit)
        s = s.bfill(limit=limit)
        if final_fill:
            s = s.ffill().bfill()
        return s

    for c in cols:
        if group_col and group_col in out.columns:
            out[c] = out.groupby(group_col)[c].transform(_fill)
        else:
            out[c] = _fill(out[c])

    after = {c: int(out[c].isna().sum()) for c in cols}
    report = {c: (before[c], after[c]) for c in cols if before[c] or after[c]}
    return out, report


def compare_detectors(df: pd.DataFrame, col: str = "engine_temp_c") -> None:
    """같은 컬럼에 세 가지 탐지기를 적용해 검출 수를 비교한다.

    차량 센서는 웜업 구간처럼 '정상인데 분포 꼬리에 있는' 값이 많다.
    IQR 과 z-score 는 이 구간을 통째로 이상값으로 몰아내므로,
    임계값을 올리거나 MAD 기반 수정 z-score 를 쓰는 편이 안전하다.
    """
    s = df[col]
    print(f"\n[탐지기 비교: {col}  (전체 {s.notna().sum():,}개 유효값)]")
    print(f"  {'기법':<22}{'임계':>8}{'검출수':>8}{'검출률':>9}")
    cases = [
        ("IQR", detect_iqr, {"k": 1.5}, 1.5),
        ("IQR", detect_iqr, {"k": 3.0}, 3.0),
        ("z-score", detect_zscore, {"thr": 3.0}, 3.0),
        ("수정 z-score(MAD)", detect_robust_zscore, {"thr": 3.5}, 3.5),
        ("수정 z-score(MAD)", detect_robust_zscore, {"thr": 6.0}, 6.0),
        ("이동 중앙값 잔차", detect_rolling_residual, {"thr": 6.0}, 6.0),
        ("이동 중앙값 잔차", detect_rolling_residual, {"thr": 10.0}, 10.0),
    ]
    n = max(s.notna().sum(), 1)
    for name, fn, kw, thr in cases:
        m = df.groupby("vehicle_id")[col].transform(
            lambda x: fn(x, **kw)).fillna(False).astype(bool)
        cnt = int(m.sum())
        print(f"  {name:<22}{thr:>8.1f}{cnt:>8}{cnt / n * 100:>8.1f}%")


def clean_pipeline(df: pd.DataFrame, method: str = "rolling",
                   k: float = 6.0, verbose: bool = True) -> pd.DataFrame:
    """전체 정제 파이프라인을 순서대로 적용한다."""
    cols = [c for c in SENSOR_COLS if c in df.columns]

    if verbose:
        print(f"입력: {len(df):,}행 x {len(df.columns)}열")
        na0 = df[cols].isna().sum().sum()
        print(f"  초기 결측 합계: {na0:,}개")

    step1, rep1 = drop_physical_outliers(df, cols)
    if verbose:
        tot = sum(rep1.values())
        print(f"\n[1단계] 물리 범위 위반 -> NaN : 총 {tot}개")
        for c, n in rep1.items():
            if n:
                lo, hi = PHYSICAL_RANGE[c]
                print(f"    {c:<18} {n:>4}개  (허용 {lo}~{hi})")

    kw: dict = {"k": k} if method == "iqr" else {"thr": k}
    step2, rep2 = drop_statistical_outliers(step1, method, cols, **kw)
    if verbose:
        tot = sum(rep2.values())
        print(f"\n[2단계] 통계 이상값({method}, 임계 {k}) -> NaN : 총 {tot}개")
        for c, n in sorted(rep2.items(), key=lambda x: -x[1])[:8]:
            if n:
                print(f"    {c:<18} {n:>4}개")

    step3, rep3 = fill_missing(step2, cols)
    if verbose:
        print("\n[3단계] 결측 보간 (선형 -> ffill -> bfill)")
        print(f"    {'컬럼':<18}{'보간전':>8}{'보간후':>8}")
        for c, (b, a) in sorted(rep3.items(), key=lambda x: -x[1][0])[:8]:
            print(f"    {c:<18}{b:>8}{a:>8}")
        left = step3[cols].isna().sum().sum()
        print(f"\n[결과] 잔여 결측 {left}개 / {len(step3) * len(cols):,}셀")

    return step3


def main() -> None:
    ap = argparse.ArgumentParser(description="센서 데이터 결측/이상값 처리")
    ap.add_argument("--csv", default="data/sensor_readings_raw.csv")
    ap.add_argument("--out", default="data/sensor_clean.csv")
    ap.add_argument("--method", default="rolling",
                    choices=["iqr", "zscore", "robust", "rolling"])
    ap.add_argument("--k", type=float, default=6.0,
                    help="IQR 배수 또는 z-score 임계값")
    args = ap.parse_args()

    path = pathlib.Path(args.csv)
    if not path.exists():
        print(f"[오류] 입력 CSV 가 없다: {path}")
        print("  먼저 결함이 주입된 데이터를 만든다.")
        print("  python 02_data/make_sensor_dataset.py --defects "
              "--out data/sensor_readings_raw.csv")
        return

    df = pd.read_csv(path)
    phys, _ = drop_physical_outliers(df)
    compare_detectors(phys, "engine_temp_c")
    compare_detectors(phys, "battery_voltage")

    print()
    clean = clean_pipeline(df, args.method, args.k)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(out, index=False, encoding="utf-8")
    print(f"\n저장: {out}")

    print("\n[정제 전후 통계 비교: engine_temp_c]")
    print(f"{'구분':<8}{'평균':>10}{'표준편차':>10}{'최소':>10}{'최대':>10}")
    for label, d in (("정제전", df), ("정제후", clean)):
        s = d["engine_temp_c"]
        print(f"{label:<8}{s.mean():>10.2f}{s.std():>10.2f}"
              f"{s.min():>10.2f}{s.max():>10.2f}")

    print("\n[정제 전후 통계 비교: battery_voltage]")
    print(f"{'구분':<8}{'평균':>10}{'표준편차':>10}{'최소':>10}{'최대':>10}")
    for label, d in (("정제전", df), ("정제후", clean)):
        s = d["battery_voltage"]
        print(f"{label:<8}{s.mean():>10.3f}{s.std():>10.3f}"
              f"{s.min():>10.3f}{s.max():>10.3f}")

    # 가장 중요한 점검: 정제가 고장 징후를 지우지 않았는지 확인한다.
    print("\n[고장 징후 보존 점검: vehicle_id=4 (failure 시나리오) 후반 10%]")
    print(f"  {'기법':<20}{'온도 최대':>10}{'전압 최소':>10}{'진동 최대':>10}")
    v4 = df[df["vehicle_id"] == 4]
    tail = slice(int(len(v4) * 0.9), None)
    s0 = v4.iloc[tail]
    print(f"  {'원본(결함 포함)':<20}{s0['engine_temp_c'].max():>10.1f}"
          f"{s0['battery_voltage'].min():>10.2f}"
          f"{s0['vibration_rms'].max():>10.3f}")
    for m, thr in (("iqr", 2.5), ("robust", 6.0), ("rolling", 6.0)):
        c = clean_pipeline(df, m, thr, verbose=False)
        g = c[c["vehicle_id"] == 4].iloc[tail]
        print(f"  {m + f'(임계 {thr})':<20}{g['engine_temp_c'].max():>10.1f}"
              f"{g['battery_voltage'].min():>10.2f}"
              f"{g['vibration_rms'].max():>10.3f}")
    print("\n  IQR 은 열화 후반의 고온/저전압을 이상값으로 보고 지워버린다.")
    print("  이동 중앙값 잔차 방식은 추세를 남기므로 징후가 보존된다.")


if __name__ == "__main__":
    main()
