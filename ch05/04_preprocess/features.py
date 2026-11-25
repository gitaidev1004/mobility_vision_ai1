# -*- coding: utf-8 -*-
"""예지보전용 특징(feature) 생성.

정제된 센서 시계열에서 모델 입력을 만든다.
  1. 스케일링   : Min-Max / Standard / Robust
  2. 이동평균   : 단순(SMA) / 지수(EMA)
  3. Rolling 통계: 평균 / 표준편차 / 최대 / 최소 / 범위
  4. 변화율     : 1차 차분, 기울기(선형회귀 계수)
  5. Window     : (N, T, F) 형태의 시계열 윈도우 텐서

중요: 스케일러의 통계량은 학습 구간에서만 계산해야 한다. 전체 데이터로
계산하면 미래 정보가 학습에 섞이는 데이터 누수(data leakage)가 된다.

numpy / pandas 만 사용한다.
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np
import pandas as pd

# 특징을 만들 대상 센서
FEATURE_COLS = [
    "engine_temp_c", "engine_rpm", "speed_kph",
    "battery_voltage", "battery_soh",
    "tire_pressure_fl", "tire_pressure_fr",
    "tire_pressure_rl", "tire_pressure_rr",
    "brake_pad_mm", "oil_quality", "vibration_rms",
]


# ----------------------------------------------------------------------
# 1. 스케일링
# ----------------------------------------------------------------------
class Scaler:
    """학습 구간에서 통계량을 적합(fit)하고 재사용하는 스케일러.

    kind = "minmax"   : (x - min) / (max - min)        -> 0~1
           "standard" : (x - mean) / std                -> 평균0 분산1
           "robust"   : (x - median) / IQR              -> 이상값에 둔감
    """

    def __init__(self, kind: str = "standard") -> None:
        if kind not in {"minmax", "standard", "robust"}:
            raise ValueError(f"알 수 없는 스케일러: {kind}")
        self.kind = kind
        self.center_: pd.Series | None = None
        self.scale_: pd.Series | None = None

    def fit(self, df: pd.DataFrame) -> "Scaler":
        if self.kind == "minmax":
            self.center_ = df.min()
            self.scale_ = (df.max() - df.min())
        elif self.kind == "standard":
            self.center_ = df.mean()
            self.scale_ = df.std(ddof=0)
        else:
            self.center_ = df.median()
            self.scale_ = df.quantile(0.75) - df.quantile(0.25)
        # 분모가 0 이면 상수 컬럼이다. 1 로 바꿔 0 나눗셈을 막는다.
        self.scale_ = self.scale_.replace(0, 1.0).fillna(1.0)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.center_ is None:
            raise RuntimeError("fit 을 먼저 호출해야 한다")
        return (df - self.center_) / self.scale_

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return df * self.scale_ + self.center_


# ----------------------------------------------------------------------
# 2. 이동평균
# ----------------------------------------------------------------------
def moving_average(s: pd.Series, window: int = 12) -> pd.Series:
    """단순 이동평균(SMA). 창 안의 값을 같은 가중으로 평균한다."""
    return s.rolling(window=window, min_periods=1).mean()


def exp_moving_average(s: pd.Series, span: int = 12) -> pd.Series:
    """지수 이동평균(EMA). 최근 값에 더 큰 가중을 준다.

    alpha = 2 / (span + 1) 이고, 최근 값의 가중이 지수적으로 감쇠한다.
    고장 징후처럼 '최근 변화'가 중요할 때 SMA 보다 반응이 빠르다.
    """
    return s.ewm(span=span, adjust=False).mean()


# ----------------------------------------------------------------------
# 3. Rolling 통계
# ----------------------------------------------------------------------
def rolling_features(df: pd.DataFrame, cols: list[str],
                     window: int = 12,
                     group_col: str | None = "vehicle_id") -> pd.DataFrame:
    """컬럼별 rolling 평균/표준편차/최대/최소/범위를 만든다.

    window=12 는 10분 간격 데이터에서 2시간 창에 해당한다.
    표준편차는 '불안정해지는 징후', 최대는 '피크 충격'을 잡는다.
    """
    out = pd.DataFrame(index=df.index)

    def _apply(g: pd.DataFrame) -> pd.DataFrame:
        res = {}
        for c in cols:
            r = g[c].rolling(window=window, min_periods=2)
            res[f"{c}_mean{window}"] = r.mean()
            res[f"{c}_std{window}"] = r.std(ddof=0)
            res[f"{c}_max{window}"] = r.max()
            res[f"{c}_min{window}"] = r.min()
            res[f"{c}_rng{window}"] = r.max() - r.min()
        return pd.DataFrame(res, index=g.index)

    if group_col and group_col in df.columns:
        parts = [_apply(g) for _, g in df.groupby(group_col, sort=False)]
        out = pd.concat(parts).reindex(df.index)
    else:
        out = _apply(df)
    return out


def rate_features(df: pd.DataFrame, cols: list[str],
                  window: int = 12,
                  group_col: str | None = "vehicle_id") -> pd.DataFrame:
    """변화율 특징: 1차 차분과 rolling 선형 기울기.

    차분은 '직전 대비 얼마나 튀었는가', 기울기는 '추세가 어디로 가는가'를
    나타낸다. 열화는 느린 추세이므로 기울기가 결정적인 신호가 된다.
    """
    # 창 안에서의 최소제곱 기울기. x 를 0..w-1 로 고정하면 가중 합으로 계산된다.
    w = window
    x = np.arange(w, dtype=float)
    x_c = x - x.mean()
    denom = float((x_c ** 2).sum())

    def _slope(a: np.ndarray) -> float:
        if len(a) < 2 or np.isnan(a).any():
            return np.nan
        xc = np.arange(len(a), dtype=float)
        xc = xc - xc.mean()
        d = float((xc ** 2).sum())
        return float((xc * a).sum() / d) if d else np.nan

    def _apply(g: pd.DataFrame) -> pd.DataFrame:
        res = {}
        for c in cols:
            res[f"{c}_diff"] = g[c].diff()
            res[f"{c}_pct"] = g[c].pct_change().replace(
                [np.inf, -np.inf], np.nan)
            res[f"{c}_slope{w}"] = g[c].rolling(
                window=w, min_periods=3).apply(_slope, raw=True)
        return pd.DataFrame(res, index=g.index)

    if group_col and group_col in df.columns:
        parts = [_apply(g) for _, g in df.groupby(group_col, sort=False)]
        return pd.concat(parts).reindex(df.index)
    return _apply(df)


# ----------------------------------------------------------------------
# 4. 시계열 윈도우
# ----------------------------------------------------------------------
def make_windows(arr: np.ndarray, seq_len: int = 24,
                 stride: int = 1, horizon: int = 0
                 ) -> tuple[np.ndarray, np.ndarray]:
    """2차원 배열 (T, F) 을 3차원 윈도우 (N, seq_len, F) 로 자른다.

    horizon > 0 이면 윈도우 끝에서 horizon 스텝 뒤의 값을 y 로 함께 돌려준다.
    (고장 예측/RUL 회귀에서 이 y 가 정답이 된다)
    stride 를 키우면 윈도우가 덜 겹쳐 샘플 수가 줄고 중복이 감소한다.
    """
    if arr.ndim != 2:
        raise ValueError(f"(T, F) 2차원 배열이 필요하다: {arr.shape}")
    t, f = arr.shape
    last = t - seq_len - horizon
    if last < 0:
        raise ValueError(
            f"데이터 길이 {t} 가 seq_len({seq_len}) + horizon({horizon}) 보다 짧다")
    starts = range(0, last + 1, stride)
    xs = np.stack([arr[s:s + seq_len] for s in starts])
    if horizon > 0:
        ys = np.stack([arr[s + seq_len + horizon - 1] for s in starts])
    else:
        ys = np.empty((len(xs), 0), dtype=arr.dtype)
    return xs, ys


def windows_by_vehicle(df: pd.DataFrame, cols: list[str],
                       seq_len: int = 24, stride: int = 4,
                       group_col: str = "vehicle_id"
                       ) -> tuple[np.ndarray, np.ndarray]:
    """차량 경계를 넘지 않도록 차량별로 윈도우를 만든 뒤 합친다.

    차량 경계를 무시하고 전체를 한 줄로 자르면 A 차량 끝과 B 차량 시작이
    한 윈도우에 섞여 존재하지 않은 운행 패턴이 만들어진다.
    """
    xs, vids = [], []
    for vid, g in df.groupby(group_col, sort=True):
        arr = g[cols].to_numpy(dtype=np.float32)
        if len(arr) < seq_len:
            continue
        x, _ = make_windows(arr, seq_len=seq_len, stride=stride)
        xs.append(x)
        vids.append(np.full(len(x), vid, dtype=np.int32))
    if not xs:
        raise ValueError("윈도우를 만들 수 있는 차량이 없다")
    return np.concatenate(xs), np.concatenate(vids)


def build_feature_table(df: pd.DataFrame, window: int = 12) -> pd.DataFrame:
    """원본 센서 + rolling 통계 + 변화율을 모두 붙인 특징 테이블."""
    cols = [c for c in FEATURE_COLS if c in df.columns]
    roll = rolling_features(df, cols, window=window)
    rate = rate_features(df, cols, window=window)
    keep = [c for c in ("vehicle_id", "measured_at", "odometer_km")
            if c in df.columns]
    out = pd.concat([df[keep + cols], roll, rate], axis=1)
    # 창 초반의 NaN 은 뒤쪽 값으로 채운다(차량별).
    num = out.select_dtypes(include=[np.number]).columns
    if "vehicle_id" in out.columns:
        out[num] = out.groupby("vehicle_id")[list(num)].transform(
            lambda s: s.bfill().ffill())
    else:
        out[num] = out[num].bfill().ffill()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="예지보전 특징 생성")
    ap.add_argument("--csv", default="data/sensor_clean.csv")
    ap.add_argument("--out", default="data/features.csv")
    ap.add_argument("--window", type=int, default=12)
    ap.add_argument("--seq-len", type=int, default=24)
    ap.add_argument("--stride", type=int, default=4)
    args = ap.parse_args()

    path = pathlib.Path(args.csv)
    if not path.exists():
        print(f"[오류] 입력 CSV 가 없다: {path}")
        print("  python 02_data/make_sensor_dataset.py --defects "
              "--out data/sensor_readings_raw.csv")
        print("  python 04_preprocess/clean.py")
        return

    df = pd.read_csv(path)
    cols = [c for c in FEATURE_COLS if c in df.columns]
    print(f"입력 {len(df):,}행, 센서 {len(cols)}개")

    # --- 1. 스케일링 비교 ------------------------------------------------
    # 학습/검증 분할은 시간 순서를 지킨다(무작위 분할 금지).
    n_train = int(len(df) * 0.7)
    train, valid = df.iloc[:n_train], df.iloc[n_train:]
    print(f"\n[시간순 분할] 학습 {len(train):,}행 / 검증 {len(valid):,}행")

    print(f"\n[스케일러별 변환 결과: engine_temp_c]")
    print(f"  {'스케일러':<12}{'학습 평균':>10}{'학습 표준편차':>14}"
          f"{'검증 평균':>10}{'검증 표준편차':>14}")
    for kind in ["minmax", "standard", "robust"]:
        sc = Scaler(kind).fit(train[cols])
        tr = sc.transform(train[cols])["engine_temp_c"]
        va = sc.transform(valid[cols])["engine_temp_c"]
        print(f"  {kind:<12}{tr.mean():>10.4f}{tr.std():>14.4f}"
              f"{va.mean():>10.4f}{va.std():>14.4f}")
    print("  검증 평균이 0 에서 벗어난 것은 분포 이동(drift)이 있다는 뜻이다.")

    # --- 2. 이동평균 비교 ------------------------------------------------
    v4 = df[df["vehicle_id"] == 4].reset_index(drop=True)
    s = v4["vibration_rms"]
    sma = moving_average(s, 12)
    ema = exp_moving_average(s, 12)
    print(f"\n[이동평균 비교: vehicle_id=4 vibration_rms]")
    print(f"  {'idx':>5}{'원본':>10}{'SMA(12)':>10}{'EMA(12)':>10}")
    for i in [0, 100, 300, 500, 650, 700, 719]:
        print(f"  {i:>5}{s.iloc[i]:>10.4f}{sma.iloc[i]:>10.4f}"
              f"{ema.iloc[i]:>10.4f}")
    print(f"  원본 표준편차 {s.std():.4f} -> SMA {sma.std():.4f}"
          f" / EMA {ema.std():.4f}")

    # --- 3. 특징 테이블 --------------------------------------------------
    feat = build_feature_table(df, window=args.window)
    print(f"\n[특징 테이블] {len(feat):,}행 x {len(feat.columns)}열 "
          f"(센서 {len(cols)} -> 특징 {len(feat.columns)})")
    print(f"  잔여 결측 {int(feat.isna().sum().sum())}개")
    sample = ["engine_temp_c", f"engine_temp_c_mean{args.window}",
              f"engine_temp_c_std{args.window}",
              f"engine_temp_c_slope{args.window}",
              "vibration_rms", f"vibration_rms_max{args.window}"]
    sample = [c for c in sample if c in feat.columns]
    print("\n  [vehicle_id=4 후반 5행]")
    print(feat[feat["vehicle_id"] == 4][sample].tail(5).to_string(index=False))

    # --- 4. 윈도우 -------------------------------------------------------
    sc = Scaler("standard").fit(feat.iloc[:n_train][cols])
    scaled = feat.copy()
    scaled[cols] = sc.transform(feat[cols])
    x, vid = windows_by_vehicle(scaled, cols, seq_len=args.seq_len,
                                stride=args.stride)
    print(f"\n[시계열 윈도우] seq_len={args.seq_len} stride={args.stride}")
    print(f"  X shape = {x.shape}  (샘플 수, 시퀀스 길이, 특징 수)")
    print(f"  dtype = {x.dtype}  메모리 = {x.nbytes / 1024 / 1024:.2f} MB")
    for v in np.unique(vid):
        print(f"    vehicle_id={v}: {int((vid == v).sum())}개 윈도우")

    print("\n[stride 별 샘플 수와 중복률]")
    print(f"  {'stride':<8}{'샘플 수':>10}{'중복 비율':>12}")
    for st in [1, 2, 4, 8, 24]:
        xx, _ = windows_by_vehicle(scaled, cols, seq_len=args.seq_len,
                                   stride=st)
        overlap = max(0.0, 1.0 - st / args.seq_len)
        print(f"  {st:<8}{len(xx):>10,}{overlap * 100:>11.1f}%")

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    feat.to_csv(out, index=False, encoding="utf-8")
    np.save(out.with_name("windows_x.npy"), x)
    print(f"\n저장: {out}")
    print(f"저장: {out.with_name('windows_x.npy')}")


if __name__ == "__main__":
    main()
