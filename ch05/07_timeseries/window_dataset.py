# -*- coding: utf-8 -*-
"""시계열 윈도우 Dataset 구성.

5.4절 (9) 에서 만든 윈도우 생성 개념을 학습용 Dataset 으로 확장한다.
핵심은 세 가지 규칙이다.

  1) 윈도우는 **차량 경계를 넘지 않는다**. A 차량 끝 + B 차량 시작을
     한 윈도우에 넣으면 존재하지 않는 패턴을 학습한다.
  2) 라벨은 윈도우의 **마지막 시점** 라벨을 쓴다(과거→현재 예측).
  3) 표준화 파라미터는 학습 분할에서만 계산한다.

torch 가 없으면 numpy 배열만 만들어 shape 를 확인하고 종료한다.

실행:  python 07_timeseries/window_dataset.py
입력:  data/labeled_dataset.csv
"""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

DATA = pathlib.Path("data") / "labeled_dataset.csv"

SENSOR_COLS = [
    "engine_temp_c", "engine_rpm", "speed_kph", "battery_voltage",
    "battery_soh", "tire_pressure_fl", "tire_pressure_fr",
    "tire_pressure_rl", "tire_pressure_rr", "brake_pad_mm",
    "oil_quality", "vibration_rms", "fuel_level",
]
WINDOW = 24          # 과거 24일을 본다
STRIDE = 1           # 1일씩 밀어 윈도우를 만든다
LABEL_COL = "will_fail_30"


def make_windows(df: pd.DataFrame, window: int = WINDOW, stride: int = STRIDE,
                 cols: list[str] | None = None, label_col: str = LABEL_COL
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(N, window, F) 입력과 (N,) 라벨, (N,) 차량 ID 를 만든다."""
    cols = cols or SENSOR_COLS
    xs, ys, vs = [], [], []
    for vid, g in df.sort_values(["vehicle_id", "day"]).groupby("vehicle_id"):
        arr = g[cols].to_numpy(dtype=np.float32)
        lab = g[label_col].to_numpy(dtype=np.int64)
        if len(arr) < window:
            continue                          # 관측이 짧은 차량은 건너뛴다
        for s in range(0, len(arr) - window + 1, stride):
            xs.append(arr[s:s + window])
            ys.append(lab[s + window - 1])    # 마지막 시점 라벨
            vs.append(vid)
    if not xs:
        return (np.empty((0, window, len(cols)), dtype=np.float32),
                np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64))
    return (np.stack(xs), np.asarray(ys, dtype=np.int64),
            np.asarray(vs, dtype=np.int64))


def fit_scaler(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """채널별 평균/표준편차를 구한다. (N,T,F) → (F,), (F,)"""
    flat = x.reshape(-1, x.shape[-1])
    mu = flat.mean(axis=0)
    sd = flat.std(axis=0)
    sd[sd < 1e-6] = 1.0                       # 상수 채널 보호
    return mu, sd


def apply_scaler(x: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """학습에서 구한 파라미터로 표준화한다."""
    return ((x - mu) / sd).astype(np.float32)


def split_vehicles(vids: np.ndarray, ratios=(0.6, 0.2, 0.2)
                   ) -> tuple[set, set, set]:
    """차량 ID 를 학습/검증/평가로 나눈다."""
    uniq = sorted(set(int(v) for v in vids))
    n = len(uniq)
    i1 = int(n * ratios[0])
    i2 = int(n * (ratios[0] + ratios[1]))
    return set(uniq[:i1]), set(uniq[i1:i2]), set(uniq[i2:])


def build_torch_loaders(batch_size: int = 64):
    """torch DataLoader 3개와 스케일러를 돌려준다."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    df = pd.read_csv(DATA)
    x, y, v = make_windows(df)
    tr_ids, va_ids, te_ids = split_vehicles(v)

    m_tr = np.isin(v, list(tr_ids))
    mu, sd = fit_scaler(x[m_tr])              # 학습 분할로만 스케일러를 학습

    out = {}
    for name, ids, shuffle in (("train", tr_ids, True),
                               ("val", va_ids, False),
                               ("test", te_ids, False)):
        m = np.isin(v, list(ids))
        xs = apply_scaler(x[m], mu, sd)
        ds = TensorDataset(torch.from_numpy(xs),
                           torch.from_numpy(y[m].astype(np.float32)))
        out[name] = DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
    return out, (mu, sd), (len(SENSOR_COLS), WINDOW)


def main() -> None:
    if not DATA.exists():
        print(f"[안내] {DATA} 가 없다. 먼저 python 06_failure/make_labels.py")
        return

    df = pd.read_csv(DATA)
    x, y, v = make_windows(df)
    print(f"윈도우 길이 {WINDOW}, 스트라이드 {STRIDE}, 채널 {len(SENSOR_COLS)}")
    print(f"생성된 윈도우: {x.shape}  라벨: {y.shape}")
    print(f"양성 비율: {y.mean():.3%}  (행 단위 {df[LABEL_COL].mean():.3%})")

    tr, va, te = split_vehicles(v)
    print(f"\n차량 분할: 학습 {len(tr)} / 검증 {len(va)} / 평가 {len(te)}")
    for name, ids in (("train", tr), ("val", va), ("test", te)):
        m = np.isin(v, list(ids))
        print(f"  {name:<6s} 윈도우 {int(m.sum()):5d}  "
              f"양성 {y[m].mean():.3%}")

    m_tr = np.isin(v, list(tr))
    mu, sd = fit_scaler(x[m_tr])
    print("\n[채널별 표준화 파라미터 — 학습 분할 기준]")
    print(f"{'채널':<20s}{'평균':>11s}{'표준편차':>11s}")
    for c, m_, s_ in zip(SENSOR_COLS, mu, sd):
        print(f"{c:<20s}{m_:11.3f}{s_:11.3f}")

    xs = apply_scaler(x[m_tr], mu, sd)
    print(f"\n표준화 후: 평균 {xs.mean():+.4f} 표준편차 {xs.std():.4f} "
          f"(0 과 1 에 가까워야 한다)")

    print(f"\n[윈도우 길이별 샘플 수와 메모리 — float32]")
    print(f"{'윈도우':>7s}{'샘플수':>8s}{'메모리(MB)':>12s}")
    for w in (12, 24, 48, 96):
        xw, yw, _ = make_windows(df, window=w)
        mb = xw.nbytes / 1024 / 1024
        print(f"{w:7d}{len(xw):8d}{mb:12.2f}")

    try:
        import torch                                        # noqa: F401
        loaders, _, _ = build_torch_loaders()
        xb, yb = next(iter(loaders["train"]))
        print(f"\n[torch] 배치 shape={tuple(xb.shape)} 라벨={tuple(yb.shape)}")
    except ImportError:
        print("\n[안내] PyTorch 가 없어 DataLoader 구성은 건너뛴다.")
        print("       pip install torch --index-url "
              "https://download.pytorch.org/whl/cpu")
        print("       위 numpy 배열만으로도 윈도우 구성은 확인할 수 있다.")


if __name__ == "__main__":
    main()
