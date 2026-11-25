# -*- coding: utf-8 -*-
"""여러 탐지기의 점수를 0~1 로 정규화해 앙상블하는 통합 이상 점수.

탐지기마다 점수의 단위와 분포가 완전히 다르다.
  관리도      : 표준화 거리(0 ~ 수십)
  IsolationForest : 경로 길이 기반(0.4 ~ 0.7)
  One-Class SVM   : 결정함수 부호(-1 ~ +0.1)
  AutoEncoder     : 재구성 MSE(0 ~ 수십)
이대로 평균하면 범위가 큰 탐지기가 결과를 지배한다. 그래서
'정상 데이터 기준 분위수'로 전부 0~1 로 맞춘 뒤 가중 결합한다.

정규화 기준점을 정상 분포의 중앙값으로 두는 것이 핵심이다.
최솟값을 0 으로 잡으면 정상 데이터 절반이 0.5 부근에 몰려 구분이 흐려진다.

이 스크립트는 numpy / pandas 만으로 동작한다.
관리도(stat)와 PCA 재구성 탐지기를 직접 구현하고, sklearn/torch 기반
점수 파일(score_iforest.csv 등)이 있으면 함께 앙상블한다.
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

# 앙상블 가중치. 합이 1 이 되도록 정규화해서 쓴다.
WEIGHTS = {
    "stat": 0.30,      # 관리도(물리적으로 해석 가능, 오경보 적음)
    "pca": 0.30,       # PCA 재구성(센서 간 상관 붕괴를 잡음)
    "iforest": 0.20,   # IsolationForest(비선형, 빠름)
    "ocsvm": 0.10,     # One-Class SVM(포화되므로 낮게)
    "ae": 0.10,        # AutoEncoder(있으면 사용)
}

# 이상 점수 구간별 조치 (CONTRACT.md 의 risk_level enum 과 일치)
RISK_BANDS = [
    (0.00, 0.25, "LOW", "정상 운행. 다음 정기 점검까지 유지"),
    (0.25, 0.50, "MEDIUM", "주의 관찰. 2주 내 점검 권고"),
    (0.50, 0.75, "HIGH", "조기 정비. 1주 내 입고 권고"),
    (0.75, 1.01, "CRITICAL", "즉시 정비. 운행 제한 검토"),
]


# ----------------------------------------------------------------------
# 정규화
# ----------------------------------------------------------------------
def normalize(raw: np.ndarray, baseline: np.ndarray,
              q_lo: float = 0.50, q_hi: float = 0.999) -> np.ndarray:
    """정상 기준 분포의 분위수로 0~1 정규화한다.

    baseline : 정상으로 확인된 데이터의 원점수
    q_lo     : 이 분위수를 0 으로 맞춘다(기본 중앙값)
    q_hi     : 이 분위수를 1 로 맞춘다
    상한을 최댓값이 아니라 99.9% 분위로 두는 이유는, 기준 데이터에
    남아 있는 단일 스파이크 하나가 전체 스케일을 망가뜨리는 것을 막기
    위해서다. 범위를 벗어난 값은 0 또는 1 로 자른다(clip).
    """
    lo = float(np.quantile(baseline, q_lo))
    hi = float(np.quantile(baseline, q_hi))
    return np.clip((raw - lo) / max(hi - lo, 1e-12), 0.0, 1.0)


# ----------------------------------------------------------------------
# 탐지기 1: 관리도(EWMA) — numpy 로 직접 구현
# ----------------------------------------------------------------------
def ewma(x: np.ndarray, lam: float = 0.2) -> np.ndarray:
    """지수가중 이동평균."""
    out = np.empty_like(x, dtype=float)
    acc = x[0]
    for i, v in enumerate(x):
        acc = lam * v + (1 - lam) * acc
        out[i] = acc
    return out


def stat_detector(df: pd.DataFrame, cols: list[str],
                  normal_mask: np.ndarray) -> np.ndarray:
    """센서별 EWMA 표준화 거리의 최댓값을 원점수로 쓴다.

    정상 차량에서 구한 중앙값/MAD 로 각 센서를 표준화한 뒤 EWMA 를 걸고,
    센서 중 가장 많이 벗어난 값을 그 시점의 이상도로 본다(최댓값 결합).
    평균이 아니라 최댓값을 쓰는 이유는, 센서 하나만 심하게 이상해도
    고장이기 때문이다. 평균을 쓰면 정상 센서들이 그 신호를 희석한다.
    """
    x = df[cols].to_numpy(dtype=float)
    ref = x[normal_mask]
    med = np.median(ref, axis=0)
    mad = np.median(np.abs(ref - med), axis=0)
    sigma = np.where(mad > 0, mad / 0.6745, ref.std(axis=0) + 1e-9)

    z = np.abs(x - med) / sigma
    # 차량별로 EWMA 를 걸어야 차량 경계가 섞이지 않는다.
    out = np.zeros(len(df))
    for vid in df["vehicle_id"].unique():
        m = (df["vehicle_id"] == vid).to_numpy()
        zz = z[m]
        sm = np.column_stack([ewma(zz[:, j]) for j in range(zz.shape[1])])
        out[m] = sm.max(axis=1)
    return out


# ----------------------------------------------------------------------
# 탐지기 2: PCA 재구성 오차 — numpy 로 직접 구현
# ----------------------------------------------------------------------
def pca_detector(df: pd.DataFrame, cols: list[str],
                 normal_mask: np.ndarray,
                 n_comp: int = 4) -> np.ndarray:
    """정상 데이터의 주성분으로 복원했을 때의 오차를 원점수로 쓴다.

    AutoEncoder 와 원리가 같은 선형 버전이다. 정상 데이터에서 분산이 큰
    방향 n_comp 개만 남기고 복원하면, 정상은 잘 복원되고 센서 간 상관이
    깨진 이상 데이터는 복원에 실패한다. SVD 하나로 끝나므로 학습이
    사실상 즉시 끝나고 CPU 비용이 거의 없다.
    """
    x = df[cols].to_numpy(dtype=float)
    ref = x[normal_mask]
    mu = ref.mean(axis=0)
    sd = ref.std(axis=0) + 1e-9
    xs = (x - mu) / sd
    rs = (ref - mu) / sd

    # 정상 데이터의 공분산 구조에서 주성분을 얻는다.
    _, _, vt = np.linalg.svd(rs, full_matrices=False)
    p = vt[:n_comp].T                      # (F, n_comp) 투영 행렬
    recon = xs @ p @ p.T                   # 저차원 통과 후 복원
    return ((xs - recon) ** 2).mean(axis=1)


# ----------------------------------------------------------------------
# 외부 점수 파일 병합
# ----------------------------------------------------------------------
def attach_external(df: pd.DataFrame, data_dir: pathlib.Path
                    ) -> list[str]:
    """sklearn/torch 스크립트가 저장한 점수 파일을 붙인다."""
    found: list[str] = []
    spec = [("iforest", "score_iforest.csv", "iforest_raw"),
            ("ocsvm", "score_ocsvm.csv", "ocsvm_raw"),
            ("ae", "score_autoencoder.csv", "ae_raw")]
    for key, fname, rawcol in spec:
        p = data_dir / fname
        if not p.exists():
            continue
        s = pd.read_csv(p)
        if rawcol not in s.columns or len(s) != len(df):
            continue
        df[f"raw_{key}"] = s[rawcol].to_numpy()
        found.append(key)
    return found


def to_risk(score: float) -> tuple[str, str]:
    """이상 점수를 risk_level 과 권고 조치로 변환한다."""
    for lo, hi, level, action in RISK_BANDS:
        if lo <= score < hi:
            return level, action
    return "CRITICAL", RISK_BANDS[-1][3]


def main() -> None:
    ap = argparse.ArgumentParser(description="통합 이상 점수 산출")
    ap.add_argument("--csv", default="data/sensor_clean.csv")
    ap.add_argument("--out", default="data/anomaly_score.csv")
    ap.add_argument("--window", type=int, default=36,
                    help="차량 판정에 쓸 최근 구간 길이(행)")
    ap.add_argument("--quantile", type=float, default=0.50,
                    help="차량 대표 점수로 쓸 분위수")
    args = ap.parse_args()

    path = pathlib.Path(args.csv)
    if not path.exists():
        print(f"[오류] 입력 CSV 가 없다: {path}")
        print("  python 02_data/make_sensor_dataset.py --defects "
              "--out data/sensor_readings_raw.csv")
        print("  python 04_preprocess/clean.py")
        return

    df = pd.read_csv(path)
    cols = [c for c in FEATURES if c in df.columns]
    df["scenario"] = df["vehicle_id"].map(SCENARIO)
    normal_mask = df["vehicle_id"].isin(NORMAL_IDS).to_numpy()
    print(f"입력 {len(df):,}행, 특징 {len(cols)}개")
    print(f"정상 기준 차량 {NORMAL_IDS} -> {normal_mask.sum():,}행")

    # --- 1. 내장 탐지기 ---------------------------------------------
    df["raw_stat"] = stat_detector(df, cols, normal_mask)
    df["raw_pca"] = pca_detector(df, cols, normal_mask, n_comp=4)
    active = ["stat", "pca"]

    # --- 2. 외부 탐지기(있으면) --------------------------------------
    found = attach_external(df, path.parent)
    active += found
    print(f"\n활성 탐지기: {active}")
    missing = [k for k in ("iforest", "ocsvm", "ae") if k not in found]
    if missing:
        print(f"  미발견: {missing}")
        print("  아래를 먼저 실행하면 앙상블에 자동으로 포함된다.")
        print("    python 05_anomaly/isolation_forest.py")
        print("    python 05_anomaly/one_class_svm.py")
        print("    python 05_anomaly/autoencoder.py")

    # --- 3. 0~1 정규화 -----------------------------------------------
    print(f"\n[탐지기별 원점수 범위와 정규화 기준]")
    print(f"  {'탐지기':<10}{'원점수 최소':>12}{'원점수 최대':>12}"
          f"{'0 기준(중앙값)':>16}{'1 기준(99.9%)':>15}")
    for key in active:
        raw = df[f"raw_{key}"].to_numpy(dtype=float)
        base = raw[normal_mask]
        df[f"s_{key}"] = normalize(raw, base)
        print(f"  {key:<10}{raw.min():>12.4f}{raw.max():>12.4f}"
              f"{np.quantile(base, 0.50):>16.4f}"
              f"{np.quantile(base, 0.999):>15.4f}")
    print("  원점수 범위가 탐지기마다 전혀 다르다는 점을 확인할 수 있다.")

    # --- 4. 가중 앙상블 ----------------------------------------------
    w = {k: WEIGHTS[k] for k in active}
    tot = sum(w.values())
    w = {k: v / tot for k, v in w.items()}
    print(f"\n[앙상블 가중치]  " + ", ".join(
        f"{k}={v:.2f}" for k, v in w.items()))

    df["anomaly_score"] = sum(df[f"s_{k}"] * v for k, v in w.items())
    # 참고용으로 최댓값 결합도 함께 본다.
    df["anomaly_max"] = df[[f"s_{k}" for k in active]].max(axis=1)

    print(f"\n[행 단위 점수 분포]")
    print(f"  {'vid':>4}{'시나리오':>10}" + "".join(
        f"{f's_{k}':>9}" for k in active) + f"{'앙상블':>9}{'최댓값':>9}")
    for vid, g in df.groupby("vehicle_id"):
        print(f"  {vid:>4}{SCENARIO.get(vid, '?'):>10}"
              + "".join(f"{g[f's_{k}'].mean():>9.4f}" for k in active)
              + f"{g['anomaly_score'].mean():>9.4f}"
              f"{g['anomaly_max'].mean():>9.4f}")

    # --- 5. 대표값 분위수 선택 ---------------------------------------
    # 차량 대표 점수를 최근 구간의 어느 분위수로 잡을지가 오경보를 좌우한다.
    # 높은 분위수(95%)는 정상 차량의 순간 스파이크 하나까지 끌어올려
    # 오경보를 만든다. 중앙값은 '지속적으로 이상한 상태'만 잡는다.
    print(f"\n[대표값 분위수 선택]  최근 {args.window}행 기준")
    print(f"  {'분위수':<10}" + "".join(
        f"{f'vid{v}({SCENARIO[v][:4]})':>16}" for v in sorted(SCENARIO)))
    for q in (0.50, 0.75, 0.90, 0.95, 1.00):
        vals = []
        for vid in sorted(SCENARIO):
            recent = df[df["vehicle_id"] == vid]["anomaly_score"].tail(
                args.window)
            r = float(np.quantile(recent, q))
            vals.append(f"{r:.3f}/{to_risk(r)[0][:4]}")
        print(f"  {q:<10.2f}" + "".join(f"{v:>16}" for v in vals))
    print("  95% 분위는 정상 차량도 MEDIUM 으로 올린다(오경보).")
    print("  중앙값은 정상을 LOW 로 유지하면서 열화/고장은 그대로 잡는다.")
    print("  '지속성'을 요구하는 쪽이 현장에서 신뢰를 얻는다.")

    # --- 6. 차량 단위 판정 -------------------------------------------
    print(f"\n[차량 단위 판정]  최근 {args.window}행의 "
          f"{args.quantile:.0%} 분위를 대표값으로")
    print(f"  {'vid':>4}{'시나리오':>10}{'대표 점수':>11}"
          f"{'risk_level':>12}  권고 조치")
    rows = []
    for vid, g in df.groupby("vehicle_id"):
        recent = g["anomaly_score"].tail(args.window)
        rep = float(np.quantile(recent, args.quantile))
        level, action = to_risk(rep)
        print(f"  {vid:>4}{SCENARIO.get(vid, '?'):>10}{rep:>11.4f}"
              f"{level:>12}  {action}")
        rows.append({"vehicle_id": vid, "scenario": SCENARIO.get(vid),
                     "anomaly_score": round(rep, 4), "risk_level": level})

    # --- 7. 단일 탐지기 대비 앙상블 비교 ------------------------------
    nm = df["scenario"].eq("normal").to_numpy()
    dm = df["scenario"].eq("degrade").to_numpy()
    fm = df["scenario"].eq("failure").to_numpy()
    anom = dm | fm

    print(f"\n[분리도 비교]  (고장 평균 - 정상 평균) / 정상 표준편차")
    print(f"  {'탐지기':<12}{'정상':>9}{'열화':>9}{'고장':>9}"
          f"{'분리도':>9}{'포화':>8}")
    for key in active + ["anomaly_score", "anomaly_max"]:
        c = f"s_{key}" if key in active else key
        v = df[c].to_numpy()
        sep = (v[anom].mean() - v[nm].mean()) / (v[nm].std() + 1e-9)
        # 포화: 이상 구간에서 점수가 1.0 에 붙어버린 비율
        sat = (v[anom] >= 0.999).mean()
        name = key if key in active else key.replace("anomaly_", "앙상블 ")
        print(f"  {name:<12}{v[nm].mean():>9.4f}{v[dm].mean():>9.4f}"
              f"{v[fm].mean():>9.4f}{sep:>9.2f}{sat * 100:>7.0f}%")

    print("\n  주의: 분리도만 보면 PCA 단독이 앙상블보다 높게 나온다.")
    print("  그러나 PCA 와 OCSVM 은 이상 구간의 거의 100% 를 1.0 으로")
    print("  포화시킨다. 열화(degrade)와 고장(failure)이 같은 값이 되어")
    print("  '얼마나 심한가'를 구분하지 못한다는 뜻이다. 정비 우선순위를")
    print("  정해야 하는 예지보전에서는 치명적인 결함이다.")
    print("  앙상블은 분리도를 충분히 유지하면서 포화를 없애")
    print("  열화와 고장 사이에 실제 점수 차이를 만든다.")
    print("  단일 지표 최적화가 아니라 '등급을 매길 수 있는 점수'가 목표다.")

    print(f"\n[이상 점수 구간별 조치 기준]")
    print(f"  {'구간':<14}{'risk_level':<12}권고 조치")
    for lo, hi, level, action in RISK_BANDS:
        rng = f"{lo:.2f} ~ {min(hi, 1.0):.2f}"
        print(f"  {rng:<14}{level:<12}{action}")

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    keep = (["vehicle_id", "measured_at", "scenario"]
            + [f"s_{k}" for k in active]
            + ["anomaly_score", "anomaly_max"])
    df[keep].to_csv(out, index=False, encoding="utf-8")
    pd.DataFrame(rows).to_csv(out.with_name("vehicle_risk.csv"),
                              index=False, encoding="utf-8")
    print(f"\n저장: {out}")
    print(f"저장: {out.with_name('vehicle_risk.csv')}")
    print("  vehicle_risk.csv 의 anomaly_score / risk_level 은")
    print("  pdm_prediction 테이블의 같은 이름 컬럼에 그대로 적재된다.")


if __name__ == "__main__":
    main()
