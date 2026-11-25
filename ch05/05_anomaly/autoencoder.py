# -*- coding: utf-8 -*-
"""PyTorch AutoEncoder 기반 이상 탐지.

AutoEncoder 는 입력을 좁은 잠재공간(latent space)으로 압축한 뒤 다시
복원하도록 학습한다. 정상 데이터만으로 학습하면 모델은 '정상의 구조'만
배우므로, 정상 입력은 잘 복원되고 이상 입력은 복원에 실패한다.
따라서 재구성 오차(reconstruction error)가 곧 이상 점수가 된다.

  이상 점수 = || x - decoder(encoder(x)) ||^2

잠재 차원이 너무 크면 입력을 그대로 외워버려(identity) 이상도 잘 복원해
버린다. 입력 12차원에 대해 잠재 3~4차원이 적절하다.

torch 미설치 시 안내 후 정상 종료한다.
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


def build_model(n_in: int, latent: int = 4):
    """대칭 구조의 완전연결 AutoEncoder 를 만든다.

    인코더: n_in -> 16 -> 8 -> latent
    디코더: latent -> 8 -> 16 -> n_in
    활성함수는 ReLU 를 쓰고 출력층은 선형으로 둔다(표준화된 값을 복원하므로
    음수도 나와야 한다). CPU 추론을 전제로 매우 작게 설계한다.
    """
    import torch.nn as nn

    class AutoEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(n_in, 16), nn.ReLU(),
                nn.Linear(16, 8), nn.ReLU(),
                nn.Linear(8, latent),
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent, 8), nn.ReLU(),
                nn.Linear(8, 16), nn.ReLU(),
                nn.Linear(16, n_in),
            )

        def forward(self, x):
            return self.decoder(self.encoder(x))

    return AutoEncoder()


def main() -> None:
    ap = argparse.ArgumentParser(description="AutoEncoder 이상 탐지")
    ap.add_argument("--csv", default="data/sensor_clean.csv")
    ap.add_argument("--out", default="data/score_autoencoder.csv")
    ap.add_argument("--latent", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=20261024)
    args = ap.parse_args()

    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        print("[안내] PyTorch 가 설치되어 있지 않다.")
        print("  CPU 버전을 아래 명령으로 설치한 뒤 다시 실행한다.")
        print("    pip install torch --index-url "
              "https://download.pytorch.org/whl/cpu")
        print("  설치 없이 진행하려면 05_anomaly/isolation_forest.py 또는")
        print("  05_anomaly/stat_threshold.py 를 사용한다.")
        return

    try:
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        print("[안내] scikit-learn 이 설치되어 있지 않다.")
        print("    pip install scikit-learn")
        return

    df = load(pathlib.Path(args.csv))
    if df is None:
        return

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(4)

    cols = [c for c in FEATURES if c in df.columns]
    df["scenario"] = df["vehicle_id"].map(SCENARIO)

    # 정상 차량만 학습한다. 이것이 AutoEncoder 이상 탐지의 핵심 전제다.
    train = df[df["vehicle_id"].isin(NORMAL_IDS)]
    scaler = StandardScaler().fit(train[cols])
    x_train = scaler.transform(train[cols]).astype(np.float32)
    x_all = scaler.transform(df[cols]).astype(np.float32)

    # 학습 안에서 다시 검증 구간을 떼어 과적합을 감시한다(시간순 분할).
    n_val = int(len(x_train) * 0.2)
    x_fit, x_val = x_train[:-n_val], x_train[-n_val:]
    print(f"학습 {len(x_fit):,}행 / 검증 {len(x_val):,}행 / "
          f"전체 평가 {len(x_all):,}행, 특징 {len(cols)}개")

    model = build_model(len(cols), args.latent)
    n_param = sum(p.numel() for p in model.parameters())
    print(f"모델 파라미터 {n_param:,}개 (잠재 차원 {args.latent})")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()
    loader = DataLoader(TensorDataset(torch.from_numpy(x_fit)),
                        batch_size=args.batch, shuffle=True)
    t_val = torch.from_numpy(x_val)

    print(f"\n[학습]")
    print(f"  {'epoch':>6}{'train MSE':>12}{'val MSE':>12}")
    for ep in range(1, args.epochs + 1):
        model.train()
        tot = 0.0
        for (xb,) in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), xb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb)
        if ep % 20 == 0 or ep == 1:
            model.eval()
            with torch.no_grad():
                v = loss_fn(model(t_val), t_val).item()
            print(f"  {ep:>6}{tot / len(x_fit):>12.6f}{v:>12.6f}")

    # 재구성 오차 계산
    model.eval()
    with torch.no_grad():
        rec_all = model(torch.from_numpy(x_all)).numpy()
        rec_tr = model(torch.from_numpy(x_train)).numpy()
    err_all = ((x_all - rec_all) ** 2).mean(axis=1)
    err_tr = ((x_train - rec_tr) ** 2).mean(axis=1)

    df["ae_raw"] = err_all
    lo, hi = np.quantile(err_tr, 0.50), np.quantile(err_tr, 0.999)
    df["ae_score"] = np.clip((err_all - lo) / max(hi - lo, 1e-12), 0.0, 1.0)
    thr = float(np.quantile(err_tr, 0.98))
    print(f"\n[정규화 기준]  학습 중앙값={lo:.6f} -> 0,  "
          f"99.9% 분위={hi:.6f} -> 1")
    print(f"[경보 임계값]  학습 98% 분위={thr:.6f}")

    print(f"\n[차량별 재구성 오차와 이상 점수]")
    print(f"  {'vid':>4}{'시나리오':>10}{'오차 평균':>12}{'오차 최대':>12}"
          f"{'점수 평균':>10}{'점수 중앙':>10}{'경보율':>9}")
    for vid, g in df.groupby("vehicle_id"):
        print(f"  {vid:>4}{SCENARIO.get(vid, '?'):>10}"
              f"{g['ae_raw'].mean():>12.6f}{g['ae_raw'].max():>12.6f}"
              f"{g['ae_score'].mean():>10.4f}{g['ae_score'].median():>10.4f}"
              f"{(g['ae_raw'] > thr).mean() * 100:>8.1f}%")

    print(f"\n[센서별 재구성 오차 기여도]  고장 차량(vid=4)에서 큰 순서")
    v4 = (df["vehicle_id"] == 4).to_numpy()
    per_col = ((x_all[v4] - rec_all[v4]) ** 2).mean(axis=0)
    for c, v in sorted(zip(cols, per_col), key=lambda x: -x[1])[:6]:
        print(f"  {c:<20}{v:>10.4f}")
    print("  어느 센서가 복원되지 않았는지 보면 '어느 부품이 이상한지'를")
    print("  설명할 수 있다. 블랙박스 모델에 해석성을 더하는 방법이다.")

    print(f"\n[잠재 차원 민감도]  차원이 크면 이상도 복원해 버린다")
    print(f"  {'latent':>8}{'정상 오차':>12}{'고장 오차':>12}{'비율':>8}")
    for lat in (2, 3, 4, 6, 12):
        torch.manual_seed(args.seed)
        m = build_model(len(cols), lat)
        o = torch.optim.Adam(m.parameters(), lr=args.lr)
        ld = DataLoader(TensorDataset(torch.from_numpy(x_fit)),
                        batch_size=args.batch, shuffle=True)
        for _ in range(60):
            m.train()
            for (xb,) in ld:
                o.zero_grad()
                l = loss_fn(m(xb), xb)
                l.backward()
                o.step()
        m.eval()
        with torch.no_grad():
            r = m(torch.from_numpy(x_all)).numpy()
        e = ((x_all - r) ** 2).mean(axis=1)
        en = e[df["scenario"].eq("normal").to_numpy()].mean()
        ef = e[df["scenario"].eq("failure").to_numpy()].mean()
        print(f"  {lat:>8}{en:>12.6f}{ef:>12.6f}{ef / max(en, 1e-12):>8.1f}x")
    print("  비율(고장/정상)이 클수록 좋은 탐지기다. 잠재 차원을 입력과")
    print("  같게(12) 두면 항등함수에 가까워져 비율이 떨어진다.")

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    keep = ["vehicle_id", "measured_at", "scenario", "ae_raw", "ae_score"]
    df[keep].to_csv(out, index=False, encoding="utf-8")
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
