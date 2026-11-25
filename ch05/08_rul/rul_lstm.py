# -*- coding: utf-8 -*-
"""LSTM 기반 RUL 예측. torch 미설치 시 안내 후 종료.

회귀 기반 RUL(08_rul/rul_regression.py)은 '현재 특징 → 남은 수명'을
직접 사상한다. LSTM 은 윈도우 전체의 열화 궤적을 보고 추정하므로
열화 속도가 변하는 구간에서 유리할 수 있다.

손실 함수로는 두 가지를 비교한다.
  - MSE        : 대칭 손실. 과소·과대예측을 같게 벌한다.
  - 비대칭 Huber : 과대예측(고장을 놓치는 쪽)에 더 큰 벌점을 준다.

실행:  python 08_rul/rul_lstm.py
입력:  data/rul_dataset.csv
출력:  data/rul_lstm_<손실>.pt
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from rul_metrics import evaluate                            # noqa: E402

OUT_DIR = pathlib.Path("data")
DATA = OUT_DIR / "rul_dataset.csv"
SENSORS = ["vibration_rms", "engine_temp_c", "oil_quality"]
WINDOW = 20
R_EARLY = 60.0


def make_windows(df: pd.DataFrame, window: int = WINDOW):
    """유닛 단위로 (N, window, F) 윈도우와 마지막 시점 RUL 을 만든다."""
    xs, ys, us = [], [], []
    for uid, g in df.sort_values(["unit_id", "day"]).groupby("unit_id"):
        arr = g[SENSORS].to_numpy(dtype=np.float32)
        rul = g["rul_days"].to_numpy(dtype=np.float32)
        if len(arr) < window:
            continue
        for s in range(len(arr) - window + 1):
            xs.append(arr[s:s + window])
            ys.append(rul[s + window - 1])
            us.append(uid)
    return (np.stack(xs), np.asarray(ys, dtype=np.float32),
            np.asarray(us, dtype=np.int64))


def build_model(n_features: int, hidden: int = 64, layers: int = 2):
    """LSTM 회귀 모델. 출력은 0~1 로 묶은 뒤 R_EARLY 를 곱한다."""
    import torch
    import torch.nn as nn

    class RULNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.rnn = nn.LSTM(n_features, hidden, num_layers=layers,
                               batch_first=True, dropout=0.2)
            self.head = nn.Sequential(
                nn.LayerNorm(hidden), nn.Linear(hidden, 32),
                nn.ReLU(), nn.Dropout(0.1), nn.Linear(32, 1))

        def forward(self, x):
            out, _ = self.rnn(x)
            # 출력에 sigmoid 를 씌워 [0, R_EARLY] 범위를 보장한다.
            # 범위를 강제하지 않으면 음수 RUL 이나 수천 일이 나온다.
            return torch.sigmoid(self.head(out[:, -1]).squeeze(-1)) * R_EARLY

    return RULNet()


def asymmetric_huber(pred, target, delta: float = 5.0,
                     late_weight: float = 3.0):
    """과대예측에 late_weight 배 벌점을 주는 Huber 손실.

    d = pred - target 이 양수면 '아직 괜찮다'고 잘못 말한 것이므로
    가중치를 키운다. Huber 를 쓰는 이유는 라벨 노이즈가 큰 RUL 에서
    MSE 가 이상치에 과하게 끌려가기 때문이다.
    """
    import torch

    d = pred - target
    a = torch.abs(d)
    base = torch.where(a <= delta, 0.5 * d * d / delta, a - 0.5 * delta)
    w = torch.where(d > 0, torch.full_like(d, late_weight),
                    torch.ones_like(d))
    return (base * w).mean()


def main() -> None:
    try:
        import torch
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        print("[안내] PyTorch 가 설치되어 있지 않다.")
        print("       pip install torch --index-url "
              "https://download.pytorch.org/whl/cpu")
        print("       설치 후 다시 실행하면 MSE 손실과 비대칭 Huber 손실로")
        print("       각각 학습해 RUL 예측 성능을 비교한다.")
        return
    if not DATA.exists():
        print(f"[안내] {DATA} 가 없다. 먼저 python 08_rul/rul_curve.py")
        return

    torch.manual_seed(42)
    torch.set_num_threads(4)

    df = pd.read_csv(DATA)
    x, y, u = make_windows(df)
    uids = sorted(set(int(v) for v in u))
    tr_ids = set(uids[:int(len(uids) * 0.7)])
    m_tr = np.isin(u, list(tr_ids))

    # 표준화는 학습 분할에서만 계산한다.
    flat = x[m_tr].reshape(-1, x.shape[-1])
    mu, sd = flat.mean(axis=0), flat.std(axis=0)
    sd[sd < 1e-6] = 1.0
    xs = ((x - mu) / sd).astype(np.float32)

    xtr = torch.from_numpy(xs[m_tr])
    ytr = torch.from_numpy(y[m_tr])
    xte = torch.from_numpy(xs[~m_tr])
    yte = torch.from_numpy(y[~m_tr])
    loader = DataLoader(TensorDataset(xtr, ytr), batch_size=64, shuffle=True)
    print(f"윈도우 {WINDOW} / 채널 {len(SENSORS)} / "
          f"학습 {len(xtr)} / 평가 {len(xte)}\n")

    results = []
    for loss_name in ("mse", "asym_huber"):
        model = build_model(len(SENSORS))
        n_param = sum(p.numel() for p in model.parameters())
        opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50)
        mse = torch.nn.MSELoss()
        ckpt = OUT_DIR / f"rul_lstm_{loss_name}.pt"

        best_rmse, bad = float("inf"), 0
        print(f"=== 손실함수: {loss_name} (파라미터 {n_param:,}) ===")
        for ep in range(1, 51):
            model.train()
            total, n = 0.0, 0
            for xb, yb in loader:
                opt.zero_grad()
                p = model(xb)
                loss = (mse(p, yb) if loss_name == "mse"
                        else asymmetric_huber(p, yb))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                total += float(loss) * len(xb)
                n += len(xb)
            sched.step()

            model.eval()
            with torch.no_grad():
                pred = model(xte).numpy()
            r = float(np.sqrt(np.mean((pred - yte.numpy()) ** 2)))
            if r < best_rmse - 1e-3:
                best_rmse, bad = r, 0
                torch.save({"state_dict": model.state_dict(), "mu": mu,
                            "sd": sd, "window": WINDOW, "epoch": ep}, ckpt)
            else:
                bad += 1
            if ep % 10 == 0 or ep == 1:
                print(f"  ep{ep:3d}  train_loss={total / n:9.4f}  "
                      f"test_RMSE={r:6.2f}")
            if bad >= 12:
                print(f"  조기 종료 (ep{ep})")
                break

        state = torch.load(ckpt, weights_only=False)
        model.load_state_dict(state["state_dict"])
        model.eval()
        with torch.no_grad():
            pred = model(xte).numpy()
        mt = evaluate(yte.numpy().astype(float), pred.astype(float))
        mt["name"] = loss_name
        results.append(mt)
        print(f"  최종: RMSE={mt['RMSE']:.2f} MAE={mt['MAE']:.2f} "
              f"Cost={mt['Cost']:.1f} Late={mt['LateRate']:.3f}\n")

    hdr = (f"{'손실함수':<13s}{'RMSE':>7s}{'MAE':>7s}{'Bias':>8s}"
           f"{'Score':>10s}{'Cost':>10s}{'Late':>7s}")
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        print(f"{r['name']:<13s}{r['RMSE']:7.2f}{r['MAE']:7.2f}{r['Bias']:8.2f}"
              f"{r['Score']:10.1f}{r['Cost']:10.1f}{r['LateRate']:7.3f}")

    print("\n비대칭 손실을 쓰면 RMSE 는 조금 나빠지고 Bias 는 음수(보수적)로")
    print("이동하면서 Cost 와 Late 가 개선되는 것이 일반적인 경향이다.")
    print("두 체크포인트는 data/rul_lstm_*.pt 에 저장된다.")


if __name__ == "__main__":
    main()
