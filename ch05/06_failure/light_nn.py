# -*- coding: utf-8 -*-
"""경량 MLP(PyTorch) 고장 분류기. torch 미설치 시 안내 후 종료.

표 형태 센서 데이터에서 MLP 가 GBDT 를 이기는 경우는 드물다. 그래도
MLP 를 다루는 이유는 두 가지다.
  1) 5.7절의 시계열 모델과 같은 학습 루프를 쓰므로 연습이 된다.
  2) 임베딩이나 멀티태스크(이진+다중 동시 예측)를 붙이기 쉽다.

실행:  python 06_failure/light_nn.py
입력:  data/labeled_dataset.csv
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

# 같은 디렉터리의 random_forest.py 에서 특징 생성과 분할 함수를 재사용한다.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from random_forest import (BASE_FEATURES, add_rolling_features,   # noqa: E402
                           split_by_vehicle)

DATA = pathlib.Path("data") / "labeled_dataset.csv"
OUT_DIR = pathlib.Path("data")
N_PART_CLASSES = 6


def build_model(n_features: int, hidden: tuple[int, ...] = (64, 32)):
    """멀티태스크 MLP 를 만든다. 공유 몸통 + 두 개의 헤드.

    고장 여부(이진)와 고장 부품(다중)은 같은 전조를 공유하므로
    몸통을 공유하면 데이터가 적을 때 서로를 돕는다(멀티태스크 학습).
    """
    import torch.nn as nn

    class LightMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            layers, prev = [], n_features
            for h in hidden:
                layers += [nn.Linear(prev, h), nn.BatchNorm1d(h),
                           nn.ReLU(), nn.Dropout(0.25)]
                prev = h
            self.body = nn.Sequential(*layers)
            self.head_bin = nn.Linear(prev, 1)              # 이진 로짓
            self.head_part = nn.Linear(prev, N_PART_CLASSES)  # 부품 로짓

        def forward(self, x):
            h = self.body(x)
            return self.head_bin(h).squeeze(-1), self.head_part(h)

    return LightMLP()


def main() -> None:
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        print("[안내] PyTorch 가 설치되어 있지 않다. CPU 버전을 설치한다.")
        print("       pip install torch --index-url "
              "https://download.pytorch.org/whl/cpu")
        print("       설치 후 다시 실행하면 경량 MLP 학습이 진행된다.")
        return
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        print("[안내] scikit-learn 이 필요하다. pip install scikit-learn")
        return
    if not DATA.exists():
        print(f"[안내] {DATA} 가 없다. 먼저 python 06_failure/make_labels.py")
        return

    torch.manual_seed(42)
    torch.set_num_threads(4)           # CPU 학습에서는 스레드 수를 고정한다

    df = add_rolling_features(pd.read_csv(DATA))
    feats = BASE_FEATURES + [c for c in df.columns
                             if c.endswith(("_ma7", "_sd7", "_slope"))]
    train, test = split_by_vehicle(df)

    # 표준화 파라미터는 **학습 데이터로만** 계산한다. 평가 데이터의
    # 평균/표준편차를 쓰면 데이터 누설(leakage)이 되어 지표가 부풀려진다.
    mu = train[feats].mean().to_numpy(dtype=np.float32)
    sd = train[feats].std().replace(0.0, 1.0).to_numpy(dtype=np.float32)

    def to_tensor(d: pd.DataFrame):
        x = ((d[feats].to_numpy(dtype=np.float32) - mu) / sd)
        return (torch.from_numpy(x),
                torch.from_numpy(d["will_fail_30"].to_numpy(dtype=np.float32)),
                torch.from_numpy(d["fail_part"].to_numpy(dtype=np.int64)))

    xtr, ybtr, yptr = to_tensor(train)
    xte, ybte, ypte = to_tensor(test)
    loader = DataLoader(TensorDataset(xtr, ybtr, yptr),
                        batch_size=128, shuffle=True)

    model = build_model(len(feats))
    n_param = sum(p.numel() for p in model.parameters())
    print(f"특징 {len(feats)}개 / 파라미터 {n_param:,}개 "
          f"/ 학습 {len(xtr)}행 / 평가 {len(xte)}행")

    # 양성 가중치를 pos_weight 로 넣어 불균형을 보정한다.
    pos_w = torch.tensor([(ybtr == 0).sum() / max((ybtr == 1).sum(), 1)])
    loss_bin = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    loss_part = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=40)

    best_auc, best_state, patience = 0.0, None, 0
    print(f"\n{'epoch':>6s}{'train_loss':>12s}{'val_AUC':>9s}{'lr':>10s}")
    for ep in range(1, 41):
        model.train()
        total = 0.0
        for xb, yb, yp in loader:
            opt.zero_grad()
            lb, lp = model(xb)
            # 보조 과제(부품 분류)에 0.4 가중치를 준다. 1.0 을 주면
            # 주 과제인 이진 분류가 희생될 수 있다.
            loss = loss_bin(lb, yb) + 0.4 * loss_part(lp, yp)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss) * len(xb)
        sched.step()

        model.eval()
        with torch.no_grad():
            lb, _ = model(xte)
            prob = torch.sigmoid(lb).numpy()
        auc = roc_auc_score(ybte.numpy(), prob)
        if ep % 5 == 0 or ep == 1:
            print(f"{ep:6d}{total / len(xtr):12.4f}{auc:9.4f}"
                  f"{opt.param_groups[0]['lr']:10.5f}")

        if auc > best_auc + 1e-4:
            best_auc, patience = auc, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 10:            # 조기 종료
                print(f"  조기 종료 (epoch {ep}, 최고 AUC {best_auc:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        lb, lp = model(xte)
        prob = torch.sigmoid(lb).numpy()
        part_pred = lp.argmax(dim=1).numpy()

    pred = (prob >= 0.5).astype(int)
    y = ybte.numpy().astype(int)
    tp = int(((y == 1) & (pred == 1)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0

    print(f"\n[최종 — 이진] AUC={best_auc:.4f} 정밀도={prec:.3f} "
          f"재현율={rec:.3f} F1={f1:.3f}")
    print(f"[최종 — 부품] 정확도={float((part_pred == ypte.numpy()).mean()):.4f}")

    OUT_DIR.mkdir(exist_ok=True)
    ckpt = OUT_DIR / "light_nn.pt"
    torch.save({"state_dict": model.state_dict(), "features": feats,
                "mu": mu, "sd": sd}, ckpt)
    print(f"\n체크포인트 저장: {ckpt}")
    print("표준화 파라미터(mu/sd)를 체크포인트에 함께 저장해야 "
          "서비스에서 같은 전처리를 재현할 수 있다.")


if __name__ == "__main__":
    main()
