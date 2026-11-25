# -*- coding: utf-8 -*-
"""시계열 모델 학습 루프 — 조기 종료, 체크포인트, 모델 교체 비교.

LSTM / GRU / TCN / Transformer 를 같은 데이터·같은 루프로 학습해
공정하게 비교한다. torch 미설치 시 안내 후 종료한다.

실행:  python 07_timeseries/train_ts.py
       python 07_timeseries/train_ts.py gru          (한 모델만)
입력:  data/labeled_dataset.csv
출력:  data/ts_<모델>.pt
"""
from __future__ import annotations

import pathlib
import sys
import time

# 같은 디렉터리의 모델 정의 파일들을 재사용한다.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

OUT_DIR = pathlib.Path("data")
DATA = OUT_DIR / "labeled_dataset.csv"
MODEL_NAMES = ("lstm", "gru", "tcn", "transformer")


def make_model(name: str, n_features: int):
    """이름으로 모델을 만든다. 각 정의 파일을 재사용한다."""
    if name in ("lstm", "gru"):
        from lstm_model import build as build_rnn
        return build_rnn(cell=name, n_features=n_features, hidden=64, layers=2)
    if name == "tcn":
        import torch.nn as nn
        from tcn_model import TCNClassifier
        m = TCNClassifier(n_features=n_features, n_classes=1,
                          channels=48, n_levels=4)
        # 이진 분류를 로짓 1개로 통일한다(BCEWithLogitsLoss 사용).
        assert isinstance(m.head, nn.Linear)
        return m
    if name == "transformer":
        from ts_transformer import build as build_tr
        return build_tr(n_features=n_features, d_model=64,
                        n_heads=4, n_layers=2)
    raise ValueError(f"알 수 없는 모델: {name}")


def forward_logits(name: str, model, xb):
    """모델별 출력 형태 차이를 한 곳에서 흡수한다."""
    out = model(xb)
    return out.squeeze(-1) if out.dim() > 1 else out


def evaluate(name: str, model, loader):
    """검증/평가 손실과 ROC-AUC 를 계산한다."""
    import torch
    from sklearn.metrics import roc_auc_score

    model.eval()
    probs, ys = [], []
    with torch.no_grad():
        for xb, yb in loader:
            p = torch.sigmoid(forward_logits(name, model, xb))
            probs.append(p)
            ys.append(yb)
    prob = torch.cat(probs).numpy()
    y = torch.cat(ys).numpy()
    auc = roc_auc_score(y, prob) if 0 < y.mean() < 1 else float("nan")
    return auc, prob, y


def train_one(name: str, loaders, n_features: int, epochs: int = 30,
              patience: int = 8, lr: float = 2e-3) -> dict:
    """한 모델을 학습한다. 검증 AUC 가 최고일 때만 체크포인트를 남긴다."""
    import torch
    import torch.nn as nn

    torch.manual_seed(42)
    model = make_model(name, n_features)
    n_param = sum(p.numel() for p in model.parameters())

    # 클래스 불균형 보정용 pos_weight 를 학습 분할에서 구한다.
    pos = sum(float(yb.sum()) for _, yb in loaders["train"])
    tot = sum(len(yb) for _, yb in loaders["train"])
    pos_weight = torch.tensor([(tot - pos) / max(pos, 1.0)])
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=0.5, patience=3)

    ckpt = OUT_DIR / f"ts_{name}.pt"
    best_auc, best_ep, bad = -1.0, 0, 0
    t0 = time.perf_counter()

    for ep in range(1, epochs + 1):
        model.train()
        total = 0.0
        n = 0
        for xb, yb in loaders["train"]:
            opt.zero_grad()
            loss = crit(forward_logits(name, model, xb), yb)
            loss.backward()
            # 기울기 폭발은 RNN 에서 흔하다. 클리핑은 선택이 아니라 필수다.
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss) * len(xb)
            n += len(xb)
        val_auc, _, _ = evaluate(name, model, loaders["val"])
        sched.step(val_auc)

        if val_auc > best_auc + 1e-4:
            best_auc, best_ep, bad = val_auc, ep, 0
            # 체크포인트에는 가중치만이 아니라 재현에 필요한 정보를 함께 담는다.
            torch.save({"name": name, "state_dict": model.state_dict(),
                        "n_features": n_features, "epoch": ep,
                        "val_auc": val_auc}, ckpt)
        else:
            bad += 1
        if ep % 5 == 0 or ep == 1:
            print(f"  ep{ep:3d}  loss={total / n:.4f}  val_AUC={val_auc:.4f}"
                  f"  lr={opt.param_groups[0]['lr']:.5f}"
                  f"{'  *best' if bad == 0 else ''}")
        if bad >= patience:
            print(f"  조기 종료: {patience} epoch 동안 개선 없음 (ep{ep})")
            break

    fit_s = time.perf_counter() - t0

    # 평가는 항상 **최고 검증 성능 체크포인트**로 한다. 마지막 epoch
    # 가중치로 평가하면 과적합된 모델을 보고하게 된다.
    state = torch.load(ckpt, weights_only=False)
    model.load_state_dict(state["state_dict"])
    t1 = time.perf_counter()
    test_auc, prob, y = evaluate(name, model, loaders["test"])
    infer_ms = (time.perf_counter() - t1) * 1000.0 / max(len(y), 1)

    pred = (prob >= 0.5).astype(int)
    tp = int(((y == 1) & (pred == 1)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {"name": name, "params": n_param, "best_ep": best_ep,
            "val_auc": best_auc, "test_auc": test_auc, "fit_s": fit_s,
            "infer_ms": infer_ms, "prec": prec, "rec": rec,
            "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0}


def main() -> None:
    try:
        import torch                                        # noqa: F401
        from sklearn.metrics import roc_auc_score           # noqa: F401
    except ImportError:
        print("[안내] PyTorch 와 scikit-learn 이 필요하다.")
        print("       pip install torch --index-url "
              "https://download.pytorch.org/whl/cpu")
        print("       pip install scikit-learn")
        print("       설치 후 다시 실행하면 LSTM/GRU/TCN/Transformer 를")
        print("       같은 조건으로 학습해 비교표를 출력한다.")
        return
    if not DATA.exists():
        print(f"[안내] {DATA} 가 없다. 먼저 python 06_failure/make_labels.py")
        return

    import torch
    torch.set_num_threads(4)

    from window_dataset import build_torch_loaders
    loaders, _, (n_features, window) = build_torch_loaders(batch_size=64)
    print(f"윈도우 {window}, 채널 {n_features}\n")

    names = sys.argv[1:] or list(MODEL_NAMES)
    results = []
    for name in names:
        print(f"=== {name} 학습 ===")
        results.append(train_one(name, loaders, n_features))
        print()

    hdr = (f"{'모델':<13s}{'파라미터':>10s}{'best_ep':>8s}{'val_AUC':>9s}"
           f"{'test_AUC':>10s}{'F1':>7s}{'학습초':>8s}{'추론ms':>8s}")
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        print(f"{r['name']:<13s}{r['params']:10d}{r['best_ep']:8d}"
              f"{r['val_auc']:9.4f}{r['test_auc']:10.4f}{r['f1']:7.3f}"
              f"{r['fit_s']:8.1f}{r['infer_ms']:8.3f}")

    print("\n검증 AUC 로 모델을 고르고, 평가 AUC 는 한 번만 본다.")
    print("평가 집합을 보며 하이퍼파라미터를 고치면 그 집합은 더 이상")
    print("평가 집합이 아니다. 체크포인트는 data/ts_<모델>.pt 에 있다.")


if __name__ == "__main__":
    main()
