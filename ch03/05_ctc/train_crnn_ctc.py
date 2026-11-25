# -*- coding: utf-8 -*-
"""CRNN + CTC 학습 루프 — 체크포인트, EarlyStopping, 문자정확도/완전일치율 로깅.

실행:  python 05_ctc/train_crnn_ctc.py --epochs 30 --batch-size 32
torch 가 없으면 학습 설정과 지표 정의만 출력하고 종료한다.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "03_dataset"))
sys.path.append(str(ROOT / "04_crnn"))
sys.path.append(str(Path(__file__).resolve().parent))
from charset import BLANK_INDEX, NUM_CLASSES  # noqa: E402

#: 기본 하이퍼파라미터. 교재 본문의 하이퍼파라미터표와 값을 일치시킨다.
DEFAULTS = {
    "epochs": 30,
    "batch_size": 32,
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "optimizer": "AdamW",
    "scheduler": "CosineAnnealingLR",
    "grad_clip": 5.0,
    "patience": 7,
    "seq_len": 40,
}


def levenshtein(a: str, b: str) -> int:
    """두 문자열의 편집 거리. 문자 오류율(CER) 계산에 쓴다.

    번호판은 글자 수가 짧아 O(len(a)·len(b)) 의 단순 DP 로 충분하다.
    단순 위치 비교(zip)로 세면 한 글자가 빠졌을 때 뒤가 전부 틀린 것으로
    집계돼 오류가 과대평가된다. 삽입·삭제를 다루는 편집 거리가 올바른 척도다.
    """
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(min(prev[j] + 1,            # 삭제
                           cur[j - 1] + 1,         # 삽입
                           prev[j - 1] + (ca != cb)))  # 치환
        prev = cur
    return prev[-1]


def compute_metrics(preds: list[str], truths: list[str]) -> dict:
    """문자 정확도, CER, 완전 일치율을 계산한다.

    번호판 시스템에서 가장 중요한 지표는 **완전 일치율**이다.
    문자 정확도가 98% 여도 7글자 중 평균 0.14자가 틀린다는 뜻이고,
    그러면 번호판 7장 중 1장꼴로 차량 조회가 실패한다. 반드시 둘을 함께 본다.
    """
    if not truths:
        return {"char_acc": 0.0, "cer": 1.0, "exact": 0.0, "count": 0}
    total_chars = sum(len(t) for t in truths)
    total_dist = sum(levenshtein(p, t) for p, t in zip(preds, truths))
    exact = sum(1 for p, t in zip(preds, truths) if p == t)
    return {
        "char_acc": round(1.0 - total_dist / max(total_chars, 1), 4),
        "cer": round(total_dist / max(total_chars, 1), 4),
        "exact": round(exact / len(truths), 4),
        "count": len(truths),
    }


class EarlyStopping:
    """검증 지표가 patience epoch 동안 개선되지 않으면 학습을 멈춘다.

    감시 지표를 손실이 아니라 완전 일치율로 두는 이유:
      CTC 손실은 계속 내려가는데 완전 일치율은 정체하는 구간이 흔하다.
      우리가 원하는 것은 낮은 손실이 아니라 정확히 읽힌 번호판이므로,
      최종 목표 지표를 직접 감시해야 과적합 직전에 멈출 수 있다.
    """

    def __init__(self, patience: int = 7, min_delta: float = 1e-4) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best = -float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, score: float) -> bool:
        """지표를 받아 개선 여부를 돌려준다(True 면 best 갱신)."""
        if score > self.best + self.min_delta:
            self.best = score
            self.counter = 0
            return True
        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return False


def print_config(args: argparse.Namespace) -> None:
    """학습 설정을 표로 출력한다."""
    print(f"{'항목':18s} {'값':>22s}")
    print("-" * 42)
    rows = [
        ("입력 크기", "32 x 160 (회색조)"),
        ("시퀀스 길이 T", str(DEFAULTS["seq_len"])),
        ("클래스 수 C", f"{NUM_CLASSES} (문자 {NUM_CLASSES - 1} + blank)"),
        ("blank 인덱스", str(BLANK_INDEX)),
        ("epochs", str(args.epochs)),
        ("batch size", str(args.batch_size)),
        ("learning rate", f"{args.lr:g}"),
        ("weight decay", f"{DEFAULTS['weight_decay']:g}"),
        ("optimizer", DEFAULTS["optimizer"]),
        ("scheduler", DEFAULTS["scheduler"]),
        ("gradient clip", str(DEFAULTS["grad_clip"])),
        ("early stop patience", str(args.patience)),
    ]
    for key, value in rows:
        print(f"{key:18s} {value:>22s}")


def main() -> None:
    parser = argparse.ArgumentParser(description="CRNN + CTC 학습")
    parser.add_argument("--train-index", type=str, default="data/index/train.jsonl")
    parser.add_argument("--val-index", type=str, default="data/index/val.jsonl")
    parser.add_argument("--out", type=str, default="runs")
    parser.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    parser.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    parser.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    parser.add_argument("--patience", type=int, default=DEFAULTS["patience"])
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    print("=== 학습 설정 ===")
    print_config(args)

    try:
        import torch
        from torch import nn
    except ImportError:
        print("\n[안내] torch 가 없어 학습을 시작할 수 없다.")
        print("       pip install torch --index-url https://download.pytorch.org/whl/cpu")
        print("\n지표 정의만 확인한다(가짜 예측으로 계산).")
        preds = ["12가3456", "12가3457", "12가345", "87로2345"]
        truths = ["12가3456", "12가3456", "12가3456", "87로2345"]
        metrics = compute_metrics(preds, truths)
        print(f"  예측 {preds}")
        print(f"  정답 {truths}")
        print(f"  문자 정확도 {metrics['char_acc']:.4f}, CER {metrics['cer']:.4f}, "
              f"완전 일치율 {metrics['exact']:.4f}")
        print("  → 문자 정확도는 0.9 를 넘지만 완전 일치율은 0.5 다. 둘을 함께 봐야 한다.")
        raise SystemExit(0)

    from crnn import CRNN
    from ctc_decoder import greedy_decode
    from dataset import build_loader

    torch.set_num_threads(args.threads)
    torch.manual_seed(42)

    train_path, val_path = Path(args.train_index), Path(args.val_index)
    if not train_path.exists() or not val_path.exists():
        print(f"\n[안내] 인덱스 파일이 없다: {train_path} / {val_path}")
        print("       synth_plate.py → make_lmdb_like.py 순서로 먼저 실행한다.")
        raise SystemExit(0)

    train_loader = build_loader(train_path, batch_size=args.batch_size, shuffle=True)
    val_loader = build_loader(val_path, batch_size=args.batch_size, shuffle=False)

    model = CRNN()
    # zero_infinity=True 가 중요하다. T < L 인 비정상 샘플이 한 건이라도 섞이면
    # 손실이 inf 가 되고 역전파에서 전체 가중치가 NaN 으로 오염된다.
    criterion = nn.CTCLoss(blank=BLANK_INDEX, reduction="mean", zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=DEFAULTS["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           T_max=args.epochs)
    stopper = EarlyStopping(patience=args.patience)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []

    print(f"\n학습 {len(train_loader.dataset):,}건 / 검증 {len(val_loader.dataset):,}건\n")
    print(f"{'ep':>3s} {'train_loss':>11s} {'val_loss':>9s} {'char_acc':>9s} "
          f"{'exact':>7s} {'lr':>9s} {'sec':>6s}")
    print("-" * 62)

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        model.train()
        train_loss, n_batches = 0.0, 0
        for images, targets, target_lengths, _ in train_loader:
            optimizer.zero_grad(set_to_none=True)
            log_probs = model(images)                         # (T, N, C)
            input_lengths = torch.full((images.size(0),), log_probs.size(0),
                                       dtype=torch.long)
            loss = criterion(log_probs, targets, input_lengths, target_lengths)
            if not torch.isfinite(loss):
                # 손실이 터진 배치는 건너뛴다. 멈추는 것보다 넘기는 편이 낫다.
                continue
            loss.backward()
            # RNN 은 기울기 폭주가 잦다. 클리핑은 선택이 아니라 필수다.
            nn.utils.clip_grad_norm_(model.parameters(), DEFAULTS["grad_clip"])
            optimizer.step()
            train_loss += float(loss)
            n_batches += 1
        train_loss /= max(n_batches, 1)

        model.eval()
        val_loss, v_batches = 0.0, 0
        preds: list[str] = []
        truths: list[str] = []
        with torch.no_grad():
            for images, targets, target_lengths, texts in val_loader:
                log_probs = model(images)
                input_lengths = torch.full((images.size(0),), log_probs.size(0),
                                           dtype=torch.long)
                loss = criterion(log_probs, targets, input_lengths, target_lengths)
                if torch.isfinite(loss):
                    val_loss += float(loss)
                    v_batches += 1
                probs = log_probs.permute(1, 0, 2).exp().numpy()   # (N, T, C)
                for i in range(probs.shape[0]):
                    preds.append(greedy_decode(probs[i])["text"])
                truths.extend(texts)
        val_loss /= max(v_batches, 1)
        metrics = compute_metrics(preds, truths)
        scheduler.step()

        elapsed = time.perf_counter() - t0
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"{epoch:3d} {train_loss:11.4f} {val_loss:9.4f} "
              f"{metrics['char_acc']:9.4f} {metrics['exact']:7.4f} "
              f"{lr_now:9.2e} {elapsed:6.1f}")

        record = {"epoch": epoch, "train_loss": round(train_loss, 5),
                  "val_loss": round(val_loss, 5), "lr": lr_now, **metrics}
        history.append(record)

        improved = stopper.step(metrics["exact"])
        checkpoint = {"model": model.state_dict(), "epoch": epoch,
                      "metrics": metrics, "num_classes": NUM_CLASSES,
                      "blank_index": BLANK_INDEX}
        torch.save(checkpoint, out_dir / "last.pt")
        if improved:
            torch.save(checkpoint, out_dir / "best.pt")
            print(f"    → best 갱신 (완전 일치율 {metrics['exact']:.4f}), best.pt 저장")
        if stopper.should_stop:
            print(f"\nEarlyStopping: {args.patience} epoch 동안 개선이 없어 중단한다.")
            break

    (out_dir / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n최고 완전 일치율: {stopper.best:.4f}")
    print(f"체크포인트: {(out_dir / 'best.pt').resolve()}")
    print(f"학습 기록  : {(out_dir / 'history.json').resolve()}")


if __name__ == "__main__":
    main()
