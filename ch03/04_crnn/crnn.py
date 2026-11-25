# -*- coding: utf-8 -*-
"""CRNN 모델 정의 — CNN 백본 + BiLSTM + Linear.

입력 32x160 회색조 번호판 → (T=40, C=78) 프레임별 문자 확률.

실행:  python 04_crnn/crnn.py
torch 가 없으면 레이어별 shape 표를 계산해 출력하고 종료한다.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "03_dataset"))
from charset import NUM_CLASSES  # noqa: E402

#: 입력 규격. 높이를 32 로 고정하는 이유는 CNN 이 높이를 1 로 줄이는 구조이기 때문이다.
IMG_H, IMG_W = 32, 160

#: 이 설정에서 CNN 이 만들어 내는 시퀀스 길이. 가로 160 을 4배 축소해 40 이 된다.
#: CTC 는 T >= L(라벨 길이) 를 요구한다. 번호판 최대 9자이므로 40 은 충분하다.
SEQ_LEN = 40


def shape_table() -> list[tuple[str, str, str]]:
    """레이어별 출력 shape 을 손으로 계산해 표로 돌려준다(torch 없이도 확인 가능)."""
    return [
        ("입력",            "(N, 1, 32, 160)",   "회색조 1채널, 높이 32 고정"),
        ("conv1 3x3/64",    "(N, 64, 32, 160)",  "padding=1 이라 크기 유지"),
        ("pool1 2x2",       "(N, 64, 16, 80)",   "높이·너비 모두 1/2"),
        ("conv2 3x3/128",   "(N, 128, 16, 80)",  "채널 2배"),
        ("pool2 2x2",       "(N, 128, 8, 40)",   "여기서 너비가 40 으로 확정된다"),
        ("conv3 3x3/256",   "(N, 256, 8, 40)",   "BatchNorm + ReLU"),
        ("conv4 3x3/256",   "(N, 256, 8, 40)",   "같은 해상도에서 한 번 더"),
        ("pool3 (2,1)",     "(N, 256, 4, 40)",   "높이만 1/2 — 너비는 지킨다"),
        ("conv5 3x3/512",   "(N, 512, 4, 40)",   "BatchNorm + ReLU"),
        ("pool4 (2,1)",     "(N, 512, 2, 40)",   "높이만 1/2"),
        ("conv6 2x2/512",   "(N, 512, 1, 39)",   "padding 없음 → 높이 1, 너비 -1"),
        ("pad + squeeze",   "(N, 512, 40)",      "너비 40 복원 후 높이 축 제거"),
        ("permute",         "(N, 40, 512)",      "(배치, 시간, 특징) 순서로 변경"),
        ("BiLSTM-1 256",    "(N, 40, 256)",      "양방향 512 를 선형으로 256 복원"),
        ("BiLSTM-2 256",    "(N, 40, 256)",      "문맥을 한 번 더 누적"),
        ("Linear 256→78",   "(N, 40, 78)",       "프레임별 클래스 로짓"),
        ("log_softmax",     "(40, N, 78)",       "CTCLoss 입력은 (T, N, C) 순서"),
    ]


def print_shape_table() -> None:
    """shape 표를 콘솔에 출력한다."""
    rows = shape_table()
    print(f"{'레이어':18s} {'출력 shape':20s} 설명")
    print("-" * 78)
    for name, shape, note in rows:
        print(f"{name:18s} {shape:20s} {note}")
    print(f"\n최종 시퀀스 길이 T = {SEQ_LEN}, 클래스 수 C = {NUM_CLASSES}")
    print(f"파라미터 수(계산값) = 약 {6_104_782:,} 개, float32 기준 약 23.3 MB")
    print("가로 축소율이 4배이므로 입력 폭 160 → 시퀀스 40 이다.")
    print("폭을 늘리면 T 가 늘어 긴 번호판에 유리하지만 추론 시간도 비례해 늘어난다.")


try:
    import torch
    from torch import nn
except ImportError:
    print("[안내] torch 가 없어 모델을 만들 수 없다. shape 표만 출력한다.")
    print("       pip install torch --index-url https://download.pytorch.org/whl/cpu\n")
    print_shape_table()
    raise SystemExit(0)


class BidirectionalLSTM(nn.Module):
    """BiLSTM 한 층 + 선형 투영.

    투영을 붙이는 이유: 양방향 LSTM 의 출력은 hidden*2 차원이다. 층을 쌓을 때마다
    차원이 두 배로 불어나므로, 매 층마다 원래 차원으로 되돌려 놓는다.
    """

    def __init__(self, in_dim: int, hidden: int, out_dim: int) -> None:
        super().__init__()
        self.rnn = nn.LSTM(in_dim, hidden, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden * 2, out_dim)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """(N, T, in_dim) -> (N, T, out_dim)"""
        out, _ = self.rnn(x)
        return self.linear(out)


class CRNN(nn.Module):
    """번호판 전용 CRNN.

    구조 설계에서 가장 중요한 결정은 **풀링을 비대칭으로 두는 것**이다.
    뒤쪽 두 풀링을 (2, 1) 로 두어 높이만 줄이고 너비는 지킨다. 너비가 곧
    시퀀스 길이이므로, 대칭 풀링을 쓰면 T 가 10 까지 줄어 9글자 번호판을
    CTC 로 표현할 수 없게 된다(blank 자리가 모자란다).
    """

    def __init__(self, num_classes: int = NUM_CLASSES,
                 in_channels: int = 1, hidden: int = 256) -> None:
        super().__init__()
        self.num_classes = num_classes

        def conv_bn(cin: int, cout: int, k: int = 3, s: int = 1, p: int = 1):
            return nn.Sequential(
                nn.Conv2d(cin, cout, k, s, p),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
            )

        self.cnn = nn.Sequential(
            conv_bn(in_channels, 64),          # (N,  64, 32, 160)
            nn.MaxPool2d(2, 2),                # (N,  64, 16,  80)
            conv_bn(64, 128),                  # (N, 128, 16,  80)
            nn.MaxPool2d(2, 2),                # (N, 128,  8,  40)
            conv_bn(128, 256),                 # (N, 256,  8,  40)
            conv_bn(256, 256),                 # (N, 256,  8,  40)
            nn.MaxPool2d((2, 1), (2, 1)),      # (N, 256,  4,  40)  높이만 축소
            conv_bn(256, 512),                 # (N, 512,  4,  40)
            nn.MaxPool2d((2, 1), (2, 1)),      # (N, 512,  2,  40)
            nn.Conv2d(512, 512, 2, 1, 0),      # (N, 512,  1,  39)
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, hidden, hidden),
            BidirectionalLSTM(hidden, hidden, hidden),
        )
        self.head = nn.Linear(hidden, num_classes)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """(N, 1, 32, 160) -> (T, N, C) 로그 확률. CTCLoss 입력 규약에 맞춘다."""
        feat = self.cnn(x)                           # (N, 512, 1, 39)
        if feat.shape[2] != 1:
            # 입력 높이가 32 가 아니면 높이가 1 로 떨어지지 않는다. 평균으로 눌러 준다.
            feat = feat.mean(dim=2, keepdim=True)
        # 마지막 conv 가 너비를 1 줄이므로 오른쪽을 복제해 40 으로 되돌린다.
        # 이렇게 해야 T 가 항상 입력폭/4 로 예측 가능해지고, 추론 코드에서
        # 문자 위치를 원본 좌표로 되돌릴 때 계산이 단순해진다.
        feat = nn.functional.pad(feat, (0, 1), mode="replicate")   # (N, 512, 1, 40)
        feat = feat.squeeze(2).permute(0, 2, 1)      # (N, 40, 512)
        seq = self.dropout(self.rnn(feat))           # (N, 40, 256)
        logits = self.head(seq)                      # (N, 40, 78)
        return logits.log_softmax(dim=-1).permute(1, 0, 2)   # (40, N, 78)

    @torch.no_grad()
    def predict_proba(self, x: "torch.Tensor") -> "torch.Tensor":
        """추론 전용 — (N, T, C) 확률을 돌려준다(디코더가 쓰기 편한 순서)."""
        self.eval()
        return self(x).permute(1, 0, 2).exp()


def main() -> None:
    torch.manual_seed(0)
    model = CRNN()
    model.eval()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"클래스 수   : {NUM_CLASSES}")
    print(f"파라미터 수 : {n_params:,}")
    print(f"모델 크기   : 약 {n_params * 4 / 1024 / 1024:.1f} MB (float32)\n")

    print_shape_table()

    # 실제 텐서를 흘려 보내 표의 값이 맞는지 확인한다.
    dummy = torch.randn(2, 1, IMG_H, IMG_W)
    with torch.no_grad():
        out = model(dummy)
        proba = model.predict_proba(dummy)
    print(f"\n[실측] 입력 {tuple(dummy.shape)} → 출력 {tuple(out.shape)} (T, N, C)")
    print(f"[실측] predict_proba {tuple(proba.shape)} (N, T, C), "
          f"행 합 = {float(proba[0, 0].sum()):.4f}")


if __name__ == "__main__":
    main()
