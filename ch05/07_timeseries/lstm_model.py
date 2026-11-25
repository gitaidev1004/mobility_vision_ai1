# -*- coding: utf-8 -*-
"""LSTM / GRU 분류 모델 정의. torch 미설치 시 안내 후 종료.

두 모델을 같은 클래스로 묶어 cell='lstm'|'gru' 로 바꿔 쓸 수 있게 한다.
GRU 는 게이트가 3개에서 2개로 줄어 파라미터가 약 3/4 이고 CPU 추론이
조금 빠르다. 어느 쪽이 정확한지는 데이터마다 다르므로 비교해야 한다.

실행:  python 07_timeseries/lstm_model.py
"""
from __future__ import annotations


def param_count(cell: str, n_features: int, hidden: int, layers: int) -> int:
    """RNN 계열 파라미터 수를 공식으로 계산한다(torch 없이도 동작).

    LSTM 한 층: 4 * (hidden*(in+hidden) + 2*hidden)
    GRU  한 층: 3 * (hidden*(in+hidden) + 2*hidden)
    (bias 가 입력·은닉 양쪽에 있어 2*hidden)
    """
    g = {"lstm": 4, "gru": 3}[cell]
    total, in_dim = 0, n_features
    for _ in range(layers):
        total += g * (hidden * (in_dim + hidden) + 2 * hidden)
        in_dim = hidden
    return total


def build(cell: str = "lstm", n_features: int = 13, hidden: int = 64,
          layers: int = 2, dropout: float = 0.2, bidirectional: bool = False):
    """RNN 분류기를 만든다. torch 가 필요하다."""
    import torch
    import torch.nn as nn

    class RNNClassifier(nn.Module):
        """RNN 백본 + 어텐션 풀링 + 분류 헤드.

        마지막 시점 은닉 상태만 쓰면 중간에 나타난 짧은 전조를 놓칠 수
        있다. 그래서 전 시점 출력에 어텐션 가중치를 학습시켜 합한다.
        """

        def __init__(self) -> None:
            super().__init__()
            rnn_cls = {"lstm": nn.LSTM, "gru": nn.GRU}[cell]
            self.rnn = rnn_cls(
                input_size=n_features, hidden_size=hidden, num_layers=layers,
                batch_first=True, dropout=dropout if layers > 1 else 0.0,
                bidirectional=bidirectional)
            out_dim = hidden * (2 if bidirectional else 1)
            # 어텐션 점수를 내는 작은 네트워크
            self.attn = nn.Sequential(nn.Linear(out_dim, out_dim // 2),
                                      nn.Tanh(), nn.Linear(out_dim // 2, 1))
            self.norm = nn.LayerNorm(out_dim)
            self.drop = nn.Dropout(dropout)
            self.head = nn.Linear(out_dim, 1)
            self.cell = cell

        def forward(self, x):                       # x: (B, T, F)
            out, _ = self.rnn(x)                    # (B, T, H)
            w = torch.softmax(self.attn(out), dim=1)   # (B, T, 1)
            ctx = (out * w).sum(dim=1)              # (B, H)
            return self.head(self.drop(self.norm(ctx))).squeeze(-1)

        def attention_weights(self, x):
            """설명용. 어느 시점을 보고 판단했는지 돌려준다."""
            out, _ = self.rnn(x)
            return torch.softmax(self.attn(out), dim=1).squeeze(-1)

    return RNNClassifier()


def main() -> None:
    print("[RNN 계열 파라미터 수 비교 — 입력 13채널]")
    print(f"{'모델':<10s}{'은닉':>6s}{'층':>4s}{'파라미터':>11s}{'LSTM 대비':>10s}")
    for hidden in (32, 64, 128):
        for layers in (1, 2):
            n_l = param_count("lstm", 13, hidden, layers)
            n_g = param_count("gru", 13, hidden, layers)
            print(f"{'LSTM':<10s}{hidden:6d}{layers:4d}{n_l:11d}{'1.00':>10s}")
            print(f"{'GRU':<10s}{hidden:6d}{layers:4d}{n_g:11d}"
                  f"{n_g / n_l:10.2f}")

    try:
        import torch
    except ImportError:
        print("\n[안내] PyTorch 가 설치되어 있지 않아 모델 생성은 건너뛴다.")
        print("       pip install torch --index-url "
              "https://download.pytorch.org/whl/cpu")
        print("       위 파라미터 수 표는 공식으로 계산한 값이므로 "
              "설치 여부와 무관하게 유효하다.")
        return

    x = torch.randn(4, 24, 13)
    for cell in ("lstm", "gru"):
        for bi in (False, True):
            m = build(cell=cell, bidirectional=bi)
            n = sum(p.numel() for p in m.parameters())
            tag = "양방향" if bi else "단방향"
            print(f"\n{cell.upper()} {tag}: 파라미터 {n:,} "
                  f"출력 shape={tuple(m(x).shape)}")
            if not bi:
                # 양방향은 미래를 보므로 실시간 예지보전에는 쓸 수 없다.
                w = m.attention_weights(x)
                print(f"  어텐션 가중치 shape={tuple(w.shape)} "
                      f"합={float(w[0].sum()):.3f}")
    print("\n주의: 양방향 RNN 은 윈도우 안의 미래 시점을 함께 본다. "
          "배치 분석에는 쓸 수 있으나 실시간 예측에는 쓰지 않는다.")


if __name__ == "__main__":
    main()
