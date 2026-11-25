# -*- coding: utf-8 -*-
"""시계열 Transformer 인코더 정의. torch 미설치 시 안내 후 종료.

NLP 의 Transformer 를 센서 시계열에 쓸 때 바꿀 것이 세 가지다.
  1) 토큰 임베딩 → 센서 벡터를 선형 투영(값 자체가 이미 수치다)
  2) 위치 인코딩 → 시간 간격이 균일하지 않으면 시간 값을 함께 넣는다
  3) 인과 마스크 → 미래를 보지 않도록 상삼각 마스크를 쓴다

실행:  python 07_timeseries/ts_transformer.py
"""
from __future__ import annotations

import math


def attention_flops(seq_len: int, d_model: int, n_layers: int,
                    ff_mult: int = 4) -> dict[str, float]:
    """인코더 1회 순전파의 대략적인 연산량을 센다(torch 없이 동작).

    셀프 어텐션은 시퀀스 길이의 **제곱**에 비례한다. 이것이 긴 시계열에
    Transformer 를 그대로 쓰기 어려운 이유다.
      QKV 투영   : 3 * L * d^2
      점수 계산  : L^2 * d  (QK^T)
      가중 합    : L^2 * d  (AV)
      출력 투영  : L * d^2
      FFN        : 2 * L * d * (ff_mult*d)
    """
    l, d = seq_len, d_model
    attn = 3 * l * d * d + 2 * l * l * d + l * d * d
    ffn = 2 * l * d * (ff_mult * d)
    per_layer = attn + ffn
    return {"attn_mflops": attn * n_layers / 1e6,
            "ffn_mflops": ffn * n_layers / 1e6,
            "total_mflops": per_layer * n_layers / 1e6,
            "attn_ratio": attn / per_layer}


def encoder_params(d_model: int, n_layers: int, ff_mult: int = 4) -> int:
    """인코더 파라미터 수. 어텐션 4개 투영 + FFN 2개 선형 + LayerNorm 2개."""
    d, f = d_model, d_model * ff_mult
    per = 4 * (d * d + d) + (d * f + f) + (f * d + d) + 4 * d
    return per * n_layers


def build(n_features: int = 13, d_model: int = 64, n_heads: int = 4,
          n_layers: int = 2, ff_mult: int = 4, dropout: float = 0.1,
          max_len: int = 512):
    """시계열 Transformer 인코더 분류기를 만든다."""
    import torch
    import torch.nn as nn

    class PositionalEncoding(nn.Module):
        """사인·코사인 고정 위치 인코딩. 학습 파라미터가 없어 가볍다."""

        def __init__(self) -> None:
            super().__init__()
            pe = torch.zeros(max_len, d_model)
            pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
            div = torch.exp(torch.arange(0, d_model, 2).float()
                            * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div)
            self.register_buffer("pe", pe.unsqueeze(0))     # (1, max_len, d)

        def forward(self, x):
            return x + self.pe[:, :x.size(1)]

    class TSTransformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            # 센서 벡터를 d_model 차원으로 투영한다(토큰 임베딩 대체).
            self.proj = nn.Linear(n_features, d_model)
            self.pos = PositionalEncoding()
            layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads,
                dim_feedforward=d_model * ff_mult, dropout=dropout,
                activation="gelu", batch_first=True, norm_first=True)
            self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)
            # CLS 토큰 하나를 앞에 붙여 그 출력으로 분류한다.
            self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.normal_(self.cls, std=0.02)
            self.norm = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, 1)

        @staticmethod
        def causal_mask(t: int) -> "torch.Tensor":
            """미래를 보지 못하게 하는 상삼각 마스크. True 인 위치가 차단된다."""
            return torch.triu(torch.ones(t, t, dtype=torch.bool), diagonal=1)

        def forward(self, x, causal: bool = False):         # x: (B, T, F)
            h = self.pos(self.proj(x))
            b = h.size(0)
            h = torch.cat([self.cls.expand(b, -1, -1), h], dim=1)
            mask = None
            if causal:
                # CLS 토큰이 앞에 붙으므로 길이를 T+1 로 맞춘다.
                mask = self.causal_mask(h.size(1))
                mask[:, 0] = False               # 모두 CLS 는 볼 수 있게 한다
            h = self.enc(h, mask=mask)
            return self.head(self.norm(h[:, 0])).squeeze(-1)

    return TSTransformer()


def main() -> None:
    print("[셀프 어텐션의 제곱 비용 — d_model=64, 2층]")
    print(f"{'윈도우 T':>9s}{'어텐션 MFLOPs':>15s}{'FFN MFLOPs':>12s}"
          f"{'합계':>10s}{'어텐션 비중':>12s}")
    for t in (12, 24, 48, 96, 192, 384):
        f = attention_flops(t, 64, 2)
        print(f"{t:9d}{f['attn_mflops']:15.2f}{f['ffn_mflops']:12.2f}"
              f"{f['total_mflops']:10.2f}{f['attn_ratio']:12.3f}")

    print("\n[모델 크기별 파라미터 수]")
    print(f"{'d_model':>9s}{'층':>4s}{'파라미터':>11s}{'FP32 MB':>10s}")
    for d in (32, 64, 128):
        for layers in (2, 4):
            n = encoder_params(d, layers)
            print(f"{d:9d}{layers:4d}{n:11d}{n * 4 / 1024 / 1024:10.3f}")

    print("\n윈도우를 2배로 늘리면 어텐션 연산은 4배가 된다. 윈도우가 96 을")
    print("넘기면 TCN(07_timeseries/tcn_model.py)이 더 경제적인 선택이 된다.")

    try:
        import torch
    except ImportError:
        print("\n[안내] PyTorch 가 설치되어 있지 않아 모델 생성은 건너뛴다.")
        print("       pip install torch --index-url "
              "https://download.pytorch.org/whl/cpu")
        print("       위 연산량·파라미터 표는 공식 계산값이라 그대로 유효하다.")
        return

    m = build()
    n = sum(p.numel() for p in m.parameters())
    x = torch.randn(4, 24, 13)
    print(f"\n[torch] 파라미터 {n:,}")
    print(f"  일반 어텐션   출력 shape={tuple(m(x).shape)}")
    print(f"  인과 마스크   출력 shape={tuple(m(x, causal=True).shape)}")
    print("  실시간 예측에는 causal=True 로 호출해 미래 누설을 막는다.")


if __name__ == "__main__":
    main()
