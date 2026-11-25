# -*- coding: utf-8 -*-
"""Temporal CNN(팽창 합성곱) 정의와 수용영역 계산.

수용영역 계산 부분은 PyTorch 없이도 동작한다. 모델 정의는 torch 가
설치되어 있을 때만 사용된다.

실행:  python 07_timeseries/tcn_model.py
"""
from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:                          # torch 미설치 환경
    HAS_TORCH = False


# -------------------------------------------------- 수용영역(순수 파이썬)
def receptive_field(kernel_size: int, n_levels: int,
                    dilation_base: int = 2, n_conv_per_block: int = 2) -> int:
    """팽창 합성곱 스택의 수용영역 길이를 계산한다.

    블록 i(0-base)의 팽창 계수는 dilation_base**i 이고, 블록 안에
    커널 k 합성곱이 n_conv_per_block 개 쌓인다. 합성곱 하나가 늘리는
    수용영역은 (k-1)*d 이므로 전체는 다음과 같다.

        RF = 1 + n_conv * sum_i (k-1) * base**i
    """
    rf = 1
    for i in range(n_levels):
        d = dilation_base ** i
        rf += n_conv_per_block * (kernel_size - 1) * d
    return rf


def levels_for_window(window: int, kernel_size: int = 3,
                      dilation_base: int = 2, n_conv_per_block: int = 2,
                      max_levels: int = 16) -> int:
    """입력 윈도우 전체를 덮는 데 필요한 최소 블록 수를 구한다."""
    for n in range(1, max_levels + 1):
        if receptive_field(kernel_size, n, dilation_base, n_conv_per_block) >= window:
            return n
    return max_levels


def param_estimate(n_levels: int, ch: int, kernel_size: int, in_ch: int) -> int:
    """TCN 파라미터 수를 대략 계산한다(합성곱 가중치와 1x1 다운샘플 기준)."""
    total = 0
    prev = in_ch
    for _ in range(n_levels):
        total += prev * ch * kernel_size + ch        # 첫 합성곱 + bias
        total += ch * ch * kernel_size + ch          # 두 번째 합성곱
        if prev != ch:
            total += prev * ch + ch                  # 1x1 잔차 정합
        prev = ch
    return total


# ----------------------------------------------------------- 모델 정의
if HAS_TORCH:

    class Chomp1d(nn.Module):
        """인과(causal) 합성곱을 만들기 위해 오른쪽 패딩을 잘라낸다.

        좌우 대칭 패딩을 쓰면 미래 시점 정보가 현재 출력에 섞인다.
        예지보전에서는 미래를 볼 수 없으므로 반드시 잘라야 한다.
        """

        def __init__(self, chomp: int) -> None:
            super().__init__()
            self.chomp = chomp

        def forward(self, x):                        # (B, C, T)
            return x[:, :, :-self.chomp].contiguous() if self.chomp > 0 else x

    class TemporalBlock(nn.Module):
        """팽창 인과 합성곱 2개 + 잔차 연결로 구성된 블록."""

        def __init__(self, in_ch: int, out_ch: int, kernel_size: int,
                     dilation: int, dropout: float = 0.2) -> None:
            super().__init__()
            pad = (kernel_size - 1) * dilation
            self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size,
                                   padding=pad, dilation=dilation)
            self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size,
                                   padding=pad, dilation=dilation)
            self.net = nn.Sequential(
                self.conv1, Chomp1d(pad), nn.ReLU(), nn.Dropout(dropout),
                self.conv2, Chomp1d(pad), nn.ReLU(), nn.Dropout(dropout),
            )
            # 채널 수가 다르면 1x1 합성곱으로 맞춘 뒤 더한다.
            self.down = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
            self.relu = nn.ReLU()

        def forward(self, x):
            out = self.net(x)
            res = x if self.down is None else self.down(x)
            return self.relu(out + res)

    class TCNClassifier(nn.Module):
        """TCN 백본 + 분류 헤드. 입력은 (B, T, F) 형태를 받는다."""

        def __init__(self, n_features: int, n_classes: int = 2,
                     channels: int = 48, n_levels: int = 4,
                     kernel_size: int = 3, dropout: float = 0.2) -> None:
            super().__init__()
            blocks, in_ch = [], n_features
            for i in range(n_levels):
                blocks.append(TemporalBlock(in_ch, channels, kernel_size,
                                            dilation=2 ** i, dropout=dropout))
                in_ch = channels
            self.tcn = nn.Sequential(*blocks)
            self.head = nn.Linear(channels, n_classes)
            self.rf = receptive_field(kernel_size, n_levels)

        def forward(self, x):                        # x: (B, T, F)
            h = self.tcn(x.transpose(1, 2))          # (B, C, T)
            return self.head(h[:, :, -1])            # 마지막 시점만 사용


def main() -> None:
    print("[팽창 합성곱 수용영역 — kernel=3, 블록당 합성곱 2개, base=2]")
    print(f"{'블록수':>6s}{'최대 팽창':>10s}{'수용영역(step)':>15s}{'48ch 파라미터':>14s}")
    for n in range(1, 9):
        rf = receptive_field(3, n)
        print(f"{n:6d}{2 ** (n - 1):10d}{rf:15d}{param_estimate(n, 48, 3, 14):14d}")

    print("\n[윈도우 길이별 필요한 블록 수 — kernel=3]")
    print(f"{'윈도우':>8s}{'필요 블록':>10s}{'실제 수용영역':>14s}")
    for w in (12, 24, 48, 96, 168, 336):
        n = levels_for_window(w)
        print(f"{w:8d}{n:10d}{receptive_field(3, n):14d}")

    print("\n[커널 크기 비교 — 윈도우 96 step 을 덮으려면]")
    print(f"{'커널':>5s}{'필요 블록':>10s}{'수용영역':>10s}{'파라미터':>12s}")
    for k in (2, 3, 5, 7):
        n = levels_for_window(96, kernel_size=k)
        print(f"{k:5d}{n:10d}{receptive_field(k, n):10d}"
              f"{param_estimate(n, 48, k, 14):12d}")

    if HAS_TORCH:
        model = TCNClassifier(n_features=14, n_classes=2, n_levels=4)
        n_param = sum(p.numel() for p in model.parameters())
        x = torch.randn(8, 48, 14)
        print(f"\n[torch] 출력 shape={tuple(model(x).shape)} "
              f"파라미터={n_param} 수용영역={model.rf}")
    else:
        print("\n[안내] PyTorch 가 설치되어 있지 않아 모델 정의는 건너뛴다.")
        print("       pip install torch --index-url https://download.pytorch.org/whl/cpu")
        print("       수용영역 계산 결과는 위 표로 확인할 수 있다.")


if __name__ == "__main__":
    main()
