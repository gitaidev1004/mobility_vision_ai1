# -*- coding: utf-8 -*-
"""Depthwise Separable Convolution 의 연산량을 직접 계산해 비교한다.

표준 합성곱 한 층을 Depthwise + Pointwise 두 단계로 쪼개면
연산량과 파라미터 수가 크게 줄어든다. MobileNet 의 핵심 아이디어다.

계산식 (입력 H x W x Cin, 커널 K x K, 출력 채널 Cout, stride 1, same padding)
  표준 합성곱
    파라미터 = K*K*Cin*Cout            (+ bias Cout)
    곱셈연산 = K*K*Cin*Cout*H*W
  Depthwise Separable
    Depthwise  파라미터 = K*K*Cin        곱셈연산 = K*K*Cin*H*W
    Pointwise  파라미터 = Cin*Cout       곱셈연산 = Cin*Cout*H*W
    합계 파라미터 = K*K*Cin + Cin*Cout
    합계 곱셈연산 = (K*K + Cout) * Cin * H * W
  절감 비율
    1/Cout + 1/(K*K)
  3x3 커널에서 Cout 이 충분히 크면 약 1/9 수준, 즉 8~9배 절감이다.

표준 라이브러리만 사용한다. PyTorch 가 있으면 실제 모듈의 파라미터 수와
비교해 수식이 맞는지 검증한다.
"""
from __future__ import annotations


def standard_conv_cost(h: int, w: int, cin: int, cout: int, k: int):
    """표준 합성곱의 (파라미터, MACs) 를 돌려준다. MAC = 곱셈-누산 1회."""
    params = k * k * cin * cout
    macs = params * h * w
    return params, macs


def dw_sep_conv_cost(h: int, w: int, cin: int, cout: int, k: int):
    """Depthwise Separable 합성곱의 (파라미터, MACs)."""
    dw_params = k * k * cin                  # 채널마다 독립 커널 1개
    dw_macs = dw_params * h * w
    pw_params = cin * cout                   # 1x1 로 채널을 섞는다
    pw_macs = pw_params * h * w
    return dw_params + pw_params, dw_macs + pw_macs


def theoretical_ratio(cout: int, k: int) -> float:
    """이론적 절감 비율 1/Cout + 1/(K*K)."""
    return 1.0 / cout + 1.0 / (k * k)


def fmt(n: float) -> str:
    """큰 수를 K/M/G 단위로 읽기 쉽게 만든다."""
    for unit, div in (("G", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(n) >= div:
            return f"{n / div:8.2f}{unit}"
    return f"{n:9.0f} "


def verify_with_torch(h, w, cin, cout, k) -> None:
    """PyTorch 모듈을 실제로 만들어 파라미터 수를 대조한다."""
    try:
        import torch                           # noqa: PLC0415
        from torch import nn                   # noqa: PLC0415
    except ImportError:
        print("\n[안내] PyTorch 가 없어 실제 모듈 검증은 건너뛴다.")
        print("       pip install torch==2.8.0  (CPU 휠)")
        return

    std = nn.Conv2d(cin, cout, k, padding=k // 2, bias=False)
    dws = nn.Sequential(
        # groups=cin 이 Depthwise 를 만든다. 각 채널이 자기 커널만 본다
        nn.Conv2d(cin, cin, k, padding=k // 2, groups=cin, bias=False),
        nn.BatchNorm2d(cin), nn.ReLU6(inplace=True),
        # 1x1 Pointwise 로 채널 간 정보를 섞는다
        nn.Conv2d(cin, cout, 1, bias=False),
        nn.BatchNorm2d(cout), nn.ReLU6(inplace=True),
    )
    n_std = sum(p.numel() for p in std.parameters())
    n_dws = sum(p.numel() for p in dws.parameters())
    print("\n=== PyTorch 실제 모듈 검증 ===")
    print(f"  표준 Conv2d            파라미터 {n_std:,}")
    print(f"  Depthwise Separable    파라미터 {n_dws:,} (BatchNorm 포함)")
    print(f"  수식 예측(표준)        {standard_conv_cost(h, w, cin, cout, k)[0]:,}")
    print(f"  수식 예측(분리형)      {dw_sep_conv_cost(h, w, cin, cout, k)[0]:,}")

    x = torch.zeros(1, cin, h, w)
    with torch.no_grad():
        print(f"  출력 형태 표준={tuple(std(x).shape)} 분리형={tuple(dws(x).shape)}")


def main() -> None:
    # 번호판 검출기 백본에서 실제로 쓰이는 규모의 층들을 가정한다
    layers = [
        # (설명, H, W, Cin, Cout, K)
        ("초반 층 320x320", 320, 320, 32, 64, 3),
        ("중반 층 80x80", 80, 80, 128, 128, 3),
        ("중반 층 40x40", 40, 40, 256, 256, 3),
        ("후반 층 20x20", 20, 20, 512, 512, 3),
        ("5x5 커널 비교", 40, 40, 256, 256, 5),
    ]

    head = (f"{'층':18s}{'표준 파라미터':>14s}{'분리형 파라미터':>16s}"
            f"{'표준 MACs':>12s}{'분리형 MACs':>14s}{'절감배수':>10s}{'이론비':>9s}")
    print("=== 표준 합성곱 vs Depthwise Separable 합성곱 ===")
    print(head)
    print("-" * len(head))
    tot_s_p = tot_d_p = tot_s_m = tot_d_m = 0
    for name, h, w, cin, cout, k in layers:
        sp, sm = standard_conv_cost(h, w, cin, cout, k)
        dp, dm = dw_sep_conv_cost(h, w, cin, cout, k)
        tot_s_p += sp
        tot_d_p += dp
        tot_s_m += sm
        tot_d_m += dm
        print(f"{name:18s}{fmt(sp):>14s}{fmt(dp):>16s}{fmt(sm):>12s}"
              f"{fmt(dm):>14s}{sm / dm:10.2f}x{theoretical_ratio(cout, k):9.4f}")
    print("-" * len(head))
    print(f"{'합계':18s}{fmt(tot_s_p):>14s}{fmt(tot_d_p):>16s}"
          f"{fmt(tot_s_m):>12s}{fmt(tot_d_m):>14s}{tot_s_m / tot_d_m:10.2f}x")

    print("\n=== 해석 ===")
    print("  3x3 커널에서 절감 비율은 1/Cout + 1/9 이다.")
    print("  Cout 이 128 이상이면 1/Cout 은 무시할 만하므로 약 9배 절감에 수렴한다.")
    print("  5x5 커널이면 1/25 + 1/Cout 이 되어 절감 배수가 더 커진다.")
    print("  단, MACs 가 9배 줄어도 실제 CPU 추론시간이 9배 빨라지지는 않는다.")
    print("  Depthwise 연산은 채널별로 메모리 접근이 흩어져 캐시 효율이 낮고,")
    print("  연산 대비 메모리 대역폭 요구가 커서 실측 속도 향상은 보통 2~4배다.")
    print("  이 때문에 이론 연산량과 실측 지연시간을 반드시 함께 측정해야 한다.")

    verify_with_torch(40, 40, 256, 256, 3)


if __name__ == "__main__":
    main()
