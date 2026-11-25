# -*- coding: utf-8 -*-
"""CPU 스레드 수에 따른 추론 성능 변화 측정.

PyTorch 가 있으면 작은 합성곱 적층으로, 없으면 numpy 행렬 곱으로 측정한다.
스레드 수를 1 부터 늘려 가며 중앙값과 1 스레드 대비 가속비를 출력한다.
실행:  python 02_cpu/thread_tuning.py
      python 02_cpu/thread_tuning.py --max-threads 4 --repeat 20
"""
from __future__ import annotations

import argparse
import os
import statistics
import time

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

import numpy as np

WARMUP = 3


def build_conv_stack():
    """검출 백본의 앞단을 모사한 작은 합성곱 적층을 만든다."""
    return nn.Sequential(
        nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.ReLU(),
        nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
        nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
        nn.Conv2d(64, 64, 3, stride=1, padding=1), nn.ReLU(),
    ).eval()


def measure_torch(threads: int, repeat: int, size: int) -> list[float]:
    """지정한 스레드 수로 합성곱 적층 추론을 측정한다."""
    torch.set_num_threads(threads)
    torch.manual_seed(0)
    model = build_conv_stack()
    x = torch.randn(1, 3, size, size)
    out: list[float] = []
    with torch.inference_mode():
        for _ in range(WARMUP):
            _ = model(x)
        for _ in range(repeat):
            t0 = time.perf_counter()
            _ = model(x)
            out.append((time.perf_counter() - t0) * 1000.0)
    return out


def measure_numpy(repeat: int, n: int = 512) -> list[float]:
    """PyTorch 가 없을 때의 대체 측정. 스레드 제어는 BLAS 가 담당한다."""
    rng = np.random.default_rng(0)
    a = rng.standard_normal((n, n), dtype=np.float32)
    b = rng.standard_normal((n, n), dtype=np.float32)
    for _ in range(WARMUP):
        _ = a @ b
    out: list[float] = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        _ = a @ b
        out.append((time.perf_counter() - t0) * 1000.0)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="스레드 수별 추론 성능 측정")
    ap.add_argument("--max-threads", type=int, default=0,
                    help="측정할 최대 스레드 수. 0 이면 코어 수를 사용")
    ap.add_argument("--repeat", type=int, default=20, help="설정별 반복 횟수")
    ap.add_argument("--size", type=int, default=320, help="입력 이미지 한 변")
    args = ap.parse_args()

    cores = os.cpu_count() or 1
    top = args.max_threads if args.max_threads > 0 else cores
    top = max(1, min(top, cores))

    if not HAS_TORCH:
        print("PyTorch 가 없어 numpy 행렬 곱으로 기준값만 측정한다.")
        s = measure_numpy(args.repeat)
        print(f"중앙값 {statistics.median(s):.2f} ms")
        print("스레드 수 비교는 PyTorch 설치 후 다시 실행한다.")
        return 0

    print(f"코어 수 {cores} / 입력 {args.size}x{args.size} / "
          f"설정별 {args.repeat}회")
    print("-" * 52)
    print(f"{'스레드':>6s} {'중앙값(ms)':>12s} {'가속비':>8s} {'효율':>8s}")

    base: float | None = None
    rows: list[tuple[int, float, float]] = []
    for t in range(1, top + 1):
        samples = measure_torch(t, args.repeat, args.size)
        med = statistics.median(samples)
        if base is None:
            base = med
        speedup = base / med if med > 0 else 0.0
        eff = speedup / t                    # 스레드당 효율. 1.0 이면 이상적
        rows.append((t, med, speedup))
        print(f"{t:>6d} {med:>12.2f} {speedup:>8.2f} {eff:>8.2f}")

    print("-" * 52)
    best = min(rows, key=lambda r: r[1])
    print(f"가장 빠른 설정: {best[0]} 스레드 ({best[1]:.2f} ms)")
    print("효율이 0.5 아래로 떨어지면 스레드를 더 늘릴 이득이 작다.")
    print("같은 장비에서 다른 서비스가 함께 동작하면 더 낮게 설정한다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
