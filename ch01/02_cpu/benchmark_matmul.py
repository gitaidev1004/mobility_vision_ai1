# -*- coding: utf-8 -*-
"""CPU 행렬 곱 성능 측정.

PyTorch 가 있으면 torch 로, 없으면 numpy 로 같은 절차를 수행한다.
예열 후 여러 번 반복해 중앙값과 95백분위를 보고한다.
실행:  python 02_cpu/benchmark_matmul.py
      python 02_cpu/benchmark_matmul.py --sizes 256,512 --repeat 20
"""
from __future__ import annotations

import argparse
import statistics
import time

# PyTorch 가 없는 환경에서도 동작하도록 선택적으로 가져온다.
try:
    import torch
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

import numpy as np

WARMUP = 3        # 예열 횟수. 결과에서 제외한다.
DEFAULT_SIZES = (256, 512, 1024)
DEFAULT_REPEAT = 15


def percentile(values: list[float], q: float) -> float:
    """정렬된 표본에서 q 백분위를 선형 보간 없이 근사한다."""
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, max(0, int(round(q / 100.0 * (len(s) - 1)))))
    return s[idx]


def bench_numpy(n: int, repeat: int) -> list[float]:
    """numpy 로 n x n 행렬 곱을 repeat 회 측정해 밀리초 목록을 돌려준다."""
    rng = np.random.default_rng(42)          # 시드 고정으로 재현성 확보
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


def bench_torch(n: int, repeat: int) -> list[float]:
    """torch 로 같은 측정을 수행한다. 장치는 CPU 로 고정한다."""
    torch.manual_seed(42)
    a = torch.randn(n, n, dtype=torch.float32, device="cpu")
    b = torch.randn(n, n, dtype=torch.float32, device="cpu")
    with torch.inference_mode():             # 그래프 추적을 끄고 측정한다
        for _ in range(WARMUP):
            _ = a @ b
        out: list[float] = []
        for _ in range(repeat):
            t0 = time.perf_counter()
            _ = a @ b
            out.append((time.perf_counter() - t0) * 1000.0)
    return out


def gflops(n: int, ms: float) -> float:
    """n x n 행렬 곱의 연산량 2*n^3 을 소요 시간으로 나눈 값(GFLOPS)."""
    if ms <= 0:
        return 0.0
    return (2.0 * n ** 3) / (ms / 1000.0) / 1e9


def main() -> int:
    ap = argparse.ArgumentParser(description="CPU 행렬 곱 성능 측정")
    ap.add_argument("--sizes", default=",".join(str(s) for s in DEFAULT_SIZES),
                    help="쉼표로 구분한 행렬 크기 목록")
    ap.add_argument("--repeat", type=int, default=DEFAULT_REPEAT,
                    help="크기별 반복 횟수")
    ap.add_argument("--backend", choices=("auto", "numpy", "torch"),
                    default="auto", help="측정 백엔드 선택")
    args = ap.parse_args()

    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    use_torch = HAS_TORCH if args.backend == "auto" else (args.backend == "torch")
    if use_torch and not HAS_TORCH:
        print("PyTorch 가 설치되지 않았다. numpy 로 측정한다.")
        use_torch = False

    backend = "torch" if use_torch else "numpy"
    print(f"백엔드: {backend}")
    if use_torch:
        print(f"torch {torch.__version__} / 스레드 {torch.get_num_threads()}")
    print(f"예열 {WARMUP}회 제외, 크기별 {args.repeat}회 반복")
    print("-" * 58)
    print(f"{'크기':>6s} {'중앙값(ms)':>12s} {'p95(ms)':>10s} {'GFLOPS':>10s}")

    for n in sizes:
        samples = bench_torch(n, args.repeat) if use_torch \
            else bench_numpy(n, args.repeat)
        med = statistics.median(samples)
        p95 = percentile(samples, 95)
        print(f"{n:>6d} {med:>12.2f} {p95:>10.2f} {gflops(n, med):>10.1f}")

    print("-" * 58)
    print("같은 장비에서 반복 측정해 값이 안정되는지 먼저 확인한다.")
    print("장비가 다르면 절대값은 달라지며, 경향 비교에만 사용한다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
