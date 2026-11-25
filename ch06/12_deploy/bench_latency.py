# -*- coding: utf-8 -*-
"""CPU 추론 지연시간 벤치마크 — 워밍업 · 반복 측정 · p50/p95/p99.

측정에서 틀리기 쉬운 네 가지를 모두 처리한다.
  1) 워밍업: 첫 호출에는 메모리 할당·스레드풀 생성·커널 선택이 섞여 있다.
     이것을 평균에 넣으면 지연시간이 실제보다 나쁘게 나온다.
  2) 평균 대신 분위수: 게이트 차단기는 평균이 아니라 최악에 맞춰 설계한다.
     p95·p99 를 보지 않으면 운영에서 "가끔 멈추는" 문제를 놓친다.
  3) 반복 수: 100회 미만이면 p99 가 사실상 단일 표본이라 의미가 없다.
  4) 타이머: ``time.perf_counter`` 를 쓴다(``time.time`` 은 시스템 시각 보정에
     영향을 받는다).

ONNX 모델이 있으면 실제 추론을, 없으면 번호판 파이프라인과 비슷한 연산량의
NumPy 워크로드를 측정한다. 어느 쪽이든 측정 방법론은 동일하다.

사용법
    python 12_deploy/bench_latency.py
    python 12_deploy/bench_latency.py --model models/crnn_ctc_kr.onnx --runs 300
    python 12_deploy/bench_latency.py --threads 1 --runs 200
"""

from __future__ import annotations

import argparse
import math
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

try:
    import numpy as np
except ImportError:  # NumPy 는 이 교재의 모든 장에서 쓰므로 사실상 필수다.
    print("[오류] numpy 가 필요하다:  pip install numpy")
    sys.exit(1)


@dataclass
class BenchResult:
    """한 구성(configuration)의 측정 결과."""

    label: str
    samples: list[float]      # 1회 호출 소요시간(ms) 목록
    warmup: int
    rss_before_mb: float
    rss_after_mb: float

    def pct(self, q: float) -> float:
        """분위수(ms). 선형 보간 없이 '넘지 않는 가장 큰 관측값' 을 쓴다.

        운영 기준을 세울 때는 보간으로 만들어낸 값보다 실제 관측값이 낫다.
        """
        xs = sorted(self.samples)
        idx = min(len(xs) - 1, max(0, math.ceil(q * len(xs)) - 1))
        return xs[idx]

    @property
    def mean(self) -> float:
        """평균(ms)."""
        return statistics.fmean(self.samples)

    @property
    def stdev(self) -> float:
        """표준편차(ms)."""
        return statistics.pstdev(self.samples) if len(self.samples) > 1 else 0.0

    @property
    def throughput(self) -> float:
        """단일 스레드 직렬 처리량(건/초) = 1000 / p50."""
        p50 = self.pct(0.50)
        return 1000.0 / p50 if p50 > 0 else 0.0

    @property
    def mem_delta(self) -> float:
        """측정 전후 RSS 증가량(MB). 누수를 잡는 1차 지표다."""
        return self.rss_after_mb - self.rss_before_mb


def rss_mb() -> float:
    """현재 프로세스의 RSS(MB)를 반환한다.

    psutil 이 있으면 그것을, 없으면 리눅스 ``/proc/self/statm`` 을 읽는다.
    둘 다 없으면 0 을 반환하되 측정은 계속한다(메모리 수치만 비게 된다).
    """
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:
        pass
    try:
        with open("/proc/self/statm", encoding="ascii") as fh:
            pages = int(fh.read().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)
    except Exception:
        return 0.0


def apply_threads(n: int) -> None:
    """스레드 수를 환경변수와 torch 양쪽에 설정한다.

    중요: OpenMP 계열 환경변수는 **라이브러리를 import 하기 전에** 설정해야
    효과가 있다. 이미 import 된 뒤 바꾸면 무시되는 경우가 많다. 그래서 이
    함수는 스크립트 맨 앞에서 호출한다.

    Args:
        n: 스레드 수(1 이상).
    """
    val = str(max(1, n))
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = val
    try:
        import torch
        torch.set_num_threads(max(1, n))
        torch.set_num_interop_threads(1)
    except Exception:
        pass


def make_numpy_workload(threads: int):
    """번호판 파이프라인과 비슷한 연산량의 NumPy 워크로드를 만든다.

    구성 근거:
        - 640x640 그레이스케일에 5x5 컨볼루션 3단(채널 8/16) → 검출기 초기 레이어 모사
        - 160x32 crop 특징에 대한 행렬곱 4회 → CRNN 의 LSTM/FC 모사
      실제 모델과 절대값이 같지는 않지만, 스레드 수·반복 횟수에 따른
      **변화 경향**은 같은 방향으로 나타난다.

    Args:
        threads: 사용할 스레드 수(로그 출력용).

    Returns:
        인수 없이 호출하면 1회 추론을 수행하는 함수.
    """
    rng = np.random.default_rng(20261024)
    img = rng.random((640, 640), dtype=np.float32)
    kernels = [rng.random((5, 5), dtype=np.float32) for _ in range(8)]
    feat = rng.random((40, 768), dtype=np.float32)
    w1 = rng.random((768, 768), dtype=np.float32)
    w2 = rng.random((768, 512), dtype=np.float32)
    w3 = rng.random((512, 256), dtype=np.float32)
    w4 = rng.random((256, 86), dtype=np.float32)   # 86 = 번호판 문자 클래스 수 예시

    def conv2d(x: np.ndarray, k: np.ndarray) -> np.ndarray:
        """stride 2 유효 영역 컨볼루션(순수 NumPy)."""
        kh, kw = k.shape
        h = (x.shape[0] - kh) // 2 + 1
        w = (x.shape[1] - kw) // 2 + 1
        out = np.zeros((h, w), dtype=np.float32)
        for i in range(kh):
            for j in range(kw):
                out += k[i, j] * x[i:i + 2 * h:2, j:j + 2 * w:2]
        return out

    def step() -> float:
        # 검출기 백본 모사: stride 2 로 해상도를 줄여 가며 다중 채널을 쌓는다.
        stage1 = [conv2d(img, k) for k in kernels[:4]]
        stage2 = [np.maximum(conv2d(x, k), 0.0)
                  for x in stage1 for k in kernels[4:6]]
        stage3 = np.maximum(conv2d(sum(stage2), kernels[6]), 0.0)
        # OCR 헤드 모사
        h1 = np.maximum(feat @ w1, 0.0)      # ReLU
        h2 = np.maximum(h1 @ w2, 0.0)
        h3 = np.maximum(h2 @ w3, 0.0)
        logits = h3 @ w4
        return float(stage3.mean() + logits.max())

    print(f"  워크로드: NumPy 모사 파이프라인 (스레드 {threads})")
    return step


def make_onnx_workload(model: Path, threads: int):
    """ONNX 모델로 실제 추론 워크로드를 만든다.

    세션 옵션에서 ``intra_op_num_threads`` 를 명시한다. 이 값을 지정하지
    않으면 onnxruntime 이 코어 수대로 스레드를 만들고, FastAPI 워커가
    여러 개일 때 코어 경합이 일어나 p95 가 크게 흔들린다.

    Args:
        model: ONNX 파일 경로.
        threads: intra-op 스레드 수.

    Returns:
        (1회 추론 함수) 또는 로드 실패 시 None.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print("  [안내] onnxruntime 미설치 → NumPy 워크로드로 대체한다.")
        return None
    if not model.is_file():
        print(f"  [안내] 모델이 없다({model}) → NumPy 워크로드로 대체한다.")
        return None
    try:
        so = ort.SessionOptions()
        so.intra_op_num_threads = max(1, threads)
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess = ort.InferenceSession(str(model), so,
                                    providers=["CPUExecutionProvider"])
        inp = sess.get_inputs()[0]
        shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
        rng = np.random.default_rng(7)
        data = rng.random(shape).astype(np.float32)
        name = inp.name

        def step():
            return sess.run(None, {name: data})

        print(f"  워크로드: ONNX {model.name} 입력 {shape} (스레드 {threads})")
        return step
    except Exception as exc:  # noqa: BLE001
        print(f"  [안내] ONNX 로드 실패({type(exc).__name__}) → NumPy 대체")
        return None


def bench(step, label: str, runs: int, warmup: int) -> BenchResult:
    """워밍업 후 runs 회를 측정한다.

    Args:
        step: 1회 추론 함수.
        label: 구성 이름.
        runs: 측정 반복 수.
        warmup: 버리는 초기 호출 수.

    Returns:
        BenchResult.
    """
    for _ in range(warmup):
        step()
    before = rss_mb()
    samples: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        step()
        samples.append((time.perf_counter() - t0) * 1000.0)
    after = rss_mb()
    return BenchResult(label=label, samples=samples, warmup=warmup,
                       rss_before_mb=before, rss_after_mb=after)


def print_result(r: BenchResult) -> None:
    """측정 결과 1건을 출력한다."""
    print(f"\n[{r.label}]")
    print(f"  워밍업 {r.warmup}회 후 {len(r.samples)}회 측정")
    print(f"  평균 {r.mean:8.2f} ms   표준편차 {r.stdev:6.2f} ms")
    print(f"  최소 {min(r.samples):8.2f} ms   최대     {max(r.samples):6.2f} ms")
    print(f"  p50  {r.pct(0.50):8.2f} ms")
    print(f"  p95  {r.pct(0.95):8.2f} ms")
    print(f"  p99  {r.pct(0.99):8.2f} ms")
    print(f"  직렬 처리량 {r.throughput:6.1f} 건/초")
    print(f"  RSS {r.rss_before_mb:.1f} → {r.rss_after_mb:.1f} MB "
          f"(증가 {r.mem_delta:+.1f} MB)")


def print_warmup_effect(step, n: int = 6) -> None:
    """초기 호출 n회의 소요시간을 그대로 보여준다.

    "워밍업이 왜 필요한가" 를 숫자로 보이는 것이 목적이다. 보통 1회차가
    2회차 이후보다 수 배 느리다.
    """
    print("\n[워밍업 효과 — 초기 호출 소요시간]")
    first: list[float] = []
    for i in range(n):
        t0 = time.perf_counter()
        step()
        ms = (time.perf_counter() - t0) * 1000.0
        first.append(ms)
        print(f"  {i + 1}회차 {ms:8.2f} ms")
    if len(first) > 1:
        rest = statistics.fmean(first[1:])
        print(f"  1회차 / 이후 평균 = {first[0] / rest:.2f}배"
              if rest > 0 else "")


def main(argv: list[str] | None = None) -> int:
    """명령행 진입점."""
    ap = argparse.ArgumentParser(description="CPU 추론 지연시간 벤치마크")
    ap.add_argument("--model", type=Path, default=None, help="ONNX 모델 경로")
    ap.add_argument("--runs", type=int, default=120, help="측정 반복 수")
    ap.add_argument("--warmup", type=int, default=10, help="워밍업 횟수")
    ap.add_argument("--threads", type=int, default=0,
                    help="스레드 수(0 이면 CPU 코어 수)")
    args = ap.parse_args(argv)

    threads = args.threads if args.threads > 0 else (os.cpu_count() or 1)
    apply_threads(threads)

    print("=" * 66)
    print("CPU 추론 지연시간 벤치마크")
    print("=" * 66)
    print(f"  CPU 코어 {os.cpu_count()} / 설정 스레드 {threads}")
    print(f"  OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}")

    step = make_onnx_workload(args.model, threads) if args.model else None
    if step is None:
        step = make_numpy_workload(threads)

    print_warmup_effect(step)
    r = bench(step, f"threads={threads}", args.runs, args.warmup)
    print_result(r)

    print("\n[해석 지침]")
    print("  - p95/p50 비율이 1.5 를 넘으면 지연 변동이 크다는 뜻이다.")
    print(f"    현재 비율 {r.pct(0.95) / r.pct(0.50):.2f}")
    print("  - RSS 증가가 반복 수에 비례해 커지면 세션·버퍼 누수를 의심한다.")
    print("  - 차단기 제어는 p99 를 기준으로 예산을 잡는다. 평균은 쓰지 않는다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
