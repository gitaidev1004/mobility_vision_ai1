# -*- coding: utf-8 -*-
"""비동기 추론 — 이벤트 루프를 막지 않고 CPU 추론을 돌리는 구조.

문제
    FastAPI 의 ``async def`` 핸들러 안에서 추론 함수를 그냥 호출하면,
    추론이 끝날 때까지 이벤트 루프 전체가 멈춘다. 동시 요청 10건이 오면
    10번째 요청은 앞의 9건이 끝날 때까지 **대기 큐에도 들어가지 못한다**.
    헬스체크(`/health`)조차 응답하지 못해 오케스트레이터가 컨테이너를
    죽이는 사고로 이어진다.

해결
    추론은 ``ThreadPoolExecutor`` 로 넘기고, 핸들러는 ``await`` 로 양보한다.
    GIL 때문에 파이썬 스레드로는 CPU 병렬화가 안 된다고들 하지만,
    NumPy·onnxruntime·torch 의 커널은 **GIL 을 놓고** C 코드에서 돌기 때문에
    실제로 병렬 실행된다. 즉 이 구조에서 스레드풀은 유효하다.

추가 장치
    1) 세마포어(동시 추론 상한) — 무제한으로 받으면 코어 경합으로 전부 느려진다.
       받아들일 수 있는 수만 넣고 나머지는 대기시키는 쪽이 p95 가 좋다.
    2) 타임아웃 — 백엔드의 RestClient 타임아웃보다 짧게 잡아야, 백엔드가
       끊어진 뒤에도 서버가 계산을 계속하는 낭비를 막는다.
    3) 큐 깊이 관측 — 대기 건수를 노출해 과부하를 조기에 감지한다.

사용법
    python 12_deploy/async_infer.py              # 직렬 vs 비동기 비교
    python 12_deploy/async_infer.py --requests 24 --limit 2
"""

from __future__ import annotations

import argparse
import asyncio
import math
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

# 중요: 수치 라이브러리 import 보다 먼저 스레드 수를 1로 고정한다.
# 커널 내부 병렬화를 끄지 않으면 "스레드풀 덕분에 빨라진 것"과
# "BLAS 가 알아서 병렬화한 것"을 구분할 수 없다.
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_key, "1")

try:
    import numpy as np
except ImportError:
    print("[오류] numpy 가 필요하다:  pip install numpy")
    sys.exit(1)


# ---------------------------------------------------------------------------
# 추론 워크로드 (동기 함수) — 실제로는 onnxruntime 세션 호출이 여기 들어간다.
# ---------------------------------------------------------------------------

_RNG = np.random.default_rng(20261024)
_IMG = _RNG.random((640, 640), dtype=np.float32)
_K = [_RNG.random((5, 5), dtype=np.float32) for _ in range(6)]
_W1 = _RNG.random((768, 768), dtype=np.float32)
_W2 = _RNG.random((768, 512), dtype=np.float32)
_W3 = _RNG.random((512, 86), dtype=np.float32)
_FEAT = _RNG.random((40, 768), dtype=np.float32)


def _conv(x: np.ndarray, k: np.ndarray) -> np.ndarray:
    """stride 2 컨볼루션(NumPy 커널 → GIL 을 놓는 구간)."""
    kh, kw = k.shape
    h = (x.shape[0] - kh) // 2 + 1
    w = (x.shape[1] - kw) // 2 + 1
    out = np.zeros((h, w), dtype=np.float32)
    for i in range(kh):
        for j in range(kw):
            out += k[i, j] * x[i:i + 2 * h:2, j:j + 2 * w:2]
    return out


def infer_sync(req_id: int) -> dict[str, object]:
    """동기 추론 1회. 실제 서비스에서는 검출 + crop + OCR 이 들어간다.

    Args:
        req_id: 요청 식별자(로그/추적용).

    Returns:
        인식 결과를 모사한 dict.
    """
    t0 = time.perf_counter()
    # 검출기 백본 모사: 같은 피라미드를 두 스케일로 돌려 연산량을 실제에 맞춘다.
    acc = 0.0
    for scale in (1, 2):
        f = _IMG[::scale, ::scale]
        for k in _K:
            if min(f.shape) < 6:
                break
            f = np.maximum(_conv(f, k), 0.0)
        acc += float(f.mean())
    # OCR 헤드 모사
    h1 = np.maximum(_FEAT @ _W1, 0.0)
    h2 = np.maximum(h1 @ _W2, 0.0)
    logits = h2 @ _W3
    elapsed = (time.perf_counter() - t0) * 1000.0
    return {
        "req_id": req_id,
        "plate_number": "12가3456",
        "confidence": float(0.90 + 0.09 * (logits.max() % 1.0)),
        "elapsed_ms": round(elapsed, 2),
    }


# ---------------------------------------------------------------------------
# 비동기 러너
# ---------------------------------------------------------------------------


@dataclass
class RunStats:
    """한 실행 방식의 측정 결과."""

    label: str
    wall_ms: float          # 전체 소요 시간(벽시계)
    latencies: list[float]  # 요청별 체감 지연(ms)
    errors: int = 0
    peak_queue: int = 0     # 최대 대기 큐 깊이

    def pct(self, q: float) -> float:
        """체감 지연 분위수(ms)."""
        if not self.latencies:
            return 0.0
        xs = sorted(self.latencies)
        idx = min(len(xs) - 1, max(0, math.ceil(q * len(xs)) - 1))
        return xs[idx]

    @property
    def throughput(self) -> float:
        """전체 처리량(건/초)."""
        return len(self.latencies) / (self.wall_ms / 1000.0) if self.wall_ms else 0.0


class InferenceRunner:
    """스레드풀 + 세마포어로 추론을 비동기 실행한다.

    Attributes:
        pool: 추론을 실행하는 스레드풀.
        sem: 동시 추론 상한 세마포어.
        waiting: 현재 대기 중인 요청 수(관측용).
    """

    def __init__(self, workers: int, limit: int, timeout_s: float = 3.0) -> None:
        """러너를 만든다.

        Args:
            workers: 스레드풀 크기. 보통 CPU 코어 수 이하로 둔다.
            limit: 동시에 추론을 수행할 최대 건수.
            timeout_s: 1건당 타임아웃(초).
        """
        self.pool = ThreadPoolExecutor(max_workers=workers,
                                       thread_name_prefix="infer")
        self.sem = asyncio.Semaphore(limit)
        self.timeout_s = timeout_s
        self.waiting = 0
        self.peak_waiting = 0

    async def infer(self, req_id: int) -> dict[str, object]:
        """비동기 추론 1건.

        세마포어로 동시 실행을 제한한 뒤, 실제 계산은 스레드풀에 넘기고
        ``await`` 로 이벤트 루프를 양보한다. 타임아웃을 넘기면 폴백 결과를
        돌려준다 — 계약서대로 500 을 던지지 않는다.

        Args:
            req_id: 요청 식별자.

        Returns:
            인식 결과 dict(타임아웃 시 status=TIMEOUT).
        """
        self.waiting += 1
        self.peak_waiting = max(self.peak_waiting, self.waiting)
        async with self.sem:
            self.waiting -= 1
            loop = asyncio.get_running_loop()
            try:
                return await asyncio.wait_for(
                    loop.run_in_executor(self.pool, infer_sync, req_id),
                    timeout=self.timeout_s)
            except asyncio.TimeoutError:
                return {"req_id": req_id, "plate_number": None,
                        "status": "TIMEOUT", "elapsed_ms": self.timeout_s * 1000}

    def shutdown(self) -> None:
        """스레드풀을 정리한다."""
        self.pool.shutdown(wait=True)


async def run_blocking(n: int) -> RunStats:
    """잘못된 구조 — async 핸들러 안에서 동기 추론을 직접 호출한다.

    이벤트 루프가 추론 중 멈추므로 요청들은 사실상 직렬 처리된다.
    뒤에 온 요청의 체감 지연이 앞 요청들의 합만큼 누적된다.
    """
    t0 = time.perf_counter()
    lat: list[float] = []
    arrive = time.perf_counter()
    for i in range(n):
        infer_sync(i)
        lat.append((time.perf_counter() - arrive) * 1000.0)
    return RunStats("동기 직접 호출(루프 차단)",
                    (time.perf_counter() - t0) * 1000.0, lat)


async def run_async(n: int, workers: int, limit: int) -> RunStats:
    """올바른 구조 — 스레드풀 + 세마포어로 비동기 실행한다."""
    runner = InferenceRunner(workers=workers, limit=limit)
    t0 = time.perf_counter()

    async def one(i: int) -> float:
        a = time.perf_counter()
        await runner.infer(i)
        return (time.perf_counter() - a) * 1000.0

    try:
        lat = list(await asyncio.gather(*(one(i) for i in range(n))))
    finally:
        runner.shutdown()
    return RunStats(f"비동기(workers={workers}, limit={limit})",
                    (time.perf_counter() - t0) * 1000.0, lat,
                    peak_queue=runner.peak_waiting)


async def heartbeat_probe(n: int, workers: int, limit: int) -> None:
    """추론 중에도 헬스체크가 응답하는지 확인한다.

    50ms 주기로 헬스체크를 흉내 내며, 추론이 진행되는 동안 몇 번
    응답했는지 센다. 루프가 막히면 이 수가 0 에 가까워진다.
    """
    runner = InferenceRunner(workers=workers, limit=limit)
    beats = 0
    stop = False

    async def health() -> None:
        nonlocal beats
        while not stop:
            beats += 1
            await asyncio.sleep(0.05)

    task = asyncio.create_task(health())
    t0 = time.perf_counter()
    try:
        await asyncio.gather(*(runner.infer(i) for i in range(n)))
    finally:
        stop = True
        task.cancel()
        runner.shutdown()
    elapsed = (time.perf_counter() - t0) * 1000.0
    print(f"\n[헬스체크 응답성] 추론 {n}건 {elapsed:.0f}ms 동안 "
          f"헬스체크 {beats}회 응답 "
          f"(50ms 주기 기대값 약 {max(1, int(elapsed / 50))}회)")
    print("  동기 직접 호출 구조에서는 추론이 끝날 때까지 한 번도 응답하지 못한다.")


def print_stats(s: RunStats) -> None:
    """측정 결과를 출력한다."""
    print(f"\n[{s.label}]")
    print(f"  전체 소요   {s.wall_ms:9.1f} ms")
    print(f"  처리량      {s.throughput:9.1f} 건/초")
    print(f"  체감 p50    {s.pct(0.50):9.1f} ms")
    print(f"  체감 p95    {s.pct(0.95):9.1f} ms")
    print(f"  체감 최대   {max(s.latencies) if s.latencies else 0:9.1f} ms")
    print(f"  체감 평균   {statistics.fmean(s.latencies) if s.latencies else 0:9.1f} ms")
    if s.peak_queue:
        print(f"  최대 대기 큐 {s.peak_queue:8d} 건")


async def amain(requests: int, limit: int, workers: int) -> None:
    """비교 실험 본체."""
    print("=" * 70)
    print("비동기 추론 구조 비교")
    print("=" * 70)
    print(f"  CPU 코어 {os.cpu_count()} / 요청 {requests}건 / "
          f"스레드풀 {workers} / 동시 상한 {limit}")

    blocking = await run_blocking(requests)
    print_stats(blocking)

    asyncr = await run_async(requests, workers, limit)
    print_stats(asyncr)

    print("\n[비교]")
    print(f"  {'지표':<14}{'동기':>12}{'비동기':>12}{'개선':>12}")
    rows = [
        ("전체 소요(ms)", blocking.wall_ms, asyncr.wall_ms),
        ("처리량(건/s)", blocking.throughput, asyncr.throughput),
        ("체감 p95(ms)", blocking.pct(0.95), asyncr.pct(0.95)),
        ("체감 최대(ms)", max(blocking.latencies), max(asyncr.latencies)),
    ]
    for name, a, b in rows:
        print(f"  {name:<14}{a:>12.1f}{b:>12.1f}{b - a:>+12.1f}")

    print("\n[이 수치를 어떻게 읽어야 하는가]")
    print("  추론은 CPU 포화 작업이므로 '총 처리량' 은 코어 수로 결정된다.")
    print(f"  코어 {os.cpu_count()}개 환경에서 비동기화만으로 처리량이 크게 늘지는 않고,")
    print("  실행마다 ±10% 정도 흔들린다. 비동기의 실익은 처리량이 아니라")
    print("  (가) 헬스체크·업로드 같은 다른 요청이 계속 처리된다는 점,")
    print("  (나) 타임아웃·동시 상한으로 과부하를 제어할 수 있다는 점이다.")
    print("  코어가 8개 이상인 서버에서는 처리량 차이도 함께 커진다.")

    await heartbeat_probe(requests, workers, limit)

    print("\n[FastAPI 적용 예]")
    print("  @app.post('/api/v1/lpr/recognize')")
    print("  async def recognize(file: UploadFile = File(...)):")
    print("      img = await file.read()                       # I/O 는 await")
    print("      return await runner.infer_image(img)          # 추론은 스레드풀")
    print("\n[주의]")
    print("  - 스레드풀 크기 × 모델 스레드 수가 코어 수를 넘지 않게 맞춘다.")
    print("  - 타임아웃은 백엔드 RestClient 읽기 타임아웃보다 짧게 둔다.")
    print("  - onnxruntime 세션은 스레드 안전하므로 하나를 공유해도 된다.")


def main(argv: list[str] | None = None) -> int:
    """명령행 진입점."""
    ap = argparse.ArgumentParser(description="비동기 추론 구조 비교")
    ap.add_argument("--requests", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0, help="0 이면 코어 수")
    ap.add_argument("--workers", type=int, default=0, help="0 이면 코어 수")
    args = ap.parse_args(argv)
    cores = os.cpu_count() or 1
    asyncio.run(amain(args.requests,
                      args.limit or cores,
                      args.workers or cores))
    return 0


if __name__ == "__main__":
    sys.exit(main())
