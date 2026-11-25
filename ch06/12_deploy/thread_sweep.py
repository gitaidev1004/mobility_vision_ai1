# -*- coding: utf-8 -*-
"""CPU 스레드 수 스윕 — 몇 개가 최적인지 측정으로 찾는다.

"코어 수만큼 스레드를 주면 가장 빠르다" 는 추측은 번호판 인식 서비스에서
자주 틀린다. 이유는 세 가지다.

  1) 번호판 입력 1장은 배치가 아니라 작은 텐서다. 스레드를 늘려도 나눌 일이
     적고 동기화 비용만 커진다.
  2) FastAPI 워커가 여러 개면 워커마다 스레드풀을 만든다. 워커 4개 ×
     스레드 4개 = 16 스레드가 4코어를 다투면 p95 가 급격히 나빠진다.
  3) 하이퍼스레딩 논리 코어는 물리 코어처럼 선형으로 보태지지 않는다.

이 스크립트는 **자식 프로세스를 새로 띄워** 각 스레드 수를 측정한다.
OpenMP 환경변수는 라이브러리 import 이전에만 적용되므로, 한 프로세스
안에서 스레드 수를 바꿔 가며 재는 것은 신뢰할 수 없다.

사용법
    python 12_deploy/thread_sweep.py
    python 12_deploy/thread_sweep.py --max-threads 8 --runs 80
    python 12_deploy/thread_sweep.py --workers 1,2,4    # 워커 동시 실행 실험
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# 자식 프로세스 모드 — 환경변수가 이미 설정된 상태에서 import 가 일어난다.
# ---------------------------------------------------------------------------


def child_measure(runs: int, warmup: int) -> dict[str, float]:
    """자식 프로세스에서 실제 측정을 수행하고 결과를 dict 로 돌려준다.

    Args:
        runs: 측정 반복 수.
        warmup: 버리는 초기 호출 수.

    Returns:
        평균/p50/p95/p99/최소/최대/RSS 를 담은 dict.
    """
    import numpy as np

    rng = np.random.default_rng(20261024)
    img = rng.random((640, 640), dtype=np.float32)
    kernels = [rng.random((5, 5), dtype=np.float32) for _ in range(8)]
    feat = rng.random((40, 768), dtype=np.float32)
    w1 = rng.random((768, 768), dtype=np.float32)
    w2 = rng.random((768, 512), dtype=np.float32)
    w3 = rng.random((512, 256), dtype=np.float32)
    w4 = rng.random((256, 86), dtype=np.float32)

    def conv2d(x, k):
        kh, kw = k.shape
        h = (x.shape[0] - kh) // 2 + 1
        w = (x.shape[1] - kw) // 2 + 1
        out = np.zeros((h, w), dtype=np.float32)
        for i in range(kh):
            for j in range(kw):
                out += k[i, j] * x[i:i + 2 * h:2, j:j + 2 * w:2]
        return out

    def step():
        stage1 = [conv2d(img, k) for k in kernels[:4]]
        stage2 = [np.maximum(conv2d(x, k), 0.0)
                  for x in stage1 for k in kernels[4:6]]
        stage3 = np.maximum(conv2d(sum(stage2), kernels[6]), 0.0)
        h1 = np.maximum(feat @ w1, 0.0)
        h2 = np.maximum(h1 @ w2, 0.0)
        h3 = np.maximum(h2 @ w3, 0.0)
        return float(stage3.mean() + (h3 @ w4).max())

    for _ in range(warmup):
        step()
    samples: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        step()
        samples.append((time.perf_counter() - t0) * 1000.0)

    xs = sorted(samples)

    def pct(q: float) -> float:
        idx = min(len(xs) - 1, max(0, math.ceil(q * len(xs)) - 1))
        return xs[idx]

    rss = 0.0
    try:
        import psutil
        rss = psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:
        try:
            with open("/proc/self/statm", encoding="ascii") as fh:
                rss = (int(fh.read().split()[1])
                       * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024))
        except Exception:
            rss = 0.0

    return {
        "mean": statistics.fmean(samples),
        "p50": pct(0.50),
        "p95": pct(0.95),
        "p99": pct(0.99),
        "min": xs[0],
        "max": xs[-1],
        "rss_mb": rss,
    }


# ---------------------------------------------------------------------------
# 부모 프로세스 모드 — 스레드 수를 바꿔 자식을 반복 실행한다.
# ---------------------------------------------------------------------------


def run_child(threads: int, runs: int, warmup: int,
              env_extra: dict[str, str] | None = None) -> dict[str, float]:
    """스레드 수를 지정해 자식 프로세스 1개를 실행하고 결과를 받는다.

    Args:
        threads: 설정할 스레드 수.
        runs: 측정 반복 수.
        warmup: 워밍업 횟수.
        env_extra: 추가 환경변수.

    Returns:
        측정 결과 dict(실패 시 빈 dict).
    """
    env = os.environ.copy()
    val = str(max(1, threads))
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = val
    env["TORCH_NUM_THREADS"] = val
    if env_extra:
        env.update(env_extra)
    cmd = [sys.executable, "-I", str(Path(__file__).resolve()),
           "--child", "--runs", str(runs), "--warmup", str(warmup)]
    try:
        out = subprocess.run(cmd, env=env, capture_output=True,
                             text=True, timeout=600, check=True)
        return json.loads(out.stdout.strip().splitlines()[-1])
    except Exception as exc:  # noqa: BLE001
        print(f"  [실패] threads={threads}: {type(exc).__name__}")
        return {}


def sweep(max_threads: int, runs: int, warmup: int) -> list[tuple[int, dict]]:
    """1부터 max_threads 까지 스레드 수를 바꿔 측정한다."""
    print("\n[스레드 수 스윕 — 요청 1건을 직렬 처리]")
    print(f"  {'스레드':>6}{'p50ms':>9}{'p95ms':>9}{'p99ms':>9}"
          f"{'처리량':>10}{'코어당':>9}{'RSS MB':>9}")
    rows: list[tuple[int, dict]] = []
    for t in range(1, max_threads + 1):
        r = run_child(t, runs, warmup)
        if not r:
            continue
        thr = 1000.0 / r["p50"] if r["p50"] else 0.0
        print(f"  {t:>6}{r['p50']:>9.2f}{r['p95']:>9.2f}{r['p99']:>9.2f}"
              f"{thr:>9.1f}건/s{thr / t:>8.1f}{r['rss_mb']:>9.1f}")
        rows.append((t, r))
    return rows


def analyze(rows: list[tuple[int, dict]]) -> None:
    """스윕 결과에서 최적 스레드 수와 확장 효율을 뽑는다."""
    if not rows:
        print("  측정값이 없다.")
        return
    base = rows[0][1]["p50"]
    print("\n[확장 효율 — 스레드 1개 대비]")
    print(f"  {'스레드':>6}{'속도향상':>10}{'선형기대':>10}{'효율':>8}  판정")
    best_t, best_p50 = rows[0][0], rows[0][1]["p50"]
    for t, r in rows:
        speedup = base / r["p50"] if r["p50"] else 0.0
        eff = speedup / t * 100
        verdict = "양호" if eff >= 70 else ("보통" if eff >= 45 else "포화")
        print(f"  {t:>6}{speedup:>9.2f}x{t:>9.2f}x{eff:>7.0f}%  {verdict}")
        if r["p50"] < best_p50:
            best_t, best_p50 = t, r["p50"]
    print(f"\n  p50 최소 지점: 스레드 {best_t} ({best_p50:.2f} ms)")
    print("  권장: 단일 요청 지연이 중요하면 위 지점,")
    print("        동시 요청 처리량이 중요하면 스레드를 줄이고 워커를 늘린다.")


def worker_experiment(worker_counts: list[int], runs: int,
                      warmup: int, cores: int) -> None:
    """워커 × 스레드 조합을 동시에 띄워 코어 경합을 관찰한다.

    uvicorn --workers N 과 같은 상황을 모사한다. 전체 스레드 수
    (= 워커 수 × 스레드 수)가 코어 수를 넘으면 p95 가 급격히 나빠진다.

    Args:
        worker_counts: 동시에 띄울 프로세스 수 목록.
        runs: 워커당 측정 반복 수.
        warmup: 워밍업 횟수.
        cores: 물리/논리 코어 수.
    """
    print(f"\n[워커 × 스레드 동시 실행 — 코어 {cores}개]")
    print(f"  {'워커':>5}{'스레드':>7}{'총스레드':>9}{'평균p50':>10}"
          f"{'최악p95':>10}{'총처리량':>11}  상태")
    for w in worker_counts:
        # 워커당 스레드 1개(권장안)와 코어를 균등 분배한 값을 비교한다.
        # 두 값이 같으면 중복 측정을 피한다.
        thread_opts = sorted({1, max(1, cores // max(1, w))})
        for t in thread_opts:
            procs = []
            env = os.environ.copy()
            val = str(t)
            for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                        "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
                env[key] = val
            cmd = [sys.executable, "-I", str(Path(__file__).resolve()),
                   "--child", "--runs", str(runs), "--warmup", str(warmup)]
            for _ in range(w):
                procs.append(subprocess.Popen(cmd, env=env,
                                              stdout=subprocess.PIPE,
                                              stderr=subprocess.DEVNULL,
                                              text=True))
            results = []
            for p in procs:
                out, _ = p.communicate(timeout=600)
                try:
                    results.append(json.loads(out.strip().splitlines()[-1]))
                except Exception:  # noqa: BLE001
                    pass
            if not results:
                continue
            p50 = statistics.fmean(r["p50"] for r in results)
            p95 = max(r["p95"] for r in results)
            total = sum(1000.0 / r["p50"] for r in results if r["p50"])
            over = w * t > cores
            print(f"  {w:>5}{t:>7}{w * t:>9}{p50:>10.2f}{p95:>10.2f}"
                  f"{total:>10.1f}건/s  {'초과' if over else '적정'}")


def main(argv: list[str] | None = None) -> int:
    """명령행 진입점."""
    ap = argparse.ArgumentParser(description="CPU 스레드 수 스윕")
    ap.add_argument("--child", action="store_true",
                    help="(내부용) 자식 프로세스 측정 모드")
    ap.add_argument("--runs", type=int, default=40)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--max-threads", type=int, default=0,
                    help="0 이면 CPU 코어 수")
    ap.add_argument("--workers", default="",
                    help="쉼표 구분 워커 수 목록(예: 1,2,4). 비우면 생략")
    args = ap.parse_args(argv)

    if args.child:
        print(json.dumps(child_measure(args.runs, args.warmup)))
        return 0

    cores = os.cpu_count() or 1
    max_t = args.max_threads if args.max_threads > 0 else cores

    print("=" * 70)
    print("CPU 스레드 설정 스윕")
    print("=" * 70)
    print(f"  CPU 코어 {cores}개 / 스윕 범위 1~{max_t} / "
          f"측정 {args.runs}회(워밍업 {args.warmup}회)")
    print("  각 스레드 수는 별도 프로세스에서 측정한다"
          " (환경변수가 import 이전에 적용되어야 하므로).")

    rows = sweep(max_t, args.runs, args.warmup)
    analyze(rows)

    if args.workers.strip():
        counts = [int(x) for x in args.workers.split(",") if x.strip()]
        worker_experiment(counts, args.runs, args.warmup, cores)

    print("\n[설정 반영 위치]")
    print("  ai-service/.env        TORCH_THREADS=<위 결과>")
    print("  컨테이너 환경변수      OMP_NUM_THREADS=<동일 값>")
    print("  uvicorn 기동           --workers <코어수 / 스레드수>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
