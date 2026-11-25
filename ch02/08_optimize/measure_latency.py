# -*- coding: utf-8 -*-
"""CPU 추론 지연시간 / 메모리 / FPS 를 측정한다.

측정 원칙(이걸 지키지 않으면 수치가 재현되지 않는다)
  1) 워밍업을 한다. 첫 호출에는 메모리 할당과 커널 초기화 비용이 섞인다
  2) 평균 대신 p50(중앙값)과 p95 를 본다. 운영에서 중요한 건 '느린 꼬리'다
  3) 스레드 수를 고정한다. 자동 설정은 장비마다 달라진다
  4) 전처리/추론/후처리를 나눠 측정한다. 병목이 추론이 아닐 수도 있다
  5) 다른 프로그램을 끄고 측정한다. 백그라운드 부하가 p95 를 크게 흔든다

측정 대상은 ONNX 모델, Ultralytics 가중치, 또는 내장 더미 연산이다.
라이브러리가 없으면 numpy 행렬곱을 측정 대상으로 삼아 측정 절차만 익힌다.

사용법:
    python 08_optimize/measure_latency.py --onnx weights/plate_yolo_int8.onnx \
        --runs 100 --warmup 10 --threads 4
"""
from __future__ import annotations

import argparse
import gc
import json
import pathlib
import sys
import time


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="CPU 지연시간/메모리/FPS 측정")
    ap.add_argument("--onnx", default="weights/plate_yolo.onnx")
    ap.add_argument("--weights", default="weights/plate_yolo.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default="runs/optimize")
    return ap.parse_args()


def percentile(vals: list[float], q: float) -> float:
    """선형 보간 분위수."""
    if not vals:
        return 0.0
    s = sorted(vals)
    if len(s) == 1:
        return s[0]
    pos = (len(s) - 1) * q / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def rss_mb() -> float:
    """현재 프로세스의 실제 메모리 사용량(MB). 측정 불가면 0.0.

    /proc/self/statm 은 현재 RSS 를, resource 모듈은 '최대' RSS 를 준다.
    증감을 보려면 현재값이 필요하므로 /proc 을 먼저 시도한다.
    """
    # 리눅스: /proc/self/statm 의 두 번째 값이 현재 RSS(페이지 수)다
    try:
        with open("/proc/self/statm", encoding="ascii") as fh:
            pages = int(fh.read().split()[1])
        import os                               # noqa: PLC0415
        return pages * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)
    except (OSError, ValueError, IndexError):
        pass
    # 윈도우/맥: psutil 이 있으면 현재 RSS 를 쓴다
    try:
        import psutil                           # noqa: PLC0415
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except ImportError:
        pass
    # 마지막 수단: 최대 RSS. 리눅스는 KB, 맥은 바이트 단위로 보고한다
    try:
        import resource                         # noqa: PLC0415
        import sys as _sys                      # noqa: PLC0415
        v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return v / (1024 * 1024) if _sys.platform == "darwin" else v / 1024.0
    except ImportError:
        return 0.0


def run_benchmark(fn, runs: int, warmup: int):
    """fn 을 반복 호출해 지연시간 분포를 구한다."""
    for _ in range(warmup):
        fn()
    gc.collect()                                # 측정 중 GC 가 튀는 것을 줄인다
    samples: list[float] = []
    t_start = time.perf_counter()
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    wall = time.perf_counter() - t_start
    return {
        "runs": runs,
        "min_ms": min(samples),
        "p50_ms": percentile(samples, 50),
        "p90_ms": percentile(samples, 90),
        "p95_ms": percentile(samples, 95),
        "p99_ms": percentile(samples, 99),
        "max_ms": max(samples),
        "mean_ms": sum(samples) / len(samples),
        "fps_sequential": runs / wall,
        "wall_s": wall,
    }


def print_result(title: str, r: dict, mem_before: float, mem_after: float) -> None:
    """측정 결과를 표로 출력한다."""
    print(f"\n=== {title} ===")
    head = f"{'지표':16s}{'값':>12s}  설명"
    print(head)
    print("-" * 66)
    rows = [
        ("최소", f"{r['min_ms']:.2f} ms", "가장 빠른 1회. 이상적 조건"),
        ("p50(중앙값)", f"{r['p50_ms']:.2f} ms", "대표 지연시간. 설계 기준"),
        ("p90", f"{r['p90_ms']:.2f} ms", "10%가 이보다 느리다"),
        ("p95", f"{r['p95_ms']:.2f} ms", "SLA 기준으로 흔히 쓴다"),
        ("p99", f"{r['p99_ms']:.2f} ms", "최악 구간. 타임아웃 설계에 쓴다"),
        ("최대", f"{r['max_ms']:.2f} ms", "GC/스케줄링 영향 포함"),
        ("평균", f"{r['mean_ms']:.2f} ms", "꼬리에 끌려 왜곡될 수 있다"),
        ("FPS(순차)", f"{r['fps_sequential']:.1f}", "1/p50 과 다를 수 있다"),
    ]
    for k, v, memo in rows:
        print(f"{k:16s}{v:>12s}  {memo}")
    print("-" * 66)
    print(f"{'메모리(측정전)':16s}{mem_before:>9.1f} MB")
    print(f"{'메모리(측정후)':16s}{mem_after:>9.1f} MB  "
          f"증가 {mem_after - mem_before:+.1f} MB")


def build_target(args):
    """측정 대상 함수를 만든다. (이름, 함수) 를 돌려준다."""
    # 1순위: ONNX 모델
    onnx_path = pathlib.Path(args.onnx)
    if onnx_path.exists():
        try:
            import numpy as np                  # noqa: PLC0415
            import onnxruntime as ort           # noqa: PLC0415
            opt = ort.SessionOptions()
            opt.intra_op_num_threads = args.threads
            opt.graph_optimization_level = \
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess = ort.InferenceSession(str(onnx_path), opt,
                                        providers=["CPUExecutionProvider"])
            inp = sess.get_inputs()[0]
            shape = [d if isinstance(d, int) else (1 if i == 0 else args.imgsz)
                     for i, d in enumerate(inp.shape)]
            x = np.zeros(shape, dtype=np.float32)
            return (f"ONNX Runtime: {onnx_path.name}",
                    lambda: sess.run(None, {inp.name: x}))
        except ImportError:
            pass

    # 2순위: Ultralytics 가중치
    wpath = pathlib.Path(args.weights)
    if wpath.exists():
        try:
            import numpy as np                  # noqa: PLC0415
            import torch                        # noqa: PLC0415
            from ultralytics import YOLO        # noqa: PLC0415
            torch.set_num_threads(args.threads)
            model = YOLO(str(wpath))
            dummy = np.zeros((args.imgsz, args.imgsz, 3), dtype=np.uint8)
            return (f"Ultralytics: {wpath.name}",
                    lambda: model.predict(dummy, imgsz=args.imgsz,
                                          device="cpu", verbose=False))
        except ImportError:
            pass

    # 3순위: numpy 행렬곱(측정 절차 시연용)
    try:
        import numpy as np                      # noqa: PLC0415
        a = np.random.rand(512, 512).astype(np.float32)
        b = np.random.rand(512, 512).astype(np.float32)
        return ("대체 측정 대상: numpy 512x512 행렬곱", lambda: a @ b)
    except ImportError:
        pass

    # 4순위: 순수 파이썬 연산
    def pure():
        s = 0.0
        for i in range(200_000):
            s += i * 0.5
        return s
    return ("대체 측정 대상: 순수 파이썬 루프", pure)


def main() -> int:
    args = parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    name, fn = build_target(args)
    if name.startswith("대체"):
        print("[안내] ONNX/YOLO 모델을 찾지 못해 대체 대상으로 측정 절차를 보여준다.")
        print("       먼저 08_optimize/export_onnx.py 로 모델을 준비한다.")

    print(f"\n측정 대상 : {name}")
    print(f"설정      : runs={args.runs} warmup={args.warmup} "
          f"threads={args.threads} imgsz={args.imgsz}")

    mem_before = rss_mb()
    r = run_benchmark(fn, args.runs, args.warmup)
    mem_after = rss_mb()

    print_result("지연시간 측정 결과", r, mem_before, mem_after)

    # 운영 환경 환산: 한 프로세스가 카메라를 순차 처리할 때의 처리량을 본다
    p50 = r["p50_ms"]
    total_fps = 1000.0 / p50 if p50 else 0.0
    print("\n=== 운영 환산 (단일 프로세스 순차 처리) ===")
    head = f"{'카메라 수':>9s}{'대당 FPS':>11s}{'프레임 간격':>13s}  판정(대당 5 FPS 기준)"
    print(head)
    print("-" * len(head))
    for cam in (1, 2, 4, 8):
        per_cam = total_fps / cam
        interval = 1000.0 / per_cam if per_cam else 0.0
        verdict = "여유" if per_cam >= 10 else ("가능" if per_cam >= 5 else "부족")
        print(f"{cam:9d}{per_cam:11.1f}{interval:11.1f}ms  {verdict}")
    print("-" * len(head))
    print(f"  총 처리량 {total_fps:.1f} FPS 를 카메라 수로 나눈 값이다.")
    print("  주차장 입출구는 보통 대당 5~10 FPS 로 충분하다.")
    print("  '부족' 이면 프로세스를 늘리거나 해상도/모델을 한 단계 낮춘다.")
    print("  p95 가 200ms 를 넘으면 차량이 차단기를 지나기 전에 결과가 안 나올 수 있다.")

    r["target"] = name
    r["threads"] = args.threads
    r["mem_before_mb"] = mem_before
    r["mem_after_mb"] = mem_after
    (out / "latency.json").write_text(
        json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n결과 저장: {out / 'latency.json'}")
    print("※ 모든 수치는 이 장비의 예시 측정값이다. 배포 장비에서 재측정한다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
