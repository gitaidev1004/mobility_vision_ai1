# -*- coding: utf-8 -*-
"""ONNX Runtime 동적 양자화(Dynamic Quantization)로 INT8 모델을 만든다.

동적 양자화는 가중치만 미리 INT8 로 바꾸고, 활성값의 스케일은 추론할 때
동적으로 계산한다. 보정 데이터셋(calibration set)이 필요 없어 적용이 쉽다.

  동적 양자화  가중치 INT8 / 활성값 런타임 계산  - 보정 불필요, 적용 간단
  정적 양자화  가중치·활성값 모두 INT8          - 보정 필요, 더 빠를 수 있음

기대 효과(예시)
  - 모델 파일 크기 약 1/4 (FP32 → INT8)
  - CPU 추론 지연 20~45% 감소 (연산/명령어 지원에 따라 다르다)
  - 정확도 소폭 하락 (mAP 0.5~2%p). 반드시 전후를 측정해 허용 범위인지 본다

주의
  - 양자화가 항상 빨라지지는 않는다. 작은 모델이나 메모리 대역폭 병목
    구간에서는 오히려 느려지는 경우도 있다.
  - 번호판처럼 작은 객체는 양자화 오차에 민감할 수 있으므로
    소형 번호판 부분집합의 Recall 을 따로 확인한다.

사용법:
    python 08_optimize/quantize_int8.py --model weights/plate_yolo.onnx \
        --out weights/plate_yolo_int8.onnx --runs 20
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

# 예시 측정값: 4코어 CPU / 경량 YOLO / imgsz 640 가정
FALLBACK = [
    # (구분, 파일크기 MB, p50 ms, p95 ms, 메모리 MB, mAP@0.5, Recall)
    ("FP32 (PyTorch)", 6.2, 66.8, 86.3, 214.0, 0.9760, 0.9780),
    ("FP32 (ONNX)", 12.1, 52.4, 68.1, 168.0, 0.9760, 0.9780),
    ("INT8 (동적양자화)", 3.3, 34.7, 46.9, 121.0, 0.9701, 0.9712),
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="ONNX 동적 양자화")
    ap.add_argument("--model", default="weights/plate_yolo.onnx",
                    help="입력 ONNX 모델(FP32)")
    ap.add_argument("--out", default="",
                    help="출력 경로. 비우면 입력명 + _int8.onnx")
    ap.add_argument("--per-channel", action="store_true",
                    help="채널별 스케일을 쓴다(정확도 유지에 유리, 약간 느림)")
    ap.add_argument("--runs", type=int, default=20,
                    help="전후 지연시간 측정 반복 횟수")
    ap.add_argument("--threads", type=int, default=4)
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


def bench(path: pathlib.Path, runs: int, threads: int):
    """ONNX 모델의 p50/p95 지연시간(ms)을 측정한다."""
    import numpy as np                          # noqa: PLC0415
    import onnxruntime as ort                   # noqa: PLC0415

    opt = ort.SessionOptions()
    opt.intra_op_num_threads = threads          # 스레드 수를 고정해 측정을 재현한다
    opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(path), opt,
                                providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    shape = [d if isinstance(d, int) else (1 if i == 0 else 640)
             for i, d in enumerate(inp.shape)]
    x = np.zeros(shape, dtype=np.float32)

    for _ in range(3):                          # 워밍업
        sess.run(None, {inp.name: x})
    samples = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(None, {inp.name: x})
        samples.append((time.perf_counter() - t0) * 1000.0)
    return percentile(samples, 50), percentile(samples, 95)


def print_table(rows) -> None:
    """양자화 전후 비교표를 출력한다."""
    head = (f"{'구분':20s}{'크기(MB)':>10s}{'p50(ms)':>10s}{'p95(ms)':>10s}"
            f"{'메모리(MB)':>12s}{'mAP@0.5':>10s}{'Recall':>9s}")
    print(head)
    print("-" * len(head))
    for name, size, p50, p95, mem, m50, rec in rows:
        print(f"{name:20s}{size:10.1f}{p50:10.1f}{p95:10.1f}"
              f"{mem:12.1f}{m50:10.4f}{rec:9.4f}")
    print("-" * len(head))


def main() -> int:
    args = parse_args()
    src = pathlib.Path(args.model)

    try:
        from onnxruntime.quantization import (  # noqa: PLC0415
            QuantType, quantize_dynamic)
        import onnxruntime as ort               # noqa: PLC0415
        have_ort = True
    except ImportError:
        have_ort = False

    if not have_ort or not src.exists():
        if not have_ort:
            print("[안내] onnxruntime 이 없다.")
            print("       pip install onnxruntime==1.22.1")
        else:
            print(f"[안내] 입력 ONNX 모델이 없다: {src}")
            print("       먼저 08_optimize/export_onnx.py 를 실행한다.")
        print("\n아래는 참고용 예시 측정값이다. 장비와 데이터에 따라 달라진다.\n")
        print_table(FALLBACK)
        print("\n판단 기준")
        print("  - 지연시간 감소가 20% 이상이고 mAP 하락이 1%p 이내면 채택을 검토한다")
        print("  - 소형 번호판 Recall 이 크게 떨어지면 양자화를 보류하거나")
        print("    해상도를 한 단계 올려 보상한다")
        return 0

    dst = pathlib.Path(args.out) if args.out else \
        src.with_name(src.stem + "_int8.onnx")
    dst.parent.mkdir(parents=True, exist_ok=True)

    print(f"onnxruntime {ort.__version__}")
    print(f"입력 : {src}  ({src.stat().st_size / (1024 * 1024):.2f} MB)")

    # 양자화 전 성능을 먼저 측정한다(같은 세션 조건에서 비교해야 의미가 있다)
    p50_fp32, p95_fp32 = bench(src, args.runs, args.threads)

    quantize_dynamic(
        model_input=str(src),
        model_output=str(dst),
        weight_type=QuantType.QInt8,            # 가중치를 INT8 로 바꾼다
        per_channel=args.per_channel,
        reduce_range=False,                     # 구형 CPU 에서는 True 가 안전하다
    )
    print(f"출력 : {dst}  ({dst.stat().st_size / (1024 * 1024):.2f} MB)")

    p50_int8, p95_int8 = bench(dst, args.runs, args.threads)

    size_fp32 = src.stat().st_size / (1024 * 1024)
    size_int8 = dst.stat().st_size / (1024 * 1024)
    rows = [
        ("FP32 (ONNX)", size_fp32, p50_fp32, p95_fp32, 0.0, 0.0, 0.0),
        ("INT8 (동적양자화)", size_int8, p50_int8, p95_int8, 0.0, 0.0, 0.0),
    ]
    print("\n=== 양자화 전후 비교 (지연시간 실측) ===")
    print_table(rows)
    print(f"  파일 크기 : {size_fp32 / size_int8:.2f} 배 축소")
    print(f"  p50 지연  : {(1 - p50_int8 / p50_fp32) * 100:+.1f}% 변화")
    print(f"  p95 지연  : {(1 - p95_int8 / p95_fp32) * 100:+.1f}% 변화")
    print("\n정확도는 이 스크립트에서 측정하지 않는다.")
    print("06_eval/eval_detector.py 를 INT8 모델로 다시 실행해")
    print("mAP 와 소형 번호판 Recall 하락폭을 반드시 확인한다.")
    print("※ 모든 수치는 이 장비의 예시 측정값이다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
