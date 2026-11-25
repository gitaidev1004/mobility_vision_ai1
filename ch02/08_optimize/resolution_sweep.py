# -*- coding: utf-8 -*-
"""입력 해상도별 정확도/지연시간 스윕(Sweep).

해상도를 키우면 작은 번호판을 더 잘 찾지만 추론시간이 해상도의 제곱에 가깝게
늘어난다. '가장 정확한 해상도' 가 아니라 '요구 지연시간 안에서 가장 정확한
해상도' 를 찾는 것이 목적이다.

측정 항목
  - imgsz 별 Recall / Precision / mAP@0.5 (가중치가 있으면 실측)
  - imgsz 별 p50 / p95 지연시간
  - 지연 예산(--budget-ms) 을 만족하는 후보 중 최고 성능 지점 추천

가중치가 없으면 해상도-연산량 관계(제곱 비례)와 공개 자료 수준의 경향을
반영한 예시 측정값 표를 출력해 판단 절차를 익히게 한다.

사용법:
    python 08_optimize/resolution_sweep.py --weights weights/plate_yolo.pt \
        --data 05_train/data.yaml --sizes 320 416 512 640 800 --budget-ms 80
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

# 예시 측정값: 4코어 CPU / 경량 YOLO / 번호판 단일 클래스 가정.
# (imgsz, Recall, Precision, mAP@0.5, p50 ms, p95 ms)
FALLBACK = [
    (320, 0.902, 0.962, 0.918, 18.4, 24.1),
    (416, 0.941, 0.967, 0.948, 29.7, 38.2),
    (512, 0.963, 0.971, 0.964, 43.1, 55.6),
    (640, 0.978, 0.974, 0.976, 66.8, 86.3),
    (800, 0.982, 0.972, 0.979, 103.5, 134.7),
    (960, 0.983, 0.969, 0.979, 148.2, 191.4),
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="입력 해상도 스윕")
    ap.add_argument("--weights", default="weights/plate_yolo.pt")
    ap.add_argument("--data", default="05_train/data.yaml")
    ap.add_argument("--sizes", type=int, nargs="+",
                    default=[320, 416, 512, 640, 800])
    ap.add_argument("--budget-ms", type=float, default=80.0,
                    help="검출 단계가 지켜야 할 p95 지연 예산(ms)")
    ap.add_argument("--runs", type=int, default=10,
                    help="지연시간 측정 반복 횟수")
    ap.add_argument("--threads", type=int, default=4,
                    help="CPU 스레드 수. 고정해야 측정이 재현된다")
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


def print_table(rows) -> None:
    """스윕 결과표를 출력한다."""
    head = (f"{'imgsz':>7s}{'Recall':>9s}{'Precision':>11s}{'mAP@0.5':>10s}"
            f"{'p50(ms)':>10s}{'p95(ms)':>10s}{'FPS':>8s}{'상대연산':>10s}")
    print(head)
    print("-" * len(head))
    base = rows[0][0] if rows else 1
    for sz, rec, prec, m50, p50, p95 in rows:
        fps = 1000.0 / p50 if p50 else 0.0
        rel = (sz / base) ** 2                 # 연산량은 대략 해상도의 제곱에 비례
        print(f"{sz:7d}{rec:9.4f}{prec:11.4f}{m50:10.4f}"
              f"{p50:10.1f}{p95:10.1f}{fps:8.1f}{rel:9.2f}x")
    print("-" * len(head))


def recommend(rows, budget_ms: float):
    """지연 예산을 만족하는 후보 중 mAP 가 가장 높은 해상도를 고른다."""
    ok = [r for r in rows if r[5] <= budget_ms]
    if not ok:
        return None, min(rows, key=lambda r: r[5])
    return max(ok, key=lambda r: r[3]), None


def main() -> int:
    args = parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows: list[tuple] = []
    mode = "fallback"

    have = False
    try:
        import numpy as np                      # noqa: PLC0415
        from ultralytics import YOLO            # noqa: PLC0415
        have = pathlib.Path(args.weights).exists()
    except ImportError:
        pass

    if have:
        import torch                            # noqa: PLC0415
        torch.set_num_threads(args.threads)     # 스레드 수를 고정한다
        model = YOLO(args.weights)
        data_ok = pathlib.Path(args.data).exists()
        print(f"=== 실측 스윕 (스레드 {args.threads}) ===")
        for sz in args.sizes:
            # 정확도: data.yaml 이 있으면 검증 세트로 측정한다
            rec = prec = m50 = 0.0
            if data_ok:
                m = model.val(data=args.data, imgsz=sz, device="cpu",
                              verbose=False)
                rec, prec, m50 = float(m.box.mr), float(m.box.mp), float(m.box.map50)
            # 지연: 같은 더미 입력으로 반복 측정한다
            dummy = np.zeros((sz, sz, 3), dtype=np.uint8)
            for _ in range(3):                  # 워밍업
                model.predict(dummy, imgsz=sz, device="cpu", verbose=False)
            samples = []
            for _ in range(args.runs):
                t0 = time.perf_counter()
                model.predict(dummy, imgsz=sz, device="cpu", verbose=False)
                samples.append((time.perf_counter() - t0) * 1000.0)
            rows.append((sz, rec, prec, m50,
                         percentile(samples, 50), percentile(samples, 95)))
            print(f"  imgsz={sz} 완료")
        mode = "measured"
    else:
        print("[안내] 가중치 또는 ultralytics 가 없어 예시 측정값으로 절차를 보여준다.")
        print("       아래 수치는 참고용이며 장비와 데이터에 따라 달라진다.\n")
        sel = set(args.sizes)
        rows = [r for r in FALLBACK if r[0] in sel] or FALLBACK

    print(f"\n=== 해상도 스윕 결과 ({mode}) ===")
    print_table(rows)

    best, fastest = recommend(rows, args.budget_ms)
    print(f"\n=== 추천 (p95 예산 {args.budget_ms:.0f}ms) ===")
    if best:
        print(f"  선택: imgsz={best[0]}  mAP@0.5={best[3]:.4f} "
              f"p95={best[5]:.1f}ms Recall={best[1]:.4f}")
        print("  근거: 예산을 만족하는 후보 중 mAP 가 가장 높다.")
        # 한 단계 올렸을 때의 이득/손실을 함께 보여준다
        idx = rows.index(best)
        if idx + 1 < len(rows):
            nxt = rows[idx + 1]
            print(f"  참고: imgsz={nxt[0]} 로 올리면 mAP +{nxt[3] - best[3]:.4f} / "
                  f"p95 +{nxt[5] - best[5]:.1f}ms (예산 초과)")
    else:
        print(f"  예산을 만족하는 해상도가 없다. 가장 빠른 imgsz={fastest[0]} "
              f"(p95 {fastest[5]:.1f}ms) 도 초과한다.")
        print("  대응: 더 작은 모델, ONNX/INT8 양자화, 또는 프레임 샘플링 간격 조정")

    (out / "resolution_sweep.json").write_text(
        json.dumps({"mode": mode, "budget_ms": args.budget_ms,
                    "rows": [{"imgsz": r[0], "recall": r[1], "precision": r[2],
                              "map50": r[3], "p50_ms": r[4], "p95_ms": r[5]}
                             for r in rows]}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"\n결과 저장: {out / 'resolution_sweep.json'}")
    print("※ 해상도를 2배로 키우면 연산량은 약 4배가 된다. 이 관계를 먼저 기억한다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
