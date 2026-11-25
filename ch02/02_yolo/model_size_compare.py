# -*- coding: utf-8 -*-
"""YOLO 계열 모델의 크기 / 파라미터 수 / 추론시간을 비교하는 표를 출력한다.

ultralytics 가 설치되어 있으면 실제 모델을 올려 측정하고,
없으면 공개 자료를 참고한 '예시 측정값' 표를 출력해 비교 방법만 익히게 한다.
실제 수치는 CPU 종류, 스레드 수, 입력 해상도에 따라 달라진다.
"""
from __future__ import annotations

import time

# 예시 측정값: 4코어 CPU, imgsz=640, 단일 이미지 기준으로 가정한 참고표
FALLBACK_ROWS = [
    # (모델, 파라미터(M), 모델크기(MB), p50 지연(ms), 상대 정확도 메모)
    ("YOLO-n (nano)", 3.2, 6.2, 42.0, "기준선. CPU 실시간에 가장 유리"),
    ("YOLO-s (small)", 11.2, 21.5, 95.0, "nano 대비 소형 번호판에 유리"),
    ("YOLO-m (medium)", 25.9, 49.7, 232.0, "CPU 단독 실시간은 어려움"),
    ("YOLO-l (large)", 43.7, 83.6, 410.0, "GPU 환경 또는 배치 처리용"),
]


def count_params(model) -> int:
    """파이토치 모듈의 전체 파라미터 개수를 센다."""
    return sum(p.numel() for p in model.parameters())


def measure_latency(predict_fn, warmup: int = 2, runs: int = 5) -> float:
    """워밍업 후 여러 번 실행한 중앙값(ms)을 돌려준다.

    첫 호출에는 메모리 할당과 커널 초기화 비용이 섞이므로 반드시 워밍업한다.
    """
    for _ in range(warmup):
        predict_fn()
    samples = []
    for _ in range(runs):
        t0 = time.perf_counter()
        predict_fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    samples.sort()
    return samples[len(samples) // 2]


def print_table(rows) -> None:
    """비교표를 고정폭으로 출력한다."""
    head = f"{'모델':18s}{'파라미터(M)':>12s}{'크기(MB)':>10s}{'지연 p50(ms)':>14s}  비고"
    print(head)
    print("-" * len(head))
    for name, params, size, ms, memo in rows:
        print(f"{name:18s}{params:12.2f}{size:10.1f}{ms:14.1f}  {memo}")
    print("-" * len(head))


def main() -> None:
    try:
        import numpy as np                      # noqa: PLC0415
        from ultralytics import YOLO            # noqa: PLC0415
    except ImportError:
        print("[안내] ultralytics/numpy 가 없어 실측을 건너뛴다.")
        print("       pip install ultralytics==8.3.200 numpy==2.1.3 후 재실행한다.")
        print("       아래는 참고용 예시 측정값이며 장비에 따라 달라진다.\n")
        print_table(FALLBACK_ROWS)
        return

    # 640x640 더미 이미지로 측정한다(실제 사진이 없어도 비교가 가능하다)
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    rows = []
    for tag in ("yolo11n.pt", "yolo11s.pt"):
        try:
            model = YOLO(tag)
        except Exception as exc:                # noqa: BLE001
            print(f"[안내] {tag} 를 불러올 수 없다({exc.__class__.__name__}). 건너뛴다.")
            continue
        params = count_params(model.model) / 1e6
        size_mb = sum(p.numel() * p.element_size()
                      for p in model.model.parameters()) / (1024 * 1024)
        ms = measure_latency(
            lambda m=model: m.predict(dummy, imgsz=640, device="cpu", verbose=False))
        rows.append((tag.replace(".pt", ""), params, size_mb, ms, "실측값"))

    if not rows:
        print("[안내] 실측 가능한 모델이 없어 예시 측정값을 출력한다.\n")
        rows = FALLBACK_ROWS
    print_table(rows)
    print("※ 위 수치는 특정 장비의 예시 측정값이다. 팀의 CPU 와 데이터로 직접 재측정한다.")


if __name__ == "__main__":
    main()
