# -*- coding: utf-8 -*-
"""NMS(Non-Maximum Suppression) 직접 구현 실습.

검출기는 같은 번호판 하나에 대해 여러 개의 박스를 내놓는다.
NMS 는 신뢰도가 가장 높은 박스를 남기고 그와 많이 겹치는 박스를 제거한다.
Soft-NMS 와의 차이도 함께 확인한다.
"""
from __future__ import annotations

import math


def iou(a, b) -> float:
    """두 박스 (x1,y1,x2,y2) 의 IoU."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def nms(boxes, scores, iou_thr: float = 0.45, score_thr: float = 0.25):
    """Greedy NMS. 남길 박스의 인덱스 목록을 반환한다.

    Args:
        boxes: [(x1,y1,x2,y2), ...]
        scores: 각 박스의 신뢰도
        iou_thr: 이 값보다 많이 겹치면 중복으로 보고 제거한다
        score_thr: 이 값보다 낮은 신뢰도는 먼저 버린다
    """
    # 1단계: 신뢰도 하한을 통과한 후보만 남긴다
    cand = [i for i, s in enumerate(scores) if s >= score_thr]
    # 2단계: 신뢰도 내림차순으로 정렬한다
    cand.sort(key=lambda i: scores[i], reverse=True)

    keep: list[int] = []
    while cand:
        best = cand.pop(0)       # 가장 신뢰도 높은 박스를 확정한다
        keep.append(best)
        # 확정 박스와 많이 겹치는 나머지 후보를 제거한다
        cand = [i for i in cand if iou(boxes[best], boxes[i]) < iou_thr]
    return keep


def soft_nms(boxes, scores, sigma: float = 0.5, score_thr: float = 0.10):
    """Soft-NMS. 겹치는 박스를 '삭제'하지 않고 신뢰도를 감쇠시킨다.

    감쇠식은 가우시안 가중치 exp(-IoU^2 / sigma) 를 사용한다.
    혼잡한 주차장처럼 번호판이 서로 붙어 있는 장면에서 누락을 줄일 수 있다.
    """
    s = list(scores)
    order = sorted(range(len(boxes)), key=lambda i: s[i], reverse=True)
    keep: list[tuple[int, float]] = []
    remaining = list(order)
    while remaining:
        remaining.sort(key=lambda i: s[i], reverse=True)
        best = remaining.pop(0)
        keep.append((best, s[best]))
        for i in remaining:
            v = iou(boxes[best], boxes[i])
            s[i] = s[i] * math.exp(-(v * v) / sigma)
    return [(i, sc) for i, sc in keep if sc >= score_thr]


def main() -> None:
    # 같은 번호판 하나(12가3456)에 대해 5개, 다른 번호판(87로2345)에 1개가 나온 상황
    boxes = [
        (412.0, 288.0, 596.0, 342.0),   # 0  가장 정확한 박스
        (408.0, 285.0, 600.0, 346.0),   # 1  0 번과 거의 동일
        (420.0, 292.0, 588.0, 338.0),   # 2  0 번과 거의 동일
        (500.0, 290.0, 684.0, 344.0),   # 3  0 번과 절반만 겹침
        (900.0, 600.0, 1084.0, 654.0),  # 4  별개의 번호판
        (905.0, 604.0, 1080.0, 650.0),  # 5  4 번과 거의 동일
    ]
    scores = [0.981, 0.944, 0.902, 0.631, 0.877, 0.812]
    names = ["A-주", "A-중복1", "A-중복2", "A-이웃", "B-주", "B-중복1"]

    print("[입력 검출 결과]")
    for i, (b, s) in enumerate(zip(boxes, scores)):
        print(f"  {i} {names[i]:8s} score={s:.3f} bbox={b}")

    print("\n[IoU 임계값별 Greedy NMS 결과]")
    print(f"{'iou_thr':>8s}{'남은개수':>10s}  남은 인덱스(신뢰도)")
    for thr in (0.30, 0.45, 0.60, 0.90):
        keep = nms(boxes, scores, iou_thr=thr, score_thr=0.25)
        detail = ", ".join(f"{i}({scores[i]:.3f})" for i in keep)
        print(f"{thr:8.2f}{len(keep):10d}  {detail}")

    print("\n[신뢰도 임계값별 NMS 결과] iou_thr=0.45 고정")
    for st in (0.10, 0.25, 0.50, 0.70):
        keep = nms(boxes, scores, iou_thr=0.45, score_thr=st)
        print(f"  score_thr={st:.2f} -> 남은 개수 {len(keep)}  {keep}")

    print("\n[Soft-NMS 결과] sigma=0.5")
    for i, sc in soft_nms(boxes, scores, sigma=0.5, score_thr=0.10):
        print(f"  {i} {names[i]:8s} 감쇠후 score={sc:.3f}")


if __name__ == "__main__":
    main()
