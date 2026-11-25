# -*- coding: utf-8 -*-
"""IoU(Intersection over Union) 계산 실습.

번호판 검출 결과가 정답과 얼마나 겹치는지 수치로 나타내는 지표를 직접 구현한다.
외부 라이브러리 없이 순수 파이썬으로 동작하므로 어떤 환경에서도 실행된다.
"""
from __future__ import annotations


def box_area(box: tuple[float, float, float, float]) -> float:
    """박스 면적을 구한다. box 는 (x1, y1, x2, y2) 순서다."""
    x1, y1, x2, y2 = box
    w = max(0.0, x2 - x1)        # 음수 폭은 0 으로 눌러 잘못된 박스를 방어한다
    h = max(0.0, y2 - y1)
    return w * h


def intersection(a, b) -> float:
    """두 박스의 교집합 면적을 구한다."""
    # 교집합 영역의 좌상단은 두 박스 좌상단 중 '더 큰' 값이다
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    # 교집합 영역의 우하단은 두 박스 우하단 중 '더 작은' 값이다
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    iw = max(0.0, ix2 - ix1)     # 겹치지 않으면 음수가 되므로 0 으로 만든다
    ih = max(0.0, iy2 - iy1)
    return iw * ih


def iou(a, b) -> float:
    """IoU = 교집합 면적 / 합집합 면적. 0.0 ~ 1.0 범위를 가진다."""
    inter = intersection(a, b)
    union = box_area(a) + box_area(b) - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def iou_matrix(preds, gts) -> list[list[float]]:
    """예측 박스 목록과 정답 박스 목록의 모든 조합에 대한 IoU 행렬을 만든다."""
    return [[iou(p, g) for g in gts] for p in preds]


def xywh_to_xyxy(cx: float, cy: float, w: float, h: float):
    """중심좌표+크기 형식을 좌상단/우하단 형식으로 바꾼다(YOLO 라벨 변환에 쓰인다)."""
    return (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)


def main() -> None:
    # 정답 번호판 박스: 1920x1080 영상에서 번호판이 (412,288)-(596,342) 에 있다고 가정
    gt = (412.0, 288.0, 596.0, 342.0)

    cases = [
        ("완전 일치", (412.0, 288.0, 596.0, 342.0)),
        ("약간 어긋남", (418.0, 292.0, 600.0, 346.0)),
        ("절반만 겹침", (504.0, 288.0, 688.0, 342.0)),
        ("박스가 너무 큼", (380.0, 260.0, 640.0, 380.0)),
        ("박스가 너무 작음", (460.0, 302.0, 548.0, 330.0)),
        ("완전히 빗나감", (900.0, 600.0, 1080.0, 654.0)),
    ]

    print("정답 박스:", gt, " 면적 =", box_area(gt))
    print("-" * 56)
    print(f"{'사례':16s}{'IoU':>8s}{'0.5판정':>10s}{'0.7판정':>10s}")
    print("-" * 56)
    for name, pred in cases:
        v = iou(gt, pred)
        ok50 = "TP" if v >= 0.5 else "FP"
        ok70 = "TP" if v >= 0.7 else "FP"
        print(f"{name:16s}{v:8.4f}{ok50:>10s}{ok70:>10s}")
    print("-" * 56)

    # YOLO 정규화 라벨(클래스 cx cy w h)을 픽셀 박스로 되돌려 IoU 를 확인한다
    img_w, img_h = 1920, 1080
    cx, cy, nw, nh = 0.26250, 0.29167, 0.09583, 0.05000
    box = xywh_to_xyxy(cx * img_w, cy * img_h, nw * img_w, nh * img_h)
    print("YOLO 라벨 복원 박스:", tuple(round(v, 1) for v in box))
    print("정답과의 IoU      :", round(iou(gt, box), 4))

    # IoU 행렬: 예측 3개 x 정답 2개
    preds = [(412.0, 288.0, 596.0, 342.0),
             (900.0, 600.0, 1084.0, 654.0),
             (430.0, 295.0, 610.0, 350.0)]
    gts = [gt, (900.0, 600.0, 1084.0, 654.0)]
    print("\nIoU 행렬(행=예측, 열=정답)")
    for r, row in enumerate(iou_matrix(preds, gts)):
        print("  pred", r, ["%.3f" % v for v in row])


if __name__ == "__main__":
    main()
