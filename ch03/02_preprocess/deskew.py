# -*- coding: utf-8 -*-
"""번호판 기울기 보정 — 최소 외접 사각형 방식과 허프 변환 방식을 비교한다.

실행:  python 02_preprocess/deskew.py
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

OUT_DIR = Path("outputs/deskew")


def make_rotated_sample(angle_deg: float = -7.5,
                        width: int = 320, height: int = 120) -> np.ndarray:
    """지정 각도로 기울어진 번호판 이미지를 합성한다."""
    plate = np.full((height, width, 3), 60, dtype=np.uint8)
    x0, y0, x1, y1 = 40, 38, 280, 82
    cv2.rectangle(plate, (x0, y0), (x1, y1), (238, 238, 238), -1)
    cv2.rectangle(plate, (x0, y0), (x1, y1), (25, 25, 25), 2)
    cv2.putText(plate, "12 3456", (x0 + 10, y1 - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2, cv2.LINE_AA)
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    return cv2.warpAffine(plate, matrix, (width, height),
                          flags=cv2.INTER_CUBIC, borderValue=(60, 60, 60))


def _foreground_mask(bgr: np.ndarray) -> np.ndarray:
    """번호판 전경(밝은 판 영역)을 이진 마스크로 만든다."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def angle_by_min_area_rect(bgr: np.ndarray) -> float | None:
    """최소 외접 사각형(minAreaRect)으로 기울기를 추정한다.

    번호판 전체가 하나의 덩어리로 잡히면 가장 간단하고 빠른 방법이다.
    주의할 점: minAreaRect 가 돌려주는 (w, h) 는 '사각형의 긴 변' 을 보장하지 않는다.
    w < h 로 나오면 사각형의 기준축이 90도 돌아간 상태이므로 각도에 90 을 더해
    환산해야 한다. 이 보정을 빼먹으면 -78도를 그대로 적용해 번호판을 세로로 세운다.
    마지막으로 [-45, 45] 범위로 정규화한다 — 번호판이 90도 누운 경우는 없기 때문이다.
    """
    mask = _foreground_mask(bgr)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 200:
        return None
    (_, _), (w, h), angle = cv2.minAreaRect(cnt)
    if w < h:                      # 기준축이 90도 돌아간 상태 → 환산한다
        angle += 90.0
    # [-45, 45] 로 정규화한다.
    while angle > 45.0:
        angle -= 90.0
    while angle < -45.0:
        angle += 90.0
    # 반환값의 규약을 허프 추정과 통일한다: '이미지를 이만큼 회전시키면 수평이 된다'.
    return float(angle)


def angle_by_hough(bgr: np.ndarray, max_abs_deg: float = 30.0) -> float | None:
    """허프 변환으로 번호판 수평 테두리 직선을 찾아 기울기를 추정한다.

    minAreaRect 는 번호판 외곽이 다른 물체(범퍼 그림자 등)와 붙으면 각도가 크게 틀린다.
    허프 변환은 '길고 직선인 테두리' 만 보므로 그런 상황에 더 강하다.
    여러 직선이 나오면 평균이 아니라 중앙값을 쓴다. 평균은 한 개의 이상치
    (대각선 그림자 등)에 쉽게 끌려간다.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
    min_len = max(20, int(bgr.shape[1] * 0.3))
    lines = cv2.HoughLinesP(edges, 1, np.pi / 360.0, threshold=60,
                            minLineLength=min_len, maxLineGap=10)
    if lines is None:
        return None

    # OpenCV 버전에 따라 반환 shape 이 (N, 1, 4) 또는 (N, 4) 다. 둘 다 받아 준다.
    segments = np.asarray(lines).reshape(-1, 4)

    angles: list[float] = []
    for x1, y1, x2, y2 in segments:
        if x2 == x1:
            continue
        deg = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        # 수직선(세로 테두리)은 제외하고 수평에 가까운 직선만 모은다.
        if abs(deg) <= max_abs_deg:
            angles.append(deg)
    if not angles:
        return None
    return float(np.median(angles))


def rotate(bgr: np.ndarray, angle_deg: float) -> np.ndarray:
    """이미지를 각도만큼 역회전해 수평으로 맞춘다.

    borderMode 를 REPLICATE 로 두는 이유: 기본값(검은색)으로 두면 네 모퉁이에
    검은 삼각형이 생기고, 그 경계가 이진화에서 글자처럼 잡힌다.
    """
    h, w = bgr.shape[:2]
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    return cv2.warpAffine(bgr, matrix, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def deskew(bgr: np.ndarray, max_abs_deg: float = 30.0) -> tuple[np.ndarray, dict]:
    """두 방법으로 각도를 추정하고 더 신뢰할 수 있는 값으로 보정한다.

    선택 규칙
      1. 두 방법의 추정값이 3도 이내로 일치하면 평균을 쓴다(교차 검증 성공).
      2. 불일치하면 허프 추정값을 쓴다. 테두리 직선이 번호판의 물리적 기준이기 때문이다.
      3. 둘 다 실패하거나 추정값이 임계치를 넘으면 보정하지 않는다.
         과한 회전은 안 하는 것보다 나쁘다 — 글자가 더 뭉개진다.
    """
    a_rect = angle_by_min_area_rect(bgr)
    a_hough = angle_by_hough(bgr, max_abs_deg=max_abs_deg)
    info = {"min_area_rect_deg": a_rect, "hough_deg": a_hough}

    chosen: float | None
    if a_rect is not None and a_hough is not None and abs(a_rect - a_hough) <= 3.0:
        chosen = (a_rect + a_hough) / 2.0
        info["method"] = "agree(avg)"
    elif a_hough is not None:
        chosen = a_hough
        info["method"] = "hough"
    elif a_rect is not None:
        chosen = a_rect
        info["method"] = "min_area_rect"
    else:
        chosen = None
        info["method"] = "none"

    if chosen is None or abs(chosen) > max_abs_deg or abs(chosen) < 0.3:
        info["applied_deg"] = 0.0
        info["skipped"] = True
        return bgr.copy(), info

    info["applied_deg"] = round(chosen, 3)
    info["skipped"] = False
    return rotate(bgr, chosen), info


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"{'정답각':>8s} {'minAreaRect':>13s} {'Hough':>9s} "
          f"{'선택방식':>14s} {'적용각':>8s} {'잔여오차':>9s}")
    print("-" * 70)

    for truth in (-12.0, -7.5, -3.0, 0.0, 4.5, 9.0):
        sample = make_rotated_sample(truth)
        fixed, info = deskew(sample)
        tag = f"{truth:+.1f}".replace("+", "p").replace("-", "m").replace(".", "_")
        cv2.imwrite(str(OUT_DIR / f"in_{tag}.png"), sample)
        cv2.imwrite(str(OUT_DIR / f"out_{tag}.png"), fixed)

        rect = f"{info['min_area_rect_deg']:+.2f}" if info["min_area_rect_deg"] is not None else "   -"
        hough = f"{info['hough_deg']:+.2f}" if info["hough_deg"] is not None else "   -"
        residual = truth + info["applied_deg"]
        print(f"{truth:+8.1f} {rect:>13s} {hough:>9s} "
              f"{info['method']:>14s} {info['applied_deg']:+8.2f} {residual:+9.2f}")

    print("\n해석: 잔여오차가 1도 이내면 OCR 정확도에 미치는 영향은 거의 없다.")
    print("      3도를 넘으면 글자 상단과 하단이 서로 다른 행으로 섞여 들어가 치명적이다.")
    print(f"\n저장 위치: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
