# -*- coding: utf-8 -*-
"""번호판 4점을 추정해 투시 변환(getPerspectiveTransform)으로 정면화한다.

실행:  python 02_preprocess/perspective.py
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

OUT_DIR = Path("outputs/perspective")

#: 번호판 정면 기준 크기(신형 1줄형 520x110mm 의 비율 약 4.7 을 반영).
TARGET_W, TARGET_H = 220, 48


def make_skewed_sample(width: int = 400, height: int = 240) -> np.ndarray:
    """비스듬히 촬영된 번호판 이미지를 합성한다(측면 진입 차량 상황)."""
    plate = np.full((TARGET_H, TARGET_W, 3), 240, dtype=np.uint8)
    cv2.rectangle(plate, (0, 0), (TARGET_W - 1, TARGET_H - 1), (25, 25, 25), 2)
    cv2.putText(plate, "12 3456", (12, TARGET_H - 13),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2, cv2.LINE_AA)

    # 원본 네 꼭짓점을 비스듬한 사변형으로 보낸다(오른쪽이 멀어 작게 보이는 구도).
    src = np.float32([[0, 0], [TARGET_W, 0], [TARGET_W, TARGET_H], [0, TARGET_H]])
    dst = np.float32([[70, 70], [330, 100], [320, 168], [60, 150]])
    matrix = cv2.getPerspectiveTransform(src, dst)
    scene = np.full((height, width, 3), 65, dtype=np.uint8)
    warped = cv2.warpPerspective(plate, matrix, (width, height),
                                 borderMode=cv2.BORDER_TRANSPARENT,
                                 dst=scene.copy())
    return warped


def order_corners(points: np.ndarray) -> np.ndarray:
    """네 점을 좌상·우상·우하·좌하 순서로 정렬한다.

    왜 정렬이 필요한가?
      findContours 나 goodFeaturesToTrack 이 주는 점의 순서는 보장되지 않는다.
      getPerspectiveTransform 은 src 와 dst 의 i 번째 점이 서로 대응한다고 보므로,
      순서가 틀리면 결과가 뒤집히거나 X 자로 꼬인다.

    판별 방법: x+y 가 최소면 좌상, 최대면 우하. x-y 가 최소면 좌하, 최대면 우상.
    이 방식은 회전이 45도를 넘지 않는 한 안정적으로 동작한다.
    """
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if pts.shape[0] != 4:
        raise ValueError(f"4개 점이 필요하다. 받은 개수={pts.shape[0]}")
    s = pts.sum(axis=1)
    d = pts[:, 0] - pts[:, 1]
    return np.float32([
        pts[int(np.argmin(s))],   # 좌상
        pts[int(np.argmax(d))],   # 우상
        pts[int(np.argmax(s))],   # 우하
        pts[int(np.argmin(d))],   # 좌하
    ])


def find_plate_quad(bgr: np.ndarray) -> np.ndarray | None:
    """번호판 윤곽에서 꼭짓점 4개를 추정한다.

    절차
      1. 회색조 → 블러 → Canny 로 에지를 뽑는다.
      2. 외곽 윤곽 중 면적이 큰 것부터 approxPolyDP 로 다각형 근사한다.
      3. 꼭짓점이 4개이고 볼록(convex)하며 가로세로비가 번호판 범위면 채택한다.

    approxPolyDP 의 epsilon 을 둘레의 2% 로 두는 것이 경험적으로 안정적이다.
    너무 작으면 꼭짓점이 5~8개로 나오고, 너무 크면 삼각형으로 뭉개진다.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 140)
    # 끊긴 에지를 이어 붙여 닫힌 윤곽을 만든다.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        quad = order_corners(approx)
        top_w = np.linalg.norm(quad[1] - quad[0])
        left_h = np.linalg.norm(quad[3] - quad[0])
        if left_h < 8 or top_w < 30:
            continue
        ratio = top_w / left_h
        if 1.6 <= ratio <= 7.0:
            return quad
    return None


def warp_plate(bgr: np.ndarray, quad: np.ndarray,
               out_w: int = TARGET_W, out_h: int = TARGET_H) -> np.ndarray:
    """추정한 사변형을 직사각형으로 펴낸다(정면화).

    dst 를 고정 크기로 두는 이유: OCR 모델 입력 규격이 고정이므로, 여기서
    바로 목표 비율에 맞춰 펴면 뒤에서 한 번 더 리사이즈할 필요가 없다.
    단계를 줄이면 보간이 한 번만 일어나 글자 윤곽이 덜 뭉개진다.
    """
    dst = np.float32([[0, 0], [out_w - 1, 0],
                      [out_w - 1, out_h - 1], [0, out_h - 1]])
    matrix = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    return cv2.warpPerspective(bgr, matrix, (out_w, out_h),
                               flags=cv2.INTER_CUBIC)


def fallback_warp(bgr: np.ndarray, box: tuple[int, int, int, int],
                  out_w: int = TARGET_W, out_h: int = TARGET_H) -> np.ndarray:
    """4점 추정이 실패했을 때 쓰는 대체 경로: 축 정렬 박스를 그냥 리사이즈한다.

    투시 왜곡이 심하지 않은 정면 진입 이미지에서는 이 정도로도 충분하다.
    실패 시 예외를 던지는 대신 품질이 낮은 결과라도 돌려주는 편이 운영에 안전하다.
    """
    x, y, w, h = box
    crop = bgr[max(0, y):y + h, max(0, x):x + w]
    if crop.size == 0:
        return np.zeros((out_h, out_w, 3), dtype=np.uint8)
    return cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_CUBIC)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scene = make_skewed_sample()
    cv2.imwrite(str(OUT_DIR / "00_skewed.png"), scene)
    print(f"입력 shape = {scene.shape}")

    quad = find_plate_quad(scene)
    if quad is None:
        print("4점 추정 실패 → 대체 경로(축 정렬 리사이즈)를 사용한다.")
        flat = fallback_warp(scene, (60, 70, 270, 100))
    else:
        print("추정된 꼭짓점 (좌상, 우상, 우하, 좌하):")
        for name, pt in zip(("좌상", "우상", "우하", "좌하"), quad):
            print(f"  {name}: ({pt[0]:.1f}, {pt[1]:.1f})")
        left_h = float(np.linalg.norm(quad[3] - quad[0]))
        right_h = float(np.linalg.norm(quad[2] - quad[1]))
        # 좌우 변 길이가 다른 것이 투시 왜곡의 직접 증거다. 카메라에서 먼 쪽이 짧다.
        print(f"  좌변 높이 {left_h:.1f}px, 우변 높이 {right_h:.1f}px "
              f"→ 투시 왜곡 {abs(left_h - right_h) / max(left_h, 1) * 100:.1f}%")
        tilt = np.degrees(np.arctan2(quad[1][1] - quad[0][1],
                                     quad[1][0] - quad[0][0]))
        print(f"  위변 기울기 {tilt:+.1f}도")

        marked = scene.copy()
        cv2.polylines(marked, [quad.astype(np.int32)], True, (0, 0, 255), 2)
        for pt in quad:
            cv2.circle(marked, (int(pt[0]), int(pt[1])), 4, (0, 255, 255), -1)
        cv2.imwrite(str(OUT_DIR / "01_quad.png"), marked)
        flat = warp_plate(scene, quad)

    cv2.imwrite(str(OUT_DIR / "02_warped.png"), flat)
    print(f"\n정면화 결과 shape = {flat.shape} (목표 {TARGET_H}x{TARGET_W})")
    print(f"저장 위치: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
