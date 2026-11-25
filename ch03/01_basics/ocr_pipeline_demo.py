# -*- coding: utf-8 -*-
"""OCR 파이프라인 4단계(검출 → 보정 → 분할 → 인식)를 단계별 중간 결과로 저장한다.

실행:  python 01_basics/ocr_pipeline_demo.py
       python 01_basics/ocr_pipeline_demo.py --image samples/car.jpg

입력 이미지를 주지 않으면 합성 번호판 이미지를 만들어 쓰므로 데이터 없이도 동작한다.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

OUT_DIR = Path("outputs/pipeline")


def make_sample_scene(width: int = 640, height: int = 360) -> np.ndarray:
    """번호판이 들어간 가짜 차량 장면을 만든다(샘플 이미지가 없을 때 사용)."""
    scene = np.full((height, width, 3), 60, dtype=np.uint8)
    cv2.rectangle(scene, (80, 120), (560, 320), (90, 95, 100), -1)   # 차체
    # 번호판: 흰 바탕 + 검은 글자(신형 비사업용 규격)
    x0, y0, x1, y1 = 240, 220, 420, 270
    cv2.rectangle(scene, (x0, y0), (x1, y1), (245, 245, 245), -1)
    cv2.rectangle(scene, (x0, y0), (x1, y1), (30, 30, 30), 2)
    cv2.putText(scene, "12 3456", (x0 + 10, y1 - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2, cv2.LINE_AA)
    # 가우시안 노이즈를 더해 실제 촬영 이미지에 가깝게 만든다.
    noise = np.random.default_rng(0).normal(0, 6, scene.shape)
    return np.clip(scene.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def stage1_detect(bgr: np.ndarray) -> tuple[int, int, int, int]:
    """1단계 — 번호판 영역 검출.

    실제 시스템은 YOLO 등 학습된 검출기를 쓴다. 여기서는 가중치 없이도 동작하도록
    '밝고 가로로 긴 사각형' 이라는 번호판의 고전적 특징으로 후보를 찾는다.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    # 번호판은 주변 차체보다 밝다 → 상위 밝기 영역만 남긴다.
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_score = -1.0
    h_img, w_img = gray.shape
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 40 or h < 12:
            continue
        ratio = w / float(h)
        # 한국 번호판 가로세로비: 신형 1변형 약 3.5, 구형 2줄형 약 2.0
        if not 1.8 <= ratio <= 6.0:
            continue
        fill = cv2.contourArea(cnt) / float(w * h)
        area_ratio = (w * h) / float(w_img * h_img)
        score = fill * 1.0 + min(area_ratio * 10.0, 1.0)
        if score > best_score:
            best_score, best = score, (x, y, w, h)

    if best is None:                      # 후보가 없으면 이미지 중앙을 돌려준다.
        return w_img // 4, h_img // 2, w_img // 2, h_img // 6
    return best


def stage2_rectify(bgr: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    """2단계 — 검출 박스를 잘라내고 OCR 입력 규격(32x160)으로 정규화한다."""
    x, y, w, h = box
    pad_x, pad_y = int(w * 0.04), int(h * 0.12)
    x0 = max(0, x - pad_x)
    y0 = max(0, y - pad_y)
    x1 = min(bgr.shape[1], x + w + pad_x)
    y1 = min(bgr.shape[0], y + h + pad_y)
    crop = bgr[y0:y1, x0:x1]
    # 가로가 긴 번호판을 확대할 때는 INTER_CUBIC 이 글자 윤곽을 덜 무너뜨린다.
    return cv2.resize(crop, (160, 32), interpolation=cv2.INTER_CUBIC)


def stage3_segment(plate_bgr: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    """3단계 — 문자 영역 분할.

    CRNN+CTC 를 쓰면 이 단계는 필요하지 않다. 다만 '어디가 글자인지' 를 눈으로
    확인하면 전처리 품질을 진단하기 쉬우므로 수직 투영(projection)으로 분할해 본다.
    """
    # 분할은 원본보다 키워서 하는 편이 유리하다. 32px 높이에서는 글자 사이 간격이
    # 1~2px 로 줄어 투영 골짜기가 사라지기 때문이다.
    big = cv2.resize(plate_bgr, (plate_bgr.shape[1] * 3, plate_bgr.shape[0] * 3),
                     interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # 번호판 테두리(검은 윤곽선)는 전경으로 잡혀 글자 구간을 모두 이어 버린다.
    # 테두리 폭만큼 안쪽을 잘라내야 투영 골짜기가 드러난다.
    mx = max(2, int(binary.shape[1] * 0.03))
    my = max(2, int(binary.shape[0] * 0.12))
    binary[:my, :] = 0
    binary[-my:, :] = 0
    binary[:, :mx] = 0
    binary[:, -mx:] = 0
    # 열별 전경 픽셀 수를 세면 글자 구간은 값이 크고 글자 사이는 0 에 가깝다.
    column_sum = (binary > 0).sum(axis=0)
    threshold = max(1, int(binary.shape[0] * 0.08))
    boxes: list[tuple[int, int, int, int]] = []
    start = None
    for col, value in enumerate(column_sum):
        if value > threshold and start is None:
            start = col
        elif value <= threshold and start is not None:
            if col - start >= 3:                     # 폭 3px 미만은 노이즈로 본다
                boxes.append((start, 0, col - start, binary.shape[0]))
            start = None
    if start is not None:
        boxes.append((start, 0, binary.shape[1] - start, binary.shape[0]))
    return binary, boxes


def stage4_recognize(boxes: list[tuple[int, int, int, int]]) -> dict:
    """4단계 — 문자 인식(여기서는 학습 가중치 없이 결과 형식만 시연한다)."""
    if not boxes:
        return {"plate_number": None, "status": "NO_PLATE", "confidence": 0.0}
    # 실제로는 CRNN 이 (T, C) 확률 행렬을 내고 CTC 디코더가 문자열을 만든다.
    # 가중치가 없으므로 계약서 예시 번호를 돌려주고 가짜임을 status 로 알린다.
    return {
        "plate_number": "12가3456",
        "status": "SUCCESS",
        "confidence": 0.0,
        "note": "학습 가중치가 없어 형식만 시연한다(3.4절에서 실제 모델을 만든다).",
        "char_count": len(boxes),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="OCR 파이프라인 단계별 중간 결과 저장")
    parser.add_argument("--image", type=str, default=None, help="입력 이미지 경로")
    args = parser.parse_args()

    if args.image and Path(args.image).exists():
        scene = cv2.imread(args.image)
        source = args.image
    else:
        if args.image:
            print(f"[안내] {args.image} 를 찾을 수 없어 합성 이미지를 사용한다.")
        scene = make_sample_scene()
        source = "(합성 이미지)"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"입력: {source}  shape={scene.shape}")
    cv2.imwrite(str(OUT_DIR / "00_input.png"), scene)

    box = stage1_detect(scene)
    marked = scene.copy()
    x, y, w, h = box
    cv2.rectangle(marked, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.imwrite(str(OUT_DIR / "01_detect.png"), marked)
    print(f"1단계 검출  : bbox=({x}, {y}, {w}, {h})  비율={w / max(h, 1):.2f}")

    plate = stage2_rectify(scene, box)
    cv2.imwrite(str(OUT_DIR / "02_rectify.png"), plate)
    print(f"2단계 보정  : {plate.shape[1]}x{plate.shape[0]} 로 정규화")

    binary, char_boxes = stage3_segment(plate)
    seg_view = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    for cx, cy, cw, ch in char_boxes:
        cv2.rectangle(seg_view, (cx, cy), (cx + cw, cy + ch - 1), (0, 160, 255), 1)
    cv2.imwrite(str(OUT_DIR / "03_segment.png"), seg_view)
    print(f"3단계 분할  : 문자 후보 {len(char_boxes)}개")

    result = stage4_recognize(char_boxes)
    print(f"4단계 인식  : {result}")
    print(f"\n중간 결과 저장 위치: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
