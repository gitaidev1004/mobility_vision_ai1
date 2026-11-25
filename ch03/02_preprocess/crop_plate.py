# -*- coding: utf-8 -*-
"""검출 박스로 번호판을 잘라내고 여백(padding)을 붙인다.

실행:  python 02_preprocess/crop_plate.py
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

OUT_DIR = Path("outputs/crop")


def make_sample(width: int = 480, height: int = 270) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """번호판이 들어간 합성 장면과 정답 박스를 함께 만든다."""
    img = np.full((height, width, 3), 70, dtype=np.uint8)
    x0, y0, x1, y1 = 150, 150, 330, 200
    cv2.rectangle(img, (x0, y0), (x1, y1), (240, 240, 240), -1)
    cv2.rectangle(img, (x0, y0), (x1, y1), (25, 25, 25), 2)
    cv2.putText(img, "12 3456", (x0 + 8, y1 - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (20, 20, 20), 2, cv2.LINE_AA)
    return img, (x0, y0, x1 - x0, y1 - y0)


def crop_with_padding(
    bgr: np.ndarray,
    box: tuple[int, int, int, int],
    pad_ratio_x: float = 0.06,
    pad_ratio_y: float = 0.18,
) -> tuple[np.ndarray, dict]:
    """박스를 비율 여백만큼 키워 잘라낸다.

    왜 여백이 필요한가?
      검출기는 번호판 테두리에 거의 딱 맞는 박스를 낸다. 그런데 OCR 입력으로는
      글자 바깥에 약간의 배경이 있어야 좋다. 테두리에 글자가 닿아 있으면
      이진화와 합성곱 경계에서 첫 글자와 마지막 글자가 깎인다.
      가로보다 세로 여백을 크게 주는 이유는 검출 박스의 세로 오차가
      가로 오차보다 상대적으로 크기 때문이다(번호판이 가로로 길어서다).

    박스가 이미지를 벗어나면 clip 하지 않고 '복제 경계'로 채워 넣는다.
    clip 만 하면 번호판이 프레임 가장자리에 걸렸을 때 글자가 잘려 나간다.
    """
    h_img, w_img = bgr.shape[:2]
    x, y, w, h = [int(v) for v in box]
    pad_x = max(2, int(round(w * pad_ratio_x)))
    pad_y = max(2, int(round(h * pad_ratio_y)))

    x0, y0 = x - pad_x, y - pad_y
    x1, y1 = x + w + pad_x, y + h + pad_y

    # 이미지 밖으로 나간 양을 따로 기록해 두고 copyMakeBorder 로 메운다.
    left = max(0, -x0)
    top = max(0, -y0)
    right = max(0, x1 - w_img)
    bottom = max(0, y1 - h_img)

    xs0, ys0 = max(0, x0), max(0, y0)
    xs1, ys1 = min(w_img, x1), min(h_img, y1)
    crop = bgr[ys0:ys1, xs0:xs1]

    if any((left, top, right, bottom)):
        crop = cv2.copyMakeBorder(crop, top, bottom, left, right,
                                  cv2.BORDER_REPLICATE)

    meta = {
        "input_box": (x, y, w, h),
        "pad_px": (pad_x, pad_y),
        "padded_box": (x0, y0, x1 - x0, y1 - y0),
        "out_of_bounds": {"left": left, "top": top, "right": right, "bottom": bottom},
        "crop_shape": crop.shape[:2],
        "aspect_ratio": round(crop.shape[1] / max(crop.shape[0], 1), 3),
    }
    return crop, meta


def resize_for_ocr(crop: np.ndarray, width: int = 160, height: int = 32,
                   keep_ratio: bool = True) -> np.ndarray:
    """OCR 입력 규격(기본 32x160)으로 맞춘다.

    keep_ratio=True 면 가로세로비를 유지한 채 축소하고 남는 폭을 회색으로 채운다.
    번호판은 구형 2줄형과 신형 1줄형의 비율이 크게 달라, 비율을 무시하고 늘리면
    구형 번호판의 글자가 심하게 찌그러진다.
    """
    h, w = crop.shape[:2]
    if not keep_ratio:
        return cv2.resize(crop, (width, height), interpolation=cv2.INTER_CUBIC)

    scale = height / float(h)
    new_w = max(1, min(width, int(round(w * scale))))
    resized = cv2.resize(crop, (new_w, height), interpolation=cv2.INTER_CUBIC)
    if new_w == width:
        return resized
    canvas = np.full((height, width, crop.shape[2]), 127, dtype=np.uint8)
    canvas[:, :new_w] = resized          # 왼쪽 정렬: 번호판은 왼쪽부터 읽는다
    return canvas


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scene, box = make_sample()
    cv2.imwrite(str(OUT_DIR / "00_scene.png"), scene)

    for label, ratio in (("tight", 0.0), ("default", 0.06), ("wide", 0.20)):
        crop, meta = crop_with_padding(scene, box,
                                       pad_ratio_x=ratio,
                                       pad_ratio_y=ratio * 3 if ratio else 0.0)
        cv2.imwrite(str(OUT_DIR / f"01_crop_{label}.png"), crop)
        print(f"[{label:8s}] pad={meta['pad_px']} crop={meta['crop_shape']} "
              f"ratio={meta['aspect_ratio']}")

    crop, meta = crop_with_padding(scene, box)
    keep = resize_for_ocr(crop, keep_ratio=True)
    stretch = resize_for_ocr(crop, keep_ratio=False)
    cv2.imwrite(str(OUT_DIR / "02_resize_keep.png"), keep)
    cv2.imwrite(str(OUT_DIR / "02_resize_stretch.png"), stretch)
    print(f"\n비율 유지 리사이즈 : {keep.shape[:2]}")
    print(f"강제 늘림 리사이즈 : {stretch.shape[:2]}")

    # 번호판이 프레임 오른쪽 경계에 걸린 상황을 재현한다.
    edge_box = (scene.shape[1] - 60, box[1], box[2], box[3])
    edge_crop, edge_meta = crop_with_padding(scene, edge_box)
    cv2.imwrite(str(OUT_DIR / "03_crop_edge.png"), edge_crop)
    print(f"\n경계 초과 케이스: out_of_bounds={edge_meta['out_of_bounds']}")
    print(f"  복제 경계로 메운 결과 shape={edge_meta['crop_shape']}")
    print(f"\n저장 위치: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
