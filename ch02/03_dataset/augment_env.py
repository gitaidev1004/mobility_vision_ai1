# -*- coding: utf-8 -*-
"""촬영 환경을 모사하는 데이터 증강(Data Augmentation) 실습.

야간 저조도, 역광, 비, 모션 블러, 기울기를 OpenCV 로 합성해
부족한 환경의 학습 데이터를 보강한다.

주의: 증강은 실제 촬영 데이터를 완전히 대체하지 못한다.
야간 센서 노이즈나 적외선 조명 특성은 합성으로 재현하기 어렵다.
증강 데이터는 '보조'로 쓰고 환경별 테스트셋은 반드시 실제 촬영으로 확보한다.

사용법:
    python 03_dataset/augment_env.py --input data/plate/images/train \
        --labels data/plate/labels/train --out data/aug --per-image 3
"""
from __future__ import annotations

import argparse
import pathlib
import random
import sys

EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="환경별 데이터 증강")
    ap.add_argument("--input", default="data/plate/images/train")
    ap.add_argument("--labels", default="data/plate/labels/train")
    ap.add_argument("--out", default="data/aug")
    ap.add_argument("--per-image", type=int, default=3,
                    help="이미지 1장당 생성할 증강 수")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--demo", action="store_true",
                    help="입력이 없을 때 합성 번호판 이미지로 시연한다")
    return ap.parse_args()


def make_night(img, np, cv2, gain: float = 0.35, noise_sigma: float = 8.0):
    """야간 저조도를 모사한다. 밝기를 낮추고 가우시안 노이즈를 더한다."""
    dark = img.astype(np.float32) * gain
    noise = np.random.normal(0.0, noise_sigma, dark.shape)
    out = np.clip(dark + noise, 0, 255).astype(np.uint8)
    # 야간 영상은 보통 채도가 낮아지므로 S 채널을 줄인다
    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] *= 0.6
    return cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)


def make_backlight(img, np, cv2, strength: float = 0.75):
    """역광(Backlight)을 모사한다. 상단에 밝은 광원 그라데이션을 겹친다."""
    h, w = img.shape[:2]
    # 세로 방향으로 위쪽이 밝은 마스크를 만든다
    ramp = np.linspace(1.0, 0.0, h, dtype=np.float32) ** 2
    mask = np.repeat(ramp[:, None], w, axis=1)[..., None]
    glare = (mask * 255.0 * strength)
    out = img.astype(np.float32) * (1.0 - mask * 0.35) + glare
    return np.clip(out, 0, 255).astype(np.uint8)


def make_rain(img, np, cv2, drops: int = 700, length: int = 18):
    """빗줄기를 모사한다. 가는 흰 선을 무작위로 그리고 블러로 번지게 한다."""
    layer = np.zeros(img.shape[:2], dtype=np.uint8)
    h, w = img.shape[:2]
    for _ in range(drops):
        x = random.randint(0, w - 1)
        y = random.randint(0, h - 1)
        dx = random.randint(-3, 3)
        cv2.line(layer, (x, y), (x + dx, min(h - 1, y + length)), 255, 1)
    layer = cv2.GaussianBlur(layer, (3, 7), 0)
    rain = cv2.cvtColor(layer, cv2.COLOR_GRAY2BGR)
    out = cv2.addWeighted(img, 0.85, rain, 0.35, 0)
    # 비 오는 날은 대비가 낮아지므로 전체 콘트라스트를 줄인다
    return np.clip(out.astype(np.float32) * 0.9 + 12.0, 0, 255).astype(np.uint8)


def make_motion_blur(img, cv2, np, ksize: int = 9, horizontal: bool = True):
    """주행 중 촬영으로 생기는 모션 블러를 모사한다."""
    k = np.zeros((ksize, ksize), dtype=np.float32)
    if horizontal:
        k[ksize // 2, :] = 1.0 / ksize       # 가로 방향 평균 필터
    else:
        k[:, ksize // 2] = 1.0 / ksize
    return cv2.filter2D(img, -1, k)


def rotate_with_boxes(img, boxes, angle, cv2, np):
    """이미지를 회전하고 YOLO 정규화 박스도 함께 회전시킨다.

    박스 네 꼭짓점을 회전한 뒤 그 외접 직사각형을 새 박스로 삼는다.
    기울어진 번호판 데이터를 만들 때 라벨을 함께 갱신해야 한다.
    """
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    out = cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REPLICATE)

    new_boxes = []
    for cls, cx, cy, bw, bh in boxes:
        px, py = cx * w, cy * h
        pw, ph = bw * w, bh * h
        corners = np.array([
            [px - pw / 2, py - ph / 2, 1.0],
            [px + pw / 2, py - ph / 2, 1.0],
            [px + pw / 2, py + ph / 2, 1.0],
            [px - pw / 2, py + ph / 2, 1.0],
        ], dtype=np.float32)
        pts = corners @ m.T                       # 2x3 회전행렬 적용
        x1, y1 = pts[:, 0].min(), pts[:, 1].min()
        x2, y2 = pts[:, 0].max(), pts[:, 1].max()
        # 화면을 벗어난 부분은 잘라낸다
        x1, y1 = max(0.0, x1), max(0.0, y1)
        x2, y2 = min(float(w), x2), min(float(h), y2)
        if x2 - x1 < 4 or y2 - y1 < 4:
            continue                              # 너무 작아진 박스는 버린다
        new_boxes.append((cls,
                          (x1 + x2) / 2 / w, (y1 + y2) / 2 / h,
                          (x2 - x1) / w, (y2 - y1) / h))
    return out, new_boxes


def read_yolo(path: pathlib.Path):
    """YOLO 라벨을 (cls, cx, cy, w, h) 튜플 목록으로 읽는다."""
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        p = line.split()
        if len(p) >= 5:
            rows.append((int(p[0]), float(p[1]), float(p[2]),
                         float(p[3]), float(p[4])))
    return rows


def write_yolo(path: pathlib.Path, boxes) -> None:
    """YOLO 라벨을 소수점 6자리로 저장한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n"
                for c, cx, cy, w, h in boxes), encoding="utf-8")


def synth_plate_image(cv2, np):
    """입력 데이터가 없을 때 쓰는 합성 차량 이미지(회색 배경 + 흰 번호판)."""
    img = np.full((480, 640, 3), 92, dtype=np.uint8)
    cv2.rectangle(img, (0, 330), (640, 480), (70, 70, 72), -1)   # 노면
    cv2.rectangle(img, (180, 180), (470, 340), (54, 58, 66), -1)  # 차량 앞면
    cv2.rectangle(img, (248, 268), (404, 312), (236, 238, 240), -1)  # 번호판
    cv2.rectangle(img, (248, 268), (404, 312), (30, 30, 30), 2)
    cv2.putText(img, "12 3456", (256, 302),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2)
    # 번호판 박스의 YOLO 라벨 (클래스 0 = plate)
    boxes = [(0, (248 + 404) / 2 / 640, (268 + 312) / 2 / 480,
              (404 - 248) / 640, (312 - 268) / 480)]
    return img, boxes


def main() -> int:
    args = parse_args()
    try:
        import cv2                             # noqa: PLC0415
        import numpy as np                     # noqa: PLC0415
    except ImportError:
        print("[안내] OpenCV/NumPy 가 필요하다.")
        print("       pip install opencv-python-headless==4.12.0.88 numpy==2.1.3")
        return 0

    random.seed(args.seed)
    np.random.seed(args.seed)

    out_img = pathlib.Path(args.out) / "images"
    out_lbl = pathlib.Path(args.out) / "labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    in_dir = pathlib.Path(args.input)
    items: list[tuple[str, object, list]] = []
    if in_dir.is_dir():
        files = sorted(p for p in in_dir.rglob("*") if p.suffix.lower() in EXTS)
        for f in files:
            img = cv2.imread(str(f))
            if img is None:
                continue
            items.append((f.stem, img, read_yolo(pathlib.Path(args.labels) / (f.stem + ".txt"))))
    if not items:
        print(f"[안내] 입력 이미지가 없다: {in_dir}")
        print("       합성 번호판 이미지 1장으로 증강 동작을 시연한다.")
        img, boxes = synth_plate_image(cv2, np)
        items = [("synth_demo", img, boxes)]
        cv2.imwrite(str(out_img / "synth_demo_origin.jpg"), img)

    recipes = ["night", "backlight", "rain", "blur", "tilt"]
    made: dict[str, int] = {r: 0 for r in recipes}

    for stem, img, boxes in items:
        picks = random.sample(recipes, k=min(args.per_image, len(recipes)))
        for r in picks:
            new_boxes = list(boxes)
            if r == "night":
                aug = make_night(img, np, cv2, gain=random.uniform(0.25, 0.45))
            elif r == "backlight":
                aug = make_backlight(img, np, cv2, strength=random.uniform(0.6, 0.9))
            elif r == "rain":
                aug = make_rain(img, np, cv2, drops=random.randint(400, 900))
            elif r == "blur":
                aug = make_motion_blur(img, cv2, np, ksize=random.choice([7, 9, 13]))
            else:  # tilt — 라벨도 함께 회전한다
                ang = random.choice([-15, -10, -6, 6, 10, 15])
                aug, new_boxes = rotate_with_boxes(img, boxes, ang, cv2, np)
            name = f"{stem}_{r}"
            cv2.imwrite(str(out_img / f"{name}.jpg"), aug)
            write_yolo(out_lbl / f"{name}.txt", new_boxes)
            made[r] += 1
            gray = cv2.cvtColor(aug, cv2.COLOR_BGR2GRAY)
            print(f"  {name:28s} 평균밝기={float(np.mean(gray)):6.1f} "
                  f"선명도={float(cv2.Laplacian(gray, cv2.CV_64F).var()):8.1f} "
                  f"박스={len(new_boxes)}")

    print(f"\n=== 증강 생성 결과 (원본 {len(items)} 장) ===")
    for r in recipes:
        print(f"  {r:10s} {made[r]:4d} 장")
    print(f"총 {sum(made.values())} 장 생성  →  {out_img}")
    print("※ 증강 데이터는 Train 에만 넣는다. Val/Test 에 섞으면 평가가 왜곡된다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
