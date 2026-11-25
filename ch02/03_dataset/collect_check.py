# -*- coding: utf-8 -*-
"""수집한 번호판 이미지의 품질 통계를 집계한다.

측정 항목
  - 해상도 분포(가로x세로)
  - 평균 밝기(0~255) 와 주간/야간 추정
  - 흐림 정도(라플라시안 분산)
  - YOLO 라벨이 함께 있으면 번호판 픽셀 폭 분포

OpenCV 가 없으면 설치 안내만 출력하고 정상 종료한다.

사용법:
    python 03_dataset/collect_check.py --images data/raw/images --labels data/raw/labels
"""
from __future__ import annotations

import argparse
import pathlib
import sys

EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="수집 이미지 품질 통계")
    ap.add_argument("--images", default="data/raw/images", help="이미지 디렉터리")
    ap.add_argument("--labels", default="data/raw/labels",
                    help="YOLO 라벨 디렉터리(없으면 생략)")
    ap.add_argument("--dark-thr", type=float, default=70.0,
                    help="이 밝기 미만을 야간 후보로 본다")
    ap.add_argument("--blur-thr", type=float, default=100.0,
                    help="라플라시안 분산이 이 값 미만이면 흐림 후보")
    ap.add_argument("--min-plate-px", type=int, default=120,
                    help="OCR 이 안정적으로 동작하는 번호판 최소 픽셀 폭")
    return ap.parse_args()


def percentile(values: list[float], q: float) -> float:
    """정렬된 분포에서 q 분위수(0~100)를 선형 보간으로 구한다."""
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    pos = (len(s) - 1) * (q / 100.0)
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    frac = pos - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def summarize(name: str, values: list[float], unit: str = "") -> None:
    """최소/p25/중앙/p75/최대/평균을 한 줄로 요약 출력한다."""
    if not values:
        print(f"  {name:14s} (데이터 없음)")
        return
    print(f"  {name:14s} min={min(values):8.1f}{unit} p25={percentile(values,25):8.1f}{unit} "
          f"med={percentile(values,50):8.1f}{unit} p75={percentile(values,75):8.1f}{unit} "
          f"max={max(values):8.1f}{unit} mean={sum(values)/len(values):8.1f}{unit}")


def read_label_widths(label_path: pathlib.Path, img_w: int) -> list[float]:
    """YOLO 라벨 파일에서 번호판 박스의 픽셀 폭 목록을 뽑는다."""
    out: list[float] = []
    if not label_path.exists():
        return out
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            nw = float(parts[3])          # 정규화된 폭(0~1)
        except ValueError:
            continue
        out.append(nw * img_w)
    return out


def main() -> int:
    args = parse_args()
    try:
        import cv2                            # noqa: PLC0415
        import numpy as np                    # noqa: PLC0415
    except ImportError:
        print("[안내] OpenCV/NumPy 가 필요하다.")
        print("       pip install opencv-python-headless==4.12.0.88 numpy==2.1.3")
        return 0

    img_dir = pathlib.Path(args.images)
    lbl_dir = pathlib.Path(args.labels)
    if not img_dir.is_dir():
        print(f"[안내] 이미지 디렉터리가 없다: {img_dir}")
        print("       data/raw/images 아래에 수집한 차량 사진을 넣고 다시 실행한다.")
        return 0

    files = sorted(p for p in img_dir.rglob("*") if p.suffix.lower() in EXTS)
    if not files:
        print(f"[안내] {img_dir} 에 이미지가 없다.")
        return 0

    widths: list[float] = []
    heights: list[float] = []
    brights: list[float] = []
    blurs: list[float] = []
    plate_px: list[float] = []
    dark_list: list[str] = []
    blur_list: list[str] = []
    small_list: list[str] = []
    res_count: dict[str, int] = {}
    broken = 0

    for f in files:
        img = cv2.imread(str(f))
        if img is None:
            broken += 1
            if broken <= 5:        # 경고가 쏟아지지 않도록 앞 5건만 보여준다
                print(f"  [경고] 읽기 실패(손상 또는 한글 경로): {f.name}")
            continue
        h, w = img.shape[:2]
        widths.append(float(w))
        heights.append(float(h))
        res_count[f"{w}x{h}"] = res_count.get(f"{w}x{h}", 0) + 1

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        b = float(np.mean(gray))              # 평균 밝기로 주간/야간을 추정한다
        brights.append(b)
        if b < args.dark_thr:
            dark_list.append(f.name)

        # 라플라시안 분산이 작을수록 에지가 약하다 = 흐린 사진이다
        lap = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        blurs.append(lap)
        if lap < args.blur_thr:
            blur_list.append(f.name)

        pw = read_label_widths(lbl_dir / (f.stem + ".txt"), w)
        plate_px.extend(pw)
        if pw and max(pw) < args.min_plate_px:
            small_list.append(f.name)

    n = len(brights)
    if broken:
        print(f"  [경고] 읽기 실패 총 {broken} 건")
    if n == 0:
        print("[안내] 읽을 수 있는 이미지가 없다. 파일 형식과 경로를 확인한다.")
        return 0
    print(f"=== 수집 이미지 통계 ({n} 장) ===")
    summarize("가로(px)", widths)
    summarize("세로(px)", heights)
    summarize("평균밝기", brights)
    summarize("선명도(lap)", blurs)
    summarize("번호판폭(px)", plate_px)

    print("\n=== 해상도 분포 (상위 5개) ===")
    for res, c in sorted(res_count.items(), key=lambda x: -x[1])[:5]:
        print(f"  {res:12s} {c:4d} 장  ({c / n * 100:5.1f}%)")

    print("\n=== 재촬영/보강 후보 ===")
    print(f"  야간(저조도) 후보 : {len(dark_list):4d} 장  {dark_list[:3]}")
    print(f"  흐림 후보         : {len(blur_list):4d} 장  {blur_list[:3]}")
    print(f"  소형 번호판 후보  : {len(small_list):4d} 장  {small_list[:3]}")

    day = n - len(dark_list)
    print("\n=== 환경 균형 점검 ===")
    print(f"  주간 추정 {day} 장 / 야간 추정 {len(dark_list)} 장")
    if n and len(dark_list) / n < 0.20:
        print("  판정: 야간 비중이 20% 미만이다. 야간 촬영을 추가 수집한다.")
    else:
        print("  판정: 주간/야간 비중이 비교적 균형 있다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
