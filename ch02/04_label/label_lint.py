# -*- coding: utf-8 -*-
"""라벨 오류 검수(Label Lint) 도구.

학습이 안 되는 원인의 상당수는 모델이 아니라 라벨이다.
학습을 돌리기 전에 아래 항목을 기계적으로 검사해 원인을 미리 제거한다.

검사 항목
  E01 토큰 수 오류        한 줄의 값이 5개가 아니다
  E02 숫자 변환 실패      정규화 좌표 자리에 숫자가 아닌 값이 있다
  E03 좌표 범위 이탈      cx/cy/w/h 가 0~1 범위를 벗어난다
  E04 박스 경계 이탈      박스가 이미지 밖으로 나간다
  E05 0 크기 박스         폭 또는 높이가 0 에 가깝다(클릭 실수)
  E06 클래스 ID 오류      정의되지 않은 클래스 번호가 쓰였다
  E07 짝 없는 라벨        라벨 파일에 대응하는 이미지가 없다
  W01 짝 없는 이미지      이미지에 라벨 파일이 없다(배경 이미지면 정상)
  W02 극단적 종횡비       번호판으로 보기 어려운 가로세로 비율이다
  W03 과소 크기 박스      번호판 폭이 너무 작아 OCR 이 어렵다
  W04 중복 박스           같은 번호판에 박스가 두 개 이상 그려졌다

표준 라이브러리만 사용한다. 실제 이미지 크기가 필요한 검사(E04/W02/W03)는
--img-width/--img-height 로 대표 해상도를 주거나 OpenCV 가 있으면 자동 측정한다.

사용법:
    python 04_label/label_lint.py --images data/plate/images/train \
        --labels data/plate/labels/train --classes plate
"""
from __future__ import annotations

import argparse
import pathlib
import sys

EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="YOLO 라벨 검수")
    ap.add_argument("--images", default="data/plate/images/train")
    ap.add_argument("--labels", default="data/plate/labels/train")
    ap.add_argument("--classes", nargs="+", default=["plate"],
                    help="허용 클래스 이름 목록(순서가 클래스 ID 다)")
    ap.add_argument("--img-width", type=int, default=1920,
                    help="이미지 크기를 못 읽을 때 쓸 대표 가로 해상도")
    ap.add_argument("--img-height", type=int, default=1080)
    ap.add_argument("--min-plate-px", type=int, default=100,
                    help="이 폭 미만이면 W03 경고")
    ap.add_argument("--ar-min", type=float, default=1.6,
                    help="허용 최소 종횡비(픽셀 기준)")
    ap.add_argument("--ar-max", type=float, default=6.5,
                    help="허용 최대 종횡비(픽셀 기준)")
    ap.add_argument("--dup-iou", type=float, default=0.80,
                    help="이 IoU 이상이면 중복 박스로 본다")
    ap.add_argument("--max-print", type=int, default=20,
                    help="유형별로 출력할 최대 건수")
    return ap.parse_args()


def iou_norm(a, b) -> float:
    """정규화 좌표 (cx,cy,w,h) 두 박스의 IoU."""
    def xyxy(t):
        cx, cy, w, h = t
        return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
    ax1, ay1, ax2, ay2 = xyxy(a)
    bx1, by1, bx2, by2 = xyxy(b)
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    u = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / u if u > 0 else 0.0


def image_size(path: pathlib.Path, default_w: int, default_h: int):
    """이미지 실제 크기를 구한다. OpenCV 가 없으면 기본값을 쓴다."""
    try:
        import cv2                           # noqa: PLC0415
    except ImportError:
        return default_w, default_h, False
    img = cv2.imread(str(path))
    if img is None:
        return default_w, default_h, False
    return img.shape[1], img.shape[0], True


def main() -> int:
    args = parse_args()
    img_dir = pathlib.Path(args.images)
    lbl_dir = pathlib.Path(args.labels)
    n_classes = len(args.classes)

    if not lbl_dir.is_dir():
        print(f"[안내] 라벨 디렉터리가 없다: {lbl_dir}")
        print("       먼저 2.4절의 라벨링 작업을 수행한다.")
        return 0

    img_files = {p.stem: p for p in img_dir.rglob("*")
                 if p.suffix.lower() in EXTS} if img_dir.is_dir() else {}
    lbl_files = {p.stem: p for p in lbl_dir.rglob("*.txt")}

    findings: dict[str, list[str]] = {}

    def add(code: str, msg: str) -> None:
        findings.setdefault(code, []).append(msg)

    # ---- 짝 검사 ----------------------------------------------------------
    for stem in sorted(set(lbl_files) - set(img_files)):
        add("E07", f"{lbl_files[stem].name}: 대응 이미지가 없다")
    for stem in sorted(set(img_files) - set(lbl_files)):
        add("W01", f"{img_files[stem].name}: 라벨 파일이 없다(배경이면 빈 .txt 생성)")

    # ---- 라벨 내용 검사 ---------------------------------------------------
    total_boxes = 0
    size_cache: dict[str, tuple[int, int, bool]] = {}
    for stem in sorted(lbl_files):
        lp = lbl_files[stem]
        boxes: list[tuple[float, float, float, float]] = []
        for ln, line in enumerate(lp.read_text(encoding="utf-8").splitlines(), 1):
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            if len(parts) != 5:
                add("E01", f"{lp.name}:{ln} 토큰 {len(parts)}개 -> {s!r}")
                continue
            try:
                cid = int(parts[0])
                cx, cy, w, h = (float(v) for v in parts[1:])
            except ValueError:
                add("E02", f"{lp.name}:{ln} 숫자 변환 실패 -> {s!r}")
                continue

            total_boxes += 1

            if not 0 <= cid < n_classes:
                add("E06", f"{lp.name}:{ln} 클래스 ID {cid} "
                           f"(허용 0~{n_classes - 1})")

            vals = {"cx": cx, "cy": cy, "w": w, "h": h}
            bad = [k for k, v in vals.items() if not 0.0 <= v <= 1.0]
            if bad:
                add("E03", f"{lp.name}:{ln} 범위 이탈 {bad} -> {s!r}")
                continue

            # 박스가 이미지 경계를 넘어가는지 확인한다
            if cx - w / 2 < -1e-6 or cx + w / 2 > 1 + 1e-6 \
               or cy - h / 2 < -1e-6 or cy + h / 2 > 1 + 1e-6:
                add("E04", f"{lp.name}:{ln} 박스가 이미지 밖으로 나간다 -> {s!r}")

            if w < 1e-4 or h < 1e-4:
                add("E05", f"{lp.name}:{ln} 0 크기 박스 w={w} h={h}")
                continue

            # 실제 이미지 크기를 알면 픽셀 기준으로 종횡비/크기를 본다
            if stem in img_files:
                if stem not in size_cache:
                    size_cache[stem] = image_size(
                        img_files[stem], args.img_width, args.img_height)
                iw, ih, _real = size_cache[stem]
            else:
                iw, ih = args.img_width, args.img_height
            pw, ph = w * iw, h * ih
            ar = pw / ph if ph > 0 else 0.0
            if not args.ar_min <= ar <= args.ar_max:
                add("W02", f"{lp.name}:{ln} 종횡비 {ar:.2f} "
                           f"(허용 {args.ar_min}~{args.ar_max}) {pw:.0f}x{ph:.0f}px")
            if pw < args.min_plate_px:
                add("W03", f"{lp.name}:{ln} 번호판 폭 {pw:.0f}px "
                           f"(권장 {args.min_plate_px}px 이상)")

            boxes.append((cx, cy, w, h))

        # 같은 파일 안에서 거의 동일한 박스가 두 번 그려졌는지 본다
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                v = iou_norm(boxes[i], boxes[j])
                if v >= args.dup_iou:
                    add("W04", f"{lp.name} 유효박스 {i}/{j} 중복 IoU={v:.3f}")

    # ---- 보고 ------------------------------------------------------------
    labels = {
        "E01": "토큰 수 오류", "E02": "숫자 변환 실패", "E03": "좌표 범위 이탈",
        "E04": "박스 경계 이탈", "E05": "0 크기 박스", "E06": "클래스 ID 오류",
        "E07": "짝 없는 라벨", "W01": "짝 없는 이미지", "W02": "극단적 종횡비",
        "W03": "과소 크기 박스", "W04": "중복 박스",
    }
    print("=== 라벨 검수 결과 ===")
    print(f"  이미지 {len(img_files)} 장 / 라벨 {len(lbl_files)} 개 / 박스 {total_boxes} 개")
    print("-" * 62)
    print(f"{'코드':6s}{'항목':16s}{'건수':>8s}  판정")
    print("-" * 62)
    n_err = n_warn = 0
    for code in sorted(labels):
        c = len(findings.get(code, []))
        if code.startswith("E"):
            n_err += c
        else:
            n_warn += c
        verdict = "통과" if c == 0 else ("수정 필요" if code[0] == "E" else "확인 권장")
        print(f"{code:6s}{labels[code]:16s}{c:8d}  {verdict}")
    print("-" * 62)
    print(f"  오류 {n_err} 건 / 경고 {n_warn} 건")

    for code in sorted(findings):
        items = findings[code]
        print(f"\n[{code}] {labels[code]} ({len(items)} 건)")
        for m in items[:args.max_print]:
            print("   -", m)
        if len(items) > args.max_print:
            print(f"   ... 외 {len(items) - args.max_print} 건")

    if n_err:
        print("\n오류가 있는 상태로 학습하면 성능이 재현되지 않는다. 먼저 수정한다.")
        return 1
    print("\n오류 없음. 학습을 진행할 수 있다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
