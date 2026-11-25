# -*- coding: utf-8 -*-
"""반자동 라벨링(Semi-automatic Labeling) 스크립트.

전략
  1) 학습된 번호판 가중치가 있으면 그 모델로 사전 라벨(pre-label)을 만든다.
  2) 가중치가 없으면 고전 영상처리(에지 + 윤곽선 + 종횡비 필터)로 후보를
     생성한다. 정확도는 낮지만 사람이 박스를 처음부터 그리는 수고를 줄인다.
  3) 어느 경로든 결과는 '초안'이다. 반드시 사람이 검수/수정한 뒤 학습에 쓴다.

신뢰도가 낮은 결과는 별도 목록(review_queue.txt)으로 빼 우선 검수하게 한다.

사용법:
    python 04_label/auto_prelabel.py --images data/raw/images \
        --out data/prelabel --weights weights/plate_yolo.pt --conf 0.25
"""
from __future__ import annotations

import argparse
import pathlib
import sys

EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
# 한국 번호판 종횡비: 구형 1줄판 약 3.4, 유럽형 장방형 약 5.0, 2줄판 약 2.0
AR_MIN, AR_MAX = 1.8, 6.0


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="번호판 반자동 사전 라벨링")
    ap.add_argument("--images", default="data/raw/images")
    ap.add_argument("--out", default="data/prelabel")
    ap.add_argument("--weights", default="weights/plate_yolo.pt")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--review-thr", type=float, default=0.60,
                    help="이 신뢰도 미만은 우선 검수 대기열로 보낸다")
    ap.add_argument("--max-candidates", type=int, default=3,
                    help="폴백 모드에서 이미지당 남길 후보 수")
    return ap.parse_args()


def classic_candidates(img, cv2, np, max_n: int):
    """고전 영상처리로 번호판 후보 박스를 찾는다.

    흐름: 그레이스케일 → 대비 보정 → Sobel 가로 에지 → 이진화 →
          모폴로지 닫기 → 윤곽선 추출 → 종횡비/면적/채움률 필터
    번호판은 '가로 방향 에지가 밀집한 밝은 직사각형' 이라는 성질을 이용한다.
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 조명 편차를 줄이기 위해 지역 대비를 균등화한다
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    gray = cv2.bilateralFilter(gray, 7, 50, 50)      # 에지는 살리고 노이즈만 줄인다

    # 문자 세로 획이 만드는 '가로 방향 기울기' 를 강조한다
    sob = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
    sob = cv2.convertScaleAbs(sob)
    _, binary = cv2.threshold(sob, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # 문자 사이 빈틈을 메워 번호판 전체를 한 덩어리로 만든다
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (19, 5))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    cands = []
    img_area = float(w * h)
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        if bh == 0:
            continue
        ar = bw / bh
        area = bw * bh
        if not AR_MIN <= ar <= AR_MAX:
            continue                                  # 번호판 비율이 아니다
        if not 0.00015 <= area / img_area <= 0.12:
            continue                                  # 너무 작거나 너무 크다
        if bw < 60:
            continue                                  # OCR 불가 크기
        fill = cv2.contourArea(c) / area              # 윤곽선이 사각형에 가까운가
        if fill < 0.45:
            continue
        # 번호판은 주변보다 밝은 경우가 많다 — 밝기 대비를 점수에 반영한다
        roi = gray[y:y + bh, x:x + bw]
        local = float(np.mean(roi)) if roi.size else 0.0
        contrast = abs(local - float(np.mean(gray))) / 128.0
        # 휴리스틱 점수. 모델 신뢰도가 아니므로 0.6 을 넘지 않게 눌러 둔다
        score = min(0.59, 0.20 + 0.25 * fill + 0.20 * min(1.0, contrast))
        cands.append((x, y, x + bw, y + bh, score))

    cands.sort(key=lambda t: -t[4])
    return cands[:max_n]


def to_yolo_line(x1, y1, x2, y2, w, h, cls=0) -> str:
    """픽셀 박스를 YOLO 라벨 한 줄로 만든다."""
    cx = (x1 + x2) / 2.0 / w
    cy = (y1 + y2) / 2.0 / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def main() -> int:
    args = parse_args()
    try:
        import cv2                              # noqa: PLC0415
        import numpy as np                      # noqa: PLC0415
    except ImportError:
        print("[안내] OpenCV/NumPy 가 필요하다.")
        print("       pip install opencv-python-headless==4.12.0.88 numpy==2.1.3")
        return 0

    img_dir = pathlib.Path(args.images)
    if not img_dir.is_dir():
        print(f"[안내] 이미지 디렉터리가 없다: {img_dir}")
        return 0
    files = sorted(p for p in img_dir.rglob("*") if p.suffix.lower() in EXTS)
    if not files:
        print(f"[안내] {img_dir} 에 이미지가 없다.")
        return 0

    # 모델 경로를 먼저 시도하고, 불가하면 고전 영상처리로 내려간다
    model = None
    mode = "classic"
    wpath = pathlib.Path(args.weights)
    try:
        from ultralytics import YOLO            # noqa: PLC0415
        if wpath.exists():
            model = YOLO(str(wpath))
            mode = "model"
        else:
            print(f"[안내] 가중치가 없다({wpath}). 고전 영상처리 후보 생성으로 진행한다.")
    except ImportError:
        print("[안내] ultralytics 가 없다. 고전 영상처리 후보 생성으로 진행한다.")

    out_lbl = pathlib.Path(args.out) / "labels"
    out_lbl.mkdir(parents=True, exist_ok=True)
    review: list[str] = []
    n_box = n_empty = 0

    print(f"\n=== 사전 라벨링 ({mode} 모드, {len(files)} 장) ===")
    for f in files:
        img = cv2.imread(str(f))
        if img is None:
            print(f"  [경고] 읽기 실패: {f.name}")
            continue
        h, w = img.shape[:2]
        rows: list[tuple[int, int, int, int, float]] = []

        if mode == "model":
            res = model.predict(img, imgsz=640, conf=args.conf,
                                device="cpu", verbose=False)[0]
            if res.boxes is not None:
                for i in range(len(res.boxes)):
                    x1, y1, x2, y2 = [float(v) for v in res.boxes.xyxy[i].tolist()]
                    rows.append((int(x1), int(y1), int(x2), int(y2),
                                 float(res.boxes.conf[i])))
        else:
            rows = classic_candidates(img, cv2, np, args.max_candidates)

        lines = [to_yolo_line(*r[:4], w, h) for r in rows]
        (out_lbl / (f.stem + ".txt")).write_text(
            "".join(s + "\n" for s in lines), encoding="utf-8")

        if not rows:
            n_empty += 1
            review.append(f"{f.name}\tNO_CANDIDATE\t0.000")
            print(f"  {f.name:30s} 후보 0 건 → 수동 라벨링 필요")
            continue

        n_box += len(rows)
        best = max(r[4] for r in rows)
        flag = "검수대기" if best < args.review_thr else "통과후보"
        if best < args.review_thr:
            review.append(f"{f.name}\tLOW_CONF\t{best:.3f}")
        print(f"  {f.name:30s} 후보 {len(rows)} 건  최고점수={best:.3f}  {flag}")

    qpath = pathlib.Path(args.out) / "review_queue.txt"
    qpath.write_text("".join(s + "\n" for s in review), encoding="utf-8")

    print(f"\n=== 요약 ===")
    print(f"  처리 이미지     : {len(files)} 장")
    print(f"  생성 박스       : {n_box} 개")
    print(f"  후보 없음       : {n_empty} 장")
    print(f"  우선 검수 대상  : {len(review)} 장  → {qpath}")
    print(f"  사전 라벨 저장  : {out_lbl}")
    print("\n이 결과는 초안이다. 라벨링 도구로 열어 사람이 확인·수정한 뒤")
    print("04_label/label_lint.py 로 검수를 통과시키고 학습에 사용한다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
