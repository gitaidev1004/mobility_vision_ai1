# -*- coding: utf-8 -*-
"""번호판 검출기 성능 평가 스크립트.

Test 세트에 대해 검출률(이미지 단위), Precision, Recall, F1, AP, mAP 를
IoU 임계값별로 계산한다.

동작 모드
  A) 가중치가 있으면 모델로 추론해 평가한다.
  B) 가중치가 없으면 predictions.json 파일(이미 저장한 추론 결과)을 읽어
     평가한다.
  C) 둘 다 없으면 내장 더미 예측으로 평가 절차만 시연한다.

예측 JSON 형식:
    [{"image": "img001.jpg", "bbox": [x1,y1,x2,y2], "score": 0.98}, ...]

사용법:
    python 06_eval/eval_detector.py --images data/plate/images/test \
        --labels data/plate/labels/test --weights weights/plate_yolo.pt
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="검출기 성능 평가")
    ap.add_argument("--images", default="data/plate/images/test")
    ap.add_argument("--labels", default="data/plate/labels/test")
    ap.add_argument("--weights", default="weights/plate_yolo.pt")
    ap.add_argument("--preds", default="runs/eval/predictions.json",
                    help="저장된 예측 결과(JSON). 가중치가 없을 때 사용")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--nms-iou", type=float, default=0.45)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out", default="runs/eval")
    return ap.parse_args()


# ---------------------------------------------------------------- 지표 계산
def iou(a, b) -> float:
    """두 박스 (x1,y1,x2,y2) 의 IoU."""
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    u = aa + bb - inter
    return inter / u if u > 0 else 0.0


def evaluate(preds, gts, iou_thr: float):
    """TP/FP/FN, Precision, Recall, F1, AP(모든점 보간)를 계산한다."""
    n_gt = sum(len(v) for v in gts.values())
    order = sorted(preds, key=lambda d: d["score"], reverse=True)
    used: dict[str, set[int]] = {}
    flags: list[int] = []
    matched_images: set[str] = set()

    for d in order:
        img = d["image"]
        cands = gts.get(img, [])
        best_iou, best_idx = 0.0, -1
        for gi, g in enumerate(cands):
            if gi in used.get(img, set()):
                continue
            v = iou(d["bbox"], g)
            if v > best_iou:
                best_iou, best_idx = v, gi
        if best_idx >= 0 and best_iou >= iou_thr:
            used.setdefault(img, set()).add(best_idx)
            flags.append(1)
            matched_images.add(img)
        else:
            flags.append(0)

    tp = sum(flags)
    fp = len(flags) - tp
    fn = n_gt - tp
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / n_gt if n_gt else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0

    # PR 곡선 → AP(모든점 보간)
    ctp = cfp = 0
    ps, rs = [], []
    for f in flags:
        ctp += f
        cfp += 1 - f
        ps.append(ctp / (ctp + cfp))
        rs.append(ctp / n_gt if n_gt else 0.0)
    mrec = [0.0] + rs + [1.0]
    mpre = [0.0] + ps + [0.0]
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    ap = 0.0
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            ap += (mrec[i] - mrec[i - 1]) * mpre[i]

    # 검출률: '번호판을 하나라도 찾아낸 이미지' 비율. 운영 지표로 자주 쓰인다
    n_img_with_gt = sum(1 for v in gts.values() if v)
    detect_rate = len(matched_images) / n_img_with_gt if n_img_with_gt else 0.0

    return {"iou": iou_thr, "tp": tp, "fp": fp, "fn": fn,
            "precision": prec, "recall": rec, "f1": f1, "ap": ap,
            "detect_rate": detect_rate}


# ---------------------------------------------------------------- 입출력
def load_gts(img_dir: pathlib.Path, lbl_dir: pathlib.Path, sizes):
    """YOLO 라벨을 픽셀 박스 딕셔너리로 읽는다."""
    gts: dict[str, list] = {}
    for stem, (w, h) in sizes.items():
        lp = lbl_dir / (stem + ".txt")
        boxes = []
        if lp.exists():
            for line in lp.read_text(encoding="utf-8").splitlines():
                p = line.split()
                if len(p) != 5:
                    continue
                _c, cx, cy, bw, bh = (float(v) for v in p)
                boxes.append([(cx - bw / 2) * w, (cy - bh / 2) * h,
                              (cx + bw / 2) * w, (cy + bh / 2) * h])
        gts[stem] = boxes
    return gts


DUMMY_SIZES = {f"img{i:03d}": (1920, 1080) for i in range(1, 13)}

DUMMY_GT_NORM = {   # (cx, cy, w, h) 정규화 정답 — 12장에 13개 번호판
    "img001": [(0.2625, 0.2917, 0.0958, 0.0500)],
    "img002": [(0.5000, 0.5000, 0.1200, 0.0360)],
    "img003": [(0.1250, 0.1500, 0.0700, 0.0220), (0.8000, 0.7000, 0.0900, 0.0280)],
    "img004": [(0.4000, 0.6000, 0.0600, 0.0190)],
    "img005": [(0.6000, 0.3000, 0.1500, 0.0450)],
    "img006": [(0.3000, 0.8000, 0.0500, 0.0160)],
    "img007": [(0.7000, 0.4000, 0.1100, 0.0330)],
    "img008": [(0.2000, 0.5000, 0.1300, 0.0400)],
    "img009": [(0.5500, 0.6500, 0.0450, 0.0140)],
    "img010": [(0.4500, 0.3500, 0.1000, 0.0300)],
    "img011": [(0.6500, 0.5500, 0.0800, 0.0250)],
    "img012": [(0.3500, 0.4500, 0.1400, 0.0420)],
}

# 더미 예측: 일부는 정답에 근접, 일부는 어긋나고, 2건은 완전 오검출이다
DUMMY_PRED_SPEC = [
    # (이미지, cx 오차, cy 오차, w 배율, h 배율, 신뢰도)
    ("img001", 0.0000, 0.0000, 1.00, 1.00, 0.981),
    ("img002", 0.0010, 0.0005, 1.01, 1.01, 0.962),
    ("img003", 0.0005, 0.0003, 0.99, 1.00, 0.948),   # img003 의 첫 번째 정답
    ("img003", 0.0020, 0.0012, 1.04, 1.06, 0.713),   # 같은 정답에 두 번째 박스(FP)
    ("img004", 0.0015, 0.0010, 1.06, 1.08, 0.604),
    ("img005", 0.0003, 0.0002, 1.01, 1.00, 0.976),
    ("img006", 0.0040, 0.0030, 1.18, 1.22, 0.412),   # 소형 번호판 — 느슨한 박스
    ("img007", 0.0006, 0.0004, 0.99, 1.01, 0.934),
    ("img008", 0.0004, 0.0002, 1.00, 1.00, 0.958),
    ("img010", 0.0010, 0.0006, 1.02, 1.02, 0.887),
    ("img011", 0.0020, 0.0015, 1.06, 1.08, 0.552),   # 야간 — 신뢰도 낮음
    ("img012", 0.0003, 0.0002, 1.00, 1.00, 0.969),
]
# img009(소형 번호판)는 아예 검출되지 않은 미검출 사례다.
DUMMY_FALSE_POSITIVES = [("img004", 0.85, 0.12, 0.08, 0.025, 0.388),
                         ("img009", 0.20, 0.80, 0.06, 0.020, 0.301)]


def build_dummy():
    """내장 더미 정답/예측을 픽셀 좌표로 만든다."""
    gts: dict[str, list] = {}
    for k, boxes in DUMMY_GT_NORM.items():
        w, h = DUMMY_SIZES[k]
        gts[k] = [[(cx - bw / 2) * w, (cy - bh / 2) * h,
                   (cx + bw / 2) * w, (cy + bh / 2) * h]
                  for cx, cy, bw, bh in boxes]

    preds = []
    for img, dx, dy, sw, sh, sc in DUMMY_PRED_SPEC:
        w, h = DUMMY_SIZES[img]
        cx, cy, bw, bh = DUMMY_GT_NORM[img][0]
        cx, cy = cx + dx, cy + dy
        bw, bh = bw * sw, bh * sh
        preds.append({"image": img, "score": sc,
                      "bbox": [(cx - bw / 2) * w, (cy - bh / 2) * h,
                               (cx + bw / 2) * w, (cy + bh / 2) * h]})
    for img, cx, cy, bw, bh, sc in DUMMY_FALSE_POSITIVES:
        w, h = DUMMY_SIZES[img]
        preds.append({"image": img, "score": sc,
                      "bbox": [(cx - bw / 2) * w, (cy - bh / 2) * h,
                               (cx + bw / 2) * w, (cy + bh / 2) * h]})
    return gts, preds


def main() -> int:
    args = parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    img_dir = pathlib.Path(args.images)
    lbl_dir = pathlib.Path(args.labels)
    gts: dict[str, list] = {}
    preds: list[dict] = []
    mode = "dummy"

    # --- 모드 A: 모델 추론 ------------------------------------------------
    have_model = False
    try:
        from ultralytics import YOLO            # noqa: PLC0415
        have_model = pathlib.Path(args.weights).exists()
    except ImportError:
        pass

    if have_model and img_dir.is_dir():
        import cv2                              # noqa: PLC0415
        model = YOLO(args.weights)
        sizes: dict[str, tuple[int, int]] = {}
        files = sorted(p for p in img_dir.rglob("*") if p.suffix.lower() in EXTS)
        for f in files:
            im = cv2.imread(str(f))
            if im is None:
                continue
            sizes[f.stem] = (im.shape[1], im.shape[0])
            r = model.predict(im, imgsz=args.imgsz, conf=args.conf,
                              iou=args.nms_iou, device="cpu", verbose=False)[0]
            if r.boxes is not None:
                for i in range(len(r.boxes)):
                    preds.append({"image": f.stem,
                                  "bbox": [float(v) for v in r.boxes.xyxy[i].tolist()],
                                  "score": float(r.boxes.conf[i])})
        gts = load_gts(img_dir, lbl_dir, sizes)
        mode = "model"
        (out / "predictions.json").write_text(
            json.dumps(preds, ensure_ascii=False, indent=1), encoding="utf-8")

    # --- 모드 B: 저장된 예측 JSON ----------------------------------------
    elif pathlib.Path(args.preds).exists() and lbl_dir.is_dir():
        preds = json.loads(pathlib.Path(args.preds).read_text(encoding="utf-8"))
        sizes = {}
        try:
            import cv2                          # noqa: PLC0415
            for f in sorted(p for p in img_dir.rglob("*")
                            if p.suffix.lower() in EXTS):
                im = cv2.imread(str(f))
                if im is not None:
                    sizes[f.stem] = (im.shape[1], im.shape[0])
        except ImportError:
            pass
        gts = load_gts(img_dir, lbl_dir, sizes)
        mode = "preds-json"

    # --- 모드 C: 더미 ------------------------------------------------------
    if not gts or not preds:
        print("[안내] 가중치/예측 파일이 없어 내장 더미 데이터로 평가 절차를 시연한다.")
        print("       실제 수치는 모델과 데이터에 따라 달라진다.\n")
        gts, preds = build_dummy()
        mode = "dummy"

    n_img = len(gts)
    n_gt = sum(len(v) for v in gts.values())
    print(f"=== 평가 대상 ({mode} 모드) ===")
    print(f"  이미지 {n_img} 장 / 정답 번호판 {n_gt} 개 / 검출 {len(preds)} 건")

    thrs = [0.30, 0.50, 0.60, 0.70, 0.75, 0.80, 0.90]
    rows = [evaluate(preds, gts, t) for t in thrs]

    head = (f"{'IoU':>6s}{'TP':>5s}{'FP':>5s}{'FN':>5s}"
            f"{'Precision':>11s}{'Recall':>9s}{'F1':>8s}{'AP':>8s}{'검출률':>9s}")
    print("\n=== IoU 임계값별 성능 ===")
    print(head)
    print("-" * len(head))
    for r in rows:
        print(f"{r['iou']:6.2f}{r['tp']:5d}{r['fp']:5d}{r['fn']:5d}"
              f"{r['precision']:11.4f}{r['recall']:9.4f}{r['f1']:8.4f}"
              f"{r['ap']:8.4f}{r['detect_rate']:9.4f}")
    print("-" * len(head))

    # COCO 방식 mAP@[.50:.95]
    coco = [evaluate(preds, gts, 0.50 + 0.05 * i) for i in range(10)]
    map5095 = sum(r["ap"] for r in coco) / len(coco)
    base = next(r for r in rows if abs(r["iou"] - 0.50) < 1e-9)
    print("\n=== 요약 (운영 보고용) ===")
    print(f"  검출률(IoU 0.5)   : {base['detect_rate']:.4f}")
    print(f"  Precision(IoU 0.5): {base['precision']:.4f}")
    print(f"  Recall(IoU 0.5)   : {base['recall']:.4f}")
    print(f"  mAP@0.5           : {base['ap']:.4f}")
    print(f"  mAP@[.50:.95]     : {map5095:.4f}")

    report = {"mode": mode, "images": n_img, "gt_boxes": n_gt,
              "predictions": len(preds), "by_iou": rows,
              "map50": base["ap"], "map50_95": map5095}
    (out / "eval_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n보고서 저장: {out / 'eval_report.json'}")
    print("※ 위 수치는 예시 측정값이다. 팀의 Test 세트로 직접 재측정한다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
