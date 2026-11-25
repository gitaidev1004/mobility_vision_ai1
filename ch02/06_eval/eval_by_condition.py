# -*- coding: utf-8 -*-
"""촬영 환경별로 Test 세트를 쪼개 평가한다.

전체 평균 mAP 만 보면 "야간에서만 Recall 이 0.6" 같은 치명적인 약점이 가려진다.
환경별로 나눠 측정해야 보강할 데이터와 개선할 전처리를 특정할 수 있다.

분류 기준
  - 조명  : 평균 밝기 → day / night
  - 크기  : 번호판 픽셀 폭 → tiny(<100) / small(100~160) / normal(160~260) / large(>=260)
  - 기울기: 파일명 태그 또는 메타 CSV 의 angle 값 → straight / tilted
  - 선명도: 라플라시안 분산 → sharp / blurry

조건 태그는 아래 순서로 결정한다.
  1) conditions.csv 가 있으면 그 값을 쓴다(image,condition,angle 열)
  2) 없으면 파일명에 포함된 태그(night, rain, tilt, backlight)를 쓴다
  3) 둘 다 없으면 이미지에서 자동 측정한다(OpenCV 필요)

사용법:
    python 06_eval/eval_by_condition.py --images data/plate/images/test \
        --labels data/plate/labels/test --preds runs/eval/predictions.json
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys

EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
NAME_TAGS = ("night", "rain", "backlight", "tilt", "blur", "far", "near")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="환경별 분할 평가")
    ap.add_argument("--images", default="data/plate/images/test")
    ap.add_argument("--labels", default="data/plate/labels/test")
    ap.add_argument("--preds", default="runs/eval/predictions.json")
    ap.add_argument("--conditions", default="data/plate/conditions.csv",
                    help="image,condition,angle 열을 가진 CSV(있으면 우선 사용)")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--dark-thr", type=float, default=70.0)
    ap.add_argument("--blur-thr", type=float, default=100.0)
    ap.add_argument("--out", default="runs/eval")
    return ap.parse_args()


def iou_box(a, b) -> float:
    """두 박스 (x1,y1,x2,y2) 의 IoU."""
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    u = aa + bb - inter
    return inter / u if u > 0 else 0.0


def size_bucket(px_w: float) -> str:
    """번호판 픽셀 폭을 크기 구간으로 나눈다."""
    if px_w < 100:
        return "tiny(<100px)"
    if px_w < 160:
        return "small(100-160px)"
    if px_w < 260:
        return "normal(160-260px)"
    return "large(>=260px)"


def eval_subset(preds, gts, iou_thr: float):
    """부분집합에 대한 TP/FP/FN, Precision, Recall, F1, 검출률을 계산한다."""
    n_gt = sum(len(v) for v in gts.values())
    order = sorted(preds, key=lambda d: d["score"], reverse=True)
    used: dict[str, set[int]] = {}
    tp = fp = 0
    hit_images: set[str] = set()
    for d in order:
        img = d["image"]
        best_iou, best_idx = 0.0, -1
        for gi, g in enumerate(gts.get(img, [])):
            if gi in used.get(img, set()):
                continue
            v = iou_box(d["bbox"], g)
            if v > best_iou:
                best_iou, best_idx = v, gi
        if best_idx >= 0 and best_iou >= iou_thr:
            used.setdefault(img, set()).add(best_idx)
            tp += 1
            hit_images.add(img)
        else:
            fp += 1
    fn = n_gt - tp
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / n_gt if n_gt else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    n_img = sum(1 for v in gts.values() if v)
    dr = len(hit_images) / n_img if n_img else 0.0
    return {"images": n_img, "gt": n_gt, "tp": tp, "fp": fp, "fn": fn,
            "precision": prec, "recall": rec, "f1": f1, "detect_rate": dr}


def build_demo():
    """내장 시연 데이터. 환경 태그가 붙은 24장을 만든다."""
    conds = (["day"] * 8 + ["night"] * 6 + ["backlight"] * 4
             + ["rain"] * 3 + ["tilted"] * 3)
    # 환경별로 (중심 신뢰도, IoU 저하폭)을 다르게 둬 현실적인 격차를 만든다
    profile = {"day": (0.96, 1.01), "night": (0.63, 1.10),
               "backlight": (0.71, 1.07), "rain": (0.68, 1.09),
               "tilted": (0.58, 1.16)}
    sizes = {}
    gts: dict[str, list] = {}
    preds: list[dict] = []
    meta: dict[str, dict] = {}
    for i, c in enumerate(conds, 1):
        name = f"t{i:03d}_{c}"
        w, h = 1920, 1080
        sizes[name] = (w, h)
        pw = [240, 200, 170, 140, 110, 86][i % 6]
        ph = int(pw * 0.30)
        x, y = 400 + (i * 17) % 900, 300 + (i * 29) % 500
        gts[name] = [[float(x), float(y), float(x + pw), float(y + ph)]]
        meta[name] = {"condition": c, "plate_px": float(pw),
                      "brightness": 36.0 if c == "night" else 118.0,
                      "sharpness": 72.0 if c in ("rain", "tilted") else 230.0}
        sc, grow = profile[c]
        score = max(0.05, min(0.995, sc - 0.03 * (i % 5)))
        # 작은 번호판은 더 크게 어긋나게 해 소형 검출 약화를 재현한다
        g = grow + (0.10 if pw < 110 else 0.0)
        cx, cy = x + pw / 2, y + ph / 2
        nw, nh = pw * g, ph * g
        # 가장 작은 번호판 일부는 아예 검출 실패(미검출)로 둔다
        if pw < 100 and c in ("night", "tilted"):
            continue
        preds.append({"image": name, "score": score,
                      "bbox": [cx - nw / 2, cy - nh / 2, cx + nw / 2, cy + nh / 2]})
    # 오검출 2건을 섞는다
    preds.append({"image": "t003_day", "score": 0.34,
                  "bbox": [1600.0, 900.0, 1700.0, 930.0]})
    preds.append({"image": "t010_night", "score": 0.29,
                  "bbox": [100.0, 100.0, 190.0, 128.0]})
    return gts, preds, meta


def load_meta(args, sizes, gts):
    """조건 태그를 결정한다(CSV → 파일명 → 자동 측정 순서)."""
    meta: dict[str, dict] = {}
    csv_path = pathlib.Path(args.conditions)
    csv_rows: dict[str, dict] = {}
    if csv_path.exists():
        with csv_path.open(encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                key = pathlib.Path(row.get("image", "")).stem
                if key:
                    csv_rows[key] = row

    try:
        import cv2                              # noqa: PLC0415
        import numpy as np                       # noqa: PLC0415
        have_cv = True
    except ImportError:
        have_cv = False

    img_dir = pathlib.Path(args.images)
    for stem, (w, h) in sizes.items():
        boxes = gts.get(stem, [])
        plate_px = max((b[2] - b[0]) for b in boxes) if boxes else 0.0
        cond, bright, sharp, angle = None, None, None, 0.0

        if stem in csv_rows:
            cond = (csv_rows[stem].get("condition") or "").strip() or None
            try:
                angle = float(csv_rows[stem].get("angle") or 0.0)
            except ValueError:
                angle = 0.0
        if cond is None:
            for tag in NAME_TAGS:
                if tag in stem.lower():
                    cond = tag
                    break
        if have_cv:
            cands = [p for p in img_dir.rglob(stem + ".*")
                     if p.suffix.lower() in EXTS]
            if cands:
                im = cv2.imread(str(cands[0]))
                if im is not None:
                    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                    bright = float(np.mean(g))
                    sharp = float(cv2.Laplacian(g, cv2.CV_64F).var())
        if cond is None:
            if bright is not None and bright < args.dark_thr:
                cond = "night"
            elif sharp is not None and sharp < args.blur_thr:
                cond = "blur"
            else:
                cond = "day"
        meta[stem] = {"condition": cond, "plate_px": plate_px,
                      "brightness": bright if bright is not None else 0.0,
                      "sharpness": sharp if sharp is not None else 0.0,
                      "angle": angle}
    return meta


def group_eval(title: str, groups: dict[str, list[str]], gts, preds, iou_thr):
    """그룹별 평가 결과를 표로 출력하고 결과 딕셔너리를 돌려준다."""
    by_img: dict[str, list[dict]] = {}
    for d in preds:
        by_img.setdefault(d["image"], []).append(d)

    head = (f"{'구분':18s}{'이미지':>7s}{'정답':>6s}{'TP':>5s}{'FP':>5s}{'FN':>5s}"
            f"{'Precision':>11s}{'Recall':>9s}{'F1':>8s}{'검출률':>9s}")
    print(f"\n=== {title} ===")
    print(head)
    print("-" * len(head))
    results: dict[str, dict] = {}
    for key in sorted(groups):
        members = groups[key]
        sub_gts = {k: gts.get(k, []) for k in members}
        sub_preds = [d for k in members for d in by_img.get(k, [])]
        r = eval_subset(sub_preds, sub_gts, iou_thr)
        results[key] = r
        print(f"{key:18s}{r['images']:7d}{r['gt']:6d}{r['tp']:5d}{r['fp']:5d}"
              f"{r['fn']:5d}{r['precision']:11.4f}{r['recall']:9.4f}"
              f"{r['f1']:8.4f}{r['detect_rate']:9.4f}")
    print("-" * len(head))
    # 가장 약한 그룹을 지목한다
    weak = min(results.items(), key=lambda kv: kv[1]["recall"]) if results else None
    if weak:
        print(f"  최저 Recall 그룹: {weak[0]} ({weak[1]['recall']:.4f}) "
              f"→ 이 조건의 학습 데이터를 보강한다")
    return results


def main() -> int:
    args = parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    lbl_dir = pathlib.Path(args.labels)
    preds_path = pathlib.Path(args.preds)
    gts: dict[str, list] = {}
    preds: list[dict] = []
    meta: dict[str, dict] = {}

    if preds_path.exists() and lbl_dir.is_dir():
        preds = json.loads(preds_path.read_text(encoding="utf-8"))
        sizes: dict[str, tuple[int, int]] = {}
        try:
            import cv2                          # noqa: PLC0415
            for f in sorted(p for p in pathlib.Path(args.images).rglob("*")
                            if p.suffix.lower() in EXTS):
                im = cv2.imread(str(f))
                if im is not None:
                    sizes[f.stem] = (im.shape[1], im.shape[0])
        except ImportError:
            pass
        for stem, (w, h) in sizes.items():
            lp = lbl_dir / (stem + ".txt")
            boxes = []
            if lp.exists():
                for line in lp.read_text(encoding="utf-8").splitlines():
                    p = line.split()
                    if len(p) == 5:
                        _c, cx, cy, bw, bh = (float(v) for v in p)
                        boxes.append([(cx - bw / 2) * w, (cy - bh / 2) * h,
                                      (cx + bw / 2) * w, (cy + bh / 2) * h])
            gts[stem] = boxes
        meta = load_meta(args, sizes, gts)

    if not gts or not preds:
        print("[안내] 예측 파일이나 라벨이 없어 내장 시연 데이터로 절차를 보여준다.")
        print("       먼저 06_eval/eval_detector.py 를 실행해 predictions.json 을 만든다.")
        print("       아래 수치는 예시 측정값이며 실제 데이터와 다르다.")
        gts, preds, meta = build_demo()

    # ---- 조건별 그룹 구성 -------------------------------------------------
    by_cond: dict[str, list[str]] = {}
    by_size: dict[str, list[str]] = {}
    by_light: dict[str, list[str]] = {}
    for stem, m in meta.items():
        by_cond.setdefault(m["condition"], []).append(stem)
        by_size.setdefault(size_bucket(m["plate_px"]), []).append(stem)
        lab = "night(저조도)" if m["brightness"] < args.dark_thr else "day(주간)"
        by_light.setdefault(lab, []).append(stem)

    print(f"=== 환경별 평가 (IoU {args.iou:.2f}) ===")
    print(f"  이미지 {len(gts)} 장 / 정답 {sum(len(v) for v in gts.values())} 개 "
          f"/ 검출 {len(preds)} 건")

    r1 = group_eval("촬영 조건별", by_cond, gts, preds, args.iou)
    r2 = group_eval("번호판 크기별", by_size, gts, preds, args.iou)
    r3 = group_eval("조명(밝기)별", by_light, gts, preds, args.iou)

    report = {"iou": args.iou, "by_condition": r1,
              "by_size": r2, "by_light": r3}
    (out / "eval_by_condition.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n보고서 저장: {out / 'eval_by_condition.json'}")
    print("※ 수치는 예시 측정값이다. 환경별 샘플 수가 적으면 신뢰구간이 넓다.")
    print("  각 그룹에 최소 50장 이상을 확보한 뒤 결론을 내린다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
