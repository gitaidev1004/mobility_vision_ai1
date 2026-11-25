# -*- coding: utf-8 -*-
"""오검출(False Positive)/미검출(False Negative) 샘플 갤러리를 만든다.

수치만 보면 "mAP 0.93" 이 어디서 깎였는지 알 수 없다.
틀린 이미지를 직접 눈으로 봐야 원인이 보인다.
이 스크립트는 다음을 수행한다.

  1) 예측과 정답을 IoU 로 매칭해 오류 유형을 분류한다
       FN_MISS      정답이 있는데 겹치는 검출이 전혀 없다(완전 미검출)
       FN_LOW_IOU   검출은 있으나 IoU 가 임계값 미만이다(박스 부정확)
       FP_GHOST     정답과 전혀 겹치지 않는 박스다(배경 오검출)
       FP_DUP       이미 다른 검출이 차지한 정답에 겹친 중복 박스다
  2) 유형별 디렉터리에 박스를 그린 이미지를 저장한다
       정답=초록, 검출=빨강, 텍스트로 IoU/신뢰도를 표시한다
  3) errors.csv 로 목록을 남겨 팀이 나눠 검토할 수 있게 한다

OpenCV 가 없으면 CSV 목록만 만든다.

사용법:
    python 06_eval/error_gallery.py --images data/plate/images/test \
        --labels data/plate/labels/test --preds runs/eval/predictions.json
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys

EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
KINDS = ("FN_MISS", "FN_LOW_IOU", "FP_GHOST", "FP_DUP")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="오류 샘플 갤러리 생성")
    ap.add_argument("--images", default="data/plate/images/test")
    ap.add_argument("--labels", default="data/plate/labels/test")
    ap.add_argument("--preds", default="runs/eval/predictions.json")
    ap.add_argument("--iou", type=float, default=0.5, help="TP 판정 IoU")
    ap.add_argument("--touch-iou", type=float, default=0.05,
                    help="이 값 이상 겹치면 '박스 부정확', 미만이면 '배경 오검출'")
    ap.add_argument("--out", default="runs/eval/errors")
    ap.add_argument("--max-per-kind", type=int, default=30,
                    help="유형별 저장 상한(디스크 보호)")
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


def classify(image: str, gt_boxes, pred_boxes, iou_thr: float, touch_iou: float):
    """한 이미지의 오류를 분류해 레코드 목록으로 돌려준다."""
    recs: list[dict] = []
    # 신뢰도 높은 검출부터 정답을 차지한다(평가 규칙과 동일하게 맞춘다)
    order = sorted(range(len(pred_boxes)),
                   key=lambda i: pred_boxes[i]["score"], reverse=True)
    taken: set[int] = set()
    matched_pred: set[int] = set()

    for pi in order:
        pb = pred_boxes[pi]["bbox"]
        best_iou, best_gi = 0.0, -1
        for gi, gb in enumerate(gt_boxes):
            v = iou_box(pb, gb)
            if v > best_iou:
                best_iou, best_gi = v, gi
        if best_gi >= 0 and best_iou >= iou_thr and best_gi not in taken:
            taken.add(best_gi)
            matched_pred.add(pi)
            continue                       # 정상 TP — 갤러리에 담지 않는다
        if best_gi >= 0 and best_iou >= iou_thr and best_gi in taken:
            recs.append({"image": image, "kind": "FP_DUP", "pred_idx": pi,
                         "gt_idx": best_gi, "iou": best_iou,
                         "score": pred_boxes[pi]["score"]})
        elif best_iou >= touch_iou:
            recs.append({"image": image, "kind": "FN_LOW_IOU", "pred_idx": pi,
                         "gt_idx": best_gi, "iou": best_iou,
                         "score": pred_boxes[pi]["score"]})
        else:
            recs.append({"image": image, "kind": "FP_GHOST", "pred_idx": pi,
                         "gt_idx": -1, "iou": best_iou,
                         "score": pred_boxes[pi]["score"]})

    # 아무 검출과도 겹치지 않은 정답은 완전 미검출이다
    for gi, gb in enumerate(gt_boxes):
        if gi in taken:
            continue
        best = max((iou_box(gb, p["bbox"]) for p in pred_boxes), default=0.0)
        if best < touch_iou:
            recs.append({"image": image, "kind": "FN_MISS", "pred_idx": -1,
                         "gt_idx": gi, "iou": best, "score": 0.0})
    return recs


def draw_and_save(cv2, img, gt_boxes, pred_boxes, rec, path: pathlib.Path) -> None:
    """정답(초록)/검출(빨강) 박스를 그려 저장한다."""
    vis = img.copy()
    for gb in gt_boxes:
        x1, y1, x2, y2 = (int(v) for v in gb)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 220, 0), 2)
        cv2.putText(vis, "GT", (x1, max(14, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1)
    for p in pred_boxes:
        x1, y1, x2, y2 = (int(v) for v in p["bbox"])
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 235), 2)
        cv2.putText(vis, f"{p['score']:.2f}", (x1, min(vis.shape[0] - 4, y2 + 16)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 235), 1)
    label = f"{rec['kind']} iou={rec['iou']:.3f} score={rec['score']:.3f}"
    cv2.rectangle(vis, (0, 0), (min(vis.shape[1], 520), 30), (0, 0, 0), -1)
    cv2.putText(vis, label, (8, 21), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), vis)


def build_demo():
    """내장 시연 데이터. 네 가지 오류 유형이 모두 들어 있다."""
    gts = {
        "e001_day":   [[412.0, 288.0, 596.0, 342.0]],
        "e002_night": [[300.0, 500.0, 470.0, 548.0]],
        "e003_tilt":  [[100.0, 120.0, 240.0, 162.0],
                       [800.0, 400.0, 940.0, 442.0]],
        "e004_far":   [[1500.0, 700.0, 1586.0, 726.0]],
    }
    preds = [
        # e001: 정상 TP + 배경 오검출
        {"image": "e001_day", "score": 0.978, "bbox": [414.0, 290.0, 598.0, 344.0]},
        {"image": "e001_day", "score": 0.331, "bbox": [1200.0, 900.0, 1310.0, 934.0]},
        # e002: 박스가 너무 커서 IoU 미달
        {"image": "e002_night", "score": 0.586,
         "bbox": [260.0, 470.0, 520.0, 590.0]},
        # e003: 첫 정답에 중복 박스 2개, 두 번째 정답은 미검출
        {"image": "e003_tilt", "score": 0.902, "bbox": [102.0, 122.0, 242.0, 164.0]},
        {"image": "e003_tilt", "score": 0.655, "bbox": [106.0, 124.0, 246.0, 166.0]},
        # e004: 소형 번호판 완전 미검출(검출 없음)
    ]
    return gts, preds


def main() -> int:
    args = parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    lbl_dir = pathlib.Path(args.labels)
    img_dir = pathlib.Path(args.images)
    preds_path = pathlib.Path(args.preds)

    gts: dict[str, list] = {}
    preds: list[dict] = []
    img_paths: dict[str, pathlib.Path] = {}
    mode = "demo"

    try:
        import cv2                              # noqa: PLC0415
        have_cv = True
    except ImportError:
        cv2 = None
        have_cv = False
        print("[안내] OpenCV 가 없어 이미지 저장은 건너뛰고 CSV 목록만 만든다.")

    if preds_path.exists() and lbl_dir.is_dir() and img_dir.is_dir() and have_cv:
        preds = json.loads(preds_path.read_text(encoding="utf-8"))
        for f in sorted(p for p in img_dir.rglob("*") if p.suffix.lower() in EXTS):
            im = cv2.imread(str(f))
            if im is None:
                continue
            img_paths[f.stem] = f
            h, w = im.shape[:2]
            lp = lbl_dir / (f.stem + ".txt")
            boxes = []
            if lp.exists():
                for line in lp.read_text(encoding="utf-8").splitlines():
                    t = line.split()
                    if len(t) == 5:
                        _c, cx, cy, bw, bh = (float(v) for v in t)
                        boxes.append([(cx - bw / 2) * w, (cy - bh / 2) * h,
                                      (cx + bw / 2) * w, (cy + bh / 2) * h])
            gts[f.stem] = boxes
        mode = "real"

    if not gts:
        print("[안내] 예측/라벨을 찾지 못해 내장 시연 데이터로 분류 절차를 보여준다.")
        gts, preds = build_demo()
        mode = "demo"

    by_img: dict[str, list[dict]] = {}
    for d in preds:
        by_img.setdefault(d["image"], []).append(d)

    all_recs: list[dict] = []
    for stem, gt_boxes in gts.items():
        all_recs.extend(classify(stem, gt_boxes, by_img.get(stem, []),
                                 args.iou, args.touch_iou))

    counts = {k: 0 for k in KINDS}
    for r in all_recs:
        counts[r["kind"]] += 1

    print(f"\n=== 오류 유형 분류 ({mode} 모드) ===")
    head = f"{'유형':12s}{'건수':>6s}  의미 / 주요 원인"
    meaning = {
        "FN_MISS": "완전 미검출 — 소형/저조도/가림. 해당 조건 데이터 보강",
        "FN_LOW_IOU": "박스 부정확 — 라벨 일관성, 해상도 상향 검토",
        "FP_GHOST": "배경 오검출 — 배경 이미지(빈 라벨) 추가, conf 상향",
        "FP_DUP": "중복 검출 — NMS IoU 임계값 하향 검토",
    }
    print(head)
    print("-" * 78)
    for k in KINDS:
        print(f"{k:12s}{counts[k]:6d}  {meaning[k]}")
    print("-" * 78)
    print(f"{'합계':12s}{len(all_recs):6d}")

    # 상세 목록
    print("\n=== 상세 목록 (상위 20건) ===")
    all_recs.sort(key=lambda r: (r["kind"], -r["score"]))
    for r in all_recs[:20]:
        print(f"  {r['kind']:12s} {r['image']:16s} "
              f"iou={r['iou']:.3f} score={r['score']:.3f} "
              f"gt#{r['gt_idx']} pred#{r['pred_idx']}")

    # CSV 저장
    csv_path = out / "errors.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=["image", "kind", "pred_idx",
                                            "gt_idx", "iou", "score"])
        wr.writeheader()
        wr.writerows(all_recs)
    print(f"\n목록 저장: {csv_path}")

    # 이미지 저장(실데이터 + OpenCV 가 있을 때만)
    if mode == "real" and have_cv:
        saved = {k: 0 for k in KINDS}
        for r in all_recs:
            if saved[r["kind"]] >= args.max_per_kind:
                continue
            src = img_paths.get(r["image"])
            if src is None:
                continue
            im = cv2.imread(str(src))
            if im is None:
                continue
            dst = out / r["kind"] / f"{r['image']}_{saved[r['kind']]:03d}.jpg"
            draw_and_save(cv2, im, gts[r["image"]],
                          by_img.get(r["image"], []), r, dst)
            saved[r["kind"]] += 1
        print("이미지 저장 건수:", saved)
        print(f"갤러리 위치: {out}")
    else:
        print("(시연 모드에서는 원본 이미지가 없어 이미지 저장을 생략한다)")

    print("\n검토 방법: 유형별 폴더를 팀원이 나눠 열어 보고,")
    print("  같은 원인이 반복되는지 적어 둔다. 원인이 데이터면 수집/라벨을 고치고,")
    print("  후처리 설정이면 2.8절의 임계값 튜닝으로 해결한다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
