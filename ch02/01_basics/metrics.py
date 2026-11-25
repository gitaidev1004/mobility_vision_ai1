# -*- coding: utf-8 -*-
"""Precision / Recall / AP / mAP 계산 실습.

검출 결과와 정답을 IoU 로 매칭해 TP/FP/FN 을 세고,
Precision-Recall 곡선과 AP(Average Precision)를 두 가지 보간 방식으로 계산한다.
 - 11점 보간(11-point interpolation): PASCAL VOC 2007 방식
 - 모든점 보간(all-point interpolation): VOC 2010 이후 및 COCO 가 쓰는 방식
"""
from __future__ import annotations


def iou(a, b) -> float:
    """두 박스 (x1,y1,x2,y2) 의 IoU."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    u = aa + bb - inter
    return inter / u if u > 0 else 0.0


def match_detections(preds, gts, iou_thr: float = 0.5):
    """검출 결과를 정답과 매칭해 TP/FP 플래그 목록을 만든다.

    Args:
        preds: [(image_id, score, bbox), ...] 전체 이미지의 검출 결과
        gts: {image_id: [bbox, ...]} 이미지별 정답 박스
        iou_thr: TP 로 인정할 IoU 하한

    Returns:
        (flags, n_gt) flags 는 신뢰도 내림차순으로 정렬된 1(TP)/0(FP) 목록
    """
    n_gt = sum(len(v) for v in gts.values())
    # 신뢰도 높은 검출부터 평가한다(순서가 PR 곡선 모양을 결정한다)
    order = sorted(preds, key=lambda x: x[1], reverse=True)
    used: dict[str, set[int]] = {k: set() for k in gts}
    flags: list[int] = []
    for image_id, _score, box in order:
        cands = gts.get(image_id, [])
        best_iou, best_idx = 0.0, -1
        for gi, g in enumerate(cands):
            if gi in used.get(image_id, set()):
                continue             # 이미 다른 검출이 차지한 정답은 재사용 금지
            v = iou(box, g)
            if v > best_iou:
                best_iou, best_idx = v, gi
        if best_idx >= 0 and best_iou >= iou_thr:
            used.setdefault(image_id, set()).add(best_idx)
            flags.append(1)          # TP
        else:
            flags.append(0)          # FP (중복 검출 또는 빗나간 검출)
    return flags, n_gt


def pr_curve(flags, n_gt: int):
    """누적 TP/FP 로 Precision-Recall 곡선 좌표를 만든다."""
    tp = fp = 0
    prec, rec = [], []
    for f in flags:
        tp += f
        fp += 1 - f
        prec.append(tp / (tp + fp))              # 맞춘 것 중 실제로 맞은 비율
        rec.append(tp / n_gt if n_gt else 0.0)   # 정답 중 찾아낸 비율
    return prec, rec


def ap_11point(prec, rec) -> float:
    """11점 보간 AP. recall 0.0, 0.1, ..., 1.0 지점의 최대 precision 평균."""
    total = 0.0
    for i in range(11):
        r = i / 10.0
        vals = [p for p, rr in zip(prec, rec) if rr >= r]
        total += max(vals) if vals else 0.0
    return total / 11.0


def ap_all_point(prec, rec) -> float:
    """모든점 보간 AP. PR 곡선 아래 면적을 단조 감소 보정 후 적분한다."""
    # 양 끝에 가상의 점을 붙여 면적 계산을 단순화한다
    mrec = [0.0] + list(rec) + [1.0]
    mpre = [0.0] + list(prec) + [0.0]
    # precision 을 뒤에서부터 단조 감소하도록 보정한다(톱니 제거)
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    area = 0.0
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            area += (mrec[i] - mrec[i - 1]) * mpre[i]
    return area


def precision_recall_f1(flags, n_gt: int):
    """최종(모든 검출 사용) Precision / Recall / F1 을 구한다."""
    tp = sum(flags)
    fp = len(flags) - tp
    fn = n_gt - tp
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1, tp, fp, fn


def map_over_thresholds(preds, gts, thrs=None):
    """IoU 임계값 목록 전체의 AP 평균(COCO 의 mAP@[.5:.95] 방식)."""
    if thrs is None:
        thrs = [0.50 + 0.05 * i for i in range(10)]
    rows = []
    for t in thrs:
        flags, n_gt = match_detections(preds, gts, iou_thr=t)
        prec, rec = pr_curve(flags, n_gt)
        rows.append((t, ap_all_point(prec, rec)))
    mean = sum(a for _, a in rows) / len(rows)
    return rows, mean


def main() -> None:
    # 테스트 이미지 4장, 정답 번호판 5개
    gts = {
        "img001": [(412, 288, 596, 342)],
        "img002": [(300, 500, 470, 548)],
        "img003": [(100, 120, 240, 162), (800, 400, 940, 442)],
        "img004": [(620, 300, 760, 344)],
    }
    # 검출 결과 7건(신뢰도 포함). 일부는 중복/오검출이다
    preds = [
        ("img001", 0.981, (414, 290, 598, 344)),   # TP
        ("img001", 0.512, (700, 700, 820, 742)),   # FP (오검출)
        ("img002", 0.944, (302, 503, 472, 551)),   # TP
        ("img003", 0.871, (104, 123, 243, 165)),   # TP
        ("img003", 0.402, (812, 410, 952, 452)),   # IoU 0.53 - 0.5 에서만 TP
        ("img003", 0.355, (110, 125, 250, 170)),   # FP (중복 검출)
        ("img004", 0.233, (900, 100, 980, 140)),   # FP (오검출), img004 는 미검출
    ]

    print("정답 박스 수:", sum(len(v) for v in gts.values()), " 검출 수:", len(preds))

    for thr in (0.50, 0.75):
        flags, n_gt = match_detections(preds, gts, iou_thr=thr)
        prec, rec = pr_curve(flags, n_gt)
        p, r, f1, tp, fp, fn = precision_recall_f1(flags, n_gt)
        print(f"\n=== IoU 임계값 {thr:.2f} ===")
        print("  TP/FP/FN   :", tp, fp, fn)
        print(f"  Precision  : {p:.4f}")
        print(f"  Recall     : {r:.4f}")
        print(f"  F1         : {f1:.4f}")
        print(f"  AP(11점)   : {ap_11point(prec, rec):.4f}")
        print(f"  AP(모든점) : {ap_all_point(prec, rec):.4f}")
        print("  PR 곡선    :")
        for i, (pp, rr) in enumerate(zip(prec, rec)):
            print(f"    {i+1}번째 검출까지  recall={rr:.3f}  precision={pp:.3f}")

    rows, mean = map_over_thresholds(preds, gts)
    print("\n=== mAP@[.50:.95] ===")
    for t, a in rows:
        print(f"  AP@{t:.2f} = {a:.4f}")
    print(f"  mAP      = {mean:.4f}")


if __name__ == "__main__":
    main()
