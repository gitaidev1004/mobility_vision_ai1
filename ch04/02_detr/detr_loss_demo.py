# -*- coding: utf-8 -*-
"""DETR 손실 함수(분류 + L1 + GIoU)의 조합을 계산해 확인한다.

L_total = sum_i [ -log p_i(c_i) + w_l1 * ||b_i - b_i_hat||_1
                  + w_giou * (1 - GIoU(b_i, b_i_hat)) ]
배경으로 매칭된 쿼리는 분류 손실만 계산하며, 가중치 eos_coef 를 곱해 줄인다.
"""
import numpy as np


def softmax(x, axis=-1):
    z = x - x.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def cxcywh_to_xyxy(b):
    b = np.asarray(b, dtype=np.float64)
    cx, cy, w, h = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=-1)


def giou(a, b):
    """단일 박스 쌍의 GIoU. a, b 는 xyxy."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    area_a = max(ax2 - ax1, 0) * max(ay2 - ay1, 0)
    area_b = max(bx2 - bx1, 0) * max(by2 - by1, 0)
    iw = max(min(ax2, bx2) - max(ax1, bx1), 0)
    ih = max(min(ay2, by2) - max(ay1, by1), 0)
    inter = iw * ih
    union = area_a + area_b - inter
    iou = inter / max(union, 1e-9)
    cw = max(ax2, bx2) - min(ax1, bx1)
    ch = max(ay2, by2) - min(ay1, by1)
    area_c = max(cw, 0) * max(ch, 0)
    return iou - (area_c - union) / max(area_c, 1e-9), iou


def detr_loss(pred_logits, pred_boxes, tgt_labels, tgt_boxes, matching,
              w_cls=1.0, w_l1=5.0, w_giou=2.0, eos_coef=0.1, verbose=True):
    """매칭이 주어졌을 때 DETR 손실을 계산한다.

    matching: {쿼리 인덱스: 정답 인덱스} 딕셔너리
    """
    n_query, n_cls_plus1 = pred_logits.shape
    bg = n_cls_plus1 - 1                     # 배경 클래스 인덱스
    prob = softmax(pred_logits, axis=1)

    loss_cls_fg = 0.0
    loss_cls_bg = 0.0
    loss_l1 = 0.0
    loss_giou = 0.0
    rows = []

    for q in range(n_query):
        if q in matching:
            g = matching[q]
            c = int(tgt_labels[g])
            lc = -np.log(max(prob[q, c], 1e-12))
            l1 = float(np.abs(pred_boxes[q] - tgt_boxes[g]).sum())
            gi, iou = giou(cxcywh_to_xyxy(pred_boxes[q]),
                           cxcywh_to_xyxy(tgt_boxes[g]))
            lg = 1.0 - gi
            loss_cls_fg += lc
            loss_l1 += l1
            loss_giou += lg
            rows.append((q, f"obj{g}", lc, l1, lg, iou))
        else:
            lc = -np.log(max(prob[q, bg], 1e-12)) * eos_coef
            loss_cls_bg += lc
            rows.append((q, "배경", lc, 0.0, 0.0, 0.0))

    n_box = max(len(matching), 1)
    total = (w_cls * (loss_cls_fg + loss_cls_bg)
             + w_l1 * loss_l1 / n_box
             + w_giou * loss_giou / n_box)

    if verbose:
        print(f"{'쿼리':>4s} {'매칭':>6s} {'분류손실':>10s} {'L1':>8s} "
              f"{'1-GIoU':>8s} {'IoU':>7s}")
        for q, m, lc, l1, lg, iou in rows:
            print(f"{q:>4d} {m:>6s} {lc:>10.4f} {l1:>8.4f} {lg:>8.4f} {iou:>7.3f}")
        print("-" * 52)
        print(f"전경 분류 손실 합      = {loss_cls_fg:.4f}")
        print(f"배경 분류 손실 합      = {loss_cls_bg:.4f}  (eos_coef={eos_coef})")
        print(f"L1 손실 평균           = {loss_l1 / n_box:.4f}")
        print(f"1-GIoU 손실 평균       = {loss_giou / n_box:.4f}")
        print(f"총 손실 (1,5,2 가중)   = {total:.4f}")
    return total


def main():
    np.set_printoptions(precision=3, suppress=True)

    # 번호판 단일 클래스: 인덱스 0 = 번호판, 1 = 배경
    pred_logits = np.array([[3.2, -1.0],
                            [-1.5, 2.0],
                            [-0.5, 1.2],
                            [0.4, 0.6],
                            [2.8, -0.7],
                            [-0.2, 1.0]])
    pred_boxes = np.array([[0.32, 0.60, 0.15, 0.06],
                          [0.10, 0.20, 0.30, 0.25],
                          [0.50, 0.50, 0.90, 0.80],
                          [0.80, 0.85, 0.08, 0.04],
                          [0.70, 0.59, 0.13, 0.05],
                          [0.45, 0.30, 0.20, 0.10]])
    tgt_boxes = np.array([[0.30, 0.62, 0.16, 0.06],
                          [0.72, 0.58, 0.12, 0.05]])
    tgt_labels = np.array([0, 0])
    matching = {0: 0, 4: 1}        # hungarian_matching.py 가 찾아낸 결과

    print("=== 기본 가중치 (cls=1, L1=5, GIoU=2) ===")
    base = detr_loss(pred_logits, pred_boxes, tgt_labels, tgt_boxes, matching)

    print("\n=== 가중치 조합별 총 손실 비교 ===")
    print(f"{'w_cls':>6s} {'w_l1':>6s} {'w_giou':>7s} {'총손실':>10s}")
    for w_cls, w_l1, w_giou in [(1, 5, 2), (1, 2, 2), (1, 5, 5),
                                (2, 5, 2), (1, 10, 2)]:
        t = detr_loss(pred_logits, pred_boxes, tgt_labels, tgt_boxes, matching,
                      w_cls=w_cls, w_l1=w_l1, w_giou=w_giou, verbose=False)
        print(f"{w_cls:>6.1f} {w_l1:>6.1f} {w_giou:>7.1f} {t:>10.4f}")

    print("\n=== 작은 번호판일 때 L1 과 GIoU 의 차이 ===")
    print(f"{'번호판 폭':>10s} {'L1':>8s} {'IoU':>7s} {'1-GIoU':>8s}")
    for w in (0.40, 0.20, 0.10, 0.05, 0.02):
        gt = np.array([0.5, 0.5, w, w * 0.35])
        pr = gt + np.array([0.01, 0.01, 0.0, 0.0])   # 중심만 1% 벗어난 예측
        l1 = float(np.abs(pr - gt).sum())
        g, iou = giou(cxcywh_to_xyxy(pr), cxcywh_to_xyxy(gt))
        print(f"{w:>10.2f} {l1:>8.4f} {iou:>7.3f} {1 - g:>8.4f}")
    print("\n동일한 L1 오차라도 박스가 작을수록 IoU 가 급격히 나빠진다.")
    print(f"기본 조합의 총 손실 = {base:.4f}")


if __name__ == "__main__":
    main()
