# -*- coding: utf-8 -*-
"""DETR 의 이분 매칭(Bipartite Matching) 비용행렬 구성과 헝가리안 알고리즘.

scipy 가 없어도 동작하도록 간이 구현(완전탐색 + 탐욕 보정)을 함께 제공한다.
"""
import itertools

import numpy as np


# ---------------------------------------------------------------------------
# 1. 박스 유틸리티
# ---------------------------------------------------------------------------
def cxcywh_to_xyxy(b):
    """(cx, cy, w, h) 정규화 좌표를 (x1, y1, x2, y2) 로 바꾼다."""
    b = np.asarray(b, dtype=np.float64)
    cx, cy, w, h = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=-1)


def pairwise_iou(a, b):
    """a: (N,4), b: (M,4) xyxy -> IoU (N,M) 과 합집합 면적 (N,M)."""
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.clip(union, 1e-9, None), union


def pairwise_giou(a, b):
    """Generalized IoU. GIoU = IoU - (C - U) / C, C 는 최소 포함 박스 면적."""
    iou, union = pairwise_iou(a, b)
    lt = np.minimum(a[:, None, :2], b[None, :, :2])
    rb = np.maximum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    area_c = wh[..., 0] * wh[..., 1]
    return iou - (area_c - union) / np.clip(area_c, 1e-9, None)


# ---------------------------------------------------------------------------
# 2. 비용행렬 구성
# ---------------------------------------------------------------------------
def build_cost_matrix(pred_logits, pred_boxes, tgt_labels, tgt_boxes,
                      w_cls=1.0, w_l1=5.0, w_giou=2.0):
    """DETR 매칭 비용행렬을 만든다.

    cost(i, j) = -w_cls * p_i(c_j) + w_l1 * ||b_i - b_j||_1 - w_giou * GIoU(b_i, b_j)

    pred_logits: (Nq, n_class+1)  마지막 열은 배경(no-object)
    pred_boxes : (Nq, 4) cxcywh 정규화
    tgt_labels : (Ng,)   정답 클래스 인덱스
    tgt_boxes  : (Ng, 4) cxcywh 정규화
    """
    # softmax 로 클래스 확률을 구한다
    z = pred_logits - pred_logits.max(axis=1, keepdims=True)
    prob = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)

    # 분류 비용: 정답 클래스의 확률이 높을수록 비용이 낮아진다
    cost_cls = -prob[:, tgt_labels]                                 # (Nq, Ng)
    # L1 비용: cxcywh 좌표의 절댓값 차이 합
    cost_l1 = np.abs(pred_boxes[:, None, :] - tgt_boxes[None, :, :]).sum(-1)
    # GIoU 비용: 겹침이 클수록 비용이 낮아진다
    cost_giou = -pairwise_giou(cxcywh_to_xyxy(pred_boxes),
                               cxcywh_to_xyxy(tgt_boxes))
    total = w_cls * cost_cls + w_l1 * cost_l1 + w_giou * cost_giou
    return total, cost_cls, cost_l1, cost_giou


# ---------------------------------------------------------------------------
# 3. 헝가리안 알고리즘(간이 구현)
# ---------------------------------------------------------------------------
def hungarian_bruteforce(cost):
    """행 수가 작을 때 모든 순열을 확인해 최적 매칭을 구한다.

    cost: (R, C), R >= C 를 가정한다(예측 쿼리가 정답보다 많다).
    반환: (행 인덱스 배열, 열 인덱스 배열)
    """
    R, C = cost.shape
    best, best_rows = None, None
    for rows in itertools.permutations(range(R), C):
        s = sum(cost[r, c] for c, r in enumerate(rows))
        if best is None or s < best:
            best, best_rows = s, rows
    return np.asarray(best_rows), np.arange(C), best


def hungarian_auction(cost, max_iter=200):
    """경매(auction) 방식의 간이 할당. 큰 행렬에서도 쓸 수 있다.

    열(정답)이 행(쿼리)에 입찰하며 가격을 올려 충돌을 해소한다.
    최적성이 보장되지는 않지만 DETR 규모에서 실용적인 근사를 준다.
    """
    R, C = cost.shape
    price = np.zeros(R)
    owner = -np.ones(R, dtype=int)     # 각 행을 차지한 열
    assign = -np.ones(C, dtype=int)    # 각 열이 받은 행
    eps = 1.0 / (C + 1)
    unassigned = list(range(C))
    it = 0
    while unassigned and it < max_iter:
        it += 1
        c = unassigned.pop(0)
        # 비용이 낮을수록 좋으므로 값(value) = -(cost + price)
        value = -(cost[:, c] + price)
        best = int(np.argmax(value))
        second = float(np.partition(value, -2)[-2]) if R >= 2 else float(value[best])
        bid = float(value[best]) - second + eps
        prev = owner[best]
        if prev >= 0:
            assign[prev] = -1
            unassigned.append(prev)
        owner[best] = c
        assign[c] = best
        price[best] += bid
    return assign, np.arange(C), it


def main():
    np.set_printoptions(precision=3, suppress=True, linewidth=140)
    rng = np.random.default_rng(7)

    n_query, n_class = 6, 1          # 번호판 단일 클래스 + 배경
    # 정답: 이미지 안의 번호판 두 개 (cx, cy, w, h) 정규화
    tgt_boxes = np.array([[0.30, 0.62, 0.16, 0.06],
                          [0.72, 0.58, 0.12, 0.05]])
    tgt_labels = np.array([0, 0])

    # 예측 쿼리 6 개. 0번과 4번이 정답과 가깝게 만들어 둔다
    pred_boxes = np.array([[0.32, 0.60, 0.15, 0.06],
                          [0.10, 0.20, 0.30, 0.25],
                          [0.50, 0.50, 0.90, 0.80],
                          [0.80, 0.85, 0.08, 0.04],
                          [0.70, 0.59, 0.13, 0.05],
                          [0.45, 0.30, 0.20, 0.10]])
    pred_logits = np.array([[3.2, -1.0],      # 번호판일 확률 높음
                            [-1.5, 2.0],
                            [-0.5, 1.2],
                            [0.4, 0.6],
                            [2.8, -0.7],      # 번호판일 확률 높음
                            [-0.2, 1.0]])
    assert pred_logits.shape == (n_query, n_class + 1)
    del rng

    total, c_cls, c_l1, c_giou = build_cost_matrix(
        pred_logits, pred_boxes, tgt_labels, tgt_boxes)

    print("=== 비용 성분별 행렬 (행=쿼리 6, 열=정답 2) ===")
    print("[분류 비용 -p(c)]\n", c_cls)
    print("[L1 비용]\n", c_l1)
    print("[GIoU 비용 -GIoU]\n", c_giou)
    print("[총 비용 1*cls + 5*L1 + 2*giou]\n", total)

    rows, cols, best = hungarian_bruteforce(total)
    print("\n=== 완전탐색 헝가리안 결과 ===")
    for c, r in zip(cols, rows):
        print(f"  정답 {c} <- 쿼리 {r}  (비용 {total[r, c]:+.4f})")
    print(f"  최소 총비용 = {best:+.4f}")

    a_assign, a_cols, iters = hungarian_auction(total)
    print("\n=== 경매 방식 간이 할당 결과 ===")
    for c in a_cols:
        print(f"  정답 {c} <- 쿼리 {a_assign[c]}  (비용 {total[a_assign[c], c]:+.4f})")
    print(f"  반복 횟수 = {iters}, 총비용 = "
          f"{sum(total[a_assign[c], c] for c in a_cols):+.4f}")
    print(f"  완전탐색과 동일한가? {list(a_assign) == list(rows)}")

    matched = set(int(r) for r in rows)
    bg = [q for q in range(n_query) if q not in matched]
    print(f"\n매칭되지 않은 쿼리 {bg} 는 배경(no-object)으로 학습된다.")


if __name__ == "__main__":
    main()
