# -*- coding: utf-8 -*-
"""DINO 의 DeNoising(DN) 학습용 노이즈 쿼리를 생성한다.

정답 박스에 일부러 노이즈를 섞어 디코더에 넣고, 원래 박스를 복원하도록 학습한다.
이렇게 하면 디코더가 "주변 쿼리를 정답으로 되돌리는" 능력을 빠르게 익혀
이분 매칭의 불안정성(같은 정답이 epoch 마다 다른 쿼리에 할당되는 문제)을 줄인다.

CDN(Contrastive DeNoising)은 노이즈 세기를 두 단계로 나눈다.
  - positive 그룹: lambda_1 이내의 약한 노이즈 -> 정답으로 복원해야 한다
  - negative 그룹: lambda_1 ~ lambda_2 의 강한 노이즈 -> 배경이라고 답해야 한다
"""
import numpy as np


def add_box_noise(boxes, scale, rng):
    """박스 좌표에 노이즈를 더한다.

    boxes: (Ng, 4) cxcywh 정규화 좌표
    scale: 노이즈 세기 lambda. 중심은 박스 크기의 scale/2 이내,
           크기는 1 +- scale 배 범위로 흔든다.
    반환: (Ng, 4) 노이즈 박스, 0~1 로 자른다
    """
    boxes = np.asarray(boxes, dtype=np.float64)
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

    # 중심 이동: 박스 폭/높이에 비례한 상대 이동이므로 작은 박스는 작게 움직인다
    dx = (rng.random(len(boxes)) * 2 - 1) * (scale / 2) * w
    dy = (rng.random(len(boxes)) * 2 - 1) * (scale / 2) * h
    # 크기 변형: 1 +- scale 배
    sw = 1.0 + (rng.random(len(boxes)) * 2 - 1) * scale
    sh = 1.0 + (rng.random(len(boxes)) * 2 - 1) * scale

    out = np.stack([cx + dx, cy + dy,
                    np.clip(w * sw, 1e-4, 1.0),
                    np.clip(h * sh, 1e-4, 1.0)], axis=1)
    return np.clip(out, 0.0, 1.0)


def flip_labels(labels, n_classes, flip_ratio, rng):
    """라벨을 일정 비율로 다른 클래스로 바꾼다(label noise).

    번호판 단일 클래스(n_classes=1)에서는 바꿀 다른 클래스가 없으므로
    플립이 일어나지 않는다. 이 점이 DN 의 효과를 제한한다.
    """
    labels = np.asarray(labels).copy()
    if n_classes <= 1:
        return labels, np.zeros(len(labels), dtype=bool)
    mask = rng.random(len(labels)) < flip_ratio
    for i in np.where(mask)[0]:
        choices = [c for c in range(n_classes) if c != labels[i]]
        labels[i] = rng.choice(choices)
    return labels, mask


def cxcywh_to_xyxy(b):
    b = np.asarray(b, dtype=np.float64)
    cx, cy, w, h = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=-1)


def iou_pairs(a, b):
    """같은 인덱스끼리의 IoU 를 계산한다. a, b 는 xyxy (N,4)."""
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    iw = np.clip(np.minimum(a[:, 2], b[:, 2]) - np.maximum(a[:, 0], b[:, 0]), 0, None)
    ih = np.clip(np.minimum(a[:, 3], b[:, 3]) - np.maximum(a[:, 1], b[:, 1]), 0, None)
    inter = iw * ih
    return inter / np.clip(area_a + area_b - inter, 1e-9, None)


def build_dn_queries(gt_boxes, gt_labels, n_classes,
                     dn_groups=3, lambda1=0.4, lambda2=0.8,
                     flip_ratio=0.2, seed=0):
    """CDN 방식의 노이즈 쿼리 집합을 만든다.

    반환: 그룹별 딕셔너리 리스트. 각 그룹은 positive/negative 쿼리를 모두 갖는다.
    """
    rng = np.random.default_rng(seed)
    gt_boxes = np.asarray(gt_boxes, dtype=np.float64)
    gt_labels = np.asarray(gt_labels)
    groups = []
    for g in range(dn_groups):
        # positive: 약한 노이즈. 정답 박스를 복원해야 한다
        pos_box = add_box_noise(gt_boxes, lambda1, rng)
        pos_lab, pos_flip = flip_labels(gt_labels, n_classes, flip_ratio, rng)
        # negative: 강한 노이즈. 배경이라고 답해야 한다
        neg_box = add_box_noise(gt_boxes, lambda2, rng)
        neg_lab, neg_flip = flip_labels(gt_labels, n_classes, flip_ratio, rng)
        groups.append({
            "group": g,
            "pos_boxes": pos_box, "pos_labels": pos_lab, "pos_flip": pos_flip,
            "neg_boxes": neg_box, "neg_labels": neg_lab, "neg_flip": neg_flip,
        })
    return groups


def main():
    np.set_printoptions(precision=4, suppress=True)

    # 번호판 2 개가 있는 이미지. 단일 클래스이므로 n_classes = 1
    gt_boxes = np.array([[0.30, 0.62, 0.160, 0.055],    # 12가3456
                         [0.72, 0.58, 0.085, 0.030]])   # 87로2345 (원거리, 작다)
    gt_labels = np.array([0, 0])
    n_classes = 1
    dn_groups, lambda1, lambda2 = 3, 0.4, 0.8
    n_obj_queries = 300

    print("=== 정답 박스 (cxcywh 정규화) ===")
    for i, b in enumerate(gt_boxes):
        print(f"  obj{i}: cx={b[0]:.4f} cy={b[1]:.4f} w={b[2]:.4f} h={b[3]:.4f}")

    groups = build_dn_queries(gt_boxes, gt_labels, n_classes,
                              dn_groups=dn_groups, lambda1=lambda1,
                              lambda2=lambda2, seed=11)

    gt_xyxy = cxcywh_to_xyxy(gt_boxes)
    print(f"\n=== DN 그룹 {dn_groups} 개 생성 "
          f"(lambda1={lambda1}, lambda2={lambda2}) ===")
    for gr in groups:
        print(f"\n[그룹 {gr['group']}]")
        p_iou = iou_pairs(cxcywh_to_xyxy(gr["pos_boxes"]), gt_xyxy)
        n_iou = iou_pairs(cxcywh_to_xyxy(gr["neg_boxes"]), gt_xyxy)
        for i in range(len(gt_boxes)):
            pb, nb = gr["pos_boxes"][i], gr["neg_boxes"][i]
            print(f"  obj{i} positive: {np.round(pb, 4)} "
                  f"IoU={p_iou[i]:.3f} 라벨={gr['pos_labels'][i]} "
                  f"플립={bool(gr['pos_flip'][i])}")
            print(f"  obj{i} negative: {np.round(nb, 4)} "
                  f"IoU={n_iou[i]:.3f} 라벨={gr['neg_labels'][i]} "
                  f"플립={bool(gr['neg_flip'][i])}")

    # 노이즈 세기별 평균 IoU 를 통계로 확인한다
    print("\n=== 노이즈 세기(lambda)별 평균 IoU (각 500회 시행) ===")
    print(f"{'lambda':>8s} {'큰 번호판':>12s} {'작은 번호판':>12s} {'IoU<0.5 비율':>14s}")
    rng = np.random.default_rng(99)
    for lam in (0.1, 0.2, 0.4, 0.6, 0.8, 1.0):
        ious = np.array([iou_pairs(cxcywh_to_xyxy(add_box_noise(gt_boxes, lam, rng)),
                                   gt_xyxy) for _ in range(500)])
        low = float((ious < 0.5).mean())
        print(f"{lam:>8.1f} {ious[:, 0].mean():>12.3f} "
              f"{ious[:, 1].mean():>12.3f} {low * 100:>13.1f}%")

    # 쿼리 수 증가와 디코더 비용
    n_dn = dn_groups * 2 * len(gt_boxes)
    print(f"\n=== 학습 시 디코더 입력 쿼리 수 ===")
    print(f"  객체 쿼리            : {n_obj_queries}")
    print(f"  DN 쿼리              : {dn_groups} 그룹 x 2 x {len(gt_boxes)} = {n_dn}")
    print(f"  합계                 : {n_obj_queries + n_dn}")
    print(f"  self-attention 행렬  : "
          f"{n_obj_queries + n_dn}^2 = {(n_obj_queries + n_dn) ** 2:,}")
    print(f"  (객체 쿼리만일 때     : {n_obj_queries ** 2:,}, "
          f"증가율 {((n_obj_queries + n_dn) / n_obj_queries) ** 2:.2f}배)")
    print("\n[주의] DN 쿼리는 학습에만 쓰이고 추론 시에는 제거된다.")
    print("       따라서 DN 은 추론 속도를 늘리지 않지만, 학습 시간은 늘린다.")
    print("[주의] 번호판 단일 클래스에서는 라벨 플립이 발생하지 않으므로")
    print("       DN 의 효과가 박스 좌표 노이즈 복원으로 한정된다.")


if __name__ == "__main__":
    main()
