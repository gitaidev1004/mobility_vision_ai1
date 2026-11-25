# -*- coding: utf-8 -*-
"""Deformable Attention 의 샘플링 좌표 생성과 bilinear 샘플링을 numpy 로 구현한다.

Deformable Attention 은 모든 위치를 보지 않고, 기준점(reference point) 주변의
K 개 위치에서만 특징을 꺼내 가중합한다.

DeformAttn(z_q, p_q, x) = sum_k A_qk * x(p_q + dp_qk)
  p_q  : 기준점 (정규화 좌표)
  dp_qk: 학습된 샘플링 오프셋
  A_qk : 학습된 어텐션 가중치 (sum_k A_qk = 1)
"""
import numpy as np


def softmax(x, axis=-1):
    z = x - x.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def bilinear_sample(fmap, pts):
    """특징맵에서 실수 좌표의 값을 bilinear 보간으로 읽는다.

    fmap: (H, W, C) 특징맵
    pts : (N, 2) 정규화 좌표 (x, y), 0~1 범위
    반환: (N, C)
    """
    H, W, C = fmap.shape
    # 정규화 좌표를 픽셀 좌표로 바꾼다 (픽셀 중심 보정)
    x = pts[:, 0] * W - 0.5
    y = pts[:, 1] * H - 0.5

    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1, y1 = x0 + 1, y0 + 1

    # 가중치는 clip 전 좌표로 계산한다
    wx1 = x - x0
    wy1 = y - y0
    wx0, wy0 = 1.0 - wx1, 1.0 - wy1

    # 경계 밖은 가장자리 값으로 고정한다 (zero padding 대신 clamp)
    x0c, x1c = np.clip(x0, 0, W - 1), np.clip(x1, 0, W - 1)
    y0c, y1c = np.clip(y0, 0, H - 1), np.clip(y1, 0, H - 1)

    v00 = fmap[y0c, x0c]      # (N, C)
    v01 = fmap[y0c, x1c]
    v10 = fmap[y1c, x0c]
    v11 = fmap[y1c, x1c]

    w00 = (wy0 * wx0)[:, None]
    w01 = (wy0 * wx1)[:, None]
    w10 = (wy1 * wx0)[:, None]
    w11 = (wy1 * wx1)[:, None]
    return v00 * w00 + v01 * w01 + v10 * w10 + v11 * w11


def make_sampling_points(ref_points, offsets):
    """기준점에 오프셋을 더해 샘플링 좌표를 만든다.

    ref_points: (Nq, 2)   정규화 기준점
    offsets   : (Nq, K, 2) 정규화 오프셋
    반환      : (Nq, K, 2) 0~1 로 자른 샘플링 좌표
    """
    pts = ref_points[:, None, :] + offsets
    return np.clip(pts, 0.0, 1.0)


def deformable_attention(fmap, ref_points, offsets, attn_logits):
    """단일 스케일 Deformable Attention 을 계산한다.

    반환: (출력 (Nq, C), 샘플링 좌표 (Nq, K, 2), 가중치 (Nq, K))
    """
    Nq, K, _ = offsets.shape
    pts = make_sampling_points(ref_points, offsets)
    weights = softmax(attn_logits, axis=-1)                   # (Nq, K)
    sampled = bilinear_sample(fmap, pts.reshape(-1, 2))       # (Nq*K, C)
    sampled = sampled.reshape(Nq, K, -1)
    out = (sampled * weights[..., None]).sum(axis=1)          # (Nq, C)
    return out, pts, weights


def main():
    np.set_printoptions(precision=3, suppress=True, linewidth=130)
    rng = np.random.default_rng(3)

    # 1) 검증용 특징맵: 채널 0 은 x 좌표, 채널 1 은 y 좌표를 그대로 담는다
    H, W, C = 12, 20, 4
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    fmap = np.zeros((H, W, C))
    fmap[:, :, 0] = (xx + 0.5) / W          # x 정규화 좌표
    fmap[:, :, 1] = (yy + 0.5) / H          # y 정규화 좌표
    fmap[:, :, 2] = rng.normal(0, 1, (H, W))
    fmap[:, :, 3] = 1.0

    print(f"[특징맵] {H} x {W} x {C}  (채널0=x좌표, 채널1=y좌표)")

    # 2) bilinear 샘플링 검증: 좌표 채널이 입력 좌표를 복원해야 한다
    probe = np.array([[0.5, 0.5], [0.025, 0.0417], [0.9, 0.8], [0.333, 0.667]])
    got = bilinear_sample(fmap, probe)
    print("\n=== bilinear 샘플링 좌표 복원 검증 ===")
    print(f"{'입력 (x, y)':>20s} {'복원 (x, y)':>20s} {'오차':>10s}")
    for p, g in zip(probe, got):
        err = abs(p[0] - g[0]) + abs(p[1] - g[1])
        print(f"  ({p[0]:.3f}, {p[1]:.3f})      "
              f"({g[0]:.3f}, {g[1]:.3f})      {err:.5f}")

    # 3) 번호판을 가리키는 기준점 2 개와 K=4 샘플링 오프셋
    Nq, K = 2, 4
    ref_points = np.array([[0.30, 0.62],     # 12가3456 번호판 위치
                           [0.72, 0.58]])    # 87로2345 번호판 위치
    # 번호판은 가로로 길기 때문에 x 방향 오프셋을 넓게 둔다
    offsets = np.array([
        [[-0.06, 0.0], [-0.02, 0.0], [0.02, 0.0], [0.06, 0.0]],
        [[-0.05, -0.02], [-0.02, 0.01], [0.02, -0.01], [0.05, 0.02]],
    ])
    attn_logits = np.array([[1.5, 0.8, 0.6, -0.4],
                            [0.2, 1.9, 0.5, 0.1]])

    out, pts, w = deformable_attention(fmap, ref_points, offsets, attn_logits)

    print("\n=== 샘플링 좌표와 어텐션 가중치 ===")
    for q in range(Nq):
        print(f"[쿼리 {q}] 기준점 ({ref_points[q, 0]:.3f}, {ref_points[q, 1]:.3f})")
        for k in range(K):
            print(f"  k={k}: 오프셋 ({offsets[q, k, 0]:+.3f}, {offsets[q, k, 1]:+.3f}) "
                  f"-> 샘플점 ({pts[q, k, 0]:.4f}, {pts[q, k, 1]:.4f}) "
                  f"가중치 {w[q, k]:.4f}")
        print(f"  가중치 합 = {w[q].sum():.4f}")
        print(f"  출력 벡터 = {out[q]}")
        print(f"  (채널0,1 은 샘플점의 가중평균 좌표: "
              f"{out[q, 0]:.4f}, {out[q, 1]:.4f})")

    # 4) 연산량 비교: 조밀 어텐션 대 희소 샘플링
    print("\n=== 조밀 어텐션 vs Deformable 샘플링 접근 횟수 ===")
    print(f"{'특징맵':>12s} {'토큰 N':>9s} {'조밀 N^2':>12s} "
          f"{'희소 N*K(4)':>13s} {'비율':>10s}")
    for h, w_ in [(12, 20), (24, 40), (48, 80), (96, 160)]:
        n = h * w_
        dense = n * n
        sparse = n * 4
        print(f"{h:>5d}x{w_:<6d} {n:>9d} {dense:>12,d} {sparse:>13,d} "
              f"{dense / sparse:>9.1f}x")

    print("\n조밀 어텐션은 토큰 수의 제곱으로 늘지만, 샘플링 수를 고정하면 선형으로 늘어난다.")


if __name__ == "__main__":
    main()
