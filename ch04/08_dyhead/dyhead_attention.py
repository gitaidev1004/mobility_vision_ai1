# -*- coding: utf-8 -*-
"""DyHead 의 세 가지 어텐션(scale / spatial / channel)을 numpy 로 순차 적용한다.

DyHead 는 특징 피라미드를 (레벨 L, 공간 S, 채널 C) 의 3차원 텐서로 보고
세 축에 대한 어텐션을 차례로 적용한다. 이 데모는 학습된 가중치 없이
결정적인 난수(시드 고정)로 텐서를 만들고, 각 어텐션이 무엇을 바꾸는지
수치로 확인한다. 태스크 어텐션(task attention)은 채널 어텐션의 한 형태로
구현되므로 채널 단계에서 함께 보여 준다.
"""
from __future__ import annotations

import numpy as np

RNG_SEED = 20261024


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x: np.ndarray, axis: int) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def hard_sigmoid(x: np.ndarray) -> np.ndarray:
    """DyHead 구현이 쓰는 값싼 게이트. [0,1] 로 자른다."""
    return np.clip((x + 1.0) / 2.0, 0.0, 1.0)


def build_pyramid(levels: int, side: int, ch: int) -> list[np.ndarray]:
    """L개 레벨의 특징 피라미드를 만든다. 각 레벨은 (H, W, C), 해상도는 절반씩."""
    rng = np.random.default_rng(RNG_SEED)
    feats = []
    for i in range(levels):
        s = side // (2 ** i)
        f = rng.normal(loc=0.3 * (i + 1), scale=1.0, size=(s, s, ch))
        feats.append(f)
    return feats


def align_to_median(feats: list[np.ndarray]) -> np.ndarray:
    """모든 레벨을 중간 레벨 해상도로 맞춰 (L, H, W, C) 텐서를 만든다.

    큰 레벨은 평균 풀링으로 줄이고, 작은 레벨은 최근접 이웃으로 키운다.
    DyHead 는 이렇게 정렬된 4차원 텐서 위에서 동작한다.
    """
    mid = len(feats) // 2
    th, tw = feats[mid].shape[:2]
    out = []
    for f in feats:
        h, w, c = f.shape
        if h > th:                                   # 평균 풀링으로 축소
            k = h // th
            g = f.reshape(th, k, tw, k, c).mean(axis=(1, 3))
        elif h < th:                                 # 최근접 이웃으로 확대
            k = th // h
            g = np.repeat(np.repeat(f, k, axis=0), k, axis=1)
        else:
            g = f
        out.append(g)
    return np.stack(out, axis=0)


def scale_attention(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """레벨(scale) 축 어텐션. 레벨마다 하나의 가중치를 만들어 곱한다.

    x: (L, H, W, C) -> 반환 (출력, 레벨 가중치 (L,))
    """
    pooled = x.mean(axis=(1, 2, 3))                  # (L,) 레벨별 전역 평균
    rng = np.random.default_rng(RNG_SEED + 1)
    w1 = rng.normal(scale=0.5, size=(len(pooled), len(pooled)))
    logit = w1 @ pooled
    alpha = hard_sigmoid(logit)                      # (L,)
    return x * alpha[:, None, None, None], alpha


def spatial_attention(x: np.ndarray, k: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """공간(spatial) 축 어텐션. 레벨-위치마다 가중치를 만든다.

    원 논문은 변형 합성곱(deformable convolution)으로 위치를 옮겨 가며 샘플링하지만,
    여기서는 k x k 평균 집계 + 게이트로 단순화해 '위치별 가중치'의 효과만 본다.
    반환: (출력, 가중치 (L, H, W))
    """
    l, h, w, c = x.shape
    pad = k // 2
    xp = np.pad(x, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode="edge")
    acc = np.zeros((l, h, w, c))
    for dy in range(k):
        for dx in range(k):
            acc += xp[:, dy:dy + h, dx:dx + w, :]
    acc /= k * k
    gate = hard_sigmoid(acc.mean(axis=3))            # (L, H, W)
    return x * gate[..., None], gate


def channel_attention(x: np.ndarray, groups: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """채널(channel) 축 어텐션. 채널마다 가중치를 만들어 곱한다.

    groups > 1 이면 그룹별로 다른 가중치를 만든다. 분류/회귀처럼 태스크마다
    다른 채널 묶음을 강조하는 태스크 어텐션(task attention)이 이 형태다.
    반환: (출력, 가중치 (groups, C))
    """
    l, h, w, c = x.shape
    pooled = x.mean(axis=(0, 1, 2))                  # (C,)
    rng = np.random.default_rng(RNG_SEED + 2)
    red = max(c // 4, 1)
    out = np.zeros_like(x)
    gates = np.zeros((groups, c))
    for g in range(groups):
        w1 = rng.normal(scale=0.4, size=(red, c))
        w2 = rng.normal(scale=0.4, size=(c, red))
        z = w1 @ pooled
        hidden = np.where(z > 0, z, 0.1 * z)         # LeakyReLU(0.1)
        gate = hard_sigmoid(w2 @ hidden)             # (C,)
        gates[g] = gate
        out += x * gate
    return out / groups, gates


def stat(name: str, a: np.ndarray) -> None:
    print(f"  {name:<22} shape={str(a.shape):<20} "
          f"mean={a.mean():+.4f} std={a.std():.4f} "
          f"min={a.min():+.4f} max={a.max():+.4f}")


def main() -> None:
    levels, side, ch = 4, 32, 16
    print("=" * 78)
    print("DyHead 세 가지 어텐션 순차 적용 데모 (학습 가중치 없이 시드 고정 난수 사용)")
    print("=" * 78)

    feats = build_pyramid(levels, side, ch)
    print()
    print(f"[1] 입력 특징 피라미드 ({levels} 레벨, 기준 {side}x{side}, 채널 {ch})")
    for i, f in enumerate(feats):
        print(f"  P{i + 3}: shape={f.shape}  mean={f.mean():+.4f}")

    x = align_to_median(feats)
    print()
    print("[2] 중간 레벨 해상도로 정렬한 4차원 텐서 (L, H, W, C)")
    stat("정렬 결과", x)

    x1, alpha = scale_attention(x)
    print()
    print("[3] Scale Attention — 레벨마다 하나의 가중치")
    for i, a in enumerate(alpha):
        print(f"  P{i + 3} 가중치 = {a:.4f}")
    stat("scale 적용 후", x1)

    x2, gate_s = spatial_attention(x1)
    print()
    print("[4] Spatial Attention — 레벨-위치마다 가중치")
    for i in range(levels):
        g = gate_s[i]
        print(f"  P{i + 3} 가중치 범위 [{g.min():.4f}, {g.max():.4f}] 평균 {g.mean():.4f}")
    stat("spatial 적용 후", x2)

    x3, gate_c = channel_attention(x2, groups=1)
    print()
    print("[5] Channel Attention — 채널마다 가중치")
    print("  채널 가중치:", np.array2string(gate_c[0], precision=3, max_line_width=72))
    stat("channel 적용 후", x3)

    x4, gate_t = channel_attention(x2, groups=2)
    print()
    print("[6] Task Attention — 태스크(분류/회귀)별로 다른 채널 가중치")
    for g, name in enumerate(("분류 분기", "회귀 분기")):
        top = np.argsort(-gate_t[g])[:4]
        print(f"  {name}: 상위 강조 채널 {top.tolist()} "
              f"(가중치 {np.round(gate_t[g][top], 3).tolist()})")
    cos = float(gate_t[0] @ gate_t[1] /
                (np.linalg.norm(gate_t[0]) * np.linalg.norm(gate_t[1])))
    print(f"  두 분기 가중치의 코사인 유사도 = {cos:.4f} "
          f"(1.0 에서 멀수록 분기별로 다른 채널을 본다)")
    stat("task 적용 후", x4)

    print()
    print("[7] 블록을 쌓았을 때의 변화 (scale -> spatial -> channel 를 1 블록으로 본다)")
    y = x.copy()
    for b in range(1, 7):
        y, _ = scale_attention(y)
        y, _ = spatial_attention(y)
        y, _ = channel_attention(y)
        print(f"  블록 {b}회 통과: mean={y.mean():+.6f} std={y.std():.6f} "
              f"활성 비율(|v|>0.1)={float((np.abs(y) > 0.1).mean()):.4f}")
    print()
    print("  블록을 쌓을수록 게이트가 곱으로 누적되어 신호가 줄어든다.")
    print("  실제 구현은 잔차 연결(residual)과 정규화로 이 감쇠를 막는다.")

    print()
    print("[8] 연산 비용 메모")
    l, h, w, c = x.shape
    print(f"  정렬 텐서 원소 수      = {l * h * w * c:,}")
    print(f"  scale   가중치 수      = {l}")
    print(f"  spatial 가중치 수      = {l * h * w:,}")
    print(f"  channel 가중치 수      = {c}")
    print("  어텐션 자체의 파라미터는 작지만, 모든 레벨을 같은 해상도로 정렬하는")
    print("  비용과 블록 수에 비례하는 메모리 사용량이 CPU 추론에서 부담이 된다.")


if __name__ == "__main__":
    main()
