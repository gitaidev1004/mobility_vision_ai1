# -*- coding: utf-8 -*-
"""Swin Transformer 의 윈도우 분할과 시프트 윈도우를 numpy 로 재현한다.

이 예제는 학습된 가중치를 쓰지 않는다. 토큰을 윈도우 단위로 어떻게 나누고
어떻게 반 윈도우만큼 밀어서(shift) 다시 나누는지 인덱스로 확인하고,
전역 어텐션(Global Attention) 과 윈도우 어텐션의 연산량을 비교한다.
"""
import numpy as np


def make_token_map(h: int, w: int) -> np.ndarray:
    """H x W 격자에 0..H*W-1 의 토큰 번호를 채운 지도를 만든다."""
    return np.arange(h * w, dtype=np.int64).reshape(h, w)


def window_partition(x: np.ndarray, win: int) -> np.ndarray:
    """(H, W, C) 특징 맵을 (윈도우개수, win*win, C) 로 분할한다.

    Swin 의 구현과 같은 순서로 reshape -> transpose 를 수행한다.
    """
    h, w, c = x.shape
    if h % win or w % win:
        raise ValueError(f"해상도 {h}x{w} 가 윈도우 크기 {win} 로 나누어지지 않는다")
    x = x.reshape(h // win, win, w // win, win, c)
    x = x.transpose(0, 2, 1, 3, 4)          # (nH, nW, win, win, C)
    return x.reshape(-1, win * win, c)


def window_reverse(windows: np.ndarray, win: int, h: int, w: int) -> np.ndarray:
    """window_partition 의 역변환. (윈도우개수, win*win, C) -> (H, W, C)"""
    c = windows.shape[-1]
    x = windows.reshape(h // win, w // win, win, win, c)
    x = x.transpose(0, 2, 1, 3, 4)
    return x.reshape(h, w, c)


def cyclic_shift(x: np.ndarray, shift: int) -> np.ndarray:
    """높이/너비 축을 각각 -shift 만큼 순환 이동한다(Shifted Window 1단계)."""
    return np.roll(x, shift=(-shift, -shift), axis=(0, 1))


def attention_mask(h: int, w: int, win: int, shift: int) -> np.ndarray:
    """시프트 후 한 윈도우에 섞여 들어온 서로 다른 구역을 구분하는 마스크.

    같은 구역끼리는 0, 다른 구역끼리는 -100.0 을 넣어 softmax 에서 제외한다.
    반환 모양은 (윈도우개수, win*win, win*win) 이다.
    """
    region = np.zeros((h, w, 1), dtype=np.int64)
    cnt = 0
    hs = (slice(0, -win), slice(-win, -shift), slice(-shift, None))
    ws = (slice(0, -win), slice(-win, -shift), slice(-shift, None))
    for a in hs:
        for b in ws:
            region[a, b, 0] = cnt
            cnt += 1
    rw = window_partition(region, win)[..., 0]          # (nWin, win*win)
    diff = rw[:, :, None] - rw[:, None, :]
    return np.where(diff == 0, 0.0, -100.0)


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def window_attention(tokens: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """윈도우 단위 Self-Attention 1회. tokens: (nWin, L, C)

    가중치가 없으므로 Q=K=V=tokens 로 두는 단순화된 형태를 쓴다.
    여기서 확인할 것은 수치가 아니라 '어텐션이 윈도우 안에서만 일어난다'는 점이다.
    """
    c = tokens.shape[-1]
    score = tokens @ tokens.transpose(0, 2, 1) / np.sqrt(c)
    if mask is not None:
        score = score + mask
    return softmax(score) @ tokens


def flops_global(n: int, c: int) -> int:
    """전역 어텐션의 QK^T + AV 곱셈 횟수(어림). 토큰 수 n 에 제곱 비례한다."""
    return 2 * n * n * c


def flops_window(n: int, c: int, win: int) -> int:
    """윈도우 어텐션의 곱셈 횟수(어림). 토큰 수 n 에 선형 비례한다."""
    return 2 * n * (win * win) * c


def main() -> None:
    h = w = 8
    win = 4
    shift = win // 2

    print("=" * 62)
    print(f"[1] 토큰 지도 {h}x{w} (총 {h * w} 토큰), 윈도우 {win}x{win}")
    print("=" * 62)
    tmap = make_token_map(h, w)
    print(tmap)

    x = tmap[:, :, None].astype(np.float64)       # (H, W, 1) 토큰 번호를 값으로 사용
    print()
    print("[2] 윈도우 분할 결과 (각 행이 하나의 윈도우, 값은 토큰 번호)")
    wins = window_partition(x, win)
    for i, blk in enumerate(wins[..., 0].astype(int)):
        print(f"  W-SA  윈도우 {i}: {blk.tolist()}")

    print()
    print(f"[3] 순환 이동 {shift} 픽셀 후 토큰 지도 (Shifted Window)")
    xs = cyclic_shift(x, shift)
    print(xs[:, :, 0].astype(int))

    print()
    print("[4] 시프트 후 윈도우 분할 결과")
    swins = window_partition(xs, win)
    for i, blk in enumerate(swins[..., 0].astype(int)):
        print(f"  SW-SA 윈도우 {i}: {blk.tolist()}")

    mask = attention_mask(h, w, win, shift)
    print()
    print("[5] 어텐션 마스크 통계 (0=허용, -100=차단)")
    for i in range(mask.shape[0]):
        blocked = int((mask[i] < 0).sum())
        total = mask[i].size
        print(f"  윈도우 {i}: 차단 쌍 {blocked:3d} / 전체 쌍 {total:3d}")

    out_w = window_attention(wins)
    out_sw = window_attention(swins, mask)
    print()
    print("[6] 어텐션 출력 모양 확인")
    print(f"  W-SA  출력: {out_w.shape} -> 복원 {window_reverse(out_w, win, h, w).shape}")
    print(f"  SW-SA 출력: {out_sw.shape}")

    print()
    print("[7] 연산량 비교 (C=96 가정, 곱셈 횟수 어림값)")
    print(f"  {'해상도':>10} {'토큰수':>8} {'전역(G)':>14} {'윈도우(G)':>14} {'배율':>8}")
    c = 96
    for res in (56, 112, 160, 320, 640):
        n = res * res
        g = flops_global(n, c)
        wv = flops_window(n, c, 7)
        print(f"  {res:>7}x{res:<3} {n:>8d} {g / 1e9:>13.2f} {wv / 1e9:>13.2f} {g / wv:>7.1f}x")
    print()
    print("  윈도우 크기를 7 로 고정하면 토큰 수가 늘어도 연산량은 선형으로만 늘어난다.")


if __name__ == "__main__":
    main()
