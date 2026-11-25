# -*- coding: utf-8 -*-
"""sin/cos 위치 인코딩(Positional Encoding)을 생성하고 값을 출력한다.

PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
"""
import numpy as np


def positional_encoding_1d(max_len, d_model, base=10000.0):
    """1차원(시퀀스) sin/cos 위치 인코딩. 반환 shape 은 (max_len, d_model)."""
    assert d_model % 2 == 0, "d_model 은 짝수여야 한다"
    pos = np.arange(max_len)[:, None].astype(np.float64)      # (max_len, 1)
    i = np.arange(d_model // 2)[None, :].astype(np.float64)   # (1, d_model/2)
    # 차원 인덱스가 커질수록 주기가 길어지는 각속도
    omega = 1.0 / (base ** (2.0 * i / d_model))
    angle = pos * omega
    pe = np.zeros((max_len, d_model), dtype=np.float64)
    pe[:, 0::2] = np.sin(angle)
    pe[:, 1::2] = np.cos(angle)
    return pe


def positional_encoding_2d(h, w, d_model, base=10000.0):
    """DETR 이 쓰는 방식과 같은 2차원 위치 인코딩.

    d_model 의 절반을 y 축, 절반을 x 축에 배분한다.
    반환 shape 은 (h * w, d_model).
    """
    assert d_model % 4 == 0, "2D 인코딩은 d_model 이 4의 배수여야 한다"
    d_half = d_model // 2
    pe_y = positional_encoding_1d(h, d_half, base)      # (h, d_half)
    pe_x = positional_encoding_1d(w, d_half, base)      # (w, d_half)
    grid = np.zeros((h, w, d_model), dtype=np.float64)
    grid[:, :, :d_half] = pe_y[:, None, :]
    grid[:, :, d_half:] = pe_x[None, :, :]
    return grid.reshape(h * w, d_model)


def main():
    np.set_printoptions(precision=3, suppress=True, linewidth=120)

    max_len, d_model = 8, 16
    pe = positional_encoding_1d(max_len, d_model)
    print(f"[1D 위치 인코딩] max_len={max_len}, d_model={d_model} -> {pe.shape}")
    print("[앞 8차원만 표시]")
    for p in range(max_len):
        print(f"  pos={p}: {pe[p, :8]}")

    print("\n[차원별 주기 확인] 짝수 차원 2i 의 파장 = 2*pi*10000^(2i/d_model)")
    for i in range(0, d_model // 2, 2):
        wave = 2 * np.pi * (10000.0 ** (2.0 * i / d_model))
        print(f"  2i={2 * i:2d} -> 파장 약 {wave:12.2f}")

    print("\n[위치 벡터의 노름] 모든 위치에서 거의 동일하다")
    print("  ", np.round(np.linalg.norm(pe, axis=1), 4))

    print("\n[인접 위치 간 코사인 유사도] 멀어질수록 감소하는 경향")
    base_vec = pe[0] / np.linalg.norm(pe[0])
    for p in range(max_len):
        v = pe[p] / np.linalg.norm(pe[p])
        print(f"  cos(pos0, pos{p}) = {float(base_vec @ v):+.4f}")

    h, w, d2 = 4, 6, 16
    pe2 = positional_encoding_2d(h, w, d2)
    print(f"\n[2D 위치 인코딩] 특징맵 {h}x{w}, d_model={d2} -> {pe2.shape}")
    print(f"  토큰 수 = {h} * {w} = {h * w}")
    print("  (0,0) 토큰 앞 8차원:", pe2[0, :8])
    print("  (0,1) 토큰 앞 8차원:", pe2[1, :8])
    print("  (1,0) 토큰 앞 8차원:", pe2[w, :8])


if __name__ == "__main__":
    main()
