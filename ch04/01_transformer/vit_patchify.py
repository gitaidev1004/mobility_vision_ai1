# -*- coding: utf-8 -*-
"""ViT 의 이미지 패치 분할과 임베딩 차원을 계산한다.

이미지가 없어도 동작하도록 테스트 이미지를 생성한다.
실제 이미지를 쓰려면 첫 번째 인자로 경로를 넘긴다.
"""
import sys

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


def make_sample_image(h=224, w=224):
    """번호판 비슷한 흰 사각형이 들어간 테스트 이미지를 만든다."""
    img = np.full((h, w, 3), 90, dtype=np.uint8)
    y0, y1 = int(h * 0.55), int(h * 0.70)
    x0, x1 = int(w * 0.30), int(w * 0.70)
    img[y0:y1, x0:x1] = 235                 # 번호판 바탕
    img[y0 + 4:y1 - 4, x0 + 6:x1 - 6:6] = 30  # 문자 흉내
    return img


def patchify(img, patch=16):
    """이미지를 patch x patch 격자로 자른다.

    반환: (n_patches, patch * patch * C) 모양의 평탄화된 패치 행렬
    """
    h, w, c = img.shape
    assert h % patch == 0 and w % patch == 0, "이미지 크기가 패치로 나누어져야 한다"
    gh, gw = h // patch, w // patch
    # (gh, patch, gw, patch, c) -> (gh, gw, patch, patch, c)
    x = img.reshape(gh, patch, gw, patch, c).transpose(0, 2, 1, 3, 4)
    return x.reshape(gh * gw, patch * patch * c), (gh, gw)


def linear_embed(flat_patches, d_model=768, seed=0):
    """패치 평탄화 벡터를 d_model 차원으로 선형 투영한다(랜덤 가중치)."""
    rng = np.random.default_rng(seed)
    dim_in = flat_patches.shape[1]
    W = rng.normal(0, 1.0 / np.sqrt(dim_in), (dim_in, d_model))
    b = np.zeros(d_model)
    x = flat_patches.astype(np.float32) / 255.0
    return x @ W + b, W.size + b.size


def main():
    if len(sys.argv) > 1:
        if cv2 is None:
            print("[안내] OpenCV 가 없어 이미지를 읽을 수 없다. pip install opencv-python")
            return
        img = cv2.imread(sys.argv[1])
        if img is None:
            print(f"[안내] 이미지를 열 수 없다: {sys.argv[1]}  테스트 이미지로 대체한다.")
            img = make_sample_image()
    else:
        img = make_sample_image()
        print("[안내] 인자가 없어 224x224 테스트 이미지를 생성했다.")

    # ViT 는 고정 입력 크기를 쓰므로 패치 크기의 배수로 리사이즈한다
    patch, d_model = 16, 768
    target = 224
    if img.shape[0] != target or img.shape[1] != target:
        if cv2 is not None:
            img = cv2.resize(img, (target, target), interpolation=cv2.INTER_AREA)
        else:
            img = make_sample_image(target, target)

    h, w, c = img.shape
    patches, (gh, gw) = patchify(img, patch)
    tokens, n_params = linear_embed(patches, d_model)

    print("\n=== ViT 패치 분할 결과 ===")
    print(f"입력 이미지        : {h} x {w} x {c}")
    print(f"패치 크기          : {patch} x {patch}")
    print(f"패치 격자          : {gh} x {gw}")
    print(f"패치 수 N          : {gh * gw}")
    print(f"패치 평탄화 차원   : {patch} x {patch} x {c} = {patch * patch * c}")
    print(f"임베딩 차원 d_model: {d_model}")
    print(f"패치 임베딩 shape  : {tokens.shape}")
    print(f"CLS 포함 토큰 길이 : {gh * gw + 1}")
    print(f"패치 임베딩 파라미터: {n_params:,}")

    # 어텐션 연산량은 토큰 수의 제곱에 비례한다
    n = gh * gw + 1
    attn_flops = 2 * n * n * d_model
    print(f"\n단일 어텐션 행렬 원소 수 : {n} x {n} = {n * n:,}")
    print(f"QK^T + AV 대략 FLOPs     : 2 x {n}^2 x {d_model} = {attn_flops / 1e9:.3f} GFLOPs")

    print("\n=== 입력 크기별 토큰 수 비교 (패치 16) ===")
    print(f"{'입력':>10s} {'격자':>9s} {'토큰(+CLS)':>12s} {'N^2':>12s}")
    for size in (224, 320, 448, 640, 800):
        g = size // patch
        nn = g * g + 1
        print(f"{size:>7d}px {g:>4d}x{g:<4d} {nn:>12d} {nn * nn:>12,d}")

    print("\n=== 번호판 검출용 비정사각 입력 (640x384, 패치 16) ===")
    gh2, gw2 = 384 // patch, 640 // patch
    n2 = gh2 * gw2
    print(f"격자 {gh2} x {gw2} -> 패치 토큰 {n2} 개, N^2 = {n2 * n2:,}")


if __name__ == "__main__":
    main()
