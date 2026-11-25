# -*- coding: utf-8 -*-
"""ViTDet 의 단순 특징 피라미드(Simple Feature Pyramid) 구성을 계산한다.

ViTDet 은 백본 내부에 FPN 을 두지 않고, ViT 의 마지막 단일 해상도 특징(보통 1/16)
하나만 받아서 스트라이드/업샘플 연산으로 P2~P5 를 만든다. 이 파일은 그 과정의
해상도·채널·파라미터 수·연산량을 가중치 없이 산술로만 계산해 비교한다.
"""
from __future__ import annotations

# (이름, 임베딩 차원, 레이어 수, 헤드 수, 패치 크기)
VIT_CONFIGS = {
    "ViT-Tiny":  dict(dim=192,  layers=12, heads=3,  patch=16),
    "ViT-Small": dict(dim=384,  layers=12, heads=6,  patch=16),
    "ViT-Base":  dict(dim=768,  layers=12, heads=12, patch=16),
    "ViT-Large": dict(dim=1024, layers=24, heads=16, patch=16),
}

# 단순 특징 피라미드가 만들 레벨: (이름, 최종 스트라이드, 변환 방식)
PYRAMID = [
    ("P2", 4,  "2회 전치합성곱 업샘플 (x4)"),
    ("P3", 8,  "1회 전치합성곱 업샘플 (x2)"),
    ("P4", 16, "항등 (ViT 출력 그대로)"),
    ("P5", 32, "stride 2 맥스풀 (x1/2)"),
]

OUT_CH = 256          # 검출 헤드가 받는 공통 채널 수


def vit_tokens(img: int, patch: int) -> tuple[int, int]:
    """입력 변 길이와 패치 크기로 (한 변 토큰 수, 전체 토큰 수) 를 구한다."""
    side = img // patch
    return side, side * side


def vit_params(dim: int, layers: int, patch: int = 16, in_ch: int = 3) -> int:
    """ViT 백본의 파라미터 수(어림). 패치 임베딩 + 블록 x layers."""
    embed = in_ch * patch * patch * dim + dim          # 패치 임베딩 합성곱
    attn = 4 * dim * dim + 4 * dim                     # Q,K,V,O 투영
    ffn = 2 * (dim * 4 * dim) + 4 * dim + dim          # MLP 비율 4
    norm = 4 * dim                                     # LayerNorm 2개
    return embed + layers * (attn + ffn + norm)


def vit_attn_gflops(tokens: int, dim: int, layers: int) -> float:
    """ViT 전역 어텐션의 QK^T + AV 곱셈 횟수(GMUL). 토큰 수에 제곱 비례한다."""
    return 2 * tokens * tokens * dim * layers / 1e9


def vit_ffn_gflops(tokens: int, dim: int, layers: int) -> float:
    """투영과 FFN 의 곱셈 횟수(GMUL). 토큰 수에 선형 비례한다."""
    per_layer = tokens * (4 * dim * dim) + tokens * (2 * 4 * dim * dim)
    return per_layer * layers / 1e9


def pyramid_rows(img: int, dim: int, patch: int) -> list[list[str]]:
    """단순 특징 피라미드의 각 레벨 형상과 변환 비용을 계산한다."""
    base_side, _ = vit_tokens(img, patch)
    rows: list[list[str]] = []
    for name, stride, how in PYRAMID:
        side = img // stride
        scale = base_side / side if side else 0
        # 변환 레이어 파라미터: 전치합성곱(2x2) 또는 1x1 측면 합성곱 + 3x3 출력 합성곱
        if stride < patch:                      # 업샘플 단계
            n_up = int(round(patch / stride)).bit_length() - 1
            up_p = sum((dim // (2 ** i)) * (dim // (2 ** (i + 1))) * 2 * 2
                       for i in range(n_up))
            in_c = max(dim // (2 ** n_up), 1)
        else:
            up_p = 0
            in_c = dim
        lateral = in_c * OUT_CH * 1 * 1
        out_conv = OUT_CH * OUT_CH * 3 * 3
        params = up_p + lateral + out_conv
        gmul = (up_p + lateral + out_conv) * side * side / 1e9
        rows.append([
            name, f"1/{stride}", f"{side}x{side}", str(OUT_CH),
            f"{side * side:,}", f"x{1 / scale:.2f}" if scale else "-",
            f"{params / 1e6:.2f}", f"{gmul:.2f}", how,
        ])
    return rows


HEADER = ["레벨", "스트라이드", "해상도", "출력채널", "격자수",
          "배율", "변환파라미터(M)", "변환(GMUL)", "변환 방식"]


def print_table(title: str, header: list[str], rows: list[list[str]]) -> None:
    widths = [max(len(header[i]), *(len(r[i]) for r in rows)) for i in range(len(header))]
    line = "-" * (sum(widths) + 3 * len(widths))
    print()
    print(title)
    print(line)
    print(" | ".join(h.ljust(widths[i]) for i, h in enumerate(header)))
    print(line)
    for r in rows:
        print(" | ".join(c.ljust(widths[i]) for i, c in enumerate(r)))
    print(line)


def main() -> None:
    print("=" * 78)
    print("ViTDet 단순 특징 피라미드 계산 (학습 가중치 불필요, 산술 추정만 수행)")
    print("=" * 78)

    img = 640
    print()
    print(f"[1] ViT 백본 자체 비용 (입력 {img}x{img}, 패치 16)")
    print(f"  {'모델':<10} {'토큰수':>8} {'파라미터(M)':>12} "
          f"{'어텐션(GMUL)':>14} {'FFN(GMUL)':>12} {'합계(GMUL)':>12}")
    for name, cfg in VIT_CONFIGS.items():
        _, n = vit_tokens(img, cfg["patch"])
        p = vit_params(cfg["dim"], cfg["layers"], cfg["patch"])
        a = vit_attn_gflops(n, cfg["dim"], cfg["layers"])
        f = vit_ffn_gflops(n, cfg["dim"], cfg["layers"])
        print(f"  {name:<10} {n:>8,} {p / 1e6:>12.1f} "
              f"{a:>14.1f} {f:>12.1f} {a + f:>12.1f}")

    print()
    print("  토큰 수 1,600개에서도 전역 어텐션 비용이 FFN 비용과 같은 규모로 올라간다.")
    print("  입력을 1024로 키우면 토큰 수는 4,096개가 되고 어텐션 비용은 약 6.5배가 된다.")

    for name in ("ViT-Small", "ViT-Base"):
        cfg = VIT_CONFIGS[name]
        rows = pyramid_rows(img, cfg["dim"], cfg["patch"])
        tot_p = sum(float(r[6]) for r in rows)
        tot_g = sum(float(r[7]) for r in rows)
        print_table(f"[2] {name} + 단순 특징 피라미드 (입력 {img})", HEADER, rows)
        print(f"  피라미드 변환 파라미터 합계 약 {tot_p:.2f} M, "
              f"변환 연산 합계 약 {tot_g:.2f} GMUL")

    print()
    print("=" * 78)
    print("[3] 계층형 백본(Swin)과의 차이")
    print("=" * 78)
    print("  Swin  : 백본이 1/4, 1/8, 1/16, 1/32 특징을 직접 만들어 FPN 에 넘긴다.")
    print("  ViTDet: 백본은 1/16 하나만 만들고, 피라미드는 그 뒤에서 합성해 만든다.")
    print("          구조가 단순한 대신 얕은 레벨(P2)의 세부 정보는 업샘플로 복원한다.")
    print()
    print("  번호판처럼 작고 가로로 긴 객체는 P2/P3 의 품질이 중요하므로,")
    print("  업샘플로만 만든 얕은 레벨이 충분한지 실제 데이터로 확인해야 한다.")


if __name__ == "__main__":
    main()
