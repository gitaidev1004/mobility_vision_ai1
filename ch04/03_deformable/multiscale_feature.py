# -*- coding: utf-8 -*-
"""다중 스케일 특징맵의 크기, 토큰 수, 연산량 추정표를 만든다.

Deformable DETR 은 백본의 C3~C5 단계와 추가 단계를 모아 L 개 스케일을 쓴다.
각 스케일의 토큰을 모두 이어붙여 하나의 시퀀스로 다루므로 총 토큰 수가 중요하다.
"""
import numpy as np


def feature_shapes(h, w, strides):
    """입력 크기와 stride 목록으로 각 스케일의 특징맵 크기를 계산한다."""
    out = []
    for s in strides:
        fh = (h + s - 1) // s        # 올림 나눗셈
        fw = (w + s - 1) // s
        out.append((s, fh, fw, fh * fw))
    return out


def print_table(title, rows, header, widths):
    print(f"\n=== {title} ===")
    print(" ".join(f"{h:>{w}s}" for h, w in zip(header, widths)))
    print(" ".join("-" * w for w in widths))
    for r in rows:
        print(" ".join(f"{str(c):>{w}s}" for c, w in zip(r, widths)))


def main():
    # 번호판 검출에서 흔히 쓰는 입력 크기
    H, W = 768, 1333
    strides = [8, 16, 32, 64]
    d_model = 256
    n_points, n_heads = 4, 8

    shapes = feature_shapes(H, W, strides)
    total_tokens = sum(s[3] for s in shapes)

    rows = []
    for s, fh, fw, n in shapes:
        rows.append([f"C{int(np.log2(s))}", f"1/{s}", f"{fh}x{fw}",
                     f"{n:,}", f"{100 * n / total_tokens:.1f}%"])
    rows.append(["합계", "-", "-", f"{total_tokens:,}", "100.0%"])
    print_table(f"입력 {H}x{W} 의 다중 스케일 특징맵",
                rows,
                ["단계", "stride", "크기 HxW", "토큰 수", "비중"],
                [6, 8, 12, 10, 8])

    # 조밀 어텐션과 Deformable 어텐션의 연산 규모 비교
    dense = total_tokens * total_tokens
    sparse = total_tokens * n_heads * n_points * len(strides)
    rows = [
        ["총 토큰 N", f"{total_tokens:,}"],
        ["조밀 어텐션 N^2", f"{dense:,}"],
        ["Deformable 샘플 수", f"{sparse:,}"],
        ["절감 비율", f"{dense / sparse:,.0f} 배"],
        ["조밀 어텐션 행렬 메모리(fp32)", f"{dense * 4 / 1e9:.2f} GB"],
        ["Deformable 샘플 메모리(fp32)", f"{sparse * 4 / 1e6:.2f} MB"],
    ]
    print_table(f"어텐션 비용 (L={len(strides)}, heads={n_heads}, K={n_points})",
                rows, ["항목", "값"], [32, 16])

    # 입력 크기별 총 토큰 수 변화
    rows = []
    for h, w in [(384, 640), (512, 854), (640, 1067), (768, 1333), (1024, 1707)]:
        sh = feature_shapes(h, w, strides)
        tot = sum(x[3] for x in sh)
        single = feature_shapes(h, w, [32])[0][3]   # DETR 은 1/32 단일 스케일
        rows.append([f"{h}x{w}", f"{single:,}", f"{tot:,}",
                     f"{tot / single:.1f}x",
                     f"{tot * tot * 4 / 1e9:.1f} GB"])
    print_table("입력 크기별 토큰 수와 조밀 어텐션 메모리",
                rows,
                ["입력 HxW", "DETR(1/32)", "다중스케일", "배수", "N^2 메모리"],
                [12, 12, 12, 8, 12])

    # 번호판 크기가 각 스케일에서 몇 픽셀인지 확인한다
    print("\n=== 번호판 크기가 스케일별로 차지하는 픽셀 ===")
    print(f"{'원본 번호판':>14s}", end="")
    for s in strides:
        print(f" {'1/' + str(s):>12s}", end="")
    print()
    for pw, ph in [(260, 90), (130, 45), (64, 22), (32, 11), (16, 6)]:
        print(f"{str(pw) + 'x' + str(ph):>14s}", end="")
        for s in strides:
            print(f" {pw / s:>5.1f}x{ph / s:<6.1f}", end="")
        print()
    print("\n한 변이 1 픽셀 미만이면 그 스케일에서는 번호판 형태 정보가 사라진다.")
    print("원거리 차량의 작은 번호판은 stride 8 특징맵에서만 의미 있는 크기로 남는다.")

    # d_model 에 따른 인코더 레이어 파라미터 추정
    print("\n=== 인코더 1 개 레이어의 주요 파라미터 (d_model=%d) ===" % d_model)
    attn_proj = 4 * d_model * d_model
    samp_off = d_model * (n_heads * len(strides) * n_points * 2)
    attn_w = d_model * (n_heads * len(strides) * n_points)
    ffn = 2 * d_model * (d_model * 4)
    rows = [
        ["value/output 투영", f"{2 * d_model * d_model:,}"],
        ["샘플링 오프셋 예측", f"{samp_off:,}"],
        ["어텐션 가중치 예측", f"{attn_w:,}"],
        ["FFN (확장 4배)", f"{ffn:,}"],
        ["레이어 합계(근사)", f"{2 * d_model * d_model + samp_off + attn_w + ffn:,}"],
    ]
    print_table("파라미터 추정", rows, ["구성", "파라미터 수"], [24, 16])
    del attn_proj


if __name__ == "__main__":
    main()
