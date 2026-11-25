# -*- coding: utf-8 -*-
"""YOLOS 의 Patch Token + Object Token(Detection Token) 구성과 연산량을 추정한다.

YOLOS 시퀀스 = [패치 토큰 N_patch] + [검출 토큰 N_det]
전체 시퀀스에 self-attention 을 적용하므로 비용이 (N_patch + N_det)^2 에 비례한다.
"""
import numpy as np

# YOLOS 논문의 모델 규모 (d_model, layers, heads)
YOLOS_VARIANTS = {
    "YOLOS-Ti": (192, 12, 3, 100),
    "YOLOS-S": (384, 12, 6, 100),
    "YOLOS-B": (768, 12, 12, 100),
}


def token_count(h, w, patch=16, n_det=100):
    """입력 크기로 패치 토큰 수와 전체 시퀀스 길이를 계산한다."""
    gh, gw = h // patch, w // patch
    n_patch = gh * gw
    return gh, gw, n_patch, n_patch + n_det


def attention_macs(n_tokens, d_model, n_layers):
    """self-attention 의 QK^T 와 AV 에 해당하는 곱셈-누산(MAC) 수.

    논문에서 'GFLOPs' 로 표기하는 값은 보통 이 MAC 수를 뜻한다.
    QK^T 가 N*N*d, AV 가 N*N*d 이므로 레이어당 2*N^2*d 이다.
    """
    return 2.0 * n_layers * (n_tokens ** 2) * d_model


def ffn_macs(n_tokens, d_model, n_layers, expand=4):
    """FFN 두 개 선형층의 MAC 수. 레이어당 2*expand*N*d^2 이다."""
    return 2.0 * expand * n_layers * n_tokens * d_model * d_model


def proj_macs(n_tokens, d_model, n_layers):
    """Q, K, V, O 네 개 투영의 MAC 수. 레이어당 4*N*d^2 이다."""
    return 4.0 * n_layers * n_tokens * d_model * d_model


def print_table(title, header, rows, widths):
    print(f"\n=== {title} ===")
    print(" ".join(f"{h:>{w}s}" for h, w in zip(header, widths)))
    print(" ".join("-" * w for w in widths))
    for r in rows:
        print(" ".join(f"{str(c):>{w}s}" for c, w in zip(r, widths)))


def main():
    patch, n_det = 16, 100

    # 1) 입력 크기별 토큰 구성
    rows = []
    for h, w in [(224, 224), (384, 384), (384, 640), (512, 864),
                 (672, 1120), (800, 1344)]:
        gh, gw, n_patch, n_tot = token_count(h, w, patch, n_det)
        rows.append([f"{h}x{w}", f"{gh}x{gw}", f"{n_patch:,}", f"{n_det}",
                     f"{n_tot:,}", f"{n_tot ** 2:,}"])
    print_table("입력 크기별 YOLOS 토큰 구성 (패치 16, 검출 토큰 100)",
                ["입력 HxW", "패치격자", "패치토큰", "검출토큰",
                 "시퀀스 N", "N^2"],
                rows, [11, 9, 9, 9, 10, 13])

    # 2) 시퀀스 길이에 따른 연산량 O(N^2)
    d_model, n_layers = 384, 12      # YOLOS-S 기준
    rows = []
    base = None
    for h, w in [(224, 224), (384, 384), (384, 640), (512, 864),
                 (672, 1120), (800, 1344)]:
        _, _, n_patch, n_tot = token_count(h, w, patch, n_det)
        a = attention_macs(n_tot, d_model, n_layers)
        f = ffn_macs(n_tot, d_model, n_layers)
        p = proj_macs(n_tot, d_model, n_layers)
        total = a + f + p
        if base is None:
            base = total
        rows.append([f"{h}x{w}", f"{n_tot:,}", f"{a / 1e9:.2f}",
                     f"{(f + p) / 1e9:.2f}", f"{total / 1e9:.2f}",
                     f"{100 * a / total:.1f}%", f"{total / base:.1f}x"])
    print_table(f"YOLOS-S (d={d_model}, L={n_layers}) 연산량 추정",
                ["입력", "N", "Attn GF", "선형 GF", "합계 GF",
                 "Attn비중", "기준배수"],
                rows, [11, 8, 9, 9, 9, 9, 9])

    # 3) 모델 규모별 비교 (512x864 입력 기준)
    h, w = 512, 864
    _, _, n_patch, n_tot = token_count(h, w, patch, n_det)
    rows = []
    for name, (d, L, heads, nd) in YOLOS_VARIANTS.items():
        _, _, _, n = token_count(h, w, patch, nd)
        total = (attention_macs(n, d, L) + ffn_macs(n, d, L)
                 + proj_macs(n, d, L))
        # 백본 파라미터 추정: 레이어당 (4*d^2 투영 + 8*d^2 FFN) = 12*d^2
        params = L * 12 * d * d + patch * patch * 3 * d
        rows.append([name, f"{d}", f"{L}", f"{heads}", f"{n:,}",
                     f"{total / 1e9:.1f}", f"{params / 1e6:.1f}M"])
    print_table(f"모델 규모별 비교 (입력 {h}x{w}, 시퀀스 {n_tot:,})",
                ["모델", "d_model", "레이어", "헤드", "N",
                 "GFLOPs", "파라미터"],
                rows, [10, 8, 7, 6, 8, 9, 10])

    # 4) 토큰 수가 2 배가 될 때 어텐션 비용은 4 배가 된다
    print("\n=== 시퀀스 길이 배수와 어텐션 비용 배수 ===")
    print(f"{'N 배수':>8s} {'Attn 비용 배수':>15s} {'선형 비용 배수':>15s}")
    for k in (1, 2, 3, 4, 6, 8):
        print(f"{k:>8d} {k * k:>15d} {k:>15d}")

    # 5) 검출 토큰 수가 늘 때의 영향
    print("\n=== 검출 토큰 수 변화의 영향 (입력 512x864) ===")
    _, _, n_patch, _ = token_count(h, w, patch, 0)
    print(f"{'검출토큰':>9s} {'시퀀스 N':>10s} {'N^2':>12s} {'기준대비':>9s}")
    ref = None
    for nd in (10, 50, 100, 200, 300):
        n = n_patch + nd
        if ref is None:
            ref = n * n
        print(f"{nd:>9d} {n:>10,d} {n * n:>12,d} {n * n / ref:>8.3f}x")
    print(f"\n패치 토큰이 {n_patch:,} 개이므로 검출 토큰을 늘려도 전체 비용 변화는 작다.")
    print("반대로 입력 해상도를 올리면 비용이 제곱으로 늘어난다.")
    print("번호판은 작아서 해상도를 낮추기 어렵고, 이것이 CPU 환경의 핵심 제약이다.")

    # 6) CPU 처리량 가정에 따른 지연시간 추정
    print("\n=== CPU 유효 처리량 가정별 지연시간 추정 (YOLOS-S) ===")
    print(f"{'입력':>11s} {'GFLOPs':>9s}", end="")
    for g in (5, 15, 30):
        print(f" {str(g) + ' GF/s':>11s}", end="")
    print()
    for hh, ww in [(384, 640), (512, 864), (672, 1120)]:
        _, _, _, n = token_count(hh, ww, patch, n_det)
        total = (attention_macs(n, d_model, n_layers)
                 + ffn_macs(n, d_model, n_layers)
                 + proj_macs(n, d_model, n_layers)) / 1e9
        print(f"{str(hh) + 'x' + str(ww):>11s} {total:>9.1f}", end="")
        for g in (5, 15, 30):
            print(f" {total / g * 1000:>8.0f} ms", end="")
        print()
    print("\n[주의] 위 값은 실제 측정치가 아니라 FLOPs 와 가정한 처리량으로 계산한 추정이다.")
    print("       실제 지연시간은 메모리 대역폭, 스레드 수, 라이브러리 구현에 따라 달라지므로")
    print("       반드시 대상 CPU 에서 직접 측정해 비교해야 한다.")


if __name__ == "__main__":
    main()
