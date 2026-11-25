# -*- coding: utf-8 -*-
"""Swin Transformer 의 단계별 해상도/채널/토큰 수를 계산해 표로 출력한다.

Swin-T / Swin-S / Swin-B 의 공개 설정값을 그대로 넣고, 입력 해상도만 바꾸어
각 단계(Stage)의 토큰 수와 어텐션 비용을 비교한다. 가중치 파일이 필요 없다.
"""
from __future__ import annotations

# (이름, 임베딩 차원 C, 단계별 블록 수, 윈도우 크기, 헤드 수)
SWIN_CONFIGS = {
    "Swin-T": dict(embed_dim=96,  depths=(2, 2, 6, 2),  window=7, heads=(3, 6, 12, 24)),
    "Swin-S": dict(embed_dim=96,  depths=(2, 2, 18, 2), window=7, heads=(3, 6, 12, 24)),
    "Swin-B": dict(embed_dim=128, depths=(2, 2, 18, 2), window=7, heads=(4, 8, 16, 32)),
}

PATCH = 4          # 패치 임베딩 크기. 1단계 스트라이드는 4 가 된다.


def stage_table(name: str, img: int) -> list[list[str]]:
    """한 모델/한 입력 해상도에 대한 단계별 계산표를 만든다."""
    cfg = SWIN_CONFIGS[name]
    rows: list[list[str]] = []
    for s in range(4):
        stride = PATCH * (2 ** s)                 # 4, 8, 16, 32
        res = img // stride
        ch = cfg["embed_dim"] * (2 ** s)          # C, 2C, 4C, 8C
        tokens = res * res
        win = cfg["window"]
        # 윈도우가 해상도보다 크면 Swin 구현은 해상도 크기로 줄인다.
        eff_win = min(win, res)
        n_win = (res // eff_win) ** 2 if eff_win else 0
        # 윈도우 어텐션의 QK^T + AV 곱셈 횟수(어림)
        attn = 2 * tokens * (eff_win * eff_win) * ch * cfg["depths"][s]
        rows.append([
            f"Stage {s + 1}",
            f"1/{stride}",
            f"{res}x{res}",
            str(ch),
            f"{tokens:,}",
            f"{eff_win}x{eff_win}",
            str(n_win),
            str(cfg["depths"][s]),
            f"{attn / 1e9:.2f}",
        ])
    return rows


HEADER = ["단계", "스트라이드", "해상도", "채널", "토큰수",
          "윈도우", "윈도우수", "블록수", "어텐션(G)"]


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
    for img in (224, 640):
        for name in SWIN_CONFIGS:
            rows = stage_table(name, img)
            total = sum(float(r[-1]) for r in rows)
            print_table(f"[{name}] 입력 {img}x{img} — 단계별 형상", HEADER, rows)
            print(f"  윈도우 어텐션 합계: 약 {total:.2f} GMUL "
                  f"(QK^T + AV 만 계산, FFN/투영 제외)")

    print()
    print("=" * 70)
    print("검출 헤드가 쓰는 특징 피라미드 (Swin-T, 입력 640)")
    print("=" * 70)
    cfg = SWIN_CONFIGS["Swin-T"]
    names = ["P2", "P3", "P4", "P5"]
    for s in range(4):
        stride = PATCH * (2 ** s)
        res = 640 // stride
        ch = cfg["embed_dim"] * (2 ** s)
        px = stride                     # 이 단계에서 1 토큰이 담당하는 원본 픽셀 변
        print(f"  {names[s]}: 해상도 {res:3d}x{res:<3d}  채널 {ch:4d}  "
              f"1토큰 = 원본 {px}x{px} 픽셀")
    print()
    print("  번호판이 원본에서 가로 40픽셀이면 P3(1/8)에서는 약 5토큰 폭,")
    print("  P5(1/32)에서는 약 1토큰 폭만 남는다. 작은 번호판은 얕은 단계가 중요하다.")


if __name__ == "__main__":
    main()
