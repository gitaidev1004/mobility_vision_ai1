# -*- coding: utf-8 -*-
"""'ViT-YOLO' 라는 이름으로 공개된 구현체 유형별 규모를 추정해 비교한다.

중요: ViT-YOLO 는 표준 아키텍처 이름이 아니다. 같은 이름을 쓰더라도
  (A) 백본만 Transformer 로 바꾼 것
  (B) 헤드(또는 넥)까지 Transformer 블록을 넣은 것
  (C) CNN 과 Transformer 를 섞은 하이브리드
처럼 구조가 전혀 다르다. 따라서 이름만으로 정확도나 속도를 비교할 수 없다.
이 파일은 유형별 '규모 차이'가 얼마나 큰지 산술로 보여 주기 위한 추정기다.
실제 값은 반드시 지정한 저장소의 코드와 체크포인트로 다시 측정해야 한다.
"""
from __future__ import annotations

IMG = 640
OUT_CH = 256


def conv_params(cin: int, cout: int, k: int) -> int:
    return cin * cout * k * k + cout


def conv_gmul(cin: int, cout: int, k: int, side: int) -> float:
    return cin * cout * k * k * side * side / 1e9


def cnn_backbone(width: float, depth: float) -> tuple[float, float]:
    """CSP 계열 경량 CNN 백본의 파라미터(M)·연산량(GMUL) 어림 추정."""
    chans = [int(64 * width), int(128 * width), int(256 * width), int(512 * width)]
    blocks = [max(1, int(3 * depth)), max(1, int(6 * depth)),
              max(1, int(6 * depth)), max(1, int(3 * depth))]
    p = conv_params(3, chans[0], 3)
    g = conv_gmul(3, chans[0], 3, IMG // 2)
    cin = chans[0]
    for i, c in enumerate(chans):
        side = IMG // (2 ** (i + 2))
        p += conv_params(cin, c, 3)
        g += conv_gmul(cin, c, 3, side)
        for _ in range(blocks[i]):
            p += 2 * conv_params(c, c, 3)
            g += 2 * conv_gmul(c, c, 3, side)
        cin = c
    return p / 1e6, g


def vit_backbone(dim: int, layers: int, patch: int) -> tuple[float, float]:
    """ViT 백본의 파라미터(M)·연산량(GMUL) 어림 추정."""
    side = IMG // patch
    n = side * side
    embed = 3 * patch * patch * dim + dim
    per = 4 * dim * dim + 2 * (dim * 4 * dim) + 9 * dim
    p = embed + layers * per
    attn = 2 * n * n * dim * layers
    proj = n * (4 * dim * dim + 2 * 4 * dim * dim) * layers
    return p / 1e6, (attn + proj + n * embed) / 1e9


def transformer_head(dim: int, blocks: int, levels: tuple[int, ...]) -> tuple[float, float]:
    """넥/헤드에 들어간 Transformer 블록의 비용. levels 는 특징 격자 변 길이."""
    per = 4 * dim * dim + 2 * (dim * 4 * dim) + 9 * dim
    p = blocks * per
    g = 0.0
    for side in levels:
        n = side * side
        g += blocks * (2 * n * n * dim + n * (4 * dim * dim + 2 * 4 * dim * dim))
    return p / 1e6, g / 1e9


def conv_head(levels: tuple[int, ...], ch: int, n_cls: int = 1) -> tuple[float, float]:
    """YOLO 계열 합성곱 검출 헤드(디커플드 헤드 2분기) 비용."""
    p = 0
    g = 0.0
    for side in levels:
        p += conv_params(ch, ch, 3) * 4
        g += conv_gmul(ch, ch, 3, side) * 4
        p += conv_params(ch, 4 * 16, 1) + conv_params(ch, n_cls, 1)
        g += conv_gmul(ch, 4 * 16, 1, side) + conv_gmul(ch, n_cls, 1, side)
    return p / 1e6, g


def fpn_cost(in_chs: tuple[int, ...], levels: tuple[int, ...]) -> tuple[float, float]:
    """측면 1x1 + 출력 3x3 로 구성된 표준 FPN 비용."""
    p = 0
    g = 0.0
    for c, side in zip(in_chs, levels):
        p += conv_params(c, OUT_CH, 1) + conv_params(OUT_CH, OUT_CH, 3)
        g += conv_gmul(c, OUT_CH, 1, side) + conv_gmul(OUT_CH, OUT_CH, 3, side)
    return p / 1e6, g


LEVELS = (80, 40, 20)            # P3, P4, P5 격자 변 길이 (입력 640)


def variants() -> list[dict]:
    """유형별 대표 구성을 만든다. 모든 숫자는 '추정'이다."""
    out: list[dict] = []

    # 기준선: 순수 경량 CNN YOLO (비교용)
    bp, bg = cnn_backbone(0.25, 0.33)
    fp, fg = fpn_cost((64, 128, 256), LEVELS)
    hp, hg = conv_head(LEVELS, OUT_CH)
    out.append(dict(name="기준선 경량 YOLO(n)", kind="순수 CNN",
                    backbone="CSP-nano", head="합성곱 디커플드",
                    params=bp + fp + hp, gmul=bg + fg + hg,
                    note="4장 비교의 기준선"))

    bp, bg = cnn_backbone(0.5, 0.33)
    out.append(dict(name="기준선 경량 YOLO(s)", kind="순수 CNN",
                    backbone="CSP-small", head="합성곱 디커플드",
                    params=bp + fp + hp, gmul=bg + fg + hg,
                    note="정확도 여유를 둔 기준선"))

    # (A) 백본만 ViT
    for tag, dim, layers in (("Tiny", 192, 12), ("Small", 384, 12), ("Base", 768, 12)):
        vp, vg = vit_backbone(dim, layers, 16)
        fp, fg = fpn_cost((dim, dim, dim), LEVELS)
        hp, hg = conv_head(LEVELS, OUT_CH)
        out.append(dict(name=f"(A) ViT-{tag} 백본 + YOLO 헤드", kind="백본만 ViT",
                        backbone=f"ViT-{tag}", head="합성곱 디커플드",
                        params=vp + fp + hp, gmul=vg + fg + hg,
                        note="헤드는 CNN 그대로, 후처리도 YOLO 와 동일"))

    # (B) 헤드까지 Transformer
    for tag, dim, layers, hb in (("Tiny", 192, 12, 3), ("Small", 384, 12, 3)):
        vp, vg = vit_backbone(dim, layers, 16)
        tp, tg = transformer_head(OUT_CH, hb, LEVELS)
        fp, fg = fpn_cost((dim, dim, dim), LEVELS)
        hp, hg = conv_head(LEVELS, OUT_CH)
        out.append(dict(name=f"(B) ViT-{tag} 백본 + Transformer 헤드",
                        kind="헤드까지 Transformer",
                        backbone=f"ViT-{tag}", head=f"Transformer x{hb} + 합성곱",
                        params=vp + fp + tp + hp, gmul=vg + fg + tg + hg,
                        note="넥/헤드의 전역 어텐션이 얕은 레벨에서 비싸다"))

    # (C) 하이브리드: CNN 백본 + 가장 깊은 레벨에만 Transformer 블록
    for tag, width, dim, hb in (("nano", 0.25, 256, 1), ("small", 0.5, 256, 2)):
        bp, bg = cnn_backbone(width, 0.33)
        tp, tg = transformer_head(dim, hb, (20,))      # P5 에만 적용
        fp, fg = fpn_cost((int(64 * width * 4), int(128 * width * 4),
                           int(256 * width * 4)), LEVELS)
        hp, hg = conv_head(LEVELS, OUT_CH)
        out.append(dict(name=f"(C) CNN({tag}) + P5 Transformer x{hb}",
                        kind="하이브리드",
                        backbone=f"CSP-{tag}", head=f"P5 Transformer x{hb} + 합성곱",
                        params=bp + fp + tp + hp, gmul=bg + fg + tg + hg,
                        note="가장 작은 격자에만 어텐션을 넣어 비용을 억제"))
    return out


def main() -> None:
    rows = variants()
    base = rows[0]
    print("=" * 104)
    print("'ViT-YOLO' 라는 이름이 가리킬 수 있는 구현체 유형별 규모 추정 (입력 640x640)")
    print("주의: 아래 값은 구조식에서 계산한 추정치다. 실제 저장소의 수치가 아니다.")
    print("=" * 104)
    print(f"{'구성':<40} {'유형':<18} {'파라미터(M)':>11} "
          f"{'연산(GMUL)':>11} {'기준선 대비':>10}")
    print("-" * 104)
    for r in rows:
        ratio = r["gmul"] / base["gmul"]
        print(f"{r['name']:<40} {r['kind']:<18} {r['params']:>11.2f} "
              f"{r['gmul']:>11.2f} {ratio:>9.1f}x")
    print("-" * 104)

    print()
    print("[유형별 특성 요약]")
    seen: set[str] = set()
    for r in rows:
        if r["kind"] in seen:
            continue
        seen.add(r["kind"])
        print(f"  - {r['kind']:<18}: {r['note']}")

    print()
    print("[결론]")
    print("  같은 'ViT-YOLO' 라는 이름 안에서도 연산량이 수십 배 차이난다.")
    print("  비교 실험을 하려면 다음 세 가지를 먼저 특정해야 한다.")
    print("    1) 저장소 URL 과 커밋 해시")
    print("    2) 모델 크기 태그(n/s/m/l 또는 Tiny/Small/Base)")
    print("    3) 입력 해상도와 헤드 구성(어텐션이 어느 레벨에 들어가는지)")
    print("  이 세 가지가 고정되지 않으면 어떤 비교표도 재현할 수 없다.")


if __name__ == "__main__":
    main()
