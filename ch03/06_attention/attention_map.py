# -*- coding: utf-8 -*-
"""어텐션 가중치 시각화 — 문자별로 번호판의 어느 위치를 봤는지 확인한다.

Attention OCR 이 CRNN + CTC 보다 디버깅하기 쉬운 가장 큰 이유가 이것이다.
출력 문자 t 에 대한 가중치 벡터 a_t 는 인코더 시퀀스 길이 T 와 같으므로,
a_t 를 원본 이미지 가로 폭으로 되돌리면 '이 글자를 읽을 때 본 영역'이 된다.

출력물
  (1) 터미널 ASCII 히트맵     — 외부 라이브러리 없이 바로 확인
  (2) attention_map.png       — matplotlib 이 있으면 이미지로 저장
  (3) 진단 리포트             — 단조성(monotonicity), 집중도(entropy)

matplotlib 이 없으면 (1)(3)만 수행하고 정상 종료한다.
"""
from __future__ import annotations

import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

try:
    import torch
except ImportError:
    print("[안내] PyTorch 가 설치되어 있지 않다.")
    print("       pip install torch --index-url "
          "https://download.pytorch.org/whl/cpu")
    raise SystemExit(0)

from attention_ocr import MAX_LEN, AttentionOCR, encode_label  # noqa: E402

RAMP = " .:-=+*#%@"      # 가중치를 농도 문자로 바꾸는 10단계 램프


def ascii_heatmap(weights: list[list[float]], labels: list[str],
                  width: int = 40) -> str:
    """어텐션 행렬을 터미널용 ASCII 히트맵 문자열로 만든다.

    weights 는 (출력 문자 수) x (인코더 시퀀스 길이) 행렬이다.
    각 행을 자기 행의 최댓값으로 정규화해 상대적인 집중 위치를 보인다.
    """
    lines = []
    ruler = "".join(str((i // 10) % 10) if i % 10 == 0 else "."
                    for i in range(width))
    lines.append("        " + ruler + "   <- 인코더 열(가로 위치)")
    for ch, row in zip(labels, weights):
        m = max(row) or 1.0
        bar = "".join(RAMP[min(9, int(v / m * 9.999))] for v in row[:width])
        peak = row.index(max(row))
        lines.append(f"  {ch:<5s} |{bar}| peak={peak:>2d}")
    return "\n".join(lines)


def peak_positions(weights: list[list[float]]) -> list[int]:
    """각 출력 스텝의 어텐션 최댓값 위치."""
    return [row.index(max(row)) for row in weights]


def monotonicity(peaks: list[int]) -> float:
    """단조성 점수. 번호판은 왼쪽에서 오른쪽으로 읽으므로 1.0 에 가까워야 한다.

    연속한 두 peak 가 증가하거나 같으면 정상으로 센다.
    값이 낮으면 어텐션이 앞뒤로 튀고 있다는 뜻이고, 이는 보통
    문자 간격이 좁거나 학습이 부족할 때 나타난다.
    """
    if len(peaks) < 2:
        return 1.0
    ok = sum(1 for a, b in zip(peaks, peaks[1:]) if b >= a)
    return ok / (len(peaks) - 1)


def entropy(row: list[float]) -> float:
    """가중치 분포의 엔트로피. 낮을수록 한 곳에 집중한 것이다."""
    s = sum(row) or 1.0
    h = 0.0
    for v in row:
        p = v / s
        if p > 1e-12:
            h -= p * math.log(p)
    return h


def coverage(weights: list[list[float]]) -> float:
    """커버리지. 인코더 열 중 어디도 보지 않은 비율의 보수(補數)다.

    1.0 에 가까우면 번호판 전 영역을 고르게 활용한 것이고,
    낮으면 일부 구간을 통째로 무시한 것이다(글자 누락의 전조).
    """
    if not weights:
        return 0.0
    t = len(weights[0])
    seen = [max(w[i] for w in weights) for i in range(t)]
    m = max(seen) or 1.0
    return sum(1 for v in seen if v / m > 0.05) / t


def save_png(weights: list[list[float]], labels: list[str],
             path: str = "attention_map.png") -> bool:
    """matplotlib 이 있으면 히트맵 PNG 로 저장한다. 없으면 False."""
    try:
        import matplotlib
        matplotlib.use("Agg")           # 화면 없는 환경에서도 저장되게 한다
        import matplotlib.pyplot as plt
    except ImportError:
        print("[안내] matplotlib 이 없어 PNG 저장을 건너뛴다.")
        print("       pip install matplotlib")
        return False
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(labels) + 1.6))
    ax.imshow(weights, aspect="auto", cmap="viridis",
              interpolation="nearest")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("encoder column (horizontal position)")
    ax.set_ylabel("decoded character")
    ax.set_title("Attention weights per decoded character")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"어텐션 히트맵을 저장했다: {path}")
    return True


def collect(model: AttentionOCR, img: "torch.Tensor",
            label: str | None = None):
    """모델에서 (문자 목록, 어텐션 행렬)을 뽑는다.

    label 이 주어지면 교사 강요로 전개해 '정답 문자를 읽을 때의 어텐션'을
    본다. 학습이 끝나지 않은 모델을 분석할 때 이 방식이 훨씬 유용하다.
    """
    model.eval()
    with torch.no_grad():
        if label:
            tgt = torch.tensor([encode_label(label)])
            _logits, attn = model(img, tgt, True)
            labels = list(label) + ["<eos>"]
        else:
            _logits, attn = model(img, None, False)
            labels = [f"t{i}" for i in range(MAX_LEN)]
    w = attn[0].tolist()
    n = min(len(labels), len(w))
    return labels[:n], w[:n]


def _demo() -> None:
    torch.manual_seed(7)
    model = AttentionOCR()

    ckpt = os.path.join(_HERE, "attention_ocr.pt")
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        print(f"학습된 가중치를 불러왔다: {ckpt}")
    else:
        print(f"[안내] 가중치 파일이 없다: {ckpt}")
        print("       랜덤 초기화 상태로 시각화 경로만 점검한다.")
        print("       학습 후에는 어텐션이 왼쪽에서 오른쪽으로 "
              "계단처럼 이동해야 정상이다.")

    img = torch.randn(1, 1, 32, 160)
    labels, weights = collect(model, img, "12가3456")

    print("\n=== 문자별 어텐션 ASCII 히트맵 ===")
    print(ascii_heatmap(weights, labels))

    peaks = peak_positions(weights)
    print("\n=== 진단 리포트 ===")
    print(f"출력 문자 수      : {len(labels)}")
    print(f"인코더 시퀀스 길이: {len(weights[0])}")
    print(f"peak 위치         : {peaks}")
    print(f"단조성(monotonic) : {monotonicity(peaks):.3f}  "
          f"(1.0 에 가까울수록 정상)")
    print(f"커버리지          : {coverage(weights):.3f}  "
          f"(무시된 구간이 없는지)")
    print(f"평균 엔트로피     : "
          f"{sum(entropy(r) for r in weights) / len(weights):.3f}  "
          f"(균등 분포 = {math.log(len(weights[0])):.3f}, 낮을수록 집중)")

    save_png(weights, labels, os.path.join(_HERE, "attention_map.png"))

    print("\n판독 요령")
    print(" - 단조성이 0.8 미만이면 어텐션이 앞뒤로 튀고 있다. "
          "문자 간격 증강이나 학습 추가가 필요하다.")
    print(" - 특정 문자의 엔트로피만 크게 높으면 그 문자의 "
          "학습 데이터가 부족하다는 신호다.")
    print(" - 커버리지가 낮으면 번호판 일부를 무시하므로 "
          "글자 누락(삭제 오류)이 늘어난다.")


if __name__ == "__main__":
    _demo()
