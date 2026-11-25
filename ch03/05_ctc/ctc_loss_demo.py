# -*- coding: utf-8 -*-
"""CTC Loss 의 원리 — 입출력 길이 불일치, blank 토큰, 경로 확률 합산.

torch 가 있으면 nn.CTCLoss 와 비교하고, 없으면 넘파이 forward 알고리즘으로
직접 손실을 계산해 같은 값을 보여 준다.

실행:  python 05_ctc/ctc_loss_demo.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1] / "03_dataset"))
from charset import BLANK_INDEX, NUM_CLASSES, encode  # noqa: E402

NEG_INF = -1e30


def _logaddexp(a: float, b: float) -> float:
    """log(exp(a)+exp(b)) 를 오버플로 없이 계산한다."""
    if a <= NEG_INF:
        return b
    if b <= NEG_INF:
        return a
    hi, lo = (a, b) if a > b else (b, a)
    return hi + math.log1p(math.exp(lo - hi))


def extend_label(labels: list[int], blank: int) -> list[int]:
    """라벨 사이사이에 blank 를 끼운 확장 라벨을 만든다.

    예: [1, 2] → [blank, 1, blank, 2, blank]  (길이 2L+1)
    왜 이렇게 하는가?
      CTC 경로는 '문자 또는 blank' 를 매 프레임 내보낸다. 확장 라벨을 만들어 두면
      forward 알고리즘이 '지금 확장 라벨의 몇 번째 칸에 있는가' 만 추적하면 되고,
      blank 를 건너뛸 수 있는지 여부가 칸의 종류(blank/문자)로 바로 결정된다.
    """
    extended = [blank]
    for lab in labels:
        extended.extend([lab, blank])
    return extended


def ctc_forward_loss(log_probs: np.ndarray, labels: list[int],
                     blank: int = BLANK_INDEX) -> float:
    """CTC 손실 = -log P(라벨 | 입력) 을 forward 알고리즘으로 계산한다.

    핵심 아이디어
      같은 라벨을 만들어 내는 **모든 정렬 경로의 확률을 합산**한다.
      경로 수는 조합적으로 폭증하지만(T=40, L=7 이면 수백만 가지),
      동적 계획법으로 O(T · 2L+1) 에 계산할 수 있다.

    점화식: alpha[t][s] = (확장라벨 s 번째 칸까지 t 프레임으로 도달할 확률)
      alpha[t][s] = (alpha[t-1][s] + alpha[t-1][s-1]) * p(ext[s] | t)
      단, ext[s] 가 문자이고 ext[s] != ext[s-2] 이면 alpha[t-1][s-2] 도 더한다.
      ext[s] == ext[s-2] 인 경우(같은 문자 반복) 사이의 blank 를 건너뛰면
      두 글자가 하나로 축약되므로, 그 경로는 허용하지 않는다. 이 한 줄이
      CTC 가 '11' 과 '1' 을 구분하는 메커니즘 전부다.

    Args:
        log_probs: (T, C) 로그 확률 행렬.
        labels: 정답 문자 인덱스 리스트.
        blank: blank 인덱스.

    Returns:
        손실값(음의 로그 가능도). 정렬이 불가능하면 inf.
    """
    n_frames = log_probs.shape[0]
    ext = extend_label(labels, blank)
    n_states = len(ext)

    # T < 필요한 최소 프레임 수면 어떤 경로도 존재하지 않는다.
    min_frames = len(labels) + sum(
        1 for i in range(1, len(labels)) if labels[i] == labels[i - 1])
    if n_frames < min_frames:
        return float("inf")

    alpha = [NEG_INF] * n_states
    # 0 프레임에서 시작 가능한 칸은 맨 앞 blank 와 첫 문자 두 곳뿐이다.
    alpha[0] = float(log_probs[0, ext[0]])
    if n_states > 1:
        alpha[1] = float(log_probs[0, ext[1]])

    for t in range(1, n_frames):
        new = [NEG_INF] * n_states
        for s in range(n_states):
            total = alpha[s]                       # 제자리에 머무는 경로
            if s > 0:
                total = _logaddexp(total, alpha[s - 1])   # 한 칸 전진
            # 두 칸 건너뛰기: blank 를 생략한다. 같은 문자 반복이면 금지.
            if s > 1 and ext[s] != blank and ext[s] != ext[s - 2]:
                total = _logaddexp(total, alpha[s - 2])
            new[s] = total + float(log_probs[t, ext[s]])
        alpha = new

    # 마지막 프레임에서는 끝 blank 또는 마지막 문자 칸에 있어야 한다.
    final = _logaddexp(alpha[-1], alpha[-2] if n_states > 1 else NEG_INF)
    if final <= NEG_INF / 2.0:
        return float("inf")
    return -final


def count_alignments(n_frames: int, labels: list[int]) -> int:
    """T 프레임으로 해당 라벨을 만들 수 있는 정렬 경로의 가짓수를 센다.

    확률 대신 경우의 수를 세는 같은 점화식이다. CTC 가 몇 개의 경로를
    한꺼번에 합산하고 있는지 체감하기 위한 보조 함수다.
    """
    ext = extend_label(labels, BLANK_INDEX)
    n_states = len(ext)
    counts = [0] * n_states
    counts[0] = 1
    if n_states > 1:
        counts[1] = 1
    for _ in range(1, n_frames):
        new = [0] * n_states
        for s in range(n_states):
            total = counts[s]
            if s > 0:
                total += counts[s - 1]
            if s > 1 and ext[s] != BLANK_INDEX and ext[s] != ext[s - 2]:
                total += counts[s - 2]
            new[s] = total
        counts = new
    return counts[-1] + (counts[-2] if n_states > 1 else 0)


def make_log_probs(text: str, n_frames: int, seed: int = 5) -> np.ndarray:
    """정답에 어느 정도 맞춰진 (T, C) 로그 확률 행렬을 만든다."""
    rng = np.random.default_rng(seed)
    labels = encode(text)
    probs = np.full((n_frames, NUM_CLASSES), 1e-3)
    probs[:, BLANK_INDEX] = 0.5
    span = max(1, n_frames // max(len(labels) * 2, 1))
    pos = 1
    for lab in labels:
        for _ in range(span):
            if pos >= n_frames:
                break
            probs[pos, lab] = 0.7 + 0.2 * rng.random()
            pos += 1
        pos += span
    probs /= probs.sum(axis=-1, keepdims=True)
    return np.log(probs)


def main() -> None:
    text = "12가3456"
    labels = encode(text)
    print(f"정답 문자열 : {text}  (L = {len(labels)}자)")
    print(f"라벨 인덱스 : {labels}")
    print(f"확장 라벨   : 길이 {2 * len(labels) + 1} "
          f"(blank 를 사이사이에 끼운 결과)")
    print(f"blank 인덱스: {BLANK_INDEX}, 클래스 수: {NUM_CLASSES}\n")

    print("① 입력과 출력 길이가 다른 문제")
    print(f"{'T(프레임)':>10s} {'정렬 경로 수':>18s} {'CTC Loss':>12s} 비고")
    print("-" * 64)
    for n_frames in (5, 7, 10, 20, 40):
        paths = count_alignments(n_frames, labels)
        log_probs = make_log_probs(text, n_frames)
        loss = ctc_forward_loss(log_probs, labels)
        if n_frames < len(labels):
            note = "T < L → 정렬 불가능"
        elif n_frames == len(labels):
            note = "유일한 정렬(여유 없음)"
        else:
            note = "여유 프레임만큼 경로 증가"
        loss_str = "inf" if math.isinf(loss) else f"{loss:.4f}"
        print(f"{n_frames:10d} {paths:18,d} {loss_str:>12s} {note}")

    print("\n해석: T 가 늘수록 경로 수가 폭발하지만 손실 계산은 O(T·(2L+1)) 로 일정하다.")
    print("      바로 이 동적 계획법 덕분에 정렬 라벨 없이 학습할 수 있다.")

    print("\n② blank 토큰이 필요한 이유 — 같은 문자 반복")
    for sample in ("1234", "1123", "1111"):
        labs = encode(sample)
        repeats = sum(1 for i in range(1, len(labs)) if labs[i] == labs[i - 1])
        print(f"  {sample}: 반복 쌍 {repeats}개 → 최소 프레임 "
              f"{len(labs) + repeats} (blank 자리가 추가로 필요하다)")
    print("  blank 가 없으면 '11' 은 '1' 로 축약돼 표현 자체가 불가능하다.")

    print("\n③ 예측이 정답에서 멀어질수록 손실이 커지는지 확인")
    n_frames = 40
    log_probs = make_log_probs(text, n_frames)
    print(f"{'비교 대상':14s} {'CTC Loss':>12s}")
    print("-" * 28)
    for candidate in (text, "12가3457", "12가345", "99하9999"):
        labs = encode(candidate)
        loss = ctc_forward_loss(log_probs, labs)
        loss_str = "inf" if math.isinf(loss) else f"{loss:.4f}"
        print(f"{candidate:14s} {loss_str:>12s}")
    print("정답의 손실이 가장 작다. 학습은 이 값을 낮추는 방향으로 진행된다.")

    try:
        import torch
    except ImportError:
        print("\n[안내] torch 가 없어 nn.CTCLoss 와의 대조는 생략한다.")
        print("       위 값은 넘파이 forward 알고리즘으로 직접 계산한 결과다.")
        return

    print("\n④ torch.nn.CTCLoss 와 대조")
    criterion = torch.nn.CTCLoss(blank=BLANK_INDEX, reduction="sum",
                                 zero_infinity=True)
    tensor = torch.from_numpy(log_probs).unsqueeze(1)             # (T, 1, C)
    targets = torch.tensor(labels, dtype=torch.long)
    input_lengths = torch.tensor([n_frames], dtype=torch.long)
    target_lengths = torch.tensor([len(labels)], dtype=torch.long)
    torch_loss = float(criterion(tensor, targets, input_lengths, target_lengths))
    numpy_loss = ctc_forward_loss(log_probs, labels)
    print(f"  numpy 구현 : {numpy_loss:.6f}")
    print(f"  torch 구현 : {torch_loss:.6f}")
    print(f"  차이       : {abs(numpy_loss - torch_loss):.2e} (부동소수 오차 범위)")


if __name__ == "__main__":
    main()
