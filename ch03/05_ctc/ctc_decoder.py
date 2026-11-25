# -*- coding: utf-8 -*-
"""CTC 디코더 — greedy 와 prefix beam search 를 넘파이만으로 직접 구현한다.

실행:  python 05_ctc/ctc_decoder.py
torch 없이 numpy 만으로 동작하므로 추론 경로에서 그대로 재사용할 수 있다.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

# 같은 장의 문자셋 모듈을 재사용한다(ch03/03_dataset/charset.py).
sys.path.append(str(Path(__file__).resolve().parents[1] / "03_dataset"))
from charset import BLANK_INDEX, INDEX_TO_CHAR, NUM_CLASSES, encode  # noqa: E402

#: log(0) 대용값. -inf 를 쓰면 logaddexp 에서 NaN 이 생길 수 있어 큰 음수로 대체한다.
NEG_INF = -1e30


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """수치적으로 안정한 softmax. 축의 최댓값을 빼서 exp 오버플로를 막는다."""
    arr = np.asarray(logits, dtype=np.float64)
    shifted = arr - np.max(arr, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    denom = np.sum(exp, axis=axis, keepdims=True)
    denom = np.where(denom <= 0.0, 1.0, denom)
    return exp / denom


def as_probabilities(matrix: np.ndarray) -> np.ndarray:
    """(T, C) 로짓/확률 행렬을 확률 행렬로 정규화한다.

    행 합이 1 에 가깝고 모든 값이 음수가 아니면 이미 확률로 간주하고,
    그렇지 않으면 softmax 를 적용한다. 독자가 실수로 로짓을 넣어도 동작한다.
    """
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.ndim == 3 and arr.shape[0] == 1:   # (1, T, C) 허용
        arr = arr[0]
    if arr.ndim != 2 or arr.size == 0:
        raise ValueError(f"(T, C) 2차원 행렬이 필요하다. shape={arr.shape}")
    row_sums = arr.sum(axis=-1)
    if bool(np.all(arr >= -1e-9)) and bool(np.allclose(row_sums, 1.0, atol=1e-3)):
        return arr
    return softmax(arr, axis=-1)


def collapse(indices, blank: int = BLANK_INDEX) -> list[int]:
    """CTC 축약 규칙: ① 연속 중복 제거 → ② blank 제거. 순서를 바꾸면 결과가 달라진다."""
    out: list[int] = []
    prev = None
    for raw in indices:
        idx = int(raw)
        if idx != prev:          # ① 직전과 다를 때만 채택
            if idx != blank:     # ② blank 는 출력하지 않음
                out.append(idx)
            prev = idx
    return out


def geometric_mean(values: list[float]) -> float:
    """0~1 신뢰도의 기하평균. 로그 공간에서 계산해 underflow 를 피한다."""
    if not values:
        return 0.0
    total = sum(math.log(max(float(v), 1e-12)) for v in values)
    return float(math.exp(total / len(values)))


def greedy_decode(matrix: np.ndarray, blank: int = BLANK_INDEX) -> dict:
    """프레임마다 argmax 를 고른 뒤 축약하는 best-path 디코딩. O(T·C)."""
    probs = as_probabilities(matrix)
    best_idx = probs.argmax(axis=-1)
    best_prob = probs.max(axis=-1)

    chars: list[str] = []
    confs: list[float] = []
    steps: list[int] = []
    prev = None
    for t, raw in enumerate(best_idx):
        idx = int(raw)
        if idx != prev:
            if idx != blank:
                ch = INDEX_TO_CHAR.get(idx)
                if ch is not None:
                    chars.append(ch)
                    confs.append(float(best_prob[t]))
                    steps.append(t)
            prev = idx
        elif idx != blank and confs:
            # 같은 문자가 이어지는 구간은 최댓값으로 대표 신뢰도를 갱신한다.
            confs[-1] = max(confs[-1], float(best_prob[t]))
    return {
        "text": "".join(chars),
        "confidence": round(geometric_mean(confs), 6),
        "char_confidences": [round(c, 4) for c in confs],
        "timesteps": steps,
        "raw_path": [int(i) for i in best_idx],
    }


def _logaddexp(a: float, b: float) -> float:
    """log(exp(a)+exp(b)) 를 오버플로 없이 계산한다. 경로 확률 병합에 쓴다."""
    if a <= NEG_INF:
        return b
    if b <= NEG_INF:
        return a
    hi, lo = (a, b) if a > b else (b, a)
    return hi + math.log1p(math.exp(lo - hi))


def beam_search_decode(
    matrix: np.ndarray,
    beam_width: int = 10,
    blank: int = BLANK_INDEX,
    top_k: int = 3,
    prune_threshold: float = 1e-4,
) -> list[dict]:
    """CTC prefix beam search.

    접두사마다 두 확률을 분리해 관리한다.
      p_blank    : 접두사가 blank 로 끝나는 경로들의 확률 합
      p_nonblank : 접두사가 마지막 문자로 끝나는 경로들의 확률 합
    같은 문자를 두 번 쓰려면 사이에 blank 가 반드시 필요하므로 이 분리가 핵심이다.
    """
    probs = as_probabilities(matrix)
    n_frames = probs.shape[0]
    beam_width = max(1, int(beam_width))

    # 초기 상태: 빈 접두사가 blank 로 끝난 것으로 본다(확률 1 = 로그 0).
    beams: dict[tuple, tuple[float, float]] = {(): (0.0, NEG_INF)}

    for t in range(n_frames):
        frame = probs[t]
        active = np.where(frame >= prune_threshold)[0]
        if active.size == 0:
            active = np.array([int(frame.argmax())])

        nxt: dict[tuple, tuple[float, float]] = {}

        def add(prefix: tuple, lb: float, lnb: float) -> None:
            cb, cnb = nxt.get(prefix, (NEG_INF, NEG_INF))
            nxt[prefix] = (_logaddexp(cb, lb), _logaddexp(cnb, lnb))

        for prefix, (lp_b, lp_nb) in beams.items():
            lp_total = _logaddexp(lp_b, lp_nb)
            last = prefix[-1] if prefix else None
            for raw_c in active:
                c = int(raw_c)
                lp_c = math.log(max(float(frame[c]), 1e-300))
                if c == blank:
                    add(prefix, lp_total + lp_c, NEG_INF)
                elif c == last:
                    add((*prefix, c), NEG_INF, lp_b + lp_c)    # blank 뒤이면 글자 추가
                    add(prefix, NEG_INF, lp_nb + lp_c)         # 같은 글자 반복은 축약
                else:
                    add((*prefix, c), NEG_INF, lp_total + lp_c)

        if not nxt:
            break
        beams = dict(sorted(nxt.items(),
                            key=lambda kv: _logaddexp(kv[1][0], kv[1][1]),
                            reverse=True)[:beam_width])

    scored = sorted(((p, _logaddexp(b, nb)) for p, (b, nb) in beams.items()),
                    key=lambda kv: kv[1], reverse=True)
    reachable = [(p, lp) for p, lp in scored if lp > NEG_INF / 2.0]
    if reachable:
        scored = reachable

    out: list[dict] = []
    for prefix, log_p in scored[: max(1, int(top_k))]:
        text = "".join(INDEX_TO_CHAR[i] for i in prefix if i in INDEX_TO_CHAR)
        # 길이로 정규화하지 않으면 짧은 후보가 항상 유리해진다(확률을 덜 곱하므로).
        norm = max(1, len(prefix))
        conf = min(1.0, math.exp(max(log_p / norm, -60.0)))
        out.append({"text": text, "confidence": round(conf, 6),
                    "log_prob": round(float(log_p), 4)})
    return out or [{"text": "", "confidence": 0.0, "log_prob": NEG_INF}]


def make_fake_logits(text: str, n_frames: int = 24, seed: int = 7) -> np.ndarray:
    """정답 문자열을 늘려 쓴 가짜 (T, C) 확률 행렬을 만든다(테스트용)."""
    rng = np.random.default_rng(seed)
    labels = encode(text)
    probs = np.full((n_frames, NUM_CLASSES), 1e-4)
    probs[:, BLANK_INDEX] = 0.6
    # 문자마다 2프레임을 배정하고 사이에 blank 프레임을 둔다.
    pos = 1
    for lab in labels:
        for _ in range(2):
            if pos >= n_frames - 1:
                break
            probs[pos, :] = 1e-4
            probs[pos, lab] = 0.9 + 0.05 * rng.random()
            pos += 1
        pos += 1  # 문자 사이 blank 프레임
    probs /= probs.sum(axis=-1, keepdims=True)
    return probs


def main() -> None:
    target = "12가3456"
    probs = make_fake_logits(target)
    print(f"입력 행렬 shape = {probs.shape}  (T=프레임, C=클래스)")
    print(f"blank 인덱스 = {BLANK_INDEX}, 클래스 수 = {NUM_CLASSES}")

    g = greedy_decode(probs)
    print(f"\n[greedy]  text={g['text']!r}  conf={g['confidence']:.4f}")
    print(f"          프레임별 문자 위치 = {g['timesteps']}")

    b = beam_search_decode(probs, beam_width=10, top_k=3)
    print("\n[beam search] 상위 후보")
    for rank, cand in enumerate(b, start=1):
        print(f"  {rank}. {cand['text']!r}  conf={cand['confidence']:.4f} "
              f"logP={cand['log_prob']:.2f}")

    print(f"\n정답 일치(greedy) = {g['text'] == target}")
    print(f"정답 일치(beam)   = {b[0]['text'] == target}")

    # 축약 규칙 확인: 같은 문자 사이에 blank 가 있어야 두 글자로 남는다.
    print("\n[축약 규칙]")
    print("  collapse([1,1,blank,1]) ->", collapse([1, 1, BLANK_INDEX, 1]))
    print("  collapse([1,1,1])       ->", collapse([1, 1, 1]))


if __name__ == "__main__":
    main()
