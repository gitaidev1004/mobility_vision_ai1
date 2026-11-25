"""CTC(Connectionist Temporal Classification) 디코더 — 순수 numpy 구현.

왜 직접 구현하는가?
    1. ``torch.nn.CTCLoss`` 는 학습용이고, ``torchaudio``/``ctcdecode`` 같은 디코딩
       라이브러리는 설치가 번거롭다. 추론 경로에서 torch 를 요구하지 않으려면
       디코더는 numpy 로 자급해야 한다(= 폴백 경로에서도 동일 코드 재사용).
    2. 교재에서 beam search 의 "접두사 확률 병합" 과정을 코드로 보여주는 것이
       개념 이해에 가장 효과적이다.

CTC 의 핵심 아이디어:
    입력 프레임 수 T 와 정답 문자 수 L 이 다를 때(T >= L) 정렬 정보 없이 학습하기 위해,
    모든 프레임이 '문자' 또는 'blank' 를 출력하도록 한 뒤
    **① 연속 중복 제거 → ② blank 제거** 순서로 축약(collapse)한다.
    이 순서가 중요하다. ``a a blank a`` → (①) ``a blank a`` → (②) ``aa``.
    즉 blank 는 "같은 문자가 두 번 연속 나온다"는 것을 표현하는 구분자 역할을 한다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from app.ocr.charset import BLANK_INDEX, INDEX_TO_CHAR

#: log(0) 대용. -inf 를 쓰면 logaddexp 에서 NaN 이 생길 수 있어 큰 음수로 대체한다.
NEG_INF: float = -1e30


@dataclass(slots=True)
class CTCResult:
    """CTC 디코딩 결과 한 건.

    Attributes:
        text: 디코딩된 문자열.
        confidence: 0~1 로 정규화된 시퀀스 신뢰도.
        char_confidences: 출력된 문자 각각의 신뢰도(``text`` 와 길이 동일).
        log_prob: 해당 경로/접두사의 로그 확률(beam search 에서만 의미 있음).
        timesteps: 각 문자가 채택된 프레임 인덱스(greedy 에서만 채워진다).
    """

    text: str
    confidence: float
    char_confidences: list[float] = field(default_factory=list)
    log_prob: float = 0.0
    timesteps: list[int] = field(default_factory=list)


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """수치적으로 안정한 softmax.

    왜 최댓값을 빼는가? ``exp(1000)`` 은 float64 에서 inf 가 된다. 각 축의 최댓값을
    빼도 softmax 결과는 수학적으로 동일하지만 overflow 가 사라진다.

    Args:
        logits: 임의 shape 의 로짓 배열.
        axis: softmax 를 적용할 축.

    Returns:
        ``logits`` 와 같은 shape 의 확률 배열(해당 축 합이 1).
    """
    arr = np.asarray(logits, dtype=np.float64)
    shifted = arr - np.max(arr, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    denom = np.sum(exp, axis=axis, keepdims=True)
    # 모든 값이 동일해 denom 이 0 이 되는 병리적 케이스 방어.
    denom = np.where(denom <= 0.0, 1.0, denom)
    return exp / denom


def log_softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """수치적으로 안정한 log-softmax.

    beam search 는 확률을 계속 곱하므로 로그 공간에서 더하는 편이 underflow 에 강하다.

    Args:
        logits: 로짓 배열.
        axis: 적용 축.

    Returns:
        로그 확률 배열.
    """
    arr = np.asarray(logits, dtype=np.float64)
    shifted = arr - np.max(arr, axis=axis, keepdims=True)
    return shifted - np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))


def _as_probabilities(matrix: np.ndarray, *, already_prob: bool | None = None) -> np.ndarray:
    """입력 행렬을 (T, C) 확률 행렬로 정규화한다.

    모델이 softmax 를 이미 적용했는지 호출자가 모를 수 있으므로,
    ``already_prob=None`` 이면 "모든 값이 0 이상이고 행 합이 1 에 가까운가"로 자동 판단한다.
    이런 자동 판단은 교재 예제에서 독자가 실수로 로짓을 넣어도 동작하게 해 준다.

    Args:
        matrix: (T, C) 로짓 또는 확률.
        already_prob: True/False 로 명시하면 자동 판단을 생략한다.

    Returns:
        (T, C) float64 확률 행렬.

    Raises:
        ValueError: 2차원이 아니거나 비어 있을 때.
    """
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.ndim == 3:
        # (1, T, C) 또는 (T, 1, C) 형태를 허용한다(torch 출력 관례).
        if arr.shape[0] == 1:
            arr = arr[0]
        elif arr.shape[1] == 1:
            arr = arr[:, 0, :]
    if arr.ndim != 2 or arr.size == 0:
        msg = f"(T, C) 2차원 행렬이 필요합니다. 받은 shape={np.asarray(matrix).shape}"
        raise ValueError(msg)

    if already_prob is True:
        return arr
    if already_prob is False:
        return softmax(arr, axis=-1)

    row_sums = arr.sum(axis=-1)
    looks_like_prob = bool(np.all(arr >= -1e-9) and np.allclose(row_sums, 1.0, atol=1e-3))
    return arr if looks_like_prob else softmax(arr, axis=-1)


def collapse(indices: list[int] | tuple[int, ...] | np.ndarray, blank: int = BLANK_INDEX) -> list[int]:
    """CTC 축약 규칙을 적용한다: ① 연속 중복 제거 → ② blank 제거.

    순서를 바꾸면 결과가 달라진다. ``[a, a, blank, a]`` 에서
    blank 를 먼저 지우면 ``[a,a,a]`` → ``[a]`` 가 되어 'aa' 를 표현할 수 없다.

    Args:
        indices: 프레임별 argmax 인덱스 시퀀스.
        blank: blank 토큰 인덱스.

    Returns:
        축약된 인덱스 리스트.

    Examples:
        >>> collapse([1, 1, 77, 1], blank=77)
        [1, 1]
        >>> collapse([1, 1, 1], blank=77)
        [1]
    """
    out: list[int] = []
    prev: int | None = None
    for raw in indices:
        idx = int(raw)
        if idx != prev:  # ① 직전 프레임과 다를 때만 채택(연속 중복 제거)
            if idx != blank:  # ② blank 는 출력하지 않음
                out.append(idx)
            prev = idx
        # idx == prev 이면 중복이므로 건너뛴다. prev 는 그대로 유지.
    return out


def greedy_decode(
    matrix: np.ndarray,
    *,
    blank: int = BLANK_INDEX,
    already_prob: bool | None = None,
) -> CTCResult:
    """CTC greedy(best-path) 디코딩.

    각 프레임에서 확률이 가장 큰 클래스를 독립적으로 고른 뒤 축약한다.
    O(T·C) 로 가장 빠르지만, "서로 다른 경로가 같은 문자열로 축약되는" 효과를
    고려하지 못해 beam search 보다 약간 정확도가 낮다.

    문자 신뢰도는 **해당 문자가 채택된 프레임들의 확률 중 최댓값**으로 정의한다.
    평균을 쓰면 문자가 길게 늘어진 프레임(blank 전환 직전의 낮은 확률)에 의해
    과소평가되므로, 대표 프레임의 확률을 쓰는 것이 체감 정확도와 더 잘 맞는다.

    Args:
        matrix: (T, C) 로짓 또는 확률 행렬.
        blank: blank 토큰 인덱스.
        already_prob: softmax 적용 여부를 명시(None 이면 자동 판단).

    Returns:
        CTCResult. 디코딩 결과가 비면 ``text=""``, ``confidence=0.0``.

    Examples:
        >>> import numpy as np
        >>> probs = np.zeros((3, 78)); probs[0, 1] = 1; probs[1, 77] = 1; probs[2, 2] = 1
        >>> greedy_decode(probs, already_prob=True).text
        '12'
    """
    probs = _as_probabilities(matrix, already_prob=already_prob)
    best_idx = probs.argmax(axis=-1)
    best_prob = probs.max(axis=-1)

    chars: list[str] = []
    char_confs: list[float] = []
    timesteps: list[int] = []

    prev: int | None = None
    for t, raw in enumerate(best_idx):
        idx = int(raw)
        if idx != prev:
            if idx != blank:
                ch = INDEX_TO_CHAR.get(idx)
                if ch is not None:
                    chars.append(ch)
                    char_confs.append(float(best_prob[t]))
                    timesteps.append(t)
            prev = idx
        elif idx != blank and chars:
            # 같은 문자가 계속 이어지는 구간: 최댓값으로 대표 신뢰도를 갱신한다.
            char_confs[-1] = max(char_confs[-1], float(best_prob[t]))

    text = "".join(chars)
    # 시퀀스 신뢰도는 문자 신뢰도의 기하평균을 쓴다.
    # 산술평균은 한 글자가 0.1 이어도 전체가 높게 보이는데, 번호판은 한 글자만 틀려도
    # 조회가 실패하므로 "가장 약한 고리" 를 반영하는 기하평균이 운영상 올바르다.
    confidence = _geometric_mean(char_confs)
    log_prob = float(np.sum(np.log(np.clip(best_prob, 1e-12, None))))

    return CTCResult(
        text=text,
        confidence=confidence,
        char_confidences=[round(c, 4) for c in char_confs],
        log_prob=log_prob,
        timesteps=timesteps,
    )


def _geometric_mean(values: list[float]) -> float:
    """0~1 값들의 기하평균을 안정적으로 계산한다.

    직접 곱하면 글자 수가 많을 때 underflow 가 나므로 로그 공간에서 평균을 낸다.

    Args:
        values: 0 이상 1 이하의 신뢰도 리스트.

    Returns:
        기하평균(빈 리스트면 0.0).
    """
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += math.log(max(float(v), 1e-12))
    return float(math.exp(total / len(values)))


def _logaddexp(a: float, b: float) -> float:
    """``log(exp(a) + exp(b))`` 를 overflow 없이 계산한다.

    beam search 에서 "서로 다른 경로가 같은 접두사로 합쳐질 때" 확률을 더하는 데 쓴다.

    Args:
        a: 로그 확률.
        b: 로그 확률.

    Returns:
        합의 로그 확률.
    """
    if a <= NEG_INF:
        return b
    if b <= NEG_INF:
        return a
    hi, lo = (a, b) if a > b else (b, a)
    return hi + math.log1p(math.exp(lo - hi))


def beam_search_decode(
    matrix: np.ndarray,
    *,
    beam_width: int = 10,
    blank: int = BLANK_INDEX,
    top_k: int = 3,
    prune_threshold: float = 1e-4,
    already_prob: bool | None = None,
) -> list[CTCResult]:
    """CTC prefix beam search 디코딩.

    알고리즘(Graves 2006, Hannun 2014 의 prefix beam search):
        각 접두사(prefix) 에 대해 두 개의 확률을 따로 관리한다.
            - ``p_blank``: 접두사가 **blank 로 끝나는** 경로들의 확률 합
            - ``p_nonblank``: 접두사가 **마지막 문자로 끝나는** 경로들의 확률 합
        이 둘을 분리하는 이유가 CTC 의 핵심이다. 같은 문자를 반복 입력할 때
        (예: '11') 두 문자 사이에 blank 가 반드시 들어가야 하므로,
        "직전이 blank 였는지" 를 알아야 중복 제거 규칙을 정확히 적용할 수 있다.

        프레임 t 에서 접두사 ``p`` 와 문자 ``c`` 에 대해:
            - ``c == blank``       → 접두사 불변, p_blank 누적
            - ``c == p 의 마지막 문자`` → 직전이 blank 인 경로만 접두사가 늘어난다
              (``p_blank`` 에서 확장). 직전이 같은 문자인 경로는 중복 제거되어
              접두사가 그대로 유지된다(``p_nonblank`` 누적).
            - 그 외                 → 두 확률 모두에서 확장 가능

    greedy 대비 장점: '1' 과 '11' 처럼 축약 결과가 갈리는 경계에서 전체 경로 확률을
    합산해 비교하므로, 흐릿한 번호판에서 글자 수를 더 정확히 맞춘다.

    Args:
        matrix: (T, C) 로짓 또는 확률 행렬.
        beam_width: 프레임마다 유지할 접두사 수. 크면 정확하지만 O(T·beam·C) 로 느려진다.
        blank: blank 토큰 인덱스.
        top_k: 반환할 상위 후보 수.
        prune_threshold: 이 확률 미만인 클래스는 확장 대상에서 제외(속도 최적화).
            번호판은 클래스가 78개라 가지치기 효과가 크다.
        already_prob: softmax 적용 여부 명시(None 이면 자동 판단).

    Returns:
        로그 확률 내림차순으로 정렬된 ``CTCResult`` 리스트(최대 ``top_k`` 개).
        입력이 비정상이어도 예외 대신 빈 결과 1건을 반환한다.

    Examples:
        >>> import numpy as np
        >>> probs = np.full((4, 78), 1e-6); probs[:, 77] = 1.0
        >>> beam_search_decode(probs, already_prob=True)[0].text
        ''
    """
    probs = _as_probabilities(matrix, already_prob=already_prob)
    n_frames, n_classes = probs.shape
    beam_width = max(1, int(beam_width))

    # beams: prefix(tuple[int]) -> (log p_blank, log p_nonblank)
    # 초기 상태: 빈 접두사가 blank 로 끝난 것으로 본다(확률 1 = log 0).
    beams: dict[tuple[int, ...], tuple[float, float]] = {(): (0.0, NEG_INF)}

    for t in range(n_frames):
        frame = probs[t]
        # 가지치기: 확률이 임계값 이상인 클래스만 확장한다.
        active = np.where(frame >= prune_threshold)[0]
        if active.size == 0:
            # 전부 가지치기되면 최댓값 하나만이라도 살린다(빈 beam 방지).
            active = np.array([int(frame.argmax())])

        next_beams: dict[tuple[int, ...], tuple[float, float]] = {}

        def _accumulate(
            prefix: tuple[int, ...],
            add_blank: float,
            add_nonblank: float,
            store: dict[tuple[int, ...], tuple[float, float]] = next_beams,
        ) -> None:
            """접두사의 (p_blank, p_nonblank) 로그 확률을 누적 병합한다."""
            cur_b, cur_nb = store.get(prefix, (NEG_INF, NEG_INF))
            store[prefix] = (
                _logaddexp(cur_b, add_blank),
                _logaddexp(cur_nb, add_nonblank),
            )

        for prefix, (lp_blank, lp_nonblank) in beams.items():
            lp_total = _logaddexp(lp_blank, lp_nonblank)
            last = prefix[-1] if prefix else None

            for raw_c in active:
                c = int(raw_c)
                lp_c = math.log(max(float(frame[c]), 1e-300))

                if c == blank:
                    # blank 출력 → 접두사 그대로, blank 로 끝나는 확률에 누적.
                    _accumulate(prefix, lp_total + lp_c, NEG_INF)
                    continue

                if c == last:
                    # ⓐ 직전이 blank 였던 경로만 접두사를 늘릴 수 있다 ('11' 생성).
                    _accumulate((*prefix, c), NEG_INF, lp_blank + lp_c)
                    # ⓑ 직전이 같은 문자였던 경로는 중복 제거되어 접두사 유지.
                    _accumulate(prefix, NEG_INF, lp_nonblank + lp_c)
                else:
                    # 새 문자 → blank/non-blank 어느 쪽에서든 확장 가능.
                    _accumulate((*prefix, c), NEG_INF, lp_total + lp_c)

        if not next_beams:
            break
        # 상위 beam_width 개만 유지한다.
        beams = dict(
            sorted(
                next_beams.items(),
                key=lambda kv: _logaddexp(kv[1][0], kv[1][1]),
                reverse=True,
            )[:beam_width]
        )

    scored = sorted(
        ((prefix, _logaddexp(b, nb)) for prefix, (b, nb) in beams.items()),
        key=lambda kv: kv[1],
        reverse=True,
    )
    # 확률이 실질적으로 0 인(도달 불가능한) 접두사는 후보에서 제외한다.
    # 이들을 남기면 API 의 ``candidates`` 에 신뢰도만 높은 쓰레기 후보가 올라간다.
    reachable = [(p, lp) for p, lp in scored if lp > NEG_INF / 2.0]
    if reachable:
        scored = reachable

    results: list[CTCResult] = []
    for prefix, log_p in scored[: max(1, int(top_k))]:
        text = "".join(INDEX_TO_CHAR[i] for i in prefix if i in INDEX_TO_CHAR)
        char_confs = _per_char_confidence(probs, prefix, blank)
        # 길이 정규화 경로 확률 = exp(log P(prefix) / 문자수).
        # 길이로 나누지 않으면 긴 후보가 항상 불리해져(확률을 더 많이 곱하므로)
        # '12가3456' 대신 '12가345' 같은 짧은 후보가 1위가 된다. 정규화는 필수다.
        norm = max(1, len(prefix))
        path_conf = math.exp(max(log_p / norm, -60.0))
        confidence = min(1.0, path_conf)
        if char_confs:
            # 문자별 근사 신뢰도와 경로 확률 중 **보수적인 쪽**을 택한다.
            # 둘 중 하나라도 낮으면 실제로 불확실한 판독이기 때문이다.
            confidence = min(confidence, _geometric_mean(char_confs))
        results.append(
            CTCResult(
                text=text,
                confidence=round(float(confidence), 6),
                char_confidences=[round(c, 4) for c in char_confs],
                log_prob=float(log_p),
            )
        )
    if not results:
        results.append(CTCResult(text="", confidence=0.0))
    return results


def _per_char_confidence(
    probs: np.ndarray, prefix: tuple[int, ...], blank: int
) -> list[float]:
    """beam 결과 접두사에 대한 문자별 신뢰도를 근사 계산한다.

    정확한 posterior 는 forward-backward 를 다시 돌려야 하지만, 번호판처럼 짧은
    시퀀스에서는 "각 문자가 가장 강하게 나타난 프레임의 확률" 로 근사해도
    순위 비교에 충분하다. 계산량을 T·L 로 억제하기 위한 실용적 타협이다.

    Args:
        probs: (T, C) 확률 행렬.
        prefix: 문자 인덱스 시퀀스.
        blank: blank 인덱스.

    Returns:
        문자별 신뢰도 리스트(prefix 와 길이 동일).
    """
    if not prefix:
        return []
    n_frames = probs.shape[0]
    confs: list[float] = []
    # 문자 순서를 유지하려면 각 문자의 탐색 시작 프레임을 단조 증가시켜야 한다.
    cursor = 0
    for pos, idx in enumerate(prefix):
        remaining = len(prefix) - pos
        # 뒤 문자들이 최소 1프레임씩 필요하므로 탐색 상한을 둔다.
        upper = max(cursor + 1, n_frames - remaining + 1)
        window = probs[cursor:upper, idx] if upper > cursor else probs[cursor : cursor + 1, idx]
        if window.size == 0:
            confs.append(0.0)
            continue
        best_offset = int(window.argmax())
        confs.append(float(window[best_offset]))
        cursor = min(cursor + best_offset + 1, n_frames - 1)
    _ = blank  # blank 는 현재 근사식에서 쓰지 않지만 시그니처 일관성을 위해 유지한다.
    return confs


def decode(
    matrix: np.ndarray,
    *,
    mode: str = "greedy",
    beam_width: int = 10,
    top_k: int = 3,
    blank: int = BLANK_INDEX,
    already_prob: bool | None = None,
) -> list[CTCResult]:
    """모드에 따라 greedy 또는 beam search 디코딩을 수행하는 단일 진입점.

    라우터/파이프라인은 이 함수만 호출하면 되므로 디코딩 전략 교체가 쉽다.

    Args:
        matrix: (T, C) 로짓 또는 확률 행렬.
        mode: ``"greedy"`` | ``"beam"``. 알 수 없는 값은 greedy 로 처리한다
            (서비스가 500 을 내지 않아야 하므로 예외를 던지지 않는다).
        beam_width: beam 모드의 beam 폭.
        top_k: 반환 후보 수.
        blank: blank 인덱스.
        already_prob: softmax 적용 여부.

    Returns:
        ``CTCResult`` 리스트(최소 1건).
    """
    try:
        if str(mode).lower().startswith("beam"):
            return beam_search_decode(
                matrix,
                beam_width=beam_width,
                blank=blank,
                top_k=top_k,
                already_prob=already_prob,
            )
        result = greedy_decode(matrix, blank=blank, already_prob=already_prob)
    except (ValueError, TypeError):
        # 잘못된 입력이 들어와도 빈 결과를 돌려 상위 계층이 NO_PLATE 로 처리하게 한다.
        return [CTCResult(text="", confidence=0.0)]
    return [result]


__all__ = [
    "NEG_INF",
    "CTCResult",
    "beam_search_decode",
    "collapse",
    "decode",
    "greedy_decode",
    "log_softmax",
    "softmax",
]
