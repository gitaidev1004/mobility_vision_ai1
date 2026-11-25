"""``app.ocr.ctc_decoder`` 단위 테스트.

numpy 만 필요하다. CTC 의 축약 규칙과 beam search 의 접두사 병합이
올바른지 **확률 행렬을 직접 구성해** 검증한다 — 모델 없이 디코더만 테스트하는
것이 가능하다는 점이 순수 numpy 구현의 장점이다.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.ocr.charset import BLANK_INDEX, CHAR_TO_INDEX, NUM_CLASSES
from app.ocr.ctc_decoder import (
    CTCResult,
    beam_search_decode,
    collapse,
    decode,
    greedy_decode,
    log_softmax,
    softmax,
)

#: 테스트에서 자주 쓰는 문자 인덱스.
GA = CHAR_TO_INDEX["가"]
NA = CHAR_TO_INDEX["나"]


def make_probs(frames: list[int], *, peak: float = 0.99) -> np.ndarray:
    """프레임별 argmax 가 지정 인덱스가 되는 확률 행렬을 만든다.

    Args:
        frames: 프레임마다 가장 확률이 높을 클래스 인덱스.
        peak: 해당 클래스에 줄 확률.

    Returns:
        ``(T, NUM_CLASSES)`` 확률 행렬(각 행 합 = 1).
    """
    matrix = np.full((len(frames), NUM_CLASSES), (1.0 - peak) / (NUM_CLASSES - 1))
    for t, idx in enumerate(frames):
        matrix[t, idx] = peak
    return matrix / matrix.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# softmax / log_softmax
# ---------------------------------------------------------------------------


def test_softmax_sums_to_one():
    """softmax 결과의 각 행 합이 1 이다."""
    logits = np.random.default_rng(0).normal(size=(5, NUM_CLASSES)) * 3
    probs = softmax(logits)
    assert np.allclose(probs.sum(axis=-1), 1.0)
    assert np.all(probs >= 0.0)


def test_softmax_numerically_stable():
    """매우 큰 로짓에서도 overflow 없이 동작한다.

    최댓값을 빼지 않는 구현은 ``exp(1000)`` 에서 inf 가 되어 NaN 을 만든다.
    """
    logits = np.array([[1000.0, 1001.0, 999.0]])
    probs = softmax(logits)
    assert np.all(np.isfinite(probs))
    assert np.allclose(probs.sum(), 1.0)


def test_log_softmax_matches_log_of_softmax():
    """``log_softmax`` 가 ``log(softmax(x))`` 와 일치한다."""
    logits = np.random.default_rng(1).normal(size=(3, 10))
    assert np.allclose(log_softmax(logits), np.log(softmax(logits)), atol=1e-10)


# ---------------------------------------------------------------------------
# collapse — CTC 축약 규칙
# ---------------------------------------------------------------------------


def test_collapse_removes_consecutive_duplicates():
    """연속 중복을 제거한다."""
    assert collapse([1, 1, 1], blank=BLANK_INDEX) == [1]


def test_collapse_blank_separates_repeats():
    """blank 가 같은 문자의 반복을 구분한다(CTC 의 핵심).

    ``[1, blank, 1]`` → ``[1, 1]``. blank 없이는 '11' 을 표현할 수 없다.
    """
    assert collapse([1, BLANK_INDEX, 1], blank=BLANK_INDEX) == [1, 1]


def test_collapse_order_matters():
    """**중복 제거 → blank 제거** 순서가 지켜진다.

    순서를 바꿔 blank 를 먼저 지우면 ``[1,1,blank,1]`` 이 ``[1,1,1]`` → ``[1]``
    이 되어 정보가 소실된다.
    """
    assert collapse([1, 1, BLANK_INDEX, 1], blank=BLANK_INDEX) == [1, 1]


def test_collapse_all_blank():
    """전부 blank 이면 빈 리스트다."""
    assert collapse([BLANK_INDEX] * 5, blank=BLANK_INDEX) == []


def test_collapse_empty():
    """빈 입력은 빈 리스트를 반환한다."""
    assert collapse([], blank=BLANK_INDEX) == []


# ---------------------------------------------------------------------------
# greedy_decode
# ---------------------------------------------------------------------------


def test_greedy_decode_simple_plate():
    """전형적인 번호판 프레임 시퀀스를 올바르게 디코딩한다."""
    frames = [1, 1, BLANK_INDEX, 2, GA, GA, BLANK_INDEX, 3, 4, 5, 6]
    result = greedy_decode(make_probs(frames), already_prob=True)
    assert result.text == "12가3456"
    assert len(result.char_confidences) == 7
    assert result.confidence > 0.9


def test_greedy_decode_double_digit():
    """blank 로 구분된 같은 숫자 반복('11')을 복원한다."""
    result = greedy_decode(make_probs([1, BLANK_INDEX, 1]), already_prob=True)
    assert result.text == "11"


def test_greedy_decode_all_blank_returns_empty():
    """모든 프레임이 blank 이면 빈 문자열이다."""
    result = greedy_decode(make_probs([BLANK_INDEX] * 6), already_prob=True)
    assert result.text == ""
    assert result.confidence == 0.0


def test_greedy_decode_timesteps_are_monotonic():
    """문자가 채택된 프레임 인덱스가 단조 증가한다."""
    frames = [1, BLANK_INDEX, 2, BLANK_INDEX, GA]
    result = greedy_decode(make_probs(frames), already_prob=True)
    assert result.timesteps == sorted(result.timesteps)
    assert len(result.timesteps) == len(result.text)


def test_greedy_decode_accepts_logits():
    """softmax 미적용 로짓도 자동 판단해 처리한다."""
    logits = np.full((3, NUM_CLASSES), -10.0)
    logits[0, 1] = 10.0
    logits[1, BLANK_INDEX] = 10.0
    logits[2, 2] = 10.0
    assert greedy_decode(logits).text == "12"


def test_greedy_confidence_is_geometric_mean():
    """시퀀스 신뢰도가 기하평균이라 "가장 약한 글자" 가 반영된다.

    산술평균이면 한 글자가 0.1 이어도 전체가 높게 보인다. 번호판은 한 글자만
    틀려도 조회가 실패하므로 기하평균이 운영상 올바르다.
    """
    # 행을 정규화하면 "작은 값" 도 상대적으로 커지므로, 경쟁 클래스를 명시적으로
    # 넣어 **진짜 불확실한** 프레임을 만든다.
    matrix = np.zeros((3, NUM_CLASSES))
    matrix[0, 1] = 0.99
    matrix[0, 7] = 0.01
    matrix[1, BLANK_INDEX] = 0.99
    matrix[1, 1] = 0.01
    # 두 번째 글자: 0.30 vs 0.28 vs ... 로 사실상 추측 수준.
    matrix[2, 2] = 0.30
    matrix[2, 3] = 0.28
    matrix[2, 4] = 0.22
    matrix[2, 5] = 0.20
    matrix /= matrix.sum(axis=1, keepdims=True)

    result = greedy_decode(matrix, already_prob=True)
    assert result.text == "12"
    assert result.char_confidences[0] > 0.95
    assert result.char_confidences[1] < 0.35

    arithmetic_mean = float(np.mean(result.char_confidences))
    # 기하평균은 값이 서로 다를 때 산술평균보다 **엄격히 작다**(AM-GM 부등식).
    assert result.confidence < arithmetic_mean
    # 약한 글자 때문에 전체 신뢰도가 크게 깎인다 — 이것이 의도한 동작이다.
    assert result.confidence < 0.60


def test_greedy_decode_handles_3d_input():
    """``(1, T, C)`` 형태(torch 출력 관례)도 받는다."""
    matrix = make_probs([1, BLANK_INDEX, 2])[None, :, :]
    assert greedy_decode(matrix, already_prob=True).text == "12"


# ---------------------------------------------------------------------------
# beam_search_decode
# ---------------------------------------------------------------------------


def test_beam_search_matches_greedy_on_confident_input():
    """확신이 높은 입력에서는 beam 과 greedy 결과가 같다."""
    frames = [1, 1, BLANK_INDEX, 2, GA, BLANK_INDEX, 3, 4, 5, 6]
    matrix = make_probs(frames)
    assert beam_search_decode(matrix, already_prob=True)[0].text == (
        greedy_decode(matrix, already_prob=True).text
    )


def test_beam_search_returns_ranked_candidates():
    """후보가 로그 확률 내림차순으로 정렬되어 반환된다."""
    matrix = np.full((5, NUM_CLASSES), 1e-6)
    matrix[0, 1] = 0.60
    matrix[0, 7] = 0.35  # 1위/2위가 경쟁
    matrix[1, BLANK_INDEX] = 0.90
    matrix[2, 2] = 0.50
    matrix[2, 3] = 0.45
    matrix[3, BLANK_INDEX] = 0.95
    matrix[4, GA] = 0.99
    matrix /= matrix.sum(axis=1, keepdims=True)

    results = beam_search_decode(matrix, beam_width=12, top_k=4, already_prob=True)
    assert len(results) >= 2
    log_probs = [r.log_prob for r in results]
    assert log_probs == sorted(log_probs, reverse=True)
    # 1위는 각 자리의 최고 확률 조합이어야 한다.
    assert results[0].text == "12가"


def test_beam_search_excludes_unreachable_prefixes():
    """도달 불가능한(확률 0) 접두사를 후보에서 제외한다.

    이 필터가 없으면 ``candidates`` 에 신뢰도만 높은 쓰레기 후보가 올라간다.
    """
    frames = [1, 1, BLANK_INDEX, 2, GA, GA, BLANK_INDEX, 3, 4, 5, 6]
    results = beam_search_decode(make_probs(frames), beam_width=8, top_k=5, already_prob=True)
    for result in results:
        assert result.log_prob > -1e20, f"도달 불가 후보가 포함됨: {result}"


def test_beam_search_length_normalized_confidence():
    """신뢰도가 길이로 정규화되어 0~1 범위를 유지한다.

    정규화하지 않으면 긴 후보가 항상 불리해져(확률을 더 많이 곱하므로)
    짧은 후보가 1위가 된다.
    """
    frames = [1, BLANK_INDEX, 2, BLANK_INDEX, GA, BLANK_INDEX, 3, BLANK_INDEX, 4]
    for result in beam_search_decode(make_probs(frames), top_k=3, already_prob=True):
        assert 0.0 <= result.confidence <= 1.0


def test_beam_search_all_blank():
    """모든 프레임이 blank 이면 빈 문자열을 반환한다."""
    assert beam_search_decode(make_probs([BLANK_INDEX] * 4), already_prob=True)[0].text == ""


def test_beam_search_double_char():
    """beam search 도 blank 로 구분된 반복을 복원한다."""
    assert beam_search_decode(make_probs([1, BLANK_INDEX, 1]), already_prob=True)[0].text == "11"


def test_beam_width_one_is_like_greedy():
    """beam 폭 1 은 사실상 greedy 와 같은 결과를 낸다."""
    frames = [1, BLANK_INDEX, 2, BLANK_INDEX, NA]
    matrix = make_probs(frames)
    assert beam_search_decode(matrix, beam_width=1, already_prob=True)[0].text == (
        greedy_decode(matrix, already_prob=True).text
    )


def test_beam_search_prune_threshold_keeps_best():
    """가지치기가 모든 클래스를 지워도 최댓값 하나는 살린다(빈 beam 방지)."""
    matrix = make_probs([1, BLANK_INDEX, 2])
    # 임계값을 1.0 보다 크게 두면 모든 클래스가 가지치기 대상이 된다.
    results = beam_search_decode(matrix, prune_threshold=2.0, already_prob=True)
    assert results[0].text == "12"


# ---------------------------------------------------------------------------
# decode — 통합 진입점
# ---------------------------------------------------------------------------


def test_decode_greedy_mode():
    """``mode="greedy"`` 는 1건을 반환한다."""
    results = decode(make_probs([1, BLANK_INDEX, 2]), mode="greedy")
    assert len(results) == 1
    assert results[0].text == "12"


def test_decode_beam_mode():
    """``mode="beam"`` 은 최대 top_k 건을 반환한다."""
    results = decode(make_probs([1, BLANK_INDEX, 2]), mode="beam", top_k=3)
    assert 1 <= len(results) <= 3


def test_decode_unknown_mode_falls_back_to_greedy():
    """알 수 없는 모드는 greedy 로 처리한다(예외를 던지지 않는다).

    서비스가 500 을 내지 않아야 하므로, 잘못된 파라미터도 흡수해야 한다.
    """
    results = decode(make_probs([1, BLANK_INDEX, 2]), mode="nonsense")
    assert results[0].text == "12"


@pytest.mark.parametrize("bad_input", [np.array([]), "garbage", None, np.zeros((0, 5))])
def test_decode_invalid_input_returns_empty(bad_input):
    """잘못된 입력에도 예외 없이 빈 결과를 돌려준다."""
    results = decode(bad_input)
    assert len(results) == 1
    assert results[0].text == ""
    assert results[0].confidence == 0.0


def test_ctc_result_defaults():
    """``CTCResult`` 의 기본값이 안전하다(가변 기본값 공유 버그 방지)."""
    first = CTCResult(text="a", confidence=0.5)
    second = CTCResult(text="b", confidence=0.5)
    first.char_confidences.append(0.9)
    # dataclass field(default_factory) 가 올바르면 서로 영향이 없다.
    assert second.char_confidences == []


def test_blank_index_matches_charset():
    """blank 인덱스가 문자셋 크기와 일치한다.

    이 불변식이 깨지면 디코더가 문자를 blank 로, blank 를 문자로 읽는다.
    모델 출력 차원도 ``NUM_CLASSES`` 와 같아야 한다.
    """
    assert BLANK_INDEX == NUM_CLASSES - 1
    # blank 인덱스로는 어떤 문자도 조회되지 않아야 한다.
    from app.ocr.charset import INDEX_TO_CHAR

    assert BLANK_INDEX not in INDEX_TO_CHAR
