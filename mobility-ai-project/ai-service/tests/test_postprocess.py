"""``app.ocr.postprocess`` 단위 테스트.

이 테스트는 **numpy 조차 필요하지 않다** — 후처리는 순수 파이썬 문자열 로직이다.
따라서 어떤 환경에서도 실행되고, 번호판 인식 품질의 핵심 규칙을 회귀 테스트로 고정한다.
"""

from __future__ import annotations

import pytest

from app.ocr.charset import USE_CHARS, is_use_char, nearest_use_char
from app.ocr.postprocess import (
    PLATE_RE_PLAIN,
    PLATE_RE_REGION,
    FrameCandidate,
    correct_confusions,
    decide_status,
    expected_layout,
    fuse_frames,
    is_valid_plate,
    normalize,
    postprocess,
    split_region,
    summarize_corrections,
)


# ---------------------------------------------------------------------------
# normalize
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("12가3456", "12가3456"),
        ("서울 12가3456", "서울12가3456"),
        ("12가-3456", "12가3456"),
        ("  12가3456  ", "12가3456"),
        ("12가.3456", "12가3456"),
        ("１２가3456", "12가3456"),  # 전각 숫자
        ("", ""),
    ],
)
def test_normalize(raw, expected):
    """공백/하이픈 제거와 전각→반각 변환이 동작한다."""
    assert normalize(raw) == expected


def test_normalize_none_safe():
    """None 입력에도 예외 없이 빈 문자열을 반환한다."""
    assert normalize(None) == ""  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 정규식 검증
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "plate",
    [
        "12가3456",      # 신형 2자리
        "123가4567",     # 신형 3자리
        "서울12가3456",   # 구형 지역명
        "서울 12가3456",  # 구형 지역명 + 공백
        "88허1004",      # 렌터카
        "34배7721",      # 택배
    ],
)
def test_valid_plates(plate):
    """유효한 한국 번호판 형식을 모두 통과시킨다."""
    assert is_valid_plate(plate) is True


@pytest.mark.parametrize(
    "plate",
    [
        "1가3456",       # 앞자리 1개 (2~3개 필요)
        "1234가5678",    # 앞자리 4개
        "12가345",       # 뒷자리 3개
        "12가34567",     # 뒷자리 5개
        "12쀍3456",      # 용도문자 화이트리스트 밖
        "가나3456",       # 한글 2개로 시작하지만 지역명 아님
        "ABC1234",       # 라틴 문자
        "12345678",      # 한글 없음
        "",
    ],
)
def test_invalid_plates(plate):
    """무효한 형식을 모두 거부한다."""
    assert is_valid_plate(plate) is False


def test_use_char_whitelist_is_enforced():
    """정규식의 ``[가-힣]`` 만으로는 부족하고 화이트리스트 검사가 실제로 동작한다.

    '쀍' 은 완성형 한글이므로 정규식은 통과하지만, 번호판 용도문자가 아니므로
    ``is_valid_plate`` 는 거부해야 한다. 이 테스트가 그 구분을 고정한다.
    """
    assert PLATE_RE_PLAIN.match("12쀍3456") is not None  # 정규식은 통과
    assert is_use_char("쀍") is False                     # 화이트리스트는 거부
    assert is_valid_plate("12쀍3456") is False            # 최종 판정은 거부


def test_region_regex_matches_both_spacings():
    """지역명 정규식이 공백 유무를 모두 허용한다."""
    assert PLATE_RE_REGION.match("서울12가3456") is not None
    assert PLATE_RE_REGION.match("서울 12가3456") is not None


# ---------------------------------------------------------------------------
# split_region
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("plate", "region", "body"),
    [
        ("서울12가3456", "서울", "12가3456"),
        ("경기12가3456", "경기", "12가3456"),
        ("12가3456", None, "12가3456"),
        ("123가4567", None, "123가4567"),
    ],
)
def test_split_region(plate, region, body):
    """지역명과 본문을 올바르게 분리한다."""
    assert split_region(plate) == (region, body)


def test_split_region_requires_digit_after_region():
    """앞 2글자가 한글이어도 3번째가 숫자가 아니면 지역명으로 보지 않는다."""
    assert split_region("가나다라") == (None, "가나다라")


# ---------------------------------------------------------------------------
# 자리 기반 혼동 보정
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "fixed"),
    [
        ("12가34S6", "12가3456"),   # S → 5
        ("I2가3456", "12가3456"),   # I → 1
        ("O2가3456", "02가3456"),   # O → 0
        ("12가34B6", "12가3486"),   # B → 8
        ("12가34Z6", "12가3426"),   # Z → 2
        ("12가34G6", "12가3466"),   # G → 6
        ("l2가3456", "12가3456"),   # 소문자 l → 1
    ],
)
def test_digit_confusion_correction(raw, fixed):
    """숫자 자리의 라틴 문자 오인식을 보정한다."""
    result, corrections = correct_confusions(raw)
    assert result == fixed
    assert len(corrections) == 1
    assert corrections[0]["rule"] == "digit-confusion"


@pytest.mark.parametrize(
    ("raw", "fixed"),
    [
        ("12갸3456", "12가3456"),  # 중성 혼동 ㅑ→ㅏ
        ("12겨3456", "12거3456"),  # ㅕ→ㅓ
        ("12며3456", "12머3456"),
        ("12간3456", "12가3456"),  # 받침 제거
        ("12강3456", "12가3456"),
    ],
)
def test_use_char_confusion_correction(raw, fixed):
    """한글 용도문자 자리의 오인식을 보정한다."""
    result, corrections = correct_confusions(raw)
    assert result == fixed
    assert corrections
    assert corrections[0]["rule"].startswith("use-char")


def test_no_reverse_correction_in_hangul_position():
    """한글 자리에서 ``0 → O`` 역보정을 **하지 않는다**(요구사항 5번).

    용도문자 자리에 숫자가 오면, 그것을 라틴 문자로 바꿔도 유효한 용도문자가
    되지 않는다. 따라서 숫자는 그대로 두고 형식 무효로 판정해야 한다.
    """
    result, _corrections = correct_confusions("1203456")
    # '0' 이 'O' 로 바뀌지 않았다.
    assert "O" not in result
    assert is_valid_plate(result) is False


def test_position_aware_not_global_replace():
    """보정이 **자리 기반**이며 전역 치환이 아니다.

    '0' 은 숫자 자리에서는 정상이므로 건드리지 않는다. 전역 치환이라면
    모든 '0' 이 바뀌어 번호가 망가진다.
    """
    result, corrections = correct_confusions("02가0000")
    assert result == "02가0000"
    assert corrections == []


def test_region_correction():
    """지역명 오인식을 닫힌 집합과 비교해 보정한다."""
    result, corrections = correct_confusions("서올12가3456")
    assert result == "서울12가3456"
    assert corrections[0]["rule"] == "region"


def test_correct_confusions_idempotent():
    """이미 올바른 입력은 보정하지 않는다(멱등)."""
    result, corrections = correct_confusions("12가3456")
    assert result == "12가3456"
    assert corrections == []


# ---------------------------------------------------------------------------
# expected_layout
# ---------------------------------------------------------------------------


def test_expected_layout():
    """7자/8자 번호판의 자리 유형을 반환하고, 그 외는 빈 리스트다."""
    assert expected_layout(7) == ["D", "D", "H", "D", "D", "D", "D"]
    assert expected_layout(8) == ["D", "D", "D", "H", "D", "D", "D", "D"]
    # 길이가 맞지 않으면 제약을 걸지 않는다(잘못된 제약이 더 위험하다).
    assert expected_layout(5) == []
    assert expected_layout(0) == []


# ---------------------------------------------------------------------------
# decide_status
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "confidence", "expected"),
    [
        ("", 0.99, "NO_PLATE"),
        ("12가3456", 0.95, "SUCCESS"),
        ("12가3456", 0.20, "LOW_CONFIDENCE"),
        ("XYZ123", 0.99, "FORMAT_INVALID"),
        ("12쀍3456", 0.99, "FORMAT_INVALID"),
    ],
)
def test_decide_status(text, confidence, expected):
    """계약서 enum 에 맞는 상태를 결정한다."""
    assert decide_status(text, confidence) == expected


def test_format_checked_before_confidence():
    """형식 검증이 신뢰도보다 **먼저** 평가된다.

    형식이 틀린 결과는 신뢰도가 높아도 DB 차량과 매칭할 수 없으므로
    ``FORMAT_INVALID`` 가 맞다. ``LOW_CONFIDENCE`` 로 보고하면
    "신뢰도만 올리면 쓸 수 있다" 는 잘못된 인상을 준다.
    """
    assert decide_status("INVALID", 0.01) == "FORMAT_INVALID"
    assert decide_status("INVALID", 0.99) == "FORMAT_INVALID"


def test_decide_status_threshold_boundary():
    """임계값 경계에서의 동작을 고정한다(>= 는 SUCCESS)."""
    assert decide_status("12가3456", 0.55, low_conf_threshold=0.55) == "SUCCESS"
    assert decide_status("12가3456", 0.549, low_conf_threshold=0.55) == "LOW_CONFIDENCE"


# ---------------------------------------------------------------------------
# postprocess
# ---------------------------------------------------------------------------


def test_postprocess_full_flow():
    """보정 + 검증 + 상태 판정이 한 번에 동작한다."""
    result = postprocess("12가34S6", 0.90)
    assert result.text == "12가3456"
    assert result.raw_text == "12가34S6"
    assert result.valid_format is True
    assert result.status == "SUCCESS"
    assert len(result.corrections) == 1


def test_postprocess_applies_correction_penalty():
    """보정이 일어나면 신뢰도를 깎는다.

    보정은 "모델이 틀렸다고 가정하고 고치는" 행위이므로, 원본의 불확실성을
    신뢰도에 반영하는 것이 정직하다.
    """
    clean = postprocess("12가3456", 0.90)
    corrected = postprocess("12가34S6", 0.90)
    assert corrected.confidence < clean.confidence
    assert clean.confidence == pytest.approx(0.90)


def test_postprocess_empty_input():
    """빈 입력은 ``NO_PLATE`` 이고 ``text`` 는 빈 문자열이다."""
    result = postprocess("", 0.90)
    assert result.status == "NO_PLATE"
    assert result.text == ""
    assert result.to_dict()["text"] is None  # 계약서는 null 을 기대한다


def test_postprocess_to_dict_shape():
    """``to_dict`` 가 기대하는 키를 모두 포함한다."""
    payload = postprocess("12가3456", 0.9).to_dict()
    for key in (
        "text",
        "raw_text",
        "confidence",
        "valid_format",
        "status",
        "corrections",
        "region",
        "char_confidences",
    ):
        assert key in payload


def test_postprocess_region_extracted():
    """지역명이 결과에 분리되어 담긴다."""
    result = postprocess("서울12가3456", 0.9)
    assert result.region == "서울"
    assert result.valid_format is True


# ---------------------------------------------------------------------------
# 다중 프레임 융합
# ---------------------------------------------------------------------------


def test_fuse_majority_vote():
    """다수결로 가장 많이 나온 결과를 고른다."""
    candidates = [
        FrameCandidate("12가3456", 0.90),
        FrameCandidate("12가3456", 0.85),
        FrameCandidate("12가3466", 0.70),
    ]
    assert fuse_frames(candidates).text == "12가3456"


def test_fuse_agreement_boosts_confidence():
    """프레임 합의가 높으면 신뢰도가 올라간다.

    3프레임 모두 같은 결과면 단일 프레임보다 확실히 믿을 만하다.
    """
    single = fuse_frames([FrameCandidate("12가3456", 0.70)])
    triple = fuse_frames([FrameCandidate("12가3456", 0.70)] * 3)
    assert triple.confidence > single.confidence


def test_fuse_char_vote_recovers_unseen_truth():
    """자리별 투표가 **어떤 프레임에도 없던 정답**을 복원한다.

    이것이 다중 프레임 융합의 가장 강력한 효과다. 세 프레임이 각각 다른
    자리에서 한 글자씩 틀렸지만, 자리별 다수결로 정답이 합성된다.
    """
    candidates = [
        FrameCandidate("12가3456", 0.80),
        FrameCandidate("12가3451", 0.70),  # 마지막 자리 오류
        FrameCandidate("12가9456", 0.75),  # 4번째 자리 오류
    ]
    assert fuse_frames(candidates, strategy="char-vote").text == "12가3456"


def test_fuse_quality_weighting():
    """품질 점수가 투표 가중치에 반영된다.

    흐린 프레임(품질 0.1)의 표가 선명한 프레임(품질 1.0)과 같은 무게를
    가지면 안 된다. 신뢰도가 더 높아도 품질이 낮으면 질 수 있다.
    """
    candidates = [
        FrameCandidate("12가3456", 0.60, quality=1.0),
        FrameCandidate("99나8888", 0.75, quality=0.05),
    ]
    assert fuse_frames(candidates).text == "12가3456"


def test_fuse_normalizes_before_voting():
    """투표 전에 후처리를 적용해 같은 결과가 표를 나누지 않게 한다.

    '12가34S6' 과 '12가3456' 은 보정하면 같은 번호다. 보정 전에 투표하면
    표가 갈려 엉뚱한 후보가 이길 수 있다.
    """
    candidates = [
        FrameCandidate("12가34S6", 0.80),
        FrameCandidate("12가3456", 0.80),
        FrameCandidate("99나8888", 0.85),
    ]
    assert fuse_frames(candidates).text == "12가3456"


def test_fuse_best_confidence_strategy():
    """``best-confidence`` 전략은 최고 신뢰도 1건을 고른다(융합 없음)."""
    candidates = [
        FrameCandidate("12가3456", 0.60),
        FrameCandidate("34나7890", 0.95),
    ]
    assert fuse_frames(candidates, strategy="best-confidence").text == "34나7890"


def test_fuse_accepts_dict_and_tuple():
    """dict 와 튜플 형태의 후보도 받는다(호출자 편의)."""
    assert fuse_frames([{"text": "12가3456", "confidence": 0.9}]).text == "12가3456"
    assert fuse_frames([("12가3456", 0.9), ("12가3456", 0.8)]).text == "12가3456"


def test_fuse_empty_returns_no_plate():
    """빈 입력은 ``NO_PLATE`` 를 반환한다(예외 없음)."""
    result = fuse_frames([])
    assert result.status == "NO_PLATE"
    assert result.text == ""


def test_fuse_all_blank_texts():
    """모든 후보가 빈 문자열이면 ``NO_PLATE`` 다."""
    assert fuse_frames([FrameCandidate("", 0.9), FrameCandidate("", 0.8)]).status == "NO_PLATE"


# ---------------------------------------------------------------------------
# 문자셋 유틸
# ---------------------------------------------------------------------------


def test_nearest_use_char():
    """초성/중성 유사도로 용도문자를 보정한다."""
    assert nearest_use_char("가") == "가"   # 이미 유효
    assert nearest_use_char("겨") == "거"   # 중성 혼동
    assert nearest_use_char("며") == "머"
    assert nearest_use_char("A") is None    # 한글 아님


def test_use_chars_are_all_complete_hangul():
    """용도문자 화이트리스트가 모두 완성형 한글 음절이다.

    자모(ㄱ, ㅏ)나 라틴 문자가 섞이면 ``nearest_use_char`` 의 초성/중성 분해가
    깨지므로, 이 불변식을 테스트로 고정한다.
    """
    for ch in USE_CHARS:
        assert len(ch) == 1, f"{ch!r} 가 한 글자가 아니다"
        assert "가" <= ch <= "힣", f"{ch!r} 는 완성형 한글이 아니다"


def test_private_use_chars_have_no_jongseong():
    """비사업용 승용 용도문자(가나다라마/거너더러머/...)에는 받침이 없다.

    일반 승용차 번호판의 용도문자는 자음 + ㅏ/ㅓ/ㅗ/ㅜ 조합이라 받침이 없다.
    받침 있는 글자(``공``, ``국``, ``합``, ``육`` 등)는 **군·관용·특수 번호판**
    전용이므로 ``SPECIAL_USE_CHARS`` 에만 존재해야 한다. 이 구분이 섞이면
    받침 제거 보정 규칙(``간`` → ``가``)이 특수 번호판을 망가뜨린다.
    """
    from app.ocr.charset import PRIVATE_USE_CHARS, SPECIAL_USE_CHARS

    for ch in PRIVATE_USE_CHARS:
        jongseong = (ord(ch) - 0xAC00) % 28
        assert jongseong == 0, f"비사업용 문자 {ch!r} 에 받침이 있다"

    # 특수 번호판 문자에는 받침이 있는 것이 정상이다(예: 공, 국, 합).
    assert any((ord(ch) - 0xAC00) % 28 != 0 for ch in SPECIAL_USE_CHARS)


def test_summarize_corrections():
    """보정 규칙별 적용 횟수를 집계한다(운영 모니터링용)."""
    results = [postprocess("12가34S6", 0.9), postprocess("I2갸3456", 0.8)]
    summary = summarize_corrections(results)
    assert summary["digit-confusion"] == 2
    assert summary["use-char-table"] == 1
