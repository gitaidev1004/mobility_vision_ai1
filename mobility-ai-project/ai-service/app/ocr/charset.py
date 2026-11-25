"""한국 자동차 번호판 문자셋 정의.

왜 문자셋을 따로 모듈로 빼는가?
    1. CTC 출력 차원 수(= ``len(CHARSET) + 1``)가 모델 정의, 디코더, 후처리에서
       모두 똑같아야 한다. 한 곳에서만 정의해 불일치 버그를 원천 차단한다.
    2. 한국 번호판은 "숫자 + 제한된 한글 용도문자 + (구형) 지역명" 구조다.
       한글 11,172자 전체를 클래스로 두면 분류 난이도가 폭증하고 오인식이 늘어난다.
       실제 사용되는 44자 정도로 제한하는 것이 정확도에 결정적이다.

번호판 체계 요약(2006년 이후 신형 / 구형 혼용):
    - 신형 비사업용 승용: ``12가3456`` (앞 2자리 숫자) 또는 ``123가4567`` 형태의
      3자리 확장형 ``123가4567``. 본 프로젝트 계약서 정규식은 ``\\d{2,3}[가-힣]\\d{4}`` 이다.
    - 구형(지역명 포함): ``서울 12가3456`` → 정규식 ``[가-힣]{2}\\s?\\d{2}[가-힣]\\d{4}``.
"""

from __future__ import annotations

from typing import Final

# ---------------------------------------------------------------------------
# 1) 숫자
# ---------------------------------------------------------------------------

#: 번호판에 쓰이는 숫자 10자.
DIGITS: Final[str] = "0123456789"


# ---------------------------------------------------------------------------
# 2) 용도 구분 한글 (가운데 한 글자)
# ---------------------------------------------------------------------------
# 한국 번호판 가운데 글자는 차량 용도를 나타낸다. 자음 ㄱ~ㅎ 에 모음 ㅏ/ㅓ/ㅗ/ㅜ 를
# 결합한 형태가 비사업용으로 쓰이고, 그 외 사업용/택배/렌터카/외교 등에 전용 문자가 있다.
# 아래 목록은 실제 도로에서 관측되는 문자만 모았다(가~흐 전체 범위가 아니다).

#: 비사업용(자가용) 승용차 용도문자. ``ㄱ~ㅁ`` 계열 + ``ㅏ/ㅓ/ㅗ/ㅜ`` 조합.
PRIVATE_USE_CHARS: Final[str] = (
    "가나다라마"  # ㅏ 계열
    "거너더러머"  # ㅓ 계열
    "고노도로모"  # ㅗ 계열
    "구누두루무"  # ㅜ 계열
)

#: 사업용(영업용) 차량 용도문자. 운수사업 차량에 배정된다.
COMMERCIAL_USE_CHARS: Final[str] = "바사아자"

#: 택시/버스/화물 등 사업용 확장 문자(``ㅂ/ㅅ/ㅇ/ㅈ`` 계열 전체).
COMMERCIAL_EXT_CHARS: Final[str] = (
    "버서어저"
    "보소오조"
    "부수우주"
)

#: 렌터카(대여사업용) 전용 문자.
RENTAL_USE_CHARS: Final[str] = "하허호"

#: 택배(배송) 전용 문자.
DELIVERY_USE_CHARS: Final[str] = "배"

#: 군용/관용/외교 등 특수 번호판 문자. 실무 운영에서 간헐적으로 등장한다.
SPECIAL_USE_CHARS: Final[str] = "육해공국합외협정대영준"

#: 번호판 가운데 자리에 올 수 있는 모든 한글 용도문자(중복 제거, 정렬).
#: 왜 ``sorted(set(...))`` 인가? 분류 라벨 인덱스를 결정적(deterministic)으로 만들어야
#: 학습 체크포인트와 추론 코드의 라벨 순서가 영원히 일치하기 때문이다.
USE_CHARS: Final[str] = "".join(
    sorted(
        set(
            PRIVATE_USE_CHARS
            + COMMERCIAL_USE_CHARS
            + COMMERCIAL_EXT_CHARS
            + RENTAL_USE_CHARS
            + DELIVERY_USE_CHARS
            + SPECIAL_USE_CHARS
        )
    )
)


# ---------------------------------------------------------------------------
# 3) 지역명 (구형 번호판 앞 2글자)
# ---------------------------------------------------------------------------

#: 구형 번호판 앞머리에 오는 시·도 지역명 17개(광역자치단체 기준).
REGION_NAMES: Final[tuple[str, ...]] = (
    "서울", "부산", "대구", "인천", "광주", "대전", "울산", "세종",
    "경기", "강원", "충북", "충남", "전북", "전남", "경북", "경남", "제주",
)

#: 지역명을 구성하는 개별 한글 음절(중복 제거). OCR 은 음절 단위로 분류하므로 필요하다.
REGION_CHARS: Final[str] = "".join(sorted({ch for name in REGION_NAMES for ch in name}))


# ---------------------------------------------------------------------------
# 4) 최종 문자셋 및 인덱스 매핑
# ---------------------------------------------------------------------------

#: 모델이 분류하는 전체 문자 집합. 순서가 곧 클래스 인덱스다(0-based).
#: 숫자 → 용도문자 → 지역명문자 순서로 고정한다(사람이 디버깅할 때 읽기 쉽다).
CHARSET: Final[str] = DIGITS + USE_CHARS + "".join(
    ch for ch in REGION_CHARS if ch not in USE_CHARS
)

#: CTC 의 blank 토큰 인덱스. 관례상 0 을 blank 로 두는 구현도 있으나,
#: 여기서는 ``len(CHARSET)`` 을 blank 로 쓴다. 이유: 문자 인덱스를 0부터 시작하게 두면
#: ``CHARSET[idx]`` 로 바로 조회할 수 있어 디코더 코드가 단순해진다.
BLANK_INDEX: Final[int] = len(CHARSET)

#: CTC/Softmax 출력 클래스 수 = 문자 수 + blank 1.
NUM_CLASSES: Final[int] = len(CHARSET) + 1

#: Attention Seq2Seq 용 특수 토큰. CTC 와 달리 시작/종료 토큰이 필요하다.
SOS_TOKEN: Final[int] = len(CHARSET) + 1  # start of sequence
EOS_TOKEN: Final[int] = len(CHARSET) + 2  # end of sequence
PAD_TOKEN: Final[int] = len(CHARSET) + 3  # padding

#: Attention 디코더의 어휘 크기(문자 + blank + SOS/EOS/PAD).
ATTENTION_VOCAB_SIZE: Final[int] = len(CHARSET) + 4

#: 문자 → 인덱스. 역방향 조회를 O(1) 로 만들기 위해 dict 로 미리 만든다.
CHAR_TO_INDEX: Final[dict[str, int]] = {ch: i for i, ch in enumerate(CHARSET)}

#: 인덱스 → 문자.
INDEX_TO_CHAR: Final[dict[int, str]] = {i: ch for i, ch in enumerate(CHARSET)}

#: 숫자 문자 집합(빠른 멤버십 검사용 frozenset).
DIGIT_SET: Final[frozenset[str]] = frozenset(DIGITS)

#: 한글 용도문자 집합.
USE_CHAR_SET: Final[frozenset[str]] = frozenset(USE_CHARS)

#: 지역명 음절 집합.
REGION_CHAR_SET: Final[frozenset[str]] = frozenset(REGION_CHARS)


def encode(text: str, *, strict: bool = False) -> list[int]:
    """문자열을 클래스 인덱스 리스트로 변환한다(학습 라벨 생성용).

    Args:
        text: 번호판 문자열. 공백은 무시한다(지역명과 번호 사이 공백 때문).
        strict: True 면 문자셋에 없는 문자를 만나면 ``ValueError``.
            False(기본) 면 해당 문자를 건너뛴다 — 학습 데이터에 오염된 라벨이
            섞여 있어도 배치 전체가 죽지 않게 하기 위한 방어적 기본값이다.

    Returns:
        인덱스 리스트.

    Raises:
        ValueError: ``strict=True`` 이고 미지의 문자가 있을 때.

    Examples:
        >>> encode("12가3456")[:3]
        [1, 2, 10]
    """
    out: list[int] = []
    for ch in text:
        if ch.isspace():
            continue
        idx = CHAR_TO_INDEX.get(ch)
        if idx is None:
            if strict:
                msg = f"문자셋에 없는 문자입니다: {ch!r} (text={text!r})"
                raise ValueError(msg)
            continue
        out.append(idx)
    return out


def decode(indices: list[int] | tuple[int, ...]) -> str:
    """클래스 인덱스 리스트를 문자열로 변환한다.

    blank/SOS/EOS/PAD 등 문자셋 범위를 벗어난 인덱스는 조용히 건너뛴다.
    디코더가 넘겨주는 시퀀스에는 특수 토큰이 섞여 있는 것이 정상이기 때문이다.

    Args:
        indices: 클래스 인덱스 시퀀스.

    Returns:
        복원된 문자열.

    Examples:
        >>> decode([1, 2, 10, 3, 4, 5, 6])[:2]
        '12'
    """
    return "".join(INDEX_TO_CHAR[i] for i in indices if i in INDEX_TO_CHAR)


def is_digit(ch: str) -> bool:
    """문자가 번호판 숫자인지 판정한다.

    ``str.isdigit()`` 을 쓰지 않는 이유: 전각 숫자('１')나 한자 숫자도 True 가 되어
    번호판 문자셋 밖의 문자를 통과시킨다. 명시적 집합 검사가 안전하다.

    Args:
        ch: 한 글자.

    Returns:
        ``0``~``9`` 이면 True.
    """
    return ch in DIGIT_SET


def is_use_char(ch: str) -> bool:
    """문자가 유효한 한글 용도문자인지 판정한다.

    Args:
        ch: 한 글자.

    Returns:
        용도문자 화이트리스트에 있으면 True.
    """
    return ch in USE_CHAR_SET


def is_region_char(ch: str) -> bool:
    """문자가 지역명 음절인지 판정한다.

    Args:
        ch: 한 글자.

    Returns:
        지역명 음절 집합에 있으면 True.
    """
    return ch in REGION_CHAR_SET


def is_hangul(ch: str) -> bool:
    """문자가 완성형 한글 음절(U+AC00~U+D7A3)인지 판정한다.

    Args:
        ch: 한 글자.

    Returns:
        한글 음절이면 True.
    """
    return len(ch) == 1 and "가" <= ch <= "힣"


def nearest_use_char(ch: str) -> str | None:
    """오인식된 한글을 가장 비슷한 용도문자로 보정한다.

    왜 초성/중성 분해로 비교하는가?
        OCR 은 '거'를 '겨', '머'를 '며' 처럼 중성(모음)을 혼동하는 경우가 많다.
        완성형 코드포인트 거리로는 이런 유사성이 드러나지 않는다. 유니코드 한글은
        ``0xAC00 + (초성*588) + (중성*28) + 종성`` 규칙을 따르므로, 분해해서
        초성이 같은 후보를 우선 고르면 실제 혼동 패턴과 잘 맞는다.

    Args:
        ch: 보정 대상 한글 한 글자.

    Returns:
        보정된 용도문자. 후보를 찾지 못하면 None.
    """
    if ch in USE_CHAR_SET:
        return ch
    if not is_hangul(ch):
        return None

    base = ord(ch) - 0xAC00
    cho, jung = base // 588, (base % 588) // 28

    # 1순위: 초성과 중성이 모두 같은 용도문자(= 종성만 다른 경우).
    # 2순위: 초성만 같은 용도문자(= 모음 혼동).
    same_cho_jung: list[str] = []
    same_cho: list[str] = []
    for cand in USE_CHARS:
        c_base = ord(cand) - 0xAC00
        c_cho, c_jung = c_base // 588, (c_base % 588) // 28
        if c_cho == cho:
            if c_jung == jung:
                same_cho_jung.append(cand)
            same_cho.append(cand)

    if same_cho_jung:
        return same_cho_jung[0]
    if same_cho:
        # 중성 인덱스 차이가 가장 작은 후보를 고른다(ㅏ↔ㅑ 는 인덱스 1 차이).
        return min(
            same_cho,
            key=lambda c: abs(((ord(c) - 0xAC00) % 588) // 28 - jung),
        )
    return None


def charset_summary() -> dict[str, object]:
    """문자셋 구성 요약을 반환한다(``/api/v1/models/info`` 응답 및 디버깅용).

    Returns:
        클래스 수, 구간별 문자 수, 각 구간의 문자열을 담은 dict.
    """
    return {
        "num_classes": NUM_CLASSES,
        "blank_index": BLANK_INDEX,
        "digits": DIGITS,
        "use_chars": USE_CHARS,
        "use_char_count": len(USE_CHARS),
        "region_names": list(REGION_NAMES),
        "region_chars": REGION_CHARS,
        "charset_length": len(CHARSET),
        "attention_vocab_size": ATTENTION_VOCAB_SIZE,
    }


__all__ = [
    "ATTENTION_VOCAB_SIZE",
    "BLANK_INDEX",
    "CHARSET",
    "CHAR_TO_INDEX",
    "COMMERCIAL_EXT_CHARS",
    "COMMERCIAL_USE_CHARS",
    "DELIVERY_USE_CHARS",
    "DIGITS",
    "DIGIT_SET",
    "EOS_TOKEN",
    "INDEX_TO_CHAR",
    "NUM_CLASSES",
    "PAD_TOKEN",
    "PRIVATE_USE_CHARS",
    "REGION_CHARS",
    "REGION_CHAR_SET",
    "REGION_NAMES",
    "RENTAL_USE_CHARS",
    "SOS_TOKEN",
    "SPECIAL_USE_CHARS",
    "USE_CHARS",
    "USE_CHAR_SET",
    "charset_summary",
    "decode",
    "encode",
    "is_digit",
    "is_hangul",
    "is_region_char",
    "is_use_char",
    "nearest_use_char",
]
