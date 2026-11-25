# -*- coding: utf-8 -*-
"""한국 자동차 번호판 문자셋 정의.

문자셋을 모듈 하나로 분리하는 이유는 두 가지다.
  1. CTC 출력 차원(len(CHARSET)+1)이 모델 정의·디코더·후처리에서 모두 같아야 한다.
  2. 한글 11,172자 전체를 클래스로 두면 난이도가 폭증한다. 실제로 쓰이는
     용도문자와 지역명 음절만 남겨 클래스 수를 80 미만으로 억제한다.

실행:  python 03_dataset/charset.py
"""
from __future__ import annotations

# --------------------------------------------------------------------------
# 1) 숫자 10자
# --------------------------------------------------------------------------
DIGITS = "0123456789"

# --------------------------------------------------------------------------
# 2) 용도 구분 한글(가운데 한 글자)
# --------------------------------------------------------------------------
#: 비사업용(자가용) 승용차. 자음 ㄱ~ㅁ 에 모음 ㅏ/ㅓ/ㅗ/ㅜ 를 결합한 20자.
PRIVATE_USE_CHARS = (
    "가나다라마"   # ㅏ 계열
    "거너더러머"   # ㅓ 계열
    "고노도로모"   # ㅗ 계열
    "구누두루무"   # ㅜ 계열
)

#: 사업용(영업용) 기본 문자. 운수사업 차량에 배정된다.
COMMERCIAL_USE_CHARS = "바사아자"

#: 사업용 확장 문자(ㅂ/ㅅ/ㅇ/ㅈ 계열).
COMMERCIAL_EXT_CHARS = "버서어저" "보소오조" "부수우주"

#: 렌터카(대여사업용) 전용.
RENTAL_USE_CHARS = "하허호"

#: 택배(화물 배송) 전용.
DELIVERY_USE_CHARS = "배"

#: 군용·관용·외교 등 특수 번호판 문자.
SPECIAL_USE_CHARS = "육해공국합외협정대영준"

#: 가운데 자리에 올 수 있는 모든 용도문자. sorted(set(...)) 로 순서를 고정해
#: 학습 체크포인트와 추론 코드의 라벨 인덱스가 영구히 일치하도록 만든다.
USE_CHARS = "".join(sorted(set(
    PRIVATE_USE_CHARS + COMMERCIAL_USE_CHARS + COMMERCIAL_EXT_CHARS
    + RENTAL_USE_CHARS + DELIVERY_USE_CHARS + SPECIAL_USE_CHARS
)))

# --------------------------------------------------------------------------
# 3) 지역명(구형 번호판 앞 2글자) — 광역자치단체 17개
# --------------------------------------------------------------------------
REGION_NAMES = (
    "서울", "부산", "대구", "인천", "광주", "대전", "울산", "세종",
    "경기", "강원", "충북", "충남", "전북", "전남", "경북", "경남", "제주",
)

#: 지역명을 구성하는 개별 음절. OCR 은 음절 단위로 분류하므로 분해해 둔다.
REGION_CHARS = "".join(sorted({ch for name in REGION_NAMES for ch in name}))

# --------------------------------------------------------------------------
# 4) 최종 문자셋과 인덱스 매핑
# --------------------------------------------------------------------------
#: 숫자 → 용도문자 → (용도문자에 없는) 지역명 음절 순서로 고정한다.
CHARSET = DIGITS + USE_CHARS + "".join(
    ch for ch in REGION_CHARS if ch not in USE_CHARS
)

#: blank 를 맨 뒤에 둔다. 그러면 CHARSET[idx] 로 바로 조회할 수 있어 디코더가 단순해진다.
BLANK_INDEX = len(CHARSET)
NUM_CLASSES = len(CHARSET) + 1

CHAR_TO_INDEX = {ch: i for i, ch in enumerate(CHARSET)}
INDEX_TO_CHAR = {i: ch for i, ch in enumerate(CHARSET)}

DIGIT_SET = frozenset(DIGITS)
USE_CHAR_SET = frozenset(USE_CHARS)
REGION_CHAR_SET = frozenset(REGION_CHARS)


def encode(text: str, strict: bool = False) -> list[int]:
    """문자열을 클래스 인덱스 리스트로 바꾼다(학습 라벨 생성용).

    공백은 무시한다(구형 번호판의 지역명과 번호 사이 공백 때문).
    strict=False 면 문자셋에 없는 글자를 건너뛴다 — 라벨 하나가 오염돼도
    배치 전체가 죽지 않게 하기 위한 방어적 기본값이다.
    """
    out: list[int] = []
    for ch in text:
        if ch.isspace():
            continue
        idx = CHAR_TO_INDEX.get(ch)
        if idx is None:
            if strict:
                raise ValueError(f"문자셋에 없는 문자: {ch!r} (text={text!r})")
            continue
        out.append(idx)
    return out


def decode(indices) -> str:
    """클래스 인덱스 리스트를 문자열로 되돌린다. 특수 토큰은 조용히 건너뛴다."""
    return "".join(INDEX_TO_CHAR[i] for i in indices if i in INDEX_TO_CHAR)


def is_digit(ch: str) -> bool:
    """번호판 숫자인지 판정한다. str.isdigit() 은 전각 숫자도 통과시켜 쓰지 않는다."""
    return ch in DIGIT_SET


def is_use_char(ch: str) -> bool:
    """유효한 용도문자(화이트리스트)인지 판정한다."""
    return ch in USE_CHAR_SET


def summary() -> dict:
    """문자셋 구성 요약을 반환한다(디버깅 및 모델 정보 API 용)."""
    return {
        "num_classes": NUM_CLASSES,
        "blank_index": BLANK_INDEX,
        "digit_count": len(DIGITS),
        "use_char_count": len(USE_CHARS),
        "region_name_count": len(REGION_NAMES),
        "region_char_count": len(REGION_CHARS),
        "charset_length": len(CHARSET),
    }


if __name__ == "__main__":
    info = summary()
    print("=== 한국 번호판 문자셋 ===")
    for key, value in info.items():
        print(f"  {key:20s} = {value}")
    print(f"\n숫자({len(DIGITS)}자)     : {DIGITS}")
    print(f"용도문자({len(USE_CHARS)}자) : {USE_CHARS}")
    print(f"지역명 음절({len(REGION_CHARS)}자): {REGION_CHARS}")
    print(f"\n전체 문자셋({len(CHARSET)}자):\n  {CHARSET}")
    for sample in ("12가3456", "서울 34나5678", "87로2345"):
        ids = encode(sample)
        print(f"\nencode({sample!r})\n  -> {ids}\n  -> decode: {decode(ids)!r}")
