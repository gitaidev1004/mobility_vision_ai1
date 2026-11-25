# -*- coding: utf-8 -*-
"""한국 차량번호판 형식 검증 규칙.

번호판 OCR 의 출력 문자열이 '한국 번호판으로 존재할 수 있는 형태'인지
정규식과 허용 문자 집합으로 판정한다. 외부 라이브러리 없이 동작한다.

판정 결과는 PlateCheck 로 돌려주며, 다음 세 가지를 분리해서 담는다.
  - ok          : 모든 규칙을 통과했는지
  - plate_type  : 어떤 형식으로 해석되었는지
  - reasons     : 통과하지 못한 이유 목록(사람이 읽을 수 있게)
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# 1) 허용 문자 집합
# ---------------------------------------------------------------------------
# 자가용(비사업용) 용도 기호. 자음 7종 x 모음 5종 조합이 실제로 쓰인다.
HANGUL_PRIVATE = set(
    "가나다라마"
    "거너더러머"
    "고노도로모"
    "구누두루무"
    "버보부"
    "서소수"
    "어오우"
    "저조주"
)
# 사업용(운수) 용도 기호.
HANGUL_COMMERCIAL = set("아바사자")
# 택배 화물.
HANGUL_DELIVERY = set("배")
# 렌터카.
HANGUL_RENTAL = set("허하호")

HANGUL_ALL = (
    HANGUL_PRIVATE | HANGUL_COMMERCIAL | HANGUL_DELIVERY | HANGUL_RENTAL
)

# 구형 번호판에 인쇄되던 지역명(17개 광역시·도).
REGION_NAMES = (
    "서울", "부산", "대구", "인천", "광주", "대전", "울산", "세종",
    "경기", "강원", "충북", "충남", "전북", "전남", "경북", "경남", "제주",
)

# 번호판에 나타날 수 있는 모든 문자(문자셋 정의와 일치해야 한다).
ALLOWED_CHARS = set("0123456789") | HANGUL_ALL | set("".join(REGION_NAMES))

# ---------------------------------------------------------------------------
# 2) 형식별 정규식
# ---------------------------------------------------------------------------
_H = "".join(sorted(HANGUL_ALL))
_REG = "|".join(REGION_NAMES)

PATTERNS: list[tuple[str, str, re.Pattern[str]]] = [
    # 신형 3자리(2019~): 123가4567
    ("NEW3", "신형 3자리(2019년 이후)",
     re.compile(rf"^[0-9]{{3}}[{_H}][0-9]{{4}}$")),
    # 신형 2자리(2004~2019): 12가3456
    ("NEW2", "신형 2자리(2004~2019년)",
     re.compile(rf"^[0-9]{{2}}[{_H}][0-9]{{4}}$")),
    # 구형 지역명(1973~2006): 서울34나5678
    ("OLD_REGION", "구형 지역명 포함(1973~2006년)",
     re.compile(rf"^(?:{_REG})[0-9]{{2}}[{_H}][0-9]{{4}}$")),
]

# 용도 기호별 분류용 표.
USAGE_TABLE: list[tuple[str, set[str]]] = [
    ("자가용", HANGUL_PRIVATE),
    ("사업용", HANGUL_COMMERCIAL),
    ("택배", HANGUL_DELIVERY),
    ("렌터카", HANGUL_RENTAL),
]

MIN_LEN, MAX_LEN = 7, 9


@dataclass
class PlateCheck:
    """번호판 한 건에 대한 검증 결과."""

    raw: str                          # 입력 원문
    text: str                         # 정규화된 문자열(공백/하이픈 제거)
    ok: bool = False                  # 모든 규칙 통과 여부
    plate_type: str = "UNKNOWN"       # NEW3 / NEW2 / OLD_REGION / UNKNOWN
    type_label: str = "미확인"
    usage: str = "미확인"             # 자가용 / 사업용 / 택배 / 렌터카
    region: str | None = None         # 구형 번호판의 지역명
    hangul: str | None = None         # 용도 기호 1자
    reasons: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """한 줄 요약 문자열."""
        state = "OK" if self.ok else "NG"
        why = "" if self.ok else " <- " + "; ".join(self.reasons)
        return f"[{state}] {self.text} | {self.plate_type} | {self.usage}{why}"


def normalize(text: str) -> str:
    """OCR 출력 문자열을 비교 가능한 형태로 정규화한다.

    - 전각 문자를 반각으로 바꾼다(NFKC).
    - 공백, 하이픈, 점 등 번호판에 인쇄되지 않는 구분 기호를 제거한다.
    - 영문 소문자는 대문자로 올린다(혼동 보정 단계에서 쓰인다).
    """
    if text is None:
        return ""
    t = unicodedata.normalize("NFKC", str(text))
    t = re.sub(r"[\s\-_.·ㆍ/]", "", t)
    return t.upper()


def check_length(text: str) -> list[str]:
    """문자열 길이 검증. 한국 번호판은 7~9자 사이다."""
    reasons: list[str] = []
    n = len(text)
    if n < MIN_LEN:
        reasons.append(f"길이 부족({n}자 < {MIN_LEN}자)")
    elif n > MAX_LEN:
        reasons.append(f"길이 초과({n}자 > {MAX_LEN}자)")
    return reasons


def check_charset(text: str) -> list[str]:
    """허용 문자 규칙 검증. 번호판 문자셋에 없는 글자를 찾아낸다."""
    bad = [c for c in text if c not in ALLOWED_CHARS]
    if bad:
        uniq = "".join(dict.fromkeys(bad))
        return [f"허용되지 않은 문자 포함('{uniq}')"]
    return []


def classify_usage(hangul: str) -> str:
    """용도 기호 1자로 차량 용도를 분류한다."""
    for label, group in USAGE_TABLE:
        if hangul in group:
            return label
    return "미확인"


def validate(text: str) -> PlateCheck:
    """번호판 문자열 하나를 검증한다."""
    norm = normalize(text)
    res = PlateCheck(raw=str(text), text=norm)

    # (1) 길이 검증
    res.reasons.extend(check_length(norm))
    # (2) 허용 문자 검증
    res.reasons.extend(check_charset(norm))

    # (3) 정규식 형식 검증 — 위에서 걸린 경우에도 형식 판정은 시도한다.
    for code, label, pat in PATTERNS:
        if pat.match(norm):
            res.plate_type, res.type_label = code, label
            break
    else:
        res.reasons.append("어떤 번호판 형식에도 맞지 않음")

    # (4) 지역명/용도 기호 분해
    if res.plate_type == "OLD_REGION":
        res.region = norm[:2]
        res.hangul = norm[4]
    elif res.plate_type in {"NEW2", "NEW3"}:
        res.hangul = norm[2] if res.plate_type == "NEW2" else norm[3]
    if res.hangul:
        res.usage = classify_usage(res.hangul)

    res.ok = not res.reasons
    return res


def is_valid(text: str) -> bool:
    """간단 호출용. 형식이 유효하면 True."""
    return validate(text).ok


def _demo() -> None:
    samples = [
        "12가3456",        # 신형 2자리 자가용
        "서울 34나5678",   # 구형 지역명
        "87로2345",        # 신형 2자리 자가용
        "123가4567",       # 신형 3자리
        "12바3456",        # 사업용
        "34허5678",        # 렌터카
        "12A3456",         # 한글 자리에 영문 -> 혼동 보정 필요
        "12가345",         # 길이 부족
        "1234가5678",      # 숫자 자리 수 초과 -> 형식 오류
        "12가34567",       # 뒷자리 5자 -> 형식 오류
    ]
    print("번호판 형식 검증 결과")
    print("-" * 78)
    for s in samples:
        print(validate(s).summary())
    print("-" * 78)
    print(f"허용 문자 수: {len(ALLOWED_CHARS)}자 "
          f"(숫자 10 + 용도 기호 {len(HANGUL_ALL)} + 지역명 음절)")


if __name__ == "__main__":
    _demo()
