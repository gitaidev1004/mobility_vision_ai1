"""한국 번호판 후처리: 정규식 검증, 자리 기반 혼동 보정, 다중 프레임 융합.

후처리가 왜 모델만큼 중요한가?
    번호판은 **형식이 정해진 구조화 문자열**이다. OCR 이 '12가34S6' 을 내놓았다면
    7번째 자리는 숫자여야 하므로 'S' 는 반드시 '5' 다. 이 한 줄의 규칙이
    모델을 더 학습시키는 것보다 훨씬 큰 정확도 개선을 준다.
    실무 LPR 시스템의 정확도는 "모델 + 후처리" 의 곱으로 결정된다.

이 모듈은 numpy 조차 필요하지 않다(순수 파이썬). 단위 테스트에서 바로 실행된다.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from app.ocr.charset import (
    DIGIT_SET,
    REGION_NAMES,
    USE_CHAR_SET,
    is_hangul,
    nearest_use_char,
)

# ---------------------------------------------------------------------------
# 1) 번호판 정규식
# ---------------------------------------------------------------------------

#: 신형(지역명 없음): 앞 2~3자리 숫자 + 용도문자 1 + 뒤 4자리 숫자.
#: 예: ``12가3456``, ``123가4567``
PLATE_RE_PLAIN: re.Pattern[str] = re.compile(r"^\d{2,3}[가-힣]\d{4}$")

#: 구형(지역명 포함): 지역명 2글자 + (공백 선택) + 2자리 숫자 + 용도문자 + 4자리 숫자.
#: 예: ``서울12가3456``, ``서울 12가3456``
PLATE_RE_REGION: re.Pattern[str] = re.compile(r"^[가-힣]{2}\s?\d{2}[가-힣]\d{4}$")

#: 두 형식을 한 번에 검사하는 패턴.
PLATE_RE_ANY: re.Pattern[str] = re.compile(
    r"^(?:\d{2,3}[가-힣]\d{4}|[가-힣]{2}\s?\d{2}[가-힣]\d{4})$"
)

#: 외교/임시 등 특수 번호판(참고용). 본 프로젝트의 ``valid_format`` 판정에는
#: 쓰지 않지만, 교재에서 "형식이 더 있다" 는 점을 보여 주기 위해 남긴다.
PLATE_RE_DIPLOMAT: re.Pattern[str] = re.compile(r"^외교\s?\d{3}-?\d{3}$")


# ---------------------------------------------------------------------------
# 2) 혼동 보정 표
# ---------------------------------------------------------------------------

#: **숫자 자리**에서 발생하는 오인식 보정표(라틴 문자/기호 → 숫자).
#: 근거: OCR 이 흔히 혼동하는 글리프 쌍이다.
#:   O/D/Q → 0 (닫힌 타원)
#:   I/l/|/! → 1 (단일 수직선)
#:   Z → 2, B → 8, S → 5, G/b → 6, T → 7, A → 4, g/q → 9
DIGIT_CONFUSION: dict[str, str] = {
    "O": "0", "o": "0", "D": "0", "Q": "0", "U": "0",
    "I": "1", "i": "1", "l": "1", "L": "1", "|": "1", "!": "1", "/": "1",
    "Z": "2", "z": "2",
    "E": "3", "e": "3",
    "A": "4", "h": "4",
    "S": "5", "s": "5", "$": "5",
    "G": "6", "b": "6", "C": "6",
    "T": "7", "t": "7", "?": "7",
    "B": "8", "R": "8", "&": "8",
    "g": "9", "q": "9", "P": "9",
}

#: 숫자 자리에서 흔히 섞여 들어오는 **한글** → 숫자 보정.
#: 모양이 유사한 쌍만 담았다(예: '이'→'0' 은 모양 근거가 없으니 넣지 않는다).
DIGIT_CONFUSION_HANGUL: dict[str, str] = {
    "ㅇ": "0",
    "ㅣ": "1",
    "ㄱ": "7",
    "ㄴ": "1",
}

#: **한글 용도문자 자리**에서 발생하는 오인식 보정표.
#: 중요: 여기에 ``"0": "O"`` 같은 **역보정은 절대 넣지 않는다.**
#:   한국 번호판 용도문자는 전부 완성형 한글이다. 숫자나 라틴 문자가
#:   용도문자 자리에 왔다면 그것은 "OCR 이 한글을 못 읽었다" 는 뜻이고,
#:   이때 숫자를 라틴 문자로 바꿔 봐야 유효한 용도문자가 되지 않는다.
#:   대신 ``nearest_use_char`` 로 초성/중성 유사도 기반 보정을 시도한다.
#: 여기에는 "한글 → 한글" 오인식 중 글리프 유사 쌍만 담는다.
USE_CHAR_CONFUSION: dict[str, str] = {
    # 중성(모음) 혼동: ㅑ↔ㅏ, ㅕ↔ㅓ, ㅛ↔ㅗ, ㅠ↔ㅜ
    "갸": "가", "겨": "거", "교": "고", "규": "구",
    "냐": "나", "녀": "너", "뇨": "노", "뉴": "누",
    "댜": "다", "뎌": "더", "됴": "도", "듀": "두",
    "랴": "라", "려": "러", "료": "로", "류": "루",
    "먀": "마", "며": "머", "묘": "모", "뮤": "무",
    "뱌": "바", "벼": "버", "뵤": "보", "뷰": "부",
    "샤": "사", "셔": "서", "쇼": "소", "슈": "수",
    "야": "아", "여": "어", "요": "오", "유": "우",
    "쟈": "자", "죠": "조", "쥬": "주",
    "햐": "하", "혀": "허", "효": "호", "휴": "허",
    # 종성(받침) 오인식: 번호판 용도문자에는 받침이 없다.
    "간": "가", "강": "가", "각": "가",
    "곤": "고", "공": "고", "곡": "고",
    "난": "나", "낭": "나",
    "단": "다", "당": "다",
    "란": "라", "랑": "라",
    "만": "마", "망": "마",
    "반": "바", "방": "바",
    "산": "사", "상": "사",
    "안": "아", "앙": "아",
    "잔": "자", "장": "자",
    "한": "하", "항": "하",
    "헌": "허", "헝": "허",
    "혼": "호", "홍": "호",
}

#: 지역명 오인식 보정표. 지역명은 17개로 닫힌 집합이라 보정 효과가 매우 크다.
#: 키는 **오인식된 문자열**, 값은 올바른 지역명이다. 올바른 지역명 자체는
#: ``REGION_NAMES`` 검사를 먼저 통과하므로 이 표에 넣지 않는다(중복 키 방지).
REGION_CONFUSION: dict[str, str] = {
    "서올": "서울", "셔울": "서울", "서율": "서울",
    "부신": "부산", "부샨": "부산", "무산": "부산",
    "데구": "대구", "대그": "대구", "다구": "대구",
    "인전": "인천", "인찬": "인천", "민천": "인천",
    "광조": "광주", "광쥬": "광주", "괌주": "광주",
    "데전": "대전", "대진": "대전",
    "을산": "울산", "올산": "울산",
    "세중": "세종", "새종": "세종",
    "경키": "경기", "곙기": "경기",
    "강윈": "강원", "강웜": "강원",
    "제조": "제주", "제쥬": "제주",
    "충묵": "충북", "충남": "충남",
    "전묵": "전북", "경묵": "경북",
}


# ---------------------------------------------------------------------------
# 3) 결과 구조
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class PlateResult:
    """후처리 결과.

    Attributes:
        text: 보정된 최종 번호판 문자열(실패 시 빈 문자열).
        raw_text: 보정 전 원본 OCR 출력.
        confidence: 0~1 신뢰도(보정으로 인한 감쇠가 반영된 값).
        valid_format: 한국 번호판 정규식을 통과했는지.
        status: ``SUCCESS`` | ``LOW_CONFIDENCE`` | ``FORMAT_INVALID`` | ``NO_PLATE``.
        corrections: 적용된 보정 내역 ``[{"pos", "from", "to", "rule"}]``.
        region: 추출된 지역명(없으면 None).
        char_confidences: 문자별 신뢰도.
    """

    text: str
    raw_text: str = ""
    confidence: float = 0.0
    valid_format: bool = False
    status: str = "NO_PLATE"
    corrections: list[dict[str, Any]] = field(default_factory=list)
    region: str | None = None
    char_confidences: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """JSON 직렬화용 dict 를 반환한다."""
        return {
            "text": self.text or None,
            "raw_text": self.raw_text,
            "confidence": round(float(self.confidence), 4),
            "valid_format": bool(self.valid_format),
            "status": self.status,
            "corrections": self.corrections,
            "region": self.region,
            "char_confidences": [round(float(c), 4) for c in self.char_confidences],
        }


# ---------------------------------------------------------------------------
# 4) 정규화 / 검증
# ---------------------------------------------------------------------------


def normalize(text: str) -> str:
    """OCR 출력을 비교 가능한 형태로 정규화한다.

    처리 내용:
        - 모든 공백/하이픈/점 제거 (``"서울 12가3456"`` → ``"서울12가3456"``)
        - 전각 문자를 반각으로 변환 (``"１２"`` → ``"12"``)
        - 라틴 대문자화 보류 — 혼동 보정표가 대소문자를 구분해 처리하므로
          여기서 대문자화하면 ``l``(소문자 L, →1)과 ``L``(→1) 구분 정보가 사라진다.

    Args:
        text: OCR 원본 출력.

    Returns:
        정규화된 문자열.

    Examples:
        >>> normalize("서울 12가-3456")
        '서울12가3456'
    """
    if not text:
        return ""
    out: list[str] = []
    for ch in str(text):
        if ch.isspace() or ch in "-–—_.·,":
            continue
        code = ord(ch)
        # 전각 ASCII(U+FF01~U+FF5E) → 반각
        if 0xFF01 <= code <= 0xFF5E:
            out.append(chr(code - 0xFEE0))
        else:
            out.append(ch)
    return "".join(out)


def is_valid_plate(text: str) -> bool:
    """한국 번호판 형식인지 검증한다.

    두 형식을 모두 허용한다:
        - 신형: ``^\\d{2,3}[가-힣]\\d{4}$``
        - 구형(지역명): ``^[가-힣]{2}\\s?\\d{2}[가-힣]\\d{4}$``

    정규식만으로 부족한 검사도 함께 한다 — **용도문자 화이트리스트**.
    정규식의 ``[가-힣]`` 는 한글 11,172자를 모두 허용하지만 번호판 용도문자는
    51자뿐이다. 화이트리스트 검사가 없으면 '12쀍3456' 같은 쓰레기가 통과한다.

    Args:
        text: 검증할 문자열(정규화 전/후 모두 허용).

    Returns:
        유효한 번호판 형식이면 True.

    Examples:
        >>> is_valid_plate("12가3456")
        True
        >>> is_valid_plate("서울12가3456")
        True
        >>> is_valid_plate("12쀍3456")
        False
        >>> is_valid_plate("1가3456")
        False
    """
    norm = normalize(text)
    if not norm:
        return False
    if not PLATE_RE_ANY.match(norm):
        return False

    region, body = split_region(norm)
    if region is not None and region not in REGION_NAMES:
        return False

    # 용도문자(본문의 유일한 한글)가 화이트리스트에 있는지 확인한다.
    hangul_chars = [ch for ch in body if is_hangul(ch)]
    if len(hangul_chars) != 1:
        return False
    return hangul_chars[0] in USE_CHAR_SET


def split_region(text: str) -> tuple[str | None, str]:
    """지역명과 본문을 분리한다.

    판정 기준: 앞 2글자가 모두 한글이고 **3번째 글자가 숫자**이면 지역명으로 본다.
    왜 3번째 글자를 보는가? ``"가나3456"`` 처럼 한글 2개로 시작하지만 지역명이
    아닌 문자열과 구분해야 한다. 번호판 구조상 지역명 뒤에는 반드시 숫자가 온다.

    Args:
        text: 정규화된 번호판 문자열.

    Returns:
        ``(지역명 또는 None, 본문)``.

    Examples:
        >>> split_region("서울12가3456")
        ('서울', '12가3456')
        >>> split_region("12가3456")
        (None, '12가3456')
    """
    norm = normalize(text)
    if len(norm) >= 3 and is_hangul(norm[0]) and is_hangul(norm[1]) and norm[2] in DIGIT_SET:
        return norm[:2], norm[2:]
    return None, norm


# ---------------------------------------------------------------------------
# 5) 자리 기반 혼동 보정
# ---------------------------------------------------------------------------


def expected_layout(length: int) -> list[str]:
    """본문 길이에 대응하는 자리 유형 배열을 반환한다.

    한국 번호판 본문 구조:
        - 7자: ``D D H D D D D``  (예: 12가3456)
        - 8자: ``D D D H D D D D`` (예: 123가4567)

    Args:
        length: 본문(지역명 제외) 길이.

    Returns:
        ``"D"``(숫자) / ``"H"``(한글) 리스트. 길이가 7/8 이 아니면 빈 리스트
        (= 제약 없음). 잘못된 제약은 멀쩡한 글자를 망치므로, 확신이 없으면
        보정을 포기하는 것이 안전하다.
    """
    if length == 7:
        return ["D", "D", "H", "D", "D", "D", "D"]
    if length == 8:
        return ["D", "D", "D", "H", "D", "D", "D", "D"]
    return []


def correct_confusions(text: str) -> tuple[str, list[dict[str, Any]]]:
    """자리 기반 혼동 보정을 적용한다.

    **자리 기반**이라는 점이 핵심이다. 같은 'O' 문자라도
        - 숫자 자리에 있으면 → '0' 으로 보정한다
        - 한글 자리에 있으면 → 숫자로 보정하면 안 된다(``0 → O`` 역보정 금지)
    맥락 없이 전역 치환하면 '12O3456' 은 고치지만 한글 자리까지 망친다.

    보정 순서:
        1. 지역명 분리 → 지역명은 닫힌 집합(17개)과 비교해 보정
        2. 본문 길이로 자리 유형 결정
        3. 숫자 자리: ``DIGIT_CONFUSION`` → ``DIGIT_CONFUSION_HANGUL`` 적용
        4. 한글 자리: ``USE_CHAR_CONFUSION`` → ``nearest_use_char`` 적용

    Args:
        text: OCR 원본 문자열.

    Returns:
        ``(보정된 문자열, 보정 내역 리스트)``. 내역의 각 항목은
        ``{"pos": 위치, "from": 원본, "to": 보정값, "rule": 규칙명}``.

    Examples:
        >>> correct_confusions("12가34S6")[0]
        '12가3456'
        >>> correct_confusions("I2가3456")[0]
        '12가3456'
    """
    norm = normalize(text)
    if not norm:
        return "", []

    corrections: list[dict[str, Any]] = []
    region, body = split_region(norm)

    # --- 지역명 보정 ---
    if region is not None and region not in REGION_NAMES:
        fixed = REGION_CONFUSION.get(region)
        if fixed is None:
            fixed = _nearest_region(region)
        if fixed is not None and fixed != region:
            corrections.append({"pos": 0, "from": region, "to": fixed, "rule": "region"})
            region = fixed

    layout = expected_layout(len(body))
    offset = 2 if region is not None else 0
    chars = list(body)

    for i, ch in enumerate(chars):
        kind = layout[i] if i < len(layout) else "?"

        if kind == "D" and ch not in DIGIT_SET:
            # 숫자 자리: 라틴/기호 → 숫자, 그 다음 한글 자모 → 숫자.
            replacement = DIGIT_CONFUSION.get(ch) or DIGIT_CONFUSION_HANGUL.get(ch)
            if replacement:
                corrections.append(
                    {"pos": offset + i, "from": ch, "to": replacement, "rule": "digit-confusion"}
                )
                chars[i] = replacement

        elif kind == "H" and ch not in USE_CHAR_SET:
            # 한글 자리: 표 우선, 그 다음 초성/중성 유사도.
            # 숫자 → 라틴 역보정은 하지 않는다(모듈 상단 주석 참조).
            replacement = USE_CHAR_CONFUSION.get(ch)
            rule = "use-char-table"
            if replacement is None and is_hangul(ch):
                replacement = nearest_use_char(ch)
                rule = "use-char-nearest"
            if replacement and replacement != ch:
                corrections.append(
                    {"pos": offset + i, "from": ch, "to": replacement, "rule": rule}
                )
                chars[i] = replacement

    fixed_body = "".join(chars)
    return (f"{region}{fixed_body}" if region else fixed_body), corrections


def _nearest_region(candidate: str) -> str | None:
    """오인식된 지역명을 가장 비슷한 실제 지역명으로 보정한다.

    지역명은 17개로 닫힌 집합이므로 **한 글자만 일치해도** 후보를 크게 좁힐 수 있다.
    두 글자 중 하나가 맞으면 그 지역명으로 보정한다(단, 후보가 유일할 때만 —
    예: '남' 으로 시작하는 지역명이 둘 이상이면 보정을 포기한다).

    Args:
        candidate: 2글자 후보 문자열.

    Returns:
        보정된 지역명 또는 None.
    """
    if len(candidate) != 2:
        return None
    matches = [
        name
        for name in REGION_NAMES
        if name[0] == candidate[0] or name[1] == candidate[1]
    ]
    return matches[0] if len(matches) == 1 else None


# ---------------------------------------------------------------------------
# 6) 상태 판정
# ---------------------------------------------------------------------------


def decide_status(
    text: str,
    confidence: float,
    *,
    low_conf_threshold: float = 0.55,
) -> str:
    """계약서 enum 에 맞는 ``status`` 를 결정한다.

    판정 순서(우선순위가 중요하다):
        1. 텍스트가 비면 ``NO_PLATE`` — 읽을 것이 없었다.
        2. 형식이 틀리면 ``FORMAT_INVALID`` — 읽었지만 번호판이 아니다.
           신뢰도보다 **형식을 먼저** 본다. 형식이 틀린 결과는 신뢰도가 높아도
           DB 의 차량과 매칭할 수 없으므로 운영상 쓸모가 없다.
        3. 신뢰도가 낮으면 ``LOW_CONFIDENCE`` — 형식은 맞지만 확신이 부족하다.
        4. 그 외 ``SUCCESS``.

    Args:
        text: 보정된 번호판 문자열.
        confidence: 0~1 신뢰도.
        low_conf_threshold: 이 값 미만이면 LOW_CONFIDENCE.

    Returns:
        ``"SUCCESS"`` | ``"LOW_CONFIDENCE"`` | ``"FORMAT_INVALID"`` | ``"NO_PLATE"``.

    Examples:
        >>> decide_status("", 0.9)
        'NO_PLATE'
        >>> decide_status("12가3456", 0.2)
        'LOW_CONFIDENCE'
        >>> decide_status("XYZ", 0.9)
        'FORMAT_INVALID'
        >>> decide_status("12가3456", 0.9)
        'SUCCESS'
    """
    if not text or not normalize(text):
        return "NO_PLATE"
    if not is_valid_plate(text):
        return "FORMAT_INVALID"
    if float(confidence) < float(low_conf_threshold):
        return "LOW_CONFIDENCE"
    return "SUCCESS"


def postprocess(
    raw_text: str,
    confidence: float = 0.0,
    *,
    char_confidences: list[float] | None = None,
    low_conf_threshold: float = 0.55,
    correction_penalty: float = 0.06,
) -> PlateResult:
    """OCR 원본 출력에 전체 후처리를 적용한다.

    보정에 **신뢰도 벌점**을 매기는 이유:
        보정은 "모델이 틀렸다고 가정하고 사람이 고치는" 행위다. 보정이 많았다는
        것은 그만큼 원본이 불확실했다는 뜻이므로 신뢰도를 깎아야 정직하다.
        벌점이 없으면 "쓰레기 입력을 억지로 고쳐 신뢰도 0.9 로 보고" 하게 된다.

    Args:
        raw_text: OCR 원본 출력.
        confidence: OCR 신뢰도(0~1).
        char_confidences: 문자별 신뢰도.
        low_conf_threshold: LOW_CONFIDENCE 판정 임계값.
        correction_penalty: 보정 1건당 차감할 신뢰도.

    Returns:
        PlateResult.

    Examples:
        >>> r = postprocess("12가34S6", 0.9)
        >>> r.text, r.status, r.valid_format
        ('12가3456', 'SUCCESS', True)
    """
    raw = str(raw_text or "")
    fixed, corrections = correct_confusions(raw)

    conf = max(0.0, min(1.0, float(confidence)))
    if corrections:
        conf = max(0.0, conf - len(corrections) * float(correction_penalty))

    region, _body = split_region(fixed)
    status = decide_status(fixed, conf, low_conf_threshold=low_conf_threshold)

    return PlateResult(
        text=fixed if status != "NO_PLATE" else "",
        raw_text=raw,
        confidence=round(conf, 4),
        valid_format=is_valid_plate(fixed),
        status=status,
        corrections=corrections,
        region=region,
        char_confidences=list(char_confidences or []),
    )


# ---------------------------------------------------------------------------
# 7) 다중 프레임 융합
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class FrameCandidate:
    """다중 프레임 융합에 들어가는 프레임별 결과 한 건.

    Attributes:
        text: 해당 프레임의 번호판 문자열.
        confidence: 해당 프레임의 신뢰도.
        quality: 해당 프레임의 영상 품질 점수(0~1). 가중치에 반영한다.
    """

    text: str
    confidence: float = 0.0
    quality: float = 1.0


def fuse_frames(
    candidates: list[FrameCandidate] | list[dict[str, Any]] | list[tuple[str, float]],
    *,
    low_conf_threshold: float = 0.55,
    strategy: str = "majority-vote",
) -> PlateResult:
    """여러 프레임의 OCR 결과를 하나로 융합한다.

    왜 다중 프레임이 효과적인가?
        한 프레임의 오인식은 **무작위**인 경향이 있다(흔들림/반사의 순간적 영향).
        반면 정답은 모든 프레임에서 **일관되게** 나타난다. 따라서 프레임 간
        다수결은 단일 프레임보다 체계적으로 정확하다. 실무 LPR 이 영상
        스트림을 쓰는 핵심 이유다.

    융합 전략(``strategy``):
        - ``"majority-vote"``(기본): 문자열 단위 가중 투표.
          가중치 = 신뢰도 x 품질점수. 형식이 유효한 후보에 1.35배 가산점을 준다
          (형식 검증을 통과한 결과가 통계적으로 더 믿을 만하다).
        - ``"char-vote"``: 자리별 문자 투표. 프레임마다 다른 글자가 틀렸을 때
          **어떤 프레임에도 없던 정답**을 복원할 수 있다. 길이가 같은 후보가
          2개 이상일 때만 의미가 있다.
        - ``"best-confidence"``: 최고 신뢰도 1건 선택(융합 없음, 기준선).

    Args:
        candidates: 프레임별 결과. ``FrameCandidate``, dict, ``(text, conf)`` 튜플
            형태를 모두 받는다(호출자 편의).
        low_conf_threshold: 최종 status 판정 임계값.
        strategy: 융합 전략.

    Returns:
        PlateResult. 입력이 비면 ``NO_PLATE`` 결과.

    Examples:
        >>> cands = [FrameCandidate("12가3456", 0.9), FrameCandidate("12가3456", 0.8),
        ...          FrameCandidate("12가3466", 0.7)]
        >>> fuse_frames(cands).text
        '12가3456'
    """
    frames = _coerce_candidates(candidates)
    frames = [f for f in frames if normalize(f.text)]
    if not frames:
        return PlateResult(text="", raw_text="", confidence=0.0, status="NO_PLATE")

    mode = str(strategy).lower()
    if mode == "best-confidence":
        best = max(frames, key=lambda f: f.confidence)
        return postprocess(best.text, best.confidence, low_conf_threshold=low_conf_threshold)

    if mode == "char-vote":
        voted = _char_vote(frames)
        if voted is not None:
            text, conf = voted
            return postprocess(text, conf, low_conf_threshold=low_conf_threshold)
        # 자리 투표가 불가능하면(길이가 모두 다름) 다수결로 폴백한다.

    # --- 가중 다수결 ---
    weights: Counter[str] = Counter()
    conf_sum: dict[str, float] = {}
    conf_count: dict[str, int] = {}

    for frame in frames:
        # 각 프레임 결과도 먼저 후처리해 비교 기준을 통일한다.
        # 보정 전 문자열끼리 투표하면 '12가34S6' 과 '12가3456' 이 다른 표로 갈라진다.
        result = postprocess(frame.text, frame.confidence, low_conf_threshold=low_conf_threshold)
        key = result.text or normalize(frame.text)
        weight = max(0.01, float(frame.confidence)) * max(0.05, float(frame.quality))
        if result.valid_format:
            weight *= 1.35
        weights[key] += weight
        conf_sum[key] = conf_sum.get(key, 0.0) + float(result.confidence)
        conf_count[key] = conf_count.get(key, 0) + 1

    winner, _ = weights.most_common(1)[0]
    votes = conf_count.get(winner, 1)
    mean_conf = conf_sum.get(winner, 0.0) / max(1, votes)

    # 합의 보너스: **동의한 프레임 수**와 **동의 비율**을 함께 본다.
    #
    # 왜 비율만으로는 안 되는가? 1프레임 중 1표도 비율 100%, 3프레임 중 3표도
    # 비율 100% 라 보너스가 같아진다. 하지만 3개가 독립적으로 같은 답을 낸 것은
    # 1개의 답보다 훨씬 강한 증거다. 통계적으로 독립 관측이 늘면 불확실성은
    # √n 에 비례해 줄어든다. 그 직관을 ``log`` 로 완만하게 반영한다
    # (1표 → 보너스 0, 3표 → 약 0.11, 9표 → 약 0.22).
    agreement_ratio = votes / len(frames)
    independence_bonus = 0.10 * math.log(max(1, votes))
    boosted = min(1.0, mean_conf * (1.0 + 0.25 * agreement_ratio + independence_bonus))

    result = postprocess(winner, boosted, low_conf_threshold=low_conf_threshold)
    result.raw_text = winner
    return result


def _coerce_candidates(
    candidates: list[FrameCandidate] | list[dict[str, Any]] | list[tuple[str, float]],
) -> list[FrameCandidate]:
    """다양한 입력 형태를 ``FrameCandidate`` 리스트로 통일한다.

    Args:
        candidates: 혼합 형태의 후보 리스트.

    Returns:
        FrameCandidate 리스트(변환 불가 항목은 건너뛴다).
    """
    out: list[FrameCandidate] = []
    for item in candidates or []:
        if isinstance(item, FrameCandidate):
            out.append(item)
        elif isinstance(item, dict):
            out.append(
                FrameCandidate(
                    text=str(item.get("text") or item.get("plate_number") or ""),
                    confidence=float(item.get("confidence") or 0.0),
                    quality=float(item.get("quality") or item.get("quality_score") or 1.0),
                )
            )
        elif isinstance(item, (tuple, list)) and len(item) >= 1:
            text = str(item[0] or "")
            conf = float(item[1]) if len(item) > 1 else 0.0
            quality = float(item[2]) if len(item) > 2 else 1.0
            out.append(FrameCandidate(text=text, confidence=conf, quality=quality))
    return out


def _char_vote(frames: list[FrameCandidate]) -> tuple[str, float] | None:
    """자리별 문자 투표로 문자열을 합성한다.

    조건: 가장 흔한 길이를 가진 후보가 **2건 이상**이어야 한다. 1건이면
    투표가 아니라 복사이므로 의미가 없다.

    각 자리에서 "가중 최다 득표 문자" 를 고르고, 자리별 득표율의 평균을
    신뢰도로 삼는다. 모든 자리가 만장일치면 신뢰도가 입력 평균보다 높아진다.

    Args:
        frames: 프레임 후보 리스트.

    Returns:
        ``(합성 문자열, 신뢰도)`` 또는 None(투표 불가).
    """
    normalized = [(normalize(f.text), f) for f in frames]
    normalized = [(t, f) for t, f in normalized if t]
    if len(normalized) < 2:
        return None

    length_counter = Counter(len(t) for t, _ in normalized)
    target_len, count = length_counter.most_common(1)[0]
    if count < 2:
        return None

    pool = [(t, f) for t, f in normalized if len(t) == target_len]
    chars: list[str] = []
    ratios: list[float] = []

    for pos in range(target_len):
        votes: Counter[str] = Counter()
        total = 0.0
        for text, frame in pool:
            weight = max(0.01, float(frame.confidence)) * max(0.05, float(frame.quality))
            votes[text[pos]] += weight
            total += weight
        best_char, best_weight = votes.most_common(1)[0]
        chars.append(best_char)
        ratios.append(best_weight / total if total > 0 else 0.0)

    mean_conf = sum(max(0.0, f.confidence) for _, f in pool) / len(pool)
    mean_ratio = sum(ratios) / len(ratios) if ratios else 0.0
    # 자리별 합의율을 신뢰도에 곱한다. 자리마다 의견이 갈렸으면(ratio 낮음)
    # 합성 결과도 믿을 수 없다.
    return "".join(chars), min(1.0, mean_conf * (0.70 + 0.45 * mean_ratio))


def summarize_corrections(results: list[PlateResult]) -> dict[str, int]:
    """여러 결과의 보정 규칙별 적용 횟수를 집계한다(운영 모니터링용).

    어떤 규칙이 자주 발동하는지 보면 모델의 약점을 알 수 있다. 예를 들어
    ``digit-confusion`` 이 급증하면 OCR 이 숫자를 라틴 문자로 읽고 있다는 뜻이고,
    이는 학습 데이터에 라틴 문자가 섞였다는 신호다.

    Args:
        results: PlateResult 리스트.

    Returns:
        ``{규칙명: 횟수}``.
    """
    counter: Counter[str] = Counter()
    for result in results or []:
        for correction in result.corrections:
            counter[str(correction.get("rule", "unknown"))] += 1
    return dict(counter)


__all__ = [
    "DIGIT_CONFUSION",
    "DIGIT_CONFUSION_HANGUL",
    "PLATE_RE_ANY",
    "PLATE_RE_DIPLOMAT",
    "PLATE_RE_PLAIN",
    "PLATE_RE_REGION",
    "REGION_CONFUSION",
    "USE_CHAR_CONFUSION",
    "FrameCandidate",
    "PlateResult",
    "correct_confusions",
    "decide_status",
    "expected_layout",
    "fuse_frames",
    "is_valid_plate",
    "normalize",
    "postprocess",
    "split_region",
    "summarize_corrections",
]
