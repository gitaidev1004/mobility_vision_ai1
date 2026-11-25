# -*- coding: utf-8 -*-
"""자리(position) 기반 숫자 ↔ 영문/한글 혼동 보정 및 후보 생성.

번호판은 각 자리에 올 수 있는 문자 종류가 정해져 있다.
  - 신형 2자리 번호판 12가3456 → [숫자][숫자][한글][숫자][숫자][숫자][숫자]
이 제약을 이용하면 'O' 가 숫자 자리에 나타난 경우 '0' 으로 되돌릴 수 있다.
반대로 숫자 자리가 아닌 용도 기호 자리에서는 숫자를 한글로 되돌려야 한다.

ch03/08_postprocess/plate_rules.py 의 형식 검증 규칙을 그대로 가져와
보정 후보가 실제로 유효한 번호판인지 확인한다.
"""
from __future__ import annotations

import itertools
import os
import sys
from dataclasses import dataclass

# 같은 디렉터리의 plate_rules.py 를 어디서 실행해도 찾을 수 있게 경로를 추가한다.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from plate_rules import HANGUL_ALL, REGION_NAMES, normalize, validate  # noqa: E402

# ---------------------------------------------------------------------------
# 1) 혼동 문자 매핑표
# ---------------------------------------------------------------------------
# 영문/기호 -> 숫자. 숫자 자리에서만 적용한다.
TO_DIGIT: dict[str, str] = {
    "O": "0", "Q": "0", "D": "0",
    "I": "1", "L": "1", "|": "1", "ㅣ": "1",
    "Z": "2",
    "J": "3",
    "A": "4",
    "S": "5",
    "G": "6", "C": "6",
    "T": "7", "Y": "7",
    "B": "8",
    "P": "9", "R": "9",
}

# 숫자 -> 영문. 숫자가 아닌 자리에 숫자가 나타났을 때의 역변환 후보다.
TO_ALPHA: dict[str, tuple[str, ...]] = {
    "0": ("O", "Q", "D"),
    "1": ("I", "L"),
    "2": ("Z",),
    "5": ("S",),
    "6": ("G",),
    "8": ("B",),
}

# 숫자 자리에서 서로 자주 바뀌는 숫자쌍(획이 비슷한 쌍만 등록).
DIGIT_PAIRS: dict[str, tuple[str, ...]] = {
    "0": ("8", "6"),
    "1": ("7", "4"),
    "3": ("8", "9"),
    "4": ("1", "9"),
    "5": ("6", "8"),
    "6": ("5", "8", "0"),
    "7": ("1", "2"),
    "8": ("0", "3", "6", "9"),
    "9": ("4", "3", "8"),
    "2": ("7", "3"),
}

# 용도 기호 자리의 한글 혼동쌍. 받침과 모음이 비슷한 음절끼리 묶는다.
HANGUL_PAIRS: dict[str, tuple[str, ...]] = {
    "가": ("기", "카", "저"),
    "나": ("너", "마", "다"),
    "다": ("라", "나", "차"),
    "라": ("다", "마"),
    "마": ("나", "머", "다"),
    "거": ("고", "너", "가"),
    "너": ("나", "노", "머"),
    "더": ("도", "다"),
    "러": ("로", "라"),
    "머": ("모", "마", "너"),
    "고": ("거", "노", "구"),
    "노": ("너", "도", "누"),
    "도": ("더", "로", "두"),
    "로": ("러", "도", "루"),
    "모": ("머", "노", "무"),
    "구": ("고", "누", "우"),
    "누": ("노", "두", "구"),
    "두": ("도", "루", "누"),
    "루": ("로", "두", "무"),
    "무": ("모", "누", "루"),
    "버": ("보", "바", "부"),
    "보": ("버", "부", "모"),
    "부": ("보", "무", "버"),
    "서": ("소", "수", "저"),
    "소": ("서", "수", "오"),
    "수": ("소", "서", "우"),
    "어": ("오", "우", "저"),
    "오": ("어", "우", "소"),
    "우": ("오", "어", "수"),
    "저": ("조", "주", "서"),
    "조": ("저", "주", "소"),
    "주": ("조", "저", "수"),
    "아": ("어", "바", "자"),
    "바": ("버", "아", "사"),
    "사": ("서", "바", "자"),
    "자": ("저", "사", "아"),
    "허": ("호", "하", "어"),
    "하": ("허", "호"),
    "호": ("허", "하", "고"),
    "배": ("바", "매"),
}

HANGUL_SORTED = tuple(sorted(HANGUL_ALL))


@dataclass
class Candidate:
    """보정 후보 하나."""

    text: str          # 보정된 문자열
    edits: int         # 원문에서 몇 자를 바꿨는지
    score: float       # 평균 신뢰도 기반 점수(높을수록 좋다)
    valid: bool        # 형식 검증 통과 여부
    plate_type: str    # 통과한 경우의 형식 코드

    def summary(self) -> str:
        flag = "valid" if self.valid else "invalid"
        return (f"{self.text}  edits={self.edits}  "
                f"score={self.score:.4f}  {flag}({self.plate_type})")


def position_mask(text: str) -> list[str]:
    """문자열 길이로부터 각 자리에 와야 할 문자 종류를 추정한다.

    반환값의 각 원소는 'D'(숫자), 'H'(용도 기호 한글), 'R'(지역명 음절)이다.
    길이만으로 유일하게 결정되지 않는 경우에는 가장 흔한 형식을 가정한다.
    """
    n = len(text)
    if n == 7:                     # 12가3456
        return list("DDHDDDD")
    if n == 8:
        # 123가4567 (신형 3자리) 또는 지역명 없는 특수 형태.
        if text[:2] in REGION_NAMES:
            return list("RRDDHDD")  # 길이가 맞지 않으므로 거의 쓰이지 않는다
        return list("DDDHDDDD")
    if n == 9:                     # 서울34나5678
        return list("RRDDHDDDD")
    # 길이가 규격 밖이면 숫자 위주로 가정한다.
    return ["D"] * n


def fix_by_position(text: str) -> tuple[str, int]:
    """자리 제약에 따라 명백한 혼동만 1차 보정한다.

    - 숫자 자리의 영문/기호 → 숫자 (O→0, I→1, B→8, S→5 …)
    - 한글 자리의 숫자/영문  → 보정 불가이므로 그대로 두고 후보 생성에 넘긴다.
    반환값은 (보정된 문자열, 바꾼 글자 수)이다.
    """
    norm = normalize(text)
    mask = position_mask(norm)
    out: list[str] = []
    edits = 0
    for ch, kind in zip(norm, mask):
        if kind == "D" and ch in TO_DIGIT:
            out.append(TO_DIGIT[ch])
            edits += 1
        else:
            out.append(ch)
    return "".join(out), edits


def _alternatives(ch: str, kind: str) -> list[str]:
    """한 자리에 대한 대체 문자 후보 목록(자기 자신 포함)."""
    alts = [ch]
    if kind == "D":
        if ch in TO_DIGIT:
            alts.append(TO_DIGIT[ch])
        alts.extend(DIGIT_PAIRS.get(ch, ()))
        if ch in HANGUL_ALL:        # 한글이 숫자 자리에 들어온 경우
            alts.extend("0123456789")
    elif kind == "H":
        if ch in HANGUL_PAIRS:
            alts.extend(HANGUL_PAIRS[ch])
        if ch.isdigit() or ch in TO_ALPHA or not ch in HANGUL_ALL:
            # 한글 자리에 숫자/영문이 들어오면 전체 용도 기호를 후보로 둔다.
            alts.extend(HANGUL_SORTED)
    elif kind == "R":
        alts.extend(sorted({r[0] for r in REGION_NAMES}
                           | {r[1] for r in REGION_NAMES}))
    # 중복 제거(입력 순서 유지)
    return list(dict.fromkeys(a for a in alts if a))


def generate_candidates(
    text: str,
    char_confidences: list[float] | None = None,
    conf_threshold: float = 0.90,
    max_edits: int = 2,
    top_k: int = 5,
) -> list[Candidate]:
    """혼동 보정 후보를 생성해 점수 순으로 돌려준다.

    char_confidences 가 주어지면 신뢰도가 conf_threshold 미만인 자리만
    대체 후보를 만든다. 이렇게 하면 탐색 공간이 급격히 줄어든다.
    신뢰도가 없으면 모든 자리를 후보 대상으로 본다(max_edits 로 제한).
    """
    base, _ = fix_by_position(text)
    mask = position_mask(base)
    confs = char_confidences or [0.0] * len(base)
    if len(confs) != len(base):
        confs = (list(confs) + [0.0] * len(base))[: len(base)]

    # 자리별 후보 목록 구성
    per_pos: list[list[str]] = []
    for idx, ch in enumerate(base):
        kind = mask[idx] if idx < len(mask) else "D"
        if confs[idx] >= conf_threshold:
            per_pos.append([ch])            # 확실한 자리는 고정
        else:
            per_pos.append(_alternatives(ch, kind))

    results: list[Candidate] = []
    seen: set[str] = set()
    for combo in itertools.product(*per_pos):
        cand = "".join(combo)
        if cand in seen:
            continue
        seen.add(cand)
        edits = sum(1 for a, b in zip(cand, base) if a != b)
        if edits > max_edits:
            continue
        chk = validate(cand)
        # 점수: 바꾸지 않은 자리의 신뢰도는 그대로, 바꾼 자리는 (1-신뢰도)로 본다.
        score = 0.0
        for i, (a, b) in enumerate(zip(cand, base)):
            c = confs[i] if confs[i] > 0 else 0.5
            score += c if a == b else max(1.0 - c, 0.01)
        score /= max(len(cand), 1)
        results.append(Candidate(cand, edits, score, chk.ok, chk.plate_type))
        if len(seen) > 200000:              # 안전장치
            break

    # 유효한 후보 우선 → 편집 거리 작은 것 → 점수 높은 것
    results.sort(key=lambda c: (not c.valid, c.edits, -c.score))
    return results[:top_k]


def correct(
    text: str,
    char_confidences: list[float] | None = None,
    conf_threshold: float = 0.90,
) -> tuple[str, bool, list[Candidate]]:
    """최종 보정 결과를 돌려준다.

    반환값은 (확정 문자열, 형식 유효 여부, 상위 후보 목록)이다.
    유효한 후보가 하나도 없으면 1차 보정 결과를 그대로 돌려주고
    valid=False 로 표시해 상위 로직이 재촬영/재판독을 요청하게 한다.
    """
    cands = generate_candidates(text, char_confidences, conf_threshold)
    for c in cands:
        if c.valid:
            return c.text, True, cands
    base, _ = fix_by_position(text)
    return base, False, cands


def _demo() -> None:
    cases: list[tuple[str, list[float] | None]] = [
        # (OCR 원문, 문자별 신뢰도)
        ("I2가3456", [0.52, 0.99, 0.98, 0.99, 0.97, 0.98, 0.96]),
        ("12가3４5B", [0.99, 0.98, 0.97, 0.99, 0.95, 0.98, 0.44]),
        ("I2가34S6", [0.48, 0.99, 0.98, 0.99, 0.97, 0.41, 0.96]),
        ("120가3456", None),
        ("87로2345", [0.99, 0.98, 0.99, 0.99, 0.98, 0.97, 0.99]),
        ("87ㄹ2345", [0.99, 0.98, 0.33, 0.99, 0.98, 0.97, 0.99]),
        ("서울34나5678", None),
    ]
    for raw, confs in cases:
        fixed, ok, cands = correct(raw, confs)
        print(f"입력 {raw!r}")
        print(f"  확정 -> {fixed}  (형식 유효: {ok})")
        for c in cands[:3]:
            print("   ", c.summary())
        print()


if __name__ == "__main__":
    _demo()
