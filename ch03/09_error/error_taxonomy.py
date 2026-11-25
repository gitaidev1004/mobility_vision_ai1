# -*- coding: utf-8 -*-
"""OCR 오류 유형 자동 분류기.

오인식 샘플 하나에 대해 '왜 틀렸는가'를 규칙으로 추정한다.
문자열 비교만으로 알 수 있는 단서(혼동쌍, 길이 변화)와
이미지 메타데이터(해상도, 밝기, 기울기, 가림률)를 함께 쓴다.

분류 결과는 ch03/09_error/hard_sample_mining.py 의 재학습 후보 추출과
재판독 정책 결정에 그대로 쓰인다.
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from dataclasses import dataclass, field

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "10_eval"))

from metrics_ocr import edit_ops, levenshtein, normalize  # noqa: E402

# ---------------------------------------------------------------------------
# 1) 오류 유형 정의
# ---------------------------------------------------------------------------
ERROR_TYPES: dict[str, str] = {
    "CONF_DIGIT_ALPHA": "숫자-영문 혼동(0/O, 1/I, 5/S, 8/B 등)",
    "CONF_DIGIT_DIGIT": "숫자 간 혼동(3/8, 6/8, 1/7 등)",
    "CONF_HANGUL": "한글 용도 기호 혼동(거/고, 수/우 등)",
    "LOW_RESOLUTION": "저해상도(번호판 픽셀 폭 부족)",
    "GLARE": "반사광·과다 노출",
    "NIGHT_NOISE": "야간 저조도 노이즈",
    "SKEW": "기울기·원근 왜곡",
    "OCCLUSION": "부분 가림(이물질·차체·눈비)",
    "LENGTH_MISMATCH": "문자 수 불일치(삽입·삭제)",
    "UNKNOWN": "규칙으로 설명되지 않는 오류",
}

# 혼동쌍 사전 — 양방향으로 조회할 수 있게 집합의 집합으로 둔다.
DIGIT_ALPHA = [("0", "O"), ("0", "D"), ("0", "Q"), ("1", "I"), ("1", "L"),
               ("2", "Z"), ("5", "S"), ("6", "G"), ("8", "B"), ("4", "A"),
               ("7", "T"), ("9", "P")]
DIGIT_DIGIT = [("0", "8"), ("0", "6"), ("3", "8"), ("3", "9"), ("5", "6"),
               ("5", "8"), ("6", "8"), ("1", "7"), ("1", "4"), ("4", "9"),
               ("2", "7"), ("8", "9")]
HANGUL_CONF = [("거", "고"), ("너", "노"), ("더", "도"), ("러", "로"),
               ("머", "모"), ("버", "보"), ("서", "소"), ("어", "오"),
               ("저", "조"), ("구", "우"), ("수", "우"), ("누", "두"),
               ("루", "무"), ("가", "카"), ("나", "다"), ("다", "라"),
               ("바", "사"), ("자", "차"), ("허", "호"), ("하", "허"),
               ("로", "도"), ("노", "도"), ("모", "보"), ("조", "소"),
               ("오", "소"), ("구", "고"), ("누", "노"), ("무", "모")]


def _pairset(pairs: list[tuple[str, str]]) -> set[frozenset[str]]:
    return {frozenset(p) for p in pairs}


PS_DIGIT_ALPHA = _pairset(DIGIT_ALPHA)
PS_DIGIT_DIGIT = _pairset(DIGIT_DIGIT)
PS_HANGUL = _pairset(HANGUL_CONF)

# 판정 임계값 — 현장 카메라 사양에 맞춰 조정한다.
TH_PLATE_PX = 120        # 번호판 가로 픽셀 폭 하한
TH_BRIGHT_HIGH = 200     # 과다 노출 판정 밝기
TH_BRIGHT_LOW = 60       # 저조도 판정 밝기
TH_BLUR = 80.0           # 라플라시안 분산 하한(낮으면 흐림)
TH_SKEW_DEG = 8.0        # 기울기 허용 각도
TH_OCCLUSION = 0.12      # 가림률 허용치
TH_NOISE = 14.0          # 노이즈 표준편차 상한


@dataclass
class Sample:
    """오인식 분석 대상 샘플 하나."""

    sample_id: str
    gt: str                          # 정답 번호판
    pred: str                        # 모델 예측
    confidence: float = 0.0
    plate_px_width: int = 0          # 번호판 가로 픽셀 폭
    brightness: float = 128.0        # 평균 밝기(0~255)
    blur: float = 200.0              # 라플라시안 분산(클수록 선명)
    skew_deg: float = 0.0            # 추정 기울기(도)
    occlusion: float = 0.0           # 가림 비율(0~1)
    noise_std: float = 5.0           # 배경 노이즈 표준편차
    is_night: bool = False           # 촬영 시각이 야간인지
    condition: str = "주간"          # 환경 라벨


@dataclass
class Diagnosis:
    """한 샘플의 오류 진단 결과."""

    sample_id: str
    gt: str
    pred: str
    correct: bool
    distance: int
    types: list[str] = field(default_factory=list)
    detail: list[str] = field(default_factory=list)

    @property
    def primary(self) -> str:
        """대표 유형. 우선순위가 높은 유형 하나를 고른다."""
        order = ["OCCLUSION", "LOW_RESOLUTION", "GLARE", "NIGHT_NOISE",
                 "SKEW", "LENGTH_MISMATCH", "CONF_DIGIT_ALPHA",
                 "CONF_HANGUL", "CONF_DIGIT_DIGIT", "UNKNOWN"]
        for t in order:
            if t in self.types:
                return t
        return "UNKNOWN"

    def line(self) -> str:
        tag = "OK" if self.correct else self.primary
        return (f"{self.sample_id:<10s} {self.gt:<11s} -> {self.pred or '(없음)':<11s} "
                f"d={self.distance}  [{tag}]  {'; '.join(self.detail)}")


def _char_pairs(gt: str, pred: str) -> list[tuple[str, str]]:
    """같은 길이일 때 자리별로 어긋난 (정답문자, 예측문자) 쌍을 모은다."""
    if len(gt) != len(pred):
        return []
    return [(a, b) for a, b in zip(gt, pred) if a != b]


def classify(s: Sample) -> Diagnosis:
    """샘플 하나의 오류 유형을 분류한다."""
    gt, pred = normalize(s.gt), normalize(s.pred)
    dist = levenshtein(gt, pred)
    dg = Diagnosis(s.sample_id, gt, pred, gt == pred, dist)
    if dg.correct:
        dg.detail.append("정답")
        return dg

    # (1) 문자열 단서 — 혼동쌍
    for a, b in _char_pairs(gt, pred):
        key = frozenset((a, b))
        if key in PS_DIGIT_ALPHA:
            dg.types.append("CONF_DIGIT_ALPHA")
            dg.detail.append(f"숫자-영문 혼동 {a}->{b}")
        elif key in PS_HANGUL:
            dg.types.append("CONF_HANGUL")
            dg.detail.append(f"한글 혼동 {a}->{b}")
        elif key in PS_DIGIT_DIGIT:
            dg.types.append("CONF_DIGIT_DIGIT")
            dg.detail.append(f"숫자 간 혼동 {a}->{b}")
        elif a.isdigit() and b.isascii() and b.isalpha():
            # 표에 없는 조합이라도 숫자 자리에 영문이 나오면 같은 계열로 본다.
            dg.types.append("CONF_DIGIT_ALPHA")
            dg.detail.append(f"숫자 자리 영문 출력 {a}->{b}")
        elif a.isdigit() and b.isdigit():
            dg.types.append("CONF_DIGIT_DIGIT")
            dg.detail.append(f"숫자 치환 {a}->{b}")
        elif not a.isascii() and not b.isascii():
            # 표에 없는 한글 조합도 한글 혼동으로 묶는다.
            dg.types.append("CONF_HANGUL")
            dg.detail.append(f"한글 치환 {a}->{b}")
        else:
            dg.types.append("UNKNOWN")
            dg.detail.append(f"설명되지 않는 치환 {a}->{b}")

    # (2) 길이 불일치
    ops = edit_ops(gt, pred)
    if ops["D"] or ops["I"]:
        dg.types.append("LENGTH_MISMATCH")
        dg.detail.append(f"삭제 {ops['D']}자 / 삽입 {ops['I']}자")

    # (3) 이미지 메타데이터 단서
    if s.plate_px_width and s.plate_px_width < TH_PLATE_PX:
        dg.types.append("LOW_RESOLUTION")
        dg.detail.append(f"번호판 폭 {s.plate_px_width}px < {TH_PLATE_PX}px")
    if s.blur < TH_BLUR:
        dg.types.append("LOW_RESOLUTION")
        dg.detail.append(f"흐림 지표 {s.blur:.0f} < {TH_BLUR:.0f}")
    if s.brightness > TH_BRIGHT_HIGH:
        dg.types.append("GLARE")
        dg.detail.append(f"과다 노출 밝기 {s.brightness:.0f}")
    if s.is_night or s.brightness < TH_BRIGHT_LOW:
        if s.noise_std > TH_NOISE or s.is_night:
            dg.types.append("NIGHT_NOISE")
            dg.detail.append(
                f"야간/저조도 밝기 {s.brightness:.0f}, 노이즈 {s.noise_std:.1f}")
    if abs(s.skew_deg) > TH_SKEW_DEG:
        dg.types.append("SKEW")
        dg.detail.append(f"기울기 {s.skew_deg:.1f}도 > {TH_SKEW_DEG}도")
    if s.occlusion > TH_OCCLUSION:
        dg.types.append("OCCLUSION")
        dg.detail.append(f"가림률 {s.occlusion:.0%}")

    if not dg.types:
        dg.types.append("UNKNOWN")
        dg.detail.append("규칙으로 설명되지 않음 — 수동 검토 필요")

    # 중복 제거(순서 유지)
    dg.types = list(dict.fromkeys(dg.types))
    return dg


def summarize(diags: list[Diagnosis]) -> Counter[str]:
    """대표 유형별 건수를 센다."""
    return Counter(d.primary for d in diags if not d.correct)


def load_samples(path: str) -> list[Sample]:
    """JSON Lines 파일에서 샘플을 읽는다. 파일이 없으면 빈 목록."""
    if not os.path.exists(path):
        print(f"[안내] 샘플 파일이 없습니다: {path}")
        print("       내장 예시 데이터로 동작을 확인한다.")
        return []
    out: list[Sample] = []
    with open(path, encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                out.append(Sample(**json.loads(line)))
    return out


def builtin_samples() -> list[Sample]:
    """교재 실습용 내장 예시. 현장에서 자주 보이는 9가지 상황을 담았다."""
    return [
        Sample("S001", "12가3456", "12가3456", 0.97, 184, 118, 230, 1.2, 0.0),
        Sample("S002", "12가3456", "12가345B", 0.61, 176, 121, 210, 0.9, 0.0),
        Sample("S003", "12가3456", "12가3458", 0.68, 168, 115, 198, 1.4, 0.0),
        Sample("S004", "87로2345", "87도2345", 0.59, 171, 120, 205, 2.1, 0.0),
        Sample("S005", "12가3456", "12가3450", 0.44, 86, 112, 61, 1.1, 0.0),
        Sample("S006", "87로2345", "87로234", 0.39, 152, 218, 150, 1.8, 0.0),
        Sample("S007", "서울34나5678", "서울34나5G78", 0.41, 160, 42, 120, 2.4,
               0.0, 22.5, True, "야간"),
        Sample("S008", "12가3456", "17가3456", 0.52, 165, 124, 170, 14.3, 0.0,
               6.0, False, "기울기"),
        Sample("S009", "87로2345", "87로2845", 0.37, 158, 117, 165, 2.0, 0.21,
               5.0, False, "가림"),
        Sample("S010", "123가4567", "123가4567", 0.95, 196, 126, 240, 0.7, 0.0),
    ]


def _demo() -> None:
    samples = load_samples("samples.jsonl") or builtin_samples()
    diags = [classify(s) for s in samples]

    print("=== 샘플별 오류 진단 ===")
    for d in diags:
        print(d.line())

    print("\n=== 대표 유형별 집계 ===")
    counts = summarize(diags)
    total_err = sum(counts.values())
    print(f"{'유형':<20s} {'건수':>4s}  비율   설명")
    for t, n in counts.most_common():
        print(f"{t:<20s} {n:>4d}  {n / total_err:5.1%}  {ERROR_TYPES[t]}")
    print(f"\n전체 {len(diags)}건 중 오류 {total_err}건 "
          f"(정확 {len(diags) - total_err}건)")


if __name__ == "__main__":
    _demo()
