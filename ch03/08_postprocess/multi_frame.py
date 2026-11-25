# -*- coding: utf-8 -*-
"""다중 프레임 OCR 결과 결합(문자 단위 다수결 + 신뢰도 가중).

한 대의 차량이 게이트를 통과하는 동안 보통 5~15 프레임을 얻는다.
프레임마다 결과가 조금씩 다르므로, 프레임 하나를 고르는 대신
자리별로 투표를 모아 가장 그럴듯한 한 개의 문자열을 만든다.

AI 서비스의 POST /api/v1/lpr/recognize-frames 응답에서
"fusion": "majority-vote" 로 표기되는 단계가 바로 이 모듈이다.
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from confusion_fix import correct                      # noqa: E402
from plate_rules import normalize, validate            # noqa: E402


@dataclass
class FrameResult:
    """프레임 한 장의 OCR 결과."""

    text: str                                   # 인식된 문자열
    confidence: float = 0.0                     # 문자열 전체 신뢰도
    char_confidences: list[float] = field(default_factory=list)
    quality: float = 1.0                        # 이미지 품질 점수(0~1)

    def weight(self) -> float:
        """프레임 가중치. 품질이 낮은 프레임의 영향력을 줄인다."""
        return max(self.confidence, 0.01) * max(self.quality, 0.05)

    def char_conf(self, i: int) -> float:
        """i 번째 자리의 문자 신뢰도. 없으면 문자열 신뢰도로 대신한다."""
        if i < len(self.char_confidences):
            return self.char_confidences[i]
        return self.confidence


@dataclass
class FusionResult:
    """결합 결과."""

    plate_number: str                  # 최종 확정 문자열
    confidence: float                  # 자리별 득표율의 평균
    frames: int                        # 사용한 프레임 수
    length: int                        # 채택한 문자열 길이
    valid_format: bool                 # 형식 검증 통과 여부
    per_char: list[tuple[str, float]] = field(default_factory=list)
    agreement: float = 0.0             # 원본 프레임 중 최종 결과와 같은 비율
    note: str = ""

    def report(self) -> str:
        lines = [
            f"최종 번호판 : {self.plate_number}",
            f"신뢰도      : {self.confidence:.4f}",
            f"프레임 수   : {self.frames} (채택 길이 {self.length}자)",
            f"형식 유효   : {self.valid_format}",
            f"프레임 일치 : {self.agreement:.1%}",
        ]
        if self.note:
            lines.append(f"비고        : {self.note}")
        lines.append("자리별 득표 : " + "  ".join(
            f"{c}({p:.2f})" for c, p in self.per_char))
        return "\n".join(lines)


def pick_length(frames: list[FrameResult]) -> int:
    """가중 투표로 문자열 길이를 먼저 정한다.

    길이가 다른 프레임을 그대로 섞으면 자리가 밀려 엉뚱한 결과가 나온다.
    따라서 길이를 먼저 확정하고, 그 길이의 프레임만 자리 투표에 넣는다.
    """
    votes: dict[int, float] = defaultdict(float)
    for f in frames:
        n = len(normalize(f.text))
        if n == 0:
            continue
        votes[n] += f.weight()
    if not votes:
        return 0
    # 득표가 같으면 7자(가장 흔한 신형 2자리)를 우선한다.
    return max(votes.items(), key=lambda kv: (kv[1], kv[0] == 7))[0]


def fuse(
    frames: list[FrameResult],
    min_frames: int = 1,
    apply_confusion_fix: bool = True,
) -> FusionResult:
    """프레임 결과 목록을 하나의 번호판 문자열로 결합한다."""
    usable = [f for f in frames if normalize(f.text)]
    if len(usable) < min_frames or not usable:
        return FusionResult("", 0.0, len(frames), 0, False,
                            note="사용 가능한 프레임이 없음")

    target_len = pick_length(usable)
    pool = [f for f in usable if len(normalize(f.text)) == target_len]
    if not pool:
        pool = usable

    # 자리별 가중 투표
    per_char: list[tuple[str, float]] = []
    chars: list[str] = []
    for i in range(target_len):
        votes: dict[str, float] = defaultdict(float)
        for f in pool:
            t = normalize(f.text)
            if i >= len(t):
                continue
            # 가중치 = 프레임 가중치 x 해당 자리의 문자 신뢰도
            votes[t[i]] += f.weight() * max(f.char_conf(i), 0.01)
        if not votes:
            chars.append("?")
            per_char.append(("?", 0.0))
            continue
        total = sum(votes.values())
        best, w = max(votes.items(), key=lambda kv: kv[1])
        chars.append(best)
        per_char.append((best, w / total if total else 0.0))

    merged = "".join(chars)
    conf = sum(p for _, p in per_char) / max(len(per_char), 1)

    note = ""
    chk = validate(merged)
    if apply_confusion_fix and not chk.ok:
        # 다수결 결과가 형식에 맞지 않으면 혼동 보정으로 한 번 더 구제한다.
        fixed, ok, _ = correct(merged, [p for _, p in per_char])
        if ok:
            note = f"다수결 결과 '{merged}' 를 혼동 보정으로 교정"
            merged, chk = fixed, validate(fixed)

    same = sum(1 for f in pool if normalize(f.text) == merged)
    return FusionResult(
        plate_number=merged,
        confidence=conf,
        frames=len(frames),
        length=target_len,
        valid_format=chk.ok,
        per_char=per_char,
        agreement=same / max(len(pool), 1),
        note=note,
    )


def _demo() -> None:
    # 실제 게이트 영상에서 흔히 나오는 패턴을 모사한 7프레임.
    frames = [
        FrameResult("12가3456", 0.97, [.99, .98, .97, .99, .97, .98, .96], 0.90),
        FrameResult("12가3456", 0.95, [.98, .97, .96, .98, .95, .97, .94], 0.85),
        FrameResult("12가3458", 0.71, [.95, .94, .93, .96, .92, .93, .48], 0.52),
        FrameResult("12가3456", 0.93, [.97, .96, .95, .97, .94, .96, .93], 0.81),
        FrameResult("12가345", 0.40, [.70, .68, .61, .72, .60, .58, .00], 0.31),
        FrameResult("17가3456", 0.66, [.93, .45, .92, .95, .91, .93, .90], 0.48),
        FrameResult("12가3456", 0.96, [.98, .98, .97, .98, .96, .97, .95], 0.88),
    ]
    print("=== 사례 1: 다수결로 안정화되는 경우 ===")
    print(fuse(frames).report())

    # 어떤 프레임도 단독으로는 정답이 아니지만 자리별로는 정답이 섞여 있는 경우.
    frames2 = [
        FrameResult("12가3458", 0.62, [.96, .95, .94, .97, .93, .95, .41], 0.55),
        FrameResult("17가3456", 0.64, [.95, .44, .93, .96, .92, .94, .90], 0.52),
        FrameResult("12나3456", 0.61, [.94, .93, .43, .95, .91, .93, .89], 0.50),
        FrameResult("12가3456", 0.69, [.95, .94, .92, .96, .92, .94, .90], 0.58),
    ]
    print("\n=== 사례 2: 자리별로 정답을 모아 복원하는 경우 ===")
    print(fuse(frames2).report())

    # 전부 실패한 경우.
    print("\n=== 사례 3: 사용 가능한 프레임이 없는 경우 ===")
    print(fuse([FrameResult("", 0.0), FrameResult("", 0.0)]).report())


if __name__ == "__main__":
    _demo()
