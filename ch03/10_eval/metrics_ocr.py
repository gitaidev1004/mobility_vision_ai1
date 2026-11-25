# -*- coding: utf-8 -*-
"""번호판 OCR 평가 지표 구현 (순수 파이썬, 외부 의존성 없음).

구현 지표
  1) Levenshtein 편집 거리          levenshtein()
  2) Character Error Rate (CER)      cer() / corpus_cer()
  3) Character Accuracy              char_accuracy() / corpus_char_accuracy()
  4) 문자열 완전 일치율(Exact Match) exact_match_rate()
  5) 목표 역산                       required_char_accuracy()

용어 정리
  - CER = (S + D + I) / N  : 치환·삭제·삽입 연산 수를 정답 길이로 나눈 값
  - Character Accuracy     : 1 - CER (단, 삽입이 많으면 음수가 될 수 있다)
  - Exact Match Rate       : 문자열 전체가 완전히 같은 샘플의 비율
"""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass


def normalize(text: str | None) -> str:
    """비교 전 정규화. 전각/반각과 공백 차이로 오답 처리되는 것을 막는다."""
    if not text:
        return ""
    t = unicodedata.normalize("NFKC", str(text))
    return "".join(ch for ch in t if not ch.isspace()).upper()


# ---------------------------------------------------------------------------
# 1) 편집 거리
# ---------------------------------------------------------------------------
def levenshtein(ref: str, hyp: str) -> int:
    """정답 ref 를 예측 hyp 로 바꾸는 데 필요한 최소 편집 연산 수.

    동적 계획법으로 (len(ref)+1) x (len(hyp)+1) 표를 채운다.
    번호판은 길이가 7~9자이므로 연산량은 무시할 수 있다.
    """
    if ref == hyp:
        return 0
    if not ref:
        return len(hyp)
    if not hyp:
        return len(ref)
    prev = list(range(len(hyp) + 1))          # 0 행: 전부 삽입
    for i, rc in enumerate(ref, start=1):
        cur = [i]                             # 0 열: 전부 삭제
        for j, hc in enumerate(hyp, start=1):
            cost = 0 if rc == hc else 1
            cur.append(min(
                prev[j] + 1,                  # 삭제
                cur[j - 1] + 1,               # 삽입
                prev[j - 1] + cost,           # 치환(같으면 비용 0)
            ))
        prev = cur
    return prev[-1]


def edit_ops(ref: str, hyp: str) -> dict[str, int]:
    """치환(S)/삭제(D)/삽입(I) 연산 수를 분해해서 돌려준다.

    표를 다시 역추적(backtrace)해 어떤 연산이 몇 번 쓰였는지 센다.
    오류 유형 분석에서 '삽입이 많은가, 삭제가 많은가'를 보려면 필요하다.
    """
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1,
                          d[i - 1][j - 1] + cost)
    ops = {"S": 0, "D": 0, "I": 0, "C": 0}
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1] \
                and d[i][j] == d[i - 1][j - 1]:
            ops["C"] += 1
            i, j = i - 1, j - 1
        elif i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + 1:
            ops["S"] += 1
            i, j = i - 1, j - 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            ops["D"] += 1
            i -= 1
        else:
            ops["I"] += 1
            j -= 1
    return ops


# ---------------------------------------------------------------------------
# 2) 샘플 단위 지표
# ---------------------------------------------------------------------------
def cer(ref: str, hyp: str) -> float:
    """샘플 하나의 Character Error Rate."""
    r, h = normalize(ref), normalize(hyp)
    if not r:
        return 0.0 if not h else 1.0
    return levenshtein(r, h) / len(r)


def char_accuracy(ref: str, hyp: str) -> float:
    """샘플 하나의 문자 정확도 = 1 - CER. 0 미만은 0 으로 자른다."""
    return max(0.0, 1.0 - cer(ref, hyp))


def is_exact(ref: str, hyp: str) -> bool:
    """문자열 완전 일치 여부."""
    return normalize(ref) == normalize(hyp) and bool(normalize(ref))


# ---------------------------------------------------------------------------
# 3) 데이터셋 단위 지표
# ---------------------------------------------------------------------------
@dataclass
class OcrMetrics:
    """데이터셋 전체 평가 결과."""

    samples: int = 0
    total_ref_chars: int = 0
    total_edits: int = 0
    subs: int = 0
    dels: int = 0
    ins: int = 0
    exact_hits: int = 0
    empty_hyp: int = 0

    @property
    def cer(self) -> float:
        """코퍼스 CER. 샘플별 CER 의 평균이 아니라 전체 합으로 계산한다."""
        if not self.total_ref_chars:
            return 0.0
        return self.total_edits / self.total_ref_chars

    @property
    def char_accuracy(self) -> float:
        """코퍼스 문자 정확도."""
        return max(0.0, 1.0 - self.cer)

    @property
    def exact_match_rate(self) -> float:
        """문자열 완전 일치율."""
        if not self.samples:
            return 0.0
        return self.exact_hits / self.samples

    @property
    def implied_exact(self) -> float:
        """문자 정확도가 자리마다 독립이라고 가정했을 때의 이론 완전 일치율.

        평균 길이 L 에 대해 acc**L 이다. 실제 완전 일치율이 이 값보다
        훨씬 낮으면 오류가 특정 샘플에 몰려 있다는 뜻이고,
        반대로 높으면 오류가 특정 자리에 반복적으로 몰려 있다는 뜻이다.
        """
        if not self.samples:
            return 0.0
        avg_len = self.total_ref_chars / self.samples
        return self.char_accuracy ** avg_len

    def report(self) -> str:
        return "\n".join([
            f"샘플 수              : {self.samples}",
            f"정답 문자 수         : {self.total_ref_chars}",
            f"편집 연산(S/D/I)     : {self.subs}/{self.dels}/{self.ins}"
            f"  합계 {self.total_edits}",
            f"CER                  : {self.cer:.6f}  ({self.cer * 100:.4f}%)",
            f"Character Accuracy   : {self.char_accuracy:.6f}"
            f"  ({self.char_accuracy * 100:.4f}%)",
            f"Exact Match Rate     : {self.exact_match_rate:.6f}"
            f"  ({self.exact_match_rate * 100:.4f}%)",
            f"이론 완전 일치율     : {self.implied_exact:.6f}"
            f"  ({self.implied_exact * 100:.4f}%)",
            f"빈 예측(미판독)      : {self.empty_hyp}",
        ])


def evaluate(pairs: list[tuple[str, str]]) -> OcrMetrics:
    """(정답, 예측) 목록을 받아 데이터셋 전체 지표를 계산한다."""
    m = OcrMetrics()
    for ref, hyp in pairs:
        r, h = normalize(ref), normalize(hyp)
        if not r:
            continue
        m.samples += 1
        m.total_ref_chars += len(r)
        if not h:
            m.empty_hyp += 1
        ops = edit_ops(r, h)
        m.subs += ops["S"]
        m.dels += ops["D"]
        m.ins += ops["I"]
        m.total_edits += ops["S"] + ops["D"] + ops["I"]
        if r == h:
            m.exact_hits += 1
    return m


def corpus_cer(pairs: list[tuple[str, str]]) -> float:
    """코퍼스 CER 단축 호출."""
    return evaluate(pairs).cer


def corpus_char_accuracy(pairs: list[tuple[str, str]]) -> float:
    """코퍼스 문자 정확도 단축 호출."""
    return evaluate(pairs).char_accuracy


def exact_match_rate(pairs: list[tuple[str, str]]) -> float:
    """완전 일치율 단축 호출."""
    return evaluate(pairs).exact_match_rate


# ---------------------------------------------------------------------------
# 4) 목표 역산
# ---------------------------------------------------------------------------
def required_char_accuracy(target_exact: float, length: int = 7) -> float:
    """완전 일치율 목표를 달성하기 위해 필요한 문자 정확도.

    자리별 오류가 독립이라고 가정하면 exact = acc**length 이므로
    acc = exact ** (1/length) 이다.
    """
    if length <= 0:
        return 0.0
    return target_exact ** (1.0 / length)


def expected_exact(char_acc: float, length: int = 7) -> float:
    """문자 정확도로부터 기대되는 완전 일치율."""
    return char_acc ** length


def pipeline_exact(detect_recall: float, char_acc: float,
                   length: int = 7) -> float:
    """검출률까지 포함한 종단(end-to-end) 완전 일치율.

    번호판을 찾지 못하면 OCR 기회조차 없으므로 두 값을 곱해야 한다.
    """
    return detect_recall * expected_exact(char_acc, length)


def _demo() -> None:
    # ----- (1) 소규모 평가 세트 -----
    pairs = [
        ("12가3456", "12가3456"),      # 정답
        ("12가3456", "12가3458"),      # 마지막 자리 치환
        ("서울34나5678", "서울34나5678"),
        ("서울34나5678", "서울34나567"),  # 삭제
        ("87로2345", "87로23456"),     # 삽입
        ("87로2345", "87도2345"),      # 한글 혼동
        ("12가3456", ""),              # 미판독
        ("123가4567", "123가4567"),
        ("123가4567", "12가4567"),     # 숫자 삭제
        ("34허5678", "34허5678"),
    ]
    m = evaluate(pairs)
    print("=== 평가 세트 지표 ===")
    print(m.report())

    print("\n=== 샘플별 상세 ===")
    print(f"{'정답':<12s} {'예측':<12s}  거리  CER     일치")
    for ref, hyp in pairs:
        d = levenshtein(normalize(ref), normalize(hyp))
        print(f"{ref:<12s} {hyp or '(없음)':<12s}  {d:>3d}  "
              f"{cer(ref, hyp):.4f}  {'O' if is_exact(ref, hyp) else 'X'}")

    # ----- (2) 99% 목표 분해 -----
    print("\n=== 완전 일치율 목표와 필요한 문자 정확도 ===")
    print("목표 완전일치  7자리 필요 문자정확도  8자리     9자리")
    for t in (0.95, 0.98, 0.99, 0.995, 0.999):
        a7 = required_char_accuracy(t, 7)
        a8 = required_char_accuracy(t, 8)
        a9 = required_char_accuracy(t, 9)
        print(f"  {t * 100:6.2f}%          {a7 * 100:8.4f}%"
              f"      {a8 * 100:8.4f}%  {a9 * 100:8.4f}%")

    print("\n=== 문자 정확도에서 기대되는 완전 일치율(7자리) ===")
    for a in (0.99, 0.995, 0.998, 0.999, 0.9999):
        print(f"  문자 정확도 {a * 100:7.3f}% -> 완전 일치율 "
              f"{expected_exact(a, 7) * 100:7.3f}%")

    print("\n=== 검출률까지 포함한 종단 완전 일치율(7자리) ===")
    for dr in (0.99, 0.995, 0.999):
        for a in (0.998, 0.9986, 0.999):
            print(f"  검출률 {dr * 100:6.2f}% x 문자정확도 {a * 100:7.3f}%"
                  f" -> {pipeline_exact(dr, a, 7) * 100:7.3f}%")


if __name__ == "__main__":
    _demo()
