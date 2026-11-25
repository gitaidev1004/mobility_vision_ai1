# -*- coding: utf-8 -*-
"""문자 혼동 행렬(confusion matrix) 생성과 상위 혼동쌍 리포트.

번호판 OCR 의 개선 지점을 찾는 가장 빠른 방법은 '어떤 문자가
어떤 문자로 잘못 읽히는가'를 세어 보는 것이다. 문자 클래스가 67종이므로
행렬 전체를 출력하면 읽기 어렵다. 따라서
  (1) 혼동이 발생한 문자만 추려 축약 행렬을 출력하고,
  (2) 발생 빈도 상위 혼동쌍을 따로 리포트한다.

외부 라이브러리 없이 동작하며, CSV 로 내보내 엑셀에서 열 수도 있다.
"""
from __future__ import annotations

import csv
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "10_eval"))

from metrics_ocr import normalize  # noqa: E402

BLANK = "_"      # 삭제(모델이 글자를 빠뜨림)를 나타내는 기호
INSERT = "+"     # 삽입(모델이 없는 글자를 추가)을 나타내는 기호


def align(ref: str, hyp: str) -> list[tuple[str, str]]:
    """편집 거리 역추적으로 정답과 예측을 자리별로 정렬한다.

    같은 길이가 아니어도 (정답문자, 예측문자) 쌍으로 맞춰 주므로
    삽입·삭제가 섞인 결과도 혼동 행렬에 넣을 수 있다.
    빠진 글자는 ('6', '_'), 추가된 글자는 ('+', '8') 형태가 된다.
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
    pairs: list[tuple[str, str]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (
                0 if ref[i - 1] == hyp[j - 1] else 1):
            pairs.append((ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            pairs.append((ref[i - 1], BLANK))
            i -= 1
        else:
            pairs.append((INSERT, hyp[j - 1]))
            j -= 1
    pairs.reverse()
    return pairs


class CharConfusion:
    """문자 단위 혼동 통계 누적기."""

    def __init__(self) -> None:
        # matrix[정답문자][예측문자] = 횟수
        self.matrix: dict[str, Counter[str]] = defaultdict(Counter)
        self.support: Counter[str] = Counter()   # 정답에 등장한 횟수
        self.samples = 0

    def add(self, ref: str, hyp: str) -> None:
        """샘플 한 건을 누적한다."""
        r, h = normalize(ref), normalize(hyp)
        self.samples += 1
        for a, b in align(r, h):
            self.matrix[a][b] += 1
            if a != INSERT:
                self.support[a] += 1

    def add_all(self, pairs: list[tuple[str, str]]) -> None:
        for ref, hyp in pairs:
            self.add(ref, hyp)

    # -- 조회 --------------------------------------------------------------
    def error_count(self, ch: str) -> int:
        """정답 문자 ch 가 다른 글자로 읽힌 총 횟수."""
        return sum(n for b, n in self.matrix[ch].items() if b != ch)

    def accuracy(self, ch: str) -> float:
        """정답 문자 ch 의 문자별 정확도."""
        sup = self.support[ch]
        if not sup:
            return 0.0
        return self.matrix[ch][ch] / sup

    def top_pairs(self, k: int = 10) -> list[tuple[str, str, int, float]]:
        """상위 혼동쌍 (정답, 예측, 횟수, 해당 문자 내 비율)."""
        out: list[tuple[str, str, int, float]] = []
        for a, row in self.matrix.items():
            for b, n in row.items():
                if a == b:
                    continue
                sup = self.support[a] or 1
                out.append((a, b, n, n / sup))
        out.sort(key=lambda x: (-x[2], -x[3]))
        return out[:k]

    def weak_chars(self, k: int = 10, min_support: int = 3) -> \
            list[tuple[str, int, int, float]]:
        """정확도가 낮은 문자 목록 (문자, 등장수, 오류수, 정확도)."""
        rows = [(c, self.support[c], self.error_count(c), self.accuracy(c))
                for c in self.support if self.support[c] >= min_support]
        rows.sort(key=lambda x: (x[3], -x[2]))
        return rows[:k]

    # -- 출력 --------------------------------------------------------------
    def print_matrix(self, max_rows: int = 12, max_cols: int = 16) -> None:
        """혼동이 1회 이상 발생한 문자만 모아 축약 행렬을 출력한다.

        행은 '오류가 난 정답 문자', 열은 '그때 나온 예측 문자'다.
        열에는 자기 자신도 포함되므로 대각선이 정답 횟수가 된다.
        """
        rows = sorted(a for a in self.matrix
                      if self.error_count(a) > 0 and self.support[a] > 0)
        rows = rows[:max_rows]
        if not rows:
            print("혼동이 발생하지 않았다.")
            return
        cols = sorted({b for a in rows for b in self.matrix[a]
                       if self.matrix[a][b] > 0})[:max_cols]
        head = "정답\\예측 " + " ".join(f"{c:>3s}" for c in cols)
        print(head)
        print("-" * (len(cols) * 4 + 10))
        for a in rows:
            cells = " ".join(f"{self.matrix[a][b] or 0:>3d}" for b in cols)
            print(f"{a:<8s} {cells}   (등장 {self.support[a]}회, "
                  f"정확도 {self.accuracy(a):.1%})")

    def print_report(self, k: int = 10) -> None:
        """상위 혼동쌍과 취약 문자 리포트."""
        print(f"\n=== 상위 혼동쌍 {k}건 ===")
        print(f"{'정답':<5s}{'예측':<5s}{'횟수':>5s}  해당 문자 내 비율")
        for a, b, n, ratio in self.top_pairs(k):
            lab_a = "(삽입)" if a == INSERT else a
            lab_b = "(삭제)" if b == BLANK else b
            print(f"{lab_a:<5s}{lab_b:<5s}{n:>5d}  {ratio:7.2%}")

        print(f"\n=== 정확도가 낮은 문자 {k}종 ===")
        print(f"{'문자':<5s}{'등장':>5s}{'오류':>5s}  정확도")
        for c, sup, err, acc in self.weak_chars(k):
            print(f"{c:<5s}{sup:>5d}{err:>5d}  {acc:7.2%}")

    def to_csv(self, path: str) -> None:
        """전체 혼동 행렬을 CSV 로 저장한다."""
        labels = sorted(set(self.matrix) | {b for r in self.matrix.values()
                                            for b in r})
        with open(path, "w", encoding="utf-8-sig", newline="") as fp:
            w = csv.writer(fp)
            w.writerow(["ref/hyp"] + labels + ["support", "accuracy"])
            for a in labels:
                row = [a] + [self.matrix[a][b] for b in labels]
                row += [self.support[a], f"{self.accuracy(a):.4f}"]
                w.writerow(row)
        print(f"\n혼동 행렬을 저장했다: {path} ({len(labels)}x{len(labels)})")


def builtin_pairs() -> list[tuple[str, str]]:
    """교재 실습용 (정답, 예측) 데이터. 실제 분포를 모사해 혼동을 섞어 두었다."""
    base = [("12가3456", "12가3456")] * 12 + [("87로2345", "87로2345")] * 10
    base += [("서울34나5678", "서울34나5678")] * 8
    base += [("123가4567", "123가4567")] * 7
    errs = [
        ("12가3456", "12가345B"),     # 6 -> B
        ("12가3456", "12가3458"),     # 6 -> 8
        ("12가3456", "I2가3456"),     # 1 -> I
        ("12가3456", "12가3450"),     # 6 -> 0
        ("87로2345", "87도2345"),     # 로 -> 도
        ("87로2345", "87로2845"),     # 3 -> 8
        ("87로2345", "87로234"),      # 마지막 5 삭제
        ("87로2345", "B7로2345"),     # 8 -> B
        ("서울34나5678", "서울34나5G78"),  # 6 -> G
        ("서울34나5678", "서울34다5678"),  # 나 -> 다
        ("123가4567", "123가456T"),   # 7 -> T
        ("123가4567", "1Z3가4567"),   # 2 -> Z
        ("123가4567", "123가4S67"),   # 5 -> S
        ("12가3456", "12가34S6"),     # 5 -> S
        ("87로2345", "87로234S"),     # 5 -> S
        ("12가3456", "12가3466"),     # 5 -> 6
    ]
    return base + errs


def _demo() -> None:
    cm = CharConfusion()
    cm.add_all(builtin_pairs())
    print(f"=== 문자 혼동 행렬 (샘플 {cm.samples}건) ===")
    cm.print_matrix()
    cm.print_report(10)
    cm.to_csv("confusion_matrix.csv")


if __name__ == "__main__":
    _demo()
