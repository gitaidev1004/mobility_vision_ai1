# -*- coding: utf-8 -*-
"""모델 조합 종합 평가 — 게이트 판정 + 가중합 점수.

평가는 반드시 두 단계로 한다.
  1단계 게이트 : 운영 요구를 만족하지 못하는 조합을 먼저 걸러낸다.
  2단계 가중합 : 게이트를 통과한 조합들 사이의 상대 우위를 계산한다.

순서를 바꾸면 "속도 점수가 만점이지만 정확도가 미달인 조합" 이 1위로 올라온다.
이 스크립트가 그 함정을 숫자로 보여준다.

측정값은 ``MEASUREMENTS`` 상수에 둔다. 자기 팀의 측정 결과로 바꿔 넣으면
그대로 평가표가 다시 계산된다. 가중치도 ``WEIGHTS`` 에서 바꿀 수 있다.

실행
    python 14_compare/score_models.py
    python 14_compare/score_models.py --no-gate        # 게이트 없이 가중합만
    python 14_compare/score_models.py --exact-weight 0.6
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# 측정값 — 모두 예시다. 자기 팀 측정 결과로 교체해 쓴다.
# (조합명, 검출재현율, 완전일치율, p50ms, p95ms, RSS MB, 모델 MB)
# ---------------------------------------------------------------------------

MEASUREMENTS: list[tuple[str, float, float, float, float, float, float]] = [
    ("YOLO-nano + CRNN/CTC",    0.991, 0.962,  61.4,  88.1,  214.0,   9.8),
    ("MobileNet-SSD + CRNN",    0.964, 0.931,  44.2,  63.5,  168.0,   6.4),
    ("YOLO-nano + PaddleOCR",   0.989, 0.958,  94.8, 131.7,  298.0,  18.2),
    ("Deformable DETR + CRNN",  0.993, 0.965, 312.0, 418.0,  724.0, 158.0),
    ("DINO + CRNN",             0.995, 0.969, 486.0, 642.0,  968.0, 191.0),
    ("Swin-T + DETR + CRNN",    0.994, 0.967, 718.0, 947.0, 1424.0, 262.0),
]

# ---------------------------------------------------------------------------
# 게이트 기준 — 운영 요구에서 역산한 값. 하나라도 못 넘으면 후보에서 제외한다.
# ---------------------------------------------------------------------------

GATE_EXACT_MIN = 0.955     # 운영 개시 하한(목표 99% 와는 다른 값)
GATE_P95_MAX = 150.0       # 차단기 동작 예산(6.11절)
GATE_MEM_MAX = 512.0       # 컨테이너 메모리 제한(워커 1개 기준)
SIZE_PENALTY_OVER_MB = 50.0   # 이 크기를 넘으면 감점(게이트는 아님)
SIZE_PENALTY_POINTS = 5.0

# ---------------------------------------------------------------------------
# 가중치 — 합계 1.0. 정확도 계열에 0.55 를 주는 것이 이 설계의 선언이다.
# ---------------------------------------------------------------------------

WEIGHTS = {
    "exact": 0.40,    # 완전일치율 — 사용자가 직접 체감하는 유일한 지표
    "recall": 0.15,   # 검출 재현율 — 완전일치율의 상한을 결정
    "p95": 0.25,      # p95 지연 — 설비 설계 기준
    "mem": 0.10,      # 메모리 — 배포 가능성
    "size": 0.10,     # 모델 크기 — 배포 편의
}


@dataclass
class Scored:
    """한 조합의 점수와 판정."""

    name: str
    recall: float
    exact: float
    p50: float
    p95: float
    mem: float
    size: float
    s_exact: float = 0.0
    s_recall: float = 0.0
    s_p95: float = 0.0
    s_mem: float = 0.0
    s_size: float = 0.0
    total: float = 0.0
    penalty: float = 0.0
    gate_fail: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        """게이트 통과 여부."""
        return not self.gate_fail

    @property
    def throughput(self) -> float:
        """단일 스레드 직렬 처리량(건/초)."""
        return 1000.0 / self.p50 if self.p50 else 0.0


def check_gate(row: Scored) -> tuple[str, ...]:
    """게이트 기준 미달 사유를 모아 돌려준다.

    Args:
        row: 평가 대상.

    Returns:
        미달 사유 문자열 튜플(통과하면 빈 튜플).
    """
    fails: list[str] = []
    if row.exact < GATE_EXACT_MIN:
        fails.append(f"정확도 미달 {row.exact * 100:.1f}%")
    if row.p95 > GATE_P95_MAX:
        fails.append(f"p95 초과 {row.p95:.0f}ms")
    if row.mem > GATE_MEM_MAX:
        fails.append(f"메모리 초과 {row.mem:.0f}MB")
    return tuple(fails)


def score_all(rows: list[Scored]) -> list[Scored]:
    """정규화 후 가중합 점수를 계산한다.

    정규화 방식이 지표에 따라 다르다.
      - 높을수록 좋은 지표(정확도·재현율): 값 ÷ 최고값 × 100
      - 낮을수록 좋은 지표(지연·메모리·크기): 최저값 ÷ 값 × 100
    둘을 섞으면 지연이 짧은 모델이 정확도에서도 높은 점수를 받는 오류가 생긴다.

    Args:
        rows: 평가 대상 목록.

    Returns:
        점수가 채워진 같은 목록.
    """
    best_exact = max(r.exact for r in rows)
    best_recall = max(r.recall for r in rows)
    low_p95 = min(r.p95 for r in rows)
    low_mem = min(r.mem for r in rows)
    low_size = min(r.size for r in rows)

    for r in rows:
        r.s_exact = r.exact / best_exact * 100
        r.s_recall = r.recall / best_recall * 100
        r.s_p95 = low_p95 / r.p95 * 100
        r.s_mem = low_mem / r.mem * 100
        r.s_size = low_size / r.size * 100
        r.penalty = SIZE_PENALTY_POINTS if r.size > SIZE_PENALTY_OVER_MB else 0.0
        r.total = (WEIGHTS["exact"] * r.s_exact
                   + WEIGHTS["recall"] * r.s_recall
                   + WEIGHTS["p95"] * r.s_p95
                   + WEIGHTS["mem"] * r.s_mem
                   + WEIGHTS["size"] * r.s_size
                   - r.penalty)
        r.gate_fail = check_gate(r)
    return rows


def print_measurements(rows: list[Scored]) -> None:
    """원시 측정값 표를 출력한다."""
    print("\n[1] 원시 측정값")
    print(f"  {'조합':<24}{'재현율':>8}{'완전일치':>9}{'p50':>8}"
          f"{'p95':>8}{'RSS':>8}{'크기':>8}{'처리량':>10}")
    for r in rows:
        print(f"  {r.name:<24}{r.recall * 100:>7.1f}%{r.exact * 100:>8.1f}%"
              f"{r.p50:>8.1f}{r.p95:>8.1f}{r.mem:>8.0f}{r.size:>8.1f}"
              f"{r.throughput:>8.1f}/s")


def print_gate(rows: list[Scored]) -> list[Scored]:
    """게이트 판정 결과를 출력하고 통과 목록을 돌려준다."""
    print("\n[2] 1단계 게이트 판정")
    print(f"  기준: 완전일치율 {GATE_EXACT_MIN * 100:.1f}% 이상 / "
          f"p95 {GATE_P95_MAX:.0f}ms 이하 / 메모리 {GATE_MEM_MAX:.0f}MB 이하")
    print(f"  {'조합':<24}{'판정':<8}사유")
    passed: list[Scored] = []
    for r in rows:
        if r.passed:
            passed.append(r)
            print(f"  {r.name:<24}{'통과':<8}-")
        else:
            print(f"  {r.name:<24}{'탈락':<8}{' / '.join(r.gate_fail)}")
    print(f"  → {len(passed)}개 조합이 통과했다.")
    return passed


def print_scores(rows: list[Scored], title: str) -> None:
    """가중합 점수표를 출력한다(총점 내림차순)."""
    print(f"\n{title}")
    print(f"  가중치: 완전일치 {WEIGHTS['exact']:.2f} / 재현율 {WEIGHTS['recall']:.2f}"
          f" / p95 {WEIGHTS['p95']:.2f} / 메모리 {WEIGHTS['mem']:.2f}"
          f" / 크기 {WEIGHTS['size']:.2f}")
    print(f"  {'조합':<24}{'정확':>7}{'재현':>7}{'지연':>7}"
          f"{'메모':>7}{'크기':>7}{'감점':>6}{'총점':>8}")
    for i, r in enumerate(sorted(rows, key=lambda x: -x.total), start=1):
        print(f"  {r.name:<24}{r.s_exact:>7.1f}{r.s_recall:>7.1f}"
              f"{r.s_p95:>7.1f}{r.s_mem:>7.1f}{r.s_size:>7.1f}"
              f"{-r.penalty:>6.0f}{r.total:>8.1f}")


def print_conclusion(all_rows: list[Scored], passed: list[Scored]) -> None:
    """선정 결과와 그 논리를 출력한다."""
    print("\n[4] 선정 결과")
    if not passed:
        print("  게이트를 통과한 조합이 없다. 기준을 재검토하거나 모델을 개선한다.")
        return
    winner = max(passed, key=lambda r: r.total)
    overall_top = max(all_rows, key=lambda r: r.total)

    print(f"  선정: {winner.name} (게이트 통과 중 총점 {winner.total:.1f})")
    if overall_top is not winner:
        print(f"  주의: 게이트를 무시하면 {overall_top.name} 이 총점 "
              f"{overall_top.total:.1f} 로 1위다.")
        print(f"        그러나 {' / '.join(overall_top.gate_fail)} 로 배포할 수 없다.")
        print("        → 게이트를 가중합보다 먼저 적용해야 하는 이유다.")

    runner = sorted(passed, key=lambda r: -r.total)
    if len(runner) >= 2:
        a, b = runner[0], runner[1]
        print(f"\n  통과 조합 비교: {a.name} vs {b.name}")
        print(f"    완전일치율  {a.exact * 100:.1f}% vs {b.exact * 100:.1f}% "
              f"({(a.exact - b.exact) * 100:+.1f}%p)")
        print(f"    p95 지연    {a.p95:.1f}ms vs {b.p95:.1f}ms "
              f"({(a.p95 / b.p95 - 1) * 100:+.0f}%)")
        print(f"    메모리      {a.mem:.0f}MB vs {b.mem:.0f}MB "
              f"({(a.mem / b.mem - 1) * 100:+.0f}%)")
        print(f"    모델 크기   {a.size:.1f}MB vs {b.size:.1f}MB "
              f"({(a.size / b.size - 1) * 100:+.0f}%)")

    ceiling = max(r.exact for r in all_rows)
    print(f"\n  정확도 상한: {ceiling * 100:.1f}% "
          f"(가장 무거운 모델도 이 이상은 내지 못했다)")
    print(f"  선정 조합과 상한의 차이 {(ceiling - winner.exact) * 100:.1f}%p")
    print("  → 이 차이가 작으면 모델을 키워 얻을 것이 거의 없다는 뜻이다.")
    print("    남은 개선은 데이터·촬영 조건에서 찾는다.")


def main(argv: list[str] | None = None) -> int:
    """명령행 진입점."""
    ap = argparse.ArgumentParser(description="모델 조합 종합 평가")
    ap.add_argument("--no-gate", action="store_true",
                    help="게이트를 적용하지 않고 가중합만 본다(비교용)")
    ap.add_argument("--exact-weight", type=float, default=None,
                    help="완전일치율 가중치를 바꿔 결론 민감도를 본다")
    args = ap.parse_args(argv)

    if args.exact_weight is not None:
        # 완전일치율 가중치를 바꾸고 나머지를 비례 축소해 합계 1.0 을 유지한다.
        new_w = max(0.0, min(1.0, args.exact_weight))
        rest = sum(v for k, v in WEIGHTS.items() if k != "exact")
        scale = (1.0 - new_w) / rest if rest else 0.0
        for k in list(WEIGHTS):
            WEIGHTS[k] = new_w if k == "exact" else WEIGHTS[k] * scale

    rows = [Scored(name=n, recall=rc, exact=ex, p50=p50, p95=p95, mem=m, size=s)
            for n, rc, ex, p50, p95, m, s in MEASUREMENTS]

    print("=" * 78)
    print("모델 조합 종합 평가 — 게이트 + 가중합")
    print("=" * 78)
    print("  주의: 아래 측정값은 모두 예시다. 자기 팀 측정 결과로 교체해 쓴다.")

    score_all(rows)
    print_measurements(rows)

    if args.no_gate:
        print("\n[2] 게이트 생략 (--no-gate)")
        passed = rows
    else:
        passed = print_gate(rows)

    print_scores(rows, "[3] 2단계 가중합 점수 (전체)")
    if passed and len(passed) != len(rows):
        print_scores(passed, "[3-1] 게이트 통과 조합만")
    print_conclusion(rows, passed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
