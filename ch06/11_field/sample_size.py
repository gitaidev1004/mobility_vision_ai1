# -*- coding: utf-8 -*-
"""실환경 검증 표본 수 산정기.

"완전일치율 99%" 를 신뢰수준 95% 로 주장하려면 몇 건을 측정해야 하는지 계산한다.
외부 패키지 없이 표준 라이브러리만 사용하므로 어떤 환경에서도 바로 실행된다.

세 가지 질문에 각각 답한다.
  (1) 실패 0건을 전제로 "참 정확도가 99% 이상" 이라고 주장하려면 최소 몇 건인가?
  (2) 실패를 k건까지 허용한다면 몇 건이 필요한가?
  (3) 정확도를 ±e 폭으로 "추정" 하려면 몇 건이 필요한가?

실행:  python 11_field/sample_size.py
"""

from __future__ import annotations

import math

# 목표 완전일치율(참값의 하한으로 주장하려는 값)
TARGET_RATE = 0.99
# 신뢰수준 95% → 유의수준 0.05 (단측)
ALPHA = 0.05
# 표준정규분포 양측 95% 분위수
Z_TWO_SIDED_95 = 1.959963985


def zero_failure_n(target: float = TARGET_RATE, alpha: float = ALPHA) -> int:
    """실패 0건 전제에서 필요한 최소 표본 수를 구한다.

    n건을 측정해 한 건도 틀리지 않았을 때, 참 정확도 p 가 target 미만일 확률은
    최대 target**n 이다. 이 값이 alpha 이하가 되어야 "p >= target" 을 신뢰수준
    (1-alpha) 로 주장할 수 있다. 즉 target**n <= alpha 를 풀면 된다.

    Args:
        target: 주장하려는 정확도 하한(0~1).
        alpha: 유의수준(0.05 = 신뢰수준 95%).

    Returns:
        필요한 최소 표본 수(건).
    """
    return math.ceil(math.log(alpha) / math.log(target))


def _binom_cdf(k: int, n: int, p: float) -> float:
    """이항분포 누적확률 P(X <= k) 를 구한다.

    조합 수를 그대로 계산하면 n 이 수백만 되어도 float 범위를 넘으므로
    로그 공간(``math.lgamma``)에서 각 항을 계산한 뒤 지수로 되돌린다.

    Args:
        k: 누적 상한.
        n: 시행 횟수.
        p: 1회 성공(여기서는 '실패 발생') 확률.

    Returns:
        P(X <= k).
    """
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 0.0
    log_p, log_q = math.log(p), math.log1p(-p)
    log_fact_n = math.lgamma(n + 1)
    total = 0.0
    for i in range(k + 1):
        log_term = (log_fact_n - math.lgamma(i + 1) - math.lgamma(n - i + 1)
                    + i * log_p + (n - i) * log_q)
        total += math.exp(log_term)
    return min(1.0, total)


def n_for_allowed_failures(k: int, target: float = TARGET_RATE,
                           alpha: float = ALPHA) -> int:
    """실패 k건까지 허용할 때 필요한 최소 표본 수를 구한다.

    참 정확도가 정확히 target 인(= 겨우 턱걸이인) 모델에서 실패 건수는
    Bin(n, 1-target) 을 따른다. 그 모델이 n건 중 k건 이하만 틀려서 잘못
    합격해 버릴 확률이 alpha 이하가 되는 가장 작은 n 을 찾는다.
    실패 허용 건수가 늘어나면 필요한 표본 수는 급격히 증가한다.

    Args:
        k: 허용 실패 건수.
        target: 주장하려는 정확도 하한.
        alpha: 유의수준.

    Returns:
        필요한 최소 표본 수(건).
    """
    fail_rate = 1.0 - target
    n = k + 1
    while _binom_cdf(k, n, fail_rate) > alpha:
        n += 1
        if n > 200000:  # 방어: 현실적으로 도달하지 않는 구간
            break
    return n


def clopper_pearson_lower(successes: int, n: int, alpha: float = ALPHA) -> float:
    """Clopper-Pearson 정확 신뢰구간의 하한을 구한다(단측 1-alpha).

    이분 탐색으로 "P(X >= successes | p) = alpha" 를 만족하는 p 를 찾는다.
    근사식(정규근사)은 성공률이 1에 가까울 때 하한을 과대평가하므로,
    99% 급 정확도를 다룰 때는 정확구간을 쓰는 것이 안전하다.

    Args:
        successes: 완전일치 건수.
        n: 전체 측정 건수.
        alpha: 유의수준.

    Returns:
        참 정확도의 하한 추정값(0~1).
    """
    if n <= 0:
        return 0.0
    if successes >= n:
        # 실패 0건이면 하한은 alpha**(1/n)
        return alpha ** (1.0 / n)
    if successes <= 0:
        return 0.0
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        # P(X >= successes | p=mid) = 1 - P(X <= successes-1)
        tail = 1.0 - _binom_cdf(successes - 1, n, mid)
        if tail < alpha:
            lo = mid
        else:
            hi = mid
    return lo


def n_for_margin(expected: float, margin: float,
                 z: float = Z_TWO_SIDED_95) -> int:
    """정확도를 ±margin 폭으로 추정하기 위한 표본 수를 구한다.

    합격/불합격 판정이 아니라 "정확도가 얼마인지" 를 보고할 때 쓴다.
    n = z^2 * p(1-p) / margin^2 의 Wald 공식을 쓰되, 결과를 올림한다.

    Args:
        expected: 예상 정확도(사전 측정값 또는 목표값).
        margin: 허용 오차 폭(0.01 = ±1%p).
        z: 정규분포 분위수(기본 양측 95%).

    Returns:
        필요한 최소 표본 수(건).
    """
    return math.ceil((z ** 2) * expected * (1.0 - expected) / (margin ** 2))


def allocate(total: int, weights: dict[str, float]) -> dict[str, int]:
    """환경별 표본 할당량을 계산한다.

    환경별 분해 평가를 하려면 각 환경 안에서도 통계적 의미가 있어야 한다.
    실제 운영 빈도(weights)를 기준으로 비례 배분하되, 어떤 환경도
    30건 미만으로 떨어지지 않게 하한을 둔다.

    Args:
        total: 전체 표본 수.
        weights: 환경명 → 운영 빈도 비율.

    Returns:
        환경명 → 할당 건수.
    """
    floor_n = 30
    denom = sum(weights.values()) or 1.0
    out = {name: max(floor_n, math.ceil(total * w / denom))
           for name, w in weights.items()}
    return out


def main() -> None:
    """표본 수 산정 결과를 표 형태로 출력한다."""
    print("=" * 66)
    print("완전일치율 99% 를 신뢰수준 95% 로 검증하기 위한 표본 수")
    print("=" * 66)

    n0 = zero_failure_n()
    print(f"\n[1] 실패 0건 전제 (가장 작은 표본)")
    print(f"    조건 : 0.99^n <= 0.05")
    print(f"    결과 : n = {n0} 건  → {n0}건 전수 정답이면 하한 99% 주장 가능")
    print(f"    검산 : 0.99^{n0} = {TARGET_RATE ** n0:.5f} (<= 0.05)")
    print(f"           0.99^{n0 - 1} = {TARGET_RATE ** (n0 - 1):.5f} (> 0.05)")

    print(f"\n[2] 실패를 k건까지 허용할 때")
    print(f"    {'허용 실패':>9} {'필요 표본':>10} {'관측 정확도':>11} {'하한(CP)':>10}")
    for k in (0, 1, 2, 3, 5, 10):
        n = n_for_allowed_failures(k)
        obs = (n - k) / n
        lower = clopper_pearson_lower(n - k, n)
        print(f"    {k:>9} {n:>10} {obs * 100:>10.2f}% {lower * 100:>9.2f}%")

    print(f"\n[3] 정확도를 ±오차 폭으로 '추정' 할 때 (예상 99%)")
    print(f"    {'허용 오차':>9} {'필요 표본':>10}")
    for m in (0.03, 0.02, 0.01, 0.005):
        print(f"    {m * 100:>8.1f}%p {n_for_margin(0.99, m):>10}")

    print(f"\n[4] 환경별 할당 (전체 {n0 * 2}건, 운영 빈도 비례 · 환경당 최소 30건)")
    weights = {
        "주간": 0.34, "야간": 0.22, "우천": 0.08, "역광": 0.10,
        "원거리": 0.08, "고속이동": 0.06, "기울기": 0.07, "부분가림": 0.05,
    }
    alloc = allocate(n0 * 2, weights)
    print(f"    {'환경':>9} {'빈도':>7} {'할당':>7} {'실패 0건 시 하한':>16}")
    for name, cnt in alloc.items():
        lower = clopper_pearson_lower(cnt, cnt)
        print(f"    {name:>9} {weights[name] * 100:>6.0f}% {cnt:>7} {lower * 100:>15.2f}%")
    print(f"    {'합계':>9} {'100%':>7} {sum(alloc.values()):>7}")

    print(f"\n[5] 해석")
    print("    - 환경당 150건 안팎으로는 그 환경만으로 99% 를 '증명' 할 수 없다.")
    print("      환경별 수치는 약점 탐색용이고, 99% 주장은 전체 집계로 한다.")
    print("    - 실패가 1건이라도 나오면 표본을 두 배 이상 늘려야 같은 주장이 가능하다.")
    print("      따라서 1차 300건에서 실패가 나오면 원인을 고치고 재측정하는 편이 빠르다.")


if __name__ == "__main__":
    main()
