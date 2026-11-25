# -*- coding: utf-8 -*-
"""정비 전략별 총비용 시뮬레이션.

사후정비(Reactive) / 예방정비(Preventive) / 상태기반정비(CBM) / 예지보전(PdM)
네 가지 전략을 같은 차량 집단에 적용했을 때의 연간 총비용을 비교한다.

총비용 = 고장비용 + 정비비용 + 가동중단비용

numpy 만 사용하므로 추가 설치 없이 실행된다.
"""
from __future__ import annotations

import numpy as np

# ----------------------------------------------------------------------
# 시뮬레이션 공통 파라미터 (단위: 원, 일)
# ----------------------------------------------------------------------
N_VEHICLE = 100          # 관리 대상 차량 수
DAYS = 365               # 시뮬레이션 기간(일)
SEED = 20261024          # 재현성을 위한 고정 시드

BASE_FAILURE_RATE = 0.0022   # 정비를 전혀 하지 않을 때의 1일 고장 확률
COST_FAILURE = 1_850_000     # 고장 1건당 수리비(부품 + 공임 + 견인)
COST_PM = 320_000            # 예방정비 1회 비용(부품 선제 교체 포함)
COST_INSPECT = 45_000        # 점검/진단 1회 비용
COST_DOWNTIME_DAY = 420_000  # 가동중단 1일당 손실(영업 기회비용)

DOWNTIME_FAILURE = 3.4       # 고장 정지 시 평균 가동중단 일수
DOWNTIME_PLANNED = 0.6       # 계획정비 시 평균 가동중단 일수

# 센서 기반 전략의 탐지 성능 (상태기반 < 예지보전)
CBM_RECALL = 0.58            # 고장 전 징후를 잡아내는 비율
CBM_FPR = 0.055              # 정상인데 정비를 부른 오경보 비율(기간 중)
PDM_RECALL = 0.86
PDM_FPR = 0.021


def simulate_reactive(rng: np.random.Generator) -> dict:
    """사후정비: 고장이 날 때까지 아무것도 하지 않는다."""
    n_fail = 0
    for _ in range(N_VEHICLE):
        # 하루 단위 베르누이 시행. 고장이 나면 수리 후 다시 운행한다.
        day = 0
        while day < DAYS:
            if rng.random() < BASE_FAILURE_RATE:
                n_fail += 1
                day += int(DOWNTIME_FAILURE) + 1   # 수리 기간 동안은 고장 없음
            else:
                day += 1
    cost_fail = n_fail * COST_FAILURE
    cost_maint = 0
    cost_down = n_fail * DOWNTIME_FAILURE * COST_DOWNTIME_DAY
    return _pack("사후정비", n_fail, 0, 0, cost_fail, cost_maint, cost_down)


def simulate_preventive(rng: np.random.Generator, interval_days: int = 90) -> dict:
    """예방정비: 주기(기본 90일)마다 무조건 부품을 교체한다."""
    n_pm_total = 0
    n_fail = 0
    for _ in range(N_VEHICLE):
        n_pm = DAYS // interval_days
        n_pm_total += n_pm
        # 주기 정비로 고장률이 낮아지지만, 주기 사이에 발생하는 고장은 남는다.
        # 교체 직후 고장률이 가장 낮고 주기 말기에 다시 올라가는 형태를 가정한다.
        for d in range(DAYS):
            phase = (d % interval_days) / interval_days      # 0 -> 1
            rate = BASE_FAILURE_RATE * (0.18 + 0.52 * phase)
            if rng.random() < rate:
                n_fail += 1
    cost_fail = n_fail * COST_FAILURE
    cost_maint = n_pm_total * COST_PM
    cost_down = (n_fail * DOWNTIME_FAILURE
                 + n_pm_total * DOWNTIME_PLANNED) * COST_DOWNTIME_DAY
    return _pack("예방정비(90일)", n_fail, n_pm_total, 0,
                 cost_fail, cost_maint, cost_down)


def simulate_condition_based(rng: np.random.Generator,
                             recall: float, fpr: float,
                             inspect_interval: int = 30,
                             label: str = "상태기반정비") -> dict:
    """상태기반정비: 주기 점검으로 상태를 확인하고 이상할 때만 정비한다."""
    n_fail = 0
    n_pm = 0
    n_inspect = N_VEHICLE * (DAYS // inspect_interval)
    for _ in range(N_VEHICLE):
        for d in range(DAYS):
            if rng.random() < BASE_FAILURE_RATE:
                # 잠재 고장 발생. 점검 주기 안에 징후를 잡으면 계획정비로 전환된다.
                if rng.random() < recall:
                    n_pm += 1
                else:
                    n_fail += 1
        # 오경보로 인한 불필요 정비
        if rng.random() < fpr * (DAYS / 365) * 4:
            n_pm += 1
    cost_fail = n_fail * COST_FAILURE
    cost_maint = n_pm * COST_PM + n_inspect * COST_INSPECT
    cost_down = (n_fail * DOWNTIME_FAILURE
                 + n_pm * DOWNTIME_PLANNED) * COST_DOWNTIME_DAY
    return _pack(label, n_fail, n_pm, n_inspect,
                 cost_fail, cost_maint, cost_down)


def simulate_predictive(rng: np.random.Generator) -> dict:
    """예지보전: 상시 센서 스트림으로 판단하므로 주기 점검비가 거의 없다."""
    n_fail = 0
    n_pm = 0
    # 상시 모니터링은 점검 인건비 대신 시스템 운영비로 치환된다.
    n_inspect = N_VEHICLE * 2            # 연 2회 정밀 검증만 수행
    for _ in range(N_VEHICLE):
        for d in range(DAYS):
            if rng.random() < BASE_FAILURE_RATE:
                if rng.random() < PDM_RECALL:
                    n_pm += 1
                else:
                    n_fail += 1
        if rng.random() < PDM_FPR * 4:
            n_pm += 1
    cost_fail = n_fail * COST_FAILURE
    cost_maint = n_pm * COST_PM + n_inspect * COST_INSPECT
    cost_down = (n_fail * DOWNTIME_FAILURE
                 + n_pm * DOWNTIME_PLANNED) * COST_DOWNTIME_DAY
    return _pack("예지보전(AI)", n_fail, n_pm, n_inspect,
                 cost_fail, cost_maint, cost_down)


def _pack(name, n_fail, n_pm, n_inspect, cost_fail, cost_maint, cost_down):
    """결과 딕셔너리를 통일된 형태로 만든다."""
    total = cost_fail + cost_maint + cost_down
    return {
        "name": name,
        "fail": n_fail,
        "pm": n_pm,
        "inspect": n_inspect,
        "cost_fail": cost_fail,
        "cost_maint": cost_maint,
        "cost_down": cost_down,
        "total": total,
        "per_vehicle": total / N_VEHICLE,
    }


def main() -> None:
    rng = np.random.default_rng(SEED)
    results = [
        simulate_reactive(rng),
        simulate_preventive(rng, interval_days=90),
        simulate_condition_based(rng, CBM_RECALL, CBM_FPR,
                                 inspect_interval=30,
                                 label="상태기반정비"),
        simulate_predictive(rng),
    ]

    print(f"차량 {N_VEHICLE}대 / {DAYS}일 시뮬레이션 (단위: 만원)")
    print("-" * 78)
    print(f"{'전략':<16}{'고장':>5}{'계획정비':>8}{'점검':>7}"
          f"{'고장비':>10}{'정비비':>10}{'중단손실':>10}{'총비용':>10}")
    print("-" * 78)
    for r in results:
        print(f"{r['name']:<16}{r['fail']:>5}{r['pm']:>8}{r['inspect']:>7}"
              f"{r['cost_fail'] / 1e4:>10,.0f}{r['cost_maint'] / 1e4:>10,.0f}"
              f"{r['cost_down'] / 1e4:>10,.0f}{r['total'] / 1e4:>10,.0f}")
    print("-" * 78)

    base = results[0]["total"]
    print("\n[사후정비 대비 절감률]")
    for r in results[1:]:
        saving = (base - r["total"]) / base * 100
        print(f"  {r['name']:<16} 총비용 {r['total'] / 1e4:>8,.0f}만원  "
              f"절감 {saving:>6.1f}%  대당 {r['per_vehicle'] / 1e4:>6.1f}만원")

    best = min(results, key=lambda x: x["total"])
    print(f"\n최저 총비용 전략: {best['name']} "
          f"({best['total'] / 1e4:,.0f}만원)")


if __name__ == "__main__":
    main()
