# -*- coding: utf-8 -*-
"""정비 권고 생성 — 우선순위, due_km, due_days.

CONTRACT.md 의 recommendations[] = {priority, item, due_km, due_days} 를
채운다. 권고는 다음 세 가지를 모두 반영해야 실무에서 쓸 수 있다.

  1) 부품 상태  : 점수가 낮은 부품을 먼저 (09_parts/part_rules.py 결과)
  2) 안전 영향  : 같은 점수라도 제동·조향 관련을 먼저
  3) 주기 정비  : 주행거리 기반 정기 항목(엔진오일 등)도 빠뜨리지 않는다

due_days 와 due_km 는 둘 중 **먼저 도달하는 쪽**이 기한이라는 정비업계
관례를 그대로 따른다.

이 파일은 표준 라이브러리만 사용한다.
실행:  python 10_service/recommend.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "09_parts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from part_rules import evaluate_parts                       # noqa: E402
from health_score import predict, SAMPLES as HS_SAMPLES     # noqa: E402

# 부품별 권고 항목 정의.
#   safety : 안전 영향 가중(1=낮음 ~ 3=인명 직결). 우선순위 정렬에 쓴다.
#   km     : CRITICAL 상태에서의 기한(km). 상태가 좋아지면 비례 확대한다.
#   days   : CRITICAL 상태에서의 기한(일).
PART_ACTIONS = {
    "brake":      {"item": "브레이크 패드·디스크 점검 및 교체",
                   "safety": 3, "km": 500, "days": 7},
    "tire":       {"item": "타이어 공기압 조정 및 마모 점검",
                   "safety": 3, "km": 300, "days": 3},
    "cooling":    {"item": "냉각 계통 점검(냉각수량·라디에이터·써모스탯)",
                   "safety": 2, "km": 500, "days": 5},
    "engine":     {"item": "엔진 정밀 점검 및 엔진오일 교환",
                   "safety": 2, "km": 1000, "days": 14},
    "ev_battery": {"item": "구동 배터리 진단(SOH·셀 밸런싱)",
                   "safety": 2, "km": 2000, "days": 21},
    "battery":    {"item": "보조 배터리 전압 점검 및 교체 검토",
                   "safety": 1, "km": 2000, "days": 30},
}

# 상태 등급별 기한 배수. 상태가 나쁠수록 기한이 짧아진다.
STATUS_FACTOR = {"CRITICAL": 1.0, "WARNING": 2.5, "WATCH": 6.0, "NORMAL": 0.0}

# 주행거리 기반 정기 정비 항목. (항목, 주기 km, 안전 가중)
PERIODIC = [
    ("엔진오일 및 오일필터 교환", 10_000, 1),
    ("에어클리너 필터 교환", 20_000, 1),
    ("브레이크 오일 교환", 40_000, 2),
    ("점화플러그 교환", 60_000, 1),
    ("타이밍 벨트 점검", 80_000, 2),
    ("미션오일 교환", 100_000, 1),
]


def part_recommendations(components: list[dict], daily_km: float) -> list[dict]:
    """부품 판정 결과를 권고 목록으로 바꾼다. NORMAL 부품은 제외한다."""
    out = []
    for c in components:
        act = PART_ACTIONS.get(c["name"])
        if act is None:
            continue
        factor = STATUS_FACTOR.get(c["status"], 0.0)
        if factor <= 0.0:
            continue                         # NORMAL 은 권고하지 않는다
        due_days = int(round(act["days"] * factor))
        due_km = int(round(act["km"] * factor))
        # 일 기준과 주행거리 기준을 교차 검증해 더 이른 쪽으로 맞춘다.
        due_km = min(due_km, max(100, int(due_days * daily_km)))
        out.append({
            "item": act["item"], "due_km": due_km, "due_days": due_days,
            "_score": c["score"], "_safety": act["safety"],
            "_part": c["name"], "_status": c["status"],
        })
    return out


def periodic_recommendations(odometer_km: int, daily_km: float,
                             lead_km: int = 2000,
                             max_items: int = 2) -> list[dict]:
    """주기 정비 중 다음 도래 시점이 lead_km 안에 든 항목만 권고한다.

    10,000/20,000/40,000km 처럼 주기가 서로 배수 관계라서 특정
    주행거리에서는 여러 항목이 동시에 도래한다. 권고가 주기 항목으로
    가득 차면 상태 기반 권고가 밀려나므로 max_items 개로 제한한다.
    """
    out = []
    for item, cycle, safety in PERIODIC:
        next_at = ((odometer_km // cycle) + 1) * cycle
        remain = next_at - odometer_km
        if remain > lead_km:
            continue
        out.append({
            "item": f"{item} (누적 {next_at:,}km 도래)",
            "due_km": int(remain),
            "due_days": max(1, int(round(remain / max(daily_km, 1.0)))),
            # 주기 항목은 상태 점수가 없으므로 중간값을 준다.
            "_score": 72.0, "_safety": safety,
            "_part": "periodic", "_status": "WATCH",
        })
    # 주기가 긴 항목(브레이크 오일, 타이밍 벨트)이 더 중요하므로 안전 가중 우선
    out.sort(key=lambda r: (-r["_safety"], r["due_km"]))
    return out[:max_items]


def urgency_tier(due_days: int) -> int:
    """기한을 구간으로 묶는다. 0=즉시 1=1주 2=1개월 3=그 이후."""
    if due_days <= 0:
        return 0
    if due_days <= 7:
        return 1
    if due_days <= 30:
        return 2
    return 3


def prioritize(cands: list[dict], limit: int = 5) -> list[dict]:
    """권고를 정렬하고 priority 1..N 을 매긴다.

    정렬 키는 (기한 구간, 안전 가중 내림차순, 상태 점수 오름차순,
    기한 오름차순) 이다. 기한을 '일' 단위로 바로 비교하면 3일 뒤 도래하는
    소모품 교환이 7일 뒤 교체해야 하는 브레이크 패드보다 앞서는 역전이
    생긴다. 그래서 기한은 구간으로만 비교하고, 같은 구간 안에서는
    안전 영향이 큰 항목을 먼저 올린다.
    """
    cands.sort(key=lambda r: (urgency_tier(r["due_days"]), -r["_safety"],
                              r["_score"], r["due_days"]))
    out = []
    for i, r in enumerate(cands[:limit], start=1):
        out.append({"priority": i, "item": r["item"],
                    "due_km": r["due_km"], "due_days": r["due_days"]})
    return out


# 위험 등급별 기한 상한(일). 등급이 높으면 모든 권고의 기한을 당긴다.
RISK_DUE_CEILING = {"CRITICAL": 1, "HIGH": 7, "MEDIUM": 14, "LOW": 365}


def build_recommendations(components: list[dict], odometer_km: int,
                          risk_level_value: str, daily_km: float = 55.0,
                          limit: int = 5) -> list[dict]:
    """부품 권고 + 주기 권고를 합쳐 최종 recommendations 배열을 만든다."""
    cands = part_recommendations(components, daily_km)
    cands += periodic_recommendations(odometer_km, daily_km)
    if risk_level_value == "CRITICAL":
        # CRITICAL 차량은 '즉시 입고' 를 최상단에 고정한다.
        cands.append({"item": "즉시 입고 정밀 진단(운행 자제 권고)",
                      "due_km": 0, "due_days": 0,
                      "_score": 0.0, "_safety": 3,
                      "_part": "service", "_status": "CRITICAL"})
    if not cands:
        # 모든 부품이 NORMAL 이고 도래한 주기 항목도 없는 경우
        cands.append({"item": "이상 없음 — 다음 정기 점검 주기 유지",
                      "due_km": 5000, "due_days": 90,
                      "_score": 100.0, "_safety": 1,
                      "_part": "none", "_status": "NORMAL"})

    # 등급 상한으로 기한을 당긴다. due_km 도 함께 줄여 모순을 없앤다.
    ceil_days = RISK_DUE_CEILING.get(risk_level_value, 365)
    for r in cands:
        if r["due_days"] > ceil_days:
            r["due_days"] = ceil_days
            r["due_km"] = min(r["due_km"], max(0, int(ceil_days * daily_km)))
    return prioritize(cands, limit)


# 위험 등급별 조치 매트릭스. 대시보드 알림 정책과 1:1 대응된다.
ACTION_MATRIX = {
    "LOW":      ("정기 점검 주기 유지", "월간 리포트", "없음"),
    "MEDIUM":   ("2주 내 점검 예약 안내", "앱 푸시 1회", "담당자 확인"),
    "HIGH":     ("7일 내 입고 예약", "앱 푸시 + SMS", "관리자 즉시 통보"),
    "CRITICAL": ("즉시 입고, 장거리 운행 금지", "전화 + SMS + 푸시", "관리자 승인 필요"),
}


def main() -> None:
    print("[위험 등급별 조치 매트릭스]")
    print(f"{'등급':<10s}{'조치':<28s}{'알림 채널':<20s}{'승인'}")
    for lvl, (act, ch, apr) in ACTION_MATRIX.items():
        print(f"{lvl:<10s}{act:<28s}{ch:<20s}{apr}")

    odo = {"정상 차량": 42_300, "브레이크 마모": 79_850,
           "엔진 과열": 118_400, "공기압 이상": 19_600,
           "EV 배터리 열화": 96_200}

    for title, pt, anom, fr, reading in HS_SAMPLES:
        out = predict(reading, anom, fr, powertrain=pt, vehicle_id=1)
        recs = build_recommendations(out["components"], odo[title],
                                     out["risk_level"])
        print(f"\n=== {title} (누적 {odo[title]:,}km, "
              f"건강점수 {out['health_score']:.1f}, {out['risk_level']}) ===")
        print(f"{'순위':>4s}  {'항목':<42s}{'due_km':>8s}{'due_days':>9s}")
        for r in recs:
            print(f"{r['priority']:4d}  {r['item']:<42s}"
                  f"{r['due_km']:8d}{r['due_days']:9d}")
        act = ACTION_MATRIX[out["risk_level"]]
        print(f"  -> 조치: {act[0]} / 알림: {act[1]}")

    print("\n[recommendations 배열 — 브레이크 마모 차량, CONTRACT 스키마]")
    out = predict(HS_SAMPLES[1][4], 0.38, 0.27, powertrain="ICE")
    for r in build_recommendations(out["components"], 79_850, out["risk_level"]):
        print(" ", r)


if __name__ == "__main__":
    main()
