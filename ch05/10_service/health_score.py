# -*- coding: utf-8 -*-
"""차량 건강점수 산출 — 부품 점수 가중합 + 이상 점수 + 고장 확률 반영.

CONTRACT.md 의 /api/v1/pdm/predict 응답 중
  health_score, anomaly_score, failure_risk, risk_level, rul_days, top_component
를 하나의 함수로 계산한다. 세 가지 서로 다른 근거를 결합한다.

  1) 부품 규칙 점수  : 09_parts/part_rules.py (설명 가능, 안전 하한)
  2) 이상 점수       : 05_anomaly 의 이상 탐지기 출력 0~1 (미지의 패턴 포착)
  3) 고장 확률       : 06_failure 의 분류 모델 출력 0~1 (학습된 패턴)

이 파일은 numpy/표준 라이브러리만 사용한다.
실행:  python 10_service/health_score.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# 같은 장의 09_parts 모듈을 재사용한다(중복 구현하지 않는다).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "09_parts"))
from part_rules import evaluate_parts, PartResult   # noqa: E402

# 부품별 건강점수 가중치. 안전 직결 부품에 더 큰 가중치를 둔다.
# 합이 1.0 이 되도록 정규화하며, 평가되지 않은 부품은 제외 후 재정규화한다.
PART_WEIGHTS = {
    "brake": 0.26,        # 제동 실패는 즉시 인명 위험
    "engine": 0.22,       # 수리비가 가장 크고 2차 손상을 유발
    "ev_battery": 0.18,   # 전기차 최고가 부품
    "tire": 0.15,         # 안전 직결이나 교체가 저렴·신속
    "cooling": 0.12,      # 엔진 손상의 선행 원인
    "battery": 0.07,      # 출발 불능이지만 수리비가 작다
}

# 위험도 등급 경계(failure_risk 기준). CONTRACT.md 의 risk_level enum 과 일치.
RISK_BANDS = ((0.70, "CRITICAL"), (0.45, "HIGH"), (0.20, "MEDIUM"), (0.0, "LOW"))

# 등급 산정에 건강점수도 함께 쓴다(확률이 낮아도 점수가 매우 낮으면 승급).
HEALTH_OVERRIDE = ((25.0, "CRITICAL"), (45.0, "HIGH"), (65.0, "MEDIUM"))
LEVEL_ORDER = ("LOW", "MEDIUM", "HIGH", "CRITICAL")


def risk_level(failure_risk: float, health: float) -> str:
    """고장 확률과 건강점수를 함께 보고 위험 등급을 정한다.

    확률 기준 등급과 건강점수 기준 등급을 각각 구한 뒤 **더 높은 쪽**을
    택한다. 모델 확률이 낮게 나와도 규칙 점수가 치명적이면 등급을
    올려야 하고, 반대로 규칙은 통과했지만 모델이 고장 전조를 잡아낸
    경우에도 등급을 올려야 한다. 둘 중 하나라도 위험하면 위험이다.
    """
    by_risk = "LOW"
    for lo, name in RISK_BANDS:
        if failure_risk >= lo:
            by_risk = name
            break
    by_health = "LOW"
    for thr, name in HEALTH_OVERRIDE:
        if health < thr:
            by_health = name
            break
    return max(by_risk, by_health, key=LEVEL_ORDER.index)


def weighted_part_score(parts: list[PartResult]) -> tuple[float, dict]:
    """부품 점수를 가중평균한다. 평가 대상만 쓰고 가중치를 재정규화한다."""
    used = {p.name: PART_WEIGHTS.get(p.name, 0.05) for p in parts}
    tw = sum(used.values())
    base = sum(p.score * used[p.name] for p in parts) / tw
    return float(base), {k: round(v / tw, 4) for k, v in used.items()}


def health_score(parts: list[PartResult], anomaly_score: float,
                 failure_risk: float) -> float:
    """최종 건강점수 0~100 을 계산한다.

    계산 순서가 중요하다.
      1) 부품 가중합을 구한다.
      2) '최악 부품 점수 + 25' 를 상한으로 걸어, 치명 부품 하나가 있으면
         나머지가 좋아도 전체 점수가 올라가지 않게 한다(안전 하한).
      3) 그 결과에서 벌점을 뺀다. 상한을 씌운 뒤에 빼야 이상 점수와
         고장 확률이 항상 점수에 반영된다.
         - 이상 점수 벌점 : 규칙에 없는 이상 패턴. 최대 12점 차감
         - 고장 확률 벌점 : 학습 모델이 본 고장 전조. 최대 18점 차감
    """
    base, _ = weighted_part_score(parts)
    if parts:
        base = min(base, min(p.score for p in parts) + 25.0)
    penalty = 12.0 * max(0.0, min(1.0, anomaly_score)) \
        + 18.0 * max(0.0, min(1.0, failure_risk))
    return float(max(0.0, min(100.0, round(base - penalty, 1))))


def rul_from_health(health: float, parts: list[PartResult],
                    daily_km: float = 55.0) -> int:
    """건강점수와 최악 부품 점수로 보수적인 RUL(일)을 추정한다.

    정식 RUL 모델(08_rul)이 있으면 그 값을 쓰고, 없을 때의 대체 추정이다.
    점수 100 에서 365일, 점수 0 에서 0일이 되도록 비선형 사상을 쓰고
    최악 부품 기준으로 한 번 더 깎는다.
    """
    worst = min((p.score for p in parts), default=health)
    eff = min(health, worst + 5.0)
    days = 365.0 * (max(0.0, eff) / 100.0) ** 1.9
    return int(max(0, round(days)))


def predict(reading: dict, anomaly_score: float, failure_risk: float,
            powertrain: str = "ICE", vehicle_id: int = 1,
            model_version: str = "pdm-ensemble-1.0.0") -> dict:
    """CONTRACT.md 응답 스키마의 핵심 필드를 산출한다.

    recommendations 는 10_service/recommend.py 가 채운다.
    """
    parts = evaluate_parts(reading, powertrain=powertrain)
    hs = health_score(parts, anomaly_score, failure_risk)
    worst = min(parts, key=lambda p: p.score)
    return {
        "vehicle_id": vehicle_id,
        "health_score": hs,
        "anomaly_score": round(float(anomaly_score), 3),
        "failure_risk": round(float(failure_risk), 3),
        "risk_level": risk_level(failure_risk, hs),
        "rul_days": rul_from_health(hs, parts),
        "top_component": worst.name,
        "components": [p.to_component() for p in parts],
        "recommendations": [],
        "model_version": model_version,
    }


SAMPLES = [
    ("정상 차량", "ICE", 0.08, 0.05, {
        "engine_temp_c": 91.2, "engine_rpm": 1820, "speed_kph": 52.0,
        "battery_voltage": 12.58, "battery_soh": 0.94, "brake_pad_mm": 7.8,
        "oil_quality": 0.88, "vibration_rms": 0.32,
        "tire_pressure_fl": 32.8, "tire_pressure_fr": 32.6,
        "tire_pressure_rl": 32.4, "tire_pressure_rr": 32.5}),
    ("브레이크 마모", "ICE", 0.38, 0.27, {
        "engine_temp_c": 92.8, "engine_rpm": 1790, "speed_kph": 48.0,
        "battery_voltage": 12.44, "battery_soh": 0.89, "brake_pad_mm": 2.6,
        "oil_quality": 0.71, "vibration_rms": 0.61,
        "tire_pressure_fl": 31.9, "tire_pressure_fr": 31.7,
        "tire_pressure_rl": 31.2, "tire_pressure_rr": 31.5}),
    ("엔진 과열", "ICE", 0.81, 0.74, {
        "engine_temp_c": 109.4, "engine_rpm": 2150, "speed_kph": 44.0,
        "battery_voltage": 12.30, "battery_soh": 0.84, "brake_pad_mm": 6.2,
        "oil_quality": 0.38, "vibration_rms": 0.74,
        "tire_pressure_fl": 32.0, "tire_pressure_fr": 31.8,
        "tire_pressure_rl": 31.6, "tire_pressure_rr": 31.9}),
    ("공기압 이상", "ICE", 0.52, 0.19, {
        "engine_temp_c": 90.6, "engine_rpm": 1805, "speed_kph": 50.0,
        "battery_voltage": 12.52, "battery_soh": 0.91, "brake_pad_mm": 6.9,
        "oil_quality": 0.80, "vibration_rms": 0.48,
        "tire_pressure_fl": 32.5, "tire_pressure_fr": 32.2,
        "tire_pressure_rl": 23.1, "tire_pressure_rr": 30.8}),
    ("EV 배터리 열화", "EV", 0.44, 0.33, {
        "engine_temp_c": 38.0, "engine_rpm": 0, "speed_kph": 46.0,
        "battery_voltage": 12.41, "battery_soh": 0.781, "brake_pad_mm": 7.1,
        "oil_quality": 0.92, "vibration_rms": 0.29,
        "tire_pressure_fl": 33.0, "tire_pressure_fr": 32.9,
        "tire_pressure_rl": 32.7, "tire_pressure_rr": 32.8,
        "cell_volt_delta": 0.094, "pack_temp_max_c": 52.4}),
]


def main() -> None:
    print("[부품별 건강점수 가중치]")
    tw = sum(PART_WEIGHTS.values())
    for k, v in PART_WEIGHTS.items():
        print(f"  {k:<11s}{v:6.2f}  ({v / tw * 100:5.1f}%)")
    print(f"  {'합계':<11s}{tw:6.2f}\n")

    hdr = (f"{'차량 상태':<16s}{'health':>8s}{'anom':>7s}{'risk':>7s}"
           f"{'등급':>10s}{'RUL일':>7s}  top_component")
    print(hdr)
    print("-" * len(hdr))
    for title, pt, anom, fr, reading in SAMPLES:
        out = predict(reading, anom, fr, powertrain=pt, vehicle_id=1)
        print(f"{title:<16s}{out['health_score']:8.1f}{out['anomaly_score']:7.2f}"
              f"{out['failure_risk']:7.2f}{out['risk_level']:>10s}"
              f"{out['rul_days']:7d}  {out['top_component']}")

    print("\n[엔진 과열 차량의 전체 응답 — recommendations 제외]")
    out = predict(SAMPLES[2][4], 0.81, 0.74, powertrain="ICE", vehicle_id=7)
    for k, v in out.items():
        if k == "components":
            print(f"  {k}:")
            for c in v:
                print(f"    {c}")
        elif k != "recommendations":
            print(f"  {k}: {v}")

    print("\n[이상 점수·고장 확률이 건강점수에 미치는 영향 — 브레이크 마모 차량]")
    print(f"{'anomaly':>9s}{'failure_risk':>14s}{'health_score':>14s}{'risk_level':>12s}")
    for anom, fr in ((0.0, 0.0), (0.2, 0.1), (0.4, 0.3), (0.6, 0.5), (0.9, 0.8)):
        o = predict(SAMPLES[1][4], anom, fr, powertrain="ICE")
        print(f"{anom:9.2f}{fr:14.2f}{o['health_score']:14.1f}{o['risk_level']:>12s}")


if __name__ == "__main__":
    main()
