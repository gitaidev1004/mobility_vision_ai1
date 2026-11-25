# -*- coding: utf-8 -*-
"""자동차 예지보전 파이프라인 골격.

1.3절의 단계를 함수로 분리한 의사코드이다. 외부 모델 파일 없이 동작하며,
출력 스키마는 AI 서비스의 POST /api/v1/pdm/predict 응답과 동일하다.
실행:  python 03_pipeline/pdm_skeleton.py
"""
from __future__ import annotations

import statistics
from datetime import datetime, timedelta

# 부품별 사용 한계 기준. RUL 을 계산할 때의 기준점이며 반드시 함께 보고한다.
LIMITS = {
    "brake_pad_mm": 3.0,      # 이 두께 미만이면 교체
    "battery_soh": 0.70,      # 이 비율 미만이면 교체
    "oil_quality": 0.30,      # 이 지표 미만이면 교환
}

# 위험 등급 경계. failure_risk 값을 네 등급으로 변환한다.
RISK_BANDS = [(0.20, "LOW"), (0.45, "MEDIUM"), (0.70, "HIGH")]


def make_sample_readings(n: int = 12) -> list[dict]:
    """시뮬레이터 대신 쓰는 예시 시계열. 브레이크 패드가 점진적으로 마모된다."""
    base = datetime(2026, 9, 1, 9, 0, 0)
    rows: list[dict] = []
    for i in range(n):
        rows.append({
            "measured_at": (base + timedelta(days=i * 7)).isoformat(),
            "odometer_km": 82000 + i * 180,
            "engine_temp_c": 91.0 + (i % 3) * 0.8,
            "engine_rpm": 1800 + (i % 4) * 40,
            "speed_kph": 46.0 + (i % 5),
            "battery_voltage": 12.5 - i * 0.01,
            "battery_soh": 0.90 - i * 0.004,
            "tire_pressure_fl": 32.0 - i * 0.05,
            "tire_pressure_fr": 31.9,
            "tire_pressure_rl": 29.4,
            "tire_pressure_rr": 31.2,
            "brake_pad_mm": 4.52 - i * 0.09,
            "oil_quality": 0.60 - i * 0.016,
            "vibration_rms": 0.40 + (i % 3) * 0.02,
            "fuel_level": 0.55,
        })
    return rows


def step3_clean(readings: list[dict]) -> list[dict]:
    """3단계: 전처리. 물리적으로 불가능한 값만 제거하고 시간순으로 정렬한다."""
    ok: list[dict] = []
    for r in readings:
        if not (0.0 <= float(r.get("battery_soh", 0.0)) <= 1.0):
            continue
        if float(r.get("engine_temp_c", 0.0)) > 200.0:
            continue
        ok.append(r)
    return sorted(ok, key=lambda x: x["measured_at"])


def step4_features(readings: list[dict]) -> dict:
    """4단계: 특징 생성. 최근값, 평균, 거리당 변화율, 타이어 편차를 만든다."""
    if not readings:
        return {}
    last = readings[-1]
    first = readings[0]
    dist = max(1.0, float(last["odometer_km"]) - float(first["odometer_km"]))

    def slope_per_1000km(key: str) -> float:
        """1000km 당 변화량. 음수면 감소 추세이다."""
        delta = float(last[key]) - float(first[key])
        return delta / dist * 1000.0

    tires = [float(last[k]) for k in
             ("tire_pressure_fl", "tire_pressure_fr",
              "tire_pressure_rl", "tire_pressure_rr")]
    return {
        "brake_pad_mm": float(last["brake_pad_mm"]),
        "brake_slope": slope_per_1000km("brake_pad_mm"),
        "battery_soh": float(last["battery_soh"]),
        "battery_slope": slope_per_1000km("battery_soh"),
        "oil_quality": float(last["oil_quality"]),
        "engine_temp_mean": statistics.fmean(
            float(r["engine_temp_c"]) for r in readings),
        "vibration_mean": statistics.fmean(
            float(r["vibration_rms"]) for r in readings),
        "tire_spread": max(tires) - min(tires),
        "odometer_km": float(last["odometer_km"]),
        "span_km": dist,
    }


def step5_anomaly(f: dict) -> float:
    """5단계: 이상 점수. 5장에서 IsolationForest 등으로 교체된다.

    지금은 각 지표의 정상 범위 이탈 정도를 평균한 규칙 기반 점수를 쓴다.
    """
    parts = [
        max(0.0, (f["engine_temp_mean"] - 100.0) / 20.0),
        max(0.0, (f["vibration_mean"] - 0.6) / 0.4),
        max(0.0, (f["tire_spread"] - 2.0) / 3.0),
        max(0.0, (0.80 - f["battery_soh"]) / 0.20),
    ]
    return round(min(1.0, statistics.fmean(parts)), 3)


def step6_failure_risk(f: dict, anomaly: float) -> float:
    """6단계: 고장 위험도. 5장에서 분류 모델로 교체된다."""
    wear = max(0.0, (LIMITS["brake_pad_mm"] + 2.0 - f["brake_pad_mm"]) / 4.0)
    oil = max(0.0, (LIMITS["oil_quality"] + 0.2 - f["oil_quality"]) / 0.5)
    risk = 0.5 * wear + 0.2 * oil + 0.3 * anomaly
    return round(min(1.0, max(0.0, risk)), 3)


def step7_rul_days(f: dict, avg_km_per_day: float) -> tuple[int, str]:
    """7단계: 잔존 수명. 가장 먼저 한계에 도달하는 부품을 찾는다.

    반환값은 (일수, 해당 부품명) 이다. 추세가 없으면 상한 365 로 제한한다.
    """
    # (현재값 키, 추세 키, 부품명) 순서로 후보를 정의한다.
    targets = (
        ("brake_pad_mm", "brake_slope", "brake"),
        ("battery_soh", "battery_slope", "battery"),
    )
    candidates: list[tuple[float, str]] = []
    for key, slope_key, comp in targets:
        slope = f[slope_key]
        if slope >= 0:                     # 감소 추세가 아니면 예측하지 않는다
            continue
        remain = f[key] - LIMITS[key]
        if remain <= 0:
            candidates.append((0.0, comp))
            continue
        km_left = remain / abs(slope) * 1000.0
        candidates.append((km_left / max(1.0, avg_km_per_day), comp))

    if not candidates:
        return 365, "none"
    days, comp = min(candidates, key=lambda x: x[0])
    return int(min(365.0, max(0.0, days))), comp


def to_risk_level(risk: float) -> str:
    """고장 위험도를 네 등급으로 변환한다."""
    for bound, label in RISK_BANDS:
        if risk < bound:
            return label
    return "CRITICAL"


def step8_recommend(f: dict, top: str) -> list[dict]:
    """8단계: 정비 권고. 우선순위를 매겨 최대 3건만 제시한다."""
    recs: list[dict] = []
    if f["brake_pad_mm"] < LIMITS["brake_pad_mm"] + 1.5:
        recs.append({"priority": 1, "item": "브레이크 패드 점검",
                     "due_km": 3000, "due_days": 30})
    if f["oil_quality"] < LIMITS["oil_quality"] + 0.2:
        recs.append({"priority": 2, "item": "엔진오일 교환",
                     "due_km": 5000, "due_days": 45})
    if f["tire_spread"] > 2.0:
        recs.append({"priority": 3, "item": "타이어 공기압 점검",
                     "due_km": 500, "due_days": 7})
    return recs[:3]


def predict(vehicle_id: int, readings: list[dict],
            avg_km_per_day: float = 26.0) -> dict:
    """AI 서비스 응답 스키마와 동일한 형태로 예측 결과를 만든다."""
    rows = step3_clean(readings)
    f = step4_features(rows)
    if not f:
        return {"vehicle_id": vehicle_id, "health_score": 0.0,
                "risk_level": "LOW", "detail": "센서 데이터 없음"}

    anomaly = step5_anomaly(f)
    risk = step6_failure_risk(f, anomaly)
    rul, top = step7_rul_days(f, avg_km_per_day)
    # 건강점수는 위험도와 이상 점수를 합쳐 0~100 으로 환산한다.
    health = round(100.0 * (1.0 - 0.7 * risk - 0.3 * anomaly), 1)
    return {
        "vehicle_id": vehicle_id,
        "health_score": max(0.0, health),
        "anomaly_score": anomaly,
        "failure_risk": risk,
        "risk_level": to_risk_level(risk),
        "rul_days": rul,
        "top_component": top,
        "recommendations": step8_recommend(f, top),
        "model_version": "pdm-skeleton-0.1.0",
    }


if __name__ == "__main__":
    out = predict(1, make_sample_readings())
    for key in ("health_score", "anomaly_score", "failure_risk",
                "risk_level", "rul_days", "top_component"):
        print(f"{key}: {out[key]}")
    print("권고:")
    for r in out["recommendations"]:
        print(f"  {r['priority']}. {r['item']} (약 {r['due_days']}일 내)")
