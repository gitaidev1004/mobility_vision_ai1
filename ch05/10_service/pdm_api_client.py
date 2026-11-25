# -*- coding: utf-8 -*-
"""/api/v1/pdm/predict 호출과 응답 파싱.

AI 서비스가 떠 있지 않아도 동작한다. 접속에 실패하면 로컬 계산
(10_service/health_score.py + recommend.py)으로 같은 스키마의 응답을
만들어 출력하므로, 스키마를 눈으로 확인하는 용도로도 쓸 수 있다.

requests 가 없으면 표준 라이브러리 urllib 로 대체한다.

실행:  python 10_service/pdm_api_client.py
       python 10_service/pdm_api_client.py http://localhost:8000
"""
from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "09_parts"))
from health_score import predict as local_predict             # noqa: E402
from recommend import ACTION_MATRIX, build_recommendations    # noqa: E402

BASE_URL = "http://localhost:8000"
ENDPOINT = "/api/v1/pdm/predict"
TIMEOUT_S = 5.0

# 응답에 반드시 있어야 하는 키. 계약 위반을 조기에 잡는다.
REQUIRED_KEYS = ("vehicle_id", "health_score", "anomaly_score", "failure_risk",
                 "risk_level", "rul_days", "top_component", "components",
                 "recommendations", "model_version")
RISK_LEVELS = ("LOW", "MEDIUM", "HIGH", "CRITICAL")


def build_request(vehicle_id: int, odometer_km: int,
                  readings: list[dict]) -> dict:
    """CONTRACT.md 의 요청 스키마를 그대로 만든다."""
    return {"vehicle_id": vehicle_id, "odometer_km": odometer_km,
            "readings": readings}


def sample_readings(n: int = 5) -> list[dict]:
    """최근 n회 센서 측정값을 만든다. 열 이름은 sensor_reading 과 동일하다."""
    base = datetime(2026, 10, 1, 9, 0, 0)
    out = []
    for i in range(n):
        # 브레이크 패드가 서서히 닳고 진동이 커지는 시나리오
        out.append({
            "measured_at": (base + timedelta(days=i * 7)).isoformat(),
            "engine_temp_c": round(92.4 + i * 0.3, 1),
            "engine_rpm": 1850 - i * 10,
            "speed_kph": 48.0,
            "battery_voltage": round(12.44 - i * 0.01, 2),
            "battery_soh": round(0.89 - i * 0.004, 3),
            "tire_pressure_fl": round(32.1 - i * 0.05, 2),
            "tire_pressure_fr": round(31.8 - i * 0.05, 2),
            "tire_pressure_rl": round(30.9 - i * 0.05, 2),
            "tire_pressure_rr": round(31.2 - i * 0.05, 2),
            "brake_pad_mm": round(4.1 - i * 0.35, 2),
            "oil_quality": round(0.62 - i * 0.02, 3),
            "vibration_rms": round(0.44 + i * 0.04, 3),
            "fuel_level": 0.55,
        })
    return out


def call_api(payload: dict, base_url: str = BASE_URL) -> dict | None:
    """AI 서비스를 호출한다. 실패하면 None 을 돌려준다.

    requests 가 있으면 그것을 쓰고, 없으면 urllib 로 같은 요청을 보낸다.
    외부 의존성을 추가하지 않고도 동작하게 하기 위한 장치다.
    """
    url = base_url.rstrip("/") + ENDPOINT
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    try:
        import requests
        r = requests.post(url, json=payload, timeout=TIMEOUT_S)
        r.raise_for_status()
        return r.json()
    except ImportError:
        pass
    except Exception as exc:                                 # noqa: BLE001
        print(f"[경고] requests 호출 실패: {exc}")
        return None

    req = urllib.request.Request(
        url, data=body, method="POST",
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
        print(f"[경고] AI 서비스({url}) 호출 실패: {exc}")
        return None


def validate(res: dict) -> list[str]:
    """응답이 계약을 지키는지 검사한다. 통합 테스트의 기본 골격이다."""
    errs = []
    for k in REQUIRED_KEYS:
        if k not in res:
            errs.append(f"필수 키 누락: {k}")
    if "risk_level" in res and res["risk_level"] not in RISK_LEVELS:
        errs.append(f"risk_level 값 오류: {res['risk_level']}")
    if "health_score" in res and not 0.0 <= float(res["health_score"]) <= 100.0:
        errs.append(f"health_score 범위 오류: {res['health_score']}")
    for k in ("anomaly_score", "failure_risk"):
        if k in res and not 0.0 <= float(res[k]) <= 1.0:
            errs.append(f"{k} 범위 오류: {res[k]}")
    if "rul_days" in res and int(res["rul_days"]) < 0:
        errs.append(f"rul_days 음수: {res['rul_days']}")
    for c in res.get("components", []):
        for k in ("name", "score", "status", "reason"):
            if k not in c:
                errs.append(f"components 항목 키 누락: {k}")
                break
    prios = [r.get("priority") for r in res.get("recommendations", [])]
    if prios and prios != sorted(prios):
        errs.append("recommendations 의 priority 가 오름차순이 아니다")
    return errs


def fallback_predict(payload: dict) -> dict:
    """서비스가 없을 때 같은 스키마를 로컬에서 산출한다.

    AI 서비스의 `app/core/fallback.py` 와 같은 발상이다. 어떤 경우에도
    화면에 빈 값이 내려가지 않게 하는 것이 목적이다.
    """
    last = payload["readings"][-1]
    # 실제 서비스에서는 05_anomaly 와 06_failure 모델이 이 두 값을 낸다.
    anomaly, risk = 0.38, 0.27
    out = local_predict(last, anomaly, risk, powertrain="ICE",
                        vehicle_id=payload["vehicle_id"],
                        model_version="pdm-rule-fallback-1.0.0")
    out["recommendations"] = build_recommendations(
        out["components"], payload["odometer_km"], out["risk_level"])
    out["computed_at"] = datetime.now().replace(microsecond=0).isoformat()
    return out


def print_result(res: dict) -> None:
    """응답을 사람이 읽기 쉽게 출력한다."""
    print(f"차량 ID      : {res['vehicle_id']}")
    print(f"건강점수     : {res['health_score']} / 100")
    print(f"이상 점수    : {res['anomaly_score']}")
    print(f"고장 위험도  : {res['failure_risk']}")
    print(f"위험 등급    : {res['risk_level']}")
    print(f"잔존 수명    : {res['rul_days']}일")
    print(f"주의 부품    : {res['top_component']}")
    print(f"모델 버전    : {res['model_version']}")
    print(f"\n{'부품':<12s}{'점수':>7s}{'상태':>10s}  근거")
    for c in res["components"]:
        print(f"{c['name']:<12s}{c['score']:7.1f}{c['status']:>10s}  "
              f"{c['reason']}")
    print(f"\n{'순위':>4s}  {'정비 항목':<42s}{'due_km':>8s}{'due_days':>9s}")
    for r in res["recommendations"]:
        print(f"{r['priority']:4d}  {r['item']:<42s}"
              f"{r['due_km']:8d}{r['due_days']:9d}")
    act = ACTION_MATRIX.get(res["risk_level"])
    if act:
        print(f"\n조치: {act[0]} / 알림: {act[1]} / 승인: {act[2]}")


def main() -> None:
    base = sys.argv[1] if len(sys.argv) > 1 else BASE_URL
    payload = build_request(vehicle_id=1, odometer_km=84_210,
                            readings=sample_readings(5))

    print("[요청 본문 — CONTRACT.md 스키마]")
    preview = dict(payload)
    preview["readings"] = [payload["readings"][0], "... (총 "
                           f"{len(payload['readings'])}건)"]
    print(json.dumps(preview, ensure_ascii=False, indent=2)[:900])

    print(f"\nPOST {base.rstrip('/')}{ENDPOINT}")
    res = call_api(payload, base)
    if res is None:
        print("        AI 서비스에 접속할 수 없어 로컬 폴백으로 계산한다.")
        print("        (6장에서 docker compose up 으로 서비스를 띄운다)\n")
        res = fallback_predict(payload)
    else:
        print("        응답 수신 성공\n")

    errs = validate(res)
    print_result(res)
    print("\n[계약 검증]", "통과" if not errs else f"{len(errs)}건 위반")
    for e in errs:
        print("  -", e)


if __name__ == "__main__":
    main()
