# -*- coding: utf-8 -*-
"""부품별 상태 판정 규칙과 점수화 — 엔진/배터리/브레이크/타이어/냉각/전기차배터리.

AI 모델의 출력만으로는 "왜 그런 판정인가" 를 설명할 수 없다. 그래서
실무 시스템은 모델 점수와 함께 **규칙 기반 부품 점수**를 반드시 둔다.
규칙은 정비 매뉴얼의 한계치에서 바로 끌어오므로 근거를 제시할 수 있고,
센서가 몇 개 빠져도 나머지로 판정할 수 있다.

각 부품 점수는 0~100 (100=신품)이며 CONTRACT.md 의
components[].{name, score, status, reason} 형태로 바로 변환된다.

이 파일은 numpy/파이썬 표준 라이브러리만 사용한다.
실행:  python 09_parts/part_rules.py
"""
from __future__ import annotations

from dataclasses import dataclass, field

# 상태 등급 경계(부품 점수 기준)
STATUS_BANDS = ((85.0, "NORMAL"), (70.0, "WATCH"), (50.0, "WARNING"), (0.0, "CRITICAL"))

# CONTRACT.md 의 components[].name 값과 1:1 대응되는 부품 코드
PART_NAMES = ("engine", "battery", "brake", "tire", "cooling", "ev_battery")


def band(score: float) -> str:
    """점수를 NORMAL/WATCH/WARNING/CRITICAL 등급으로 바꾼다."""
    for lo, name in STATUS_BANDS:
        if score >= lo:
            return name
    return "CRITICAL"


def linear_score(value: float, good: float, bad: float) -> float:
    """good 에서 100점, bad 에서 0점이 되도록 선형 보간한다.

    good > bad 인 경우(값이 클수록 좋음)와 good < bad 인 경우(값이
    작을수록 좋음)를 모두 처리한다. 범위를 벗어나면 0 또는 100 으로 자른다.
    """
    if good == bad:
        return 100.0
    r = (value - bad) / (good - bad)
    return float(max(0.0, min(1.0, r)) * 100.0)


@dataclass
class Finding:
    """하나의 판정 근거. 점수와 사람이 읽을 수 있는 이유를 함께 담는다."""
    key: str                # 센서/지표 이름
    value: float            # 관측값
    score: float            # 0~100
    reason: str             # 한국어 설명


@dataclass
class PartResult:
    """부품 단위 판정 결과."""
    name: str
    score: float
    status: str
    reason: str
    findings: list[Finding] = field(default_factory=list)

    def to_component(self) -> dict:
        """CONTRACT.md components[] 항목 형태로 변환한다."""
        return {"name": self.name, "score": round(self.score, 1),
                "status": self.status, "reason": self.reason}


def _combine(name: str, findings: list[Finding],
             weights: dict[str, float]) -> PartResult:
    """지표별 점수를 가중평균하되, 최저 점수에 하한을 씌운다.

    가중평균만 쓰면 한 지표가 치명적으로 나빠도 다른 지표가 좋아서
    전체 점수가 평탄해진다. 안전 부품에서는 이것이 위험하므로
    '최저 점수 + 20' 을 상한으로 두어 최악 지표를 반영한다.
    """
    if not findings:
        return PartResult(name, 100.0, "NORMAL", "판정 가능한 센서 데이터가 없다")
    tw = sum(weights.get(f.key, 1.0) for f in findings)
    avg = sum(f.score * weights.get(f.key, 1.0) for f in findings) / tw
    worst = min(findings, key=lambda f: f.score)
    score = min(avg, worst.score + 20.0)
    score = float(max(0.0, min(100.0, score)))
    status = band(score)
    # 종합 등급이 NORMAL 인데 개별 지표 문구만 경고조로 남는 것을 막는다.
    if status == "NORMAL" and worst.score < 85.0:
        reason = f"{worst.key} 지표가 다소 낮으나 종합 상태는 정상 범위"
    else:
        reason = worst.reason
    return PartResult(name, score, status, reason, findings)


# 동력계통별로 평가할 부품 목록. 내연기관 차량에 ev_battery 를 돌리거나
# 전기차에 engine 규칙을 돌리면 의미 없는 판정이 나온다.
PARTS_BY_POWERTRAIN = {
    "ICE": ("engine", "battery", "brake", "tire", "cooling"),
    "EV": ("battery", "brake", "tire", "cooling", "ev_battery"),
    "HEV": ("engine", "battery", "brake", "tire", "cooling", "ev_battery"),
}


# ------------------------------------------------------------------ 엔진
def engine_rules(r: dict) -> PartResult:
    """엔진: 냉각수 온도, 진동 RMS, 엔진오일 품질, 공회전 RPM."""
    f: list[Finding] = []
    if (v := r.get("engine_temp_c")) is not None:
        s = linear_score(v, good=92.0, bad=115.0)
        f.append(Finding("engine_temp_c", v, s,
                         f"냉각수 온도 {v:.1f}℃ "
                         + ("정상 범위" if s >= 85 else "상승 추세 — 과열 점검 필요")))
    if (v := r.get("vibration_rms")) is not None:
        s = linear_score(v, good=0.35, bad=1.10)
        f.append(Finding("vibration_rms", v, s,
                         f"진동 RMS {v:.2f}g "
                         + ("정상" if s >= 85 else "기준 초과 — 마운트·베어링 점검")))
    if (v := r.get("oil_quality")) is not None:
        s = linear_score(v, good=0.85, bad=0.20)
        f.append(Finding("oil_quality", v, s,
                         f"엔진오일 품질지수 {v:.2f} "
                         + ("양호" if s >= 85 else "열화 — 오일 교환 권고")))
    if (v := r.get("engine_rpm")) is not None:
        # 공회전 기준에서 벗어난 정도를 양방향으로 평가한다.
        dev = abs(v - 1800.0)
        s = linear_score(dev, good=150.0, bad=700.0)
        f.append(Finding("engine_rpm", v, s,
                         f"엔진 회전수 {v:.0f}rpm "
                         + ("안정" if s >= 85 else "불안정 — 흡기·점화 계통 점검")))
    return _combine("engine", f, {"engine_temp_c": 1.4, "vibration_rms": 1.3,
                                  "oil_quality": 1.0, "engine_rpm": 0.7})


# ---------------------------------------------------------------- 배터리
def battery_rules(r: dict) -> PartResult:
    """12V 납축전지: 단자 전압과 SOH."""
    f: list[Finding] = []
    if (v := r.get("battery_voltage")) is not None:
        s = linear_score(v, good=12.6, bad=11.6)
        f.append(Finding("battery_voltage", v, s,
                         f"배터리 전압 {v:.2f}V "
                         + ("정상" if s >= 85 else "저하 — 충전계통·수명 점검")))
    if (v := r.get("battery_soh")) is not None:
        s = linear_score(v, good=0.95, bad=0.65)
        f.append(Finding("battery_soh", v, s,
                         f"배터리 SOH {v * 100:.0f}% "
                         + ("양호" if s >= 85 else "교체 시점 접근")))
    return _combine("battery", f, {"battery_voltage": 1.0, "battery_soh": 1.5})


# -------------------------------------------------------------- 브레이크
def brake_rules(r: dict) -> PartResult:
    """브레이크: 패드 잔량(법정 최소 2mm)과 제동 시 진동."""
    f: list[Finding] = []
    if (v := r.get("brake_pad_mm")) is not None:
        s = linear_score(v, good=8.0, bad=2.0)
        f.append(Finding("brake_pad_mm", v, s,
                         f"브레이크 패드 잔량 {v:.1f}mm "
                         + ("충분" if s >= 85 else
                            ("교체 임박 — 2mm 가 법정 한계" if v < 3.5
                             else "마모 진행 중"))))
    if (v := r.get("vibration_rms")) is not None:
        s = linear_score(v, good=0.35, bad=0.95)
        f.append(Finding("vibration_rms", v, s,
                         f"제동 진동 {v:.2f}g "
                         + ("정상" if s >= 85 else "디스크 변형 의심")))
    return _combine("brake", f, {"brake_pad_mm": 2.0, "vibration_rms": 0.6})


# ---------------------------------------------------------------- 타이어
def tire_rules(r: dict) -> PartResult:
    """타이어: 4륜 공기압의 절대값과 좌우 편차."""
    keys = ("tire_pressure_fl", "tire_pressure_fr",
            "tire_pressure_rl", "tire_pressure_rr")
    vals = [r[k] for k in keys if r.get(k) is not None]
    f: list[Finding] = []
    if vals:
        lo = min(vals)
        s = linear_score(lo, good=33.0, bad=18.0)
        f.append(Finding("pressure_min", lo, s,
                         f"최저 공기압 {lo:.1f}psi "
                         + ("정상" if s >= 85 else "부족 — 보충 또는 펑크 점검")))
        imb = max(vals) - lo
        s2 = linear_score(imb, good=1.5, bad=10.0)
        f.append(Finding("pressure_imbalance", imb, s2,
                         f"4륜 공기압 편차 {imb:.1f}psi "
                         + ("균일" if s2 >= 85 else "불균형 — 편마모·조향 쏠림 유발")))
    return _combine("tire", f, {"pressure_min": 1.5, "pressure_imbalance": 1.0})


# ------------------------------------------------------------ 냉각 시스템
def cooling_rules(r: dict) -> PartResult:
    """냉각 시스템: 온도 절대값 + 부하 대비 온도(열방출 능력)."""
    f: list[Finding] = []
    temp = r.get("engine_temp_c")
    if temp is not None:
        s = linear_score(temp, good=90.0, bad=112.0)
        f.append(Finding("engine_temp_c", temp, s,
                         f"냉각수 온도 {temp:.1f}℃ "
                         + ("정상" if s >= 85 else "과열 경향")))
    rpm, spd = r.get("engine_rpm"), r.get("speed_kph")
    if temp is not None and rpm is not None and spd is not None:
        # 같은 부하에서 온도가 높을수록 라디에이터·워터펌프 성능 저하를 의심한다.
        load = max(rpm / 1800.0, 0.3)
        idx = (temp - 88.0) / load
        s = linear_score(idx, good=3.0, bad=22.0)
        f.append(Finding("thermal_load_idx", idx, s,
                         f"부하 보정 온도지수 {idx:.1f} "
                         + ("정상" if s >= 85 else
                            "열방출 저하 — 냉각수량·라디에이터·써모스탯 점검")))
    return _combine("cooling", f, {"engine_temp_c": 1.2, "thermal_load_idx": 1.5})


# -------------------------------------------------------- 전기차 배터리
def ev_battery_rules(r: dict) -> PartResult:
    """전기차 구동 배터리: SOH, 셀 전압 편차, 최고 셀 온도."""
    f: list[Finding] = []
    if (v := r.get("battery_soh")) is not None:
        # 전기차는 SOH 80% 를 보증·교체 기준으로 쓰는 경우가 많다.
        s = linear_score(v, good=0.95, bad=0.75)
        f.append(Finding("battery_soh", v, s,
                         f"구동 배터리 SOH {v * 100:.1f}% "
                         + ("양호" if s >= 85 else "용량 감소 — 보증 기준 80% 확인")))
    if (v := r.get("cell_volt_delta")) is not None:
        s = linear_score(v, good=0.02, bad=0.12)
        f.append(Finding("cell_volt_delta", v, s,
                         f"셀 전압 편차 {v * 1000:.0f}mV "
                         + ("균형" if s >= 85 else "불균형 — BMS 밸런싱·셀 불량 점검")))
    if (v := r.get("pack_temp_max_c")) is not None:
        s = linear_score(v, good=35.0, bad=58.0)
        f.append(Finding("pack_temp_max_c", v, s,
                         f"최고 셀 온도 {v:.1f}℃ "
                         + ("정상" if s >= 85 else "열관리 점검 필요")))
    return _combine("ev_battery", f, {"battery_soh": 1.6, "cell_volt_delta": 1.3,
                                      "pack_temp_max_c": 1.0})


RULES = {
    "engine": engine_rules, "battery": battery_rules, "brake": brake_rules,
    "tire": tire_rules, "cooling": cooling_rules, "ev_battery": ev_battery_rules,
}


def evaluate_parts(reading: dict, parts: tuple[str, ...] | None = None,
                   powertrain: str = "ICE") -> list[PartResult]:
    """센서 한 행에 대해 요청된 부품 전부를 판정한다.

    parts 를 주지 않으면 powertrain(ICE/EV/HEV)에 맞는 부품 집합을 쓴다.
    """
    if parts is None:
        parts = PARTS_BY_POWERTRAIN.get(powertrain, PART_NAMES)
    return [RULES[p](reading) for p in parts if p in RULES]


# ------------------------------------------------------------------ 데모
SAMPLES = {
    "정상 차량": {
        "engine_temp_c": 91.2, "engine_rpm": 1820, "speed_kph": 52.0,
        "battery_voltage": 12.58, "battery_soh": 0.94, "brake_pad_mm": 7.8,
        "oil_quality": 0.88, "vibration_rms": 0.32,
        "tire_pressure_fl": 32.8, "tire_pressure_fr": 32.6,
        "tire_pressure_rl": 32.4, "tire_pressure_rr": 32.5,
        "cell_volt_delta": 0.018, "pack_temp_max_c": 31.0,
    },
    "브레이크 마모": {
        "engine_temp_c": 92.8, "engine_rpm": 1790, "speed_kph": 48.0,
        "battery_voltage": 12.44, "battery_soh": 0.89, "brake_pad_mm": 2.6,
        "oil_quality": 0.71, "vibration_rms": 0.61,
        "tire_pressure_fl": 31.9, "tire_pressure_fr": 31.7,
        "tire_pressure_rl": 31.2, "tire_pressure_rr": 31.5,
        "cell_volt_delta": 0.024, "pack_temp_max_c": 33.5,
    },
    "엔진 과열": {
        "engine_temp_c": 109.4, "engine_rpm": 2150, "speed_kph": 44.0,
        "battery_voltage": 12.30, "battery_soh": 0.84, "brake_pad_mm": 6.2,
        "oil_quality": 0.38, "vibration_rms": 0.74,
        "tire_pressure_fl": 32.0, "tire_pressure_fr": 31.8,
        "tire_pressure_rl": 31.6, "tire_pressure_rr": 31.9,
        "cell_volt_delta": 0.031, "pack_temp_max_c": 41.0,
    },
    "타이어 공기압 이상": {
        "engine_temp_c": 90.6, "engine_rpm": 1805, "speed_kph": 50.0,
        "battery_voltage": 12.52, "battery_soh": 0.91, "brake_pad_mm": 6.9,
        "oil_quality": 0.80, "vibration_rms": 0.48,
        "tire_pressure_fl": 32.5, "tire_pressure_fr": 32.2,
        "tire_pressure_rl": 23.1, "tire_pressure_rr": 30.8,
        "cell_volt_delta": 0.020, "pack_temp_max_c": 32.0,
    },
    "EV 배터리 열화": {
        "engine_temp_c": 38.0, "engine_rpm": 0, "speed_kph": 46.0,
        "battery_voltage": 12.41, "battery_soh": 0.781, "brake_pad_mm": 7.1,
        "oil_quality": 0.92, "vibration_rms": 0.29,
        "tire_pressure_fl": 33.0, "tire_pressure_fr": 32.9,
        "tire_pressure_rl": 32.7, "tire_pressure_rr": 32.8,
        "cell_volt_delta": 0.094, "pack_temp_max_c": 52.4,
    },
}


POWERTRAIN = {"정상 차량": "ICE", "브레이크 마모": "ICE", "엔진 과열": "ICE",
              "타이어 공기압 이상": "ICE", "EV 배터리 열화": "EV"}


def main() -> None:
    for title, reading in SAMPLES.items():
        pt = POWERTRAIN[title]
        print(f"=== {title} (동력계통 {pt}) ===")
        results = evaluate_parts(reading, powertrain=pt)
        print(f"{'부품':<12s}{'점수':>7s}{'상태':>10s}  근거")
        for pr in results:
            print(f"{pr.name:<12s}{pr.score:7.1f}{pr.status:>10s}  {pr.reason}")
        worst = min(results, key=lambda p: p.score)
        print(f"-> top_component = {worst.name} (점수 {worst.score:.1f})\n")

    print("[components 배열 변환 예 — EV 배터리 열화 차량]")
    for p in evaluate_parts(SAMPLES["EV 배터리 열화"], powertrain="EV"):
        print(" ", p.to_component())


if __name__ == "__main__":
    main()
