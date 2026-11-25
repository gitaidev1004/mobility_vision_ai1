"""고장 분류: RandomForest / GradientBoosting (sklearn) + 규칙 기반 폴백.

이상탐지와 고장분류의 역할 차이:
    - **이상탐지**(``anomaly.py``): "뭔가 평소와 다르다" — 라벨 없이 학습(단일 클래스)
    - **고장분류**(이 모듈): "**어느 부품**이 고장날 것이다" — 라벨 필요(지도학습)

    이상탐지만으로는 정비사에게 지시를 줄 수 없다. "브레이크일 확률 72%" 라고
    말해야 작업 지시가 된다. 그래서 두 모델을 함께 쓴다.

부품 분류 체계:
    계약서 ``pdm_prediction.top_component`` 와 ``components[].name`` 에 들어가는 값이다.
    ``engine | battery | brake | tire | oil | transmission`` 6개로 고정한다.

규칙 기반 폴백이 왜 쓸만한가?
    차량 정비는 **도메인 규칙이 매우 잘 정립된** 분야다. "브레이크 패드 3mm 미만
    → 교체" 는 법정 기준이고 머신러닝이 필요 없다. 규칙 기반 폴백의 정확도는
    데이터가 적을 때 오히려 학습 모델보다 높을 수 있고, 설명 가능성은 압도적이다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from app.core.logging import get_logger
from app.pdm.features import FeatureFrame, aggregate_features, feature_vector

logger = get_logger(__name__)

#: 분류 대상 부품(계약서 enum 과 동일). 순서가 모델 클래스 인덱스 순서다.
COMPONENTS: tuple[str, ...] = (
    "engine",
    "battery",
    "brake",
    "tire",
    "oil",
    "transmission",
)

#: 부품별 한국어 표시명(응답 ``reason`` 생성용).
COMPONENT_LABELS: dict[str, str] = {
    "engine": "엔진",
    "battery": "배터리",
    "brake": "브레이크",
    "tire": "타이어",
    "oil": "엔진오일",
    "transmission": "변속기",
}

#: 부품 상태 enum. 계약서 ``components[].status`` 에 들어간다.
#: ``NORMAL`` → ``WATCH`` → ``WARNING`` → ``CRITICAL`` 4단계.
#: 왜 4단계인가? 2단계(정상/고장)는 "지금은 괜찮지만 다음 정비 때 보자" 를
#: 표현할 수 없다. 운영자가 실제로 쓰는 구분을 반영했다.
STATUS_NORMAL = "NORMAL"
STATUS_WATCH = "WATCH"
STATUS_WARNING = "WARNING"
STATUS_CRITICAL = "CRITICAL"


@dataclass(slots=True)
class ComponentDiagnosis:
    """부품 하나의 진단 결과.

    Attributes:
        name: 부품명(``COMPONENTS`` 중 하나).
        score: 0~100 건강 점수(100 = 완전 건강).
            왜 0~1 이 아니라 0~100 인가? 계약서 ``components[].score`` 가
            ``81.2`` 형태이고, 운영자가 "점수" 로 직관적으로 읽기 쉽다.
        risk: 0~1 고장 위험도(= 대략 ``1 - score/100``).
        status: 4단계 상태 enum.
        reason: 한국어 판단 근거.
        evidence: 판단에 쓰인 센서값 ``{센서명: 값}``.
    """

    name: str
    score: float
    risk: float
    status: str
    reason: str
    evidence: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """계약서 ``components[]`` 스키마로 직렬화한다.

        Returns:
            ``{"name", "score", "status", "reason"}`` + 부가 필드.
        """
        return {
            "name": self.name,
            "score": round(float(self.score), 1),
            "status": self.status,
            "reason": self.reason,
            "risk": round(float(self.risk), 4),
            "evidence": {k: round(float(v), 3) for k, v in self.evidence.items()},
        }


@dataclass(slots=True)
class FailureResult:
    """고장 분류 전체 결과.

    Attributes:
        failure_risk: 0~1 종합 고장 위험도.
        top_component: 위험도가 가장 높은 부품명.
        components: 부품별 진단 리스트.
        method: 사용한 방법.
        probabilities: 부품별 고장 확률(학습 모델 경로에서만 채워진다).
    """

    failure_risk: float
    top_component: str
    components: list[ComponentDiagnosis] = field(default_factory=list)
    method: str = "rule-based"
    probabilities: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON 직렬화용 dict 를 반환한다."""
        return {
            "failure_risk": round(float(self.failure_risk), 4),
            "top_component": self.top_component,
            "components": [c.to_dict() for c in self.components],
            "method": self.method,
            "probabilities": {k: round(float(v), 4) for k, v in self.probabilities.items()},
        }


# ---------------------------------------------------------------------------
# 1) 규칙 기반 진단 (폴백 — 가중치 불필요)
# ---------------------------------------------------------------------------


def _score_from_risk(risk: float) -> float:
    """0~1 위험도를 0~100 건강 점수로 변환한다.

    선형 변환(``100 * (1 - risk)``)이 아니라 **곡선**을 쓴다.
    이유: 위험도 0.1(거의 정상)과 0.3(관찰 필요)의 운영적 차이는 크지만,
    0.7 과 0.9 의 차이는 "둘 다 당장 정비" 로 같다. 낮은 위험 구간을 넓게
    펼치면 점수가 더 유용한 정보를 담는다.

    Args:
        risk: 0~1 위험도.

    Returns:
        0~100 점수.
    """
    r = float(np.clip(risk, 0.0, 1.0))
    # (1-r)^0.75: r=0 → 100, r=0.3 → 77, r=0.5 → 59, r=0.9 → 17
    return float(np.clip(100.0 * ((1.0 - r) ** 0.75), 0.0, 100.0))


def _status_from_risk(risk: float) -> str:
    """위험도를 4단계 상태로 변환한다.

    경계값(0.25 / 0.50 / 0.75)은 운영 조치와 대응된다:
        - < 0.25 NORMAL   : 조치 없음
        - < 0.50 WATCH    : 다음 정기 점검 시 확인
        - < 0.75 WARNING  : 2주 내 정비 예약
        - >= 0.75 CRITICAL: 즉시 정비

    Args:
        risk: 0~1 위험도.

    Returns:
        상태 문자열.
    """
    r = float(risk)
    if r >= 0.75:
        return STATUS_CRITICAL
    if r >= 0.50:
        return STATUS_WARNING
    if r >= 0.25:
        return STATUS_WATCH
    return STATUS_NORMAL


def _linear_risk(value: float, good: float, bad: float) -> float:
    """값을 ``good``(위험 0) ~ ``bad``(위험 1) 로 선형 매핑한다.

    ``good > bad`` 이면 "작을수록 위험" 으로, ``good < bad`` 이면
    "클수록 위험" 으로 자동 해석한다 — 호출부에서 방향을 따로 넘길 필요가 없다.

    Args:
        value: 현재값.
        good: 위험 0 에 해당하는 값.
        bad: 위험 1 에 해당하는 값.

    Returns:
        0~1 위험도.
    """
    v, g, b = float(value), float(good), float(bad)
    if abs(b - g) < 1e-9:
        return 0.0
    return float(np.clip((v - g) / (b - g), 0.0, 1.0))


def diagnose_engine(features: dict[str, float]) -> ComponentDiagnosis:
    """엔진 상태를 진단한다.

    세 신호를 결합한다:
        1. 냉각수 온도 수준 (90도 정상 → 112도 위험)
        2. 온도 **상승 추세** — 수준이 아직 정상이어도 오르고 있으면 조기 경고다.
           냉각수 누설/워터펌프 고장은 서서히 진행되므로 추세가 핵심 신호다.
        3. 온도 변동성 — 서모스탯 고착은 온도가 들쭉날쭉해진다.

    Args:
        features: ``aggregate_features`` 출력.

    Returns:
        ComponentDiagnosis.
    """
    temp = features.get("engine_temp_c_last", 90.0)
    temp_mean = features.get("engine_temp_c_mean", 90.0)
    slope = features.get("engine_temp_c_slope", 0.0)
    volatility = features.get("engine_temp_c_std", 0.0)

    level_risk = _linear_risk(max(temp, temp_mean), 95.0, 112.0)
    # 샘플당 0.5도 이상 오르면 명백한 악화 추세다.
    trend_risk = _linear_risk(slope, 0.05, 0.80)
    volatility_risk = _linear_risk(volatility, 1.5, 8.0)

    risk = float(np.clip(0.55 * level_risk + 0.30 * trend_risk + 0.15 * volatility_risk, 0.0, 1.0))

    if level_risk >= 0.75:
        reason = f"냉각수 온도 {temp:.1f}°C — 과열 위험 구간"
    elif trend_risk >= 0.5:
        reason = f"냉각수 온도 상승 추세({slope:+.2f}°C/회) — 냉각 계통 점검 필요"
    elif volatility_risk >= 0.5:
        reason = f"냉각수 온도 변동 큼(σ={volatility:.1f}°C) — 서모스탯 점검 권장"
    elif risk >= 0.25:
        reason = f"냉각수 온도 {temp:.1f}°C — 정상 범위 상단"
    else:
        reason = f"냉각수 온도 {temp:.1f}°C — 정상 범위"

    return ComponentDiagnosis(
        name="engine",
        score=_score_from_risk(risk),
        risk=risk,
        status=_status_from_risk(risk),
        reason=reason,
        evidence={"engine_temp_c": temp, "temp_slope": slope, "temp_std": volatility},
    )


def diagnose_battery(features: dict[str, float]) -> ComponentDiagnosis:
    """배터리 상태를 진단한다.

    전압과 SOH 를 **곱이 아니라 최댓값**으로 결합한다.
    이유: 둘 중 하나만 나빠도 교체 대상이다. SOH 0.95(건강)인데 전압 11.5V 면
    발전기 고장이고, 전압 12.6V(정상)인데 SOH 0.65 면 용량 저하다 —
    원인은 달라도 둘 다 조치가 필요하다. 곱으로 결합하면 서로를 가린다.

    Args:
        features: 집계 특징.

    Returns:
        ComponentDiagnosis.
    """
    voltage = features.get("battery_voltage_last", 12.6)
    soh = features.get("battery_soh_last", 0.95)
    volt_slope = features.get("battery_voltage_slope", 0.0)

    voltage_risk = _linear_risk(voltage, 12.4, 11.6)
    soh_risk = _linear_risk(soh, 0.85, 0.65)
    # 전압이 내려가는 추세(음수 기울기)면 위험. 부호를 뒤집어 매핑한다.
    trend_risk = _linear_risk(-volt_slope, 0.005, 0.08)

    risk = float(np.clip(max(voltage_risk, soh_risk) * 0.85 + trend_risk * 0.15, 0.0, 1.0))

    if soh_risk >= 0.75:
        reason = f"배터리 건강도 {soh:.2f} — 용량 저하, 교체 권고"
    elif voltage_risk >= 0.75:
        reason = f"배터리 전압 {voltage:.2f}V — 충전 계통 점검 필요"
    elif trend_risk >= 0.5:
        reason = f"배터리 전압 하락 추세({volt_slope:+.3f}V/회) — 발전기 점검 권장"
    elif risk >= 0.25:
        reason = f"배터리 전압 {voltage:.2f}V / SOH {soh:.2f} — 관찰 필요"
    else:
        reason = f"배터리 전압 {voltage:.2f}V / SOH {soh:.2f} — 정상"

    return ComponentDiagnosis(
        name="battery",
        score=_score_from_risk(risk),
        risk=risk,
        status=_status_from_risk(risk),
        reason=reason,
        evidence={"battery_voltage": voltage, "battery_soh": soh, "voltage_slope": volt_slope},
    )


def diagnose_brake(features: dict[str, float]) -> ComponentDiagnosis:
    """브레이크 상태를 진단한다.

    패드 두께는 **법정 기준이 명확한** 드문 항목이다(3mm 미만 부적합).
    따라서 추세보다 절대값 가중치를 높게 둔다. 다만 마모 속도(기울기)는
    RUL 계산에 결정적이므로 함께 평가한다.

    Args:
        features: 집계 특징.

    Returns:
        ComponentDiagnosis.
    """
    pad = features.get("brake_pad_mm_last", 8.0)
    slope = features.get("brake_pad_mm_slope", 0.0)

    level_risk = _linear_risk(pad, 6.0, 2.5)
    # 패드가 빠르게 줄고 있으면(음수 기울기 큼) 위험.
    wear_risk = _linear_risk(-slope, 0.02, 0.50)

    risk = float(np.clip(0.80 * level_risk + 0.20 * wear_risk, 0.0, 1.0))

    if pad <= 3.0:
        reason = f"브레이크 패드 {pad:.1f}mm — 법정 교체 기준(3mm) 이하, 즉시 교체"
    elif pad <= 4.0:
        reason = f"브레이크 패드 {pad:.1f}mm — 교체 임박"
    elif wear_risk >= 0.5:
        reason = f"브레이크 패드 마모 빠름({slope:+.2f}mm/회) — 주행 습관/캘리퍼 점검"
    elif risk >= 0.25:
        reason = f"브레이크 패드 {pad:.1f}mm — 다음 점검 시 확인"
    else:
        reason = f"브레이크 패드 {pad:.1f}mm — 정상"

    return ComponentDiagnosis(
        name="brake",
        score=_score_from_risk(risk),
        risk=risk,
        status=_status_from_risk(risk),
        reason=reason,
        evidence={"brake_pad_mm": pad, "wear_slope": slope},
    )


def diagnose_tire(features: dict[str, float]) -> ComponentDiagnosis:
    """타이어 상태를 진단한다.

    **불균형(imbalance)을 평균보다 중요하게 본다.**
    네 바퀴가 모두 30psi 면 "약간 부족"(공기 보충)이지만, 한 바퀴만 24psi 면
    펑크/밸브 누설이라는 전혀 다른 문제이고 주행 안정성에 직접 영향을 준다.
    평균만 보는 시스템은 이 차이를 완전히 놓친다.

    Args:
        features: 집계 특징.

    Returns:
        ComponentDiagnosis.
    """
    mean_pressure = features.get("tire_pressure_mean", 33.0)
    imbalance = features.get("tire_pressure_imbalance", 0.0)
    pressures = {
        "FL": features.get("tire_pressure_fl_last", 33.0),
        "FR": features.get("tire_pressure_fr_last", 33.0),
        "RL": features.get("tire_pressure_rl_last", 33.0),
        "RR": features.get("tire_pressure_rr_last", 33.0),
    }
    lowest_wheel = min(pressures, key=lambda k: pressures[k])
    lowest = pressures[lowest_wheel]

    level_risk = _linear_risk(lowest, 30.0, 24.0)
    imbalance_risk = _linear_risk(imbalance, 2.0, 7.0)

    risk = float(np.clip(max(level_risk, imbalance_risk * 0.9), 0.0, 1.0))

    if level_risk >= 0.75:
        reason = f"{lowest_wheel} 타이어 공기압 {lowest:.1f}psi — 저압 위험, 즉시 보충"
    elif imbalance_risk >= 0.6:
        reason = f"타이어 공기압 편차 {imbalance:.1f}psi — {lowest_wheel} 누설 점검 필요"
    elif risk >= 0.25:
        reason = f"타이어 평균 공기압 {mean_pressure:.1f}psi — 보충 권장"
    else:
        reason = f"타이어 평균 공기압 {mean_pressure:.1f}psi — 정상"

    return ComponentDiagnosis(
        name="tire",
        score=_score_from_risk(risk),
        risk=risk,
        status=_status_from_risk(risk),
        reason=reason,
        evidence={"lowest_pressure": lowest, "imbalance": imbalance, "mean": mean_pressure},
    )


def diagnose_oil(features: dict[str, float]) -> ComponentDiagnosis:
    """엔진오일 상태를 진단한다.

    Args:
        features: 집계 특징.

    Returns:
        ComponentDiagnosis.
    """
    quality = features.get("oil_quality_last", 0.8)
    slope = features.get("oil_quality_slope", 0.0)

    level_risk = _linear_risk(quality, 0.55, 0.20)
    degradation_risk = _linear_risk(-slope, 0.002, 0.05)
    risk = float(np.clip(0.80 * level_risk + 0.20 * degradation_risk, 0.0, 1.0))

    if quality <= 0.25:
        reason = f"엔진오일 상태 {quality:.2f} — 즉시 교환 필요"
    elif quality <= 0.40:
        reason = f"엔진오일 상태 {quality:.2f} — 교환 임박"
    elif risk >= 0.25:
        reason = f"엔진오일 상태 {quality:.2f} — 다음 점검 시 확인"
    else:
        reason = f"엔진오일 상태 {quality:.2f} — 정상"

    return ComponentDiagnosis(
        name="oil",
        score=_score_from_risk(risk),
        risk=risk,
        status=_status_from_risk(risk),
        reason=reason,
        evidence={"oil_quality": quality, "quality_slope": slope},
    )


def diagnose_transmission(features: dict[str, float]) -> ComponentDiagnosis:
    """변속기 상태를 진단한다.

    변속기에는 전용 센서가 없으므로 **간접 지표**를 쓴다:
        - 진동 RMS: 변속기 마운트/베어링 이상은 진동으로 나타난다.
        - 저속 고RPM: 같은 속도에서 RPM 이 높아지면 클러치/토크컨버터 슬립 신호다.
          이 복합 지표가 단일 센서로는 보이지 않는 변속기 문제를 드러낸다.

    이것이 "파생 특징이 왜 필요한가" 의 가장 좋은 예시다.

    Args:
        features: 집계 특징.

    Returns:
        ComponentDiagnosis.
    """
    vibration = features.get("vibration_rms_last", 0.25)
    vib_slope = features.get("vibration_rms_slope", 0.0)
    speed = max(1.0, features.get("speed_kph_mean", 45.0))
    rpm = features.get("engine_rpm_mean", 1500.0)
    # rpm/speed 비율: 정상 주행에서 약 30~45(=1500/45). 60 이상이면 슬립 의심.
    rpm_per_kph = rpm / speed

    vibration_risk = _linear_risk(vibration, 0.45, 1.10)
    trend_risk = _linear_risk(vib_slope, 0.005, 0.08)
    slip_risk = _linear_risk(rpm_per_kph, 50.0, 95.0)

    risk = float(
        np.clip(0.50 * vibration_risk + 0.20 * trend_risk + 0.30 * slip_risk, 0.0, 1.0)
    )

    if vibration_risk >= 0.7:
        reason = f"진동 RMS {vibration:.2f}g — 구동계 이상 의심, 점검 필요"
    elif slip_risk >= 0.6:
        reason = f"속도 대비 엔진회전수 과다({rpm_per_kph:.0f}rpm/kph) — 변속기 슬립 의심"
    elif trend_risk >= 0.5:
        reason = f"진동 증가 추세({vib_slope:+.3f}g/회) — 마운트 점검 권장"
    elif risk >= 0.25:
        reason = f"진동 RMS {vibration:.2f}g — 관찰 필요"
    else:
        reason = f"진동 RMS {vibration:.2f}g — 정상"

    return ComponentDiagnosis(
        name="transmission",
        score=_score_from_risk(risk),
        risk=risk,
        status=_status_from_risk(risk),
        reason=reason,
        evidence={
            "vibration_rms": vibration,
            "rpm_per_kph": rpm_per_kph,
            "vibration_slope": vib_slope,
        },
    )


#: 부품명 → 진단 함수. 새 부품을 추가할 때 이 dict 한 줄만 늘리면 된다.
DIAGNOSERS = {
    "engine": diagnose_engine,
    "battery": diagnose_battery,
    "brake": diagnose_brake,
    "tire": diagnose_tire,
    "oil": diagnose_oil,
    "transmission": diagnose_transmission,
}


def rule_based_failure(frame: FeatureFrame) -> FailureResult:
    """도메인 규칙으로 부품별 고장 위험을 진단한다(가중치 불필요).

    종합 위험도는 부품별 위험도의 **최댓값 70% + 평균 30%** 다.
    브레이크 하나가 치명적이면 다른 부품이 멀쩡해도 차량은 위험하다
    (최댓값 중심). 동시에 여러 부품이 노후한 차량도 구별해야 한다(평균 보정).

    Args:
        frame: 센서 FeatureFrame.

    Returns:
        FailureResult(method=``"rule-based"``).
    """
    if frame.rows == 0:
        # 데이터가 없으면 "알 수 없음" 이다. 0 위험으로 보고하면
        # 센서가 죽은 차량이 "가장 건강한 차량" 으로 집계된다 — 위험한 착각이다.
        # 중립값 0.25(WATCH)를 주고 사유를 명시한다.
        components = [
            ComponentDiagnosis(
                name=name,
                score=_score_from_risk(0.25),
                risk=0.25,
                status=STATUS_WATCH,
                reason="센서 데이터 없음 — 진단 불가, 데이터 수집 점검 필요",
            )
            for name in COMPONENTS
        ]
        return FailureResult(
            failure_risk=0.25,
            top_component="engine",
            components=components,
            method="rule-based(no-data)",
        )

    features = aggregate_features(frame)
    components = [DIAGNOSERS[name](features) for name in COMPONENTS]
    risks = np.array([c.risk for c in components], dtype=np.float64)

    failure_risk = float(np.clip(0.70 * risks.max() + 0.30 * risks.mean(), 0.0, 1.0))
    top = max(components, key=lambda c: c.risk)

    return FailureResult(
        failure_risk=failure_risk,
        top_component=top.name,
        components=sorted(components, key=lambda c: c.risk, reverse=True),
        method="rule-based",
        probabilities={c.name: round(c.risk, 4) for c in components},
    )


# ---------------------------------------------------------------------------
# 2) sklearn 분류 모델
# ---------------------------------------------------------------------------


def sklearn_failure(frame: FeatureFrame, model_path: str) -> FailureResult | None:
    """학습된 sklearn 분류기로 부품별 고장 확률을 예측한다.

    모델 체크포인트는 joblib dict 형식을 기대한다::

        {"model": <classifier>, "classes": ["engine", "battery", ...],
         "feature_names": [...]}

    왜 ``classes`` 를 체크포인트에 저장해야 하는가?
        sklearn 의 ``predict_proba`` 는 ``model.classes_`` 순서로 확률을 준다.
        그 순서가 ``COMPONENTS`` 와 다르면 "브레이크 확률" 을 "엔진 확률" 로
        읽는다. 라벨 순서를 명시적으로 저장/검증하는 것이 유일한 안전장치다.

    Args:
        frame: FeatureFrame.
        model_path: joblib 파일 경로.

    Returns:
        FailureResult 또는 None(모델 없음 → 호출자가 폴백).
    """
    path = Path(str(model_path))
    if not path.is_file():
        return None
    try:
        import joblib  # noqa: PLC0415

        bundle = joblib.load(path)
    except Exception as exc:  # noqa: BLE001
        logger.info("고장분류 모델 로딩 실패(%s): %s", model_path, exc)
        return None

    if frame.rows == 0:
        return rule_based_failure(frame)

    try:
        model = bundle.get("model") if isinstance(bundle, dict) else bundle
        if model is None or not hasattr(model, "predict_proba"):
            return None

        classes = list(bundle.get("classes") or getattr(model, "classes_", [])) if isinstance(
            bundle, dict
        ) else list(getattr(model, "classes_", []))
        classes = [str(c) for c in classes]
        if not classes:
            return None

        vector, _names = feature_vector(frame)
        proba = np.asarray(model.predict_proba(vector.reshape(1, -1))[0], dtype=np.float64)
        if proba.size != len(classes):
            logger.warning(
                "고장분류 클래스 수 불일치(proba=%d, classes=%d) → 규칙 기반으로 폴백",
                proba.size,
                len(classes),
            )
            return None

        probabilities = {cls: float(p) for cls, p in zip(classes, proba, strict=False)}

        # 학습 모델은 확률만 주고 "왜" 를 말하지 않는다. 규칙 기반 진단에서
        # 사유/증거를 가져와 결합하면 설명 가능성을 잃지 않는다 —
        # 이 하이브리드가 실무에서 가장 잘 작동하는 패턴이다.
        rule_result = rule_based_failure(frame)
        rule_by_name = {c.name: c for c in rule_result.components}

        components: list[ComponentDiagnosis] = []
        for name in COMPONENTS:
            model_risk = probabilities.get(name)
            base = rule_by_name.get(name)
            if base is None:
                continue
            if model_risk is None:
                components.append(base)
                continue
            # 모델 60% + 규칙 40%. 모델을 더 신뢰하되 규칙이 안전망 역할을 한다.
            blended = float(np.clip(0.60 * model_risk + 0.40 * base.risk, 0.0, 1.0))
            components.append(
                ComponentDiagnosis(
                    name=name,
                    score=_score_from_risk(blended),
                    risk=blended,
                    status=_status_from_risk(blended),
                    reason=f"{base.reason} (모델 확률 {model_risk:.0%})",
                    evidence=base.evidence,
                )
            )

        if not components:
            return None
        risks = np.array([c.risk for c in components], dtype=np.float64)
        failure_risk = float(np.clip(0.70 * risks.max() + 0.30 * risks.mean(), 0.0, 1.0))
        top = max(components, key=lambda c: c.risk)

        return FailureResult(
            failure_risk=failure_risk,
            top_component=top.name,
            components=sorted(components, key=lambda c: c.risk, reverse=True),
            method=f"sklearn-{type(model).__name__}+rules",
            probabilities=probabilities,
        )
    except Exception as exc:  # noqa: BLE001
        logger.info("고장분류 추론 실패: %s", exc)
        return None


def train_failure_classifier(
    matrix: np.ndarray,
    labels: list[str],
    *,
    model_type: str = "random_forest",
    random_state: int = 42,
) -> dict[str, Any] | None:
    """고장 분류기를 학습한다(학습 스크립트용).

    ``class_weight="balanced"`` 를 쓰는 이유: 고장 데이터는 극도로 불균형하다
    (정상 95% vs 고장 5%). 가중치 없이 학습하면 모델이 "항상 정상" 이라고
    예측해도 95% 정확도가 나와, 쓸모없는 모델이 좋아 보인다.

    XGBoost 를 쓰지 않는 이유: requirements 에 없는 선택 의존성이고,
    sklearn 의 ``GradientBoostingClassifier`` 또는 ``HistGradientBoosting`` 이
    이 규모(특징 56개, 샘플 수천)에서는 충분한 성능을 낸다.

    Args:
        matrix: ``(N, F)`` 특징 행렬.
        labels: 길이 N 의 부품명 라벨 리스트.
        model_type: ``"random_forest"`` | ``"gradient_boosting"``.
        random_state: 재현성 시드.

    Returns:
        ``{"model", "classes", "feature_count"}`` dict 또는 None(sklearn 미설치).
    """
    arr = np.asarray(matrix, dtype=np.float64)
    label_list = [str(v) for v in (labels or [])]
    if arr.ndim != 2 or arr.shape[0] != len(label_list) or arr.shape[0] < 4:
        return None

    try:
        if str(model_type) == "gradient_boosting":
            from sklearn.ensemble import GradientBoostingClassifier  # noqa: PLC0415

            model: Any = GradientBoostingClassifier(random_state=int(random_state))
        else:
            from sklearn.ensemble import RandomForestClassifier  # noqa: PLC0415

            model = RandomForestClassifier(
                n_estimators=300,
                max_depth=12,
                class_weight="balanced",
                random_state=int(random_state),
                n_jobs=1,
            )
    except ImportError:
        logger.info("scikit-learn 미설치 → 고장분류 학습 생략")
        return None

    model.fit(arr, label_list)
    return {
        "model": model,
        "classes": [str(c) for c in model.classes_],
        "feature_count": int(arr.shape[1]),
    }


def classify_failure(frame: FeatureFrame, *, model_path: str | None = None) -> FailureResult:
    """사용 가능한 최선의 방법으로 고장을 분류한다.

    Args:
        frame: FeatureFrame.
        model_path: sklearn 모델 경로(선택).

    Returns:
        FailureResult(항상 유효).
    """
    if model_path:
        result = sklearn_failure(frame, model_path)
        if result is not None:
            return result
    return rule_based_failure(frame)


__all__ = [
    "COMPONENTS",
    "COMPONENT_LABELS",
    "DIAGNOSERS",
    "STATUS_CRITICAL",
    "STATUS_NORMAL",
    "STATUS_WARNING",
    "STATUS_WATCH",
    "ComponentDiagnosis",
    "FailureResult",
    "classify_failure",
    "diagnose_battery",
    "diagnose_brake",
    "diagnose_engine",
    "diagnose_oil",
    "diagnose_tire",
    "diagnose_transmission",
    "rule_based_failure",
    "sklearn_failure",
    "train_failure_classifier",
]
