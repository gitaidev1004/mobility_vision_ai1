"""예지보전 서비스: 이상탐지 + 고장분류 + RUL 을 묶어 ``/pdm/predict`` 응답을 만든다.

계약서 응답 스키마(3장)::

    {"vehicle_id":1,"health_score":72.4,"anomaly_score":0.38,"failure_risk":0.27,
     "risk_level":"MEDIUM","rul_days":41,"top_component":"brake",
     "components":[{"name":"engine","score":81.2,"status":"NORMAL","reason":"..."}],
     "recommendations":[{"priority":1,"item":"브레이크 패드 점검","due_km":3000,"due_days":30}],
     "model_version":"pdm-ensemble-1.0.0","computed_at":"2026-10-24T12:00:00"}

왜 앙상블인가?
    세 모델이 서로 다른 질문에 답하고, 각자 놓치는 것을 보완한다:
        - 이상탐지: "평소와 다른가" — 미지의 고장 패턴을 잡는다
        - 고장분류: "어디가 문제인가" — 조치 대상을 지목한다
        - RUL:      "언제까지인가" — 정비 일정을 만든다
    세 신호를 합성해 단일 ``health_score`` 와 ``risk_level`` 을 낸다.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np

from app.config import get_settings
from app.core.logging import get_logger
from app.pdm.anomaly import AnomalyResult, detect_anomaly
from app.pdm.failure import (
    COMPONENT_LABELS,
    STATUS_CRITICAL,
    STATUS_WARNING,
    STATUS_WATCH,
    FailureResult,
    classify_failure,
)
from app.pdm.features import build_feature_frame
from app.pdm.rul import MAX_RUL_DAYS, RulResult, predict_rul, rul_to_km

logger = get_logger(__name__)

#: 계약서 ``risk_level`` enum (DB ``pdm_prediction.risk_level`` 과 동일).
RISK_LEVELS: tuple[str, ...] = ("LOW", "MEDIUM", "HIGH", "CRITICAL")

#: health_score 구간별 위험 등급 경계.
#: 점수가 높을수록 건강하므로, 경계는 **내림차순**으로 읽는다.
#:   85 이상 LOW / 65 이상 MEDIUM / 40 이상 HIGH / 그 미만 CRITICAL
#:
#: 경계값 근거: 운영 조치와 1:1 대응시켰다.
#:   LOW      — 정기 점검만
#:   MEDIUM   — 다음 점검 시 해당 부품 확인
#:   HIGH     — 2주 내 정비 예약
#:   CRITICAL — 운행 전 점검(필요시 운행 중단)
HEALTH_LOW_BOUND: float = 85.0
HEALTH_MEDIUM_BOUND: float = 65.0
HEALTH_HIGH_BOUND: float = 40.0

#: 부품별 권장 조치 문구와 표준 교체 주기(km).
#: ``due_km`` 는 RUL 환산값과 이 주기 중 **작은 쪽**을 쓴다 —
#: "추세로는 1만km 남았지만 제조사 권장 주기가 5천km" 면 5천km 가 맞다.
MAINTENANCE_ACTIONS: dict[str, tuple[str, int]] = {
    "brake": ("브레이크 패드 점검 및 교체", 10000),
    "oil": ("엔진오일 및 필터 교환", 10000),
    "battery": ("배터리 점검 및 교체", 40000),
    "engine": ("냉각 계통(냉각수/워터펌프/서모스탯) 점검", 20000),
    "tire": ("타이어 공기압 보충 및 상태 점검", 5000),
    "transmission": ("변속기 오일 및 구동계 점검", 40000),
}

#: 일 평균 주행거리 기본값(한국 승용차 연평균 약 1.5만 km 기준).
DEFAULT_DAILY_KM: float = 45.0


@dataclass(slots=True)
class PdmPrediction:
    """예지보전 예측 결과(계약서 응답과 1:1).

    Attributes:
        vehicle_id: 차량 ID.
        health_score: 0~100 종합 건강 점수.
        anomaly_score: 0~1 이상 점수.
        failure_risk: 0~1 고장 위험도.
        risk_level: ``LOW`` | ``MEDIUM`` | ``HIGH`` | ``CRITICAL``.
        rul_days: 잔여수명(일).
        top_component: 가장 위험한 부품.
        components: 부품별 진단.
        recommendations: 권장 조치.
        model_version: 모델 버전 문자열.
        computed_at: 계산 시각(ISO 8601).
        detail: 진단 내역(이상탐지/고장분류/RUL 각 방법과 데이터 품질).
    """

    vehicle_id: int
    health_score: float
    anomaly_score: float
    failure_risk: float
    risk_level: str
    rul_days: int
    top_component: str
    components: list[dict[str, Any]]
    recommendations: list[dict[str, Any]]
    model_version: str
    computed_at: str
    detail: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """계약서 ``POST /api/v1/pdm/predict`` 응답 스키마로 직렬화한다.

        Returns:
            계약서와 키/타입이 일치하는 dict.
        """
        return {
            "vehicle_id": int(self.vehicle_id),
            "health_score": round(float(self.health_score), 1),
            "anomaly_score": round(float(self.anomaly_score), 4),
            "failure_risk": round(float(self.failure_risk), 4),
            "risk_level": self.risk_level,
            "rul_days": int(self.rul_days),
            "top_component": self.top_component,
            "components": self.components,
            "recommendations": self.recommendations,
            "model_version": self.model_version,
            "computed_at": self.computed_at,
            "detail": self.detail,
        }


def compute_health_score(
    anomaly: AnomalyResult, failure: FailureResult, rul: RulResult
) -> float:
    """세 신호를 합성해 0~100 종합 건강 점수를 만든다.

    가중치 설계:
        - 고장위험 0.45 — "어느 부품이 얼마나 위험한가" 가 가장 직접적인 신호다.
        - 이상점수 0.30 — 미지의 패턴을 잡는 보완 신호.
        - RUL      0.25 — 시간 여유. 임박할수록 점수를 깎는다.

    RUL 을 점수로 바꿀 때 **로그 스케일**을 쓴다.
        7일 vs 30일의 차이는 크지만 300일 vs 600일은 운영상 같다.
        ``log(1 + days) / log(1 + 180)`` 로 180일에서 1.0 에 도달하게 한다.

    Args:
        anomaly: 이상탐지 결과.
        failure: 고장분류 결과.
        rul: RUL 예측 결과.

    Returns:
        0~100 건강 점수.
    """
    failure_health = 1.0 - float(np.clip(failure.failure_risk, 0.0, 1.0))
    anomaly_health = 1.0 - float(np.clip(anomaly.score, 0.0, 1.0))
    rul_health = float(
        np.clip(np.log1p(max(0, int(rul.rul_days))) / np.log1p(180.0), 0.0, 1.0)
    )

    combined = 0.45 * failure_health + 0.30 * anomaly_health + 0.25 * rul_health
    return float(np.clip(combined * 100.0, 0.0, 100.0))


def decide_risk_level(health_score: float, failure: FailureResult, rul_days: int) -> str:
    """건강 점수와 보조 신호로 위험 등급을 결정한다.

    왜 점수만으로 정하지 않는가 — **안전 상한(safety override)**:
        건강 점수는 가중 평균이므로, 브레이크가 CRITICAL 인데 다른 다섯 부품이
        완벽하면 점수가 70(MEDIUM)으로 나올 수 있다. 하지만 브레이크 패드
        2.5mm 차량을 "중간 위험" 으로 보고하면 안 된다.
        따라서 ① CRITICAL 부품이 하나라도 있거나 ② RUL 이 7일 미만이면
        **무조건 CRITICAL** 로 올린다. 안전 관련 시스템의 기본 원칙이다.

    Args:
        health_score: 0~100 건강 점수.
        failure: 고장분류 결과(부품별 상태 확인용).
        rul_days: 잔여수명.

    Returns:
        ``LOW`` | ``MEDIUM`` | ``HIGH`` | ``CRITICAL``.
    """
    statuses = {c.status for c in failure.components}

    # --- 안전 상한 ---
    if STATUS_CRITICAL in statuses or int(rul_days) < 7:
        return "CRITICAL"

    # --- 점수 기반 등급 ---
    score = float(health_score)
    if score >= HEALTH_LOW_BOUND:
        level = "LOW"
    elif score >= HEALTH_MEDIUM_BOUND:
        level = "MEDIUM"
    elif score >= HEALTH_HIGH_BOUND:
        level = "HIGH"
    else:
        level = "CRITICAL"

    # WARNING 부품이 있으면 LOW 로는 두지 않는다(상향 조정).
    if level == "LOW" and STATUS_WARNING in statuses:
        return "MEDIUM"
    # RUL 30일 미만이면 최소 HIGH.
    if int(rul_days) < 30 and level in {"LOW", "MEDIUM"}:
        return "HIGH"
    return level


def build_recommendations(
    failure: FailureResult,
    rul: RulResult,
    *,
    daily_km: float = DEFAULT_DAILY_KM,
    max_items: int = 5,
) -> list[dict[str, Any]]:
    """부품 진단과 RUL 로 권장 조치 목록을 만든다.

    계약서 스키마: ``{"priority": 1, "item": "...", "due_km": 3000, "due_days": 30}``

    우선순위 규칙:
        위험도 내림차순으로 1 부터 매긴다. ``WATCH`` 미만(NORMAL)인 부품은
        권고에 넣지 않는다 — 할 일이 없는 항목으로 목록을 채우면 운영자가
        정작 중요한 항목을 놓친다.

    ``due_days`` 는 **부품별 RUL** 을 쓴다(종합 RUL 이 아니다).
    브레이크가 20일이고 오일이 150일이면 각각 다른 일정으로 안내해야 한다.

    Args:
        failure: 고장분류 결과.
        rul: RUL 예측 결과.
        daily_km: 일 평균 주행거리.
        max_items: 최대 권고 수.

    Returns:
        권고 리스트. 모든 부품이 정상이면 "정기 점검" 1건을 반환한다.
    """
    # 권고 대상 선정에는 **두 가지 조건**을 모두 본다:
    #   ① 상태가 WATCH 이상 (현재 나쁨)
    #   ② 또는 해당 부품 RUL 이 90일 미만 (곧 나빠짐)
    #
    # ②가 없으면 어떤 일이 벌어지는가 — 실측 사례:
    #   브레이크 패드가 5.5mm(위험도 0.12 → NORMAL)이고 하루 0.05mm 마모 중이면
    #   RUL 은 51일이다. 즉 두 달 뒤 교체가 필요한데, 상태만 보면 NORMAL 이라
    #   "정기 점검(이상 징후 없음)" 만 안내되어 정비 계획을 세울 수 없다.
    #   예지보전의 존재 이유가 바로 이 "아직 정상이지만 곧 아닌" 구간이다.
    rul_horizon_days = 90
    actionable = [
        c
        for c in failure.components
        if c.status in {STATUS_WATCH, STATUS_WARNING, STATUS_CRITICAL}
        or int(rul.per_component.get(c.name, MAX_RUL_DAYS)) < rul_horizon_days
    ]
    # 정렬: 위험도 내림차순, 같으면 RUL 오름차순(임박한 것 먼저).
    actionable.sort(
        key=lambda c: (-c.risk, int(rul.per_component.get(c.name, 730)))
    )

    out: list[dict[str, Any]] = []
    for priority, component in enumerate(actionable[: max(1, int(max_items))], start=1):
        action, interval_km = MAINTENANCE_ACTIONS.get(
            component.name, (f"{COMPONENT_LABELS.get(component.name, component.name)} 점검", 10000)
        )
        days = int(rul.per_component.get(component.name, rul.rul_days))

        # 상태 기반 기한 상한(status cap).
        #
        # 왜 필요한가 — 실측 모순: 배터리 전압이 11.6V 로 **이미 위험 기준 이하**
        # 인데 값이 일정해 추세 기울기가 0 이었다. 그러면 선형 외삽 RUL 이
        # 730일(상한)로 나와 "CRITICAL 배터리를 2년 뒤 점검" 이라는 권고가 생긴다.
        # RUL 은 "언제 나빠지는가" 를 답하지만, 이미 나쁜 부품에는 질문 자체가
        # 성립하지 않는다. 상태가 답을 덮어써야 한다.
        if component.status == STATUS_CRITICAL:
            days = min(days, 3)
        elif component.status == STATUS_WARNING:
            days = min(days, 14)
        elif component.status == STATUS_WATCH:
            days = min(days, 60)

        # 추세 기반 거리와 제조사 권장 주기 중 보수적인(작은) 쪽을 택한다.
        due_km = min(rul_to_km(days, daily_km=daily_km), int(interval_km))
        out.append(
            {
                "priority": priority,
                "item": action,
                "component": component.name,
                "due_km": int(due_km),
                "due_days": int(days),
                "status": component.status,
                "reason": component.reason,
            }
        )

    if not out:
        out.append(
            {
                "priority": 1,
                "item": "정기 점검(이상 징후 없음)",
                "component": "general",
                "due_km": 10000,
                "due_days": min(180, int(rul.rul_days)),
                "status": "NORMAL",
                "reason": "모든 부품이 정상 범위입니다.",
            }
        )
    return out


class PdmService:
    """예지보전 예측 서비스.

    사용 예::

        service = get_pdm_service()
        result = service.predict(vehicle_id=1, readings=[...], odometer_km=84210)
        payload = result.to_dict()

    ``predict`` 는 예외를 던지지 않는다 — 계약서의 "500 금지" 원칙을 따른다.
    """

    def __init__(self) -> None:
        """설정에서 가중치 경로를 읽어 초기화한다."""
        settings = get_settings()
        self.model_version = str(settings.pdm_model_version)
        # 가중치가 있으면 쓰고 없으면 None → 각 모듈이 폴백으로 동작한다.
        self._anomaly_path = (
            settings.pdm_anomaly_weights if settings.weight_exists("pdm_anomaly_weights") else None
        )
        self._failure_path = (
            settings.pdm_failure_weights if settings.weight_exists("pdm_failure_weights") else None
        )
        self._rul_path = (
            settings.pdm_rul_weights if settings.weight_exists("pdm_rul_weights") else None
        )
        # AutoEncoder 는 별도 환경변수 없이 규약으로 찾는다.
        ae_candidate = settings.model_root / "pdm_autoencoder.pt"
        self._autoencoder_path = str(ae_candidate) if ae_candidate.is_file() else None

    def predict(
        self,
        *,
        vehicle_id: int,
        readings: list[dict[str, Any]],
        odometer_km: float | None = None,
        daily_km: float = DEFAULT_DAILY_KM,
    ) -> PdmPrediction:
        """센서 시계열로 예지보전 예측을 수행한다.

        Args:
            vehicle_id: 차량 ID.
            readings: 센서 판독 리스트(계약서 ``readings`` 필드).
            odometer_km: 현재 주행거리(권고의 ``due_km`` 보정에 사용).
            daily_km: 일 평균 주행거리.

        Returns:
            PdmPrediction. 어떤 입력에도 유효한 결과를 돌려준다.
        """
        try:
            return self._predict_impl(
                vehicle_id=int(vehicle_id),
                readings=readings,
                odometer_km=odometer_km,
                daily_km=float(daily_km),
            )
        except Exception as exc:  # noqa: BLE001 - 서비스는 예외를 올리지 않는다
            logger.warning("PdM 예측 실패 → 중립 결과 반환: %s", exc, extra={"stage": "pdm"})
            return self._neutral_result(int(vehicle_id), reason=f"{type(exc).__name__}: {exc}")

    def _predict_impl(
        self,
        *,
        vehicle_id: int,
        readings: list[dict[str, Any]],
        odometer_km: float | None,
        daily_km: float,
    ) -> PdmPrediction:
        """실제 예측 로직."""
        frame = build_feature_frame(readings or [])

        anomaly = detect_anomaly(
            frame,
            sklearn_path=self._anomaly_path,
            autoencoder_path=self._autoencoder_path,
        )
        failure = classify_failure(frame, model_path=self._failure_path)
        rul = predict_rul(frame, weights_path=self._rul_path)

        health = compute_health_score(anomaly, failure, rul)
        risk_level = decide_risk_level(health, failure, rul.rul_days)
        recommendations = build_recommendations(failure, rul, daily_km=daily_km)

        # 데이터가 없으면 "건강하다" 고 단정하지 않는다.
        #
        # 왜 특별 취급하는가 — 실측 모순: 빈 readings 로 요청했을 때
        # health=88.8 / risk=LOW 가 나왔다. 규칙 기반 진단이 모든 부품에
        # 중립값 0.25 를 주고 RUL 이 상한(730일)을 돌려주니 점수가 높게 합성된 것이다.
        # 그 결과 **센서가 고장난 차량이 가장 건강한 차량으로 집계**된다.
        # 센서 결함은 그 자체로 조치 대상이므로 MEDIUM 으로 고정하고 사유를 남긴다.
        if frame.rows == 0:
            health = min(health, 60.0)
            risk_level = "MEDIUM"
            recommendations = [
                {
                    "priority": 1,
                    "item": "센서 데이터 수집 상태 점검",
                    "component": "general",
                    "due_km": 0,
                    "due_days": 7,
                    "status": STATUS_WATCH,
                    "reason": "센서 판독 데이터가 없어 진단할 수 없습니다. 통신/센서 점검이 필요합니다.",
                }
            ]

        # 주행거리 기준 권고 보정: due_km 를 현재 오도미터에 더해 "언제" 를
        # 절대 주행거리로도 알 수 있게 한다(백엔드가 정비 이력과 비교하기 쉽다).
        if odometer_km is not None:
            try:
                base = float(odometer_km)
                for item in recommendations:
                    item["due_odometer_km"] = int(base + float(item.get("due_km", 0)))
            except (TypeError, ValueError):
                pass

        return PdmPrediction(
            vehicle_id=vehicle_id,
            health_score=health,
            anomaly_score=anomaly.score,
            failure_risk=failure.failure_risk,
            risk_level=risk_level,
            rul_days=rul.rul_days,
            top_component=failure.top_component,
            components=[c.to_dict() for c in failure.components],
            recommendations=recommendations,
            model_version=self.model_version,
            computed_at=datetime.now().isoformat(timespec="seconds"),
            detail={
                "anomaly": anomaly.to_dict(),
                "rul": rul.to_dict(),
                "failure_method": failure.method,
                "data_quality": frame.quality_summary(),
                "weights": {
                    "anomaly": self._anomaly_path,
                    "failure": self._failure_path,
                    "rul": self._rul_path,
                    "autoencoder": self._autoencoder_path,
                },
            },
        )

    def _neutral_result(self, vehicle_id: int, *, reason: str) -> PdmPrediction:
        """예측 실패 시 반환할 중립 결과를 만든다.

        왜 "건강 100점" 이 아니라 중립값(60점/MEDIUM)인가?
            예측이 실패했는데 "완전히 건강하다" 고 보고하면 운영자가 문제 차량을
            방치한다. 반대로 "CRITICAL" 로 보고하면 거짓 경보로 신뢰를 잃는다.
            "알 수 없으니 확인하라" 는 뜻의 중립값이 정직하다.

        Args:
            vehicle_id: 차량 ID.
            reason: 실패 사유.

        Returns:
            PdmPrediction.
        """
        return PdmPrediction(
            vehicle_id=vehicle_id,
            health_score=60.0,
            anomaly_score=0.0,
            failure_risk=0.25,
            risk_level="MEDIUM",
            rul_days=180,
            top_component="unknown",
            components=[],
            recommendations=[
                {
                    "priority": 1,
                    "item": "센서 데이터 수집 상태 점검",
                    "component": "general",
                    "due_km": 0,
                    "due_days": 7,
                    "status": STATUS_WATCH,
                    "reason": "예측에 필요한 데이터를 처리할 수 없습니다.",
                }
            ],
            model_version=self.model_version,
            computed_at=datetime.now().isoformat(timespec="seconds"),
            detail={"error": reason, "fallback": "neutral"},
        )

    def info(self) -> dict[str, Any]:
        """서비스 메타데이터를 반환한다(레지스트리/모델 정보 API 용)."""
        return pdm_spec()


def pdm_spec() -> dict[str, Any]:
    """PdM 모델 스펙을 반환한다(가중치 로딩 없이).

    Returns:
        구성 모델과 가중치 존재 여부를 담은 dict.
    """
    settings = get_settings()
    ae = settings.model_root / "pdm_autoencoder.pt"
    return {
        "name": "pdm-ensemble",
        "model_version": str(settings.pdm_model_version),
        "architecture": "이상탐지(IsolationForest/AutoEncoder/통계) + 고장분류(RandomForest/규칙) + RUL(LSTM/선형외삽)",
        "components": list(COMPONENT_LABELS.keys()),
        "risk_levels": list(RISK_LEVELS),
        "weights": {
            "anomaly": {
                "path": settings.pdm_anomaly_weights,
                "exists": settings.weight_exists("pdm_anomaly_weights"),
            },
            "failure": {
                "path": settings.pdm_failure_weights,
                "exists": settings.weight_exists("pdm_failure_weights"),
            },
            "rul": {
                "path": settings.pdm_rul_weights,
                "exists": settings.weight_exists("pdm_rul_weights"),
            },
            "autoencoder": {"path": str(ae), "exists": ae.is_file()},
        },
        "fallback_available": True,
        "weights_required": False,
    }


#: 서비스 싱글턴.
_SERVICE: PdmService | None = None


def get_pdm_service() -> PdmService:
    """PdM 서비스 싱글턴을 반환한다.

    Returns:
        PdmService.
    """
    global _SERVICE  # noqa: PLW0603 - 프로세스 싱글턴
    if _SERVICE is None:
        _SERVICE = PdmService()
    return _SERVICE


def reset_pdm_service() -> None:
    """서비스 싱글턴을 초기화한다(가중치 교체/테스트용)."""
    global _SERVICE  # noqa: PLW0603
    _SERVICE = None


def predict_pdm(
    vehicle_id: int,
    readings: list[dict[str, Any]],
    *,
    odometer_km: float | None = None,
) -> dict[str, Any]:
    """예지보전 예측 편의 함수(계약서 응답 dict 를 바로 반환).

    Args:
        vehicle_id: 차량 ID.
        readings: 센서 판독 리스트.
        odometer_km: 현재 주행거리.

    Returns:
        계약서 응답 dict.
    """
    return get_pdm_service().predict(
        vehicle_id=vehicle_id, readings=readings, odometer_km=odometer_km
    ).to_dict()


__all__ = [
    "DEFAULT_DAILY_KM",
    "HEALTH_HIGH_BOUND",
    "HEALTH_LOW_BOUND",
    "HEALTH_MEDIUM_BOUND",
    "MAINTENANCE_ACTIONS",
    "RISK_LEVELS",
    "PdmPrediction",
    "PdmService",
    "build_recommendations",
    "compute_health_score",
    "decide_risk_level",
    "get_pdm_service",
    "pdm_spec",
    "predict_pdm",
    "reset_pdm_service",
]
