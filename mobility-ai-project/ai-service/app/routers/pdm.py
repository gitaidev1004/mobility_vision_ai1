"""예지보전(PdM) 라우터.

계약서 3장: ``POST /api/v1/pdm/predict`` (application/json)

요청::

    {"vehicle_id": 1, "odometer_km": 84210,
     "readings": [{"measured_at":"2026-10-01T09:00:00","engine_temp_c":92.4, ...}]}

이 라우터도 500 을 내지 않는다. ``PdmService.predict`` 가 모든 예외를 흡수해
중립 결과(health 60 / MEDIUM)를 돌려주므로, 라우터는 입력 정규화만 담당한다.
"""

from __future__ import annotations

from typing import Any

from app.core.logging import get_logger
from app.pdm.service import DEFAULT_DAILY_KM, get_pdm_service, pdm_spec

logger = get_logger(__name__)

try:
    from fastapi import APIRouter, Body

    from app.schemas import PdmRequest

    router: Any = APIRouter(prefix="/api/v1/pdm", tags=["pdm"])
    _HAS_FASTAPI = True
except ImportError:  # pragma: no cover
    router = None
    PdmRequest = None  # type: ignore[assignment,misc]
    _HAS_FASTAPI = False


def _as_dict(reading: Any) -> dict[str, Any]:
    """센서 판독 항목을 dict 로 정규화한다.

    pydantic 모델, dict, 임의 객체를 모두 받는다.
    왜 세 가지를 지원하는가? 라우터는 pydantic 모델을 받지만, 테스트와
    배치 작업은 dict 를 바로 넘긴다. 입구에서 통일하면 하위 로직이 단순해진다.

    Args:
        reading: 센서 판독 항목.

    Returns:
        dict(변환 불가 시 빈 dict).
    """
    if isinstance(reading, dict):
        return reading
    dump = getattr(reading, "model_dump", None)
    if callable(dump):
        try:
            return dict(dump())
        except Exception:  # noqa: BLE001
            pass
    if hasattr(reading, "__dict__"):
        return {k: v for k, v in vars(reading).items() if not k.startswith("_")}
    return {}


def predict_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """요청 dict 로 예지보전 예측을 수행한다(HTTP 와 무관한 재사용 함수).

    Args:
        payload: ``{"vehicle_id", "odometer_km", "readings", "daily_km"}``.

    Returns:
        계약서 PdM 응답 dict.
    """
    try:
        vehicle_id = int(payload.get("vehicle_id") or 0)
    except (TypeError, ValueError):
        vehicle_id = 0

    readings = [_as_dict(r) for r in (payload.get("readings") or [])]

    odometer = payload.get("odometer_km")
    try:
        odometer_km = float(odometer) if odometer is not None else None
    except (TypeError, ValueError):
        odometer_km = None

    daily = payload.get("daily_km")
    try:
        daily_km = float(daily) if daily else DEFAULT_DAILY_KM
    except (TypeError, ValueError):
        daily_km = DEFAULT_DAILY_KM
    # 비현실적인 값은 기본값으로 되돌린다(0 이나 음수가 들어오면 due_km 가 깨진다).
    if not (0.1 <= daily_km <= 2000.0):
        daily_km = DEFAULT_DAILY_KM

    result = get_pdm_service().predict(
        vehicle_id=vehicle_id,
        readings=readings,
        odometer_km=odometer_km,
        daily_km=daily_km,
    )
    return result.to_dict()


if _HAS_FASTAPI:

    @router.post(
        "/predict",
        summary="예지보전 예측",
        description=(
            "센서 시계열로 건강 점수·이상 점수·고장 위험·잔여수명(RUL)과 "
            "부품별 진단, 권장 조치를 계산한다. 학습된 가중치가 없으면 "
            "도메인 규칙과 통계 기법으로 대체하며, 500 을 반환하지 않는다."
        ),
    )
    def post_predict(request: PdmRequest = Body(...)) -> dict[str, Any]:  # type: ignore[valid-type]
        """``POST /api/v1/pdm/predict`` 핸들러.

        Args:
            request: PdmRequest 본문.

        Returns:
            계약서 PdM 응답.
        """
        try:
            payload = request.model_dump()
        except Exception:  # noqa: BLE001 - pydantic 버전 차이 방어
            payload = _as_dict(request)
        return predict_payload(payload)

    @router.get(
        "/spec",
        summary="예지보전 모델 스펙",
        description="구성 모델과 가중치 존재 여부를 반환한다(모델을 로드하지 않는다).",
    )
    def get_spec() -> dict[str, Any]:
        """``GET /api/v1/pdm/spec`` 핸들러.

        계약서에는 없는 보조 엔드포인트다. 운영 중 "지금 어떤 경로로 예측하고
        있는가(학습 모델인가 규칙 기반인가)" 를 확인할 수단이 필요하다.

        Returns:
            PdM 스펙 dict.
        """
        return pdm_spec()


__all__ = ["predict_payload", "router"]
