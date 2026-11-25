"""헬스체크 라우터.

계약서 1장: ``GET /health`` (``/api/v1`` prefix **없음**)
    ``{"status":"UP","version":"1.0.0","device":"cpu","detector":"yolo-nano-plate",
    "recognizer":"crnn-ctc-kr","torch":"2.x","threads":4}``

왜 prefix 가 없는가? 컨테이너 오케스트레이터(docker-compose healthcheck,
k8s liveness probe)는 보통 루트 경로의 헬스체크를 기대하고, API 버전이 올라가도
헬스체크 경로는 바뀌지 않아야 한다.

중요: **헬스체크는 모델을 로드하지 않는다.**
    헬스체크가 모델 로딩을 유발하면 ① 기동 직후 프로브가 타임아웃되고,
    ② 프로브마다 수 초가 걸린다. 가중치 **파일 존재 여부**만 확인해
    보고할 이름을 결정한다(``-fallback`` 접미사 판정).
"""

from __future__ import annotations

from typing import Any

from app import __version__
from app.config import get_settings, settings_backend

try:
    from fastapi import APIRouter

    router: Any = APIRouter(tags=["health"])
    _HAS_FASTAPI = True
except ImportError:  # pragma: no cover - FastAPI 환경에서는 항상 존재
    router = None
    _HAS_FASTAPI = False


def torch_version() -> str:
    """torch 버전 문자열을 반환한다(미설치 시 ``"not-installed"``).

    import 를 try 로 감싸는 이유는 자명하지만, **import 자체가 느릴 수 있다**는
    점도 중요하다(torch import 는 1~3초). 헬스체크마다 import 하지 않도록
    모듈 수준 캐시를 쓰는 편이 좋지만, 파이썬의 모듈 캐시가 이미 그 역할을 한다
    (두 번째 import 는 즉시 반환).

    Returns:
        예: ``"2.8.0"`` 또는 ``"not-installed"``.
    """
    try:
        import torch  # noqa: PLC0415

        return str(torch.__version__)
    except ImportError:
        return "not-installed"


def _resolved_name(requested: str, weight_attrs: list[str]) -> str:
    """가중치 파일 존재 여부로 노출 이름을 결정한다(모델 로딩 없이).

    계약서 요구사항: 가중치가 없으면 ``-fallback`` 접미사를 붙인다.
    모델을 실제로 로드하면 더 정확하지만 헬스체크가 느려지므로,
    파일 존재 확인으로 근사한다.

    Args:
        requested: 요청/기본 모델 이름.
        weight_attrs: 확인할 설정 필드 이름들(하나라도 있으면 가중치 있음).

    Returns:
        ``"yolo"`` 또는 ``"yolo-fallback"``.
    """
    settings = get_settings()
    has_weights = any(settings.weight_exists(attr) for attr in weight_attrs)
    return str(requested) if has_weights else f"{requested}-fallback"


def health_payload() -> dict[str, Any]:
    """헬스체크 응답 본문을 만든다.

    라우터 함수와 분리한 이유: 테스트와 ``/api/v1/models/info`` 에서 재사용하고,
    FastAPI 없이도 호출 가능하게 하기 위함이다.

    Returns:
        계약서 ``GET /health`` 스키마 dict.
    """
    settings = get_settings()
    detector = _resolved_name(
        str(settings.default_detector), ["detector_weights", "detector_onnx"]
    )
    recognizer = _resolved_name(str(settings.default_recognizer), ["crnn_weights"])

    return {
        "status": "UP",
        "version": __version__,
        "device": settings.resolved_device,
        "detector": detector,
        "recognizer": recognizer,
        "torch": torch_version(),
        "threads": settings.effective_threads,
        # 아래는 계약서에 없는 추가 진단 정보다. 계약서는 최소 집합을 정의하므로
        # 추가 필드는 호환성을 깨지 않는다(백엔드는 필요한 필드만 읽는다).
        "settings_backend": settings_backend(),
        "fallback_ready": True,
    }


if _HAS_FASTAPI:

    @router.get(
        "/health",
        summary="헬스체크",
        description="서비스 상태, 버전, 디바이스, 현재 모델 이름을 반환한다. 모델을 로드하지 않는다.",
    )
    def get_health() -> dict[str, Any]:
        """``GET /health`` 핸들러.

        Returns:
            계약서 헬스체크 스키마.
        """
        return health_payload()

    @router.get(
        "/healthz",
        include_in_schema=False,
        summary="헬스체크 별칭(k8s 관례)",
    )
    def get_healthz() -> dict[str, Any]:
        """k8s 관례 경로 별칭.

        왜 별칭을 두는가? 운영 팀이 ``/healthz`` 를 기본으로 설정하는 경우가
        많다. 경로 하나를 더 노출하는 비용은 0 이고, 배포 시 혼란을 줄인다.

        Returns:
            계약서 헬스체크 스키마.
        """
        return health_payload()


__all__ = ["health_payload", "router", "torch_version"]
