"""모델 정보 / 벤치마크 라우터.

계약서 3장:
    - ``GET /api/v1/models/info`` → 모델 레지스트리 목록
    - ``POST /api/v1/models/benchmark`` → 벤치마크 실행 결과 **배열**

벤치마크는 수 초~수십 초가 걸릴 수 있다. 동기 엔드포인트로 두는 이유:
    백엔드(``POST /api/v1/benchmarks/run``)가 결과를 받아 DB 에 저장해야 하므로,
    "요청하면 결과를 준다" 는 단순한 계약이 필요하다. 비동기 작업 큐를 쓰면
    백엔드가 폴링 로직을 구현해야 하고 교재 범위를 넘는 복잡도가 생긴다.
    대신 ``samples`` 기본값을 작게(20) 두어 수 초 내 끝나게 했다.
"""

from __future__ import annotations

import platform
import sys
from typing import Any

from app import __version__
from app.config import get_settings, settings_backend
from app.core.logging import get_logger
from app.core.registry import registry_summary
from app.ocr.charset import charset_summary

logger = get_logger(__name__)

try:
    from fastapi import APIRouter, Body, Query

    from app.schemas import BenchmarkRequest

    router: Any = APIRouter(prefix="/api/v1/models", tags=["models"])
    _HAS_FASTAPI = True
except ImportError:  # pragma: no cover
    router = None
    BenchmarkRequest = None  # type: ignore[assignment,misc]
    _HAS_FASTAPI = False


def runtime_info() -> dict[str, Any]:
    """런타임 환경 정보를 수집한다.

    왜 필요한가? "교재와 결과가 다르다" 는 문의의 대부분은 환경 차이다.
    설치된 라이브러리 버전을 한 번에 보여 주면 원인 파악이 즉시 끝난다.

    Returns:
        파이썬/OS/라이브러리 버전 dict.
    """
    settings = get_settings()

    def _version(module_name: str) -> str:
        """모듈 버전을 조회한다(미설치 시 ``"not-installed"``)."""
        try:
            module = __import__(module_name)
            return str(getattr(module, "__version__", "unknown"))
        except ImportError:
            return "not-installed"
        except Exception:  # noqa: BLE001
            return "error"

    return {
        "service_version": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "device": settings.resolved_device,
        "threads": settings.effective_threads,
        "settings_backend": settings_backend(),
        "libraries": {
            "numpy": _version("numpy"),
            "cv2": _version("cv2"),
            "PIL": _version("PIL"),
            "torch": _version("torch"),
            "torchvision": _version("torchvision"),
            "ultralytics": _version("ultralytics"),
            "onnxruntime": _version("onnxruntime"),
            "sklearn": _version("sklearn"),
            "pandas": _version("pandas"),
            "fastapi": _version("fastapi"),
            "pydantic": _version("pydantic"),
        },
    }


def models_info_payload(*, load: bool = False) -> dict[str, Any]:
    """``GET /api/v1/models/info`` 응답 본문을 만든다.

    Args:
        load: True 면 각 모델을 실제 로드해 폴백 여부까지 정확히 보고한다.
            기본 False — 정보 조회가 모델 로딩을 유발하면 응답이 수 초로 늘어난다.

    Returns:
        계약서 모델 정보 응답 dict.
    """
    summary = registry_summary(load=load)
    summary["charset"] = charset_summary()
    summary["runtime"] = runtime_info()
    return summary


def benchmark_payload(
    *,
    samples: int = 20,
    detectors: list[str] | None = None,
    recognizers: list[str] | None = None,
    seed: int = 20261024,
) -> list[dict[str, Any]]:
    """``POST /api/v1/models/benchmark`` 응답 본문을 만든다.

    Args:
        samples: 조합당 샘플 수.
        detectors: 평가할 검출기(None 이면 전체).
        recognizers: 평가할 인식기(None 이면 전체).
        seed: 합성 데이터 시드.

    Returns:
        ``BenchmarkEntry`` dict 리스트. 실패해도 빈 배열 대신 오류 항목을 포함한다.
    """
    from app.train.benchmark import run_benchmark  # noqa: PLC0415 - 지연 import

    try:
        return run_benchmark(
            samples=int(max(1, min(200, int(samples)))),
            detectors=detectors,
            recognizers=recognizers,
            seed=int(seed),
            include_pdm=True,
        )
    except Exception as exc:  # noqa: BLE001 - 벤치마크 실패도 500 이 아니다
        logger.warning("벤치마크 실행 실패: %s", exc)
        return [
            {
                "model_name": "benchmark-failed",
                "task": "detect+ocr",
                "detect_recall": 0.0,
                "exact_match_rate": 0.0,
                "latency_p50_ms": 0.0,
                "latency_p95_ms": 0.0,
                "memory_mb": 0.0,
                "model_size_mb": 0.0,
                "quantized": False,
                "detail": {"error": f"{type(exc).__name__}: {exc}"},
            }
        ]


if _HAS_FASTAPI:

    @router.get(
        "/info",
        summary="모델 레지스트리 정보",
        description=(
            "등록된 검출기/인식기/예지보전 모델과 가중치 존재 여부, 문자셋, "
            "런타임 환경을 반환한다. `load=true` 를 주면 모델을 실제 로드해 "
            "폴백 여부까지 정확히 보고한다(느림)."
        ),
    )
    def get_models_info(
        load: bool = Query(
            False, description="true 면 모델을 로드해 폴백 여부를 정확히 보고(느림)"
        ),
    ) -> dict[str, Any]:
        """``GET /api/v1/models/info`` 핸들러.

        Args:
            load: 모델 로딩 여부.

        Returns:
            모델 정보 응답.
        """
        return models_info_payload(load=bool(load))

    @router.post(
        "/benchmark",
        summary="벤치마크 실행",
        description=(
            "합성 데이터로 검출기/인식기 조합의 지연(p50/p95), 메모리, "
            "검출 재현율, 완전 일치율을 측정해 배열로 반환한다. "
            "합성 데이터이므로 **모델 간 상대 비교** 용도로 해석해야 한다."
        ),
    )
    def post_benchmark(
        request: BenchmarkRequest | None = Body(None),  # type: ignore[valid-type]
    ) -> list[dict[str, Any]]:
        """``POST /api/v1/models/benchmark`` 핸들러.

        본문을 생략할 수 있게 한 이유: 백엔드가 "그냥 실행해 줘" 로 호출하는
        경우가 많고, 기본값으로 충분하다. 빈 본문도 받아들인다.

        Args:
            request: 벤치마크 요청(생략 가능).

        Returns:
            벤치마크 결과 배열.
        """
        samples = 20
        detectors: list[str] | None = None
        recognizers: list[str] | None = None
        seed = 20261024

        if request is not None:
            samples = int(getattr(request, "samples", samples) or samples)
            detectors = getattr(request, "detectors", None)
            recognizers = getattr(request, "recognizers", None)
            seed = int(getattr(request, "seed", seed) or seed)

        return benchmark_payload(
            samples=samples, detectors=detectors, recognizers=recognizers, seed=seed
        )


__all__ = ["benchmark_payload", "models_info_payload", "router", "runtime_info"]
