"""FastAPI 애플리케이션 진입점.

구성:
    - CORS 미들웨어 (프론트엔드 5173, 백엔드 8080 허용)
    - lifespan (기동/종료 훅 — 스레드 수 설정, 폴백 준비 확인)
    - 라우터 등록 (health, lpr, pdm, models)
    - 요청 ID 미들웨어 + 접근 로그
    - **전역 예외 처리** — 계약서의 "절대 500 을 던지지 않는다" 를 최종 보장

전역 예외 처리 설계가 이 모듈의 핵심이다:
    파이프라인과 라우터가 이미 예외를 흡수하지만, 그래도 빠져나오는 예외가 있을
    수 있다(메모리 부족, 서드파티 라이브러리의 ``SystemExit`` 등).
    전역 핸들러가 **경로별로 적절한 계약 응답**을 200 으로 만들어 준다.
    왜 500 을 피하는가? 백엔드가 500 을 받으면 ``RestClient`` 가 예외를 던지고,
    게이트 운영자에게는 "시스템 오류" 만 보인다. 반면 ``status: NO_PLATE`` 를
    받으면 "번호판을 읽지 못했습니다 — 수동 입력" 이라는 정상 운영 흐름으로
    이어진다. 같은 실패를 **복구 가능한 형태**로 전달하는 것이 핵심이다.

실행::

    uvicorn app.main:app --host 0.0.0.0 --port 8000
    python -m app.main                      # 직접 실행도 지원
"""

from __future__ import annotations

import os
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

from app import __version__
from app.config import get_settings
from app.core.logging import configure_logging, get_logger, get_request_id, set_request_id

logger = get_logger(__name__)

try:
    from fastapi import FastAPI, Request
    from fastapi.exceptions import RequestValidationError
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from starlette.exceptions import HTTPException as StarletteHTTPException

    _HAS_FASTAPI = True
except ImportError:  # pragma: no cover - FastAPI 없는 환경(순수 로직 테스트)
    FastAPI = None  # type: ignore[assignment,misc]
    _HAS_FASTAPI = False


def _apply_thread_limits() -> None:
    """추론 스레드 수를 제한한다(기동 시 1회).

    왜 스레드를 제한하는가?
        BLAS/OpenMP 는 기본적으로 **모든 코어**를 쓴다. 게이트 서버에서
        여러 요청이 동시에 들어오면 각 요청이 모든 코어를 요구해 컨텍스트
        스위칭으로 오히려 전체 처리량이 떨어진다(thrashing).
        요청당 스레드를 제한하고 요청 간 병렬성에 맡기는 편이 처리량이 높다.

    환경변수는 **라이브러리 import 전에** 설정해야 효과가 있으므로,
    lifespan 이 아니라 모듈 로딩 시점에 호출한다.
    """
    settings = get_settings()
    threads = str(settings.effective_threads)
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(key, threads)

    # torch 가 이미 로드된 경우를 위해 런타임 설정도 시도한다.
    try:
        import torch  # noqa: PLC0415

        torch.set_num_threads(settings.effective_threads)
    except (ImportError, RuntimeError, AttributeError):
        pass

    # OpenCV 도 자체 스레드풀을 쓴다.
    try:
        import cv2  # noqa: PLC0415

        cv2.setNumThreads(settings.effective_threads)
    except (ImportError, AttributeError):
        pass


_apply_thread_limits()


def _startup_diagnostics() -> dict[str, Any]:
    """기동 시 환경 진단 정보를 수집하고 로그에 남긴다.

    "이 환경에서 어떤 경로로 동작할 것인가" 를 기동 로그에 명시하면,
    나중에 "왜 정확도가 낮은가" 를 조사할 때 즉시 답이 나온다
    (가중치가 없어 폴백으로 돌고 있었다는 사실이 로그에 있다).

    Returns:
        진단 정보 dict.
    """
    settings = get_settings()
    weights = {
        "detector_pt": settings.weight_exists("detector_weights"),
        "detector_onnx": settings.weight_exists("detector_onnx"),
        "crnn": settings.weight_exists("crnn_weights"),
        "attention": settings.weight_exists("attention_weights"),
        "pdm_anomaly": settings.weight_exists("pdm_anomaly_weights"),
        "pdm_failure": settings.weight_exists("pdm_failure_weights"),
        "pdm_rul": settings.weight_exists("pdm_rul_weights"),
    }
    try:
        from app.core.fallback import fallback_available  # noqa: PLC0415

        fallback = fallback_available()
    except Exception as exc:  # noqa: BLE001
        fallback = {"error": str(exc)}

    info = {
        "version": __version__,
        "device": settings.resolved_device,
        "threads": settings.effective_threads,
        "weights": weights,
        "fallback": fallback,
        "any_weights": any(weights.values()),
    }
    if not info["any_weights"]:
        logger.warning(
            "학습된 가중치가 없습니다 — 모든 추론이 OpenCV/규칙 기반 폴백으로 동작합니다. "
            "응답의 detector/recognizer 에 '-fallback' 접미사가 붙습니다."
        )
    return info


@asynccontextmanager
async def lifespan(_app: Any) -> AsyncIterator[None]:
    """애플리케이션 생명주기 훅.

    왜 ``@app.on_event("startup")`` 대신 lifespan 인가?
        ``on_event`` 는 FastAPI 0.1xx 에서 deprecated 되었다. lifespan 은
        시작과 종료를 하나의 컨텍스트 매니저로 묶어 **자원 정리가 보장**된다
        (예외가 나도 ``finally`` 가 실행된다).

    기동 시 **모델을 로드하지 않는다.** 지연 로딩이 원칙이다 —
    첫 요청이 조금 느린 대신 기동이 즉시 끝나고, 쓰지 않는 모델이
    메모리를 점유하지 않는다.

    Args:
        _app: FastAPI 인스턴스(사용하지 않음).

    Yields:
        None.
    """
    settings = get_settings()
    configure_logging(settings.log_level)
    diagnostics = _startup_diagnostics()
    logger.info(
        "%s v%s 기동 (device=%s, threads=%d, weights=%s)",
        settings.app_name,
        __version__,
        diagnostics["device"],
        diagnostics["threads"],
        "있음" if diagnostics["any_weights"] else "없음(폴백 모드)",
        extra={"stage": "startup"},
    )
    try:
        yield
    finally:
        # 모델 캐시를 비워 재기동 시 깨끗한 상태를 보장한다.
        try:
            from app.core.registry import get_registry  # noqa: PLC0415

            get_registry().clear()
        except Exception:  # noqa: BLE001
            pass
        logger.info("%s 종료", settings.app_name, extra={"stage": "shutdown"})


def _error_body(status: int, error: str, message: str, path: str) -> dict[str, Any]:
    """계약서 6장 공통 오류 응답 본문을 만든다.

    백엔드의 ``GlobalExceptionHandler`` 와 **같은 형태**를 쓰는 이유:
    프론트엔드가 오류 포맷을 하나만 다루면 되기 때문이다.

    Args:
        status: HTTP 상태 코드.
        error: 오류 종류.
        message: 오류 메시지.
        path: 요청 경로.

    Returns:
        ``{"timestamp","status","error","message","path"}``.
    """
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "status": int(status),
        "error": str(error),
        "message": str(message),
        "path": str(path),
        "request_id": get_request_id(),
    }


def _contract_safe_response(path: str, detail: str) -> dict[str, Any] | None:
    """경로에 맞는 "계약 준수 실패 응답" 을 만든다.

    이것이 "500 금지" 의 핵심 장치다. 예외가 어디서 나든, 그 경로가 기대하는
    스키마로 **실패를 표현**해 200 으로 돌려준다.
        - LPR 경로   → ``status: NO_PLATE`` 응답
        - PdM 경로   → 중립 결과(health 60 / MEDIUM)
        - 벤치마크   → 오류 항목 1건이 담긴 배열
        - 그 외      → None(일반 오류 응답을 쓰라는 뜻)

    Args:
        path: 요청 경로.
        detail: 실패 사유.

    Returns:
        계약 준수 응답 dict 또는 None.
    """
    if "/lpr/recognize" in path:
        try:
            from app.routers.lpr import no_plate_response  # noqa: PLC0415

            settings = get_settings()
            payload = no_plate_response(
                detector=f"{settings.default_detector}-fallback",
                recognizer=f"{settings.default_recognizer}-fallback",
                reason=f"internal-error: {detail}",
            )
            if "frames" in path:
                payload["frames"] = 0
                payload["fusion"] = "majority-vote"
                payload["per_frame"] = []
            return payload
        except Exception:  # noqa: BLE001
            return None

    if "/pdm/predict" in path:
        try:
            from app.pdm.service import get_pdm_service  # noqa: PLC0415

            service = get_pdm_service()
            # 중립 결과 생성기를 직접 쓴다(예측을 재시도하지 않는다).
            return service._neutral_result(0, reason=detail).to_dict()  # noqa: SLF001
        except Exception:  # noqa: BLE001
            return None

    if "/models/benchmark" in path:
        return None  # 배열을 기대하므로 아래에서 별도 처리한다.

    return None


def create_app() -> Any:
    """FastAPI 애플리케이션을 생성한다.

    Returns:
        FastAPI 인스턴스.

    Raises:
        RuntimeError: FastAPI 미설치.
    """
    if not _HAS_FASTAPI:
        msg = (
            "FastAPI 가 설치되지 않았습니다. "
            "`pip install -r requirements.txt` 를 실행하세요."
        )
        raise RuntimeError(msg)

    settings = get_settings()
    configure_logging(settings.log_level)

    application = FastAPI(
        title="모빌리티 비전 AI 서비스",
        description=(
            "번호판 인식(LPR)과 예지보전(PdM) REST API.\n\n"
            "**가중치 파일이 없어도 전체 엔드포인트가 동작한다.** "
            "가중치가 없으면 OpenCV 기반 결정적 알고리즘과 도메인 규칙으로 "
            "자동 대체되며, 응답의 모델 이름에 `-fallback` 접미사가 붙는다. "
            "어떤 경우에도 500 을 반환하지 않는다."
        ),
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs" if settings.enable_docs else None,
        redoc_url="/redoc" if settings.enable_docs else None,
        openapi_url="/openapi.json" if settings.enable_docs else None,
    )

    # ---- CORS ----
    # 프론트엔드(5173)가 AI 서비스를 직접 호출하는 경우(디버깅/시연)와
    # 백엔드(8080)의 서버 간 호출을 모두 허용한다.
    origins = settings.cors_origin_list
    application.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        # allow_origins 가 ["*"] 일 때 allow_credentials=True 는 브라우저가 거부한다.
        # 와일드카드와 자격증명은 CORS 규격상 함께 쓸 수 없으므로 분기한다.
        allow_credentials=origins != ["*"],
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-Process-Time-Ms"],
    )

    # ---- 요청 ID + 접근 로그 미들웨어 ----
    @application.middleware("http")
    async def request_context(request: Request, call_next: Callable[..., Any]) -> Any:
        """요청마다 상관관계 ID 를 부여하고 처리 시간을 로깅한다.

        외부(백엔드/게이트웨이)에서 ``X-Request-ID`` 를 보내면 그것을 이어받는다.
        분산 추적에서 하나의 트랜잭션을 여러 서비스에 걸쳐 따라가려면
        ID 전파가 필수다.

        Args:
            request: 들어온 요청.
            call_next: 다음 핸들러.

        Returns:
            응답(헤더에 요청 ID 와 처리 시간 추가).
        """
        incoming = request.headers.get("X-Request-ID")
        request_id = set_request_id(incoming)
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            # 여기서 잡히는 예외는 아래 전역 핸들러로 전달된다.
            elapsed = (time.perf_counter() - started) * 1000.0
            logger.exception(
                "요청 처리 중 예외: %s %s (%.1fms)",
                request.method,
                request.url.path,
                elapsed,
                extra={"stage": "middleware", "ms": round(elapsed, 2)},
            )
            raise
        elapsed = (time.perf_counter() - started) * 1000.0
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time-Ms"] = f"{elapsed:.2f}"
        # 헬스체크는 수십 초마다 호출되므로 로그를 남기지 않는다(노이즈 방지).
        if not request.url.path.startswith(("/health", "/healthz")):
            logger.info(
                "%s %s → %d (%.1fms)",
                request.method,
                request.url.path,
                response.status_code,
                elapsed,
                extra={"stage": "access", "ms": round(elapsed, 2), "status": response.status_code},
            )
        return response

    # ---- 전역 예외 처리 ----
    @application.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> Any:
        """처리되지 않은 모든 예외를 계약 준수 응답으로 변환한다.

        계약서: "절대 500 을 던지지 않는다." LPR/PdM 경로는 각자의 스키마로
        실패를 표현해 **200** 으로 응답한다. 그 외 경로는 공통 오류 포맷으로
        500 을 돌려준다(그런 경로는 백엔드 계약에 없으므로 안전하다).

        Args:
            request: 요청.
            exc: 발생한 예외.

        Returns:
            JSONResponse.
        """
        path = str(request.url.path)
        detail = f"{type(exc).__name__}: {exc}"
        logger.exception("처리되지 않은 예외: %s", path, extra={"stage": "error"})

        if "/models/benchmark" in path:
            return JSONResponse(
                status_code=200,
                content=[
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
                        "detail": {"error": detail},
                    }
                ],
            )

        safe = _contract_safe_response(path, detail)
        if safe is not None:
            return JSONResponse(status_code=200, content=safe)

        return JSONResponse(
            status_code=500,
            content=_error_body(500, "Internal Server Error", detail, path),
        )

    @application.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> Any:
        """요청 검증 실패(422)를 공통 오류 포맷으로 변환한다.

        422 는 **유지한다** — 500 과 달리 "호출자가 요청을 잘못 보냈다" 는
        뜻이고, 이를 200 으로 감추면 백엔드 개발자가 버그를 발견할 수 없다.
        다만 LPR 경로는 예외로 둔다: 게이트 운영 중 파라미터 오류로 차량이
        막히는 것을 막기 위해 기본값으로 처리하고 NO_PLATE 를 돌려준다.

        Args:
            request: 요청.
            exc: 검증 예외.

        Returns:
            JSONResponse.
        """
        path = str(request.url.path)
        try:
            errors = exc.errors()
            message = "; ".join(
                f"{'.'.join(str(p) for p in e.get('loc', []))}: {e.get('msg', '')}"
                for e in errors[:5]
            )
        except Exception:  # noqa: BLE001
            message = str(exc)

        logger.info("요청 검증 실패: %s — %s", path, message, extra={"stage": "validation"})

        if "/lpr/recognize" in path:
            safe = _contract_safe_response(path, f"validation-error: {message}")
            if safe is not None:
                return JSONResponse(status_code=200, content=safe)

        return JSONResponse(
            status_code=422,
            content=_error_body(422, "Unprocessable Entity", message, path),
        )

    @application.exception_handler(StarletteHTTPException)
    async def http_exception_handler(
        request: Request, exc: StarletteHTTPException
    ) -> Any:
        """HTTP 예외(404 등)를 공통 오류 포맷으로 변환한다.

        Args:
            request: 요청.
            exc: HTTP 예외.

        Returns:
            JSONResponse.
        """
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_body(
                exc.status_code,
                str(getattr(exc, "detail", "HTTP Error")),
                str(getattr(exc, "detail", "")),
                str(request.url.path),
            ),
        )

    # ---- 라우터 등록 ----
    from app.routers import health as health_router  # noqa: PLC0415
    from app.routers import lpr as lpr_router  # noqa: PLC0415
    from app.routers import models as models_router  # noqa: PLC0415
    from app.routers import pdm as pdm_router  # noqa: PLC0415

    # health 는 prefix 없음(계약서 1장), 나머지는 /api/v1 prefix 를
    # 각 라우터가 자체적으로 선언한다.
    application.include_router(health_router.router)
    application.include_router(lpr_router.router)
    application.include_router(pdm_router.router)
    application.include_router(models_router.router)

    @application.get("/", include_in_schema=False)
    def root() -> dict[str, Any]:
        """루트 경로 — 서비스 소개와 주요 엔드포인트 목록.

        왜 두는가? 개발자가 ``localhost:8000`` 을 열었을 때 404 가 아니라
        "무엇을 할 수 있는지" 를 보여 주면 실습 경험이 훨씬 낫다.

        Returns:
            서비스 소개 dict.
        """
        return {
            "service": "모빌리티 비전 AI 서비스",
            "version": __version__,
            "docs": "/docs" if settings.enable_docs else None,
            "endpoints": {
                "health": "GET /health",
                "lpr_recognize": "POST /api/v1/lpr/recognize",
                "lpr_recognize_frames": "POST /api/v1/lpr/recognize-frames",
                "pdm_predict": "POST /api/v1/pdm/predict",
                "models_info": "GET /api/v1/models/info",
                "models_benchmark": "POST /api/v1/models/benchmark",
            },
        }

    return application


#: ASGI 애플리케이션. FastAPI 가 없으면 None 이 되어, 순수 로직 테스트가
#: 이 모듈 import 때문에 실패하지 않는다.
app: Any = create_app() if _HAS_FASTAPI else None


def main() -> int:
    """개발용 실행 진입점(``python -m app.main``).

    운영에서는 ``uvicorn app.main:app --workers N`` 을 쓰는 것이 맞다.
    이 함수는 실습 편의를 위한 것이다.

    Returns:
        종료 코드.
    """
    if not _HAS_FASTAPI:
        print("FastAPI 가 설치되지 않았습니다: pip install -r requirements.txt")
        return 2
    try:
        import uvicorn  # noqa: PLC0415
    except ImportError:
        print("uvicorn 이 설치되지 않았습니다: pip install -r requirements.txt")
        return 2

    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=int(settings.port),
        log_level=str(settings.log_level).lower(),
        reload=False,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["app", "create_app", "lifespan", "main"]
