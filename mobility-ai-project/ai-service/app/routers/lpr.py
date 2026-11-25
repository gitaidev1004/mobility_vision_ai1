"""번호판 인식(LPR) 라우터.

계약서 3장:
    - ``POST /api/v1/lpr/recognize`` (multipart/form-data)
      form: ``file``(필수), ``min_conf``(0.25), ``detector``(yolo), ``recognizer``(crnn)
    - ``POST /api/v1/lpr/recognize-frames`` (multipart, ``files`` 다중)

**절대 500 을 내지 않는다** 는 요구사항을 이 라우터에서 어떻게 보장하는가:
    1. 파이프라인이 모든 내부 예외를 흡수한다(``LprPipeline.run``).
    2. 이미지 디코딩 실패는 ``NO_PLATE`` 응답으로 표현한다(예외가 아니다).
    3. 잘못된 ``detector``/``recognizer`` 값은 기본값으로 대체한다.
    4. 그래도 남는 예외는 ``main.py`` 의 전역 핸들러가 200 + ``NO_PLATE`` 로 변환한다.

왜 잘못된 파라미터를 422 로 거부하지 않는가?
    게이트 시스템에서 "파라미터 오타 때문에 차량이 통과하지 못하는 것" 은
    최악의 실패다. 알 수 없는 값은 기본값으로 처리하고 응답에 실제 사용한
    모델명을 담아, 호출자가 사후에 확인할 수 있게 하는 편이 안전하다.
"""

from __future__ import annotations

from typing import Any

from app.config import get_settings
from app.core.logging import get_logger
from app.detection.preprocess import decode_image_bytes
from app.ocr.recognizer import ALLOWED_RECOGNIZERS
from app.pipeline.lpr_pipeline import get_pipeline
from app.pipeline.quality import QualityReport

logger = get_logger(__name__)

#: 계약서가 허용하는 검출기 이름.
ALLOWED_DETECTORS: tuple[str, ...] = ("yolo", "mobilenet")

#: 계약서가 허용하는 융합 전략.
ALLOWED_FUSIONS: tuple[str, ...] = ("majority-vote", "char-vote", "best-confidence")

try:
    from fastapi import APIRouter, File, Form, UploadFile

    router: Any = APIRouter(prefix="/api/v1/lpr", tags=["lpr"])
    _HAS_FASTAPI = True
except ImportError:  # pragma: no cover
    router = None
    _HAS_FASTAPI = False


def normalize_detector(name: str | None) -> str:
    """검출기 이름을 허용 목록으로 정규화한다.

    Args:
        name: 요청값.

    Returns:
        허용된 이름(알 수 없으면 설정 기본값).
    """
    settings = get_settings()
    value = str(name or "").strip().lower()
    if value in ALLOWED_DETECTORS:
        return value
    if value:
        logger.info("알 수 없는 detector '%s' → 기본값 사용", value)
    return str(settings.default_detector).strip().lower()


def normalize_recognizer(name: str | None) -> str:
    """인식기 이름을 허용 목록으로 정규화한다.

    Args:
        name: 요청값.

    Returns:
        허용된 이름(알 수 없으면 설정 기본값).
    """
    settings = get_settings()
    value = str(name or "").strip().lower()
    if value in ALLOWED_RECOGNIZERS:
        return value
    if value:
        logger.info("알 수 없는 recognizer '%s' → 기본값 사용", value)
    return str(settings.default_recognizer).strip().lower()


def normalize_min_conf(value: float | None) -> float:
    """최소 신뢰도를 0~1 로 클램프한다.

    Args:
        value: 요청값.

    Returns:
        0.0~1.0 범위의 값(None 이면 설정 기본값).
    """
    settings = get_settings()
    if value is None:
        return float(settings.default_min_conf)
    try:
        return float(min(1.0, max(0.0, float(value))))
    except (TypeError, ValueError):
        return float(settings.default_min_conf)


def normalize_fusion(value: str | None) -> str:
    """융합 전략을 허용 목록으로 정규화한다.

    Args:
        value: 요청값.

    Returns:
        허용된 전략(알 수 없으면 ``"majority-vote"``).
    """
    strategy = str(value or "").strip().lower()
    return strategy if strategy in ALLOWED_FUSIONS else "majority-vote"


def exposed_model_names(detector: str, recognizer: str) -> tuple[str, str]:
    """실제 사용될 모델의 노출 이름을 구한다(``-fallback`` 접미사 포함).

    왜 이 함수가 필요한가 — 실측 불일치:
        정상 경로의 응답은 ``yolo-fallback`` 을 보고하는데, 이미지 디코딩 실패
        같은 **조기 반환 경로**는 요청값 ``yolo`` 를 그대로 돌려줬다.
        같은 서비스가 같은 환경에서 두 가지 이름을 보고하면, 백엔드가
        "폴백으로 처리된 건수" 를 집계할 때 숫자가 어긋난다.
        조기 반환에서도 같은 이름을 쓰도록 이름 결정을 한 곳에 모은다.

    모델을 로드하지 않고 **가중치 파일 존재 여부**로 판단한다 —
    디코딩조차 실패한 요청에서 모델을 로드하는 것은 낭비다.

    Args:
        detector: 정규화된 검출기 이름.
        recognizer: 정규화된 인식기 이름.

    Returns:
        ``(검출기 노출 이름, 인식기 노출 이름)``.
    """
    settings = get_settings()

    det_has_weights = settings.weight_exists("detector_weights") or settings.weight_exists(
        "detector_onnx"
    )
    if str(detector) == "mobilenet":
        # MobileNet 은 규약 경로(plate_ssdlite_mobilenet.pt)를 먼저 찾는다.
        det_has_weights = (settings.model_root / "plate_ssdlite_mobilenet.pt").is_file() or (
            settings.model_root / "plate_mobilenet.pt"
        ).is_file()
    det_name = str(detector) if det_has_weights else f"{detector}-fallback"

    ocr_weight_attr = {"crnn": "crnn_weights", "attention": "attention_weights"}.get(
        str(recognizer)
    )
    if ocr_weight_attr is not None:
        ocr_has_weights = settings.weight_exists(ocr_weight_attr)
    else:
        # paddle 은 가중치 파일이 아니라 패키지 설치 여부로 판단한다.
        try:
            from app.ocr import paddle_adapter  # noqa: PLC0415

            ocr_has_weights = paddle_adapter.is_available()
        except Exception:  # noqa: BLE001
            ocr_has_weights = False
    ocr_name = str(recognizer) if ocr_has_weights else f"{recognizer}-fallback"

    return det_name, ocr_name


def no_plate_response(
    *,
    detector: str,
    recognizer: str,
    reason: str,
) -> dict[str, Any]:
    """이미지를 처리할 수 없을 때의 ``NO_PLATE`` 응답을 만든다.

    계약서 스키마를 **완전히** 채운다 — 필드를 빼면 백엔드의 DTO 역직렬화가
    실패하거나 프론트엔드에서 undefined 접근 오류가 난다.

    모델 이름은 ``exposed_model_names`` 로 보정해, 정상 경로와 동일한 형태
    (``yolo-fallback`` 등)로 보고한다. 이미 접미사가 붙어 있으면 유지한다.

    Args:
        detector: 요청된 검출기 이름.
        recognizer: 요청된 인식기 이름.
        reason: 품질 필드에 담을 사유.

    Returns:
        계약서 LPR 응답 스키마 dict.
    """
    if detector.endswith("-fallback") or recognizer.endswith("-fallback"):
        det_name, ocr_name = detector, recognizer
    else:
        det_name, ocr_name = exposed_model_names(detector, recognizer)

    quality = QualityReport(
        score=0.0,
        blur=0.0,
        brightness=0.0,
        contrast=0.0,
        plate_px_width=0,
        readable=False,
        reason=reason,
    )
    return {
        "plate_number": None,
        "confidence": 0.0,
        "status": "NO_PLATE",
        "valid_format": False,
        "detector": det_name,
        "recognizer": ocr_name,
        "elapsed_ms": {"detect": 0.0, "ocr": 0.0, "total": 0.0},
        "quality": quality.to_dict(),
        "detections": [],
        "candidates": [],
        "char_confidences": [],
    }


def recognize_bytes(
    data: bytes,
    *,
    detector: str = "yolo",
    recognizer: str = "crnn",
    min_conf: float | None = None,
) -> dict[str, Any]:
    """이미지 바이트에서 번호판을 인식한다(HTTP 와 무관한 재사용 함수).

    Args:
        data: 이미지 파일 바이트.
        detector: 검출기 이름.
        recognizer: 인식기 이름.
        min_conf: 최소 신뢰도.

    Returns:
        계약서 LPR 응답 dict.
    """
    det = normalize_detector(detector)
    ocr = normalize_recognizer(recognizer)
    conf = normalize_min_conf(min_conf)

    settings = get_settings()
    if not data:
        return no_plate_response(detector=det, recognizer=ocr, reason="empty-upload")
    if len(data) > int(settings.max_upload_bytes):
        # 거부 대신 NO_PLATE 로 응답한다 — 413 을 던지면 백엔드가 별도 처리를
        # 해야 하고, 게이트 운영에서는 "처리 못 함" 이 곧 "번호판 없음" 이다.
        return no_plate_response(
            detector=det,
            recognizer=ocr,
            reason=f"upload-too-large({len(data)}>{settings.max_upload_bytes})",
        )

    image = decode_image_bytes(data)
    if image is None:
        return no_plate_response(detector=det, recognizer=ocr, reason="image-decode-failed")

    result = get_pipeline().run(image, detector=det, recognizer=ocr, min_conf=conf)
    return result.to_dict()


def recognize_frames_bytes(
    frames: list[bytes],
    *,
    detector: str = "yolo",
    recognizer: str = "crnn",
    min_conf: float | None = None,
    fusion: str = "majority-vote",
) -> dict[str, Any]:
    """여러 프레임 바이트를 인식하고 융합한다.

    Args:
        frames: 이미지 바이트 리스트.
        detector: 검출기 이름.
        recognizer: 인식기 이름.
        min_conf: 최소 신뢰도.
        fusion: 융합 전략.

    Returns:
        계약서 다중 프레임 응답 dict.
    """
    det = normalize_detector(detector)
    ocr = normalize_recognizer(recognizer)
    conf = normalize_min_conf(min_conf)
    strategy = normalize_fusion(fusion)

    settings = get_settings()
    images = []
    for data in (frames or [])[: int(settings.max_frames)]:
        if not data or len(data) > int(settings.max_upload_bytes):
            continue
        decoded = decode_image_bytes(data)
        if decoded is not None:
            images.append(decoded)

    if not images:
        payload = no_plate_response(
            detector=det, recognizer=ocr, reason="no-decodable-frames"
        )
        payload["frames"] = 0
        payload["fusion"] = strategy
        payload["per_frame"] = []
        return payload

    return get_pipeline().run_frames(
        images, detector=det, recognizer=ocr, min_conf=conf, fusion=strategy
    )


if _HAS_FASTAPI:

    @router.post(
        "/recognize",
        summary="단일 이미지 번호판 인식",
        description=(
            "이미지 1장에서 번호판을 검출·인식한다. "
            "가중치가 없으면 OpenCV 폴백으로 동작하며 detector/recognizer 에 "
            "`-fallback` 접미사가 붙는다. 어떤 경우에도 500 을 반환하지 않는다."
        ),
    )
    async def post_recognize(
        file: UploadFile = File(..., description="번호판이 포함된 이미지"),
        min_conf: float = Form(0.25, description="검출 최소 신뢰도(0~1)"),
        detector: str = Form("yolo", description="yolo | mobilenet"),
        recognizer: str = Form("crnn", description="crnn | attention | paddle"),
    ) -> dict[str, Any]:
        """``POST /api/v1/lpr/recognize`` 핸들러.

        Args:
            file: 업로드 이미지.
            min_conf: 검출 최소 신뢰도.
            detector: 검출기 이름.
            recognizer: 인식기 이름.

        Returns:
            계약서 LPR 응답.
        """
        det = normalize_detector(detector)
        ocr = normalize_recognizer(recognizer)
        try:
            data = await file.read()
        except Exception as exc:  # noqa: BLE001 - 업로드 스트림 오류
            logger.warning("업로드 읽기 실패: %s", exc)
            return no_plate_response(
                detector=det, recognizer=ocr, reason=f"upload-read-error: {type(exc).__name__}"
            )
        return recognize_bytes(data, detector=det, recognizer=ocr, min_conf=min_conf)

    @router.post(
        "/recognize-frames",
        summary="다중 프레임 번호판 인식(융합)",
        description=(
            "여러 프레임을 인식하고 다수결/자리투표로 융합한다. "
            "단일 프레임보다 정확도가 높다. 응답은 단일 스키마 + frames/fusion/per_frame."
        ),
    )
    async def post_recognize_frames(
        files: list[UploadFile] = File(..., description="번호판 프레임 이미지 여러 장"),
        min_conf: float = Form(0.25, description="검출 최소 신뢰도(0~1)"),
        detector: str = Form("yolo", description="yolo | mobilenet"),
        recognizer: str = Form("crnn", description="crnn | attention | paddle"),
        fusion: str = Form(
            "majority-vote", description="majority-vote | char-vote | best-confidence"
        ),
    ) -> dict[str, Any]:
        """``POST /api/v1/lpr/recognize-frames`` 핸들러.

        Args:
            files: 업로드 이미지 리스트.
            min_conf: 검출 최소 신뢰도.
            detector: 검출기 이름.
            recognizer: 인식기 이름.
            fusion: 융합 전략.

        Returns:
            계약서 다중 프레임 응답.
        """
        det = normalize_detector(detector)
        ocr = normalize_recognizer(recognizer)
        payloads: list[bytes] = []
        for upload in files or []:
            try:
                payloads.append(await upload.read())
            except Exception as exc:  # noqa: BLE001 - 한 프레임 실패가 전체를 막지 않는다
                logger.info("프레임 읽기 실패(건너뜀): %s", exc)
                continue
        return recognize_frames_bytes(
            payloads, detector=det, recognizer=ocr, min_conf=min_conf, fusion=fusion
        )


__all__ = [
    "ALLOWED_DETECTORS",
    "ALLOWED_FUSIONS",
    "exposed_model_names",
    "no_plate_response",
    "normalize_detector",
    "normalize_fusion",
    "normalize_min_conf",
    "normalize_recognizer",
    "recognize_bytes",
    "recognize_frames_bytes",
    "router",
]
