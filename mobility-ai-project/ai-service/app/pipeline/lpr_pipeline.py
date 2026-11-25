"""LPR(번호판 인식) 전체 파이프라인: 검출 → 보정 → OCR → 후처리.

파이프라인이 별도 모듈인 이유:
    라우터(``app.routers.lpr``)는 HTTP 관심사(멀티파트 파싱, 응답 직렬화)만
    담당해야 한다. "번호판을 어떻게 읽는가" 는 HTTP 와 무관한 도메인 로직이므로
    여기에 둔다. 덕분에 파이프라인을 CLI(벤치마크)나 배치 작업에서도 재사용할 수 있다.

핵심 설계 — **후보 경쟁(candidate competition)**:
    검출기는 여러 후보를 준다. 특히 폴백 검출기는 번호판 높이 가설을 여러 개
    만들어 낸다(``PLATE_HEIGHT_HYPOTHESES``). 1위 박스만 OCR 하면 그 박스가
    약간 어긋났을 때 전체가 실패한다. 상위 N개를 모두 OCR 해 보고
    **"번호판 형식을 만족하면서 신뢰도가 가장 높은" 결과**를 고르면 재현율이 크게 오른다.
    OCR 비용이 검출보다 훨씬 싸기 때문에 가능한 전략이다.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from app.config import get_settings
from app.core.logging import get_logger
from app.detection.base import Detection
from app.detection.preprocess import crop, ensure_bgr
from app.ocr.postprocess import FrameCandidate, PlateResult, fuse_frames, postprocess
from app.pipeline.quality import QualityReport, assess
from app.pipeline.rectify import rectify_plate

logger = get_logger(__name__)

#: OCR 을 시도할 최대 검출 후보 수.
#: 3 으로 둔 이유: 폴백 검출기의 높이 가설이 3개라 1순위 밴드의 모든 가설을
#: 커버한다. 더 늘려도 이득이 적고 지연만 증가한다.
MAX_OCR_CANDIDATES: int = 3

#: 검출 박스를 OCR 전에 확장할 비율. 문자 주변 여백이 이진화 임계값을 안정시킨다.
CROP_EXPAND_RATIO: float = 0.04


@dataclass(slots=True)
class LprResult:
    """LPR 파이프라인 결과(계약서 응답 스키마와 1:1 대응).

    Attributes:
        plate_number: 최종 번호판 문자열(미검출/실패 시 None).
        confidence: 0~1 신뢰도.
        status: ``SUCCESS`` | ``LOW_CONFIDENCE`` | ``NO_PLATE`` | ``FORMAT_INVALID``.
        valid_format: 번호판 정규식 통과 여부.
        detector: 사용한 검출기 노출 이름(폴백이면 ``-fallback``).
        recognizer: 사용한 인식기 노출 이름.
        elapsed_ms: ``{"detect", "ocr", "total"}`` 밀리초.
        quality: 품질 평가 결과(계약서 ``quality`` 필드).
        detections: 검출 박스 리스트.
        candidates: OCR 후보 리스트.
        char_confidences: 문자별 신뢰도.
        corrections: 후처리 보정 내역.
        rectify: 보정 단계 메타데이터.
    """

    plate_number: str | None = None
    confidence: float = 0.0
    status: str = "NO_PLATE"
    valid_format: bool = False
    detector: str = ""
    recognizer: str = ""
    elapsed_ms: dict[str, float] = field(default_factory=dict)
    quality: dict[str, Any] = field(default_factory=dict)
    detections: list[dict[str, Any]] = field(default_factory=list)
    candidates: list[dict[str, Any]] = field(default_factory=list)
    char_confidences: list[float] = field(default_factory=list)
    corrections: list[dict[str, Any]] = field(default_factory=list)
    rectify: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """계약서 ``POST /api/v1/lpr/recognize`` 응답 스키마로 직렬화한다.

        Returns:
            계약서와 키/타입이 일치하는 dict.
        """
        return {
            "plate_number": self.plate_number,
            "confidence": round(float(self.confidence), 4),
            "status": self.status,
            "valid_format": bool(self.valid_format),
            "detector": self.detector,
            "recognizer": self.recognizer,
            "elapsed_ms": {k: round(float(v), 2) for k, v in self.elapsed_ms.items()},
            "quality": self.quality,
            "detections": self.detections,
            "candidates": self.candidates,
            "char_confidences": [round(float(c), 4) for c in self.char_confidences],
        }


class LprPipeline:
    """번호판 인식 파이프라인.

    사용 예::

        pipeline = LprPipeline()
        result = pipeline.run(image, detector="yolo", recognizer="crnn", min_conf=0.25)
        payload = result.to_dict()

    모든 단계가 예외를 흡수하므로 ``run`` 은 어떤 입력에도 유효한 결과를 돌려준다.
    """

    def __init__(self, *, max_ocr_candidates: int = MAX_OCR_CANDIDATES) -> None:
        """파이프라인을 초기화한다.

        Args:
            max_ocr_candidates: OCR 을 시도할 최대 검출 후보 수.
        """
        self.max_ocr_candidates = max(1, int(max_ocr_candidates))

    # -----------------------------------------------------------------
    # 단일 이미지
    # -----------------------------------------------------------------

    def run(
        self,
        image: np.ndarray,
        *,
        detector: str = "yolo",
        recognizer: str = "crnn",
        min_conf: float | None = None,
    ) -> LprResult:
        """단일 이미지에 대해 전체 파이프라인을 실행한다.

        Args:
            image: BGR uint8 이미지.
            detector: ``"yolo"`` | ``"mobilenet"``.
            recognizer: ``"crnn"`` | ``"attention"`` | ``"paddle"``.
            min_conf: 검출 최소 신뢰도. None 이면 설정 기본값.

        Returns:
            LprResult. 어떤 실패에도 예외를 던지지 않는다.
        """
        settings = get_settings()
        threshold = float(settings.default_min_conf if min_conf is None else min_conf)
        started = time.perf_counter()

        try:
            bgr = ensure_bgr(image)
        except (ValueError, TypeError) as exc:
            logger.warning("이미지 정규화 실패: %s", exc)
            total = (time.perf_counter() - started) * 1000.0
            return LprResult(
                status="NO_PLATE",
                detector=detector,
                recognizer=recognizer,
                elapsed_ms={"detect": 0.0, "ocr": 0.0, "total": round(total, 2)},
                quality=QualityReport(
                    score=0.0, blur=0.0, brightness=0.0, contrast=0.0,
                    plate_px_width=0, readable=False, reason="invalid-image",
                ).to_dict(),
            )

        # ---- 1. 검출 ----
        from app.core.registry import get_detector, get_ocr  # noqa: PLC0415 - 순환 import 방지

        det_model = get_detector(detector)
        detect_started = time.perf_counter()
        detections = det_model.detect(bgr, threshold)
        detect_ms = (time.perf_counter() - detect_started) * 1000.0

        ocr_model = get_ocr(recognizer)
        detector_name = getattr(det_model, "exposed_name", detector)
        recognizer_name = getattr(ocr_model, "exposed_name", recognizer)

        if not detections:
            # 번호판 미검출: 전체 이미지 품질을 평가해 "왜 못 찾았는지" 를 알려 준다.
            quality = assess(
                bgr,
                plate_px_width=0,
                blur_min=settings.quality_blur_min,
                brightness_min=settings.quality_brightness_min,
                brightness_max=settings.quality_brightness_max,
                plate_px_min=settings.quality_plate_px_min,
            )
            total = (time.perf_counter() - started) * 1000.0
            return LprResult(
                plate_number=None,
                confidence=0.0,
                status="NO_PLATE",
                valid_format=False,
                detector=detector_name,
                recognizer=recognizer_name,
                elapsed_ms={
                    "detect": round(detect_ms, 2),
                    "ocr": 0.0,
                    "total": round(total, 2),
                },
                quality=quality.to_dict(),
                detections=[],
                candidates=[],
            )

        # ---- 2~4. 후보별 보정 → OCR → 후처리 (후보 경쟁) ----
        img_h, img_w = bgr.shape[:2]
        ocr_started = time.perf_counter()

        attempts: list[tuple[PlateResult, Detection, QualityReport, dict[str, Any], list[dict[str, Any]]]] = []
        for detection in detections[: self.max_ocr_candidates]:
            expanded = detection.expand(CROP_EXPAND_RATIO, img_w, img_h)
            plate_crop = crop(bgr, expanded.bbox)
            if plate_crop.size == 0:
                continue

            # 2. 보정: 검출기가 4점을 줬으면 투시, 아니면 기울기 회전.
            rect = rectify_plate(
                plate_crop,
                quad=detection.quad,
                angle=detection.angle if abs(detection.angle) > 0.5 else None,
            )

            # 3. OCR
            recognition = ocr_model.recognize(rect.image)

            # 4. 후처리(정규식 보정 + 상태 판정)
            processed = postprocess(
                recognition.text,
                recognition.confidence,
                char_confidences=recognition.char_confidences,
                low_conf_threshold=settings.low_conf_threshold,
            )

            quality = assess(
                rect.image,
                plate_px_width=expanded.width,
                blur_min=settings.quality_blur_min,
                brightness_min=settings.quality_brightness_min,
                brightness_max=settings.quality_brightness_max,
                plate_px_min=settings.quality_plate_px_min,
            )
            attempts.append(
                (processed, detection, quality, rect.to_dict(), list(recognition.candidates))
            )

        ocr_ms = (time.perf_counter() - ocr_started) * 1000.0

        if not attempts:
            quality = assess(bgr, plate_px_width=0)
            total = (time.perf_counter() - started) * 1000.0
            return LprResult(
                status="NO_PLATE",
                detector=detector_name,
                recognizer=recognizer_name,
                elapsed_ms={
                    "detect": round(detect_ms, 2),
                    "ocr": round(ocr_ms, 2),
                    "total": round(total, 2),
                },
                quality=quality.to_dict(),
                detections=[d.to_dict() for d in detections],
            )

        best_processed, best_detection, best_quality, best_rect, best_candidates = max(
            attempts, key=lambda item: _attempt_score(item[0], item[1], item[2])
        )

        # 후보 목록: 모든 시도의 결과를 신뢰도순으로 모은다(운영자 검수용).
        candidates: list[dict[str, Any]] = []
        seen: set[str] = set()
        for processed, _det, _q, _r, raw_candidates in attempts:
            for item in [
                {"text": processed.text, "confidence": processed.confidence},
                *raw_candidates,
            ]:
                text = str(item.get("text") or "")
                if not text or text in seen:
                    continue
                seen.add(text)
                candidates.append(
                    {"text": text, "confidence": round(float(item.get("confidence") or 0.0), 4)}
                )
        candidates.sort(key=lambda c: c["confidence"], reverse=True)

        total = (time.perf_counter() - started) * 1000.0
        return LprResult(
            plate_number=best_processed.text or None,
            confidence=best_processed.confidence,
            status=best_processed.status,
            valid_format=best_processed.valid_format,
            detector=detector_name,
            recognizer=recognizer_name,
            elapsed_ms={
                "detect": round(detect_ms, 2),
                "ocr": round(ocr_ms, 2),
                "total": round(total, 2),
            },
            quality=best_quality.to_dict(),
            detections=[d.to_dict() for d in detections],
            candidates=candidates[:5],
            char_confidences=list(best_processed.char_confidences),
            corrections=list(best_processed.corrections),
            rectify=best_rect,
        )

    # -----------------------------------------------------------------
    # 다중 프레임
    # -----------------------------------------------------------------

    def run_frames(
        self,
        images: list[np.ndarray],
        *,
        detector: str = "yolo",
        recognizer: str = "crnn",
        min_conf: float | None = None,
        fusion: str = "majority-vote",
    ) -> dict[str, Any]:
        """여러 프레임을 처리하고 결과를 융합한다.

        계약서 ``POST /api/v1/lpr/recognize-frames`` 응답 스키마를 만든다:
            단일 응답 스키마 + ``frames``, ``fusion``, ``per_frame``.

        Args:
            images: BGR 이미지 리스트.
            detector: 검출기 이름.
            recognizer: 인식기 이름.
            min_conf: 검출 최소 신뢰도.
            fusion: ``"majority-vote"`` | ``"char-vote"`` | ``"best-confidence"``.

        Returns:
            계약서 스키마에 맞는 dict.
        """
        settings = get_settings()
        frames = list(images or [])[: max(1, int(settings.max_frames))]
        started = time.perf_counter()

        per_frame: list[dict[str, Any]] = []
        results: list[LprResult] = []
        for image in frames:
            result = self.run(
                image, detector=detector, recognizer=recognizer, min_conf=min_conf
            )
            results.append(result)
            per_frame.append(
                {
                    "plate_number": result.plate_number,
                    "confidence": round(float(result.confidence), 4),
                    "status": result.status,
                    "valid_format": result.valid_format,
                    "quality_score": result.quality.get("score", 0.0),
                    "elapsed_ms": result.elapsed_ms,
                }
            )

        if not results:
            return {
                **LprResult(status="NO_PLATE", detector=detector, recognizer=recognizer).to_dict(),
                "frames": 0,
                "fusion": fusion,
                "per_frame": [],
            }

        # 융합: 각 프레임의 결과를 품질 점수로 가중해 투표한다.
        # 흐린 프레임의 표가 선명한 프레임과 같은 무게를 가지면 안 된다.
        candidates = [
            FrameCandidate(
                text=r.plate_number or "",
                confidence=float(r.confidence),
                quality=float(r.quality.get("score") or 0.0) or 0.3,
            )
            for r in results
        ]
        fused = fuse_frames(
            candidates, low_conf_threshold=settings.low_conf_threshold, strategy=fusion
        )

        # 메타데이터는 "가장 좋은 프레임" 의 것을 쓴다.
        best = max(results, key=lambda r: (r.valid_format, r.confidence))
        total = (time.perf_counter() - started) * 1000.0

        payload = LprResult(
            plate_number=fused.text or None,
            confidence=fused.confidence,
            status=fused.status,
            valid_format=fused.valid_format,
            detector=best.detector,
            recognizer=best.recognizer,
            elapsed_ms={
                "detect": round(sum(float(r.elapsed_ms.get("detect", 0.0)) for r in results), 2),
                "ocr": round(sum(float(r.elapsed_ms.get("ocr", 0.0)) for r in results), 2),
                "total": round(total, 2),
            },
            quality=best.quality,
            detections=best.detections,
            candidates=best.candidates,
            char_confidences=list(fused.char_confidences or best.char_confidences),
            corrections=list(fused.corrections),
        ).to_dict()

        payload["frames"] = len(results)
        payload["fusion"] = str(fusion)
        payload["per_frame"] = per_frame
        return payload


def _attempt_score(
    processed: PlateResult, detection: Detection, quality: QualityReport
) -> tuple[int, float]:
    """후보 경쟁에서 쓸 정렬 키를 만든다.

    **형식 유효성을 1순위로 둔다.** 형식이 맞는 결과는 DB 의 차량과 매칭할 수
    있으므로 운영 가치가 있고, 형식이 틀린 고신뢰도 결과는 쓸모가 없다.
    따라서 (유효성, 가중점수) 튜플로 정렬한다.

    2순위 점수는 OCR 신뢰도 x 검출 신뢰도 x 품질을 혼합한다. 세 신호가 모두
    높아야 정말 좋은 후보이므로 곱에 가까운 가중합을 쓴다.

    Args:
        processed: 후처리 결과.
        detection: 해당 후보의 검출 결과.
        quality: 해당 crop 의 품질 평가.

    Returns:
        ``(형식유효 1/0, 가중 점수)``.
    """
    valid = 1 if processed.valid_format else 0
    score = (
        0.60 * float(processed.confidence)
        + 0.25 * float(detection.score)
        + 0.15 * float(quality.score)
    )
    return valid, score


#: 파이프라인 싱글턴. 내부 상태가 거의 없지만, 재사용으로 객체 생성을 줄인다.
_PIPELINE: LprPipeline | None = None


def get_pipeline() -> LprPipeline:
    """파이프라인 싱글턴을 반환한다.

    Returns:
        LprPipeline.
    """
    global _PIPELINE  # noqa: PLW0603 - 프로세스 싱글턴
    if _PIPELINE is None:
        _PIPELINE = LprPipeline()
    return _PIPELINE


def recognize_image(
    image: np.ndarray,
    *,
    detector: str = "yolo",
    recognizer: str = "crnn",
    min_conf: float | None = None,
) -> dict[str, Any]:
    """단일 이미지 인식 편의 함수(계약서 응답 dict 를 바로 반환).

    Args:
        image: BGR 이미지.
        detector: 검출기 이름.
        recognizer: 인식기 이름.
        min_conf: 최소 신뢰도.

    Returns:
        계약서 응답 dict.
    """
    return get_pipeline().run(
        image, detector=detector, recognizer=recognizer, min_conf=min_conf
    ).to_dict()


def recognize_frames(
    images: list[np.ndarray],
    *,
    detector: str = "yolo",
    recognizer: str = "crnn",
    min_conf: float | None = None,
    fusion: str = "majority-vote",
) -> dict[str, Any]:
    """다중 프레임 인식 편의 함수.

    Args:
        images: BGR 이미지 리스트.
        detector: 검출기 이름.
        recognizer: 인식기 이름.
        min_conf: 최소 신뢰도.
        fusion: 융합 전략.

    Returns:
        계약서 응답 dict.
    """
    return get_pipeline().run_frames(
        images, detector=detector, recognizer=recognizer, min_conf=min_conf, fusion=fusion
    )


__all__ = [
    "CROP_EXPAND_RATIO",
    "MAX_OCR_CANDIDATES",
    "LprPipeline",
    "LprResult",
    "get_pipeline",
    "recognize_frames",
    "recognize_image",
]
