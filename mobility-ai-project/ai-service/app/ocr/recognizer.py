"""인식기(recognizer) 공통 인터페이스와 팩토리.

설계 의도:
    파이프라인은 "번호판 crop 을 주면 문자열과 신뢰도를 돌려주는 것" 만 알면 된다.
    그 뒤에 CRNN 이 있든, Attention 이 있든, PaddleOCR 이 있든, OpenCV 템플릿
    매칭이 있든 상관없어야 한다. 이 모듈이 그 경계를 정의한다.

폴백 전환 규칙 (계약서 요구사항):
    요청한 인식기의 가중치/런타임이 없으면 자동으로 ``FallbackOCR`` 로 전환하고,
    ``exposed_name`` 에 ``-fallback`` 접미사를 붙인다. 절대 500 을 던지지 않는다.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from app.config import get_settings
from app.core.logging import get_logger
from app.ocr.charset import NUM_CLASSES

logger = get_logger(__name__)

#: 계약서가 허용하는 인식기 이름.
ALLOWED_RECOGNIZERS: tuple[str, ...] = ("crnn", "attention", "paddle")


@dataclass(slots=True)
class RecognitionResult:
    """인식 결과 한 건.

    Attributes:
        text: 인식된 문자열(후처리 전 원본).
        confidence: 0~1 신뢰도.
        char_confidences: 문자별 신뢰도.
        candidates: 상위 후보 ``[{"text", "confidence"}]``.
            계약서 응답의 ``candidates`` 필드와 대응한다.
        reason: 실패 사유(성공 시 None).
        backend: 실제로 사용한 백엔드 이름.
    """

    text: str
    confidence: float = 0.0
    char_confidences: list[float] = field(default_factory=list)
    candidates: list[dict[str, Any]] = field(default_factory=list)
    reason: str | None = None
    backend: str = ""


@dataclass(slots=True)
class RecognizerInfo:
    """인식기 메타데이터.

    Attributes:
        name: 노출 이름.
        backend: 실행 백엔드.
        weights: 가중치 경로(없으면 None).
        loaded: 로딩 성공 여부.
        fallback: 폴백 동작 중인지.
        detail: 추가 정보.
    """

    name: str
    backend: str
    weights: str | None = None
    loaded: bool = False
    fallback: bool = True
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON 직렬화용 dict 를 반환한다."""
        return {
            "name": self.name,
            "backend": self.backend,
            "weights": self.weights,
            "loaded": self.loaded,
            "fallback": self.fallback,
            "detail": self.detail,
        }


class PlateRecognizer(ABC):
    """번호판 인식기 추상 베이스.

    구현체 계약:
        1. ``recognize`` 는 예외를 던지지 않는다. 실패는 ``reason`` 으로 표현한다.
        2. 반환 ``text`` 는 **후처리 전 원본** 이다. 정규식 보정은
           ``app.ocr.postprocess`` 가 담당한다(책임 분리).
        3. ``candidates`` 는 신뢰도 내림차순으로 정렬한다.
    """

    #: 계약서에 노출되는 기본 이름.
    name: str = "base"

    @abstractmethod
    def recognize(self, plate_bgr: np.ndarray) -> RecognitionResult:
        """번호판 crop 에서 문자열을 인식한다.

        Args:
            plate_bgr: 번호판 영역 BGR uint8 배열.

        Returns:
            RecognitionResult.
        """
        raise NotImplementedError

    @abstractmethod
    def info(self) -> RecognizerInfo:
        """인식기 메타데이터를 반환한다."""
        raise NotImplementedError

    @property
    def is_fallback(self) -> bool:
        """폴백 경로로 동작 중인지."""
        return self.info().fallback

    @property
    def exposed_name(self) -> str:
        """응답에 노출할 이름(폴백이면 ``-fallback`` 접미사).

        Returns:
            노출용 이름.
        """
        base = self.name
        if self.is_fallback and not base.endswith("-fallback"):
            return f"{base}-fallback"
        return base


# ---------------------------------------------------------------------------
# 1) 폴백 인식기 (OpenCV 템플릿)
# ---------------------------------------------------------------------------


class FallbackRecognizer(PlateRecognizer):
    """``app.core.fallback.FallbackOCR`` 을 감싸는 인식기.

    가중치가 없는 모든 경로의 최종 착지점이다. 이 구현이 있기 때문에
    "어떤 경우에도 엔드포인트가 500 을 내지 않는다" 가 성립한다.
    """

    name = "template-ocr"

    def __init__(self, *, declared_as: str | None = None) -> None:
        """폴백 인식기를 초기화한다.

        Args:
            declared_as: 사용자가 원래 요청한 인식기 이름(``crnn`` 등).
                응답에 ``crnn-fallback`` 처럼 표기하기 위해 보관한다 —
                운영자가 "요청한 모델이 폴백으로 처리됐다" 를 알 수 있어야 한다.
        """
        from app.core.fallback import FallbackOCR  # noqa: PLC0415 - 지연 import

        self._ocr = FallbackOCR()
        if declared_as:
            self.name = str(declared_as)

    def recognize(self, plate_bgr: np.ndarray) -> RecognitionResult:
        """템플릿 매칭으로 번호판을 판독한다.

        Args:
            plate_bgr: 번호판 crop.

        Returns:
            RecognitionResult.
        """
        raw = self._ocr.read(plate_bgr)
        text = str(raw.get("text") or "")
        conf = float(raw.get("confidence") or 0.0)
        return RecognitionResult(
            text=text,
            confidence=conf,
            char_confidences=list(raw.get("char_confidences") or []),
            candidates=[{"text": text, "confidence": round(conf, 4)}] if text else [],
            reason=raw.get("reason"),
            backend="opencv-template",
        )

    def info(self) -> RecognizerInfo:
        """메타데이터를 반환한다."""
        raw = self._ocr.info()
        return RecognizerInfo(
            name=self.name,
            backend=str(raw.get("backend", "opencv-template")),
            weights=None,
            loaded=True,
            fallback=True,
            detail=dict(raw.get("detail") or {}),
        )


# ---------------------------------------------------------------------------
# 2) CRNN 인식기
# ---------------------------------------------------------------------------


class CRNNRecognizer(PlateRecognizer):
    """CRNN + CTC 기반 인식기.

    가중치가 있으면 torch 로 추론하고, 없으면 생성 시점에 ``RuntimeError`` 를
    던져 팩토리가 폴백으로 전환하게 한다 — 실패를 **생성 시점에** 드러내는 것이
    요청마다 실패하는 것보다 훨씬 다루기 쉽다(fail fast).
    """

    name = "crnn"

    def __init__(
        self,
        weights_path: str,
        *,
        device: str = "cpu",
        decode_mode: str = "beam",
        beam_width: int = 10,
        top_k: int = 3,
    ) -> None:
        """CRNN 인식기를 초기화하고 가중치를 로드한다.

        Args:
            weights_path: ``.pt`` 체크포인트 경로.
            device: ``"cpu"`` | ``"cuda"``.
            decode_mode: ``"greedy"`` | ``"beam"``.
                기본값을 beam 으로 둔 이유: 번호판은 시퀀스가 짧아(T=40)
                beam search 비용이 1ms 미만이면서 경계 케이스 정확도가 눈에 띄게 오른다.
            beam_width: beam 폭.
            top_k: 반환할 후보 수.

        Raises:
            RuntimeError: torch 미설치 또는 가중치 로딩 실패.
        """
        from app.ocr.crnn import load_crnn  # noqa: PLC0415

        self._weights = str(weights_path)
        self._device = str(device)
        self._decode_mode = str(decode_mode)
        self._beam_width = int(beam_width)
        self._top_k = int(top_k)
        self._model = load_crnn(self._weights, device=self._device, num_classes=NUM_CLASSES)

    def recognize(self, plate_bgr: np.ndarray) -> RecognitionResult:
        """CRNN 추론 → CTC 디코딩으로 번호판을 판독한다.

        Args:
            plate_bgr: 번호판 crop.

        Returns:
            RecognitionResult. 추론 실패 시 빈 텍스트와 사유.
        """
        try:
            return self._recognize_impl(plate_bgr)
        except Exception as exc:  # noqa: BLE001 - 인식기는 예외를 올리지 않는다
            logger.warning("CRNN 추론 실패: %s", exc, extra={"stage": "ocr"})
            return RecognitionResult(
                text="",
                confidence=0.0,
                reason=f"crnn-error: {type(exc).__name__}",
                backend="torch",
            )

    def _recognize_impl(self, plate_bgr: np.ndarray) -> RecognitionResult:
        """실제 추론 로직."""
        import torch  # noqa: PLC0415

        from app.ocr.ctc_decoder import decode as ctc_decode  # noqa: PLC0415
        from app.pipeline.rectify import resize_for_ocr  # noqa: PLC0415

        settings = get_settings()
        # 전처리: 그레이스케일 → OCR 입력 크기 → [0,1] 정규화.
        arr = np.asarray(plate_bgr)
        if arr.ndim == 3:
            # BT.601 가중치로 그레이 변환(학습 전처리와 동일해야 한다).
            weights = np.array([0.114, 0.587, 0.299], dtype=np.float32)
            gray = np.clip(arr[:, :, :3].astype(np.float32) @ weights, 0, 255).astype(np.uint8)
        else:
            gray = arr.astype(np.uint8)

        resized = resize_for_ocr(
            gray, height=settings.ocr_img_height, width=settings.ocr_img_width
        )
        # (1, 1, H, W) 로 만들고 평균 0.5 / 표준편차 0.5 로 정규화한다.
        tensor = torch.from_numpy(
            ((resized.astype(np.float32) / 255.0) - 0.5) / 0.5
        ).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            logits = self._model(tensor.to(self._device))   # (1, T, C)
            probs = logits.softmax(dim=-1)[0].cpu().numpy()  # (T, C)

        results = ctc_decode(
            probs,
            mode=self._decode_mode,
            beam_width=self._beam_width,
            top_k=self._top_k,
            already_prob=True,
        )
        best = results[0]
        return RecognitionResult(
            text=best.text,
            confidence=float(best.confidence),
            char_confidences=list(best.char_confidences),
            candidates=[
                {"text": r.text, "confidence": round(float(r.confidence), 4)} for r in results
            ],
            reason=None if best.text else "ctc-empty",
            backend="torch",
        )

    def info(self) -> RecognizerInfo:
        """메타데이터를 반환한다."""
        from app.ocr.crnn import crnn_info  # noqa: PLC0415

        detail = crnn_info()
        detail["decode_mode"] = self._decode_mode
        detail["beam_width"] = self._beam_width
        return RecognizerInfo(
            name=self.name,
            backend="torch",
            weights=self._weights,
            loaded=True,
            fallback=False,
            detail=detail,
        )


# ---------------------------------------------------------------------------
# 3) Attention 인식기
# ---------------------------------------------------------------------------


class AttentionRecognizer(PlateRecognizer):
    """CNN + Attention Seq2Seq 기반 인식기."""

    name = "attention"

    def __init__(self, weights_path: str, *, device: str = "cpu") -> None:
        """Attention 인식기를 초기화하고 가중치를 로드한다.

        Args:
            weights_path: ``.pt`` 체크포인트 경로.
            device: 디바이스.

        Raises:
            RuntimeError: torch 미설치 또는 로딩 실패.
        """
        from app.ocr.attention_ocr import load_attention_ocr  # noqa: PLC0415

        self._weights = str(weights_path)
        self._device = str(device)
        self._model = load_attention_ocr(self._weights, device=self._device)

    def recognize(self, plate_bgr: np.ndarray) -> RecognitionResult:
        """Attention 디코딩으로 번호판을 판독한다.

        Args:
            plate_bgr: 번호판 crop.

        Returns:
            RecognitionResult.
        """
        try:
            return self._recognize_impl(plate_bgr)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Attention 추론 실패: %s", exc, extra={"stage": "ocr"})
            return RecognitionResult(
                text="",
                confidence=0.0,
                reason=f"attention-error: {type(exc).__name__}",
                backend="torch",
            )

    def _recognize_impl(self, plate_bgr: np.ndarray) -> RecognitionResult:
        """실제 추론 로직."""
        import torch  # noqa: PLC0415

        from app.ocr.attention_ocr import decode_tokens  # noqa: PLC0415
        from app.pipeline.rectify import resize_for_ocr  # noqa: PLC0415

        settings = get_settings()
        arr = np.asarray(plate_bgr)
        if arr.ndim == 3:
            weights = np.array([0.114, 0.587, 0.299], dtype=np.float32)
            gray = np.clip(arr[:, :, :3].astype(np.float32) @ weights, 0, 255).astype(np.uint8)
        else:
            gray = arr.astype(np.uint8)

        resized = resize_for_ocr(
            gray, height=settings.ocr_img_height, width=settings.ocr_img_width
        )
        tensor = torch.from_numpy(
            ((resized.astype(np.float32) / 255.0) - 0.5) / 0.5
        ).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            tokens, probs, _attn = self._model.greedy_decode(tensor.to(self._device))

        idx_list = [int(v) for v in tokens[0].cpu().numpy().tolist()]
        prob_list = [float(v) for v in probs[0].cpu().numpy().tolist()]
        text = decode_tokens(idx_list)
        # 실제로 출력된 글자 수만큼의 확률을 취한다(EOS 이후는 버린다).
        char_confs = prob_list[: len(text)]
        confidence = float(np.exp(np.mean(np.log(np.clip(char_confs, 1e-6, 1.0))))) if char_confs else 0.0

        return RecognitionResult(
            text=text,
            confidence=confidence,
            char_confidences=[round(c, 4) for c in char_confs],
            candidates=[{"text": text, "confidence": round(confidence, 4)}] if text else [],
            reason=None if text else "attention-empty",
            backend="torch",
        )

    def info(self) -> RecognizerInfo:
        """메타데이터를 반환한다."""
        from app.ocr.attention_ocr import attention_info  # noqa: PLC0415

        return RecognizerInfo(
            name=self.name,
            backend="torch",
            weights=self._weights,
            loaded=True,
            fallback=False,
            detail=attention_info(),
        )


# ---------------------------------------------------------------------------
# 4) PaddleOCR 인식기
# ---------------------------------------------------------------------------


class PaddleRecognizer(PlateRecognizer):
    """PaddleOCR 어댑터를 감싸는 인식기."""

    name = "paddle"

    def __init__(self, *, lang: str = "korean") -> None:
        """PaddleOCR 인식기를 초기화한다.

        Args:
            lang: 인식 언어.

        Raises:
            RuntimeError: PaddleOCR 미설치 또는 초기화 실패.
        """
        from app.ocr import paddle_adapter  # noqa: PLC0415

        self._lang = str(lang)
        self._adapter = paddle_adapter
        if not paddle_adapter.is_available():
            msg = "PaddleOCR 이 설치되지 않았습니다(선택 의존성)."
            raise RuntimeError(msg)
        if paddle_adapter.get_paddle_ocr(lang=self._lang) is None:
            msg = f"PaddleOCR 초기화 실패: {paddle_adapter.info().get('reason')}"
            raise RuntimeError(msg)

    def recognize(self, plate_bgr: np.ndarray) -> RecognitionResult:
        """PaddleOCR 로 번호판을 판독한다.

        Args:
            plate_bgr: 번호판 crop.

        Returns:
            RecognitionResult.
        """
        raw = self._adapter.recognize(plate_bgr, lang=self._lang)
        text = str(raw.get("text") or "")
        conf = float(raw.get("confidence") or 0.0)
        return RecognitionResult(
            text=text,
            confidence=conf,
            char_confidences=[],
            candidates=list(raw.get("lines") or []),
            reason=raw.get("reason"),
            backend="paddlepaddle",
        )

    def info(self) -> RecognizerInfo:
        """메타데이터를 반환한다."""
        raw = self._adapter.info()
        return RecognizerInfo(
            name=self.name,
            backend="paddlepaddle",
            weights=None,
            loaded=bool(raw.get("loaded")),
            fallback=not bool(raw.get("loaded")),
            detail=dict(raw.get("detail") or {}),
        )


# ---------------------------------------------------------------------------
# 5) 팩토리
# ---------------------------------------------------------------------------


def create_recognizer(name: str = "crnn", *, device: str | None = None) -> PlateRecognizer:
    """이름으로 인식기를 생성한다. **절대 예외를 던지지 않는다.**

    전환 순서:
        1. 요청한 인식기를 시도한다.
        2. 실패하면(가중치/런타임 없음) ``FallbackRecognizer`` 를 반환하되
           ``declared_as`` 에 원래 이름을 넣어 ``crnn-fallback`` 으로 표기한다.

    왜 예외를 던지지 않는가? 라우터가 매번 try/except 로 감싸야 한다면
    "500 금지" 규칙이 라우터 구현자의 성실함에 의존하게 된다. 팩토리가
    실패를 흡수하면 규칙이 구조적으로 보장된다.

    Args:
        name: ``"crnn"`` | ``"attention"`` | ``"paddle"``. 알 수 없는 값은 기본값으로.
        device: 디바이스. None 이면 설정에서 가져온다.

    Returns:
        PlateRecognizer 구현체(항상 유효한 객체).
    """
    settings = get_settings()
    requested = str(name or settings.default_recognizer).strip().lower()
    if requested not in ALLOWED_RECOGNIZERS:
        logger.info("알 수 없는 인식기 '%s' → 기본값 '%s' 사용", requested, settings.default_recognizer)
        requested = str(settings.default_recognizer).strip().lower()
    dev = device or settings.resolved_device

    try:
        if requested == "crnn":
            if settings.weight_exists("crnn_weights"):
                return CRNNRecognizer(settings.crnn_weights, device=dev)
            logger.info("CRNN 가중치 없음(%s) → 폴백", settings.crnn_weights)
        elif requested == "attention":
            if settings.weight_exists("attention_weights"):
                return AttentionRecognizer(settings.attention_weights, device=dev)
            logger.info("Attention 가중치 없음(%s) → 폴백", settings.attention_weights)
        elif requested == "paddle":
            return PaddleRecognizer()
    except Exception as exc:  # noqa: BLE001 - 어떤 실패든 폴백으로 흡수
        logger.warning("인식기 '%s' 생성 실패 → 폴백: %s", requested, exc)

    return FallbackRecognizer(declared_as=requested)


#: 인식기 인스턴스 캐시. 모델 로딩은 수백 ms~수 초이므로 요청마다 하면 안 된다.
_RECOGNIZER_CACHE: dict[str, PlateRecognizer] = {}


def get_recognizer(name: str = "crnn", *, device: str | None = None) -> PlateRecognizer:
    """인식기 싱글턴을 반환한다(이름+디바이스별 캐시).

    Args:
        name: 인식기 이름.
        device: 디바이스.

    Returns:
        캐시된 PlateRecognizer.
    """
    settings = get_settings()
    key = f"{str(name).strip().lower()}::{device or settings.resolved_device}"
    cached = _RECOGNIZER_CACHE.get(key)
    if cached is None:
        cached = create_recognizer(name, device=device)
        _RECOGNIZER_CACHE[key] = cached
    return cached


def clear_recognizer_cache() -> None:
    """인식기 캐시를 비운다(테스트/가중치 교체용)."""
    _RECOGNIZER_CACHE.clear()


def available_recognizers() -> list[dict[str, Any]]:
    """모든 인식기의 가용 상태를 보고한다(``/api/v1/models/info`` 용).

    주의: 각 인식기를 **실제로 생성**하므로 폴백 전환 여부가 정확히 반영된다.
    캐시를 쓰므로 반복 호출 비용은 낮다.

    Returns:
        인식기별 메타데이터 리스트.
    """
    out: list[dict[str, Any]] = []
    for name in ALLOWED_RECOGNIZERS:
        recognizer = get_recognizer(name)
        payload = recognizer.info().to_dict()
        payload["requested"] = name
        payload["exposed_name"] = recognizer.exposed_name
        out.append(payload)
    return out


__all__ = [
    "ALLOWED_RECOGNIZERS",
    "AttentionRecognizer",
    "CRNNRecognizer",
    "FallbackRecognizer",
    "PaddleRecognizer",
    "PlateRecognizer",
    "RecognitionResult",
    "RecognizerInfo",
    "available_recognizers",
    "clear_recognizer_cache",
    "create_recognizer",
    "get_recognizer",
]
