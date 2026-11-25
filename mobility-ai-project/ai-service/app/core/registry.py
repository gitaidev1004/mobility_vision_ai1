"""모델 레지스트리: 이름 → 로더 매핑, 지연 로딩, 싱글턴 캐시.

왜 레지스트리 패턴인가?
    1. **지연 로딩(lazy loading)**: 서비스가 뜰 때 모든 모델을 로드하면 기동이
       수십 초로 늘어나고, 쓰지도 않는 모델이 메모리를 점유한다. 레지스트리는
       "이름 → 로더 함수" 만 들고 있다가 첫 요청에 로드한다.
    2. **싱글턴**: 모델 로딩은 비싸다(CRNN 33MB, YOLO 수십 MB). 요청마다 로드하면
       지연이 수백 ms 늘어난다. 이름별로 한 번만 만들어 재사용한다.
    3. **교체 용이성**: 새 모델을 추가할 때 ``register`` 한 줄만 쓰면 되고,
       라우터/파이프라인은 전혀 수정하지 않는다.

스레드 안전성:
    FastAPI 는 동기 엔드포인트를 스레드풀에서 실행한다. 두 요청이 동시에 같은
    모델을 처음 요청하면 로딩이 두 번 일어날 수 있다. ``threading.Lock`` 으로
    이름별 로딩을 직렬화한다 — 모델 로딩은 CPU/IO 를 크게 쓰므로 중복 로딩은
    단순한 낭비가 아니라 OOM 위험이다.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class RegistryEntry:
    """레지스트리 항목 하나.

    Attributes:
        name: 모델 이름(요청에서 쓰는 키).
        task: 용도 — ``"detect"`` | ``"ocr"`` | ``"pdm"``.
        loader: 호출하면 모델 객체를 돌려주는 함수(지연 실행).
        description: 사람이 읽는 설명.
        info_fn: 모델을 **로드하지 않고** 메타데이터만 돌려주는 함수(선택).
            ``/api/v1/models/info`` 가 무거운 로딩 없이 스펙을 보고할 수 있게 한다.
    """

    name: str
    task: str
    loader: Callable[[], Any]
    description: str = ""
    info_fn: Callable[[], dict[str, Any]] | None = None


@dataclass(slots=True)
class ModelRegistry:
    """모델 레지스트리.

    사용 예::

        registry = get_registry()
        detector = registry.get("yolo")        # 첫 호출에 로드, 이후 캐시
        info = registry.describe_all()         # 로딩 없이 메타데이터만
    """

    _entries: dict[str, RegistryEntry] = field(default_factory=dict)
    _instances: dict[str, Any] = field(default_factory=dict)
    _locks: dict[str, threading.Lock] = field(default_factory=dict)
    _global_lock: threading.Lock = field(default_factory=threading.Lock)

    def register(
        self,
        name: str,
        task: str,
        loader: Callable[[], Any],
        *,
        description: str = "",
        info_fn: Callable[[], dict[str, Any]] | None = None,
        overwrite: bool = True,
    ) -> None:
        """모델 로더를 등록한다.

        Args:
            name: 모델 이름(소문자 권장).
            task: ``"detect"`` | ``"ocr"`` | ``"pdm"``.
            loader: 모델을 생성/로드해 반환하는 함수.
            description: 설명 문구.
            info_fn: 로딩 없이 메타데이터를 주는 함수.
            overwrite: False 면 같은 이름이 이미 있을 때 무시한다.
        """
        key = str(name).strip().lower()
        with self._global_lock:
            if not overwrite and key in self._entries:
                return
            self._entries[key] = RegistryEntry(
                name=key, task=str(task), loader=loader, description=description, info_fn=info_fn
            )
            # 로더가 바뀌면 기존 인스턴스는 무효하다.
            self._instances.pop(key, None)

    def get(self, name: str) -> Any | None:
        """모델 인스턴스를 반환한다(최초 호출 시 로드).

        로딩이 실패하면 **None 을 반환하고 예외를 올리지 않는다.** 호출자(파이프라인)가
        폴백으로 전환할 수 있어야 하므로, 레지스트리 수준에서 예외를 흡수한다.

        Args:
            name: 등록된 모델 이름.

        Returns:
            모델 객체 또는 None(미등록/로딩 실패).
        """
        key = str(name).strip().lower()
        cached = self._instances.get(key)
        if cached is not None:
            return cached

        entry = self._entries.get(key)
        if entry is None:
            logger.warning("레지스트리에 없는 모델: %s", key)
            return None

        # 이름별 락으로 중복 로딩을 막는다.
        with self._global_lock:
            lock = self._locks.setdefault(key, threading.Lock())

        with lock:
            # 락을 기다리는 동안 다른 스레드가 로드했을 수 있다(double-checked locking).
            cached = self._instances.get(key)
            if cached is not None:
                return cached
            try:
                instance = entry.loader()
            except Exception as exc:  # noqa: BLE001 - 레지스트리는 예외를 올리지 않는다
                logger.warning("모델 '%s' 로딩 실패: %s", key, exc)
                return None
            if instance is not None:
                self._instances[key] = instance
            return instance

    def names(self, task: str | None = None) -> list[str]:
        """등록된 모델 이름 목록을 반환한다.

        Args:
            task: 지정하면 해당 task 만 필터링한다.

        Returns:
            이름 리스트(정렬).
        """
        with self._global_lock:
            items = list(self._entries.values())
        if task:
            items = [e for e in items if e.task == str(task)]
        return sorted(e.name for e in items)

    def describe(self, name: str, *, load: bool = False) -> dict[str, Any]:
        """모델 하나의 메타데이터를 반환한다.

        Args:
            name: 모델 이름.
            load: True 면 실제로 로드해 ``info()`` 를 호출한다(정확하지만 느리다).
                False 면 ``info_fn`` 만 쓴다(빠르지만 폴백 여부를 모른다).

        Returns:
            메타데이터 dict. 미등록이면 ``{"name", "registered": False}``.
        """
        key = str(name).strip().lower()
        entry = self._entries.get(key)
        if entry is None:
            return {"name": key, "registered": False}

        payload: dict[str, Any] = {
            "name": entry.name,
            "task": entry.task,
            "description": entry.description,
            "registered": True,
            "instantiated": key in self._instances,
        }

        if entry.info_fn is not None:
            try:
                payload["spec"] = entry.info_fn()
            except Exception as exc:  # noqa: BLE001
                payload["spec_error"] = f"{type(exc).__name__}: {exc}"

        if load:
            instance = self.get(key)
            if instance is None:
                payload["loaded"] = False
            else:
                payload["loaded"] = True
                info = getattr(instance, "info", None)
                if callable(info):
                    try:
                        raw = info()
                        payload["runtime"] = raw.to_dict() if hasattr(raw, "to_dict") else raw
                    except Exception as exc:  # noqa: BLE001
                        payload["runtime_error"] = f"{type(exc).__name__}: {exc}"
                exposed = getattr(instance, "exposed_name", None)
                if exposed:
                    payload["exposed_name"] = exposed
        return payload

    def describe_all(self, *, task: str | None = None, load: bool = False) -> list[dict[str, Any]]:
        """등록된 모든 모델의 메타데이터를 반환한다.

        Args:
            task: task 필터.
            load: 실제 로딩 여부.

        Returns:
            메타데이터 리스트.
        """
        return [self.describe(name, load=load) for name in self.names(task)]

    def clear(self) -> None:
        """인스턴스 캐시를 비운다(가중치 교체/테스트용). 등록 정보는 유지한다."""
        with self._global_lock:
            self._instances.clear()

    def reset(self) -> None:
        """등록 정보와 캐시를 모두 비운다(테스트 전용)."""
        with self._global_lock:
            self._entries.clear()
            self._instances.clear()
            self._locks.clear()


#: 프로세스 전역 레지스트리 인스턴스.
_REGISTRY: ModelRegistry | None = None
_REGISTRY_LOCK = threading.Lock()


def get_registry() -> ModelRegistry:
    """전역 레지스트리 싱글턴을 반환한다(최초 호출 시 기본 모델 등록).

    Returns:
        ModelRegistry.
    """
    global _REGISTRY  # noqa: PLW0603 - 프로세스 싱글턴
    if _REGISTRY is not None:
        return _REGISTRY
    with _REGISTRY_LOCK:
        if _REGISTRY is None:
            registry = ModelRegistry()
            _register_defaults(registry)
            _REGISTRY = registry
    return _REGISTRY


def _register_defaults(registry: ModelRegistry) -> None:
    """기본 모델들을 등록한다.

    등록되는 로더는 모두 **실패해도 폴백 객체를 반환**하는 팩토리다.
    그래서 ``registry.get("yolo")`` 는 가중치가 없어도 None 이 아니라
    ``FallbackPlateDetector`` 를 돌려준다 — "500 금지" 원칙이 여기서 보장된다.

    Args:
        registry: 등록 대상 레지스트리.
    """

    # ---- 검출기 ----
    def load_yolo() -> Any:
        """YOLO 검출기(또는 폴백)를 로드한다."""
        from app.detection.yolo_detector import create_yolo_detector  # noqa: PLC0415

        return create_yolo_detector()

    def load_mobilenet() -> Any:
        """MobileNet 검출기(또는 폴백)를 로드한다."""
        from app.detection.mobilenet_detector import create_mobilenet_detector  # noqa: PLC0415

        return create_mobilenet_detector()

    def load_opencv_detector() -> Any:
        """OpenCV 폴백 검출기를 로드한다(명시 선택 가능)."""
        from app.core.fallback import FallbackPlateDetector  # noqa: PLC0415

        return FallbackPlateDetector()

    def yolo_spec() -> dict[str, Any]:
        """YOLO 스펙(로딩 없이)."""
        from app.config import get_settings  # noqa: PLC0415

        settings = get_settings()
        return {
            "architecture": "YOLO-nano (single class: plate)",
            "img_size": settings.detect_img_size,
            "backends": ["ultralytics(.pt)", "onnxruntime(.onnx)"],
            "pt_weights": settings.detector_weights,
            "pt_exists": settings.weight_exists("detector_weights"),
            "onnx_weights": settings.detector_onnx,
            "onnx_exists": settings.weight_exists("detector_onnx"),
            "license": "AGPL-3.0 (ultralytics)",
        }

    def mobilenet_spec() -> dict[str, Any]:
        """MobileNet 스펙(로딩 없이)."""
        from app.detection.mobilenet_detector import mobilenet_info  # noqa: PLC0415

        return mobilenet_info()

    def opencv_spec() -> dict[str, Any]:
        """OpenCV 폴백 스펙."""
        from app.core.fallback import (  # noqa: PLC0415
            ASPECT_MAX,
            ASPECT_MIN,
            MIN_RECTANGULARITY,
            fallback_available,
        )

        spec = fallback_available()
        spec.update(
            {
                "architecture": "CLAHE + Sobel-x + morphology + contour (2-stage band→plate)",
                "aspect_range": [ASPECT_MIN, ASPECT_MAX],
                "min_rectangularity": MIN_RECTANGULARITY,
                "weights_required": False,
            }
        )
        return spec

    registry.register(
        "yolo",
        "detect",
        load_yolo,
        description="YOLO-nano 번호판 검출기 (ultralytics 또는 ONNXRuntime)",
        info_fn=yolo_spec,
    )
    registry.register(
        "mobilenet",
        "detect",
        load_mobilenet,
        description="SSDLite320-MobileNetV3 경량 번호판 검출기 (torchvision)",
        info_fn=mobilenet_spec,
    )
    registry.register(
        "opencv",
        "detect",
        load_opencv_detector,
        description="OpenCV 결정적 폴백 검출기 (가중치 불필요)",
        info_fn=opencv_spec,
    )

    # ---- 인식기 ----
    def make_recognizer_loader(name: str) -> Callable[[], Any]:
        """인식기 로더를 만드는 클로저 팩토리.

        왜 클로저 팩토리인가? 루프 안에서 ``lambda: get_recognizer(n)`` 을 쓰면
        모든 람다가 루프 변수의 **마지막 값**을 캡처하는 고전적 버그가 생긴다.
        팩토리 함수로 감싸면 각 호출의 ``name`` 이 독립적으로 바인딩된다.

        Args:
            name: 인식기 이름.

        Returns:
            인자 없는 로더 함수.
        """

        def loader() -> Any:
            """캡처된 ``name`` 으로 인식기를 가져온다."""
            from app.ocr.recognizer import get_recognizer  # noqa: PLC0415

            return get_recognizer(name)

        return loader

    def crnn_spec() -> dict[str, Any]:
        """CRNN 스펙(torch 없이)."""
        from app.config import get_settings  # noqa: PLC0415
        from app.ocr.crnn import crnn_info  # noqa: PLC0415

        spec = crnn_info()
        settings = get_settings()
        spec["weights"] = settings.crnn_weights
        spec["weights_exists"] = settings.weight_exists("crnn_weights")
        return spec

    def attention_spec() -> dict[str, Any]:
        """Attention OCR 스펙(torch 없이)."""
        from app.config import get_settings  # noqa: PLC0415
        from app.ocr.attention_ocr import attention_info  # noqa: PLC0415

        spec = attention_info()
        settings = get_settings()
        spec["weights"] = settings.attention_weights
        spec["weights_exists"] = settings.weight_exists("attention_weights")
        return spec

    def paddle_spec() -> dict[str, Any]:
        """PaddleOCR 스펙(초기화 없이)."""
        from app.ocr import paddle_adapter  # noqa: PLC0415

        return paddle_adapter.info()

    registry.register(
        "crnn",
        "ocr",
        make_recognizer_loader("crnn"),
        description="CRNN(CNN+BiLSTM) + CTC 한국 번호판 인식기",
        info_fn=crnn_spec,
    )
    registry.register(
        "attention",
        "ocr",
        make_recognizer_loader("attention"),
        description="CNN + Bahdanau Attention Seq2Seq 번호판 인식기",
        info_fn=attention_spec,
    )
    registry.register(
        "paddle",
        "ocr",
        make_recognizer_loader("paddle"),
        description="PaddleOCR 선택적 어댑터(미설치 시 폴백)",
        info_fn=paddle_spec,
    )

    # ---- 예지보전 ----
    def load_pdm() -> Any:
        """PdM 서비스 인스턴스를 로드한다."""
        from app.pdm.service import get_pdm_service  # noqa: PLC0415

        return get_pdm_service()

    def pdm_spec() -> dict[str, Any]:
        """PdM 스펙."""
        from app.pdm.service import pdm_spec as spec  # noqa: PLC0415

        return spec()

    registry.register(
        "pdm-ensemble",
        "pdm",
        load_pdm,
        description="이상탐지 + 고장분류 + RUL 앙상블 예지보전 모델",
        info_fn=pdm_spec,
    )


def get_detector(name: str = "yolo") -> Any:
    """검출기를 레지스트리에서 가져온다(없으면 폴백).

    레지스트리에 없는 이름이 오면 ``opencv`` 폴백을 돌려준다 —
    계약서가 허용하는 이름은 ``yolo|mobilenet`` 이지만, 잘못된 값이 와도
    500 을 내지 않아야 한다.

    Args:
        name: ``"yolo"`` | ``"mobilenet"`` | ``"opencv"``.

    Returns:
        ``PlateDetector`` 구현체(항상 유효).
    """
    registry = get_registry()
    detector = registry.get(name)
    if detector is None:
        logger.info("검출기 '%s' 사용 불가 → opencv 폴백", name)
        detector = registry.get("opencv")
    if detector is None:
        # 레지스트리조차 실패하는 극단적 상황 → 직접 생성.
        from app.core.fallback import FallbackPlateDetector  # noqa: PLC0415

        detector = FallbackPlateDetector()
    return detector


def get_ocr(name: str = "crnn") -> Any:
    """인식기를 레지스트리에서 가져온다(없으면 폴백).

    Args:
        name: ``"crnn"`` | ``"attention"`` | ``"paddle"``.

    Returns:
        ``PlateRecognizer`` 구현체(항상 유효).
    """
    registry = get_registry()
    recognizer = registry.get(name)
    if recognizer is None:
        from app.ocr.recognizer import get_recognizer  # noqa: PLC0415

        recognizer = get_recognizer(name)
    return recognizer


def registry_summary(*, load: bool = False) -> dict[str, Any]:
    """레지스트리 전체 요약(``/api/v1/models/info`` 응답 본문).

    Args:
        load: True 면 각 모델을 실제 로드해 폴백 여부까지 정확히 보고한다.

    Returns:
        task 별로 묶인 모델 메타데이터.
    """
    registry = get_registry()
    return {
        "detectors": registry.describe_all(task="detect", load=load),
        "recognizers": registry.describe_all(task="ocr", load=load),
        "pdm": registry.describe_all(task="pdm", load=load),
    }


__all__ = [
    "ModelRegistry",
    "RegistryEntry",
    "get_detector",
    "get_ocr",
    "get_registry",
    "registry_summary",
]
