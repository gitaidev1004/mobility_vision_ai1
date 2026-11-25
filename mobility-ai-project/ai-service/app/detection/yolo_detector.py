"""YOLO 번호판 검출기 — ultralytics 와 ONNXRuntime 양쪽 경로 지원.

왜 두 경로를 모두 지원하는가?
    | | ultralytics(.pt) | ONNXRuntime(.onnx) |
    |---|---|---|
    | 의존성 | torch + ultralytics (약 2.5GB) | onnxruntime (약 60MB) |
    | CPU 추론 속도 | 기준 | 1.5~2.5배 빠름 |
    | 전/후처리 | 라이브러리가 해 줌 | **직접 구현해야 함** |
    | 배포 적합성 | 실험/학습용 | **운영 배포용** |

    교재 독자는 학습 단계에서 ultralytics 를, 배포 단계에서 ONNX 를 쓰게 된다.
    같은 인터페이스로 두 경로를 제공하면 "학습한 모델을 어떻게 배포하는가" 라는
    실무 흐름을 코드로 보여줄 수 있다.

ONNX 경로에서는 letterbox → 추론 → 디코딩 → NMS → 역변환을 직접 해야 한다.
그 구현이 이 모듈의 교육적 핵심이다(``_decode_onnx_output``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from app.config import get_settings
from app.core.logging import get_logger
from app.detection.base import Detection, DetectorInfo, PlateDetector
from app.detection.preprocess import (
    ensure_bgr,
    letterbox,
    nms,
    normalize_for_nn,
    unletterbox_boxes,
    xywh_to_xyxy,
)

logger = get_logger(__name__)


class YoloPlateDetector(PlateDetector):
    """YOLO 기반 번호판 검출기.

    백엔드 선택 우선순위:
        1. ``.onnx`` 파일이 있고 onnxruntime 이 설치됨 → ONNX 경로(빠름)
        2. ``.pt`` 파일이 있고 ultralytics 가 설치됨 → ultralytics 경로
        3. 둘 다 안 되면 생성 시점에 ``RuntimeError`` → 팩토리가 폴백 전환

    ONNX 를 먼저 보는 이유: 운영 환경에서 더 빠르고 의존성이 가볍기 때문이다.
    """

    name = "yolo"

    def __init__(
        self,
        *,
        pt_weights: str | None = None,
        onnx_weights: str | None = None,
        img_size: int = 640,
        nms_iou: float = 0.45,
        device: str = "cpu",
        prefer_onnx: bool = True,
    ) -> None:
        """검출기를 초기화하고 가능한 백엔드를 연다.

        Args:
            pt_weights: ultralytics ``.pt`` 경로.
            onnx_weights: ONNX ``.onnx`` 경로.
            img_size: letterbox 정사각 입력 크기.
            nms_iou: NMS IoU 임계값.
            device: ``"cpu"`` | ``"cuda"``.
            prefer_onnx: True 면 ONNX 를 우선 시도한다.

        Raises:
            RuntimeError: 사용 가능한 백엔드가 하나도 없을 때.
        """
        self.img_size = max(64, int(img_size))
        self.nms_iou = float(nms_iou)
        self.device = str(device)
        self._backend = "none"
        self._weights: str | None = None
        self._session: Any = None
        self._model: Any = None
        self._input_name: str | None = None

        onnx_ok = bool(onnx_weights) and Path(str(onnx_weights)).is_file()
        pt_ok = bool(pt_weights) and Path(str(pt_weights)).is_file()

        order = ["onnx", "pt"] if prefer_onnx else ["pt", "onnx"]
        errors: list[str] = []
        for kind in order:
            if kind == "onnx" and onnx_ok:
                try:
                    self._init_onnx(str(onnx_weights))
                    return
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"onnx: {type(exc).__name__}: {exc}")
            elif kind == "pt" and pt_ok:
                try:
                    self._init_ultralytics(str(pt_weights))
                    return
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"ultralytics: {type(exc).__name__}: {exc}")

        detail = "; ".join(errors) if errors else "가중치 파일 없음"
        msg = f"YOLO 백엔드를 열 수 없습니다 ({detail})"
        raise RuntimeError(msg)

    def _init_onnx(self, path: str) -> None:
        """ONNXRuntime 세션을 생성한다.

        세션 옵션 설명:
            - ``intra_op_num_threads``: 한 연산 내부 병렬화 스레드 수. CPU 코어 수를
              넘기면 컨텍스트 스위칭으로 오히려 느려지므로 설정값으로 제한한다.
            - ``graph_optimization_level=ALL``: 상수 폴딩, 연산 융합을 모두 적용한다.
              번호판 검출 모델에서 보통 10~20% 추가 속도 이득이 있다.

        Args:
            path: ``.onnx`` 파일 경로.

        Raises:
            ImportError: onnxruntime 미설치.
            Exception: 세션 생성 실패.
        """
        import onnxruntime as ort  # noqa: PLC0415

        settings = get_settings()
        options = ort.SessionOptions()
        options.intra_op_num_threads = settings.effective_threads
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        providers = ["CPUExecutionProvider"]
        if self.device == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers():
            providers.insert(0, "CUDAExecutionProvider")

        self._session = ort.InferenceSession(path, sess_options=options, providers=providers)
        self._input_name = self._session.get_inputs()[0].name
        self._backend = "onnxruntime"
        self._weights = path
        logger.info("YOLO ONNX 로드 완료: %s (providers=%s)", path, providers)

    def _init_ultralytics(self, path: str) -> None:
        """ultralytics YOLO 모델을 로드한다.

        Args:
            path: ``.pt`` 파일 경로.

        Raises:
            ImportError: ultralytics 미설치.
            Exception: 로딩 실패.
        """
        from ultralytics import YOLO  # noqa: PLC0415

        self._model = YOLO(path)
        self._backend = "ultralytics"
        self._weights = path
        logger.info("YOLO ultralytics 로드 완료: %s", path)

    # -----------------------------------------------------------------
    # 추론
    # -----------------------------------------------------------------

    def detect(self, image: np.ndarray, min_conf: float = 0.25) -> list[Detection]:
        """번호판을 검출한다. 예외를 바깥으로 던지지 않는다.

        Args:
            image: BGR uint8 이미지.
            min_conf: 최소 신뢰도.

        Returns:
            신뢰도 내림차순 Detection 리스트.
        """
        try:
            if self._backend == "onnxruntime":
                return self._detect_onnx(image, float(min_conf))
            if self._backend == "ultralytics":
                return self._detect_ultralytics(image, float(min_conf))
        except Exception as exc:  # noqa: BLE001
            logger.warning("YOLO 검출 실패: %s", exc, extra={"stage": "detect"})
        return []

    def _detect_ultralytics(self, image: np.ndarray, min_conf: float) -> list[Detection]:
        """ultralytics 경로 추론.

        ultralytics 는 letterbox/NMS/역변환을 모두 내부에서 처리하므로
        결과 박스가 이미 원본 좌표계다.

        Args:
            image: BGR 이미지.
            min_conf: 최소 신뢰도.

        Returns:
            Detection 리스트.
        """
        bgr = ensure_bgr(image)
        results = self._model.predict(
            source=bgr,
            imgsz=self.img_size,
            conf=min_conf,
            iou=self.nms_iou,
            device=self.device,
            verbose=False,
        )
        detections: list[Detection] = []
        img_h, img_w = bgr.shape[:2]
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.asarray(boxes.xyxy)
            conf = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else np.asarray(boxes.conf)
            for i in range(xyxy.shape[0]):
                x1, y1, x2, y2 = (int(round(float(v))) for v in xyxy[i][:4])
                detections.append(
                    Detection(
                        bbox=(x1, y1, x2, y2),
                        score=round(float(conf[i]), 4),
                        label="plate",
                        source="yolo-ultralytics",
                    ).clip(img_w, img_h)
                )
        detections.sort(key=lambda d: d.score, reverse=True)
        return detections

    def _detect_onnx(self, image: np.ndarray, min_conf: float) -> list[Detection]:
        """ONNX 경로 추론 — 전처리/후처리를 직접 수행한다.

        단계:
            1. letterbox 로 정사각 입력 생성(종횡비 보존)
            2. RGB 변환 + [0,1] 정규화 + CHW + 배치 차원
            3. 세션 실행
            4. 출력 디코딩 (``_decode_onnx_output``)
            5. NMS
            6. letterbox 역변환으로 원본 좌표 복원

        Args:
            image: BGR 이미지.
            min_conf: 최소 신뢰도.

        Returns:
            Detection 리스트.
        """
        bgr = ensure_bgr(image)
        img_h, img_w = bgr.shape[:2]
        canvas, meta = letterbox(bgr, self.img_size)
        tensor = normalize_for_nn(canvas, to_rgb=True, chw=True)[None, ...]

        outputs = self._session.run(None, {self._input_name: tensor})
        boxes, scores = _decode_onnx_output(outputs, min_conf=min_conf, img_size=self.img_size)
        if boxes.size == 0:
            return []

        keep = nms(boxes, scores, self.nms_iou, max_output=20)
        if not keep:
            return []

        kept_boxes = unletterbox_boxes(boxes[keep], meta)
        detections: list[Detection] = []
        for i, idx in enumerate(keep):
            x1, y1, x2, y2 = (int(round(float(v))) for v in kept_boxes[i])
            detections.append(
                Detection(
                    bbox=(x1, y1, x2, y2),
                    score=round(float(scores[idx]), 4),
                    label="plate",
                    source="yolo-onnx",
                ).clip(img_w, img_h)
            )
        detections.sort(key=lambda d: d.score, reverse=True)
        return detections

    def info(self) -> DetectorInfo:
        """검출기 메타데이터를 반환한다."""
        return DetectorInfo(
            name=self.name,
            backend=self._backend,
            weights=self._weights,
            loaded=self._backend in {"onnxruntime", "ultralytics"},
            fallback=False,
            detail={
                "img_size": self.img_size,
                "nms_iou": self.nms_iou,
                "device": self.device,
                "classes": ["plate"],
            },
        )


def _decode_onnx_output(
    outputs: list[np.ndarray],
    *,
    min_conf: float,
    img_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """YOLO ONNX 출력을 ``(박스 xyxy, 점수)`` 로 디코딩한다.

    YOLO 출력 레이아웃은 버전마다 다르다 — 이 함수가 그 차이를 흡수한다:

        **YOLOv8/v11 (NMS 미포함)**: ``(1, 4+nc, N)``
            채널 축에 ``[cx, cy, w, h, cls0, cls1, ...]`` 이 들어 있다.
            전치하면 ``(N, 4+nc)``. objectness 가 없고 클래스 점수가 곧 신뢰도다.

        **YOLOv5 (NMS 미포함)**: ``(1, N, 5+nc)``
            ``[cx, cy, w, h, obj, cls0, ...]``. 최종 점수 = obj x cls.

        **NMS 포함 export**: ``(1, N, 6)`` = ``[x1, y1, x2, y2, score, cls]``
            이미 코너 좌표이므로 변환하지 않는다.

    판별 방법: 축 길이로 추론한다. 채널 수가 5 또는 6 이면 "이미 디코딩된" 형태,
    첫 축이 둘째 축보다 훨씬 작으면 v8 의 ``(4+nc, N)`` 전치 필요 형태다.

    좌표 스케일: 모델이 정규화 좌표(0~1)를 출력하는 경우가 있어,
    최댓값이 2.0 미만이면 ``img_size`` 를 곱해 픽셀 좌표로 바꾼다.

    Args:
        outputs: 세션 실행 결과 리스트.
        min_conf: 최소 신뢰도.
        img_size: letterbox 입력 크기(정규화 좌표 복원에 사용).

    Returns:
        ``(xyxy (M, 4), 점수 (M,))``. 유효 검출이 없으면 빈 배열 쌍.
    """
    empty = (np.zeros((0, 4), dtype=np.float64), np.zeros((0,), dtype=np.float64))
    if not outputs:
        return empty

    arr = np.asarray(outputs[0], dtype=np.float64)
    if arr.size == 0:
        return empty
    # 배치 차원 제거.
    while arr.ndim > 2:
        arr = arr[0]
    if arr.ndim != 2:
        return empty

    rows, cols = arr.shape
    # ``(4+nc, N)`` 형태면 전치한다. 번호판은 클래스가 1개라 채널이 5 이고,
    # 후보 수 N 은 보통 8400 이므로 "행 << 열" 이면 전치가 필요하다.
    if rows < cols and rows <= 85:
        arr = arr.T
        rows, cols = arr.shape

    if cols < 5:
        return empty

    if cols == 6:
        # NMS 포함 export: [x1, y1, x2, y2, score, cls]
        boxes = arr[:, :4].copy()
        scores = arr[:, 4]
    elif cols == 5:
        # [cx, cy, w, h, score] — 단일 클래스 v8 의 전형적 형태.
        boxes = xywh_to_xyxy(arr[:, :4])
        scores = arr[:, 4]
    else:
        # cols > 6: v5 형태(obj + 클래스들) 인지, v8 형태(클래스들만) 인지 구분한다.
        # v5 는 5번째 열이 objectness 라 0~1 범위이고 클래스 점수와 곱해진다.
        # 실용적 판별: 열 수가 6보다 크면 v5 로 가정하고 obj x max(cls) 를 쓴다.
        # 단일 클래스 번호판 모델에서는 어느 쪽으로 해석해도 결과가 거의 같다.
        objectness = arr[:, 4]
        class_scores = arr[:, 5:].max(axis=1)
        boxes = xywh_to_xyxy(arr[:, :4])
        scores = objectness * class_scores

    # 정규화 좌표(0~1) → 픽셀 좌표.
    if boxes.size > 0 and float(np.nanmax(np.abs(boxes))) <= 2.0:
        boxes = boxes * float(img_size)

    mask = np.isfinite(scores) & (scores >= float(min_conf))
    if not np.any(mask):
        return empty
    return boxes[mask], scores[mask]


def create_yolo_detector(*, device: str | None = None) -> PlateDetector:
    """설정 기반으로 YOLO 검출기를 생성하고, 실패하면 폴백을 반환한다.

    Args:
        device: 디바이스. None 이면 설정값.

    Returns:
        ``YoloPlateDetector`` 또는 ``FallbackPlateDetector``(절대 예외 없음).
    """
    settings = get_settings()
    try:
        return YoloPlateDetector(
            pt_weights=settings.detector_weights,
            onnx_weights=settings.detector_onnx,
            img_size=settings.detect_img_size,
            nms_iou=settings.nms_iou,
            device=device or settings.resolved_device,
        )
    except Exception as exc:  # noqa: BLE001
        logger.info("YOLO 사용 불가 → OpenCV 폴백으로 전환: %s", exc)
        from app.core.fallback import FallbackPlateDetector  # noqa: PLC0415

        detector = FallbackPlateDetector()
        # 사용자가 요청한 이름을 유지해 ``yolo-fallback`` 으로 표기되게 한다.
        detector.name = "yolo"
        return detector


__all__ = ["YoloPlateDetector", "create_yolo_detector"]
