"""torchvision SSDLite320-MobileNetV3-Large 기반 경량 번호판 검출기.

왜 YOLO 외에 MobileNet-SSD 도 제공하는가 (교재의 비교 포인트):

    | 항목 | YOLOv8-nano | SSDLite320-MobileNetV3 |
    |---|---|---|
    | 파라미터 | 약 3.2M | 약 3.4M |
    | 입력 크기 | 640x640 | 320x320 |
    | CPU 지연(1코어) | 기준 | **약 40% 빠름** (입력이 1/4) |
    | 작은 객체 | 강함 | 약함 (320 해상도의 한계) |
    | 의존성 | ultralytics (별도 설치) | **torchvision 만** |
    | 라이선스 | AGPL-3.0 | BSD |

    AGPL 라이선스가 부담스러운 상용 프로젝트에서 SSDLite 는 현실적인 선택이다.
    또한 "입력 해상도를 낮춰 속도를 얻고 작은 번호판을 포기하는" 트레이드오프를
    독자가 체감할 수 있게 한다 — 게이트 카메라처럼 번호판이 크게 잡히는
    환경에서는 320 해상도로도 충분하다.

``torchvision`` 미설치 또는 가중치 없음 → 생성 시점 ``RuntimeError`` → 팩토리 폴백.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from app.config import get_settings
from app.core.logging import get_logger
from app.detection.base import Detection, DetectorInfo, PlateDetector
from app.detection.preprocess import ensure_bgr, nms

logger = get_logger(__name__)

#: SSDLite320 의 표준 입력 크기. 모델 구조가 이 크기에 맞춰 설계되어 있다.
SSD_INPUT_SIZE: int = 320

#: 번호판 단일 클래스 + 배경. torchvision 검출 모델은 0번을 배경으로 예약한다.
NUM_CLASSES_WITH_BG: int = 2


def _require_torchvision() -> tuple[Any, Any]:
    """torch 와 torchvision 을 지연 import 한다.

    Returns:
        ``(torch, torchvision)`` 모듈 쌍.

    Raises:
        RuntimeError: 둘 중 하나라도 없을 때.
    """
    try:
        import torch  # noqa: PLC0415
        import torchvision  # noqa: PLC0415

        return torch, torchvision
    except ImportError as exc:
        msg = (
            "torchvision 이 설치되지 않았습니다. MobileNet 검출기를 쓰려면 "
            "`pip install torch==2.8.0 torchvision==0.23.0` 를 실행하거나, "
            "폴백 검출기를 사용하세요."
        )
        raise RuntimeError(msg) from exc


def build_ssdlite_detector(num_classes: int = NUM_CLASSES_WITH_BG) -> Any:
    """번호판 클래스 수에 맞춘 SSDLite320-MobileNetV3-Large 모델을 만든다.

    핵심: **분류 헤드만 교체한다.**
        사전학습 모델은 COCO 91 클래스용 헤드를 갖고 있다. 번호판은 1 클래스
        (+배경)이므로 헤드를 새로 만들어야 한다. 백본(MobileNetV3)은 그대로 쓴다 —
        저수준 특징(에지/텍스처)은 도메인이 달라도 재사용 가능하기 때문이다.
        이것이 전이학습(transfer learning)의 표준 패턴이다.

    Args:
        num_classes: 배경을 포함한 클래스 수(번호판 1개 → 2).

    Returns:
        ``torch.nn.Module`` 검출 모델.

    Raises:
        RuntimeError: torchvision 미설치.
    """
    torch, torchvision = _require_torchvision()

    from torchvision.models.detection import ssdlite320_mobilenet_v3_large  # noqa: PLC0415
    from torchvision.models.detection.ssdlite import SSDLiteClassificationHead  # noqa: PLC0415

    # weights=None: 가중치 다운로드를 하지 않는다. 네트워크 없는 환경과
    # "테스트는 네트워크 없이" 요구사항을 지키기 위함이다.
    # 전이학습을 하려면 학습 스크립트에서 weights="DEFAULT" 로 바꾼다.
    model = ssdlite320_mobilenet_v3_large(weights=None, weights_backbone=None)

    # 기존 헤드에서 앵커 수와 입력 채널 수를 읽어 와 새 헤드를 만든다.
    # 하드코딩하면 torchvision 버전이 올라가 구조가 바뀔 때 조용히 깨진다.
    in_channels = [
        layer[0].in_channels for layer in model.head.classification_head.module_list
    ]
    num_anchors = model.anchor_generator.num_anchors_per_location()
    norm_layer = torch.nn.BatchNorm2d

    model.head.classification_head = SSDLiteClassificationHead(
        in_channels=in_channels,
        num_anchors=num_anchors,
        num_classes=int(num_classes),
        norm_layer=norm_layer,
    )
    return model


class MobileNetPlateDetector(PlateDetector):
    """SSDLite320-MobileNetV3 기반 번호판 검출기."""

    name = "mobilenet"

    def __init__(
        self,
        weights_path: str,
        *,
        device: str = "cpu",
        nms_iou: float = 0.45,
        num_classes: int = NUM_CLASSES_WITH_BG,
    ) -> None:
        """검출기를 초기화하고 가중치를 로드한다.

        Args:
            weights_path: ``.pt`` 체크포인트 경로.
            device: 디바이스.
            nms_iou: 추가 NMS IoU 임계값.
                torchvision SSD 는 내부에서 이미 NMS 를 하지만, 번호판은 단일
                객체인 경우가 많아 한 번 더 걸러 중복 후보를 줄인다.
            num_classes: 배경 포함 클래스 수.

        Raises:
            RuntimeError: torchvision 미설치, 파일 없음, 또는 로딩 실패.
        """
        torch, _ = _require_torchvision()
        path = Path(str(weights_path))
        if not path.is_file():
            msg = f"MobileNet 가중치가 없습니다: {weights_path}"
            raise RuntimeError(msg)

        self._weights = str(path)
        self._device = str(device)
        self.nms_iou = float(nms_iou)
        self._torch = torch

        try:
            checkpoint = torch.load(self._weights, map_location="cpu", weights_only=True)
        except Exception as exc:
            msg = f"MobileNet 체크포인트 로딩 실패: {type(exc).__name__}: {exc}"
            raise RuntimeError(msg) from exc

        state = checkpoint
        classes = int(num_classes)
        if isinstance(checkpoint, dict):
            if "num_classes" in checkpoint:
                classes = int(checkpoint["num_classes"])
            for key in ("model", "state_dict", "model_state_dict"):
                if key in checkpoint and isinstance(checkpoint[key], dict):
                    state = checkpoint[key]
                    break

        model = build_ssdlite_detector(num_classes=classes)
        try:
            model.load_state_dict(state, strict=True)
        except Exception as exc:
            msg = f"MobileNet state_dict 불일치: {type(exc).__name__}: {exc}"
            raise RuntimeError(msg) from exc

        model.to(self._device).eval()
        self._model = model
        logger.info("MobileNet SSDLite 로드 완료: %s", self._weights)

    def detect(self, image: np.ndarray, min_conf: float = 0.25) -> list[Detection]:
        """번호판을 검출한다. 예외를 바깥으로 던지지 않는다.

        Args:
            image: BGR uint8 이미지.
            min_conf: 최소 신뢰도.

        Returns:
            신뢰도 내림차순 Detection 리스트.
        """
        try:
            return self._detect_impl(image, float(min_conf))
        except Exception as exc:  # noqa: BLE001
            logger.warning("MobileNet 검출 실패: %s", exc, extra={"stage": "detect"})
            return []

    def _detect_impl(self, image: np.ndarray, min_conf: float) -> list[Detection]:
        """실제 검출 로직.

        torchvision 검출 모델의 입력 규약:
            - ``list[Tensor]`` 형태로 **이미지마다 다른 크기**를 허용한다.
            - 각 텐서는 ``(3, H, W)`` **RGB**, ``float32``, **0~1 범위**.
            - 모델이 내부에서 리사이즈/정규화를 수행한다(``GeneralizedRCNNTransform``).
              따라서 letterbox 를 직접 할 필요가 없고, **출력 박스도 원본 좌표계**다.
              이 점이 ONNX 경로와의 큰 차이다.

        Args:
            image: BGR 이미지.
            min_conf: 최소 신뢰도.

        Returns:
            Detection 리스트.
        """
        torch = self._torch
        bgr = ensure_bgr(image)
        img_h, img_w = bgr.shape[:2]

        # BGR → RGB, HWC → CHW, 0~255 → 0~1
        rgb = np.ascontiguousarray(bgr[:, :, ::-1].transpose(2, 0, 1), dtype=np.float32) / 255.0
        tensor = torch.from_numpy(rgb).to(self._device)

        with torch.no_grad():
            outputs = self._model([tensor])

        if not outputs:
            return []
        result = outputs[0]
        boxes = result["boxes"].cpu().numpy()
        scores = result["scores"].cpu().numpy()
        labels = result["labels"].cpu().numpy()

        # 배경(0)을 제외하고 신뢰도 필터를 적용한다.
        mask = (scores >= min_conf) & (labels != 0)
        if not np.any(mask):
            return []
        boxes, scores = boxes[mask], scores[mask]

        keep = nms(boxes, scores, self.nms_iou, max_output=20)
        detections: list[Detection] = []
        for i in keep:
            x1, y1, x2, y2 = (int(round(float(v))) for v in boxes[i][:4])
            detections.append(
                Detection(
                    bbox=(x1, y1, x2, y2),
                    score=round(float(scores[i]), 4),
                    label="plate",
                    source="mobilenet-ssdlite",
                ).clip(img_w, img_h)
            )
        detections.sort(key=lambda d: d.score, reverse=True)
        return detections

    def info(self) -> DetectorInfo:
        """검출기 메타데이터를 반환한다."""
        return DetectorInfo(
            name=self.name,
            backend="torchvision",
            weights=self._weights,
            loaded=True,
            fallback=False,
            detail={
                "architecture": "SSDLite320-MobileNetV3-Large",
                "input_size": SSD_INPUT_SIZE,
                "nms_iou": self.nms_iou,
                "device": self._device,
                "classes": ["background", "plate"],
                "license": "BSD (torchvision)",
            },
        )


def mobilenet_info() -> dict[str, Any]:
    """MobileNet 검출기 구조 메타데이터를 반환한다(torch 없이 호출 가능).

    Returns:
        구조 설명 dict.
    """
    return {
        "architecture": "SSDLite320-MobileNetV3-Large",
        "input_size": [3, SSD_INPUT_SIZE, SSD_INPUT_SIZE],
        "params_estimate": 3_440_000,
        "size_fp32_mb": 13.1,
        "classes": ["background", "plate"],
        "license": "BSD (torchvision)",
        "notes": "YOLO 대비 입력 1/4(320 vs 640)로 CPU 약 40% 빠름, 작은 번호판엔 약함",
    }


def create_mobilenet_detector(*, device: str | None = None) -> PlateDetector:
    """설정 기반으로 MobileNet 검출기를 생성하고, 실패하면 폴백을 반환한다.

    가중치 경로 규칙: ``DETECTOR_WEIGHTS`` 와 같은 디렉터리의
    ``plate_ssdlite_mobilenet.pt`` 를 먼저 찾고, 없으면 ``DETECTOR_WEIGHTS`` 자체를
    시도한다. 검출기별로 환경변수를 따로 두면 설정이 번잡해지므로,
    규약으로 해결한다.

    Args:
        device: 디바이스. None 이면 설정값.

    Returns:
        ``MobileNetPlateDetector`` 또는 ``FallbackPlateDetector``(절대 예외 없음).
    """
    settings = get_settings()
    candidates = [
        settings.model_root / "plate_ssdlite_mobilenet.pt",
        settings.model_root / "plate_mobilenet.pt",
        Path(str(settings.detector_weights)),
    ]
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            return MobileNetPlateDetector(
                str(candidate),
                device=device or settings.resolved_device,
                nms_iou=settings.nms_iou,
            )
        except Exception as exc:  # noqa: BLE001
            logger.info("MobileNet 로딩 실패(%s): %s", candidate, exc)

    logger.info("MobileNet 사용 불가 → OpenCV 폴백으로 전환")
    from app.core.fallback import FallbackPlateDetector  # noqa: PLC0415

    detector = FallbackPlateDetector()
    detector.name = "mobilenet"
    return detector


__all__ = [
    "NUM_CLASSES_WITH_BG",
    "SSD_INPUT_SIZE",
    "MobileNetPlateDetector",
    "build_ssdlite_detector",
    "create_mobilenet_detector",
    "mobilenet_info",
]
