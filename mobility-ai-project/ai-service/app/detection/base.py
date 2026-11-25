"""검출기 공통 인터페이스.

왜 추상 클래스를 두는가?
    YOLO, MobileNet-SSD, OpenCV 폴백은 내부 구현이 완전히 다르지만
    파이프라인(``app.pipeline.lpr_pipeline``) 입장에서는 "이미지를 주면
    ``Detection`` 리스트를 돌려주는 것" 이면 충분하다. 인터페이스를 고정하면
    모델 교체가 파이프라인 코드 수정 없이 이루어진다(= 의존성 역전).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # numpy 는 타입 체크 시에만 필요하므로 런타임 import 를 피한다.
    import numpy as np


@dataclass(slots=True)
class Detection:
    """번호판 검출 결과 한 건.

    Attributes:
        bbox: ``(x1, y1, x2, y2)`` 정수 픽셀 좌표(좌상단 포함, 우하단 배타적이 아님).
            계약서 응답의 ``detections[].bbox`` 와 동일한 순서다.
        score: 0~1 신뢰도.
        label: 클래스 이름. 본 프로젝트는 단일 클래스 ``"plate"`` 만 쓴다.
        angle: 번호판 기울기(도). 폴백 검출기는 ``minAreaRect`` 로 추정해 채운다.
            보정(rectify) 단계가 이 값을 활용한다. 모르면 0.0.
        quad: 4점 좌표 ``[(x,y) x 4]``. 투시 변환에 쓰며, 축 정렬 박스만 아는
            검출기는 None 을 반환한다.
        source: 결과를 만든 구현 이름(예: ``"yolo"``, ``"opencv-fallback"``).
    """

    bbox: tuple[int, int, int, int]
    score: float
    label: str = "plate"
    angle: float = 0.0
    quad: list[tuple[float, float]] | None = None
    source: str = ""

    @property
    def width(self) -> int:
        """박스의 가로 픽셀 폭(최소 0)."""
        return max(0, int(self.bbox[2]) - int(self.bbox[0]))

    @property
    def height(self) -> int:
        """박스의 세로 픽셀 높이(최소 0)."""
        return max(0, int(self.bbox[3]) - int(self.bbox[1]))

    @property
    def area(self) -> int:
        """박스 면적(픽셀)."""
        return self.width * self.height

    @property
    def aspect_ratio(self) -> float:
        """가로/세로 종횡비. 높이가 0 이면 0.0.

        한국 번호판은 신형 단판 기준 약 3.3:1(520x110mm), 구형 2판형은 약 2:1 이다.
        이 값이 필터링의 1차 기준이 된다.
        """
        h = self.height
        return (self.width / h) if h > 0 else 0.0

    def clip(self, img_w: int, img_h: int) -> Detection:
        """박스를 이미지 경계 안으로 잘라낸 새 ``Detection`` 을 반환한다.

        letterbox 역변환이나 박스 확장(padding) 후에는 좌표가 이미지를 벗어날 수
        있다. crop 전에 반드시 호출해야 numpy 슬라이싱이 빈 배열을 내지 않는다.

        Args:
            img_w: 원본 이미지 너비.
            img_h: 원본 이미지 높이.

        Returns:
            경계 내로 클리핑된 새 Detection.
        """
        x1, y1, x2, y2 = (int(v) for v in self.bbox)
        x1 = max(0, min(x1, img_w - 1))
        y1 = max(0, min(y1, img_h - 1))
        x2 = max(x1 + 1, min(x2, img_w))
        y2 = max(y1 + 1, min(y2, img_h))
        return Detection(
            bbox=(x1, y1, x2, y2),
            score=self.score,
            label=self.label,
            angle=self.angle,
            quad=self.quad,
            source=self.source,
        )

    def expand(self, ratio: float, img_w: int, img_h: int) -> Detection:
        """박스를 비율만큼 확장한다(경계 클리핑 포함).

        왜 확장이 필요한가? 검출기는 번호판을 타이트하게 잡는 경향이 있고,
        OCR 은 문자 주변에 약간의 여백이 있을 때 이진화 임계값이 안정된다.
        보통 3~8% 확장이 인식률을 올린다.

        Args:
            ratio: 확장 비율(0.05 = 각 방향 5%).
            img_w: 원본 너비.
            img_h: 원본 높이.

        Returns:
            확장된 새 Detection.
        """
        dx = int(round(self.width * ratio))
        dy = int(round(self.height * ratio))
        x1, y1, x2, y2 = self.bbox
        return Detection(
            bbox=(x1 - dx, y1 - dy, x2 + dx, y2 + dy),
            score=self.score,
            label=self.label,
            angle=self.angle,
            quad=self.quad,
            source=self.source,
        ).clip(img_w, img_h)

    def to_dict(self) -> dict[str, Any]:
        """계약서 응답 스키마(``detections[]``)에 맞는 dict 로 직렬화한다.

        Returns:
            ``{"bbox": [x1,y1,x2,y2], "score": float, "label": str}``
        """
        return {
            "bbox": [int(v) for v in self.bbox],
            "score": round(float(self.score), 4),
            "label": self.label,
        }


@dataclass(slots=True)
class DetectorInfo:
    """검출기 메타데이터(``/api/v1/models/info`` 응답에 사용).

    Attributes:
        name: 계약서상의 노출 이름(폴백이면 ``-fallback`` 접미사 포함).
        backend: 실제 실행 백엔드(``ultralytics`` / ``onnxruntime`` / ``opencv`` 등).
        weights: 사용 중인 가중치 파일 경로(없으면 None).
        loaded: 가중치 로딩 성공 여부.
        fallback: 폴백 경로로 동작 중인지.
        detail: 추가 정보(입력 크기, 임계값 등).
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


class PlateDetector(ABC):
    """번호판 검출기 추상 베이스.

    구현체가 지켜야 할 계약:
        1. ``detect`` 는 **절대 예외를 던지지 않는다.** 내부 실패는 빈 리스트로 표현한다.
           엔드포인트가 500 을 내지 않아야 한다는 상위 요구사항 때문이다.
        2. 반환 ``Detection.bbox`` 는 **원본 이미지 좌표계**다(letterbox 역변환 완료).
        3. ``score`` 내림차순으로 정렬해서 반환한다.
    """

    #: 계약서에 노출되는 기본 이름. 구현체가 오버라이드한다.
    name: str = "base"

    @abstractmethod
    def detect(self, image: np.ndarray, min_conf: float = 0.25) -> list[Detection]:
        """이미지에서 번호판을 검출한다.

        Args:
            image: BGR 채널 순서의 ``(H, W, 3)`` uint8 numpy 배열.
            min_conf: 이 값 미만의 신뢰도는 버린다.

        Returns:
            신뢰도 내림차순 ``Detection`` 리스트. 검출 없으면 빈 리스트.
        """
        raise NotImplementedError

    @abstractmethod
    def info(self) -> DetectorInfo:
        """검출기 메타데이터를 반환한다."""
        raise NotImplementedError

    @property
    def is_fallback(self) -> bool:
        """폴백 경로로 동작 중인지 여부."""
        return self.info().fallback

    @property
    def exposed_name(self) -> str:
        """응답에 노출할 최종 이름.

        계약서 요구사항: 폴백이면 ``-fallback`` 접미사를 붙인다.
        이름 생성 로직을 한 곳에 모아 라우터마다 중복되지 않게 한다.
        """
        base = self.name
        if self.is_fallback and not base.endswith("-fallback"):
            return f"{base}-fallback"
        return base


__all__ = ["Detection", "DetectorInfo", "PlateDetector"]
