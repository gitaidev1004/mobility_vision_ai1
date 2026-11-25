"""검출 전처리 / 후처리 유틸리티.

이 모듈은 numpy 만 필수이고 OpenCV 는 선택이다(없으면 numpy 로 최근접 리사이즈).
딥러닝 검출기와 OpenCV 폴백 검출기가 **동일한 전처리/NMS 코드를 공유**하도록
여기에 모아 두어, 둘의 좌표 체계가 어긋나는 버그를 구조적으로 막는다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:  # OpenCV 는 있으면 쓰고 없으면 numpy 폴백.
    import cv2

    _HAS_CV2 = True
except ImportError:  # pragma: no cover - 실습 환경에서는 보통 설치되어 있다.
    cv2 = None  # type: ignore[assignment]
    _HAS_CV2 = False


@dataclass(slots=True)
class LetterboxMeta:
    """letterbox 변환 메타데이터(역변환에 필요한 모든 정보).

    Attributes:
        scale: 원본 → 리사이즈 배율(가로/세로 동일).
        pad_x: 좌측 패딩 픽셀.
        pad_y: 상단 패딩 픽셀.
        orig_w: 원본 너비.
        orig_h: 원본 높이.
        new_w: letterbox 캔버스 너비.
        new_h: letterbox 캔버스 높이.
    """

    scale: float
    pad_x: int
    pad_y: int
    orig_w: int
    orig_h: int
    new_w: int
    new_h: int


def ensure_bgr(image: np.ndarray) -> np.ndarray:
    """입력 배열을 ``(H, W, 3)`` uint8 BGR 로 정규화한다.

    왜 필요한가? 업로드된 이미지는 그레이스케일(2D), RGBA(4채널), float 등
    형태가 제각각이다. 파이프라인 전체가 BGR 3채널 uint8 을 가정하므로
    입구에서 한 번만 정규화하고 이후에는 분기하지 않는다.

    Args:
        image: 임의 shape/dtype 의 이미지 배열.

    Returns:
        ``(H, W, 3)`` uint8 BGR 배열(입력이 이미 맞으면 그대로 반환).

    Raises:
        ValueError: 2/3차원이 아니거나 비어 있을 때.
    """
    arr = np.asarray(image)
    if arr.size == 0:
        msg = "빈 이미지 배열입니다."
        raise ValueError(msg)

    if arr.dtype != np.uint8:
        # float(0~1) 로 들어온 경우를 고려해 최댓값으로 스케일을 판단한다.
        peak = float(np.nanmax(arr)) if arr.size else 1.0
        scale = 255.0 if 0.0 < peak <= 1.0 + 1e-6 else 1.0
        arr = np.clip(np.nan_to_num(arr.astype(np.float32)) * scale, 0, 255).astype(np.uint8)

    if arr.ndim == 2:
        return np.repeat(arr[:, :, None], 3, axis=2)
    if arr.ndim == 3:
        ch = arr.shape[2]
        if ch == 1:
            return np.repeat(arr, 3, axis=2)
        if ch == 3:
            return arr
        if ch == 4:
            # 알파 채널은 버린다(번호판 인식에 기여하지 않는다).
            return arr[:, :, :3]
        return arr[:, :, :3] if ch > 3 else np.repeat(arr[:, :, :1], 3, axis=2)

    msg = f"지원하지 않는 이미지 차원입니다: ndim={arr.ndim}"
    raise ValueError(msg)


def to_gray(image: np.ndarray) -> np.ndarray:
    """BGR 이미지를 그레이스케일로 변환한다.

    OpenCV 가 없을 때는 ITU-R BT.601 가중치(``0.114B + 0.587G + 0.299R``)로 직접
    계산한다. 단순 평균보다 사람의 밝기 지각에 가까워 이진화 결과가 좋다.

    Args:
        image: BGR 또는 그레이스케일 배열.

    Returns:
        ``(H, W)`` uint8 그레이스케일 배열.
    """
    arr = np.asarray(image)
    if arr.ndim == 2:
        return arr.astype(np.uint8, copy=False)
    bgr = ensure_bgr(arr)
    if _HAS_CV2:
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    weights = np.array([0.114, 0.587, 0.299], dtype=np.float32)
    return np.clip(bgr.astype(np.float32) @ weights, 0, 255).astype(np.uint8)


def resize(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """이미지를 지정 크기로 리사이즈한다.

    OpenCV 가 있으면 축소에는 ``INTER_AREA``, 확대에는 ``INTER_LINEAR`` 를 쓴다.
    축소에 LINEAR 를 쓰면 에일리어싱으로 문자 획이 끊기고, 확대에 AREA 를 쓰면
    블록이 생긴다 — 번호판 OCR 정확도에 직접 영향을 주는 선택이다.

    Args:
        image: 입력 배열.
        width: 목표 너비(1 이상).
        height: 목표 높이(1 이상).

    Returns:
        리사이즈된 배열.
    """
    arr = np.asarray(image)
    target_w, target_h = max(1, int(width)), max(1, int(height))
    src_h, src_w = arr.shape[:2]
    if (src_w, src_h) == (target_w, target_h):
        return arr

    if _HAS_CV2:
        shrinking = target_w * target_h < src_w * src_h
        interp = cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR
        return cv2.resize(arr, (target_w, target_h), interpolation=interp)

    # numpy 최근접 폴백.
    ys = np.clip((np.arange(target_h) * src_h / target_h).astype(np.int64), 0, src_h - 1)
    xs = np.clip((np.arange(target_w) * src_w / target_w).astype(np.int64), 0, src_w - 1)
    return arr[ys][:, xs]


def letterbox(
    image: np.ndarray,
    size: int = 640,
    *,
    color: tuple[int, int, int] = (114, 114, 114),
) -> tuple[np.ndarray, LetterboxMeta]:
    """종횡비를 유지하며 정사각 캔버스에 맞추고 남는 부분을 패딩한다.

    왜 단순 ``resize`` 가 아니라 letterbox 인가?
        YOLO 는 정사각 입력을 요구한다. 그냥 늘리면 번호판의 종횡비가 깨져
        학습 시 분포와 달라지고 검출률이 떨어진다. letterbox 는 비율을 보존하므로
        학습/추론 간 도메인 차이를 없앤다.

    패딩 색 114 회색은 YOLO 계열의 관례값이다(0=검정은 테두리를 윤곽으로 오인시킬 수 있다).

    Args:
        image: BGR 입력 이미지.
        size: 출력 정사각 한 변 크기.
        color: 패딩 색(BGR).

    Returns:
        ``(letterbox 이미지, LetterboxMeta)``. 메타는 좌표 역변환에 쓴다.
    """
    bgr = ensure_bgr(image)
    orig_h, orig_w = bgr.shape[:2]
    side = max(1, int(size))

    scale = min(side / orig_w, side / orig_h)
    new_w = max(1, int(round(orig_w * scale)))
    new_h = max(1, int(round(orig_h * scale)))
    resized = resize(bgr, new_w, new_h)

    canvas = np.full((side, side, 3), 0, dtype=np.uint8)
    canvas[:, :] = np.array(color, dtype=np.uint8)
    # 중앙 정렬 패딩. 한쪽에만 패딩하면 객체가 가장자리에 치우쳐 검출이 불안정하다.
    pad_x = (side - new_w) // 2
    pad_y = (side - new_h) // 2
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized

    meta = LetterboxMeta(
        scale=scale,
        pad_x=pad_x,
        pad_y=pad_y,
        orig_w=orig_w,
        orig_h=orig_h,
        new_w=side,
        new_h=side,
    )
    return canvas, meta


def unletterbox_boxes(boxes: np.ndarray, meta: LetterboxMeta) -> np.ndarray:
    """letterbox 좌표계의 박스를 원본 이미지 좌표계로 되돌린다.

    역변환 순서가 중요하다: **패딩을 먼저 빼고 그 다음 배율로 나눈다.**
    순서를 바꾸면 패딩만큼의 오프셋이 배율로 왜곡된다(흔한 버그).

    Args:
        boxes: ``(N, 4)`` ``[x1, y1, x2, y2]`` 배열.
        meta: letterbox 변환 메타.

    Returns:
        원본 좌표계로 변환되고 이미지 경계로 클리핑된 ``(N, 4)`` float 배열.
    """
    arr = np.asarray(boxes, dtype=np.float64).reshape(-1, 4).copy()
    if arr.size == 0:
        return arr
    arr[:, [0, 2]] -= meta.pad_x
    arr[:, [1, 3]] -= meta.pad_y
    scale = meta.scale if meta.scale > 1e-9 else 1.0
    arr /= scale
    arr[:, [0, 2]] = np.clip(arr[:, [0, 2]], 0, meta.orig_w)
    arr[:, [1, 3]] = np.clip(arr[:, [1, 3]], 0, meta.orig_h)
    return arr


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """``(cx, cy, w, h)`` 중심 좌표 포맷을 ``(x1, y1, x2, y2)`` 로 변환한다.

    YOLO 계열 출력은 중심 좌표 포맷이므로 NMS 전에 반드시 변환해야 한다.

    Args:
        boxes: ``(N, 4)`` 중심 좌표 배열.

    Returns:
        ``(N, 4)`` 코너 좌표 배열.
    """
    arr = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    out = np.empty_like(arr)
    half_w = arr[:, 2] / 2.0
    half_h = arr[:, 3] / 2.0
    out[:, 0] = arr[:, 0] - half_w
    out[:, 1] = arr[:, 1] - half_h
    out[:, 2] = arr[:, 0] + half_w
    out[:, 3] = arr[:, 1] + half_h
    return out


def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """두 박스 집합 간 IoU 행렬을 계산한다.

    Args:
        boxes_a: ``(N, 4)`` xyxy.
        boxes_b: ``(M, 4)`` xyxy.

    Returns:
        ``(N, M)`` IoU 행렬.
    """
    a = np.asarray(boxes_a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(boxes_b, dtype=np.float64).reshape(-1, 4)
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float64)

    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)

    # 브로드캐스팅으로 모든 쌍의 교집합을 한 번에 구한다(루프보다 수십 배 빠르다).
    inter_x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    inter_y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    inter_x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    inter_y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(inter_x2 - inter_x1, 0, None) * np.clip(inter_y2 - inter_y1, 0, None)

    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.45,
    *,
    max_output: int = 50,
) -> list[int]:
    """Non-Maximum Suppression(순수 numpy).

    동작: 점수가 가장 높은 박스를 채택하고, 그것과 IoU 가 임계값을 넘는 박스를
    제거한다. 남은 것에 대해 반복.

    왜 직접 구현하는가? ``torchvision.ops.nms`` 는 torch 를 요구한다. 폴백 검출기도
    NMS 가 필요하므로 torch 없이 돌아가는 구현이 하나는 있어야 한다.

    Args:
        boxes: ``(N, 4)`` xyxy 배열.
        scores: ``(N,)`` 신뢰도 배열.
        iou_threshold: 이 값을 초과하면 중복으로 보고 제거한다.
        max_output: 최대 출력 개수(과도한 후보로 인한 지연 방지).

    Returns:
        채택된 박스의 원본 인덱스 리스트(점수 내림차순).
    """
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    if b.shape[0] == 0 or s.shape[0] == 0:
        return []
    n = min(b.shape[0], s.shape[0])
    b, s = b[:n], s[:n]

    order = np.argsort(-s)
    keep: list[int] = []
    thr = float(iou_threshold)

    while order.size > 0 and len(keep) < max_output:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break
        rest = order[1:]
        ious = iou_matrix(b[current : current + 1], b[rest])[0]
        order = rest[ious <= thr]

    return keep


def crop(image: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    """박스 영역을 안전하게 잘라낸다(경계 밖이면 빈 배열 대신 1픽셀 보장).

    Args:
        image: 입력 이미지.
        bbox: ``(x1, y1, x2, y2)``.

    Returns:
        잘라낸 부분 배열(원본 메모리 뷰).
    """
    arr = np.asarray(image)
    h, w = arr.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in bbox)
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(x1 + 1, min(x2, w))
    y2 = max(y1 + 1, min(y2, h))
    return arr[y1:y2, x1:x2]


def normalize_for_nn(
    image: np.ndarray,
    *,
    mean: tuple[float, float, float] = (0.0, 0.0, 0.0),
    std: tuple[float, float, float] = (1.0, 1.0, 1.0),
    to_rgb: bool = True,
    chw: bool = True,
) -> np.ndarray:
    """신경망 입력용 정규화 텐서를 만든다.

    OpenCV 는 BGR 로 읽지만 torchvision/ultralytics 는 RGB 를 기대한다.
    이 변환을 잊으면 검출률이 "조금" 떨어져 디버깅이 매우 어려워지므로
    기본값을 ``to_rgb=True`` 로 둔다.

    Args:
        image: ``(H, W, 3)`` uint8 BGR.
        mean: 채널별 평균(0~1 스케일 기준).
        std: 채널별 표준편차.
        to_rgb: True 면 채널 순서를 RGB 로 뒤집는다.
        chw: True 면 ``(3, H, W)``, False 면 ``(H, W, 3)``.

    Returns:
        float32 정규화 배열.
    """
    bgr = ensure_bgr(image)
    arr = bgr[:, :, ::-1] if to_rgb else bgr
    out = arr.astype(np.float32) / 255.0
    mean_arr = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
    std_arr = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)
    out = (out - mean_arr) / np.where(std_arr == 0, 1.0, std_arr)
    return np.ascontiguousarray(out.transpose(2, 0, 1)) if chw else np.ascontiguousarray(out)


def decode_image_bytes(data: bytes) -> np.ndarray | None:
    """업로드된 바이트를 BGR 이미지 배열로 디코딩한다.

    OpenCV → Pillow 순으로 시도한다. OpenCV 는 빠르지만 일부 포맷(WebP 변종 등)에서
    실패할 수 있고, Pillow 는 범용성이 높다. 둘 다 실패하면 None 을 반환하여
    상위 계층이 ``NO_PLATE`` 로 응답하게 한다(500 금지 원칙).

    Args:
        data: 이미지 파일 바이트.

    Returns:
        BGR uint8 배열 또는 None.
    """
    if not data:
        return None

    if _HAS_CV2:
        try:
            buf = np.frombuffer(data, dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if img is not None and img.size > 0:
                return ensure_bgr(img)
        except Exception:
            pass

    try:
        import io  # noqa: PLC0415

        from PIL import Image  # noqa: PLC0415

        with Image.open(io.BytesIO(data)) as im:
            rgb = np.asarray(im.convert("RGB"))
        # Pillow 는 RGB 이므로 BGR 로 뒤집는다.
        return ensure_bgr(rgb[:, :, ::-1].copy())
    except Exception:
        return None


def preprocess_summary() -> dict[str, Any]:
    """전처리 모듈의 가용 기능을 보고한다(헬스체크/진단용).

    Returns:
        OpenCV 가용 여부 등을 담은 dict.
    """
    return {"opencv": _HAS_CV2, "opencv_version": getattr(cv2, "__version__", None) if _HAS_CV2 else None}


__all__ = [
    "LetterboxMeta",
    "crop",
    "decode_image_bytes",
    "ensure_bgr",
    "iou_matrix",
    "letterbox",
    "nms",
    "normalize_for_nn",
    "preprocess_summary",
    "resize",
    "to_gray",
    "unletterbox_boxes",
    "xywh_to_xyxy",
]
