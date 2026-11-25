"""번호판 보정(rectification): 기울기 보정, 투시 변환, CLAHE, 샤프닝.

왜 OCR 앞에 보정 단계가 필요한가?
    OCR 모델(CRNN/CTC)은 입력을 "왼쪽에서 오른쪽으로 읽는 가로 띠" 로 가정한다.
    번호판이 7도만 기울어도 문자가 세로로 어긋나 CTC 의 프레임-문자 정렬이 흐트러진다.
    폴백 OCR(연결요소 분할)은 더 민감하다 — 기울면 문자 x 구간이 겹쳐 분할이 실패한다.
    실측에서 증강(투시+블러+조명) 영상의 폴백 OCR 정확도가 0.09 까지 떨어졌고,
    그 주된 원인이 기울기와 저대비였다.

처리 순서가 중요하다:
    1. **기하 보정 먼저** (deskew / perspective) — 기하 왜곡이 있는 상태에서
       샤프닝을 하면 왜곡된 에지를 강화해 버린다.
    2. **대비 보정 다음** (CLAHE) — 조명 불균일을 지역적으로 평탄화한다.
    3. **샤프닝 마지막** (unsharp mask) — 보정으로 생긴 보간 흐림을 복원한다.
    순서를 바꾸면 각 단계가 앞 단계의 아티팩트를 증폭시킨다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import cv2

    _HAS_CV2 = True
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]
    _HAS_CV2 = False


#: OCR 입력 표준 크기. CRNN 은 높이를 고정하고 너비로 시퀀스 길이를 만든다.
#: 높이 32 는 CRNN 계열의 사실상 표준값이다(5번의 2배 다운샘플링 후 1픽셀이 됨).
OCR_TARGET_HEIGHT: int = 32
OCR_TARGET_WIDTH: int = 160

#: 기울기 보정 허용 최대 각도(도). 이보다 크면 검출 오류일 가능성이 높아
#: 보정하지 않는다 — 잘못된 회전은 원본보다 나쁘다.
MAX_DESKEW_ANGLE: float = 25.0


@dataclass(slots=True)
class RectifyResult:
    """보정 결과.

    Attributes:
        image: 보정된 이미지.
        angle: 적용된 회전 각도(도). 0 이면 회전하지 않음.
        method: 적용 방법 설명(``"perspective"`` / ``"deskew"`` / ``"none"``).
        steps: 실제로 적용된 단계 이름 목록(교재 설명/디버깅용).
    """

    image: np.ndarray
    angle: float = 0.0
    method: str = "none"
    steps: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        """메타데이터만 직렬화한다(이미지는 제외).

        Returns:
            각도/방법/단계를 담은 dict.
        """
        return {
            "angle": round(float(self.angle), 2),
            "method": self.method,
            "steps": list(self.steps or []),
            "shape": list(self.image.shape[:2]) if self.image is not None else [],
        }


def estimate_skew_angle(image: np.ndarray, *, max_angle: float = MAX_DESKEW_ANGLE) -> float:
    """번호판 crop 의 기울기 각도를 추정한다.

    방법: **문자 밑변 직선 검출(Hough 변환)**
        번호판 문자들의 아래쪽 끝은 하나의 직선 위에 놓인다. 에지에서 선분을 뽑아
        그 각도들의 **중앙값**을 기울기로 쓴다.
        왜 평균이 아니라 중앙값인가? 번호판 테두리나 배경에서 들어온 이상치 선분
        한두 개가 평균을 크게 왜곡한다. 중앙값은 이상치에 강건하다.

    대안으로 ``minAreaRect`` 를 쓸 수도 있지만, crop 경계가 번호판과 어긋나 있으면
    박스 각도가 번호판 기울기와 무관해진다. 문자 기반 추정이 더 신뢰할 만하다.

    Args:
        image: 번호판 crop(BGR 또는 그레이).
        max_angle: 이 각도를 넘는 추정값은 신뢰하지 않고 0 을 반환한다.

    Returns:
        -max_angle ~ max_angle 범위의 기울기(도). 추정 실패 시 0.0.
    """
    if not _HAS_CV2:
        return 0.0
    try:
        gray = image if np.asarray(image).ndim == 2 else cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2GRAY)
        gray = np.asarray(gray, dtype=np.uint8)
        if gray.shape[0] < 12 or gray.shape[1] < 32:
            return 0.0

        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        # 최소 선분 길이를 너비의 30% 로 둔다. 짧은 선분은 문자 내부 획이라
        # 기울기 정보가 아니라 노이즈다.
        min_len = max(16, int(gray.shape[1] * 0.30))
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 360.0,  # 0.5도 해상도
            threshold=max(20, min_len // 2),
            minLineLength=min_len,
            maxLineGap=max(4, gray.shape[1] // 20),
        )
        segments = _normalize_hough_output(lines)
        if segments.size == 0:
            return 0.0

        plate_h, plate_w = gray.shape[:2]
        angles: list[float] = []
        weights: list[float] = []
        for x1, y1, x2, y2 in segments[:120]:
            dx, dy = x2 - x1, y2 - y1
            if abs(dx) < 1e-6:
                continue
            deg = float(np.degrees(np.arctan2(dy, dx)))
            # 수평에 가까운 선분만 쓴다. 수직 선분(문자 세로획)은 기울기 정보가 없다.
            if abs(deg) > max_angle:
                continue
            # crop 경계를 **따라 달리는** 선분만 제외한다.
            #
            # 중요한 구분: 선분의 한쪽 끝이 경계에 닿는 것은 정상이다. crop 이
            # 번호판에 딱 맞으면 번호판 테두리선(= 진짜 기울기 정보)은 좌변에서
            # 우변까지 이어지므로 양 끝이 경계에 닿는다. 이를 제외하면 올바른
            # 선분이 전부 버려져 기울기가 항상 0 이 된다(실측 확인: -8도 입력에
            # 대해 유효 선분 4개가 모두 +7.9~8.0 도였는데 전부 탈락했다).
            #
            # 반면 BORDER_REPLICATE 로 생긴 줄무늬는 **양 끝이 같은 변에** 붙어
            # 수평으로 달린다. 그런 경우만 걸러낸다.
            along_top = y1 <= 1.0 and y2 <= 1.0
            along_bottom = y1 >= plate_h - 2.0 and y2 >= plate_h - 2.0
            along_left = x1 <= 1.0 and x2 <= 1.0
            along_right = x1 >= plate_w - 2.0 and x2 >= plate_w - 2.0
            if along_top or along_bottom or along_left or along_right:
                continue
            angles.append(deg)
            # 긴 선분에 더 큰 가중치를 준다. 긴 선분이 번호판 구조를 더 잘 반영한다.
            weights.append(float(np.hypot(dx, dy)))

        if not angles:
            return 0.0
        median = _weighted_median(angles, weights)
        return 0.0 if abs(median) > max_angle else round(median, 2)
    except Exception:
        return 0.0


def _normalize_hough_output(lines: Any) -> np.ndarray:
    """``HoughLinesP`` 출력을 ``(N, 4)`` 배열로 정규화한다.

    왜 필요한가? OpenCV 버전에 따라 반환 shape 이 다르다.
    4.x 는 ``(N, 1, 4)``, 5.x 는 ``(N, 4)`` 를 준다. 4.x 관례대로 ``line[0]`` 로
    풀면 5.x 에서는 스칼라가 나와 ``TypeError: 'numpy.int32' object is not iterable``
    로 터진다(실측 확인). 입구에서 한 번 정규화하면 양쪽 버전에서 동작한다.

    Args:
        lines: HoughLinesP 반환값.

    Returns:
        ``(N, 4)`` float64 배열. 입력이 None/비정상이면 빈 배열.
    """
    if lines is None:
        return np.zeros((0, 4), dtype=np.float64)
    arr = np.asarray(lines, dtype=np.float64)
    if arr.size == 0:
        return np.zeros((0, 4), dtype=np.float64)
    return arr.reshape(-1, 4)


def _weighted_median(values: list[float], weights: list[float]) -> float:
    """가중 중앙값을 계산한다.

    왜 평균이 아니라 (가중) 중앙값인가? 테두리/배경에서 들어온 이상치 선분
    한두 개가 평균을 크게 왜곡한다. 중앙값은 이상치 비율이 50% 미만이면
    영향을 받지 않는다. 여기에 선분 길이를 가중치로 주어 "긴 선분이 더 믿을 만하다"
    는 사전지식을 반영한다.

    Args:
        values: 값 리스트.
        weights: 같은 길이의 가중치 리스트(모두 양수).

    Returns:
        가중 중앙값. 입력이 비면 0.0.
    """
    if not values:
        return 0.0
    if len(weights) != len(values) or not any(w > 0 for w in weights):
        return float(np.median(values))
    order = np.argsort(np.asarray(values, dtype=np.float64))
    v = np.asarray(values, dtype=np.float64)[order]
    w = np.asarray(weights, dtype=np.float64)[order]
    cumulative = np.cumsum(w)
    half = cumulative[-1] / 2.0
    idx = int(np.searchsorted(cumulative, half))
    return float(v[min(idx, len(v) - 1)])


def deskew(image: np.ndarray, angle: float | None = None, *, max_angle: float = MAX_DESKEW_ANGLE) -> RectifyResult:
    """회전으로 기울기를 보정한다.

    경계 처리에 ``BORDER_REPLICATE`` 를 쓰는 이유: 회전 후 생기는 빈 모서리를
    0(검정)으로 채우면 이진화에서 거대한 전경 덩어리가 되어 문자 분할을 망친다.
    가장자리 픽셀을 복제하면 배경색이 자연스럽게 이어진다.

    Args:
        image: 입력 이미지.
        angle: 보정 각도(도). None 이면 ``estimate_skew_angle`` 로 추정한다.
        max_angle: 허용 최대 각도.

    Returns:
        RectifyResult. OpenCV 가 없거나 각도가 0 이면 원본을 그대로 담아 반환.
    """
    arr = np.asarray(image)
    deg = estimate_skew_angle(arr, max_angle=max_angle) if angle is None else float(angle)

    if not _HAS_CV2 or abs(deg) < 0.5 or abs(deg) > max_angle:
        return RectifyResult(image=arr, angle=0.0, method="none", steps=[])

    h, w = arr.shape[:2]
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, deg, 1.0)
    rotated = cv2.warpAffine(
        arr,
        matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,  # 문자 획 보존에는 CUBIC 이 LINEAR 보다 낫다
        borderMode=cv2.BORDER_REPLICATE,
    )
    return RectifyResult(image=rotated, angle=deg, method="deskew", steps=["deskew"])


def order_quad(points: np.ndarray) -> np.ndarray:
    """4점을 ``좌상 → 우상 → 우하 → 좌하`` 순서로 정렬한다.

    왜 순서가 중요한가? ``getPerspectiveTransform`` 은 src/dst 의 **대응 순서**로
    변환을 계산한다. 순서가 어긋나면 이미지가 뒤집히거나 대각선으로 꼬인다.

    방법: x+y 합이 최소인 점이 좌상, 최대가 우하. x-y 차가 최소면 우상, 최대면 좌하.
    (좌표계가 "오른쪽/아래로 증가" 이므로 성립한다.)

    Args:
        points: ``(4, 2)`` 좌표 배열.

    Returns:
        ``(4, 2)`` float32 정렬 배열.
    """
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if pts.shape[0] != 4:
        msg = f"4점이 필요합니다. 받은 개수={pts.shape[0]}"
        raise ValueError(msg)

    ordered = np.zeros((4, 2), dtype=np.float32)
    total = pts.sum(axis=1)
    ordered[0] = pts[int(np.argmin(total))]  # 좌상
    ordered[2] = pts[int(np.argmax(total))]  # 우하
    diff = pts[:, 0] - pts[:, 1]
    ordered[1] = pts[int(np.argmax(diff))]  # 우상 (x 큼, y 작음)
    ordered[3] = pts[int(np.argmin(diff))]  # 좌하 (x 작음, y 큼)
    return ordered


def perspective_correct(
    image: np.ndarray,
    quad: np.ndarray | list[tuple[float, float]],
    *,
    target_size: tuple[int, int] | None = None,
) -> RectifyResult:
    """4점 투시 변환으로 비스듬한 번호판을 정면으로 펼친다.

    출력 크기를 자동 계산할 때, 변의 길이를 **양쪽 변의 최댓값**으로 잡는다.
    평균을 쓰면 가까운 쪽이 축소되어 정보가 손실되고, 최댓값을 쓰면 먼 쪽이
    보간으로 확대되어 정보 손실이 없다(화질 보존 우선 원칙).

    Args:
        image: 원본 이미지.
        quad: 번호판 네 꼭짓점 좌표.
        target_size: ``(너비, 높이)``. None 이면 변 길이로 자동 결정.

    Returns:
        RectifyResult. OpenCV 가 없거나 좌표가 부적절하면 원본을 반환.
    """
    arr = np.asarray(image)
    if not _HAS_CV2:
        return RectifyResult(image=arr, angle=0.0, method="none", steps=[])

    try:
        src = order_quad(np.asarray(quad, dtype=np.float32))
    except ValueError:
        return RectifyResult(image=arr, angle=0.0, method="none", steps=[])

    (tl, tr, br, bl) = src
    width = int(round(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))))
    height = int(round(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))))
    if target_size is not None:
        width, height = int(target_size[0]), int(target_size[1])
    width = max(16, min(width, 2048))
    height = max(8, min(height, 1024))

    dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32
    )
    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(
        arr, matrix, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    return RectifyResult(image=warped, angle=0.0, method="perspective", steps=["perspective"])


def apply_clahe(image: np.ndarray, *, clip: float = 2.5, grid: int = 8) -> np.ndarray:
    """CLAHE 로 지역 대비를 평탄화한다.

    컬러 이미지는 **LAB 색공간의 L 채널에만** 적용한다.
    왜? BGR 각 채널에 따로 적용하면 채널 간 비율이 깨져 색이 왜곡된다
    (흰 번호판이 분홍빛으로 변하는 등). L(명도) 채널만 건드리면 색상(a,b)은
    보존되면서 대비만 개선된다.

    Args:
        image: BGR 또는 그레이 이미지.
        clip: 대비 제한값.
        grid: 타일 격자 크기.

    Returns:
        대비가 개선된 이미지(입력과 같은 shape).
    """
    arr = np.asarray(image)
    if not _HAS_CV2:
        return arr

    try:
        clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(int(grid), int(grid)))
        if arr.ndim == 2:
            return clahe.apply(np.asarray(arr, dtype=np.uint8))
        lab = cv2.cvtColor(np.asarray(arr, dtype=np.uint8), cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    except Exception:
        return arr


def sharpen(image: np.ndarray, *, amount: float = 1.0, radius: int = 3) -> np.ndarray:
    """언샤프 마스킹(unsharp mask)으로 샤프닝한다.

    원리: ``결과 = 원본 + amount * (원본 - 블러)``
        ``원본 - 블러`` 는 고주파 성분(에지)이다. 이를 원본에 더하면 에지가 강조된다.
        고정 커널(라플라시안 샤프닝)보다 **반경을 조절할 수 있어** 번호판 문자 획
        굵기에 맞출 수 있고, 링잉 아티팩트가 적다.

    과도한 샤프닝은 노이즈도 증폭하므로 ``amount`` 는 1.0 전후를 권장한다.

    Args:
        image: 입력 이미지.
        amount: 샤프닝 강도.
        radius: 가우시안 블러 반경(홀수로 보정됨).

    Returns:
        샤프닝된 이미지.
    """
    arr = np.asarray(image)
    if not _HAS_CV2 or float(amount) <= 0.0:
        return arr
    try:
        k = max(3, int(radius) | 1)  # 홀수 보정
        blurred = cv2.GaussianBlur(arr, (k, k), 0)
        # addWeighted 로 한 번에 계산한다(중간 배열 생성을 피해 빠르다).
        return cv2.addWeighted(arr, 1.0 + float(amount), blurred, -float(amount), 0)
    except Exception:
        return arr


def denoise(image: np.ndarray, *, strength: int = 3) -> np.ndarray:
    """중간값 필터로 점 노이즈(salt-and-pepper)를 제거한다.

    왜 가우시안이 아니라 중간값인가? 가우시안은 노이즈를 "번지게" 할 뿐이고
    문자 획도 함께 흐린다. 중간값 필터는 고립된 이상치를 완전히 제거하면서
    에지는 보존한다 — 이진화 전처리에 최적이다.

    Args:
        image: 입력 이미지.
        strength: 커널 크기(홀수로 보정, 3~7 권장).

    Returns:
        노이즈가 줄어든 이미지.
    """
    arr = np.asarray(image)
    if not _HAS_CV2:
        return arr
    try:
        k = max(3, int(strength) | 1)
        return cv2.medianBlur(arr, k)
    except Exception:
        return arr


def resize_for_ocr(
    image: np.ndarray,
    *,
    height: int = OCR_TARGET_HEIGHT,
    width: int | None = OCR_TARGET_WIDTH,
    keep_aspect: bool = True,
) -> np.ndarray:
    """OCR 입력 크기로 리사이즈한다.

    ``keep_aspect=True`` 이면 높이를 맞춘 뒤 너비를 종횡비대로 계산하고,
    목표 너비보다 짧으면 **오른쪽에 패딩**한다(왼쪽 정렬).
    왜 오른쪽 패딩인가? CTC 는 왼→오 순서로 디코딩하므로 문자가 왼쪽에
    정렬되어 있어야 패딩 구간이 전부 blank 로 예측되어 무해하다.
    가운데 정렬하면 양쪽에 blank 가 생겨 시퀀스 길이 추정이 흔들린다.

    Args:
        image: 번호판 crop.
        height: 목표 높이.
        width: 목표 너비. None 이면 종횡비대로 자유롭게 둔다.
        keep_aspect: 종횡비 유지 여부.

    Returns:
        리사이즈된 배열.
    """
    arr = np.asarray(image)
    if arr.size == 0:
        return arr
    src_h, src_w = arr.shape[:2]
    target_h = max(8, int(height))

    if not keep_aspect and width is not None:
        return _resize(arr, int(width), target_h)

    scale = target_h / max(1, src_h)
    new_w = max(8, int(round(src_w * scale)))
    resized = _resize(arr, new_w, target_h)

    if width is None:
        return resized
    target_w = max(8, int(width))
    if new_w == target_w:
        return resized
    if new_w > target_w:
        # 너무 길면 줄인다(비율이 약간 깨지지만 모델 입력 폭은 고정이어야 한다).
        return _resize(resized, target_w, target_h)

    # 오른쪽 패딩. 패딩 색은 가장자리 평균을 쓴다(검정 패딩은 거짓 에지를 만든다).
    pad_w = target_w - new_w
    edge = resized[:, -1:, ...] if resized.ndim == 3 else resized[:, -1:]
    fill = np.repeat(edge, pad_w, axis=1)
    return np.concatenate([resized, fill], axis=1)


def _resize(arr: np.ndarray, width: int, height: int) -> np.ndarray:
    """내부 리사이즈 헬퍼(OpenCV 있으면 보간, 없으면 최근접).

    Args:
        arr: 입력 배열.
        width: 목표 너비.
        height: 목표 높이.

    Returns:
        리사이즈된 배열.
    """
    w, h = max(1, int(width)), max(1, int(height))
    src_h, src_w = arr.shape[:2]
    if (src_w, src_h) == (w, h):
        return arr
    if _HAS_CV2:
        interp = cv2.INTER_AREA if w * h < src_w * src_h else cv2.INTER_CUBIC
        return cv2.resize(arr, (w, h), interpolation=interp)
    ys = np.clip((np.arange(h) * src_h / h).astype(np.int64), 0, src_h - 1)
    xs = np.clip((np.arange(w) * src_w / w).astype(np.int64), 0, src_w - 1)
    return arr[ys][:, xs]


def rectify_plate(
    plate_bgr: np.ndarray,
    *,
    quad: np.ndarray | list[tuple[float, float]] | None = None,
    angle: float | None = None,
    use_clahe: bool = True,
    use_sharpen: bool = True,
    use_denoise: bool = False,
    clahe_clip: float = 2.5,
    sharpen_amount: float = 0.8,
) -> RectifyResult:
    """번호판 crop 에 전체 보정 파이프라인을 적용한다.

    순서: **투시(또는 기울기) → 노이즈 제거 → CLAHE → 샤프닝**

    기하 보정을 먼저 하는 이유는 모듈 docstring 에 설명했다. 노이즈 제거를
    CLAHE 앞에 두는 이유: CLAHE 는 지역 대비를 증폭하므로 노이즈가 남아 있으면
    노이즈까지 증폭된다. 먼저 지우는 것이 맞다.

    Args:
        plate_bgr: 번호판 영역 crop(BGR).
        quad: 4점 좌표가 있으면 투시 변환을 쓴다(기울기 회전보다 정확하다).
        angle: 기울기 각도. quad 가 없을 때 사용. None 이면 자동 추정.
        use_clahe: CLAHE 적용 여부.
        use_sharpen: 샤프닝 적용 여부.
        use_denoise: 중간값 필터 적용 여부(노이즈가 심한 야간 영상에 권장).
        clahe_clip: CLAHE 대비 제한값.
        sharpen_amount: 샤프닝 강도.

    Returns:
        RectifyResult. 어떤 단계가 실패해도 예외 없이 가능한 만큼만 적용한다.
    """
    arr = np.asarray(plate_bgr)
    steps: list[str] = []
    applied_angle = 0.0
    method = "none"

    # 1. 기하 보정
    if quad is not None:
        geo = perspective_correct(arr, quad)
        if geo.method == "perspective":
            arr = geo.image
            method = "perspective"
            steps.append("perspective")
    if method == "none":
        geo = deskew(arr, angle)
        if geo.method == "deskew":
            arr = geo.image
            applied_angle = geo.angle
            method = "deskew"
            steps.append("deskew")

    # 2. 노이즈 제거
    if use_denoise:
        arr = denoise(arr)
        steps.append("denoise")

    # 3. 지역 대비 평탄화
    if use_clahe:
        arr = apply_clahe(arr, clip=clahe_clip)
        steps.append("clahe")

    # 4. 샤프닝
    if use_sharpen:
        arr = sharpen(arr, amount=sharpen_amount)
        steps.append("sharpen")

    return RectifyResult(image=arr, angle=applied_angle, method=method, steps=steps)


def rectify_summary() -> dict[str, Any]:
    """보정 모듈의 가용 기능을 보고한다(진단용).

    Returns:
        OpenCV 가용 여부와 기본 파라미터.
    """
    return {
        "opencv": _HAS_CV2,
        "ocr_target": [OCR_TARGET_HEIGHT, OCR_TARGET_WIDTH],
        "max_deskew_angle": MAX_DESKEW_ANGLE,
    }


__all__ = [
    "MAX_DESKEW_ANGLE",
    "OCR_TARGET_HEIGHT",
    "OCR_TARGET_WIDTH",
    "RectifyResult",
    "apply_clahe",
    "denoise",
    "deskew",
    "estimate_skew_angle",
    "order_quad",
    "perspective_correct",
    "rectify_plate",
    "rectify_summary",
    "resize_for_ocr",
    "sharpen",
]
