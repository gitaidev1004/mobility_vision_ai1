"""영상 품질 평가.

왜 품질 평가가 별도 단계로 필요한가?
    번호판 인식이 실패했을 때 원인은 둘 중 하나다 — ① 모델이 약하다,
    ② 입력 영상이 애초에 판독 불가다. 운영자에게 "흐려서 못 읽었습니다"라고
    알려 주면 **카메라 초점/조명을 고치는 조치**로 이어진다. 반대로 원인을
    알려 주지 않으면 모델 탓만 하다 끝난다. 품질 점수는 운영 개선의 출발점이다.

계약서 응답 스키마(``quality`` 필드)와 1:1 대응한다:
    ``{"score": 0.86, "blur": 142.3, "brightness": 118.4, "plate_px_width": 184,
       "readable": true, "reason": null}``

의존성: numpy 만 필수(OpenCV 는 있으면 더 빠른 경로를 쓴다).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

try:
    import cv2

    _HAS_CV2 = True
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]
    _HAS_CV2 = False


# ---------------------------------------------------------------------------
# 기본 임계값 — config 에서 덮어쓸 수 있다.
# ---------------------------------------------------------------------------

#: 라플라시안 분산이 이 값 미만이면 "흐리다" 고 판단한다.
#: 왜 60 인가? 320x240 이상 해상도의 선명한 번호판 crop 은 보통 150~600 이 나오고,
#: 초점이 빗나간 영상은 10~40 에 머문다. 60 은 그 사이의 보수적 경계다.
DEFAULT_BLUR_MIN: float = 60.0

#: 평균 밝기 허용 범위. 40 미만은 야간 노출 부족, 215 초과는 역광 포화다.
#: 포화된 영역은 정보가 소실되어 어떤 알고리즘으로도 복원할 수 없다.
DEFAULT_BRIGHTNESS_MIN: float = 40.0
DEFAULT_BRIGHTNESS_MAX: float = 215.0

#: 번호판 가로 픽셀폭 최소값.
#: 왜 72px 인가? 한국 번호판은 7~8자다. 문자 하나당 최소 8~9px 폭이 있어야
#: 획이 1px 이상으로 표현되어 OCR 이 가능하다(7자 x 9px ≈ 63px + 여백 ≈ 72px).
#: 이 값은 "광학적 하한" 이라 모델을 바꿔도 변하지 않는다.
DEFAULT_PLATE_PX_MIN: int = 72

#: 종합 점수가 이 값 미만이면 ``readable=False``.
DEFAULT_READABLE_SCORE: float = 0.35


@dataclass(slots=True)
class QualityReport:
    """품질 평가 결과.

    Attributes:
        score: 0~1 종합 품질 점수.
        blur: 라플라시안 분산(선명도). 클수록 선명하다.
        brightness: 평균 밝기(0~255).
        contrast: 밝기 표준편차. 낮으면 대비가 부족하다.
        plate_px_width: 번호판 가로 픽셀폭(검출 전이면 0).
        readable: 판독 가능 여부.
        reason: 판독 불가 사유(가능하면 None).
        detail: 하위 점수 내역(디버깅/교재 설명용).
    """

    score: float
    blur: float
    brightness: float
    contrast: float
    plate_px_width: int
    readable: bool
    reason: str | None = None
    detail: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """계약서 ``quality`` 필드 스키마로 직렬화한다.

        Returns:
            계약서와 키/타입이 일치하는 dict.
        """
        return {
            "score": round(float(self.score), 4),
            "blur": round(float(self.blur), 2),
            "brightness": round(float(self.brightness), 2),
            "plate_px_width": int(self.plate_px_width),
            "readable": bool(self.readable),
            "reason": self.reason,
        }


def to_gray(image: np.ndarray) -> np.ndarray:
    """BGR 또는 그레이스케일 입력을 ``(H, W)`` uint8 그레이스케일로 만든다.

    ``app.detection.preprocess.to_gray`` 와 같은 일을 하지만, 이 모듈이
    detection 패키지에 의존하지 않도록 독립 구현을 둔다 — 품질 평가는
    검출 이전/이후 어디서든 호출되므로 의존 방향을 단순하게 유지한다.

    Args:
        image: 2D 또는 3D 이미지 배열.

    Returns:
        uint8 그레이스케일 배열.

    Raises:
        ValueError: 입력이 비었거나 차원이 2/3 이 아닐 때.
    """
    arr = np.asarray(image)
    if arr.size == 0:
        msg = "빈 이미지입니다."
        raise ValueError(msg)
    if arr.dtype != np.uint8:
        peak = float(np.nanmax(arr))
        scale = 255.0 if 0.0 < peak <= 1.0 + 1e-6 else 1.0
        arr = np.clip(np.nan_to_num(arr.astype(np.float32)) * scale, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        if arr.shape[2] == 1:
            return arr[:, :, 0]
        if _HAS_CV2 and arr.shape[2] == 3:
            return cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        weights = np.array([0.114, 0.587, 0.299], dtype=np.float32)
        return np.clip(arr[:, :, :3].astype(np.float32) @ weights, 0, 255).astype(np.uint8)
    msg = f"지원하지 않는 차원: ndim={arr.ndim}"
    raise ValueError(msg)


def laplacian_variance(image: np.ndarray) -> float:
    """라플라시안 분산으로 선명도를 측정한다.

    원리:
        라플라시안은 2차 미분 연산자로, 밝기 변화가 급한 곳(에지)에서 큰 값을 낸다.
        선명한 영상은 에지가 뚜렷해 라플라시안 응답의 **분산**이 크고, 흐린 영상은
        에지가 번져 응답이 0 근처에 몰려 분산이 작다.
        평균이 아니라 분산을 쓰는 이유: 라플라시안의 평균은 부호가 상쇄되어
        항상 0 에 가깝다. 변동의 크기를 봐야 선명도를 알 수 있다.

    사용 커널(3x3)::

        0   1   0
        1  -4   1
        0   1   0

    Args:
        image: 입력 이미지(BGR 또는 그레이).

    Returns:
        라플라시안 분산(0 이상). 입력이 너무 작으면 0.0.
    """
    try:
        gray = to_gray(image)
    except ValueError:
        return 0.0
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return 0.0

    if _HAS_CV2:
        lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
        return float(np.var(lap))

    # numpy 폴백: 4-이웃 라플라시안을 슬라이싱으로 계산.
    f = gray.astype(np.float64)
    lap = np.zeros_like(f)
    lap[1:-1, 1:-1] = (
        f[:-2, 1:-1] + f[2:, 1:-1] + f[1:-1, :-2] + f[1:-1, 2:] - 4.0 * f[1:-1, 1:-1]
    )
    return float(np.var(lap[1:-1, 1:-1]))


def brightness_stats(image: np.ndarray) -> tuple[float, float]:
    """평균 밝기와 표준편차(대비)를 계산한다.

    Args:
        image: 입력 이미지.

    Returns:
        ``(평균 밝기, 표준편차)``. 둘 다 0~255 스케일.
    """
    try:
        gray = to_gray(image)
    except ValueError:
        return 0.0, 0.0
    return float(gray.mean()), float(gray.std())


def saturation_ratio(image: np.ndarray, *, low: int = 4, high: int = 251) -> tuple[float, float]:
    """포화(클리핑) 픽셀 비율을 계산한다.

    왜 평균 밝기만으로 부족한가?
        역광에서 번호판만 하얗게 날아가도 전체 평균은 정상 범위일 수 있다.
        포화 비율은 "복원 불가능하게 소실된 정보량" 을 직접 측정하므로
        평균 밝기가 놓치는 실패를 잡아낸다.

    Args:
        image: 입력 이미지.
        low: 이 값 이하를 암부 포화로 본다.
        high: 이 값 이상을 명부 포화로 본다.

    Returns:
        ``(암부 포화 비율, 명부 포화 비율)``. 각 0~1.
    """
    try:
        gray = to_gray(image)
    except ValueError:
        return 0.0, 0.0
    total = float(gray.size)
    if total <= 0:
        return 0.0, 0.0
    return float((gray <= low).sum() / total), float((gray >= high).sum() / total)


def _blur_score(blur: float, blur_min: float) -> float:
    """선명도 값을 0~1 점수로 변환한다.

    선형이 아니라 **포화 곡선**(``x / (x + k)`` 형태)을 쓰는 이유:
        선명도는 임계값 부근에서 판독 가능성이 급격히 바뀌고, 충분히 선명해진
        뒤에는 더 선명해져도 의미가 없다. 포화 곡선이 이 성질과 맞는다.
        ``blur == blur_min`` 일 때 정확히 0.5 가 되도록 계수를 잡았다.

    Args:
        blur: 라플라시안 분산.
        blur_min: 기준 임계값.

    Returns:
        0~1 점수.
    """
    b = max(0.0, float(blur))
    k = max(1e-6, float(blur_min))
    return float(b / (b + k))


def _brightness_score(brightness: float, lo: float, hi: float) -> float:
    """평균 밝기를 0~1 점수로 변환한다(허용 범위 내부는 1.0, 밖으로 갈수록 감소).

    Args:
        brightness: 평균 밝기.
        lo: 허용 하한.
        hi: 허용 상한.

    Returns:
        0~1 점수.
    """
    v = float(brightness)
    if lo <= v <= hi:
        return 1.0
    # 범위를 벗어난 거리를 허용 범위 폭의 40% 로 정규화해 감쇠시킨다.
    span = max(1.0, (hi - lo) * 0.40)
    distance = (lo - v) if v < lo else (v - hi)
    return float(np.clip(1.0 - distance / span, 0.0, 1.0))


def _width_score(plate_px_width: int, px_min: int) -> float:
    """번호판 픽셀폭을 0~1 점수로 변환한다.

    ``px_min`` 에서 0.5, ``px_min * 2.5`` 에서 1.0 에 가까워지는 포화 곡선이다.
    너비는 "문자당 몇 픽셀" 로 환산되는 물리량이라 하한이 명확하다.

    Args:
        plate_px_width: 번호판 가로 픽셀폭. 0 이면 검출 전이므로 중립값 0.5 반환.
        px_min: 최소 요구 픽셀폭.

    Returns:
        0~1 점수.
    """
    w = int(plate_px_width)
    if w <= 0:
        # 검출 전에는 너비를 알 수 없다. 0 점을 주면 종합 점수가 부당하게
        # 낮아지므로 중립값을 준다.
        return 0.5
    k = max(1.0, float(px_min))
    return float(np.clip(w / (w + k), 0.0, 1.0))


def _contrast_score(contrast: float) -> float:
    """대비(표준편차)를 0~1 점수로 변환한다.

    번호판 crop 의 표준편차는 선명할 때 45~80 이고, 안개/저대비에서는 15 이하다.
    표준편차 50 에서 1.0 에 도달하도록 선형 포화시킨다.

    Args:
        contrast: 밝기 표준편차.

    Returns:
        0~1 점수.
    """
    return float(np.clip(float(contrast) / 50.0, 0.0, 1.0))


def assess(
    image: np.ndarray,
    *,
    plate_px_width: int = 0,
    blur_min: float = DEFAULT_BLUR_MIN,
    brightness_min: float = DEFAULT_BRIGHTNESS_MIN,
    brightness_max: float = DEFAULT_BRIGHTNESS_MAX,
    plate_px_min: int = DEFAULT_PLATE_PX_MIN,
    readable_score: float = DEFAULT_READABLE_SCORE,
) -> QualityReport:
    """영상 품질을 종합 평가한다.

    가중치 설계 근거:
        - 선명도 0.38 — 번호판 OCR 실패의 압도적 1위 원인이다.
        - 픽셀폭 0.27 — 광학적 하한이라 후처리로 복원할 수 없다.
        - 밝기   0.20 — CLAHE 로 상당 부분 보정 가능하므로 가중치를 낮춘다.
        - 대비   0.15 — 밝기와 상관이 높아 중복이므로 가장 낮게 둔다.
      그리고 포화 비율이 높으면 **곱셈 벌점**을 적용한다. 가산 벌점으로는
      "완전히 날아간 영상" 이 다른 항목 점수로 가려지기 때문이다.

    Args:
        image: 평가 대상 이미지(전체 장면 또는 번호판 crop).
        plate_px_width: 검출된 번호판 가로 픽셀폭(없으면 0).
        blur_min: 선명도 임계값.
        brightness_min: 밝기 하한.
        brightness_max: 밝기 상한.
        plate_px_min: 번호판 최소 픽셀폭.
        readable_score: 이 점수 미만이면 판독 불가로 표시한다.

    Returns:
        QualityReport. 예외는 던지지 않으며, 평가 불가 시 score 0 의 보고서를 준다.
    """
    try:
        gray = to_gray(image)
    except ValueError:
        return QualityReport(
            score=0.0,
            blur=0.0,
            brightness=0.0,
            contrast=0.0,
            plate_px_width=int(plate_px_width),
            readable=False,
            reason="invalid-image",
        )

    blur = laplacian_variance(gray)
    brightness, contrast = brightness_stats(gray)
    dark_ratio, bright_ratio = saturation_ratio(gray)

    s_blur = _blur_score(blur, blur_min)
    s_bright = _brightness_score(brightness, brightness_min, brightness_max)
    s_width = _width_score(plate_px_width, plate_px_min)
    s_contrast = _contrast_score(contrast)

    base = 0.38 * s_blur + 0.27 * s_width + 0.20 * s_bright + 0.15 * s_contrast

    # 포화 벌점: 암부/명부 포화가 30% 를 넘으면 급격히 깎는다.
    worst_saturation = max(dark_ratio, bright_ratio)
    saturation_penalty = float(np.clip(1.0 - max(0.0, worst_saturation - 0.30) * 2.0, 0.0, 1.0))
    score = float(np.clip(base * saturation_penalty, 0.0, 1.0))

    reason = _failure_reason(
        blur=blur,
        blur_min=blur_min,
        brightness=brightness,
        brightness_min=brightness_min,
        brightness_max=brightness_max,
        plate_px_width=plate_px_width,
        plate_px_min=plate_px_min,
        contrast=contrast,
        dark_ratio=dark_ratio,
        bright_ratio=bright_ratio,
    )
    readable = score >= float(readable_score) and reason is None

    return QualityReport(
        score=score,
        blur=blur,
        brightness=brightness,
        contrast=contrast,
        plate_px_width=int(plate_px_width),
        readable=readable,
        # 판독 가능하면 사유를 노출하지 않는다(계약서: readable 이면 reason null).
        reason=None if readable else reason,
        detail={
            "blur_score": round(s_blur, 4),
            "brightness_score": round(s_bright, 4),
            "width_score": round(s_width, 4),
            "contrast_score": round(s_contrast, 4),
            "dark_ratio": round(dark_ratio, 4),
            "bright_ratio": round(bright_ratio, 4),
            "saturation_penalty": round(saturation_penalty, 4),
        },
    )


def _failure_reason(
    *,
    blur: float,
    blur_min: float,
    brightness: float,
    brightness_min: float,
    brightness_max: float,
    plate_px_width: int,
    plate_px_min: int,
    contrast: float,
    dark_ratio: float,
    bright_ratio: float,
) -> str | None:
    """판독 불가 사유를 **가장 심각한 하나**로 정한다.

    여러 사유를 나열하지 않는 이유: 운영자는 "무엇을 먼저 고칠까" 를 알고 싶어한다.
    우선순위는 "조치 난이도가 낮고 효과가 큰 순서" 다 — 해상도(카메라 위치/줌) →
    초점 → 노출 → 대비.

    Args:
        blur: 라플라시안 분산.
        blur_min: 선명도 임계값.
        brightness: 평균 밝기.
        brightness_min: 밝기 하한.
        brightness_max: 밝기 상한.
        plate_px_width: 번호판 픽셀폭.
        plate_px_min: 최소 픽셀폭.
        contrast: 표준편차.
        dark_ratio: 암부 포화 비율.
        bright_ratio: 명부 포화 비율.

    Returns:
        사유 문자열(영문 slug) 또는 None(문제 없음).
    """
    if 0 < plate_px_width < plate_px_min:
        return f"plate-too-small({plate_px_width}px<{plate_px_min}px)"
    if blur < blur_min:
        return f"too-blurry(laplacian-var={blur:.1f}<{blur_min:.0f})"
    if bright_ratio > 0.35:
        return f"overexposed(saturated={bright_ratio:.0%})"
    if dark_ratio > 0.35:
        return f"underexposed(dark={dark_ratio:.0%})"
    if brightness < brightness_min:
        return f"too-dark(mean={brightness:.0f}<{brightness_min:.0f})"
    # 평균 밝기 상한 초과는 **포화 증거가 함께 있을 때만** 실패로 본다.
    #
    # 왜 평균만으로는 안 되는가? 한국 비사업용 번호판은 흰 바탕(RGB 245)이다.
    # 번호판 crop 의 평균 밝기는 정상적으로도 210~220 이 나온다
    # (실측: 선명한 흰 번호판 crop 평균 215.25). 평균만 보면 완벽하게 선명한
    # 번호판을 "노출 과다" 로 반려하게 된다.
    # 진짜 역광 포화는 **정보 소실**을 동반하므로, 명부 클리핑 비율(>=251)이
    # 10% 를 넘거나 대비가 무너진 경우에만 실패로 판정한다.
    if brightness > brightness_max and (bright_ratio > 0.10 or contrast < 25.0):
        return f"too-bright(mean={brightness:.0f}>{brightness_max:.0f}, saturated={bright_ratio:.0%})"
    if contrast < 12.0:
        return f"low-contrast(std={contrast:.1f})"
    return None


def assess_plate_crop(
    crop_image: np.ndarray,
    *,
    blur_min: float = DEFAULT_BLUR_MIN,
    plate_px_min: int = DEFAULT_PLATE_PX_MIN,
    **kwargs: Any,
) -> QualityReport:
    """번호판 crop 전용 품질 평가(픽셀폭을 crop 너비로 자동 설정).

    Args:
        crop_image: 번호판 영역 crop.
        blur_min: 선명도 임계값.
        plate_px_min: 최소 픽셀폭.
        **kwargs: ``assess`` 에 그대로 전달.

    Returns:
        QualityReport.
    """
    arr = np.asarray(crop_image)
    width = int(arr.shape[1]) if arr.ndim >= 2 else 0
    return assess(
        arr,
        plate_px_width=width,
        blur_min=blur_min,
        plate_px_min=plate_px_min,
        **kwargs,
    )


__all__ = [
    "DEFAULT_BLUR_MIN",
    "DEFAULT_BRIGHTNESS_MAX",
    "DEFAULT_BRIGHTNESS_MIN",
    "DEFAULT_PLATE_PX_MIN",
    "DEFAULT_READABLE_SCORE",
    "QualityReport",
    "assess",
    "assess_plate_crop",
    "brightness_stats",
    "laplacian_variance",
    "saturation_ratio",
    "to_gray",
]
