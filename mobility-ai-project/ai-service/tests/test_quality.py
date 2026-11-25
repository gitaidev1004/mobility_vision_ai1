"""``app.pipeline.quality`` 단위 테스트.

numpy 만 필요하다(OpenCV 가 있으면 더 빠른 경로를 쓰지만 결과는 같다).

품질 평가는 "흐린 영상을 흐리다고 말하는가" 라는 단순한 질문을 검증하는 것이
핵심이다. 그래서 테스트도 **인공적으로 열화시킨 영상**을 만들어 비교한다.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.pipeline.quality import (
    DEFAULT_BLUR_MIN,
    DEFAULT_PLATE_PX_MIN,
    QualityReport,
    assess,
    assess_plate_crop,
    brightness_stats,
    laplacian_variance,
    saturation_ratio,
    to_gray,
)


#: 테스트 체커보드의 명/암 값.
#:
#: 왜 0/255 가 아니라 30/225 인가?
#:   0 과 255 는 **센서 포화(클리핑) 값**이다. 순수 0/255 영상은 모든 픽셀이
#:   포화 상태여서 품질 평가가 (정당하게) 포화 벌점을 최대로 매긴다.
#:   실제 번호판은 검정 문자 약 24, 흰 바탕 약 245 로 포화 직전에 머문다.
#:   테스트 픽스처도 그 현실을 반영해야 평가 로직을 올바르게 검증할 수 있다.
DARK_TONE: int = 30
LIGHT_TONE: int = 225


def make_checkerboard(height: int = 64, width: int = 256, cell: int = 8) -> np.ndarray:
    """고주파 성분이 많은 선명한 테스트 이미지를 만든다.

    체커보드는 에지가 규칙적으로 많아 라플라시안 분산이 크다 — "선명한 영상"
    의 극단적 예시로 쓰기에 적합하다. 다만 톤은 포화를 피해
    ``DARK_TONE``/``LIGHT_TONE`` 을 쓴다(상단 주석 참조).

    Args:
        height: 이미지 높이.
        width: 이미지 너비.
        cell: 격자 한 칸 크기.

    Returns:
        ``(H, W, 3)`` uint8 BGR 배열.
    """
    ys, xs = np.mgrid[0:height, 0:width]
    mask = (((ys // cell) + (xs // cell)) % 2).astype(bool)
    pattern = np.where(mask, LIGHT_TONE, DARK_TONE).astype(np.uint8)
    return np.repeat(pattern[:, :, None], 3, axis=2)


def make_clipped_checkerboard(height: int = 64, width: int = 256, cell: int = 8) -> np.ndarray:
    """포화된(0/255) 체커보드 — 포화 벌점 검증용.

    Args:
        height: 이미지 높이.
        width: 이미지 너비.
        cell: 격자 한 칸 크기.

    Returns:
        ``(H, W, 3)`` uint8 BGR 배열(모든 픽셀이 0 또는 255).
    """
    ys, xs = np.mgrid[0:height, 0:width]
    pattern = (((ys // cell) + (xs // cell)) % 2).astype(np.uint8) * 255
    return np.repeat(pattern[:, :, None], 3, axis=2)


def blur_image(image: np.ndarray, ksize: int = 15) -> np.ndarray:
    """이미지를 흐리게 만든다(OpenCV 없으면 박스 평균으로 대체).

    Args:
        image: 입력 이미지.
        ksize: 커널 크기.

    Returns:
        흐려진 이미지.
    """
    try:
        import cv2

        return cv2.GaussianBlur(image, (ksize | 1, ksize | 1), 0)
    except ImportError:
        # numpy 박스 블러: 반복 평균으로 가우시안을 근사한다.
        out = image.astype(np.float32)
        for _ in range(4):
            padded = np.pad(out, ((2, 2), (2, 2), (0, 0)), mode="edge")
            out = sum(
                padded[dy : dy + out.shape[0], dx : dx + out.shape[1]]
                for dy in range(5)
                for dx in range(5)
            ) / 25.0
        return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# to_gray
# ---------------------------------------------------------------------------


def test_to_gray_from_bgr():
    """BGR 3채널을 2D 그레이스케일로 변환한다."""
    gray = to_gray(make_checkerboard())
    assert gray.ndim == 2
    assert gray.dtype == np.uint8


def test_to_gray_passthrough_2d():
    """이미 2D 인 입력은 그대로 통과시킨다."""
    source = np.full((10, 10), 128, dtype=np.uint8)
    assert to_gray(source).shape == (10, 10)


def test_to_gray_single_channel():
    """``(H, W, 1)`` 입력도 처리한다."""
    assert to_gray(np.zeros((8, 8, 1), dtype=np.uint8)).shape == (8, 8)


def test_to_gray_float_input_scaled():
    """0~1 범위 float 입력을 0~255 로 스케일링한다."""
    gray = to_gray(np.ones((4, 4, 3), dtype=np.float32) * 0.5)
    assert 120 <= int(gray.mean()) <= 135


def test_to_gray_empty_raises():
    """빈 배열은 ``ValueError`` 를 던진다(상위 계층이 처리한다)."""
    with pytest.raises(ValueError, match="빈 이미지"):
        to_gray(np.zeros((0, 0, 3), dtype=np.uint8))


# ---------------------------------------------------------------------------
# laplacian_variance
# ---------------------------------------------------------------------------


def test_sharp_has_higher_laplacian_than_blurred():
    """선명한 영상의 라플라시안 분산이 흐린 영상보다 크다.

    이것이 블러 탐지의 전체 원리다. 이 관계가 깨지면 품질 평가가 무의미해진다.
    """
    sharp = make_checkerboard()
    blurred = blur_image(sharp, 15)
    assert laplacian_variance(sharp) > laplacian_variance(blurred)


def test_flat_image_has_near_zero_laplacian():
    """단색 영상은 에지가 없어 분산이 0 에 가깝다."""
    flat = np.full((64, 64, 3), 128, dtype=np.uint8)
    assert laplacian_variance(flat) < 1.0


def test_laplacian_on_tiny_image():
    """3x3 미만 이미지는 0 을 반환한다(예외 없음)."""
    assert laplacian_variance(np.zeros((2, 2, 3), dtype=np.uint8)) == 0.0


def test_laplacian_on_invalid_input():
    """빈 입력에도 예외 없이 0 을 반환한다."""
    assert laplacian_variance(np.zeros((0, 0, 3), dtype=np.uint8)) == 0.0


# ---------------------------------------------------------------------------
# brightness / saturation
# ---------------------------------------------------------------------------


def test_brightness_stats():
    """평균 밝기와 표준편차를 올바르게 계산한다."""
    mean, std = brightness_stats(np.full((16, 16, 3), 100, dtype=np.uint8))
    assert mean == pytest.approx(100.0, abs=1.0)
    assert std == pytest.approx(0.0, abs=0.5)


def test_brightness_stats_checkerboard():
    """명/암이 반반인 체커보드는 평균이 두 톤의 중간, 표준편차가 크다."""
    mean, std = brightness_stats(make_checkerboard())
    assert (DARK_TONE + LIGHT_TONE) / 2 - 20 <= mean <= (DARK_TONE + LIGHT_TONE) / 2 + 20
    assert std > 80


def test_saturation_ratio_detects_clipping():
    """포화(클리핑) 픽셀 비율을 측정한다."""
    white = np.full((16, 16, 3), 255, dtype=np.uint8)
    dark_ratio, bright_ratio = saturation_ratio(white)
    assert bright_ratio == pytest.approx(1.0)
    assert dark_ratio == pytest.approx(0.0)

    black = np.zeros((16, 16, 3), dtype=np.uint8)
    dark_ratio, bright_ratio = saturation_ratio(black)
    assert dark_ratio == pytest.approx(1.0)
    assert bright_ratio == pytest.approx(0.0)


def test_saturation_ratio_catches_what_mean_misses():
    """포화 비율이 평균 밝기가 놓치는 실패를 잡는다.

    절반은 완전 흰색, 절반은 완전 검정이면 평균은 127(정상)이지만
    **모든 픽셀의 정보가 소실**된 상태다. 포화 비율은 이를 드러낸다.
    """
    half = np.zeros((16, 16, 3), dtype=np.uint8)
    half[:8] = 255
    mean, _std = brightness_stats(half)
    assert 120 <= mean <= 135  # 평균은 정상 범위

    dark_ratio, bright_ratio = saturation_ratio(half)
    assert dark_ratio + bright_ratio == pytest.approx(1.0)  # 전부 포화


# ---------------------------------------------------------------------------
# assess
# ---------------------------------------------------------------------------


def test_assess_sharp_is_readable():
    """선명하고 충분히 큰 영상은 판독 가능으로 판정된다."""
    report = assess(make_checkerboard(80, 320), plate_px_width=320)
    assert report.readable is True
    assert report.reason is None
    assert report.score > 0.5


def test_assess_blurred_is_not_readable():
    """흐린 영상은 판독 불가로 판정되고 사유가 블러임을 밝힌다."""
    blurred = blur_image(np.full((80, 320, 3), 128, dtype=np.uint8), 15)
    report = assess(blurred, plate_px_width=320)
    assert report.readable is False
    assert report.reason is not None
    assert "blur" in report.reason


def test_assess_small_plate_rejected():
    """픽셀폭이 하한 미만이면 판독 불가이고 사유에 크기가 명시된다.

    이 하한은 광학적 제약이라 모델을 바꿔도 변하지 않는다 —
    7자 번호판에서 문자당 최소 8~9px 가 필요하다.
    """
    report = assess(make_checkerboard(20, 50), plate_px_width=50)
    assert report.readable is False
    assert report.reason is not None
    assert "too-small" in report.reason


def test_assess_dark_image_rejected():
    """너무 어두운 영상은 판독 불가이고 사유가 밝기임을 밝힌다."""
    dark = (make_checkerboard(80, 320).astype(np.float32) * 0.06).astype(np.uint8)
    report = assess(dark, plate_px_width=320)
    assert report.readable is False
    assert report.reason is not None
    assert "dark" in report.reason


def test_white_plate_is_not_rejected_as_too_bright():
    """흰 바탕 번호판을 "노출 과다" 로 잘못 반려하지 않는다.

    한국 비사업용 번호판은 흰 바탕(RGB 245)이라 crop 평균 밝기가 정상적으로도
    210~220 이 나온다. 평균만으로 판정하면 **완벽하게 선명한 번호판을 반려**한다.
    포화 증거(명부 클리핑 >= 10%)가 함께 있을 때만 실패로 봐야 한다.
    """
    # 흰 바탕(245) + 검정 문자(24) 패턴 — 실제 번호판 구조를 모사.
    plate = np.full((80, 320, 3), 245, dtype=np.uint8)
    for i in range(7):
        x = 20 + i * 42
        plate[20:60, x : x + 22] = 24

    report = assess_plate_crop(plate)
    assert report.brightness > 180  # 평균은 높다
    assert report.readable is True  # 그래도 판독 가능해야 한다
    assert report.reason is None


def test_clipping_penalty_lowers_score():
    """포화(0/255)된 영상은 선명해도 점수가 깎인다.

    0/255 체커보드는 라플라시안 분산이 매우 크지만(에지가 날카롭다) 모든 픽셀이
    클리핑되어 **계조 정보가 전부 소실**된 상태다. 실제 센서에서 이런 영상이
    나오면 문자 획의 굵기/농담 정보가 사라져 OCR 이 어려워진다.
    선명도만 보고 "좋은 영상" 이라 판정하면 안 된다 — 포화 벌점이 이를 막는다.
    """
    clipped = assess(make_clipped_checkerboard(80, 320), plate_px_width=320)
    normal = assess(make_checkerboard(80, 320), plate_px_width=320)

    assert clipped.detail["dark_ratio"] + clipped.detail["bright_ratio"] > 0.95
    assert clipped.detail["saturation_penalty"] < 1.0
    # 선명도는 오히려 더 높지만 종합 점수는 낮다.
    assert clipped.blur > normal.blur
    assert clipped.score < normal.score


def test_assess_truly_overexposed_rejected():
    """정보가 실제로 소실된 과노출 영상은 반려한다."""
    overexposed = np.full((80, 320, 3), 254, dtype=np.uint8)
    report = assess_plate_crop(overexposed)
    assert report.readable is False
    assert report.reason is not None


def test_assess_invalid_image():
    """빈 이미지는 예외 없이 score 0 보고서를 반환한다."""
    report = assess(np.zeros((0, 0, 3), dtype=np.uint8))
    assert report.score == 0.0
    assert report.readable is False
    assert report.reason == "invalid-image"


def test_assess_score_ordering():
    """선명 > 흐림 > 단색 순으로 점수가 매겨진다."""
    sharp = assess(make_checkerboard(80, 320), plate_px_width=320).score
    blurred = assess(blur_image(make_checkerboard(80, 320), 15), plate_px_width=320).score
    flat = assess(np.full((80, 320, 3), 128, dtype=np.uint8), plate_px_width=320).score
    assert sharp > blurred > flat


def test_assess_plate_crop_infers_width():
    """``assess_plate_crop`` 이 crop 너비를 자동으로 픽셀폭으로 쓴다."""
    report = assess_plate_crop(make_checkerboard(40, 200))
    assert report.plate_px_width == 200


def test_assess_width_zero_is_neutral():
    """검출 전(픽셀폭 0)에는 너비 점수를 중립으로 둔다.

    0 점을 주면 종합 점수가 부당하게 낮아져, 검출 실패 원인을 "너비" 로
    잘못 지목하게 된다.
    """
    sharp = make_checkerboard(80, 320)
    without_width = assess(sharp, plate_px_width=0)
    with_width = assess(sharp, plate_px_width=320)
    # 너비를 알면 점수가 오르지만, 모를 때도 0 이 되지는 않는다.
    assert 0.0 < without_width.score < with_width.score


def test_quality_report_to_dict_contract_shape():
    """``to_dict`` 가 계약서 ``quality`` 필드 스키마와 일치한다.

    계약서: ``{"score","blur","brightness","plate_px_width","readable","reason"}``
    """
    payload = assess(make_checkerboard(), plate_px_width=256).to_dict()
    assert set(payload.keys()) == {
        "score",
        "blur",
        "brightness",
        "plate_px_width",
        "readable",
        "reason",
    }
    assert isinstance(payload["score"], float)
    assert isinstance(payload["plate_px_width"], int)
    assert isinstance(payload["readable"], bool)


def test_quality_report_detail_breakdown():
    """하위 점수 내역이 채워져 디버깅이 가능하다."""
    report = assess(make_checkerboard(80, 320), plate_px_width=320)
    for key in ("blur_score", "brightness_score", "width_score", "contrast_score"):
        assert key in report.detail
        assert 0.0 <= report.detail[key] <= 1.0


def test_default_thresholds_are_sane():
    """기본 임계값이 물리적으로 타당한 범위에 있다."""
    assert 10.0 <= DEFAULT_BLUR_MIN <= 500.0
    # 7자 번호판에서 문자당 최소 8px → 약 56px 이상이어야 한다.
    assert 56 <= DEFAULT_PLATE_PX_MIN <= 200


def test_quality_report_dataclass_defaults_isolated():
    """``detail`` 기본값이 인스턴스 간 공유되지 않는다(가변 기본값 버그 방지)."""
    first = QualityReport(
        score=0.5, blur=1.0, brightness=1.0, contrast=1.0, plate_px_width=1, readable=True
    )
    second = QualityReport(
        score=0.5, blur=1.0, brightness=1.0, contrast=1.0, plate_px_width=1, readable=True
    )
    first.detail["x"] = 1.0
    assert second.detail == {}
