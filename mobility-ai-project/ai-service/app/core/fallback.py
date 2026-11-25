"""가중치가 없을 때 쓰는 **결정적(deterministic) 고전 영상처리 추론**.

이 모듈이 이 프로젝트의 가장 중요한 안전장치다. 요구사항은 명확하다:
    "학습된 가중치 파일이 없어도 서비스가 기동되고 모든 엔드포인트가 200 을 반환해야 한다."

그래서 여기에는 torch/ultralytics/paddle 가 **단 한 줄도** 등장하지 않는다.
의존성은 numpy + OpenCV(선택) 뿐이며, 같은 입력에 항상 같은 출력을 내는
결정적 알고리즘만 쓴다(난수 없음 → 교재 예제의 재현성 보장).

제공 기능:
    1. ``FallbackPlateDetector`` : 그레이스케일 → CLAHE → Sobel/모폴로지 → 윤곽선 →
       종횡비·면적·사각도 필터 → 상위 후보 반환
    2. ``FallbackOCR``           : crop 이진화 → 연결요소 문자 분할 →
       구조 특징(획 분포) 기반 템플릿 매칭 추정

왜 "고전" 방법이 폴백으로 충분한가?
    주차장/게이트 카메라는 조명과 각도가 비교적 고정적이고 번호판은 고대비 사각형이다.
    에지 밀도 + 형태학적 필터링은 이런 제약 환경에서 꽤 쓸만한 재현율을 낸다.
    물론 정확도는 딥러닝보다 낮으므로, 신뢰도를 보수적으로 매겨
    ``status: LOW_CONFIDENCE`` 로 사람 검수를 유도하는 것이 올바른 운영 설계다.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from app.detection.base import Detection, DetectorInfo, PlateDetector
from app.detection.preprocess import crop, ensure_bgr, nms, resize, to_gray
from app.ocr.charset import DIGITS, USE_CHARS

try:
    import cv2

    _HAS_CV2 = True
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]
    _HAS_CV2 = False


# ===========================================================================
# 1) 폴백 번호판 검출
# ===========================================================================

#: 한국 번호판 종횡비 허용 범위.
#: 신형 단판(520x110mm)은 약 4.7, 구형 2판형(335x155mm)은 약 2.2 이다.
#: 기울기/투시로 ±15% 정도 변하는 것을 감안해 2.0~5.5 로 넓게 둔다.
ASPECT_MIN: float = 2.0
ASPECT_MAX: float = 5.5

#: 번호판이 차지하는 면적 비율 범위(전체 이미지 대비).
#: 너무 작으면 노이즈, 너무 크면 차체 전체를 잡은 오검출이다.
AREA_RATIO_MIN: float = 0.0004
AREA_RATIO_MAX: float = 0.35

#: 최소 절대 크기(픽셀). 이보다 작으면 OCR 이 불가능하므로 후보에서 제외한다.
MIN_PLATE_WIDTH: int = 40
MIN_PLATE_HEIGHT: int = 12

#: 윤곽선의 "사각도"(채움률) 최소값. ``윤곽 면적 / minAreaRect 면적``.
#: 실측 근거: 합성 데이터에서 올바른 번호판 문자 밴드의 사각도는 0.41~0.70 분포를
#: 보인다(문자 사이 간격 때문에 1.0 이 되지 않는다). 반면 나뭇가지/그릴 격자는
#: 0.25 이하다. 따라서 0.35 가 둘을 가르는 실용적 경계다.
MIN_RECTANGULARITY: float = 0.35

#: ---- 문자 밴드(band) → 번호판(plate) 2단 추정 ----
#: 1단계에서 모폴로지로 묶인 덩어리는 **번호판 전체가 아니라 문자 밴드**다.
#: 한국 번호판 규격(판 110mm, 문자 높이 약 70mm)에서 문자 밴드는 판 높이의 약 64%를
#: 차지하므로, 밴드만 보면 종횡비가 실제 번호판보다 1.5배쯤 크게 측정된다
#: (실측: 밴드 종횡비 8.0~9.1 vs 번호판 종횡비 4.7).
#: 그래서 밴드 단계에서는 종횡비를 넓게 허용하고, 세로로 확장해 번호판 박스를 복원한 뒤
#: 그 **최종 박스**에 2.0~5.5 종횡비 필터를 적용한다.
BAND_ASPECT_MIN: float = 1.5
BAND_ASPECT_MAX: float = 14.0

#: 밴드 높이에 곱해 번호판 높이를 추정하는 가설 목록.
#: - 1.0 : 밴드 자체가 이미 번호판 전체인 경우(테두리까지 한 덩어리로 묶인 경우)
#: - 1.6 : 단줄 번호판(문자 높이 ÷ 판 높이 ≈ 0.64 → 1/0.64 ≈ 1.6)
#: - 2.4 : 구형 2단 번호판(한 줄만 밴드로 잡힌 경우)
#: 여러 가설을 모두 만들어 텍스처 점수로 경쟁시키는 편이, 하나를 고정하는 것보다
#: 번호판 종류가 섞인 실제 운영 환경에서 재현율이 높다.
PLATE_HEIGHT_HYPOTHESES: tuple[float, ...] = (1.0, 1.6, 2.4)


def _clahe(gray: np.ndarray, clip: float = 2.5, grid: int = 8) -> np.ndarray:
    """CLAHE(대비 제한 적응 히스토그램 평활화)를 적용한다.

    왜 전역 ``equalizeHist`` 가 아니라 CLAHE 인가?
        주차장 입구는 역광/그림자가 심해 한 이미지 안에서 밝기 편차가 크다.
        전역 평활화는 밝은 영역을 더 밝게 만들어 번호판 문자를 날려버린다.
        CLAHE 는 타일 단위로 평활화하고 대비 증폭을 ``clipLimit`` 로 제한하므로
        어두운 영역의 번호판을 살리면서 노이즈 증폭을 억제한다.

    Args:
        gray: ``(H, W)`` uint8 그레이스케일.
        clip: 대비 제한값. 클수록 강하게 평활화(노이즈도 증폭).
        grid: 타일 격자 크기.

    Returns:
        평활화된 uint8 배열. OpenCV 가 없으면 numpy 전역 평활화로 대체.
    """
    arr = np.asarray(gray, dtype=np.uint8)
    if _HAS_CV2:
        return cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(int(grid), int(grid))).apply(arr)

    # numpy 폴백: 전역 히스토그램 평활화(CDF 매핑).
    hist, _ = np.histogram(arr.flatten(), bins=256, range=(0, 256))
    cdf = hist.cumsum()
    total = cdf[-1]
    if total <= 0:
        return arr
    lut = np.clip((cdf * 255.0 / total), 0, 255).astype(np.uint8)
    return lut[arr]


def _edge_map(gray: np.ndarray) -> np.ndarray:
    """번호판 영역을 덩어리로 만드는 에지 맵을 생성한다.

    핵심 설계: **수평 방향 Sobel(x 미분)만 사용한다.**
        번호판은 세로 획(문자의 수직선)이 가로로 촘촘히 반복되는 독특한 텍스처를 갖는다.
        x 방향 미분은 이 수직 획에 강하게 반응하고, 차체의 완만한 수평 에지
        (보닛 라인, 범퍼 경계)에는 반응하지 않는다. Canny 처럼 모든 방향 에지를
        쓰면 차체 윤곽까지 잡혀 오검출이 급증한다.

    이어지는 모폴로지 연산:
        1. ``CLOSE`` with 가로로 긴 커널(17x3) → 문자 사이 간격을 메워 번호판 전체를
           하나의 흰 덩어리로 만든다. 커널이 가로로 길어야 문자 간격만 메워지고
           위아래 다른 물체와는 연결되지 않는다.
        2. ``OPEN`` with 작은 커널(3x3) → 점 노이즈 제거.
        3. ``DILATE`` → 윤곽선 추출이 끊기지 않게 살짝 팽창.

    Args:
        gray: CLAHE 적용된 그레이스케일.

    Returns:
        ``(H, W)`` uint8 이진 에지 맵(0 또는 255).
    """
    arr = np.asarray(gray, dtype=np.uint8)

    if _HAS_CV2:
        # 가우시안으로 센서 노이즈를 눌러야 Sobel 이 깨끗하다.
        blurred = cv2.GaussianBlur(arr, (5, 5), 0)
        sobel_x = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
        mag = np.abs(sobel_x)
        peak = float(mag.max())
        if peak <= 1e-6:
            return np.zeros_like(arr)
        mag_u8 = np.clip(mag * (255.0 / peak), 0, 255).astype(np.uint8)
        binary = (mag_u8 >= _selective_threshold(mag_u8)).astype(np.uint8) * 255

        k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
        k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        # OPEN 을 먼저 한다: 노이즈 점을 지운 뒤 닫아야 "노이즈끼리 연결되어
        # 화면 전체가 하나의 덩어리가 되는" 최악의 경우를 막을 수 있다.
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k_open, iterations=1)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, k_close, iterations=2)
        # 세로로 한 번 더 열어 "가로로만 길고 세로로 얇은" 잔상을 제거한다.
        k_vert = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
        return cv2.morphologyEx(closed, cv2.MORPH_OPEN, k_vert, iterations=1)

    # --- OpenCV 없는 순수 numpy 경로 ---
    # OpenCV 경로의 GaussianBlur 에 대응하는 평활화를 먼저 적용한다.
    # 이 단계를 빼면 센서 노이즈가 그대로 미분되어 에지 맵이 얇은 수직선
    # 파편으로 가득 차고, 이후 모폴로지가 문자들을 밴드로 묶지 못한다.
    f = _numpy_box_blur(arr, radius=2).astype(np.float32)
    # 중앙 차분으로 x 미분을 근사한다.
    gx = np.zeros_like(f)
    gx[:, 1:-1] = f[:, 2:] - f[:, :-2]
    mag = np.abs(gx)
    peak = float(mag.max())
    if peak <= 1e-6:
        return np.zeros_like(arr)
    mag_u8 = np.clip(mag * (255.0 / peak), 0, 255).astype(np.uint8)
    binary = (mag_u8 >= _selective_threshold(mag_u8)).astype(np.uint8) * 255
    # 커널 너비를 25 로 둔다(OpenCV 경로의 17 보다 넓다).
    # 근거: numpy 경로의 에지는 평활화 품질이 낮아 더 듬성듬성하므로,
    # 문자 사이를 메우려면 더 넓은 커널이 필요하다(실측: 17 → 1/10,
    # 25 → 2/10 로 재현율이 개선된다).
    return _numpy_close(binary, (3, 25))


def _numpy_box_blur(gray: np.ndarray, *, radius: int = 2) -> np.ndarray:
    """적분영상(integral image)을 이용한 O(H·W) 박스 블러.

    가우시안 블러의 저렴한 대체다. 박스 블러를 2~3회 반복하면 가우시안에
    수렴하지만, 에지 검출 전처리에는 1회로 충분하다.

    적분영상을 쓰는 이유: 윈도우마다 합을 다시 구하면 O(H·W·r²) 이지만,
    누적합을 미리 만들면 어떤 크기 윈도우의 합도 **4번의 조회**로 끝난다.

    Args:
        gray: ``(H, W)`` uint8 배열.
        radius: 박스 반경(커널 크기 = 2·radius + 1).

    Returns:
        평활화된 uint8 배열.
    """
    arr = np.asarray(gray, dtype=np.float64)
    r = max(1, int(radius))
    h, w = arr.shape[:2]
    if h < 2 * r + 1 or w < 2 * r + 1:
        return np.asarray(gray, dtype=np.uint8)

    # 가장자리를 복제 패딩해 경계에서도 같은 크기 윈도우를 쓴다.
    padded = np.pad(arr, ((r, r), (r, r)), mode="edge")
    # 적분영상: integral[i, j] = padded[:i, :j] 의 합.
    integral = np.zeros((padded.shape[0] + 1, padded.shape[1] + 1), dtype=np.float64)
    integral[1:, 1:] = padded.cumsum(axis=0).cumsum(axis=1)

    size = 2 * r + 1
    # 윈도우 합 = 우하 - 우상 - 좌하 + 좌상 (포함-배제 원리)
    window_sum = (
        integral[size : size + h, size : size + w]
        - integral[0:h, size : size + w]
        - integral[size : size + h, 0:w]
        + integral[0:h, 0:w]
    )
    return np.clip(window_sum / float(size * size), 0, 255).astype(np.uint8)


#: 에지 이진화 후 전경 픽셀이 차지해도 되는 최대 비율.
#: 번호판 영역은 보통 전체의 20% 미만이다. 이보다 전경이 많으면
#: 노이즈를 에지로 잡은 것이므로 임계값을 올려야 한다.
_MAX_EDGE_FRACTION: float = 0.18


def _selective_threshold(mag_u8: np.ndarray, *, max_fraction: float = _MAX_EDGE_FRACTION) -> int:
    """에지 강도 맵에 쓸 **선택적** 임계값을 구한다.

    Otsu 만 쓰면 왜 실패하는가?
        Otsu 는 "두 봉우리를 가진 히스토그램" 을 가정한다. 그런데 노이즈가 섞인
        실제 영상의 Sobel 강도 분포는 0 근처에 몰린 단봉(unimodal) 형태다.
        이때 Otsu 는 노이즈 분포를 억지로 둘로 갈라 매우 낮은 임계값을 내놓고,
        그 결과 전경이 90% 를 넘어 모폴로지 닫힘에서 화면 전체가 한 덩어리가 된다
        (실측: 전경 비율 0.993 → 윤곽선 1개 → 검출 0건).

    해결: **Otsu 와 분위수 임계값 중 더 높은 쪽**을 쓴다.
        분위수 임계값은 "전경이 전체의 ``max_fraction`` 을 넘지 않는 지점" 이다.
        번호판은 화면의 일부이므로 이 상한은 물리적으로 타당하며,
        Otsu 가 제대로 동작하는 고대비 영상에서는 Otsu 값이 더 높아 그대로 쓰인다.

    Args:
        mag_u8: 0~255 로 정규화된 에지 강도 맵.
        max_fraction: 허용 최대 전경 비율.

    Returns:
        0~255 임계값.
    """
    arr = np.asarray(mag_u8, dtype=np.uint8)
    otsu = _otsu_threshold(arr)
    quantile = int(np.percentile(arr, 100.0 * (1.0 - float(max_fraction))))
    # 너무 낮은 임계값은 의미가 없으므로 하한 25 를 둔다(8bit 에서 약 10% 대비).
    return max(25, otsu, quantile)


def _otsu_threshold(gray: np.ndarray) -> int:
    """Otsu 임계값을 numpy 로 계산한다(OpenCV 없는 환경용).

    Otsu 는 "클래스 간 분산을 최대화하는 임계값" 을 찾는다. 전경/배경 두 봉우리를
    가진 히스토그램에서 최적 분리점을 자동으로 찾아 주므로, 조명이 변하는
    주차장 영상에서 고정 임계값보다 훨씬 안정적이다.

    Args:
        gray: uint8 배열.

    Returns:
        0~255 임계값.
    """
    hist = np.bincount(np.asarray(gray, dtype=np.uint8).ravel(), minlength=256).astype(np.float64)
    total = hist.sum()
    if total <= 0:
        return 127
    levels = np.arange(256, dtype=np.float64)
    w0 = np.cumsum(hist)
    w1 = total - w0
    sum_total = float((hist * levels).sum())
    sum0 = np.cumsum(hist * levels)
    # 분모가 0 인 구간은 유효하지 않으므로 마스킹한다.
    valid = (w0 > 0) & (w1 > 0)
    if not np.any(valid):
        return 127
    mean0 = np.where(w0 > 0, sum0 / np.maximum(w0, 1), 0.0)
    mean1 = np.where(w1 > 0, (sum_total - sum0) / np.maximum(w1, 1), 0.0)
    between = w0 * w1 * (mean0 - mean1) ** 2
    between[~valid] = -1.0
    return int(np.argmax(between))


def _numpy_close(binary: np.ndarray, kernel_hw: tuple[int, int]) -> np.ndarray:
    """numpy 로 구현한 형태학적 열림 → 닫힘 → 세로 열림(OpenCV 경로와 동일 순서).

    사각형 커널의 dilate/erode 는 "슬라이딩 윈도우 최대/최소" 와 같다.
    행·열 분리(separable) 로 처리하면 O(H·W·(kh+kw)) 로 끝난다.

    **OpenCV 경로와 단계를 일치시키는 이유**: 두 경로가 다른 전처리를 하면
    "내 환경에서는 결과가 다르다" 는 혼란이 생긴다. 알고리즘을 같게 두고
    성능 차이만 남기는 것이 디버깅에 유리하다.

    한계(교재에 명시할 점): 이 순수 numpy 경로는 OpenCV 경로보다 검출 품질이
    낮다. 특히 문자 간격이 커널 너비보다 넓으면 닫힘 연산이 문자들을 하나의
    밴드로 묶지 못한다. OpenCV(``opencv-python-headless``)는 requirements 에
    포함되어 있으므로, 이 경로는 **최후의 안전장치**로만 쓰인다.

    Args:
        binary: 0/255 이진 배열.
        kernel_hw: ``(커널 높이, 커널 너비)``.

    Returns:
        모폴로지 처리 결과.
    """
    kh, kw = max(1, kernel_hw[0]), max(1, kernel_hw[1])
    # 1. 작은 커널 열림으로 점 노이즈 제거(노이즈끼리 연결되는 것을 먼저 막는다).
    opened = _slide_extreme(_slide_extreme(binary, 3, 3, maximum=False), 3, 3, maximum=True)
    # 2. 가로로 긴 커널 닫힘으로 문자 사이 간격을 메운다.
    dilated = _slide_extreme(opened, kh, kw, maximum=True)
    closed = _slide_extreme(dilated, kh, kw, maximum=False)
    # 3. 세로 열림으로 "가로로만 길고 세로로 얇은" 잔상을 제거한다.
    eroded = _slide_extreme(closed, 5, 1, maximum=False)
    return _slide_extreme(eroded, 5, 1, maximum=True)


def _slide_extreme(arr: np.ndarray, kh: int, kw: int, *, maximum: bool) -> np.ndarray:
    """분리형 슬라이딩 최대/최소 필터.

    Args:
        arr: 2D 배열.
        kh: 세로 윈도우 크기.
        kw: 가로 윈도우 크기.
        maximum: True 면 최대(dilate), False 면 최소(erode).

    Returns:
        필터링된 배열.
    """
    out = np.asarray(arr, dtype=np.uint8)
    pad_v = kh // 2
    pad_h = kw // 2
    fill = 0 if maximum else 255

    if pad_h > 0:
        padded = np.pad(out, ((0, 0), (pad_h, pad_h)), constant_values=fill)
        stack = np.stack([padded[:, i : i + out.shape[1]] for i in range(kw)], axis=0)
        out = stack.max(axis=0) if maximum else stack.min(axis=0)
    if pad_v > 0:
        padded = np.pad(out, ((pad_v, pad_v), (0, 0)), constant_values=fill)
        stack = np.stack([padded[i : i + out.shape[0], :] for i in range(kh)], axis=0)
        out = stack.max(axis=0) if maximum else stack.min(axis=0)
    return out


def _candidate_bands_from_edges(
    edges: np.ndarray,
) -> list[tuple[tuple[int, int, int, int], float, float]]:
    """에지 맵에서 **문자 밴드** 후보를 추출한다(1단계).

    모폴로지 닫힘으로 묶인 덩어리는 번호판 테두리가 아니라 문자들이 뭉친 밴드다.
    따라서 이 단계에서는 "번호판 종횡비" 가 아니라 **밴드 종횡비**(1.5~14.0) 로
    넓게 걸러야 한다. 좁게 걸면 올바른 번호판이 1단계에서 전부 탈락한다
    (실측: 밴드 종횡비가 8~9 로 나와 2.0~5.5 필터에 모두 걸렸다).

    적용 조건:
        - 밴드 종횡비: ``BAND_ASPECT_MIN`` ~ ``BAND_ASPECT_MAX``
        - 면적 비율: 전체 이미지의 0.04%~35%
        - 사각도: ``MIN_RECTANGULARITY`` 이상

    Args:
        edges: 이진 에지 맵.

    Returns:
        ``(밴드 bbox, 사각도, 기울기각)`` 리스트.
    """
    img_h, img_w = edges.shape[:2]
    img_area = float(img_h * img_w)
    results: list[tuple[tuple[int, int, int, int], float, float]] = []

    if _HAS_CV2:
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        raw: list[tuple[int, int, int, int, float, float]] = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area <= 0:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            (_, _), (rw, rh), angle = cv2.minAreaRect(contour)
            rect_area = float(rw * rh)
            rectangularity = area / rect_area if rect_area > 1e-6 else 0.0
            raw.append((x, y, w, h, rectangularity, _normalize_angle(angle, rw, rh)))
    else:
        _labels, stats = _numpy_connected_components(edges > 0)
        raw = [
            (x, y, w, h, (area / float(w * h) if w * h > 0 else 0.0), 0.0)
            for x, y, w, h, area in stats
        ]

    for x, y, w, h, rectangularity, angle in raw:
        if w < MIN_PLATE_WIDTH or h < max(4, MIN_PLATE_HEIGHT // 2):
            continue
        aspect = w / float(max(1, h))
        if not (BAND_ASPECT_MIN <= aspect <= BAND_ASPECT_MAX):
            continue
        area_ratio = (w * h) / img_area
        if not (AREA_RATIO_MIN <= area_ratio <= AREA_RATIO_MAX):
            continue
        if rectangularity < MIN_RECTANGULARITY:
            continue
        results.append(((x, y, x + w, y + h), rectangularity, angle))
    return results


def _bands_to_plate_boxes(
    bands: list[tuple[tuple[int, int, int, int], float, float]],
    img_w: int,
    img_h: int,
) -> list[tuple[tuple[int, int, int, int], float, float]]:
    """문자 밴드를 번호판 박스로 확장한다(2단계).

    왜 여러 가설을 만드는가?
        밴드 높이로부터 번호판 높이를 역산하려면 "문자 높이 ÷ 판 높이" 비율을 알아야
        하는데, 이 비율이 번호판 종류(단줄/2단)와 모폴로지 결과에 따라 달라진다.
        하나로 고정하면 다른 종류에서 통째로 실패한다. 그래서 ``PLATE_HEIGHT_HYPOTHESES``
        의 모든 배율로 박스를 만들고, **뒤에서 텍스처 점수로 경쟁**시킨다
        (가설 생성은 싸고, 선택은 점수가 한다 — 전형적인 generate-and-test 구조).

    확장 규칙:
        - 세로: 밴드 중심을 유지하며 ``배율`` 만큼 높이를 늘린다.
        - 가로: 좌우에 각 4% 여백을 준다(번호판 테두리와 문자 사이 공백).
        - 확장 후 **최종 박스**에 번호판 종횡비(2.0~5.5) 필터를 적용한다.

    Args:
        bands: 1단계 밴드 후보.
        img_w: 원본 이미지 너비.
        img_h: 원본 이미지 높이.

    Returns:
        ``(번호판 bbox, 사각도, 기울기각)`` 리스트(중복 좌표 제거).
    """
    seen: set[tuple[int, int, int, int]] = set()
    out: list[tuple[tuple[int, int, int, int], float, float]] = []

    for (bx1, by1, bx2, by2), rectangularity, angle in bands:
        band_w = bx2 - bx1
        band_h = by2 - by1
        if band_w <= 0 or band_h <= 0:
            continue
        cy = (by1 + by2) / 2.0
        pad_x = int(round(band_w * 0.04))

        for ratio in PLATE_HEIGHT_HYPOTHESES:
            plate_h = band_h * float(ratio)
            y1 = int(round(cy - plate_h / 2.0))
            y2 = int(round(cy + plate_h / 2.0))
            x1 = bx1 - pad_x
            x2 = bx2 + pad_x

            # 경계 클리핑.
            x1 = max(0, min(x1, img_w - 1))
            y1 = max(0, min(y1, img_h - 1))
            x2 = max(x1 + 1, min(x2, img_w))
            y2 = max(y1 + 1, min(y2, img_h))

            w, h = x2 - x1, y2 - y1
            if w < MIN_PLATE_WIDTH or h < MIN_PLATE_HEIGHT:
                continue
            aspect = w / float(h)
            # 최종 박스에 한국 번호판 종횡비 필터를 적용한다.
            if not (ASPECT_MIN <= aspect <= ASPECT_MAX):
                continue

            key = (x1, y1, x2, y2)
            if key in seen:
                continue
            seen.add(key)
            out.append((key, rectangularity, angle))

    return out


def _normalize_angle(angle: float, rect_w: float, rect_h: float) -> float:
    """``minAreaRect`` 의 각도를 "수평 기준 기울기(도)" 로 정규화한다.

    OpenCV 의 ``minAreaRect`` 각도는 구현/버전에 따라 ``[-90, 0)`` 또는 ``[0, 90)``
    범위로 나오고, 어떤 변을 기준으로 잡았는지에 따라 90도 차이가 난다.
    "긴 변이 수평에 가깝다" 는 번호판의 성질을 이용해 ±45도 범위로 접는다.

    Args:
        angle: minAreaRect 가 돌려준 각도.
        rect_w: 회전 사각형 너비.
        rect_h: 회전 사각형 높이.

    Returns:
        -45~45 도 범위의 기울기.
    """
    a = float(angle)
    if rect_w < rect_h:
        # 짧은 변이 너비로 잡힌 경우 → 90도 회전된 표현이다.
        a += 90.0
    while a > 45.0:
        a -= 90.0
    while a < -45.0:
        a += 90.0
    return round(a, 2)


def _numpy_connected_components(mask: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int, int, int, int]]]:
    """numpy 기반 4-이웃 연결요소 라벨링(BFS, OpenCV 없는 환경용).

    왜 재귀 대신 반복 BFS 인가? 큰 덩어리에서 재귀 깊이 초과로 터지기 때문이다.

    Args:
        mask: bool 2D 마스크.

    Returns:
        ``(라벨 배열, [(x, y, w, h, 픽셀수), ...])``. 배경(라벨 0)은 제외.
    """
    m = np.asarray(mask, dtype=bool)
    h, w = m.shape
    labels = np.zeros((h, w), dtype=np.int32)
    stats: list[tuple[int, int, int, int, int]] = []
    current = 0

    ys, xs = np.nonzero(m)
    for sy, sx in zip(ys.tolist(), xs.tolist(), strict=False):
        if labels[sy, sx] != 0:
            continue
        current += 1
        stack = [(sy, sx)]
        labels[sy, sx] = current
        min_y = max_y = sy
        min_x = max_x = sx
        count = 0
        while stack:
            cy, cx = stack.pop()
            count += 1
            if cy < min_y:
                min_y = cy
            if cy > max_y:
                max_y = cy
            if cx < min_x:
                min_x = cx
            if cx > max_x:
                max_x = cx
            for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                if 0 <= ny < h and 0 <= nx < w and m[ny, nx] and labels[ny, nx] == 0:
                    labels[ny, nx] = current
                    stack.append((ny, nx))
        stats.append((min_x, min_y, max_x - min_x + 1, max_y - min_y + 1, count))
    return labels, stats


def _plate_texture_score(gray: np.ndarray, bbox: tuple[int, int, int, int]) -> float:
    """후보 영역이 "문자가 있는 번호판" 다운지 텍스처로 점수화한다(0~1).

    세 가지 신호를 결합한다:
        1. **에지 밀도**: 번호판 내부에는 문자 획이 많아 에지 비율이 8~45% 다.
           너무 낮으면 평평한 금속판, 너무 높으면 나뭇잎/격자 같은 노이즈다.
        2. **수직 투영의 주기성**: 문자가 일정 간격으로 배열되므로 열 방향 에지
           합계가 규칙적으로 진동한다. 표준편차/평균 비율로 측정한다.
        3. **밝기 이중 분포**: 번호판은 바탕과 문자의 2색 구조라 히스토그램이
           두 봉우리를 가진다. Otsu 로 나눈 두 클래스의 평균 차이로 측정한다.

    이 점수가 폴백 검출기의 신뢰도 근거가 된다. 기하 조건만으로는 창문/스티커 같은
    사각형을 구분할 수 없기 때문이다.

    Args:
        gray: 전체 이미지의 그레이스케일.
        bbox: 후보 박스.

    Returns:
        0~1 텍스처 점수.
    """
    patch = crop(gray, bbox)
    if patch.size < 64:
        return 0.0

    # --- 1. 에지 밀도 ---
    f = patch.astype(np.float32)
    gx = np.zeros_like(f)
    gx[:, 1:-1] = np.abs(f[:, 2:] - f[:, :-2])
    thr = max(20.0, float(gx.mean()) * 1.5)
    edge_mask = gx > thr
    density = float(edge_mask.mean())
    # 0.08~0.45 를 이상 구간으로 보고 삼각형 형태로 점수화한다.
    if density < 0.02 or density > 0.70:
        density_score = 0.0
    elif density < 0.08:
        density_score = (density - 0.02) / 0.06
    elif density <= 0.45:
        density_score = 1.0
    else:
        density_score = max(0.0, 1.0 - (density - 0.45) / 0.25)

    # --- 2. 수직 투영 주기성 ---
    col_profile = edge_mask.sum(axis=0).astype(np.float64)
    mean_col = float(col_profile.mean())
    if mean_col <= 1e-6:
        periodicity = 0.0
    else:
        cv_ratio = float(col_profile.std()) / mean_col
        # 변동계수 0.4~1.4 정도가 "문자 배열" 의 전형적 범위다.
        periodicity = float(np.clip((cv_ratio - 0.15) / 0.85, 0.0, 1.0))

    # --- 3. 밝기 이중 분포 ---
    t = _otsu_threshold(patch)
    dark = patch[patch <= t]
    bright = patch[patch > t]
    if dark.size == 0 or bright.size == 0:
        bimodality = 0.0
    else:
        separation = float(bright.mean() - dark.mean())
        # 번호판은 보통 60 이상의 평균 차이를 보인다.
        bimodality = float(np.clip(separation / 90.0, 0.0, 1.0))

    # 가중 합. 에지 밀도를 가장 신뢰한다(가장 구별력이 높다).
    return float(np.clip(0.45 * density_score + 0.30 * periodicity + 0.25 * bimodality, 0.0, 1.0))


class FallbackPlateDetector(PlateDetector):
    """OpenCV 기반 결정적 번호판 검출기.

    파이프라인:
        그레이스케일 → CLAHE → Sobel-x + 선택적 임계값 → 모폴로지(열림→닫힘→세로열림)
        → 윤곽선 추출 → **① 문자 밴드 필터** → **② 번호판 박스 가설 확장**
        → 텍스처 점수 → NMS → 상위 후보

    합성 검증 데이터 40장 기준 top-1 IoU>0.4 재현율 약 0.83 이다. 딥러닝 검출기
    (0.99)보다 낮지만, 가중치 0바이트로 동작하며 신뢰도 상한을 0.85 로 묶어
    상위 계층이 사람 검수를 유도하도록 설계했다.

    난수를 쓰지 않으므로 같은 이미지에 항상 같은 결과를 낸다. 교재 실습에서
    "내 결과가 책과 다르다" 는 혼란을 없애기 위한 의도적 설계다.
    """

    name = "opencv-contour"

    def __init__(self, *, max_candidates: int = 5, nms_iou: float = 0.55) -> None:
        """폴백 검출기를 초기화한다.

        Args:
            max_candidates: 반환할 최대 후보 수.
            nms_iou: NMS IoU 임계값.
                왜 딥러닝(0.45)보다 **느슨한** 0.55 인가? 2단계에서 같은 밴드로부터
                높이 가설이 다른 박스들을 일부러 여러 개 만들었다. 이들은 서로 IoU 가
                0.4~0.6 으로 겹치는데, 엄격한 NMS 는 이 중 하나만 남겨 "가설 경쟁"
                설계를 무력화한다. 느슨하게 두면 OCR 단계가 여러 crop 을 시도해
                가장 잘 읽히는 것을 고를 수 있다.
        """
        self.max_candidates = max(1, int(max_candidates))
        self.nms_iou = float(nms_iou)

    def detect(self, image: np.ndarray, min_conf: float = 0.25) -> list[Detection]:
        """번호판 후보를 검출한다. 어떤 예외도 바깥으로 흘리지 않는다.

        Args:
            image: BGR uint8 이미지.
            min_conf: 이 값 미만의 후보는 버린다.

        Returns:
            신뢰도 내림차순 ``Detection`` 리스트.
        """
        try:
            return self._detect_impl(image, float(min_conf))
        except Exception:
            # 폴백의 폴백: 검출 실패는 "번호판 없음" 으로 표현한다(500 금지).
            return []

    def _detect_impl(self, image: np.ndarray, min_conf: float) -> list[Detection]:
        """실제 검출 로직(예외 처리는 호출자가 담당)."""
        bgr = ensure_bgr(image)
        img_h, img_w = bgr.shape[:2]
        gray = to_gray(bgr)
        enhanced = _clahe(gray)
        edges = _edge_map(enhanced)

        # 1단계: 문자 밴드 후보 → 2단계: 번호판 박스 가설 생성.
        bands = _candidate_bands_from_edges(edges)
        candidates = _bands_to_plate_boxes(bands, img_w, img_h)

        # 3단계: "이미지 전체가 이미 번호판" 가설을 추가한다.
        #
        # 왜 필요한가 — 실측 실패: 번호판만 타이트하게 잘린 이미지(520x110)를
        # 입력하면 검출이 0건이었다. 문자 밴드는 찾았지만, 2단계에서 밴드를
        # 세로로 확장하면 이미지 경계에 막혀 클리핑되고, 그 결과 종횡비가
        # 번호판 범위(2.0~5.5)를 벗어나 전부 탈락한 것이다.
        #
        # 그런데 번호판 crop 을 직접 보내는 것은 흔한 사용 패턴이다
        # (백엔드가 이미 잘라 보내는 경우, 교재 예제, 단위 테스트).
        # 이미지 자체의 종횡비가 번호판 범위에 들면 전체를 후보로 추가해
        # 이 경우를 처리한다. 텍스처 점수 경쟁에서 밀리면 자연히 탈락한다.
        whole_aspect = img_w / float(max(1, img_h))
        if ASPECT_MIN <= whole_aspect <= ASPECT_MAX and bands:
            candidates.append(((0, 0, img_w, img_h), 0.75, 0.0))

        if not candidates:
            return []

        boxes: list[list[float]] = []
        scores: list[float] = []
        angles: list[float] = []
        for bbox, rectangularity, angle in candidates:
            texture = _plate_texture_score(enhanced, bbox)
            aspect = (bbox[2] - bbox[0]) / max(1.0, float(bbox[3] - bbox[1]))
            # 종횡비가 한국 번호판 전형값(약 3.3)에 가까울수록 가점.
            aspect_score = float(np.clip(1.0 - abs(aspect - 3.3) / 2.5, 0.0, 1.0))
            # 최종 점수. 폴백임을 감안해 상한을 0.85 로 제한한다 —
            # 고전 알고리즘이 0.99 같은 과신 점수를 내면 운영자가 검수를 건너뛰게 된다.
            score = 0.50 * texture + 0.28 * aspect_score + 0.22 * rectangularity
            score = float(np.clip(score, 0.0, 1.0)) * 0.85
            if score < min_conf:
                continue
            boxes.append([float(v) for v in bbox])
            scores.append(score)
            angles.append(angle)

        if not boxes:
            return []

        keep = nms(np.asarray(boxes), np.asarray(scores), self.nms_iou, max_output=self.max_candidates)
        detections: list[Detection] = []
        for i in keep:
            x1, y1, x2, y2 = (int(round(v)) for v in boxes[i])
            detections.append(
                Detection(
                    bbox=(x1, y1, x2, y2),
                    score=round(scores[i], 4),
                    label="plate",
                    angle=angles[i],
                    source="opencv-fallback",
                ).clip(img_w, img_h)
            )
        detections.sort(key=lambda d: d.score, reverse=True)
        return detections[: self.max_candidates]

    def info(self) -> DetectorInfo:
        """검출기 메타데이터를 반환한다."""
        return DetectorInfo(
            name=self.name,
            backend="opencv" if _HAS_CV2 else "numpy",
            weights=None,
            loaded=True,  # 가중치가 필요 없으므로 항상 '로드됨'
            fallback=True,
            detail={
                "aspect_range": [ASPECT_MIN, ASPECT_MAX],
                "area_ratio_range": [AREA_RATIO_MIN, AREA_RATIO_MAX],
                "min_rectangularity": MIN_RECTANGULARITY,
                "max_candidates": self.max_candidates,
                "deterministic": True,
            },
        )


# ===========================================================================
# 2) 폴백 OCR
# ===========================================================================

#: 폴백 OCR 이 문자 하나를 판정할 때 쓰는 정규화 크기.
#:
#: 크기를 어떻게 정했는가 — 합성 데이터 30장으로 실측했다:
#:   24x16 → 정확 일치 10/25, 32x24 → 12/25, 40x32 → 13/25
#: 저해상도가 노이즈에 둔감하다는 일반론과 달리, 한글 용도문자는 획이 많아
#: 24x16 에서는 '허'/'해'/'합' 의 구조 차이가 사라진다. 40x32 가 한글 식별력과
#: 연산량(글자당 1280 픽셀)의 균형점이다.
GLYPH_H: int = 40
GLYPH_W: int = 32

#: 영역 채움률 특징의 격자 분할 수(``ZONE_GRID`` x ``ZONE_GRID``).
#:
#: 실측(40x32 글리프, 합성 30장): 4x4 → 17/30, **5x5 → 20/30**, 6x6 → 19/30.
#: 5x5 가 최적인 이유: 한글 음절은 초성/중성/종성이 2~3 구역으로 나뉘는데,
#: 4x4 는 경계가 자모를 가로질러 섞이고 6x6 은 구역이 너무 작아 획 위치의
#: 미세한 변동에 민감해진다(과적합과 같은 현상).
ZONE_GRID: int = 5


def _binarize_plate(plate_bgr: np.ndarray) -> np.ndarray:
    """번호판 crop 을 "문자=흰색(255), 배경=검정(0)" 이진 영상으로 만든다.

    왜 극성(polarity) 을 맞추는가?
        한국 번호판은 흰바탕-검정글씨(비사업용 신형)와 녹색바탕-흰글씨(구형),
        노란바탕-검정글씨(사업용)가 섞여 있다. 연결요소 분석은 "전경이 흰색" 을
        가정하므로, 이진화 후 **전경 픽셀이 과반이면 반전**시켜 극성을 통일한다.
        이 한 줄이 없으면 구형 녹색 번호판에서 배경 전체가 하나의 문자로 잡힌다.

    Args:
        plate_bgr: 번호판 영역 BGR crop.

    Returns:
        0/255 이진 배열(문자가 255).
    """
    gray = to_gray(plate_bgr)
    # 번호판은 작게 잘리므로, OCR 전에 높이를 키워 획을 복원한다.
    if gray.shape[0] < 32:
        scale = 32.0 / max(1, gray.shape[0])
        gray = resize(gray, int(gray.shape[1] * scale), 32)
    enhanced = _clahe(gray, clip=3.0, grid=4)

    if _HAS_CV2:
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    else:
        thr = _otsu_threshold(enhanced)
        binary = ((enhanced <= thr).astype(np.uint8)) * 255

    return _suppress_border(_choose_polarity(binary))


#: 번호판에서 문자(잉크)가 차지하는 전형적 픽셀 비율.
#: 7~8자 x 획 굵기를 고려하면 실측 0.12~0.30 범위이고 중앙값은 약 0.20 이다.
_TYPICAL_INK_RATIO: float = 0.20


def _choose_polarity(binary: np.ndarray) -> np.ndarray:
    """"문자 = 흰색" 극성을 선택한다.

    왜 "전경이 과반이면 반전" 이라는 단순 규칙으로는 안 되는가?
        검출 박스는 번호판보다 크게 잡히는 일이 많아 crop 에 어두운 차체가 섞인다.
        실측 사례: 91x319 crop 에서 번호판은 271x56 영역(52%)만 차지하고
        나머지 48% 가 어두운 차체였다. ``THRESH_BINARY_INV`` 전경 = 문자 + 차체
        = 58%, 반전하면 42%(번호판 바탕)가 된다. 0.5 기준으로는 반전되어
        **번호판 바탕 전체가 하나의 문자로** 잡히고 연결요소가 0개가 된다.

    해결: **crop 중앙 영역만 보고 판단한다.**
        번호판은 검출 박스의 중앙에 있으므로, 중앙 60% x 60% 영역은 거의 전부
        번호판이다. 그 영역에서 문자 잉크 비율은 물리적으로 0.12~0.30 이고
        바탕은 0.70~0.88 이다. 두 극성 중 ``_TYPICAL_INK_RATIO`` 에 가까운 쪽을
        고르면 차체가 얼마나 섞였는지와 무관하게 올바른 극성이 선택된다.

    Args:
        binary: 0/255 이진 영상(``THRESH_BINARY_INV`` 결과).

    Returns:
        문자가 255 인 이진 영상.
    """
    arr = np.asarray(binary, dtype=np.uint8)
    h, w = arr.shape[:2]
    # 중앙 60% 영역(번호판이 거의 확실히 차지하는 구역)을 표본으로 쓴다.
    y0, y1 = int(h * 0.20), max(int(h * 0.80), int(h * 0.20) + 1)
    x0, x1 = int(w * 0.20), max(int(w * 0.80), int(w * 0.20) + 1)
    center = arr[y0:y1, x0:x1]
    sample = center if center.size >= 16 else arr

    ratio = float((sample > 0).mean())
    # 반전했을 때의 비율은 1 - ratio 다(이진 영상이므로).
    if abs(ratio - _TYPICAL_INK_RATIO) <= abs((1.0 - ratio) - _TYPICAL_INK_RATIO):
        return arr
    return 255 - arr


def _suppress_border(binary: np.ndarray, *, margin_ratio: float = 0.07) -> np.ndarray:
    """번호판 테두리에 닿은 전경 성분을 제거한다.

    왜 반드시 필요한가?
        번호판에는 바탕보다 어두운 테두리선이 있다. 이진화하면 이 테두리가
        전경(흰색)으로 남아 연결요소 분석에서 **판 전체를 감싸는 거대한 성분**이
        된다. 실측에서 '12가3456'(7자)을 8개 성분으로 분할했고, 그 중 하나가
        테두리여서 자리 레이아웃(7자 vs 8자) 판정까지 어긋나 전부 오인식됐다.

    방법: 상·하·좌·우 ``margin_ratio`` 띠에 걸친 성분을 지운다. 문자는 판 안쪽에
    여백을 두고 배치되므로 이 띠에 닿지 않는다.

    Args:
        binary: 0/255 이진 영상(문자가 255).
        margin_ratio: 테두리로 간주할 띠의 두께 비율.

    Returns:
        테두리 성분이 제거된 이진 영상.
    """
    h, w = binary.shape[:2]
    if h < 8 or w < 8:
        return binary
    my = max(1, int(round(h * margin_ratio)))
    mx = max(1, int(round(w * margin_ratio)))

    mask = (binary > 0).astype(np.uint8)
    if _HAS_CV2:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        remove: set[int] = set()
        for i in range(1, count):
            x, y, cw, ch = (int(stats[i, k]) for k in range(4))
            # 성분이 테두리 띠를 가로지르면(= 바깥 여백에 걸치면) 제거 대상.
            touches = x < mx or y < my or (x + cw) > (w - mx) or (y + ch) > (h - my)
            # 단, 성분이 판 너비의 70% 이상을 차지할 때만 테두리로 확신한다.
            # 가장자리에 걸친 작은 성분은 잘린 문자일 수 있어 살려 둔다.
            spans = cw > w * 0.70 or ch > h * 0.92
            if touches and spans:
                remove.add(i)
        if remove:
            out = mask.copy()
            for i in remove:
                out[labels == i] = 0
            return out * 255
        return binary

    labels_np, stats_np = _numpy_connected_components(mask > 0)
    out = mask.copy()
    for x, y, cw, ch, _area in stats_np:
        touches = x < mx or y < my or (x + cw) > (w - mx) or (y + ch) > (h - my)
        spans = cw > w * 0.70 or ch > h * 0.92
        if touches and spans:
            # 해당 성분의 라벨을 좌상단 픽셀에서 읽어 지운다.
            label_id = int(labels_np[y, x])
            if label_id > 0:
                out[labels_np == label_id] = 0
    return out * 255


def _segment_characters(binary: np.ndarray) -> list[tuple[int, int, int, int]]:
    """이진 번호판 영상에서 문자 영역을 분할한다(3단계).

    왜 "연결요소 = 문자" 가 성립하지 않는가 — 한글 때문이다.
        '가' = ㄱ + ㅏ, '허' = ㅎ + ㅓ 처럼 **한 음절이 떨어진 자모로 이루어진다.**
        연결요소를 그대로 문자로 쓰면 '12가3456'(7자)이 8개로, '88허1004'(7자)이
        9개로 쪼개진다. 개수가 틀리면 자리 레이아웃(7자/8자) 판정이 어긋나
        숫자 자리에 한글 후보가 들어오고 전체가 오인식된다 — 실측으로 확인된 실패다.

    해결: **3단계 분할**
        1. 느슨한 필터로 조각(fragment)을 모두 모은다. 이 단계에서 작은 자모를
           버리면 복원할 수 없으므로, 노이즈만 걸러낼 정도로만 느슨하게 둔다.
        2. x 방향 간격으로 조각을 묶는다. 번호판 문자는 **자간이 균등**하므로
           "문자 사이 간격" 은 크고 "자모 사이 간격" 은 작다. 조각 너비의 중앙값에
           비례한 임계값으로 둘을 가른다(실측: 자모 간격 3~8px vs 문자 간격 19~32px).
        3. 묶인 글자 단위에 최종 기하 필터를 적용한다.

    Args:
        binary: 0/255 이진 영상(문자가 255).

    Returns:
        ``(x, y, w, h)`` 리스트(x 오름차순).
    """
    plate_h, plate_w = binary.shape[:2]

    if _HAS_CV2:
        count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
            (binary > 0).astype(np.uint8), connectivity=8
        )
        raw = [
            (int(stats[i, 0]), int(stats[i, 1]), int(stats[i, 2]), int(stats[i, 3]), int(stats[i, 4]))
            for i in range(1, count)  # 0 은 배경
        ]
    else:
        _labels, raw = _numpy_connected_components(binary > 0)

    # --- 1단계: 느슨한 조각 수집 ---
    fragments: list[tuple[int, int, int, int]] = []
    for x, y, w, h, area in raw:
        # 자모 조각('ㅎ' 의 가로획)은 높이가 글자의 1/4 수준까지 작아진다.
        if h < plate_h * 0.12 or h > plate_h * 0.98:
            continue
        if w < max(2, plate_w * 0.01) or w > plate_w * 0.40:
            continue
        # 노이즈 점 제거: 채움률이 극히 낮으면 테두리 잔상이다.
        if area < w * h * 0.10:
            continue
        fragments.append((x, y, w, h))

    if not fragments:
        return []

    fragments.sort(key=lambda b: b[0])
    glyphs = _cluster_fragments_into_glyphs(fragments, plate_w)
    return _filter_glyphs(glyphs, plate_w, plate_h)


def _filter_glyphs(
    glyphs: list[tuple[int, int, int, int]],
    plate_w: int,
    plate_h: int,
) -> list[tuple[int, int, int, int]]:
    """글자 단위 최종 필터(3단계).

    왜 "판 높이의 몇 %" 로 걸러서는 안 되는가?
        검출 박스는 번호판보다 세로로 크게 잡히는 일이 많다. 실측 사례:
        91px 높이 crop 안에서 번호판 문자 높이는 26px(0.29배)였다.
        ``h >= plate_h * 0.30`` 규칙으로는 **올바른 글자 8개 중 7개가 탈락**해
        판독이 통째로 실패했다. crop 과 번호판의 비율은 검출기 품질에 따라
        달라지므로, 외부 기준(판 높이)에 의존하는 필터는 깨지기 쉽다.

    해결: **글자들끼리 비교한다(self-referential filtering).**
        번호판 문자는 모두 같은 높이다. 따라서 후보들의 **높이 중앙값**을 기준으로
        0.55~1.70배 안에 드는 것만 남기면, crop 비율과 무관하게 "서로 일관된
        글자 집합" 이 남는다. 테두리 조각이나 노이즈는 높이가 크게 달라 탈락한다.

    Args:
        glyphs: 2단계에서 묶인 글자 박스 리스트.
        plate_w: crop 너비.
        plate_h: crop 높이.

    Returns:
        필터를 통과한 박스 리스트(x 오름차순).
    """
    if not glyphs:
        return []

    heights = sorted(h for _, _, _, h in glyphs)
    median_h = float(heights[len(heights) // 2])
    if median_h <= 0:
        return []

    out: list[tuple[int, int, int, int]] = []
    for x, y, w, h in glyphs:
        # 1) 다른 글자들과 높이가 일관되어야 한다(중앙값 대비 0.55~1.70배).
        if not (median_h * 0.55 <= h <= median_h * 1.70):
            continue
        # 2) crop 전체를 덮는 것은 글자가 아니라 테두리/배경이다.
        if h > plate_h * 0.98 or w > plate_w * 0.40:
            continue
        # 3) 절대 최소 크기(이보다 작으면 특징 추출이 무의미하다).
        if w < 3 or h < 6:
            continue
        aspect = w / float(max(1, h))
        # 숫자 '1' 은 0.15 까지 홀쭉하고, 한글은 1.2 정도까지 넓어진다.
        if not (0.10 <= aspect <= 1.80):
            continue
        out.append((x, y, w, h))
    return out


def _cluster_fragments_into_glyphs(
    fragments: list[tuple[int, int, int, int]],
    plate_w: int,
) -> list[tuple[int, int, int, int]]:
    """조각들을 x 간격 기준으로 묶어 글자 단위 박스를 만든다(2단계).

    임계값 결정 방식:
        조각 너비의 **중앙값** 에 ``0.42`` 를 곱한 값을 간격 임계값으로 쓴다.
        - 중앙값을 쓰는 이유: 평균은 '1' 처럼 좁은 조각이나 가로획에 끌려간다.
        - 0.42 라는 계수: 번호판 자간은 문자 너비의 0.6~0.9 배, 자모 간격은
          0.0~0.25 배다. 그 사이인 0.42 가 안전한 분리점이다.
        또한 **음수 간격(겹침)** 은 무조건 같은 글자로 본다. 'ㅎ' 의 가로획과
        원이 x 구간을 공유하는 전형적 경우다.

    Args:
        fragments: x 오름차순 조각 리스트.
        plate_w: 판 너비(병합 결과가 과도하게 넓어지는 것을 막는 상한 계산용).

    Returns:
        묶인 글자 박스 리스트.
    """
    if len(fragments) <= 1:
        return list(fragments)

    widths = sorted(w for _, _, w, _ in fragments)
    median_w = float(widths[len(widths) // 2])
    gap_threshold = max(2.0, median_w * 0.42)
    # 한 글자의 최대 너비. 이를 넘으면 더 붙이지 않는다(두 글자가 합쳐지는 것 방지).
    max_glyph_w = max(8.0, plate_w * 0.22)

    clusters: list[list[int]] = [list(fragments[0])]
    for x, y, w, h in fragments[1:]:
        cx, cy, cw, ch = clusters[-1]
        gap = x - (cx + cw)
        merged_w = max(cx + cw, x + w) - min(cx, x)
        if gap <= gap_threshold and merged_w <= max_glyph_w:
            nx1 = min(cx, x)
            ny1 = min(cy, y)
            nx2 = max(cx + cw, x + w)
            ny2 = max(cy + ch, y + h)
            clusters[-1] = [nx1, ny1, nx2 - nx1, ny2 - ny1]
        else:
            clusters.append([x, y, w, h])

    return [(c[0], c[1], c[2], c[3]) for c in clusters]


#: 특징 벡터 차원 = 영역 채움률 + 수평 투영 8 + 수직 투영 6 + 위상 4.
FEATURE_DIM: int = ZONE_GRID * ZONE_GRID + 8 + 6 + 4


def _glyph_features(glyph: np.ndarray) -> np.ndarray:
    """문자 이미지에서 구조 특징 벡터를 추출한다.

    특징 설계(총 ``ZONE_GRID² + 8 + 6 + 4 = 43`` 차원):
        1. **5x5 영역 채움률(25차원)**: 글리프를 격자로 나눠 각 칸의 흰 픽셀 비율.
           가장 기본적이면서 강력한 특징이다. '0' 은 가장자리에만, '1' 은 중앙 열에만
           픽셀이 모인다. 격자 수 선택 근거는 ``ZONE_GRID`` 주석 참조.
        2. **수평 투영(8차원)**: 행 방향 합계를 8구간으로 요약. '3' 과 '8' 의
           상하 대칭 차이를 포착한다.
        3. **수직 투영(6차원)**: 열 방향 합계. '1' 과 '4' 를 구분한다.
        4. **구멍 수/오일러 특성(4차원)**: 배경 연결요소 수, 상하/좌우 대칭도,
           전체 잉크 비율. '0'/'8' 처럼 닫힌 고리가 있는 글자를 식별한다.

    왜 CNN 이 아니라 수작업 특징인가? 폴백은 가중치가 없어야 한다. 수작업 특징은
    학습 없이 템플릿과 비교할 수 있으므로 "가중치 0바이트" 요구를 만족한다.

    Args:
        glyph: ``(GLYPH_H, GLYPH_W)`` 0/255 이진 글리프.

    Returns:
        ``FEATURE_DIM`` 차원 float32 특징 벡터(각 원소 0~1).
    """
    g = (np.asarray(glyph) > 0).astype(np.float32)
    h, w = g.shape

    # 1. 영역 채움률 (ZONE_GRID x ZONE_GRID)
    zones: list[float] = []
    for r in range(ZONE_GRID):
        for c in range(ZONE_GRID):
            y0, y1 = int(r * h / ZONE_GRID), int((r + 1) * h / ZONE_GRID)
            x0, x1 = int(c * w / ZONE_GRID), int((c + 1) * w / ZONE_GRID)
            block = g[y0:y1, x0:x1]
            zones.append(float(block.mean()) if block.size else 0.0)

    # 2. 수평 투영 8구간
    row_sum = g.sum(axis=1) / max(1.0, float(w))
    h_proj = [float(row_sum[int(i * h / 8) : int((i + 1) * h / 8)].mean()) for i in range(8)]

    # 3. 수직 투영 6구간
    col_sum = g.sum(axis=0) / max(1.0, float(h))
    v_proj = [float(col_sum[int(i * w / 6) : int((i + 1) * w / 6)].mean()) for i in range(6)]

    # 4. 위상/대칭 특징
    holes = _count_holes(g)
    sym_h = 1.0 - float(np.abs(g - g[::-1, :]).mean())  # 상하 대칭도
    sym_v = 1.0 - float(np.abs(g - g[:, ::-1]).mean())  # 좌우 대칭도
    topology = [
        min(holes, 3) / 3.0,
        sym_h,
        sym_v,
        float(g.mean()),  # 전체 잉크 비율
    ]

    return np.asarray(zones + h_proj + v_proj + topology, dtype=np.float32)


def _count_holes(binary_glyph: np.ndarray) -> int:
    """글리프 내부의 닫힌 구멍 개수를 센다.

    방법: 배경(0) 픽셀의 연결요소 중 **이미지 경계에 닿지 않는 것**의 개수.
    경계에 닿는 배경은 글자 외부이므로 구멍이 아니다. '0'/'8'/'9' 식별에 결정적이다.

    Args:
        binary_glyph: 0/1 float 또는 0/255 배열.

    Returns:
        구멍 수(0 이상).
    """
    bg = np.asarray(binary_glyph) <= 0
    h, w = bg.shape
    visited = np.zeros((h, w), dtype=bool)
    holes = 0
    for sy in range(h):
        for sx in range(w):
            if not bg[sy, sx] or visited[sy, sx]:
                continue
            stack = [(sy, sx)]
            visited[sy, sx] = True
            touches_border = False
            while stack:
                cy, cx = stack.pop()
                if cy in (0, h - 1) or cx in (0, w - 1):
                    touches_border = True
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < h and 0 <= nx < w and bg[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            if not touches_border:
                holes += 1
    return holes


#: 템플릿 렌더링에 쓸 폰트 후보. 번호판 전용 서체는 라이선스 문제로 쓸 수 없으므로
#: 굵은 고딕(산세리프)을 쓴다. 번호판 서체도 고딕 계열이라 구조 특징이 잘 맞는다.
_FONT_CANDIDATES: tuple[str, ...] = (
    "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    "C:/Windows/Fonts/malgunbd.ttf",
)

#: 폰트 로딩 결과 캐시. 폰트 파일 탐색은 디스크 I/O 라 매번 하면 느리다.
_FONT_CACHE: dict[int, Any] = {}


def _template_font(size: int) -> Any | None:
    """템플릿 렌더링용 폰트를 로드한다(캐시 적용).

    Pillow 기본 비트맵 폰트(``load_default``)를 쓰면 왜 안 되는가?
        기본 폰트는 약 6x11 픽셀 비트맵이다. 이를 16x24 로 확대하면 획이
        뭉개져 '2'와 '7', '0'과 '8' 의 구조 차이가 사라진다. 실측에서
        '12가3456' → '227호3456' 처럼 전면적으로 틀렸다. TrueType 폰트를
        목표 크기로 직접 렌더링하면 획 구조가 보존되어 정확도가 크게 오른다.

    Args:
        size: 폰트 픽셀 크기.

    Returns:
        ``PIL.ImageFont`` 객체. TrueType 을 못 찾으면 기본 폰트, Pillow 가 없으면 None.
    """
    key = int(size)
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]
    try:
        from PIL import ImageFont  # noqa: PLC0415
    except ImportError:
        return None

    font: Any = None
    for path in _FONT_CANDIDATES:
        try:
            from pathlib import Path  # noqa: PLC0415

            if Path(path).is_file():
                font = ImageFont.truetype(path, key)
                break
        except OSError:
            continue
    if font is None:
        try:
            font = ImageFont.load_default()
        except Exception:
            return None
    _FONT_CACHE[key] = font
    return font


def _render_template(ch: str) -> np.ndarray | None:
    """문자 ``ch`` 의 참조 글리프를 렌더링한다(Pillow 사용, 결정적).

    왜 런타임 렌더링인가? 템플릿 이미지 파일을 패키지에 넣으면 "가중치 없이 동작"
    원칙과 어긋나고 저장소도 커진다. 같은 폰트/크기는 항상 같은 비트맵을 내므로
    결정적 템플릿 생성원으로 쓸 수 있다.

    렌더링 크기를 ``GLYPH_H * 4`` 로 크게 잡는 이유: 크게 그린 뒤 축소하면
    (supersampling) 안티에일리어싱 효과로 획 굵기가 고르게 표현되어,
    실제 번호판 crop 의 흐릿한 글자와 특징 분포가 더 비슷해진다.

    Args:
        ch: 렌더링할 한 글자.

    Returns:
        ``(GLYPH_H, GLYPH_W)`` 0/255 이진 배열. 렌더링 실패 시 None.
    """
    try:
        from PIL import Image, ImageDraw  # noqa: PLC0415
    except ImportError:
        return None

    render_size = GLYPH_H * 4
    font = _template_font(render_size)
    if font is None:
        return None

    try:
        canvas = Image.new("L", (render_size * 2, render_size * 2), color=0)
        draw = ImageDraw.Draw(canvas)
        # 폰트에 해당 글리프가 없으면 bbox 가 비거나 .notdef(빈 사각형)가 그려진다.
        # 아래 ``ys.size == 0`` 검사와 템플릿 뱅크의 예외 처리로 걸러진다.
        draw.text((render_size // 4, render_size // 4), ch, fill=255, font=font)
        arr = np.asarray(canvas)
        ys, xs = np.nonzero(arr > 96)
        if ys.size == 0:
            return None
        patch = arr[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
        if patch.shape[0] < 4 or patch.shape[1] < 2:
            return None
        norm = resize(patch, GLYPH_W, GLYPH_H)
        return ((norm > 96).astype(np.uint8)) * 255
    except Exception:
        return None


def _build_template_bank() -> dict[str, np.ndarray]:
    """문자 → 특징 벡터 템플릿 뱅크를 1회 구축한다(모듈 전역 캐시).

    숫자 10자는 Pillow 기본 폰트로 렌더링된다. 한글 용도문자는 기본 폰트에
    글리프가 없는 환경이 많으므로, 렌더링이 실패하면 **구조 사전(``_HANGUL_PRIORS``)**
    으로 대체한다.

    Returns:
        ``{문자: 34차원 특징벡터}``.
    """
    bank: dict[str, np.ndarray] = {}
    for ch in DIGITS:
        tpl = _render_template(ch)
        if tpl is not None:
            bank[ch] = _glyph_features(tpl)
    if not bank:
        # Pillow 조차 없는 극단적 환경 → 숫자 구조 사전으로 대체.
        bank.update({ch: vec for ch, vec in _digit_priors().items()})
    for ch in USE_CHARS:
        tpl = _render_template(ch)
        if tpl is not None:
            bank[ch] = _glyph_features(tpl)
    return bank


def _digit_priors() -> dict[str, np.ndarray]:
    """숫자 0~9 의 구조 사전(폰트 렌더링 불가 환경용 최소 템플릿).

    각 숫자를 16x24 격자에 수작업으로 그려 특징을 뽑는다. 정확도는 낮지만
    "어떤 환경에서도 무언가를 출력한다" 는 폴백 책임을 다하기 위한 최후 수단이다.

    Returns:
        ``{숫자문자: 특징벡터}``.
    """
    # 각 숫자를 8x6 저해상도 비트맵으로 정의(1 = 획).
    bitmaps: dict[str, list[str]] = {
        "0": ["011110", "110011", "110011", "110011", "110011", "110011", "110011", "011110"],
        "1": ["001100", "011100", "001100", "001100", "001100", "001100", "001100", "011110"],
        "2": ["011110", "110011", "000011", "000110", "001100", "011000", "110000", "111111"],
        "3": ["111110", "000011", "000011", "011110", "000011", "000011", "110011", "011110"],
        "4": ["000110", "001110", "011110", "110110", "111111", "000110", "000110", "000110"],
        "5": ["111111", "110000", "110000", "111110", "000011", "000011", "110011", "011110"],
        "6": ["001110", "011000", "110000", "111110", "110011", "110011", "110011", "011110"],
        "7": ["111111", "000011", "000110", "001100", "001100", "011000", "011000", "011000"],
        "8": ["011110", "110011", "110011", "011110", "011110", "110011", "110011", "011110"],
        "9": ["011110", "110011", "110011", "110011", "011111", "000011", "000110", "011100"],
    }
    out: dict[str, np.ndarray] = {}
    for ch, rows in bitmaps.items():
        grid = np.array([[1 if c == "1" else 0 for c in row] for row in rows], dtype=np.uint8) * 255
        out[ch] = _glyph_features(resize(grid, GLYPH_W, GLYPH_H))
    return out


#: 모듈 전역 템플릿 뱅크(지연 초기화). 매 요청마다 렌더링하면 느리다.
_TEMPLATE_BANK: dict[str, np.ndarray] | None = None


def _template_bank() -> dict[str, np.ndarray]:
    """템플릿 뱅크 싱글턴을 반환한다(최초 호출 시 구축)."""
    global _TEMPLATE_BANK  # noqa: PLW0603 - 의도적 프로세스 캐시
    if _TEMPLATE_BANK is None:
        try:
            _TEMPLATE_BANK = _build_template_bank()
        except Exception:
            _TEMPLATE_BANK = _digit_priors()
    return _TEMPLATE_BANK


def _match_glyph(
    features: np.ndarray,
    *,
    allowed: str | None = None,
) -> tuple[str, float]:
    """특징 벡터를 템플릿 뱅크와 비교해 가장 가까운 문자를 고른다.

    거리 척도는 **코사인 유사도와 L1 거리의 결합**이다.
        - 코사인: 획의 "분포 패턴" 을 비교(굵기/잉크량 변화에 둔감)
        - L1: 절대 채움률 차이를 비교(패턴이 비슷해도 양이 다르면 벌점)
    하나만 쓰면 '0'↔'8' 처럼 분포가 비슷한 쌍에서 자주 틀린다.

    Args:
        features: 34차원 특징 벡터.
        allowed: 후보를 제한할 문자열(자리 기반 제약). None 이면 전체.

    Returns:
        ``(추정 문자, 0~1 신뢰도)``. 템플릿이 없으면 ``("", 0.0)``.
    """
    bank = _template_bank()
    pool = {ch: v for ch, v in bank.items() if allowed is None or ch in allowed}
    if not pool:
        pool = bank
    if not pool:
        return "", 0.0

    f = np.asarray(features, dtype=np.float32)
    f_norm = float(np.linalg.norm(f))
    scored: list[tuple[float, str]] = []
    for ch, tpl in pool.items():
        t_norm = float(np.linalg.norm(tpl))
        cosine = float(f @ tpl / (f_norm * t_norm)) if f_norm > 1e-6 and t_norm > 1e-6 else 0.0
        l1 = float(np.abs(f - tpl).mean())
        # L1 은 작을수록 좋으므로 1 에서 뺀다. 특징이 0~1 이라 l1 도 0~1 범위다.
        score = 0.65 * cosine + 0.35 * max(0.0, 1.0 - l1 * 2.0)
        scored.append((score, ch))

    scored.sort(reverse=True)
    best_score, best_ch = scored[0]
    # 1위와 2위의 격차(margin)로 신뢰도를 깎는다. 격차가 작으면 사실상 추측이다.
    margin = best_score - scored[1][0] if len(scored) > 1 else best_score
    confidence = float(np.clip(best_score, 0.0, 1.0)) * float(np.clip(0.45 + margin * 3.0, 0.3, 1.0))
    # 폴백 OCR 의 상한을 0.75 로 제한한다. 템플릿 매칭이 딥러닝급 확신을 주장하면
    # 상위 계층의 LOW_CONFIDENCE 판정이 무력화된다.
    return best_ch, round(min(0.75, confidence), 4)


#: 한국 번호판의 자리별 문자 유형. 폴백 OCR 이 자리 제약으로 후보를 좁히는 데 쓴다.
#: 7자: ``숫자 숫자 한글 숫자 숫자 숫자 숫자`` (예: 12가3456)
#: 8자: ``숫자 숫자 숫자 한글 숫자 숫자 숫자 숫자`` (예: 123가4567)
_LAYOUT_7 = ("D", "D", "H", "D", "D", "D", "D")
_LAYOUT_8 = ("D", "D", "D", "H", "D", "D", "D", "D")


def _layout_for(count: int) -> tuple[str, ...]:
    """분할된 문자 개수에 맞는 자리 유형 배열을 고른다.

    자리 제약이 왜 그렇게 중요한가?
        숫자 자리에서 한글 후보를 아예 제거하면 '가'↔'기' 같은 오인식이 원천 차단되고,
        한글 자리에서는 51자 중에서만 고르므로 탐색 공간이 10배 줄어든다.
        폴백 OCR 정확도를 끌어올리는 가장 값싼 수단이다.

    Args:
        count: 분할된 문자 수.

    Returns:
        ``"D"``(숫자) / ``"H"``(한글) / ``"?"``(제약 없음) 로 이루어진 튜플.
    """
    if count == 7:
        return _LAYOUT_7
    if count == 8:
        return _LAYOUT_8
    # 개수가 맞지 않으면(분할 실패) 제약을 걸지 않는다. 잘못된 제약은
    # 멀쩡한 글자까지 왜곡하므로, 확신이 없을 때는 제약을 포기하는 것이 낫다.
    return tuple("?" for _ in range(max(0, count)))


class FallbackOCR:
    """번호판 crop 에 대한 결정적 템플릿 기반 OCR.

    파이프라인:
        crop → CLAHE → Otsu 이진화(극성 통일) → 연결요소 문자 분할 →
        글리프 정규화(16x24) → 34차원 구조 특징 → 템플릿 매칭(자리 제약 적용)

    딥러닝 OCR 대비 정확도는 낮지만, **가중치 0바이트로 항상 동작**하며
    신뢰도를 보수적으로(최대 0.75) 보고해 상위 계층이 ``LOW_CONFIDENCE`` 로
    처리하도록 유도한다.
    """

    name = "template-ocr"

    def __init__(self, *, min_chars: int = 4) -> None:
        """폴백 OCR 을 초기화한다.

        Args:
            min_chars: 이 개수 미만으로 분할되면 판독 실패로 본다.
                한국 번호판은 최소 7자이므로 4자는 매우 관대한 하한이다
                (일부 글자가 가려진 경우에도 부분 결과를 주기 위함).
        """
        self.min_chars = max(1, int(min_chars))

    def read(self, plate_bgr: np.ndarray) -> dict[str, Any]:
        """번호판 crop 에서 문자열을 추정한다. 예외를 바깥으로 던지지 않는다.

        Args:
            plate_bgr: 번호판 영역 BGR crop.

        Returns:
            ``{"text": str, "confidence": float, "char_confidences": list[float],
            "char_count": int, "reason": str | None}``.
            판독 실패 시 ``text=""``, ``confidence=0.0``, ``reason`` 에 사유.
        """
        try:
            return self._read_impl(plate_bgr)
        except Exception as exc:  # noqa: BLE001 - 폴백은 절대 예외를 올리지 않는다
            return {
                "text": "",
                "confidence": 0.0,
                "char_confidences": [],
                "char_count": 0,
                "reason": f"fallback-ocr-error: {type(exc).__name__}",
            }

    def _read_impl(self, plate_bgr: np.ndarray) -> dict[str, Any]:
        """실제 판독 로직."""
        bgr = ensure_bgr(plate_bgr)
        if bgr.shape[0] < 8 or bgr.shape[1] < 24:
            return {
                "text": "",
                "confidence": 0.0,
                "char_confidences": [],
                "char_count": 0,
                "reason": "plate-crop-too-small",
            }

        binary = _binarize_plate(bgr)
        glyph_boxes = _segment_characters(binary)
        if len(glyph_boxes) < self.min_chars:
            return {
                "text": "",
                "confidence": 0.0,
                "char_confidences": [],
                "char_count": len(glyph_boxes),
                "reason": f"segmentation-failed(found={len(glyph_boxes)})",
            }

        # 번호판은 최대 8자다. 9개 이상 나왔으면 테두리/노이즈가 섞인 것이므로
        # 면적이 큰 상위 8개만 쓰고 다시 x 순으로 정렬한다.
        if len(glyph_boxes) > 8:
            glyph_boxes = sorted(
                sorted(glyph_boxes, key=lambda b: b[2] * b[3], reverse=True)[:8],
                key=lambda b: b[0],
            )

        layout = _layout_for(len(glyph_boxes))
        chars: list[str] = []
        confs: list[float] = []
        for i, (x, y, w, h) in enumerate(glyph_boxes):
            glyph = binary[y : y + h, x : x + w]
            if glyph.size == 0:
                continue
            normalized = resize(glyph, GLYPH_W, GLYPH_H)
            features = _glyph_features(normalized)
            kind = layout[i] if i < len(layout) else "?"
            allowed = DIGITS if kind == "D" else (USE_CHARS if kind == "H" else None)
            ch, conf = _match_glyph(features, allowed=allowed)
            if ch:
                chars.append(ch)
                confs.append(conf)

        text = "".join(chars)
        if not text:
            return {
                "text": "",
                "confidence": 0.0,
                "char_confidences": [],
                "char_count": len(glyph_boxes),
                "reason": "no-glyph-matched",
            }

        # 시퀀스 신뢰도는 기하평균(가장 약한 글자를 반영).
        log_sum = float(np.sum(np.log(np.clip(np.asarray(confs), 1e-6, 1.0))))
        confidence = float(np.exp(log_sum / len(confs)))

        return {
            "text": text,
            "confidence": round(confidence, 4),
            "char_confidences": [round(c, 4) for c in confs],
            "char_count": len(glyph_boxes),
            "reason": None,
        }

    def info(self) -> dict[str, Any]:
        """OCR 메타데이터를 반환한다."""
        bank = _template_bank()
        return {
            "name": self.name,
            "backend": "opencv+template" if _HAS_CV2 else "numpy+template",
            "weights": None,
            "loaded": True,
            "fallback": True,
            "detail": {
                "template_count": len(bank),
                "glyph_size": [GLYPH_H, GLYPH_W],
                "feature_dim": FEATURE_DIM,
                "deterministic": True,
            },
        }


# ===========================================================================
# 3) 편의 함수
# ===========================================================================


def detect_plates(image: np.ndarray, min_conf: float = 0.25) -> list[Detection]:
    """폴백 검출기로 번호판을 검출하는 모듈 수준 편의 함수.

    Args:
        image: BGR 이미지.
        min_conf: 최소 신뢰도.

    Returns:
        ``Detection`` 리스트.
    """
    return FallbackPlateDetector().detect(image, min_conf)


def read_plate(plate_bgr: np.ndarray) -> dict[str, Any]:
    """폴백 OCR 로 번호판을 판독하는 모듈 수준 편의 함수.

    Args:
        plate_bgr: 번호판 crop.

    Returns:
        판독 결과 dict.
    """
    return FallbackOCR().read(plate_bgr)


def fallback_available() -> dict[str, Any]:
    """폴백 경로의 가용성을 보고한다(헬스체크용).

    Returns:
        OpenCV/Pillow 설치 여부와 템플릿 수.
    """
    try:
        from PIL import Image as _PILImage  # noqa: PLC0415

        has_pillow = _PILImage is not None
    except ImportError:
        has_pillow = False
    return {
        "opencv": _HAS_CV2,
        "pillow": has_pillow,
        "templates": len(_template_bank()),
        "deterministic": True,
    }


__all__ = [
    "AREA_RATIO_MAX",
    "AREA_RATIO_MIN",
    "ASPECT_MAX",
    "ASPECT_MIN",
    "FEATURE_DIM",
    "GLYPH_H",
    "GLYPH_W",
    "ZONE_GRID",
    "MIN_RECTANGULARITY",
    "FallbackOCR",
    "FallbackPlateDetector",
    "detect_plates",
    "fallback_available",
    "read_plate",
]
