"""합성 한국 번호판 이미지 생성기.

왜 합성 데이터가 필요한가?
    실제 번호판 사진은 개인정보(차량 소유자 식별 가능)에 해당해 공개 데이터셋이
    극히 드물다. 번호판은 **규격이 법으로 정해진 인공물**이라 합성으로 실제와
    매우 비슷하게 만들 수 있고, 라벨이 100% 정확하다는 장점도 크다.
    실무에서도 "합성으로 사전학습 → 소량 실데이터로 미세조정" 이 표준 레시피다.

규격(자동차등록번호판 기준, 단위 mm):
    - 신형 비사업용 승용 단판: 520 x 110  (종횡비 약 4.7)
    - 구형 비사업용 승용 단판: 335 x 155  (종횡비 약 2.2)
    - 사업용(노란 바탕): 520 x 110
    본 생성기는 픽셀 단위로 비례 축소해 그린다.

증강(augmentation) 설계 의도:
    실제 게이트 카메라에서 발생하는 열화만 모사한다 — 투시(비스듬한 설치각),
    모션 블러(주행 중), 조명 불균일(역광/야간 조명), JPEG 압축, 센서 노이즈.
    색상 반전이나 상하 뒤집기 같은 "현실에 없는" 증강은 넣지 않는다.
    현실 분포에 없는 증강은 학습을 방해할 뿐이다.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from app.ocr.charset import DIGITS, REGION_NAMES, USE_CHARS

try:
    import cv2

    _HAS_CV2 = True
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]
    _HAS_CV2 = False

try:
    from PIL import Image, ImageDraw, ImageFilter, ImageFont

    _HAS_PIL = True
except ImportError:  # pragma: no cover
    Image = ImageDraw = ImageFilter = ImageFont = None  # type: ignore[assignment]
    _HAS_PIL = False


#: 번호판 유형별 (픽셀 너비, 픽셀 높이, 바탕색 RGB, 글자색 RGB).
#: 실제 규격 비율을 유지하면서 OCR 학습에 충분한 해상도로 확대했다.
PLATE_STYLES: dict[str, tuple[int, int, tuple[int, int, int], tuple[int, int, int]]] = {
    # 신형 비사업용: 흰 바탕 + 검정 글씨
    "new_private": (520, 110, (245, 245, 243), (24, 24, 24)),
    # 사업용(택시/버스/화물): 노란 바탕 + 검정 글씨
    "commercial": (520, 110, (238, 205, 48), (24, 24, 24)),
    # 구형 비사업용: 녹색 바탕 + 흰 글씨 (2006년 이전)
    "old_private": (335, 155, (36, 112, 72), (242, 242, 240)),
    # 전기차(친환경): 파란 바탕 + 흰 글씨
    "electric": (520, 110, (40, 92, 164), (244, 244, 244)),
}

#: 리눅스/도커 환경에서 흔히 찾을 수 있는 한글 지원 폰트 경로 후보.
#: 번호판 전용 서체(헤드라인체)는 배포 라이선스 문제가 있어 시스템 폰트를 쓴다.
FONT_CANDIDATES: tuple[str, ...] = (
    "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJKkr-Bold.otf",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    "C:/Windows/Fonts/malgunbd.ttf",
)


@dataclass(slots=True)
class SynthSample:
    """합성 샘플 한 건.

    Attributes:
        image: ``(H, W, 3)`` uint8 BGR 이미지(배경 포함 전체 장면).
        text: 정답 번호판 문자열(공백 없음, 예 ``"12가3456"``).
        bbox: 장면 좌표계의 번호판 박스 ``(x1, y1, x2, y2)``.
        style: 사용한 번호판 유형 키.
        meta: 증강 파라미터 등 부가 정보(재현/디버깅용).
    """

    image: np.ndarray
    text: str
    bbox: tuple[int, int, int, int]
    style: str
    meta: dict[str, Any] = field(default_factory=dict)


def find_korean_font(size: int) -> Any:
    """한글 렌더링이 가능한 폰트를 찾아 로드한다.

    후보 경로를 순서대로 시도하고, 모두 실패하면 Pillow 기본 비트맵 폰트를
    반환한다(한글은 □ 로 렌더링되지만 숫자 학습은 가능하다).

    Args:
        size: 폰트 크기(px).

    Returns:
        ``PIL.ImageFont`` 객체.

    Raises:
        RuntimeError: Pillow 자체가 설치되지 않은 경우.
    """
    if not _HAS_PIL:
        msg = "Pillow 가 필요합니다: pip install pillow"
        raise RuntimeError(msg)
    for path in FONT_CANDIDATES:
        if Path(path).is_file():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


def random_plate_text(rng: random.Random, *, with_region: bool = False) -> str:
    """무작위 한국 번호판 문자열을 생성한다.

    Args:
        rng: 재현성을 위한 ``random.Random`` 인스턴스.
            전역 ``random`` 을 쓰지 않는 이유: 같은 시드로 같은 데이터셋이
            재생성되어야 교재 실습 결과가 일치하기 때문이다.
        with_region: True 면 구형 지역명 포함 형식(``서울12가3456``).

    Returns:
        번호판 문자열(공백 없음).
    """
    use = rng.choice(USE_CHARS)
    tail = "".join(rng.choice(DIGITS) for _ in range(4))
    if with_region:
        region = rng.choice(REGION_NAMES)
        head = "".join(rng.choice(DIGITS) for _ in range(2))
        return f"{region}{head}{use}{tail}"
    # 신형은 앞자리가 2자리 또는 3자리다. 3자리형이 2019년 이후 신규 발급분이라
    # 비중을 조금 더 준다(실제 분포 모사).
    head_len = 3 if rng.random() < 0.55 else 2
    head = "".join(rng.choice(DIGITS) for _ in range(head_len))
    return f"{head}{use}{tail}"


def render_plate(text: str, style: str = "new_private", *, scale: float = 1.0) -> np.ndarray:
    """번호판 평면 이미지를 렌더링한다(증강 없음, 정면 뷰).

    레이아웃 규칙:
        - 지역명이 있으면 상단 작은 글씨 또는 좌측에 배치(구형 2단 번호판 모사).
        - 숫자/한글의 자간을 균등 분배하되, 한글 용도문자 앞뒤에 약간 더 여백을 둔다
          (실제 번호판의 시각적 리듬을 따른 것).
        - 테두리는 바탕보다 어두운 선 + 안쪽 여백. 검출기 학습 시 테두리가
          번호판 경계의 중요한 단서가 된다.

    Args:
        text: 번호판 문자열(공백 없이 입력. 지역명은 앞 2글자로 자동 인식).
        style: ``PLATE_STYLES`` 의 키.
        scale: 해상도 배율(1.0 = 규격 비례 기본 크기).

    Returns:
        ``(H, W, 3)`` uint8 BGR 번호판 이미지.

    Raises:
        RuntimeError: Pillow 미설치.
    """
    if not _HAS_PIL:
        msg = "Pillow 가 필요합니다: pip install pillow"
        raise RuntimeError(msg)

    width, height, bg, fg = PLATE_STYLES.get(style, PLATE_STYLES["new_private"])
    width = max(64, int(width * scale))
    height = max(24, int(height * scale))

    canvas = Image.new("RGB", (width, height), color=bg)
    draw = ImageDraw.Draw(canvas)

    # 테두리: 바탕색을 30% 어둡게 한 선.
    border_color = tuple(max(0, int(c * 0.7)) for c in bg)
    inset = max(2, int(height * 0.045))
    draw.rounded_rectangle(
        [inset, inset, width - inset - 1, height - inset - 1],
        radius=max(3, int(height * 0.08)),
        outline=border_color,
        width=max(2, int(height * 0.035)),
    )

    # 지역명 분리: 앞 2글자가 모두 한글이고 지역명 목록에 있으면 지역명으로 본다.
    region = ""
    body = text
    if len(text) >= 2 and text[:2] in REGION_NAMES:
        region = text[:2]
        body = text[2:]

    is_two_line = style == "old_private" and bool(region)
    if is_two_line:
        # 구형 2단: 상단에 지역명 + 분류번호, 하단에 용도문자 + 4자리.
        region_font = find_korean_font(max(10, int(height * 0.34)))
        body_font = find_korean_font(max(14, int(height * 0.50)))
        head = body[:2]
        tail = body[2:]
        _draw_centered(draw, f"{region}{head}", region_font, width, int(height * 0.28), fg)
        _draw_centered(draw, tail, body_font, width, int(height * 0.68), fg)
    else:
        font_size = max(14, int(height * 0.62))
        font = find_korean_font(font_size)
        label = f"{region} {body}".strip() if region else body
        _draw_tracked(draw, label, font, width, height, fg)

    rgb = np.asarray(canvas)
    # Pillow 는 RGB, 파이프라인은 BGR 이다.
    return np.ascontiguousarray(rgb[:, :, ::-1])


def _draw_centered(
    draw: Any, text: str, font: Any, width: int, center_y: int, color: tuple[int, int, int]
) -> None:
    """텍스트를 지정한 y 중심선에 가로 중앙 정렬로 그린다.

    Args:
        draw: ``ImageDraw.Draw`` 객체.
        text: 그릴 문자열.
        font: 폰트 객체.
        width: 캔버스 너비.
        center_y: 텍스트 세로 중심 y.
        color: 글자색 RGB.
    """
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        off_x, off_y = bbox[0], bbox[1]
    except Exception:
        tw, th, off_x, off_y = len(text) * 10, 12, 0, 0
    draw.text((width // 2 - tw // 2 - off_x, center_y - th // 2 - off_y), text, font=font, fill=color)


def _draw_tracked(
    draw: Any,
    text: str,
    font: Any,
    width: int,
    height: int,
    color: tuple[int, int, int],
) -> None:
    """자간(tracking)을 균등 분배해 한 줄 번호판을 그린다.

    왜 ``draw.text`` 한 번으로 끝내지 않는가?
        폰트 기본 자간은 번호판 서체보다 촘촘하다. 문자를 개별 배치해 남는 폭을
        고르게 나누면 실제 번호판의 넓은 자간을 재현할 수 있고, 덤으로
        **문자별 bbox 를 알 수 있어** 문자 단위 라벨(검출 기반 OCR 학습)도 뽑을 수 있다.

    Args:
        draw: ImageDraw 객체.
        text: 문자열(공백 포함 가능).
        font: 폰트 객체.
        width: 캔버스 너비.
        height: 캔버스 높이.
        color: 글자색 RGB.
    """
    chars = list(text)
    widths: list[int] = []
    heights: list[int] = []
    for ch in chars:
        try:
            bbox = draw.textbbox((0, 0), ch, font=font)
            widths.append(bbox[2] - bbox[0])
            heights.append(bbox[3] - bbox[1])
        except Exception:
            widths.append(int(height * 0.4))
            heights.append(int(height * 0.6))

    side_margin = int(width * 0.06)
    usable = width - 2 * side_margin
    total_glyph = sum(widths)
    gaps = max(1, len(chars) - 1)
    # 남는 폭을 자간으로 분배. 음수가 되면(글자가 넘침) 0 으로 눌러 겹치게 둔다.
    gap = max(0, (usable - total_glyph) // gaps)

    x = side_margin
    baseline_y = height // 2
    for ch, w, h in zip(chars, widths, heights, strict=False):
        if ch != " ":
            try:
                bbox = draw.textbbox((0, 0), ch, font=font)
                draw.text((x - bbox[0], baseline_y - h // 2 - bbox[1]), ch, font=font, fill=color)
            except Exception:
                draw.text((x, baseline_y - h // 2), ch, font=font, fill=color)
        x += w + gap


# ---------------------------------------------------------------------------
# 증강
# ---------------------------------------------------------------------------


def augment_perspective(
    plate: np.ndarray, rng: random.Random, *, max_shift: float = 0.14
) -> tuple[np.ndarray, np.ndarray]:
    """투시 변환으로 카메라 설치각을 모사한다.

    게이트 카메라는 보통 차량 위쪽 비스듬한 각도에 설치되므로 번호판이
    사다리꼴로 보인다. 네 꼭짓점을 각각 무작위로 옮겨 이를 재현한다.

    Args:
        plate: 정면 번호판 이미지.
        rng: 난수 생성기.
        max_shift: 각 꼭짓점의 최대 이동 비율(너비/높이 대비).

    Returns:
        ``(변환된 이미지, 3x3 호모그래피 행렬)``. OpenCV 가 없으면 원본과 단위행렬.
    """
    if not _HAS_CV2:
        return plate, np.eye(3, dtype=np.float32)

    h, w = plate.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])

    def jitter() -> tuple[float, float]:
        """꼭짓점 하나의 (dx, dy) 무작위 이동량을 만든다."""
        return (
            rng.uniform(-max_shift, max_shift) * w,
            rng.uniform(-max_shift, max_shift) * h,
        )

    dst = np.float32([[x + dx, y + dy] for (x, y), (dx, dy) in zip(src, [jitter() for _ in range(4)], strict=False)])
    # 변환 후 좌표가 음수가 되면 잘리므로 전체를 양수 영역으로 평행이동한다.
    min_xy = dst.min(axis=0)
    dst -= np.minimum(min_xy, 0.0)
    out_w = int(np.ceil(dst[:, 0].max()))
    out_h = int(np.ceil(dst[:, 1].max()))
    out_w = max(w // 2, min(out_w, w * 2))
    out_h = max(h // 2, min(out_h, h * 2))

    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(
        plate, matrix, (out_w, out_h), borderMode=cv2.BORDER_REPLICATE
    )
    return warped, matrix


def augment_motion_blur(image: np.ndarray, rng: random.Random, *, max_kernel: int = 9) -> np.ndarray:
    """주행 중 촬영으로 생기는 모션 블러를 모사한다.

    방향성 커널(한 줄만 1인 행렬)로 합성곱한다. 가우시안 블러와 달리
    **한 방향으로만 번지므로** 실제 모션 블러와 더 비슷하고, 번호판 세로획이
    가로로 늘어지는 특유의 열화를 재현한다.

    Args:
        image: 입력 이미지.
        rng: 난수 생성기.
        max_kernel: 최대 커널 크기(홀수로 보정됨).

    Returns:
        블러 적용된 이미지.
    """
    if not _HAS_CV2:
        return image
    size = rng.randrange(3, max(5, int(max_kernel)) + 1, 2)
    kernel = np.zeros((size, size), dtype=np.float32)
    if rng.random() < 0.7:
        kernel[size // 2, :] = 1.0  # 수평 블러(차량 진행 방향)
    else:
        kernel[:, size // 2] = 1.0  # 수직 블러(카메라 흔들림)
    kernel /= kernel.sum()
    return cv2.filter2D(image, -1, kernel)


def augment_lighting(image: np.ndarray, rng: random.Random) -> np.ndarray:
    """조명 불균일(역광/야간 조명/그림자)을 모사한다.

    기법: 2차원 가우시안 형태의 밝기 맵을 만들어 곱한다. 한쪽이 밝고 반대쪽이
    어두워지므로 전역 임계값 이진화를 무력화시킨다 — 즉 CLAHE/적응 이진화의
    필요성을 학습 데이터에 심어 주는 증강이다.

    Args:
        image: 입력 이미지.
        rng: 난수 생성기.

    Returns:
        조명 변화가 적용된 이미지.
    """
    h, w = image.shape[:2]
    cx = rng.uniform(0.0, 1.0) * w
    cy = rng.uniform(0.0, 1.0) * h
    sigma = max(w, h) * rng.uniform(0.35, 0.9)

    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    dist2 = (xs - cx) ** 2 + (ys - cy) ** 2
    spot = np.exp(-dist2 / (2.0 * sigma**2))

    gain_lo = rng.uniform(0.45, 0.80)
    gain_hi = rng.uniform(1.05, 1.60)
    gain = gain_lo + (gain_hi - gain_lo) * spot
    out = image.astype(np.float32) * gain[:, :, None]
    # 전역 밝기/대비도 함께 흔든다.
    out = out * rng.uniform(0.85, 1.15) + rng.uniform(-18.0, 18.0)
    return np.clip(out, 0, 255).astype(np.uint8)


def augment_noise(image: np.ndarray, rng: random.Random, *, max_sigma: float = 14.0) -> np.ndarray:
    """센서 노이즈(가우시안)를 추가한다.

    야간/저조도에서 ISO 를 올리면 노이즈가 급증한다. 노이즈 없는 데이터로만
    학습한 모델은 야간 영상에서 성능이 급락하므로 반드시 포함해야 하는 증강이다.

    Args:
        image: 입력 이미지.
        rng: 난수 생성기(시드 전파용).
        max_sigma: 최대 노이즈 표준편차.

    Returns:
        노이즈가 추가된 이미지.
    """
    # numpy 난수도 같은 시드 체계를 쓰도록 rng 에서 시드를 뽑는다(재현성).
    nprng = np.random.default_rng(rng.randrange(0, 2**32))
    sigma = rng.uniform(1.0, float(max_sigma))
    noise = nprng.normal(0.0, sigma, size=image.shape)
    return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def augment_jpeg(image: np.ndarray, rng: random.Random, *, min_quality: int = 35) -> np.ndarray:
    """JPEG 압축 열화를 모사한다.

    CCTV/업로드 경로에서 이미지는 거의 항상 JPEG 로 압축된다. 블록 아티팩트는
    문자 경계를 뭉개는 주요 원인이므로 학습에 포함해야 한다.

    Args:
        image: 입력 이미지.
        rng: 난수 생성기.
        min_quality: 최저 JPEG 품질.

    Returns:
        압축/복원된 이미지. OpenCV 가 없으면 원본.
    """
    if not _HAS_CV2:
        return image
    quality = rng.randint(int(min_quality), 92)
    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return image
    decoded = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return image if decoded is None else decoded


def compose_scene(
    plate: np.ndarray,
    rng: random.Random,
    *,
    scene_size: tuple[int, int] = (720, 1280),
    plate_width_ratio: float = 0.28,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """번호판을 차량 배경 장면에 합성한다.

    배경을 왜 단색이 아니라 그라데이션 + 질감으로 만드는가?
        단색 배경에서 학습한 검출기는 "에지가 있는 곳 = 번호판" 이라고 배워 버린다.
        실제 차량 사진에는 범퍼 라인, 그릴 격자, 그림자 등 에지가 많으므로
        배경에도 방해 에지(distractor)를 넣어야 검출기가 번호판의 고유 텍스처를 배운다.

    Args:
        plate: 번호판 이미지(증강 적용 후).
        rng: 난수 생성기.
        scene_size: ``(높이, 너비)`` 장면 크기.
        plate_width_ratio: 장면 너비 대비 번호판 너비 비율.

    Returns:
        ``(장면 이미지, 번호판 bbox)``.
    """
    scene_h, scene_w = scene_size
    # --- 배경: 수직 그라데이션(차체 색) + 질감 ---
    base_color = np.array(
        [rng.randint(40, 200), rng.randint(40, 200), rng.randint(40, 200)], dtype=np.float32
    )
    gradient = np.linspace(0.65, 1.15, scene_h, dtype=np.float32)[:, None, None]
    # ``np.broadcast_to`` 는 읽기 전용 뷰를 주므로, 이후 in-place 연산을 위해
    # 명시적으로 (H, W, 3) 실배열로 확장(복사)한다.
    scene = np.empty((scene_h, scene_w, 3), dtype=np.float32)
    scene[:] = base_color[None, None, :] * gradient

    nprng = np.random.default_rng(rng.randrange(0, 2**32))
    scene += nprng.normal(0.0, 7.0, size=(scene_h, scene_w, 3))

    # --- 방해 에지: 수평 라인(범퍼) + 격자(그릴) ---
    for _ in range(rng.randint(2, 6)):
        y = rng.randint(0, scene_h - 1)
        thickness = rng.randint(1, 5)
        shade = rng.uniform(0.45, 1.5)
        scene[y : y + thickness, :, :] *= shade
    if rng.random() < 0.6:
        grill_y = rng.randint(0, max(1, scene_h // 2))
        grill_h = rng.randint(30, max(40, scene_h // 3))
        step = rng.randint(6, 18)
        scene[grill_y : grill_y + grill_h : step, :, :] *= 0.55

    scene_img = np.clip(scene, 0, 255).astype(np.uint8)

    # --- 번호판 리사이즈 및 배치 ---
    target_w = max(32, int(scene_w * plate_width_ratio))
    ph, pw = plate.shape[:2]
    target_h = max(12, int(target_w * ph / max(1, pw)))
    if _HAS_CV2:
        plate_resized = cv2.resize(plate, (target_w, target_h), interpolation=cv2.INTER_AREA)
    else:
        ys = np.clip((np.arange(target_h) * ph / target_h).astype(int), 0, ph - 1)
        xs = np.clip((np.arange(target_w) * pw / target_w).astype(int), 0, pw - 1)
        plate_resized = plate[ys][:, xs]

    max_x = max(0, scene_w - target_w)
    max_y = max(0, scene_h - target_h)
    # 번호판은 보통 화면 중하단에 있다. 위쪽 1/3 은 제외해 현실성을 높인다.
    x0 = rng.randint(0, max_x) if max_x > 0 else 0
    y0 = rng.randint(max_y // 3, max_y) if max_y > 0 else 0

    scene_img[y0 : y0 + target_h, x0 : x0 + target_w] = plate_resized
    return scene_img, (x0, y0, x0 + target_w, y0 + target_h)


def generate_sample(
    rng: random.Random,
    *,
    style: str | None = None,
    with_region: bool | None = None,
    scene: bool = True,
    augment: bool = True,
    scene_size: tuple[int, int] = (720, 1280),
) -> SynthSample:
    """합성 샘플 한 건을 생성한다.

    Args:
        rng: 난수 생성기(시드 고정으로 재현 가능).
        style: 번호판 유형. None 이면 무작위(실제 분포를 반영해 신형 비중 높음).
        with_region: 지역명 포함 여부. None 이면 구형 스타일일 때만 확률적으로 포함.
        scene: True 면 배경 장면에 합성(검출기 학습용),
            False 면 번호판만 반환(OCR 학습용).
        augment: 증강 적용 여부.
        scene_size: 장면 크기 ``(H, W)``.

    Returns:
        SynthSample.
    """
    if style is None:
        # 실제 도로 분포 근사: 신형 비사업용이 압도적으로 많다.
        style = rng.choices(
            list(PLATE_STYLES.keys()), weights=[0.62, 0.14, 0.14, 0.10], k=1
        )[0]
    if with_region is None:
        with_region = style == "old_private" and rng.random() < 0.8

    text = random_plate_text(rng, with_region=bool(with_region))
    plate = render_plate(text, style=style, scale=rng.uniform(0.85, 1.25))

    meta: dict[str, Any] = {"augment": bool(augment), "with_region": bool(with_region)}

    if augment:
        if rng.random() < 0.75:
            plate, _ = augment_perspective(plate, rng)
            meta["perspective"] = True
        if rng.random() < 0.45:
            plate = augment_motion_blur(plate, rng)
            meta["motion_blur"] = True
        if rng.random() < 0.80:
            plate = augment_lighting(plate, rng)
            meta["lighting"] = True

    if scene:
        image, bbox = compose_scene(
            plate, rng, scene_size=scene_size, plate_width_ratio=rng.uniform(0.14, 0.42)
        )
    else:
        image = plate
        bbox = (0, 0, plate.shape[1], plate.shape[0])

    if augment:
        if rng.random() < 0.70:
            image = augment_noise(image, rng)
            meta["noise"] = True
        if rng.random() < 0.65:
            image = augment_jpeg(image, rng)
            meta["jpeg"] = True

    return SynthSample(image=image, text=text, bbox=bbox, style=style, meta=meta)


def generate_dataset(
    out_dir: str | Path,
    count: int,
    *,
    seed: int = 20261024,
    scene: bool = True,
    augment: bool = True,
    yolo_labels: bool = True,
    scene_size: tuple[int, int] = (720, 1280),
) -> dict[str, Any]:
    """합성 데이터셋을 디스크에 생성한다.

    출력 구조(ultralytics YOLO 관례와 호환):
        ``out_dir/images/000000.jpg``
        ``out_dir/labels/000000.txt``  (``cls cx cy w h``, 0~1 정규화)
        ``out_dir/labels.jsonl``       (OCR 학습용 ``{"image", "text", "bbox"}``)

    Args:
        out_dir: 출력 디렉터리.
        count: 생성할 샘플 수.
        seed: 난수 시드(재현성).
        scene: 장면 합성 여부.
        augment: 증강 적용 여부.
        yolo_labels: YOLO txt 라벨 생성 여부.
        scene_size: 장면 크기.

    Returns:
        생성 통계 dict.

    Raises:
        RuntimeError: Pillow 또는 OpenCV 가 없어 저장할 수 없을 때.
    """
    if not _HAS_PIL:
        msg = "Pillow 가 필요합니다: pip install pillow"
        raise RuntimeError(msg)

    root = Path(out_dir)
    img_dir = root / "images"
    lbl_dir = root / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    if yolo_labels:
        lbl_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    jsonl_path = root / "labels.jsonl"
    styles: dict[str, int] = {}
    written = 0

    with jsonl_path.open("w", encoding="utf-8") as jsonl:
        for i in range(max(0, int(count))):
            sample = generate_sample(
                rng, scene=scene, augment=augment, scene_size=scene_size
            )
            stem = f"{i:06d}"
            img_path = img_dir / f"{stem}.jpg"
            if not _save_image(img_path, sample.image):
                continue

            if yolo_labels:
                h, w = sample.image.shape[:2]
                x1, y1, x2, y2 = sample.bbox
                cx = (x1 + x2) / 2.0 / w
                cy = (y1 + y2) / 2.0 / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                (lbl_dir / f"{stem}.txt").write_text(
                    f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n", encoding="utf-8"
                )

            jsonl.write(
                json.dumps(
                    {
                        "image": f"images/{stem}.jpg",
                        "text": sample.text,
                        "bbox": [int(v) for v in sample.bbox],
                        "style": sample.style,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            styles[sample.style] = styles.get(sample.style, 0) + 1
            written += 1

    # ultralytics 학습용 data.yaml 도 함께 만들어 둔다(train_detector 가 사용).
    (root / "data.yaml").write_text(
        "\n".join(
            [
                f"path: {root.resolve()}",
                "train: images",
                "val: images",
                "nc: 1",
                "names: [plate]",
                "",
            ]
        ),
        encoding="utf-8",
    )

    return {
        "out_dir": str(root.resolve()),
        "written": written,
        "requested": int(count),
        "styles": styles,
        "seed": seed,
        "labels_jsonl": str(jsonl_path.resolve()),
    }


def _save_image(path: Path, image: np.ndarray) -> bool:
    """이미지를 저장한다(OpenCV → Pillow 순서로 시도).

    Args:
        path: 저장 경로.
        image: BGR 배열.

    Returns:
        성공하면 True.
    """
    if _HAS_CV2:
        try:
            return bool(cv2.imwrite(str(path), image))
        except Exception:
            pass
    if _HAS_PIL:
        try:
            Image.fromarray(image[:, :, ::-1]).save(path, quality=92)
            return True
        except Exception:
            return False
    return False


def build_arg_parser() -> argparse.ArgumentParser:
    """CLI 인자 파서를 만든다.

    Returns:
        argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(
        prog="python -m app.train.synth_plate",
        description="합성 한국 번호판 데이터셋 생성기",
    )
    parser.add_argument("--out", default="./data/synth", help="출력 디렉터리")
    parser.add_argument("--count", type=int, default=200, help="생성 샘플 수")
    parser.add_argument("--seed", type=int, default=20261024, help="난수 시드(재현성)")
    parser.add_argument(
        "--no-scene",
        action="store_true",
        help="배경 합성 없이 번호판만 생성(OCR 학습용)",
    )
    parser.add_argument("--no-augment", action="store_true", help="증강 비활성화")
    parser.add_argument("--height", type=int, default=720, help="장면 높이")
    parser.add_argument("--width", type=int, default=1280, help="장면 너비")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI 진입점.

    Args:
        argv: 인자 리스트(None 이면 ``sys.argv``).

    Returns:
        프로세스 종료 코드(0 = 성공).
    """
    args = build_arg_parser().parse_args(argv)
    if not _HAS_PIL:
        print("[ERROR] Pillow 가 설치되지 않았습니다. pip install pillow", file=sys.stderr)
        return 2
    try:
        stats = generate_dataset(
            args.out,
            args.count,
            seed=args.seed,
            scene=not args.no_scene,
            augment=not args.no_augment,
            scene_size=(args.height, args.width),
        )
    except Exception as exc:  # noqa: BLE001 - CLI 는 스택트레이스 대신 메시지를 보여준다
        print(f"[ERROR] 생성 실패: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "FONT_CANDIDATES",
    "PLATE_STYLES",
    "SynthSample",
    "augment_jpeg",
    "augment_lighting",
    "augment_motion_blur",
    "augment_noise",
    "augment_perspective",
    "compose_scene",
    "find_korean_font",
    "generate_dataset",
    "generate_sample",
    "main",
    "random_plate_text",
    "render_plate",
]
