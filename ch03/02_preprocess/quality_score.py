# -*- coding: utf-8 -*-
"""번호판 이미지 품질 점수 — 블러·밝기·픽셀폭·대비를 합산해 판독 가능성을 추정한다.

계약서 응답의 quality 필드와 같은 구조를 만든다.
    {"score": 0.86, "blur": 142.3, "brightness": 118.4,
     "plate_px_width": 184, "readable": true, "reason": null}

실행:  python 02_preprocess/quality_score.py
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

OUT_DIR = Path("outputs/quality")

#: 라플라시안 분산 하한. 이보다 낮으면 글자 윤곽이 사실상 사라진 상태다.
#: 이 값은 카메라와 설치 환경마다 다르므로, 운영 전에 실제 수집 이미지로
#: '읽을 수 있는 것' 과 '읽을 수 없는 것' 의 분산 분포를 그려 경계를 다시 잡아야 한다.
BLUR_MIN = 120.0
#: 라플라시안 분산 포화점. 이 이상은 더 좋아져도 OCR 정확도가 거의 변하지 않는다.
BLUR_GOOD = 800.0
#: 평균 밝기 적정 구간. 번호판은 흰 바탕이라 중간보다 약간 밝은 쪽이 좋다.
BRIGHT_LOW, BRIGHT_HIGH = 70.0, 190.0
#: 번호판 가로 픽셀폭 하한. 7~8글자가 들어가므로 글자당 최소 12px 은 확보해야 한다.
WIDTH_MIN, WIDTH_GOOD = 96, 200


def blur_metric(gray: np.ndarray) -> float:
    """라플라시안 분산으로 선명도를 측정한다.

    라플라시안은 2차 미분이므로 에지에서 큰 값을 낸다. 초점이 맞은 이미지는
    글자 테두리에서 값이 크게 흔들려 분산이 커지고, 흐린 이미지는 평탄해져 분산이 작다.

    두 가지 함정을 반드시 보정해야 한다.
      1. 이미지 크기 의존: 큰 이미지는 에지 픽셀이 많아 값이 커진다.
         → 측정 전에 고정 높이로 정규화한다(evaluate 가 처리한다).
      2. 노이즈 의존: 픽셀 단위 잡음도 2차 미분값을 키운다. 보정하지 않으면
         '노이즈 가득한 흐린 이미지' 가 선명한 이미지보다 높은 점수를 받는다.
         → 3x3 중간값 필터로 점 잡음만 걷어낸 뒤 측정한다. 중간값 필터는
           글자 테두리 같은 넓은 에지는 보존하므로 선명도 정보는 남는다.
    """
    denoised = cv2.medianBlur(gray, 3)
    return float(cv2.Laplacian(denoised, cv2.CV_64F).var())


def brightness_metric(gray: np.ndarray) -> float:
    """평균 밝기(0~255). 역광이면 낮고, 반사광이 강하면 240 근처로 포화된다."""
    return float(gray.mean())


def contrast_metric(gray: np.ndarray) -> float:
    """표준편차 기반 대비. 번호판 바탕과 글자의 밝기 차가 충분한지 본다."""
    return float(gray.std())


def saturated_ratio(gray: np.ndarray, high: int = 250) -> float:
    """포화(흰색으로 날아간) 픽셀 비율. 반사광·플래시 과노출 탐지에 쓴다."""
    return float((gray >= high).mean())


def _ramp(value: float, low: float, high: float) -> float:
    """low 이하는 0, high 이상은 1, 사이는 선형으로 매핑한다."""
    if high <= low:
        return 1.0
    return float(min(1.0, max(0.0, (value - low) / (high - low))))


def _band(value: float, low: float, high: float, margin: float = 40.0) -> float:
    """[low, high] 구간은 1, 구간 밖은 margin 만큼 벗어날 때까지 선형 감소한다."""
    if low <= value <= high:
        return 1.0
    if value < low:
        return float(max(0.0, 1.0 - (low - value) / margin))
    return float(max(0.0, 1.0 - (value - high) / margin))


def evaluate(plate_bgr: np.ndarray, plate_px_width: int | None = None) -> dict:
    """번호판 crop 의 품질을 0~1 점수로 평가한다.

    가중치 설계 근거
      - 선명도 0.40: 블러는 복구가 불가능한 손실이다. 가장 큰 비중을 둔다.
      - 픽셀폭 0.25: 해상도가 모자라면 전처리로 만들 수 없는 정보다.
      - 밝기   0.20: 역광/과노출은 CLAHE 로 상당 부분 복구되므로 비중이 낮다.
      - 대비   0.15: 밝기와 상관이 높아 중복 가중을 피하려고 낮게 둔다.

    readable 판정은 점수 한 개가 아니라 '치명적 결격 사유' 를 따로 본다.
    총점이 0.6 인데 픽셀폭이 60px 이면 읽을 수 없으므로, 개별 하한 미달은
    총점과 무관하게 readable=False 로 떨어뜨린다.
    """
    gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY) \
        if plate_bgr.ndim == 3 else plate_bgr
    if plate_px_width is None:
        plate_px_width = int(plate_bgr.shape[1])

    # 지표 비교 가능성을 위해 측정은 고정 높이 32px 로 정규화한 사본에서 한다.
    scale = 32.0 / max(gray.shape[0], 1)
    norm = cv2.resize(gray, (max(8, int(gray.shape[1] * scale)), 32),
                      interpolation=cv2.INTER_AREA)

    blur = blur_metric(norm)
    bright = brightness_metric(gray)
    contrast = contrast_metric(gray)
    saturated = saturated_ratio(gray)

    s_blur = _ramp(blur, BLUR_MIN * 0.3, BLUR_GOOD)
    s_width = _ramp(float(plate_px_width), WIDTH_MIN * 0.6, float(WIDTH_GOOD))
    s_bright = _band(bright, BRIGHT_LOW, BRIGHT_HIGH)
    s_contrast = _ramp(contrast, 18.0, 60.0)

    score = (0.40 * s_blur + 0.25 * s_width
             + 0.20 * s_bright + 0.15 * s_contrast)
    # 포화 픽셀이 25% 를 넘으면 반사광으로 글자가 지워진 상태다. 총점을 깎는다.
    if saturated > 0.25:
        score *= max(0.3, 1.0 - (saturated - 0.25) * 2.0)

    reason: str | None = None
    if blur < BLUR_MIN:
        reason = f"흐림(라플라시안 분산 {blur:.1f} < {BLUR_MIN})"
    elif plate_px_width < WIDTH_MIN:
        reason = f"번호판 폭 부족({plate_px_width}px < {WIDTH_MIN}px)"
    elif bright < BRIGHT_LOW - 25:
        reason = f"과도한 저조도(평균 밝기 {bright:.1f})"
    elif bright > BRIGHT_HIGH + 35:
        reason = f"과노출(평균 밝기 {bright:.1f})"
    elif saturated > 0.35:
        reason = f"반사광 포화({saturated * 100:.0f}% 픽셀)"
    elif score < 0.45:
        reason = f"종합 품질 미달(점수 {score:.2f})"

    return {
        "score": round(float(min(1.0, max(0.0, score))), 4),
        "blur": round(blur, 1),
        "brightness": round(bright, 1),
        "contrast": round(contrast, 1),
        "saturated_ratio": round(saturated, 4),
        "plate_px_width": int(plate_px_width),
        "readable": reason is None,
        "reason": reason,
        "parts": {"blur": round(s_blur, 3), "width": round(s_width, 3),
                  "bright": round(s_bright, 3), "contrast": round(s_contrast, 3)},
    }


def make_case(name: str) -> np.ndarray:
    """품질 시나리오별 테스트 이미지를 만든다.

    모든 케이스에 약한 블러(sigma 0.7)와 약한 노이즈(sigma 4)를 공통으로 넣는다.
    합성 이미지는 글자 경계가 수학적으로 완벽해 실제 카메라 영상보다 라플라시안
    분산이 수십 배 높게 나온다. 그 상태로 임계값을 정하면 실제 이미지에서
    전부 '흐림' 판정이 난다. 기준선을 실제 촬영에 맞춰 둬야 임계값이 쓸모 있다.
    """
    w, h = {"narrow": (72, 18)}.get(name, (240, 54))
    img = np.full((h, w, 3), 205, dtype=np.uint8)
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), (35, 35, 35), 2)
    fs = 0.85 * (h / 54.0)
    cv2.putText(img, "12 3456", (int(w * 0.05), int(h * 0.78)),
                cv2.FONT_HERSHEY_SIMPLEX, fs, (45, 45, 45),
                max(1, int(2 * h / 54)), cv2.LINE_AA)

    sigma_blur, sigma_noise, gain, offset = 0.7, 4.0, 1.0, 0.0
    if name == "blurred":
        sigma_blur = 3.2
    elif name == "dark":
        gain = 0.20
    elif name == "glare":
        gain, offset = 1.35, 60.0
    elif name == "noisy":
        sigma_noise = 34.0

    img = cv2.GaussianBlur(img, (0, 0), sigma_blur)
    img = np.clip(img.astype(np.float32) * gain + offset, 0, 255)
    noise = np.random.default_rng(11).normal(0, sigma_noise, img.shape)
    return np.clip(img + noise, 0, 255).astype(np.uint8)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cases = ("good", "blurred", "dark", "glare", "narrow", "noisy")

    print(f"{'케이스':8s} {'점수':>6s} {'블러':>9s} {'밝기':>7s} "
          f"{'대비':>6s} {'폭':>5s} {'판독':>5s}  사유")
    print("-" * 82)
    for name in cases:
        img = make_case(name)
        cv2.imwrite(str(OUT_DIR / f"{name}.png"), img)
        q = evaluate(img)
        mark = "O" if q["readable"] else "X"
        print(f"{name:8s} {q['score']:6.3f} {q['blur']:9.1f} "
              f"{q['brightness']:7.1f} {q['contrast']:6.1f} "
              f"{q['plate_px_width']:5d} {mark:>5s}  {q['reason'] or '-'}")

    print("\n세부 점수 분해(good 케이스):")
    for key, value in evaluate(make_case("good"))["parts"].items():
        print(f"  {key:9s} = {value}")
    print("\n운영 지침: readable=False 이면 OCR 을 호출하지 않고 추가 프레임을 요청한다.")
    print("           흐린 이미지를 억지로 읽으면 틀린 번호가 자신 있게 나온다.")
    print(f"\n저장 위치: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
