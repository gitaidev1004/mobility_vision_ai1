# -*- coding: utf-8 -*-
"""전처리 기법을 단계별로 적용하고 결과를 비교 저장한다.

Grayscale -> CLAHE -> Thresholding -> Denoise -> Sharpen 순서로 진행하며
각 단계의 결과 이미지와 정량 지표(선명도/대비)를 함께 출력한다.

실행:  python 02_preprocess/enhance.py
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

OUT_DIR = Path("outputs/enhance")


def make_hard_sample(width: int = 320, height: int = 72) -> np.ndarray:
    """저대비·저조도·노이즈가 섞인 '어려운' 번호판 이미지를 합성한다."""
    img = np.full((height, width, 3), 150, dtype=np.uint8)
    cv2.putText(img, "12 3456", (16, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (95, 95, 95), 3, cv2.LINE_AA)
    # 좌우 밝기 기울기(한쪽만 조명을 받은 상황)
    gradient = np.linspace(0.55, 1.1, width, dtype=np.float32)[None, :, None]
    img = np.clip(img.astype(np.float32) * gradient, 0, 255)
    # 가우시안 노이즈 + 약한 블러(야간 촬영 특성)
    noise = np.random.default_rng(3).normal(0, 14, img.shape)
    img = np.clip(img + noise, 0, 255).astype(np.uint8)
    return cv2.GaussianBlur(img, (3, 3), 0.8)


def to_grayscale(bgr: np.ndarray) -> np.ndarray:
    """회색조 변환.

    번호판 글자는 색이 아니라 밝기 차이로 구분된다. 3채널을 1채널로 줄이면
    연산량이 1/3 로 줄고, 조명 색온도(주차장 나트륨등 등)의 영향도 사라진다.
    cvtColor 는 BT.601 가중치(0.299R+0.587G+0.114B)를 쓴다. 단순 평균보다
    사람 눈의 밝기 인지에 가깝고, 초록 번호판(구형 사업용)의 글자 대비가 더 잘 남는다.
    """
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def apply_clahe(gray: np.ndarray, clip_limit: float = 2.5,
                tile: int = 8) -> np.ndarray:
    """CLAHE(대비 제한 적응 히스토그램 평활화)로 국소 대비를 올린다.

    equalizeHist 는 이미지 전체 히스토그램을 늘리므로, 번호판 절반만 그늘진
    이미지에서는 밝은 쪽이 포화되고 어두운 쪽은 여전히 어둡다. CLAHE 는 타일마다
    따로 평활화하고 clipLimit 로 증폭 한도를 둬 노이즈 폭주를 막는다.
    tileGridSize 가 너무 작으면(2x2 등) 타일 경계가 격자 무늬로 보인다.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))
    return clahe.apply(gray)


def apply_threshold(gray: np.ndarray, mode: str = "adaptive") -> np.ndarray:
    """이진화. 'otsu' 는 전역 임계값, 'adaptive' 는 국소 임계값을 쓴다.

    번호판은 조명 불균일이 흔하므로 adaptive 가 기본값으로 더 안전하다.
    blockSize 는 글자 굵기의 2~3배가 적당하고 반드시 홀수여야 한다.
    """
    if mode == "otsu":
        _, binary = cv2.threshold(gray, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary
    block = max(3, (gray.shape[0] // 3) | 1)     # 홀수 보정
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, block, 5)


def apply_denoise(gray: np.ndarray, mode: str = "bilateral") -> np.ndarray:
    """노이즈 제거. 'median' 은 소금후추 노이즈, 'bilateral' 은 가우시안 노이즈에 쓴다.

    GaussianBlur 를 쓰지 않는 이유: 노이즈와 함께 글자 윤곽도 같이 흐려진다.
    bilateralFilter 는 밝기 차가 큰 경계(글자 테두리)는 보존하고 평탄한 영역만
    평활화하므로 OCR 전처리에 적합하다. 다만 느리므로 작은 번호판 crop 에만 쓴다.
    """
    if mode == "median":
        return cv2.medianBlur(gray, 3)
    return cv2.bilateralFilter(gray, d=5, sigmaColor=45, sigmaSpace=45)


def apply_sharpen(gray: np.ndarray, amount: float = 1.0) -> np.ndarray:
    """언샤프 마스킹으로 글자 윤곽을 살린다.

    원본에서 흐린 사본을 빼면 고주파(윤곽) 성분만 남는다. 이를 원본에 더한다.
    amount 를 1.5 이상으로 올리면 글자 주변에 흰 테가 생기고(halo) 오히려
    이진화 결과가 나빠진다. 1.0 전후를 기본으로 두고 데이터로 조정한다.
    """
    blurred = cv2.GaussianBlur(gray, (0, 0), 1.6)
    return cv2.addWeighted(gray, 1.0 + amount, blurred, -amount, 0)


def sharpness(gray: np.ndarray) -> float:
    """라플라시안 분산 — 값이 클수록 선명하다."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def contrast(gray: np.ndarray) -> float:
    """표준편차 기반 대비 지표."""
    return float(gray.std())


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bgr = make_hard_sample()
    cv2.imwrite(str(OUT_DIR / "00_input.png"), bgr)

    stages: list[tuple[str, np.ndarray]] = []
    gray = to_grayscale(bgr)
    stages.append(("1_gray", gray))
    stages.append(("2_clahe", apply_clahe(gray)))
    stages.append(("3_denoise", apply_denoise(stages[-1][1])))
    stages.append(("4_sharpen", apply_sharpen(stages[-1][1])))
    stages.append(("5_thresh_adaptive", apply_threshold(stages[-1][1], "adaptive")))
    stages.append(("5_thresh_otsu", apply_threshold(stages[3][1], "otsu")))

    print(f"{'단계':20s} {'선명도(Lap.Var)':>16s} {'대비(Std)':>12s}")
    print("-" * 52)
    for name, img in stages:
        cv2.imwrite(str(OUT_DIR / f"{name}.png"), img)
        print(f"{name:20s} {sharpness(img):16.1f} {contrast(img):12.2f}")

    # 가로로 이어 붙인 비교 이미지도 저장한다(교재 그림용).
    strip = np.hstack([cv2.resize(img, (bgr.shape[1], bgr.shape[0]))
                       for _, img in stages])
    cv2.imwrite(str(OUT_DIR / "99_compare_strip.png"), strip)

    print("\n해석: CLAHE 는 대비를 크게 올리지만 노이즈도 함께 증폭한다.")
    print("      그래서 CLAHE 다음에 노이즈 제거를 두고, 그 뒤에 선명화를 둔다.")
    print("      순서를 뒤집으면(선명화 → 노이즈 제거) 살린 윤곽을 다시 지운다.")
    print(f"\n저장 위치: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
