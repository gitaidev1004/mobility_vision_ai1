# -*- coding: utf-8 -*-
"""차량번호판 인식 파이프라인 골격.

1.2절의 여덟 단계를 함수로 분리한 의사코드이다. 모델이 없어도 실행되며,
각 단계는 뒤의 장에서 실제 구현으로 교체된다.
실행:  python 03_pipeline/pipeline_skeleton.py
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

# 한국 번호판 형식: 숫자 2~3 + 용도 한글 1 + 숫자 4 (지역명은 선택)
PLATE_RE = re.compile(r"^(?:[가-힣]{2}\s)?\d{2,3}[가-힣]\d{4}$")

# 신뢰도 임계값. 이 값 미달이면 LOW_CONFIDENCE 로 분류한다.
CONF_THRESHOLD = 0.70


@dataclass
class PlateResult:
    """파이프라인의 최종 산출물. AI 서비스 응답 스키마와 대응된다."""

    plate_number: str | None = None
    confidence: float = 0.0
    status: str = "NO_PLATE"          # SUCCESS|LOW_CONFIDENCE|NO_PLATE|FORMAT_INVALID
    valid_format: bool = False
    detections: list[dict] = field(default_factory=list)
    elapsed_ms: dict = field(default_factory=dict)
    quality: dict = field(default_factory=dict)


def step1_load_frame(source: str) -> dict:
    """1단계: 영상 입력. 실제로는 cv2.imread 또는 VideoCapture 를 사용한다."""
    # 여기서는 모델과 파일이 없어도 동작하도록 가상의 프레임 정보만 만든다.
    return {"source": source, "width": 1920, "height": 1080}


def step2_detect_vehicle(frame: dict) -> list[dict]:
    """2단계: 차량 검출(선택). 게이트 환경이면 생략하고 빈 목록을 반환한다."""
    return []


def step3_detect_plate(frame: dict, rois: list[dict]) -> list[dict]:
    """3단계: 번호판 검출. bbox 는 원본 좌표계 [x1, y1, x2, y2] 로 반환한다."""
    # 2장에서 학습한 검출 모델로 교체된다. 지금은 고정 좌표를 돌려준다.
    return [{"bbox": [412, 288, 596, 342], "score": 0.981, "label": "plate"}]


def step4_preprocess(frame: dict, box: dict) -> dict:
    """4단계: 크롭과 보정. 품질 지표를 함께 계산해 반환한다."""
    x1, y1, x2, y2 = box["bbox"]
    px_width = x2 - x1
    quality = {
        "plate_px_width": px_width,
        "blur": 142.3,          # 라플라시안 분산(3장에서 실제 계산)
        "brightness": 118.4,    # 그레이스케일 평균
        "score": 0.86,
        "readable": px_width >= 96,
        "reason": None if px_width >= 96 else "번호판 폭 부족",
    }
    return {"image": None, "quality": quality}


def step5_recognize(prep: dict) -> tuple[str, float, list[float]]:
    """5단계: 문자 인식. CRNN+CTC 모델이 문자 시퀀스와 확률을 돌려준다."""
    # 3장에서 학습한 OCR 모델로 교체된다.
    text = "12가3456"
    char_confs = [0.99, 0.98, 0.95, 0.99, 0.97, 0.98, 0.96]
    return text, min(char_confs), char_confs


def step6_postprocess(text: str) -> tuple[str, bool]:
    """6단계: 후처리. 공백을 정리하고 형식을 검증한다."""
    cleaned = text.strip().replace("  ", " ")
    return cleaned, bool(PLATE_RE.match(cleaned))


def step7_finalize(text: str, conf: float, valid: bool,
                   has_plate: bool, readable: bool) -> str:
    """7단계: 최종 확정. 네 가지 상태 중 하나를 결정한다."""
    if not has_plate or not readable:
        return "NO_PLATE"
    if not valid:
        return "FORMAT_INVALID"
    if conf < CONF_THRESHOLD:
        return "LOW_CONFIDENCE"
    return "SUCCESS"


def step8_save(result: PlateResult) -> dict:
    """8단계: 저장. 실제로는 백엔드가 plate_recognition 행을 생성한다."""
    # 성공 여부와 무관하게 인식 이력을 남기는 것이 원칙이다.
    return {"saved": True, "status": result.status,
            "plate_number": result.plate_number}


def run_pipeline(source: str) -> PlateResult:
    """여덟 단계를 순서대로 실행하고 단계별 소요 시간을 측정한다."""
    t0 = time.perf_counter()
    frame = step1_load_frame(source)
    rois = step2_detect_vehicle(frame)
    boxes = step3_detect_plate(frame, rois)
    t_detect = (time.perf_counter() - t0) * 1000.0

    result = PlateResult(detections=boxes)
    if not boxes:
        result.status = "NO_PLATE"
        result.elapsed_ms = {"detect": round(t_detect, 1), "ocr": 0.0,
                             "total": round(t_detect, 1)}
        step8_save(result)
        return result

    # 신뢰도와 면적을 함께 고려해 대표 박스를 고른다.
    def rank(b: dict) -> float:
        x1, y1, x2, y2 = b["bbox"]
        area = max(1, (x2 - x1) * (y2 - y1))
        return b["score"] * (area ** 0.25)

    best = max(boxes, key=rank)

    t1 = time.perf_counter()
    prep = step4_preprocess(frame, best)
    raw_text, conf, _char_confs = step5_recognize(prep)
    text, valid = step6_postprocess(raw_text)
    t_ocr = (time.perf_counter() - t1) * 1000.0

    status = step7_finalize(text, conf, valid, True,
                            prep["quality"]["readable"])

    result.plate_number = text if status != "NO_PLATE" else None
    result.confidence = round(conf, 3)
    result.status = status
    result.valid_format = valid
    result.quality = prep["quality"]
    result.elapsed_ms = {"detect": round(t_detect, 1), "ocr": round(t_ocr, 1),
                         "total": round(t_detect + t_ocr, 1)}
    step8_save(result)
    return result


if __name__ == "__main__":
    r = run_pipeline("samples/gate_entry_01.jpg")
    print("번호:", r.plate_number)
    print("신뢰도:", r.confidence)
    print("상태:", r.status)
    print("형식검증:", r.valid_format)
    print("품질:", r.quality)
    print("소요시간(ms):", r.elapsed_ms)
