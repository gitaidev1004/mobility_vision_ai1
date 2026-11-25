# -*- coding: utf-8 -*-
"""PaddleOCR 한국어 모바일 모델로 번호판을 읽는다.

PaddleOCR 은 검출(det) + 방향 분류(cls) + 인식(rec) 세 모듈로 구성된다.
번호판은 이미 2장에서 검출해 잘라 두었으므로, 보통 det 를 끄고
rec 만 돌리는 쪽이 빠르고 정확하다. 이 스크립트는 두 경로를 모두 보인다.

PaddleOCR 이 설치되어 있지 않으면 설치 안내를 출력한 뒤 정상 종료한다
(종료 코드 0). 따라서 CI 나 교육용 환경에서도 그대로 실행할 수 있다.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_CH03 = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_CH03, "08_postprocess"))

INSTALL_GUIDE = """\
[안내] PaddleOCR 가 설치되어 있지 않다. 아래 순서로 설치한다.

  (1) PaddlePaddle CPU 버전
      pip install paddlepaddle==2.6.1

  (2) PaddleOCR
      pip install paddleocr==2.8.1

  (3) 첫 실행 시 모델이 자동으로 내려온다(약 20MB).
      사내망이라면 아래 경로에 미리 받아 둔 모델을 넣는다.
      Windows : C:\\Users\\<사용자>\\.paddleocr\\whl\\
      Linux   : ~/.paddleocr/whl/

  (4) 한국어 인식 모델은 lang="korean" 으로 지정한다.
      문자 사전은 korean_dict.txt 이며 한글 음절과 숫자가 모두 들어 있다.

설치하지 않아도 이 스크립트는 정상 종료한다(종료 코드 0).
"""


def load_ocr(lang: str = "korean", det: bool = True):
    """PaddleOCR 인스턴스를 만든다. 실패하면 None 과 이유를 돌려준다."""
    try:
        from paddleocr import PaddleOCR
    except ImportError:
        return None, "paddleocr 모듈이 없음"
    try:
        # use_angle_cls: 180도 뒤집힌 번호판을 바로잡는 방향 분류기.
        # rec_image_shape: 번호판은 가로로 길어 48x320 이 안정적이다.
        ocr = PaddleOCR(
            lang=lang,
            use_angle_cls=True,
            det=det,
            use_gpu=False,                 # CPU 전용 환경
            rec_image_shape="3, 48, 320",
            drop_score=0.3,                # 이보다 낮은 결과는 버린다
            show_log=False,
        )
        return ocr, ""
    except Exception as exc:               # noqa: BLE001
        return None, f"초기화 실패: {exc}"


def flatten(result) -> list[tuple[str, float]]:
    """PaddleOCR 결과를 (문자열, 신뢰도) 목록으로 평탄화한다.

    PaddleOCR 2.x 의 반환 구조는 버전에 따라
      [[ [box, (text, score)], ... ]]  또는  [ [box, (text, score)], ... ]
    로 달라진다. 두 경우를 모두 처리한다.
    """
    out: list[tuple[str, float]] = []
    if not result:
        return out
    pages = result if isinstance(result[0], list) else [result]
    for page in pages:
        if not page:
            continue
        for line in page:
            try:
                text, score = line[1][0], float(line[1][1])
            except (IndexError, TypeError, ValueError):
                continue
            out.append((text, score))
    return out


def merge_lines(items: list[tuple[str, float]]) -> tuple[str, float]:
    """두 줄 번호판처럼 여러 줄로 쪼개진 결과를 하나로 합친다.

    PaddleOCR 은 줄 단위로 결과를 주므로, 구형 두 줄 번호판은
    '서울34' 와 '나5678' 로 나뉘어 나온다. 신뢰도는 길이 가중 평균을 쓴다.
    """
    if not items:
        return "", 0.0
    text = "".join(t for t, _ in items)
    total = sum(len(t) for t, _ in items) or 1
    score = sum(len(t) * s for t, s in items) / total
    return text, score


def postprocess(text: str, score: float):
    """3.8절의 후처리를 적용한다(혼동 보정 + 형식 검증)."""
    try:
        from confusion_fix import correct
        from plate_rules import validate
    except ImportError:
        print("[안내] ch03/08_postprocess 모듈을 찾지 못해 후처리를 건너뛴다.")
        return text, False
    fixed, ok, _cands = correct(text, None)
    chk = validate(fixed)
    return fixed, chk.ok and ok


def recognize(path: str, lang: str = "korean", det: bool = True) -> int:
    """이미지 한 장을 읽어 결과를 출력한다. 반환값은 종료 코드."""
    if not os.path.exists(path):
        print(f"[안내] 이미지가 없다: {path}")
        print("       2장에서 잘라 둔 번호판 이미지 경로를 넘긴다.")
        print("       예) python 07_paddle/paddle_infer.py "
              "--image data/plates/12가3456.jpg")
        return 0

    ocr, why = load_ocr(lang, det)
    if ocr is None:
        print(INSTALL_GUIDE)
        print(f"(사유: {why})")
        return 0

    t0 = time.perf_counter()
    result = ocr.ocr(path, det=det, cls=True)
    elapsed = (time.perf_counter() - t0) * 1000.0

    items = flatten(result)
    print(f"입력 이미지   : {path}")
    print(f"추론 시간     : {elapsed:.1f} ms (det={det}, lang={lang})")
    print(f"검출 줄 수    : {len(items)}")
    for i, (t, s) in enumerate(items):
        print(f"  [{i}] '{t}' score={s:.4f}")

    raw, score = merge_lines(items)
    fixed, valid = postprocess(raw, score)
    print(f"\n원문 결합     : {raw} (score={score:.4f})")
    print(f"후처리 결과   : {fixed}")
    print(f"형식 유효     : {valid}")
    if not valid:
        print("형식이 유효하지 않으므로 상위 로직은 "
              "추가 프레임 확보나 재촬영을 요청해야 한다.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="PaddleOCR 한국어 모바일 모델 번호판 추론")
    ap.add_argument("--image", default=os.path.join(
        _CH03, "07_paddle", "sample_plate.jpg"),
        help="번호판 이미지 경로(크롭된 이미지 권장)")
    ap.add_argument("--lang", default="korean", help="인식 언어 코드")
    ap.add_argument("--no-det", action="store_true",
                    help="검출을 끄고 인식만 수행(크롭 이미지일 때 권장)")
    args = ap.parse_args()
    return recognize(args.image, args.lang, det=not args.no_det)


if __name__ == "__main__":
    raise SystemExit(main())
