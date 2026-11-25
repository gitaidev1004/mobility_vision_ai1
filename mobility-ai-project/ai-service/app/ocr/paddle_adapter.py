"""PaddleOCR 선택적 어댑터.

PaddleOCR 을 requirements.txt 에 넣지 않은 이유:
    1. ``paddlepaddle`` 휠은 플랫폼/CPU 명령어셋별로 나뉘어 있고 용량이 크다
       (CPU 버전도 약 100MB). 교재 독자가 ``pip install -r requirements.txt``
       한 번에 실패하는 일을 막아야 한다.
    2. 첫 실행 시 인식/검출 모델을 **네트워크로 다운로드**한다. 오프라인 환경과
       "네트워크 없이 테스트 통과" 요구사항에 정면으로 충돌한다.
    3. 한국어 번호판 전용 모델이 아니라, 한글 일반 OCR 모델을 쓰게 되어
       번호판 특화 CRNN 보다 정확도가 낮은 경우가 많다.

그래서 이 어댑터는 **있으면 쓰고, 없으면 조용히 비활성화**된다.
``recognizer=paddle`` 로 요청이 와도 500 이 아니라 폴백으로 처리되며,
응답의 ``recognizer`` 필드에 ``-fallback`` 접미사가 붙는다.
"""

from __future__ import annotations

from typing import Any

import numpy as np

#: PaddleOCR 인스턴스 캐시. 초기화에 수 초가 걸리므로 프로세스당 1회만 만든다.
_PADDLE_INSTANCE: Any | None = None

#: 초기화 시도 결과 캐시. 실패를 매 요청마다 재시도하면 응답이 수 초씩 느려진다.
#: ``None`` = 아직 시도 안 함, ``str`` = 실패 사유.
_PADDLE_ERROR: str | None = None
_PADDLE_TRIED: bool = False


def is_available() -> bool:
    """PaddleOCR 이 설치되어 있는지 확인한다(초기화는 하지 않음).

    import 가능성만 확인한다. 실제 초기화는 모델 다운로드를 유발할 수 있으므로
    ``get_paddle_ocr`` 가 호출될 때까지 미룬다.

    Returns:
        ``paddleocr`` 패키지를 import 할 수 있으면 True.
    """
    try:
        import importlib.util  # noqa: PLC0415

        return importlib.util.find_spec("paddleocr") is not None
    except (ImportError, ValueError):
        return False


def get_paddle_ocr(*, lang: str = "korean", use_gpu: bool = False) -> Any | None:
    """PaddleOCR 싱글턴을 반환한다(최초 호출 시 초기화).

    Args:
        lang: 인식 언어. 한국 번호판은 ``"korean"``.
        use_gpu: GPU 사용 여부. 교재 환경은 CPU 기본.

    Returns:
        ``PaddleOCR`` 인스턴스. 설치/초기화 실패 시 None(예외를 던지지 않는다).
    """
    global _PADDLE_INSTANCE, _PADDLE_ERROR, _PADDLE_TRIED  # noqa: PLW0603 - 프로세스 캐시

    if _PADDLE_INSTANCE is not None:
        return _PADDLE_INSTANCE
    if _PADDLE_TRIED:
        # 이미 실패했으므로 재시도하지 않는다(지연 누적 방지).
        return None

    _PADDLE_TRIED = True
    try:
        from paddleocr import PaddleOCR  # noqa: PLC0415

        # PaddleOCR 의 생성자 인자는 버전마다 바뀌어 왔다(2.6 → 2.8 → 3.x).
        # 알려진 조합을 순서대로 시도해 최대한 넓은 버전 범위를 지원한다.
        attempts: list[dict[str, Any]] = [
            {"lang": lang, "use_angle_cls": True, "use_gpu": use_gpu, "show_log": False},
            {"lang": lang, "use_angle_cls": True, "use_gpu": use_gpu},
            {"lang": lang, "use_textline_orientation": True},
            {"lang": lang},
        ]
        last_error: Exception | None = None
        for kwargs in attempts:
            try:
                _PADDLE_INSTANCE = PaddleOCR(**kwargs)
                _PADDLE_ERROR = None
                return _PADDLE_INSTANCE
            except TypeError as exc:
                # 인자 조합이 맞지 않음 → 다음 조합 시도.
                last_error = exc
                continue
            except Exception as exc:  # noqa: BLE001 - 모델 다운로드 실패 등
                last_error = exc
                break
        _PADDLE_ERROR = f"초기화 실패: {type(last_error).__name__}: {last_error}"
        return None
    except ImportError:
        _PADDLE_ERROR = "paddleocr 미설치"
        return None
    except Exception as exc:  # noqa: BLE001
        _PADDLE_ERROR = f"예상치 못한 오류: {type(exc).__name__}: {exc}"
        return None


def recognize(plate_bgr: np.ndarray, *, lang: str = "korean") -> dict[str, Any]:
    """PaddleOCR 로 번호판을 판독한다.

    PaddleOCR 반환 형식은 버전마다 다르다:
        - 2.x: ``[[[box, (text, score)], ...]]`` (페이지 → 라인 → [박스, (텍스트, 점수)])
        - 3.x: ``[{"rec_texts": [...], "rec_scores": [...]}]`` (dict 형태)
    두 형식을 모두 파싱한다. 교재 독자가 어떤 버전을 설치하든 동작해야 한다.

    여러 라인이 검출되면 **점수가 가장 높은 라인**을 고른다. 번호판 crop 에는
    보통 한 줄만 있지만, 구형 2단 번호판이나 테두리 잔상이 라인으로 잡힐 수 있다.

    Args:
        plate_bgr: 번호판 영역 BGR crop.
        lang: 인식 언어.

    Returns:
        ``{"text": str, "confidence": float, "available": bool, "reason": str | None,
        "lines": list}``. 사용 불가 시 ``available=False``, ``text=""``.
    """
    engine = get_paddle_ocr(lang=lang)
    if engine is None:
        return {
            "text": "",
            "confidence": 0.0,
            "available": False,
            "reason": _PADDLE_ERROR or "paddleocr 사용 불가",
            "lines": [],
        }

    try:
        arr = np.asarray(plate_bgr)
        # PaddleOCR 2.x 는 ``cls`` 인자를 받고 3.x 는 받지 않는다.
        try:
            raw = engine.ocr(arr, cls=True)
        except TypeError:
            raw = engine.ocr(arr)

        lines = _parse_paddle_output(raw)
        if not lines:
            return {
                "text": "",
                "confidence": 0.0,
                "available": True,
                "reason": "no-text-detected",
                "lines": [],
            }
        best = max(lines, key=lambda item: item[1])
        return {
            "text": str(best[0]),
            "confidence": round(float(best[1]), 4),
            "available": True,
            "reason": None,
            "lines": [{"text": t, "confidence": round(float(s), 4)} for t, s in lines],
        }
    except Exception as exc:  # noqa: BLE001 - 어댑터는 절대 예외를 올리지 않는다
        return {
            "text": "",
            "confidence": 0.0,
            "available": True,
            "reason": f"paddle-ocr-error: {type(exc).__name__}: {exc}",
            "lines": [],
        }


def _parse_paddle_output(raw: Any) -> list[tuple[str, float]]:
    """PaddleOCR 출력을 ``[(텍스트, 점수)]`` 로 정규화한다.

    Args:
        raw: ``engine.ocr()`` 반환값.

    Returns:
        ``(텍스트, 점수)`` 리스트. 파싱 실패 시 빈 리스트.
    """
    out: list[tuple[str, float]] = []
    if not raw:
        return out

    try:
        for page in raw:
            if page is None:
                continue
            # --- 3.x dict 형식 ---
            if isinstance(page, dict):
                texts = page.get("rec_texts") or []
                scores = page.get("rec_scores") or []
                for i, text in enumerate(texts):
                    score = float(scores[i]) if i < len(scores) else 0.0
                    out.append((str(text), score))
                continue
            # --- 2.x 리스트 형식 ---
            for line in page:
                if not line:
                    continue
                # line = [box, (text, score)] 또는 [box, [text, score]]
                if len(line) >= 2 and isinstance(line[1], (tuple, list)) and len(line[1]) >= 2:
                    out.append((str(line[1][0]), float(line[1][1])))
    except (TypeError, ValueError, IndexError, KeyError):
        # 알 수 없는 형식 → 빈 결과. 어댑터가 터지면 안 된다.
        return out
    return out


def info() -> dict[str, Any]:
    """PaddleOCR 어댑터 상태를 반환한다(``/api/v1/models/info`` 용).

    주의: 이 함수는 **초기화를 유발하지 않는다**. 모델 정보 조회 API 가
    PaddleOCR 모델 다운로드를 트리거하면 응답이 수십 초로 늘어난다.

    Returns:
        설치 여부, 초기화 상태, 실패 사유를 담은 dict.
    """
    installed = is_available()
    return {
        "name": "paddleocr",
        "backend": "paddlepaddle",
        "installed": installed,
        "initialized": _PADDLE_INSTANCE is not None,
        "loaded": _PADDLE_INSTANCE is not None,
        "fallback": not installed,
        "weights": None,
        "reason": _PADDLE_ERROR,
        "detail": {
            "optional": True,
            "install_hint": "pip install paddlepaddle==2.6.2 paddleocr==2.8.1",
            "note": "requirements.txt 에서 제외됨(플랫폼별 휠 + 첫 실행 시 모델 다운로드)",
        },
    }


def reset() -> None:
    """캐시를 초기화한다(테스트 전용).

    설치 상태가 바뀐 뒤 재시도하거나, 테스트에서 실패 경로를 검증할 때 쓴다.
    """
    global _PADDLE_INSTANCE, _PADDLE_ERROR, _PADDLE_TRIED  # noqa: PLW0603
    _PADDLE_INSTANCE = None
    _PADDLE_ERROR = None
    _PADDLE_TRIED = False


__all__ = ["get_paddle_ocr", "info", "is_available", "recognize", "reset"]
