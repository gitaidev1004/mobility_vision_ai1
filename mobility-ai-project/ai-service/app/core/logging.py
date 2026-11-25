"""로깅 설정.

운영에서 로그를 수집기(Loki/CloudWatch 등)로 보낼 때 파싱이 쉬워야 하므로
``LOG_FORMAT=json`` 이면 한 줄 JSON 으로, 기본값에서는 사람이 읽기 좋은
텍스트로 출력한다.

요청 ID 를 ``contextvars`` 로 전파하는 이유:
    FastAPI 는 하나의 프로세스에서 여러 요청을 비동기로 처리한다. 전역 변수에
    요청 ID 를 담으면 요청끼리 값이 섞인다. ``ContextVar`` 는 비동기 태스크마다
    독립적인 값을 가지므로 안전하다.
"""

from __future__ import annotations

import json
import logging
import sys
import uuid
from contextvars import ContextVar
from typing import Any

#: 현재 처리 중인 요청의 상관관계 ID. 미들웨어가 설정하고 포매터가 읽는다.
_request_id: ContextVar[str] = ContextVar("request_id", default="-")

_CONFIGURED = False


def new_request_id() -> str:
    """새 요청 ID(짧은 UUID 앞 12자)를 생성한다.

    전체 UUID 는 로그 한 줄을 길게 만들어 가독성을 떨어뜨리므로 12자만 쓴다.
    12 hex 문자는 약 2^48 공간이라 단일 서비스의 상관관계 추적에는 충분하다.

    Returns:
        예: ``"3f9a12bc77de"``
    """
    return uuid.uuid4().hex[:12]


def set_request_id(request_id: str | None = None) -> str:
    """현재 컨텍스트의 요청 ID 를 설정한다.

    Args:
        request_id: 외부(게이트웨이/백엔드)에서 전달된 ID. None 이면 새로 생성한다.

    Returns:
        설정된 요청 ID.
    """
    rid = request_id or new_request_id()
    _request_id.set(rid)
    return rid


def get_request_id() -> str:
    """현재 컨텍스트의 요청 ID 를 반환한다(없으면 ``"-"``)."""
    return _request_id.get()


class RequestIdFilter(logging.Filter):
    """모든 LogRecord 에 ``request_id`` 속성을 주입하는 필터.

    포매터에서 ``%(request_id)s`` 를 쓰려면 레코드에 해당 속성이 반드시 있어야 한다.
    속성이 없는 레코드가 하나라도 있으면 포매팅이 깨지므로 필터로 일괄 주입한다.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """레코드에 요청 ID 를 붙이고 항상 통과시킨다."""
        if not hasattr(record, "request_id"):
            record.request_id = get_request_id()
        return True


class JsonFormatter(logging.Formatter):
    """한 줄 JSON 포매터.

    ``extra={"stage": "detect", "ms": 12.3}`` 처럼 넘긴 임의 필드도 함께 직렬화하여
    지연 시간/모델명 같은 운영 지표를 로그 검색으로 집계할 수 있게 한다.
    """

    #: LogRecord 의 표준 속성(= extra 가 아닌 것). 직렬화에서 제외한다.
    _RESERVED = frozenset(
        {
            "args", "asctime", "created", "exc_info", "exc_text", "filename",
            "funcName", "levelname", "levelno", "lineno", "module", "msecs",
            "message", "msg", "name", "pathname", "process", "processName",
            "relativeCreated", "stack_info", "thread", "threadName", "taskName",
        }
    )

    def format(self, record: logging.LogRecord) -> str:
        """LogRecord 를 JSON 문자열로 변환한다."""
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "request_id": getattr(record, "request_id", "-"),
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key in self._RESERVED or key == "request_id":
                continue
            # 직렬화 불가능한 객체는 문자열로 떨어뜨려 로깅 자체가 실패하지 않게 한다.
            try:
                json.dumps(value)
                payload[key] = value
            except (TypeError, ValueError):
                payload[key] = repr(value)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(level: str = "INFO", json_output: bool | None = None) -> None:
    """루트 로거를 1회만 설정한다(멱등).

    uvicorn 이 자체 핸들러를 추가하기 때문에, 중복 설정하면 같은 줄이 두 번 찍힌다.
    모듈 전역 플래그로 재진입을 막는다.

    **로그는 stderr 로 보낸다.** 이유: CLI 스크립트(``benchmark``, ``synth_plate``)가
    결과 JSON 을 stdout 으로 출력하므로, 로그가 stdout 에 섞이면 파이프로
    받은 JSON 파싱이 깨진다(실측: ``JSONDecodeError: Extra data``).
    "데이터는 stdout, 진단은 stderr" 는 유닉스 CLI 의 기본 규약이다.
    컨테이너 로그 수집기는 두 스트림을 모두 수집하므로 운영에도 영향이 없다.

    Args:
        level: 로그 레벨 문자열(``"DEBUG"``/``"INFO"``/...).
        json_output: True 면 JSON, False 면 텍스트. None 이면 환경변수
            ``LOG_FORMAT`` 이 ``json`` 인지로 판단한다.
    """
    global _CONFIGURED  # noqa: PLW0603 - 멱등 보장을 위한 모듈 플래그
    if _CONFIGURED:
        return

    import os

    use_json = json_output
    if use_json is None:
        use_json = os.environ.get("LOG_FORMAT", "").strip().lower() == "json"

    handler = logging.StreamHandler(stream=sys.stderr)
    handler.addFilter(RequestIdFilter())
    if use_json:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s %(levelname)-7s [%(request_id)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    try:
        root.setLevel(getattr(logging, str(level).upper()))
    except AttributeError:
        root.setLevel(logging.INFO)

    # 접근 로그는 미들웨어에서 따로 찍으므로 uvicorn 기본 access 로그는 낮춘다.
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """모듈용 로거를 반환한다.

    Args:
        name: 보통 ``__name__``.

    Returns:
        logging.Logger
    """
    return logging.getLogger(name)


__all__ = [
    "JsonFormatter",
    "RequestIdFilter",
    "configure_logging",
    "get_logger",
    "get_request_id",
    "new_request_id",
    "set_request_id",
]
