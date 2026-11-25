"""애플리케이션 설정.

``pydantic-settings`` 의 ``BaseSettings`` 를 1순위로 사용한다. 다만 이 서비스는
"의존성이 하나도 없어도 테스트가 돌아야 한다"는 요구사항이 있으므로,
``pydantic_settings`` 가 없으면 ``pydantic.BaseModel`` 로, 그것조차 없으면
순수 ``dataclass`` 로 3단 강등(graceful degradation)한다.

왜 이렇게까지 하는가?
    설정 모듈은 거의 모든 모듈이 import 한다. 여기서 ImportError 가 나면
    "순수 로직 테스트(후처리/CTC 디코더/품질평가)" 까지 전부 무너진다.
    설정은 서비스의 가장 바닥 계층이므로 가장 견고해야 한다.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

# ---------------------------------------------------------------------------
# 1) .env 로딩 — python-dotenv 가 있으면 사용, 없으면 직접 파싱한다.
# ---------------------------------------------------------------------------


def _load_dotenv_if_present(path: str = ".env") -> None:
    """``.env`` 파일을 ``os.environ`` 에 주입한다(이미 존재하는 키는 덮어쓰지 않음).

    python-dotenv 가 설치돼 있으면 그것을 쓰고, 없으면 ``KEY=VALUE`` 한 줄 포맷만
    지원하는 최소 파서를 사용한다. 교재 독자가 dotenv 설치 없이도 실습할 수 있도록
    의존성을 선택사항으로 둔 것이다.

    Args:
        path: .env 파일 경로(상대경로는 프로세스 작업 디렉터리 기준).
    """
    try:
        from dotenv import load_dotenv  # noqa: PLC0415 - 지연 import 의도

        load_dotenv(path, override=False)
        return
    except ImportError:
        pass

    env_file = Path(path)
    if not env_file.is_file():
        return
    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        # 따옴표로 감싼 값과 인라인 주석을 제거한다.
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv_if_present()


# ---------------------------------------------------------------------------
# 2) 설정 필드 기본값 — 어떤 백엔드를 쓰더라도 이 한 벌의 정의를 공유한다.
# ---------------------------------------------------------------------------

#: (필드명, 기본값) 목록. 타입은 기본값의 타입으로부터 유추한다.
_FIELD_DEFAULTS: dict[str, Any] = {
    # 서버
    "app_name": "mobility-ai-service",
    "app_version": "1.0.0",
    "host": "0.0.0.0",  # noqa: S104 - 컨테이너 내부 바인딩이므로 의도적
    "port": 8000,
    "log_level": "INFO",
    "enable_docs": True,
    # CORS — 문자열(쉼표 구분)로 받고 프로퍼티에서 리스트로 변환한다.
    "cors_origins": "http://localhost:5173,http://localhost:8080,http://127.0.0.1:5173",
    # 가중치 경로
    "model_dir": "./models",
    "detector_weights": "./models/plate_yolo_nano.pt",
    "detector_onnx": "./models/plate_yolo_nano.onnx",
    "crnn_weights": "./models/crnn_ctc_kr.pt",
    "attention_weights": "./models/attention_ocr_kr.pt",
    "pdm_anomaly_weights": "./models/pdm_isoforest.joblib",
    "pdm_failure_weights": "./models/pdm_rf.joblib",
    "pdm_rul_weights": "./models/pdm_lstm_rul.pt",
    # 기본 모델 선택
    "default_detector": "yolo",
    "default_recognizer": "crnn",
    "default_min_conf": 0.25,
    # 추론 파라미터
    "device": "cpu",
    "torch_threads": 4,
    "nms_iou": 0.45,
    "detect_img_size": 640,
    "ocr_img_height": 32,
    "ocr_img_width": 160,
    "max_upload_bytes": 12 * 1024 * 1024,
    "max_frames": 12,
    # 품질 임계값
    "quality_blur_min": 60.0,
    "quality_brightness_min": 40.0,
    "quality_brightness_max": 215.0,
    "quality_plate_px_min": 72,
    # 신뢰도
    "low_conf_threshold": 0.55,
    # PdM
    "pdm_model_version": "pdm-ensemble-1.0.0",
}


#: 환경변수 이름 접두사 후보.
#:
#: 왜 두 가지를 모두 지원하는가?
#:   ``.env.example`` 과 계약서는 접두사 없는 이름(``PORT``, ``TORCH_THREADS``)을
#:   쓰지만, 루트의 ``docker-compose.yml`` 과 ``Dockerfile`` 은 다른 서비스의
#:   변수와 섞이지 않도록 ``AI_`` 접두사를 붙인다(``AI_PORT``, ``AI_TORCH_THREADS``).
#:   둘 중 하나만 읽으면 컨테이너 배포에서 설정이 조용히 무시된다
#:   (예: ``AI_TORCH_THREADS=4`` 를 줘도 스레드 수가 기본값으로 동작).
#:   접두사 있는 이름을 **우선**하는 이유: 컨테이너 환경이 더 명시적인 의도다.
ENV_PREFIXES: tuple[str, ...] = ("AI_", "")


def _read_env(field: str) -> str | None:
    """필드 이름에 대응하는 환경변수를 접두사 후보 순서로 찾는다.

    Args:
        field: 소문자 스네이크 케이스 필드명(예 ``"torch_threads"``).

    Returns:
        찾은 환경변수 값. 없으면 None.
    """
    upper = field.upper()
    for prefix in ENV_PREFIXES:
        value = os.environ.get(f"{prefix}{upper}")
        if value is not None:
            return value
    return None


def _coerce(value: Any, default: Any) -> Any:
    """환경변수 문자열을 기본값의 타입으로 변환한다.

    왜 직접 변환하는가? pydantic 이 없는 폴백 경로에서도 ``PORT=9000`` 같은
    문자열이 int 로 들어와야 하기 때문이다.

    Args:
        value: 환경변수에서 읽은 원본 값(보통 str).
        default: 타입 기준이 되는 기본값.

    Returns:
        기본값과 동일한 타입으로 변환된 값. 변환 실패 시 기본값.
    """
    if value is None:
        return default
    if isinstance(default, bool):
        return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}
    try:
        if isinstance(default, int):
            return int(float(str(value).strip()))
        if isinstance(default, float):
            return float(str(value).strip())
    except (TypeError, ValueError):
        return default
    return str(value)


class _SettingsMixin:
    """설정 백엔드(pydantic-settings / pydantic / dataclass)가 공유하는 파생 프로퍼티."""

    cors_origins: str
    model_dir: str
    device: str
    torch_threads: int

    @property
    def cors_origin_list(self) -> list[str]:
        """CORS 허용 오리진 목록.

        쉼표 구분 문자열을 리스트로 변환한다. ``*`` 단독이면 전체 허용으로 본다.
        """
        raw = (self.cors_origins or "").strip()
        if raw == "*":
            return ["*"]
        return [item.strip() for item in raw.split(",") if item.strip()]

    @property
    def model_root(self) -> Path:
        """가중치 디렉터리 ``Path``. 존재하지 않아도 예외를 던지지 않는다."""
        return Path(self.model_dir).expanduser()

    @property
    def resolved_device(self) -> Literal["cpu", "cuda"]:
        """실제 사용 가능한 디바이스.

        ``DEVICE=cuda`` 로 설정했더라도 torch 가 없거나 GPU 가 없으면 ``cpu`` 로
        강등한다. 교재 독자 대부분이 CPU 환경이므로 이 강등은 필수다.
        """
        if str(self.device).lower() != "cuda":
            return "cpu"
        try:
            import torch  # noqa: PLC0415 - 지연 import 의도

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            # torch 미설치 또는 CUDA 초기화 실패 → 조용히 cpu.
            return "cpu"

    @property
    def effective_threads(self) -> int:
        """추론에 쓸 스레드 수. 0 이하이면 CPU 코어 수로 자동 결정한다."""
        if self.torch_threads and self.torch_threads > 0:
            return int(self.torch_threads)
        return max(1, os.cpu_count() or 1)

    def weight_exists(self, attr: str) -> bool:
        """지정한 가중치 설정 필드가 실제 파일로 존재하는지 확인한다.

        Args:
            attr: ``detector_weights`` 처럼 이 설정 객체의 필드 이름.

        Returns:
            파일이 존재하고 크기가 0 보다 크면 True.
        """
        raw = getattr(self, attr, None)
        if not raw:
            return False
        path = Path(str(raw)).expanduser()
        try:
            return path.is_file() and path.stat().st_size > 0
        except OSError:
            return False


# ---------------------------------------------------------------------------
# 3) 3단 강등 구현
# ---------------------------------------------------------------------------

_BACKEND: str

try:  # 1순위: pydantic-settings
    from pydantic_settings import BaseSettings, SettingsConfigDict

    class Settings(_SettingsMixin, BaseSettings):  # type: ignore[misc]
        """환경변수/``.env`` 기반 애플리케이션 설정 (pydantic-settings 백엔드).

        필드명은 소문자 스네이크 케이스이며, 환경변수는 대문자(``PORT``, ``MODEL_DIR``)로
        매칭된다(``case_sensitive=False``).

        ``AI_`` 접두사 지원: 컨테이너 배포는 ``AI_PORT`` 처럼 접두사를 쓴다.
        pydantic-settings 의 ``env_prefix`` 는 **하나만** 지정할 수 있어
        두 형식을 동시에 받을 수 없으므로, ``__init__`` 에서 접두사 있는
        변수를 먼저 읽어 명시적 기본값으로 주입한다
        (``ENV_PREFIXES`` 주석에 그 필요성이 설명되어 있다).
        """

        model_config = SettingsConfigDict(
            env_file=".env",
            env_file_encoding="utf-8",
            case_sensitive=False,
            extra="ignore",
        )

        def __init__(self, **overrides: Any) -> None:
            """``AI_`` 접두사 환경변수를 먼저 반영한 뒤 초기화한다.

            Args:
                **overrides: 호출자가 명시한 값(최우선).
            """
            merged: dict[str, Any] = {}
            for name in _FIELD_DEFAULTS:
                if name in overrides:
                    continue
                # 접두사 없는 이름은 pydantic-settings 가 알아서 읽으므로,
                # 접두사 있는 변수만 명시적으로 주입한다.
                prefixed = os.environ.get(f"AI_{name.upper()}")
                if prefixed is not None:
                    merged[name] = prefixed
            merged.update(overrides)
            super().__init__(**merged)

        app_name: str = _FIELD_DEFAULTS["app_name"]
        app_version: str = _FIELD_DEFAULTS["app_version"]
        host: str = _FIELD_DEFAULTS["host"]
        port: int = _FIELD_DEFAULTS["port"]
        log_level: str = _FIELD_DEFAULTS["log_level"]
        enable_docs: bool = _FIELD_DEFAULTS["enable_docs"]
        cors_origins: str = _FIELD_DEFAULTS["cors_origins"]
        model_dir: str = _FIELD_DEFAULTS["model_dir"]
        detector_weights: str = _FIELD_DEFAULTS["detector_weights"]
        detector_onnx: str = _FIELD_DEFAULTS["detector_onnx"]
        crnn_weights: str = _FIELD_DEFAULTS["crnn_weights"]
        attention_weights: str = _FIELD_DEFAULTS["attention_weights"]
        pdm_anomaly_weights: str = _FIELD_DEFAULTS["pdm_anomaly_weights"]
        pdm_failure_weights: str = _FIELD_DEFAULTS["pdm_failure_weights"]
        pdm_rul_weights: str = _FIELD_DEFAULTS["pdm_rul_weights"]
        default_detector: str = _FIELD_DEFAULTS["default_detector"]
        default_recognizer: str = _FIELD_DEFAULTS["default_recognizer"]
        default_min_conf: float = _FIELD_DEFAULTS["default_min_conf"]
        device: str = _FIELD_DEFAULTS["device"]
        torch_threads: int = _FIELD_DEFAULTS["torch_threads"]
        nms_iou: float = _FIELD_DEFAULTS["nms_iou"]
        detect_img_size: int = _FIELD_DEFAULTS["detect_img_size"]
        ocr_img_height: int = _FIELD_DEFAULTS["ocr_img_height"]
        ocr_img_width: int = _FIELD_DEFAULTS["ocr_img_width"]
        max_upload_bytes: int = _FIELD_DEFAULTS["max_upload_bytes"]
        max_frames: int = _FIELD_DEFAULTS["max_frames"]
        quality_blur_min: float = _FIELD_DEFAULTS["quality_blur_min"]
        quality_brightness_min: float = _FIELD_DEFAULTS["quality_brightness_min"]
        quality_brightness_max: float = _FIELD_DEFAULTS["quality_brightness_max"]
        quality_plate_px_min: int = _FIELD_DEFAULTS["quality_plate_px_min"]
        low_conf_threshold: float = _FIELD_DEFAULTS["low_conf_threshold"]
        pdm_model_version: str = _FIELD_DEFAULTS["pdm_model_version"]

    _BACKEND = "pydantic-settings"

except ImportError:  # 2·3순위: pydantic BaseModel 또는 순수 객체
    try:
        from pydantic import BaseModel, create_model

        _BACKEND = "pydantic"
    except ImportError:
        BaseModel = object  # type: ignore[assignment,misc]
        create_model = None  # type: ignore[assignment]
        _BACKEND = "plain"

    class _BehaviourMixin(_SettingsMixin):
        """env 읽기 동작(생성자)을 담은 믹스인.

        ``pydantic_settings`` 가 제공하는 자동 env 바인딩이 없으므로
        ``os.environ`` 을 직접 읽어 타입 변환한다. 동작/필드는 1순위 백엔드와
        완전히 동일하다.
        """

        def __init__(self, **overrides: Any) -> None:
            """환경변수와 명시 인자를 병합해 모든 필드를 설정한다.

            Args:
                **overrides: 호출자가 명시한 값(환경변수보다 우선).
            """
            values: dict[str, Any] = {}
            for name, default in _FIELD_DEFAULTS.items():
                if name in overrides:
                    values[name] = _coerce(overrides[name], default)
                else:
                    # ``AI_`` 접두사를 우선 확인한다(ENV_PREFIXES 주석 참조).
                    values[name] = _coerce(_read_env(name), default)
            # pydantic 경로에서는 BaseModel.__init__ 이 검증을 수행한다.
            # plain 경로에서는 object.__init__ 이 인자를 받지 않으므로 직접 설정한다.
            if _BACKEND == "pydantic":
                super().__init__(**values)  # type: ignore[call-arg]
            else:
                for key, val in values.items():
                    object.__setattr__(self, key, val)

    if _BACKEND == "pydantic" and create_model is not None:

        class _SettingsBehaviour(_BehaviourMixin, BaseModel):  # type: ignore[misc,valid-type]
            """``create_model`` 의 베이스로 쓸 BaseModel 서브클래스.

            ``create_model(__base__=...)`` 은 베이스가 **BaseModel 서브클래스**
            여야 한다. 순수 믹스인을 넘기면 메타클래스가 키워드 인자를 받지 못해
            ``TypeError: __init_subclass__() takes no keyword arguments`` 가 난다.
            """

            model_config = {"extra": "ignore", "protected_namespaces": ()}

        # 왜 ``create_model`` 인가?
        #   클래스 본문 뒤에 ``__annotations__`` 를 주입하고 ``model_rebuild`` 를
        #   호출하는 방식은 **동작하지 않는다**. pydantic v2 는 클래스 생성 시점에
        #   메타클래스가 필드를 수집하므로, 나중에 추가한 애너테이션은 무시된다.
        #   실측: 32개 필드 중 믹스인에 선언된 4개만 등록되고, ``Settings(port=1234)``
        #   조차 기본값 8000 을 반환했다(환경변수도 전부 무시).
        #   ``create_model`` 은 필드 정의를 받아 **클래스를 처음부터** 만들므로
        #   이 문제가 없다.
        Settings = create_model(  # type: ignore[misc,assignment,call-overload]
            "Settings",
            __base__=_SettingsBehaviour,
            __module__=__name__,
            **{
                name: (type(default), default) for name, default in _FIELD_DEFAULTS.items()
            },
        )
        Settings.__doc__ = (
            "환경변수 기반 애플리케이션 설정 (pydantic BaseModel 폴백 백엔드).\n\n"
            "``pydantic_settings`` 가 없을 때 쓰이며, 필드와 동작은 1순위 백엔드와 동일하다."
        )
    else:

        class Settings(_BehaviourMixin):  # type: ignore[misc,no-redef]
            """환경변수 기반 애플리케이션 설정 (의존성 없는 순수 파이썬 백엔드).

            pydantic 조차 없는 환경에서 쓰인다. 타입 검증은 ``_coerce`` 가
            수행하며, 필드와 동작은 1순위 백엔드와 동일하다.
            """


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """프로세스 전역 싱글턴 설정 객체를 반환한다.

    ``lru_cache`` 로 캐싱하므로 .env 파싱과 타입 변환은 1회만 수행된다.
    테스트에서 값을 바꾸려면 ``get_settings.cache_clear()`` 를 호출하면 된다.

    Returns:
        Settings: 설정 인스턴스.
    """
    return Settings()


def settings_backend() -> str:
    """현재 사용 중인 설정 백엔드 이름을 반환한다(진단/헬스체크용).

    Returns:
        ``"pydantic-settings"`` | ``"pydantic"`` | ``"plain"``
    """
    return _BACKEND


__all__ = ["ENV_PREFIXES", "Settings", "get_settings", "settings_backend"]
