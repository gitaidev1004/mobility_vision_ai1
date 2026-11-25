"""pydantic v2 요청/응답 모델 — 계약서 3장 JSON 과 1:1 대응.

왜 스키마를 명시적으로 정의하는가?
    1. **계약 검증**: 백엔드(Spring Boot)가 기대하는 필드명/타입을 코드로 고정한다.
       필드 하나가 달라지면 pydantic 이 즉시 알려 준다 — 런타임에 null 이
       흘러들어가 프론트엔드에서 터지는 것보다 훨씬 낫다.
    2. **자동 문서화**: ``/docs`` (OpenAPI)가 그대로 API 명세서가 된다.
    3. **입력 방어**: ``min_conf=-5`` 같은 값을 엔드포인트 진입 전에 거부한다.

pydantic 이 없으면?
    이 모듈은 pydantic 이 없을 때 **경량 대체 클래스**로 강등된다.
    FastAPI 자체가 pydantic 을 요구하므로 실제 서비스에서는 항상 pydantic 이
    있지만, 순수 로직 테스트(후처리/CTC/품질)가 이 모듈 import 때문에
    실패하는 것은 막아야 한다.
"""

from __future__ import annotations

from typing import Any, Literal

try:
    from pydantic import BaseModel, ConfigDict, Field

    _HAS_PYDANTIC = True
except ImportError:  # pragma: no cover - FastAPI 환경에서는 항상 존재
    _HAS_PYDANTIC = False

    class BaseModel:  # type: ignore[no-redef]
        """pydantic 미설치 시 쓰는 최소 대체 클래스.

        타입 검증은 하지 않지만, 속성 접근과 ``model_dump`` 를 제공해
        import 와 간단한 사용이 깨지지 않게 한다.
        """

        def __init__(self, **data: Any) -> None:
            """전달된 키워드를 그대로 속성으로 설정한다."""
            for key, value in data.items():
                setattr(self, key, value)

        def model_dump(self, **_kwargs: Any) -> dict[str, Any]:
            """인스턴스 속성을 dict 로 반환한다."""
            return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    def Field(default: Any = None, **_kwargs: Any) -> Any:  # type: ignore[no-redef] # noqa: N802
        """pydantic ``Field`` 의 무동작 대체(기본값만 반환)."""
        return default

    def ConfigDict(**kwargs: Any) -> dict[str, Any]:  # type: ignore[no-redef] # noqa: N802
        """pydantic ``ConfigDict`` 의 dict 대체."""
        return dict(kwargs)


#: 계약서의 ``plate_recognition.status`` enum(DB 와 동일).
PlateStatus = Literal["SUCCESS", "LOW_CONFIDENCE", "NO_PLATE", "FORMAT_INVALID"]

#: 계약서의 ``pdm_prediction.risk_level`` enum.
RiskLevel = Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]

#: 허용 검출기/인식기 이름(계약서 3장 form field 제약).
DetectorName = Literal["yolo", "mobilenet"]
RecognizerName = Literal["crnn", "attention", "paddle"]


# ===========================================================================
# 공통
# ===========================================================================


class ElapsedMs(BaseModel):
    """단계별 소요 시간(밀리초).

    계약서: ``{"detect": 38.2, "ocr": 19.7, "total": 61.4}``
    """

    model_config = ConfigDict(extra="allow")

    detect: float = Field(0.0, description="검출 소요 시간(ms)")
    ocr: float = Field(0.0, description="OCR 소요 시간(ms)")
    total: float = Field(0.0, description="전체 소요 시간(ms)")


class QualityInfo(BaseModel):
    """영상 품질 평가 결과.

    계약서: ``{"score":0.86,"blur":142.3,"brightness":118.4,
    "plate_px_width":184,"readable":true,"reason":null}``
    """

    model_config = ConfigDict(extra="allow")

    score: float = Field(0.0, ge=0.0, le=1.0, description="0~1 종합 품질 점수")
    blur: float = Field(0.0, description="라플라시안 분산(선명도). 클수록 선명")
    brightness: float = Field(0.0, description="평균 밝기(0~255)")
    plate_px_width: int = Field(0, description="번호판 가로 픽셀폭")
    readable: bool = Field(False, description="판독 가능 여부")
    reason: str | None = Field(None, description="판독 불가 사유(가능하면 null)")


class DetectionBox(BaseModel):
    """검출 박스 하나.

    계약서: ``{"bbox":[412,288,596,342],"score":0.981,"label":"plate"}``
    """

    model_config = ConfigDict(extra="allow")

    bbox: list[int] = Field(default_factory=list, description="[x1, y1, x2, y2] 픽셀 좌표")
    score: float = Field(0.0, ge=0.0, le=1.0, description="검출 신뢰도")
    label: str = Field("plate", description="클래스 이름")


class OcrCandidate(BaseModel):
    """OCR 후보 하나.

    계약서: ``{"text":"12가3456","confidence":0.973}``
    """

    model_config = ConfigDict(extra="allow")

    text: str = Field("", description="인식된 문자열")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="신뢰도")


# ===========================================================================
# 1) GET /health
# ===========================================================================


class HealthResponse(BaseModel):
    """헬스체크 응답.

    계약서: ``{"status":"UP","version":"1.0.0","device":"cpu",
    "detector":"yolo-nano-plate","recognizer":"crnn-ctc-kr","torch":"2.x","threads":4}``
    """

    model_config = ConfigDict(extra="allow")

    status: str = Field("UP", description="서비스 상태")
    version: str = Field("1.0.0", description="서비스 버전")
    device: str = Field("cpu", description="추론 디바이스")
    detector: str = Field("", description="현재 검출기 이름(폴백이면 -fallback)")
    recognizer: str = Field("", description="현재 인식기 이름")
    torch: str = Field("not-installed", description="torch 버전(미설치 시 not-installed)")
    threads: int = Field(1, description="추론 스레드 수")


# ===========================================================================
# 2) POST /api/v1/lpr/recognize
# ===========================================================================


class LprResponse(BaseModel):
    """단일 이미지 번호판 인식 응답(계약서 3장).

    ``plate_number`` 가 None 이면 ``status`` 는 ``NO_PLATE`` 다.
    """

    model_config = ConfigDict(extra="allow")

    plate_number: str | None = Field(None, description="인식된 번호판(미검출 시 null)")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="0~1 종합 신뢰도")
    status: PlateStatus = Field("NO_PLATE", description="인식 상태(DB enum 과 동일)")
    valid_format: bool = Field(False, description="한국 번호판 정규식 통과 여부")
    detector: str = Field("", description="사용한 검출기(폴백이면 -fallback)")
    recognizer: str = Field("", description="사용한 인식기(폴백이면 -fallback)")
    elapsed_ms: ElapsedMs = Field(default_factory=ElapsedMs, description="단계별 소요 시간")
    quality: QualityInfo = Field(default_factory=QualityInfo, description="영상 품질")
    detections: list[DetectionBox] = Field(default_factory=list, description="검출 박스 목록")
    candidates: list[OcrCandidate] = Field(default_factory=list, description="OCR 후보 목록")
    char_confidences: list[float] = Field(default_factory=list, description="문자별 신뢰도")


class PerFrameResult(BaseModel):
    """다중 프레임 응답의 프레임별 요약."""

    model_config = ConfigDict(extra="allow")

    plate_number: str | None = Field(None, description="해당 프레임 인식 결과")
    confidence: float = Field(0.0, description="해당 프레임 신뢰도")
    status: PlateStatus = Field("NO_PLATE", description="해당 프레임 상태")
    valid_format: bool = Field(False, description="형식 유효성")
    quality_score: float = Field(0.0, description="해당 프레임 품질 점수")
    elapsed_ms: ElapsedMs = Field(default_factory=ElapsedMs, description="소요 시간")


class LprFramesResponse(LprResponse):
    """다중 프레임 인식 응답(계약서: 단일 스키마 + frames/fusion/per_frame)."""

    frames: int = Field(0, description="처리한 프레임 수")
    fusion: str = Field("majority-vote", description="융합 전략")
    per_frame: list[PerFrameResult] = Field(default_factory=list, description="프레임별 결과 요약")


# ===========================================================================
# 3) POST /api/v1/pdm/predict
# ===========================================================================


class SensorReading(BaseModel):
    """센서 판독 한 건(계약서 ``readings[]``, DB ``sensor_reading`` 과 대응).

    모든 센서 필드가 Optional 인 이유: 차량/센서 구성에 따라 일부 값이 없을 수
    있다. 결측은 ``app.pdm.features`` 가 전방 채움으로 보간한다 —
    "필드 하나가 없으면 400" 이라고 하면 실제 운영 데이터를 받을 수 없다.
    """

    model_config = ConfigDict(extra="allow")

    measured_at: str | None = Field(None, description="측정 시각(ISO 8601)")
    odometer_km: float | None = Field(None, description="주행거리(km)")
    engine_temp_c: float | None = Field(None, description="냉각수 온도(°C)")
    engine_rpm: float | None = Field(None, description="엔진 회전수(rpm)")
    speed_kph: float | None = Field(None, description="속도(km/h)")
    battery_voltage: float | None = Field(None, description="배터리 전압(V)")
    battery_soh: float | None = Field(None, description="배터리 건강도(0~1)")
    tire_pressure_fl: float | None = Field(None, description="좌전 타이어 공기압(psi)")
    tire_pressure_fr: float | None = Field(None, description="우전 타이어 공기압(psi)")
    tire_pressure_rl: float | None = Field(None, description="좌후 타이어 공기압(psi)")
    tire_pressure_rr: float | None = Field(None, description="우후 타이어 공기압(psi)")
    brake_pad_mm: float | None = Field(None, description="브레이크 패드 두께(mm)")
    oil_quality: float | None = Field(None, description="엔진오일 상태(0~1)")
    vibration_rms: float | None = Field(None, description="진동 RMS(g)")
    fuel_level: float | None = Field(None, description="연료 잔량(0~1)")


class PdmRequest(BaseModel):
    """예지보전 예측 요청(계약서 3장).

    계약서 예시::

        {"vehicle_id": 1, "odometer_km": 84210, "readings": [{...}]}
    """

    model_config = ConfigDict(extra="allow")

    vehicle_id: int = Field(..., ge=0, description="차량 ID")
    odometer_km: float | None = Field(None, ge=0, description="현재 주행거리(km)")
    readings: list[SensorReading] = Field(
        default_factory=list, description="센서 시계열(시간 오름차순 권장, 아니어도 내부 정렬)"
    )
    daily_km: float | None = Field(
        None, gt=0, description="일 평균 주행거리(km). 미지정 시 45km 가정"
    )


class ComponentDiagnosisOut(BaseModel):
    """부품 진단 결과(계약서 ``components[]``).

    계약서: ``{"name":"engine","score":81.2,"status":"NORMAL","reason":"냉각수 온도 정상 범위"}``
    """

    model_config = ConfigDict(extra="allow")

    name: str = Field("", description="부품명(engine|battery|brake|tire|oil|transmission)")
    score: float = Field(0.0, ge=0.0, le=100.0, description="0~100 건강 점수")
    status: str = Field("NORMAL", description="NORMAL|WATCH|WARNING|CRITICAL")
    reason: str = Field("", description="한국어 판단 근거")
    risk: float = Field(0.0, ge=0.0, le=1.0, description="0~1 고장 위험도")


class RecommendationOut(BaseModel):
    """권장 조치(계약서 ``recommendations[]``).

    계약서: ``{"priority":1,"item":"브레이크 패드 점검","due_km":3000,"due_days":30}``
    """

    model_config = ConfigDict(extra="allow")

    priority: int = Field(1, ge=1, description="우선순위(1 = 가장 급함)")
    item: str = Field("", description="조치 내용")
    due_km: int = Field(0, ge=0, description="조치 기한(잔여 주행거리 km)")
    due_days: int = Field(0, ge=0, description="조치 기한(잔여 일수)")
    component: str | None = Field(None, description="대상 부품")
    status: str | None = Field(None, description="대상 부품 상태")
    reason: str | None = Field(None, description="권고 사유")


class PdmResponse(BaseModel):
    """예지보전 예측 응답(계약서 3장)."""

    model_config = ConfigDict(extra="allow")

    vehicle_id: int = Field(0, description="차량 ID")
    health_score: float = Field(0.0, ge=0.0, le=100.0, description="0~100 종합 건강 점수")
    anomaly_score: float = Field(0.0, ge=0.0, le=1.0, description="0~1 이상 점수")
    failure_risk: float = Field(0.0, ge=0.0, le=1.0, description="0~1 고장 위험도")
    risk_level: RiskLevel = Field("LOW", description="위험 등급(DB enum 과 동일)")
    rul_days: int = Field(0, ge=0, description="잔여수명(일)")
    top_component: str = Field("", description="가장 위험한 부품")
    components: list[ComponentDiagnosisOut] = Field(
        default_factory=list, description="부품별 진단"
    )
    recommendations: list[RecommendationOut] = Field(
        default_factory=list, description="권장 조치(우선순위순)"
    )
    model_version: str = Field("", description="모델 버전")
    computed_at: str = Field("", description="계산 시각(ISO 8601)")


# ===========================================================================
# 4) GET /api/v1/models/info, POST /api/v1/models/benchmark
# ===========================================================================


class ModelEntry(BaseModel):
    """모델 레지스트리 항목."""

    model_config = ConfigDict(extra="allow")

    name: str = Field("", description="모델 이름")
    task: str = Field("", description="용도(detect|ocr|pdm)")
    description: str = Field("", description="설명")
    registered: bool = Field(True, description="레지스트리 등록 여부")
    instantiated: bool = Field(False, description="인스턴스 생성 여부")
    loaded: bool | None = Field(None, description="가중치 로딩 성공 여부")
    exposed_name: str | None = Field(None, description="응답에 노출되는 이름")
    spec: dict[str, Any] | None = Field(None, description="구조 스펙(로딩 없이 조회)")
    runtime: dict[str, Any] | None = Field(None, description="런타임 정보(로딩 후)")


class ModelsInfoResponse(BaseModel):
    """모델 정보 응답(계약서 ``GET /api/v1/models/info``)."""

    model_config = ConfigDict(extra="allow")

    detectors: list[ModelEntry] = Field(default_factory=list, description="검출기 목록")
    recognizers: list[ModelEntry] = Field(default_factory=list, description="인식기 목록")
    pdm: list[ModelEntry] = Field(default_factory=list, description="예지보전 모델 목록")
    charset: dict[str, Any] = Field(default_factory=dict, description="번호판 문자셋 요약")
    runtime: dict[str, Any] = Field(default_factory=dict, description="런타임 환경 정보")


class BenchmarkEntry(BaseModel):
    """벤치마크 결과 한 건(계약서 ``POST /api/v1/models/benchmark``, DB ``model_benchmark``).

    계약서: ``{"model_name":"YOLO-nano + CRNN/CTC","task":"detect+ocr",
    "detect_recall":0.991,"exact_match_rate":0.962,"latency_p50_ms":61.4,
    "latency_p95_ms":88.1,"memory_mb":214.0,"model_size_mb":9.8,"quantized":false}``
    """

    model_config = ConfigDict(extra="allow")

    model_name: str = Field("", description="모델 조합 이름")
    task: str = Field("", description="작업 종류(detect+ocr 등)")
    detect_recall: float = Field(0.0, ge=0.0, le=1.0, description="검출 재현율")
    exact_match_rate: float = Field(0.0, ge=0.0, le=1.0, description="완전 일치율")
    latency_p50_ms: float = Field(0.0, ge=0.0, description="지연 중앙값(ms)")
    latency_p95_ms: float = Field(0.0, ge=0.0, description="지연 95분위(ms)")
    memory_mb: float = Field(0.0, ge=0.0, description="메모리 사용량(MB)")
    model_size_mb: float = Field(0.0, ge=0.0, description="모델 파일 크기(MB)")
    quantized: bool = Field(False, description="양자화 여부")


class BenchmarkRequest(BaseModel):
    """벤치마크 실행 요청."""

    model_config = ConfigDict(extra="allow")

    samples: int = Field(
        20, ge=1, le=200, description="벤치마크 샘플 수(합성 데이터로 생성)"
    )
    detectors: list[str] | None = Field(None, description="평가할 검출기(미지정 시 전체)")
    recognizers: list[str] | None = Field(None, description="평가할 인식기(미지정 시 전체)")
    seed: int = Field(20261024, description="합성 데이터 시드(재현성)")


# ===========================================================================
# 5) 공통 오류 응답
# ===========================================================================


class ErrorResponse(BaseModel):
    """오류 응답(백엔드 ``GlobalExceptionHandler`` 와 동일 형태).

    계약서 6장: ``{"timestamp","status","error","message","path"}``
    AI 서비스는 500 을 던지지 않도록 설계했지만, 422(검증 실패) 등은 발생할 수
    있으므로 같은 형태로 통일한다 — 백엔드가 두 가지 오류 포맷을 다루지 않아도 된다.
    """

    model_config = ConfigDict(extra="allow")

    timestamp: str = Field("", description="발생 시각(ISO 8601)")
    status: int = Field(500, description="HTTP 상태 코드")
    error: str = Field("", description="오류 종류")
    message: str = Field("", description="오류 메시지")
    path: str = Field("", description="요청 경로")


def pydantic_available() -> bool:
    """pydantic 사용 가능 여부를 반환한다(진단용).

    Returns:
        pydantic 이 설치되어 실제 검증이 동작하면 True.
    """
    return _HAS_PYDANTIC


__all__ = [
    "BenchmarkEntry",
    "BenchmarkRequest",
    "ComponentDiagnosisOut",
    "DetectionBox",
    "DetectorName",
    "ElapsedMs",
    "ErrorResponse",
    "HealthResponse",
    "LprFramesResponse",
    "LprResponse",
    "ModelEntry",
    "ModelsInfoResponse",
    "OcrCandidate",
    "PdmRequest",
    "PdmResponse",
    "PerFrameResult",
    "PlateStatus",
    "QualityInfo",
    "RecognizerName",
    "RecommendationOut",
    "RiskLevel",
    "SensorReading",
    "pydantic_available",
]
