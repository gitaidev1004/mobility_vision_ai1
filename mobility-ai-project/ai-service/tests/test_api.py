"""REST API 통합 테스트 — ``TestClient`` 로 모든 엔드포인트가 200 을 반환하는지 검증.

**가중치 없이** 실행된다는 점이 이 테스트의 핵심이다. 계약서의
"학습된 가중치 파일이 없어도 모든 엔드포인트가 200 을 반환해야 한다" 를
실제로 검증하는 유일한 수단이다.

FastAPI/httpx 가 없는 환경(순수 로직 테스트만 돌리는 환경)에서는
``pytest.importorskip`` 으로 전체 모듈을 스킵한다. 이렇게 하면
``pytest tests`` 가 어떤 환경에서도 실패하지 않는다.
"""

from __future__ import annotations

import io

import numpy as np
import pytest

# FastAPI 와 TestClient(httpx 필요)가 없으면 이 모듈 전체를 스킵한다.
fastapi = pytest.importorskip("fastapi", reason="FastAPI 미설치 — API 테스트 스킵")
pytest.importorskip("httpx", reason="httpx 미설치 — TestClient 사용 불가")

from fastapi.testclient import TestClient  # noqa: E402

from app.main import create_app  # noqa: E402


@pytest.fixture(scope="module")
def client() -> TestClient:
    """모듈 범위 ``TestClient``.

    범위를 module 로 둔 이유: 앱 생성과 모델 레지스트리 초기화를 1회만 하면
    테스트가 훨씬 빠르다. 테스트 간 상태 공유가 문제되는 엔드포인트는 없다
    (모든 엔드포인트가 상태를 변경하지 않는다).

    Returns:
        TestClient 인스턴스.
    """
    with TestClient(create_app()) as test_client:
        yield test_client


def encode_png(image: np.ndarray) -> bytes:
    """numpy 이미지를 PNG 바이트로 인코딩한다.

    Args:
        image: ``(H, W, 3)`` uint8 BGR 배열.

    Returns:
        PNG 바이트.
    """
    try:
        import cv2

        ok, buffer = cv2.imencode(".png", image)
        if ok:
            return buffer.tobytes()
    except ImportError:
        pass

    from PIL import Image

    stream = io.BytesIO()
    # OpenCV 는 BGR, Pillow 는 RGB 이므로 채널을 뒤집는다.
    Image.fromarray(image[:, :, ::-1]).save(stream, format="PNG")
    return stream.getvalue()


@pytest.fixture(scope="module")
def plate_png() -> bytes:
    """합성 번호판 이미지 PNG 바이트.

    Returns:
        PNG 바이트.
    """
    try:
        from app.train.synth_plate import render_plate

        return encode_png(render_plate("12가3456", "new_private"))
    except Exception:
        # Pillow/폰트가 없으면 단순 패턴으로 대체한다(엔드포인트 200 검증이 목적).
        fallback = np.full((110, 520, 3), 230, dtype=np.uint8)
        for i in range(7):
            x = 30 + i * 68
            fallback[30:80, x : x + 34] = 20
        return encode_png(fallback)


@pytest.fixture(scope="module")
def scene_png() -> bytes:
    """번호판이 포함된 장면 이미지 PNG 바이트.

    Returns:
        PNG 바이트.
    """
    try:
        import random

        from app.train.synth_plate import generate_sample

        return encode_png(generate_sample(random.Random(20261024), augment=False).image)
    except Exception:
        return encode_png(np.full((360, 640, 3), 120, dtype=np.uint8))


def sample_readings(count: int = 8) -> list[dict[str, float | str]]:
    """계약서 예시 형태의 센서 판독 리스트를 만든다.

    Args:
        count: 생성할 판독 수.

    Returns:
        센서 판독 리스트.
    """
    return [
        {
            "measured_at": f"2026-10-{i + 1:02d}T09:00:00",
            "engine_temp_c": 92.4 + i * 0.5,
            "engine_rpm": 1850.0,
            "speed_kph": 48.0,
            "battery_voltage": 12.4 - i * 0.02,
            "battery_soh": 0.86,
            "tire_pressure_fl": 32.1,
            "tire_pressure_fr": 31.8,
            "tire_pressure_rl": 30.9,
            "tire_pressure_rr": 31.2,
            "brake_pad_mm": 4.1 - i * 0.05,
            "oil_quality": 0.62,
            "vibration_rms": 0.44,
            "fuel_level": 0.55,
        }
        for i in range(count)
    ]


# ===========================================================================
# GET /health
# ===========================================================================


def test_health_returns_200(client):
    """헬스체크가 200 과 계약서 스키마를 반환한다."""
    response = client.get("/health")
    assert response.status_code == 200

    payload = response.json()
    for key in ("status", "version", "device", "detector", "recognizer", "torch", "threads"):
        assert key in payload, f"계약서 필드 누락: {key}"
    assert payload["status"] == "UP"
    assert isinstance(payload["threads"], int)


def test_health_reports_fallback_without_weights(client):
    """가중치가 없으면 모델 이름에 ``-fallback`` 접미사가 붙는다(계약서 요구사항)."""
    payload = client.get("/health").json()
    assert payload["detector"].endswith("-fallback")
    assert payload["recognizer"].endswith("-fallback")


def test_healthz_alias(client):
    """``/healthz`` 별칭도 동작한다."""
    assert client.get("/healthz").status_code == 200


def test_root_endpoint(client):
    """루트 경로가 엔드포인트 목록을 반환한다."""
    response = client.get("/")
    assert response.status_code == 200
    assert "endpoints" in response.json()


# ===========================================================================
# POST /api/v1/lpr/recognize
# ===========================================================================


def assert_lpr_contract(payload: dict) -> None:
    """LPR 응답이 계약서 스키마를 만족하는지 검증한다.

    Args:
        payload: 응답 JSON.
    """
    for key in (
        "plate_number",
        "confidence",
        "status",
        "valid_format",
        "detector",
        "recognizer",
        "elapsed_ms",
        "quality",
        "detections",
        "candidates",
        "char_confidences",
    ):
        assert key in payload, f"계약서 필드 누락: {key}"

    assert payload["status"] in {"SUCCESS", "LOW_CONFIDENCE", "NO_PLATE", "FORMAT_INVALID"}
    assert 0.0 <= payload["confidence"] <= 1.0
    assert isinstance(payload["valid_format"], bool)

    for key in ("detect", "ocr", "total"):
        assert key in payload["elapsed_ms"]
        assert payload["elapsed_ms"][key] >= 0.0

    for key in ("score", "blur", "brightness", "plate_px_width", "readable", "reason"):
        assert key in payload["quality"]

    # 미검출이면 plate_number 가 null 이어야 한다(계약서 명시).
    if payload["status"] == "NO_PLATE":
        assert payload["plate_number"] is None


def test_recognize_plate_image(client, plate_png):
    """번호판 이미지 인식이 200 과 계약 스키마를 반환한다."""
    response = client.post(
        "/api/v1/lpr/recognize",
        files={"file": ("plate.png", plate_png, "image/png")},
        data={"min_conf": "0.15", "detector": "yolo", "recognizer": "crnn"},
    )
    assert response.status_code == 200
    assert_lpr_contract(response.json())


def test_recognize_scene_image(client, scene_png):
    """장면 이미지(배경 포함) 인식도 200 을 반환한다."""
    response = client.post(
        "/api/v1/lpr/recognize",
        files={"file": ("scene.png", scene_png, "image/png")},
        data={"min_conf": "0.15"},
    )
    assert response.status_code == 200
    assert_lpr_contract(response.json())


def test_recognize_uses_defaults_when_form_omitted(client, plate_png):
    """form 필드를 생략해도 기본값으로 동작한다(계약서 기본값)."""
    response = client.post(
        "/api/v1/lpr/recognize",
        files={"file": ("plate.png", plate_png, "image/png")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert_lpr_contract(payload)
    assert payload["detector"].startswith("yolo")
    assert payload["recognizer"].startswith("crnn")


@pytest.mark.parametrize("detector", ["yolo", "mobilenet"])
def test_recognize_all_detectors(client, plate_png, detector):
    """계약서가 허용하는 모든 검출기가 200 을 반환한다."""
    response = client.post(
        "/api/v1/lpr/recognize",
        files={"file": ("plate.png", plate_png, "image/png")},
        data={"detector": detector, "min_conf": "0.15"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert_lpr_contract(payload)
    assert payload["detector"].startswith(detector)


@pytest.mark.parametrize("recognizer", ["crnn", "attention", "paddle"])
def test_recognize_all_recognizers(client, plate_png, recognizer):
    """계약서가 허용하는 모든 인식기가 200 을 반환한다.

    ``paddle`` 은 미설치 환경에서 폴백으로 전환되지만, 그래도 200 이어야 한다.
    """
    response = client.post(
        "/api/v1/lpr/recognize",
        files={"file": ("plate.png", plate_png, "image/png")},
        data={"recognizer": recognizer, "min_conf": "0.15"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert_lpr_contract(payload)
    assert payload["recognizer"].startswith(recognizer)


def test_recognize_invalid_model_name_falls_back(client, plate_png):
    """알 수 없는 모델명도 422 가 아니라 200 + 기본값으로 처리된다.

    게이트 운영에서 "파라미터 오타 때문에 차량이 통과하지 못하는 것" 이
    최악의 실패이므로, 잘못된 값은 기본값으로 대체한다.
    """
    response = client.post(
        "/api/v1/lpr/recognize",
        files={"file": ("plate.png", plate_png, "image/png")},
        data={"detector": "nonexistent", "recognizer": "alsobogus"},
    )
    assert response.status_code == 200
    assert_lpr_contract(response.json())


def test_recognize_garbage_bytes_returns_no_plate(client):
    """이미지가 아닌 바이트는 500 이 아니라 ``NO_PLATE`` 200 이다."""
    response = client.post(
        "/api/v1/lpr/recognize",
        files={"file": ("junk.png", b"this-is-not-an-image", "image/png")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "NO_PLATE"
    assert payload["plate_number"] is None
    assert payload["quality"]["reason"] is not None


def test_recognize_empty_file_returns_no_plate(client):
    """빈 파일도 ``NO_PLATE`` 200 이다."""
    response = client.post(
        "/api/v1/lpr/recognize",
        files={"file": ("empty.png", b"", "image/png")},
    )
    assert response.status_code == 200
    assert response.json()["status"] == "NO_PLATE"


def test_recognize_tiny_image(client):
    """극단적으로 작은 이미지도 200 을 반환한다(예외 없음)."""
    tiny = encode_png(np.zeros((4, 4, 3), dtype=np.uint8))
    response = client.post(
        "/api/v1/lpr/recognize", files={"file": ("tiny.png", tiny, "image/png")}
    )
    assert response.status_code == 200
    assert_lpr_contract(response.json())


def test_recognize_out_of_range_min_conf(client, plate_png):
    """``min_conf`` 가 범위를 벗어나도 클램프되어 200 이다."""
    for value in ("-5", "99", "1.5"):
        response = client.post(
            "/api/v1/lpr/recognize",
            files={"file": ("plate.png", plate_png, "image/png")},
            data={"min_conf": value},
        )
        assert response.status_code == 200, f"min_conf={value} 에서 실패"


def test_recognize_missing_file_returns_no_plate(client):
    """필수 ``file`` 이 없으면 계약 준수 200 응답을 돌려준다.

    계약서는 ``file`` 을 필수로 정의하지만, LPR 경로의 검증 실패는
    422 대신 ``NO_PLATE`` 로 변환한다(``main.py`` 의 검증 예외 핸들러).
    """
    response = client.post("/api/v1/lpr/recognize", data={"min_conf": "0.3"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "NO_PLATE"
    assert payload["plate_number"] is None


# ===========================================================================
# POST /api/v1/lpr/recognize-frames
# ===========================================================================


def test_recognize_frames(client, plate_png):
    """다중 프레임 인식이 200 과 확장 스키마를 반환한다."""
    response = client.post(
        "/api/v1/lpr/recognize-frames",
        files=[
            ("files", ("f1.png", plate_png, "image/png")),
            ("files", ("f2.png", plate_png, "image/png")),
            ("files", ("f3.png", plate_png, "image/png")),
        ],
        data={"min_conf": "0.15", "fusion": "majority-vote"},
    )
    assert response.status_code == 200

    payload = response.json()
    assert_lpr_contract(payload)
    # 계약서: 단일 스키마 + frames/fusion/per_frame
    assert payload["frames"] == 3
    assert payload["fusion"] == "majority-vote"
    assert len(payload["per_frame"]) == 3
    for frame in payload["per_frame"]:
        for key in ("plate_number", "confidence", "status", "quality_score", "elapsed_ms"):
            assert key in frame


@pytest.mark.parametrize("fusion", ["majority-vote", "char-vote", "best-confidence"])
def test_recognize_frames_all_fusions(client, plate_png, fusion):
    """모든 융합 전략이 200 을 반환한다."""
    response = client.post(
        "/api/v1/lpr/recognize-frames",
        files=[
            ("files", ("f1.png", plate_png, "image/png")),
            ("files", ("f2.png", plate_png, "image/png")),
        ],
        data={"fusion": fusion, "min_conf": "0.15"},
    )
    assert response.status_code == 200
    assert response.json()["fusion"] == fusion


def test_recognize_frames_single_frame(client, plate_png):
    """프레임 1장만 보내도 동작한다."""
    response = client.post(
        "/api/v1/lpr/recognize-frames",
        files=[("files", ("f1.png", plate_png, "image/png"))],
        data={"min_conf": "0.15"},
    )
    assert response.status_code == 200
    assert response.json()["frames"] == 1


def test_recognize_frames_all_garbage(client):
    """모든 프레임이 디코딩 불가면 ``NO_PLATE`` 200 이다."""
    response = client.post(
        "/api/v1/lpr/recognize-frames",
        files=[
            ("files", ("a.png", b"junk", "image/png")),
            ("files", ("b.png", b"junk", "image/png")),
        ],
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "NO_PLATE"
    assert payload["frames"] == 0


def test_recognize_frames_unknown_fusion_falls_back(client, plate_png):
    """알 수 없는 융합 전략은 기본값으로 대체된다."""
    response = client.post(
        "/api/v1/lpr/recognize-frames",
        files=[("files", ("f1.png", plate_png, "image/png"))],
        data={"fusion": "nonsense"},
    )
    assert response.status_code == 200
    assert response.json()["fusion"] == "majority-vote"


# ===========================================================================
# POST /api/v1/pdm/predict
# ===========================================================================


def assert_pdm_contract(payload: dict) -> None:
    """PdM 응답이 계약서 스키마를 만족하는지 검증한다.

    Args:
        payload: 응답 JSON.
    """
    for key in (
        "vehicle_id",
        "health_score",
        "anomaly_score",
        "failure_risk",
        "risk_level",
        "rul_days",
        "top_component",
        "components",
        "recommendations",
        "model_version",
        "computed_at",
    ):
        assert key in payload, f"계약서 필드 누락: {key}"

    assert payload["risk_level"] in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
    assert 0.0 <= payload["health_score"] <= 100.0
    assert 0.0 <= payload["anomaly_score"] <= 1.0
    assert 0.0 <= payload["failure_risk"] <= 1.0
    assert payload["rul_days"] >= 0

    for component in payload["components"]:
        for key in ("name", "score", "status", "reason"):
            assert key in component
        assert 0.0 <= component["score"] <= 100.0

    for recommendation in payload["recommendations"]:
        for key in ("priority", "item", "due_km", "due_days"):
            assert key in recommendation


def test_pdm_predict_contract_example(client):
    """계약서 예시 요청이 200 과 계약 스키마를 반환한다."""
    response = client.post(
        "/api/v1/pdm/predict",
        json={"vehicle_id": 1, "odometer_km": 84210, "readings": sample_readings()},
    )
    assert response.status_code == 200

    payload = response.json()
    assert_pdm_contract(payload)
    assert payload["vehicle_id"] == 1


def test_pdm_predict_single_reading(client):
    """센서 판독 1건으로도 예측이 동작한다."""
    response = client.post(
        "/api/v1/pdm/predict",
        json={"vehicle_id": 7, "odometer_km": 10000, "readings": sample_readings(1)},
    )
    assert response.status_code == 200
    assert_pdm_contract(response.json())


def test_pdm_predict_empty_readings(client):
    """센서 데이터가 없어도 200 이고, "건강하다" 고 단정하지 않는다.

    센서가 죽은 차량이 "가장 건강한 차량" 으로 집계되는 것을 막아야 한다.
    """
    response = client.post(
        "/api/v1/pdm/predict", json={"vehicle_id": 3, "readings": []}
    )
    assert response.status_code == 200
    payload = response.json()
    assert_pdm_contract(payload)
    assert payload["risk_level"] != "LOW"
    assert payload["health_score"] <= 70.0


def test_pdm_predict_partial_sensors(client):
    """일부 센서 필드만 있어도 200 이다(결측은 보간된다)."""
    response = client.post(
        "/api/v1/pdm/predict",
        json={
            "vehicle_id": 4,
            "readings": [
                {"measured_at": "2026-10-01T09:00:00", "engine_temp_c": 95.0},
                {"measured_at": "2026-10-02T09:00:00", "brake_pad_mm": 3.5},
            ],
        },
    )
    assert response.status_code == 200
    assert_pdm_contract(response.json())


def test_pdm_predict_critical_vehicle(client):
    """심각한 열화 데이터는 높은 위험 등급과 권고를 반환한다."""
    readings = [
        {
            "measured_at": f"2026-10-{i + 1:02d}T09:00:00",
            "engine_temp_c": 113.0,
            "battery_voltage": 11.5,
            "battery_soh": 0.60,
            "brake_pad_mm": 2.2,
            "oil_quality": 0.15,
            "vibration_rms": 1.3,
            "tire_pressure_fl": 24.0,
            "tire_pressure_fr": 33.0,
            "tire_pressure_rl": 33.0,
            "tire_pressure_rr": 33.0,
            "engine_rpm": 1800.0,
            "speed_kph": 40.0,
            "fuel_level": 0.3,
        }
        for i in range(6)
    ]
    response = client.post(
        "/api/v1/pdm/predict", json={"vehicle_id": 9, "readings": readings}
    )
    assert response.status_code == 200

    payload = response.json()
    assert_pdm_contract(payload)
    assert payload["risk_level"] in {"HIGH", "CRITICAL"}
    assert payload["recommendations"]
    assert payload["recommendations"][0]["priority"] == 1


def test_pdm_predict_malformed_values(client):
    """비정상 센서값(문자열/극단값)에도 200 을 반환한다."""
    response = client.post(
        "/api/v1/pdm/predict",
        json={
            "vehicle_id": 5,
            "odometer_km": 0,
            "readings": [
                {
                    "measured_at": "not-a-date",
                    "engine_temp_c": -999.0,
                    "battery_voltage": 99999.0,
                    "brake_pad_mm": -1.0,
                }
            ],
        },
    )
    assert response.status_code == 200
    assert_pdm_contract(response.json())


def test_pdm_spec_endpoint(client):
    """``GET /api/v1/pdm/spec`` 이 200 과 가중치 정보를 반환한다."""
    response = client.get("/api/v1/pdm/spec")
    assert response.status_code == 200
    payload = response.json()
    assert "weights" in payload
    assert payload["fallback_available"] is True


def test_pdm_predict_missing_vehicle_id_is_422(client):
    """필수 ``vehicle_id`` 누락은 422 다(LPR 과 달리 유지한다).

    PdM 은 게이트 통행을 막지 않으므로, 잘못된 요청을 숨기지 않고
    호출자에게 알리는 편이 디버깅에 유리하다.
    """
    response = client.post("/api/v1/pdm/predict", json={"readings": []})
    assert response.status_code == 422
    payload = response.json()
    # 계약서 6장 공통 오류 포맷
    for key in ("timestamp", "status", "error", "message", "path"):
        assert key in payload


# ===========================================================================
# GET /api/v1/models/info, POST /api/v1/models/benchmark
# ===========================================================================


def test_models_info(client):
    """모델 정보가 200 과 레지스트리 목록을 반환한다."""
    response = client.get("/api/v1/models/info")
    assert response.status_code == 200

    payload = response.json()
    for key in ("detectors", "recognizers", "pdm", "charset", "runtime"):
        assert key in payload

    names = {entry["name"] for entry in payload["detectors"]}
    assert {"yolo", "mobilenet"}.issubset(names)

    ocr_names = {entry["name"] for entry in payload["recognizers"]}
    assert {"crnn", "attention", "paddle"}.issubset(ocr_names)

    # 문자셋 정보가 모델 출력 차원과 일치한다.
    from app.ocr.charset import NUM_CLASSES

    assert payload["charset"]["num_classes"] == NUM_CLASSES


def test_models_info_with_load(client):
    """``load=true`` 로 호출하면 실제 로딩 결과(폴백 여부)가 포함된다."""
    response = client.get("/api/v1/models/info", params={"load": "true"})
    assert response.status_code == 200

    payload = response.json()
    for entry in payload["detectors"]:
        assert "loaded" in entry
    # 가중치가 없으므로 노출 이름에 -fallback 이 붙어야 한다.
    exposed = [e.get("exposed_name") for e in payload["detectors"] if e.get("exposed_name")]
    assert any(str(name).endswith("-fallback") for name in exposed)


def test_benchmark_returns_array(client):
    """벤치마크가 200 과 계약서 스키마 **배열**을 반환한다."""
    response = client.post(
        "/api/v1/models/benchmark",
        json={"samples": 3, "detectors": ["yolo"], "recognizers": ["crnn"]},
    )
    assert response.status_code == 200

    payload = response.json()
    assert isinstance(payload, list)
    assert payload

    for entry in payload:
        for key in (
            "model_name",
            "task",
            "detect_recall",
            "exact_match_rate",
            "latency_p50_ms",
            "latency_p95_ms",
            "memory_mb",
            "model_size_mb",
            "quantized",
        ):
            assert key in entry, f"계약서 필드 누락: {key}"
        assert 0.0 <= entry["detect_recall"] <= 1.0
        assert 0.0 <= entry["exact_match_rate"] <= 1.0
        assert entry["latency_p50_ms"] >= 0.0
        assert isinstance(entry["quantized"], bool)


def test_benchmark_without_body(client):
    """본문 없이 호출해도 기본값으로 동작한다."""
    response = client.post("/api/v1/models/benchmark")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


# ===========================================================================
# 공통 동작
# ===========================================================================


def test_request_id_header_is_returned(client):
    """응답에 ``X-Request-ID`` 와 처리 시간 헤더가 포함된다."""
    response = client.get("/health")
    assert "X-Request-ID" in response.headers
    assert "X-Process-Time-Ms" in response.headers


def test_request_id_is_propagated(client):
    """외부에서 보낸 ``X-Request-ID`` 를 그대로 이어받는다(분산 추적)."""
    response = client.get("/health", headers={"X-Request-ID": "trace-abc-123"})
    assert response.headers["X-Request-ID"] == "trace-abc-123"


def test_unknown_path_returns_404_with_error_schema(client):
    """없는 경로는 404 와 공통 오류 포맷을 반환한다."""
    response = client.get("/api/v1/does-not-exist")
    assert response.status_code == 404
    payload = response.json()
    for key in ("timestamp", "status", "error", "message", "path"):
        assert key in payload


def test_openapi_schema_available(client):
    """OpenAPI 스키마가 생성된다(문서화 검증)."""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    assert "/api/v1/lpr/recognize" in schema["paths"]
    assert "/api/v1/pdm/predict" in schema["paths"]
    assert "/health" in schema["paths"]


def test_cors_headers_present(client):
    """허용된 오리진에 대해 CORS 헤더가 설정된다."""
    response = client.get("/health", headers={"Origin": "http://localhost:5173"})
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "http://localhost:5173"


def test_no_endpoint_returns_500(client, plate_png):
    """**계약서 핵심 요구사항**: 어떤 입력에도 500 을 반환하지 않는다.

    정상/비정상 입력을 섞어 모든 엔드포인트를 호출하고 500 이 없음을 확인한다.
    """
    cases = [
        ("GET", "/health", {}),
        ("GET", "/api/v1/models/info", {}),
        ("GET", "/api/v1/pdm/spec", {}),
        (
            "POST",
            "/api/v1/lpr/recognize",
            {"files": {"file": ("p.png", plate_png, "image/png")}},
        ),
        (
            "POST",
            "/api/v1/lpr/recognize",
            {"files": {"file": ("junk.png", b"\x00\x01\x02", "image/png")}},
        ),
        (
            "POST",
            "/api/v1/lpr/recognize-frames",
            {"files": [("files", ("p.png", plate_png, "image/png"))]},
        ),
        ("POST", "/api/v1/pdm/predict", {"json": {"vehicle_id": 1, "readings": []}}),
        (
            "POST",
            "/api/v1/pdm/predict",
            {"json": {"vehicle_id": 1, "readings": sample_readings(2)}},
        ),
        ("POST", "/api/v1/models/benchmark", {"json": {"samples": 2}}),
    ]

    for method, path, kwargs in cases:
        response = client.request(method, path, **kwargs)
        assert response.status_code != 500, f"{method} {path} 가 500 을 반환했다"
        assert response.status_code < 500, f"{method} {path} → {response.status_code}"
