"""이상탐지: IsolationForest / OneClassSVM / AutoEncoder + 통계 임계.

예지보전 이상탐지의 근본적 어려움:
    고장 데이터는 **극히 드물다**. 차량 1000대 중 브레이크 고장은 연간 몇 건이다.
    지도학습(정상/고장 라벨)으로는 데이터가 부족하다. 그래서 "정상이 무엇인지만
    배우고, 그것에서 벗어나면 이상" 이라고 보는 **단일 클래스 학습**을 쓴다.

세 가지 접근을 모두 제공하는 이유 (교재 비교 포인트):

    | 방법 | 원리 | 강점 | 약점 |
    |---|---|---|---|
    | IsolationForest | 무작위 분할로 격리되는 깊이 | 빠름, 고차원 OK, 튜닝 쉬움 | 지역 밀도 차이에 둔감 |
    | OneClassSVM | 정상 영역을 감싸는 초평면 | 복잡한 경계 학습 | O(N²), 스케일 민감 |
    | AutoEncoder | 복원 오차 | 비선형 패턴, 센서 간 상관 학습 | 학습 데이터/시간 필요 |
    | **통계 임계(폴백)** | z-score + 도메인 범위 | 가중치 불필요, 설명 가능 | 센서 간 상관 무시 |

    sklearn/torch 가 없으면 **통계 임계** 경로가 동작한다. 정확도는 낮지만
    "왜 이상인가" 를 센서 이름으로 설명할 수 있어 운영 가치가 오히려 높다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from app.core.logging import get_logger
from app.pdm.features import (
    SENSOR_NOMINAL,
    FeatureFrame,
    aggregate_features,
    feature_vector,
)

logger = get_logger(__name__)

#: 통계 임계 폴백에서 "이상" 으로 보는 z-score 절댓값 하한.
#: 정규분포 가정 시 |z| > 2.5 는 상위/하위 약 0.6% 다. 센서 노이즈를 감안해
#: 3.0 대신 2.5 로 다소 민감하게 둔다 — 예지보전은 놓치는 것(FN)이
#: 잘못 경고하는 것(FP)보다 비용이 크다.
ZSCORE_THRESHOLD: float = 2.5

#: 각 센서의 **위험 방향**과 임계값. ``(방향, 경고값, 위험값)``.
#: 방향 ``"high"`` = 값이 클수록 위험, ``"low"`` = 작을수록 위험.
#:
#: 이 표가 통계 임계 폴백의 핵심이다. z-score 만 쓰면 "평소보다 타이어 압력이
#: 높다" 도 이상으로 잡지만, 과압은 저압보다 훨씬 덜 위험하다.
#: 방향성 지식을 넣으면 같은 편차에 대해 올바른 심각도를 매길 수 있다.
RISK_THRESHOLDS: dict[str, tuple[str, float, float]] = {
    "engine_temp_c": ("high", 103.0, 110.0),
    "engine_rpm": ("high", 4500.0, 6000.0),
    "battery_voltage": ("low", 12.2, 11.8),
    "battery_soh": ("low", 0.80, 0.70),
    "tire_pressure_fl": ("low", 29.0, 26.0),
    "tire_pressure_fr": ("low", 29.0, 26.0),
    "tire_pressure_rl": ("low", 29.0, 26.0),
    "tire_pressure_rr": ("low", 29.0, 26.0),
    "brake_pad_mm": ("low", 4.0, 3.0),
    "oil_quality": ("low", 0.40, 0.25),
    "vibration_rms": ("high", 0.70, 1.00),
}


@dataclass(slots=True)
class AnomalyResult:
    """이상탐지 결과.

    Attributes:
        score: 0~1 이상 점수(1 = 매우 이상).
        method: 사용한 방법 이름.
        is_anomaly: 이상 판정 여부.
        contributors: 이상에 기여한 센서 ``[{"sensor", "value", "severity", "reason"}]``.
            이 필드가 운영 가치의 핵심이다 — "이상입니다" 만으로는 조치할 수 없다.
        detail: 방법별 추가 정보.
    """

    score: float
    method: str
    is_anomaly: bool = False
    contributors: list[dict[str, Any]] = field(default_factory=list)
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON 직렬화용 dict 를 반환한다."""
        return {
            "score": round(float(self.score), 4),
            "method": self.method,
            "is_anomaly": bool(self.is_anomaly),
            "contributors": self.contributors,
            "detail": self.detail,
        }


# ---------------------------------------------------------------------------
# 1) 통계 임계 (폴백 — 가중치 불필요)
# ---------------------------------------------------------------------------


def statistical_anomaly(frame: FeatureFrame) -> AnomalyResult:
    """도메인 임계값 + z-score 로 이상을 탐지한다(가중치 불필요).

    두 신호를 결합한다:
        1. **절대 임계값 위반**: ``RISK_THRESHOLDS`` 의 경고/위험 구간 진입.
           "브레이크 패드 2.8mm" 는 과거 이력과 무관하게 위험하다.
        2. **상대 편차(z-score)**: 자기 과거 분포 대비 이탈.
           "이 차는 늘 95도였는데 갑자기 100도" 는 절대값이 정상 범위여도 신호다.

    왜 둘을 모두 쓰는가? 절대값만 보면 "서서히 나빠지는" 조기 신호를 놓치고,
    z-score 만 보면 "처음부터 나쁜" 차량을 정상으로 본다(과거 분포가 이미 나쁨).

    최종 점수는 센서별 심각도의 **최댓값과 평균의 가중합**이다.
    최댓값만 쓰면 센서 하나의 오작동이 전체 점수를 지배하고, 평균만 쓰면
    심각한 단일 고장이 다른 정상 센서들에 희석된다.

    Args:
        frame: 센서 FeatureFrame.

    Returns:
        AnomalyResult(method=``"statistical-threshold"``).
    """
    if frame.rows == 0:
        return AnomalyResult(
            score=0.0,
            method="statistical-threshold",
            is_anomaly=False,
            detail={"reason": "no-data"},
        )

    features = aggregate_features(frame)
    severities: list[float] = []
    contributors: list[dict[str, Any]] = []

    for sensor, (direction, warn, danger) in RISK_THRESHOLDS.items():
        column = frame.column(sensor)
        if column.size == 0:
            continue
        current = float(features.get(f"{sensor}_last", SENSOR_NOMINAL.get(sensor, 0.0)))

        # --- 1. 절대 임계값 심각도 (0~1) ---
        if direction == "high":
            if current >= danger:
                absolute = 1.0
            elif current >= warn:
                absolute = 0.5 + 0.5 * (current - warn) / max(1e-6, danger - warn)
            else:
                # 경고 미달 구간에서도 접근도를 0~0.5 로 매긴다(조기 신호).
                span = max(1e-6, warn - SENSOR_NOMINAL.get(sensor, warn * 0.8))
                absolute = max(0.0, 0.5 * (current - SENSOR_NOMINAL.get(sensor, 0.0)) / span)
        else:  # "low"
            if current <= danger:
                absolute = 1.0
            elif current <= warn:
                absolute = 0.5 + 0.5 * (warn - current) / max(1e-6, warn - danger)
            else:
                span = max(1e-6, SENSOR_NOMINAL.get(sensor, warn * 1.2) - warn)
                absolute = max(0.0, 0.5 * (SENSOR_NOMINAL.get(sensor, 0.0) - current) / span)
        absolute = float(np.clip(absolute, 0.0, 1.0))

        # --- 2. z-score 심각도 ---
        if column.size >= 3:
            std = max(float(column.std()), 1e-6)
            z = abs((current - float(column.mean())) / std)
            relative = float(np.clip(z / ZSCORE_THRESHOLD, 0.0, 1.0))
        else:
            z = 0.0
            relative = 0.0

        # 결합 규칙: z-score 는 심각도를 **올리기만 하고 내리지 않는다.**
        #
        # 왜 가중합(0.7a + 0.3r)이 틀렸는가 — 실측으로 드러난 모순:
        #   ① 서서히 악화(브레이크 5.0→2.9mm): 변동이 있어 z-score 가 높음 → 0.657
        #   ② 이미 위험 상태로 고정(브레이크 2.2mm 유지): 변동이 없어 z=0 → 0.571
        # 즉 "계속 위험한 차" 가 "막 위험해진 차" 보다 안전하다고 보고된다.
        # z-score 는 **변화의 탐지**이고 절대 임계는 **상태의 평가**다. 상태가
        # 이미 위험하면 변화가 없다는 사실이 위험을 줄여 주지 않는다.
        # 따라서 둘 중 큰 값을 쓰되, 둘 다 높을 때만 가중합으로 가점한다.
        severity = float(np.clip(max(absolute, 0.70 * absolute + 0.30 * relative), 0.0, 1.0))
        severities.append(severity)

        if severity >= 0.5:
            contributors.append(
                {
                    "sensor": sensor,
                    "value": round(current, 3),
                    "severity": round(severity, 4),
                    "zscore": round(z, 2),
                    "reason": _threshold_reason(sensor, current, direction, warn, danger),
                }
            )

    if not severities:
        return AnomalyResult(
            score=0.0, method="statistical-threshold", detail={"reason": "no-known-sensors"}
        )

    arr = np.asarray(severities)
    # 최댓값 0.65 + 평균 0.35. 단일 심각 고장과 전반적 노화를 모두 반영한다.
    score = float(np.clip(0.65 * arr.max() + 0.35 * arr.mean(), 0.0, 1.0))
    contributors.sort(key=lambda c: c["severity"], reverse=True)

    return AnomalyResult(
        score=score,
        method="statistical-threshold",
        is_anomaly=score >= 0.5,
        contributors=contributors[:6],
        detail={
            "max_severity": round(float(arr.max()), 4),
            "mean_severity": round(float(arr.mean()), 4),
            "sensors_evaluated": len(severities),
            "weights_required": False,
        },
    )


def _threshold_reason(
    sensor: str, value: float, direction: str, warn: float, danger: float
) -> str:
    """임계값 위반 사유를 한국어로 설명한다.

    Args:
        sensor: 센서명.
        value: 현재값.
        direction: ``"high"`` | ``"low"``.
        warn: 경고 임계값.
        danger: 위험 임계값.

    Returns:
        한국어 사유 문장.
    """
    labels = {
        "engine_temp_c": ("냉각수 온도", "°C"),
        "engine_rpm": ("엔진 회전수", "rpm"),
        "battery_voltage": ("배터리 전압", "V"),
        "battery_soh": ("배터리 건강도", ""),
        "tire_pressure_fl": ("좌전 타이어 공기압", "psi"),
        "tire_pressure_fr": ("우전 타이어 공기압", "psi"),
        "tire_pressure_rl": ("좌후 타이어 공기압", "psi"),
        "tire_pressure_rr": ("우후 타이어 공기압", "psi"),
        "brake_pad_mm": ("브레이크 패드 두께", "mm"),
        "oil_quality": ("엔진오일 상태", ""),
        "vibration_rms": ("진동 RMS", "g"),
    }
    label, unit = labels.get(sensor, (sensor, ""))
    if direction == "high":
        if value >= danger:
            return f"{label} {value:.1f}{unit} — 위험 기준({danger:.1f}{unit}) 초과"
        if value >= warn:
            return f"{label} {value:.1f}{unit} — 경고 기준({warn:.1f}{unit}) 초과"
        return f"{label} {value:.1f}{unit} — 경고 기준 접근 중"
    if value <= danger:
        return f"{label} {value:.2f}{unit} — 위험 기준({danger:.2f}{unit}) 이하"
    if value <= warn:
        return f"{label} {value:.2f}{unit} — 경고 기준({warn:.2f}{unit}) 이하"
    return f"{label} {value:.2f}{unit} — 경고 기준 접근 중"


# ---------------------------------------------------------------------------
# 2) IsolationForest / OneClassSVM (sklearn)
# ---------------------------------------------------------------------------


def _load_sklearn_model(path: str) -> Any | None:
    """joblib 로 저장된 sklearn 모델을 로드한다.

    Args:
        path: ``.joblib`` 파일 경로.

    Returns:
        모델 객체 또는 None(파일 없음/joblib 없음/로딩 실패).
    """
    file_path = Path(str(path))
    if not file_path.is_file():
        return None
    try:
        import joblib  # noqa: PLC0415

        return joblib.load(file_path)
    except Exception as exc:  # noqa: BLE001
        logger.info("sklearn 모델 로딩 실패(%s): %s", path, exc)
        return None


def sklearn_anomaly(frame: FeatureFrame, model_path: str) -> AnomalyResult | None:
    """학습된 sklearn 이상탐지 모델로 점수를 계산한다.

    IsolationForest 의 ``score_samples`` 는 음수 로그 밀도 형태로, **클수록 정상**
    이다(보통 -0.5 ~ -0.1 범위). 이를 0~1 이상 점수로 바꾸려면 부호를 뒤집고
    스케일링해야 한다. 이 변환을 틀리면 "정상을 이상으로" 보고하게 되므로
    실무에서 자주 나는 버그다.

    Args:
        frame: FeatureFrame.
        model_path: 모델 파일 경로.

    Returns:
        AnomalyResult 또는 None(모델 없음 → 호출자가 폴백).
    """
    model = _load_sklearn_model(model_path)
    if model is None:
        return None
    if frame.rows == 0:
        return AnomalyResult(score=0.0, method="sklearn", detail={"reason": "no-data"})

    try:
        vector, names = feature_vector(frame)
        x = vector.reshape(1, -1)

        if hasattr(model, "score_samples"):
            raw = float(model.score_samples(x)[0])
            # IsolationForest: score_samples 는 -1(이상) ~ 0(정상) 근처.
            # offset_ 을 기준으로 정규화하면 모델별 스케일 차이를 흡수한다.
            offset = float(getattr(model, "offset_", -0.5))
            score = float(np.clip((offset - raw) / max(1e-6, abs(offset)) * 0.5 + 0.5, 0.0, 1.0))
        elif hasattr(model, "decision_function"):
            raw = float(model.decision_function(x)[0])
            # OneClassSVM: 양수 = 정상, 음수 = 이상.
            score = float(np.clip(0.5 - raw * 0.5, 0.0, 1.0))
        else:
            return None

        method = type(model).__name__
        # 기여 센서는 통계 기법으로 보완한다. sklearn 모델은 "왜" 를 말해 주지
        # 않으므로, 설명은 통계 경로에서 가져오는 것이 실용적이다.
        statistical = statistical_anomaly(frame)
        return AnomalyResult(
            score=score,
            method=f"sklearn-{method}",
            is_anomaly=score >= 0.5,
            contributors=statistical.contributors,
            detail={
                "raw_score": round(raw, 6),
                "feature_count": len(names),
                "model_file": str(model_path),
            },
        )
    except Exception as exc:  # noqa: BLE001
        logger.info("sklearn 이상탐지 실패: %s", exc)
        return None


def train_isolation_forest(
    matrix: np.ndarray, *, contamination: float = 0.05, random_state: int = 42
) -> Any | None:
    """IsolationForest 를 학습한다(학습 스크립트/테스트용).

    ``contamination=0.05`` 의 의미: 학습 데이터의 5% 가 이상이라고 가정한다.
    이 값이 임계값을 결정하므로 중요하다 — 실제 고장률보다 크게 잡으면
    정상 차량을 이상으로 보고하고, 작게 잡으면 고장을 놓친다.
    차량 fleet 의 경험적 고장률(연 3~8%)에 맞춘 값이다.

    Args:
        matrix: ``(N, F)`` 학습 특징 행렬.
        contamination: 이상 비율 가정.
        random_state: 재현성 시드.

    Returns:
        학습된 모델 또는 None(sklearn 미설치).
    """
    try:
        from sklearn.ensemble import IsolationForest  # noqa: PLC0415
    except ImportError:
        logger.info("scikit-learn 미설치 → IsolationForest 학습 생략")
        return None

    arr = np.asarray(matrix, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 4:
        return None
    model = IsolationForest(
        n_estimators=200,
        contamination=float(contamination),
        random_state=int(random_state),
        n_jobs=1,  # 게이트 서버는 코어가 적다. 병렬화 오버헤드가 더 크다.
    )
    model.fit(arr)
    return model


def train_one_class_svm(matrix: np.ndarray, *, nu: float = 0.05, gamma: str = "scale") -> Any | None:
    """OneClassSVM 을 학습한다.

    ``nu`` 는 "이상으로 허용할 비율의 상한" 이자 "서포트 벡터 비율의 하한" 이다.
    IsolationForest 의 ``contamination`` 과 비슷한 역할이다.

    주의: OneClassSVM 은 **스케일에 매우 민감**하다. 엔진 RPM(0~7000)과
    오일 상태(0~1)를 그대로 넣으면 RPM 이 거리 계산을 지배한다.
    실제 학습 파이프라인에서는 ``StandardScaler`` 를 앞에 붙여야 한다.

    Args:
        matrix: ``(N, F)`` 학습 행렬.
        nu: 이상 비율 상한.
        gamma: RBF 커널 폭.

    Returns:
        학습된 ``Pipeline``(스케일러 포함) 또는 None.
    """
    try:
        from sklearn.pipeline import make_pipeline  # noqa: PLC0415
        from sklearn.preprocessing import StandardScaler  # noqa: PLC0415
        from sklearn.svm import OneClassSVM  # noqa: PLC0415
    except ImportError:
        logger.info("scikit-learn 미설치 → OneClassSVM 학습 생략")
        return None

    arr = np.asarray(matrix, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 4:
        return None
    # 스케일러를 파이프라인에 포함시켜, 추론 시 스케일링을 잊는 사고를 막는다.
    model = make_pipeline(StandardScaler(), OneClassSVM(nu=float(nu), gamma=gamma, kernel="rbf"))
    model.fit(arr)
    return model


# ---------------------------------------------------------------------------
# 3) AutoEncoder (torch)
# ---------------------------------------------------------------------------


def build_autoencoder(input_dim: int, *, latent_dim: int = 8) -> Any:
    """센서 이상탐지용 AutoEncoder 를 만든다.

    구조: ``input → 32 → 16 → latent(8) → 16 → 32 → input``

    왜 병목(bottleneck)이 이상탐지에 쓰이는가?
        AE 는 입력을 작은 잠재 벡터로 압축한 뒤 복원하도록 학습된다.
        정상 데이터로만 학습하면, 모델은 **정상 패턴의 저차원 매니폴드**를 배운다.
        이상 데이터는 그 매니폴드 밖에 있으므로 복원이 잘 안 되고,
        복원 오차(MSE)가 곧 이상 점수가 된다.

    왜 latent_dim=8 인가? 센서 56개 특징 중 실제 독립 자유도는 훨씬 적다
    (타이어 4개는 거의 같이 움직이고, 온도와 RPM 은 상관이 높다).
    너무 크게 잡으면 이상까지 복원해 버려 탐지력이 사라진다.

    Args:
        input_dim: 입력 특징 수.
        latent_dim: 잠재 차원.

    Returns:
        ``torch.nn.Module``.

    Raises:
        RuntimeError: torch 미설치.
    """
    try:
        import torch  # noqa: PLC0415
    except ImportError as exc:
        msg = "PyTorch 가 필요합니다: pip install torch==2.8.0"
        raise RuntimeError(msg) from exc

    nn = torch.nn

    class SensorAutoEncoder(nn.Module):
        """센서 특징 복원용 AutoEncoder."""

        def __init__(self) -> None:
            """인코더/디코더를 구성한다."""
            super().__init__()
            dim = max(4, int(input_dim))
            latent = max(2, min(int(latent_dim), dim // 2))
            self.encoder = nn.Sequential(
                nn.Linear(dim, 32),
                nn.ReLU(inplace=True),
                nn.Linear(32, 16),
                nn.ReLU(inplace=True),
                nn.Linear(16, latent),
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent, 16),
                nn.ReLU(inplace=True),
                nn.Linear(16, 32),
                nn.ReLU(inplace=True),
                nn.Linear(32, dim),
            )

        def forward(self, x: Any) -> Any:
            """순전파: 압축 후 복원.

            Args:
                x: ``(B, input_dim)`` 입력.

            Returns:
                복원된 ``(B, input_dim)`` 텐서.
            """
            return self.decoder(self.encoder(x))

        def reconstruction_error(self, x: Any) -> Any:
            """샘플별 복원 MSE 를 반환한다.

            Args:
                x: ``(B, input_dim)`` 입력.

            Returns:
                ``(B,)`` 오차 텐서.
            """
            with torch.no_grad():
                recon = self.forward(x)
                return ((recon - x) ** 2).mean(dim=-1)

    return SensorAutoEncoder()


def autoencoder_anomaly(frame: FeatureFrame, weights_path: str) -> AnomalyResult | None:
    """학습된 AutoEncoder 로 복원 오차 기반 이상 점수를 계산한다.

    체크포인트에는 모델 가중치와 함께 **정규화 통계(mean/std)** 와
    **오차 임계값(threshold)** 이 들어 있어야 한다. 임계값 없이는 복원 오차
    0.03 이 큰지 작은지 판단할 수 없다 — 학습 시 정상 데이터의 오차 분포에서
    95 분위수를 임계값으로 저장하는 것이 표준 관행이다.

    Args:
        frame: FeatureFrame.
        weights_path: ``.pt`` 체크포인트 경로.

    Returns:
        AnomalyResult 또는 None(torch/가중치 없음 → 호출자가 폴백).
    """
    if not Path(str(weights_path)).is_file():
        return None
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        return None
    if frame.rows == 0:
        return AnomalyResult(score=0.0, method="autoencoder", detail={"reason": "no-data"})

    try:
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict) or "model" not in checkpoint:
            return None

        vector, _names = feature_vector(frame)
        mean = np.asarray(checkpoint.get("feature_mean", np.zeros_like(vector)), dtype=np.float64)
        std = np.asarray(checkpoint.get("feature_std", np.ones_like(vector)), dtype=np.float64)
        std = np.where(std <= 1e-9, 1.0, std)
        normalized = (vector - mean) / std

        model = build_autoencoder(int(normalized.size), latent_dim=int(checkpoint.get("latent_dim", 8)))
        model.load_state_dict(checkpoint["model"], strict=True)
        model.eval()

        tensor = torch.from_numpy(normalized.astype(np.float32)).unsqueeze(0)
        error = float(model.reconstruction_error(tensor)[0].item())
        threshold = float(checkpoint.get("threshold", 1.0)) or 1.0
        # 오차/임계값 비율을 0~1 로 매핑한다. 비율 1.0 에서 점수 0.5 가 되도록
        # 설계해 "임계값 = 경계" 라는 직관을 유지한다.
        score = float(np.clip(error / (2.0 * threshold), 0.0, 1.0))

        statistical = statistical_anomaly(frame)
        return AnomalyResult(
            score=score,
            method="autoencoder",
            is_anomaly=error >= threshold,
            contributors=statistical.contributors,
            detail={
                "reconstruction_error": round(error, 6),
                "threshold": round(threshold, 6),
                "latent_dim": int(checkpoint.get("latent_dim", 8)),
            },
        )
    except Exception as exc:  # noqa: BLE001
        logger.info("AutoEncoder 이상탐지 실패: %s", exc)
        return None


# ---------------------------------------------------------------------------
# 4) 통합 진입점
# ---------------------------------------------------------------------------


def detect_anomaly(
    frame: FeatureFrame,
    *,
    sklearn_path: str | None = None,
    autoencoder_path: str | None = None,
) -> AnomalyResult:
    """사용 가능한 최선의 방법으로 이상을 탐지한다.

    우선순위: AutoEncoder → sklearn → 통계 임계.
    AE 를 1순위로 두는 이유: 센서 간 상관관계를 학습하므로 "각 센서는 정상
    범위인데 조합이 이상한" 패턴(예: 저속인데 고온)을 유일하게 잡아낸다.

    Args:
        frame: FeatureFrame.
        sklearn_path: sklearn 모델 경로(선택).
        autoencoder_path: AE 체크포인트 경로(선택).

    Returns:
        AnomalyResult(항상 유효한 결과).
    """
    if autoencoder_path:
        result = autoencoder_anomaly(frame, autoencoder_path)
        if result is not None:
            return result
    if sklearn_path:
        result = sklearn_anomaly(frame, sklearn_path)
        if result is not None:
            return result
    return statistical_anomaly(frame)


__all__ = [
    "RISK_THRESHOLDS",
    "ZSCORE_THRESHOLD",
    "AnomalyResult",
    "autoencoder_anomaly",
    "build_autoencoder",
    "detect_anomaly",
    "sklearn_anomaly",
    "statistical_anomaly",
    "train_isolation_forest",
    "train_one_class_svm",
]
