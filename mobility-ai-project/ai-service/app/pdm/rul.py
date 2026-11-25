"""RUL(Remaining Useful Life, 잔여수명) 예측: LSTM + 선형 외삽 폴백.

RUL 이란?
    "이 부품이 교체 기준에 도달하기까지 남은 시간/주행거리" 다.
    이상탐지가 "지금 이상한가", 고장분류가 "어디가 고장날까" 를 답한다면,
    RUL 은 **"언제 조치해야 하나"** 를 답한다. 정비 일정을 세우는 데 직접 쓰인다.

두 접근:
    1. **선형 외삽(폴백)** — 소모성 부품에는 놀랍도록 잘 맞는다.
       브레이크 패드와 엔진오일은 거의 선형으로 마모/열화한다.
       "현재 4.1mm, 하루 0.05mm 감소, 기준 3.0mm → 22일" 이라는 계산은
       물리적으로 타당하고 정비사도 같은 방식으로 추정한다.
    2. **LSTM(가중치 있을 때)** — 비선형 열화와 센서 간 상호작용을 학습한다.
       배터리는 온도/충방전 패턴에 따라 비선형으로 노화하므로 LSTM 이 유리하다.

왜 선형 폴백을 "임시 방편" 이상으로 보는가?
    NASA C-MAPSS 같은 벤치마크에서도 잘 설계된 선형/통계 모델이 딥러닝과
    비슷한 성능을 내는 경우가 많다. 데이터가 적을 때는 선형이 더 안전하다
    (과적합 위험이 없다). 교재에서 이 점을 강조할 가치가 있다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from app.core.logging import get_logger
from app.pdm.features import FeatureFrame, aggregate_features, make_windows, trend_slope

logger = get_logger(__name__)

#: RUL 상한(일). 2년 이상은 "충분히 멀다" 로 보고 더 세분하지 않는다.
#: 왜 상한을 두는가? 열화 추세가 거의 0 이면 선형 외삽이 수만 일을 내놓는데,
#: 그런 숫자는 정보가 아니라 노이즈다. 운영자에게 "730일 이상" 이 더 정직하다.
MAX_RUL_DAYS: int = 730

#: RUL 하한(일). 음수나 0 은 "이미 기준 초과" 를 뜻하므로 1 로 자른다.
MIN_RUL_DAYS: int = 1

#: 부품별 **수명 종료 기준값**과 **센서 필드**. ``(센서, 기준값, 방향)``.
#: 방향 ``"decreasing"`` = 값이 줄어서 기준에 도달, ``"increasing"`` = 늘어서 도달.
#:
#: 근거:
#:   brake_pad_mm 3.0 — 법정 교체 기준
#:   oil_quality  0.25 — 윤활 성능 급락 구간
#:   battery_soh  0.70 — 시동 불량 발생 시작점
#:   engine_temp  110 — 오버히트 경고 온도
#:   vibration    1.00 — 구동계 손상 진행 구간
END_OF_LIFE: dict[str, tuple[str, float, str]] = {
    "brake": ("brake_pad_mm", 3.0, "decreasing"),
    "oil": ("oil_quality", 0.25, "decreasing"),
    "battery": ("battery_soh", 0.70, "decreasing"),
    "engine": ("engine_temp_c", 110.0, "increasing"),
    "transmission": ("vibration_rms", 1.00, "increasing"),
    "tire": ("tire_pressure_mean", 24.0, "decreasing"),
}

#: 센서 샘플 1건이 몇 일에 해당하는지의 기본 가정.
#: 계약서 ``sensor_reading`` 은 측정 시각을 주므로 실제 간격을 계산하지만,
#: 타임스탬프가 없을 때의 폴백 값이 필요하다. 일 1회 수집을 가정한다.
DEFAULT_SAMPLE_INTERVAL_DAYS: float = 1.0


@dataclass(slots=True)
class RulResult:
    """RUL 예측 결과.

    Attributes:
        rul_days: 종합 잔여수명(일). 가장 임박한 부품 기준.
        limiting_component: 수명을 결정하는(가장 임박한) 부품.
        per_component: 부품별 RUL ``{부품: 일수}``.
        method: 사용한 방법.
        confidence: 0~1 예측 신뢰도. 데이터 수와 추세 명확성으로 결정한다.
        detail: 추가 정보(샘플 간격, 추세 등).
    """

    rul_days: int
    limiting_component: str
    per_component: dict[str, int] = field(default_factory=dict)
    method: str = "linear-extrapolation"
    confidence: float = 0.0
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON 직렬화용 dict 를 반환한다."""
        return {
            "rul_days": int(self.rul_days),
            "limiting_component": self.limiting_component,
            "per_component": {k: int(v) for k, v in self.per_component.items()},
            "method": self.method,
            "confidence": round(float(self.confidence), 4),
            "detail": self.detail,
        }


def estimate_sample_interval_days(frame: FeatureFrame) -> float:
    """센서 샘플 간 평균 간격(일)을 추정한다.

    **중앙값**을 쓰는 이유: 수집이 며칠 끊겼다가 재개되면 그 한 번의 큰 간격이
    평균을 왜곡한다. 중앙값은 "보통 얼마나 자주 수집되는가" 를 정확히 반영한다.

    Args:
        frame: FeatureFrame(타임스탬프 포함).

    Returns:
        평균 간격(일). 추정 불가 시 ``DEFAULT_SAMPLE_INTERVAL_DAYS``.
    """
    stamps = [t for t in (frame.timestamps or []) if t is not None]
    if len(stamps) < 2:
        return DEFAULT_SAMPLE_INTERVAL_DAYS

    deltas: list[float] = []
    for prev, curr in zip(stamps, stamps[1:], strict=False):
        try:
            seconds = (curr - prev).total_seconds()
        except (TypeError, ValueError):
            continue
        if seconds > 0:
            deltas.append(seconds / 86400.0)

    if not deltas:
        return DEFAULT_SAMPLE_INTERVAL_DAYS
    median = float(np.median(deltas))
    # 간격이 비현실적이면(1분 미만 또는 1년 초과) 기본값으로 되돌린다.
    if median < (1.0 / 1440.0) or median > 365.0:
        return DEFAULT_SAMPLE_INTERVAL_DAYS
    return median


def linear_rul(frame: FeatureFrame) -> RulResult:
    """선형 외삽으로 부품별 RUL 을 계산한다(가중치 불필요).

    계산::

        남은 여유 = |현재값 - 기준값|
        변화율    = |추세 기울기| / 샘플당 일수      (단위: 값/일)
        RUL(일)   = 남은 여유 / 변화율

    경계 처리가 중요하다:
        - **이미 기준 초과** → RUL = 1 일(즉시 조치). 음수를 주면 호출자가
          "수명이 남았다" 로 오해할 수 있다.
        - **추세가 거의 0** → ``MAX_RUL_DAYS``. "변하지 않으니 영원히 괜찮다" 는
          결론은 위험하지만, 데이터로는 그 이상 말할 수 없다. 상한으로 정직하게 표현한다.
        - **추세가 개선 방향** → ``MAX_RUL_DAYS``. 정비 직후 패드가 두꺼워진
          경우가 실제로 있다(교체). 이를 "음수 RUL" 로 만들면 안 된다.

    Args:
        frame: 센서 FeatureFrame.

    Returns:
        RulResult(method=``"linear-extrapolation"``).
    """
    if frame.rows == 0:
        return RulResult(
            rul_days=MAX_RUL_DAYS,
            limiting_component="unknown",
            method="linear-extrapolation(no-data)",
            confidence=0.0,
            detail={"reason": "no-data"},
        )

    features = aggregate_features(frame)
    interval = estimate_sample_interval_days(frame)
    per_component: dict[str, int] = {}
    trends: dict[str, float] = {}

    for component, (sensor, limit, direction) in END_OF_LIFE.items():
        # tire_pressure_mean 은 파생 특징이므로 집계값에서 가져온다.
        if sensor == "tire_pressure_mean":
            current = float(features.get("tire_pressure_mean", 33.0))
            series = np.array(
                [
                    frame.column("tire_pressure_fl"),
                    frame.column("tire_pressure_fr"),
                    frame.column("tire_pressure_rl"),
                    frame.column("tire_pressure_rr"),
                ]
            ).mean(axis=0)
            slope = trend_slope(series)
        else:
            column = frame.column(sensor)
            if column.size == 0:
                per_component[component] = MAX_RUL_DAYS
                trends[component] = 0.0
                continue
            current = float(column[-1])
            slope = trend_slope(column)

        trends[component] = slope
        remaining = (current - limit) if direction == "decreasing" else (limit - current)

        if remaining <= 0.0:
            # 이미 기준 도달/초과 → 즉시 조치.
            per_component[component] = MIN_RUL_DAYS
            continue

        # 악화 방향의 기울기만 의미가 있다.
        worsening = -slope if direction == "decreasing" else slope
        rate_per_day = worsening / max(1e-9, interval)

        if rate_per_day <= 1e-9:
            per_component[component] = MAX_RUL_DAYS
            continue

        days = remaining / rate_per_day
        per_component[component] = int(np.clip(round(days), MIN_RUL_DAYS, MAX_RUL_DAYS))

    if not per_component:
        return RulResult(
            rul_days=MAX_RUL_DAYS,
            limiting_component="unknown",
            method="linear-extrapolation",
            confidence=0.0,
        )

    limiting = min(per_component, key=lambda k: per_component[k])
    rul_days = per_component[limiting]

    return RulResult(
        rul_days=rul_days,
        limiting_component=limiting,
        per_component=per_component,
        method="linear-extrapolation",
        confidence=_linear_confidence(frame, trends.get(limiting, 0.0)),
        detail={
            "sample_interval_days": round(interval, 4),
            "slopes": {k: round(v, 6) for k, v in trends.items()},
            "weights_required": False,
        },
    )


def _linear_confidence(frame: FeatureFrame, slope: float) -> float:
    """선형 외삽 RUL 의 신뢰도를 추정한다.

    두 요소를 본다:
        1. **데이터 수** — 3건으로 추세를 외삽하는 것은 거의 추측이다.
           20건 이상이면 추세가 꽤 안정적이다.
        2. **추세 명확성** — 기울기가 0 에 가까우면 "변화 없음" 인지
           "노이즈에 묻힘" 인지 구별할 수 없어 예측이 불확실하다.

    데이터 품질(보간 비율)도 곱해 반영한다 — 절반이 보간된 데이터로 낸
    예측은 신뢰할 수 없다.

    Args:
        frame: FeatureFrame.
        slope: 결정 부품의 추세 기울기.

    Returns:
        0~1 신뢰도.
    """
    # 데이터 수: 3건 → 0.2, 20건 → 1.0 로 포화.
    sample_score = float(np.clip((frame.rows - 2) / 18.0, 0.0, 1.0))
    # 추세 명확성: |slope| 가 클수록 명확. 스케일은 센서마다 달라
    # 정규화가 어려우므로, "0 이 아니다" 정도만 반영한다.
    trend_score = 1.0 if abs(float(slope)) > 1e-6 else 0.35
    completeness = float(frame.quality_summary().get("completeness", 1.0))
    return float(np.clip(0.55 * sample_score + 0.25 * trend_score + 0.20 * completeness, 0.0, 1.0))


# ---------------------------------------------------------------------------
# LSTM RUL
# ---------------------------------------------------------------------------


def build_lstm_rul(input_dim: int, *, hidden_size: int = 64, num_layers: int = 2) -> Any:
    """LSTM 기반 RUL 회귀 모델을 만든다.

    구조: ``(B, T, F) → LSTM(64) x2 → 마지막 시점 hidden → Linear(32) → Linear(1)``

    **마지막 시점의 은닉 상태만 쓰는 이유**:
        RUL 은 시퀀스 하나에 대한 단일 숫자다(sequence-to-one). LSTM 의 마지막
        은닉 상태는 시퀀스 전체 정보를 요약하고 있으므로 이것만으로 충분하다.
        모든 시점을 평균하면 오래된 정보가 최신 상태를 희석시킨다 —
        열화 예측에서는 최신 상태가 압도적으로 중요하다.

    출력에 ``ReLU`` 를 쓰지 않는 이유: 학습 중 음수 예측이 나오면 그 기울기로
    학습이 계속되어야 한다. ReLU 로 막으면 기울기가 0 이 되어 죽는다.
    음수 클리핑은 **추론 시점**에만 한다.

    Args:
        input_dim: 시점당 특징 수.
        hidden_size: LSTM 은닉 차원.
        num_layers: LSTM 층 수.

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

    class LstmRul(nn.Module):
        """센서 시계열 → 잔여수명(일) 회귀 모델."""

        def __init__(self) -> None:
            """계층을 구성한다."""
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=max(1, int(input_dim)),
                hidden_size=int(hidden_size),
                num_layers=int(num_layers),
                batch_first=True,
                dropout=0.1 if int(num_layers) > 1 else 0.0,
            )
            self.head = nn.Sequential(
                nn.Linear(int(hidden_size), 32),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(32, 1),
            )

        def forward(self, x: Any) -> Any:
            """순전파.

            Args:
                x: ``(B, T, F)`` 시계열 텐서.

            Returns:
                ``(B,)`` RUL 예측(정규화 스케일).
            """
            output, _ = self.lstm(x)
            # 마지막 시점의 출력을 쓴다.
            return self.head(output[:, -1, :]).squeeze(-1)

    return LstmRul()


def lstm_rul(frame: FeatureFrame, weights_path: str, *, window: int = 10) -> RulResult | None:
    """학습된 LSTM 으로 RUL 을 예측한다.

    체크포인트 필수 키::

        {"model": state_dict, "input_dim": int, "window": int,
         "feature_mean": [...], "feature_std": [...],
         "rul_scale": float}   # 예측값을 일(day) 단위로 환산하는 배율

    ``rul_scale`` 이 필요한 이유: 회귀 타깃을 0~1 로 정규화해 학습하는 것이
    수렴에 유리하다. 그러면 추론 결과를 다시 일 단위로 되돌려야 한다.
    이 배율을 체크포인트에 넣지 않으면 "0.37일" 같은 무의미한 값이 나온다.

    Args:
        frame: FeatureFrame.
        weights_path: ``.pt`` 경로.
        window: 입력 윈도우 길이(체크포인트 값이 우선).

    Returns:
        RulResult 또는 None(torch/가중치 없음 → 호출자가 폴백).
    """
    if not Path(str(weights_path)).is_file():
        return None
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        return None
    if frame.rows == 0:
        return None

    try:
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict) or "model" not in checkpoint:
            return None

        win = int(checkpoint.get("window", window))
        windows = make_windows(frame.matrix, window=win)
        if windows.shape[0] == 0:
            return None
        # 가장 최근 윈도우만 쓴다(현재 상태 예측이므로).
        latest = windows[-1]

        mean = np.asarray(checkpoint.get("feature_mean", np.zeros(latest.shape[1])), dtype=np.float64)
        std = np.asarray(checkpoint.get("feature_std", np.ones(latest.shape[1])), dtype=np.float64)
        std = np.where(std <= 1e-9, 1.0, std)
        normalized = (latest - mean) / std

        model = build_lstm_rul(
            int(checkpoint.get("input_dim", latest.shape[1])),
            hidden_size=int(checkpoint.get("hidden_size", 64)),
            num_layers=int(checkpoint.get("num_layers", 2)),
        )
        model.load_state_dict(checkpoint["model"], strict=True)
        model.eval()

        tensor = torch.from_numpy(normalized.astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            raw = float(model(tensor)[0].item())

        scale = float(checkpoint.get("rul_scale", 1.0)) or 1.0
        days = int(np.clip(round(raw * scale), MIN_RUL_DAYS, MAX_RUL_DAYS))

        # 어느 부품이 제한 요소인지는 LSTM 이 말해 주지 않는다.
        # 선형 외삽에서 가져와 결합한다(하이브리드 설명 가능성).
        linear = linear_rul(frame)
        return RulResult(
            rul_days=days,
            limiting_component=linear.limiting_component,
            per_component=linear.per_component,
            method="lstm",
            # LSTM 은 학습 데이터 분포 내에서는 선형보다 신뢰할 만하다.
            confidence=float(np.clip(linear.confidence * 0.5 + 0.5, 0.0, 1.0)),
            detail={
                "raw_output": round(raw, 6),
                "rul_scale": scale,
                "window": win,
                "linear_reference_days": linear.rul_days,
            },
        )
    except Exception as exc:  # noqa: BLE001
        logger.info("LSTM RUL 예측 실패: %s", exc)
        return None


def predict_rul(frame: FeatureFrame, *, weights_path: str | None = None) -> RulResult:
    """사용 가능한 최선의 방법으로 RUL 을 예측한다.

    Args:
        frame: FeatureFrame.
        weights_path: LSTM 체크포인트 경로(선택).

    Returns:
        RulResult(항상 유효).
    """
    if weights_path:
        result = lstm_rul(frame, weights_path)
        if result is not None:
            return result
    return linear_rul(frame)


def rul_to_km(rul_days: int, *, daily_km: float = 45.0) -> int:
    """RUL(일)을 주행거리(km)로 환산한다.

    계약서 ``recommendations[].due_km`` 를 채우는 데 쓴다.
    기본 일일 주행거리 45km 는 한국 승용차 연평균 주행거리(약 1.5만 km)에서
    나온 값이다. 차량별 실측 평균이 있으면 그것을 넘기는 것이 정확하다.

    Args:
        rul_days: 잔여일수.
        daily_km: 일 평균 주행거리.

    Returns:
        잔여 주행거리(km, 100 단위로 내림 — 과도한 정밀도는 거짓 신뢰를 준다).
    """
    km = max(0.0, float(rul_days) * float(daily_km))
    return int(km // 100 * 100)


__all__ = [
    "DEFAULT_SAMPLE_INTERVAL_DAYS",
    "END_OF_LIFE",
    "MAX_RUL_DAYS",
    "MIN_RUL_DAYS",
    "RulResult",
    "build_lstm_rul",
    "estimate_sample_interval_days",
    "linear_rul",
    "lstm_rul",
    "predict_rul",
    "rul_to_km",
]
