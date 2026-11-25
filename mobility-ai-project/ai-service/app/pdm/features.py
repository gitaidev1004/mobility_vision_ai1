"""센서 시계열 특징 공학.

예지보전에서 특징 공학이 모델보다 중요한 이유:
    차량 센서의 "현재 값" 하나로는 고장을 예측할 수 없다. 엔진 온도 95도는
    정상일 수도 있고(여름 정체), 이상일 수도 있다(냉각수 부족). 결정적인 것은
    **추세(trend)** 와 **변동성(volatility)** 이다 — 온도가 지난 주보다 8도
    올랐다면 그것이 신호다. 이 모듈은 그런 파생 특징을 만든다.

모든 함수가 numpy 만 쓴다. pandas 가 있으면 더 편하지만, 의존성을 줄이기 위해
rolling 연산도 직접 구현했다(교재에서 "rolling 이 내부적으로 무엇을 하는가" 를
보여 주는 효과도 있다).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

#: 계약서 ``sensor_reading`` 테이블과 1:1 대응하는 센서 필드 목록.
#: 순서가 곧 특징 벡터의 열 순서다 — 학습/추론에서 반드시 일치해야 한다.
SENSOR_FIELDS: tuple[str, ...] = (
    "engine_temp_c",
    "engine_rpm",
    "speed_kph",
    "battery_voltage",
    "battery_soh",
    "tire_pressure_fl",
    "tire_pressure_fr",
    "tire_pressure_rl",
    "tire_pressure_rr",
    "brake_pad_mm",
    "oil_quality",
    "vibration_rms",
    "fuel_level",
)

#: 각 센서의 정상 범위 ``(하한, 상한)``. 도메인 지식에 기반한 값이다.
#: 이 범위는 두 곳에서 쓰인다 — ① 이상값(outlier) 클리핑, ② 규칙 기반 폴백 진단.
#:
#: 근거:
#:   engine_temp_c   : 정상 운전 온도 80~105도. 110 이상은 오버히트 경고.
#:   engine_rpm      : 아이들 600 ~ 레드존 6500.
#:   battery_voltage : 12V 납축전지. 12.4V 이상이 정상, 11.8V 미만은 방전.
#:   battery_soh     : State of Health. 0.8 미만이면 교체 권고 구간.
#:   tire_pressure_* : 승용차 권장 32~36 psi. 28 미만은 저압 경고.
#:   brake_pad_mm    : 신품 10~12mm, 3mm 미만은 법정 교체 기준.
#:   oil_quality     : 0(폐유)~1(신유). 0.3 미만 교체 권고.
#:   vibration_rms   : 정상 0.1~0.5g. 0.8 이상은 베어링/마운트 이상 신호.
SENSOR_RANGES: dict[str, tuple[float, float]] = {
    "engine_temp_c": (60.0, 120.0),
    "engine_rpm": (0.0, 7000.0),
    "speed_kph": (0.0, 200.0),
    "battery_voltage": (10.5, 15.0),
    "battery_soh": (0.0, 1.0),
    "tire_pressure_fl": (20.0, 45.0),
    "tire_pressure_fr": (20.0, 45.0),
    "tire_pressure_rl": (20.0, 45.0),
    "tire_pressure_rr": (20.0, 45.0),
    "brake_pad_mm": (0.0, 14.0),
    "oil_quality": (0.0, 1.0),
    "vibration_rms": (0.0, 3.0),
    "fuel_level": (0.0, 1.0),
}

#: 각 센서의 "정상 중심값". 결측 보간의 최후 수단으로 쓴다.
#: 평균 대신 도메인 기준값을 쓰는 이유: 데이터가 몇 건 없을 때 표본 평균은
#: 신뢰할 수 없고, 고장 데이터만 들어오면 "고장이 정상" 이 되어 버린다.
SENSOR_NOMINAL: dict[str, float] = {
    "engine_temp_c": 90.0,
    "engine_rpm": 1500.0,
    "speed_kph": 45.0,
    "battery_voltage": 12.6,
    "battery_soh": 0.95,
    "tire_pressure_fl": 33.0,
    "tire_pressure_fr": 33.0,
    "tire_pressure_rl": 33.0,
    "tire_pressure_rr": 33.0,
    "brake_pad_mm": 8.0,
    "oil_quality": 0.8,
    "vibration_rms": 0.25,
    "fuel_level": 0.5,
}


@dataclass(slots=True)
class FeatureFrame:
    """센서 시계열을 정리한 결과.

    Attributes:
        matrix: ``(N, F)`` float64 배열. 행 = 시점, 열 = ``SENSOR_FIELDS`` 순서.
        fields: 열 이름 튜플(= ``SENSOR_FIELDS``).
        timestamps: 각 행의 측정 시각(파싱 실패 시 None).
        imputed: 보간된 셀 수(데이터 품질 지표).
        clipped: 범위 밖으로 벗어나 클리핑된 셀 수.
        rows: 유효 행 수.
    """

    matrix: np.ndarray
    fields: tuple[str, ...] = SENSOR_FIELDS
    timestamps: list[datetime | None] = field(default_factory=list)
    imputed: int = 0
    clipped: int = 0
    rows: int = 0

    def column(self, name: str) -> np.ndarray:
        """센서 이름으로 열을 가져온다.

        Args:
            name: 센서 필드명.

        Returns:
            ``(N,)`` 배열. 없는 이름이면 빈 배열.
        """
        if name not in self.fields:
            return np.zeros((0,), dtype=np.float64)
        idx = self.fields.index(name)
        if self.matrix.ndim != 2 or self.matrix.shape[0] == 0:
            return np.zeros((0,), dtype=np.float64)
        return self.matrix[:, idx]

    def latest(self, name: str) -> float:
        """센서의 최신값을 반환한다(없으면 nominal 값).

        Args:
            name: 센서 필드명.

        Returns:
            최신 측정값.
        """
        col = self.column(name)
        if col.size == 0:
            return float(SENSOR_NOMINAL.get(name, 0.0))
        return float(col[-1])

    def quality_summary(self) -> dict[str, Any]:
        """데이터 품질 요약을 반환한다.

        보간/클리핑 비율이 높으면 예측 신뢰도가 낮다는 뜻이므로,
        응답에 포함해 운영자가 "센서를 고쳐야 한다" 를 알 수 있게 한다.

        Returns:
            행 수, 보간 수, 클리핑 수, 완전성 비율.
        """
        total_cells = max(1, self.rows * len(self.fields))
        return {
            "rows": self.rows,
            "imputed_cells": self.imputed,
            "clipped_cells": self.clipped,
            "completeness": round(1.0 - self.imputed / total_cells, 4),
        }


def parse_timestamp(value: Any) -> datetime | None:
    """ISO 8601 문자열 또는 datetime 을 ``datetime`` 으로 파싱한다.

    여러 포맷을 허용한다 — 백엔드(Jackson)는 ``2026-10-01T09:00:00`` 를 보내지만,
    DB 에서 직접 읽으면 ``2026-10-01 09:00:00`` 이 올 수 있고, 타임존 표기
    (``Z``, ``+09:00``)가 붙기도 한다. 파싱 실패는 예외가 아니라 None 이다 —
    타임스탬프가 없어도 센서값으로 예측은 가능하므로 요청 전체를 거부할 이유가 없다.

    Args:
        value: 문자열 또는 datetime.

    Returns:
        datetime 또는 None.
    """
    if isinstance(value, datetime):
        return value
    if not value:
        return None
    text = str(value).strip()
    # 'Z'(UTC) 는 fromisoformat 이 파이썬 3.11+ 에서만 지원하므로 명시 변환한다.
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    candidates = (text, text.replace(" ", "T"), text.split(".")[0])
    for candidate in candidates:
        try:
            return datetime.fromisoformat(candidate)
        except (ValueError, TypeError):
            continue
    return None


def build_feature_frame(readings: list[dict[str, Any]]) -> FeatureFrame:
    """센서 판독 리스트를 ``FeatureFrame`` 으로 정리한다.

    처리 순서(순서가 중요하다):
        1. 시간순 정렬 — 추세 계산은 시간순을 전제한다. 백엔드가 순서를
           보장하지 않을 수 있으므로 여기서 한 번 정렬한다.
        2. 결측 보간 — **전방 채움(forward fill) → 후방 채움 → nominal**.
           왜 선형 보간이 아니라 전방 채움인가? 센서 결측은 보통 통신 단절이고,
           그 구간의 실제 값은 "직전 값과 비슷했을 것" 이 가장 안전한 가정이다.
           선형 보간은 존재하지 않는 추세를 만들어 내 이상탐지를 오작동시킨다.
        3. 이상값 클리핑 — 센서 오류(예: 온도 -999)를 물리적 범위로 자른다.
           제거(행 삭제)하지 않는 이유: 다른 센서 값은 멀쩡하므로 행 전체를
           버리면 정보 손실이 더 크다.

    Args:
        readings: ``{"measured_at": ..., "engine_temp_c": ..., ...}`` 리스트.

    Returns:
        FeatureFrame. 입력이 비면 ``rows=0`` 인 빈 프레임.
    """
    rows = list(readings or [])
    if not rows:
        return FeatureFrame(
            matrix=np.zeros((0, len(SENSOR_FIELDS)), dtype=np.float64), rows=0
        )

    # --- 1. 시간순 정렬 ---
    parsed: list[tuple[datetime | None, dict[str, Any]]] = [
        (parse_timestamp(r.get("measured_at")), r) for r in rows if isinstance(r, dict)
    ]
    # 타임스탬프가 없는 행은 원래 순서를 유지해야 하므로, 정렬 키에
    # (타임스탬프 유무, 타임스탬프, 원래 인덱스) 를 쓴다.
    indexed = [(i, ts, r) for i, (ts, r) in enumerate(parsed)]
    indexed.sort(key=lambda item: (item[1] is None, item[1] or datetime.min, item[0]))

    timestamps = [ts for _i, ts, _r in indexed]
    raw = np.full((len(indexed), len(SENSOR_FIELDS)), np.nan, dtype=np.float64)
    for row_idx, (_i, _ts, record) in enumerate(indexed):
        for col_idx, name in enumerate(SENSOR_FIELDS):
            value = record.get(name)
            if value is None:
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(number):
                raw[row_idx, col_idx] = number

    # --- 2. 결측 보간 ---
    missing_before = int(np.isnan(raw).sum())
    filled = _forward_backward_fill(raw)
    for col_idx, name in enumerate(SENSOR_FIELDS):
        column = filled[:, col_idx]
        still_nan = np.isnan(column)
        if np.any(still_nan):
            column[still_nan] = float(SENSOR_NOMINAL.get(name, 0.0))

    # --- 3. 이상값 클리핑 ---
    clipped_count = 0
    for col_idx, name in enumerate(SENSOR_FIELDS):
        lo, hi = SENSOR_RANGES.get(name, (-np.inf, np.inf))
        column = filled[:, col_idx]
        out_of_range = (column < lo) | (column > hi)
        clipped_count += int(out_of_range.sum())
        filled[:, col_idx] = np.clip(column, lo, hi)

    return FeatureFrame(
        matrix=filled,
        fields=SENSOR_FIELDS,
        timestamps=timestamps,
        imputed=missing_before,
        clipped=clipped_count,
        rows=filled.shape[0],
    )


def _forward_backward_fill(matrix: np.ndarray) -> np.ndarray:
    """NaN 을 전방 채움 후 후방 채움으로 보간한다.

    전방 채움을 numpy 로 구현하는 방법(pandas 없이):
        각 열에서 "유효 값의 인덱스" 를 누적 최대값으로 전파한다.
        ``np.maximum.accumulate`` 가 핵심이다 — NaN 위치에는 직전 유효 인덱스가
        남으므로 그 인덱스의 값을 가져오면 전방 채움이 된다.

    Args:
        matrix: ``(N, F)`` NaN 포함 배열.

    Returns:
        보간된 새 배열(원본은 변경하지 않는다).
    """
    out = np.array(matrix, dtype=np.float64, copy=True)
    n_rows = out.shape[0]
    if n_rows == 0:
        return out

    for col in range(out.shape[1]):
        column = out[:, col]
        valid = ~np.isnan(column)
        if not np.any(valid):
            continue
        # 전방 채움: 유효 인덱스를 누적 최대로 전파.
        indices = np.where(valid, np.arange(n_rows), -1)
        forward_idx = np.maximum.accumulate(indices)
        has_prev = forward_idx >= 0
        column[has_prev] = column[forward_idx[has_prev]]
        # 후방 채움: 뒤에서부터 같은 방법을 쓴다(선두 NaN 처리).
        still_nan = np.isnan(column)
        if np.any(still_nan):
            rev_indices = np.where(valid[::-1], np.arange(n_rows), -1)
            backward_idx = np.maximum.accumulate(rev_indices)[::-1]
            backward_idx = n_rows - 1 - backward_idx
            can_fill = still_nan & (backward_idx >= 0) & (backward_idx < n_rows)
            column[can_fill] = column[backward_idx[can_fill]]
        out[:, col] = column
    return out


def moving_average(series: np.ndarray, window: int = 5) -> np.ndarray:
    """이동평균을 계산한다(끝단은 사용 가능한 값만으로 평균).

    왜 끝단을 버리지 않는가? 예지보전에서 가장 중요한 값은 **가장 최근 값**이다.
    ``valid`` 모드 합성곱은 앞뒤 ``window-1`` 개를 버려 최신 추세를 잃는다.
    누적합 기반으로 구현해 모든 시점에 값을 준다.

    Args:
        series: ``(N,)`` 1차원 배열.
        window: 윈도우 크기.

    Returns:
        ``(N,)`` 이동평균 배열.
    """
    arr = np.asarray(series, dtype=np.float64).ravel()
    n = arr.size
    if n == 0:
        return arr
    w = max(1, min(int(window), n))
    # 누적합으로 O(N) 계산. 각 시점 i 의 윈도우는 [max(0, i-w+1), i].
    cumulative = np.concatenate([[0.0], np.cumsum(arr)])
    starts = np.maximum(0, np.arange(n) - w + 1)
    ends = np.arange(n) + 1
    sums = cumulative[ends] - cumulative[starts]
    counts = (ends - starts).astype(np.float64)
    return sums / counts


def rolling_stat(series: np.ndarray, window: int = 5, stat: str = "std") -> np.ndarray:
    """rolling 통계를 계산한다.

    Args:
        series: 1차원 배열.
        window: 윈도우 크기.
        stat: ``"std"`` | ``"min"`` | ``"max"`` | ``"mean"`` | ``"range"``.

    Returns:
        ``(N,)`` 통계 배열. 알 수 없는 ``stat`` 은 평균으로 처리한다.
    """
    arr = np.asarray(series, dtype=np.float64).ravel()
    n = arr.size
    if n == 0:
        return arr
    w = max(1, min(int(window), n))
    mode = str(stat).lower()

    if mode == "mean":
        return moving_average(arr, w)

    out = np.zeros(n, dtype=np.float64)
    for i in range(n):
        chunk = arr[max(0, i - w + 1) : i + 1]
        if mode == "std":
            # ddof=0 (모표준편차). 윈도우가 1 일 때 NaN 이 되지 않게 하기 위함.
            out[i] = float(chunk.std())
        elif mode == "min":
            out[i] = float(chunk.min())
        elif mode == "max":
            out[i] = float(chunk.max())
        elif mode == "range":
            out[i] = float(chunk.max() - chunk.min())
        else:
            out[i] = float(chunk.mean())
    return out


def trend_slope(series: np.ndarray, window: int | None = None) -> float:
    """선형 회귀 기울기로 추세를 추정한다(단위: 값/샘플).

    최소제곱 해를 직접 계산한다::

        slope = Σ((x - x̄)(y - ȳ)) / Σ((x - x̄)²)

    왜 "마지막 값 - 첫 값" 이 아니라 회귀인가? 두 점 차이는 노이즈 한 번에
    완전히 왜곡된다. 회귀 기울기는 모든 점을 쓰므로 노이즈에 강하다.

    Args:
        series: 1차원 배열.
        window: 최근 N개만 사용(None 이면 전체).

    Returns:
        기울기. 데이터가 2개 미만이면 0.0.
    """
    arr = np.asarray(series, dtype=np.float64).ravel()
    if window is not None and int(window) > 0:
        arr = arr[-int(window) :]
    n = arr.size
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=np.float64)
    x_mean = x.mean()
    y_mean = arr.mean()
    denominator = float(((x - x_mean) ** 2).sum())
    if denominator <= 1e-12:
        return 0.0
    return float(((x - x_mean) * (arr - y_mean)).sum() / denominator)


def zscore(value: float, series: np.ndarray, *, min_std: float = 1e-6) -> float:
    """값의 z-score 를 계산한다.

    Args:
        value: 평가 대상 값.
        series: 기준 분포.
        min_std: 표준편차 하한(0 나눗셈 방지).

    Returns:
        z-score. 데이터가 2개 미만이면 0.0.
    """
    arr = np.asarray(series, dtype=np.float64).ravel()
    if arr.size < 2:
        return 0.0
    std = max(float(arr.std()), float(min_std))
    return float((float(value) - float(arr.mean())) / std)


def make_windows(
    matrix: np.ndarray, window: int = 10, stride: int = 1
) -> np.ndarray:
    """시계열을 LSTM 학습/추론용 윈도우로 자른다.

    데이터가 윈도우보다 짧으면 **앞쪽을 첫 행으로 패딩**해 최소 1개 윈도우를 만든다.
    왜 패딩하는가? 신차처럼 데이터가 3건뿐인 차량도 예측값을 받아야 한다.
    "데이터 부족" 으로 거부하면 서비스가 쓸모없어진다 — 대신 신뢰도를 낮춘다.

    Args:
        matrix: ``(N, F)`` 특징 행렬.
        window: 윈도우 길이.
        stride: 윈도우 간 간격.

    Returns:
        ``(W, window, F)`` 배열. 입력이 비면 ``(0, window, F)``.
    """
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.ndim != 2:
        arr = arr.reshape(-1, len(SENSOR_FIELDS))
    n_rows, n_features = arr.shape
    w = max(1, int(window))
    s = max(1, int(stride))

    if n_rows == 0:
        return np.zeros((0, w, n_features), dtype=np.float64)

    if n_rows < w:
        # 앞쪽 패딩: 첫 행을 반복한다(과거는 현재와 비슷했다는 가정).
        pad = np.repeat(arr[:1], w - n_rows, axis=0)
        padded = np.concatenate([pad, arr], axis=0)
        return padded[None, :, :]

    starts = list(range(0, n_rows - w + 1, s))
    return np.stack([arr[i : i + w] for i in starts], axis=0)


def aggregate_features(frame: FeatureFrame, *, window: int = 5) -> dict[str, float]:
    """센서별 집계 특징을 만든다(이상탐지/고장분류 모델 입력).

    센서마다 4개 특징을 만든다:
        - ``<name>_last``  : 최신값 — 현재 상태
        - ``<name>_mean``  : 최근 윈도우 평균 — 노이즈 제거된 수준
        - ``<name>_std``   : 최근 윈도우 표준편차 — 변동성(불안정 신호)
        - ``<name>_slope`` : 추세 기울기 — 악화 속도

    왜 이 4개인가? 고장은 "수준이 나쁨"(last/mean), "불안정함"(std),
    "빠르게 나빠짐"(slope) 세 가지 방식으로 드러난다. 이 조합이
    도메인 전문가가 계기판을 볼 때 실제로 보는 것과 일치한다.

    Args:
        frame: FeatureFrame.
        window: 집계 윈도우 크기.

    Returns:
        ``{특징명: 값}`` dict (센서 13개 x 4 = 52개 + 파생 4개).
    """
    out: dict[str, float] = {}
    for name in frame.fields:
        column = frame.column(name)
        if column.size == 0:
            nominal = float(SENSOR_NOMINAL.get(name, 0.0))
            out[f"{name}_last"] = nominal
            out[f"{name}_mean"] = nominal
            out[f"{name}_std"] = 0.0
            out[f"{name}_slope"] = 0.0
            continue
        recent = column[-max(1, int(window)) :]
        out[f"{name}_last"] = float(column[-1])
        out[f"{name}_mean"] = float(recent.mean())
        out[f"{name}_std"] = float(recent.std())
        out[f"{name}_slope"] = trend_slope(column, window=max(3, int(window) * 2))

    # --- 파생 특징: 단일 센서로는 보이지 않는 복합 신호 ---
    pressures = np.array(
        [
            out["tire_pressure_fl_last"],
            out["tire_pressure_fr_last"],
            out["tire_pressure_rl_last"],
            out["tire_pressure_rr_last"],
        ],
        dtype=np.float64,
    )
    # 타이어 압력 **불균형**. 네 바퀴 모두 30psi 면 "약간 부족" 이지만,
    # 하나만 24psi 면 펑크/밸브 누설이라는 전혀 다른 문제다. 평균만 보면 놓친다.
    out["tire_pressure_imbalance"] = float(pressures.max() - pressures.min())
    out["tire_pressure_mean"] = float(pressures.mean())

    # 엔진 부하 지표: 같은 RPM 에서 온도가 높으면 냉각 성능 저하를 의미한다.
    rpm = max(1.0, out["engine_rpm_mean"])
    out["thermal_load_index"] = float(out["engine_temp_c_mean"] / (rpm / 1000.0))

    # 배터리 전압 x SOH: 둘 다 낮으면 위험이 곱으로 커진다.
    out["battery_health_index"] = float(
        out["battery_voltage_last"] / 12.6 * out["battery_soh_last"]
    )
    return out


def feature_vector(frame: FeatureFrame, *, window: int = 5) -> tuple[np.ndarray, list[str]]:
    """집계 특징을 고정 순서 벡터로 변환한다(모델 입력용).

    **정렬된 키 순서**를 쓰는 이유: dict 순서에 의존하면 파이썬 버전이나
    코드 수정으로 열 순서가 바뀌어, 학습된 모델이 조용히 틀린 열을 보게 된다.
    정렬은 결정적이므로 이 사고를 구조적으로 막는다.

    Args:
        frame: FeatureFrame.
        window: 집계 윈도우.

    Returns:
        ``(벡터 (F,), 특징명 리스트)``.
    """
    features = aggregate_features(frame, window=window)
    names = sorted(features.keys())
    return np.array([features[k] for k in names], dtype=np.float64), names


__all__ = [
    "SENSOR_FIELDS",
    "SENSOR_NOMINAL",
    "SENSOR_RANGES",
    "FeatureFrame",
    "aggregate_features",
    "build_feature_frame",
    "feature_vector",
    "make_windows",
    "moving_average",
    "parse_timestamp",
    "rolling_stat",
    "trend_slope",
    "zscore",
]
