# -*- coding: utf-8 -*-
"""OBD-II 응답 바이트를 물리값으로 변환한다.

OBD-II(On-Board Diagnostics II)는 Mode 01(현재 데이터) 요청에 대해
`41 <PID> <A> <B> ...` 형태의 응답을 돌려준다.
여기서는 예지보전에 자주 쓰이는 PID 를 디코딩하는 함수를 구현한다.

표준 라이브러리만 사용하므로 추가 설치 없이 실행된다.
"""
from __future__ import annotations

from typing import Callable, NamedTuple


class PidSpec(NamedTuple):
    """PID 한 개의 메타 정보."""
    pid: int                      # PID 번호
    name: str                     # 신호 이름
    n_bytes: int                  # 데이터 바이트 수
    unit: str                     # 물리 단위
    decode: Callable[..., float]  # 바이트 -> 물리값 변환 함수
    lo: float                     # 물리 범위 하한
    hi: float                     # 물리 범위 상한


# ----------------------------------------------------------------------
# PID 별 변환식 (SAE J1979 표준)
# ----------------------------------------------------------------------
def dec_coolant_temp(a: int) -> float:
    """0x05 엔진 냉각수 온도: A - 40 (섭씨)."""
    return a - 40.0


def dec_engine_rpm(a: int, b: int) -> float:
    """0x0C 엔진 회전수: (256*A + B) / 4 (rpm)."""
    return (256 * a + b) / 4.0


def dec_vehicle_speed(a: int) -> float:
    """0x0D 차량 속도: A (km/h)."""
    return float(a)


def dec_intake_temp(a: int) -> float:
    """0x0F 흡기 온도: A - 40 (섭씨)."""
    return a - 40.0


def dec_engine_load(a: int) -> float:
    """0x04 계산된 엔진 부하: A * 100 / 255 (%)."""
    return a * 100.0 / 255.0


def dec_fuel_level(a: int) -> float:
    """0x2F 연료 탱크 잔량: A * 100 / 255 (%)."""
    return a * 100.0 / 255.0


def dec_module_voltage(a: int, b: int) -> float:
    """0x42 제어 모듈 전압: (256*A + B) / 1000 (V)."""
    return (256 * a + b) / 1000.0


def dec_throttle(a: int) -> float:
    """0x11 스로틀 위치: A * 100 / 255 (%)."""
    return a * 100.0 / 255.0


def dec_maf(a: int, b: int) -> float:
    """0x10 공기 흐름량(MAF): (256*A + B) / 100 (g/s)."""
    return (256 * a + b) / 100.0


def dec_runtime(a: int, b: int) -> float:
    """0x1F 엔진 시동 후 경과 시간: 256*A + B (초)."""
    return float(256 * a + b)


PID_TABLE: dict[int, PidSpec] = {
    0x04: PidSpec(0x04, "engine_load", 1, "%", dec_engine_load, 0, 100),
    0x05: PidSpec(0x05, "engine_temp_c", 1, "℃", dec_coolant_temp, -40, 215),
    0x0C: PidSpec(0x0C, "engine_rpm", 2, "rpm", dec_engine_rpm, 0, 16383),
    0x0D: PidSpec(0x0D, "speed_kph", 1, "km/h", dec_vehicle_speed, 0, 255),
    0x0F: PidSpec(0x0F, "intake_temp_c", 1, "℃", dec_intake_temp, -40, 215),
    0x10: PidSpec(0x10, "maf_gps", 2, "g/s", dec_maf, 0, 655.35),
    0x11: PidSpec(0x11, "throttle_pct", 1, "%", dec_throttle, 0, 100),
    0x1F: PidSpec(0x1F, "runtime_s", 2, "s", dec_runtime, 0, 65535),
    0x2F: PidSpec(0x2F, "fuel_level", 1, "%", dec_fuel_level, 0, 100),
    0x42: PidSpec(0x42, "battery_voltage", 2, "V", dec_module_voltage, 0, 65.5),
}


class ObdDecodeError(ValueError):
    """OBD 응답을 해석할 수 없을 때 발생한다."""


def parse_response(raw: str) -> tuple[str, float, str]:
    """`41 05 5A` 같은 문자열 응답을 (신호명, 물리값, 단위) 로 변환한다.

    raw : 공백으로 구분된 16진수 문자열. ELM327 계열 어댑터 출력 형식.
    """
    tokens = raw.replace("\r", " ").replace("\n", " ").split()
    try:
        data = [int(t, 16) for t in tokens]
    except ValueError as exc:
        raise ObdDecodeError(f"16진수가 아닌 토큰이 있다: {raw!r}") from exc

    if len(data) < 2:
        raise ObdDecodeError(f"응답이 너무 짧다: {raw!r}")
    mode, pid, payload = data[0], data[1], data[2:]

    # Mode 01 요청의 긍정 응답 코드는 0x41 (= 0x01 + 0x40)
    if mode != 0x41:
        raise ObdDecodeError(f"Mode 01 응답(0x41)이 아니다: 0x{mode:02X}")
    spec = PID_TABLE.get(pid)
    if spec is None:
        raise ObdDecodeError(f"지원하지 않는 PID: 0x{pid:02X}")
    if len(payload) < spec.n_bytes:
        raise ObdDecodeError(
            f"{spec.name} 은 {spec.n_bytes}바이트가 필요하지만 "
            f"{len(payload)}바이트만 왔다")

    value = spec.decode(*payload[:spec.n_bytes])
    return spec.name, round(value, 3), spec.unit


def in_physical_range(name: str, value: float) -> bool:
    """디코딩된 값이 PID 의 물리 범위 안에 있는지 확인한다.

    물리 범위를 벗어난 값은 센서/배선 고장이거나 디코딩 오류이므로
    이상 탐지 이전 단계에서 버려야 한다.
    """
    for spec in PID_TABLE.values():
        if spec.name == name:
            return spec.lo <= value <= spec.hi
    return True


# 정상 운용 범위(Operating Range). 물리 범위보다 좁으며 경고 판정에 쓴다.
OPERATING_RANGE: dict[str, tuple[float, float]] = {
    "engine_temp_c": (80.0, 105.0),
    "engine_rpm": (600.0, 4500.0),
    "speed_kph": (0.0, 120.0),
    "battery_voltage": (12.2, 14.8),
    "engine_load": (0.0, 85.0),
    "fuel_level": (10.0, 100.0),
    "intake_temp_c": (-10.0, 60.0),
    "maf_gps": (1.0, 120.0),
}


def judge(name: str, value: float) -> str:
    """물리 범위 → 운용 범위 순서로 2단 판정한다."""
    if not in_physical_range(name, value):
        return "범위밖"
    lo, hi = OPERATING_RANGE.get(name, (float("-inf"), float("inf")))
    if value < lo:
        return "낮음"
    if value > hi:
        return "높음"
    return "정상"


def main() -> None:
    # 실제 차량에서 수집한 형태의 샘플 응답들
    samples = [
        ("41 05 5A", "냉각수 온도(웜업 중)"),
        ("41 05 93", "냉각수 온도(과열 의심)"),
        ("41 0C 1A F8", "엔진 회전수(주행)"),
        ("41 0C 0B B8", "엔진 회전수(아이들)"),
        ("41 0D 3C", "차량 속도"),
        ("41 2F 8C", "연료 잔량"),
        ("41 42 30 B4", "제어 모듈 전압"),
        ("41 42 2B 5C", "제어 모듈 전압(저전압 의심)"),
        ("41 04 66", "엔진 부하"),
        ("41 10 09 C4", "MAF"),
        ("41 1F 0E 10", "시동 후 경과"),
    ]

    print(f"{'원시 응답':<14}{'설명':<22}{'신호명':<18}{'값':>12} {'단위':<6}{'판정':<6}")
    print("-" * 82)
    for raw, desc in samples:
        name, value, unit = parse_response(raw)
        print(f"{raw:<14}{desc:<22}{name:<18}{value:>12,.3f} {unit:<6}"
              f"{judge(name, value):<6}")

    print("\n[오류 처리 테스트]")
    bad = ["41 99 00", "41", "01 05 5A", "41 0C 1A", "4Z 05 00"]
    for raw in bad:
        try:
            parse_response(raw)
        except ObdDecodeError as exc:
            print(f"  {raw!r:<14} -> {exc}")

    print("\n[검증: 변환식 역산]")
    # 냉각수 92도 -> 바이트 (92 + 40) = 132 = 0x84
    assert parse_response("41 05 84")[1] == 92.0
    # RPM 1850 -> 1850 * 4 = 7400 = 0x1CE8
    assert parse_response("41 0C 1C E8")[1] == 1850.0
    # 전압 12.4V -> 12400 = 0x3070
    assert parse_response("41 42 30 70")[1] == 12.4
    print("  냉각수 92.0℃ / RPM 1850 / 전압 12.4V 역산 일치")


if __name__ == "__main__":
    main()
