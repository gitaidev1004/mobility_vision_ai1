# -*- coding: utf-8 -*-
"""CAN 프레임 파싱과 신호 추출.

CAN(Controller Area Network) 데이터 프레임은
  식별자(ID) + DLC(Data Length Code) + 데이터 필드(0~8바이트)
로 구성된다. 데이터 필드 안에 여러 신호가 비트 단위로 묶여 있으므로
`비트 오프셋 / 비트 길이 / 스케일 / 오프셋` 네 가지 정보로 복원한다.

물리값 = raw * scale + offset

표준 라이브러리만 사용하므로 추가 설치 없이 실행된다.
"""
from __future__ import annotations

from typing import NamedTuple


class CanFrame(NamedTuple):
    """파싱된 CAN 데이터 프레임."""
    can_id: int          # 11비트(표준) 또는 29비트(확장) 식별자
    extended: bool       # 확장 ID 여부
    dlc: int             # 데이터 길이(바이트)
    data: bytes          # 데이터 필드


class Signal(NamedTuple):
    """CAN 메시지 안에 들어 있는 신호 한 개의 정의(DBC 의 축약형)."""
    name: str
    start_bit: int       # 데이터 필드 시작부터의 비트 오프셋
    length: int          # 비트 길이
    scale: float         # 스케일 계수
    offset: float        # 오프셋
    unit: str
    signed: bool = False     # 2의 보수 부호 여부
    big_endian: bool = True  # Motorola(big) / Intel(little)


class CanMessage(NamedTuple):
    """CAN ID 하나에 대응하는 메시지 정의."""
    can_id: int
    name: str
    dlc: int
    signals: tuple[Signal, ...]


# ----------------------------------------------------------------------
# 실습용 메시지 정의 (실제 차량의 DBC 를 단순화한 형태)
# ----------------------------------------------------------------------
DBC: dict[int, CanMessage] = {
    0x0C1: CanMessage(0x0C1, "ENGINE_STATUS", 8, (
        # RPM: 상위 2바이트, 0.25 rpm/bit
        Signal("engine_rpm", 0, 16, 0.25, 0.0, "rpm"),
        # 냉각수 온도: 1바이트, 1℃/bit, -40℃ 오프셋
        Signal("engine_temp_c", 16, 8, 1.0, -40.0, "℃"),
        # 엔진 부하: 1바이트, 0.3922 %/bit (= 100/255)
        Signal("engine_load", 24, 8, 100.0 / 255.0, 0.0, "%"),
        # 스로틀: 1바이트
        Signal("throttle_pct", 32, 8, 100.0 / 255.0, 0.0, "%"),
        # 흡기 온도: 1바이트
        Signal("intake_temp_c", 40, 8, 1.0, -40.0, "℃"),
        # 오일 압력: 12비트, 0.05 bar/bit
        Signal("oil_press_bar", 48, 12, 0.05, 0.0, "bar"),
        # 엔진 상태 플래그: 4비트
        Signal("engine_flags", 60, 4, 1.0, 0.0, "-"),
    )),
    0x1A3: CanMessage(0x1A3, "VEHICLE_DYNAMICS", 8, (
        # 차속: 16비트, 0.01 km/h per bit
        Signal("speed_kph", 0, 16, 0.01, 0.0, "km/h"),
        # 종방향 가속도: 12비트 부호 있음, 0.01 m/s^2
        Signal("accel_long", 16, 12, 0.01, 0.0, "m/s^2", signed=True),
        # 요레이트: 12비트 부호 있음, 0.02 deg/s
        Signal("yaw_rate", 28, 12, 0.02, 0.0, "deg/s", signed=True),
        # 브레이크 압력: 16비트, 0.1 bar
        Signal("brake_press_bar", 40, 16, 0.1, 0.0, "bar"),
        # 주행거리 누적 하위: 8비트
        Signal("trip_counter", 56, 8, 1.0, 0.0, "-"),
    )),
    0x2F5: CanMessage(0x2F5, "BATTERY_TPMS", 8, (
        # 배터리 전압: 16비트, 0.001 V
        Signal("battery_voltage", 0, 16, 0.001, 0.0, "V"),
        # 배터리 SOH: 8비트, 0.4 %
        Signal("battery_soh", 16, 8, 0.4, 0.0, "%"),
        # 타이어 공기압 4개: 각 8비트, 0.25 psi
        Signal("tire_pressure_fl", 24, 8, 0.25, 0.0, "psi"),
        Signal("tire_pressure_fr", 32, 8, 0.25, 0.0, "psi"),
        Signal("tire_pressure_rl", 40, 8, 0.25, 0.0, "psi"),
        Signal("tire_pressure_rr", 48, 8, 0.25, 0.0, "psi"),
        # 경고 비트맵: 8비트
        Signal("tpms_warn", 56, 8, 1.0, 0.0, "-"),
    )),
}


class CanParseError(ValueError):
    """CAN 프레임을 해석할 수 없을 때 발생한다."""


def parse_frame(line: str) -> CanFrame:
    """candump 형식의 한 줄을 CanFrame 으로 변환한다.

    형식: `<ID>#<데이터 16진수>`  예) `0C1#1CE8540A6428012F`
    ID 가 8자리면 확장 ID(29비트)로 간주한다.
    """
    if "#" not in line:
        raise CanParseError(f"'#' 구분자가 없다: {line!r}")
    id_part, data_part = line.strip().split("#", 1)
    id_part = id_part.strip()
    data_part = data_part.replace(" ", "").strip()

    try:
        can_id = int(id_part, 16)
    except ValueError as exc:
        raise CanParseError(f"ID 가 16진수가 아니다: {id_part!r}") from exc

    extended = len(id_part) > 3
    limit = 0x1FFFFFFF if extended else 0x7FF
    if can_id > limit:
        raise CanParseError(
            f"ID 0x{can_id:X} 가 {'29' if extended else '11'}비트 범위를 넘는다")

    if len(data_part) % 2 != 0:
        raise CanParseError(f"데이터 길이가 홀수다: {data_part!r}")
    try:
        data = bytes.fromhex(data_part)
    except ValueError as exc:
        raise CanParseError(f"데이터가 16진수가 아니다: {data_part!r}") from exc
    if len(data) > 8:
        raise CanParseError(f"클래식 CAN 데이터는 최대 8바이트다: {len(data)}바이트")

    return CanFrame(can_id=can_id, extended=extended,
                    dlc=len(data), data=data)


def extract_bits(data: bytes, start_bit: int, length: int,
                 big_endian: bool = True) -> int:
    """데이터 필드에서 비트 구간을 잘라 정수(raw)로 돌려준다.

    big_endian=True 면 MSB 우선(Motorola), False 면 LSB 우선(Intel) 순서다.
    """
    total_bits = len(data) * 8
    if start_bit < 0 or start_bit + length > total_bits:
        raise CanParseError(
            f"비트 구간 [{start_bit}:{start_bit + length}] 이 "
            f"{total_bits}비트 프레임을 벗어난다")

    if big_endian:
        # 전체를 하나의 큰 정수로 보고 왼쪽부터 자른다.
        whole = int.from_bytes(data, "big")
        shift = total_bits - start_bit - length
        return (whole >> shift) & ((1 << length) - 1)
    # Intel 순서: 바이트를 역순으로 이어 붙인 뒤 오른쪽부터 자른다.
    whole = int.from_bytes(data, "little")
    return (whole >> start_bit) & ((1 << length) - 1)


def to_signed(raw: int, length: int) -> int:
    """2의 보수 표현을 부호 있는 정수로 변환한다."""
    sign_bit = 1 << (length - 1)
    return raw - (1 << length) if raw & sign_bit else raw


def decode_signal(data: bytes, sig: Signal) -> float:
    """신호 정의를 적용해 물리값을 계산한다."""
    raw = extract_bits(data, sig.start_bit, sig.length, sig.big_endian)
    if sig.signed:
        raw = to_signed(raw, sig.length)
    return raw * sig.scale + sig.offset


def decode_frame(frame: CanFrame) -> tuple[str, dict[str, float]]:
    """DBC 정의를 찾아 프레임 전체를 신호 딕셔너리로 변환한다."""
    msg = DBC.get(frame.can_id)
    if msg is None:
        raise CanParseError(f"DBC 에 없는 CAN ID: 0x{frame.can_id:03X}")
    if frame.dlc != msg.dlc:
        raise CanParseError(
            f"{msg.name} 은 DLC {msg.dlc} 인데 프레임은 {frame.dlc} 이다")
    out: dict[str, float] = {}
    for sig in msg.signals:
        out[sig.name] = round(decode_signal(frame.data, sig), 4)
    return msg.name, out


def main() -> None:
    # 실제 버스에서 캡처한 형태의 프레임들
    lines = [
        "0C1#1CE8849E2E440402",   # ENGINE_STATUS (정상 주행)
        "0C1#307094E0C2560521",   # ENGINE_STATUS (고부하 + 과열 의심)
        "1A3#12C0FDD050007C1F",   # VEHICLE_DYNAMICS (감속 중)
        "2F5#3714E6807F7E7D00",   # BATTERY_TPMS (정상)
        "2F5#2E189A647E7D7B01",   # BATTERY_TPMS (저전압 + 타이어 경고)
    ]

    for line in lines:
        frame = parse_frame(line)
        name, signals = decode_frame(frame)
        kind = "확장" if frame.extended else "표준"
        print(f"\n[{line}]")
        print(f"  ID=0x{frame.can_id:03X} ({kind})  DLC={frame.dlc}  "
              f"DATA={frame.data.hex().upper()}  MSG={name}")
        msg = DBC[frame.can_id]
        for sig in msg.signals:
            raw = extract_bits(frame.data, sig.start_bit, sig.length,
                               sig.big_endian)
            print(f"    {sig.name:<18} bit[{sig.start_bit:>2}:"
                  f"{sig.start_bit + sig.length:>2}] raw={raw:<7} "
                  f"-> {signals[sig.name]:>10,.3f} {sig.unit}")

    print("\n[오류 처리 테스트]")
    bad = ["0C1 1CE8", "ZZZ#0011", "0C1#1CE85", "7FF#00112233445566778899",
           "999#00112233445566", "0C1#1CE854"]
    for line in bad:
        try:
            decode_frame(parse_frame(line))
        except CanParseError as exc:
            print(f"  {line!r:<28} -> {exc}")

    print("\n[검증: 신호 역산]")
    # RPM 1850 -> raw 7400 = 0x1CE8, 냉각수 92℃ -> raw 132 = 0x84
    f = parse_frame("0C1#1CE88400000000 00".replace(" ", ""))
    _, s = decode_frame(f)
    assert abs(s["engine_rpm"] - 1850.0) < 1e-6
    assert abs(s["engine_temp_c"] - 92.0) < 1e-6
    print("  engine_rpm=1850.0, engine_temp_c=92.0 역산 일치")
    # 전압 12.4V -> raw 12400 = 0x3070
    f = parse_frame("2F5#3070000000000000")
    _, s = decode_frame(f)
    assert abs(s["battery_voltage"] - 12.4) < 1e-9
    print("  battery_voltage=12.4 역산 일치")


if __name__ == "__main__":
    main()
