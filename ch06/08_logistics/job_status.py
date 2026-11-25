# -*- coding: utf-8 -*-
"""하역 작업 상태 기계 (6.8절).

물류센터 확장의 핵심은 상태 전이 규칙이다. 전이를 코드 곳곳의 if 문으로
흩어 놓으면 "출차한 작업을 다시 접안" 하거나 "도착하지 않은 작업을 하역 완료"
처리하는 호출이 어딘가에서 들어온다. 허용 전이를 하나의 표로 모아 두고 모든
상태 변경이 그 표를 통과하게 하면 모순이 구조적으로 막힌다.

이 모듈을 Spring Boot 로 옮길 때는 STATUSES 를 JobStatus enum 으로,
TRANSITIONS 를 enum 의 allowedNext() 메서드로, TransitionError 를
BusinessException 으로 바꾼다(GlobalExceptionHandler 가 400 으로 매핑한다).

실행:
    (mobility) d:\\mobility_ai\\ch06> python 08_logistics/job_status.py
"""
from __future__ import annotations

#: 하역 작업 상태. DB 의 dock_job.status 컬럼 값과 1:1 대응한다.
STATUSES: tuple[str, ...] = (
    "RESERVED",   # 배차 예약만 등록된 상태
    "ARRIVED",    # 번호판 인식으로 도착이 확인된 상태
    "WAITING",    # 대기열에 등록되어 도크 배정을 기다리는 상태
    "LOADING",    # 도크에 접안해 상/하차 중
    "DONE",       # 하역 완료, 출차 대기
    "DEPARTED",   # 출차 이벤트까지 끝난 종결 상태
    "NO_SHOW",    # 예약 시각 + 유예시간을 넘겨 미도착 처리
    "CANCELED",   # 예약 취소
)

#: 허용된 전이. 키에 없는 전이는 TransitionError 로 거부한다.
#: 이 표가 교재 본문의 "물류 상태 전이표" 와 같은 내용이다.
#:
#: 설계 판단 두 가지:
#:   - LOADING → CANCELED 는 금지한다. 이미 접안해 상차를 시작한 작업은
#:     취소할 수 없다. 화물이 절반만 실린 상태를 표현할 수 없으므로
#:     DONE 으로 마치고 memo 에 사유를 남긴다.
#:   - NO_SHOW → ARRIVED 는 허용한다. 예약보다 한참 늦게 도착하는 일이
#:     흔하고, 원래 예약과 연결을 끊으면 "예약했는데 안 왔다" 와
#:     "예약 없이 왔다" 가 동시에 집계되어 통계가 이중으로 왜곡된다.
TRANSITIONS: dict[str, tuple[str, ...]] = {
    "RESERVED": ("ARRIVED", "NO_SHOW", "CANCELED"),
    "ARRIVED": ("WAITING", "CANCELED"),
    "WAITING": ("LOADING", "CANCELED"),
    "LOADING": ("DONE",),
    "DONE": ("DEPARTED",),
    "DEPARTED": (),
    "NO_SHOW": ("ARRIVED",),   # 늦게라도 도착하면 되살린다
    "CANCELED": (),
}

#: 전이가 완료되면 기록할 타임스탬프 필드명. 상태 변경과 시각 기록을
#: 한 곳에 묶어 두면 "상태는 LOADING 인데 접안 시각이 비어 있는" 불일치가
#: 생길 수 없다.
TIMESTAMP_FIELD: dict[str, str] = {
    "ARRIVED": "arrived_at",
    "WAITING": "queued_at",
    "LOADING": "loading_at",
    "DONE": "done_at",
    "DEPARTED": "departed_at",
}

#: 더 이상 전이할 수 없는 종결 상태.
TERMINAL: frozenset[str] = frozenset(s for s, nxt in TRANSITIONS.items() if not nxt)


class TransitionError(RuntimeError):
    """허용되지 않은 상태 전이를 시도했을 때 발생한다."""


def allowed_next(status: str) -> tuple[str, ...]:
    """현재 상태에서 갈 수 있는 상태 목록을 돌려준다.

    Args:
        status: 현재 상태.

    Returns:
        허용 전이 튜플. 알 수 없는 상태면 빈 튜플.
    """
    return TRANSITIONS.get(status, ())


def can_transit(from_status: str, to_status: str) -> bool:
    """전이가 허용되는지 판정한다.

    Args:
        from_status: 현재 상태.
        to_status: 목표 상태.

    Returns:
        허용되면 True.
    """
    return to_status in allowed_next(from_status)


def check_transit(from_status: str, to_status: str, label: str = "") -> None:
    """전이를 검사하고 허용되지 않으면 예외를 던진다.

    오류 메시지에 허용 전이 목록을 함께 넣는 것이 디버깅을 크게 돕는다.
    "전이할 수 없다" 만 던지면 호출자가 무엇이 가능한지 찾아야 하지만,
    허용 목록을 보여주면 즉시 올바른 호출을 알 수 있다.

    Args:
        from_status: 현재 상태.
        to_status: 목표 상태.
        label: 오류 메시지에 붙일 식별자(보통 번호판).

    Raises:
        TransitionError: 허용되지 않은 전이일 때.
    """
    if can_transit(from_status, to_status):
        return
    allowed = allowed_next(from_status)
    prefix = f"{label}: " if label else ""
    raise TransitionError(
        f"{prefix}{from_status} → {to_status} 는 허용되지 않는다. "
        f"허용: {allowed or '(종결 상태)'}"
    )


def transition_table() -> list[tuple[str, str, str]]:
    """전이표를 (현재 상태, 허용 전이, 진입 시 기록 필드) 행 목록으로 펼친다.

    세 번째 열은 "이 상태로 진입할 때 어느 타임스탬프를 남기는가" 이다.
    RESERVED 는 예약 등록 시점에 reserved_at 이 이미 채워지고,
    NO_SHOW/CANCELED 는 시각을 따로 남기지 않으므로 '—' 다.

    Returns:
        표 출력용 행 목록.
    """
    rows: list[tuple[str, str, str]] = []
    for status in STATUSES:
        nxt = TRANSITIONS.get(status, ())
        rows.append((
            status,
            " / ".join(nxt) if nxt else "(없음)",
            TIMESTAMP_FIELD.get(status, "—"),
        ))
    return rows


def main() -> None:
    """전이표를 출력하고 허용/거부 사례를 각각 확인한다."""
    print("=" * 72)
    print(" 하역 작업 상태 전이표")
    print("=" * 72)
    print(f"{'현재 상태':<12}{'허용 전이':<36}{'기록 시각':<14}")
    print("-" * 72)
    for status, nxt, field in transition_table():
        print(f"{status:<12}{nxt:<36}{field:<14}")
    print("-" * 72)
    print(f"종결 상태: {', '.join(sorted(TERMINAL))}")

    print("\n[허용되는 전이]")
    for a, b in (("RESERVED", "ARRIVED"), ("WAITING", "LOADING"), ("NO_SHOW", "ARRIVED")):
        check_transit(a, b, "12가3456")
        print(f"  통과: {a} → {b}")

    print("\n[거부되는 전이]")
    for a, b in (("RESERVED", "LOADING"), ("LOADING", "CANCELED"), ("DEPARTED", "WAITING")):
        try:
            check_transit(a, b, "99하9999")
        except TransitionError as exc:
            print(f"  거부됨: {exc}")


if __name__ == "__main__":
    main()
