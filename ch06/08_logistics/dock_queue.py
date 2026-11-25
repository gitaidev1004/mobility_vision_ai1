# -*- coding: utf-8 -*-
"""물류센터 하역 대기열 시뮬레이터 (6.8절).

완성 프로젝트의 `access_event` / `parking_session` 도메인을 물류센터로 확장할 때
핵심이 되는 **대기열 정렬 규칙**과 **도크 배정**을 파이썬으로 먼저 검증한다.
여기서 확정한 규칙을 Spring Boot 서비스(`DockQueueService`)로 옮긴다.

상태 전이 규칙은 같은 디렉터리의 `job_status.py` 에 있다(단일 출처).

실행:
    (mobility) d:\\mobility_ai\\ch06> python 08_logistics/dock_queue.py
"""
from __future__ import annotations

import pathlib
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta

# 같은 디렉터리의 job_status.py 를 재사용한다(전이 규칙을 중복 구현하지 않는다).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from job_status import TIMESTAMP_FIELD, TransitionError, check_transit   # noqa: E402

# =====================================================================
# 대기열 정렬 규칙
# =====================================================================

#: 화물 종류별 우선순위. 숫자가 작을수록 먼저 배정된다.
#: 냉장/냉동은 품질 저하 위험이 있어 상온 화물보다 앞에 둔다.
#:
#: 근거는 "지연의 비용" 순이다.
#:   REEFER : 적재함 문을 열 때마다 온도가 오르고, 대기가 길어지면 폐기 위험.
#:   FRESH  : 그보다 완만하지만 유통기한이 짧다.
#:   NORMAL : 몇 시간 대기해도 품질에 영향이 없다.
#:   RETURN : 센터 입장에서 매출이 아니라 비용이므로 가장 뒤.
CARGO_PRIORITY: dict[str, int] = {
    "REEFER": 1,   # 냉장·냉동
    "FRESH": 2,    # 신선식품
    "NORMAL": 3,   # 상온 일반
    "RETURN": 4,   # 반품 회수
}

#: 예약 시각 대비 이 분을 넘겨 도착하면 지연 도착으로 본다(통계 분류 기준).
LATE_GRACE_MINUTES = 15
#: 예약 시각 + 이 분을 넘겨도 도착하지 않으면 NO_SHOW 로 돌린다(상태 전이 임계값).
#: 두 상수를 하나로 합치면 "20분 늦게 온 차량" 을 미도착으로 돌릴지
#: 지연으로 셀지 구분할 수 없다.
NO_SHOW_MINUTES = 60


@dataclass(slots=True)
class DockJob:
    """하역 작업 1건. DB `dock_job` 레코드에 대응한다.

    Attributes:
        plate: 화물차 번호판.
        carrier: 운송사명.
        cargo_type: CARGO_PRIORITY 의 키.
        reserved_at: 배차 예약 시각(walk-in 이면 None).
        status: 현재 상태.
        arrived_at: 도착 인식 시각.
        queued_at: 대기열 등록 시각.
        loading_at: 도크 접안 시각.
        done_at: 하역 완료 시각.
        departed_at: 출차 시각.
        dock_no: 배정된 도크 번호.
        history: (시각, 이전상태, 다음상태) 감사 로그.
    """

    plate: str
    carrier: str
    cargo_type: str
    reserved_at: datetime | None
    status: str = "RESERVED"
    arrived_at: datetime | None = None
    queued_at: datetime | None = None
    loading_at: datetime | None = None
    done_at: datetime | None = None
    departed_at: datetime | None = None
    dock_no: int | None = None
    history: list[tuple[datetime, str, str]] = field(default_factory=list)

    def transit(self, to_status: str, at: datetime) -> None:
        """상태를 전이하고 해당 타임스탬프를 기록한다.

        상태 변경과 시각 기록을 한 메서드에 묶은 것이 핵심이다. 호출 측이
        둘을 따로 쓰면 한쪽을 빠뜨릴 수 있고, 그러면 상태는 LOADING 인데
        접안 시각이 비어 대기시간 집계에서 그 건이 사라진다.

        Args:
            to_status: 목표 상태.
            at: 전이 시각.

        Raises:
            TransitionError: 허용되지 않은 전이일 때.
        """
        check_transit(self.status, to_status, self.plate)
        self.history.append((at, self.status, to_status))
        self.status = to_status

        field_name = TIMESTAMP_FIELD.get(to_status)
        if field_name:
            setattr(self, field_name, at)

    @property
    def wait_minutes(self) -> float | None:
        """대기열 등록 ~ 도크 접안까지의 대기시간(분)."""
        if self.queued_at is None or self.loading_at is None:
            return None
        return (self.loading_at - self.queued_at).total_seconds() / 60.0

    @property
    def load_minutes(self) -> float | None:
        """접안 ~ 하역 완료까지의 작업시간(분)."""
        if self.loading_at is None or self.done_at is None:
            return None
        return (self.done_at - self.loading_at).total_seconds() / 60.0

    @property
    def turnaround_minutes(self) -> float | None:
        """도착 ~ 출차까지의 총 체류시간(분). 물류센터 핵심 KPI."""
        if self.arrived_at is None or self.departed_at is None:
            return None
        return (self.departed_at - self.arrived_at).total_seconds() / 60.0

    @property
    def late_minutes(self) -> float:
        """예약 시각 대비 지연 도착 분. 예약/도착이 없으면 0."""
        if self.reserved_at is None or self.arrived_at is None:
            return 0.0
        delta = (self.arrived_at - self.reserved_at).total_seconds() / 60.0
        return max(0.0, delta)


def queue_key(job: DockJob, now: datetime) -> tuple[int, float, float]:
    """대기열 정렬 키를 만든다. 작은 값이 먼저 배정된다.

    세 단계로 비교한다.
      1. 화물 우선순위(냉장 > 신선 > 상온 > 반품)
      2. 예약 시각(이른 예약이 먼저) — 예약이 없으면 맨 뒤
      3. 대기열 등록 시각(먼저 줄 선 차가 먼저)

    예약을 1순위로 두지 않는 이유: 예약 시각만으로 정렬하면 냉장 화물이
    상온 화물 뒤에 밀려 품질 사고가 난다. 반대로 화물 종류만 보면 같은
    종류끼리 순서가 뒤바뀌어 운송사 불만이 생기므로 2·3순위를 둔다.

    Args:
        job: 대기 중인 작업.
        now: 기준 시각(예약 없는 건의 정렬값 산정용).

    Returns:
        (우선순위, 예약 경과분, 대기 경과분) 튜플.
    """
    priority = CARGO_PRIORITY.get(job.cargo_type, 9)
    if job.reserved_at is None:
        # 예약 없는 walk-in 은 같은 우선순위 안에서 맨 뒤로 보낸다.
        reserved_rank = 1e9
    else:
        reserved_rank = (job.reserved_at - now).total_seconds() / 60.0
    queued_rank = 0.0 if job.queued_at is None else (job.queued_at - now).total_seconds() / 60.0
    return (priority, reserved_rank, queued_rank)


def assign_docks(jobs: list[DockJob], docks: list[int], now: datetime) -> list[tuple[int, DockJob]]:
    """비어 있는 도크에 대기 차량을 배정한다.

    Args:
        jobs: 전체 작업 목록.
        docks: 도크 번호 목록.
        now: 배정 기준 시각.

    Returns:
        (도크 번호, 배정된 작업) 목록.
    """
    busy = {j.dock_no for j in jobs if j.status == "LOADING" and j.dock_no is not None}
    free = [d for d in docks if d not in busy]
    waiting = sorted([j for j in jobs if j.status == "WAITING"],
                     key=lambda j: queue_key(j, now))

    assigned: list[tuple[int, DockJob]] = []
    for dock_no, job in zip(free, waiting):
        job.dock_no = dock_no
        job.transit("LOADING", now)
        assigned.append((dock_no, job))
    return assigned


# =====================================================================
# 집계
# =====================================================================

def summarize(jobs: list[DockJob]) -> dict[str, float]:
    """대기/작업/체류 시간을 집계한다(운영 대시보드 숫자).

    세 지표가 각각 None 을 돌려줄 수 있으므로 집계에서 걸러낸다.
    진행 중인 작업을 0 으로 세면 평균이 낮게 나온다 — SQL 의 AVG 가
    NULL 을 자동 제외하는 것과 같은 처리다.

    Args:
        jobs: 작업 목록.

    Returns:
        지표 dict.
    """
    waits = [j.wait_minutes for j in jobs if j.wait_minutes is not None]
    loads = [j.load_minutes for j in jobs if j.load_minutes is not None]
    turns = [j.turnaround_minutes for j in jobs if j.turnaround_minutes is not None]
    lates = [j.late_minutes for j in jobs if j.late_minutes > LATE_GRACE_MINUTES]

    def mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    def p90(values: list[float]) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        idx = min(len(ordered) - 1, int(round(0.9 * (len(ordered) - 1))))
        return ordered[idx]

    return {
        "jobs": len(jobs),
        "completed": sum(1 for j in jobs if j.status == "DEPARTED"),
        "avg_wait": mean(waits),
        "max_wait": max(waits) if waits else 0.0,
        "p90_wait": p90(waits),
        "avg_load": mean(loads),
        "avg_turnaround": mean(turns),
        "late_arrivals": len(lates),
    }


# =====================================================================
# 시뮬레이션
# =====================================================================

def build_scenario(base: datetime) -> list[DockJob]:
    """교재 본문의 예시 시나리오(8대, 도크 2개)를 만든다.

    Args:
        base: 기준 시각(08:00 로 사용).

    Returns:
        작업 목록.
    """
    rows = [
        ("12가3456", "한결로지스", "NORMAL", 0),
        ("34나5678", "미래운수", "REEFER", 20),
        ("87로2345", "대한통운", "NORMAL", 40),
        ("78라1234", "한결로지스", "FRESH", 60),
        ("56다7890", "미래운수", "RETURN", 80),
        ("90마2345", "제일택배", "REEFER", 100),
        ("11바3456", "대한통운", "NORMAL", 120),
        ("22사4567", None, "NORMAL", None),   # 예약 없는 walk-in
    ]
    jobs: list[DockJob] = []
    for plate, carrier, cargo, offset in rows:
        reserved = None if offset is None else base + timedelta(minutes=offset)
        jobs.append(DockJob(
            plate=plate,
            carrier=carrier or "미등록",
            cargo_type=cargo,
            reserved_at=reserved,
        ))
    return jobs


def run_simulation() -> tuple[list[DockJob], dict[str, float]]:
    """시나리오를 1분 단위로 진행해 전체 흐름을 재현한다.

    Returns:
        (작업 목록, 집계 결과).
    """
    base = datetime(2026, 10, 24, 8, 0)
    jobs = build_scenario(base)
    docks = [1, 2]

    #: 번호판별 (도착 분, 하역 소요 분, 출차 지연 분)
    plan = {
        "12가3456": (5, 35, 4),
        "34나5678": (8, 40, 3),
        "87로2345": (12, 30, 5),
        "78라1234": (18, 45, 4),
        "56다7890": (25, 25, 6),
        "90마2345": (30, 35, 3),
        "11바3456": (36, 30, 5),
        "22사4567": (44, 20, 4),
    }
    by_plate = {j.plate: j for j in jobs}

    for minute in range(0, 420):
        now = base + timedelta(minutes=minute)

        # (1) 도착 인식 → ARRIVED → 즉시 WAITING 등록
        for plate, (arrive, _load, _depart) in plan.items():
            job = by_plate[plate]
            if minute == arrive and job.status in ("RESERVED", "NO_SHOW"):
                job.transit("ARRIVED", now)
                job.transit("WAITING", now)

        # (2) 하역 완료 판정
        for job in jobs:
            if job.status == "LOADING" and job.loading_at is not None:
                need = plan[job.plate][1]
                if (now - job.loading_at).total_seconds() / 60.0 >= need:
                    job.transit("DONE", now)

        # (3) 출차 처리
        for job in jobs:
            if job.status == "DONE" and job.done_at is not None:
                need = plan[job.plate][2]
                if (now - job.done_at).total_seconds() / 60.0 >= need:
                    job.transit("DEPARTED", now)

        # (4) 빈 도크에 대기 차량 배정
        assign_docks(jobs, docks, now)

        # (5) 예약 시각 + NO_SHOW_MINUTES 초과 미도착 처리
        for job in jobs:
            if (job.status == "RESERVED" and job.reserved_at is not None
                    and now >= job.reserved_at + timedelta(minutes=NO_SHOW_MINUTES)):
                job.transit("NO_SHOW", now)

    return jobs, summarize(jobs)


def main() -> None:
    """시뮬레이션을 실행하고 결과를 표로 출력한다."""
    jobs, stats = run_simulation()

    print("=" * 74)
    print(" 물류센터 하역 대기열 시뮬레이션 (도크 2개 / 차량 8대)")
    print("=" * 74)
    print(f"{'번호판':<11}{'화물':<8}{'도착':<7}{'접안':<7}{'완료':<7}"
          f"{'대기(분)':>9}{'하역(분)':>9}{'체류(분)':>9}")
    print("-" * 74)
    for job in sorted(jobs, key=lambda j: j.arrived_at or datetime.max):
        def hm(value: datetime | None) -> str:
            return "-" if value is None else value.strftime("%H:%M")

        def num(value: float | None) -> str:
            return "-" if value is None else f"{value:.0f}"

        print(f"{job.plate:<11}{job.cargo_type:<8}{hm(job.arrived_at):<7}"
              f"{hm(job.loading_at):<7}{hm(job.done_at):<7}"
              f"{num(job.wait_minutes):>9}{num(job.load_minutes):>9}"
              f"{num(job.turnaround_minutes):>9}")

    print("-" * 74)
    print(f"  처리 완료 {stats['completed']}/{stats['jobs']}대")
    print(f"  평균 대기시간 {stats['avg_wait']:.1f}분 / 최대 {stats['max_wait']:.0f}분 "
          f"/ p90 {stats['p90_wait']:.0f}분")
    print(f"  평균 하역시간 {stats['avg_load']:.1f}분")
    print(f"  평균 체류시간 {stats['avg_turnaround']:.1f}분")
    print(f"  지연 도착({LATE_GRACE_MINUTES}분 초과) {stats['late_arrivals']}건")

    print("\n[상태 전이 검증] 허용되지 않은 전이는 거부되어야 한다.")
    probe = DockJob("99하9999", "테스트", "NORMAL", None)
    try:
        probe.transit("LOADING", datetime(2026, 10, 24, 9, 0))
    except TransitionError as exc:
        print(f"  거부됨: {exc}")


if __name__ == "__main__":
    main()
