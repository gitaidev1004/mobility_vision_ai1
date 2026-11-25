# -*- coding: utf-8 -*-
"""환경별(주간/야간/우천/역광/원거리/기울기) 성능 분해 리포트.

전체 평균 하나로는 어디를 고쳐야 할지 알 수 없다. REFERENCE.md 의
평가 기준 3번 '낮·밤, 비·역광, 원거리, 기울어진 번호판 등 환경별로
성능을 나눠 확인한다'를 그대로 구현한 스크립트다.

출력
  (1) 환경별 검출률 / 문자 정확도 / 완전 일치율 / 지연시간
  (2) 전체 평균 대비 격차(gap)와 개선 우선순위
  (3) 환경별 가중 평균(운영 트래픽 비중을 반영한 기대 성능)
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from metrics_ocr import (OcrMetrics, evaluate, expected_exact,   # noqa: E402
                         required_char_accuracy)

# 실제 운영 트래픽 비중(현장 로그에서 집계해 넣는다). 합이 1.0 이어야 한다.
TRAFFIC_MIX: dict[str, float] = {
    "주간": 0.52,
    "야간": 0.21,
    "우천": 0.08,
    "역광": 0.09,
    "원거리": 0.06,
    "기울기": 0.04,
}

# 환경별 운영 목표(완전 일치율). 환경이 나쁠수록 목표를 낮춰 잡는다.
TARGET_EXACT: dict[str, float] = {
    "주간": 0.995,
    "야간": 0.980,
    "우천": 0.970,
    "역광": 0.970,
    "원거리": 0.950,
    "기울기": 0.950,
}


@dataclass
class EvalRecord:
    """평가 레코드 하나(이벤트 단위 결과)."""

    condition: str
    gt: str
    pred: str
    detected: bool = True
    latency_ms: float = 0.0


@dataclass
class ConditionReport:
    """한 환경의 성능 분해 결과."""

    condition: str
    n: int = 0
    detected: int = 0
    metrics: OcrMetrics = field(default_factory=OcrMetrics)
    latencies: list[float] = field(default_factory=list)

    @property
    def detect_rate(self) -> float:
        return self.detected / self.n if self.n else 0.0

    @property
    def char_acc(self) -> float:
        return self.metrics.char_accuracy

    @property
    def exact(self) -> float:
        """종단 완전 일치율. 미검출은 실패로 센다."""
        return self.metrics.exact_hits / self.n if self.n else 0.0

    @property
    def p50(self) -> float:
        return _percentile(self.latencies, 50)

    @property
    def p95(self) -> float:
        return _percentile(self.latencies, 95)

    @property
    def target(self) -> float:
        return TARGET_EXACT.get(self.condition, 0.99)

    @property
    def gap(self) -> float:
        """목표 대비 부족분. 양수면 목표 미달이다."""
        return self.target - self.exact


def _percentile(values: list[float], p: float) -> float:
    """정렬 후 선형 보간 없이 가장 가까운 순위값을 돌려준다."""
    if not values:
        return 0.0
    s = sorted(values)
    k = min(len(s) - 1, max(0, int(round((p / 100.0) * (len(s) - 1)))))
    return s[k]


def analyze(records: list[EvalRecord]) -> dict[str, ConditionReport]:
    """환경별로 레코드를 묶어 지표를 계산한다."""
    buckets: dict[str, list[EvalRecord]] = {}
    for r in records:
        buckets.setdefault(r.condition, []).append(r)

    out: dict[str, ConditionReport] = {}
    for cond, rs in buckets.items():
        rep = ConditionReport(cond, n=len(rs))
        rep.detected = sum(1 for r in rs if r.detected)
        # 미검출은 예측을 빈 문자열로 둬서 CER 에 불이익이 반영되게 한다.
        rep.metrics = evaluate([(r.gt, r.pred if r.detected else "")
                                for r in rs])
        rep.latencies = [r.latency_ms for r in rs if r.latency_ms > 0]
        out[cond] = rep
    return out


def weighted_exact(reports: dict[str, ConditionReport],
                   mix: dict[str, float] | None = None) -> float:
    """운영 트래픽 비중을 반영한 기대 완전 일치율."""
    mix = mix or TRAFFIC_MIX
    total_w = sum(mix.get(c, 0.0) for c in reports)
    if total_w <= 0:
        return 0.0
    return sum(reports[c].exact * mix.get(c, 0.0) for c in reports) / total_w


def print_table(reports: dict[str, ConditionReport]) -> None:
    """환경별 평가표를 출력한다(교재의 환경별 평가표 양식과 같은 구성)."""
    order = [c for c in TRAFFIC_MIX if c in reports]
    order += [c for c in reports if c not in TRAFFIC_MIX]
    print(f"{'환경':<7s}{'건수':>5s}{'비중':>7s}{'검출률':>9s}"
          f"{'문자정확도':>11s}{'완전일치':>10s}{'목표':>8s}{'격차':>8s}"
          f"{'p50(ms)':>9s}{'p95(ms)':>9s}")
    print("-" * 86)
    for c in order:
        r = reports[c]
        w = TRAFFIC_MIX.get(c, 0.0)
        print(f"{c:<7s}{r.n:>5d}{w:>7.0%}{r.detect_rate:>9.2%}"
              f"{r.char_acc:>11.3%}{r.exact:>10.2%}{r.target:>8.1%}"
              f"{r.gap:>+8.2%}{r.p50:>9.1f}{r.p95:>9.1f}")
    print("-" * 86)
    allm = OcrMetrics()
    tot = det = 0
    lat: list[float] = []
    for r in reports.values():
        allm.samples += r.metrics.samples
        allm.total_ref_chars += r.metrics.total_ref_chars
        allm.total_edits += r.metrics.total_edits
        allm.exact_hits += r.metrics.exact_hits
        tot += r.n
        det += r.detected
        lat.extend(r.latencies)
    print(f"{'전체':<7s}{tot:>5d}{1.0:>7.0%}{det / tot:>9.2%}"
          f"{allm.char_accuracy:>11.3%}{allm.exact_hits / tot:>10.2%}"
          f"{0.99:>8.1%}{0.99 - allm.exact_hits / tot:>+8.2%}"
          f"{_percentile(lat, 50):>9.1f}{_percentile(lat, 95):>9.1f}")
    print(f"\n트래픽 가중 기대 완전 일치율: {weighted_exact(reports):.3%}")


def print_priority(reports: dict[str, ConditionReport]) -> None:
    """개선 우선순위. 격차 x 트래픽 비중이 큰 환경부터 손본다."""
    rows = []
    for c, r in reports.items():
        w = TRAFFIC_MIX.get(c, 0.0)
        impact = max(r.gap, 0.0) * w          # 전체 지표에 주는 손실
        rows.append((c, r.gap, w, impact, r.n))
    rows.sort(key=lambda x: -x[3])
    print(f"\n{'환경':<7s}{'격차':>9s}{'비중':>7s}{'전체 손실':>11s}  판단")
    print("-" * 62)
    for c, gap, w, impact, n in rows:
        if gap <= 0:
            verdict = "목표 달성 — 유지"
        elif impact >= 0.012:
            verdict = "최우선 개선"
        elif impact >= 0.009:
            verdict = "차순위 개선"
        elif n < 30:
            verdict = "샘플 부족 — 데이터 추가 수집 먼저"
        else:
            verdict = "관찰"
        print(f"{c:<7s}{gap:>+9.2%}{w:>7.0%}{impact:>11.3%}  {verdict}")


def print_target_breakdown(reports: dict[str, ConditionReport]) -> None:
    """목표 완전 일치율에 필요한 문자 정확도를 환경별로 역산한다."""
    print(f"\n{'환경':<7s}{'목표 완전일치':>13s}{'평균 길이':>10s}"
          f"{'필요 문자정확도':>15s}{'현재 문자정확도':>15s}  판단")
    print("-" * 74)
    for c, r in reports.items():
        avg_len = (r.metrics.total_ref_chars / r.metrics.samples
                   if r.metrics.samples else 7.0)
        need = required_char_accuracy(r.target, int(round(avg_len)))
        now = r.char_acc
        verdict = "충족" if now >= need else "부족"
        print(f"{c:<7s}{r.target:>13.1%}{avg_len:>10.1f}"
              f"{need:>15.4%}{now:>15.4%}  {verdict}"
              f"(기대 완전일치 {expected_exact(now, int(round(avg_len))):.2%})")


def builtin_records() -> list[EvalRecord]:
    """교재 실습용 평가 데이터. 환경별 난이도 차이를 모사했다."""
    recs: list[EvalRecord] = []

    def add(cond: str, n_ok: int, errs: list[tuple[str, str]],
            n_miss: int, lat: float) -> None:
        for i in range(n_ok):
            gt = ["12가3456", "87로2345", "서울34나5678",
                  "123가4567", "34허5678"][i % 5]
            recs.append(EvalRecord(cond, gt, gt, True, lat + (i % 5) * 2.0))
        for gt, pred in errs:
            recs.append(EvalRecord(cond, gt, pred, True, lat + 6.0))
        for i in range(n_miss):
            recs.append(EvalRecord(cond, "12가3456", "", False, lat + 12.0))

    add("주간", 196, [("12가3456", "12가3458"),
                      ("87로2345", "87도2345")], 2, 58.0)
    add("야간", 78, [("12가3456", "12가345B"), ("87로2345", "87로234"),
                     ("서울34나5678", "서울34나5G78")], 4, 66.0)
    add("우천", 28, [("12가3456", "12가3450"),
                     ("123가4567", "123가456")], 2, 71.0)
    add("역광", 30, [("87로2345", "87도2345"),
                     ("34허5678", "34허5B78")], 3, 69.0)
    add("원거리", 18, [("서울34나5678", "서울34나567"),
                       ("12가3456", "12가3466")], 3, 63.0)
    add("기울기", 12, [("123가4567", "128가4567"),
                       ("12가3456", "17가3456")], 2, 74.0)
    return recs


def _demo() -> None:
    reports = analyze(builtin_records())
    print("=== 환경별 성능 분해 리포트 ===")
    print_table(reports)
    print_priority(reports)
    print_target_breakdown(reports)


if __name__ == "__main__":
    _demo()
