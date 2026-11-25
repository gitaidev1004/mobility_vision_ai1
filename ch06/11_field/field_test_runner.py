# -*- coding: utf-8 -*-
"""실환경 성능 검증 실행·집계기.

환경(주간/야간/우천/역광/원거리/고속이동/기울기/부분가림)별 테스트 표본을 돌려
REFERENCE.md 의 평가 기준 5개 항목을 한 번에 집계한다.

  1) 번호판 검출률과 OCR 문자 정확도를 따로 측정
  2) 전체 차량번호 완전일치율을 별도 측정
  3) 환경별로 성능을 분해
  4) 미학습 차량·미학습 촬영 세션만으로 독립 테스트
  5) CPU 추론 시간·메모리와 INT8 변환 전후 비교

사용법
    python 11_field/field_test_runner.py                      # 내장 데모 표본으로 실행
    python 11_field/field_test_runner.py --manifest cases.json # 실제 측정 결과 집계
    python 11_field/field_test_runner.py --csv report.csv      # 집계 결과 CSV 저장

manifest(JSON) 한 건의 형식 — AI 서비스 응답을 그대로 붙이면 된다.
    {"env": "야간", "truth": "12가3456", "pred": "12가3456",
     "detected": true, "session": "S03", "vehicle_id": "V118",
     "trained_vehicle": false, "detect_ms": 41.2, "ocr_ms": 20.1,
     "rss_mb": 212.4, "quant": "fp32"}
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# 환경 정의 — 표본 할당량과 합격 기준을 한 곳에 모은다.
# ---------------------------------------------------------------------------

#: 환경명 → (할당 표본 수, 검출률 하한, 완전일치율 하한)
ENV_PLAN: dict[str, tuple[int, float, float]] = {
    "주간":     (204, 0.995, 0.985),
    "야간":     (132, 0.985, 0.960),
    "우천":     (48, 0.970, 0.930),
    "역광":     (60, 0.975, 0.940),
    "원거리":   (48, 0.960, 0.920),
    "고속이동": (36, 0.970, 0.930),
    "기울기":   (42, 0.980, 0.950),
    "부분가림": (30, 0.930, 0.850),
}

#: 전체 집계 합격 기준
OVERALL_DETECT_MIN = 0.99
OVERALL_EXACT_MIN = 0.99
OVERALL_CHAR_MIN = 0.995
#: INT8 변환으로 허용하는 완전일치율 하락 폭(%p)
INT8_DROP_MAX = 0.02
#: p95 지연시간 상한(ms) — 게이트 차단기 동작 시간 예산에서 역산한 값
LATENCY_P95_MAX = 150.0


@dataclass
class Case:
    """테스트 표본 1건의 측정 결과."""

    env: str
    truth: str
    pred: str | None
    detected: bool
    session: str
    vehicle_id: str
    trained_vehicle: bool
    detect_ms: float
    ocr_ms: float
    rss_mb: float
    quant: str = "fp32"

    @property
    def total_ms(self) -> float:
        """검출 + OCR 합계 지연시간(ms)."""
        return self.detect_ms + self.ocr_ms

    @property
    def exact(self) -> bool:
        """전체 문자열 완전일치 여부."""
        return self.detected and self.pred == self.truth


@dataclass
class Agg:
    """한 그룹(환경 등)의 집계 결과."""

    name: str
    n: int = 0
    detected: int = 0
    exact: int = 0
    char_hit: int = 0
    char_total: int = 0
    latencies: list[float] = field(default_factory=list)
    rss: list[float] = field(default_factory=list)

    @property
    def detect_rate(self) -> float:
        """검출률 = 검출 성공 / 전체."""
        return self.detected / self.n if self.n else 0.0

    @property
    def exact_rate(self) -> float:
        """완전일치율 = 완전일치 / 전체."""
        return self.exact / self.n if self.n else 0.0

    @property
    def char_acc(self) -> float:
        """문자 정확도 = 맞춘 문자 수 / 정답 문자 수."""
        return self.char_hit / self.char_total if self.char_total else 0.0

    def pct(self, q: float) -> float:
        """지연시간 분위수(ms). q=0.95 면 p95."""
        if not self.latencies:
            return 0.0
        xs = sorted(self.latencies)
        idx = min(len(xs) - 1, max(0, math.ceil(q * len(xs)) - 1))
        return xs[idx]

    @property
    def mem_peak(self) -> float:
        """측정된 RSS 최대값(MB)."""
        return max(self.rss) if self.rss else 0.0


def char_hits(truth: str, pred: str | None) -> tuple[int, int]:
    """자리별 문자 일치 수와 정답 문자 수를 센다.

    번호판은 길이가 7~9자로 고정적이므로 편집거리 대신 자리 비교를 쓴다.
    길이가 다르면 짧은 쪽까지만 비교하고 나머지는 오인식으로 처리한다.

    Args:
        truth: 정답 번호판 문자열.
        pred: 예측 문자열(미검출이면 None).

    Returns:
        (일치 문자 수, 정답 문자 수).
    """
    total = len(truth)
    if not pred:
        return 0, total
    hit = sum(1 for a, b in zip(truth, pred) if a == b)
    return hit, total


def aggregate(cases: list[Case], key) -> dict[str, Agg]:
    """표본을 key 함수 기준으로 묶어 집계한다.

    Args:
        cases: 표본 목록.
        key: Case → 그룹명 함수.

    Returns:
        그룹명 → Agg.
    """
    out: dict[str, Agg] = {}
    for c in cases:
        name = key(c)
        agg = out.setdefault(name, Agg(name=name))
        agg.n += 1
        if c.detected:
            agg.detected += 1
        if c.exact:
            agg.exact += 1
        hit, tot = char_hits(c.truth, c.pred)
        agg.char_hit += hit
        agg.char_total += tot
        agg.latencies.append(c.total_ms)
        agg.rss.append(c.rss_mb)
    return out


def wilson_lower(successes: int, n: int, z: float = 1.959963985) -> float:
    """Wilson 점수 구간의 하한을 구한다(양측 95%).

    표본이 수십 건일 때 정규근사(Wald)보다 극단값 처리가 안정적이다.
    보고서에는 관측값과 함께 이 하한을 적어 "표본이 작아서 높게 나온 것"
    과 "실제로 좋은 것" 을 구분한다.

    Args:
        successes: 성공 건수.
        n: 전체 건수.
        z: 정규분포 분위수.

    Returns:
        비율의 하한 추정값(0~1).
    """
    if n <= 0:
        return 0.0
    p = successes / n
    denom = 1.0 + z * z / n
    center = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (center - margin) / denom)


def demo_cases(seed: int = 20261024) -> list[Case]:
    """내장 데모 표본을 생성한다.

    가중치 파일이 없어도 집계·판정 로직을 그대로 확인할 수 있도록,
    환경별 난이도를 반영한 결정적(seed 고정) 가상 측정값을 만든다.
    실제 검증에서는 ``--manifest`` 로 측정 결과를 넣는다.

    Args:
        seed: 난수 시드(고정하면 실행마다 같은 결과).

    Returns:
        표본 목록(fp32 + INT8 동일 표본 2벌).
    """
    rng = random.Random(seed)
    plates = ["12가3456", "서울34나5678", "87로2345", "11바3456", "45다7890"]
    # 환경별 (검출 실패율, 문자 오인식률, 기준 지연 ms)
    profile = {
        "주간":     (0.000, 0.0003, 56.0),
        "야간":     (0.004, 0.0016, 62.0),
        "우천":     (0.008, 0.0030, 64.0),
        "역광":     (0.006, 0.0022, 60.0),
        "원거리":   (0.014, 0.0048, 58.0),
        "고속이동": (0.010, 0.0032, 59.0),
        "기울기":   (0.005, 0.0018, 63.0),
        "부분가림": (0.030, 0.0085, 61.0),
    }
    cases: list[Case] = []
    for quant, lat_scale, err_scale, mem in (("fp32", 1.0, 1.0, 212.0),
                                             ("int8", 0.62, 1.35, 96.0)):
        for env, (n, _, _) in ENV_PLAN.items():
            miss, cerr, base = profile[env]
            for i in range(n):
                truth = plates[i % len(plates)]
                detected = rng.random() >= miss
                pred: str | None = None
                if detected:
                    chars = list(truth)
                    for pos, ch in enumerate(chars):
                        if rng.random() < cerr * err_scale:
                            chars[pos] = "8" if ch.isdigit() else "하"
                    pred = "".join(chars)
                cases.append(Case(
                    env=env,
                    truth=truth,
                    pred=pred,
                    detected=detected,
                    session=f"S{(i % 6) + 1:02d}",
                    vehicle_id=f"V{100 + (i % 23)}",
                    trained_vehicle=False,   # 독립 테스트 세트
                    detect_ms=round((base * 0.66 + rng.uniform(-4, 9)) * lat_scale, 1),
                    ocr_ms=round((base * 0.34 + rng.uniform(-2, 6)) * lat_scale, 1),
                    rss_mb=round(mem + rng.uniform(-6, 14), 1),
                    quant=quant,
                ))
    return cases


def load_manifest(path: Path) -> list[Case]:
    """측정 결과 JSON 파일을 Case 목록으로 읽는다.

    JSON 배열 또는 JSON Lines 둘 다 받아들인다. 누락 필드는 기본값으로 채우고,
    알 수 없는 필드는 무시한다(AI 응답을 그대로 붙여 넣을 수 있게).

    Args:
        path: 파일 경로.

    Returns:
        표본 목록.
    """
    raw = path.read_text(encoding="utf-8").strip()
    records = (json.loads(raw) if raw.startswith("[")
               else [json.loads(line) for line in raw.splitlines() if line.strip()])
    out: list[Case] = []
    for r in records:
        out.append(Case(
            env=str(r.get("env", "미분류")),
            truth=str(r.get("truth", "")),
            pred=r.get("pred"),
            detected=bool(r.get("detected", r.get("pred") is not None)),
            session=str(r.get("session", "S00")),
            vehicle_id=str(r.get("vehicle_id", "V000")),
            trained_vehicle=bool(r.get("trained_vehicle", False)),
            detect_ms=float(r.get("detect_ms", 0.0)),
            ocr_ms=float(r.get("ocr_ms", 0.0)),
            rss_mb=float(r.get("rss_mb", 0.0)),
            quant=str(r.get("quant", "fp32")),
        ))
    return out


def print_env_table(cases: list[Case], label: str) -> dict[str, Agg]:
    """환경별 분해 결과표를 출력하고 집계를 반환한다."""
    by_env = aggregate(cases, lambda c: c.env)
    print(f"\n[환경별 분해 — {label}]")
    print(f"  {'환경':<9}{'표본':>5}{'검출률':>9}{'문자정확':>9}"
          f"{'완전일치':>9}{'하한':>8}{'p95ms':>8}  판정")
    for env, (_, dmin, emin) in ENV_PLAN.items():
        a = by_env.get(env)
        if a is None:
            continue
        lower = wilson_lower(a.exact, a.n)
        ok = a.detect_rate >= dmin and a.exact_rate >= emin
        print(f"  {env:<9}{a.n:>5}{a.detect_rate * 100:>8.2f}%"
              f"{a.char_acc * 100:>8.2f}%{a.exact_rate * 100:>8.2f}%"
              f"{lower * 100:>7.1f}%{a.pct(0.95):>8.1f}  "
              f"{'합격' if ok else '재작업'}")
    return by_env


def print_overall(cases: list[Case], label: str) -> Agg:
    """전체 집계와 합격 판정을 출력한다."""
    a = aggregate(cases, lambda c: "전체")["전체"]
    print(f"\n[전체 집계 — {label}]")
    print(f"  표본 수          : {a.n}건 "
          f"(미학습 차량 전용: {all(not c.trained_vehicle for c in cases)})")
    print(f"  검출률           : {a.detect_rate * 100:.2f}% "
          f"(기준 {OVERALL_DETECT_MIN * 100:.1f}%) "
          f"{'OK' if a.detect_rate >= OVERALL_DETECT_MIN else 'NG'}")
    print(f"  OCR 문자 정확도  : {a.char_acc * 100:.2f}% "
          f"(기준 {OVERALL_CHAR_MIN * 100:.1f}%) "
          f"{'OK' if a.char_acc >= OVERALL_CHAR_MIN else 'NG'}")
    print(f"  완전일치율       : {a.exact_rate * 100:.2f}% "
          f"[95% 하한 {wilson_lower(a.exact, a.n) * 100:.2f}%] "
          f"(기준 {OVERALL_EXACT_MIN * 100:.1f}%) "
          f"{'OK' if a.exact_rate >= OVERALL_EXACT_MIN else 'NG'}")
    print(f"  지연 p50/p95/p99 : {a.pct(0.50):.1f} / {a.pct(0.95):.1f} "
          f"/ {a.pct(0.99):.1f} ms "
          f"(p95 기준 {LATENCY_P95_MAX:.0f}ms) "
          f"{'OK' if a.pct(0.95) <= LATENCY_P95_MAX else 'NG'}")
    print(f"  RSS 최대         : {a.mem_peak:.1f} MB")
    return a


def print_independence(cases: list[Case]) -> None:
    """세션 독립성 점검 — 세션별로 성능이 크게 갈리는지 본다."""
    by_s = aggregate(cases, lambda c: c.session)
    rates = [a.exact_rate for a in by_s.values()]
    print("\n[세션 독립 테스트]")
    print(f"  {'세션':<6}{'표본':>6}{'완전일치':>10}")
    for name in sorted(by_s):
        a = by_s[name]
        print(f"  {name:<6}{a.n:>6}{a.exact_rate * 100:>9.2f}%")
    spread = (max(rates) - min(rates)) * 100 if rates else 0.0
    print(f"  세션 간 편차 {spread:.2f}%p "
          f"→ {'허용(3%p 이내)' if spread <= 3.0 else '세션 의존 의심'}")


def print_quant_compare(cases: list[Case]) -> None:
    """INT8 변환 전후를 같은 표본에서 비교한다."""
    by_q = aggregate(cases, lambda c: c.quant)
    fp = by_q.get("fp32")
    q8 = by_q.get("int8")
    print("\n[INT8 변환 전후 비교 — 동일 표본]")
    if not fp or not q8:
        print("  두 구성의 측정값이 모두 있어야 비교할 수 있다.")
        return
    print(f"  {'지표':<16}{'FP32':>10}{'INT8':>10}{'차이':>10}")
    rows = [
        ("검출률(%)", fp.detect_rate * 100, q8.detect_rate * 100),
        ("문자정확도(%)", fp.char_acc * 100, q8.char_acc * 100),
        ("완전일치율(%)", fp.exact_rate * 100, q8.exact_rate * 100),
        ("p50 지연(ms)", fp.pct(0.50), q8.pct(0.50)),
        ("p95 지연(ms)", fp.pct(0.95), q8.pct(0.95)),
        ("RSS 최대(MB)", fp.mem_peak, q8.mem_peak),
    ]
    for name, a, b in rows:
        print(f"  {name:<16}{a:>10.2f}{b:>10.2f}{b - a:>+10.2f}")
    drop = fp.exact_rate - q8.exact_rate
    print(f"  완전일치율 하락 {drop * 100:.2f}%p "
          f"(허용 {INT8_DROP_MAX * 100:.0f}%p) "
          f"→ {'INT8 채택 가능' if drop <= INT8_DROP_MAX else 'INT8 보류'}")


def write_csv(path: Path, by_env: dict[str, Agg]) -> None:
    """환경별 집계를 CSV 로 저장한다(보고서 첨부용)."""
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["환경", "표본수", "검출률", "문자정확도",
                    "완전일치율", "완전일치_95하한", "p50_ms", "p95_ms", "RSS_MB"])
        for env, a in by_env.items():
            w.writerow([env, a.n, f"{a.detect_rate:.4f}", f"{a.char_acc:.4f}",
                        f"{a.exact_rate:.4f}", f"{wilson_lower(a.exact, a.n):.4f}",
                        f"{a.pct(0.50):.1f}", f"{a.pct(0.95):.1f}",
                        f"{a.mem_peak:.1f}"])
    print(f"\nCSV 저장: {path}")


def main(argv: list[str] | None = None) -> int:
    """명령행 진입점."""
    ap = argparse.ArgumentParser(description="실환경 성능 검증 집계기")
    ap.add_argument("--manifest", type=Path, default=None,
                    help="측정 결과 JSON/JSONL 경로(없으면 내장 데모 표본)")
    ap.add_argument("--csv", type=Path, default=None, help="환경별 집계 CSV 저장 경로")
    args = ap.parse_args(argv)

    if args.manifest:
        if not args.manifest.is_file():
            print(f"[안내] manifest 를 찾을 수 없다: {args.manifest}")
            print("       내장 데모 표본으로 대체 실행한다.")
            cases = demo_cases()
        else:
            cases = load_manifest(args.manifest)
    else:
        print("[안내] manifest 미지정 → 내장 데모 표본(결정적 생성)으로 실행한다.")
        cases = demo_cases()

    print("=" * 74)
    print("실환경 성능 검증 리포트")
    print("=" * 74)

    fp32 = [c for c in cases if c.quant == "fp32"]
    by_env = print_env_table(fp32, "FP32")
    print_overall(fp32, "FP32")
    print_independence(fp32)
    print_quant_compare(cases)

    weak = [e for e, (_, d, x) in ENV_PLAN.items()
            if e in by_env and (by_env[e].detect_rate < d
                                or by_env[e].exact_rate < x)]
    print("\n[조치 대상]")
    if weak:
        for e in weak:
            print(f"  - {e}: 기준 미달 → 해당 환경 데이터 증강 후 재학습·재측정")
    else:
        print("  - 모든 환경이 기준을 충족한다.")

    if args.csv:
        write_csv(args.csv, by_env)
    return 0


if __name__ == "__main__":
    sys.exit(main())
