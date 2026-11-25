# -*- coding: utf-8 -*-
"""프레임 / 이벤트 / 차량 단위 인식률 집계.

같은 데이터로도 '무엇을 1건으로 세는가'에 따라 인식률이 크게 달라진다.
  - 프레임 단위: 이미지 한 장을 1건으로 센다. 가장 보수적이고 낮게 나온다.
  - 이벤트 단위: 차량 1회 통과를 1건으로 센다. 운영 지표로 쓰는 값이다.
  - 차량 단위: 차량 1대를 1건으로 센다. 특정 차량이 반복 실패하는지 본다.

이 모듈은 프레임 레코드만 입력으로 받아 세 단계를 모두 계산한다.
프레임 → 이벤트 결합은 ch03/08_postprocess/multi_frame.py 와 같은
'자리별 가중 다수결'을 쓴다.
"""
from __future__ import annotations

import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "08_postprocess"))

from metrics_ocr import evaluate, is_exact, normalize   # noqa: E402

try:
    from multi_frame import FrameResult, fuse           # noqa: E402
    _HAS_FUSION = True
except Exception:                                        # noqa: BLE001
    _HAS_FUSION = False
    print("[안내] multi_frame 모듈을 찾지 못해 내장 다수결로 대체한다.")


@dataclass
class FrameRecord:
    """프레임 한 장의 평가 레코드."""

    event_id: str              # 통과 이벤트 식별자(차량 1회 통과)
    vehicle_id: str            # 차량 식별자(정답 번호판과 1:1)
    gt: str                    # 정답 번호판
    pred: str                  # 프레임 단위 예측
    confidence: float = 0.0
    quality: float = 1.0
    detected: bool = True      # 번호판 검출에 성공했는지
    condition: str = "주간"    # 환경 라벨


@dataclass
class LevelStat:
    """한 집계 단위의 통계."""

    name: str
    total: int = 0
    exact: int = 0
    detected: int = 0
    failures: list[str] = field(default_factory=list)

    @property
    def exact_rate(self) -> float:
        return self.exact / self.total if self.total else 0.0

    @property
    def detect_rate(self) -> float:
        return self.detected / self.total if self.total else 0.0

    def line(self) -> str:
        return (f"{self.name:<12s} 건수 {self.total:>5d}  "
                f"검출률 {self.detect_rate:7.3%}  "
                f"완전일치율 {self.exact_rate:7.3%}  "
                f"실패 {self.total - self.exact:>4d}건")


def _fallback_vote(preds: list[tuple[str, float]]) -> str:
    """multi_frame 을 못 쓸 때의 간단 다수결(문자열 단위 가중 투표)."""
    votes: dict[str, float] = defaultdict(float)
    for t, w in preds:
        t = normalize(t)
        if t:
            votes[t] += max(w, 0.01)
    if not votes:
        return ""
    return max(votes.items(), key=lambda kv: kv[1])[0]


def fuse_event(records: list[FrameRecord]) -> str:
    """이벤트에 속한 프레임들을 결합해 이벤트 확정 번호판을 만든다."""
    usable = [r for r in records if r.detected and normalize(r.pred)]
    if not usable:
        return ""
    if _HAS_FUSION:
        frames = [FrameResult(r.pred, r.confidence, [], r.quality)
                  for r in usable]
        return fuse(frames).plate_number
    return _fallback_vote([(r.pred, r.confidence * r.quality) for r in usable])


def aggregate(records: list[FrameRecord]) -> dict[str, LevelStat]:
    """프레임/이벤트/차량 세 단위를 한 번에 집계한다."""
    out: dict[str, LevelStat] = {
        "frame": LevelStat("프레임 단위"),
        "event": LevelStat("이벤트 단위"),
        "vehicle": LevelStat("차량 단위"),
    }

    # (1) 프레임 단위
    fs = out["frame"]
    for r in records:
        fs.total += 1
        if r.detected:
            fs.detected += 1
        if r.detected and is_exact(r.gt, r.pred):
            fs.exact += 1
        else:
            fs.failures.append(f"{r.event_id}/{r.gt}->{r.pred or '(없음)'}")

    # (2) 이벤트 단위 — 프레임을 결합한 뒤 비교
    by_event: dict[str, list[FrameRecord]] = defaultdict(list)
    for r in records:
        by_event[r.event_id].append(r)

    es = out["event"]
    event_results: dict[str, tuple[str, str, bool]] = {}
    for eid, rs in by_event.items():
        gt = normalize(rs[0].gt)
        merged = fuse_event(rs)
        ok = merged == gt and bool(gt)
        es.total += 1
        if any(r.detected for r in rs):
            es.detected += 1
        if ok:
            es.exact += 1
        else:
            es.failures.append(f"{eid} {gt} -> {merged or '(없음)'}")
        event_results[eid] = (rs[0].vehicle_id, gt, ok)

    # (3) 차량 단위 — 그 차량의 모든 이벤트가 성공해야 성공으로 센다
    by_vehicle: dict[str, list[bool]] = defaultdict(list)
    for _eid, (vid, _gt, ok) in event_results.items():
        by_vehicle[vid].append(ok)

    vs = out["vehicle"]
    for vid, oks in by_vehicle.items():
        vs.total += 1
        vs.detected += 1
        if all(oks):
            vs.exact += 1
        else:
            bad = sum(1 for o in oks if not o)
            vs.failures.append(f"{vid} 이벤트 {len(oks)}건 중 {bad}건 실패")
    return out


def per_vehicle_table(records: list[FrameRecord]) -> list[tuple[str, str, int, int, float]]:
    """차량별 이벤트 성공률 표를 만든다(반복 실패 차량 식별용)."""
    by_event: dict[str, list[FrameRecord]] = defaultdict(list)
    for r in records:
        by_event[r.event_id].append(r)
    agg: dict[str, list[bool]] = defaultdict(list)
    plate: dict[str, str] = {}
    for _eid, rs in by_event.items():
        vid = rs[0].vehicle_id
        plate[vid] = normalize(rs[0].gt)
        agg[vid].append(fuse_event(rs) == normalize(rs[0].gt))
    rows = []
    for vid, oks in agg.items():
        n, ok = len(oks), sum(oks)
        rows.append((vid, plate[vid], n, ok, ok / n if n else 0.0))
    rows.sort(key=lambda x: (x[4], -x[2]))
    return rows


def builtin_records() -> list[FrameRecord]:
    """교재 실습용 데이터. 차량 5대 / 이벤트 8건 / 프레임 40장."""
    recs: list[FrameRecord] = []

    def add(eid: str, vid: str, gt: str, preds: list[tuple[str, float, float]],
            cond: str = "주간") -> None:
        for t, c, q in preds:
            recs.append(FrameRecord(eid, vid, gt, t, c, q,
                                    detected=bool(t), condition=cond))

    # V1: 전 프레임 양호
    add("E1", "V1", "12가3456", [("12가3456", .97, .92), ("12가3456", .96, .90),
                                 ("12가3456", .95, .88), ("12가3456", .94, .86),
                                 ("12가345", .41, .35)])
    # V1 두 번째 통과 — 야간, 프레임 품질 저하
    add("E2", "V1", "12가3456", [("12가3456", .72, .55), ("12가3458", .61, .48),
                                 ("12가3456", .69, .52), ("", .00, .20),
                                 ("12가3456", .66, .50)], "야간")
    # V2: 역광이지만 다수결로 복구
    add("E3", "V2", "87로2345", [("87도2345", .58, .45), ("87로2345", .66, .52),
                                 ("87로2345", .64, .50), ("87로2845", .51, .42),
                                 ("87로2345", .62, .49)], "역광")
    # V2 두 번째 통과 — 실패(다수가 같은 오류)
    add("E4", "V2", "87로2345", [("87도2345", .61, .47), ("87도2345", .59, .45),
                                 ("87도2345", .57, .44), ("87로2345", .48, .40),
                                 ("", .00, .18)], "우천")
    # V3: 원거리, 프레임 단위로는 자주 틀리지만 이벤트로는 성공
    add("E5", "V3", "서울34나5678",
        [("서울34나5678", .62, .50), ("서울34나567", .44, .36),
         ("서울34나5678", .59, .48), ("서울34다5678", .47, .39),
         ("서울34나5678", .61, .49)], "원거리")
    # V4: 기울기 심함 — 이벤트 실패
    add("E6", "V4", "123가4567",
        [("128가4567", .49, .38), ("123가456", .42, .33),
         ("128가4567", .52, .41), ("", .00, .15),
         ("128가4567", .50, .40)], "기울기")
    # V5: 정상
    add("E7", "V5", "34허5678", [("34허5678", .95, .90), ("34허5678", .94, .89),
                                 ("34허5678", .93, .87), ("34허5678", .92, .86),
                                 ("34허5678", .91, .85)])
    add("E8", "V5", "34허5678", [("34허5678", .93, .88), ("34허5B78", .55, .44),
                                 ("34허5678", .91, .86), ("34허5678", .90, .85),
                                 ("34허5678", .89, .84)])
    return recs


def _demo() -> None:
    recs = builtin_records()
    stats = aggregate(recs)

    print("=== 집계 단위별 인식률 ===")
    for key in ("frame", "event", "vehicle"):
        print(stats[key].line())

    # 프레임 단위 CER 도 함께 본다.
    pairs = [(r.gt, r.pred) for r in recs]
    m = evaluate(pairs)
    print("\n=== 프레임 단위 문자 지표 ===")
    print(f"CER                : {m.cer:.6f} ({m.cer * 100:.3f}%)")
    print(f"Character Accuracy : {m.char_accuracy * 100:.3f}%")
    print(f"Exact Match Rate   : {m.exact_match_rate * 100:.3f}%")

    print("\n=== 차량별 이벤트 성공률 ===")
    print(f"{'차량':<6s}{'번호판':<12s}{'이벤트':>6s}{'성공':>5s}  성공률")
    for vid, plate, n, ok, rate in per_vehicle_table(recs):
        print(f"{vid:<6s}{plate:<12s}{n:>6d}{ok:>5d}  {rate:7.1%}")

    print("\n=== 이벤트 단위 실패 목록 ===")
    for f in stats["event"].failures:
        print(" -", f)

    print("\n=== 환경 라벨별 프레임 수 ===")
    for cond, n in Counter(r.condition for r in recs).most_common():
        print(f"{cond:<8s}{n:>4d}장")


if __name__ == "__main__":
    _demo()
