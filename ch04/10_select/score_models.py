# -*- coding: utf-8 -*-
"""가중합 종합 점수를 계산해 최종 모델을 선정한다.

점수 산정 절차
  1) 하드 제약(운영 요건) 위반 모델을 먼저 탈락시킨다.
  2) 남은 모델의 각 지표를 0~1 로 정규화한다.
       높을수록 좋은 지표: (v - min) / (max - min)
       낮을수록 좋은 지표: (max - v) / (max - min)
     지연은 값의 폭이 매우 크므로 로그 스케일로 정규화하는 선택지를 둔다.
  3) 가중합으로 종합 점수를 만든다.
       완전 일치율 0.40 + 검출 재현율 0.20 + 지연 0.25 + 메모리 0.15
  4) 가중치를 흔들어(민감도 분석) 1위가 바뀌는지 확인한다.

fair_eval.py 의 결과 JSON 이 있으면 그것을 읽고, 없으면 내장 샘플로 동작한다.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

# 예시 측정 환경(8코어 CPU, 입력 640, 단일 스레드)에서 얻은 한 번의 관측값.
# 하드웨어·데이터셋이 바뀌면 달라지므로 이 값을 결론으로 삼지 않는다.
SAMPLE = [
    dict(key="yolo_n", name="경량 YOLO-nano", exact_match_rate=0.958,
         detect_recall=0.988, p50_ms=38.4, p95_ms=54.1, memory_mb=210.0,
         size_mb=5.3),
    dict(key="yolo_s", name="경량 YOLO-small", exact_match_rate=0.964,
         detect_recall=0.993, p50_ms=74.6, p95_ms=101.3, memory_mb=268.0,
         size_mb=18.9),
    dict(key="mbnet_det", name="MobileNetV3+SSDLite", exact_match_rate=0.931,
         detect_recall=0.961, p50_ms=29.8, p95_ms=41.7, memory_mb=186.0,
         size_mb=13.9),
    dict(key="detr_r50", name="DETR(R50)", exact_match_rate=0.941,
         detect_recall=0.972, p50_ms=612.4, p95_ms=781.9, memory_mb=742.0,
         size_mb=166.0),
    dict(key="ddetr_r50", name="Deformable DETR(R50)", exact_match_rate=0.953,
         detect_recall=0.986, p50_ms=948.7, p95_ms=1224.5, memory_mb=864.0,
         size_mb=161.0),
    dict(key="dino_r50", name="DINO(R50)", exact_match_rate=0.959,
         detect_recall=0.991, p50_ms=1287.3, p95_ms=1662.8, memory_mb=1018.0,
         size_mb=189.0),
    dict(key="yolos_s", name="YOLOS-small", exact_match_rate=0.918,
         detect_recall=0.948, p50_ms=1104.2, p95_ms=1398.6, memory_mb=906.0,
         size_mb=123.0),
    dict(key="swin_dino", name="Swin-T + DINO", exact_match_rate=0.961,
         detect_recall=0.992, p50_ms=1742.9, p95_ms=2210.4, memory_mb=1264.0,
         size_mb=193.0),
    dict(key="vitdet_b", name="ViTDet(ViT-Base)", exact_match_rate=0.955,
         detect_recall=0.989, p50_ms=3418.6, p95_ms=4102.7, memory_mb=2186.0,
         size_mb=445.0),
]

WEIGHTS = {"exact_match_rate": 0.40, "detect_recall": 0.20,
           "latency": 0.25, "memory": 0.15}

DIRECTION = {"exact_match_rate": "high", "detect_recall": "high",
             "latency": "low", "memory": "low"}

CONSTRAINTS = {"max_p95_ms": 300.0, "max_memory_mb": 600.0,
               "min_detect_recall": 0.95}

METRIC_FIELD = {"exact_match_rate": "exact_match_rate",
                "detect_recall": "detect_recall",
                "latency": "p95_ms", "memory": "memory_mb"}


def load_rows(path: str) -> list[dict]:
    """fair_eval 결과를 읽는다. 없으면 내장 샘플을 쓴다."""
    p = pathlib.Path(path)
    if not p.exists():
        print(f"[알림] {path} 가 없다. 내장 샘플(예시 측정값)로 계산한다.\n",
              file=sys.stderr)
        return [dict(r) for r in SAMPLE]
    data = json.loads(p.read_text(encoding="utf-8"))
    rows = []
    for r in data.get("results", []):
        row = dict(r)
        row.setdefault("name", row.get("key", "?"))
        rows.append(row)
    if not rows:
        return [dict(r) for r in SAMPLE]
    return rows


def check_constraints(rows: list[dict]) -> tuple[list[dict], list[tuple[dict, str]]]:
    """하드 제약을 검사해 통과 목록과 탈락 사유를 돌려준다."""
    passed, failed = [], []
    for r in rows:
        why = []
        if r.get("p95_ms", 0) > CONSTRAINTS["max_p95_ms"]:
            why.append(f"p95 {r['p95_ms']:.1f}ms > {CONSTRAINTS['max_p95_ms']:.0f}ms")
        if r.get("memory_mb", 0) > CONSTRAINTS["max_memory_mb"]:
            why.append(f"메모리 {r['memory_mb']:.0f}MB > "
                       f"{CONSTRAINTS['max_memory_mb']:.0f}MB")
        if r.get("detect_recall", 1.0) < CONSTRAINTS["min_detect_recall"]:
            why.append(f"재현율 {r['detect_recall']:.3f} < "
                       f"{CONSTRAINTS['min_detect_recall']:.2f}")
        if why:
            failed.append((r, ", ".join(why)))
        else:
            passed.append(r)
    return passed, failed


def normalize(rows: list[dict], metric: str, log_scale: bool) -> dict[str, float]:
    """한 지표를 0~1 로 정규화한다. 모든 값이 같으면 1.0 을 준다."""
    field = METRIC_FIELD[metric]
    vals = {}
    for r in rows:
        v = float(r.get(field, 0.0))
        if log_scale and DIRECTION[metric] == "low":
            v = math.log10(max(v, 1e-6))
        vals[r["key"]] = v
    lo, hi = min(vals.values()), max(vals.values())
    if hi - lo < 1e-12:
        return {k: 1.0 for k in vals}
    if DIRECTION[metric] == "high":
        return {k: (v - lo) / (hi - lo) for k, v in vals.items()}
    return {k: (hi - v) / (hi - lo) for k, v in vals.items()}


def score(rows: list[dict], weights: dict[str, float],
          log_scale: bool) -> list[dict]:
    """가중합 종합 점수를 계산해 내림차순으로 돌려준다."""
    norms = {m: normalize(rows, m, log_scale) for m in weights}
    out = []
    for r in rows:
        parts = {m: norms[m][r["key"]] * w for m, w in weights.items()}
        out.append({**r, "norm": {m: norms[m][r["key"]] for m in weights},
                    "parts": parts, "total": sum(parts.values())})
    return sorted(out, key=lambda x: -x["total"])


def print_scores(ranked: list[dict], weights: dict[str, float]) -> None:
    head = (f"{'순위':>3} {'모델':<22} {'일치율':>7} {'재현율':>7} "
            f"{'p95(ms)':>9} {'메모리':>8} {'종합점수':>9}")
    print(head)
    print("-" * 80)
    for i, r in enumerate(ranked, 1):
        print(f"{i:>3} {r['name']:<22} {r['exact_match_rate']:>7.3f} "
              f"{r['detect_recall']:>7.3f} {r['p95_ms']:>9.1f} "
              f"{r['memory_mb']:>8.0f} {r['total']:>9.4f}")
    print("-" * 80)
    print()
    print("기여도 분해 (가중치 x 정규화 점수)")
    print(f"{'모델':<22} " + " ".join(f"{m:>18}" for m in weights) + f" {'합계':>9}")
    for r in ranked:
        cells = " ".join(f"{r['parts'][m]:>18.4f}" for m in weights)
        print(f"{r['name']:<22} {cells} {r['total']:>9.4f}")


def sensitivity(rows: list[dict], log_scale: bool) -> None:
    """가중치를 바꿔 1위가 유지되는지 확인한다."""
    scenarios = {
        "기본(정확도 0.40/0.20, 비용 0.25/0.15)": WEIGHTS,
        "정확도 더 중시(0.50/0.25, 0.15/0.10)":
            {"exact_match_rate": 0.50, "detect_recall": 0.25,
             "latency": 0.15, "memory": 0.10},
        "속도 더 중시(0.30/0.15, 0.40/0.15)":
            {"exact_match_rate": 0.30, "detect_recall": 0.15,
             "latency": 0.40, "memory": 0.15},
        "균등(0.25 x 4)":
            {"exact_match_rate": 0.25, "detect_recall": 0.25,
             "latency": 0.25, "memory": 0.25},
    }
    print()
    print("가중치 민감도 분석 — 1위가 바뀌는지 확인한다")
    print("-" * 80)
    print(f"{'시나리오':<40} {'1위':<22} {'점수':>8} {'2위와 차':>9}")
    for label, w in scenarios.items():
        assert abs(sum(w.values()) - 1.0) < 1e-9, "가중치 합계가 1.0 이어야 한다"
        r = score(rows, w, log_scale)
        gap = r[0]["total"] - r[1]["total"] if len(r) > 1 else 0.0
        print(f"{label:<40} {r[0]['name']:<22} {r[0]['total']:>8.4f} {gap:>9.4f}")
    print("-" * 80)


def main() -> None:
    ap = argparse.ArgumentParser(description="가중합 종합 점수 및 최종 모델 선정")
    ap.add_argument("--input", default="fair_eval_result.json")
    ap.add_argument("--log-latency", action="store_true",
                    help="지연/메모리를 로그 스케일로 정규화")
    ap.add_argument("--no-constraints", action="store_true",
                    help="하드 제약을 적용하지 않고 전체 순위를 본다")
    ap.add_argument("--out", default="", help="결과 JSON 저장 경로")
    args = ap.parse_args()

    rows = load_rows(args.input)
    print("=" * 80)
    print("검출기 최종 선정 — 가중합 종합 점수")
    print("=" * 80)
    print(f"  가중치: " + ", ".join(f"{m} {w}" for m, w in WEIGHTS.items()))
    print(f"  합계 검증: {sum(WEIGHTS.values()):.2f}")
    print(f"  하드 제약: p95 <= {CONSTRAINTS['max_p95_ms']:.0f}ms, "
          f"메모리 <= {CONSTRAINTS['max_memory_mb']:.0f}MB, "
          f"재현율 >= {CONSTRAINTS['min_detect_recall']:.2f}")
    print(f"  정규화  : {'로그 스케일' if args.log_latency else '선형'}")
    print(f"  후보 수 : {len(rows)}")
    print()

    if args.no_constraints:
        passed, failed = rows, []
    else:
        passed, failed = check_constraints(rows)

    print("[1] 하드 제약 심사")
    if failed:
        for r, why in failed:
            print(f"  탈락  {r['name']:<22} <- {why}")
    for r in passed:
        print(f"  통과  {r['name']:<22}")
    print()

    if not passed:
        print("통과한 모델이 없다. 제약을 재검토하거나 입력 해상도를 낮춰 다시 측정한다.")
        return

    print("[2] 통과 모델 종합 점수")
    ranked = score(passed, WEIGHTS, args.log_latency)
    print_scores(ranked, WEIGHTS)

    sensitivity(passed, args.log_latency)

    print()
    print("[3] 참고: 제약을 풀었을 때의 전체 순위")
    all_ranked = score(rows, WEIGHTS, args.log_latency)
    for i, r in enumerate(all_ranked, 1):
        flag = "" if any(p["key"] == r["key"] for p in passed) else "  (제약 위반)"
        print(f"  {i:>2}. {r['name']:<22} {r['total']:.4f}{flag}")

    best = ranked[0]
    print()
    print("=" * 80)
    print(f"[4] 최종 선정: {best['name']}")
    print("=" * 80)
    print(f"  종합 점수       : {best['total']:.4f}")
    print(f"  완전 일치율     : {best['exact_match_rate']:.3f} "
          f"(정규화 {best['norm']['exact_match_rate']:.3f})")
    print(f"  검출 재현율     : {best['detect_recall']:.3f} "
          f"(정규화 {best['norm']['detect_recall']:.3f})")
    print(f"  CPU p95 지연    : {best['p95_ms']:.1f} ms "
          f"(정규화 {best['norm']['latency']:.3f})")
    print(f"  메모리          : {best['memory_mb']:.0f} MB "
          f"(정규화 {best['norm']['memory']:.3f})")
    if len(ranked) > 1:
        print(f"  2위와 점수 차   : {best['total'] - ranked[1]['total']:.4f} "
              f"({ranked[1]['name']})")
    print()
    print("  선정 근거는 점수 하나가 아니라 다음 세 가지를 함께 본다.")
    print("    (a) 하드 제약을 통과했는가")
    print("    (b) 가중치를 흔들어도 순위가 유지되는가")
    print("    (c) 떨어진 모델과의 정확도 차이가 운영상 의미 있는 크기인가")

    if args.out:
        payload = {"weights": WEIGHTS, "constraints": CONSTRAINTS,
                   "log_latency": args.log_latency,
                   "ranked": [{k: v for k, v in r.items() if k != "parts"}
                              for r in ranked],
                   "selected": best["name"]}
        pathlib.Path(args.out).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print()
        print(f"  결과를 {args.out} 에 저장했다.")


if __name__ == "__main__":
    main()
