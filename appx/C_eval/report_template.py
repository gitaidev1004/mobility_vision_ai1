# -*- coding: utf-8 -*-
"""부록 C — 평가 결과 JSON 을 교재 표 양식으로 자동 변환하는 도구.

6장의 평가 스크립트나 `POST /api/v1/models/benchmark` 응답을 그대로 받아
부록 C 의 일곱 가지 평가표를 채워 출력한다.

생성하는 표
    1) 번호판 검출 평가표      2) OCR 평가표
    3) 환경별 평가표           4) CPU 성능 평가표
    5) 메모리 사용량 평가표    6) 모델 비교표
    7) 최종 모델 선정표(가중 종합 점수)

입력 JSON 스키마(필요한 키만 있으면 되고, 없는 칸은 `-` 로 비워 둔다):
    {
      "project": "...", "measured_at": "...", "test_set": {...},
      "models": [
        {"name": "...", "detector": "...", "recognizer": "...",
         "quantized": false,
         "detect": {"tp":.., "fp":.., "fn":.., "map50":.., "map50_95":..},
         "ocr": {"char_total":.., "char_correct":.., "edit_distance":..,
                 "exact_match":.., "sample_total":..},
         "by_env": {"DY": {"exact_match_rate":.., "detect_recall":..}, ...},
         "cpu": {"latency_ms": [..], "threads":.., "fps":..},
         "memory": {"rss_peak_mb":.., "model_size_mb":..}}
      ]
    }

사용법:
    python appx/C_eval/report_template.py result.json
    python appx/C_eval/report_template.py result.json --markdown
    python appx/C_eval/report_template.py --demo
    python appx/C_eval/report_template.py --blank
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys

DASH = "-"

# 환경 코드 → 표시 이름(부록 B 와 동일)
ENV_LABEL = {
    "DY": "주간 정상", "NI": "야간", "RA": "우천·강설", "BL": "역광",
    "FA": "원거리", "TI": "기울어짐", "OC": "부분 가림",
}

# 최종 선정표의 지표별 가중치 — 합계 1.00
WEIGHTS = {
    "exact_match_rate": 0.40,   # 문자열 완전 일치율
    "detect_recall": 0.20,      # 검출 재현율
    "latency": 0.25,            # CPU p95 지연시간(낮을수록 좋음)
    "memory": 0.10,             # 최대 RSS(낮을수록 좋음)
    "size": 0.05,               # 모델 파일 크기(낮을수록 좋음)
}


# ---------------------------------------------------------------- 표 렌더링

def width(text: object) -> int:
    """한글을 폭 2로 계산한 표시 폭."""
    return sum(2 if ord(ch) > 0x1100 else 1 for ch in str(text))


def render(header: list[str], rows: list[list[object]],
           markdown: bool = False) -> str:
    """고정폭 표 또는 마크다운 표를 문자열로 만든다."""
    if markdown:
        out = ["| " + " | ".join(header) + " |",
               "|" + "|".join("---" for _ in header) + "|"]
        out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
        return "\n".join(out)

    data = [header, *rows] if rows else [header]
    widths = [max(width(r[c]) for r in data) for c in range(len(header))]

    def pad(text: object, w: int) -> str:
        return str(text) + " " * max(0, w - width(text))

    sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    out = [sep, "| " + " | ".join(pad(header[c], widths[c])
                                  for c in range(len(header))) + " |", sep]
    for r in rows:
        out.append("| " + " | ".join(pad(r[c], widths[c])
                                     for c in range(len(header))) + " |")
    out.append(sep)
    return "\n".join(out)


def fmt(value: object, digits: int = 3) -> str:
    """수치를 고정 소수점으로, 값이 없으면 `-` 로 표시한다."""
    if value is None:
        return DASH
    if isinstance(value, bool):
        return "O" if value else "X"
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}" if isinstance(value, float) else str(value)
    return str(value)


def pctf(value: float | None) -> str:
    """0~1 비율을 백분율 문자열로 바꾼다."""
    return DASH if value is None else f"{value * 100:.2f}%"


# ---------------------------------------------------------------- 지표 계산

def detect_metrics(d: dict) -> dict:
    """TP/FP/FN 에서 정밀도·재현율·F1 을 계산한다."""
    tp = d.get("tp")
    fp = d.get("fp")
    fn = d.get("fn")
    out = {"tp": tp, "fp": fp, "fn": fn,
           "map50": d.get("map50"), "map50_95": d.get("map50_95")}
    if tp is None or fp is None or fn is None:
        out.update(precision=None, recall=None, f1=None)
        return out
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    out.update(precision=prec, recall=rec, f1=f1)
    return out


def ocr_metrics(o: dict) -> dict:
    """문자 정확도·CER·완전 일치율을 계산한다."""
    total = o.get("char_total")
    correct = o.get("char_correct")
    dist = o.get("edit_distance")
    exact = o.get("exact_match")
    n = o.get("sample_total")
    char_acc = (correct / total) if (total and correct is not None) else None
    cer = (dist / total) if (total and dist is not None) else None
    emr = (exact / n) if (n and exact is not None) else None
    return {"char_total": total, "char_acc": char_acc, "cer": cer,
            "exact_match": exact, "sample_total": n, "exact_match_rate": emr}


def cpu_metrics(c: dict) -> dict:
    """지연시간 표본에서 p50·p95·p99·평균·표준편차와 처리량을 계산한다."""
    samples = [float(v) for v in c.get("latency_ms", []) if v is not None]
    out = {"threads": c.get("threads"), "n": len(samples),
           "fps": c.get("fps")}
    if not samples:
        out.update(p50=None, p95=None, p99=None, mean=None, stdev=None)
        return out
    ordered = sorted(samples)

    def q(p: float) -> float:
        # 백분위는 선형 보간 없이 가장 가까운 상위 순위를 쓴다(보고 간결성).
        idx = min(len(ordered) - 1, max(0, int(round(p * (len(ordered) - 1)))))
        return ordered[idx]

    out.update(p50=q(0.50), p95=q(0.95), p99=q(0.99),
               mean=statistics.fmean(ordered),
               stdev=statistics.stdev(ordered) if len(ordered) > 1 else 0.0)
    if out["fps"] is None and out["mean"]:
        out["fps"] = 1000.0 / out["mean"]
    return out


def summarize(model: dict) -> dict:
    """모델 1건의 모든 지표를 한 딕셔너리로 모은다."""
    return {
        "name": model.get("name", "(이름 없음)"),
        "detector": model.get("detector", DASH),
        "recognizer": model.get("recognizer", DASH),
        "quantized": model.get("quantized", False),
        "detect": detect_metrics(model.get("detect", {})),
        "ocr": ocr_metrics(model.get("ocr", {})),
        "by_env": model.get("by_env", {}),
        "cpu": cpu_metrics(model.get("cpu", {})),
        "memory": model.get("memory", {}),
    }


# ---------------------------------------------------------------- 종합 점수

def normalize(values: list[float | None], lower_better: bool) -> list[float]:
    """측정값을 0~1 로 정규화한다. 값이 없으면 0 으로 둔다."""
    valid = [v for v in values if v is not None]
    if not valid:
        return [0.0 for _ in values]
    lo, hi = min(valid), max(valid)
    span = hi - lo
    out = []
    for v in values:
        if v is None:
            out.append(0.0)
        elif span == 0:
            out.append(1.0)
        elif lower_better:
            out.append((hi - v) / span)
        else:
            out.append((v - lo) / span)
    return out


def total_scores(models: list[dict]) -> list[float]:
    """가중 종합 점수(100점 만점)를 계산한다."""
    emr = normalize([m["ocr"]["exact_match_rate"] for m in models], False)
    rec = normalize([m["detect"]["recall"] for m in models], False)
    lat = normalize([m["cpu"]["p95"] for m in models], True)
    mem = normalize([m["memory"].get("rss_peak_mb") for m in models], True)
    siz = normalize([m["memory"].get("model_size_mb") for m in models], True)
    scores = []
    for i in range(len(models)):
        s = (emr[i] * WEIGHTS["exact_match_rate"]
             + rec[i] * WEIGHTS["detect_recall"]
             + lat[i] * WEIGHTS["latency"]
             + mem[i] * WEIGHTS["memory"]
             + siz[i] * WEIGHTS["size"])
        scores.append(round(s * 100, 1))
    return scores


# ---------------------------------------------------------------- 일곱 표

def table_detect(models: list[dict], md: bool) -> str:
    """1) 번호판 검출 평가표."""
    header = ["모델", "TP", "FP", "FN", "Precision", "Recall", "F1",
              "mAP@50", "mAP@50-95"]
    rows = []
    for m in models:
        d = m["detect"]
        rows.append([m["name"], fmt(d["tp"]), fmt(d["fp"]), fmt(d["fn"]),
                     fmt(d["precision"]), fmt(d["recall"]), fmt(d["f1"]),
                     fmt(d["map50"]), fmt(d["map50_95"])])
    return render(header, rows, md)


def table_ocr(models: list[dict], md: bool) -> str:
    """2) OCR 평가표."""
    header = ["모델", "문자 수", "문자 정확도", "CER",
              "완전 일치", "표본 수", "완전 일치율"]
    rows = []
    for m in models:
        o = m["ocr"]
        rows.append([m["name"], fmt(o["char_total"]), pctf(o["char_acc"]),
                     fmt(o["cer"], 4), fmt(o["exact_match"]),
                     fmt(o["sample_total"]), pctf(o["exact_match_rate"])])
    return render(header, rows, md)


def table_env(models: list[dict], md: bool) -> str:
    """3) 환경별 평가표."""
    codes = []
    for m in models:
        for c in m["by_env"]:
            if c not in codes:
                codes.append(c)
    codes = [c for c in ENV_LABEL if c in codes] + \
            [c for c in codes if c not in ENV_LABEL]
    if not codes:
        return render(["모델", "환경"], [["(환경별 데이터 없음)", DASH]], md)
    header = ["모델", "지표", *[f"{c}({ENV_LABEL.get(c, '?')})" for c in codes]]
    rows = []
    for m in models:
        for key, label in (("exact_match_rate", "완전 일치율"),
                           ("detect_recall", "검출 재현율")):
            cells = []
            for c in codes:
                v = m["by_env"].get(c, {}).get(key)
                cells.append(pctf(v) if isinstance(v, (int, float)) else DASH)
            rows.append([m["name"], label, *cells])
    return render(header, rows, md)


def table_cpu(models: list[dict], md: bool) -> str:
    """4) CPU 성능 평가표."""
    header = ["모델", "스레드", "표본 수", "평균(ms)", "표준편차",
              "p50(ms)", "p95(ms)", "p99(ms)", "처리량(fps)"]
    rows = []
    for m in models:
        c = m["cpu"]
        rows.append([m["name"], fmt(c["threads"]), fmt(c["n"]),
                     fmt(c["mean"], 1), fmt(c["stdev"], 1),
                     fmt(c["p50"], 1), fmt(c["p95"], 1), fmt(c["p99"], 1),
                     fmt(c["fps"], 1)])
    return render(header, rows, md)


def table_memory(models: list[dict], md: bool) -> str:
    """5) 메모리 사용량 평가표."""
    header = ["모델", "최대 RSS(MB)", "모델 파일(MB)", "양자화",
              "RSS/모델 비율"]
    rows = []
    for m in models:
        mem = m["memory"]
        rss = mem.get("rss_peak_mb")
        size = mem.get("model_size_mb")
        ratio = (rss / size) if (rss and size) else None
        rows.append([m["name"], fmt(rss, 1), fmt(size, 1),
                     fmt(m["quantized"]), fmt(ratio, 1)])
    return render(header, rows, md)


def table_compare(models: list[dict], md: bool) -> str:
    """6) 모델 비교표 — 핵심 지표만 한 줄로 모은다."""
    header = ["모델", "검출기", "인식기", "검출 Recall",
              "완전 일치율", "p95(ms)", "RSS(MB)", "크기(MB)"]
    rows = []
    for m in models:
        rows.append([m["name"], m["detector"], m["recognizer"],
                     fmt(m["detect"]["recall"]),
                     pctf(m["ocr"]["exact_match_rate"]),
                     fmt(m["cpu"]["p95"], 1),
                     fmt(m["memory"].get("rss_peak_mb"), 1),
                     fmt(m["memory"].get("model_size_mb"), 1)])
    return render(header, rows, md)


def table_final(models: list[dict], md: bool) -> str:
    """7) 최종 모델 선정표 — 가중 종합 점수와 순위."""
    scores = total_scores(models)
    order = sorted(range(len(models)), key=lambda i: -scores[i])
    rank_of = {idx: r + 1 for r, idx in enumerate(order)}
    header = ["순위", "모델", "완전일치(40)", "Recall(20)", "지연(25)",
              "메모리(10)", "크기(5)", "종합(100)", "판정"]
    emr = normalize([m["ocr"]["exact_match_rate"] for m in models], False)
    rec = normalize([m["detect"]["recall"] for m in models], False)
    lat = normalize([m["cpu"]["p95"] for m in models], True)
    mem = normalize([m["memory"].get("rss_peak_mb") for m in models], True)
    siz = normalize([m["memory"].get("model_size_mb") for m in models], True)
    rows = []
    for i in order:
        m = models[i]
        rows.append([
            rank_of[i], m["name"],
            f"{emr[i] * 40:.1f}", f"{rec[i] * 20:.1f}",
            f"{lat[i] * 25:.1f}", f"{mem[i] * 10:.1f}", f"{siz[i] * 5:.1f}",
            f"{scores[i]:.1f}",
            "채택" if rank_of[i] == 1 else "대조군",
        ])
    return render(header, rows, md)


# ---------------------------------------------------------------- 빈 양식

def blank_forms() -> str:
    """기입 전 빈 양식 일곱 벌을 출력한다(인쇄해 손으로 채울 용도)."""
    out = ["=" * 72, " 부록 C — 평가표 빈 양식", "=" * 72]
    specs = [
        ("1) 번호판 검출 평가표",
         ["모델", "TP", "FP", "FN", "Precision", "Recall", "F1",
          "mAP@50", "mAP@50-95"]),
        ("2) OCR 평가표",
         ["모델", "문자 수", "문자 정확도", "CER", "완전 일치",
          "표본 수", "완전 일치율"]),
        ("3) 환경별 평가표",
         ["모델", "지표", "DY", "NI", "RA", "BL", "FA", "TI", "OC"]),
        ("4) CPU 성능 평가표",
         ["모델", "스레드", "표본 수", "평균(ms)", "표준편차",
          "p50(ms)", "p95(ms)", "p99(ms)", "처리량(fps)"]),
        ("5) 메모리 사용량 평가표",
         ["모델", "최대 RSS(MB)", "모델 파일(MB)", "양자화", "RSS/모델 비율"]),
        ("6) 모델 비교표",
         ["모델", "검출기", "인식기", "검출 Recall", "완전 일치율",
          "p95(ms)", "RSS(MB)", "크기(MB)"]),
        ("7) 최종 모델 선정표",
         ["순위", "모델", "완전일치(40)", "Recall(20)", "지연(25)",
          "메모리(10)", "크기(5)", "종합(100)", "판정"]),
    ]
    for title, header in specs:
        out.append(f"\n[{title}]")
        out.append(render(header, [[DASH] * len(header) for _ in range(3)]))
    return "\n".join(out)


# ---------------------------------------------------------------- 데모 데이터

def demo_payload() -> dict:
    """교재 예시용 데모 측정 결과를 만든다(수치는 예시이며 환경마다 다르다)."""
    def lat(base: float, n: int = 40) -> list[float]:
        # 결정적(deterministic) 의사 난수로 지연시간 표본을 만든다.
        return [round(base + ((i * 37) % 23) - 8 + (i % 5) * 1.7, 1)
                for i in range(n)]

    return {
        "project": "모빌리티 비전 AI 프로젝트",
        "measured_at": "2026-10-24T12:00:00",
        "test_set": {"images": 1200, "sessions": 24, "vehicles": 180},
        "models": [
            {"name": "YOLO-nano + CRNN/CTC",
             "detector": "yolo", "recognizer": "crnn", "quantized": False,
             "detect": {"tp": 1189, "fp": 14, "fn": 11,
                        "map50": 0.987, "map50_95": 0.742},
             "ocr": {"char_total": 8400, "char_correct": 8341,
                     "edit_distance": 71, "exact_match": 1154,
                     "sample_total": 1200},
             "by_env": {"DY": {"exact_match_rate": 0.981, "detect_recall": 0.997},
                        "NI": {"exact_match_rate": 0.942, "detect_recall": 0.988},
                        "RA": {"exact_match_rate": 0.918, "detect_recall": 0.974},
                        "BL": {"exact_match_rate": 0.905, "detect_recall": 0.969},
                        "FA": {"exact_match_rate": 0.871, "detect_recall": 0.942},
                        "TI": {"exact_match_rate": 0.933, "detect_recall": 0.981},
                        "OC": {"exact_match_rate": 0.804, "detect_recall": 0.915}},
             "cpu": {"latency_ms": lat(61.0), "threads": 4},
             "memory": {"rss_peak_mb": 214.0, "model_size_mb": 9.8}},
            {"name": "MobileNet-SSD + CRNN/CTC",
             "detector": "mobilenet", "recognizer": "crnn", "quantized": False,
             "detect": {"tp": 1168, "fp": 26, "fn": 32,
                        "map50": 0.964, "map50_95": 0.701},
             "ocr": {"char_total": 8176, "char_correct": 8071,
                     "edit_distance": 128, "exact_match": 1118,
                     "sample_total": 1200},
             "by_env": {"DY": {"exact_match_rate": 0.972, "detect_recall": 0.991},
                        "NI": {"exact_match_rate": 0.914, "detect_recall": 0.972},
                        "RA": {"exact_match_rate": 0.886, "detect_recall": 0.951},
                        "BL": {"exact_match_rate": 0.872, "detect_recall": 0.943},
                        "FA": {"exact_match_rate": 0.818, "detect_recall": 0.901},
                        "TI": {"exact_match_rate": 0.901, "detect_recall": 0.962},
                        "OC": {"exact_match_rate": 0.759, "detect_recall": 0.874}},
             "cpu": {"latency_ms": lat(44.0), "threads": 4},
             "memory": {"rss_peak_mb": 168.0, "model_size_mb": 6.2}},
            {"name": "YOLO-nano + PaddleOCR-mobile",
             "detector": "yolo", "recognizer": "paddle", "quantized": False,
             "detect": {"tp": 1189, "fp": 14, "fn": 11,
                        "map50": 0.987, "map50_95": 0.742},
             "ocr": {"char_total": 8386, "char_correct": 8280,
                     "edit_distance": 142, "exact_match": 1131,
                     "sample_total": 1200},
             "by_env": {"DY": {"exact_match_rate": 0.975, "detect_recall": 0.997},
                        "NI": {"exact_match_rate": 0.921, "detect_recall": 0.988},
                        "RA": {"exact_match_rate": 0.894, "detect_recall": 0.974},
                        "BL": {"exact_match_rate": 0.881, "detect_recall": 0.969},
                        "FA": {"exact_match_rate": 0.842, "detect_recall": 0.942},
                        "TI": {"exact_match_rate": 0.912, "detect_recall": 0.981},
                        "OC": {"exact_match_rate": 0.778, "detect_recall": 0.915}},
             "cpu": {"latency_ms": lat(118.0), "threads": 4},
             "memory": {"rss_peak_mb": 396.0, "model_size_mb": 18.4}},
            {"name": "Deformable DETR-R50 + CRNN/CTC",
             "detector": "deformable-detr", "recognizer": "crnn",
             "quantized": False,
             "detect": {"tp": 1193, "fp": 9, "fn": 7,
                        "map50": 0.991, "map50_95": 0.776},
             "ocr": {"char_total": 8428, "char_correct": 8377,
                     "edit_distance": 64, "exact_match": 1161,
                     "sample_total": 1200},
             "by_env": {"DY": {"exact_match_rate": 0.986, "detect_recall": 0.998},
                        "NI": {"exact_match_rate": 0.951, "detect_recall": 0.992},
                        "RA": {"exact_match_rate": 0.929, "detect_recall": 0.981},
                        "BL": {"exact_match_rate": 0.917, "detect_recall": 0.977},
                        "FA": {"exact_match_rate": 0.896, "detect_recall": 0.963},
                        "TI": {"exact_match_rate": 0.944, "detect_recall": 0.988},
                        "OC": {"exact_match_rate": 0.831, "detect_recall": 0.934}},
             "cpu": {"latency_ms": lat(742.0), "threads": 4},
             "memory": {"rss_peak_mb": 1480.0, "model_size_mb": 158.0}},
            {"name": "YOLO-nano(INT8) + CRNN/CTC(INT8)",
             "detector": "yolo-int8", "recognizer": "crnn-int8",
             "quantized": True,
             "detect": {"tp": 1182, "fp": 18, "fn": 18,
                        "map50": 0.979, "map50_95": 0.724},
             "ocr": {"char_total": 8330, "char_correct": 8258,
                     "edit_distance": 88, "exact_match": 1142,
                     "sample_total": 1200},
             "by_env": {"DY": {"exact_match_rate": 0.977, "detect_recall": 0.994},
                        "NI": {"exact_match_rate": 0.931, "detect_recall": 0.982},
                        "RA": {"exact_match_rate": 0.906, "detect_recall": 0.966},
                        "BL": {"exact_match_rate": 0.893, "detect_recall": 0.961},
                        "FA": {"exact_match_rate": 0.858, "detect_recall": 0.929},
                        "TI": {"exact_match_rate": 0.922, "detect_recall": 0.974},
                        "OC": {"exact_match_rate": 0.791, "detect_recall": 0.902}},
             "cpu": {"latency_ms": lat(34.0), "threads": 4},
             "memory": {"rss_peak_mb": 141.0, "model_size_mb": 3.1}},
        ],
    }


# ---------------------------------------------------------------- 진입점

def emit(payload: dict, md: bool) -> None:
    """입력 payload 로 일곱 표를 차례대로 출력한다."""
    models = [summarize(m) for m in payload.get("models", [])]
    if not models:
        print("[안내] models 배열이 비어 있다. --blank 로 빈 양식을 출력할 수 있다.")
        return
    ts = payload.get("test_set", {})
    print("=" * 72)
    print(f" {payload.get('project', '평가 리포트')} — AI 성능 평가표")
    print("=" * 72)
    print(f"측정 시각: {payload.get('measured_at', DASH)}   "
          f"테스트셋: 이미지 {ts.get('images', DASH)}장 / "
          f"세션 {ts.get('sessions', DASH)}개 / 차량 {ts.get('vehicles', DASH)}대")
    print(f"모델 {len(models)}종\n")

    for title, fn in (("1) 번호판 검출 평가표", table_detect),
                      ("2) OCR 평가표", table_ocr),
                      ("3) 환경별 평가표", table_env),
                      ("4) CPU 성능 평가표", table_cpu),
                      ("5) 메모리 사용량 평가표", table_memory),
                      ("6) 모델 비교표", table_compare),
                      ("7) 최종 모델 선정표", table_final)):
        print(f"[{title}]")
        print(fn(models, md))
        print()

    scores = total_scores(models)
    best = max(range(len(models)), key=lambda i: scores[i])
    print("가중치: 완전일치율 40 / 검출 Recall 20 / p95 지연 25 / "
          "메모리 10 / 모델 크기 5")
    print(f"종합 1위: {models[best]['name']} ({scores[best]:.1f}점) — "
          f"채택 근거를 부록 D (7) 성능 평가서에 그대로 옮겨 적는다.")


def main() -> int:
    """명령줄 인자를 해석해 평가표를 출력한다."""
    ap = argparse.ArgumentParser(
        description="평가 결과 JSON → 부록 C 평가표 자동 생성")
    ap.add_argument("json_file", nargs="?",
                    help="평가 결과 JSON 파일 경로")
    ap.add_argument("--demo", action="store_true",
                    help="내장 데모 데이터로 기입 예시를 출력")
    ap.add_argument("--blank", action="store_true",
                    help="빈 양식만 출력")
    ap.add_argument("--markdown", action="store_true",
                    help="마크다운 표로 출력(문서에 붙여 쓸 때)")
    args = ap.parse_args()

    if args.blank:
        print(blank_forms())
        return 0

    if args.demo or not args.json_file:
        if not args.demo:
            print("[안내] 입력 파일이 없어 내장 데모 데이터로 출력한다."
                  " 실제 측정값은 JSON 파일로 전달한다.\n")
        emit(demo_payload(), args.markdown)
        return 0

    path = pathlib.Path(args.json_file)
    if not path.exists():
        print(f"[안내] 파일이 없다: {path}")
        print("       --demo 로 기입 예시를, --blank 로 빈 양식을 볼 수 있다.")
        return 0
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"[오류] JSON 을 읽을 수 없다: {exc}")
        return 0
    if not isinstance(payload, dict):
        # /api/v1/models/benchmark 응답처럼 배열만 온 경우를 감싸 준다.
        payload = {"models": payload if isinstance(payload, list) else []}
    emit(payload, args.markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
