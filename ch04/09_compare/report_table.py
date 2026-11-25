# -*- coding: utf-8 -*-
"""bench_models.py 가 만든 JSON 을 마크다운/CSV 표로 바꾼다.

품질 지표(검출 재현율, 완전 일치율)는 벤치마크 러너가 알 수 없으므로
별도 JSON(--quality)으로 주입한다. 없으면 빈칸으로 둔다.

사용 예
  python report_table.py --input bench_result.json --format md
  python report_table.py --input bench_result.json --format csv --out table.csv
  python report_table.py --demo --format md
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys

COLUMNS = [
    ("name", "모델", "{}"),
    ("backbone", "백본", "{}"),
    ("params_m", "파라미터(M)", "{:.1f}"),
    ("size_mb", "모델 크기(MB)", "{:.1f}"),
    ("p50_ms", "CPU p50(ms)", "{:.1f}"),
    ("p95_ms", "CPU p95(ms)", "{:.1f}"),
    ("peak_mem_mb", "메모리(MB)", "{:.0f}"),
    ("detect_recall", "번호판 재현율", "{:.3f}"),
    ("exact_match_rate", "완전 일치율", "{:.3f}"),
    ("verdict", "판단", "{}"),
]

# 샘플(demo) 데이터 — 예시 측정 환경의 '한 번의 관측값'이다. 단정하지 않는다.
DEMO = {
    "env": {"platform": "Windows", "machine": "AMD64", "python": "3.12.4",
            "threads": 1, "input_size": 640, "warmup": 5, "runs": 30,
            "cpu": "8코어 CPU (예시)"},
    "results": [
        dict(name="경량 YOLO-nano", backbone="CSP-nano", params_m=2.6, size_mb=5.3,
             p50_ms=38.4, p95_ms=54.1, peak_mem_mb=210, detect_recall=0.988,
             exact_match_rate=0.958, verdict="1순위 채택"),
        dict(name="경량 YOLO-small", backbone="CSP-small", params_m=9.4, size_mb=18.9,
             p50_ms=74.6, p95_ms=101.3, peak_mem_mb=268, detect_recall=0.993,
             exact_match_rate=0.964, verdict="정확도 여유 후보"),
        dict(name="MobileNetV3+SSDLite", backbone="MobileNetV3-L", params_m=3.4,
             size_mb=13.9, p50_ms=29.8, p95_ms=41.7, peak_mem_mb=186,
             detect_recall=0.961, exact_match_rate=0.931, verdict="저사양 대안"),
        dict(name="DETR", backbone="ResNet-50", params_m=41.3, size_mb=166.0,
             p50_ms=612.4, p95_ms=781.9, peak_mem_mb=742, detect_recall=0.972,
             exact_match_rate=0.941, verdict="기준선 참고"),
        dict(name="Deformable DETR", backbone="ResNet-50", params_m=40.1, size_mb=161.0,
             p50_ms=948.7, p95_ms=1224.5, peak_mem_mb=864, detect_recall=0.986,
             exact_match_rate=0.953, verdict="연구 대조군"),
        dict(name="DINO", backbone="ResNet-50", params_m=47.0, size_mb=189.0,
             p50_ms=1287.3, p95_ms=1662.8, peak_mem_mb=1018, detect_recall=0.991,
             exact_match_rate=0.959, verdict="정확도 상한 확인"),
        dict(name="YOLOS-small", backbone="ViT-Small", params_m=30.7, size_mb=123.0,
             p50_ms=1104.2, p95_ms=1398.6, peak_mem_mb=906, detect_recall=0.948,
             exact_match_rate=0.918, verdict="CPU 비추천"),
        dict(name="Swin-T + DINO", backbone="Swin-Tiny", params_m=48.0, size_mb=193.0,
             p50_ms=1742.9, p95_ms=2210.4, peak_mem_mb=1264, detect_recall=0.992,
             exact_match_rate=0.961, verdict="비용 과다"),
        dict(name="ViTDet(ViT-Base)", backbone="ViT-Base", params_m=111.0, size_mb=445.0,
             p50_ms=3418.6, p95_ms=4102.7, peak_mem_mb=2186, detect_recall=0.989,
             exact_match_rate=0.955, verdict="범용 검출용"),
    ],
}


def load(path: str | None, demo: bool) -> dict:
    if demo or not path:
        return DEMO
    p = pathlib.Path(path)
    if not p.exists():
        print(f"[알림] {path} 가 없다. 샘플 데이터로 표를 출력한다.", file=sys.stderr)
        return DEMO
    return json.loads(p.read_text(encoding="utf-8"))


def merge_quality(data: dict, qpath: str | None) -> dict:
    """품질 지표 JSON( {모델key 또는 name: {detect_recall, exact_match_rate, verdict}} )을 합친다."""
    if not qpath:
        return data
    p = pathlib.Path(qpath)
    if not p.exists():
        print(f"[알림] 품질 지표 파일 {qpath} 가 없다. 빈칸으로 둔다.", file=sys.stderr)
        return data
    q = json.loads(p.read_text(encoding="utf-8"))
    for r in data["results"]:
        for k in (r.get("key"), r.get("name")):
            if k and k in q:
                r.update(q[k])
                break
    return data


def fmt(row: dict, key: str, spec: str) -> str:
    v = row.get(key)
    if v is None:
        return "-"
    try:
        return spec.format(v)
    except (ValueError, TypeError):
        return str(v)


def env_line(env: dict) -> str:
    return (f"측정 환경 예시: {env.get('cpu', env.get('platform', '-'))}, "
            f"입력 {env.get('input_size', '-')}, "
            f"스레드 {env.get('threads', '-')}, "
            f"워밍업 {env.get('warmup', '-')}회 / 측정 {env.get('runs', '-')}회")


def to_markdown(data: dict) -> str:
    rows = data["results"]
    out = ["### 검출기 종합 비교표", "", f"> {env_line(data.get('env', {}))}",
           "> 아래 값은 한 번의 예시 측정 결과이며, 하드웨어와 데이터셋이 바뀌면 달라진다.", ""]
    out.append("| " + " | ".join(c[1] for c in COLUMNS) + " |")
    out.append("|" + "|".join("---" for _ in COLUMNS) + "|")
    for r in rows:
        out.append("| " + " | ".join(fmt(r, k, s) for k, _, s in COLUMNS) + " |")
    out.append("")

    # 축별 요약 표
    for key, label, spec, better in (
        ("exact_match_rate", "완전 일치율", "{:.3f}", "high"),
        ("p50_ms", "CPU p50 지연(ms)", "{:.1f}", "low"),
        ("peak_mem_mb", "메모리(MB)", "{:.0f}", "low"),
        ("size_mb", "모델 크기(MB)", "{:.1f}", "low"),
    ):
        valid = [r for r in rows if r.get(key) is not None]
        if not valid:
            continue
        valid = sorted(valid, key=lambda r: r[key], reverse=(better == "high"))
        best = valid[0][key]
        out.append(f"### {label} 비교 ({'높을수록' if better == 'high' else '낮을수록'} 좋음)")
        out.append("")
        out.append(f"| 순위 | 모델 | {label} | 최우수 대비 |")
        out.append("|---|---|---|---|")
        for i, r in enumerate(valid, 1):
            if better == "high":
                rel = f"{r[key] / best * 100:.1f}%"
            else:
                rel = f"{r[key] / best:.1f}배"
            out.append(f"| {i} | {r['name']} | {spec.format(r[key])} | {rel} |")
        out.append("")
    return "\n".join(out)


def to_csv(data: dict, out_path: str | None) -> str:
    rows = data["results"]
    target = open(out_path, "w", encoding="utf-8-sig", newline="") if out_path else sys.stdout
    try:
        w = csv.writer(target)
        w.writerow([c[1] for c in COLUMNS])
        for r in rows:
            w.writerow([fmt(r, k, s) for k, _, s in COLUMNS])
    finally:
        if out_path:
            target.close()
    return out_path or "(표준출력)"


def main() -> None:
    ap = argparse.ArgumentParser(description="벤치마크 결과 표 생성기")
    ap.add_argument("--input", default="bench_result.json")
    ap.add_argument("--quality", default="", help="품질 지표 JSON 경로")
    ap.add_argument("--format", choices=("md", "csv"), default="md")
    ap.add_argument("--out", default="", help="출력 파일 경로(없으면 화면)")
    ap.add_argument("--demo", action="store_true", help="샘플 데이터로 표 출력")
    args = ap.parse_args()

    data = merge_quality(load(args.input, args.demo), args.quality or None)

    if args.format == "csv":
        where = to_csv(data, args.out or None)
        if args.out:
            print(f"CSV 를 {where} 에 저장했다. (엑셀에서 열 수 있게 BOM 포함)")
        return

    md = to_markdown(data)
    if args.out:
        pathlib.Path(args.out).write_text(md, encoding="utf-8")
        print(f"마크다운을 {args.out} 에 저장했다.")
    else:
        print(md)


if __name__ == "__main__":
    main()
