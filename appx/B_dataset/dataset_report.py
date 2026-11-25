# -*- coding: utf-8 -*-
"""부록 B — 번호판 데이터셋 통계·품질 리포트 생성기.

YOLO 형식 데이터셋 디렉터리를 훑어
  1) 분할(train/val/test)별 이미지·라벨 수
  2) 촬영 환경 코드별 분포와 목표 대비 달성률
  3) 번호판 박스의 픽셀 폭·높이·가로세로비 분포
  4) 품질 결함 목록(라벨 누락, 좌표 범위 초과, 과소 해상도 등)
을 표로 출력한다. 외부 패키지 없이 표준 라이브러리만 사용한다.

파일명 규칙(부록 B (2) 수집 기준과 동일):
    S<세션3자리>_<환경코드>_<YYYYMMDD>_<연번4자리>.jpg
    예) S003_NI_20260412_0147.jpg   → 세션 S003, 야간(NI)

환경 코드: DY(주간) NI(야간) RA(우천) BL(역광) FA(원거리) TI(기울어짐) OC(가림)

사용법:
    python appx/B_dataset/dataset_report.py datasets/plate_det
    python appx/B_dataset/dataset_report.py datasets/plate_det --json
    python appx/B_dataset/dataset_report.py --make-sample datasets/sample_det
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import re
import statistics
import sys

SPLITS = ("train", "val", "test")
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}

# 환경 코드 → (표시 이름, 전체 목표 장수)
ENV_TARGET: dict[str, tuple[str, int]] = {
    "DY": ("주간 정상", 3000),
    "NI": ("야간",      1500),
    "RA": ("우천·강설",  800),
    "BL": ("역광",       600),
    "FA": ("원거리",     700),
    "TI": ("기울어짐",   500),
    "OC": ("부분 가림",  400),
}

# 번호판 최소 픽셀 폭 기준(부록 B (1) 촬영 가이드와 동일)
MIN_PLATE_PX = 96
NAME_RE = re.compile(r"^S(\d{3})_([A-Z]{2})_(\d{8})_(\d{4})$")


# ---------------------------------------------------------------- 자료 구조

class Record:
    """이미지 1장과 그 라벨에서 뽑아낸 정보."""

    def __init__(self, split: str, path: pathlib.Path) -> None:
        self.split = split
        self.path = path
        self.stem = path.stem
        self.session = "UNKNOWN"
        self.env = "??"
        self.date = ""
        self.boxes: list[tuple[float, float, float, float]] = []
        self.defects: list[str] = []

        m = NAME_RE.match(self.stem)
        if m:
            self.session = "S" + m.group(1)
            self.env = m.group(2)
            self.date = m.group(3)
            if self.env not in ENV_TARGET:
                self.defects.append(f"환경 코드 미정의({self.env})")
        else:
            self.defects.append("파일명 규칙 불일치")


# ---------------------------------------------------------------- 수집

def label_path(img: pathlib.Path) -> pathlib.Path:
    """images/<split>/x.jpg → labels/<split>/x.txt 로 바꾼다."""
    parts = list(img.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    return pathlib.Path(*parts).with_suffix(".txt")


def parse_label(txt: pathlib.Path, rec: Record, img_w: int, img_h: int) -> None:
    """YOLO 라벨 파일을 읽어 박스를 수집하고 결함을 기록한다."""
    if not txt.exists():
        rec.defects.append("라벨 파일 없음")
        return
    try:
        lines = [ln.strip() for ln in
                 txt.read_text(encoding="utf-8").splitlines() if ln.strip()]
    except OSError as exc:
        rec.defects.append(f"라벨 읽기 실패({exc.__class__.__name__})")
        return
    if not lines:
        rec.defects.append("라벨 내용 비어 있음")
        return
    for ln_no, line in enumerate(lines, 1):
        cols = line.split()
        if len(cols) != 5:
            rec.defects.append(f"{ln_no}행 열 개수 {len(cols)}≠5")
            continue
        try:
            cls = int(float(cols[0]))
            cx, cy, w, h = (float(v) for v in cols[1:])
        except ValueError:
            rec.defects.append(f"{ln_no}행 숫자 변환 실패")
            continue
        if cls != 0:
            rec.defects.append(f"{ln_no}행 클래스 {cls}(0 이어야 함)")
        if not all(0.0 <= v <= 1.0 for v in (cx, cy, w, h)):
            rec.defects.append(f"{ln_no}행 정규화 좌표 범위 초과")
            continue
        if w <= 0 or h <= 0:
            rec.defects.append(f"{ln_no}행 폭/높이 0")
            continue
        px_w = w * img_w
        px_h = h * img_h
        if px_w < MIN_PLATE_PX:
            rec.defects.append(f"{ln_no}행 번호판 폭 {px_w:.0f}px < {MIN_PLATE_PX}px")
        rec.boxes.append((cx, cy, px_w, px_h))


def read_image_size(path: pathlib.Path) -> tuple[int, int]:
    """이미지 헤더만 읽어 (폭, 높이)를 구한다. 실패하면 (1920, 1080) 가정."""
    try:
        data = path.read_bytes()
    except OSError:
        return 1920, 1080
    # PNG: IHDR 청크의 폭/높이
    if data[:8] == b"\x89PNG\r\n\x1a\n" and len(data) >= 24:
        w = int.from_bytes(data[16:20], "big")
        h = int.from_bytes(data[20:24], "big")
        if w and h:
            return w, h
    # JPEG: SOF 마커에서 폭/높이
    if data[:2] == b"\xff\xd8":
        i = 2
        while i + 9 < len(data):
            if data[i] != 0xFF:
                i += 1
                continue
            marker = data[i + 1]
            if marker in (0xC0, 0xC1, 0xC2, 0xC3):
                h = int.from_bytes(data[i + 5:i + 7], "big")
                w = int.from_bytes(data[i + 7:i + 9], "big")
                if w and h:
                    return w, h
                break
            if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
                i += 2
                continue
            seg = int.from_bytes(data[i + 2:i + 4], "big")
            if seg <= 0:
                break
            i += 2 + seg
    # 측정 불가 시 교재 권장 촬영 해상도를 가정한다.
    return 1920, 1080


def collect(root: pathlib.Path) -> list[Record]:
    """데이터셋 루트에서 분할별 레코드를 모두 수집한다."""
    records: list[Record] = []
    for split in SPLITS:
        img_dir = root / "images" / split
        if not img_dir.is_dir():
            continue
        for img in sorted(img_dir.iterdir()):
            if img.suffix.lower() not in IMG_EXT:
                continue
            rec = Record(split, img)
            w, h = read_image_size(img)
            parse_label(label_path(img), rec, w, h)
            records.append(rec)
    return records


# ---------------------------------------------------------------- 표 출력

def width(text: str) -> int:
    """한글을 폭 2로 계산한 표시 폭."""
    return sum(2 if ord(ch) > 0x1100 else 1 for ch in text)


def table(header: list[str], rows: list[list[str]]) -> str:
    """고정폭 표를 문자열로 만든다."""
    data = [header, *rows] if rows else [header]
    cols = len(header)
    widths = [max(width(str(r[c])) for r in data) for c in range(cols)]

    def pad(text: str, w: int) -> str:
        return str(text) + " " * max(0, w - width(str(text)))

    sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    out = [sep, "| " + " | ".join(pad(header[c], widths[c])
                                  for c in range(cols)) + " |", sep]
    for r in rows:
        out.append("| " + " | ".join(pad(r[c], widths[c])
                                     for c in range(cols)) + " |")
    out.append(sep)
    return "\n".join(out)


def pct(part: int, total: int) -> str:
    """비율을 백분율 문자열로 만든다."""
    return f"{(part / total * 100):.1f}%" if total else "-"


# ---------------------------------------------------------------- 리포트

def build_report(records: list[Record]) -> dict:
    """레코드 목록에서 리포트용 집계 딕셔너리를 만든다."""
    total = len(records)
    by_split = {s: [r for r in records if r.split == s] for s in SPLITS}
    by_env: dict[str, list[Record]] = {}
    for r in records:
        by_env.setdefault(r.env, []).append(r)
    sessions = {r.session for r in records}
    widths = [b[2] for r in records for b in r.boxes]
    heights = [b[3] for r in records for b in r.boxes]
    ratios = [b[2] / b[3] for r in records for b in r.boxes if b[3] > 0]
    defect_rows = [r for r in records if r.defects]
    return {
        "total_images": total,
        "total_boxes": sum(len(r.boxes) for r in records),
        "sessions": sorted(sessions),
        "by_split": {s: len(v) for s, v in by_split.items()},
        "by_env": {e: len(v) for e, v in sorted(by_env.items())},
        "plate_px_width": _stat(widths),
        "plate_px_height": _stat(heights),
        "aspect_ratio": _stat(ratios),
        "defect_image_count": len(defect_rows),
        "defects": [{"file": r.path.name, "split": r.split,
                     "issues": r.defects} for r in defect_rows],
    }


def _stat(values: list[float]) -> dict:
    """최소·중앙·평균·최대를 묶어 돌려준다."""
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "min": round(min(values), 1),
        "median": round(statistics.median(values), 1),
        "mean": round(statistics.fmean(values), 1),
        "max": round(max(values), 1),
    }


def print_report(rep: dict) -> None:
    """집계 결과를 표 형태로 출력한다."""
    print("=" * 72)
    print(" 번호판 데이터셋 통계·품질 리포트")
    print("=" * 72)
    print(f"이미지 {rep['total_images']}장 / 박스 {rep['total_boxes']}개 / "
          f"촬영 세션 {len(rep['sessions'])}개\n")

    print("[1] 분할별 구성")
    total = rep["total_images"]
    rows = [[s, str(rep["by_split"].get(s, 0)),
             pct(rep["by_split"].get(s, 0), total),
             {"train": "70%", "val": "15%", "test": "15%"}[s]]
            for s in SPLITS]
    print(table(["분할", "장수", "비율", "목표 비율"], rows))

    print("\n[2] 촬영 환경별 분포와 목표 달성률")
    rows = []
    for code, (label, target) in ENV_TARGET.items():
        got = rep["by_env"].get(code, 0)
        rate = f"{got / target * 100:.1f}%" if target else "-"
        verdict = "충족" if got >= target else "부족"
        rows.append([code, label, str(got), str(target), rate, verdict])
    for code, got in rep["by_env"].items():
        if code not in ENV_TARGET:
            rows.append([code, "(미정의)", str(got), "-", "-", "확인"])
    print(table(["코드", "환경", "보유", "목표", "달성률", "판정"], rows))

    print("\n[3] 번호판 박스 크기 분포 (단위 px)")
    rows = []
    for key, label in (("plate_px_width", "번호판 폭"),
                       ("plate_px_height", "번호판 높이"),
                       ("aspect_ratio", "가로세로비")):
        s = rep[key]
        if s["n"] == 0:
            rows.append([label, "0", "-", "-", "-", "-"])
        else:
            rows.append([label, str(s["n"]), str(s["min"]), str(s["median"]),
                         str(s["mean"]), str(s["max"])])
    print(table(["측정값", "개수", "최소", "중앙값", "평균", "최대"], rows))
    pw = rep["plate_px_width"]
    if pw["n"]:
        print(f"    기준: 번호판 폭 {MIN_PLATE_PX}px 이상. "
              f"최소값 {pw['min']}px "
              f"{'→ 기준 충족' if pw['min'] >= MIN_PLATE_PX else '→ 기준 미달 이미지 존재'}")

    print("\n[4] 품질 결함 목록")
    if not rep["defects"]:
        print("    결함 없음. 모든 이미지가 라벨·좌표·해상도 기준을 통과했다.")
    else:
        rows = [[d["split"], d["file"], "; ".join(d["issues"])[:52]]
                for d in rep["defects"][:25]]
        print(table(["분할", "파일", "결함 내용"], rows))
        print(f"    결함 이미지 {rep['defect_image_count']}장 / "
              f"전체 {rep['total_images']}장 "
              f"({pct(rep['defect_image_count'], rep['total_images'])})")


# ---------------------------------------------------------------- 샘플 생성

def make_sample(root: pathlib.Path, seed: int = 20261024) -> None:
    """리포트 동작 확인용 소형 샘플 데이터셋을 만든다.

    1x1 PNG 를 쓰지 않고 헤더에 1920x1080 을 기록한 최소 PNG 를 생성해
    픽셀 폭 계산이 실제로 동작하도록 한다.
    """
    rng = random.Random(seed)
    plan = [("train", 0, 14), ("val", 1, 3), ("test", 2, 3)]
    codes = list(ENV_TARGET)
    session_no = 1
    for split, _idx, count in plan:
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
        for i in range(count):
            # 세션은 5장 단위로 끊어 분할 간에 겹치지 않게 배정한다.
            if i % 5 == 0:
                session_no += 1
            env = codes[i % len(codes)]
            stem = f"S{session_no:03d}_{env}_2026041{i % 9}_{i:04d}"
            img = root / "images" / split / f"{stem}.jpg"
            img.write_bytes(_fake_jpeg(1920, 1080))
            # 번호판 폭은 100~340px 범위, 가로세로비는 2.0~5.2 사이로 둔다.
            px_w = rng.uniform(100, 340)
            px_h = px_w / rng.uniform(2.0, 5.2)
            line = (f"0 {rng.uniform(0.3, 0.7):.6f} {rng.uniform(0.4, 0.8):.6f} "
                    f"{px_w / 1920:.6f} {px_h / 1080:.6f}\n")
            (root / "labels" / split / f"{stem}.txt").write_text(
                line, encoding="utf-8")

    # 리포트가 결함을 실제로 찾아내는지 보이기 위해 의도적 결함 3건을 심는다.
    bad1 = root / "images" / "train" / "S002_NI_20260412_9001.jpg"
    bad1.write_bytes(_fake_jpeg(1920, 1080))          # 라벨 파일 없음
    bad2 = root / "images" / "train" / "S002_NI_20260412_9002.jpg"
    bad2.write_bytes(_fake_jpeg(1920, 1080))
    (root / "labels" / "train" / "S002_NI_20260412_9002.txt").write_text(
        "0 0.5 0.5 0.030000 0.012000\n", encoding="utf-8")  # 폭 57px (기준 미달)
    bad3 = root / "images" / "train" / "noname_0003.jpg"      # 파일명 규칙 위반
    bad3.write_bytes(_fake_jpeg(1920, 1080))
    (root / "labels" / "train" / "noname_0003.txt").write_text(
        "0 0.5 0.5 0.100000 0.030000\n", encoding="utf-8")


def _fake_jpeg(w: int, h: int) -> bytes:
    """폭·높이만 올바른 최소 JPEG 바이트열을 만든다(샘플 생성 전용)."""
    sof = (b"\xff\xc0\x00\x11\x08"
           + h.to_bytes(2, "big") + w.to_bytes(2, "big")
           + b"\x03\x01\x11\x00\x02\x11\x01\x03\x11\x01")
    return b"\xff\xd8" + sof + b"\xff\xd9"


# ---------------------------------------------------------------- 진입점

def main() -> int:
    """명령줄 인자를 해석해 리포트를 생성한다."""
    ap = argparse.ArgumentParser(description="번호판 데이터셋 통계·품질 리포트")
    ap.add_argument("root", nargs="?", default="datasets/plate_det",
                    help="데이터셋 루트 (images/, labels/ 를 포함)")
    ap.add_argument("--json", action="store_true", help="JSON 으로 출력")
    ap.add_argument("--make-sample", metavar="DIR",
                    help="동작 확인용 샘플 데이터셋을 만든 뒤 리포트를 출력")
    args = ap.parse_args()

    root = pathlib.Path(args.make_sample or args.root)
    if args.make_sample:
        make_sample(root)
        print(f"샘플 데이터셋을 생성했다: {root}\n")

    if not root.is_dir():
        print(f"[안내] 데이터셋 디렉터리가 없다: {root}")
        print("       --make-sample <경로> 로 샘플을 만들어 동작을 확인할 수 있다.")
        return 0

    records = collect(root)
    if not records:
        print(f"[안내] {root} 아래 images/train|val|test 에 이미지가 없다.")
        print("       부록 B (4) 의 디렉터리 구조를 먼저 만든다.")
        return 0

    rep = build_report(records)
    if args.json:
        json.dump(rep, sys.stdout, ensure_ascii=False, indent=2)
        print()
    else:
        print_report(rep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
