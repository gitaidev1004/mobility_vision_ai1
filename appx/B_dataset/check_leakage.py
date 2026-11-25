# -*- coding: utf-8 -*-
"""부록 B — 촬영 세션 기준 데이터 누수 검사기.

Train/Validation/Test 를 이미지 단위로 무작위 분할하면, 같은 차량을
같은 자리에서 연속 촬영한 프레임이 학습과 평가에 동시에 들어간다.
이때 측정된 성능은 실제보다 높게 나오며 이를 데이터 누수(Data Leakage)라 한다.

이 스크립트는 다음 네 가지 누수를 검사한다.
  1) 세션 누수   — 같은 촬영 세션(S###)이 두 개 이상의 분할에 걸쳐 있음
  2) 차량 누수   — 같은 차량번호가 두 개 이상의 분할에 걸쳐 있음
  3) 촬영일 누수 — 같은 촬영 날짜가 train 과 test 에 함께 있음(경고 수준)
  4) 중복 파일   — 바이트가 완전히 동일한 이미지가 다른 분할에 존재

차량번호는 라벨 사이드카 `labels/<split>/<stem>.gt.txt` 또는
`meta.csv`(stem,plate) 에서 읽는다. 없으면 차량 누수 검사는 건너뛴다.

사용법:
    python appx/B_dataset/check_leakage.py datasets/plate_det
    python appx/B_dataset/check_leakage.py datasets/plate_det --json
    python appx/B_dataset/check_leakage.py --make-sample datasets/leak_demo
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pathlib
import re
import sys

SPLITS = ("train", "val", "test")
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
NAME_RE = re.compile(r"^S(\d{3})_([A-Z]{2})_(\d{8})_(\d{4})$")


# ---------------------------------------------------------------- 수집

def load_meta(root: pathlib.Path) -> dict[str, str]:
    """meta.csv (stem,plate) 가 있으면 stem → 번호판 매핑을 읽는다."""
    meta_file = root / "meta.csv"
    mapping: dict[str, str] = {}
    if not meta_file.exists():
        return mapping
    try:
        with meta_file.open(encoding="utf-8", newline="") as fp:
            for row in csv.DictReader(fp):
                stem = (row.get("stem") or "").strip()
                plate = (row.get("plate") or "").strip()
                if stem and plate:
                    mapping[stem] = plate
    except (OSError, csv.Error):
        return {}
    return mapping


def plate_of(root: pathlib.Path, split: str, stem: str,
             meta: dict[str, str]) -> str | None:
    """사이드카 또는 meta.csv 에서 해당 이미지의 정답 번호판을 찾는다."""
    if stem in meta:
        return meta[stem]
    side = root / "labels" / split / f"{stem}.gt.txt"
    if side.exists():
        try:
            text = side.read_text(encoding="utf-8").strip()
        except OSError:
            return None
        return text.splitlines()[0].strip() if text else None
    return None


def file_digest(path: pathlib.Path) -> str:
    """파일 내용의 SHA-256 요약을 구한다(중복 파일 탐지용)."""
    h = hashlib.sha256()
    try:
        with path.open("rb") as fp:
            for chunk in iter(lambda: fp.read(65536), b""):
                h.update(chunk)
    except OSError:
        return ""
    return h.hexdigest()


def collect(root: pathlib.Path) -> list[dict]:
    """분할별 이미지의 세션·환경·날짜·번호판·해시를 모은다."""
    meta = load_meta(root)
    items: list[dict] = []
    for split in SPLITS:
        img_dir = root / "images" / split
        if not img_dir.is_dir():
            continue
        for img in sorted(img_dir.iterdir()):
            if img.suffix.lower() not in IMG_EXT:
                continue
            stem = img.stem
            m = NAME_RE.match(stem)
            items.append({
                "split": split,
                "file": img.name,
                "stem": stem,
                "session": ("S" + m.group(1)) if m else "UNKNOWN",
                "env": m.group(2) if m else "??",
                "date": m.group(3) if m else "",
                "plate": plate_of(root, split, stem, meta),
                "sha256": file_digest(img),
            })
    return items


# ---------------------------------------------------------------- 검사

def group_by(items: list[dict], key: str) -> dict[str, dict[str, list[str]]]:
    """key 값 → {분할: [파일명...]} 형태로 묶는다."""
    out: dict[str, dict[str, list[str]]] = {}
    for it in items:
        val = it.get(key)
        if not val:
            continue
        out.setdefault(str(val), {}).setdefault(it["split"], []).append(it["file"])
    return out


def find_cross_split(groups: dict[str, dict[str, list[str]]]) -> list[dict]:
    """두 개 이상의 분할에 걸쳐 있는 그룹만 뽑아낸다."""
    found = []
    for val, per_split in sorted(groups.items()):
        if len(per_split) < 2:
            continue
        found.append({
            "value": val,
            "splits": {s: len(v) for s, v in sorted(per_split.items())},
            "examples": [v[0] for _s, v in sorted(per_split.items())],
        })
    return found


def check_all(items: list[dict]) -> dict:
    """네 가지 누수를 모두 검사해 결과 딕셔너리를 돌려준다."""
    session_leaks = find_cross_split(group_by(items, "session"))
    plate_leaks = find_cross_split(group_by(items, "plate"))

    # 촬영일은 train 과 test 가 겹칠 때만 경고한다(val 은 허용 가능).
    date_groups = group_by(items, "date")
    date_leaks = [g for g in find_cross_split(date_groups)
                  if "train" in g["splits"] and "test" in g["splits"]]

    dup_groups = group_by(items, "sha256")
    dup_leaks = find_cross_split(dup_groups)
    # 같은 분할 안의 완전 중복도 품질 문제이므로 따로 센다.
    dup_within = 0
    for _val, per_split in dup_groups.items():
        for _s, files in per_split.items():
            if len(files) > 1:
                dup_within += len(files) - 1

    unknown = [it["file"] for it in items if it["session"] == "UNKNOWN"]
    no_plate = sum(1 for it in items if not it["plate"])

    return {
        "total_images": len(items),
        "by_split": {s: sum(1 for it in items if it["split"] == s)
                     for s in SPLITS},
        "session_count": len({it["session"] for it in items}),
        "session_leaks": session_leaks,
        "plate_leaks": plate_leaks,
        "plate_known": len(items) - no_plate,
        "date_leaks": date_leaks,
        "duplicate_cross_split": dup_leaks,
        "duplicate_within_split": dup_within,
        "unnamed_files": unknown,
        "verdict": "FAIL" if (session_leaks or plate_leaks or dup_leaks)
                   else ("WARN" if (date_leaks or unknown or dup_within)
                         else "PASS"),
    }


# ---------------------------------------------------------------- 출력

def width(text: str) -> int:
    """한글을 폭 2로 계산한 표시 폭."""
    return sum(2 if ord(ch) > 0x1100 else 1 for ch in str(text))


def table(header: list[str], rows: list[list[str]]) -> str:
    """고정폭 표를 문자열로 만든다."""
    data = [header, *rows] if rows else [header]
    widths = [max(width(r[c]) for r in data) for c in range(len(header))]

    def pad(text: str, w: int) -> str:
        return str(text) + " " * max(0, w - width(text))

    sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    out = [sep, "| " + " | ".join(pad(header[c], widths[c])
                                  for c in range(len(header))) + " |", sep]
    for r in rows:
        out.append("| " + " | ".join(pad(r[c], widths[c])
                                     for c in range(len(header))) + " |")
    out.append(sep)
    return "\n".join(out)


def splits_text(splits: dict[str, int]) -> str:
    """{'train': 3, 'test': 2} → 'train 3 / test 2'."""
    return " / ".join(f"{s} {n}" for s, n in splits.items())


def print_result(res: dict) -> None:
    """검사 결과를 표로 출력한다."""
    print("=" * 72)
    print(" 촬영 세션 기준 데이터 누수 검사")
    print("=" * 72)
    print(f"이미지 {res['total_images']}장 "
          f"({splits_text(res['by_split'])}) / "
          f"촬영 세션 {res['session_count']}개 / "
          f"번호판 정답 보유 {res['plate_known']}장\n")

    print("[1] 검사 요약")
    rows = [
        ["세션 누수", str(len(res["session_leaks"])),
         "0", "FAIL" if res["session_leaks"] else "PASS"],
        ["차량번호 누수", str(len(res["plate_leaks"])),
         "0", "FAIL" if res["plate_leaks"] else "PASS"],
        ["촬영일 누수(train↔test)", str(len(res["date_leaks"])),
         "0 권장", "WARN" if res["date_leaks"] else "PASS"],
        ["분할 간 중복 파일", str(len(res["duplicate_cross_split"])),
         "0", "FAIL" if res["duplicate_cross_split"] else "PASS"],
        ["분할 내 중복 파일", str(res["duplicate_within_split"]),
         "0", "WARN" if res["duplicate_within_split"] else "PASS"],
        ["파일명 규칙 위반", str(len(res["unnamed_files"])),
         "0", "WARN" if res["unnamed_files"] else "PASS"],
    ]
    print(table(["검사 항목", "검출", "기준", "판정"], rows))

    if res["session_leaks"]:
        print("\n[2] 세션 누수 상세 — 같은 촬영 세션이 두 분할에 걸쳐 있다")
        rows = [[g["value"], splits_text(g["splits"]),
                 ", ".join(g["examples"][:2])]
                for g in res["session_leaks"][:20]]
        print(table(["세션", "분할별 장수", "예시 파일"], rows))

    if res["plate_leaks"]:
        print("\n[3] 차량번호 누수 상세 — 같은 차량이 두 분할에 걸쳐 있다")
        rows = [[g["value"], splits_text(g["splits"]),
                 ", ".join(g["examples"][:2])]
                for g in res["plate_leaks"][:20]]
        print(table(["차량번호", "분할별 장수", "예시 파일"], rows))

    if res["duplicate_cross_split"]:
        print("\n[4] 분할 간 중복 파일 — 내용이 완전히 같은 이미지")
        rows = [[g["value"][:16] + "...", splits_text(g["splits"]),
                 ", ".join(g["examples"][:2])]
                for g in res["duplicate_cross_split"][:20]]
        print(table(["SHA-256", "분할별 장수", "예시 파일"], rows))

    if res["date_leaks"]:
        print("\n[5] 촬영일 누수(경고) — 같은 날짜가 train 과 test 에 함께 있다")
        rows = [[g["value"], splits_text(g["splits"])]
                for g in res["date_leaks"][:20]]
        print(table(["촬영일", "분할별 장수"], rows))

    print(f"\n최종 판정: {res['verdict']}")
    if res["verdict"] == "FAIL":
        print("세션 또는 차량 단위로 분할을 다시 만들어야 한다. "
              "부록 B (4) 의 세션 기준 분할 절차를 따른다.")
    elif res["verdict"] == "WARN":
        print("치명적 누수는 없으나 경고 항목을 확인한다.")
    else:
        print("세션·차량·중복 기준 누수가 없다. 평가 결과를 신뢰할 수 있다.")


# ---------------------------------------------------------------- 샘플 생성

def make_sample(root: pathlib.Path) -> None:
    """누수가 실제로 검출되는지 보여 주는 샘플 데이터셋을 만든다.

    S001~S004 는 정상 분할, S005 는 train/test 에 걸친 누수 사례로 둔다.
    같은 차량번호 `12가3456` 도 train 과 test 양쪽에 넣어 차량 누수를 만든다.
    """
    plan = {
        "train": [("S001", "DY", "20260401", "12가3456"),
                  ("S001", "DY", "20260401", "12가3456"),
                  ("S002", "NI", "20260402", "서울 34나5678"),
                  ("S005", "RA", "20260405", "87로2345")],
        "val":   [("S003", "BL", "20260403", "11가1111")],
        "test":  [("S004", "FA", "20260404", "22나2222"),
                  ("S005", "RA", "20260405", "87로2345"),
                  ("S006", "TI", "20260406", "12가3456")],
    }
    rows = [("stem", "plate")]
    for split, entries in plan.items():
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
        for i, (sess, env, date, plate) in enumerate(entries):
            stem = f"{sess}_{env}_{date}_{i:04d}"
            # 분할 간 중복 파일 사례를 만들기 위해 내용을 세션별로 고정한다.
            (root / "images" / split / f"{stem}.jpg").write_bytes(
                b"\xff\xd8" + sess.encode() + b"\xff\xd9")
            (root / "labels" / split / f"{stem}.txt").write_text(
                "0 0.500000 0.600000 0.120000 0.040000\n", encoding="utf-8")
            rows.append((stem, plate))
    with (root / "meta.csv").open("w", encoding="utf-8", newline="") as fp:
        csv.writer(fp).writerows(rows)


# ---------------------------------------------------------------- 진입점

def main() -> int:
    """명령줄 인자를 해석해 누수 검사를 수행한다."""
    ap = argparse.ArgumentParser(description="촬영 세션 기준 데이터 누수 검사")
    ap.add_argument("root", nargs="?", default="datasets/plate_det",
                    help="데이터셋 루트 (images/, labels/ 를 포함)")
    ap.add_argument("--json", action="store_true", help="JSON 으로 출력")
    ap.add_argument("--make-sample", metavar="DIR",
                    help="누수 사례가 포함된 샘플을 만든 뒤 검사 결과를 출력")
    args = ap.parse_args()

    root = pathlib.Path(args.make_sample or args.root)
    if args.make_sample:
        make_sample(root)
        print(f"샘플 데이터셋을 생성했다: {root}\n")

    if not root.is_dir():
        print(f"[안내] 데이터셋 디렉터리가 없다: {root}")
        print("       --make-sample <경로> 로 샘플을 만들어 동작을 확인할 수 있다.")
        return 0

    items = collect(root)
    if not items:
        print(f"[안내] {root} 아래 images/train|val|test 에 이미지가 없다.")
        return 0

    res = check_all(items)
    if args.json:
        json.dump(res, sys.stdout, ensure_ascii=False, indent=2)
        print()
    else:
        print_result(res)
    # 누수가 있어도 스크립트 자체는 정상 종료한다(판정은 출력으로 전달).
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
