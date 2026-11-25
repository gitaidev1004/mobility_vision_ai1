# -*- coding: utf-8 -*-
"""PaddleOCR rec(인식) 학습용 데이터셋 변환.

PaddleOCR 의 인식 학습은 아주 단순한 형식을 쓴다.

    <이미지 상대경로>\t<정답 문자열>

줄마다 탭 하나로 구분한 텍스트 파일(라벨 txt) 하나와, 그 경로의 기준이
되는 이미지 루트 디렉터리만 있으면 된다. 이 스크립트는 3.3절에서 만든
번호판 크롭 데이터셋을 그 형식으로 바꾸고, 함께 필요한
  - train.txt / val.txt (학습/검증 분할)
  - plate_dict.txt      (번호판 전용 문자 사전)
를 생성한다.

문자 사전을 번호판 문자로만 줄이는 것이 핵심이다. 한국어 범용 사전은
수천 자이지만 번호판은 67자뿐이므로, 사전을 줄이면 출력층이 작아져
학습이 빨라지고 '번호판에 없는 글자'로 오인식할 여지가 사라진다.

이미지 디렉터리가 없어도 내장 예시로 변환 과정을 보여 주고 정상 종료한다.
"""
from __future__ import annotations

import argparse
import os
import random
import re
import sys
from collections import Counter

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "08_postprocess"))

try:
    from plate_rules import ALLOWED_CHARS, normalize, validate
    _HAS_RULES = True
except ImportError:
    _HAS_RULES = False
    ALLOWED_CHARS = set("0123456789")
    print("[안내] ch03/08_postprocess/plate_rules.py 를 찾지 못했다. "
          "형식 검증 없이 변환만 수행한다.")

    def normalize(t: str) -> str:              # type: ignore[misc]
        return (t or "").strip().upper()

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")
# 파일명에서 정답을 뽑는 패턴. 예) 0012_12가3456.jpg / 12가3456.jpg
NAME_RE = re.compile(r"(?:^|_)([0-9]{2,3}[가-힣][0-9]{4}"
                     r"|[가-힣]{2}[0-9]{2}[가-힣][0-9]{4})(?:_|\.|$)")


def label_from_name(filename: str) -> str | None:
    """파일명에서 정답 번호판 문자열을 뽑는다."""
    stem = os.path.splitext(os.path.basename(filename))[0]
    m = NAME_RE.search(stem + ".")
    if m:
        return m.group(1)
    # 패턴에 맞지 않으면 파일명 전체를 후보로 본다.
    cand = normalize(stem)
    return cand or None


def scan(image_root: str) -> list[tuple[str, str]]:
    """이미지 루트를 훑어 (상대경로, 정답) 목록을 만든다."""
    out: list[tuple[str, str]] = []
    for dirpath, _dirs, files in os.walk(image_root):
        for f in sorted(files):
            if not f.lower().endswith(IMG_EXT):
                continue
            label = label_from_name(f)
            if not label:
                continue
            rel = os.path.relpath(os.path.join(dirpath, f), image_root)
            out.append((rel.replace(os.sep, "/"), label))
    return out


def filter_valid(pairs: list[tuple[str, str]], strict: bool = True):
    """형식이 유효한 샘플만 남긴다. 반환값은 (채택, 제외[(경로,이유)])."""
    keep: list[tuple[str, str]] = []
    drop: list[tuple[str, str]] = []
    for rel, label in pairs:
        lab = normalize(label)
        bad = [c for c in lab if c not in ALLOWED_CHARS]
        if bad:
            drop.append((rel, f"사전 밖 문자 '{''.join(set(bad))}'"))
            continue
        if strict and _HAS_RULES and not validate(lab).ok:
            drop.append((rel, "번호판 형식 불일치"))
            continue
        keep.append((rel, lab))
    return keep, drop


def split(pairs: list[tuple[str, str]], val_ratio: float = 0.1,
          seed: int = 42):
    """학습/검증 분할. 같은 번호판이 양쪽에 섞이지 않게 번호 단위로 나눈다.

    REFERENCE.md 의 평가 기준 4번('학습에 사용하지 않은 차량과 촬영
    세션으로 독립 테스트를 수행한다')을 지키기 위한 분할 방식이다.
    """
    by_plate: dict[str, list[tuple[str, str]]] = {}
    for rel, lab in pairs:
        by_plate.setdefault(lab, []).append((rel, lab))
    plates = sorted(by_plate)
    rng = random.Random(seed)
    rng.shuffle(plates)
    n_val = max(1, int(len(plates) * val_ratio)) if plates else 0
    val_plates = set(plates[:n_val])
    train = [p for lab in plates if lab not in val_plates
             for p in by_plate[lab]]
    val = [p for lab in plates if lab in val_plates for p in by_plate[lab]]
    return train, val, len(val_plates)


def write_label_file(pairs: list[tuple[str, str]], path: str) -> None:
    """PaddleOCR rec 라벨 txt 를 쓴다. 구분자는 탭 하나다."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fp:
        for rel, lab in pairs:
            fp.write(f"{rel}\t{lab}\n")
    print(f"라벨 파일 생성: {path} ({len(pairs)}줄)")


def write_dict(pairs: list[tuple[str, str]], path: str) -> list[str]:
    """번호판 전용 문자 사전을 쓴다. 한 줄에 한 글자다."""
    chars = sorted({c for _rel, lab in pairs for c in lab})
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fp:
        for c in chars:
            fp.write(c + "\n")
    print(f"문자 사전 생성: {path} ({len(chars)}자)")
    return chars


def report_distribution(pairs: list[tuple[str, str]], top: int = 12) -> None:
    """문자 분포를 출력한다. 희소 문자가 있으면 증강 대상이 된다."""
    cnt = Counter(c for _rel, lab in pairs for c in lab)
    print(f"\n문자 종류 {len(cnt)}자 / 총 {sum(cnt.values())}자")
    print("가장 많은 문자 :", ", ".join(
        f"{c}({n})" for c, n in cnt.most_common(top)))
    print("가장 적은 문자 :", ", ".join(
        f"{c}({n})" for c, n in cnt.most_common()[-top:]))
    rare = [c for c, n in cnt.items() if n < 5]
    if rare:
        print(f"5회 미만 등장 문자 {len(rare)}자: {''.join(sorted(rare))}")
        print("  -> 3.3절의 합성 번호판 생성으로 보강한다.")


def builtin_pairs() -> list[tuple[str, str]]:
    """이미지 디렉터리가 없을 때 변환 과정을 보이기 위한 내장 예시."""
    plates = ["12가3456", "87로2345", "서울34나5678", "123가4567",
              "34허5678", "12바3456", "56누7890", "78고1234"]
    out: list[tuple[str, str]] = []
    for i, p in enumerate(plates):
        for k in range(3):                      # 번호판마다 3장씩 있다고 가정
            out.append((f"plates/{i:04d}_{p}_{k}.jpg", p))
    # 잘못된 샘플도 섞어 필터 동작을 보인다.
    out.append(("plates/9999_12ABC456_0.jpg", "12ABC456"))
    out.append(("plates/9998_12가345_0.jpg", "12가345"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="PaddleOCR rec 데이터셋 변환")
    ap.add_argument("--image-root", default="data/plates_crop",
                    help="번호판 크롭 이미지 루트 디렉터리")
    ap.add_argument("--out", default="data/paddle_rec",
                    help="라벨/사전 출력 디렉터리")
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--no-strict", action="store_true",
                    help="번호판 형식 검증을 건너뛴다")
    args = ap.parse_args()

    if os.path.isdir(args.image_root):
        pairs = scan(args.image_root)
        print(f"이미지 {len(pairs)}장을 찾았다: {args.image_root}")
    else:
        print(f"[안내] 이미지 디렉터리가 없다: {args.image_root}")
        print("       내장 예시 목록으로 변환 과정을 보인다.")
        pairs = builtin_pairs()

    keep, drop = filter_valid(pairs, strict=not args.no_strict)
    print(f"\n채택 {len(keep)}장 / 제외 {len(drop)}장")
    for rel, why in drop[:10]:
        print(f"  제외: {rel}  <- {why}")

    train, val, n_val_plates = split(keep, args.val_ratio)
    print(f"\n학습 {len(train)}장 / 검증 {len(val)}장 "
          f"(검증에만 쓰는 번호판 {n_val_plates}종)")

    out = args.out
    write_label_file(train, os.path.join(out, "train.txt"))
    write_label_file(val, os.path.join(out, "val.txt"))
    chars = write_dict(keep, os.path.join(out, "plate_dict.txt"))
    report_distribution(keep)

    print("\n=== paddle_rec_config.yml 에 넣을 값 ===")
    print(f"  Train.dataset.data_dir   : {args.image_root}")
    print(f"  Train.dataset.label_file_list : [{out}/train.txt]")
    print(f"  Eval.dataset.label_file_list  : [{out}/val.txt]")
    print(f"  Global.character_dict_path    : {out}/plate_dict.txt")
    print(f"  Architecture.Head.out_channels 기준 문자 수 : {len(chars)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
