# -*- coding: utf-8 -*-
"""데이터셋을 Train/Validation/Test 로 분할한다.

핵심 원칙: '촬영 세션(또는 차량)' 단위로 그룹을 묶어 분할한다.
같은 세션에서 연속 촬영한 프레임이 Train 과 Test 에 나뉘어 들어가면
테스트 성능이 실제보다 높게 나오는 데이터 누수(Data Leakage)가 발생한다.

세션 키 추출 규칙(기본): 파일명의 첫 두 토큰, 예)
    20260311_gate1_0007.jpg -> 세션 "20260311_gate1"
    plate_12가3456_03.jpg   -> 세션 "plate_12가3456"

표준 라이브러리만 사용하므로 어떤 환경에서도 실행된다.

사용법:
    python 03_dataset/split_dataset.py --images data/raw/images --labels data/raw/labels \
        --out data/plate --ratio 0.7 0.15 0.15 --seed 42
"""
from __future__ import annotations

import argparse
import pathlib
import random
import shutil
import sys

EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="세션 기준 데이터셋 분할")
    ap.add_argument("--images", default="data/raw/images")
    ap.add_argument("--labels", default="data/raw/labels")
    ap.add_argument("--out", default="data/plate", help="출력 루트")
    ap.add_argument("--ratio", type=float, nargs=3, default=[0.7, 0.15, 0.15],
                    metavar=("TRAIN", "VAL", "TEST"))
    ap.add_argument("--seed", type=int, default=42, help="재현성을 위한 난수 시드")
    ap.add_argument("--session-tokens", type=int, default=2,
                    help="파일명 앞 N개 토큰을 세션 키로 쓴다")
    ap.add_argument("--copy", action="store_true",
                    help="지정하면 실제로 파일을 복사한다(기본은 모의 실행)")
    return ap.parse_args()


def session_key(stem: str, n_tokens: int) -> str:
    """파일명에서 촬영 세션 키를 뽑는다."""
    parts = stem.split("_")
    if len(parts) <= n_tokens:
        return stem
    return "_".join(parts[:n_tokens])


def group_by_session(files: list[pathlib.Path], n_tokens: int) -> dict[str, list[pathlib.Path]]:
    """세션 키별로 파일 목록을 모은다."""
    groups: dict[str, list[pathlib.Path]] = {}
    for f in files:
        groups.setdefault(session_key(f.stem, n_tokens), []).append(f)
    return groups


def split_groups(groups: dict[str, list[pathlib.Path]], ratio, seed: int):
    """세션 단위로 섞어 목표 비율에 가장 가깝게 배분한다.

    세션마다 이미지 수가 다르므로 큰 세션부터 '가장 부족한 분할'에 배정하는
    탐욕적(greedy) 방법을 쓴다. 단순 무작위 배분보다 비율 오차가 작다.
    """
    keys = sorted(groups)                    # 정렬 후 섞어 결과를 재현 가능하게 한다
    rng = random.Random(seed)
    rng.shuffle(keys)
    # 큰 세션부터 배치하면 마지막에 비율이 크게 틀어지는 일이 줄어든다
    keys.sort(key=lambda k: len(groups[k]), reverse=True)

    total = sum(len(groups[k]) for k in keys)
    targets = [total * r for r in ratio]
    assigned: list[list[str]] = [[], [], []]
    counts = [0, 0, 0]
    for k in keys:
        # 목표 대비 가장 덜 채워진 분할을 고른다
        deficit = [targets[i] - counts[i] for i in range(3)]
        idx = max(range(3), key=lambda i: deficit[i])
        assigned[idx].append(k)
        counts[idx] += len(groups[k])
    return assigned, counts, total


def main() -> int:
    args = parse_args()
    if abs(sum(args.ratio) - 1.0) > 1e-6:
        print(f"[오류] ratio 합이 1.0 이 아니다: {args.ratio}")
        return 1

    img_dir = pathlib.Path(args.images)
    lbl_dir = pathlib.Path(args.labels)
    if not img_dir.is_dir():
        print(f"[안내] 이미지 디렉터리가 없다: {img_dir}")
        print("       먼저 데이터를 수집해 data/raw/images 에 넣는다.")
        return 0

    files = sorted(p for p in img_dir.rglob("*") if p.suffix.lower() in EXTS)
    if not files:
        print(f"[안내] {img_dir} 에 이미지가 없다.")
        return 0

    groups = group_by_session(files, args.session_tokens)
    assigned, counts, total = split_groups(groups, args.ratio, args.seed)
    names = ["train", "val", "test"]

    print(f"=== 분할 계획 (이미지 {total} 장 / 세션 {len(groups)} 개) ===")
    for i, nm in enumerate(names):
        pct = counts[i] / total * 100 if total else 0.0
        print(f"  {nm:5s} 세션 {len(assigned[i]):3d} 개  이미지 {counts[i]:5d} 장 "
              f"({pct:5.1f}%, 목표 {args.ratio[i]*100:5.1f}%)")

    print("\n=== 세션 중복 검사 ===")
    sets = [set(a) for a in assigned]
    overlap = (sets[0] & sets[1]) | (sets[1] & sets[2]) | (sets[0] & sets[2])
    if overlap:
        print(f"  [오류] 세션이 두 분할에 동시 포함되었다: {sorted(overlap)[:5]}")
        return 1
    print("  통과: 같은 촬영 세션이 두 분할에 걸쳐 있지 않다(누수 없음).")

    if not args.copy:
        print("\n[모의 실행] --copy 를 붙이면 실제로 파일을 복사한다.")
        for i, nm in enumerate(names):
            sample = assigned[i][:3]
            print(f"  {nm} 예시 세션: {sample}")
        return 0

    out = pathlib.Path(args.out)
    for i, nm in enumerate(names):
        (out / "images" / nm).mkdir(parents=True, exist_ok=True)
        (out / "labels" / nm).mkdir(parents=True, exist_ok=True)
        for key in assigned[i]:
            for f in groups[key]:
                shutil.copy2(f, out / "images" / nm / f.name)
                lab = lbl_dir / (f.stem + ".txt")
                if lab.exists():
                    shutil.copy2(lab, out / "labels" / nm / lab.name)
                else:
                    # 번호판이 없는 배경 이미지는 빈 라벨 파일을 둔다(FP 억제에 도움)
                    (out / "labels" / nm / (f.stem + ".txt")).write_text(
                        "", encoding="utf-8")
    print(f"\n복사 완료: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
