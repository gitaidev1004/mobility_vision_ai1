# -*- coding: utf-8 -*-
"""학습용 인덱스(JSONL)를 만든다 — LMDB 를 쓰지 않는 경량 대안.

전통적인 OCR 학습 코드는 수십만 장을 LMDB 한 덩어리로 묶는다. 작은 파일 수십만 개를
매 epoch 열고 닫는 비용이 크기 때문이다. 다만 LMDB 는 설치와 디버깅이 번거롭고,
본 프로젝트 규모(수만 장)에서는 JSONL 인덱스 + 이미지 파일로 충분하다.
인덱스를 쓰면 디렉터리 스캔이 사라져 DataLoader 기동이 빨라지고, split 이 고정된다.

실행:  python 03_dataset/make_lmdb_like.py --src data/synth --out data/index
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
from charset import CHARSET, encode  # noqa: E402

#: 라벨로 허용할 최대/최소 글자 수. 한국 번호판은 7~9자다
#: (신형 2자리 7자, 신형 3자리 8자, 구형 지역명 포함 9자).
MIN_LEN, MAX_LEN = 7, 9


def stable_split(key: str, ratios: tuple[float, float, float]) -> str:
    """키 해시로 train/val/test 를 결정한다.

    왜 난수 셔플이 아니라 해시인가?
      난수로 나누면 데이터를 추가할 때마다 split 이 전부 바뀐다. 그러면
      어제의 검증셋 일부가 오늘의 학습셋에 들어가 성능이 부풀려진다.
      해시는 파일명만 같으면 몇 번을 다시 돌려도 같은 분할을 준다.
      md5 를 쓰는 이유는 보안이 아니라 '실행 간 재현성' 이다. 파이썬의 hash() 는
      프로세스마다 시드가 달라져 재현되지 않는다.
    """
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()  # noqa: S324
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    train_r, val_r, _ = ratios
    if bucket < train_r:
        return "train"
    if bucket < train_r + val_r:
        return "val"
    return "test"


def validate_record(rec: dict, src_dir: Path) -> tuple[bool, str]:
    """레코드 한 건의 유효성을 검사한다.

    학습 직전 검증을 하는 이유: 깨진 라벨 한 건이 수 시간짜리 학습을 중간에
    죽이거나, 더 나쁘게는 조용히 모델 품질을 깎는다. 인덱스를 만드는 지금이
    가장 싸게 걸러낼 수 있는 시점이다.
    """
    image = rec.get("image")
    text = (rec.get("text") or "").strip()
    if not image:
        return False, "image 필드 없음"
    if not (src_dir / image).exists():
        return False, f"이미지 파일 없음: {image}"
    if not text:
        return False, "라벨 비어 있음"
    if not (MIN_LEN <= len(text) <= MAX_LEN):
        return False, f"길이 이상({len(text)}자, 허용 {MIN_LEN}~{MAX_LEN})"
    unknown = [ch for ch in text if ch not in CHARSET]
    if unknown:
        return False, f"문자셋 밖 문자 {unknown}"
    return True, ""


def build_index(src_dir: Path, out_dir: Path,
                ratios: tuple[float, float, float]) -> dict:
    """원본 라벨을 검증하고 split 별 JSONL 인덱스를 쓴다."""
    labels_path = src_dir / "labels.jsonl"
    if not labels_path.exists():
        raise FileNotFoundError(
            f"{labels_path} 가 없다. 먼저 synth_plate.py 로 데이터를 만든다.")

    out_dir.mkdir(parents=True, exist_ok=True)
    buckets: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    rejected: list[dict] = []
    seen_keys: set[str] = set()

    with labels_path.open(encoding="utf-8") as fp:
        for lineno, line in enumerate(fp, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                rejected.append({"line": lineno, "reason": f"JSON 오류 {exc.msg}"})
                continue

            ok, reason = validate_record(rec, src_dir)
            if not ok:
                rejected.append({"line": lineno, "image": rec.get("image"),
                                 "reason": reason})
                continue

            # 같은 이미지가 두 번 들어오면 split 이 꼬이므로 중복을 제거한다.
            if rec["image"] in seen_keys:
                rejected.append({"line": lineno, "image": rec["image"],
                                 "reason": "중복 이미지"})
                continue
            seen_keys.add(rec["image"])

            text = rec["text"].strip()
            entry = {
                "image": str((src_dir / rec["image"]).as_posix()),
                "text": text,
                "label": encode(text),          # 인덱스에 미리 인코딩해 둔다
                "length": len(text),
                "spec": rec.get("spec", "unknown"),
            }
            buckets[stable_split(rec["image"], ratios)].append(entry)

    stats: dict = {"rejected": len(rejected), "splits": {}}
    for split, items in buckets.items():
        path = out_dir / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as fp:
            for entry in items:
                fp.write(json.dumps(entry, ensure_ascii=False) + "\n")
        char_count = Counter(ch for e in items for ch in e["text"])
        stats["splits"][split] = {
            "count": len(items),
            "path": str(path),
            "unique_classes": len(char_count),
            "avg_length": round(
                sum(e["length"] for e in items) / len(items), 2) if items else 0.0,
        }

    if rejected:
        with (out_dir / "rejected.jsonl").open("w", encoding="utf-8") as fp:
            for item in rejected:
                fp.write(json.dumps(item, ensure_ascii=False) + "\n")

    meta = {
        "charset": CHARSET,
        "num_classes": len(CHARSET) + 1,
        "blank_index": len(CHARSET),
        "min_len": MIN_LEN,
        "max_len": MAX_LEN,
        "ratios": {"train": ratios[0], "val": ratios[1], "test": ratios[2]},
        "stats": stats,
    }
    (out_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    stats["rejected_samples"] = rejected[:5]
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="학습용 JSONL 인덱스 생성")
    parser.add_argument("--src", type=str, default="data/synth")
    parser.add_argument("--out", type=str, default="data/index")
    parser.add_argument("--train", type=float, default=0.8)
    parser.add_argument("--val", type=float, default=0.1)
    args = parser.parse_args()

    test_ratio = max(0.0, 1.0 - args.train - args.val)
    try:
        stats = build_index(Path(args.src), Path(args.out),
                            (args.train, args.val, test_ratio))
    except FileNotFoundError as exc:
        print(f"[안내] {exc}")
        raise SystemExit(0) from None

    print(f"{'split':8s} {'건수':>8s} {'평균길이':>9s} {'등장클래스':>10s}  경로")
    print("-" * 72)
    for split in ("train", "val", "test"):
        info = stats["splits"][split]
        print(f"{split:8s} {info['count']:8,d} {info['avg_length']:9.2f} "
              f"{info['unique_classes']:10d}  {info['path']}")

    print(f"\n거부된 레코드: {stats['rejected']}건")
    for item in stats.get("rejected_samples", []):
        print(f"  line {item.get('line')}: {item.get('reason')}")
    print(f"\nmeta.json 에 문자셋과 클래스 수를 함께 저장했다.")
    print("학습·추론 양쪽이 이 파일을 읽으면 문자셋 불일치 사고를 막을 수 있다.")


if __name__ == "__main__":
    main()
