# -*- coding: utf-8 -*-
"""문자별 등장 빈도를 분석하고 불균형 리포트를 출력한다.

실행:  python 03_dataset/class_balance.py --labels data/synth/labels.jsonl
라벨 파일이 없으면 합성 번호판 문자열을 만들어 분석한다.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
from charset import (CHARSET, DIGITS, PRIVATE_USE_CHARS,  # noqa: E402
                     REGION_NAMES, USE_CHARS, is_digit, is_use_char)


def load_texts(labels_path: Path | None, fallback_count: int = 4000) -> tuple[list[str], str]:
    """라벨 파일에서 문자열을 읽는다. 없으면 합성 문자열을 생성한다."""
    if labels_path and labels_path.exists():
        texts: list[str] = []
        with labels_path.open(encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                try:
                    texts.append(json.loads(line)["text"])
                except (json.JSONDecodeError, KeyError):
                    continue
        if texts:
            return texts, str(labels_path)

    # 실제 등록 분포에 가깝게 치우친 합성 데이터를 만든다.
    # 현실에서는 '가/나/다' 같은 앞쪽 용도문자가 '주/준' 류보다 훨씬 흔하다.
    rng = random.Random(0)
    weights = [max(1, 40 - i * 2) for i in range(len(PRIVATE_USE_CHARS))]
    texts = []
    for _ in range(fallback_count):
        use = rng.choices(PRIVATE_USE_CHARS, weights=weights, k=1)[0]
        tail = "".join(str(rng.randrange(10)) for _ in range(4))
        if rng.random() < 0.12:        # 구형 지역명 번호판 비중
            region = rng.choice(REGION_NAMES[:5])   # 수도권/대도시에 집중
            texts.append(f"{region}{rng.randrange(10)}{rng.randrange(10)}{use}{tail}")
        else:
            texts.append(f"{rng.randrange(1, 10)}{rng.randrange(10)}{use}{tail}")
    return texts, "(합성 문자열)"


def analyze(texts: list[str]) -> dict:
    """문자 빈도와 불균형 지표를 계산한다.

    불균형 지표 세 가지
      - 불균형비(imbalance ratio) = 최다 빈도 / 최소 빈도.
        10 을 넘으면 소수 클래스의 재현율이 눈에 띄게 떨어지기 시작한다.
      - 미등장 클래스 수: 한 번도 안 나온 문자는 학습 자체가 불가능하다.
        번호판 OCR 에서 이것이 가장 위험한 신호다. 실제 운영 중 그 문자가
        들어온 차량을 통째로 놓친다.
      - 상위 10개 점유율: 소수 클래스가 꼬리로 밀려난 정도를 본다.
    """
    counter = Counter(ch for text in texts for ch in text if ch in CHARSET)
    total = sum(counter.values())
    present = [counter[ch] for ch in CHARSET if counter[ch] > 0]
    missing = [ch for ch in CHARSET if counter[ch] == 0]

    top10 = counter.most_common(10)
    top10_share = sum(c for _, c in top10) / total if total else 0.0

    return {
        "total_chars": total,
        "total_texts": len(texts),
        "classes_total": len(CHARSET),
        "classes_present": len(present),
        "classes_missing": len(missing),
        "missing_chars": "".join(missing),
        "max_count": max(present) if present else 0,
        "min_count": min(present) if present else 0,
        "imbalance_ratio": round(max(present) / min(present), 2) if present else 0.0,
        "top10_share": round(top10_share, 4),
        "counter": counter,
    }


def sampling_weights(counter: Counter, power: float = 0.5) -> dict[str, float]:
    """역빈도 기반 샘플링 가중치를 만든다.

    왜 1/f 가 아니라 1/f^0.5 인가?
      완전한 역빈도(power=1.0)를 쓰면 한 번만 등장한 문자가 수천 배 가중을 받아
      그 한 장을 과적합한다. 제곱근으로 완화하면 소수 클래스를 끌어올리면서도
      다수 클래스의 다양성을 잃지 않는다. 실무에서 흔히 쓰는 절충값이다.
    """
    if not counter:
        return {}
    weights = {ch: (1.0 / (count ** power)) for ch, count in counter.items()}
    mean = sum(weights.values()) / len(weights)
    return {ch: round(w / mean, 3) for ch, w in weights.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="문자 클래스 불균형 리포트")
    parser.add_argument("--labels", type=str, default="data/synth/labels.jsonl")
    parser.add_argument("--top", type=int, default=12, help="상·하위 표시 개수")
    args = parser.parse_args()

    texts, source = load_texts(Path(args.labels))
    report = analyze(texts)
    counter: Counter = report["counter"]

    print(f"데이터 출처 : {source}")
    print(f"문자열 수   : {report['total_texts']:,}")
    print(f"총 문자 수  : {report['total_chars']:,}")
    print(f"클래스      : {report['classes_present']} / {report['classes_total']} 등장")
    print(f"미등장      : {report['classes_missing']}개  "
          f"[{report['missing_chars'][:40]}{'...' if report['classes_missing'] > 40 else ''}]")
    print(f"불균형비    : {report['imbalance_ratio']} (최다 {report['max_count']:,} / "
          f"최소 {report['min_count']:,})")
    print(f"상위10 점유 : {report['top10_share'] * 100:.1f}%")

    weights = sampling_weights(counter)
    print(f"\n[상위 {args.top}개 — 가중치를 낮춰야 할 문자]")
    print(f"{'문자':4s} {'종류':8s} {'빈도':>8s} {'비율':>7s} {'가중치':>7s}")
    for ch, cnt in counter.most_common(args.top):
        kind = "숫자" if is_digit(ch) else ("용도문자" if is_use_char(ch) else "지역명")
        print(f"{ch:4s} {kind:8s} {cnt:8,d} {cnt / report['total_chars'] * 100:6.2f}% "
              f"{weights.get(ch, 0):7.3f}")

    print(f"\n[하위 {args.top}개 — 데이터를 더 모아야 할 문자]")
    print(f"{'문자':4s} {'종류':8s} {'빈도':>8s} {'비율':>7s} {'가중치':>7s}")
    for ch, cnt in counter.most_common()[-args.top:]:
        kind = "숫자" if is_digit(ch) else ("용도문자" if is_use_char(ch) else "지역명")
        print(f"{ch:4s} {kind:8s} {cnt:8,d} {cnt / report['total_chars'] * 100:6.2f}% "
              f"{weights.get(ch, 0):7.3f}")

    # 구간별 요약 — 숫자는 과잉, 한글은 부족한 구조적 편향을 확인한다.
    print("\n[문자 종류별 요약]")
    groups = {"숫자": DIGITS, "용도문자": USE_CHARS,
              "지역명 음절": "".join(ch for ch in CHARSET
                                if not is_digit(ch) and not is_use_char(ch))}
    for name, chars in groups.items():
        counts = [counter[ch] for ch in chars]
        seen = [c for c in counts if c > 0]
        total = sum(counts)
        print(f"  {name:12s} 클래스 {len(seen):3d}/{len(chars):3d}  "
              f"총 {total:8,d}회  평균 {total / max(len(chars), 1):8.1f}회/클래스")

    print("\n권고")
    if report["classes_missing"]:
        print(f"  - 미등장 클래스 {report['classes_missing']}개는 합성 데이터로 반드시 채운다.")
    if report["imbalance_ratio"] > 10:
        print(f"  - 불균형비 {report['imbalance_ratio']} 는 과도하다. 역빈도 샘플링이나")
        print("    소수 클래스 집중 합성으로 10 이하까지 낮춘다.")
    print("  - 평가 지표는 전체 정확도가 아니라 클래스별 재현율로 본다.")
    print("    숫자가 70% 를 차지하면 한글을 다 틀려도 전체 정확도는 높게 나온다.")


if __name__ == "__main__":
    main()
