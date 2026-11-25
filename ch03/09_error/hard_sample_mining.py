# -*- coding: utf-8 -*-
"""오류 샘플 재학습 후보(hard sample) 추출.

오류를 전부 재학습에 넣으면 오히려 성능이 떨어진다. 라벨이 틀린 샘플,
사람도 못 읽는 샘플, 이미 과대 표집된 유형이 섞이기 때문이다.
이 스크립트는 다음 기준으로 '배울 가치가 있는' 샘플만 골라낸다.

  (1) 틀렸지만 모델이 자신 있었던 샘플(high-confidence error) — 최우선
  (2) 편집 거리가 1~2 인 샘플 — 조금만 더 배우면 맞출 수 있다
  (3) 희소 문자(등장 횟수가 적은 용도 기호)를 포함한 샘플
  (4) 이미지 품질이 판독 가능 범위인 샘플 — 품질이 너무 낮으면 제외
  (5) 유형별 상한(quota)을 둬서 한 유형이 재학습 세트를 지배하지 않게 한다

ch03/09_error/error_taxonomy.py 의 분류 결과를 입력으로 받는다.
"""
from __future__ import annotations

import csv
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "10_eval"))

from error_taxonomy import (ERROR_TYPES, Sample, builtin_samples,  # noqa: E402
                            classify)
from metrics_ocr import levenshtein, normalize                     # noqa: E402

# 재학습 후보 선정 기준값
CONF_HIGH = 0.55          # 이 값 이상인데 틀렸으면 최우선 후보
DIST_MAX = 2              # 편집 거리 상한
MIN_PLATE_PX = 70         # 이보다 작으면 사람도 읽기 어려워 제외
MIN_BLUR = 30.0           # 너무 흐리면 제외
MAX_OCCLUSION = 0.35      # 가림이 심하면 제외
QUOTA_PER_TYPE = 40       # 유형별 최대 샘플 수

# 희소 문자(학습 데이터에 적게 등장하는 용도 기호) — 3.3절의 분포 분석 결과.
RARE_CHARS = set("배허하호아바사자")


@dataclass
class MiningResult:
    """재학습 후보 한 건."""

    sample_id: str
    gt: str
    pred: str
    error_type: str
    confidence: float
    distance: int
    priority: float            # 높을수록 먼저 재학습에 넣는다
    reasons: str
    augment: str               # 권장 증강 방식

    def line(self) -> str:
        return (f"{self.sample_id:<8s} {self.gt:<11s} {self.pred or '(없음)':<11s} "
                f"{self.error_type:<18s} p={self.priority:5.2f}  "
                f"{self.augment:<14s} {self.reasons}")


# 오류 유형별 권장 증강 방식 — 재학습 시 어떤 변형을 늘릴지 정한다.
AUGMENT_BY_TYPE: dict[str, str] = {
    "CONF_DIGIT_ALPHA": "폰트변형+대비",
    "CONF_DIGIT_DIGIT": "폰트변형+블러",
    "CONF_HANGUL": "한글폰트확장",
    "LOW_RESOLUTION": "다운스케일",
    "GLARE": "밝기+반사광",
    "NIGHT_NOISE": "저조도+노이즈",
    "SKEW": "원근+회전",
    "OCCLUSION": "랜덤이레이징",
    "LENGTH_MISMATCH": "문자간격변형",
    "UNKNOWN": "원본재검수",
}


def quality_ok(s: Sample) -> tuple[bool, str]:
    """이미지 품질이 재학습에 쓸 만한지 판정한다."""
    if s.plate_px_width and s.plate_px_width < MIN_PLATE_PX:
        return False, f"번호판 폭 {s.plate_px_width}px 가 너무 작다"
    if s.blur < MIN_BLUR:
        return False, f"흐림 지표 {s.blur:.0f} 가 너무 낮다"
    if s.occlusion > MAX_OCCLUSION:
        return False, f"가림률 {s.occlusion:.0%} 가 너무 높다"
    return True, ""


def priority_score(s: Sample, error_type: str, dist: int) -> tuple[float, str]:
    """재학습 우선순위 점수와 근거를 계산한다."""
    score = 0.0
    why: list[str] = []

    # (1) 자신 있게 틀린 샘플: 모델이 잘못된 패턴을 '확신'하고 있다는 뜻이다.
    if s.confidence >= CONF_HIGH:
        score += 3.0 + (s.confidence - CONF_HIGH) * 4.0
        why.append(f"고신뢰 오류(conf={s.confidence:.2f})")
    else:
        score += 1.0
        why.append(f"저신뢰 오류(conf={s.confidence:.2f})")

    # (2) 거의 맞춘 샘플에 가중치를 둔다.
    if dist == 1:
        score += 2.0
        why.append("1자 차이")
    elif dist == 2:
        score += 1.0
        why.append("2자 차이")

    # (3) 희소 문자 포함 여부
    rare = RARE_CHARS & set(normalize(s.gt))
    if rare:
        score += 1.5
        why.append(f"희소 문자 포함({''.join(sorted(rare))})")

    # (4) 환경 오류는 데이터 증강으로 바로 보강할 수 있으므로 가중치를 준다.
    if error_type in {"NIGHT_NOISE", "GLARE", "SKEW", "LOW_RESOLUTION"}:
        score += 0.8
        why.append("증강으로 보강 가능한 환경 오류")

    # (5) 설명되지 않는 오류는 라벨 오류 가능성이 있으므로 감점한다.
    if error_type == "UNKNOWN":
        score -= 1.5
        why.append("원인 불명 — 라벨 재검수 필요")

    return score, "; ".join(why)


def mine(samples: list[Sample], quota: int = QUOTA_PER_TYPE) -> \
        tuple[list[MiningResult], list[tuple[str, str]]]:
    """재학습 후보를 추출한다.

    반환값은 (채택 목록, 제외 목록[(샘플ID, 이유)])이다.
    """
    picked: list[MiningResult] = []
    dropped: list[tuple[str, str]] = []

    for s in samples:
        dg = classify(s)
        if dg.correct:
            dropped.append((s.sample_id, "정답 샘플"))
            continue
        dist = levenshtein(normalize(s.gt), normalize(s.pred))
        if dist > DIST_MAX:
            dropped.append((s.sample_id, f"편집 거리 {dist} > {DIST_MAX}"))
            continue
        ok, why = quality_ok(s)
        if not ok:
            dropped.append((s.sample_id, why))
            continue
        et = dg.primary
        score, reasons = priority_score(s, et, dist)
        picked.append(MiningResult(
            s.sample_id, normalize(s.gt), normalize(s.pred), et,
            s.confidence, dist, score, reasons,
            AUGMENT_BY_TYPE.get(et, "원본재검수")))

    # 우선순위 내림차순 정렬 후 유형별 상한 적용
    picked.sort(key=lambda r: -r.priority)
    used: Counter[str] = Counter()
    final: list[MiningResult] = []
    for r in picked:
        if used[r.error_type] >= quota:
            dropped.append((r.sample_id, f"{r.error_type} 유형 상한 초과"))
            continue
        used[r.error_type] += 1
        final.append(r)
    return final, dropped


def export_csv(rows: list[MiningResult], path: str) -> None:
    """재학습 작업 지시서를 CSV 로 저장한다."""
    with open(path, "w", encoding="utf-8-sig", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["sample_id", "gt", "pred", "error_type", "confidence",
                    "distance", "priority", "augment", "reasons"])
        for r in rows:
            w.writerow([r.sample_id, r.gt, r.pred, r.error_type,
                        f"{r.confidence:.3f}", r.distance,
                        f"{r.priority:.2f}", r.augment, r.reasons])
    print(f"재학습 지시서를 저장했다: {path} ({len(rows)}건)")


def export_jsonl(rows: list[MiningResult], path: str) -> None:
    """학습 파이프라인이 바로 읽을 수 있는 JSON Lines 로도 저장한다."""
    with open(path, "w", encoding="utf-8") as fp:
        for r in rows:
            fp.write(json.dumps({
                "sample_id": r.sample_id, "label": r.gt,
                "error_type": r.error_type, "augment": r.augment,
                "priority": round(r.priority, 3),
            }, ensure_ascii=False) + "\n")
    print(f"학습용 JSONL 을 저장했다: {path}")


def _demo() -> None:
    # 내장 예시에 저품질/대거리 샘플을 추가해 필터 동작까지 보인다.
    samples = builtin_samples() + [
        Sample("S011", "12가3456", "34나7890", 0.22, 150, 110, 180, 3.0, 0.0),
        Sample("S012", "34허5678", "34허5B78", 0.72, 178, 119, 215, 1.0, 0.0),
        Sample("S013", "12바3456", "12바345", 0.58, 95, 108, 25, 2.0, 0.0),
        Sample("S014", "87로2345", "87로2345", 0.91, 180, 120, 220, 1.0, 0.0),
        Sample("S015", "12배3456", "12바3456", 0.66, 172, 116, 205, 1.5, 0.0),
    ]
    picked, dropped = mine(samples)

    print(f"=== 재학습 후보 {len(picked)}건 (우선순위 순) ===")
    print(f"{'ID':<8s} {'정답':<11s} {'예측':<11s} {'유형':<18s} "
          f"{'점수':<8s} {'증강':<14s} 근거")
    for r in picked:
        print(r.line())

    print(f"\n=== 제외 {len(dropped)}건 ===")
    for sid, why in dropped:
        print(f"{sid:<8s} {why}")

    print("\n=== 유형별 채택 건수 ===")
    cnt = Counter(r.error_type for r in picked)
    for t, n in cnt.most_common():
        print(f"{t:<18s} {n:>3d}건  {ERROR_TYPES[t]}")

    export_csv(picked, "hard_samples.csv")
    export_jsonl(picked, "hard_samples.jsonl")


if __name__ == "__main__":
    _demo()
