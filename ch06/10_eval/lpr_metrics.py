# -*- coding: utf-8 -*-
"""번호판 인식 평가지표 — 검출 / OCR 문자 / 완전 일치 / 차량 단위 (6.10절).

네 가지를 구분해 계산한다. 같은 모델이 지표에 따라 전혀 다른 점수를 받으므로
어느 지표로 보고하는지를 분명히 해야 한다.

    ① 검출 정확도   : 번호판 영역을 찾았는가 (IoU 기반 재현율/정밀도)
    ② 문자 정확도   : 읽은 글자가 몇 %나 맞았는가 (1 - CER)
    ③ 완전 일치율   : 번호판 전체가 한 글자도 안 틀렸는가 (EMR)
    ④ 차량 단위 인식률: 여러 프레임을 합쳐 차량 1대를 맞췄는가

외부 의존성은 numpy 하나뿐이고 데이터가 파일 안에 내장되어 있으므로
가중치 파일이나 네트워크 없이 그대로 실행된다.

실행:
    (mobility) d:\\mobility_ai\\ch06> python 10_eval/lpr_metrics.py
"""
from __future__ import annotations

import numpy as np

# =====================================================================
# 1. 검출 평가 — IoU 기반 TP/FP/FN
# =====================================================================

#: 검출 평가용 소표본. 이미지 10장, 각 이미지의 정답 박스와 예측 박스.
#: 박스는 (x1, y1, x2, y2) 픽셀 좌표이고, 예측에는 신뢰도 점수가 붙는다.
#: 6번째는 미검출(FN), 7번째는 오검출(FP)이 섞인 사례다.
DETECTION_SAMPLES: list[dict] = [
    {"gt": [(412, 288, 596, 342)], "pred": [((410, 286, 598, 344), 0.98)]},
    {"gt": [(120, 410, 268, 456)], "pred": [((124, 414, 262, 452), 0.95)]},
    {"gt": [(300, 200, 420, 240)], "pred": [((301, 201, 419, 239), 0.93)]},
    {"gt": [(55, 330, 190, 372)], "pred": [((60, 338, 182, 366), 0.88)]},
    # 박스가 크게 어긋나 IoU 가 임계값에 못 미친다 → FP + FN 동시 발생
    {"gt": [(500, 150, 640, 196)], "pred": [((540, 160, 700, 200), 0.71)]},
    # 번호판을 아예 못 찾았다 → FN
    {"gt": [(210, 260, 350, 304)], "pred": []},
    # 올바른 검출 1건 + 엉뚱한 곳에 저신뢰 검출 1건 → TP + FP
    {"gt": [(140, 120, 268, 160)], "pred": [((142, 122, 266, 158), 0.96),
                                            ((400, 400, 480, 430), 0.42)]},
    {"gt": [(80, 90, 210, 134)], "pred": [((82, 92, 208, 132), 0.97)]},
    {"gt": [(330, 420, 470, 468)], "pred": [((332, 418, 468, 470), 0.91)]},
    {"gt": [(260, 310, 398, 356)], "pred": [((258, 312, 400, 354), 0.94)]},
]

#: 검출 성공으로 인정할 IoU 임계값.
#: 일반 객체 검출은 0.5 를 쓰고, 번호판은 가늘고 길어 같은 0.5 에서도
#: 박스가 조금 어긋나면 쉽게 탈락한다. 임계값을 낮추면 재현율이 오르지만
#: OCR 에 넘기는 잘린 영역의 품질이 떨어져 문자 정확도가 내려간다.
IOU_THRESHOLD = 0.5

#: 이 점수 미만의 예측은 버린다. 백엔드 app.recognition.min-confidence 와
#: 같은 역할이고, 값을 올리면 FP 가 줄고 FN 이 늘어난다.
SCORE_THRESHOLD = 0.25


def iou(box_a: tuple[float, float, float, float],
        box_b: tuple[float, float, float, float]) -> float:
    """두 박스의 IoU(Intersection over Union)를 계산한다.

    정의식::

        IoU = |A ∩ B| / |A ∪ B| = 교집합 면적 / (A면적 + B면적 - 교집합 면적)

    Args:
        box_a: (x1, y1, x2, y2)
        box_b: (x1, y1, x2, y2)

    Returns:
        0.0 ~ 1.0 의 겹침 비율. 겹치지 않으면 0.0.
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    # 교집합 사각형. 겹치지 않으면 폭이나 높이가 음수가 되므로 0 으로 가둔다.
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def evaluate_detection(samples: list[dict],
                       iou_thr: float = IOU_THRESHOLD,
                       score_thr: float = SCORE_THRESHOLD) -> dict[str, float]:
    """검출 재현율/정밀도/F1 을 계산한다.

    정의식::

        Recall    = TP / (TP + FN)   찾아야 할 번호판 중 찾은 비율
        Precision = TP / (TP + FP)   찾았다고 한 것 중 실제인 비율
        F1        = 2PR / (P + R)    두 값의 조화평균

    매칭 규칙: 한 정답 박스에는 예측을 하나만 매칭한다(점수 높은 것 우선).
    이미 매칭된 정답에 또 맞은 예측은 중복 검출이므로 FP 로 센다.
    이 규칙이 없으면 같은 번호판에 박스를 열 개 그려도 재현율이 떨어지지
    않아 "많이 그리면 이기는" 평가가 된다.

    Args:
        samples: {"gt": [박스], "pred": [(박스, 점수)]} 목록.
        iou_thr: 성공 인정 IoU 임계값.
        score_thr: 점수 하한.

    Returns:
        tp/fp/fn/recall/precision/f1/mean_iou 를 담은 dict.
    """
    tp = fp = fn = 0
    matched_ious: list[float] = []

    for sample in samples:
        gts = list(sample["gt"])
        preds = [p for p in sample["pred"] if p[1] >= score_thr]
        # 점수 내림차순으로 욕심껏(greedy) 매칭한다.
        preds.sort(key=lambda p: p[1], reverse=True)
        used = [False] * len(gts)

        for box, _score in preds:
            best_i, best_iou = -1, 0.0
            for gi, gt in enumerate(gts):
                if used[gi]:
                    continue
                value = iou(box, gt)
                if value > best_iou:
                    best_i, best_iou = gi, value
            if best_i >= 0 and best_iou >= iou_thr:
                used[best_i] = True
                tp += 1
                matched_ious.append(best_iou)
            else:
                fp += 1
        fn += used.count(False)

    recall = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    mean_iou = float(np.mean(matched_ious)) if matched_ious else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "recall": recall,
            "precision": precision, "f1": f1, "mean_iou": mean_iou}


# =====================================================================
# 2. OCR 평가 — 문자 정확도 / 완전 일치율
# =====================================================================

#: OCR 평가용 (정답, 예측) 쌍 12건. 실무에서 자주 보는 실패 유형을 섞었다.
#: 숫자 혼동(6→0), 한글 혼동(로→루), 문자 누락, 영문 혼동(0→O).
OCR_SAMPLES: list[tuple[str, str]] = [
    ("12가3456", "12가3456"),      # 정상
    ("서울34나5678", "서울34나5678"),
    ("87로2345", "87로2345"),
    ("123다4567", "123다4567"),
    ("45마6789", "45마6789"),
    ("12가3456", "12가3450"),      # 6 → 0 (숫자 혼동)
    ("87로2345", "87루2345"),      # 로 → 루 (한글 혼동)
    ("56바7890", "56바789"),        # 마지막 문자 누락
    ("90사1234", "9O사1234"),      # 0 → O (영문 혼동)
    ("11아2233", "11아2233"),
    ("34자5566", "34자5566"),
    ("78차9900", "78차99OO"),      # 0 → O 2자
]


def levenshtein(a: str, b: str) -> int:
    """두 문자열의 편집 거리(삽입/삭제/치환 최소 횟수)를 구한다.

    동적 계획법 표를 한 행씩만 유지해 메모리를 O(len(b)) 로 줄인다.

    문자 정확도를 "같은 위치끼리 비교" 로 재면 안 되는 이유가 이 함수에 있다.
    문자 하나가 누락되면 그 뒤 모든 글자의 위치가 한 칸 밀려 전부 오답이 된다.
    편집 거리는 삽입·삭제를 1회로 세므로 누락 1건을 오류 1건으로 센다.

    Args:
        a: 기준 문자열(정답).
        b: 비교 문자열(예측).

    Returns:
        편집 거리.
    """
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cur[j] = min(
                prev[j] + 1,               # 삭제
                cur[j - 1] + 1,            # 삽입
                prev[j - 1] + (ca != cb),  # 치환
            )
        prev = cur
    return prev[-1]


def evaluate_ocr(samples: list[tuple[str, str]]) -> dict[str, float]:
    """문자 정확도(CAR), 문자 오류율(CER), 완전 일치율(EMR)을 계산한다.

    정의식::

        CER = Σ 편집거리(정답, 예측) / Σ len(정답)
        CAR = 1 - CER
        EMR = (정답 == 예측 인 표본 수) / 전체 표본 수

    CER 을 표본별로 평균하지 않고 전체 편집거리를 전체 글자 수로 나누는
    것에 주의한다. 표본별 평균을 쓰면 짧은 번호판의 오류 1건이 긴 번호판의
    오류 1건보다 더 큰 영향을 준다(7자 중 1자 = 14.3%, 10자 중 1자 = 10%).
    전체 합으로 계산하면 글자 하나가 어디에 있든 같은 무게를 가진다.

    Args:
        samples: (정답, 예측) 쌍 목록.

    Returns:
        지표 dict.
    """
    total_chars = sum(len(truth) for truth, _ in samples)
    total_edits = sum(levenshtein(truth, pred) for truth, pred in samples)
    exact = sum(1 for truth, pred in samples if truth == pred)

    cer = total_edits / total_chars if total_chars else 0.0
    return {
        "samples": len(samples),
        "total_chars": total_chars,
        "edits": total_edits,
        "char_accuracy": 1.0 - cer,
        "cer": cer,
        "exact": exact,
        "exact_match_rate": exact / len(samples) if samples else 0.0,
    }


# =====================================================================
# 3. 차량 단위 인식률 — 여러 프레임을 다수결로 합쳐 1대로 판정
# =====================================================================

#: 차량 6대. 각 차량을 3프레임씩 찍어 프레임별 OCR 결과를 얻었다고 가정한다.
#: None 은 그 프레임에서 번호판을 못 읽은 경우다.
VEHICLE_FRAMES: list[dict] = [
    {"truth": "12가3456", "frames": ["12가3456", "12가3456", "12가3450"]},
    {"truth": "서울34나5678", "frames": ["서울34나5678", "서울34나5678", "서울34나5678"]},
    {"truth": "87로2345", "frames": ["87루2345", "87로2345", "87로2345"]},
    {"truth": "56바7890", "frames": ["56바789", "56바7890", None]},
    # 세 프레임 중 둘이 같은 오답이면 다수결이 오답을 택한다(2:1).
    {"truth": "90사1234", "frames": ["9O사1234", "9O사1234", "90사1234"]},
    # 두 프레임이 미검출이어도 남은 한 프레임이 맞으면 성공한다.
    {"truth": "78차9900", "frames": [None, None, "78차9900"]},
]


def majority_vote(frames: list[str | None]) -> str | None:
    """프레임별 판독 결과를 다수결로 합친다.

    동률이면 먼저 나온 후보를 택한다(결정적 동작 보장). 무작위로 고르면
    같은 입력에 다른 결과가 나와 재현이 불가능해진다.

    Args:
        frames: 프레임별 문자열. 미검출은 None.

    Returns:
        최다 득표 문자열. 유효 프레임이 없으면 None.
    """
    counts: dict[str, int] = {}
    for text in frames:
        if not text:
            continue
        counts[text] = counts.get(text, 0) + 1
    if not counts:
        return None
    best = max(counts.items(), key=lambda kv: kv[1])
    return best[0]


def evaluate_vehicle_level(vehicles: list[dict]) -> dict[str, float]:
    """차량 단위 인식률을 프레임 단위와 함께 비교한다.

    정의식::

        프레임 단위 = (정답과 일치한 프레임 수) / (전체 프레임 수)
        차량 단위   = (다수결 결과가 정답인 차량 수) / (전체 차량 수)

    두 숫자가 다르다는 것이 핵심이다. 운영자가 알고 싶은 것은 "차단기가
    차량 1대를 맞게 처리했는가" 이므로 차량 단위가 보고해야 할 지표다.
    논문이 흔히 보고하는 프레임 단위 수치는 그보다 낮게 나온다.

    Args:
        vehicles: {"truth": 정답, "frames": [프레임 결과]} 목록.

    Returns:
        프레임 단위 정확도와 차량 단위 인식률.
    """
    frame_total = frame_ok = 0
    vehicle_ok = 0

    for item in vehicles:
        for text in item["frames"]:
            frame_total += 1
            if text == item["truth"]:
                frame_ok += 1
        if majority_vote(item["frames"]) == item["truth"]:
            vehicle_ok += 1

    return {
        "frames": frame_total,
        "frame_ok": frame_ok,
        "frame_rate": frame_ok / frame_total if frame_total else 0.0,
        "vehicles": len(vehicles),
        "vehicle_ok": vehicle_ok,
        "vehicle_rate": vehicle_ok / len(vehicles) if vehicles else 0.0,
    }


def pct(value: float, digits: int = 1) -> str:
    """0~1 비율을 퍼센트 문자열로 바꾼다."""
    return f"{value * 100:.{digits}f}%"


def main() -> None:
    """세 가지 번호판 인식 지표를 계산해 출력한다."""
    print("=" * 62)
    print(" 번호판 인식 평가 (검출 / OCR / 차량 단위)")
    print("=" * 62)

    det = evaluate_detection(DETECTION_SAMPLES)
    print("\n[1] 번호판 검출 (IoU >= %.2f, score >= %.2f)" % (IOU_THRESHOLD, SCORE_THRESHOLD))
    print(f"  TP={det['tp']}  FP={det['fp']}  FN={det['fn']}")
    print(f"  재현율(Recall)   = {det['recall']:.4f}  ({pct(det['recall'])})")
    print(f"  정밀도(Precision)= {det['precision']:.4f}  ({pct(det['precision'])})")
    print(f"  F1              = {det['f1']:.4f}")
    print(f"  매칭 박스 평균 IoU = {det['mean_iou']:.4f}")

    ocr = evaluate_ocr(OCR_SAMPLES)
    print("\n[2] OCR 문자 정확도 / 완전 일치율")
    print(f"  표본 {ocr['samples']}건, 정답 문자 {ocr['total_chars']}자, 편집 거리 합 {ocr['edits']}")
    print(f"  문자 정확도(CAR) = {ocr['char_accuracy']:.4f}  ({pct(ocr['char_accuracy'], 2)})")
    print(f"  문자 오류율(CER) = {ocr['cer']:.4f}  ({pct(ocr['cer'], 2)})")
    print(f"  완전 일치 {ocr['exact']}/{ocr['samples']}")
    print(f"  완전 일치율(EMR) = {ocr['exact_match_rate']:.4f}  ({pct(ocr['exact_match_rate'], 2)})")

    veh = evaluate_vehicle_level(VEHICLE_FRAMES)
    print("\n[3] 차량 단위 인식률 (3프레임 다수결)")
    print(f"  프레임 단위 : {veh['frame_ok']}/{veh['frames']} = {pct(veh['frame_rate'], 2)}")
    print(f"  차량 단위   : {veh['vehicle_ok']}/{veh['vehicles']} = {pct(veh['vehicle_rate'], 2)}")
    print(f"  다수결 이득 : {pct(veh['vehicle_rate'] - veh['frame_rate'], 2)}p")


if __name__ == "__main__":
    main()
