# -*- coding: utf-8 -*-
"""예지보전 평가지표 — 혼동행렬 / Accuracy / Precision·Recall / F1 / ROC-AUC / RUL (6.10절).

분류 지표와 회귀 지표를 모두 다룬다.

    ① 혼동행렬      : TP/FP/FN/TN 네 칸. 모든 분류 지표의 출발점
    ② Accuracy      : 전체 중 맞춘 비율. 불균형 데이터에서 위험한 지표
    ③ Precision/Recall : 양성 예측의 정확성 / 양성 탐지의 완전성
    ④ F1 / F-beta   : 두 값의 (가중) 조화평균
    ⑤ ROC-AUC       : 임계값과 무관한 순위 품질
    ⑥ RUL 오차      : MAE / RMSE / 편향 / 비대칭 점수

외부 의존성은 numpy 하나뿐이고 데이터가 파일 안에 내장되어 있으므로
가중치 파일이나 네트워크 없이 그대로 실행된다.

실행:
    (mobility) d:\\mobility_ai\\ch06> python 10_eval/pdm_metrics.py
"""
from __future__ import annotations

import numpy as np

# =====================================================================
# 1. 분류 평가 데이터
# =====================================================================

#: 차량 20대의 30일 내 고장 여부 정답(1=고장)과 모델이 낸 고장 확률.
#: 양성 7건 / 음성 13건으로 약간 불균형이다. 실제 예지보전 데이터는
#: 양성이 1~5% 뿐인 극단적 불균형이 흔하므로, 이 표본은 설명용으로
#: 완만하게 만든 것이다.
#:
#: 19번째(인덱스 18)가 중요한 사례다. 정답은 정상(0)인데 점수가 0.93 으로
#: 가장 높은 쪽에 있다 — 모델이 확신을 가지고 틀린 FP 다. 이런 건 하나가
#: ROC-AUC 를 눈에 띄게 떨어뜨린다.
PDM_TRUTH = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 1, 1, 1, 1, 1, 1, 0, 1], dtype=np.int64)
PDM_SCORE = np.array([0.04, 0.07, 0.11, 0.14, 0.18, 0.22, 0.27, 0.31, 0.36, 0.41,
                      0.46, 0.58, 0.49, 0.62, 0.68, 0.74, 0.81, 0.88, 0.93, 0.96],
                     dtype=np.float64)

#: 양성(고장 예측) 판정 임계값. 계약서의 risk_level 경계와 같은 역할이다.
PDM_THRESHOLD = 0.50


def confusion_matrix(truth: np.ndarray, pred: np.ndarray) -> dict[str, int]:
    """이진 혼동행렬을 만든다.

    네 칸의 뜻::

        TP (True Positive)  : 고장날 차를 고장난다고 했다 — 맞게 잡음
        FP (False Positive) : 정상 차를 고장난다고 했다 — 헛 정비
        FN (False Negative) : 고장날 차를 정상이라 했다 — 놓침(가장 위험)
        TN (True Negative)  : 정상 차를 정상이라 했다 — 맞게 통과

    Args:
        truth: 0/1 정답 배열.
        pred: 0/1 예측 배열.

    Returns:
        tp/fp/fn/tn 건수.
    """
    truth = np.asarray(truth).astype(np.int64)
    pred = np.asarray(pred).astype(np.int64)
    tp = int(np.sum((truth == 1) & (pred == 1)))
    fp = int(np.sum((truth == 0) & (pred == 1)))
    fn = int(np.sum((truth == 1) & (pred == 0)))
    tn = int(np.sum((truth == 0) & (pred == 0)))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def classification_metrics(truth: np.ndarray, score: np.ndarray,
                           threshold: float = PDM_THRESHOLD) -> dict[str, float]:
    """Accuracy / Precision / Recall / Specificity / F1 을 계산한다.

    정의식::

        Accuracy    = (TP + TN) / (TP + FP + FN + TN)
        Precision   = TP / (TP + FP)      정비하라고 한 것 중 실제 고장 비율
        Recall      = TP / (TP + FN)      실제 고장 중 잡아낸 비율(민감도)
        Specificity = TN / (TN + FP)      실제 정상 중 통과시킨 비율
        F1          = 2PR / (P + R)       정밀도와 재현율의 조화평균

    조화평균을 쓰는 이유: 산술평균이면 Precision 1.0 / Recall 0.0 인
    모델이 0.5 를 받는다. 조화평균은 둘 중 하나가 0 이면 0 이 되므로
    "한쪽만 잘하는" 모델에 점수를 주지 않는다.

    Args:
        truth: 0/1 정답.
        score: 0~1 양성 확률.
        threshold: 양성 판정 임계값.

    Returns:
        혼동행렬 + 지표 dict.
    """
    pred = (np.asarray(score) >= threshold).astype(np.int64)
    cm = confusion_matrix(truth, pred)
    tp, fp, fn, tn = cm["tp"], cm["fp"], cm["fn"], cm["tn"]
    total = tp + fp + fn + tn

    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    out = dict(cm)
    out.update({"threshold": threshold, "accuracy": accuracy, "precision": precision,
                "recall": recall, "specificity": specificity, "f1": f1})
    return out


def fbeta(precision: float, recall: float, beta: float) -> float:
    """F-beta 점수. beta > 1 이면 재현율을 더 중시한다.

    정의식::

        F_beta = (1 + β²) · P · R / (β² · P + R)

    β = 1 이면 F1(동등 가중), β = 2 면 재현율에 4배(=β²) 가중,
    β = 0.5 면 정밀도에 4배 가중이다. 예지보전은 놓치는 비용(FN)이
    헛 정비 비용(FP)보다 크므로 F2 를 보고하는 것이 적절하다.

    Args:
        precision: 정밀도.
        recall: 재현율.
        beta: 가중치.

    Returns:
        F-beta 값.
    """
    b2 = beta * beta
    denominator = b2 * precision + recall
    if denominator <= 0:
        return 0.0
    return (1 + b2) * precision * recall / denominator


# =====================================================================
# 2. ROC-AUC
# =====================================================================

def roc_curve(truth: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """ROC 곡선 좌표(FPR, TPR)를 임계값 내림차순으로 만든다.

    점수를 높은 순으로 정렬하고 하나씩 양성으로 바꿔 가며 누적 TP/FP 를
    세는 방식이다. 임계값을 1.0 에서 0.0 까지 내리는 것과 같다.

    Args:
        truth: 0/1 정답.
        score: 양성 확률.

    Returns:
        (fpr 배열, tpr 배열). 원점 (0,0) 과 (1,1) 을 포함한다.
    """
    truth = np.asarray(truth).astype(np.int64)
    score = np.asarray(score, dtype=np.float64)
    order = np.argsort(-score)            # 점수 높은 순
    truth_sorted = truth[order]

    positives = int(truth_sorted.sum())
    negatives = int(truth_sorted.size - positives)
    # 한쪽 클래스만 있으면 ROC 를 정의할 수 없다 — 대각선을 돌려준다.
    if positives == 0 or negatives == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])

    tps = np.cumsum(truth_sorted)                       # 누적 TP
    fps = np.cumsum(1 - truth_sorted)                   # 누적 FP
    tpr = np.concatenate(([0.0], tps / positives, [1.0]))
    fpr = np.concatenate(([0.0], fps / negatives, [1.0]))
    return fpr, tpr


def roc_auc(truth: np.ndarray, score: np.ndarray) -> float:
    """ROC 곡선 아래 면적을 사다리꼴 적분으로 구한다.

    Args:
        truth: 0/1 정답.
        score: 양성 확률.

    Returns:
        0~1 의 AUC.
    """
    fpr, tpr = roc_curve(truth, score)
    return float(np.trapezoid(tpr, fpr))


def roc_auc_rank(truth: np.ndarray, score: np.ndarray) -> float:
    """순위 합(Mann-Whitney U) 방식으로 ROC-AUC 를 계산한다.

    정의식::

        AUC = P(양성 점수 > 음성 점수) + 0.5 · P(동점)

    즉 "양성 1건과 음성 1건을 무작위로 뽑았을 때 양성 점수가 더 높을 확률"
    이다. 이 해석이 AUC 가 임계값과 무관한 이유를 설명한다 — 점수의 절대값이
    아니라 순위만 쓴다.

    사다리꼴 적분과 같은 값이 나오는지 대조하는 용도로도 쓴다.

    Args:
        truth: 0/1 정답.
        score: 양성 확률.

    Returns:
        0~1 의 AUC.
    """
    truth = np.asarray(truth).astype(np.int64)
    score = np.asarray(score, dtype=np.float64)
    pos = score[truth == 1]
    neg = score[truth == 0]
    if pos.size == 0 or neg.size == 0:
        return 0.5
    wins = 0.0
    for p in pos:
        wins += float(np.sum(p > neg)) + 0.5 * float(np.sum(p == neg))
    return wins / (pos.size * neg.size)


# =====================================================================
# 3. RUL 예측 오차
# =====================================================================

#: 차량 12대의 실제 잔존수명(일)과 모델 예측값.
#: 1번째(12일 → 20일)와 11번째(95일 → 108일)가 과대 예측(늦은 예측)이고,
#: 2번째(18일 → 16일)는 과소 예측(이른 예측)이다.
RUL_TRUTH = np.array([12, 18, 25, 30, 41, 48, 55, 63, 70, 84, 95, 120], dtype=np.float64)
RUL_PRED = np.array([20, 16, 31, 27, 38, 57, 49, 71, 66, 79, 108, 112], dtype=np.float64)


def rul_metrics(truth: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    """RUL 회귀 오차를 계산한다.

    정의식::

        err  = 예측 - 실제        (+ 면 수명을 길게 본 '늦은 예측')
        MAE  = mean(|err|)       평균 절대 오차. 단위가 일(day)로 해석 가능
        RMSE = sqrt(mean(err²))  큰 오차에 더 큰 벌점
        Bias = mean(err)         체계적 편향. 0 에 가까워야 한다
        MAPE = mean(|err| / 실제) 상대 오차. 짧은 수명의 오차를 크게 본다

    ``late_ratio`` 는 "실제보다 늦게(=남은 수명을 과대) 예측한 비율"이다.
    늦은 예측은 고장을 놓치므로 이른 예측보다 위험하다.

    Args:
        truth: 실제 잔존수명(일).
        pred: 예측 잔존수명(일).

    Returns:
        지표 dict.
    """
    truth = np.asarray(truth, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    err = pred - truth                       # + 면 과대(늦은) 예측

    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    bias = float(np.mean(err))
    mape = float(np.mean(np.abs(err) / np.maximum(truth, 1e-9)))
    late = int(np.sum(err > 0))
    return {
        "n": int(truth.size), "mae": mae, "rmse": rmse, "bias": bias,
        "mape": mape, "late": late, "late_ratio": late / truth.size,
        "asym": asymmetric_score(truth, pred),
    }


def asymmetric_score(truth: np.ndarray, pred: np.ndarray,
                     a_late: float = 10.0, a_early: float = 13.0) -> float:
    """NASA C-MAPSS 류의 비대칭 점수. 낮을수록 좋다.

    정의식::

        err > 0 (늦은 예측) : exp(err / a_late)  - 1
        err ≤ 0 (이른 예측) : exp(-err / a_early) - 1
        점수 = Σ 위 벌점

    분모가 작은 쪽(a_late=10)이 벌점이 더 급하게 커진다. 같은 10일 오차가
    늦은 쪽은 exp(1)-1 = 1.72, 이른 쪽은 exp(10/13)-1 = 1.16 이다.
    MAE 나 RMSE 는 두 방향을 똑같이 벌하므로, "늦게 예측해 고장을 놓치는"
    위험을 지표에 반영하려면 이런 비대칭 손실이 필요하다.

    Args:
        truth: 실제 잔존수명.
        pred: 예측 잔존수명.
        a_late: 과대 예측 쪽 완충 계수(작을수록 벌점이 커진다).
        a_early: 과소 예측 쪽 완충 계수.

    Returns:
        점수 합계.
    """
    err = np.asarray(pred, dtype=np.float64) - np.asarray(truth, dtype=np.float64)
    penalty = np.where(err > 0,
                       np.exp(err / a_late) - 1.0,
                       np.exp(-err / a_early) - 1.0)
    return float(np.sum(penalty))


def pct(value: float, digits: int = 1) -> str:
    """0~1 비율을 퍼센트 문자열로 바꾼다."""
    return f"{value * 100:.{digits}f}%"


def main() -> None:
    """예지보전 분류·회귀 지표를 계산해 출력한다."""
    print("=" * 62)
    print(" 예지보전 평가 (분류 지표 + RUL 오차)")
    print("=" * 62)

    clf = classification_metrics(PDM_TRUTH, PDM_SCORE)
    print("\n[1] 고장 분류 혼동행렬 (임계값 %.2f)" % clf["threshold"])
    print("           예측 고장  예측 정상")
    print(f"  실제 고장   TP={clf['tp']:2d}      FN={clf['fn']:2d}")
    print(f"  실제 정상   FP={clf['fp']:2d}      TN={clf['tn']:2d}")

    print("\n[2] 분류 지표")
    print(f"  Accuracy    = {clf['accuracy']:.4f}  ({pct(clf['accuracy'], 2)})")
    print(f"  Precision   = {clf['precision']:.4f}  ({pct(clf['precision'], 2)})")
    print(f"  Recall      = {clf['recall']:.4f}  ({pct(clf['recall'], 2)})")
    print(f"  Specificity = {clf['specificity']:.4f}")
    print(f"  F1          = {clf['f1']:.4f}")
    print(f"  F2 (재현율 가중) = {fbeta(clf['precision'], clf['recall'], 2.0):.4f}")
    print(f"  F0.5(정밀도 가중)= {fbeta(clf['precision'], clf['recall'], 0.5):.4f}")

    print("\n[3] 임계값 민감도")
    print("  thr   Prec    Recall  F1      F2")
    for thr in (0.30, 0.40, 0.50, 0.60, 0.70):
        m = classification_metrics(PDM_TRUTH, PDM_SCORE, thr)
        print("  %.2f  %.4f  %.4f  %.4f  %.4f"
              % (thr, m["precision"], m["recall"], m["f1"],
                 fbeta(m["precision"], m["recall"], 2.0)))

    auc_trap = roc_auc(PDM_TRUTH, PDM_SCORE)
    auc_rank = roc_auc_rank(PDM_TRUTH, PDM_SCORE)
    print("\n[4] ROC-AUC")
    print(f"  사다리꼴 적분 AUC = {auc_trap:.4f}")
    print(f"  순위 합(U) AUC    = {auc_rank:.4f}")
    print(f"  두 방식 차이      = {abs(auc_trap - auc_rank):.6f}")

    rul = rul_metrics(RUL_TRUTH, RUL_PRED)
    print("\n[5] RUL 예측 오차 (차량 %d대)" % rul["n"])
    print(f"  MAE  = {rul['mae']:.2f}일")
    print(f"  RMSE = {rul['rmse']:.2f}일")
    print(f"  Bias = {rul['bias']:+.2f}일 (양수면 수명을 길게 본 것)")
    print(f"  MAPE = {pct(rul['mape'], 2)}")
    print(f"  늦은 예측 {rul['late']}건 / {rul['n']}건 = {pct(rul['late_ratio'], 1)}")
    print(f"  비대칭 점수 = {rul['asym']:.2f} (낮을수록 좋음)")


if __name__ == "__main__":
    main()
