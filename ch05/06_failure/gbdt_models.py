# -*- coding: utf-8 -*-
"""XGBoost / LightGBM 학습과 비교. 미설치 라이브러리는 건너뛴다.

실행:  python 06_failure/gbdt_models.py
입력:  data/labeled_dataset.csv
"""
from __future__ import annotations

import pathlib
import sys
import time

import pandas as pd

# 같은 디렉터리의 random_forest.py 에서 특징 생성과 분할 함수를 재사용한다.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from random_forest import (BASE_FEATURES, add_rolling_features,   # noqa: E402
                           split_by_vehicle)

DATA = pathlib.Path("data") / "labeled_dataset.csv"


def available() -> dict[str, bool]:
    """설치 여부를 먼저 확인한다. 하나도 없으면 안내만 하고 끝낸다."""
    have = {}
    for name in ("xgboost", "lightgbm", "sklearn"):
        try:
            __import__(name)
            have[name] = True
        except ImportError:
            have[name] = False
    return have


def train_xgboost(xtr, ytr, xte, yte):
    """XGBoost 학습. scale_pos_weight 로 클래스 불균형을 보정한다."""
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score

    # 양성이 희소하므로 음성/양성 비율을 가중치로 넣는다.
    spw = float((ytr == 0).sum()) / max(float((ytr == 1).sum()), 1.0)
    model = xgb.XGBClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.06,
        subsample=0.85, colsample_bytree=0.85,
        reg_lambda=1.5, scale_pos_weight=spw,
        eval_metric="aucpr",            # 불균형에서는 AUC-PR 이 더 민감하다
        early_stopping_rounds=40,
        n_jobs=4, tree_method="hist", random_state=42)
    t0 = time.perf_counter()
    model.fit(xtr, ytr, eval_set=[(xte, yte)], verbose=False)
    fit_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    prob = model.predict_proba(xte)[:, 1]
    infer_ms = (time.perf_counter() - t0) * 1000.0 / len(xte)
    return {"name": "XGBoost", "auc": roc_auc_score(yte, prob),
            "fit_s": fit_s, "infer_ms": infer_ms,
            "best_iter": int(getattr(model, "best_iteration", 0) or 0),
            "model": model, "prob": prob}


def train_lightgbm(xtr, ytr, xte, yte):
    """LightGBM 학습. 리프 기반 성장이라 num_leaves 가 핵심 하이퍼파라미터다."""
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score

    model = lgb.LGBMClassifier(
        n_estimators=600, num_leaves=31, learning_rate=0.05,
        min_child_samples=20, subsample=0.85, subsample_freq=1,
        colsample_bytree=0.85, reg_lambda=1.5,
        class_weight="balanced",        # XGBoost 의 scale_pos_weight 와 같은 역할
        n_jobs=4, random_state=42, verbosity=-1)
    t0 = time.perf_counter()
    model.fit(xtr, ytr, eval_set=[(xte, yte)], eval_metric="auc",
              callbacks=[lgb.early_stopping(40, verbose=False)])
    fit_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    prob = model.predict_proba(xte)[:, 1]
    infer_ms = (time.perf_counter() - t0) * 1000.0 / len(xte)
    return {"name": "LightGBM", "auc": roc_auc_score(yte, prob),
            "fit_s": fit_s, "infer_ms": infer_ms,
            "best_iter": int(model.best_iteration_ or 0),
            "model": model, "prob": prob}


def main() -> None:
    have = available()
    if not have["sklearn"]:
        print("[안내] scikit-learn 이 필요하다. pip install scikit-learn")
        return
    if not (have["xgboost"] or have["lightgbm"]):
        print("[안내] XGBoost / LightGBM 이 모두 설치되어 있지 않다.")
        print("       pip install xgboost lightgbm")
        print("       두 라이브러리는 CPU 환경에서도 잘 동작하며, 표 형태")
        print("       센서 데이터에서는 보통 RandomForest 보다 조금 더 높은")
        print("       성능을 보인다. 설치 후 다시 실행해 비교하면 된다.")
        return
    if not DATA.exists():
        print(f"[안내] {DATA} 가 없다. 먼저 python 06_failure/make_labels.py")
        return

    df = add_rolling_features(pd.read_csv(DATA))
    feats = BASE_FEATURES + [c for c in df.columns
                             if c.endswith(("_ma7", "_sd7", "_slope"))]
    train, test = split_by_vehicle(df)
    xtr, ytr = train[feats], train["will_fail_30"]
    xte, yte = test[feats], test["will_fail_30"]
    print(f"특징 {len(feats)}개 / 학습 {len(xtr)}행 / 평가 {len(xte)}행")
    print(f"양성 비율: 학습 {ytr.mean():.3%} / 평가 {yte.mean():.3%}\n")

    results = []
    if have["xgboost"]:
        results.append(train_xgboost(xtr, ytr, xte, yte))
    else:
        print("[건너뜀] xgboost 미설치")
    if have["lightgbm"]:
        results.append(train_lightgbm(xtr, ytr, xte, yte))
    else:
        print("[건너뜀] lightgbm 미설치")

    hdr = (f"{'모델':<12s}{'ROC-AUC':>9s}{'학습(초)':>10s}"
           f"{'추론(ms/건)':>12s}{'최적 트리':>10s}")
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        print(f"{r['name']:<12s}{r['auc']:9.4f}{r['fit_s']:10.2f}"
              f"{r['infer_ms']:12.4f}{r['best_iter']:10d}")

    # 두 모델 확률을 단순 평균한 앙상블도 함께 본다.
    if len(results) == 2:
        from sklearn.metrics import roc_auc_score
        ens = (results[0]["prob"] + results[1]["prob"]) / 2.0
        print(f"\n단순 평균 앙상블 ROC-AUC = {roc_auc_score(yte, ens):.4f}")
        print("두 모델의 오류 패턴이 다를 때만 앙상블이 이득이 된다.")

    print("\n[특징 중요도 상위 10개]")
    for r in results:
        imp = pd.Series(r["model"].feature_importances_,
                        index=feats).sort_values(ascending=False)
        print(f"  {r['name']}: " + ", ".join(imp.head(10).index))


if __name__ == "__main__":
    main()
