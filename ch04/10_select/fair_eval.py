# -*- coding: utf-8 -*-
"""동일 조건 평가 러너 — 공정 비교를 위한 고정 장치를 모아 둔 스크립트.

이 파일이 보장하는 것
  1) 시드 고정: 파이썬/numpy/torch(있으면) 난수를 모두 같은 값으로 맞춘다.
  2) 스레드 고정: BLAS/OpenMP 스레드 수를 import 전에 환경변수로 박는다.
  3) 워밍업: 첫 호출의 지연(캐시 적재, 그래프 최적화)을 측정에서 제외한다.
  4) 반복 측정: 평균이 아니라 p50/p95 를 쓴다.
  5) 동일 전처리/동일 임계값: experiment_config.yaml 의 값을 모든 모델에 그대로 적용한다.

가중치나 데이터셋이 없어도 합성 데이터와 더미 모델로 전 과정을 실행해 볼 수 있다.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import platform
import random
import statistics
import sys
import time

DEFAULT_CONFIG = {
    "seed": 20261024,
    "threads": 1,
    "input_size": 640,
    "warmup": 5,
    "runs": 30,
    "letterbox": True,
    "normalize": "div255",
    "conf_threshold": 0.25,
    "iou_threshold": 0.50,
    "match_iou": 0.50,
    "models": ["yolo_n", "yolo_s", "mbnet_det", "detr_r50", "ddetr_r50"],
}


# ---------------------------------------------------------------------------
# 1) 환경 고정
# ---------------------------------------------------------------------------

def fix_threads(n: int) -> None:
    """스레드 수 고정. numpy/torch 를 import 하기 전에 호출해야 효과가 있다."""
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = str(n)


def fix_seed(seed: int) -> None:
    """모든 난수 생성기의 시드를 고정한다."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def load_config(path: str) -> dict:
    """YAML 설정을 읽는다. PyYAML 이 없거나 파일이 없으면 기본값을 쓴다."""
    p = pathlib.Path(path)
    if not p.exists():
        print(f"[알림] 설정 {path} 가 없다. 내장 기본값을 사용한다.", file=sys.stderr)
        return dict(DEFAULT_CONFIG)
    try:
        import yaml
        cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except ModuleNotFoundError:
        print("[알림] PyYAML 이 없다. 내장 기본값을 사용한다.", file=sys.stderr)
        return dict(DEFAULT_CONFIG)
    merged = dict(DEFAULT_CONFIG)
    for section in ("runtime", "preprocess", "threshold", "models"):
        v = cfg.get(section)
        if isinstance(v, dict):
            merged.update(v)
        elif isinstance(v, list):
            merged["models"] = v
    for k, v in cfg.items():
        if not isinstance(v, (dict, list)):
            merged[k] = v
    return merged


# ---------------------------------------------------------------------------
# 2) 동일 전처리 — 모든 모델이 똑같은 함수를 쓴다
# ---------------------------------------------------------------------------

def letterbox(img, size: int, pad_value: int = 114):
    """가로세로 비율을 유지한 채 size x size 로 맞춘다.

    번호판은 가로로 긴 객체라 단순 리사이즈로 비율이 깨지면 문자 형태가 왜곡된다.
    모든 후보 모델에 같은 레터박스를 적용해야 비교가 공정해진다.
    반환: (패딩된 이미지, 배율, (왼쪽 패딩, 위쪽 패딩))
    """
    import numpy as np
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = max(1, int(round(h * r))), max(1, int(round(w * r)))
    ys = (np.arange(nh) / r).astype(int).clip(0, h - 1)
    xs = (np.arange(nw) / r).astype(int).clip(0, w - 1)
    resized = img[ys][:, xs]
    out = np.full((size, size, img.shape[2]), pad_value, dtype=img.dtype)
    top, left = (size - nh) // 2, (size - nw) // 2
    out[top:top + nh, left:left + nw] = resized
    return out, r, (left, top)


def normalize(img, mode: str):
    import numpy as np
    x = img.astype(np.float32)
    if mode == "div255":
        return x / 255.0
    if mode == "imagenet":
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        return (x / 255.0 - mean) / std
    return x


# ---------------------------------------------------------------------------
# 3) 동일 임계값 평가 — IoU 매칭과 지표 산출
# ---------------------------------------------------------------------------

def iou(a: list[float], b: list[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def evaluate(preds: list[dict], gts: list[dict],
             conf_th: float, match_iou: float) -> dict:
    """검출 재현율/정밀도와 번호 문자열 완전 일치율을 한 번에 계산한다.

    preds[i] = {"boxes": [[x1,y1,x2,y2],...], "scores": [...], "texts": [...]}
    gts[i]   = {"boxes": [[...]], "texts": ["12가3456", ...]}
    """
    tp = fp = fn = 0
    exact = total_text = 0
    for pr, gt in zip(preds, gts):
        keep = [i for i, s in enumerate(pr["scores"]) if s >= conf_th]
        pboxes = [pr["boxes"][i] for i in keep]
        ptexts = [pr.get("texts", [None] * len(pr["boxes"]))[i] for i in keep]
        used: set[int] = set()
        for gi, gbox in enumerate(gt["boxes"]):
            best, bi = 0.0, -1
            for pi, pbox in enumerate(pboxes):
                if pi in used:
                    continue
                v = iou(pbox, gbox)
                if v > best:
                    best, bi = v, pi
            total_text += 1
            if best >= match_iou and bi >= 0:
                used.add(bi)
                tp += 1
                if ptexts[bi] is not None and ptexts[bi] == gt["texts"][gi]:
                    exact += 1
            else:
                fn += 1
        fp += len(pboxes) - len(used)
    recall = tp / (tp + fn) if tp + fn else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "detect_recall": round(recall, 4),
        "precision": round(precision, 4),
        "f1": round(2 * recall * precision / (recall + precision), 4)
        if recall + precision else 0.0,
        "exact_match_rate": round(exact / total_text, 4) if total_text else 0.0,
    }


# ---------------------------------------------------------------------------
# 4) 지연 측정
# ---------------------------------------------------------------------------

def percentile(values: list[float], q: float) -> float:
    s = sorted(values)
    if not s:
        return 0.0
    if len(s) == 1:
        return s[0]
    pos = (len(s) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def measure_latency(fn, img, warmup: int, runs: int) -> dict:
    """워밍업 후 runs 회 반복 측정하고 분위수를 돌려준다."""
    for _ in range(warmup):
        fn(img)
    samples: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn(img)
        samples.append((time.perf_counter() - t0) * 1000.0)
    return {
        "p50_ms": round(percentile(samples, 0.50), 2),
        "p90_ms": round(percentile(samples, 0.90), 2),
        "p95_ms": round(percentile(samples, 0.95), 2),
        "p99_ms": round(percentile(samples, 0.99), 2),
        "mean_ms": round(statistics.fmean(samples), 2),
        "stdev_ms": round(statistics.pstdev(samples), 2),
        "min_ms": round(min(samples), 2),
        "max_ms": round(max(samples), 2),
        "runs": runs,
    }


# ---------------------------------------------------------------------------
# 5) 더미 모델 — 데이터셋/가중치 없이도 전 과정을 돌려 보기 위한 자리표
# ---------------------------------------------------------------------------

DUMMY_COST = {"yolo_n": 60, "yolo_s": 160, "mbnet_det": 40,
              "detr_r50": 900, "ddetr_r50": 1500, "dino_r50": 2200,
              "yolos_s": 1500, "swin_dino": 2600, "vitdet_b": 5000}

DUMMY_QUALITY = {"yolo_n": (0.988, 0.958), "yolo_s": (0.993, 0.964),
                 "mbnet_det": (0.961, 0.931), "detr_r50": (0.972, 0.941),
                 "ddetr_r50": (0.986, 0.953), "dino_r50": (0.991, 0.959),
                 "yolos_s": (0.948, 0.918), "swin_dino": (0.992, 0.961),
                 "vitdet_b": (0.989, 0.955)}

# 실제 측정이 불가능한 환경에서 쓰는 메모리 자리표(MB).
DUMMY_MEM = {"yolo_n": 210, "yolo_s": 268, "mbnet_det": 186,
             "detr_r50": 742, "ddetr_r50": 864, "dino_r50": 1018,
             "yolos_s": 906, "swin_dino": 1264, "vitdet_b": 2186}


def measure_memory_mb() -> float:
    """현재 프로세스의 상주 메모리(MB). 측정 수단이 없으면 0 을 반환한다."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1024 / 1024
    except Exception:
        try:
            import resource
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        except Exception:
            return 0.0


def make_dummy(key: str):
    """key 별 연산량 자리표에 비례한 계산을 수행하는 더미 추론 함수."""
    import numpy as np
    n = 128
    rng = np.random.default_rng(hash(key) % (2 ** 32))
    a = rng.standard_normal((n, n), dtype=np.float32)
    b = rng.standard_normal((n, n), dtype=np.float32)
    reps = max(1, DUMMY_COST.get(key, 100) // 4)

    def infer(_img):
        acc = 0.0
        for _ in range(reps):
            acc += float((a @ b)[0, 0])
        return {"boxes": [[412, 288, 596, 342]], "scores": [0.9],
                "texts": ["12가3456"], "sig": acc}
    return infer


def make_dataset(n_images: int, size: int, seed: int):
    """합성 테스트셋. 번호판 1개가 들어간 결정적 이미지와 정답을 만든다."""
    import numpy as np
    rng = np.random.default_rng(seed)
    plates = ["12가3456", "서울 34나5678", "87로2345"]
    imgs, gts = [], []
    for i in range(n_images):
        img = rng.integers(40, 200, size=(size, size, 3), dtype=np.uint8)
        x1 = 380 + (i * 7) % 60
        y1 = 270 + (i * 5) % 40
        x2, y2 = x1 + 184, y1 + 54
        img[y1:y2, x1:x2] = 235
        imgs.append(img)
        gts.append({"boxes": [[float(x1), float(y1), float(x2), float(y2)]],
                    "texts": [plates[i % len(plates)]]})
    return imgs, gts


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="동일 조건 평가 러너")
    ap.add_argument("--config", default="experiment_config.yaml")
    ap.add_argument("--images", type=int, default=24, help="합성 테스트 이미지 수")
    ap.add_argument("--out", default="fair_eval_result.json")
    args = ap.parse_args()

    cfg = load_config(args.config)
    fix_threads(int(cfg["threads"]))        # numpy import 전에 고정
    fix_seed(int(cfg["seed"]))

    import numpy as np                      # 스레드 고정 후 import

    print("=" * 78)
    print("동일 조건 평가 러너 — 모든 후보에 같은 고정 조건을 적용한다")
    print("=" * 78)
    print(f"  시드         : {cfg['seed']}")
    print(f"  스레드       : {cfg['threads']}")
    print(f"  입력 해상도  : {cfg['input_size']}")
    print(f"  전처리       : letterbox={cfg['letterbox']} normalize={cfg['normalize']}")
    print(f"  신뢰도 임계값: {cfg['conf_threshold']}  NMS IoU: {cfg['iou_threshold']}")
    print(f"  매칭 IoU     : {cfg['match_iou']}")
    print(f"  워밍업/측정  : {cfg['warmup']} / {cfg['runs']} 회")
    print(f"  플랫폼       : {platform.system()} {platform.machine()} "
          f"/ 파이썬 {platform.python_version()}")
    print()

    size = int(cfg["input_size"])
    imgs, gts = make_dataset(args.images, size, int(cfg["seed"]))
    print(f"[1] 테스트셋 생성: {len(imgs)} 장 (합성, 시드 고정)")

    # 동일 전처리를 한 번만 수행해 모든 모델에 같은 텐서를 넘긴다.
    pre = []
    for im in imgs:
        x, r, pad = (letterbox(im, size) if cfg["letterbox"] else (im, 1.0, (0, 0)))
        pre.append(normalize(x, str(cfg["normalize"])))
    print(f"[2] 동일 전처리 적용 완료: shape={pre[0].shape} "
          f"dtype={pre[0].dtype} 값범위=[{pre[0].min():.3f}, {pre[0].max():.3f}]")
    print()

    results = []
    for key in cfg["models"]:
        infer = make_dummy(key)
        lat = measure_latency(infer, pre[0], int(cfg["warmup"]), int(cfg["runs"]))
        preds = [infer(x) for x in pre]
        # 더미는 품질을 모르므로 자리표 품질 지표를 쓴다.
        dr, em = DUMMY_QUALITY.get(key, (0.0, 0.0))
        metric = evaluate(preds, gts, float(cfg["conf_threshold"]),
                          float(cfg["match_iou"]))
        metric["detect_recall"] = dr
        metric["exact_match_rate"] = em
        mem = round(max(measure_memory_mb(), float(DUMMY_MEM.get(key, 0))), 1)
        row = {"key": key, "dummy": True, "memory_mb": mem, **lat, **metric}
        results.append(row)
        print(f"[{key:<10}] p50={lat['p50_ms']:>8.2f}ms  p95={lat['p95_ms']:>8.2f}ms  "
              f"표준편차={lat['stdev_ms']:>6.2f}  메모리={mem:>7.1f}MB  "
              f"재현율={dr:.3f}  일치율={em:.3f}")

    print()
    print("[3] p95/p50 비율 — 지연의 흔들림 정도")
    for r in results:
        print(f"  {r['key']:<10} p95/p50 = {r['p95_ms'] / r['p50_ms']:.2f} "
              f"(1.0 에 가까울수록 안정적)")

    payload = {"config": cfg,
               "env": {"platform": platform.system(), "machine": platform.machine(),
                       "python": platform.python_version()},
               "note": "더미 모델 측정값이다. 실제 모델 성능이 아니다.",
               "results": results}
    pathlib.Path(args.out).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print()
    print(f"[4] 결과를 {args.out} 에 저장했다. score_models.py 가 이 파일을 읽는다.")
    print()
    print("주의: 위 수치는 더미 모델로 만든 자리표다. 실제 비교를 하려면")
    print("      같은 설정 파일을 그대로 두고 모델 어댑터만 실제 구현으로 바꾼다.")


if __name__ == "__main__":
    main()
