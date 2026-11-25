# -*- coding: utf-8 -*-
"""여러 검출기를 같은 조건으로 벤치마크하는 공통 러너.

설계 원칙
  1) 모델을 '어댑터'로 감싸 같은 인터페이스(load / infer)로 다룬다.
  2) 실제 모델 패키지가 없으면 더미 모델(DummyDetector)로 대체해 동작한다.
     더미 모델은 지정한 GMUL 추정값에 비례한 계산을 수행하므로,
     상대적인 지연 경향을 체험하는 용도로만 쓴다. 절대 성능값이 아니다.
  3) 워밍업 후 반복 측정하고 p50/p95 를 낸다. 평균만 보지 않는다.

사용 예
  python bench_models.py --runs 20 --warmup 5 --threads 1 --out bench.json
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import time
from dataclasses import dataclass, asdict, field

import numpy as np

# ---------------------------------------------------------------------------
# 측정 환경 고정 — 스레드 수는 BLAS 라이브러리가 import 되기 전에 정해야 한다.
# ---------------------------------------------------------------------------


def fix_threads(n: int) -> None:
    """BLAS/OpenMP 스레드 수를 고정한다. 반드시 모델 import 전에 호출한다."""
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = str(n)
    try:
        import torch
        torch.set_num_threads(n)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# 모델 레지스트리 — 비교할 후보와 공개된 규모 추정값
# ---------------------------------------------------------------------------

@dataclass
class ModelSpec:
    """비교 후보 한 건의 메타데이터.

    params_m / size_mb / gmul 은 구조식에서 계산한 추정값이다.
    실제 측정 대상 체크포인트가 있으면 그 값으로 덮어써야 한다.
    """
    key: str
    name: str
    backbone: str
    params_m: float
    size_mb: float
    gmul: float
    note: str = ""
    loader: str = "dummy"          # "dummy" | "ultralytics" | "torchvision" | "hf"
    weight: str = ""


REGISTRY: list[ModelSpec] = [
    ModelSpec("yolo_n", "경량 YOLO-nano", "CSP-nano", 2.6, 5.3, 8.7,
              "1순위 후보", loader="ultralytics", weight="yolo11n.pt"),
    ModelSpec("yolo_s", "경량 YOLO-small", "CSP-small", 9.4, 18.9, 29.4,
              "정확도 여유 확보", loader="ultralytics", weight="yolo11s.pt"),
    ModelSpec("mbnet_det", "MobileNetV3 + SSDLite", "MobileNetV3-L", 3.4, 13.9, 1.6,
              "CPU 효율 최우선", loader="torchvision",
              weight="ssdlite320_mobilenet_v3_large"),
    ModelSpec("detr_r50", "DETR", "ResNet-50", 41.3, 166.0, 86.0,
              "기준선 Transformer 검출기", loader="hf", weight="facebook/detr-resnet-50"),
    ModelSpec("ddetr_r50", "Deformable DETR", "ResNet-50", 40.1, 161.0, 173.0,
              "다중 스케일 희소 어텐션", loader="hf",
              weight="SenseTime/deformable-detr"),
    ModelSpec("dino_r50", "DINO", "ResNet-50", 47.0, 189.0, 279.0,
              "정확도 우선", loader="dummy"),
    ModelSpec("yolos_s", "YOLOS-small", "ViT-Small", 30.7, 123.0, 194.0,
              "순수 ViT 검출기", loader="hf", weight="hustvl/yolos-small"),
    ModelSpec("swin_dino", "Swin-T + DINO", "Swin-Tiny", 48.0, 193.0, 312.0,
              "계층형 Transformer 백본", loader="dummy"),
    ModelSpec("vitdet_b", "ViTDet (ViT-Base)", "ViT-Base", 111.0, 445.0, 741.0,
              "범용 검출 프레임워크", loader="dummy"),
]


# ---------------------------------------------------------------------------
# 어댑터 — 실제 모델이 없으면 더미로 대체
# ---------------------------------------------------------------------------

class DummyDetector:
    """모델 패키지/가중치가 없을 때 쓰는 대체 구현.

    spec.gmul 에 비례하는 행렬곱을 수행해 '연산량이 크면 느리다'는 경향만 만든다.
    여기서 나온 지연값은 교재의 표를 재현하는 수치가 아니며, 교육용 자리표다.
    """

    def __init__(self, spec: ModelSpec, size: int = 640) -> None:
        self.spec = spec
        self.size = size
        # 연산량 추정값을 행렬 변으로 환산한다. 2*n^3 FLOP 기준.
        n = int((spec.gmul * 1e9 / 2.0) ** (1.0 / 3.0))
        self.n = max(32, min(n, 512))
        rng = np.random.default_rng(1234)
        self.a = rng.standard_normal((self.n, self.n), dtype=np.float32)
        self.b = rng.standard_normal((self.n, self.n), dtype=np.float32)
        self.reps = max(1, int(spec.gmul * 1e9 / (2.0 * self.n ** 3)))

    def infer(self, img: np.ndarray) -> dict:
        acc = 0.0
        for _ in range(self.reps):
            acc += float((self.a @ self.b)[0, 0])
        # 결정적인 가짜 검출 결과. 번호판 1개를 반환한다.
        return {"boxes": [[412, 288, 596, 342]], "scores": [0.90], "sig": acc}


def load_model(spec: ModelSpec, size: int):
    """실제 모델을 불러오고, 실패하면 더미로 대체한다. (모델, 실제여부) 반환."""
    if spec.loader == "ultralytics":
        try:
            from ultralytics import YOLO                        # noqa: PLC0415
            m = YOLO(spec.weight)

            class _Real:
                def infer(self, img):
                    r = m.predict(img, imgsz=size, verbose=False, device="cpu")[0]
                    b = r.boxes
                    return {"boxes": b.xyxy.tolist() if b is not None else [],
                            "scores": b.conf.tolist() if b is not None else []}
            return _Real(), True
        except Exception as exc:
            print(f"  [알림] {spec.name}: ultralytics 사용 불가 → 더미 사용 ({exc.__class__.__name__})")
    elif spec.loader == "torchvision":
        try:
            import torch                                        # noqa: PLC0415
            import torchvision                                   # noqa: PLC0415
            fn = getattr(torchvision.models.detection, spec.weight)
            m = fn(weights=None).eval()

            class _Real:
                def infer(self, img):
                    t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
                    with torch.no_grad():
                        r = m([t])[0]
                    return {"boxes": r["boxes"].tolist(), "scores": r["scores"].tolist()}
            return _Real(), True
        except Exception as exc:
            print(f"  [알림] {spec.name}: torchvision 사용 불가 → 더미 사용 ({exc.__class__.__name__})")
    elif spec.loader == "hf":
        try:
            import torch                                        # noqa: PLC0415
            from transformers import AutoImageProcessor, AutoModelForObjectDetection
            proc = AutoImageProcessor.from_pretrained(spec.weight)
            m = AutoModelForObjectDetection.from_pretrained(spec.weight).eval()

            class _Real:
                def infer(self, img):
                    inp = proc(images=img, return_tensors="pt")
                    with torch.no_grad():
                        out = m(**inp)
                    res = proc.post_process_object_detection(
                        out, threshold=0.25,
                        target_sizes=[(img.shape[0], img.shape[1])])[0]
                    return {"boxes": res["boxes"].tolist(),
                            "scores": res["scores"].tolist()}
            return _Real(), True
        except Exception as exc:
            print(f"  [알림] {spec.name}: transformers 사용 불가 → 더미 사용 ({exc.__class__.__name__})")
    return DummyDetector(spec, size), False


# ---------------------------------------------------------------------------
# 측정
# ---------------------------------------------------------------------------

@dataclass
class BenchResult:
    key: str
    name: str
    backbone: str
    params_m: float
    size_mb: float
    gmul: float
    real_model: bool
    p50_ms: float
    p95_ms: float
    mean_ms: float
    min_ms: float
    max_ms: float
    peak_mem_mb: float
    runs: int
    note: str = ""
    samples_ms: list[float] = field(default_factory=list)


def percentile(values: list[float], q: float) -> float:
    """선형 보간 백분위수. numpy 없이도 같은 결과가 나오도록 직접 구현한다."""
    s = sorted(values)
    if not s:
        return 0.0
    if len(s) == 1:
        return s[0]
    pos = (len(s) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def measure_memory_mb() -> float:
    """현재 프로세스의 상주 메모리(MB). psutil 이 없으면 0 을 반환한다."""
    try:
        import psutil                                           # noqa: PLC0415
        return psutil.Process().memory_info().rss / 1024 / 1024
    except Exception:
        try:
            import resource                                     # noqa: PLC0415
            ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            return ru / 1024                 # 리눅스는 KB 단위
        except Exception:
            return 0.0


def make_input(size: int, seed: int = 777) -> np.ndarray:
    """결정적인 테스트 입력. 같은 시드면 항상 같은 이미지가 나온다."""
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
    # 번호판 모양의 밝은 사각형을 심어 둔다(결정적).
    img[288:342, 412:596] = 230
    return img


def bench_one(spec: ModelSpec, img: np.ndarray, runs: int,
              warmup: int, size: int) -> BenchResult:
    print(f"[{spec.key}] {spec.name} 측정 중 ...")
    model, real = load_model(spec, size)

    for _ in range(warmup):                     # 워밍업: 캐시/JIT 영향 제거
        model.infer(img)

    mem0 = measure_memory_mb()
    samples: list[float] = []
    peak = mem0
    for _ in range(runs):
        t0 = time.perf_counter()
        model.infer(img)
        samples.append((time.perf_counter() - t0) * 1000.0)
        peak = max(peak, measure_memory_mb())

    return BenchResult(
        key=spec.key, name=spec.name, backbone=spec.backbone,
        params_m=spec.params_m, size_mb=spec.size_mb, gmul=spec.gmul,
        real_model=real,
        p50_ms=round(percentile(samples, 0.50), 2),
        p95_ms=round(percentile(samples, 0.95), 2),
        mean_ms=round(sum(samples) / len(samples), 2),
        min_ms=round(min(samples), 2), max_ms=round(max(samples), 2),
        peak_mem_mb=round(peak, 1), runs=runs, note=spec.note,
        samples_ms=[round(s, 3) for s in samples],
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="검출기 공통 벤치마크 러너")
    ap.add_argument("--runs", type=int, default=20, help="측정 반복 횟수")
    ap.add_argument("--warmup", type=int, default=5, help="워밍업 횟수")
    ap.add_argument("--threads", type=int, default=1, help="CPU 스레드 수")
    ap.add_argument("--size", type=int, default=640, help="입력 한 변 길이")
    ap.add_argument("--only", default="", help="쉼표로 구분한 모델 key 만 측정")
    ap.add_argument("--out", default="bench_result.json", help="결과 JSON 경로")
    args = ap.parse_args()

    fix_threads(args.threads)

    keys = {k.strip() for k in args.only.split(",") if k.strip()}
    specs = [s for s in REGISTRY if not keys or s.key in keys]

    print("=" * 78)
    print("검출기 공통 벤치마크 — 모든 후보에 같은 조건을 적용한다")
    print("=" * 78)
    print(f"  플랫폼      : {platform.system()} {platform.machine()}")
    print(f"  파이썬      : {platform.python_version()}")
    print(f"  CPU 스레드  : {args.threads}")
    print(f"  입력 해상도 : {args.size}x{args.size}")
    print(f"  워밍업/측정 : {args.warmup} / {args.runs} 회")
    print(f"  측정 대상   : {len(specs)} 개")
    print()

    img = make_input(args.size)
    results = [bench_one(s, img, args.runs, args.warmup, args.size) for s in specs]

    print()
    print(f"{'모델':<22} {'백본':<14} {'실제':>4} {'p50(ms)':>9} "
          f"{'p95(ms)':>9} {'메모리(MB)':>11}")
    print("-" * 78)
    for r in results:
        print(f"{r.name:<22} {r.backbone:<14} {'O' if r.real_model else 'X':>4} "
              f"{r.p50_ms:>9.2f} {r.p95_ms:>9.2f} {r.peak_mem_mb:>11.1f}")
    print("-" * 78)

    payload = {
        "env": {"platform": platform.system(), "machine": platform.machine(),
                "python": platform.python_version(), "threads": args.threads,
                "input_size": args.size, "warmup": args.warmup, "runs": args.runs},
        "results": [asdict(r) for r in results],
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"결과를 {args.out} 에 저장했다.")
    real_cnt = sum(1 for r in results if r.real_model)
    if real_cnt < len(results):
        print()
        print(f"주의: {len(results) - real_cnt} 개 모델이 더미로 측정되었다.")
        print("      더미 지연값은 경향 확인용이며 실제 모델의 성능이 아니다.")
        print("      실제 비교표를 만들려면 해당 패키지와 가중치를 설치한 뒤 다시 실행한다.")


if __name__ == "__main__":
    main()
