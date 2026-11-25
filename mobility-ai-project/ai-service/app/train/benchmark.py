"""CPU 지연 / 메모리 / 정확도 벤치마크.

계약서 ``POST /api/v1/models/benchmark`` 와 DB ``model_benchmark`` 테이블을 채운다.

측정 방법론 — 왜 이렇게 하는가:

    1. **워밍업(warmup)을 버린다.** 첫 추론은 지연 로딩, 메모리 할당, CPU 캐시
       미스로 평상시보다 5~20배 느리다. 이를 포함하면 p95 가 완전히 왜곡된다.
       운영에서 중요한 것은 "정상 상태(steady state)" 지연이다.

    2. **평균이 아니라 p50/p95 를 쓴다.** 평균은 이상치 하나에 끌려간다.
       게이트 시스템의 SLA 는 "95% 의 차량이 N초 내 통과" 로 정의되므로
       p95 가 운영상 가장 의미 있는 지표다.

    3. **detect_recall 과 exact_match_rate 를 분리한다.** 번호판을 찾았는데
       못 읽은 것과 애초에 못 찾은 것은 전혀 다른 문제다(카메라 각도 vs 해상도).
       두 지표를 분리해야 어디를 개선할지 알 수 있다.

    4. **메모리는 RSS 증가분으로 측정한다.** 절대 RSS 는 파이썬 인터프리터와
       다른 모듈을 포함해 모델 크기를 알 수 없다. 모델 로딩 전후 차이를 본다.

torch 없이도 완전히 동작한다(폴백 경로를 벤치마크한다).
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from app.core.logging import get_logger

logger = get_logger(__name__)

#: 측정 전에 버릴 워밍업 반복 수.
DEFAULT_WARMUP: int = 3

#: 기본 벤치마크 샘플 수. 20 이면 p95 가 "두 번째로 느린 값" 이 되어
#: 통계적 의미가 생기는 최소 수준이다.
DEFAULT_SAMPLES: int = 20


@dataclass(slots=True)
class BenchmarkResult:
    """벤치마크 결과 한 건(계약서 스키마와 1:1).

    Attributes:
        model_name: 모델 조합 이름(예 ``"yolo-fallback + crnn-fallback"``).
        task: 작업 종류.
        detect_recall: 번호판 검출 재현율(IoU 0.4 기준).
        exact_match_rate: 번호판 문자열 완전 일치율.
        latency_p50_ms: 지연 중앙값.
        latency_p95_ms: 지연 95분위.
        memory_mb: 모델 로딩으로 증가한 메모리(MB).
        model_size_mb: 가중치 파일 크기(MB). 폴백은 0.
        quantized: 양자화 모델 여부.
        detail: 부가 지표(문자 정확도, 형식 유효율 등).
    """

    model_name: str
    task: str = "detect+ocr"
    detect_recall: float = 0.0
    exact_match_rate: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    memory_mb: float = 0.0
    model_size_mb: float = 0.0
    quantized: bool = False
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """계약서 벤치마크 응답 스키마로 직렬화한다."""
        return {
            "model_name": self.model_name,
            "task": self.task,
            "detect_recall": round(float(self.detect_recall), 4),
            "exact_match_rate": round(float(self.exact_match_rate), 4),
            "latency_p50_ms": round(float(self.latency_p50_ms), 2),
            "latency_p95_ms": round(float(self.latency_p95_ms), 2),
            "memory_mb": round(float(self.memory_mb), 2),
            "model_size_mb": round(float(self.model_size_mb), 2),
            "quantized": bool(self.quantized),
            "detail": self.detail,
        }


def current_rss_mb() -> float:
    """현재 프로세스의 RSS(실제 사용 메모리)를 MB 로 반환한다.

    측정 수단을 세 단계로 시도한다:
        1. ``/proc/self/statm``(리눅스, 의존성 없음, 가장 정확)
        2. ``resource.getrusage``(유닉스 계열, 플랫폼별 단위 주의)
        3. 실패 시 0.0

    psutil 을 쓰지 않는 이유: requirements 에 없는 의존성이고,
    리눅스 컨테이너 환경에서는 ``/proc`` 으로 충분하다.

    Returns:
        RSS(MB). 측정 불가 시 0.0.
    """
    try:
        statm = Path("/proc/self/statm")
        if statm.is_file():
            fields = statm.read_text(encoding="utf-8").split()
            if len(fields) >= 2:
                pages = int(fields[1])
                page_size = os.sysconf("SC_PAGE_SIZE")
                return float(pages * page_size) / (1024.0 * 1024.0)
    except (OSError, ValueError, IndexError):
        pass

    try:
        import resource  # noqa: PLC0415

        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # 리눅스는 KB, macOS 는 bytes 를 반환한다. 1e6 을 넘으면 bytes 로 본다.
        return float(usage) / (1024.0 * 1024.0) if usage > 1e6 else float(usage) / 1024.0
    except (ImportError, AttributeError, OSError):
        return 0.0


def weight_size_mb(paths: list[str | None]) -> float:
    """가중치 파일들의 총 크기를 MB 로 반환한다.

    Args:
        paths: 파일 경로 리스트(None 과 없는 파일은 무시).

    Returns:
        총 크기(MB). 가중치가 없으면 0.0.
    """
    total = 0
    for path in paths or []:
        if not path:
            continue
        try:
            file_path = Path(str(path))
            if file_path.is_file():
                total += file_path.stat().st_size
        except OSError:
            continue
    return float(total) / (1024.0 * 1024.0)


def _iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    """두 박스의 IoU 를 계산한다.

    Args:
        box_a: ``(x1, y1, x2, y2)``.
        box_b: ``(x1, y1, x2, y2)``.

    Returns:
        0~1 IoU.
    """
    ax1, ay1, ax2, ay2 = (float(v) for v in box_a)
    bx1, by1, bx2, by2 = (float(v) for v in box_b)
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def _percentile(values: list[float], q: float) -> float:
    """백분위수를 계산한다(numpy 의 선형 보간 방식).

    Args:
        values: 값 리스트.
        q: 0~100 백분위.

    Returns:
        백분위 값. 빈 리스트면 0.0.
    """
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), float(q)))


def build_eval_set(
    samples: int = DEFAULT_SAMPLES, *, seed: int = 20261024, augment: bool = False
) -> list[tuple[np.ndarray, str, tuple[int, int, int, int]]]:
    """합성 평가 데이터셋을 만든다.

    왜 합성 데이터로 벤치마크하는가?
        실제 번호판 사진은 개인정보이므로 교재/저장소에 포함할 수 없다.
        합성 데이터는 **정답 라벨이 100% 정확**하고 시드로 재현 가능하므로
        "같은 환경에서 같은 숫자가 나온다" 는 벤치마크의 기본 요건을 만족한다.
        단, 절대 정확도는 실제 데이터보다 높게 나오므로 **모델 간 상대 비교**
        용도로 해석해야 한다 — 이 한계를 교재에 명시해야 한다.

    Args:
        samples: 생성할 샘플 수.
        seed: 난수 시드.
        augment: 증강 적용 여부. False(기본)면 깨끗한 영상으로 평가한다.

    Returns:
        ``(이미지, 정답 번호판, 정답 bbox)`` 리스트. 생성 실패 시 빈 리스트.
    """
    try:
        import random  # noqa: PLC0415

        from app.train.synth_plate import generate_sample  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        logger.warning("평가 데이터 생성 불가: %s", exc)
        return []

    out: list[tuple[np.ndarray, str, tuple[int, int, int, int]]] = []
    for i in range(max(1, int(samples))):
        try:
            sample = generate_sample(random.Random(int(seed) + i), augment=bool(augment))
            out.append((sample.image, sample.text, sample.bbox))
        except Exception as exc:  # noqa: BLE001
            logger.info("샘플 %d 생성 실패: %s", i, exc)
            continue
    return out


def benchmark_pair(
    detector: str,
    recognizer: str,
    *,
    samples: int = DEFAULT_SAMPLES,
    seed: int = 20261024,
    warmup: int = DEFAULT_WARMUP,
    augment: bool = False,
    iou_threshold: float = 0.4,
) -> BenchmarkResult:
    """검출기 + 인식기 조합 하나를 벤치마크한다.

    Args:
        detector: 검출기 이름.
        recognizer: 인식기 이름.
        samples: 샘플 수.
        seed: 시드.
        warmup: 버릴 워밍업 반복 수.
        augment: 평가 데이터 증강 여부.
        iou_threshold: 검출 성공으로 볼 IoU 하한.

    Returns:
        BenchmarkResult. 실패해도 예외 없이 0 값 결과를 반환한다.
    """
    from app.config import get_settings  # noqa: PLC0415
    from app.core.registry import get_detector, get_ocr  # noqa: PLC0415
    from app.pipeline.lpr_pipeline import get_pipeline  # noqa: PLC0415

    dataset = build_eval_set(samples, seed=seed, augment=augment)
    if not dataset:
        return BenchmarkResult(
            model_name=f"{detector} + {recognizer}",
            detail={"error": "평가 데이터 생성 실패"},
        )

    # --- 메모리: 모델 로딩 전후 RSS 차이 ---
    gc.collect()
    rss_before = current_rss_mb()
    det_model = get_detector(detector)
    ocr_model = get_ocr(recognizer)
    gc.collect()
    rss_after = current_rss_mb()

    detector_name = getattr(det_model, "exposed_name", detector)
    recognizer_name = getattr(ocr_model, "exposed_name", recognizer)

    pipeline = get_pipeline()
    settings = get_settings()

    # --- 워밍업: 측정에 포함하지 않는다 ---
    for i in range(max(0, int(warmup))):
        image = dataset[i % len(dataset)][0]
        pipeline.run(image, detector=detector, recognizer=recognizer, min_conf=0.15)

    latencies: list[float] = []
    detect_hits = 0
    exact_hits = 0
    valid_format_hits = 0
    char_correct = 0
    char_total = 0

    for image, truth, truth_box in dataset:
        started = time.perf_counter()
        result = pipeline.run(image, detector=detector, recognizer=recognizer, min_conf=0.15)
        latencies.append((time.perf_counter() - started) * 1000.0)

        # 검출 재현율: 상위 후보 중 하나라도 정답 박스와 충분히 겹치면 성공.
        for box in result.detections:
            coords = box.get("bbox") or []
            if len(coords) == 4 and _iou(tuple(int(v) for v in coords), truth_box) >= iou_threshold:  # type: ignore[arg-type]
                detect_hits += 1
                break

        predicted = result.plate_number or ""
        if predicted == truth:
            exact_hits += 1
        if result.valid_format:
            valid_format_hits += 1
        # 문자 정확도: 길이가 같을 때만 자리별로 비교한다.
        if len(predicted) == len(truth):
            char_correct += sum(1 for a, b in zip(predicted, truth, strict=False) if a == b)
        char_total += len(truth)

    total = len(dataset)
    sizes = weight_size_mb(
        [
            settings.detector_weights if detector == "yolo" else None,
            settings.detector_onnx if detector == "yolo" else None,
            settings.crnn_weights if recognizer == "crnn" else None,
            settings.attention_weights if recognizer == "attention" else None,
        ]
    )

    return BenchmarkResult(
        model_name=f"{detector_name} + {recognizer_name}",
        task="detect+ocr",
        detect_recall=detect_hits / max(1, total),
        exact_match_rate=exact_hits / max(1, total),
        latency_p50_ms=_percentile(latencies, 50.0),
        latency_p95_ms=_percentile(latencies, 95.0),
        memory_mb=max(0.0, rss_after - rss_before),
        model_size_mb=sizes,
        quantized=False,
        detail={
            "samples": total,
            "seed": int(seed),
            "augmented": bool(augment),
            "iou_threshold": float(iou_threshold),
            "valid_format_rate": round(valid_format_hits / max(1, total), 4),
            "char_accuracy": round(char_correct / max(1, char_total), 4),
            "latency_mean_ms": round(statistics.fmean(latencies), 2) if latencies else 0.0,
            "latency_min_ms": round(min(latencies), 2) if latencies else 0.0,
            "latency_max_ms": round(max(latencies), 2) if latencies else 0.0,
            "detector_fallback": getattr(det_model, "is_fallback", True),
            "recognizer_fallback": getattr(ocr_model, "is_fallback", True),
            "rss_total_mb": round(rss_after, 2),
            "note": "합성 데이터 기준. 실제 데이터에서는 정확도가 낮아지므로 상대 비교용으로 해석하라.",
        },
    )


def benchmark_pdm(*, samples: int = 50, seed: int = 20261024) -> BenchmarkResult:
    """예지보전 파이프라인의 지연을 벤치마크한다.

    PdM 에는 "정답 라벨" 이 없으므로(합성 데이터로 고장 시점을 만들어야 함)
    정확도 대신 **지연과 처리량**만 측정한다. 백엔드가 차량 수천 대를
    배치로 예측할 때 이 수치가 용량 산정의 근거가 된다.

    Args:
        samples: 반복 횟수.
        seed: 난수 시드.

    Returns:
        BenchmarkResult(task=``"pdm"``).
    """
    import random  # noqa: PLC0415

    from app.pdm.service import get_pdm_service, pdm_spec  # noqa: PLC0415

    rng = random.Random(int(seed))
    service = get_pdm_service()

    def make_readings(count: int = 20) -> list[dict[str, Any]]:
        """무작위 센서 시계열을 만든다."""
        return [
            {
                "measured_at": f"2026-10-{(i % 28) + 1:02d}T09:00:00",
                "engine_temp_c": rng.uniform(85.0, 108.0),
                "engine_rpm": rng.uniform(800.0, 3200.0),
                "speed_kph": rng.uniform(0.0, 110.0),
                "battery_voltage": rng.uniform(11.8, 13.2),
                "battery_soh": rng.uniform(0.65, 0.99),
                "tire_pressure_fl": rng.uniform(27.0, 36.0),
                "tire_pressure_fr": rng.uniform(27.0, 36.0),
                "tire_pressure_rl": rng.uniform(27.0, 36.0),
                "tire_pressure_rr": rng.uniform(27.0, 36.0),
                "brake_pad_mm": rng.uniform(2.5, 10.0),
                "oil_quality": rng.uniform(0.2, 0.95),
                "vibration_rms": rng.uniform(0.1, 1.2),
                "fuel_level": rng.uniform(0.05, 1.0),
            }
            for i in range(count)
        ]

    gc.collect()
    rss_before = current_rss_mb()

    # 워밍업
    for _ in range(DEFAULT_WARMUP):
        service.predict(vehicle_id=1, readings=make_readings())

    latencies: list[float] = []
    for i in range(max(1, int(samples))):
        readings = make_readings()
        started = time.perf_counter()
        service.predict(vehicle_id=i + 1, readings=readings, odometer_km=50000.0 + i * 100)
        latencies.append((time.perf_counter() - started) * 1000.0)

    gc.collect()
    rss_after = current_rss_mb()
    spec = pdm_spec()

    return BenchmarkResult(
        model_name=str(spec.get("model_version", "pdm-ensemble-1.0.0")),
        task="pdm",
        # PdM 에는 검출/일치 개념이 없으므로 0 으로 둔다(계약서 필드 유지).
        detect_recall=0.0,
        exact_match_rate=0.0,
        latency_p50_ms=_percentile(latencies, 50.0),
        latency_p95_ms=_percentile(latencies, 95.0),
        memory_mb=max(0.0, rss_after - rss_before),
        model_size_mb=0.0,
        quantized=False,
        detail={
            "samples": len(latencies),
            "readings_per_sample": 20,
            "throughput_per_sec": round(1000.0 / max(1e-6, _percentile(latencies, 50.0)), 1),
            "weights_present": {
                k: v.get("exists") for k, v in (spec.get("weights") or {}).items()
            },
            "note": "정답 라벨이 없어 지연/처리량만 측정한다.",
        },
    )


def run_benchmark(
    *,
    samples: int = DEFAULT_SAMPLES,
    detectors: list[str] | None = None,
    recognizers: list[str] | None = None,
    seed: int = 20261024,
    include_pdm: bool = True,
    augment: bool = False,
) -> list[dict[str, Any]]:
    """모든 조합을 벤치마크하고 계약서 응답 배열을 만든다.

    Args:
        samples: 조합당 샘플 수.
        detectors: 평가할 검출기(None 이면 ``["yolo", "mobilenet"]``).
        recognizers: 평가할 인식기(None 이면 ``["crnn", "attention"]``).
            ``paddle`` 은 기본에서 제외한다 — 미설치 환경에서 폴백과 동일한
            결과를 내 중복 행만 늘린다.
        seed: 시드.
        include_pdm: PdM 벤치마크 포함 여부.
        augment: 평가 데이터 증강 여부.

    Returns:
        ``BenchmarkEntry`` dict 리스트. 지연 p50 오름차순 정렬.
    """
    det_list = list(detectors or ["yolo", "mobilenet"])
    ocr_list = list(recognizers or ["crnn", "attention"])

    results: list[dict[str, Any]] = []
    for det in det_list:
        for ocr in ocr_list:
            try:
                result = benchmark_pair(
                    det, ocr, samples=samples, seed=seed, augment=augment
                )
                results.append(result.to_dict())
            except Exception as exc:  # noqa: BLE001 - 한 조합 실패가 전체를 막지 않는다
                logger.warning("벤치마크 실패(%s+%s): %s", det, ocr, exc)
                results.append(
                    BenchmarkResult(
                        model_name=f"{det} + {ocr}", detail={"error": str(exc)}
                    ).to_dict()
                )

    if include_pdm:
        try:
            results.append(benchmark_pdm(samples=max(10, samples), seed=seed).to_dict())
        except Exception as exc:  # noqa: BLE001
            logger.warning("PdM 벤치마크 실패: %s", exc)

    results.sort(key=lambda r: float(r.get("latency_p50_ms") or 0.0))
    return results


def build_arg_parser() -> argparse.ArgumentParser:
    """CLI 인자 파서를 만든다.

    Returns:
        argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(
        prog="python -m app.train.benchmark",
        description="LPR/PdM CPU 벤치마크(지연·메모리·정확도)",
    )
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES, help="조합당 샘플 수")
    parser.add_argument("--seed", type=int, default=20261024, help="난수 시드")
    parser.add_argument(
        "--detectors", default="yolo,mobilenet", help="평가 검출기(쉼표 구분)"
    )
    parser.add_argument(
        "--recognizers", default="crnn,attention", help="평가 인식기(쉼표 구분)"
    )
    parser.add_argument("--no-pdm", action="store_true", help="PdM 벤치마크 제외")
    parser.add_argument("--augment", action="store_true", help="평가 데이터에 증강 적용")
    parser.add_argument("--out", default="", help="결과 JSON 저장 경로(미지정 시 표준출력)")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI 진입점.

    Args:
        argv: 인자 리스트.

    Returns:
        종료 코드.
    """
    args = build_arg_parser().parse_args(argv)
    from app.core.logging import configure_logging  # noqa: PLC0415

    configure_logging("INFO")

    try:
        results = run_benchmark(
            samples=args.samples,
            detectors=[s.strip() for s in str(args.detectors).split(",") if s.strip()],
            recognizers=[s.strip() for s in str(args.recognizers).split(",") if s.strip()],
            seed=args.seed,
            include_pdm=not args.no_pdm,
            augment=args.augment,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] 벤치마크 실패: {exc}", file=sys.stderr)
        return 1

    payload = json.dumps(results, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).write_text(payload, encoding="utf-8")
        print(f"결과 저장: {args.out}")
    else:
        print(payload)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "DEFAULT_SAMPLES",
    "DEFAULT_WARMUP",
    "BenchmarkResult",
    "benchmark_pair",
    "benchmark_pdm",
    "build_eval_set",
    "current_rss_mb",
    "main",
    "run_benchmark",
    "weight_size_mb",
]
