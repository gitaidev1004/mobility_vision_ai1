"""INT8 동적 양자화 (CLI).

양자화란?
    float32(4바이트) 가중치를 int8(1바이트)로 바꾸는 것이다.
    모델 크기가 **1/4** 로 줄고, 정수 연산이 부동소수점보다 빨라
    CPU 추론이 **1.5~3배** 빨라진다.

**동적(dynamic)** 양자화를 쓰는 이유 — 정적(static)과의 비교:

    | | 동적 양자화 | 정적 양자화 |
    |---|---|---|
    | 보정 데이터 | **불필요** | 수백 장 필요(calibration) |
    | 가중치 | int8 (사전 변환) | int8 |
    | 활성값 | 런타임에 동적 스케일 | 사전 계산된 고정 스케일 |
    | 정확도 손실 | 작음 (0.5~2%) | 중간 (1~4%) |
    | 속도 | 빠름 | **가장 빠름** |
    | 적용 난이도 | **한 줄** | 모델 수정 + 보정 파이프라인 |

    CRNN 은 **LSTM 과 Linear 가 연산의 대부분**을 차지한다. 동적 양자화는
    정확히 이 두 계층을 지원하므로 가성비가 최고다. Conv 계층은 동적 양자화
    대상이 아니지만, CRNN 의 Conv 는 전체 연산의 작은 부분이다.

왜 YOLO 는 동적 양자화로 이득이 적은가?
    YOLO 는 거의 전부 Conv 다. Conv 를 양자화하려면 정적 양자화나
    ONNX Runtime 의 양자화 도구가 필요하다. 이 스크립트는 ONNX 경로도
    함께 제공해 그 선택지를 보여 준다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from app.core.logging import configure_logging, get_logger

logger = get_logger(__name__)


def check_torch() -> bool:
    """torch 설치 여부를 확인한다.

    Returns:
        설치되어 있으면 True.
    """
    try:
        import importlib.util  # noqa: PLC0415

        return importlib.util.find_spec("torch") is not None
    except (ImportError, ValueError):
        return False


def quantize_crnn_dynamic(
    weights: str,
    output: str,
    *,
    height: int = 32,
    width: int = 160,
    compare: bool = True,
) -> dict[str, Any]:
    """CRNN 을 INT8 동적 양자화한다.

    ``torch.quantization.quantize_dynamic`` 에 ``{nn.LSTM, nn.Linear}`` 를 지정한다.
    이 두 계층만 양자화되고 Conv/BatchNorm 은 float32 로 남는다 —
    "혼합 정밀도" 상태가 되지만, 추론 결과는 동일한 인터페이스로 쓸 수 있다.

    주의: 양자화된 모델은 ``state_dict`` 구조가 달라져 **원본 모델 클래스로
    로드할 수 없다**. 따라서 ``torch.save(model)`` 로 모델 객체 전체를 저장하거나,
    양자화를 추론 시점에 매번 적용해야 한다. 이 스크립트는 전자를 쓰고,
    그 사실을 체크포인트 메타데이터에 명시한다.

    Args:
        weights: 원본 ``.pt`` 체크포인트 경로.
        output: 양자화 모델 저장 경로.
        height: 입력 높이(크기 비교용 더미 입력).
        width: 입력 너비.
        compare: True 면 원본과 출력을 비교해 정확도 변화를 측정한다.

    Returns:
        양자화 결과 요약 dict.

    Raises:
        RuntimeError: torch 미설치, 가중치 없음, 또는 양자화 실패.
    """
    if not check_torch():
        msg = "PyTorch 가 필요합니다: pip install torch==2.8.0"
        raise RuntimeError(msg)
    source = Path(str(weights))
    if not source.is_file():
        msg = f"CRNN 가중치가 없습니다: {weights}"
        raise RuntimeError(msg)

    import torch  # noqa: PLC0415

    from app.ocr.crnn import load_crnn  # noqa: PLC0415

    model = load_crnn(str(source), device="cpu")
    model.eval()

    original_size_mb = source.stat().st_size / (1024 * 1024)

    quantized = torch.quantization.quantize_dynamic(
        model,
        # LSTM 과 Linear 만 양자화한다(동적 양자화가 지원하는 계층).
        {torch.nn.LSTM, torch.nn.Linear},
        dtype=torch.qint8,
    )
    quantized.eval()

    output_path = Path(str(output))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # 모델 객체 전체를 저장한다(양자화 구조를 복원하려면 필요 —
    # docstring 의 주의사항 참조).
    torch.save(
        {
            "model_object": quantized,
            "quantized": True,
            "quantization": "dynamic-int8",
            "quantized_layers": ["LSTM", "Linear"],
            "input_height": int(height),
            "input_width": int(width),
            "note": "torch.load(weights_only=False) 로 로드해야 한다(모델 객체 포함).",
        },
        output_path,
    )

    quantized_size_mb = output_path.stat().st_size / (1024 * 1024)
    result: dict[str, Any] = {
        "model": "crnn",
        "source": str(source),
        "output": str(output_path),
        "original_size_mb": round(original_size_mb, 2),
        "quantized_size_mb": round(quantized_size_mb, 2),
        "compression_ratio": round(original_size_mb / max(1e-6, quantized_size_mb), 2),
        "quantization": "dynamic-int8",
        "quantized_layers": ["LSTM", "Linear"],
    }

    if compare:
        result["comparison"] = _compare_outputs(model, quantized, height, width)
    return result


def _compare_outputs(
    original: Any, quantized: Any, height: int, width: int, *, trials: int = 8
) -> dict[str, Any]:
    """원본과 양자화 모델의 출력 차이와 속도를 비교한다.

    왜 argmax 일치율을 보는가?
        절대 오차만 보면 "0.02 차이" 가 괜찮은지 알 수 없다. CTC 디코딩은
        각 프레임의 **argmax** 만 쓰므로, argmax 가 같으면 디코딩 결과도 같다.
        즉 argmax 일치율이 실질적인 정확도 보존 지표다.

    Args:
        original: 원본 모델.
        quantized: 양자화 모델.
        height: 입력 높이.
        width: 입력 너비.
        trials: 비교 반복 횟수.

    Returns:
        오차/일치율/속도 비교 dict.
    """
    try:
        import time  # noqa: PLC0415

        import torch  # noqa: PLC0415
    except ImportError:
        return {"error": "torch 미설치"}

    try:
        torch.manual_seed(20261024)  # 재현성
        diffs: list[float] = []
        argmax_matches: list[float] = []
        original_times: list[float] = []
        quantized_times: list[float] = []

        with torch.no_grad():
            # 워밍업(첫 호출의 지연 로딩을 측정에서 제외).
            dummy = torch.randn(1, 1, int(height), int(width))
            original(dummy)
            quantized(dummy)

            for _ in range(max(1, int(trials))):
                sample = torch.randn(1, 1, int(height), int(width))

                started = time.perf_counter()
                out_original = original(sample)
                original_times.append((time.perf_counter() - started) * 1000.0)

                started = time.perf_counter()
                out_quantized = quantized(sample)
                quantized_times.append((time.perf_counter() - started) * 1000.0)

                a = out_original.cpu().numpy()
                b = out_quantized.cpu().numpy()
                diffs.append(float(np.max(np.abs(a - b))))
                argmax_matches.append(
                    float(np.mean(a.argmax(axis=-1) == b.argmax(axis=-1)))
                )

        mean_original = float(np.mean(original_times))
        mean_quantized = float(np.mean(quantized_times))
        return {
            "max_abs_diff": round(float(np.max(diffs)), 6),
            "mean_abs_diff": round(float(np.mean(diffs)), 6),
            # 이 값이 1.0 에 가까우면 디코딩 결과가 사실상 동일하다.
            "argmax_agreement": round(float(np.mean(argmax_matches)), 6),
            "latency_original_ms": round(mean_original, 3),
            "latency_quantized_ms": round(mean_quantized, 3),
            "speedup": round(mean_original / max(1e-6, mean_quantized), 2),
            "trials": int(trials),
        }
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}


def quantize_onnx(
    onnx_path: str,
    output: str,
    *,
    per_channel: bool = True,
) -> dict[str, Any]:
    """ONNX 모델을 INT8 동적 양자화한다(onnxruntime.quantization).

    ONNX 경로가 PyTorch 동적 양자화보다 나은 점:
        **Conv 계층도 양자화된다.** 따라서 YOLO 처럼 Conv 가 대부분인 모델에서
        의미 있는 속도 향상을 얻는다. PyTorch 동적 양자화로는 불가능하다.

    ``per_channel=True`` 의 의미: 채널마다 별도의 스케일을 쓴다. 채널 간
    값 범위 차이가 큰 Conv 에서 정확도 손실을 크게 줄인다(비용은 메타데이터 약간 증가).

    Args:
        onnx_path: 원본 ``.onnx`` 경로.
        output: 양자화 ``.onnx`` 출력 경로.
        per_channel: 채널별 양자화 사용 여부.

    Returns:
        양자화 결과 요약 dict.

    Raises:
        RuntimeError: onnxruntime 미설치, 파일 없음, 또는 실패.
    """
    source = Path(str(onnx_path))
    if not source.is_file():
        msg = f"ONNX 파일이 없습니다: {onnx_path}"
        raise RuntimeError(msg)

    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic  # noqa: PLC0415
    except ImportError as exc:
        msg = (
            "onnxruntime 의 양자화 모듈을 찾을 수 없습니다.\n"
            "  pip install onnxruntime==1.22.1"
        )
        raise RuntimeError(msg) from exc

    output_path = Path(str(output))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    quantize_dynamic(
        model_input=str(source),
        model_output=str(output_path),
        weight_type=QuantType.QInt8,
        per_channel=bool(per_channel),
    )

    original_mb = source.stat().st_size / (1024 * 1024)
    quantized_mb = output_path.stat().st_size / (1024 * 1024)
    return {
        "model": "onnx",
        "source": str(source),
        "output": str(output_path),
        "original_size_mb": round(original_mb, 2),
        "quantized_size_mb": round(quantized_mb, 2),
        "compression_ratio": round(original_mb / max(1e-6, quantized_mb), 2),
        "quantization": "dynamic-int8",
        "per_channel": bool(per_channel),
        "note": "ONNX 경로는 Conv 계층도 양자화되므로 YOLO 에 유효하다.",
    }


def build_arg_parser() -> argparse.ArgumentParser:
    """CLI 인자 파서를 만든다.

    Returns:
        argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(
        prog="python -m app.train.quantize",
        description="INT8 동적 양자화(PyTorch CRNN / ONNX)",
    )
    parser.add_argument("--crnn", default="", help="CRNN .pt 경로(PyTorch 동적 양자화)")
    parser.add_argument(
        "--crnn-out", default="./models/crnn_ctc_kr_int8.pt", help="CRNN 양자화 출력 경로"
    )
    parser.add_argument("--onnx", default="", help="ONNX 파일 경로(ONNX 동적 양자화)")
    parser.add_argument(
        "--onnx-out", default="", help="ONNX 양자화 출력 경로(기본: 원본명 + _int8.onnx)"
    )
    parser.add_argument("--height", type=int, default=32, help="CRNN 입력 높이")
    parser.add_argument("--width", type=int, default=160, help="CRNN 입력 너비")
    parser.add_argument("--no-compare", action="store_true", help="원본 비교 생략")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI 진입점.

    Args:
        argv: 인자 리스트.

    Returns:
        종료 코드(0 = 성공, 2 = 의존성/입력 없음, 1 = 실패).
    """
    args = build_arg_parser().parse_args(argv)
    configure_logging("INFO")

    if not args.crnn and not args.onnx:
        print(
            "[ERROR] --crnn 또는 --onnx 중 하나는 지정해야 합니다.\n"
            "  예: python -m app.train.quantize --crnn ./models/crnn_ctc_kr.pt",
            file=sys.stderr,
        )
        return 2

    results: list[dict[str, Any]] = []

    if args.crnn:
        try:
            results.append(
                quantize_crnn_dynamic(
                    args.crnn,
                    args.crnn_out,
                    height=args.height,
                    width=args.width,
                    compare=not args.no_compare,
                )
            )
        except RuntimeError as exc:
            print(f"[SKIP] CRNN 양자화: {exc}", file=sys.stderr)
        except Exception as exc:  # noqa: BLE001
            print(f"[ERROR] CRNN 양자화 실패: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1

    if args.onnx:
        onnx_out = args.onnx_out or str(
            Path(args.onnx).with_name(Path(args.onnx).stem + "_int8.onnx")
        )
        try:
            results.append(quantize_onnx(args.onnx, onnx_out))
        except RuntimeError as exc:
            print(f"[SKIP] ONNX 양자화: {exc}", file=sys.stderr)
        except Exception as exc:  # noqa: BLE001
            print(f"[ERROR] ONNX 양자화 실패: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1

    if not results:
        return 2
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["check_torch", "main", "quantize_crnn_dynamic", "quantize_onnx"]
