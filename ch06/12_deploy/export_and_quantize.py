# -*- coding: utf-8 -*-
"""PyTorch 모델을 ONNX 로 내보내고 INT8 동적 양자화까지 수행한다.

파이프라인
    PyTorch (.pt)  ──export──▶  ONNX FP32 (.onnx)  ──quantize──▶  ONNX INT8 (.onnx)

의존성이 없으면 중단하지 않고 "무엇을 설치해야 하는지" 를 안내한 뒤,
양자화 이론 압축률만 계산해 보여준다. 교재 독자가 어떤 환경에서든
파일을 실행해 다음 단계를 알 수 있게 하기 위한 설계다.

사용법
    python 12_deploy/export_and_quantize.py --weights models/crnn_ctc_kr.pt
    python 12_deploy/export_and_quantize.py --check     # 환경 점검만
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ONNX opset — 17 이상이면 LayerNorm 이 단일 노드로 내려가 CRNN/Transformer 변환이 쉽다.
OPSET = 17
#: 검출기 입력 크기(CONTRACT.md 의 detect_img_size)
DETECT_SIZE = 640
#: OCR 입력 크기(CONTRACT.md 의 ocr_img_height / ocr_img_width)
OCR_H, OCR_W = 32, 160


def probe() -> dict[str, str | None]:
    """변환에 필요한 패키지의 설치 여부와 버전을 확인한다.

    Returns:
        패키지명 → 버전 문자열(미설치면 None).
    """
    found: dict[str, str | None] = {}
    for name in ("torch", "onnx", "onnxruntime"):
        try:
            mod = __import__(name)
            found[name] = str(getattr(mod, "__version__", "unknown"))
        except ImportError:
            found[name] = None
    try:
        from onnxruntime.quantization import quantize_dynamic  # noqa: F401
        found["onnxruntime.quantization"] = "available"
    except Exception:
        found["onnxruntime.quantization"] = None
    return found


def print_probe(found: dict[str, str | None]) -> bool:
    """환경 점검 결과를 출력하고 변환 가능 여부를 반환한다."""
    print("[환경 점검]")
    for name, ver in found.items():
        mark = "OK " if ver else "없음"
        print(f"  {mark} {name:<26} {ver or '-'}")
    ready = bool(found["torch"] and found["onnx"] and found["onnxruntime.quantization"])
    if not ready:
        print("\n[안내] 변환에 필요한 패키지가 부족하다. 아래로 설치한다.")
        print("  pip install torch --index-url https://download.pytorch.org/whl/cpu")
        print("  pip install onnx onnxruntime")
        print("  (onnxruntime 만 있으면 추론은 되지만 변환·양자화는 onnx 가 필요하다)")
    return ready


def size_mb(path: Path) -> float:
    """파일 크기를 MB 로 반환한다(없으면 0)."""
    try:
        return path.stat().st_size / (1024 * 1024)
    except OSError:
        return 0.0


def theoretical_compression(params: int) -> None:
    """파라미터 수로 FP32 / FP16 / INT8 이론 크기를 계산해 보여준다.

    실제 파일 크기는 그래프 메타데이터와 양자화 스케일·제로포인트 때문에
    이론값보다 조금 크다. 그래도 "4배 가까이 줄어든다" 는 기대치를 미리
    잡아 두면 변환 결과가 이상할 때 바로 알아차릴 수 있다.

    Args:
        params: 모델 파라미터 개수.
    """
    print(f"\n[이론 압축률 — 파라미터 {params:,}개]")
    print(f"  {'정밀도':<10}{'바이트/파라미터':>14}{'크기(MB)':>12}{'FP32 대비':>10}")
    base = params * 4 / (1024 * 1024)
    for name, nbytes in (("FP32", 4), ("FP16", 2), ("INT8", 1)):
        mb = params * nbytes / (1024 * 1024)
        print(f"  {name:<10}{nbytes:>14}{mb:>12.2f}{mb / base * 100:>9.0f}%")
    print("  주의: 가중치만 줄어든다. 활성값 버퍼와 런타임 상주 메모리는 별도다.")


def export_detector(weights: Path, out: Path) -> bool:
    """검출기를 ONNX 로 내보낸다.

    동적 축을 batch 에만 준다. 입력 해상도까지 동적으로 두면 런타임이
    매 호출마다 메모리 계획을 다시 세워 CPU 추론이 오히려 느려진다.

    Args:
        weights: PyTorch 가중치 경로.
        out: 출력 ONNX 경로.

    Returns:
        성공 여부.
    """
    try:
        import torch
    except ImportError:
        print("  [건너뜀] torch 미설치")
        return False
    if not weights.is_file():
        print(f"  [건너뜀] 가중치가 없다: {weights}")
        return False
    try:
        model = torch.load(weights, map_location="cpu", weights_only=False)
        if hasattr(model, "model"):      # ultralytics 래퍼 대응
            model = model.model
        model.eval().float()
        dummy = torch.zeros(1, 3, DETECT_SIZE, DETECT_SIZE)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.onnx.export(
            model, dummy, str(out),
            input_names=["images"], output_names=["output"],
            opset_version=OPSET,
            dynamic_axes={"images": {0: "batch"}, "output": {0: "batch"}},
            do_constant_folding=True,   # 상수 접기로 노드 수를 줄인다
        )
        print(f"  검출기 ONNX 생성: {out} ({size_mb(out):.2f} MB)")
        return True
    except Exception as exc:  # noqa: BLE001 - 변환 실패 원인을 그대로 보여준다
        print(f"  [실패] 검출기 변환: {type(exc).__name__}: {exc}")
        return False


def export_recognizer(weights: Path, out: Path) -> bool:
    """CRNN 인식기를 ONNX 로 내보낸다.

    CRNN 은 LSTM 을 포함하므로 opset 17 이상을 쓰고, 시퀀스 길이는 입력 폭에서
    결정되므로 폭을 고정한다(OCR_W). 폭을 동적으로 두면 CTC 디코더가
    받는 타임스텝 수가 호출마다 바뀌어 디버깅이 어려워진다.

    Args:
        weights: PyTorch 가중치 경로.
        out: 출력 ONNX 경로.

    Returns:
        성공 여부.
    """
    try:
        import torch
    except ImportError:
        print("  [건너뜀] torch 미설치")
        return False
    if not weights.is_file():
        print(f"  [건너뜀] 가중치가 없다: {weights}")
        return False
    try:
        model = torch.load(weights, map_location="cpu", weights_only=False)
        model.eval().float()
        dummy = torch.zeros(1, 1, OCR_H, OCR_W)   # 그레이스케일 1채널
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.onnx.export(
            model, dummy, str(out),
            input_names=["crop"], output_names=["logits"],
            opset_version=OPSET,
            dynamic_axes={"crop": {0: "batch"}, "logits": {1: "batch"}},
            do_constant_folding=True,
        )
        print(f"  인식기 ONNX 생성: {out} ({size_mb(out):.2f} MB)")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"  [실패] 인식기 변환: {type(exc).__name__}: {exc}")
        return False


def quantize(src: Path, dst: Path) -> bool:
    """ONNX FP32 모델을 INT8 동적 양자화한다.

    왜 동적(dynamic) 양자화인가?
        정적(static) 양자화는 대표 데이터로 활성값 범위를 보정(calibration)해야
        해서 추가 데이터셋과 시간이 필요하다. 동적 양자화는 가중치만 미리
        INT8 로 바꾸고 활성값 스케일은 추론 시 계산하므로, 보정 데이터 없이
        바로 적용되고 MatMul·LSTM 비중이 큰 OCR 모델에서 특히 효과가 좋다.
        반대로 Conv 비중이 큰 검출기는 동적 양자화 효과가 제한적이므로
        정적 양자화를 별도로 비교해야 한다.

    Args:
        src: 입력 FP32 ONNX 경로.
        dst: 출력 INT8 ONNX 경로.

    Returns:
        성공 여부.
    """
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except Exception as exc:  # noqa: BLE001
        print(f"  [건너뜀] 양자화 모듈 없음: {type(exc).__name__}")
        return False
    if not src.is_file():
        print(f"  [건너뜀] 원본 ONNX 가 없다: {src}")
        return False
    try:
        quantize_dynamic(
            model_input=str(src),
            model_output=str(dst),
            weight_type=QuantType.QInt8,   # 대칭 INT8. QUInt8 은 일부 CPU 에서 더 빠르다
            per_channel=True,              # 채널별 스케일 → 정확도 손실을 줄인다
        )
        before, after = size_mb(src), size_mb(dst)
        ratio = (after / before * 100) if before else 0.0
        print(f"  INT8 생성: {dst}")
        print(f"    {before:.2f} MB → {after:.2f} MB ({ratio:.0f}%)")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"  [실패] 양자화: {type(exc).__name__}: {exc}")
        return False


def main(argv: list[str] | None = None) -> int:
    """명령행 진입점."""
    ap = argparse.ArgumentParser(description="ONNX 변환 + INT8 양자화")
    ap.add_argument("--model-dir", type=Path, default=Path("models"),
                    help="가중치/출력 디렉터리")
    ap.add_argument("--detector", default="plate_yolo_nano.pt")
    ap.add_argument("--recognizer", default="crnn_ctc_kr.pt")
    ap.add_argument("--check", action="store_true", help="환경 점검만 수행")
    args = ap.parse_args(argv)

    print("=" * 66)
    print("ONNX 변환 · INT8 양자화")
    print("=" * 66)

    found = probe()
    ready = print_probe(found)
    if args.check:
        return 0

    md = args.model_dir
    print("\n[1단계] PyTorch → ONNX FP32")
    det_ok = export_detector(md / args.detector, md / "plate_yolo_nano.onnx")
    rec_ok = export_recognizer(md / args.recognizer, md / "crnn_ctc_kr.onnx")

    print("\n[2단계] ONNX FP32 → ONNX INT8")
    if det_ok:
        quantize(md / "plate_yolo_nano.onnx", md / "plate_yolo_nano.int8.onnx")
    else:
        print("  [건너뜀] 검출기 FP32 ONNX 없음")
    if rec_ok:
        quantize(md / "crnn_ctc_kr.onnx", md / "crnn_ctc_kr.int8.onnx")
    else:
        print("  [건너뜀] 인식기 FP32 ONNX 없음")

    # 변환이 안 되는 환경에서도 기대치를 잡을 수 있도록 이론값을 보여준다.
    theoretical_compression(params=2_570_000)   # YOLO-nano 급 파라미터 수 예시
    theoretical_compression(params=8_420_000)   # CRNN/CTC 급 파라미터 수 예시

    print("\n[다음 단계]")
    print("  1) bench_latency.py 로 FP32/INT8 지연시간을 p50/p95/p99 로 비교한다.")
    print("  2) field_test_runner.py 로 INT8 의 완전일치율 하락 폭을 확인한다.")
    print("  3) 지연 이득과 정확도 손실을 함께 보고 채택을 결정한다.")
    return 0 if ready else 0


if __name__ == "__main__":
    sys.exit(main())
