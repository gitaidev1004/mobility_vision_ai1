"""ONNX 내보내기 (CLI).

왜 ONNX 로 내보내는가?
    | | PyTorch(.pt) | ONNX Runtime(.onnx) |
    |---|---|---|
    | 런타임 크기 | 약 2.5GB | 약 60MB |
    | CPU 추론 속도 | 기준 | **1.5~2.5배 빠름** |
    | 기동 시간 | torch import 1~3초 | 0.1초 미만 |
    | 배포 | 파이썬 필수 | C++/Java/Node 등 다양 |

    게이트 서버는 CPU 뿐이고 컨테이너 크기가 중요하다. ONNX 가 배포에 압도적으로 유리하다.

내보낼 때 반드시 지켜야 할 것들:

    1. **``model.eval()``** — BatchNorm/Dropout 이 학습 모드로 고정되면
       추론 결과가 달라진다. 가장 흔한 실수다.
    2. **동적 축(dynamic_axes)** — 배치 크기를 고정하면 1장씩 처리할 때만
       동작한다. 배치 축은 반드시 동적으로 둔다.
    3. **수치 검증** — 내보낸 뒤 PyTorch 출력과 비교해야 한다. opset 차이나
       지원되지 않는 연산 때문에 조용히 다른 결과가 나오는 경우가 있다.
       이 스크립트는 ``--verify`` 로 최대 절대 오차를 측정한다.
    4. **opset 선택** — 17 은 PyTorch 2.x 와 ONNX Runtime 1.16+ 에서 안정적이다.
       너무 높으면 런타임이 지원하지 않고, 너무 낮으면 연산이 분해되어 느려진다.
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

#: 기본 ONNX opset 버전(PyTorch 2.8 / onnxruntime 1.22 조합에서 안정적).
DEFAULT_OPSET: int = 17

#: 수치 검증 통과 기준(최대 절대 오차). float32 연산 순서 차이로 1e-4 수준의
#: 오차는 정상이다. 1e-3 을 넘으면 연산이 잘못 변환된 것이다.
VERIFY_TOLERANCE: float = 1e-3


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


def export_crnn(
    weights: str,
    output: str,
    *,
    height: int = 32,
    width: int = 160,
    opset: int = DEFAULT_OPSET,
    verify: bool = True,
) -> dict[str, Any]:
    """CRNN 체크포인트를 ONNX 로 내보낸다.

    Args:
        weights: ``.pt`` 체크포인트 경로.
        output: ``.onnx`` 출력 경로.
        height: 입력 높이.
        width: 입력 너비.
        opset: ONNX opset 버전.
        verify: True 면 PyTorch 와 수치 비교를 수행한다.

    Returns:
        내보내기 결과 요약 dict.

    Raises:
        RuntimeError: torch 미설치, 가중치 없음, 또는 내보내기 실패.
    """
    if not check_torch():
        msg = "PyTorch 가 필요합니다: pip install torch==2.8.0"
        raise RuntimeError(msg)
    if not Path(str(weights)).is_file():
        msg = f"CRNN 가중치가 없습니다: {weights}"
        raise RuntimeError(msg)

    import torch  # noqa: PLC0415

    from app.ocr.crnn import load_crnn  # noqa: PLC0415

    model = load_crnn(str(weights), device="cpu")
    model.eval()  # 필수 — 모듈 docstring 1번 참조.

    dummy = torch.randn(1, 1, int(height), int(width), dtype=torch.float32)
    output_path = Path(str(output))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        input_names=["images"],
        output_names=["logits"],
        # 배치 축을 동적으로 둔다. 시퀀스 길이(T)는 입력 너비에서 결정되므로
        # 너비도 동적으로 둘 수 있으나, 고정하면 ONNX Runtime 이 더 최적화한다.
        dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=int(opset),
        do_constant_folding=True,
        export_params=True,
    )

    result: dict[str, Any] = {
        "model": "crnn",
        "weights": str(weights),
        "output": str(output_path),
        "input_shape": [1, 1, int(height), int(width)],
        "opset": int(opset),
        "size_mb": round(output_path.stat().st_size / (1024 * 1024), 2),
    }

    if verify:
        result["verification"] = _verify_onnx(model, dummy, output_path)
    return result


def export_detector(
    weights: str,
    output: str,
    *,
    img_size: int = 640,
    opset: int = DEFAULT_OPSET,
    simplify: bool = True,
) -> dict[str, Any]:
    """YOLO 검출기를 ONNX 로 내보낸다(ultralytics 내장 export 사용).

    왜 ultralytics 의 ``export`` 를 쓰는가?
        YOLO 는 후처리(앵커 디코딩, NMS)가 모델 구조에 얽혀 있어
        직접 ``torch.onnx.export`` 하면 출력 형식이 달라진다. ultralytics 의
        export 는 이 부분을 올바르게 처리하고, 우리 ``_decode_onnx_output``
        이 그 출력 형식을 전제로 작성되어 있다.

    Args:
        weights: ``.pt`` 경로.
        output: ``.onnx`` 출력 경로(ultralytics 가 만든 파일을 이 경로로 옮긴다).
        img_size: 입력 크기.
        opset: opset 버전.
        simplify: onnx-simplifier 적용 여부(설치돼 있을 때만 동작).

    Returns:
        내보내기 결과 요약 dict.

    Raises:
        RuntimeError: ultralytics 미설치, 가중치 없음, 또는 실패.
    """
    try:
        import importlib.util  # noqa: PLC0415

        if importlib.util.find_spec("ultralytics") is None:
            msg = "ultralytics 가 필요합니다: pip install ultralytics==8.3.200"
            raise RuntimeError(msg)
    except (ImportError, ValueError) as exc:
        msg = "ultralytics 가 필요합니다: pip install ultralytics==8.3.200"
        raise RuntimeError(msg) from exc

    if not Path(str(weights)).is_file():
        msg = f"검출기 가중치가 없습니다: {weights}"
        raise RuntimeError(msg)

    from ultralytics import YOLO  # noqa: PLC0415

    model = YOLO(str(weights))
    exported = model.export(
        format="onnx",
        imgsz=int(img_size),
        opset=int(opset),
        simplify=bool(simplify),
        dynamic=False,  # 고정 입력이 CPU 추론에서 더 빠르다
        half=False,     # CPU 는 fp16 이득이 없다
    )

    # ultralytics 는 가중치 옆에 파일을 만든다. 지정 경로로 옮긴다.
    source = Path(str(exported)) if exported else Path(str(weights)).with_suffix(".onnx")
    target = Path(str(output))
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.is_file() and source.resolve() != target.resolve():
        try:
            import shutil  # noqa: PLC0415

            shutil.move(str(source), str(target))
        except OSError as exc:
            logger.warning("ONNX 파일 이동 실패(%s → %s): %s", source, target, exc)
            target = source

    return {
        "model": "yolo",
        "weights": str(weights),
        "output": str(target),
        "img_size": int(img_size),
        "opset": int(opset),
        "size_mb": round(target.stat().st_size / (1024 * 1024), 2) if target.is_file() else 0.0,
    }


def _verify_onnx(model: Any, dummy: Any, onnx_path: Path) -> dict[str, Any]:
    """내보낸 ONNX 의 출력을 PyTorch 와 비교한다.

    이 검증이 왜 필수인가?
        ONNX 변환은 "조용히 틀리는" 실패 모드가 있다. 예외 없이 내보내졌지만
        특정 연산이 다르게 해석되어 결과가 어긋나는 경우다. 이를 배포 후에
        발견하면 원인 추적이 매우 어렵다. 내보내기 직후 1회 비교로 잡아낸다.

    Args:
        model: PyTorch 모델(eval 모드).
        dummy: 더미 입력 텐서.
        onnx_path: 내보낸 ONNX 경로.

    Returns:
        ``{"passed", "max_abs_diff", "tolerance"}`` 또는 사유가 담긴 dict.
    """
    try:
        import onnxruntime as ort  # noqa: PLC0415
        import torch  # noqa: PLC0415
    except ImportError:
        return {"passed": None, "reason": "onnxruntime 미설치 — 검증 생략"}

    try:
        with torch.no_grad():
            torch_output = model(dummy).cpu().numpy()

        session = ort.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"]
        )
        input_name = session.get_inputs()[0].name
        onnx_output = session.run(None, {input_name: dummy.cpu().numpy()})[0]

        if torch_output.shape != np.asarray(onnx_output).shape:
            return {
                "passed": False,
                "reason": f"shape 불일치: torch={torch_output.shape} onnx={np.asarray(onnx_output).shape}",
            }

        max_diff = float(np.max(np.abs(torch_output - np.asarray(onnx_output))))
        return {
            "passed": bool(max_diff <= VERIFY_TOLERANCE),
            "max_abs_diff": round(max_diff, 8),
            "tolerance": VERIFY_TOLERANCE,
        }
    except Exception as exc:  # noqa: BLE001
        return {"passed": False, "reason": f"{type(exc).__name__}: {exc}"}


def build_arg_parser() -> argparse.ArgumentParser:
    """CLI 인자 파서를 만든다.

    Returns:
        argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(
        prog="python -m app.train.export_onnx",
        description="CRNN / YOLO 모델을 ONNX 로 내보낸다",
    )
    parser.add_argument("--crnn", default="", help="CRNN .pt 경로(지정 시 CRNN 내보내기)")
    parser.add_argument(
        "--crnn-out", default="./models/crnn_ctc_kr.onnx", help="CRNN ONNX 출력 경로"
    )
    parser.add_argument(
        "--detector", default="", help="YOLO .pt 경로(지정 시 검출기 내보내기)"
    )
    parser.add_argument(
        "--detector-out", default="./models/plate_yolo_nano.onnx", help="검출기 ONNX 출력 경로"
    )
    parser.add_argument("--height", type=int, default=32, help="CRNN 입력 높이")
    parser.add_argument("--width", type=int, default=160, help="CRNN 입력 너비")
    parser.add_argument("--imgsz", type=int, default=640, help="검출기 입력 크기")
    parser.add_argument("--opset", type=int, default=DEFAULT_OPSET, help="ONNX opset 버전")
    parser.add_argument("--no-verify", action="store_true", help="수치 검증 생략")
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

    if not args.crnn and not args.detector:
        print(
            "[ERROR] --crnn 또는 --detector 중 하나는 지정해야 합니다.\n"
            "  예: python -m app.train.export_onnx --crnn ./models/crnn_ctc_kr.pt",
            file=sys.stderr,
        )
        return 2

    results: list[dict[str, Any]] = []

    if args.crnn:
        try:
            results.append(
                export_crnn(
                    args.crnn,
                    args.crnn_out,
                    height=args.height,
                    width=args.width,
                    opset=args.opset,
                    verify=not args.no_verify,
                )
            )
        except RuntimeError as exc:
            print(f"[SKIP] CRNN 내보내기: {exc}", file=sys.stderr)
        except Exception as exc:  # noqa: BLE001
            print(f"[ERROR] CRNN 내보내기 실패: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1

    if args.detector:
        try:
            results.append(
                export_detector(
                    args.detector,
                    args.detector_out,
                    img_size=args.imgsz,
                    opset=args.opset,
                )
            )
        except RuntimeError as exc:
            print(f"[SKIP] 검출기 내보내기: {exc}", file=sys.stderr)
        except Exception as exc:  # noqa: BLE001
            print(f"[ERROR] 검출기 내보내기 실패: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1

    if not results:
        return 2
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "DEFAULT_OPSET",
    "VERIFY_TOLERANCE",
    "check_torch",
    "export_crnn",
    "export_detector",
    "main",
]
