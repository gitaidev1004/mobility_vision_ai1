# -*- coding: utf-8 -*-
"""학습된 검출 모델을 ONNX 로 내보낸다.

ONNX(Open Neural Network Exchange)로 바꾸면
  - PyTorch 런타임 없이 ONNX Runtime 만으로 추론할 수 있어 배포가 가벼워진다
  - 그래프 수준 최적화(연산 융합, 상수 접기)가 적용돼 CPU 추론이 빨라진다
  - 동적 양자화(INT8) 같은 후처리 최적화를 적용할 수 있다

주의할 점
  - opset 버전이 낮으면 일부 연산이 변환되지 않는다(13 이상 권장)
  - 동적 배치/해상도를 쓰려면 dynamic_axes 를 지정해야 한다
  - 변환 후에는 반드시 PyTorch 결과와 수치를 비교해 검증한다

사용법:
    python 08_optimize/export_onnx.py --weights weights/plate_yolo.pt --imgsz 640
    python 08_optimize/export_onnx.py --mode torchvision --out weights/ssdlite.onnx
"""
from __future__ import annotations

import argparse
import pathlib
import sys


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="ONNX 내보내기")
    ap.add_argument("--mode", default="auto",
                    choices=["auto", "yolo", "torchvision"],
                    help="yolo: ultralytics export, torchvision: SSDLite 모델")
    ap.add_argument("--weights", default="weights/plate_yolo.pt")
    ap.add_argument("--out", default="weights/plate_yolo.onnx")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--opset", type=int, default=17,
                    help="ONNX opset 버전. 13 이상을 쓴다")
    ap.add_argument("--dynamic", action="store_true",
                    help="배치/해상도를 동적 축으로 내보낸다")
    ap.add_argument("--simplify", action="store_true",
                    help="그래프를 단순화한다(onnxsim 필요)")
    return ap.parse_args()


def export_yolo(args) -> pathlib.Path | None:
    """Ultralytics 모델을 ONNX 로 내보낸다."""
    from ultralytics import YOLO                # noqa: PLC0415

    model = YOLO(args.weights)
    # ultralytics 의 export 는 전처리/후처리 일부를 그래프에 포함할 수 있다.
    # 내보낸 파일은 가중치와 같은 디렉터리에 생성된다.
    path = model.export(format="onnx", imgsz=args.imgsz, opset=args.opset,
                        dynamic=args.dynamic, simplify=args.simplify,
                        device="cpu")
    return pathlib.Path(path)


def export_torchvision(args) -> pathlib.Path | None:
    """torchvision SSDLite 모델을 ONNX 로 내보낸다."""
    import torch                                # noqa: PLC0415

    # 07_mobilenet 은 숫자로 시작해 패키지 이름이 될 수 없으므로
    # 파일 경로로 직접 불러온다.
    model = None
    helper = (pathlib.Path(__file__).resolve().parents[1]
              / "07_mobilenet" / "mobilenet_detector.py")
    if helper.exists():
        import importlib.util                    # noqa: PLC0415
        spec = importlib.util.spec_from_file_location("mn_detector", helper)
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            model = mod.build_detector(pretrained=False)
    if model is None:
        from torchvision.models.detection import (  # noqa: PLC0415
            ssdlite320_mobilenet_v3_large)
        model = ssdlite320_mobilenet_v3_large(weights=None, num_classes=2)

    model.eval()
    dummy = torch.zeros(1, 3, 320, 320)
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 검출 모델은 출력이 가변 길이 리스트라서 변환이 까다롭다.
    # 입력만 동적 축으로 두고 출력은 고정 이름으로 지정한다.
    dynamic_axes = ({"images": {0: "batch"}} if args.dynamic else None)
    torch.onnx.export(
        model, (dummy,), str(out_path),
        input_names=["images"],
        output_names=["boxes", "scores", "labels"],
        opset_version=args.opset,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,               # 상수 연산을 미리 계산해 접는다
    )
    return out_path


def verify(onnx_path: pathlib.Path, imgsz: int) -> None:
    """내보낸 ONNX 모델을 ONNX Runtime 으로 한 번 실행해 본다."""
    try:
        import numpy as np                      # noqa: PLC0415
        import onnxruntime as ort               # noqa: PLC0415
    except ImportError:
        print("[안내] onnxruntime 이 없어 검증을 건너뛴다.")
        print("       pip install onnxruntime==1.22.1")
        return

    sess = ort.InferenceSession(str(onnx_path),
                                providers=["CPUExecutionProvider"])
    print("\n=== ONNX 모델 정보 ===")
    for i in sess.get_inputs():
        print(f"  입력  {i.name:12s} shape={i.shape} type={i.type}")
    for o in sess.get_outputs():
        print(f"  출력  {o.name:12s} shape={o.shape} type={o.type}")

    inp = sess.get_inputs()[0]
    shape = [d if isinstance(d, int) else (1 if j == 0 else imgsz)
             for j, d in enumerate(inp.shape)]
    x = np.zeros(shape, dtype=np.float32)
    outs = sess.run(None, {inp.name: x})
    print("\n=== 더미 추론 성공 ===")
    for o, v in zip(sess.get_outputs(), outs):
        arr = np.asarray(v)
        print(f"  {o.name:12s} shape={arr.shape} dtype={arr.dtype}")
    size_mb = onnx_path.stat().st_size / (1024 * 1024)
    print(f"\n  파일 크기: {size_mb:.2f} MB  경로: {onnx_path}")


def main() -> int:
    args = parse_args()

    mode = args.mode
    if mode == "auto":
        mode = "yolo" if pathlib.Path(args.weights).exists() else "torchvision"

    onnx_path: pathlib.Path | None = None
    if mode == "yolo":
        try:
            if not pathlib.Path(args.weights).exists():
                print(f"[안내] 가중치가 없다: {args.weights}")
                print("       먼저 05_train/train_yolo.py 로 학습한다.")
                return 0
            onnx_path = export_yolo(args)
        except ImportError:
            print("[안내] ultralytics 가 없다. pip install ultralytics==8.3.200")
            return 0
        except Exception as exc:                # noqa: BLE001
            print(f"[안내] 내보내기 실패: {exc!r}")
            print("       onnx 패키지가 필요할 수 있다: pip install onnx")
            return 0
    else:
        try:
            onnx_path = export_torchvision(args)
        except ImportError:
            print("[안내] torch / torchvision 이 없어 내보내기를 진행할 수 없다.")
            print("       pip install torch==2.8.0 torchvision==0.23.0")
            print("\nONNX 변환이 하는 일(요약)")
            print("  1) 파이썬 코드로 된 모델을 연산 그래프로 고정한다")
            print("  2) 상수 접기/연산 융합으로 그래프를 단순화한다")
            print("  3) 결과를 .onnx 한 파일로 저장해 런타임만으로 추론한다")
            return 0
        except Exception as exc:                # noqa: BLE001
            print(f"[안내] 내보내기 실패: {exc!r}")
            print("       opset 을 올리거나(--opset 18) 동적 축을 끄고 재시도한다.")
            return 0

    if onnx_path is None or not onnx_path.exists():
        print("[안내] ONNX 파일이 생성되지 않았다.")
        return 0

    print(f"내보내기 완료: {onnx_path}")
    verify(onnx_path, args.imgsz)
    print("\n다음 단계: 08_optimize/quantize_int8.py 로 INT8 양자화를 적용한다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
