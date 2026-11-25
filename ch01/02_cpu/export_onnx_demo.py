# -*- coding: utf-8 -*-
"""ONNX 내보내기 최소 예시.

학습된 가중치 없이, 작은 합성곱 모델을 그 자리에서 만들어 내보낸다.
torch.onnx.export 의 인자 구성과 동적 축 지정 방법을 확인하는 것이 목적이다.
PyTorch 가 없으면 안내만 출력하고 정상 종료한다.
실행:  python 02_cpu/export_onnx_demo.py
"""
from __future__ import annotations

import os

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

OUT_DIR = "models"
OUT_PATH = os.path.join(OUT_DIR, "demo_backbone.onnx")


def build_demo_model():
    """내보내기 예시용 소형 백본. 실제 검출 모델의 앞단을 모사한다."""
    return nn.Sequential(
        nn.Conv2d(3, 16, 3, stride=2, padding=1),
        nn.BatchNorm2d(16),           # ONNX 최적화 시 합성곱과 병합될 수 있다
        nn.ReLU(),
        nn.Conv2d(16, 32, 3, stride=2, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(32, 8),
    ).eval()


def export(path: str) -> None:
    """모델을 ONNX 로 내보낸다. 배치와 해상도를 동적 축으로 지정한다."""
    model = build_demo_model()
    # 예시 입력: 그래프 추적에 사용되며 실제 값은 결과에 영향을 주지 않는다.
    dummy = torch.randn(1, 3, 320, 320)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        path,
        input_names=["images"],       # 런타임에서 입력을 지칭할 이름
        output_names=["features"],    # 출력 이름
        dynamic_axes={                # 배치와 해상도를 가변으로 선언한다
            "images": {0: "batch", 2: "height", 3: "width"},
            "features": {0: "batch"},
        },
        opset_version=17,             # 런타임이 지원하는 연산 집합 버전
        do_constant_folding=True,     # 상수로 계산 가능한 부분을 미리 접는다
    )


def verify(path: str) -> None:
    """내보낸 파일을 ONNX 런타임으로 불러 추론 한 번을 수행한다."""
    try:
        import numpy as np
        import onnxruntime as ort
    except Exception:
        print("onnxruntime 이 없어 검증은 생략한다.")
        return
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    x = np.random.default_rng(0).standard_normal(
        (1, 3, 256, 256)).astype("float32")
    out = sess.run(["features"], {"images": x})[0]
    print(f"런타임 검증 통과. 입력 256x256 -> 출력 {out.shape}")
    print("동적 축을 지정했기 때문에 320 이외의 해상도도 받아들인다.")


def main() -> int:
    if not HAS_TORCH:
        print("PyTorch 가 설치되지 않았다. 1.4절 (3) 을 참고해 설치한다.")
        print("설치 후 다시 실행하면 ONNX 파일이 생성된다.")
        return 0
    export(OUT_PATH)
    size_kb = os.path.getsize(OUT_PATH) / 1024.0
    print(f"내보내기 완료: {OUT_PATH} ({size_kb:.1f} KB)")
    verify(OUT_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
