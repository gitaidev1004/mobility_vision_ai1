# -*- coding: utf-8 -*-
"""torchvision 의 SSDLite320 + MobileNetV3-Large 로 번호판 검출기를 정의한다.

SSDLite 는 SSD 의 검출 헤드에 Depthwise Separable Convolution 을 적용한
경량 변형이고, 백본은 MobileNetV3-Large 다. CPU 추론을 전제한 조합이다.

이 파일이 제공하는 것
  - build_detector(num_classes): 클래스 수를 번호판 전용으로 교체한 모델
  - freeze_backbone(model, n): 데이터가 적을 때 백본 앞단을 동결
  - count_params / model_size_mb: 모델 규모 측정
  - predict_boxes(model, image): 추론 후 (bbox, score) 목록 반환
  - 단독 실행 시: 모델을 만들어 구조/파라미터/더미 추론시간을 출력

torch/torchvision 이 없으면 설치 안내만 출력하고 정상 종료한다.
"""
from __future__ import annotations

import argparse
import sys
import time

# 번호판 단일 클래스 + 배경 = 2. torchvision 검출기는 배경을 0번으로 둔다
NUM_CLASSES = 2
CLASS_NAMES = {0: "__background__", 1: "plate"}

GUIDE = """\
[안내] torch / torchvision 이 필요하다.
  CPU 전용 설치 예시:
    pip install torch==2.8.0 torchvision==0.23.0 \\
        --index-url https://download.pytorch.org/whl/cpu
설치 후 다시 실행하면 모델 구조와 파라미터 수, 더미 추론시간을 볼 수 있다.
"""


def head_in_channels(model) -> list[int]:
    """분류 헤드 각 스케일의 입력 채널 수를 읽어 온다.

    SSDLite 의 예측 블록은 [Conv2dNormActivation(depthwise), Conv2d(1x1)] 구조다.
    즉 module_list[i][0][0] 이 그 스케일의 첫 Conv2d 다.
    torchvision 버전에 따라 중첩 구조가 다를 수 있으므로 재귀로 찾는다.
    """
    from torch import nn                        # noqa: PLC0415

    def first_conv(m):
        if isinstance(m, nn.Conv2d):
            return m
        for child in m.children():
            found = first_conv(child)
            if found is not None:
                return found
        return None

    chans = []
    for layer in model.head.classification_head.module_list:
        conv = first_conv(layer)
        if conv is None:
            raise RuntimeError("분류 헤드에서 Conv2d 를 찾지 못했다")
        # depthwise Conv 는 groups == in_channels 이므로 in_channels 가 곧 입력 채널이다
        chans.append(conv.in_channels)
    return chans


def build_detector(num_classes: int = NUM_CLASSES, pretrained: bool = True):
    """SSDLite320 + MobileNetV3-Large 검출기를 만든다.

    SSDLite320 은 입력을 320x320 고정 크기로 변환한 뒤 추론한다.
    앵커 배치가 이 해상도에 맞춰 설계되어 있으므로 해상도를 바꾸려면
    앵커 생성기까지 함께 조정해야 한다(여기서는 기본값을 유지한다).

    Args:
        num_classes: 배경을 포함한 클래스 수(번호판만 쓰면 2)
        pretrained: COCO 사전학습 가중치로 시작할지 여부(전이학습)

    Returns:
        분류 헤드를 번호판용으로 교체한 모델
    """
    from torch import nn                        # noqa: PLC0415
    from torchvision.models.detection import (  # noqa: PLC0415
        SSDLite320_MobileNet_V3_Large_Weights, ssdlite320_mobilenet_v3_large)
    from torchvision.models.detection.ssdlite import (  # noqa: PLC0415
        SSDLiteClassificationHead)

    weights = (SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
               if pretrained else None)
    model = ssdlite320_mobilenet_v3_large(weights=weights)

    # COCO 91 클래스 헤드를 번호판 2 클래스 헤드로 교체한다.
    # 백본과 특징 추출부는 그대로 재사용하므로 전이학습이 된다.
    in_channels = head_in_channels(model)
    num_anchors = model.anchor_generator.num_anchors_per_location()
    model.head.classification_head = SSDLiteClassificationHead(
        in_channels, num_anchors, num_classes, nn.BatchNorm2d)
    return model


def freeze_backbone(model, n_blocks: int = 6) -> int:
    """백본 앞쪽 n_blocks 개 하위 블록의 가중치를 동결한다.

    번호판 데이터가 수천 장 수준이면 백본 전체를 학습시키면 과적합되기 쉽다.
    저수준 특징(에지, 질감)은 COCO 사전학습으로 충분하므로 앞단을 고정한다.

    torchvision 의 SSDLite 백본은 features 가 2단(C4 이전 / C4 이후)으로
    묶여 있으므로, 실제 MobileNet 블록은 features[0] 안에 들어 있다.

    Returns:
        동결된 파라미터 개수
    """
    features = model.backbone.features
    # 첫 단이 Sequential 이면 그 안의 블록 단위로 동결한다
    target = features[0] if len(list(features.children())) and \
        len(list(features[0].children())) > n_blocks else features
    frozen = 0
    for i, block in enumerate(target.children()):
        if i >= n_blocks:
            break
        for p in block.parameters():
            p.requires_grad = False
            frozen += p.numel()
    return frozen


def count_params(model) -> tuple[int, int]:
    """(전체 파라미터, 학습 대상 파라미터) 개수를 돌려준다."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def model_size_mb(model) -> float:
    """파라미터와 버퍼가 차지하는 메모리 크기(MB)."""
    nbytes = sum(p.numel() * p.element_size() for p in model.parameters())
    nbytes += sum(b.numel() * b.element_size() for b in model.buffers())
    return nbytes / (1024 * 1024)


def predict_boxes(model, image, score_thr: float = 0.25):
    """단일 이미지를 추론해 (bbox, score, label) 목록을 돌려준다.

    Args:
        model: build_detector 로 만든 모델(eval 모드여야 한다)
        image: (3, H, W) 범위 0~1 의 torch.Tensor
        score_thr: 이 점수 미만은 버린다
    """
    import torch                                # noqa: PLC0415

    model.eval()
    with torch.no_grad():                       # 추론에는 기울기가 필요 없다
        out = model([image])[0]
    rows = []
    for box, score, label in zip(out["boxes"], out["scores"], out["labels"]):
        s = float(score)
        if s < score_thr:
            continue
        rows.append(([float(v) for v in box.tolist()], s,
                     CLASS_NAMES.get(int(label), str(int(label)))))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="MobileNet 기반 검출기 정의/점검")
    ap.add_argument("--no-pretrained", action="store_true",
                    help="사전학습 가중치를 내려받지 않는다(오프라인)")
    ap.add_argument("--freeze", type=int, default=0,
                    help="백본 앞 N개 블록을 동결한다")
    ap.add_argument("--size", type=int, default=320,
                    help="더미 입력 해상도(모델 내부는 320 으로 재조정된다)")
    ap.add_argument("--runs", type=int, default=5, help="지연시간 측정 횟수")
    args = ap.parse_args()

    try:
        import torch                            # noqa: PLC0415
        import torchvision                      # noqa: PLC0415
    except ImportError:
        print(GUIDE)
        return 0

    print(f"torch {torch.__version__} / torchvision {torchvision.__version__}")
    torch.set_num_threads(4)                    # CPU 스레드 수를 고정해 측정을 재현한다

    try:
        model = build_detector(NUM_CLASSES, pretrained=not args.no_pretrained)
    except Exception as exc:                    # noqa: BLE001
        # 사전학습 가중치 다운로드가 막힌 환경에서도 진행할 수 있게 한다
        print(f"[안내] 사전학습 가중치를 받을 수 없다({exc.__class__.__name__}).")
        print("       무작위 초기화 모델을 만들어 구조만 확인한다.")
        try:
            model = build_detector(NUM_CLASSES, pretrained=False)
        except Exception as exc2:               # noqa: BLE001
            print(f"[안내] 모델 생성 실패: {exc2!r}")
            print("       torchvision 버전을 0.23.0 으로 맞춰 다시 시도한다.")
            return 0

    if args.freeze > 0:
        frozen = freeze_backbone(model, args.freeze)
        print(f"백본 앞 {args.freeze} 블록 동결: 파라미터 {frozen:,} 개 고정")

    total, trainable = count_params(model)
    print("\n=== 모델 규모 ===")
    print(f"  클래스 수        : {NUM_CLASSES} (배경 + plate)")
    print("  내부 입력 해상도 : 320x320 (SSDLite320 고정)")
    print(f"  전체 파라미터    : {total:,} ({total / 1e6:.2f}M)")
    print(f"  학습 대상        : {trainable:,} ({trainable / 1e6:.2f}M)")
    print(f"  FP32 모델 크기   : {model_size_mb(model):.1f} MB")

    # 더미 이미지로 추론시간을 측정한다
    model.eval()
    dummy = torch.zeros(3, args.size, args.size)
    with torch.no_grad():
        model([dummy])                          # 워밍업(첫 호출은 초기화 비용 포함)
        model([dummy])
    samples = []
    with torch.no_grad():
        for _ in range(args.runs):
            t0 = time.perf_counter()
            model([dummy])
            samples.append((time.perf_counter() - t0) * 1000.0)
    samples.sort()
    p50 = samples[len(samples) // 2]
    print("\n=== CPU 추론시간 (더미 입력) ===")
    print(f"  측정 {args.runs} 회  min={samples[0]:.1f}ms "
          f"p50={p50:.1f}ms max={samples[-1]:.1f}ms")
    print(f"  환산 FPS(단일 스레드 배치1): {1000.0 / p50:.1f}")
    print("  ※ 예시 측정값이다. CPU 모델/스레드 수/해상도에 따라 크게 달라진다.")

    rows = predict_boxes(model, dummy, score_thr=0.25)
    print(f"\n더미 입력 검출 결과: {len(rows)} 건 "
          f"(무작위 초기화 모델이면 0 건이 정상이다)")
    print("\n학습은 torchvision 검출 학습 루프를 쓰거나,")
    print("YOLO 와 동일 조건에서 비교하려면 2.5절 학습 설정을 맞춰 구성한다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
