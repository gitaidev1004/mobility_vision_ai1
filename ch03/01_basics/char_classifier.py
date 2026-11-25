# -*- coding: utf-8 -*-
"""단일 문자 분류 CNN 정의와 더미 추론.

시퀀스 인식(CRNN)으로 넘어가기 전에, 문자 한 글자를 분류하는 가장 단순한 형태를
먼저 만들어 본다. 입력은 32x32 회색조 한 글자, 출력은 문자셋 크기만큼의 로짓이다.

실행:  python 01_basics/char_classifier.py
torch 가 없으면 설치 안내만 출력하고 정상 종료한다.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "03_dataset"))
from charset import CHARSET, INDEX_TO_CHAR  # noqa: E402

try:
    import torch
    from torch import nn
except ImportError:
    print("[안내] torch 가 설치되어 있지 않다.")
    print("       pip install torch --index-url https://download.pytorch.org/whl/cpu")
    print(f"       참고: 분류 대상 문자 수 = {len(CHARSET)}")
    raise SystemExit(0)


class CharClassifier(nn.Module):
    """32x32 회색조 문자 이미지를 문자셋 클래스로 분류하는 소형 CNN.

    설계 근거
      - 3x3 합성곱 3단: 번호판 글자는 획이 굵고 단순해 깊은 망이 필요하지 않다.
      - BatchNorm: 야간/역광 이미지가 섞이면 입력 분포가 크게 흔들린다. 정규화가
        학습 안정성에 직접 기여한다.
      - GlobalAvgPool: Flatten 대신 쓰면 파라미터가 급감하고 CPU 추론이 빨라진다.
    """

    def __init__(self, num_classes: int = len(CHARSET)) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                              # 32x32 -> 16x16
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                              # 16x16 -> 8x8
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),                      # 8x8 -> 1x1
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """(N, 1, 32, 32) -> (N, num_classes) 로짓을 반환한다."""
        return self.classifier(self.features(x))


def count_parameters(model: nn.Module) -> int:
    """학습 대상 파라미터 개수를 센다(모델 크기 가늠용)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main() -> None:
    torch.manual_seed(0)
    model = CharClassifier()
    model.eval()

    n_params = count_parameters(model)
    print(f"클래스 수        : {len(CHARSET)}")
    print(f"파라미터 수      : {n_params:,} 개")
    print(f"float32 모델 크기: 약 {n_params * 4 / 1024 / 1024:.2f} MB")

    # 더미 입력 4장으로 shape 변화를 확인한다(학습되지 않은 모델이므로 결과는 무의미).
    dummy = torch.randn(4, 1, 32, 32)
    with torch.no_grad():
        logits = model(dummy)
        probs = torch.softmax(logits, dim=-1)
        conf, pred = probs.max(dim=-1)

    print(f"\n입력 shape : {tuple(dummy.shape)}")
    print(f"출력 shape : {tuple(logits.shape)}")
    print("\n[더미 추론 결과] 학습 전이므로 예측은 무작위에 가깝다.")
    for i in range(dummy.shape[0]):
        ch = INDEX_TO_CHAR.get(int(pred[i]), "?")
        print(f"  샘플 {i}: 예측 '{ch}' (클래스 {int(pred[i])}), "
              f"신뢰도 {float(conf[i]):.4f}")

    print("\n단일 문자 분류는 문자 분할이 완벽해야 성립한다. 번호판은 글자가 붙거나")
    print("가려지는 경우가 많아 3.4절부터는 분할 없이 전체를 읽는 CRNN 으로 전환한다.")


if __name__ == "__main__":
    main()
