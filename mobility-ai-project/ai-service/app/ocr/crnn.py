"""CRNN(CNN + BiLSTM + CTC) 번호판 OCR 모델 정의.

CRNN 이 번호판 OCR 의 표준 선택인 이유:
    1. **문자 분할이 필요 없다.** 연결요소 분할은 한글 자모가 떨어져 있거나
       문자가 붙으면 실패한다(이 프로젝트 폴백 OCR 이 실제로 겪은 문제다).
       CRNN 은 이미지를 가로 방향 시퀀스로 보고 CTC 로 정렬을 학습하므로
       분할 단계를 완전히 제거한다.
    2. **가변 길이를 다룬다.** 번호판은 7자/8자가 섞여 있고 지역명이 붙으면 9~10자다.
       CTC 는 출력 길이를 고정하지 않는다.
    3. **가볍다.** 번호판은 문자 종류가 78개뿐이라 작은 모델로 충분하다.
       CPU 에서 10~20ms 로 동작해야 하는 게이트 환경에 적합하다.

구조:
    입력 (B, 1, 32, 160) 그레이스케일
      → CNN 7층: 높이를 32 → 1 로, 너비를 160 → 40 으로 줄인다
      → (B, 40, 512) 시퀀스
      → BiLSTM 2층 (hidden 256, 양방향) → (B, 40, 512)
      → Linear → (B, 40, 78) 클래스 로짓
      → CTC

높이를 1 까지 줄이고 너비는 1/4 만 줄이는 비대칭 풀링이 핵심이다.
세로 정보는 "문자 모양" 으로 CNN 채널에 압축되고, 가로 해상도는 "몇 번째 문자" 를
구분하기 위해 남겨 둔다. 문자 하나당 약 5 프레임(40/8자)이 할당되어
CTC 의 blank 삽입에 충분한 여유가 생긴다.

torch 가 없으면 이 모듈을 import 해도 예외가 나지 않고, 모델 생성 시점에만
``RuntimeError`` 를 던진다(상위 계층이 폴백으로 전환한다).
"""

from __future__ import annotations

from typing import Any

from app.ocr.charset import NUM_CLASSES

#: 입력 표준 크기. 학습과 추론이 반드시 일치해야 한다.
INPUT_HEIGHT: int = 32
INPUT_WIDTH: int = 160

#: CNN 출력 시퀀스 길이(= INPUT_WIDTH // 4). 아래 풀링 설계에서 유도된다.
SEQUENCE_LENGTH: int = INPUT_WIDTH // 4

#: BiLSTM 은닉 차원.
HIDDEN_SIZE: int = 256


def _require_torch() -> Any:
    """torch 를 지연 import 한다.

    모듈 최상단에서 import 하지 않는 이유: torch 는 2GB 가 넘고, 설치되지 않은
    환경에서 ``app.ocr.crnn`` 을 import 하는 것만으로 서비스가 죽으면
    "가중치 없이도 동작" 원칙이 깨진다.

    Returns:
        torch 모듈.

    Raises:
        RuntimeError: torch 미설치.
    """
    try:
        import torch  # noqa: PLC0415

        return torch
    except ImportError as exc:
        msg = (
            "PyTorch 가 설치되지 않았습니다. CRNN 을 쓰려면 "
            "`pip install torch==2.8.0` 를 실행하거나, 폴백 OCR 을 사용하세요."
        )
        raise RuntimeError(msg) from exc


def build_crnn(
    num_classes: int = NUM_CLASSES,
    *,
    input_channels: int = 1,
    hidden_size: int = HIDDEN_SIZE,
    lstm_layers: int = 2,
    dropout: float = 0.1,
) -> Any:
    """CRNN 모델 인스턴스를 생성한다.

    torch 를 지연 import 하기 위해 클래스 정의를 함수 안에 둔다.
    (모듈 최상단에 ``class CRNN(nn.Module)`` 을 두면 import 시 torch 가 필요해진다.)

    Args:
        num_classes: 출력 클래스 수(기본 = 문자셋 + blank).
        input_channels: 입력 채널(1 = 그레이스케일, 3 = BGR).
            그레이스케일을 기본으로 쓰는 이유: 번호판 문자 판독에 색 정보는
            거의 기여하지 않으면서 연산량을 3배로 늘린다.
        hidden_size: BiLSTM 은닉 차원.
        lstm_layers: BiLSTM 층 수.
        dropout: LSTM 층 간 드롭아웃.

    Returns:
        ``torch.nn.Module`` 인스턴스.

    Raises:
        RuntimeError: torch 미설치.
    """
    torch = _require_torch()
    nn = torch.nn

    class ConvBlock(nn.Module):
        """Conv → BatchNorm → ReLU 묶음.

        BatchNorm 을 쓰는 이유: 번호판 영상은 조명에 따라 밝기 분포가 크게 변한다.
        BatchNorm 이 각 층의 입력 분포를 정규화해 주면 조명 변화에 둔감해지고
        학습이 훨씬 빨리 수렴한다.
        """

        def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, stride: int = 1, pad: int = 1) -> None:
            """합성곱 블록을 초기화한다.

            Args:
                in_ch: 입력 채널 수.
                out_ch: 출력 채널 수.
                kernel: 커널 크기.
                stride: 스트라이드.
                pad: 패딩.
            """
            super().__init__()
            # BatchNorm 이 bias 역할을 하므로 Conv 의 bias 는 끈다(파라미터 절약).
            self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride, pad, bias=False)
            self.bn = nn.BatchNorm2d(out_ch)
            self.act = nn.ReLU(inplace=True)

        def forward(self, x: Any) -> Any:
            """순전파."""
            return self.act(self.bn(self.conv(x)))

    class CRNN(nn.Module):
        """CNN + BiLSTM + CTC 기반 번호판 인식 모델.

        입력 ``(B, C, 32, 160)`` → 출력 ``(B, 40, num_classes)`` 로짓.

        CTC 손실을 쓸 때는 ``forward`` 출력을 ``(T, B, C)`` 로 permute 하고
        ``log_softmax`` 를 적용해야 한다. ``forward`` 가 ``(B, T, C)`` 를 돌려주는
        이유는 추론 코드와 ONNX 내보내기가 batch-first 를 선호하기 때문이다.
        """

        def __init__(self) -> None:
            """모델 계층을 구성한다."""
            super().__init__()
            self.num_classes = int(num_classes)
            self.input_height = INPUT_HEIGHT
            self.input_width = INPUT_WIDTH

            # --- CNN: 높이 32 → 1, 너비 160 → 40 ---
            # 풀링 설계 의도가 이 모델의 핵심이다.
            #   stage1: (2,2) → 16 x 80   (양방향 절반)
            #   stage2: (2,2) →  8 x 40   (양방향 절반, 여기서 너비 축소 종료)
            #   stage3: (2,1) →  4 x 40   (높이만 절반)
            #   stage4: (2,1) →  2 x 40   (높이만 절반)
            #   stage5: (2,1) →  1 x 40   (높이 소멸 → 순수 시퀀스)
            # 너비를 40 에 고정하는 이유: 8자 번호판에서 문자당 5프레임이 확보되어
            # CTC 가 "문자-blank-문자" 패턴을 표현할 공간이 생긴다. 너비를 더 줄이면
            # 인접 문자가 같은 프레임에 겹쳐 중복 제거 규칙이 글자를 삼킨다.
            self.cnn = nn.Sequential(
                ConvBlock(int(input_channels), 64),
                nn.MaxPool2d(2, 2),               # 16 x 80
                ConvBlock(64, 128),
                nn.MaxPool2d(2, 2),               # 8 x 40
                ConvBlock(128, 256),
                ConvBlock(256, 256),
                nn.MaxPool2d((2, 1), (2, 1)),     # 4 x 40
                ConvBlock(256, 512),
                ConvBlock(512, 512),
                nn.MaxPool2d((2, 1), (2, 1)),     # 2 x 40
                # 마지막은 2x2 커널로 높이 2 를 1 로 만든다(패딩 0).
                ConvBlock(512, 512, kernel=2, stride=1, pad=0),
            )

            # --- RNN: 양방향 LSTM ---
            # 왜 양방향인가? 번호판 문자는 좌우 문맥이 모두 유용하다. 예를 들어
            # '가' 를 판정할 때 앞에 숫자 2개가 있고 뒤에 숫자 4개가 있다는 정보가
            # 들어오면 "여기는 용도문자 자리" 라는 사전지식이 암묵적으로 학습된다.
            self.rnn = nn.LSTM(
                input_size=512,
                hidden_size=int(hidden_size),
                num_layers=int(lstm_layers),
                bidirectional=True,
                batch_first=True,
                dropout=float(dropout) if int(lstm_layers) > 1 else 0.0,
            )

            # 양방향이므로 출력 차원은 hidden_size x 2.
            self.classifier = nn.Linear(int(hidden_size) * 2, self.num_classes)

        def forward(self, x: Any) -> Any:
            """순전파.

            Args:
                x: ``(B, C, H, W)`` 입력 텐서. H=32, W=160 권장.

            Returns:
                ``(B, T, num_classes)`` 로짓. T = W // 4.
            """
            features = self.cnn(x)                      # (B, 512, 1, T)
            # 높이 차원을 제거한다. squeeze(2) 대신 mean 을 쓰면 H>1 인 입력에도
            # 안전하지만, 설계상 H=1 이 되어야 정상이므로 assert 대신 mean 으로
            # 방어한다(ONNX 내보내기에서 동적 shape 문제를 피하는 실용적 선택).
            if features.size(2) != 1:
                features = features.mean(dim=2, keepdim=True)
            features = features.squeeze(2)              # (B, 512, T)
            sequence = features.permute(0, 2, 1)        # (B, T, 512)
            recurrent, _ = self.rnn(sequence)           # (B, T, 2H)
            return self.classifier(recurrent)           # (B, T, C)

        def log_probs_for_ctc(self, x: Any) -> Any:
            """CTC 손실에 바로 넣을 수 있는 ``(T, B, C)`` log-softmax 를 반환한다.

            ``nn.CTCLoss`` 는 time-first 입력을 요구한다. 이 변환을 모델 쪽에
            둬서 학습 스크립트가 permute 순서를 틀릴 가능성을 없앤다.

            Args:
                x: ``(B, C, H, W)`` 입력.

            Returns:
                ``(T, B, num_classes)`` 로그 확률.
            """
            logits = self.forward(x)                     # (B, T, C)
            return logits.permute(1, 0, 2).log_softmax(2)

        @torch.no_grad()
        def predict_probs(self, x: Any) -> Any:
            """추론용: ``(B, T, C)`` softmax 확률을 반환한다.

            Args:
                x: 입력 텐서.

            Returns:
                확률 텐서.
            """
            self.eval()
            return self.forward(x).softmax(dim=-1)

    return CRNN()


def crnn_info(num_classes: int = NUM_CLASSES) -> dict[str, Any]:
    """CRNN 구조 메타데이터를 반환한다(torch 없이도 호출 가능).

    ``/api/v1/models/info`` 가 torch 미설치 환경에서도 모델 스펙을 보고할 수
    있어야 하므로, 구조 정보는 torch 없이 계산한다.

    Args:
        num_classes: 출력 클래스 수.

    Returns:
        입력 크기, 시퀀스 길이, 파라미터 추정치 등.
    """
    # 파라미터 수는 구조에서 해석적으로 계산한다(torch 없이도 정확하다).
    conv_params = (
        1 * 64 * 9 + 64 * 2            # ConvBlock(1,64) + BN
        + 64 * 128 * 9 + 128 * 2
        + 128 * 256 * 9 + 256 * 2
        + 256 * 256 * 9 + 256 * 2
        + 256 * 512 * 9 + 512 * 2
        + 512 * 512 * 9 + 512 * 2
        + 512 * 512 * 4 + 512 * 2      # kernel=2 → 4
    )
    # LSTM: 층당 4 x (input x hidden + hidden x hidden + 2 x hidden), 양방향 x2
    lstm_l1 = 2 * 4 * (512 * HIDDEN_SIZE + HIDDEN_SIZE * HIDDEN_SIZE + 2 * HIDDEN_SIZE)
    lstm_l2 = 2 * 4 * (
        (HIDDEN_SIZE * 2) * HIDDEN_SIZE + HIDDEN_SIZE * HIDDEN_SIZE + 2 * HIDDEN_SIZE
    )
    fc_params = (HIDDEN_SIZE * 2) * int(num_classes) + int(num_classes)
    total = conv_params + lstm_l1 + lstm_l2 + fc_params

    return {
        "architecture": "CRNN (7-layer CNN + 2-layer BiLSTM + CTC)",
        "input_shape": [1, INPUT_HEIGHT, INPUT_WIDTH],
        "sequence_length": SEQUENCE_LENGTH,
        "hidden_size": HIDDEN_SIZE,
        "num_classes": int(num_classes),
        "params_estimate": int(total),
        "size_fp32_mb": round(total * 4 / (1024 * 1024), 2),
        "size_int8_mb": round(total * 1 / (1024 * 1024), 2),
    }


def load_crnn(weights_path: str, *, device: str = "cpu", num_classes: int = NUM_CLASSES) -> Any:
    """체크포인트에서 CRNN 을 로드한다.

    체크포인트 형식을 두 가지 모두 허용한다:
        - ``{"model": state_dict, "num_classes": int, ...}`` (``train_crnn`` 이 저장하는 형식)
        - ``state_dict`` 단독
    교재 독자가 외부에서 받은 가중치도 그대로 쓸 수 있게 하기 위함이다.

    Args:
        weights_path: ``.pt`` 파일 경로.
        device: ``"cpu"`` 또는 ``"cuda"``.
        num_classes: 출력 클래스 수(체크포인트에 기록돼 있으면 그 값이 우선).

    Returns:
        eval 모드의 모델.

    Raises:
        RuntimeError: torch 미설치 또는 로딩 실패.
    """
    torch = _require_torch()
    try:
        # weights_only=True 로 임의 코드 실행(pickle 취약점)을 막는다.
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        msg = f"CRNN 체크포인트 로딩 실패: {weights_path} ({type(exc).__name__}: {exc})"
        raise RuntimeError(msg) from exc

    state = checkpoint
    classes = int(num_classes)
    if isinstance(checkpoint, dict):
        if "num_classes" in checkpoint:
            classes = int(checkpoint["num_classes"])
        for key in ("model", "state_dict", "model_state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                state = checkpoint[key]
                break

    model = build_crnn(num_classes=classes)
    try:
        model.load_state_dict(state, strict=True)
    except Exception as exc:
        msg = f"CRNN state_dict 불일치: {type(exc).__name__}: {exc}"
        raise RuntimeError(msg) from exc

    model.to(device).eval()
    return model


__all__ = [
    "HIDDEN_SIZE",
    "INPUT_HEIGHT",
    "INPUT_WIDTH",
    "SEQUENCE_LENGTH",
    "build_crnn",
    "crnn_info",
    "load_crnn",
]
