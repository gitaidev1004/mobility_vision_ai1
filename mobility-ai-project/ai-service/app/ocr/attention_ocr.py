"""CNN + Attention Seq2Seq 번호판 OCR 모델 정의.

CTC 와 Attention 의 차이 (교재의 핵심 비교 포인트):

    | 항목 | CTC (CRNN) | Attention Seq2Seq |
    |---|---|---|
    | 정렬 가정 | **단조(monotonic)** — 입력과 출력 순서가 같다 | 자유 — 어디든 참조 가능 |
    | 조건부 독립 | 각 프레임이 독립 (언어모델 없음) | 직전 출력에 조건부 (암묵적 언어모델) |
    | 추론 | 1회 forward | 글자 수만큼 순차 decode (느림) |
    | 짧은 시퀀스 | 매우 강함 | 과적합 경향 |
    | 긴/왜곡 시퀀스 | 정렬이 깨짐 | 상대적으로 강건 |

    번호판은 **단조 정렬이 보장되는 짧은 시퀀스**라 CTC 가 기본값으로 더 적합하다.
    그런데도 Attention 을 함께 제공하는 이유:
        1. 심하게 기울어지거나 가려진 번호판에서는 Attention 이 더 잘 버틴다
           (프레임 정렬에 묶이지 않으므로).
        2. Attention 가중치를 시각화하면 "모델이 어디를 보고 이 글자를 읽었는지"
           알 수 있어, 오인식 원인 분석에 결정적인 도구가 된다.
        3. 앙상블 시 두 모델의 오류 패턴이 달라 상호 보완된다.

구조:
    인코더: CNN (CRNN 과 동일 백본) → (B, T=40, 512) → BiLSTM → (B, 40, 512)
    디코더: GRU(512) + Bahdanau(additive) attention → 글자별 출력
"""

from __future__ import annotations

from typing import Any

from app.ocr.charset import ATTENTION_VOCAB_SIZE, EOS_TOKEN, SOS_TOKEN

#: 입력 크기(CRNN 과 동일하게 맞춘다 — 두 모델이 같은 전처리를 공유해야 한다).
INPUT_HEIGHT: int = 32
INPUT_WIDTH: int = 160

#: 인코더/디코더 은닉 차원.
ENCODER_HIDDEN: int = 256
DECODER_HIDDEN: int = 256

#: 최대 디코딩 길이. 지역명 포함 번호판이 최대 10자이므로 여유를 둬 12 로 잡는다.
MAX_DECODE_LENGTH: int = 12


def _require_torch() -> Any:
    """torch 를 지연 import 한다.

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
            "PyTorch 가 설치되지 않았습니다. Attention OCR 을 쓰려면 "
            "`pip install torch==2.8.0` 를 실행하거나, 폴백 OCR 을 사용하세요."
        )
        raise RuntimeError(msg) from exc


def build_attention_ocr(
    vocab_size: int = ATTENTION_VOCAB_SIZE,
    *,
    input_channels: int = 1,
    encoder_hidden: int = ENCODER_HIDDEN,
    decoder_hidden: int = DECODER_HIDDEN,
    embed_dim: int = 128,
    dropout: float = 0.1,
) -> Any:
    """Attention Seq2Seq OCR 모델을 생성한다.

    Args:
        vocab_size: 어휘 크기(문자 + blank + SOS/EOS/PAD).
        input_channels: 입력 채널(1 = 그레이스케일).
        encoder_hidden: 인코더 BiLSTM 은닉 차원.
        decoder_hidden: 디코더 GRU 은닉 차원.
        embed_dim: 글자 임베딩 차원.
        dropout: 드롭아웃 비율.

    Returns:
        ``torch.nn.Module`` 인스턴스.

    Raises:
        RuntimeError: torch 미설치.
    """
    torch = _require_torch()
    nn = torch.nn
    functional = torch.nn.functional

    class BahdanauAttention(nn.Module):
        """Bahdanau(additive) attention.

        점수 계산: ``score = v^T tanh(W_enc h_enc + W_dec h_dec)``

        왜 dot-product 가 아니라 additive 인가?
            dot-product attention 은 인코더와 디코더의 은닉 차원이 같아야 하고,
            차원이 커지면 점수 분산이 커져 softmax 가 포화된다(그래서 Transformer 는
            √d 로 나눈다). additive 는 학습 가능한 선형 변환을 거치므로 차원이
            달라도 되고, 번호판처럼 짧은 시퀀스(T=40)에서는 연산 부담도 미미하다.
        """

        def __init__(self, enc_dim: int, dec_dim: int, attn_dim: int = 128) -> None:
            """Attention 모듈을 초기화한다.

            Args:
                enc_dim: 인코더 출력 차원.
                dec_dim: 디코더 은닉 차원.
                attn_dim: attention 내부 차원.
            """
            super().__init__()
            self.enc_proj = nn.Linear(enc_dim, attn_dim, bias=False)
            self.dec_proj = nn.Linear(dec_dim, attn_dim, bias=False)
            self.score = nn.Linear(attn_dim, 1, bias=False)

        def forward(self, encoder_out: Any, decoder_hidden_state: Any) -> tuple[Any, Any]:
            """문맥 벡터와 attention 가중치를 계산한다.

            Args:
                encoder_out: ``(B, T, enc_dim)`` 인코더 출력.
                decoder_hidden_state: ``(B, dec_dim)`` 디코더 은닉 상태.

            Returns:
                ``(문맥 벡터 (B, enc_dim), attention 가중치 (B, T))``.
            """
            # (B, T, attn) + (B, 1, attn) → 브로드캐스팅으로 모든 시점에 더한다.
            energy = torch.tanh(
                self.enc_proj(encoder_out) + self.dec_proj(decoder_hidden_state).unsqueeze(1)
            )
            scores = self.score(energy).squeeze(-1)        # (B, T)
            weights = functional.softmax(scores, dim=-1)   # (B, T)
            # 가중 평균으로 문맥 벡터를 만든다.
            context = torch.bmm(weights.unsqueeze(1), encoder_out).squeeze(1)
            return context, weights

    class AttentionOCR(nn.Module):
        """CNN 인코더 + Attention GRU 디코더 OCR 모델."""

        def __init__(self) -> None:
            """모델 계층을 구성한다."""
            super().__init__()
            self.vocab_size = int(vocab_size)
            self.input_height = INPUT_HEIGHT
            self.input_width = INPUT_WIDTH
            self.max_length = MAX_DECODE_LENGTH

            def conv_bn(in_ch: int, out_ch: int, kernel: int = 3, pad: int = 1) -> Any:
                """Conv-BN-ReLU 블록 생성 헬퍼."""
                return nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel, 1, pad, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                )

            # --- CNN 인코더 (CRNN 과 동일 설계: 높이 소멸, 너비 1/4) ---
            self.cnn = nn.Sequential(
                conv_bn(int(input_channels), 64),
                nn.MaxPool2d(2, 2),               # 16 x 80
                conv_bn(64, 128),
                nn.MaxPool2d(2, 2),               # 8 x 40
                conv_bn(128, 256),
                conv_bn(256, 256),
                nn.MaxPool2d((2, 1), (2, 1)),     # 4 x 40
                conv_bn(256, 512),
                nn.MaxPool2d((2, 1), (2, 1)),     # 2 x 40
                conv_bn(512, 512, kernel=2, pad=0),  # 1 x 40
            )

            enc_out_dim = int(encoder_hidden) * 2
            self.encoder_rnn = nn.LSTM(
                512, int(encoder_hidden), num_layers=1, bidirectional=True, batch_first=True
            )
            self.attention = BahdanauAttention(enc_out_dim, int(decoder_hidden))
            self.embedding = nn.Embedding(self.vocab_size, int(embed_dim))
            # 디코더 입력 = 직전 글자 임베딩 + 문맥 벡터
            self.decoder_rnn = nn.GRU(
                int(embed_dim) + enc_out_dim, int(decoder_hidden), num_layers=1, batch_first=True
            )
            self.dropout = nn.Dropout(float(dropout))
            # 출력층은 은닉 상태와 문맥을 함께 본다(deep output layer).
            # 문맥을 직접 연결하면 "지금 보고 있는 이미지 영역" 정보가 분류에
            # 바로 전달되어 수렴이 빨라진다.
            self.classifier = nn.Linear(int(decoder_hidden) + enc_out_dim, self.vocab_size)

        def encode(self, images: Any) -> Any:
            """이미지를 인코더 출력 시퀀스로 변환한다.

            Args:
                images: ``(B, C, H, W)`` 입력.

            Returns:
                ``(B, T, encoder_hidden*2)`` 인코더 출력.
            """
            features = self.cnn(images)
            if features.size(2) != 1:
                features = features.mean(dim=2, keepdim=True)
            features = features.squeeze(2).permute(0, 2, 1)  # (B, T, 512)
            encoder_out, _ = self.encoder_rnn(features)
            return encoder_out

        def forward(
            self,
            images: Any,
            targets: Any | None = None,
            *,
            teacher_forcing_ratio: float = 1.0,
            max_length: int | None = None,
        ) -> tuple[Any, Any]:
            """순전파(학습/추론 공용).

            **Teacher forcing** 이란?
                학습 중 디코더의 다음 입력으로 "모델의 직전 예측" 대신
                "정답의 직전 글자" 를 넣는 기법이다. 초기 학습에서 모델 예측이
                엉망일 때 오류가 누적되는 것을 막아 수렴을 크게 가속한다.
                단, 비율을 1.0 으로 끝까지 두면 추론 시(정답 없음) 성능이 떨어지는
                **exposure bias** 가 생긴다. 학습 후반에 비율을 0.5 로 낮추는
                스케줄링이 정석이다.

            Args:
                images: ``(B, C, H, W)`` 입력 이미지.
                targets: ``(B, L)`` 정답 인덱스(SOS 제외, EOS 포함). None 이면 추론 모드.
                teacher_forcing_ratio: 0~1. 1.0 이면 항상 정답을 다음 입력으로 쓴다.
                max_length: 추론 시 최대 디코딩 길이.

            Returns:
                ``(로짓 (B, L, vocab), attention 가중치 (B, L, T))``.
            """
            encoder_out = self.encode(images)
            batch = encoder_out.size(0)
            device = encoder_out.device

            steps = int(max_length or self.max_length)
            if targets is not None:
                steps = int(targets.size(1))

            # 디코더 초기 입력은 SOS 토큰.
            token = torch.full((batch, 1), int(SOS_TOKEN), dtype=torch.long, device=device)
            hidden = torch.zeros(1, batch, self.decoder_rnn.hidden_size, device=device)

            logits_list: list[Any] = []
            attn_list: list[Any] = []

            for step in range(steps):
                embedded = self.embedding(token)                    # (B, 1, E)
                context, weights = self.attention(encoder_out, hidden[-1])
                rnn_in = torch.cat([embedded, context.unsqueeze(1)], dim=-1)
                output, hidden = self.decoder_rnn(rnn_in, hidden)   # (B, 1, D)
                combined = torch.cat([output.squeeze(1), context], dim=-1)
                step_logits = self.classifier(self.dropout(combined))  # (B, vocab)

                logits_list.append(step_logits.unsqueeze(1))
                attn_list.append(weights.unsqueeze(1))

                if targets is not None and float(teacher_forcing_ratio) >= 1.0:
                    token = targets[:, step : step + 1]
                elif targets is not None and torch.rand(1).item() < float(teacher_forcing_ratio):
                    token = targets[:, step : step + 1]
                else:
                    token = step_logits.argmax(dim=-1, keepdim=True)

            return torch.cat(logits_list, dim=1), torch.cat(attn_list, dim=1)

        @torch.no_grad()
        def greedy_decode(self, images: Any, *, max_length: int | None = None) -> tuple[Any, Any, Any]:
            """추론: EOS 를 만나면 조기 종료하는 greedy 디코딩.

            Args:
                images: ``(B, C, H, W)`` 입력.
                max_length: 최대 길이.

            Returns:
                ``(인덱스 (B, L), 확률 (B, L), attention (B, L, T))``.
            """
            self.eval()
            encoder_out = self.encode(images)
            batch = encoder_out.size(0)
            device = encoder_out.device
            steps = int(max_length or self.max_length)

            token = torch.full((batch, 1), int(SOS_TOKEN), dtype=torch.long, device=device)
            hidden = torch.zeros(1, batch, self.decoder_rnn.hidden_size, device=device)
            # 이미 EOS 를 낸 샘플은 더 갱신하지 않도록 완료 플래그를 둔다.
            finished = torch.zeros(batch, dtype=torch.bool, device=device)

            tokens: list[Any] = []
            probs: list[Any] = []
            attns: list[Any] = []

            for _ in range(steps):
                embedded = self.embedding(token)
                context, weights = self.attention(encoder_out, hidden[-1])
                output, hidden = self.decoder_rnn(
                    torch.cat([embedded, context.unsqueeze(1)], dim=-1), hidden
                )
                step_logits = self.classifier(
                    torch.cat([output.squeeze(1), context], dim=-1)
                )
                step_probs = step_logits.softmax(dim=-1)
                best_prob, best_idx = step_probs.max(dim=-1)

                # 완료된 샘플은 PAD 로 채운다(이후 디코딩 결과에서 제거됨).
                best_idx = torch.where(finished, torch.full_like(best_idx, int(EOS_TOKEN)), best_idx)
                tokens.append(best_idx.unsqueeze(1))
                probs.append(best_prob.unsqueeze(1))
                attns.append(weights.unsqueeze(1))

                finished = finished | (best_idx == int(EOS_TOKEN))
                if bool(finished.all()):
                    break
                token = best_idx.unsqueeze(1)

            return (
                torch.cat(tokens, dim=1),
                torch.cat(probs, dim=1),
                torch.cat(attns, dim=1),
            )

    return AttentionOCR()


def attention_info(vocab_size: int = ATTENTION_VOCAB_SIZE) -> dict[str, Any]:
    """Attention OCR 구조 메타데이터를 반환한다(torch 없이 호출 가능).

    Args:
        vocab_size: 어휘 크기.

    Returns:
        구조 설명 dict.
    """
    conv_params = (
        1 * 64 * 9 + 128
        + 64 * 128 * 9 + 256
        + 128 * 256 * 9 + 512
        + 256 * 256 * 9 + 512
        + 256 * 512 * 9 + 1024
        + 512 * 512 * 4 + 1024
    )
    enc_lstm = 2 * 4 * (512 * ENCODER_HIDDEN + ENCODER_HIDDEN**2 + 2 * ENCODER_HIDDEN)
    enc_dim = ENCODER_HIDDEN * 2
    attn_params = enc_dim * 128 + DECODER_HIDDEN * 128 + 128
    embed = int(vocab_size) * 128
    dec_gru = 3 * ((128 + enc_dim) * DECODER_HIDDEN + DECODER_HIDDEN**2 + 2 * DECODER_HIDDEN)
    fc = (DECODER_HIDDEN + enc_dim) * int(vocab_size) + int(vocab_size)
    total = conv_params + enc_lstm + attn_params + embed + dec_gru + fc

    return {
        "architecture": "CNN encoder + Bahdanau Attention GRU decoder (Seq2Seq)",
        "input_shape": [1, INPUT_HEIGHT, INPUT_WIDTH],
        "encoder_hidden": ENCODER_HIDDEN,
        "decoder_hidden": DECODER_HIDDEN,
        "vocab_size": int(vocab_size),
        "max_decode_length": MAX_DECODE_LENGTH,
        "params_estimate": int(total),
        "size_fp32_mb": round(total * 4 / (1024 * 1024), 2),
        "notes": "CTC 대비 장점: 비단조 왜곡에 강건, attention 시각화로 설명 가능",
    }


def load_attention_ocr(
    weights_path: str, *, device: str = "cpu", vocab_size: int = ATTENTION_VOCAB_SIZE
) -> Any:
    """체크포인트에서 Attention OCR 을 로드한다.

    Args:
        weights_path: ``.pt`` 파일 경로.
        device: 디바이스.
        vocab_size: 어휘 크기(체크포인트에 있으면 그 값이 우선).

    Returns:
        eval 모드 모델.

    Raises:
        RuntimeError: torch 미설치 또는 로딩 실패.
    """
    torch = _require_torch()
    try:
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        msg = f"Attention OCR 체크포인트 로딩 실패: {weights_path} ({type(exc).__name__}: {exc})"
        raise RuntimeError(msg) from exc

    state = checkpoint
    vocab = int(vocab_size)
    if isinstance(checkpoint, dict):
        if "vocab_size" in checkpoint:
            vocab = int(checkpoint["vocab_size"])
        for key in ("model", "state_dict", "model_state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                state = checkpoint[key]
                break

    model = build_attention_ocr(vocab_size=vocab)
    try:
        model.load_state_dict(state, strict=True)
    except Exception as exc:
        msg = f"Attention OCR state_dict 불일치: {type(exc).__name__}: {exc}"
        raise RuntimeError(msg) from exc
    model.to(device).eval()
    return model


def decode_tokens(indices: list[int] | tuple[int, ...]) -> str:
    """Attention 디코더 출력 인덱스를 문자열로 변환한다.

    EOS 를 만나면 즉시 중단한다. EOS 이후 토큰은 패딩이므로 포함하면 쓰레기가 붙는다.

    Args:
        indices: 토큰 인덱스 시퀀스.

    Returns:
        복원된 문자열.
    """
    from app.ocr.charset import INDEX_TO_CHAR  # noqa: PLC0415 - 순환 import 방지

    out: list[str] = []
    for raw in indices:
        idx = int(raw)
        if idx == int(EOS_TOKEN):
            break
        ch = INDEX_TO_CHAR.get(idx)
        if ch is not None:
            out.append(ch)
    return "".join(out)


__all__ = [
    "DECODER_HIDDEN",
    "ENCODER_HIDDEN",
    "INPUT_HEIGHT",
    "INPUT_WIDTH",
    "MAX_DECODE_LENGTH",
    "attention_info",
    "build_attention_ocr",
    "decode_tokens",
    "load_attention_ocr",
]
