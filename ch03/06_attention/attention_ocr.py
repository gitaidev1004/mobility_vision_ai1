# -*- coding: utf-8 -*-
"""CNN 인코더 + GRU 디코더 + Bahdanau Attention 기반 번호판 OCR 모델.

구조
    입력 이미지 (1, 32, 160)  — 회색조, 높이 32px 고정
      └ CNN 인코더            → 특징맵 (C, 1, W') → 시퀀스 (W', C)
          └ 양방향 GRU        → 인코더 은닉 상태 H = (W', 2*hidden)
              └ Bahdanau Attention → 각 출력 스텝에서 H 를 가중 합
                  └ GRU 디코더 → 문자 1자 예측(자기회귀)

CRNN + CTC 와 다른 점
  - CTC 는 입력 프레임마다 문자를 예측한 뒤 중복/blank 를 제거한다.
  - Attention 디코더는 출력 길이만큼만 반복하며, 매 스텝에서 입력의
    어느 위치를 볼지 스스로 정한다. 따라서 문자 수를 모델이 결정한다.

문자셋은 3.3절에서 만든 ch03/03_dataset/charset.py 를 재사용한다.
가중치 파일이 없어도 랜덤 초기화 상태로 구조와 출력 형태를 확인할 수 있다.
"""
from __future__ import annotations

import os
import sys

# 3.3절의 문자셋 정의를 재사용한다. 경로가 다르면 CHARSET 를 직접 넘겨도 된다.
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "03_dataset"))

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    print("[안내] PyTorch 가 설치되어 있지 않다.")
    print("       아래 명령으로 CPU 버전을 설치한 뒤 다시 실행한다.")
    print("       pip install torch --index-url "
          "https://download.pytorch.org/whl/cpu")
    raise SystemExit(0)

try:
    from charset import CHARSET            # 3.3절에서 정의한 문자셋
except ImportError:
    # 교재 예제를 단독 실행할 때를 위한 대체 정의(3.3절 정의와 동일 구성).
    CHARSET = ("0123456789"
               "가나다라마거너더러머고노도로모구누두루무"
               "버보부서소수어오우저조주아바사자배허하호")
    print("[안내] charset.py 를 찾지 못해 내장 문자셋으로 대체한다.")

# 특수 토큰. Attention 디코더는 시작/종료 토큰이 반드시 필요하다.
PAD, SOS, EOS = 0, 1, 2
SPECIALS = ["<pad>", "<sos>", "<eos>"]
ITOS = SPECIALS + list(CHARSET)
STOI = {c: i for i, c in enumerate(ITOS)}
NUM_CLASSES = len(ITOS)
MAX_LEN = 11            # 서울34나5678(9자) + EOS 여유


class CnnEncoder(nn.Module):
    """번호판 이미지를 가로 방향 특징 시퀀스로 바꾸는 CNN 인코더.

    높이는 단계적으로 1까지 줄이고 가로는 1/4 만 줄여, 문자 하나가
    여러 열에 걸쳐 남도록 만든다. 160px 입력이면 40열 시퀀스가 된다.
    """

    def __init__(self, in_ch: int = 1, out_ch: int = 256) -> None:
        super().__init__()
        self.body = nn.Sequential(
            # (1,32,160) -> (64,16,80)
            nn.Conv2d(in_ch, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            # -> (128,8,40)
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            # 높이만 줄인다 -> (256,4,40)
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),
            # -> (256,2,40)
            nn.Conv2d(256, out_ch, 3, 1, 1), nn.BatchNorm2d(out_ch),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),
            # -> (out_ch,1,40)
            nn.Conv2d(out_ch, out_ch, (2, 3), 1, (0, 1)),
            nn.BatchNorm2d(out_ch), nn.ReLU(True),
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """(B,1,32,160) -> (B,W',C) 형태의 시퀀스를 돌려준다."""
        f = self.body(x)                       # (B,C,1,W')
        if f.size(2) != 1:                     # 안전장치: 높이를 평균으로 눌러준다
            f = f.mean(dim=2, keepdim=True)
        f = f.squeeze(2)                       # (B,C,W')
        return f.permute(0, 2, 1).contiguous()  # (B,W',C)


class BahdanauAttention(nn.Module):
    """Bahdanau(additive) Attention.

    점수 e_i = v^T tanh(W_h h_i + W_s s)  로 계산한다.
    h_i 는 인코더의 i 번째 위치 표현, s 는 디코더의 현재 상태다.
    곱셈(dot-product) 방식보다 차원이 달라도 쓰기 쉽고,
    번호판처럼 짧은 시퀀스에서는 연산 비용 차이가 거의 없다.
    """

    def __init__(self, enc_dim: int, dec_dim: int, attn_dim: int = 128) -> None:
        super().__init__()
        self.Wh = nn.Linear(enc_dim, attn_dim, bias=False)
        self.Ws = nn.Linear(dec_dim, attn_dim, bias=False)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, enc: "torch.Tensor", dec_state: "torch.Tensor"):
        """enc (B,T,E), dec_state (B,D) -> (context (B,E), weights (B,T))."""
        proj = self.Wh(enc) + self.Ws(dec_state).unsqueeze(1)   # (B,T,A)
        score = self.v(torch.tanh(proj)).squeeze(-1)            # (B,T)
        weight = F.softmax(score, dim=1)                        # (B,T)
        context = torch.bmm(weight.unsqueeze(1), enc).squeeze(1)  # (B,E)
        return context, weight


class AttentionDecoder(nn.Module):
    """GRU 기반 자기회귀 디코더. 한 스텝에 문자 1자를 예측한다."""

    def __init__(self, enc_dim: int, hidden: int = 256,
                 num_classes: int = NUM_CLASSES, emb: int = 64) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_classes, emb)
        self.attn = BahdanauAttention(enc_dim, hidden)
        # 입력: 직전 문자 임베딩 + 컨텍스트 벡터
        self.cell = nn.GRUCell(emb + enc_dim, hidden)
        self.out = nn.Linear(hidden + enc_dim, num_classes)
        self.hidden = hidden

    def step(self, enc: "torch.Tensor", prev_tok: "torch.Tensor",
             state: "torch.Tensor"):
        """한 스텝 전개. (logits, 새 상태, 어텐션 가중치)를 돌려준다."""
        ctx, w = self.attn(enc, state)
        e = self.embedding(prev_tok)                      # (B,emb)
        state = self.cell(torch.cat([e, ctx], dim=1), state)
        logits = self.out(torch.cat([state, ctx], dim=1))  # (B,C)
        return logits, state, w


class AttentionOCR(nn.Module):
    """CNN 인코더 + 양방향 GRU + Attention 디코더를 묶은 OCR 모델."""

    def __init__(self, hidden: int = 256, enc_ch: int = 256) -> None:
        super().__init__()
        self.cnn = CnnEncoder(1, enc_ch)
        # 양방향 GRU 로 좌우 문맥을 함께 본다. 출력 차원은 2*hidden 이다.
        self.rnn = nn.GRU(enc_ch, hidden, num_layers=1,
                          bidirectional=True, batch_first=True)
        self.enc_dim = hidden * 2
        self.dec = AttentionDecoder(self.enc_dim, hidden)
        self.init_state = nn.Linear(self.enc_dim, hidden)
        self.hidden = hidden

    def encode(self, img: "torch.Tensor"):
        """이미지를 인코더 표현으로 바꾼다."""
        seq = self.cnn(img)                 # (B,T,enc_ch)
        enc, _ = self.rnn(seq)              # (B,T,2*hidden)
        s0 = torch.tanh(self.init_state(enc.mean(dim=1)))   # (B,hidden)
        return enc, s0

    def forward(self, img: "torch.Tensor",
                targets: "torch.Tensor | None" = None,
                teacher_forcing: bool = True):
        """학습/추론 공통 전개.

        targets 가 주어지면 교사 강요(teacher forcing)로 학습하고,
        없으면 직전 예측을 다음 입력으로 넣어 자기회귀 추론한다.
        반환값은 (logits (B,L,C), attention (B,L,T)) 이다.
        """
        enc, state = self.encode(img)
        b = img.size(0)
        steps = targets.size(1) if targets is not None else MAX_LEN
        tok = torch.full((b,), SOS, dtype=torch.long, device=img.device)

        logits_all, attn_all = [], []
        for t in range(steps):
            logits, state, w = self.dec.step(enc, tok, state)
            logits_all.append(logits)
            attn_all.append(w)
            if targets is not None and teacher_forcing:
                tok = targets[:, t]
            else:
                tok = logits.argmax(dim=1)
        return torch.stack(logits_all, 1), torch.stack(attn_all, 1)

    @torch.no_grad()
    def predict(self, img: "torch.Tensor"):
        """추론. (문자열 목록, 문자별 신뢰도 목록, 어텐션 텐서)를 돌려준다."""
        self.eval()
        logits, attn = self.forward(img, None, False)
        prob = F.softmax(logits, dim=-1)
        conf, idx = prob.max(dim=-1)              # (B,L)
        texts, confs = [], []
        for b in range(idx.size(0)):
            chars, cs = [], []
            for t in range(idx.size(1)):
                k = int(idx[b, t])
                if k == EOS:
                    break
                if k in (PAD, SOS):
                    continue
                chars.append(ITOS[k])
                cs.append(float(conf[b, t]))
            texts.append("".join(chars))
            confs.append(cs)
        return texts, confs, attn


def encode_label(text: str) -> list[int]:
    """정답 문자열을 디코더 타깃 인덱스 열로 바꾼다(끝에 EOS, 뒤는 PAD)."""
    ids = [STOI[c] for c in text if c in STOI] + [EOS]
    return (ids + [PAD] * MAX_LEN)[:MAX_LEN]


def ctc_free_loss(logits: "torch.Tensor", targets: "torch.Tensor"):
    """Attention OCR 의 손실. 자리별 교차 엔트로피이며 PAD 는 무시한다."""
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
        ignore_index=PAD)


def count_params(model: "nn.Module") -> int:
    return sum(p.numel() for p in model.parameters())


def _demo() -> None:
    torch.manual_seed(0)
    model = AttentionOCR()
    print(f"문자 클래스 수 : {NUM_CLASSES} (특수 토큰 3 + 문자 {len(CHARSET)})")
    print(f"파라미터 수     : {count_params(model):,}개 "
          f"({count_params(model) * 4 / 1e6:.1f} MB, float32 기준)")

    # (1) 인코더 출력 형태 확인
    img = torch.randn(2, 1, 32, 160)
    enc, s0 = model.encode(img)
    print(f"\n입력 이미지     : {tuple(img.shape)}")
    print(f"인코더 시퀀스   : {tuple(enc.shape)}  (배치, 시퀀스 길이, 차원)")
    print(f"디코더 초기 상태: {tuple(s0.shape)}")

    # (2) 학습 경로(교사 강요) 확인
    labels = torch.tensor([encode_label("12가3456"),
                           encode_label("87로2345")])
    logits, attn = model(img, labels, True)
    loss = ctc_free_loss(logits, labels)
    print(f"\n[학습 경로] logits {tuple(logits.shape)}  "
          f"attention {tuple(attn.shape)}")
    print(f"[학습 경로] 초기 손실 = {loss.item():.4f} "
          f"(균등 분포 기준 ln({NUM_CLASSES}) ≈ "
          f"{__import__('math').log(NUM_CLASSES):.4f})")

    # (3) 추론 경로 확인 — 학습 전이므로 결과 문자열은 무의미하다.
    texts, confs, attn2 = model.predict(img)
    print(f"\n[추론 경로] attention {tuple(attn2.shape)}")
    for i, (t, c) in enumerate(zip(texts, confs)):
        avg = sum(c) / len(c) if c else 0.0
        print(f"  샘플 {i}: '{t}' (길이 {len(t)}, 평균 신뢰도 {avg:.3f})")
    print("\n학습 전 가중치이므로 문자열은 의미가 없다. "
          "구조와 텐서 형태가 맞는지만 확인한다.")


if __name__ == "__main__":
    _demo()
