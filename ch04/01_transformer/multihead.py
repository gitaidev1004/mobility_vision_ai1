# -*- coding: utf-8 -*-
"""Multi-Head Attention 을 numpy 로 구현하고 헤드별 출력 shape 을 확인한다.

MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W_O
head_i = Attention(Q W_Qi, K W_Ki, V W_Vi)
"""
import numpy as np


def softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - x_max)
    return e / np.sum(e, axis=axis, keepdims=True)


def attention(q, k, v):
    """단일 헤드 어텐션. q,k,v 는 (n, d_head)."""
    d = q.shape[-1]
    w = softmax((q @ k.T) / np.sqrt(d), axis=-1)
    return w @ v, w


class MultiHeadAttention:
    """학습 없이 구조만 확인하기 위한 Multi-Head Attention 구현."""

    def __init__(self, d_model=64, n_heads=8, seed=0):
        assert d_model % n_heads == 0, "d_model 은 n_heads 로 나누어져야 한다"
        rng = np.random.default_rng(seed)
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        scale = 1.0 / np.sqrt(d_model)
        # 전체 헤드를 한 번에 투영하는 큰 가중치 행렬 (d_model, d_model)
        self.Wq = rng.normal(0, scale, (d_model, d_model))
        self.Wk = rng.normal(0, scale, (d_model, d_model))
        self.Wv = rng.normal(0, scale, (d_model, d_model))
        self.Wo = rng.normal(0, scale, (d_model, d_model))

    def split_heads(self, x):
        """(n, d_model) -> (n_heads, n, d_head) 로 재배열한다."""
        n = x.shape[0]
        x = x.reshape(n, self.n_heads, self.d_head)
        return x.transpose(1, 0, 2)

    def __call__(self, x, verbose=False):
        n = x.shape[0]
        q = self.split_heads(x @ self.Wq)
        k = self.split_heads(x @ self.Wk)
        v = self.split_heads(x @ self.Wv)
        if verbose:
            print(f"입력 x            : {x.shape}")
            print(f"헤드 분할 후 Q    : {q.shape}  (n_heads, n_tokens, d_head)")

        heads, weights = [], []
        for h in range(self.n_heads):
            out_h, w_h = attention(q[h], k[h], v[h])
            heads.append(out_h)
            weights.append(w_h)
            if verbose:
                print(f"  head {h}: out {out_h.shape}, attn {w_h.shape}")

        # 헤드 출력을 이어 붙여 (n, d_model) 로 복원한 뒤 W_O 로 투영한다
        concat = np.concatenate(heads, axis=-1)
        out = concat @ self.Wo
        if verbose:
            print(f"Concat 후         : {concat.shape}")
            print(f"W_O 투영 후 출력  : {out.shape}")
        return out, np.stack(weights), concat


def main():
    np.set_printoptions(precision=3, suppress=True)
    n_tokens, d_model, n_heads = 6, 64, 8
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, (n_tokens, d_model))

    mha = MultiHeadAttention(d_model=d_model, n_heads=n_heads, seed=0)
    print(f"[설정] d_model={d_model}, n_heads={n_heads}, d_head={mha.d_head}")
    out, attn, concat = mha(x, verbose=True)

    print("\n[어텐션 텐서 shape]", attn.shape, "(n_heads, n_tokens, n_tokens)")
    print("[헤드 0 어텐션 가중치]")
    print(attn[0])
    print("\n[헤드별 가중치 합이 1인지 확인]",
          np.allclose(attn.sum(axis=-1), 1.0))
    print("[출력 일부 (앞 4차원)]")
    print(out[:, :4])

    # 파라미터 수 계산: 4 개의 (d_model x d_model) 투영 행렬
    params = 4 * d_model * d_model
    print(f"\n[투영 파라미터 수] 4 x {d_model} x {d_model} = {params:,}")


if __name__ == "__main__":
    main()
