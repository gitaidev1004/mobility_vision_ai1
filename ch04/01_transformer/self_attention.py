# -*- coding: utf-8 -*-
"""Scaled Dot-Product Attention 을 numpy 로 직접 구현한다.

Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
"""
import numpy as np


def softmax(x, axis=-1):
    """수치적으로 안정적인 softmax. 최댓값을 빼고 지수함수를 적용한다."""
    x_max = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - x_max)
    return e / np.sum(e, axis=axis, keepdims=True)


def scaled_dot_product_attention(q, k, v, mask=None):
    """Scaled Dot-Product Attention.

    q: (n_q, d_k)  질의(Query)
    k: (n_k, d_k)  키(Key)
    v: (n_k, d_v)  값(Value)
    mask: (n_q, n_k) 불리언 배열. True 인 위치를 가린다.
    반환: (출력 (n_q, d_v), 어텐션 가중치 (n_q, n_k))
    """
    d_k = q.shape[-1]
    # 1) 유사도 점수: Q 와 K 의 내적
    scores = q @ k.T
    # 2) 스케일링: sqrt(d_k) 로 나눠 softmax 의 기울기 소실을 막는다
    scores = scores / np.sqrt(d_k)
    # 3) 마스킹: 가릴 위치를 매우 작은 값으로 바꿔 softmax 결과를 0 에 가깝게 만든다
    if mask is not None:
        scores = np.where(mask, -1e9, scores)
    # 4) 정규화: 각 질의마다 키에 대한 확률분포를 만든다
    weights = softmax(scores, axis=-1)
    # 5) 가중합: 확률로 Value 를 섞는다
    out = weights @ v
    return out, weights


def main():
    np.random.seed(0)
    np.set_printoptions(precision=3, suppress=True)

    n_q, n_k, d_k, d_v = 3, 4, 8, 5
    q = np.random.randn(n_q, d_k)
    k = np.random.randn(n_k, d_k)
    v = np.random.randn(n_k, d_v)

    out, w = scaled_dot_product_attention(q, k, v)
    print("[입력 shape] Q:", q.shape, "K:", k.shape, "V:", v.shape)
    print("[출력 shape] out:", out.shape, "weights:", w.shape)
    print("\n[어텐션 가중치]")
    print(w)
    print("\n[행별 가중치 합]", np.sum(w, axis=1))
    print("\n[출력 벡터]")
    print(out)

    # 마지막 키를 가리는 마스크를 적용해 변화를 확인한다
    mask = np.zeros((n_q, n_k), dtype=bool)
    mask[:, -1] = True
    out_m, w_m = scaled_dot_product_attention(q, k, v, mask=mask)
    print("\n[마스킹 후 어텐션 가중치] (마지막 열이 0 이 된다)")
    print(w_m)
    print("[마스킹 전후 출력 차이 norm]", round(float(np.linalg.norm(out - out_m)), 4))


if __name__ == "__main__":
    main()
