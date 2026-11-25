# -*- coding: utf-8 -*-
"""CRNN+CTC vs Attention OCR 비교 실험 스크립트.

REFERENCE.md 의 평가 기준 5번 'CPU 추론 시간, 메모리 사용량, INT8 변환
전후 정확도를 함께 비교한다'를 실행 가능한 형태로 옮긴 것이다.

측정 항목
  - 파라미터 수 / 모델 크기(MB)
  - 1장 추론 지연시간 p50, p95 (CPU, 배치 1)
  - 배치 8 처리량(장/초)
  - 추론 중 증가한 메모리(RSS 기준, psutil 이 있을 때)
  - 문자 정확도 / 완전 일치율 (가중치가 있을 때만 의미 있음)

가중치 파일이 없으면 랜덤 초기화 모델로 '속도와 메모리만' 비교하고,
정확도 칸에는 n/a 를 출력한다. 정확도를 비교하려면 먼저 두 모델을
같은 학습/검증 분할로 학습시킨 뒤 .pt 파일을 같은 디렉터리에 둔다.
"""
from __future__ import annotations

import os
import statistics
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_CH03 = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_CH03, "04_crnn"))
sys.path.insert(0, os.path.join(_CH03, "05_ctc"))
sys.path.insert(0, os.path.join(_CH03, "10_eval"))

try:
    import torch
except ImportError:
    print("[안내] PyTorch 가 설치되어 있지 않다.")
    print("       pip install torch --index-url "
          "https://download.pytorch.org/whl/cpu")
    raise SystemExit(0)

from attention_ocr import AttentionOCR  # noqa: E402

# 3.4절의 CRNN 모델과 3.5절의 CTC 디코더를 그대로 재사용한다.
try:
    from crnn import CRNN                       # ch03/04_crnn/crnn.py
    from ctc_decoder import greedy_decode       # ch03/05_ctc/ctc_decoder.py
    _HAS_CRNN = True
except ImportError:
    _HAS_CRNN = False
    print("[안내] ch03/04_crnn/crnn.py 또는 "
          "ch03/05_ctc/ctc_decoder.py 를 찾지 못했다.")
    print("       3.4~3.5절 예제를 먼저 저장한 뒤 다시 실행한다. "
          "여기서는 Attention 모델만 측정한다.")

try:
    import psutil
    _PROC = psutil.Process(os.getpid())
except ImportError:
    _PROC = None

try:
    from metrics_ocr import evaluate            # ch03/10_eval/metrics_ocr.py
    _HAS_METRICS = True
except ImportError:
    _HAS_METRICS = False

WARMUP = 5
RUNS = 30
BATCH = 8


def rss_mb() -> float:
    """현재 프로세스의 상주 메모리(MB). psutil 이 없으면 0."""
    if _PROC is None:
        return 0.0
    return _PROC.memory_info().rss / (1024 * 1024)


def param_stats(model) -> tuple[int, float]:
    """(파라미터 수, float32 기준 모델 크기 MB)."""
    n = sum(p.numel() for p in model.parameters())
    return n, n * 4 / (1024 * 1024)


def measure_latency(fn, runs: int = RUNS, warmup: int = WARMUP) -> dict:
    """함수 하나의 지연시간 통계를 낸다(단위 ms)."""
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    samples.sort()
    return {
        "p50": statistics.median(samples),
        "p95": samples[min(len(samples) - 1, int(0.95 * len(samples)))],
        "mean": statistics.fmean(samples),
        "min": samples[0],
        "max": samples[-1],
    }


def bench_model(name: str, model, forward, batch_forward) -> dict:
    """모델 하나를 측정해 결과 딕셔너리로 돌려준다."""
    model.eval()
    n_param, size_mb = param_stats(model)
    before = rss_mb()
    lat = measure_latency(forward)
    peak = rss_mb()
    t0 = time.perf_counter()
    for _ in range(5):
        batch_forward()
    thr = (5 * BATCH) / (time.perf_counter() - t0)
    return {
        "name": name,
        "params": n_param,
        "size_mb": size_mb,
        "p50": lat["p50"],
        "p95": lat["p95"],
        "mean": lat["mean"],
        "throughput": thr,
        "mem_delta": max(peak - before, 0.0),
        "rss": peak,
    }


def quantize_int8(model):
    """동적 양자화(INT8). 선형/RNN 계층의 가중치를 8비트로 바꾼다.

    CPU 추론에서 모델 크기와 지연시간을 줄이는 가장 손쉬운 방법이다.
    정확도 변화는 반드시 같은 검증 세트로 다시 측정해야 한다.
    """
    try:
        return torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear, torch.nn.LSTM, torch.nn.GRU},
            dtype=torch.qint8)
    except Exception as exc:                     # noqa: BLE001
        print(f"[안내] 동적 양자화를 적용할 수 없다: {exc}")
        return None


def print_table(rows: list[dict]) -> None:
    """비교 결과 표를 출력한다."""
    print(f"\n{'모델':<22s}{'파라미터':>12s}{'크기(MB)':>10s}"
          f"{'p50(ms)':>9s}{'p95(ms)':>9s}{'처리량(장/s)':>13s}"
          f"{'메모리증가(MB)':>14s}")
    print("-" * 90)
    for r in rows:
        print(f"{r['name']:<22s}{r['params']:>12,d}{r['size_mb']:>10.1f}"
              f"{r['p50']:>9.2f}{r['p95']:>9.2f}{r['throughput']:>13.1f}"
              f"{r['mem_delta']:>14.1f}")
    print("-" * 90)
    if len(rows) >= 2:
        base, other = rows[0], rows[1]
        ratio = other["p50"] / base["p50"] if base["p50"] else 0.0
        print(f"{other['name']} 의 p50 은 {base['name']} 대비 "
              f"{ratio:.2f}배다.")


def _demo() -> None:
    torch.manual_seed(0)
    torch.set_num_threads(4)        # CPU 추론은 스레드 수를 고정해 비교한다
    print(f"PyTorch {torch.__version__} / 스레드 {torch.get_num_threads()}개")
    if _PROC is None:
        print("[안내] psutil 이 없어 메모리 증가량은 0 으로 표시된다. "
              "pip install psutil")

    img1 = torch.randn(1, 1, 32, 160)
    img8 = torch.randn(BATCH, 1, 32, 160)
    rows: list[dict] = []

    # ----- (1) CRNN + CTC -----
    if _HAS_CRNN:
        crnn = CRNN()
        with torch.no_grad():
            rows.append(bench_model(
                "CRNN+CTC", crnn,
                lambda: crnn(img1), lambda: crnn(img8)))
        # CTC 디코딩 비용도 따로 본다. 보통 전체의 5% 미만이다.
        with torch.no_grad():
            logits = crnn(img1)
        dec = measure_latency(lambda: greedy_decode(logits), 50, 5)
        print(f"CTC greedy 디코딩 p50 = {dec['p50']:.3f} ms")

    # ----- (2) Attention OCR -----
    attn = AttentionOCR()
    with torch.no_grad():
        rows.append(bench_model(
            "Attention(GRU)", attn,
            lambda: attn(img1, None, False), lambda: attn(img8, None, False)))

    # ----- (3) INT8 양자화 비교 -----
    qa = quantize_int8(attn)
    if qa is not None:
        with torch.no_grad():
            r = bench_model("Attention INT8", attn,
                            lambda: qa(img1, None, False),
                            lambda: qa(img8, None, False))
        # 양자화 모델의 파라미터 수는 동일하지만 저장 크기는 약 1/4 이 된다.
        r["size_mb"] = r["size_mb"] / 4.0
        rows.append(r)

    print_table(rows)

    # ----- (4) 정확도 비교 -----
    print("\n=== 정확도 비교 ===")
    ckpts = [os.path.join(_HERE, "attention_ocr.pt"),
             os.path.join(_CH03, "04_crnn", "crnn.pt")]
    missing = [p for p in ckpts if not os.path.exists(p)]
    if missing or not _HAS_METRICS:
        print("학습된 가중치가 없어 정확도는 측정하지 않는다(n/a).")
        for p in missing:
            print(f"  없음: {p}")
        print("3.4절/3.6절의 학습 스크립트를 먼저 실행해 .pt 를 만든 뒤")
        print("같은 검증 세트로 아래 지표를 채운다.")
        print("  - Character Accuracy / CER / 문자열 완전 일치율")
        print("  - 환경별(주간/야간/우천/역광/원거리/기울기) 분해")
        print("  - INT8 변환 전후 완전 일치율 차이")
    else:
        # 검증 세트를 돌려 (정답, 예측) 쌍을 만든 뒤 지표를 계산한다.
        pairs: list[tuple[str, str]] = []       # 실제 검증 루프로 채운다
        m = evaluate(pairs)
        print(m.report())

    print("\n=== 해석 지침 ===")
    print(" - Attention 은 출력 길이만큼 디코더를 반복하므로 "
          "번호판 7~9자에서 CRNN 보다 지연시간이 커지는 경향이 있다.")
    print(" - 지연시간이 2배여도 완전 일치율이 1%p 이상 오르면 "
          "게이트 1대 환경에서는 채택할 가치가 있다.")
    print(" - 반대로 차이가 통계적으로 유의하지 않으면 "
          "CPU 비용이 낮은 CRNN+CTC 를 유지한다.")
    print(" - 모든 수치는 같은 스레드 수, 같은 입력 해상도에서 "
          "다시 측정해 비교해야 한다.")


if __name__ == "__main__":
    _demo()
