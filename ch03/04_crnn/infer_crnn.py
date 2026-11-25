# -*- coding: utf-8 -*-
"""CRNN 추론 스크립트 — 이미지 한 장 또는 폴더를 읽어 번호판 문자열을 출력한다.

실행:  python 04_crnn/infer_crnn.py --image data/synth/images/00000.png
       python 04_crnn/infer_crnn.py --dir data/synth/images --weights runs/best.pt

가중치 파일이 없으면 무작위 초기화 모델로 형식만 보여 주고, torch 가 없으면
넘파이 가짜 확률 행렬에 CTC 디코더만 적용해 디코딩 경로를 확인한다.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "03_dataset"))
sys.path.append(str(ROOT / "05_ctc"))
from charset import BLANK_INDEX, NUM_CLASSES, encode  # noqa: E402
from ctc_decoder import beam_search_decode, greedy_decode  # noqa: E402

IMG_H, IMG_W = 32, 160


def collect_images(args: argparse.Namespace) -> list[Path]:
    """--image 또는 --dir 로 지정된 이미지 목록을 모은다."""
    if args.image:
        path = Path(args.image)
        return [path] if path.exists() else []
    if args.dir:
        folder = Path(args.dir)
        if not folder.exists():
            return []
        return sorted(p for p in folder.iterdir()
                      if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"})
    return []


def fake_logits(text: str, n_frames: int = 40, seed: int = 0) -> np.ndarray:
    """가중치가 없을 때 쓰는 가짜 (T, C) 확률 행렬(디코딩 경로 점검용)."""
    rng = np.random.default_rng(seed)
    labels = encode(text)
    probs = np.full((n_frames, NUM_CLASSES), 1e-4)
    probs[:, BLANK_INDEX] = 0.7
    span = max(1, (n_frames - 2) // max(len(labels) * 2, 1))
    pos = 1
    for lab in labels:
        for _ in range(span):
            if pos >= n_frames - 1:
                break
            probs[pos, :] = 1e-4
            probs[pos, lab] = 0.85 + 0.1 * rng.random()
            pos += 1
        pos += span   # 문자 사이 blank 구간
    return probs / probs.sum(axis=-1, keepdims=True)


def run_without_torch(images: list[Path], mode: str) -> None:
    """torch 없이 디코더 동작만 확인한다."""
    print("[안내] torch 가 없다. 가짜 확률 행렬로 디코딩 경로만 점검한다.")
    print("       pip install torch --index-url https://download.pytorch.org/whl/cpu\n")
    samples = ["12가3456", "서울34나5678", "87로2345"]
    print(f"{'정답':12s} {'greedy':12s} {'beam':12s} {'conf':>7s} {'ms':>6s}")
    print("-" * 56)
    for text in samples:
        probs = fake_logits(text)
        t0 = time.perf_counter()
        g = greedy_decode(probs)
        t_g = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        b = beam_search_decode(probs, beam_width=10, top_k=1)[0]
        t_b = (time.perf_counter() - t0) * 1000
        print(f"{text:12s} {g['text']:12s} {b['text']:12s} "
              f"{g['confidence']:7.4f} {t_g:6.2f}")
        print(f"{'':12s} {'':12s} {'(beam)':12s} "
              f"{b['confidence']:7.4f} {t_b:6.2f}")
    print(f"\n대상 이미지 {len(images)}장은 모델이 없어 건너뛴다.")


def main() -> None:
    parser = argparse.ArgumentParser(description="CRNN 번호판 추론")
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--dir", type=str, default=None)
    parser.add_argument("--weights", type=str, default="runs/best.pt")
    parser.add_argument("--mode", type=str, default="greedy",
                        choices=["greedy", "beam"])
    parser.add_argument("--beam-width", type=int, default=10)
    args = parser.parse_args()

    images = collect_images(args)

    try:
        import torch
    except ImportError:
        run_without_torch(images, args.mode)
        raise SystemExit(0)

    sys.path.append(str(Path(__file__).resolve().parent))
    from crnn import CRNN
    from dataset import preprocess_image

    # CPU 스레드를 제한한다. 기본값은 코어 수만큼 쓰는데, 웹 서비스에서는
    # 요청 하나가 모든 코어를 점유해 동시 처리량이 오히려 떨어진다.
    torch.set_num_threads(4)

    model = CRNN()
    weights_path = Path(args.weights)
    if weights_path.exists():
        state = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state.get("model", state))
        print(f"가중치 로드: {weights_path}")
    else:
        print(f"[안내] 가중치 {weights_path} 가 없다. 무작위 초기화 모델로 진행한다.")
        print("       결과 문자열은 의미가 없고 입출력 형식만 확인하는 용도다.")
    model.eval()

    if not images:
        print("\n[안내] 대상 이미지가 없다. --image 또는 --dir 로 지정한다.")
        print("       data/synth 가 없으면 synth_plate.py 를 먼저 실행한다.")
        raise SystemExit(0)

    print(f"\n{'파일':24s} {'인식 결과':12s} {'신뢰도':>8s} {'ms':>7s}")
    print("-" * 58)
    total_ms = 0.0
    for path in images[:20]:
        arr = preprocess_image(str(path))
        tensor = torch.from_numpy(arr).unsqueeze(0)      # (1, 1, 32, 160)
        t0 = time.perf_counter()
        with torch.no_grad():
            probs = model.predict_proba(tensor)[0].numpy()   # (T, C)
        if args.mode == "beam":
            result = beam_search_decode(probs, beam_width=args.beam_width,
                                        top_k=1)[0]
        else:
            result = greedy_decode(probs)
        elapsed = (time.perf_counter() - t0) * 1000
        total_ms += elapsed
        text = result["text"] or "(없음)"
        print(f"{path.name:24s} {text:12s} {result['confidence']:8.4f} "
              f"{elapsed:7.2f}")

    n = min(len(images), 20)
    print(f"\n평균 추론 시간: {total_ms / max(n, 1):.2f} ms/장 "
          f"({args.mode} 디코딩, 스레드 4)")
    print("실제 운영에서는 전처리와 검출 시간이 더해지므로 계약서의 elapsed_ms 로 분리 측정한다.")


if __name__ == "__main__":
    main()
