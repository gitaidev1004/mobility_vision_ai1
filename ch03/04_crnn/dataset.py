# -*- coding: utf-8 -*-
"""CRNN 학습용 Dataset/DataLoader — 리사이즈·정규화·라벨 인코딩.

실행:  python 04_crnn/dataset.py --index data/index/train.jsonl
torch 가 없으면 전처리 로직만 numpy 로 시연한다.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parents[1] / "03_dataset"))
from charset import BLANK_INDEX, CHARSET, NUM_CLASSES, decode, encode  # noqa: E402

IMG_H, IMG_W = 32, 160

#: 정규화 상수. 회색조 번호판은 평균이 0.5 근처이므로 (x-0.5)/0.5 로 [-1, 1] 에 맞춘다.
#: ImageNet 통계를 쓰지 않는 이유: 사전학습 백본을 쓰지 않고 처음부터 학습하므로
#: 데이터 자체의 분포에 맞추는 편이 수렴이 빠르다.
MEAN, STD = 0.5, 0.5


def load_index(path: Path) -> list[dict]:
    """JSONL 인덱스를 읽는다."""
    items: list[dict] = []
    with path.open(encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def preprocess_image(path: str, keep_ratio: bool = True) -> np.ndarray:
    """이미지를 (1, 32, 160) float32 배열로 전처리한다.

    keep_ratio=True 면 높이 32 에 맞춰 축소한 뒤 남는 폭을 회색(127)으로 채운다.
    구형 2줄형(비율 2.0)과 신형 1줄형(비율 4.7)이 섞인 데이터에서 비율을 무시하고
    늘리면 같은 글자가 전혀 다른 모양이 되어 학습이 어려워진다.
    """
    img = Image.open(path).convert("L")
    if keep_ratio:
        scale = IMG_H / float(img.height)
        new_w = max(1, min(IMG_W, int(round(img.width * scale))))
        img = img.resize((new_w, IMG_H), Image.BICUBIC)
        if new_w < IMG_W:
            canvas = Image.new("L", (IMG_W, IMG_H), 127)
            canvas.paste(img, (0, 0))      # 왼쪽 정렬 — 번호판은 왼쪽부터 읽는다
            img = canvas
    else:
        img = img.resize((IMG_W, IMG_H), Image.BICUBIC)

    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    return arr[None, :, :]                 # (1, H, W)


def encode_label(text: str) -> tuple[list[int], int]:
    """라벨 문자열을 (인덱스 리스트, 길이) 로 인코딩한다."""
    indices = encode(text)
    return indices, len(indices)


try:
    import torch
    from torch.utils.data import DataLoader, Dataset
except ImportError:
    print("[안내] torch 가 없어 DataLoader 는 만들 수 없다. 전처리만 시연한다.\n")

    def main_no_torch(index_path: Path) -> None:
        if not index_path.exists():
            print(f"[안내] {index_path} 가 없다. make_lmdb_like.py 를 먼저 실행한다.")
            return
        items = load_index(index_path)
        print(f"인덱스 {len(items):,}건  (문자셋 {len(CHARSET)}자, 클래스 {NUM_CLASSES})")
        print(f"\n{'이미지':30s} {'라벨':12s} {'길이':>4s} 인코딩(앞 7개)")
        print("-" * 78)
        for item in items[:5]:
            arr = preprocess_image(item["image"])
            ids, length = encode_label(item["text"])
            name = Path(item["image"]).name
            print(f"{name:30s} {item['text']:12s} {length:4d} {ids[:7]}")
        arr = preprocess_image(items[0]["image"])
        print(f"\n전처리 결과 shape = {arr.shape}, "
              f"범위 [{arr.min():.3f}, {arr.max():.3f}], 평균 {arr.mean():.3f}")
        print(f"역변환 확인: decode({ids[:7]}) = {decode(ids)!r}")

    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=str, default="data/index/train.jsonl")
    main_no_torch(Path(parser.parse_args().index))
    raise SystemExit(0)


class PlateDataset(Dataset):
    """번호판 OCR 학습용 Dataset.

    __getitem__ 이 (이미지 텐서, 라벨 인덱스 리스트, 라벨 길이) 를 돌려주고,
    실제 패딩은 collate 함수에서 한다. 배치마다 라벨 길이가 다르기 때문에
    Dataset 단계에서 고정 길이로 맞추면 CTC 가 쓰는 target_lengths 가 왜곡된다.
    """

    def __init__(self, index_path: str | Path, keep_ratio: bool = True) -> None:
        self.items = load_index(Path(index_path))
        self.keep_ratio = keep_ratio
        if not self.items:
            raise ValueError(f"인덱스가 비어 있다: {index_path}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        item = self.items[idx]
        arr = preprocess_image(item["image"], keep_ratio=self.keep_ratio)
        # 인덱스에 미리 인코딩해 둔 label 이 있으면 재사용한다(문자열 파싱 비용 절약).
        ids = item.get("label") or encode(item["text"])
        return torch.from_numpy(arr), torch.tensor(ids, dtype=torch.long), item["text"]


def ctc_collate(batch):
    """CTC 학습에 필요한 4종 텐서로 배치를 묶는다.

    CTCLoss 는 다음 네 가지를 요구한다.
      log_probs      (T, N, C) — 모델 출력
      targets        (sum(L),) — 라벨을 한 줄로 이어 붙인 1차원 텐서
      input_lengths  (N,)      — 각 샘플의 T (여기서는 전부 같다)
      target_lengths (N,)      — 각 샘플의 라벨 길이

    targets 를 패딩된 2차원이 아니라 1차원 연결로 주는 이유:
    패딩 토큰을 blank 와 구분할 방법이 없기 때문이다. 길이를 따로 넘기는 방식이
    패딩 값에 의존하지 않아 안전하다.
    """
    images = torch.stack([b[0] for b in batch], dim=0)
    labels = [b[1] for b in batch]
    texts = [b[2] for b in batch]
    targets = torch.cat(labels, dim=0)
    target_lengths = torch.tensor([len(x) for x in labels], dtype=torch.long)
    return images, targets, target_lengths, texts


def build_loader(index_path: str | Path, batch_size: int = 32,
                 shuffle: bool = True, num_workers: int = 0) -> "DataLoader":
    """DataLoader 를 만든다.

    윈도우에서 num_workers>0 을 쓰면 워커마다 프로세스를 새로 띄우며 모듈을
    다시 import 한다. 스크립트에 if __name__ == "__main__" 가드가 없으면
    무한 재귀로 멈추므로, 교재 예제의 기본값은 0 으로 둔다.
    """
    dataset = PlateDataset(index_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, collate_fn=ctc_collate,
                      drop_last=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="CRNN Dataset/DataLoader 점검")
    parser.add_argument("--index", type=str, default="data/index/train.jsonl")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    index_path = Path(args.index)
    if not index_path.exists():
        print(f"[안내] {index_path} 가 없다. make_lmdb_like.py 를 먼저 실행한다.")
        raise SystemExit(0)

    loader = build_loader(index_path, batch_size=args.batch_size, shuffle=False)
    print(f"샘플 {len(loader.dataset):,}건, 배치 {len(loader)}개\n")

    images, targets, target_lengths, texts = next(iter(loader))
    print(f"images         : {tuple(images.shape)}  dtype={images.dtype}")
    print(f"targets        : {tuple(targets.shape)} (라벨을 1차원으로 연결)")
    print(f"target_lengths : {target_lengths.tolist()}")
    print(f"input_lengths  : 전부 40 (CRNN 의 고정 T)")
    print(f"\n픽셀 범위 [{float(images.min()):.3f}, {float(images.max()):.3f}], "
          f"평균 {float(images.mean()):.3f}")

    print(f"\n{'원본 라벨':12s} {'길이':>4s} 인코딩")
    print("-" * 60)
    offset = 0
    for text, length in zip(texts, target_lengths.tolist()):
        ids = targets[offset:offset + length].tolist()
        offset += length
        print(f"{text:12s} {length:4d} {ids}")

    print(f"\nblank 인덱스 {BLANK_INDEX} 는 라벨에 절대 들어가지 않는다.")
    print("라벨에 blank 가 섞이면 CTCLoss 가 NaN 을 낸다 — 인코딩 단계에서 차단된다.")


if __name__ == "__main__":
    main()
