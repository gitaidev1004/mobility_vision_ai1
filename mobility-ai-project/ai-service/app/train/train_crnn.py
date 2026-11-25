"""CRNN + CTC 학습 루프 (CLI, 체크포인트 / EarlyStopping).

사용법::

    # 1) OCR 학습용 데이터(번호판만, 배경 없음) 생성
    python -m app.train.synth_plate --out ./data/ocr --count 20000 --no-scene

    # 2) 학습
    python -m app.train.train_crnn --data ./data/ocr --epochs 30 --batch 64

    # 3) 서비스에 반영 (CRNN_WEIGHTS 경로로 저장됨)
    #    .env 의 CRNN_WEIGHTS=./models/crnn_ctc_kr.pt

CTC 학습에서 흔히 틀리는 부분 — 이 코드가 짚는 함정들:

    1. **입력 형태**: ``nn.CTCLoss`` 는 ``(T, B, C)`` log-softmax 를 요구한다.
       모델이 ``(B, T, C)`` 를 내므로 permute 가 필요하다. 이 변환을 잊으면
       손실이 수렴하지 않으면서도 오류는 나지 않아 디버깅이 매우 어렵다.
       ``CRNN.log_probs_for_ctc`` 가 이 변환을 캡슐화한다.

    2. **길이 조건**: ``input_length >= target_length`` 여야 한다. T=40 이고
       번호판이 최대 10자이므로 여유가 충분하지만, 라벨이 깨져 길이가 폭증하면
       손실이 inf 가 된다. 배치 구성 시 길이를 검증한다.

    3. **``zero_infinity=True``**: 위 조건을 위반한 샘플의 손실을 0 으로 만든다.
       한 샘플 때문에 배치 전체의 기울기가 NaN 이 되는 것을 막는 안전장치다.

    4. **blank 인덱스 일치**: 모델 출력 차원, 디코더, 손실 함수가 모두 같은
       blank 인덱스를 써야 한다. ``charset.BLANK_INDEX`` 하나만 참조한다.

torch 가 없으면 설치 안내 후 종료한다(import 단계에서 터지지 않는다).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from app.core.logging import configure_logging, get_logger
from app.ocr.charset import BLANK_INDEX, NUM_CLASSES, encode

logger = get_logger(__name__)

#: 기본 학습 설정.
DEFAULT_EPOCHS: int = 30
DEFAULT_BATCH: int = 64
DEFAULT_LR: float = 1e-3
DEFAULT_PATIENCE: int = 6


def check_torch() -> bool:
    """torch 설치 여부를 확인한다.

    Returns:
        설치되어 있으면 True.
    """
    try:
        import importlib.util  # noqa: PLC0415

        return importlib.util.find_spec("torch") is not None
    except (ImportError, ValueError):
        return False


@dataclass(slots=True)
class OcrSample:
    """OCR 학습 샘플 하나.

    Attributes:
        image_path: 이미지 파일 경로.
        text: 정답 번호판 문자열.
    """

    image_path: Path
    text: str


@dataclass(slots=True)
class TrainState:
    """학습 상태(EarlyStopping / 체크포인트 관리).

    Attributes:
        best_loss: 지금까지의 최저 검증 손실.
        best_accuracy: 지금까지의 최고 검증 정확도.
        epochs_without_improvement: 개선 없이 지난 에폭 수.
        history: 에폭별 지표 기록.
    """

    best_loss: float = float("inf")
    best_accuracy: float = 0.0
    epochs_without_improvement: int = 0
    history: list[dict[str, float]] = field(default_factory=list)


def load_dataset(data_dir: str | Path) -> list[OcrSample]:
    """``labels.jsonl`` 에서 OCR 학습 샘플을 읽는다.

    ``synth_plate.generate_dataset`` 가 만드는 형식을 기대한다::

        {"image": "images/000000.jpg", "text": "12가3456", "bbox": [...]}

    Args:
        data_dir: 데이터셋 루트(``labels.jsonl`` 이 있는 디렉터리).

    Returns:
        OcrSample 리스트. 파일이 없거나 비면 빈 리스트.
    """
    root = Path(data_dir)
    jsonl = root / "labels.jsonl"
    if not jsonl.is_file():
        logger.error(
            "라벨 파일이 없습니다: %s\n"
            "먼저 데이터를 생성하세요: python -m app.train.synth_plate --out %s --count 20000 --no-scene",
            jsonl,
            root,
        )
        return []

    samples: list[OcrSample] = []
    for line in jsonl.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        text = str(record.get("text") or "")
        rel = str(record.get("image") or "")
        if not text or not rel:
            continue
        image_path = root / rel
        if image_path.is_file():
            samples.append(OcrSample(image_path=image_path, text=text))
    logger.info("데이터셋 로드: %d 샘플 (%s)", len(samples), root)
    return samples


def load_image_gray(path: Path, height: int, width: int) -> np.ndarray | None:
    """이미지를 그레이스케일로 읽어 OCR 입력 크기로 맞춘다.

    **추론과 동일한 전처리**를 써야 한다. 학습에서 ``cv2.INTER_AREA`` 를 쓰고
    추론에서 ``INTER_LINEAR`` 를 쓰면 입력 분포가 미묘하게 달라져 정확도가
    떨어진다 — ``app.pipeline.rectify.resize_for_ocr`` 를 공유해 이를 막는다.

    Args:
        path: 이미지 경로.
        height: 목표 높이.
        width: 목표 너비.

    Returns:
        ``(H, W)`` uint8 배열 또는 None(로딩 실패).
    """
    try:
        import cv2  # noqa: PLC0415

        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            return None
    except ImportError:
        try:
            from PIL import Image  # noqa: PLC0415

            with Image.open(path) as im:
                image = np.asarray(im.convert("L"))
        except Exception:  # noqa: BLE001
            return None
    except Exception:  # noqa: BLE001
        return None

    from app.pipeline.rectify import resize_for_ocr  # noqa: PLC0415

    return resize_for_ocr(image, height=height, width=width)


def collate_batch(
    samples: list[OcrSample], height: int, width: int
) -> tuple[np.ndarray, list[list[int]]] | None:
    """샘플 리스트를 배치 배열과 라벨로 변환한다.

    Args:
        samples: 배치에 들어갈 샘플들.
        height: 입력 높이.
        width: 입력 너비.

    Returns:
        ``(이미지 (B,1,H,W) float32, 라벨 인덱스 리스트)`` 또는 None(유효 샘플 없음).
    """
    images: list[np.ndarray] = []
    labels: list[list[int]] = []
    for sample in samples:
        gray = load_image_gray(sample.image_path, height, width)
        if gray is None:
            continue
        indices = encode(sample.text)
        # 라벨이 비거나 시퀀스 길이(T = width//4)를 넘으면 CTC 조건을 위반한다.
        if not indices or len(indices) > (width // 4):
            continue
        # 추론과 동일한 정규화: [0,1] → 평균 0.5 / 표준편차 0.5
        images.append(((gray.astype(np.float32) / 255.0) - 0.5) / 0.5)
        labels.append(indices)

    if not images:
        return None
    batch = np.stack(images, axis=0)[:, None, :, :]
    return batch, labels


def evaluate(
    model: Any,
    samples: list[OcrSample],
    *,
    height: int,
    width: int,
    batch_size: int,
    criterion: Any,
    device: str,
) -> tuple[float, float]:
    """검증 손실과 완전 일치 정확도를 계산한다.

    왜 손실과 정확도를 모두 보는가?
        CTC 손실이 낮아져도 정확도가 오르지 않는 구간이 있다(정렬은 배웠지만
        글자를 혼동). 반대로 정확도가 같은데 손실만 낮아지는 구간도 있다.
        **정확도를 EarlyStopping 기준**으로 쓰는 것이 실무 목표에 부합한다 —
        번호판은 한 글자만 틀려도 쓸모가 없으므로 "완전 일치율" 이 최종 지표다.

    Args:
        model: CRNN 모델.
        samples: 검증 샘플.
        height: 입력 높이.
        width: 입력 너비.
        batch_size: 배치 크기.
        criterion: CTC 손실 함수.
        device: 디바이스.

    Returns:
        ``(평균 손실, 완전 일치 정확도)``.
    """
    import torch  # noqa: PLC0415

    from app.ocr.ctc_decoder import greedy_decode  # noqa: PLC0415

    model.eval()
    total_loss = 0.0
    batches = 0
    exact = 0
    total = 0

    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            chunk = samples[start : start + batch_size]
            prepared = collate_batch(chunk, height, width)
            if prepared is None:
                continue
            images, labels = prepared
            tensor = torch.from_numpy(images).to(device)

            log_probs = model.log_probs_for_ctc(tensor)  # (T, B, C)
            input_lengths = torch.full(
                (tensor.size(0),), log_probs.size(0), dtype=torch.long
            )
            target_lengths = torch.tensor([len(label) for label in labels], dtype=torch.long)
            targets = torch.tensor(
                [idx for label in labels for idx in label], dtype=torch.long
            )

            loss = criterion(log_probs, targets, input_lengths, target_lengths)
            if torch.isfinite(loss):
                total_loss += float(loss.item())
                batches += 1

            # 정확도: greedy 디코딩 결과를 정답과 비교한다.
            probs = log_probs.permute(1, 0, 2).exp().cpu().numpy()  # (B, T, C)
            for i, label in enumerate(labels):
                from app.ocr.charset import decode as decode_indices  # noqa: PLC0415

                predicted = greedy_decode(probs[i], already_prob=True).text
                if predicted == decode_indices(label):
                    exact += 1
                total += 1

    mean_loss = total_loss / max(1, batches)
    accuracy = exact / max(1, total)
    return mean_loss, accuracy


def train(
    data_dir: str,
    *,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH,
    learning_rate: float = DEFAULT_LR,
    patience: int = DEFAULT_PATIENCE,
    val_ratio: float = 0.1,
    height: int = 32,
    width: int = 160,
    device: str = "cpu",
    output: str = "./models/crnn_ctc_kr.pt",
    seed: int = 20261024,
    max_samples: int = 0,
) -> dict[str, Any]:
    """CRNN + CTC 를 학습한다.

    Args:
        data_dir: 데이터셋 루트.
        epochs: 최대 에폭 수.
        batch_size: 배치 크기.
        learning_rate: 초기 학습률.
        patience: EarlyStopping 인내 에폭 수.
        val_ratio: 검증 분할 비율.
        height: 입력 높이.
        width: 입력 너비.
        device: 디바이스.
        output: 체크포인트 저장 경로.
        seed: 재현성 시드.
        max_samples: 0 이면 전체, 양수면 그 개수만 사용(빠른 실험용).

    Returns:
        학습 결과 요약 dict.

    Raises:
        RuntimeError: torch 미설치 또는 데이터 없음.
    """
    if not check_torch():
        msg = (
            "PyTorch 가 설치되지 않았습니다.\n"
            "  pip install torch==2.8.0\n"
            "학습 없이 서비스를 쓰려면 템플릿 기반 폴백 OCR 이 자동으로 동작합니다."
        )
        raise RuntimeError(msg)

    import torch  # noqa: PLC0415

    from app.ocr.crnn import build_crnn  # noqa: PLC0415

    samples = load_dataset(data_dir)
    if not samples:
        msg = f"학습 데이터가 없습니다: {data_dir}"
        raise RuntimeError(msg)
    if max_samples > 0:
        samples = samples[: int(max_samples)]

    # 재현성: 모든 난수원의 시드를 고정한다.
    rng = random.Random(int(seed))
    torch.manual_seed(int(seed))
    np.random.seed(int(seed) % (2**32))
    rng.shuffle(samples)

    split = max(1, int(len(samples) * float(val_ratio)))
    val_samples = samples[:split]
    train_samples = samples[split:]
    if not train_samples:
        msg = "학습 분할이 비었습니다. 샘플 수를 늘리거나 val_ratio 를 줄이세요."
        raise RuntimeError(msg)

    model = build_crnn(num_classes=NUM_CLASSES).to(device)
    # zero_infinity=True 이유는 모듈 docstring 3번 참조.
    criterion = torch.nn.CTCLoss(blank=BLANK_INDEX, reduction="mean", zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=1e-4)
    # ReduceLROnPlateau: 정확도가 정체되면 학습률을 낮춘다.
    # 코사인 스케줄보다 이쪽이 "언제 수렴할지 모르는" 상황에 적합하다.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    state = TrainState()
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    logger.info(
        "학습 시작: train=%d val=%d epochs=%d batch=%d lr=%.2e device=%s classes=%d",
        len(train_samples),
        len(val_samples),
        epochs,
        batch_size,
        learning_rate,
        device,
        NUM_CLASSES,
    )

    for epoch in range(1, int(epochs) + 1):
        model.train()
        rng.shuffle(train_samples)
        epoch_loss = 0.0
        batches = 0

        for start in range(0, len(train_samples), batch_size):
            chunk = train_samples[start : start + batch_size]
            prepared = collate_batch(chunk, height, width)
            if prepared is None:
                continue
            images, labels = prepared
            tensor = torch.from_numpy(images).to(device)

            log_probs = model.log_probs_for_ctc(tensor)  # (T, B, C)
            input_lengths = torch.full(
                (tensor.size(0),), log_probs.size(0), dtype=torch.long
            )
            target_lengths = torch.tensor([len(label) for label in labels], dtype=torch.long)
            targets = torch.tensor(
                [idx for label in labels for idx in label], dtype=torch.long
            )

            loss = criterion(log_probs, targets, input_lengths, target_lengths)
            if not torch.isfinite(loss):
                # inf/NaN 배치는 건너뛴다. 기울기를 적용하면 가중치가 전부 NaN 이 된다.
                logger.warning("에폭 %d: 비유한 손실 배치 건너뜀", epoch)
                continue

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            # 기울기 클리핑: RNN 은 기울기 폭발이 흔하다. 5.0 은 CRNN 관례값이다.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_loss += float(loss.item())
            batches += 1

        train_loss = epoch_loss / max(1, batches)
        val_loss, val_accuracy = evaluate(
            model,
            val_samples,
            height=height,
            width=width,
            batch_size=batch_size,
            criterion=criterion,
            device=device,
        )
        scheduler.step(val_accuracy)

        state.history.append(
            {
                "epoch": float(epoch),
                "train_loss": round(train_loss, 6),
                "val_loss": round(val_loss, 6),
                "val_accuracy": round(val_accuracy, 6),
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
        )
        logger.info(
            "에폭 %d/%d — train_loss=%.4f val_loss=%.4f val_acc=%.4f lr=%.2e",
            epoch,
            epochs,
            train_loss,
            val_loss,
            val_accuracy,
            optimizer.param_groups[0]["lr"],
        )

        # --- 체크포인트: 정확도 기준으로 최고 모델만 저장한다 ---
        if val_accuracy > state.best_accuracy:
            state.best_accuracy = val_accuracy
            state.best_loss = val_loss
            state.epochs_without_improvement = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "num_classes": NUM_CLASSES,
                    "blank_index": BLANK_INDEX,
                    "input_height": height,
                    "input_width": width,
                    "epoch": epoch,
                    "val_accuracy": val_accuracy,
                    "val_loss": val_loss,
                },
                output_path,
            )
            logger.info("체크포인트 저장: %s (val_acc=%.4f)", output_path, val_accuracy)
        else:
            state.epochs_without_improvement += 1
            if state.epochs_without_improvement >= int(patience):
                logger.info(
                    "EarlyStopping: %d 에폭 동안 개선 없음 (최고 val_acc=%.4f)",
                    patience,
                    state.best_accuracy,
                )
                break

    elapsed = time.perf_counter() - started
    return {
        "output": str(output_path),
        "best_val_accuracy": round(state.best_accuracy, 6),
        "best_val_loss": round(state.best_loss, 6),
        "epochs_run": len(state.history),
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "elapsed_sec": round(elapsed, 1),
        "history": state.history,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    """CLI 인자 파서를 만든다.

    Returns:
        argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(
        prog="python -m app.train.train_crnn",
        description="CRNN + CTC 번호판 OCR 학습",
    )
    parser.add_argument("--data", default="./data/ocr", help="데이터셋 루트(labels.jsonl 포함)")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="최대 에폭")
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH, help="배치 크기")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help="초기 학습률")
    parser.add_argument(
        "--patience", type=int, default=DEFAULT_PATIENCE, help="EarlyStopping 인내 에폭"
    )
    parser.add_argument("--val-ratio", type=float, default=0.1, help="검증 분할 비율")
    parser.add_argument("--height", type=int, default=32, help="입력 높이")
    parser.add_argument("--width", type=int, default=160, help="입력 너비")
    parser.add_argument("--device", default="cpu", help="cpu | cuda")
    parser.add_argument(
        "--out", default="./models/crnn_ctc_kr.pt", help="체크포인트 저장 경로"
    )
    parser.add_argument("--seed", type=int, default=20261024, help="재현성 시드")
    parser.add_argument(
        "--max-samples", type=int, default=0, help="0 = 전체, 양수 = 빠른 실험용 제한"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI 진입점.

    Args:
        argv: 인자 리스트.

    Returns:
        종료 코드(0 = 성공, 2 = 의존성/데이터 없음, 1 = 학습 실패).
    """
    args = build_arg_parser().parse_args(argv)
    configure_logging("INFO")

    if not check_torch():
        print(
            "[SKIP] PyTorch 가 설치되지 않았습니다.\n"
            "  pip install torch==2.8.0\n"
            "학습을 건너뛰어도 서비스는 템플릿 기반 폴백 OCR 로 정상 동작합니다.",
            file=sys.stderr,
        )
        return 2

    try:
        summary = train(
            args.data,
            epochs=args.epochs,
            batch_size=args.batch,
            learning_rate=args.lr,
            patience=args.patience,
            val_ratio=args.val_ratio,
            height=args.height,
            width=args.width,
            device=args.device,
            output=args.out,
            seed=args.seed,
            max_samples=args.max_samples,
        )
    except RuntimeError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] 학습 실패: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    # history 는 길어서 요약만 출력한다.
    brief = {k: v for k, v in summary.items() if k != "history"}
    print(json.dumps(brief, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "DEFAULT_BATCH",
    "DEFAULT_EPOCHS",
    "DEFAULT_LR",
    "DEFAULT_PATIENCE",
    "OcrSample",
    "TrainState",
    "check_torch",
    "collate_batch",
    "evaluate",
    "load_dataset",
    "load_image_gray",
    "main",
    "train",
]
