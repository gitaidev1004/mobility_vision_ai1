"""YOLO 번호판 검출기 학습 스크립트 (CLI).

사용법::

    # 1) 합성 데이터 생성
    python -m app.train.synth_plate --out ./data/synth --count 2000

    # 2) 학습
    python -m app.train.train_detector --data ./data/synth/data.yaml --epochs 50

    # 3) ONNX 내보내기
    python -m app.train.export_onnx --detector ./models/plate_yolo_nano.pt

하이퍼파라미터 선택 근거(번호판 검출의 특수성):

    **단일 클래스**: 번호판 하나뿐이므로 분류 손실 가중치를 낮춰도 된다.
    박스 회귀 정확도가 훨씬 중요하다 — 박스가 어긋나면 OCR 이 문자를 잘라 먹는다.

    **imgsz=640**: 번호판은 화면에서 작다(폭 100~300px). 416 으로 줄이면
    문자 구조가 사라져 검출률이 급락한다. 640 이 속도/정확도 균형점이다.

    **mosaic 증강 비활성화 권장**: mosaic 은 4장을 합쳐 작은 객체를 더 작게 만든다.
    번호판처럼 "이미 작은 단일 객체" 에는 역효과가 크다. 대신 합성 데이터
    생성기(``synth_plate``)에서 현실적인 증강을 이미 적용했다.

    **fliplr=0.0 (좌우 반전 금지)**: 번호판 문자는 좌우 반전하면 읽을 수 없는
    형태가 된다. 검출기만 학습할 때는 영향이 적지만, 같은 데이터로 OCR 을
    학습하면 치명적이다. 일관성을 위해 끈다.

``ultralytics`` 가 없으면 설치 안내를 출력하고 종료한다(import 단계에서 터지지 않는다).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from app.core.logging import configure_logging, get_logger

logger = get_logger(__name__)

#: 기본 모델. ``yolov8n`` 은 약 3.2M 파라미터로 CPU 추론이 가능한 가장 작은 급이다.
DEFAULT_MODEL: str = "yolov8n.pt"

#: 권장 학습 설정. 교재 독자가 그대로 써도 수렴하는 값으로 잡았다.
DEFAULT_EPOCHS: int = 50
DEFAULT_BATCH: int = 16
DEFAULT_IMG_SIZE: int = 640


def check_ultralytics() -> bool:
    """ultralytics 설치 여부를 확인한다.

    Returns:
        설치되어 있으면 True.
    """
    try:
        import importlib.util  # noqa: PLC0415

        return importlib.util.find_spec("ultralytics") is not None
    except (ImportError, ValueError):
        return False


def ensure_dataset(data_yaml: str) -> bool:
    """데이터셋 YAML 이 존재하는지 확인한다.

    Args:
        data_yaml: ``data.yaml`` 경로.

    Returns:
        존재하면 True.
    """
    path = Path(str(data_yaml))
    if path.is_file():
        return True
    logger.error(
        "데이터셋 YAML 이 없습니다: %s\n"
        "먼저 합성 데이터를 생성하세요: python -m app.train.synth_plate --out ./data/synth --count 2000",
        data_yaml,
    )
    return False


def train(
    data_yaml: str,
    *,
    model: str = DEFAULT_MODEL,
    epochs: int = DEFAULT_EPOCHS,
    batch: int = DEFAULT_BATCH,
    img_size: int = DEFAULT_IMG_SIZE,
    device: str = "cpu",
    project: str = "./runs/detect",
    name: str = "plate",
    patience: int = 15,
    output_weights: str | None = None,
) -> dict[str, Any]:
    """YOLO 검출기를 학습한다.

    Args:
        data_yaml: ultralytics 데이터셋 YAML 경로.
        model: 사전학습 모델 또는 체크포인트.
        epochs: 학습 에폭 수.
        batch: 배치 크기. CPU 학습 시에는 4~8 로 줄이는 것이 현실적이다.
        img_size: 입력 크기.
        device: ``"cpu"`` | ``"0"``(GPU 인덱스).
        project: 결과 저장 루트.
        name: 실행 이름.
        patience: EarlyStopping 인내 에폭 수.
        output_weights: 학습 완료 후 best.pt 를 복사할 경로.

    Returns:
        학습 결과 요약 dict.

    Raises:
        RuntimeError: ultralytics 미설치 또는 학습 실패.
    """
    if not check_ultralytics():
        msg = (
            "ultralytics 가 설치되지 않았습니다.\n"
            "  pip install ultralytics==8.3.200\n"
            "학습 없이 서비스를 쓰려면 OpenCV 폴백 검출기가 자동으로 동작합니다."
        )
        raise RuntimeError(msg)
    if not ensure_dataset(data_yaml):
        msg = f"데이터셋을 찾을 수 없습니다: {data_yaml}"
        raise RuntimeError(msg)

    from ultralytics import YOLO  # noqa: PLC0415

    detector = YOLO(str(model))
    logger.info(
        "학습 시작: model=%s data=%s epochs=%d batch=%d imgsz=%d device=%s",
        model,
        data_yaml,
        epochs,
        batch,
        img_size,
        device,
    )

    results = detector.train(
        data=str(data_yaml),
        epochs=int(epochs),
        batch=int(batch),
        imgsz=int(img_size),
        device=str(device),
        project=str(project),
        name=str(name),
        exist_ok=True,
        patience=int(patience),       # EarlyStopping
        # --- 번호판 특화 증강 설정(모듈 docstring 의 근거 참조) ---
        mosaic=0.0,                   # 작은 객체를 더 작게 만들어 역효과
        fliplr=0.0,                   # 번호판 문자는 좌우 반전 불가
        flipud=0.0,                   # 상하 반전도 현실에 없다
        degrees=7.0,                  # 카메라 설치각 정도의 회전만 허용
        translate=0.08,
        scale=0.25,                   # 거리 변화 모사
        shear=3.0,                    # 투시 왜곡 근사
        perspective=0.0005,
        hsv_h=0.012,                  # 번호판 색은 중요한 단서라 색조 변화는 최소
        hsv_s=0.5,
        hsv_v=0.4,                    # 명도는 크게 흔든다(조명 변화)
        # --- 최적화 ---
        optimizer="AdamW",
        lr0=0.002,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        cos_lr=True,                  # 코사인 스케줄이 후반 수렴에 유리
        # 단일 클래스이므로 분류 손실 가중치를 낮추고 박스 회귀를 강조한다.
        box=8.0,
        cls=0.3,
        dfl=1.5,
        plots=True,
        verbose=True,
    )

    # best.pt 를 지정 경로로 복사한다 — 서비스는 ``DETECTOR_WEIGHTS`` 를 읽는다.
    best_path = Path(project) / name / "weights" / "best.pt"
    copied: str | None = None
    if output_weights and best_path.is_file():
        try:
            import shutil  # noqa: PLC0415

            target = Path(str(output_weights))
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(best_path, target)
            copied = str(target)
            logger.info("가중치 복사 완료: %s → %s", best_path, target)
        except OSError as exc:
            logger.warning("가중치 복사 실패: %s", exc)

    metrics: dict[str, Any] = {}
    try:
        # results.results_dict 는 버전에 따라 키가 다르므로 방어적으로 읽는다.
        raw = getattr(results, "results_dict", None) or {}
        metrics = {str(k): float(v) for k, v in raw.items() if isinstance(v, (int, float))}
    except Exception:  # noqa: BLE001
        metrics = {}

    return {
        "model": str(model),
        "data": str(data_yaml),
        "epochs": int(epochs),
        "best_weights": str(best_path) if best_path.is_file() else None,
        "copied_to": copied,
        "metrics": metrics,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    """CLI 인자 파서를 만든다.

    Returns:
        argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(
        prog="python -m app.train.train_detector",
        description="YOLO 번호판 검출기 학습",
    )
    parser.add_argument(
        "--data", default="./data/synth/data.yaml", help="ultralytics 데이터셋 YAML"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="사전학습 모델 또는 체크포인트")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="학습 에폭")
    parser.add_argument(
        "--batch", type=int, default=DEFAULT_BATCH, help="배치 크기(CPU 학습 시 4~8 권장)"
    )
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMG_SIZE, help="입력 크기")
    parser.add_argument("--device", default="cpu", help="cpu 또는 GPU 인덱스(0)")
    parser.add_argument("--project", default="./runs/detect", help="결과 저장 루트")
    parser.add_argument("--name", default="plate", help="실행 이름")
    parser.add_argument("--patience", type=int, default=15, help="EarlyStopping 인내 에폭")
    parser.add_argument(
        "--out",
        default="./models/plate_yolo_nano.pt",
        help="학습 완료 후 best.pt 복사 경로(서비스가 읽는 위치)",
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

    if not check_ultralytics():
        print(
            "[SKIP] ultralytics 가 설치되지 않았습니다.\n"
            "  pip install ultralytics==8.3.200\n"
            "학습을 건너뛰어도 서비스는 OpenCV 폴백 검출기로 정상 동작합니다.",
            file=sys.stderr,
        )
        return 2

    try:
        summary = train(
            args.data,
            model=args.model,
            epochs=args.epochs,
            batch=args.batch,
            img_size=args.imgsz,
            device=args.device,
            project=args.project,
            name=args.name,
            patience=args.patience,
            output_weights=args.out,
        )
    except RuntimeError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] 학습 실패: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "DEFAULT_BATCH",
    "DEFAULT_EPOCHS",
    "DEFAULT_IMG_SIZE",
    "DEFAULT_MODEL",
    "check_ultralytics",
    "ensure_dataset",
    "main",
    "train",
]
